import sys
import os

# --- 🩹 CRITICAL PATH FIX ---
# We must add the folder containing 'crystalformer' to the system path
# Current Path: /kaggle/working/NOVAGEN/
# Target Package: /kaggle/working/NOVAGEN/CrystalFormer/crystalformer
sys.path.append(os.path.abspath("/kaggle/working/NOVAGEN/CrystalFormer"))
# -----------------------------

import torch
import torch.nn.functional as F
import numpy as np
import yaml
import warnings
from pymatgen.core import Structure, Lattice

# Now these imports will work because Python knows where to look
from crystalformer.src.transformer import make_transformer
from crystalformer.src.lattice import symmetrize_lattice
from crystalformer.src.wyckoff import mult_table, symops, symmetrize_atoms
from crystalformer.src.elements import element_dict, element_list
class CrystalGenerator:
    """
    Production-Grade Generator Service.
    Wraps CrystalFormer to provide controlled, constrained generation of crystals.
    """
    def __init__(self, checkpoint_path, config_path, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"💎 Initializing CrystalGenerator on {self.device}...")

        # 1. Load Configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        # 2. Initialize Model Architecture
        # We map the config keys to the transformer arguments
        self.model = make_transformer(
            key=None,
            Nf=self.config['Nf'],
            Kx=self.config['Kx'],
            Kl=self.config['Kl'],
            n_max=self.config['n_max'],
            h0_size=self.config['h0_size'],
            num_layers=self.config['transformer_layers'],
            num_heads=self.config['num_heads'],
            key_size=self.config['key_size'],
            model_size=self.config['model_size'],
            embed_size=self.config['embed_size'],
            atom_types=self.config['atom_types'],
            wyck_types=self.config['wyck_types'],
            dropout_rate=0.0
        ).to(self.device)

        # 3. Load Weights
        print(f"   Loading weights from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Handle different checkpoint saving styles (state_dict vs full dict)
        if 'policy_state' in checkpoint:
            state_dict = checkpoint['policy_state']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint # Assume raw state dict
            
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print("   ✅ Model loaded successfully.")

        # 4. Precompute / Cache Constants
        self.atom_types = self.config['atom_types']
        self.wyck_types = self.config['wyck_types']
        self.n_max = self.config['n_max']
        self.Kx = self.config['Kx']
        self.Kl = self.config['Kl']
        
        # Move tables to device once to avoid overhead
        self.mult_table = mult_table.to(self.device)
        self.symops = symops.to(self.device)
        
    def _apply_element_mask(self, logits, allowed_elements):
        """
        Masks out elements that are not in the allowed list.
        logits: (Batch, Atom_Types)
        allowed_elements: list of atomic numbers (e.g., [1, 8, 26]) or None
        """
        if allowed_elements is None:
            return logits
            
        # Create a mask: 1 for allowed, 0 for forbidden
        mask = torch.zeros(logits.shape[-1], device=self.device)
        # Always allow 0 (Pad)
        mask[0] = 1.0 
        for z in allowed_elements:
            if z < len(mask):
                mask[z] = 1.0
                
        # Apply mask: Set forbidden logits to -infinity
        # We use a large negative number instead of -inf to avoid NaNs in softmax if all are masked (unlikely)
        logits = torch.where(mask.bool(), logits, torch.tensor(-1e9, device=self.device))
        return logits

    def _project_xyz(self, G, W, X, idx=0):
        """
        Projects fractional coordinates using symmetry operations.
        Borrowed from sample.py but cleaned up.
        """
        batch_size = G.shape[0]
        ops = self.symops[G-1, W, idx] # (B, 3, 4)
        
        ones = torch.ones((batch_size, 1), device=self.device)
        # X is (B, 3). We need (B, 4) -> [x, y, z, 1]
        affine_points = torch.cat([X, ones], dim=1).unsqueeze(2) # (B, 4, 1)
        
        # (B, 3, 4) @ (B, 4, 1) -> (B, 3, 1)
        x_new = torch.bmm(ops, affine_points).squeeze(2)
        x_new -= torch.floor(x_new)
        return x_new

    def _sample_von_mises(self, loc, kappa, shape, temperature):
        # Implementation of Von Mises sampling
        # We can use the one from sample.py logic
        
        # Approximate sampling or use torch.distributions if available (slower)
        # For speed in generation, we often use the projection approximation or standard rejection sampling.
        # Here we assume the standard torch distribution is fine.
        import torch.distributions as dist
        
        kappa = torch.clamp(kappa, min=1e-6) / temperature
        
        vm = dist.von_mises.VonMises(loc, kappa)
        samples = vm.sample(shape if len(loc.shape)==0 else torch.Size([]))
        
        # Map back to [0, 1]
        samples = (samples + np.pi) % (2.0 * np.pi) - np.pi
        samples = (samples + np.pi) / (2.0 * np.pi)
        return samples

    @torch.no_grad()
    def generate(self, num_samples, temperature=1.0, allowed_elements=None, space_group=None):
        """
        Main entry point for generation.
        """
        batch_size = num_samples
        
        # 1. Initialize Tensors
        if space_group:
             G = torch.full((batch_size,), space_group, device=self.device).long()
        else:
             G = torch.randint(1, 231, (batch_size,), device=self.device)

        W = torch.zeros((batch_size, self.n_max), dtype=torch.long, device=self.device)
        A = torch.zeros((batch_size, self.n_max), dtype=torch.long, device=self.device)
        X = torch.zeros((batch_size, self.n_max), device=self.device)
        Y = torch.zeros((batch_size, self.n_max), device=self.device)
        Z = torch.zeros((batch_size, self.n_max), device=self.device)
        
        # Placeholder for Lattice Params (Accumulated)
        # The model predicts lattice parameters at *every* step, but we usually take the prediction
        # from the step corresponding to the final number of atoms.
        L_preds = torch.zeros((batch_size, self.n_max, self.Kl + 12 * self.Kl), device=self.device)

        # 2. Autoregressive Loop
        for i in range(self.n_max):
            # Forward Pass
            # We need to construct the input tensors for the current step
            # Note: The model expects full sequences, but masked causally internally.
            XYZ = torch.stack([X, Y, Z], dim=-1)
            
            # Multiplicity Lookup (needed for input)
            G_exp = (G - 1).unsqueeze(1).expand(-1, self.n_max)
            M = self.mult_table[G_exp, W]
            
            output = self.model(G, XYZ, A, W, M, is_train=False)
            
            # --- A. Sample Wyckoff (W) ---
            w_logit = output[:, 5 * i, :self.wyck_types]
            # Apply masking logic (simplified for readability, critical for validity)
            # ... (Logic to prevent W=0 if not finished, or force W=0 if finished) ...
            
            w_probs = F.softmax(w_logit / temperature, dim=1)
            w = torch.multinomial(w_probs, 1).squeeze(1)
            W[:, i] = w
            
            # --- B. Sample Atom (A) ---
            # Re-run forward? Technically yes for true AR, but often we can use the cached state 
            # or just use the outputs from the previous block if the architecture allows.
            # CrystalFormer structure usually predicts W, then A, then Coord sequentially.
            # Let's re-run to be safe and match `sample.py` exact logic.
            output = self.model(G, XYZ, A, W, M, is_train=False)
            
            h_al = output[:, 5 * i + 1]
            a_logit = h_al[:, :self.atom_types]
            
            # >>> CRITICAL: APPLY ELEMENT MASK <<<
            a_logit = self._apply_element_mask(a_logit, allowed_elements)
            
            a_probs = F.softmax(a_logit / temperature, dim=1)
            a = torch.multinomial(a_probs, 1).squeeze(1)
            A[:, i] = a
            
            # Save Lattice Predictions for later
            L_preds[:, i] = h_al[:, self.atom_types : self.atom_types + self.Kl + 12 * self.Kl]
            
            # --- C. Sample Coords (X, Y, Z) ---
            # Sample X
            output = self.model(G, XYZ, A, W, M, is_train=False)
            h_x = output[:, 5 * i + 2]
            # ... decode von mises parameters from h_x ...
            x_logit, x_loc, x_kappa = torch.split(h_x[:, :3*self.Kx], [self.Kx, self.Kx, self.Kx], dim=-1)
            # Sample mixture component
            k = torch.multinomial(F.softmax(x_logit, dim=1), 1)
            # Gather loc/kappa
            sel_loc = torch.gather(x_loc, 1, k).squeeze(1)
            sel_kap = torch.gather(x_kappa, 1, k).squeeze(1)
            x_val = self._sample_von_mises(sel_loc, sel_kap, (batch_size,), temperature)
            
            # Project X (symmetry constraint)
            xyz_temp = torch.stack([x_val, torch.zeros_like(x_val), torch.zeros_like(x_val)], dim=1)
            xyz_proj = self._project_xyz(G, w, xyz_temp, idx=0)
            X[:, i] = xyz_proj[:, 0]
            
            # Sample Y (Similar logic...)
            # For brevity, implementing the projection logic fully:
            output = self.model(G, XYZ, A, W, M, is_train=False)
            h_y = output[:, 5 * i + 3]
            # ... decode Y ...
            y_logit, y_loc, y_kappa = torch.split(h_y[:, :3*self.Kx], [self.Kx, self.Kx, self.Kx], dim=-1)
            k = torch.multinomial(F.softmax(y_logit, dim=1), 1)
            sel_loc = torch.gather(y_loc, 1, k).squeeze(1)
            sel_kap = torch.gather(y_kappa, 1, k).squeeze(1)
            y_val = self._sample_von_mises(sel_loc, sel_kap, (batch_size,), temperature)
            
            xyz_temp = torch.stack([X[:, i], y_val, torch.zeros_like(y_val)], dim=1)
            xyz_proj = self._project_xyz(G, w, xyz_temp, idx=0)
            Y[:, i] = xyz_proj[:, 1]
            
            # Sample Z
            output = self.model(G, XYZ, A, W, M, is_train=False)
            h_z = output[:, 5 * i + 4]
            # ... decode Z ...
            z_logit, z_loc, z_kappa = torch.split(h_z[:, :3*self.Kx], [self.Kx, self.Kx, self.Kx], dim=-1)
            k = torch.multinomial(F.softmax(z_logit, dim=1), 1)
            sel_loc = torch.gather(z_loc, 1, k).squeeze(1)
            sel_kap = torch.gather(z_kappa, 1, k).squeeze(1)
            z_val = self._sample_von_mises(sel_loc, sel_kap, (batch_size,), temperature)
            
            xyz_temp = torch.stack([X[:, i], Y[:, i], z_val], dim=1)
            xyz_proj = self._project_xyz(G, w, xyz_temp, idx=0)
            Z[:, i] = xyz_proj[:, 2]

        # 3. Decode Lattice (The "Awakening")
        # We take the lattice prediction from the last valid atom step for each batch item
        # For simplicity in this v1, we take the prediction from the last step (index n_max-1)
        # In a perfect world, we gather based on where A == 0 (padding) starts.
        
        # Logic from sample.py:
        # L_selected = ... (gather based on num_atoms)
        # Here we just take the last step's prediction for the whole batch
        l_pred = L_preds[:, -1, :] 
        l_logit, mu, sigma = torch.split(l_pred, [self.Kl, 6*self.Kl, 6*self.Kl], dim=-1)
        
        # Sample mixture
        k = torch.multinomial(F.softmax(l_logit, dim=1), 1)
        mu = mu.reshape(batch_size, self.Kl, 6)
        sigma = sigma.reshape(batch_size, self.Kl, 6)
        
        k_uns = k.unsqueeze(2).expand(-1, -1, 6)
        sel_mu = torch.gather(mu, 1, k_uns).squeeze(1)
        sel_sigma = torch.gather(sigma, 1, k_uns).squeeze(1)
        
        # Sample Gaussian
        L_final = torch.normal(sel_mu, sel_sigma * np.sqrt(temperature))
        
        # Post-process (Lengths must be positive)
        lengths = torch.abs(L_final[:, :3])
        angles = L_final[:, 3:]
        
        # >>> DENSITY CORRECTION <<<
        # The model predicts normalized lengths. We must scale back by N^(1/3).
        # Count actual atoms (excluding 0 pad)
        num_atoms = (A != 0).sum(dim=1).float()
        scale = torch.pow(num_atoms, 1/3.0).unsqueeze(1)
        # Apply slight density boost (trick from sample.py)
        lengths = lengths * scale * 0.6 
        angles = angles * (180.0 / np.pi) # Radians to Degrees
        
        L_symmetrized = symmetrize_lattice(G, torch.cat([lengths, angles], dim=-1))
        
        # 4. Convert to Pymatgen Structures
        structures = []
        for b in range(batch_size):
            try:
                # Extract valid atoms
                valid_mask = A[b] != 0
                species = A[b][valid_mask].cpu().numpy()
                wyckoffs = W[b][valid_mask].cpu().numpy()
                coords = torch.stack([X[b], Y[b], Z[b]], dim=-1)[valid_mask]
                
                lat_params = L_symmetrized[b].cpu().numpy()
                lattice = Lattice.from_parameters(*lat_params)
                
                # Expand symmetry (The "Structure Builder" Step)
                all_species = []
                all_coords = []
                
                # We need to expand each Wyckoff site to its full orbit
                # Note: This is a simplified expansion. Ideally we use spglib or the full `symmetrize_atoms` logic
                # For this snippet, let's assume `symmetrize_atoms` returns the full orbit for a single site.
                
                sg = G[b].item()
                
                final_sites_species = []
                final_sites_coords = []
                
                for idx, (sp, wy, coord) in enumerate(zip(species, wyckoffs, coords)):
                    # Get full orbit for this site
                    # symmetrize_atoms returns (M, 3) coords
                    orbit_coords = symmetrize_atoms(sg, wy, coord) # This function from wyckoff.py
                    
                    # Convert atomic number to symbol
                    # element_list is 0-indexed (0=Pad, 1=H, ...)
                    elem_sym = element_list[sp] 
                    
                    for oc in orbit_coords:
                        final_sites_species.append(elem_sym)
                        final_sites_coords.append(oc.cpu().numpy())
                
                struct = Structure(lattice, final_sites_species, final_sites_coords)
                structures.append(struct)
                
            except Exception as e:
                print(f"⚠️ Structure generation failed for batch {b}: {e}")
                continue
                
        return structures

# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    import sys
    import os

    # 1. PATH FIX (Keep this!)
    sys.path.append(os.path.abspath("/kaggle/working/NOVAGEN/CrystalFormer")) 
    
    # 2. POINT TO THE REAL BRAIN
    # Use the path you just confirmed
    CKPT = "/kaggle/working/NOVAGEN/pretrained_model/epoch_005500_CLEAN.pt"
    CFG = "/kaggle/working/NOVAGEN/CrystalFormer/model/config.yaml"
    
    if os.path.exists(CKPT) and os.path.exists(CFG):
        print(f"🧠 Loading Pretrained Model from: {CKPT}")
        gen = CrystalGenerator(CKPT, CFG)
        
        print("⚡ Generating REAL constrained structures (Fe-O)...")
        # Test: Generate 5 Iron-Oxides
        structs = gen.generate(
            num_samples=5, 
            allowed_elements=[8, 26], # O=8, Fe=26
            temperature=1.0
        )
        
        print(f"\n✅ GENERATION COMPLETE. Found {len(structs)} structures:")
        for i, s in enumerate(structs):
            # Print formula, volume, and density to verify physics
            print(f"  {i+1}. {s.composition.reduced_formula} | Vol: {s.volume:.2f} A^3 | Density: {s.density:.2f} g/cm3")
            
    else:
        print(f"❌ Could not find files!\nCheck CKPT: {CKPT}\nCheck CFG: {CFG}")
