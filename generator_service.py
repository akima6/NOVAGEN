import sys
import os
import time

# --- 1. PATH SETUP ---
sys.path.append(os.path.abspath("/kaggle/working/NOVAGEN/CrystalFormer")) 

import torch
import torch.nn.functional as F
import numpy as np
import yaml
import warnings
from tqdm import tqdm  # <--- NEW: Progress Bar

from pymatgen.core import Structure, Lattice

# Import internal modules
from crystalformer.src.transformer import make_transformer
from crystalformer.src.lattice import symmetrize_lattice
from crystalformer.src.wyckoff import mult_table, symops, symmetrize_atoms
from crystalformer.src.elements import element_dict, element_list

# --- 2. SPEED PATCH ---
# We suppress the warnings that slow down the loop
warnings.filterwarnings("ignore", category=UserWarning)

class CrystalGenerator:
    def __init__(self, checkpoint_path, config_path, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"💎 Initializing CrystalGenerator on {self.device}...")

        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        # Initialize Model
        self.model = make_transformer(
            key=None, Nf=self.config['Nf'], Kx=self.config['Kx'], Kl=self.config['Kl'], n_max=self.config['n_max'],
            h0_size=self.config['h0_size'], num_layers=self.config['transformer_layers'], num_heads=self.config['num_heads'],
            key_size=self.config['key_size'], model_size=self.config['model_size'], embed_size=self.config['embed_size'],
            atom_types=self.config['atom_types'], wyck_types=self.config['wyck_types'], dropout_rate=0.0
        ).to(self.device)

        # Load Weights
        print(f"   Loading weights from {os.path.basename(checkpoint_path)}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        state_dict = checkpoint.get('policy_state', checkpoint.get('model_state_dict', checkpoint))
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        # Cache Constants
        self.n_max = self.config['n_max']
        self.atom_types = self.config['atom_types']
        self.Kl = self.config['Kl']
        self.Kx = self.config['Kx']
        self.wyck_types = self.config['wyck_types']
        
        # GPU Tables
        self.mult_table = mult_table.to(self.device)
        self.symops = symops.to(self.device)

    def _apply_element_mask(self, logits, allowed_elements):
        if allowed_elements is None: return logits
        mask = torch.zeros(logits.shape[-1], device=self.device)
        mask[0] = 1.0 
        for z in allowed_elements:
            if z < len(mask): mask[z] = 1.0
        return torch.where(mask.bool(), logits, torch.tensor(-1e9, device=self.device))

    def _project_xyz(self, G, W, X, idx=0):
        batch_size = G.shape[0]
        ops = self.symops[G-1, W, idx] 
        ones = torch.ones((batch_size, 1), device=self.device)
        affine_points = torch.cat([X, ones], dim=1).unsqueeze(2)
        x_new = torch.bmm(ops, affine_points).squeeze(2)
        x_new -= torch.floor(x_new)
        return x_new

    def _sample_von_mises(self, loc, kappa, shape, temperature):
        import torch.distributions as dist
        kappa = torch.clamp(kappa, min=1e-6) / temperature
        vm = dist.von_mises.VonMises(loc, kappa)
        samples = vm.sample(shape if len(loc.shape)==0 else torch.Size([]))
        return (samples + np.pi) / (2.0 * np.pi)

    @torch.no_grad()
    def generate(self, num_samples, temperature=1.0, allowed_elements=None):
        batch_size = num_samples
        G = torch.randint(1, 231, (batch_size,), device=self.device)
        
        W = torch.zeros((batch_size, self.n_max), dtype=torch.long, device=self.device)
        A = torch.zeros((batch_size, self.n_max), dtype=torch.long, device=self.device)
        X = torch.zeros((batch_size, self.n_max), device=self.device)
        Y = torch.zeros((batch_size, self.n_max), device=self.device)
        Z = torch.zeros((batch_size, self.n_max), device=self.device)
        L_preds = torch.zeros((batch_size, self.n_max, self.Kl + 12 * self.Kl), device=self.device)

        print(f"   🌊 Autoregressive Sampling ({self.n_max} steps)...")
        # 1. MODEL LOOP (The slow part)
        for i in tqdm(range(self.n_max), desc="Sampling"):
            XYZ = torch.stack([X, Y, Z], dim=-1)
            G_exp = (G - 1).unsqueeze(1).expand(-1, self.n_max)
            M = self.mult_table[G_exp, W]
            
            output = self.model(G, XYZ, A, W, M, is_train=False)
            
            # Sample W
            w_logit = output[:, 5 * i, :self.wyck_types]
            w_probs = F.softmax(w_logit / temperature, dim=1)
            w = torch.multinomial(w_probs, 1).squeeze(1)
            W[:, i] = w
            
            # Sample A
            output = self.model(G, XYZ, A, W, M, is_train=False)
            a_logit = output[:, 5 * i + 1, :self.atom_types]
            a_logit = self._apply_element_mask(a_logit, allowed_elements)
            a_probs = F.softmax(a_logit / temperature, dim=1)
            a = torch.multinomial(a_probs, 1).squeeze(1)
            A[:, i] = a
            
            L_preds[:, i] = output[:, 5 * i + 1, self.atom_types : self.atom_types + self.Kl + 12 * self.Kl]

            # Sample Coordinates (X, Y, Z) - Simplified logic for brevity
            # (In a real run, this would be the full Von Mises logic)
            # For this test, we just project random noise to check pipeline speed
            # If you want full physics, uncomment the detailed sampling from before.
            # Here we use the valid projection logic but dummy sampling to speed up the test.
            
            # REAL SAMPLING LOGIC (Uncommented for correctness)
            output = self.model(G, XYZ, A, W, M, is_train=False)
            h_x = output[:, 5 * i + 2]
            x_logit, x_loc, x_kappa = torch.split(h_x[:, :3*self.Kx], [self.Kx, self.Kx, self.Kx], dim=-1)
            k = torch.multinomial(F.softmax(x_logit, dim=1), 1)
            sel_loc = torch.gather(x_loc, 1, k).squeeze(1)
            sel_kap = torch.gather(x_kappa, 1, k).squeeze(1)
            x_val = self._sample_von_mises(sel_loc, sel_kap, (batch_size,), temperature)
            xyz_temp = torch.stack([x_val, torch.zeros_like(x_val), torch.zeros_like(x_val)], dim=1)
            X[:, i] = self._project_xyz(G, w, xyz_temp, idx=0)[:, 0]

            output = self.model(G, XYZ, A, W, M, is_train=False)
            h_y = output[:, 5 * i + 3]
            y_logit, y_loc, y_kappa = torch.split(h_y[:, :3*self.Kx], [self.Kx, self.Kx, self.Kx], dim=-1)
            k = torch.multinomial(F.softmax(y_logit, dim=1), 1)
            sel_loc = torch.gather(y_loc, 1, k).squeeze(1)
            sel_kap = torch.gather(y_kappa, 1, k).squeeze(1)
            y_val = self._sample_von_mises(sel_loc, sel_kap, (batch_size,), temperature)
            xyz_temp = torch.stack([X[:, i], y_val, torch.zeros_like(y_val)], dim=1)
            Y[:, i] = self._project_xyz(G, w, xyz_temp, idx=0)[:, 1]

            output = self.model(G, XYZ, A, W, M, is_train=False)
            h_z = output[:, 5 * i + 4]
            z_logit, z_loc, z_kappa = torch.split(h_z[:, :3*self.Kx], [self.Kx, self.Kx, self.Kx], dim=-1)
            k = torch.multinomial(F.softmax(z_logit, dim=1), 1)
            sel_loc = torch.gather(z_loc, 1, k).squeeze(1)
            sel_kap = torch.gather(z_kappa, 1, k).squeeze(1)
            z_val = self._sample_von_mises(sel_loc, sel_kap, (batch_size,), temperature)
            xyz_temp = torch.stack([X[:, i], Y[:, i], z_val], dim=1)
            Z[:, i] = self._project_xyz(G, w, xyz_temp, idx=0)[:, 2]


        print("   🔨 Reconstructing Lattices...")
        l_pred = L_preds[:, -1, :] 
        l_logit, mu, sigma = torch.split(l_pred, [self.Kl, 6*self.Kl, 6*self.Kl], dim=-1)
        k = torch.multinomial(F.softmax(l_logit, dim=1), 1)
        mu = mu.reshape(batch_size, self.Kl, 6)
        sigma = sigma.reshape(batch_size, self.Kl, 6)
        k_uns = k.unsqueeze(2).expand(-1, -1, 6)
        sel_mu = torch.gather(mu, 1, k_uns).squeeze(1)
        sel_sigma = torch.gather(sigma, 1, k_uns).squeeze(1)
        L_final = torch.normal(sel_mu, sel_sigma * np.sqrt(temperature))
        lengths = torch.abs(L_final[:, :3])
        angles = L_final[:, 3:]
        num_atoms = (A != 0).sum(dim=1).float()
        scale = torch.pow(num_atoms, 1/3.0).unsqueeze(1)
        lengths = lengths * scale * 0.6 
        angles = angles * (180.0 / np.pi)
        L_symmetrized = symmetrize_lattice(G, torch.cat([lengths, angles], dim=-1))

        structures = []
        print("   🏗️ Building Pymatgen Objects...")
        for b in range(batch_size):
            try:
                valid_mask = A[b] != 0
                species = A[b][valid_mask].cpu().numpy()
                wyckoffs = W[b][valid_mask].cpu().numpy()
                coords = torch.stack([X[b], Y[b], Z[b]], dim=-1)[valid_mask]
                lat_params = L_symmetrized[b].cpu().numpy()
                lattice = Lattice.from_parameters(*lat_params)
                
                sg = G[b].item()
                final_sites_species = []
                final_sites_coords = []
                
                # --- OPTIMIZATION: Only expand if not too huge ---
                if len(species) > 50:
                    print(f"   ⚠️ Skipping full symmetry expansion for structure {b} (Too many atoms: {len(species)})")
                    # Just add the unique sites for the test
                    for idx, (sp, coord) in enumerate(zip(species, coords)):
                        final_sites_species.append(element_list[sp])
                        final_sites_coords.append(coord.cpu().numpy())
                else:
                    for idx, (sp, wy, coord) in enumerate(zip(species, wyckoffs, coords)):
                        orbit_coords = symmetrize_atoms(sg, wy, coord)
                        elem_sym = element_list[sp] 
                        for oc in orbit_coords:
                            final_sites_species.append(elem_sym)
                            final_sites_coords.append(oc.cpu().numpy())
                
                struct = Structure(lattice, final_sites_species, final_sites_coords)
                structures.append(struct)
            except Exception as e:
                print(f"   ⚠️ Batch {b} failed: {e}")
                continue
                
        return structures

if __name__ == "__main__":
    CKPT = "/kaggle/working/NOVAGEN/pretrained_model/epoch_005500_CLEAN.pt"
    CFG = "/kaggle/working/NOVAGEN/CrystalFormer/model/config.yaml"
    
    if os.path.exists(CKPT):
        gen = CrystalGenerator(CKPT, CFG)
        print("⚡ Start Generation...")
        structs = gen.generate(3, allowed_elements=[8, 26]) # Reduced to 3 samples for speed
        for s in structs:
            print(f"✅ {s.composition.reduced_formula} | Vol: {s.volume:.1f}")
