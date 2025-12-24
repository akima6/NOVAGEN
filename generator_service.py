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

# Import internal modules
from crystalformer.src.transformer import make_transformer
from crystalformer.src.lattice import symmetrize_lattice
from crystalformer.src.wyckoff import mult_table, symops, symmetrize_atoms
from crystalformer.src.elements import element_dict, element_list
from pymatgen.core import Structure, Lattice

# Suppress the spammy warnings
warnings.filterwarnings("ignore")

class CrystalGenerator:
    def __init__(self, checkpoint_path, config_path):
        # ⚠️ FORCE CPU FOR SAFETY & DEBUGGING
        self.device = "cpu" 
        print(f"💎 Initializing CrystalGenerator on {self.device} (Safe Mode)...")

        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        # Initialize Model
        print("   Building Model Architecture...")
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
        
        # Tables (Move to CPU)
        self.mult_table = mult_table.to(self.device)
        self.symops = symops.to(self.device)

    def _apply_element_mask(self, logits, allowed_elements):
        if allowed_elements is None: return logits
        mask = torch.zeros(logits.shape[-1], device=self.device)
        mask[0] = 1.0 
        for z in allowed_elements:
            if z < len(mask): mask[z] = 1.0
        return torch.where(mask.bool(), logits, torch.tensor(-1e9))

    def _project_xyz(self, G, W, X, idx=0):
        batch_size = G.shape[0]
        ops = self.symops[G-1, W, idx] 
        ones = torch.ones((batch_size, 1), device=self.device)
        affine_points = torch.cat([X, ones], dim=1).unsqueeze(2)
        x_new = torch.bmm(ops, affine_points).squeeze(2)
        x_new -= torch.floor(x_new)
        return x_new

    def _sample_von_mises(self, loc, kappa, shape, temperature):
        # Simplified Von Mises for CPU Test
        # Just return the location (mode) to avoid complex sampling logic errors
        return loc / (2.0 * np.pi) 

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

        print(f"   🌊 Sampling {self.n_max} steps (Verbose Mode)...")
        
        for i in range(self.n_max):
            # Print dot every step to prove life
            print(f".", end="", flush=True) 
            
            XYZ = torch.stack([X, Y, Z], dim=-1)
            G_exp = (G - 1).unsqueeze(1).expand(-1, self.n_max)
            M = self.mult_table[G_exp, W]
            
            # 1. Wyckoff
            output = self.model(G, XYZ, A, W, M, is_train=False)
            w_logit = output[:, 5 * i, :self.wyck_types]
            w_probs = F.softmax(w_logit / temperature, dim=1)
            w = torch.multinomial(w_probs, 1).squeeze(1)
            W[:, i] = w
            
            # 2. Atom
            output = self.model(G, XYZ, A, W, M, is_train=False)
            a_logit = output[:, 5 * i + 1, :self.atom_types]
            a_logit = self._apply_element_mask(a_logit, allowed_elements)
            a_probs = F.softmax(a_logit / temperature, dim=1)
            a = torch.multinomial(a_probs, 1).squeeze(1)
            A[:, i] = a
            
            # Lattice storage
            L_preds[:, i] = output[:, 5 * i + 1, self.atom_types : self.atom_types + self.Kl + 12 * self.Kl]

            # 3. Coords (Simplified projection for speed)
            output = self.model(G, XYZ, A, W, M, is_train=False)
            h_x = output[:, 5 * i + 2]
            # Just take the first mode for testing stability
            x_logit, x_loc, x_kappa = torch.split(h_x[:, :3*self.Kx], [self.Kx, self.Kx, self.Kx], dim=-1)
            k = torch.argmax(x_logit, dim=1)
            sel_loc = torch.gather(x_loc, 1, k.unsqueeze(1)).squeeze(1)
            x_val = self._sample_von_mises(sel_loc, None, (batch_size,), temperature)
            
            xyz_temp = torch.stack([x_val, torch.zeros_like(x_val), torch.zeros_like(x_val)], dim=1)
            X[:, i] = self._project_xyz(G, w, xyz_temp, idx=0)[:, 0]
            
            # (Repeat for Y and Z - skipping for brevity in debug mode, just filling X is enough to test loop)
            Y[:, i] = X[:, i] 
            Z[:, i] = X[:, i]

        print("\n   🔨 Reconstructing Lattices...")
        # ... (Rest of reconstruction code logic) ...
        # For the test, let's just return a success message if we passed the loop
        
        return ["Success"]

if __name__ == "__main__":
    CKPT = "/kaggle/working/NOVAGEN/pretrained_model/epoch_005500_CLEAN.pt"
    CFG = "/kaggle/working/NOVAGEN/CrystalFormer/model/config.yaml"
    
    if os.path.exists(CKPT):
        gen = CrystalGenerator(CKPT, CFG)
        print("⚡ Start Generation...")
        # Generate just 1 sample to be fast
        try:
            structs = gen.generate(1, allowed_elements=[8, 26]) 
            print("\n✅ DEBUG TEST PASSED: The loop runs correctly on CPU.")
        except Exception as e:
            print(f"\n❌ CRASHED: {e}")
            import traceback
            traceback.print_exc()
