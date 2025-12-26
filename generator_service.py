import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import yaml
import warnings
from tqdm import tqdm

from pymatgen.core import Structure, Lattice

# Import internal modules
sys.path.append(os.path.abspath("/kaggle/working/NOVAGEN/CrystalFormer"))
from crystalformer.src.transformer import make_transformer
from crystalformer.src.lattice import symmetrize_lattice
from crystalformer.src.wyckoff import mult_table, symops, symmetrize_atoms
from crystalformer.src.elements import element_dict, element_list

warnings.filterwarnings("ignore")

class CrystalGenerator:
    def __init__(self, checkpoint_path, config_path, device=None):
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        print(f"💎 Initializing CrystalGenerator on {self.device}...")

        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.model = make_transformer(
            key=None, Nf=self.config['Nf'], Kx=self.config['Kx'], Kl=self.config['Kl'], n_max=self.config['n_max'],
            h0_size=self.config['h0_size'], num_layers=self.config['transformer_layers'], num_heads=self.config['num_heads'],
            key_size=self.config['key_size'], model_size=self.config['model_size'], embed_size=self.config['embed_size'],
            atom_types=self.config['atom_types'], wyck_types=self.config['wyck_types'], dropout_rate=0.0
        ).to(self.device)

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
        
        self.mult_table = mult_table.to(self.device)
        self.symops = symops.to(self.device)

    def _apply_element_mask(self, logits, allowed_elements):
        if allowed_elements is None: return logits
        mask_val = torch.tensor(float('-inf'), device=self.device)
        masked_logits = torch.full_like(logits, mask_val)
        masked_logits[:, 0] = logits[:, 0]
        for z in allowed_elements:
            if z < logits.shape[-1]:
                masked_logits[:, z] = logits[:, z]
        return masked_logits

    def _project_xyz(self, G, W, X, idx=0):
        batch_size = G.shape[0]
        ops = self.symops[G-1, W, idx] 
        ones = torch.ones((batch_size, 1), device=self.device)
        affine_points = torch.cat([X, ones], dim=1).unsqueeze(2)
        x_new = torch.bmm(ops, affine_points).squeeze(2)
        x_new -= torch.floor(x_new)
        return x_new

    def _sample_von_mises(self, loc, kappa, shape, temperature):
        loc_cpu = loc.detach().cpu()
        kappa_cpu = torch.clamp(kappa, min=1e-6, max=1000.0).detach().cpu()
        sigma = 1.0 / torch.sqrt(kappa_cpu)
        sigma = sigma * np.sqrt(temperature)
        samples = torch.normal(loc_cpu, sigma)
        samples = (samples + np.pi) % (2.0 * np.pi) - np.pi
        final_val = (samples + np.pi) / (2.0 * np.pi)
        return final_val.to(self.device)

    def _safe_multinomial(self, probs):
        if torch.isnan(probs).any():
            probs = torch.nan_to_num(probs, nan=0.0)
        if probs.sum(dim=1).min() == 0:
            return torch.randint(0, probs.shape[1], (probs.shape[0],), device=self.device)
        try:
            return torch.multinomial(probs.detach().cpu(), 1).to(self.device).squeeze(1)
        except Exception:
            return torch.randint(0, probs.shape[1], (probs.shape[0],), device=self.device)

    def generate(self, num_samples, temperature=1.0, allowed_elements=None):
        with torch.no_grad():
            result = self._run_generation(num_samples, temperature, allowed_elements, with_grads=False)
            return result["structures"]

    def generate_with_grads(self, num_samples, allowed_elements=None, temperature=1.0):
        return self._run_generation(num_samples, temperature, allowed_elements, with_grads=True)

    def _run_generation(self, num_samples, temperature, allowed_elements, with_grads):
        batch_size = num_samples
        
        G = torch.randint(1, 231, (batch_size,), device=self.device)
        W = torch.zeros((batch_size, self.n_max), dtype=torch.long, device=self.device)
        A = torch.zeros((batch_size, self.n_max), dtype=torch.long, device=self.device)
        X = torch.zeros((batch_size, self.n_max), device=self.device)
        Y = torch.zeros((batch_size, self.n_max), device=self.device)
        Z = torch.zeros((batch_size, self.n_max), device=self.device)
        L_preds = torch.zeros((batch_size, self.n_max, self.Kl + 12 * self.Kl), device=self.device)
        
        log_probs_sum = torch.zeros(batch_size, device=self.device)

        iterator = range(self.n_max)
        if not with_grads:
            iterator = tqdm(iterator, desc="   Sampling")

        for i in iterator:
            # Prepare Input tensors (Read-Only)
            XYZ = torch.stack([X, Y, Z], dim=-1)
            G_exp = (G - 1).unsqueeze(1).expand(-1, self.n_max)
            M = self.mult_table[G_exp, W]
            
            # 1. WYCKOFF
            output = self.model(G, XYZ, A, W, M, is_train=False)
            w_logit = output[:, 5 * i, :self.wyck_types]
            
            if with_grads:
                w_dist = torch.distributions.Categorical(logits=w_logit / temperature)
                sample_w = w_dist.sample()
                log_probs_sum = log_probs_sum + w_dist.log_prob(sample_w)
                
                # UPDATE SAFEGUARD: Clone, then Update
                W = W.clone() 
                W[:, i] = sample_w
            else:
                w_probs = F.softmax(w_logit / temperature, dim=1)
                W[:, i] = self._safe_multinomial(w_probs)
            
            # 2. ATOM TYPES
            # Re-compute M/XYZ is not strictly needed if only W changed and is used for next step,
            # but for safety in the graph we assume dependencies.
            M = self.mult_table[G_exp, W] # Update M with new W
            
            output = self.model(G, XYZ, A, W, M, is_train=False)
            a_logit = output[:, 5 * i + 1, :self.atom_types]
            a_logit = self._apply_element_mask(a_logit, allowed_elements)
            
            if with_grads:
                a_dist = torch.distributions.Categorical(logits=a_logit / temperature)
                sample_a = a_dist.sample()
                log_probs_sum = log_probs_sum + a_dist.log_prob(sample_a)
                
                # UPDATE SAFEGUARD
                A = A.clone()
                A[:, i] = sample_a
            else:
                a_probs = F.softmax(a_logit / temperature, dim=1)
                A[:, i] = self._safe_multinomial(a_probs)
            
            L_preds[:, i] = output[:, 5 * i + 1, self.atom_types : self.atom_types + self.Kl + 12 * self.Kl]

            # 3. COORDINATES
            # X
            output = self.model(G, XYZ, A, W, M, is_train=False)
            h_x = output[:, 5 * i + 2]
            x_logit, x_loc, x_kappa = torch.split(h_x[:, :3*self.Kx], [self.Kx, self.Kx, self.Kx], dim=-1)
            k = self._safe_multinomial(F.softmax(x_logit, dim=1))
            sel_loc = torch.gather(x_loc, 1, k.unsqueeze(1)).squeeze(1)
            sel_kap = torch.gather(x_kappa, 1, k.unsqueeze(1)).squeeze(1)
            x_val = self._sample_von_mises(sel_loc, sel_kap, (batch_size,), temperature)
            
            # Update X
            if with_grads: X = X.clone()
            
            xyz_temp = torch.stack([x_val, torch.zeros_like(x_val), torch.zeros_like(x_val)], dim=1)
            X[:, i] = self._project_xyz(G, W[:, i], xyz_temp, idx=0)[:, 0]

            # Y
            XYZ = torch.stack([X, Y, Z], dim=-1) # Re-stack with new X
            output = self.model(G, XYZ, A, W, M, is_train=False)
            h_y = output[:, 5 * i + 3]
            y_logit, y_loc, y_kappa = torch.split(h_y[:, :3*self.Kx], [self.Kx, self.Kx, self.Kx], dim=-1)
            k = self._safe_multinomial(F.softmax(y_logit, dim=1))
            sel_loc = torch.gather(y_loc, 1, k.unsqueeze(1)).squeeze(1)
            sel_kap = torch.gather(y_kappa, 1, k.unsqueeze(1)).squeeze(1)
            y_val = self._sample_von_mises(sel_loc, sel_kap, (batch_size,), temperature)
            
            # Update Y
            if with_grads: Y = Y.clone()
            
            xyz_temp = torch.stack([X[:, i], y_val, torch.zeros_like(y_val)], dim=1)
            Y[:, i] = self._project_xyz(G, W[:, i], xyz_temp, idx=0)[:, 1]

            # Z
            XYZ = torch.stack([X, Y, Z], dim=-1) # Re-stack with new Y
            output = self.model(G, XYZ, A, W, M, is_train=False)
            h_z = output[:, 5 * i + 4]
            z_logit, z_loc, z_kappa = torch.split(h_z[:, :3*self.Kx], [self.Kx, self.Kx, self.Kx], dim=-1)
            k = self._safe_multinomial(F.softmax(z_logit, dim=1))
            sel_loc = torch.gather(z_loc, 1, k.unsqueeze(1)).squeeze(1)
            sel_kap = torch.gather(z_kappa, 1, k.unsqueeze(1)).squeeze(1)
            z_val = self._sample_von_mises(sel_loc, sel_kap, (batch_size,), temperature)
            
            # Update Z
            if with_grads: Z = Z.clone()
            
            xyz_temp = torch.stack([X[:, i], Y[:, i], z_val], dim=1)
            Z[:, i] = self._project_xyz(G, W[:, i], xyz_temp, idx=0)[:, 2]

        # 4. LATTICE
        l_pred = L_preds[:, -1, :] 
        l_logit, mu, sigma = torch.split(l_pred, [self.Kl, 6*self.Kl, 6*self.Kl], dim=-1)
        k = self._safe_multinomial(F.softmax(l_logit, dim=1))
        mu = mu.reshape(batch_size, self.Kl, 6)
        sigma = sigma.reshape(batch_size, self.Kl, 6)
        
        k_uns = k.unsqueeze(1).unsqueeze(2).expand(-1, -1, 6)
        sel_mu = torch.gather(mu, 1, k_uns).squeeze(1)
        sel_sigma = torch.gather(sigma, 1, k_uns).squeeze(1)
        sel_sigma = torch.nan_to_num(sel_sigma, nan=1.0)
        sel_sigma = torch.clamp(sel_sigma, max=100.0)
        
        sel_mu_cpu = sel_mu.detach().cpu()
        sel_sigma_cpu = sel_sigma.detach().cpu()
        L_final_cpu = torch.normal(sel_mu_cpu, sel_sigma_cpu * np.sqrt(temperature))
        L_final = L_final_cpu.to(self.device)
        
        lengths = torch.abs(L_final[:, :3]) + 0.1
        angles = L_final[:, 3:]
        num_atoms = (A != 0).sum(dim=1).float()
        scale = torch.pow(num_atoms, 1/3.0).unsqueeze(1)
        lengths = lengths * scale 
        angles = angles * (180.0 / np.pi)
        
        L_symmetrized = symmetrize_lattice(G, torch.cat([lengths, angles], dim=-1))

        structures = []
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
                
                if len(species) > 50:
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
            except Exception:
                continue
                
        return {"structures": structures, "log_probs": log_probs_sum}
