import sys
import os
import torch
import torch.nn as nn
import numpy as np
import yaml
import warnings
from pymatgen.core import Structure, Lattice

# Import internal modules
current_dir = os.path.dirname(os.path.abspath(__file__))
crystal_former_path = os.path.join(current_dir, "CrystalFormer")
sys.path.append(crystal_former_path)

from crystalformer.src.transformer import make_transformer
from crystalformer.src.wyckoff import mult_table, symops, symmetrize_atoms
from crystalformer.src.elements import element_dict, element_list

warnings.filterwarnings("ignore")


class CrystalGenerator(nn.Module): 
    def __init__(self, checkpoint_path, config_path, device=None):
        super().__init__() 
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"💎 Initializing CrystalGenerator on {self.device}...")

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.model = make_transformer(
            key=None,
            Nf=self.config["Nf"],
            Kx=self.config["Kx"],
            Kl=self.config["Kl"],
            n_max=self.config["n_max"],
            h0_size=self.config["h0_size"],
            num_layers=self.config["transformer_layers"],
            num_heads=self.config["num_heads"],
            key_size=self.config["key_size"],
            model_size=self.config["model_size"],
            embed_size=self.config["embed_size"],
            atom_types=self.config["atom_types"],
            wyck_types=self.config["wyck_types"],
            dropout_rate=0.0,
        ).to(self.device)
        
        # Load weights safely
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"   Loading weights from {os.path.basename(checkpoint_path)}...")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            if "model_state" in checkpoint:
                state_dict = checkpoint["model_state"]
            else:
                state_dict = checkpoint
                
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        
        self.model.eval()

        self.n_max = self.config["n_max"]
        self.atom_types = self.config["atom_types"]
        self.wyck_types = self.config["wyck_types"]
        self.Kx = self.config["Kx"]
        self.Kl = self.config["Kl"]

        self.mult_table = mult_table.to(self.device)
        self.symops = symops.to(self.device)

    # ---------------- EXISTING FUNCTIONS ----------------

    def _apply_element_mask(self, logits, allowed_elements):
        if allowed_elements is None:
            return logits
        masked = torch.full_like(logits, float("-inf"))
        masked[:, 0] = logits[:, 0]
        for z in allowed_elements:
            if z < logits.shape[-1]:
                masked[:, z] = logits[:, z]
        return masked

    def _apply_complexity_mask(self, logits, A_past, max_elements=5):
        """
        Dynamically masks out elements if the batch item has already 
        selected the maximum allowed number of unique elements.
        """
        B, num_classes = logits.shape
        if A_past.shape[1] == 0:
            return logits # First step, no restrictions yet

        mask = torch.ones_like(logits, dtype=torch.bool)

        for b in range(B):
            chosen = torch.unique(A_past[b])
            chosen = chosen[chosen != 0] # Exclude padding/empty token (0)

            if len(chosen) >= max_elements:
                mask[b, :] = False
                mask[b, 0] = True 
                mask[b, chosen] = True
        
        return logits.masked_fill(~mask, float('-inf'))

    # ---------------- PROJECTION & SAMPLING ----------------

    def _project_xyz(self, G, W, X, idx=0):
        ops = self.symops[G - 1, W, idx]
        ones = torch.ones((X.shape[0], 1), device=self.device)
        affine = torch.cat([X, ones], dim=1).unsqueeze(2)
        out = torch.bmm(ops, affine).squeeze(2)
        out -= torch.floor(out)
        return out

    def _sample_von_mises(self, loc, kappa, shape, temperature):
        loc = loc.detach().cpu()
        kappa = torch.clamp(kappa, 1e-6, 1000.0).detach().cpu()
        sigma = (1.0 / torch.sqrt(kappa)) * np.sqrt(temperature)
        samples = torch.normal(loc, sigma)
        samples = (samples + np.pi) % (2 * np.pi) - np.pi
        return ((samples + np.pi) / (2 * np.pi)).to(self.device)

    # ---------------- 🔹 NEW PURE PYTORCH LATTICE SYMMETRIZATION ----------------
    def _symmetrize_lattice_pt(self, sg, L):
        """
        Forces the 6 predicted lattice parameters to obey strict geometric
        Space Group rules, written purely in PyTorch to protect autograd graphs.
        """
        a, b, c, alpha, beta, gamma = L[:, 0], L[:, 1], L[:, 2], L[:, 3], L[:, 4], L[:, 5]
        ninety = torch.full_like(a, 90.0)
        one_twenty = torch.full_like(a, 120.0)

        # Build all possible symmetry states
        triclinic = L
        monoclinic = torch.stack([a, b, c, ninety, beta, ninety], dim=-1)
        orthorhombic = torch.stack([a, b, c, ninety, ninety, ninety], dim=-1)
        tetragonal = torch.stack([a, a, c, ninety, ninety, ninety], dim=-1)
        hexagonal = torch.stack([a, a, c, ninety, ninety, one_twenty], dim=-1)
        cubic = torch.stack([a, a, a, ninety, ninety, ninety], dim=-1)

        sg = sg.unsqueeze(-1)
        
        sym_L = torch.where(sg <= 2, triclinic,
                torch.where(sg <= 15, monoclinic,
                torch.where(sg <= 74, orthorhombic,
                torch.where(sg <= 142, tetragonal,
                torch.where(sg <= 194, hexagonal, cubic)))))
        return sym_L

    # ---------------- GENERATION ----------------

    def generate(self, num_samples, allowed_elements=None, temperature=0.5, G=None):
        with torch.no_grad():
            return self._run_generation(
                num_samples, temperature, allowed_elements, with_grads=False, G=G
            )

    def generate_with_grads(self, num_samples, allowed_elements=None, temperature=0.5, G=None):
        return self._run_generation(
            num_samples, temperature, allowed_elements, with_grads=True, G=G
        )

    def _run_generation(self, num_samples, temperature, allowed_elements, with_grads,G=None):
        B = num_samples

        if G is None: G = torch.randint(1, 231, (B,), device=self.device)
        W = torch.zeros((B, self.n_max), dtype=torch.long, device=self.device)
        A = torch.zeros((B, self.n_max), dtype=torch.long, device=self.device)
        X = torch.zeros((B, self.n_max), device=self.device)
        Y = torch.zeros((B, self.n_max), device=self.device)
        Z = torch.zeros((B, self.n_max), device=self.device)

        # 🔹 NEW: Accumulator for lattice GMM logits generated at every atomic step
        L_accum = torch.zeros((B, self.n_max, self.Kl + 2 * 6 * self.Kl), device=self.device)

        log_probs = torch.zeros(B, device=self.device)
        statuses = ["ok"] * B

        for i in range(self.n_max):
            XYZ = torch.stack([X, Y, Z], dim=-1)
            G_exp = (G - 1).unsqueeze(1).expand(-1, self.n_max)
            M = self.mult_table[G_exp, W]

            # Wyckoff
            out = self.model(G, XYZ.clone(), A.clone(), W.clone(), M, is_train=False)
            w_logit = out[:, 5 * i, : self.wyck_types]
            w_dist = torch.distributions.Categorical(logits=w_logit / temperature)
            w = w_dist.sample()
            if with_grads:
                log_probs += w_dist.log_prob(w)
            W[:, i] = w

            # Atom
            out = self.model(G, XYZ.clone(), A.clone(), W.clone(), M, is_train=False)
            h_al = out[:, 5 * i + 1]
            a_logit = h_al[:, : self.atom_types]
            
            a_logit = self._apply_element_mask(a_logit, allowed_elements)
            if i > 0:
                a_logit = self._apply_complexity_mask(a_logit, A[:, :i], max_elements=5)
                
            a_dist = torch.distributions.Categorical(logits=a_logit / temperature)
            a = a_dist.sample()
            if with_grads:
                log_probs += a_dist.log_prob(a)
            A[:, i] = a

            # 🔹 NEW: Capture the lattice GMM parameters outputted at this step
            L_accum[:, i] = h_al[:, self.atom_types : self.atom_types + self.Kl + 2 * 6 * self.Kl]

            # X
            out = self.model(G, XYZ.clone(), A.clone(), W.clone(), M, is_train=False)
            h = out[:, 5 * i + 2]
            x_logit, x_loc, x_kap = torch.split(h[:, : 3 * self.Kx], self.Kx, dim=-1)
            k_dist = torch.distributions.Categorical(logits=x_logit)
            k = k_dist.sample()
            if with_grads:
                log_probs += k_dist.log_prob(k)
                
            x_val = self._sample_von_mises(
                torch.gather(x_loc, 1, k[:, None]).squeeze(1),
                torch.gather(x_kap, 1, k[:, None]).squeeze(1),
                (B,),
                temperature,
            )
            X[:, i] = self._project_xyz(
                G, W[:, i],
                torch.stack([x_val, torch.zeros(B, device=self.device), torch.zeros(B, device=self.device)], dim=1),
                idx=0
            )[:, 0]

            # Y
            out = self.model(G, torch.stack([X, Y, Z], dim=-1), A.clone(), W.clone(), M, False)
            h = out[:, 5 * i + 3]
            y_logit, y_loc, y_kap = torch.split(h[:, : 3 * self.Kx], self.Kx, dim=-1)
            k_dist = torch.distributions.Categorical(logits=y_logit)
            k = k_dist.sample()
            if with_grads:
                log_probs += k_dist.log_prob(k)
                
            y_val = self._sample_von_mises(
                torch.gather(y_loc, 1, k[:, None]).squeeze(1),
                torch.gather(y_kap, 1, k[:, None]).squeeze(1),
                (B,),
                temperature,
            )
            Y[:, i] = self._project_xyz(
                G, W[:, i],
                torch.stack([X[:, i], y_val, torch.zeros(B, device=self.device)], dim=1),
                idx=0
            )[:, 1]

            # Z
            out = self.model(G, torch.stack([X, Y, Z], dim=-1), A.clone(), W.clone(), M, False)
            h = out[:, 5 * i + 4]
            z_logit, z_loc, z_kap = torch.split(h[:, : 3 * self.Kx], self.Kx, dim=-1)
            k_dist = torch.distributions.Categorical(logits=z_logit)
            k = k_dist.sample()
            if with_grads:
                log_probs += k_dist.log_prob(k)
                
            z_val = self._sample_von_mises(
                torch.gather(z_loc, 1, k[:, None]).squeeze(1),
                torch.gather(z_kap, 1, k[:, None]).squeeze(1),
                (B,),
                temperature,
            )
            Z[:, i] = self._project_xyz(
                G, W[:, i],
                torch.stack([X[:, i], Y[:, i], z_val], dim=1),
                idx=0
            )[:, 2]


        # ---------------- 🔹 NEW TRUE LATTICE SAMPLING ----------------
        G_exp = (G - 1).unsqueeze(1).expand(-1, self.n_max)
        M_final = self.mult_table[G_exp, W]
        num_atoms = M_final.sum(dim=1)
        
        # Find the index where generation stopped (the padding token)
        num_sites = (A != 0).sum(dim=1)
        # Ensure we don't go out of bounds if the max length is perfectly filled
        valid_indices = torch.clamp(num_sites, max=self.n_max - 1)
        
        # Extract the GMM parameters from the exact step where generation halted
        l_out = L_accum[torch.arange(B), valid_indices]
        l_logit, mu, sigma = torch.split(l_out, [self.Kl, self.Kl * 6, self.Kl * 6], dim=-1)
        
        k_dist = torch.distributions.Categorical(logits=l_logit / temperature)
        k = k_dist.sample()
        if with_grads:
            log_probs += k_dist.log_prob(k)
            
        mu = mu.view(B, self.Kl, 6)[torch.arange(B), k]
        sigma = sigma.view(B, self.Kl, 6)[torch.arange(B), k]
        
        l_dist = torch.distributions.Normal(loc=mu, scale=sigma * np.sqrt(temperature))
        L_sample = l_dist.sample()
        if with_grads:
            log_probs += l_dist.log_prob(L_sample).sum(dim=-1)

        # Scale Lengths and Angles
        length, angle = torch.split(L_sample, 3, dim=-1)
        length = length * (num_atoms.float().unsqueeze(1) ** (1.0 / 3.0))
        angle = angle * (180.0 / np.pi)
        L_val = torch.cat([length, angle], dim=-1)
        
        # Apply space group rules
        L_final = self._symmetrize_lattice_pt(G, L_val)
        
        # ---------------- STRUCTURE ASSEMBLY ----------------
        structures = []
        for b in range(B):
            try:
                mask = A[b] != 0
                species = A[b][mask].cpu().numpy()
                wyck = W[b][mask].cpu().numpy()
                coords = torch.stack([X[b], Y[b], Z[b]], dim=-1)[mask]

                final_species, final_coords = [], []
                for sp, wy, c in zip(species, wyck, coords):
                    orbit = symmetrize_atoms(G[b].item(), wy, c)
                    for oc in orbit:
                        final_species.append(element_list[sp])
                        final_coords.append(oc.cpu().numpy())

                lattice = Lattice.from_parameters(*L_final[b].detach().cpu().numpy())
                structures.append(Structure(lattice, final_species, final_coords))
            except Exception:
                structures.append(None)
                statuses[b] = "assembly_fail"

        return {
            "structures": structures,
            "log_probs": log_probs,
            "statuses": statuses,
            "G": G, # Return these for log probability calculations later
            "L": L_final,
            "XYZ": torch.stack([X, Y, Z], dim=-1),
            "A": A,
            "W": W
        }