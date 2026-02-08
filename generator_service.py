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
from crystalformer.src.lattice import symmetrize_lattice
from crystalformer.src.wyckoff import mult_table, symops, symmetrize_atoms
from crystalformer.src.elements import element_dict, element_list

warnings.filterwarnings("ignore")


class CrystalGenerator(nn.Module): # [CHANGE] Inherit from nn.Module to register parameters
    def __init__(self, checkpoint_path, config_path, device=None):
        super().__init__() # [CHANGE] Initialize parent
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

        # [CRITICAL FIX] Add a LEARNABLE parameter for lattice scaling
        # We initialize it to 3.0 (approx 20 A^3) to start in the "Safe Zone"
        self.lattice_bias = nn.Parameter(torch.tensor(2.2, device=self.device))
        
        # Load weights safely
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"   Loading weights from {os.path.basename(checkpoint_path)}...")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Handle nested checkpoints (from training) vs raw weights
            if "model_state" in checkpoint:
                state_dict = checkpoint["model_state"]
            else:
                state_dict = checkpoint
                
            # Load Transformer weights (ignoring the new lattice_bias if missing)
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            
            # If loading a fine-tuned model, try to load the lattice bias too
            if "lattice_bias" in state_dict:
                self.lattice_bias.data = state_dict["lattice_bias"]
                print("   ✅ Loaded learned lattice bias.")
            else:
                print("   ⚠️  New Lattice Head initialized to default (3.0).")
        
        self.model.eval()

        self.n_max = self.config["n_max"]
        self.atom_types = self.config["atom_types"]
        self.wyck_types = self.config["wyck_types"]
        self.Kx = self.config["Kx"]

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

    # ---------------- FIXED: LATTICE SCALE SAMPLER ----------------

    def _sample_lattice_scale(self, B, temperature, with_grads):
        """
        Learnable lattice size using self.lattice_bias
        """
        # [FIX] Use the learnable parameter as the mean
        mu = self.lattice_bias 
        
        dist = torch.distributions.Normal(
            loc=mu, 
            scale=0.15 * temperature,
        )
        
        # Reparameterization trick happens automatically here if mu requires_grad
        log_s = dist.sample((B,))
        s = torch.exp(log_s)

        if with_grads:
            log_prob = dist.log_prob(log_s)
        else:
            log_prob = torch.zeros(B, device=self.device)

        return s, log_prob

    # ---------------- GENERATION ----------------

    def generate(self, num_samples, allowed_elements=None, temperature=0.5):
        with torch.no_grad():
            return self._run_generation(
                num_samples, temperature, allowed_elements, with_grads=False
            )

    def generate_with_grads(self, num_samples, allowed_elements=None, temperature=0.5):
        return self._run_generation(
            num_samples, temperature, allowed_elements, with_grads=True
        )

    def _run_generation(self, num_samples, temperature, allowed_elements, with_grads):
        B = num_samples

        G = torch.randint(1, 231, (B,), device=self.device)
        W = torch.zeros((B, self.n_max), dtype=torch.long, device=self.device)
        A = torch.zeros((B, self.n_max), dtype=torch.long, device=self.device)
        X = torch.zeros((B, self.n_max), device=self.device)
        Y = torch.zeros((B, self.n_max), device=self.device)
        Z = torch.zeros((B, self.n_max), device=self.device)

        log_probs = torch.zeros(B, device=self.device)
        statuses = ["ok"] * B

        # Lattice Scale
        lattice_scale, lp = self._sample_lattice_scale(B, temperature, with_grads)
        log_probs += lp

        # ... (Rest of generation logic remains the same) ...
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
            a_logit = out[:, 5 * i + 1, : self.atom_types]
            a_logit = self._apply_element_mask(a_logit, allowed_elements)
            a_dist = torch.distributions.Categorical(logits=a_logit / temperature)
            a = a_dist.sample()
            if with_grads:
                log_probs += a_dist.log_prob(a)
            A[:, i] = a

            # X
            out = self.model(G, XYZ.clone(), A.clone(), W.clone(), M, is_train=False)
            h = out[:, 5 * i + 2]
            x_logit, x_loc, x_kap = torch.split(h[:, : 3 * self.Kx], self.Kx, dim=-1)
            k = torch.distributions.Categorical(logits=x_logit).sample()
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
            k = torch.distributions.Categorical(logits=y_logit).sample()
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
            k = torch.distributions.Categorical(logits=z_logit).sample()
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

                lat_params = torch.tensor(
                    [lattice_scale[b], lattice_scale[b], lattice_scale[b], 90.0, 90.0, 90.0],
                    device=self.device,
                )
                lat = symmetrize_lattice(G[b], lat_params)
                lattice = Lattice.from_parameters(*lat.cpu().numpy())

                structures.append(Structure(lattice, final_species, final_coords))
            except Exception:
                structures.append(None)
                statuses[b] = "assembly_fail"

        return {
            "structures": structures,
            "log_probs": log_probs,
            "statuses": statuses,
        }
