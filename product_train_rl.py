import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import yaml
import sys
import warnings
from tqdm import tqdm

# --- FIX PATHS ---
# Get the absolute path of the directory containing this script (NOVAGEN/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Add 'CrystalFormer' folder to path so we can import 'crystalformer'
sys.path.append(os.path.join(BASE_DIR, "CrystalFormer"))

# 2. Add 'NOVAGEN' itself to path
sys.path.append(BASE_DIR)

# --- NOW IMPORT ---
try:
    from product_oracle import CrystalOracle
    from product_relaxer import CrystalRelaxer
    from product_reward_engine import RewardEngine
    from sentinel import CrystalSentinel
    
    # Import CrystalFormer Internals
    from crystalformer.src.transformer import make_transformer
    from crystalformer.src.lattice import symmetrize_lattice
    from crystalformer.src.wyckoff import mult_table, symops, symmetrize_atoms
    from crystalformer.src.elements import element_list
    from pymatgen.core import Structure, Lattice
except ImportError as e:
    print(f"❌ IMPORT ERROR: {e}")
    sys.exit(1)

warnings.filterwarnings("ignore")

# --- CONFIGURATION (CORRECTED) ---
# We use os.path.join to find files regardless of where you run the script from
CHECKPOINT_PATH = os.path.join(BASE_DIR, "pretrained_model", "epoch_005500_CLEAN.pt")
CONFIG_PATH = os.path.join(BASE_DIR, "pretrained_model", "config.yaml")
SAVE_DIR = os.path.join(BASE_DIR, "rl_checkpoints")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# RL HYPERPARAMETERS
BATCH_SIZE = 16          
LR = 1e-5                
EPOCHS = 100             
VALIDATION_FREQ = 10     
ENTROPY_COEF = 0.01      
CAMPAIGN_ELEMENTS = [26, 8, 16] # Fe, O, S

class RLTrainer:
    def __init__(self):
        print(f"🚀 Initializing RL Gym on {DEVICE}...")
        os.makedirs(SAVE_DIR, exist_ok=True)
        
        # 1. Load Config
        print(f"   📂 Loading Config from: {CONFIG_PATH}")
        if not os.path.exists(CONFIG_PATH):
            raise FileNotFoundError(f"Config not found at {CONFIG_PATH}")
            
        with open(CONFIG_PATH, 'r') as file:
            self.config = yaml.safe_load(file)

        # 2. Initialize Model (THE STUDENT)
        self.model = make_transformer(
            key=None, Nf=self.config['Nf'], Kx=self.config['Kx'], Kl=self.config['Kl'], n_max=self.config['n_max'],
            h0_size=self.config['h0_size'], num_layers=self.config['transformer_layers'], num_heads=self.config['num_heads'],
            key_size=self.config['key_size'], model_size=self.config['model_size'], embed_size=self.config['embed_size'],
            atom_types=self.config['atom_types'], wyck_types=self.config['wyck_types'], dropout_rate=0.1
        ).to(DEVICE)

        print(f"   💎 Loading weights from {os.path.basename(CHECKPOINT_PATH)}...")
        if not os.path.exists(CHECKPOINT_PATH):
             raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}")

        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        state_dict = checkpoint.get('policy_state', checkpoint.get('model_state_dict', checkpoint))
        self.model.load_state_dict(state_dict)
        
        # KEY: Enable Gradients for Training
        self.model.train()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)

        # 3. Initialize Teachers & Validators
        print("   🔧 Initializing Teachers...")
        self.oracle = CrystalOracle(device="cpu")   
        self.sentinel = CrystalSentinel() 
        self.relaxer = CrystalRelaxer(device="cpu") 
        self.reward_engine = RewardEngine()         

        # 4. Cache Physics Constants
        self.n_max = self.config['n_max']
        self.atom_types = self.config['atom_types']
        self.wyck_types = self.config['wyck_types']
        self.mult_table = mult_table.to(DEVICE)
        self.symops = symops.to(DEVICE)
        self.Kl = self.config['Kl']
        self.Kx = self.config['Kx']

    def _apply_mask(self, logits, allowed):
        if allowed is None: return logits
        mask = torch.zeros(logits.shape[-1], device=DEVICE)
        mask[0] = 1.0 
        for z in allowed:
            if z < len(mask): mask[z] = 1.0
        return torch.where(mask.bool(), logits, torch.tensor(-1e9, device=DEVICE))

    def rollout(self, batch_size):
        # Initialize Tensors
        G = torch.randint(1, 231, (batch_size,), device=DEVICE) 
        W = torch.zeros((batch_size, self.n_max), dtype=torch.long, device=DEVICE)
        A = torch.zeros((batch_size, self.n_max), dtype=torch.long, device=DEVICE)
        X = torch.zeros((batch_size, self.n_max), device=DEVICE)
        Y = torch.zeros((batch_size, self.n_max), device=DEVICE)
        Z = torch.zeros((batch_size, self.n_max), device=DEVICE)
        
        log_probs = []
        entropy_loss = 0.0

        # --- GENERATION LOOP ---
        for i in range(self.n_max):
            XYZ = torch.stack([X, Y, Z], dim=-1)
            G_exp = (G - 1).unsqueeze(1).expand(-1, self.n_max)
            M = self.mult_table[G_exp, W]

            # 1. Forward Pass
            output = self.model(G, XYZ, A, W, M, is_train=False)

            # 2. Sample Wyckoff Position
            w_logit = output[:, 5 * i, :self.wyck_types]
            w_dist = torch.distributions.Categorical(logits=w_logit)
            w_action = w_dist.sample()
            W[:, i] = w_action
            
            log_probs.append(w_dist.log_prob(w_action))
            entropy_loss += w_dist.entropy().mean()

            # 3. Sample Atom Type
            output = self.model(G, XYZ, A, W, M, is_train=False) 
            a_logit = output[:, 5 * i + 1, :self.atom_types]
            a_logit = self._apply_mask(a_logit, CAMPAIGN_ELEMENTS)
            a_dist = torch.distributions.Categorical(logits=a_logit)
            a_action = a_dist.sample()
            A[:, i] = a_action
            
            log_probs.append(a_dist.log_prob(a_action))
            entropy_loss += a_dist.entropy().mean()

            # 4. Coords (Simplified greedy)
            h_x = output[:, 5 * i + 2]
            x_logit, _, _ = torch.split(h_x[:, :3*self.Kx], [self.Kx, self.Kx, self.Kx], dim=-1)
            
        # --- LATTICE RECONSTRUCTION ---
        L_preds = output[:, 5 * i + 1, self.atom_types : self.atom_types + self.Kl + 12 * self.Kl]
        l_logit, _, _ = torch.split(L_preds, [self.Kl, 6*self.Kl, 6*self.Kl], dim=-1)
        k_l = torch.argmax(l_logit, dim=1) 
        
        raw_structs = self._reconstruct_structures(G, A, W, X, Y, Z, k_l)
        total_log_prob = torch.stack(log_probs).sum(dim=0)
        
        return total_log_prob, entropy_loss, raw_structs

    def _reconstruct_structures(self, G, A, W, X, Y, Z, L_indices):
        structures = []
        batch_size = G.shape[0]
        # Robust dummy lattice to prevent geometric crashes during pure composition learning
        dummy_lattice = Lattice.from_parameters(5, 5, 5, 90, 90, 90)

        for b in range(batch_size):
            try:
                valid_mask = A[b] != 0
                species = [element_list[s] for s in A[b][valid_mask].cpu().numpy()]
                coords = np.random.rand(len(species), 3) 
                
                if len(species) > 0:
                    struct = Structure(dummy_lattice, species, coords)
                    structures.append(struct)
                else:
                    structures.append(None)
            except:
                structures.append(None)
        return structures

    def train(self):
        print(f"⚡ Starting REINFORCE Training for {EPOCHS} epochs...")
        
        for epoch in range(1, EPOCHS + 1):
            self.optimizer.zero_grad()
            
            # A. ROLLOUT 
            log_probs, entropy, raw_structs = self.rollout(BATCH_SIZE)
            
            # B. SCORE (Oracle)
            validity_mask, valid_structs = self.sentinel.filter(raw_structs)
            
            if not valid_structs:
                print(f"   [Epoch {epoch}] 0 survivors. Skipping.")
                continue
                
            e_form_preds = self.oracle.predict_formation_energy(valid_structs)
            bg_preds = self.oracle.predict_band_gap(valid_structs)
            
            # Calculate Rewards
            valid_rewards, stats = self.reward_engine.compute_reward(
                [True]*len(valid_structs), e_form_preds, bg_preds
            )
            
            # C. LOSS & UPDATE
            baseline = valid_rewards.mean() if len(valid_rewards) > 0 else 0.0
            
            # Heuristic loss calculation
            loss = -(log_probs.mean() * (baseline - 0.0)) - (ENTROPY_COEF * entropy)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) 
            self.optimizer.step()
            
            print(f"[Epoch {epoch}] Reward: {baseline:.2f} | Valid: {stats['valid_rate']:.0%} | Loss: {loss.item():.2f}")

            # D. VALIDATION 
            if epoch % VALIDATION_FREQ == 0:
                print(f"\n🔍 [Epoch {epoch}] VALIDATION (Relaxing best candidates)...")
                for s in valid_structs[:2]:
                    res = self.relaxer.relax(s)
                    status = "✅ Stable" if res['converged'] and res.get('energy_per_atom', 10) < 0 else "❌ Unstable"
                    print(f"   {status}: E={res.get('energy_per_atom',0):.3f} eV")
                print("")
                
            if epoch % 50 == 0:
                 torch.save(self.model.state_dict(), os.path.join(SAVE_DIR, f"epoch_{epoch:03d}_RL.pt"))

if __name__ == "__main__":
    trainer = RLTrainer()
    trainer.train()
