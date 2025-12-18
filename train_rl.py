import os
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import csv
import yaml
import pandas as pd
from collections import deque
from tqdm import tqdm  # Clean progress bars

# --- PATH SETUP ---
ROOT = os.path.dirname(os.path.abspath("__file__"))
sys.path.append(os.path.join(ROOT, "CrystalFormer"))

# --- IMPORTS ---
from crystalformer.src.transformer import make_transformer
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter

# Local Modules
from relaxer import Relaxer
from oracle import Oracle

# --- CONFIGURATION (UPDATED) ---
PRETRAINED_DIR = os.path.join(ROOT, "pretrained_model")
CONFIG = {
    "BATCH_SIZE": 128,      # Increased for GPU efficiency
    "LR": 1e-6,             # Low learning rate for stability
    "EPOCHS": 50,           # Set to 2000 for full discovery run
    "KL_COEF": 0.05,        # Penalty for drifting too far from "Professor"
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "REPLAY_RATIO": 0.2     # 20% Memory (Safety), 80% Exploration (Discovery)
}

class PPOAgent_Online:
    def __init__(self):
        print("--- Initializing PPO Agent (Student-Professor Mode) ---")
        self.device = CONFIG["DEVICE"]
        
        # 1. Load Config
        with open(os.path.join(PRETRAINED_DIR, "config.yaml"), "r") as f:
            cfg = yaml.safe_load(f)

        # 2. Create Models
        # Student (Policy): The active learner
        self.policy = make_transformer(
            key=None, Nf=cfg['Nf'], Kx=cfg['Kx'], Kl=cfg['Kl'], n_max=cfg['n_max'],
            h0_size=cfg['h0_size'], num_layers=cfg['transformer_layers'], num_heads=cfg['num_heads'],
            key_size=cfg['key_size'], model_size=cfg['model_size'], embed_size=cfg['embed_size'],
            atom_types=cfg['atom_types'], wyck_types=cfg['wyck_types'], dropout_rate=0.0, widening_factor=4
        ).to(self.device)

        # Professor (Reference): The frozen expert
        self.ref_model = make_transformer(
            key=None, Nf=cfg['Nf'], Kx=cfg['Kx'], Kl=cfg['Kl'], n_max=cfg['n_max'],
            h0_size=cfg['h0_size'], num_layers=cfg['transformer_layers'], num_heads=cfg['num_heads'],
            key_size=cfg['key_size'], model_size=cfg['model_size'], embed_size=cfg['embed_size'],
            atom_types=cfg['atom_types'], wyck_types=cfg['wyck_types'], dropout_rate=0.0, widening_factor=4
        ).to(self.device)
        
        # --- 3. HYBRID LOADING ---
        author_model_path = os.path.join(PRETRAINED_DIR, "epoch_005500_CLEAN.pt")
        my_model_path = os.path.join(ROOT, "best_rl_model.pt")

        # A. Load STUDENT (Policy)
        if os.path.exists(my_model_path):
            print(f"🧠 Loading YOUR Smart Agent: {os.path.basename(my_model_path)}")
            self.policy.load_state_dict(torch.load(my_model_path, map_location=self.device), strict=True)
        else:
            print("⚠️ No previous RL model found. Starting fresh from Author's weights.")
            self.policy.load_state_dict(torch.load(author_model_path, map_location=self.device), strict=True)

        # B. Load PROFESSOR (Reference) - ALWAYS Author's weights
        print(f"📚 Loading Reference Anchor: {os.path.basename(author_model_path)}")
        self.ref_model.load_state_dict(torch.load(author_model_path, map_location=self.device), strict=True)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # 4. Optimizer & Memory
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=CONFIG["LR"])
        self.memory = deque(maxlen=2000) # Rolling memory of best 2000 crystals
        
        # --- UPGRADE: EXPANDED ELEMENT LIST ---
        # Added: Cu (29), In (49), Sn (50), Ge (32), Te (52), Si (14)
        # Targeted Materials: CIGS, CZTS, CdTe, SiGe
        self.idx_to_atom = {
            0: 30, 1: 16, 2: 48, 3: 34, 4: 8, 5: 31, 6: 33,  # Original (Zn, S, Cd, Se, O, Ga, As)
            7: 29, 8: 49, 9: 50, 10: 32, 11: 52, 12: 14      # New Additions
        }

    def prepare_input(self, G, XYZ, A, W, M):
        return (
            torch.tensor([G], device=self.device),
            torch.tensor([XYZ], device=self.device),
            torch.tensor([A], device=self.device),
            torch.tensor([W], device=self.device),
            torch.tensor([M], device=self.device)
        )

# --- REWARD ENGINE ---
class RewardEngine:
    def compute_reward(self, relax_result, oracle_props):
        if not relax_result.get('is_converged', False):
            return -5.0, "Fail"
        
        e_form = oracle_props.get('formation_energy', 0.0)
        gap = oracle_props.get('band_gap_scalar', 0.0)

        # Reward Logic: Boost stable crystals with 1.0 - 2.5 eV bandgap
        r_stability = (10 / (1 + np.exp(2 * e_form))) - 5
        r_gap = min(gap * 5.0, 10.0)
        
        return r_stability + r_gap, f"G:{gap:.2f}|E:{e_form:.2f}"

def build_structure(A, X, lattice_scale):
    """
    Builds structure with variable initial lattice size (Randomized Guess).
    """
    species = [a for a in A if a != 0]
    coords = [x for a, x in zip(A, X) if a != 0]
    if len(species) < 1: return None
    try:
        # UPGRADE: Randomized Lattice Guess (Avoids "Everything is 5x5x5")
        # Gives the relaxer a better chance to find the true shape
        a = b = c = lattice_scale
        lattice = Lattice.from_parameters(a, b, c, 90, 90, 90)
        return Structure(lattice, species, coords)
    except: return None

# --- MAIN LOOP ---
def main():
    agent = PPOAgent_Online()
    relaxer = Relaxer()
    oracle = Oracle()
    reward_engine = RewardEngine()
    
    disc_dir = os.path.join(ROOT, "rl_discoveries")
    os.makedirs(disc_dir, exist_ok=True)
    
    log_file = open("training_log.csv", "a", newline="") # Append mode
    writer = csv.writer(log_file)
    # Check if empty before writing header
    if os.path.getsize("training_log.csv") == 0:
        writer.writerow(["Epoch", "Reward", "Stable_Count", "Top_Formula"])

    best_avg_reward = -10.0
    
    # Track top candidates for final report
    top_candidates = [] # List of dicts

    print(f"\n🚀 STARTING TRAINING: {CONFIG['EPOCHS']} Epochs | Batch: {CONFIG['BATCH_SIZE']}")
    
    # TQDM Progress Bar for Epochs
    pbar = tqdm(range(CONFIG["EPOCHS"]), desc="Training Progress", unit="epoch")

    for epoch in pbar:
        epoch_rewards = []
        stable_count = 0
        best_formula_epoch = "None"
        
        for batch_i in range(CONFIG["BATCH_SIZE"]):
            
            # --- MEMORY REPLAY LOGIC ---
            use_memory = (len(agent.memory) > 10) and (random.random() < CONFIG["REPLAY_RATIO"])
            
            if use_memory:
                # Replay a past success (Teach the Student to remember)
                struct = random.choice(agent.memory)
                reward = 5.0 
                loss = torch.tensor(0.0, requires_grad=True)
            else:
                # --- EXPLORATION ---
                G_raw = random.randint(1, 230)
                num_atoms = random.randint(2, 8) # Variable atom count
                
                # Randomized Lattice Initial Guess (3.0 to 7.0 Angstroms)
                lattice_guess = random.uniform(3.0, 7.0)
                
                inputs = agent.prepare_input(G_raw, [[0.5]*3]*num_atoms, [0]*num_atoms, [0]*num_atoms, [1]*num_atoms)
                
                # Forward Pass
                logits_policy = agent.policy(*inputs, is_train=False).squeeze(0)
                with torch.no_grad():
                    logits_ref = agent.ref_model(*inputs, is_train=False).squeeze(0)
                
                # Action Sampling
                log_probs_list = []
                actions_list = []
                for j in range(num_atoms):
                    # Only sample from expanded element list (indices 0-12)
                    valid_logits = logits_policy[j][:13] 
                    dist = torch.distributions.Categorical(logits=valid_logits)
                    action = dist.sample()
                    log_probs_list.append(dist.log_prob(action))
                    actions_list.append(agent.idx_to_atom.get(action.item(), 6))
                
                # Coordinate Decoding
                X_list = []
                c_start = num_atoms + 1
                for j in range(num_atoms):
                    idx_base = c_start + (j * 3)
                    if idx_base + 2 < len(logits_policy):
                        x = torch.sigmoid(logits_policy[idx_base][0]).item()
                        y = torch.sigmoid(logits_policy[idx_base + 1][0]).item()
                        z = torch.sigmoid(logits_policy[idx_base + 2][0]).item()
                        X_list.append([x, y, z])
                    else:
                        X_list.append([random.random(), random.random(), random.random()])

                # Build & Relax (With Randomized Lattice)
                struct = build_structure(actions_list, X_list, lattice_guess)
                
                reward = -5.0
                
                if struct:
                    relax_res = relaxer.relax(struct)
                    
                    if relax_res["is_converged"]:
                        # Property Prediction
                        final_s = relax_res["final_structure"]
                        p_list = oracle.predict_properties([final_s])
                        props = p_list[0] if p_list else {}
                        
                        reward, info = reward_engine.compute_reward(relax_res, props)
                        
                        # SAVE DISCOVERIES
                        if reward > 0.0:
                            stable_count += 1
                            agent.memory.append(final_s)
                            
                            formula = final_s.composition.reduced_formula
                            best_formula_epoch = formula
                            filename = f"{disc_dir}/{formula}_Ep{epoch}_B{batch_i}.cif"
                            CifWriter(final_s).write_file(filename)
                            
                            # Add to Top Candidates List
                            top_candidates.append({
                                "Formula": formula,
                                "Formation_Energy": props.get('formation_energy', 0),
                                "Band_Gap": props.get('band_gap_scalar', 0),
                                "Reward": reward,
                                "Epoch": epoch
                            })

                # Loss Calculation (PPO-ish)
                total_log_prob = torch.stack(log_probs_list).sum()
                kl_div = F.mse_loss(logits_policy, logits_ref.detach())
                pg_loss = -(reward * total_log_prob)
                loss = pg_loss + (CONFIG["KL_COEF"] * kl_div)

            # Optimization Step
            agent.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), 1.0)
            agent.optimizer.step()
            
            epoch_rewards.append(reward)

        # End of Epoch Stats
        avg_r = np.mean(epoch_rewards)
        
        # Update Progress Bar Description
        pbar.set_postfix({"Avg Reward": f"{avg_r:.2f}", "Stable": stable_count})
        
        writer.writerow([epoch, avg_r, stable_count, best_formula_epoch])
        
        # Save Best Model Logic
        if avg_r > best_avg_reward:
            best_avg_reward = avg_r
            torch.save(agent.policy.state_dict(), os.path.join(ROOT, "best_rl_model.pt"))

    # --- FINAL REPORTING ---
    print("\n" + "="*60)
    print("🏆 TOP 10 MATERIAL CANDIDATES (Ranked by Reward)")
    print("="*60)
    
    if top_candidates:
        df = pd.DataFrame(top_candidates)
        # Deduplicate by Formula, keeping the one with lowest energy
        df = df.sort_values("Formation_Energy", ascending=True).drop_duplicates("Formula")
        # Sort by best mix of properties (Reward)
        df = df.sort_values("Reward", ascending=False).head(10)
        
        print(df[["Formula", "Formation_Energy", "Band_Gap", "Reward"]].to_string(index=False))
        
        # Save Report
        df.to_csv("final_candidates.csv", index=False)
        print("\n✅ Full candidate list saved to 'final_candidates.csv'")
    else:
        print("No stable candidates found in this run.")

if __name__ == "__main__":
    main()
