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
from tqdm import tqdm

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

# --- CONFIGURATION ---
PRETRAINED_DIR = os.path.join(ROOT, "pretrained_model")
CONFIG = {
    "BATCH_SIZE": 64,       
    "LR": 1e-5,             
    "EPOCHS": 300,          # Phase 2 Duration
    "KL_COEF": 0.05,        
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "REPLAY_RATIO": 0.2     
}

class PPOAgent_Online:
    def __init__(self):
        print("--- Initializing Robust PPO Agent (Anti-Cheat Active) ---")
        self.device = CONFIG["DEVICE"]
        
        # 1. Load Config
        with open(os.path.join(PRETRAINED_DIR, "config.yaml"), "r") as f:
            cfg = yaml.safe_load(f)

        # 2. Create Models
        self.policy = make_transformer(
            key=None, Nf=cfg['Nf'], Kx=cfg['Kx'], Kl=cfg['Kl'], n_max=cfg['n_max'],
            h0_size=cfg['h0_size'], num_layers=cfg['transformer_layers'], num_heads=cfg['num_heads'],
            key_size=cfg['key_size'], model_size=cfg['model_size'], embed_size=cfg['embed_size'],
            atom_types=cfg['atom_types'], wyck_types=cfg['wyck_types'], dropout_rate=0.0, widening_factor=4
        ).to(self.device)

        self.ref_model = make_transformer(
            key=None, Nf=cfg['Nf'], Kx=cfg['Kx'], Kl=cfg['Kl'], n_max=cfg['n_max'],
            h0_size=cfg['h0_size'], num_layers=cfg['transformer_layers'], num_heads=cfg['num_heads'],
            key_size=cfg['key_size'], model_size=cfg['model_size'], embed_size=cfg['embed_size'],
            atom_types=cfg['atom_types'], wyck_types=cfg['wyck_types'], dropout_rate=0.0, widening_factor=4
        ).to(self.device)
        
        # --- 3. HYBRID LOADING ---
        author_model_path = os.path.join(PRETRAINED_DIR, "epoch_005500_CLEAN.pt")
        my_model_path = os.path.join(ROOT, "best_rl_model.pt")

        # Load Student (Policy)
        if os.path.exists(my_model_path):
            print(f"🧠 Loading YOUR Smart Agent: {os.path.basename(my_model_path)}")
            self.policy.load_state_dict(torch.load(my_model_path, map_location=self.device), strict=True)
        else:
            print("⚠️ No previous RL model found. Starting fresh from Author's weights.")
            self.policy.load_state_dict(torch.load(author_model_path, map_location=self.device), strict=True)

        # Load Professor (Reference)
        print(f"📚 Loading Reference Anchor: {os.path.basename(author_model_path)}")
        self.ref_model.load_state_dict(torch.load(author_model_path, map_location=self.device), strict=True)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # 4. Optimizer & Memory
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=CONFIG["LR"])
        self.memory = deque(maxlen=2000) 
        
        # --- EXPANDED ELEMENT LIST ---
        self.idx_to_atom = {
            0: 30, 1: 16, 2: 48, 3: 34, 4: 8, 5: 31, 6: 33,  # Zn, S, Cd, Se, O, Ga, As
            7: 29, 8: 49, 9: 50, 10: 32, 11: 52, 12: 14      # Cu, In, Sn, Ge, Te, Si
        }

    def prepare_input(self, G, XYZ, A, W, M):
        return (
            torch.tensor([G], device=self.device),
            torch.tensor([XYZ], device=self.device),
            torch.tensor([A], device=self.device),
            torch.tensor([W], device=self.device),
            torch.tensor([M], device=self.device)
        )

# --- ROBUST REWARD ENGINE ---
class RewardEngine:
    def check_geometry_validity(self, struct):
        """
        ANTI-CHEAT: Returns False if atoms are overlapping (< 0.8 Angstrom).
        """
        try:
            dm = struct.distance_matrix
            np.fill_diagonal(dm, 10.0)
            min_dist = np.min(dm)
            
            if min_dist < 0.8: 
                return False
            return True
        except:
            return False

    def compute_reward(self, relax_result, oracle_props):
        if not relax_result.get('is_converged', False):
            return -5.0, "Fail_Conv"
            
        struct = relax_result["final_structure"]

        # Double Check Geometry (Post-Relaxation)
        if not self.check_geometry_validity(struct):
            return -10.0, "Glitch_Collapse"
        
        e_form = oracle_props.get('formation_energy', 0.0)
        gap = oracle_props.get('band_gap_scalar', 0.0)

        # Sanity Check
        if e_form < -50.0:
            return -10.0, "Physics_Err"

        # Reward Logic
        r_stability = (10 / (1 + np.exp(2 * e_form))) - 5
        r_gap = min(gap * 5.0, 10.0)
        
        return r_stability + r_gap, f"G:{gap:.2f}|E:{e_form:.2f}"

def build_structure(A, X, lattice_scale):
    species = [a for a in A if a != 0]
    coords = [x for a, x in zip(A, X) if a != 0]
    if len(species) < 1: return None
    try:
        a = b = c = lattice_scale
        lattice = Lattice.from_parameters(a, b, c, 90, 90, 90)
        return Structure(lattice, species, coords)
    except: return None

def summarize_results(cand_file="final_candidates.csv"):
    print("\n" + "="*80)
    print("🏁 SEMICONDUCTOR DISCOVERY SUMMARY")
    print("="*80)

    if not os.path.exists(cand_file):
        print("❌ No candidates file found.")
        return

    df = pd.read_csv(cand_file)

    if df.empty:
        print("❌ No stable materials were discovered.")
        return

    # Basic Counts
    total_entries = len(df)
    total_unique = df["Formula"].nunique()

    print(f"📦 Total Stable Structures Found    : {total_entries}")
    print(f"🧪 Unique Material Formulas         : {total_unique}")

    # Semiconductor Filters
    df = df[
        (df["Band_Gap"] > 0.1) &          # Not metallic
        (df["Band_Gap"] < 3.5) &          # Not deep insulator
        (df["Formation_Energy"] < 0.5)    # Reasonably stable
    ]

    if df.empty:
        print("❌ No materials passed semiconductor filters.")
        return

    # Deduplicate by formula (keep lowest energy)
    df = df.sort_values("Formation_Energy", ascending=True)
    df = df.drop_duplicates("Formula")

    # Semiconductor Scoring
    df["Stability_Score"] = -df["Formation_Energy"]
    df["BandGap_Score"] = np.exp(-((df["Band_Gap"] - 1.5) ** 2) / (2 * 0.5 ** 2))
    df["Semiconductor_Score"] = df["Stability_Score"] + 2.0 * df["BandGap_Score"]

    # Top 10
    top10 = df.sort_values("Semiconductor_Score", ascending=False).head(10)

    print("\n🏆 TOP 10 INORGANIC SEMICONDUCTORS")
    print("-" * 80)
    print(top10[["Formula", "Formation_Energy", "Band_Gap", "Semiconductor_Score"]].to_string(index=False))

# --- MAIN LOOP ---
def main():
    agent = PPOAgent_Online()
    relaxer = Relaxer()
    oracle = Oracle()
    reward_engine = RewardEngine()
    
    disc_dir = os.path.join(ROOT, "rl_discoveries")
    os.makedirs(disc_dir, exist_ok=True)
    
    # 1. Initialize Log File
    with open("training_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Reward", "Stable_Count", "Top_Formula"])

    # 2. Initialize Candidates File
    cand_file = "final_candidates.csv"
    with open(cand_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Formula", "Formation_Energy", "Band_Gap", "Reward", "Epoch"])

    best_avg_reward = -10.0
    
    print(f"\n🚀 STARTING TRAINING: {CONFIG['EPOCHS']} Epochs | Batch: {CONFIG['BATCH_SIZE']}")
    pbar = tqdm(range(CONFIG["EPOCHS"]), desc="Training Progress", unit="epoch")

    for epoch in pbar:
        epoch_rewards = []
        stable_count = 0
        best_formula_epoch = "None"
        
        for batch_i in range(CONFIG["BATCH_SIZE"]):
            
            # Memory Replay
            use_memory = (len(agent.memory) > 10) and (random.random() < CONFIG["REPLAY_RATIO"])
            
            if use_memory:
                struct = random.choice(agent.memory)
                reward = 5.0 
                loss = torch.tensor(0.0, requires_grad=True)
            else:
                # Exploration
                G_raw = random.randint(1, 230)
                num_atoms = random.randint(2, 6)
                lattice_guess = random.uniform(3.5, 6.0)
                
                inputs = agent.prepare_input(G_raw, [[0.5]*3]*num_atoms, [0]*num_atoms, [0]*num_atoms, [1]*num_atoms)
                
                logits_policy = agent.policy(*inputs, is_train=False).squeeze(0)
                with torch.no_grad():
                    logits_ref = agent.ref_model(*inputs, is_train=False).squeeze(0)
                
                log_probs_list = []
                actions_list = []
                X_list = []
                
                # Atoms
                for j in range(num_atoms):
                    valid_logits = logits_policy[j][:13] 
                    dist = torch.distributions.Categorical(logits=valid_logits)
                    action = dist.sample()
                    log_probs_list.append(dist.log_prob(action))
                    actions_list.append(agent.idx_to_atom.get(action.item(), 6))
                
                # Coords
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

                # Build
                struct = build_structure(actions_list, X_list, lattice_guess)
                reward = -5.0
                
                if struct:
                    # Check Geometry (Fast Fail)
                    if reward_engine.check_geometry_validity(struct):
                        relax_res = relaxer.relax(struct)
                        
                        if relax_res["is_converged"]:
                            final_s = relax_res["final_structure"]
                            
                            # Check Geometry (Post-Relax)
                            if reward_engine.check_geometry_validity(final_s):
                                p_list = oracle.predict_properties([final_s])
                                props = p_list[0] if p_list else {}
                                reward, info = reward_engine.compute_reward(relax_res, props)
                                
                                # SAVE SUCCESSFUL CANDIDATE
                                if reward > 0.0:
                                    stable_count += 1
                                    agent.memory.append(final_s)
                                    formula = final_s.composition.reduced_formula
                                    best_formula_epoch = formula
                                    
                                    # 1. Save CIF
                                    filename = f"{disc_dir}/{formula}_Ep{epoch}_B{batch_i}.cif"
                                    CifWriter(final_s).write_file(filename)
                                    
                                    # 2. Save Data to CSV
                                    with open(cand_file, "a", newline="") as f:
                                        c_writer = csv.writer(f)
                                        c_writer.writerow([
                                            formula, 
                                            props.get('formation_energy', 0), 
                                            props.get('band_gap_scalar', 0), 
                                            reward, 
                                            epoch
                                        ])
                            else:
                                reward = -10.0 # Collapsed
                        else:
                            reward = -5.0 # Failed Relax
                    else:
                        reward = -10.0 # Initial Overlap

                # Backprop
                total_log_prob = torch.stack(log_probs_list).sum()
                kl_div = F.mse_loss(logits_policy, logits_ref.detach())
                pg_loss = -(reward * total_log_prob)
                loss = pg_loss + (CONFIG["KL_COEF"] * kl_div)

            agent.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), 1.0)
            agent.optimizer.step()
            epoch_rewards.append(reward)

        # End Epoch
        avg_r = np.mean(epoch_rewards)
        pbar.set_postfix({"Avg R": f"{avg_r:.2f}", "Stable": stable_count})
        
        # Log Stats
        with open("training_log.csv", "a", newline="") as f:
            csv.writer(f).writerow([epoch, avg_r, stable_count, best_formula_epoch])
        
        # Save Model
        if avg_r > best_avg_reward:
            best_avg_reward = avg_r
            torch.save(agent.policy.state_dict(), os.path.join(ROOT, "best_rl_model_v2.pt"))

    print("\n✅ Training Complete. All stable materials saved to 'final_candidates.csv'.")
    summarize_results(cand_file)

if __name__ == "__main__":
    main()
