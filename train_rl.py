import os
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import csv
import yaml
from collections import deque

# --- PATH SETUP ---
ROOT = os.path.dirname(__file__)
sys.path.append(os.path.join(ROOT, "CrystalFormer"))

# --- IMPORTS ---
from crystalformer.src.transformer import make_transformer
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter # To save discoveries immediately

# Local Modules
from relaxer import Relaxer
from oracle import Oracle

# --- CONFIGURATION ---
PRETRAINED_DIR = os.path.join(ROOT, "pretrained_model")
CONFIG = {
    "BATCH_SIZE": 4, 
    "LR": 1e-6,           
    "EPOCHS": 50,
    "KL_COEF": 0.05,
    "DEVICE": "cuda",
    "REPLAY_RATIO": 0.5   # 50% of batch comes from your own discoveries
}

class PPOAgent_Online:
    def __init__(self):
        print("--- Initializing PPO Agent (Online Memory Mode) ---")
        self.device = CONFIG["DEVICE"]
        
        # 1. Load Config & Model
        with open(os.path.join(PRETRAINED_DIR, "config.yaml"), "r") as f:
            cfg = yaml.safe_load(f)

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
        
        # Load Weights
        state_dict = torch.load(os.path.join(PRETRAINED_DIR, "epoch_005500_CLEAN.pt"), map_location=self.device)
        self.policy.load_state_dict(state_dict, strict=True)
        self.ref_model.load_state_dict(state_dict, strict=True)
        self.ref_model.eval()
        
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=CONFIG["LR"])
        
        # 2. ONLINE MEMORY (Starts Empty)
        self.memory = [] 
        print("📝 Memory Bank initialized (Empty). Will fill with your discoveries.")

        # Simplified Atom Map (Zn, S, Cd, Se, O, Ga, As)
        self.idx_to_atom = {0: 30, 1: 16, 2: 48, 3: 34, 4: 8, 5: 31, 6: 33} 

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
            return -5.0, "Unstable"
        
        e_form = oracle_props.get('formation_energy', 0.0)
        gap = oracle_props.get('band_gap_scalar', 0.0)

        # Reward Logic
        r_stability = (10 / (1 + np.exp(2 * e_form))) - 5
        r_gap = min(gap * 5.0, 10.0)
        
        return r_stability + r_gap, f"G:{gap:.2f}|E:{e_form:.2f}"

def build_structure(A, X):
    species = [a for a in A if a != 0]
    coords = [x for a, x in zip(A, X) if a != 0]
    if len(species) < 1: return None
    try:
        lattice = Lattice.from_parameters(5.0, 5.0, 5.0, 90, 90, 90)
        return Structure(lattice, species, coords)
    except: return None

# --- MAIN LOOP ---
def main():
    agent = PPOAgent_Online()
    relaxer = Relaxer()
    oracle = Oracle()
    reward_engine = RewardEngine()
    
    # Discovery Folder
    disc_dir = os.path.join(ROOT, "rl_discoveries")
    os.makedirs(disc_dir, exist_ok=True)
    
    log_file = open("training_log.csv", "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["Epoch", "Batch", "Reward", "KL", "Info"])

    best_avg_reward = -10.0 # Track best performance

    print(f"\n🚀 STARTING ONLINE TRAINING - {CONFIG['EPOCHS']} Epochs")

    for epoch in range(CONFIG["EPOCHS"]):
        epoch_rewards = []
        
        for batch_i in range(CONFIG["BATCH_SIZE"]):
            
            # --- A. DECIDE SOURCE: MEMORY OR EXPLORATION? ---
            # Only use memory if we actually FOUND something good previously
            use_memory = (len(agent.memory) > 5) and (random.random() < CONFIG["REPLAY_RATIO"])
            
            if use_memory:
                # === PATH 1: REPLAY SUCCESS ===
                # This keeps the "Good Yields" alive in the brain
                struct = random.choice(agent.memory)
                reward = 10.0 # Reinforce "This was good!"
                info = "Memory_Replay"
                loss = torch.tensor(0.0, requires_grad=True) 
                
            else:
                # === PATH 2: EXPLORE NEW ===
                G_raw = random.randint(1, 230)
                num_atoms = 4
                inputs = agent.prepare_input(G_raw, [[0.5]*3]*num_atoms, [0]*num_atoms, [0]*num_atoms, [1]*num_atoms)
                
                # Forward
                logits_policy = agent.policy(*inputs, is_train=False).squeeze(0)
                with torch.no_grad():
                    logits_ref = agent.ref_model(*inputs, is_train=False).squeeze(0)
                
                # Sample
                log_probs_list = []
                actions_list = []
                for j in range(num_atoms):
                    dist = torch.distributions.Categorical(logits=logits_policy[j][:7]) 
                    action = dist.sample()
                    log_probs_list.append(dist.log_prob(action))
                    actions_list.append(agent.idx_to_atom.get(action.item(), 6))
                
                # Build & Relax
                struct = build_structure(actions_list, [[0.5, 0.5, 0.5]] * num_atoms)
                
                reward = -5.0
                info = "Build_Fail"
                
                if struct:
                    relax_res = relaxer.relax(struct)
                    props = {"formation_energy": 0.0, "band_gap_scalar": 0.0}
                    if relax_res["is_converged"]:
                        p_list = oracle.predict_properties([relax_res["final_structure"]])
                        if p_list: props = p_list[0]
                    
                    reward, info = reward_engine.compute_reward(relax_res, props)

                    # --- CRITICAL: SAVE THE GOOD STUFF ---
                    if reward > 0.0:
                        # 1. Add to Memory for future training
                        agent.memory.append(relax_res["final_structure"])
                        # 2. Save file immediately
                        formula = relax_res["final_structure"].composition.reduced_formula
                        filename = f"{disc_dir}/{formula}_Ep{epoch}_B{batch_i}.cif"
                        CifWriter(relax_res["final_structure"]).write_file(filename)
                        print(f"   💎 DISCOVERY SAVED: {formula} (R:{reward:.2f})")

                # Loss
                total_log_prob = torch.stack(log_probs_list).sum()
                kl_div = F.mse_loss(logits_policy, logits_ref.detach())
                pg_loss = -(reward * total_log_prob)
                loss = pg_loss + (CONFIG["KL_COEF"] * kl_div)

            # --- OPTIMIZE ---
            agent.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), 1.0)
            agent.optimizer.step()

            print(f"   Ep {epoch}.{batch_i} | R: {reward:.2f} ({info})")
            epoch_rewards.append(reward)

        # --- END OF EPOCH STATS ---
        avg_r = sum(epoch_rewards)/len(epoch_rewards) if epoch_rewards else 0.0
        print(f"✅ EPOCH {epoch} END. Avg Reward: {avg_r:.3f} | Memory Size: {len(agent.memory)}")
        
        # SAVE BEST MODEL
        if avg_r > best_avg_reward:
            best_avg_reward = avg_r
            torch.save(agent.policy.state_dict(), os.path.join(ROOT, "best_rl_model.pt"))
            print(f"   💾 NEW RECORD! Model saved to 'best_rl_model.pt'")

if __name__ == "__main__":
    main()
