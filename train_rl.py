import os
import sys
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import csv
import yaml
import pandas as pd
from collections import deque
from tqdm import tqdm
import time

# --- PATH SETUP ---
ROOT = os.path.dirname(os.path.abspath("__file__"))
sys.path.append(os.path.join(ROOT, "CrystalFormer"))

from crystalformer.src.transformer import make_transformer
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter
from relaxer import Relaxer 
from oracle import Oracle   

# --- CONFIGURATION ---
PRETRAINED_DIR = os.path.join(ROOT, "pretrained_model")
CONFIG = {
    "BATCH_SIZE": 16,       # Lower batch size since we relax EVERYTHING
    "LR": 1e-4,             
    "EPOCHS": 100,
    "KL_COEF": 0.05,
    "ENTROPY_COEF": 0.1,    
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "EPSILON": 0.20,        # 20% Chance to FORCE random atoms (The "Shortcut")
    "NUM_WORKERS": 1        
}

# --- WORKER ---
_relaxer = None
def worker_relax_task(task_data):
    global _relaxer
    torch.set_num_threads(1)
    if _relaxer is None: _relaxer = Relaxer()
    idx, struct = task_data
    try:
        # TIMEOUT: Don't spend forever on garbage crystals
        # We rely on the relaxer's internal step limit, but this catches hangs
        result = _relaxer.relax(struct) 
        return (idx, result)
    except Exception as e:
        return (idx, {"is_converged": False, "error": str(e)})

# --- UTILS ---
def build_structure(A, X, lattice_scale):
    species = [a for a in A if a != 0]
    coords = [x for a, x in zip(A, X) if a != 0]
    if len(species) < 1: return None
    try:
        # Give them space! 4.0 to 8.0 Angstroms
        a = b = c = lattice_scale
        lattice = Lattice.from_parameters(a, b, c, 90, 90, 90)
        return Structure(lattice, species, coords)
    except: return None

# --- AGENT ---
class PPOAgent_Direct:
    def __init__(self):
        print(f"--- Initializing Direct PPO (No Filters) ---")
        self.device = CONFIG["DEVICE"]
        self.memory = deque(maxlen=5000)
        
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
        
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=CONFIG["LR"])
        
        # Fresh Start Only
        author_path = os.path.join(PRETRAINED_DIR, "epoch_005500_CLEAN.pt")
        print("⚠️ Starting Fresh from Author Weights.")
        state = torch.load(author_path, map_location=self.device)
        self.policy.load_state_dict(state, strict=True)
        self.ref_model.load_state_dict(state, strict=True)
        self.ref_model.eval()

        # Mapping (Zinc is 0, Silicon is 12, etc)
        self.idx_to_atom = {
            0: 30, 1: 16, 2: 48, 3: 34, 4: 8, 5: 31, 6: 33,
            7: 29, 8: 49, 9: 50, 10: 32, 11: 52, 12: 14
        }
        # Reverse mapping for "Forced" exploration
        self.atom_keys = list(self.idx_to_atom.keys())

    def prepare_input(self, G, XYZ, A, W, M):
        return (
            torch.tensor([G], device=self.device),
            torch.tensor([XYZ], device=self.device),
            torch.tensor([A], device=self.device),
            torch.tensor([W], device=self.device),
            torch.tensor([M], device=self.device)
        )

# --- MAIN ---
def main():
    mp.set_start_method('spawn', force=True)
    agent = PPOAgent_Direct()
    oracle = Oracle(device=CONFIG["DEVICE"])
    os.makedirs("rl_discoveries", exist_ok=True)
    
    # Clean Logs
    with open("training_log.csv", "w") as f:
        csv.writer(f).writerow(["Epoch", "Avg_Reward", "Valid_Count", "Semi_Count", "Best_Formula"])
    with open("all_attempts.csv", "w") as f:
        csv.writer(f).writerow(["Epoch", "Formula", "Result", "Energy", "Gap", "Reward"])

    print(f"\n🚀 STARTING DIRECT TRAINING (Epsilon={CONFIG['EPSILON']*100}%)")
    pool = mp.Pool(processes=CONFIG["NUM_WORKERS"])
    
    try:
        for epoch in range(CONFIG["EPOCHS"]):
            start = time.time()
            batch_data = []
            
            # --- 1. GENERATION (With Forced Exploration) ---
            for _ in range(CONFIG["BATCH_SIZE"]):
                G_raw = random.randint(1, 230)
                num_atoms = random.randint(2, 6) # Keep it small for speed
                lattice_guess = random.uniform(4.0, 8.0) # BIG BOX
                
                inputs = agent.prepare_input(G_raw, [[0.5]*3]*num_atoms, [0]*num_atoms, [0]*num_atoms, [1]*num_atoms)
                logits_policy = agent.policy(*inputs, is_train=False).squeeze(0)
                with torch.no_grad():
                    logits_ref = agent.ref_model(*inputs, is_train=False).squeeze(0)
                
                log_probs = []
                actions = []
                X_list = []
                
                for j in range(num_atoms):
                    base = 1 + 5 * j
                    
                    # --- A. ATOM SELECTION ---
                    atom_logits = logits_policy[base][:13]
                    atom_dist = torch.distributions.Categorical(logits=atom_logits)
                    
                    # EPSILON-GREEDY: Force diversity!
                    if random.random() < CONFIG["EPSILON"]:
                        # IGNORE model, pick random atom (forces non-Zinc)
                        atom_idx = torch.tensor(random.choice(agent.atom_keys), device=agent.device)
                        # We still need log_prob for the update, even if forced
                        log_probs.append(atom_dist.log_prob(atom_idx))
                    else:
                        # Use Model
                        atom_idx = atom_dist.sample()
                        log_probs.append(atom_dist.log_prob(atom_idx))
                        
                    actions.append(agent.idx_to_atom.get(atom_idx.item(), 6))
                    
                    # --- B. WYCKOFF ---
                    w_dist = torch.distributions.Categorical(logits=logits_policy[base+4][:agent.policy.wyck_types])
                    w_act = w_dist.sample()
                    log_probs.append(w_dist.log_prob(w_act))
                    
                    # --- C. COORDINATES (Mixture Sampling) ---
                    Kx = agent.policy.Kx
                    def sample_c(blk):
                        w_dist = torch.distributions.Categorical(logits=blk[:Kx])
                        idx = w_dist.sample()
                        log_probs.append(w_dist.log_prob(idx))
                        return torch.sigmoid(blk[Kx : 2*Kx][idx]).item()

                    X_list.append([
                        sample_c(logits_policy[base+1]),
                        sample_c(logits_policy[base+2]),
                        sample_c(logits_policy[base+3])
                    ])
                
                struct = build_structure(actions, X_list, lattice_guess)
                batch_data.append({
                    "type": "gen",
                    "struct": struct,
                    "log_probs": log_probs,
                    "logits_policy": logits_policy,
                    "logits_ref": logits_ref
                })

            # --- 2. RELAX EVERYTHING (No Filter) ---
            relax_tasks = []
            for i, item in enumerate(batch_data):
                if item["struct"]:
                    relax_tasks.append((i, item["struct"]))
                else:
                    item["result"] = "invalid_build"

            if relax_tasks:
                results = pool.map(worker_relax_task, relax_tasks)
                for (idx, res) in results:
                    batch_data[idx]["relax_res"] = res

            # --- 3. ORACLE & REWARD ---
            oracle_tasks = []
            oracle_map = []
            for idx, item in enumerate(batch_data):
                if "relax_res" in item:
                    res = item["relax_res"]
                    if res["is_converged"]:
                        oracle_tasks.append(res["final_structure"])
                        oracle_map.append(idx)
                    else:
                        item["result"] = "diverged" # Physics failed
            
            if oracle_tasks:
                preds = oracle.predict_batch(oracle_tasks)
                for k, pred in enumerate(preds):
                    idx = oracle_map[k]
                    batch_data[idx]["oracle"] = pred
                    batch_data[idx]["result"] = "success"

            # --- 4. CALC REWARDS ---
            agent.optimizer.zero_grad()
            loss_accum = torch.tensor(0.0, device=agent.device)
            rewards = []
            
            valid_count = 0
            semi_count = 0
            best_f = "-"
            
            for item in batch_data:
                reward = -2.0 # Default penalty (Explosion/Invalid)
                
                form = "Invalid"
                e = 0.0
                g = 0.0
                
                if item["result"] == "success":
                    form = item["relax_res"]["final_structure"].composition.reduced_formula
                    e = item["oracle"]["formation_energy"]
                    g = item["oracle"]["band_gap_scalar"]
                    
                    # REWARD LOGIC
                    # 1. Base Stability
                    if e <= 0.2: # Loose stability threshold
                        r_stab = 0.5
                    else:
                        r_stab = -0.5
                        
                    # 2. Band Gap (The Goal)
                    if g > 0.1:
                        r_gap = 5.0 # BIG BONUS
                        semi_count += 1
                    else:
                        r_gap = 0.0
                        
                    reward = r_stab + r_gap
                    valid_count += 1
                    best_f = form
                    
                    # Save Good Stuff
                    if reward > 2.0:
                        item["relax_res"]["final_structure"].to(filename=f"rl_discoveries/{form}_{epoch}.cif")

                rewards.append(reward)
                
                # Log Attempt
                with open("all_attempts.csv", "a") as f:
                    csv.writer(f).writerow([epoch, form, item["result"], f"{e:.2f}", f"{g:.2f}", f"{reward:.2f}"])

                # Loss Calculation
                if item["result"] == "success" or item["result"] == "diverged":
                     # We learn from Failures too! (Diverged = Negative Reward)
                     log_sum = torch.stack(item["log_probs"]).sum()
                     kl = F.kl_div(F.log_softmax(item["logits_policy"], dim=-1), F.softmax(item["logits_ref"].detach(), dim=-1), reduction="batchmean")
                     loss = -(reward * log_sum) + (CONFIG["KL_COEF"] * kl)
                     loss_accum += loss

            if loss_accum.requires_grad:
                loss_accum.backward()
                torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), 1.0)
                agent.optimizer.step()
                
            # Log
            avg_r = np.mean(rewards) if rewards else 0
            print(f"[Epoch {epoch+1}] R={avg_r:.2f} | Valid={valid_count} | Semi={semi_count} | Last: {best_f}")
            
            with open("training_log.csv", "a") as f:
                csv.writer(f).writerow([epoch, avg_r, valid_count, semi_count, best_f])

    except KeyboardInterrupt: pass
    finally:
        pool.close(); pool.join()

if __name__ == "__main__":
    main()
