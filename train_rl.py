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

# --- IMPORTS ---
from crystalformer.src.transformer import make_transformer
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter
from relaxer import Relaxer 
from oracle import Oracle   

# --- CONFIGURATION ---
PRETRAINED_DIR = os.path.join(ROOT, "pretrained_model")
CONFIG = {
    "BATCH_SIZE": 16,       # Low batch size for CPU safety
    "LR": 1e-4,             
    "EPOCHS": 300,          
    "KL_COEF": 0.05,
    "ENTROPY_COEF": 0.1,    
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "EPSILON": 0.20,        # 20% Forced Exploration
    "NUM_WORKERS": 1        
}

# --- WORKER FUNCTION ---
_relaxer = None

def worker_relax_task(task_data):
    global _relaxer
    torch.set_num_threads(1)
    if _relaxer is None: _relaxer = Relaxer()
    idx, struct = task_data
    try:
        # Relax everything (No pre-filter)
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
        # Random Lattice 3.5 - 6.0 (Tighter box to force bonding)
        a = b = c = lattice_scale
        lattice = Lattice.from_parameters(a, b, c, 90, 90, 90)
        return Structure(lattice, species, coords)
    except: return None

# --- AGENT CLASS ---
class PPOAgent_Final:
    def __init__(self):
        print(f"--- Initializing PPO Agent (Device: {CONFIG['DEVICE']}) ---")
        self.device = CONFIG["DEVICE"]
        self.memory = deque(maxlen=5000)
        self.best_avg_reward = -10.0
        
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
        
        # Load Weights
        checkpoint_path = os.path.join(ROOT, "checkpoint.pt")
        author_path = os.path.join(PRETRAINED_DIR, "epoch_005500_CLEAN.pt")
        
        if os.path.exists(checkpoint_path):
            print(f"🔄 Resuming from Checkpoint...")
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.policy.load_state_dict(ckpt['policy_state'])
            self.ref_model.load_state_dict(ckpt['ref_state'])
            self.optimizer.load_state_dict(ckpt['optimizer_state'])
            self.memory = ckpt['memory'] 
        else:
            print("⚠️ Starting Fresh from Author Weights.")
            state = torch.load(author_path, map_location=self.device)
            self.policy.load_state_dict(state, strict=True)
            self.ref_model.load_state_dict(state, strict=True)

        self.ref_model.eval()

        self.idx_to_atom = {
            0: 30, 1: 16, 2: 48, 3: 34, 4: 8, 5: 31, 6: 33,
            7: 29, 8: 49, 9: 50, 10: 32, 11: 52, 12: 14
        }
        self.atom_keys = list(self.idx_to_atom.keys())

    def save_checkpoint(self, epoch, current_reward):
        ckpt = {
            'epoch': epoch,
            'policy_state': self.policy.state_dict(),
            'ref_state': self.ref_model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'best_reward': self.best_avg_reward,
            'memory': self.memory
        }
        torch.save(ckpt, os.path.join(ROOT, "checkpoint.pt"))

    def prepare_input(self, G, XYZ, A, W, M):
        return (
            torch.tensor([G], device=self.device),
            torch.tensor([XYZ], device=self.device),
            torch.tensor([A], device=self.device),
            torch.tensor([W], device=self.device),
            torch.tensor([M], device=self.device)
        )

# --- MAIN PIPELINE ---
def main():
    mp.set_start_method('spawn', force=True)
    agent = PPOAgent_Final()
    
    # 1. Initialize Oracle with CPU Fallback (handled inside your oracle.py)
    print("🔮 Initializing Oracle...")
    oracle = Oracle(device=CONFIG["DEVICE"]) 
    
    # --- ORACLE WARMUP TEST ---
    print("\n🧪 Running Oracle Warmup Test (Silicon)...")
    try:
        # Create a fake Silicon structure
        si_struct = Structure(Lattice.cubic(5.43), ["Si"]*2, [[0,0,0], [0.25,0.25,0.25]])
        pred = oracle.predict_batch([si_struct])[0]
        e_test = pred["formation_energy"]
        g_test = pred["band_gap_scalar"]
        print(f"   -> Result: Energy={e_test:.4f} eV, Gap={g_test:.4f} eV")
        
        if e_test == 0.0 and g_test == 0.0:
            print("🚨 WARNING: Oracle returned EXACT ZEROS. Check installation.")
        elif e_test < -0.1:
            print("✅ PASS: Oracle is returning valid negative energy.")
    except Exception as e:
        print(f"❌ Oracle Warmup Failed: {e}")
        # We continue anyway, but expect issues if this failed
    
    disc_dir = os.path.join(ROOT, "rl_discoveries")
    os.makedirs(disc_dir, exist_ok=True)
    
    # Initialize Logs
    if not os.path.exists("training_log.csv"):
        with open("training_log.csv", "w") as f:
            csv.writer(f).writerow(["Epoch", "Avg_Reward", "Valid_Count", "Best_Formula", "Time_Sec"])
        with open("final_candidates.csv", "w") as f:
            csv.writer(f).writerow(["Formula", "Formation_Energy", "Band_Gap", "Reward", "Epoch"])
        with open("relaxed_all.csv", "w") as f:
            csv.writer(f).writerow(["Epoch", "Formula", "Energy", "BandGap", "Reward"])

    WINDOW = 10
    w_reward = 0.0
    w_valid = 0
    w_best_f = "-"
    w_best_r = -10.0
    
    print(f"\n🚀 STARTING FINAL PIPELINE: {CONFIG['EPOCHS']} Epochs")
    pool = mp.Pool(processes=CONFIG["NUM_WORKERS"])
    
    try:
        for epoch in range(CONFIG["EPOCHS"]):
            start_time = time.time()
            batch_data = []
            
            # --- 1. GENERATION ---
            for _ in range(CONFIG["BATCH_SIZE"]):
                G_raw = random.randint(1, 230)
                num_atoms = random.randint(2, 6) 
                # Tighter lattice to encourage bonding (was 4.0-8.0)
                lattice_guess = random.uniform(3.5, 6.0) 
                
                inputs = agent.prepare_input(G_raw, [[0.5]*3]*num_atoms, [0]*num_atoms, [0]*num_atoms, [1]*num_atoms)
                logits_policy = agent.policy(*inputs, is_train=False).squeeze(0)
                with torch.no_grad():
                    logits_ref = agent.ref_model(*inputs, is_train=False).squeeze(0)
                
                log_probs = []
                actions = []
                X_list = []
                
                for j in range(num_atoms):
                    base = 1 + 5 * j
                    atom_logits = logits_policy[base][:13]
                    atom_dist = torch.distributions.Categorical(logits=atom_logits)
                    
                    # EPSILON-GREEDY
                    if random.random() < CONFIG["EPSILON"]:
                        atom_idx = torch.tensor(random.choice(agent.atom_keys), device=agent.device)
                        log_probs.append(atom_dist.log_prob(atom_idx))
                    else:
                        atom_idx = atom_dist.sample()
                        log_probs.append(atom_dist.log_prob(atom_idx))
                        
                    actions.append(agent.idx_to_atom.get(atom_idx.item(), 6))
                    
                    w_dist = torch.distributions.Categorical(logits=logits_policy[base+4][:agent.policy.wyck_types])
                    w_act = w_dist.sample()
                    log_probs.append(w_dist.log_prob(w_act))
                    
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

            # --- 2. RELAXATION ---
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

            # --- 3. ORACLE ---
            oracle_tasks = []
            oracle_map = []
            for idx, item in enumerate(batch_data):
                if "relax_res" in item:
                    res = item["relax_res"]
                    if res["is_converged"]:
                        oracle_tasks.append(res["final_structure"])
                        oracle_map.append(idx)
                    else:
                        item["result"] = "diverged"
            
            if oracle_tasks:
                preds = oracle.predict_batch(oracle_tasks)
                for k, pred in enumerate(preds):
                    idx = oracle_map[k]
                    batch_data[idx]["oracle"] = pred
                    batch_data[idx]["result"] = "success"

            # --- 4. REAL REWARDS ---
            agent.optimizer.zero_grad()
            loss_accum = torch.tensor(0.0, device=agent.device)
            rewards = []
            
            epoch_valid_count = 0
            epoch_best_f = "-"
            
            for item in batch_data:
                reward = -2.0 # Default failure
                
                if item["result"] == "success":
                    final_s = item["relax_res"]["final_structure"]
                    form = final_s.composition.reduced_formula
                    e = item["oracle"]["formation_energy"]
                    g = item["oracle"]["band_gap_scalar"]
                    
                    # === REAL PHYSICS LOGIC ===
                    # 1. Stability (Must be negative energy)
                    # e < -0.1 implies true bonding
                    if e < -0.1: 
                        r_stab = 1.0
                    else:
                        r_stab = -0.5 # Penalty for unstable/positive energy
                        
                    # 2. Diversity Penalty (Anti-Zinc)
                    # If it's just Zn, cap the reward so it explores
                    if form == "Zn" or form == "Zn1":
                         r_stab = min(r_stab, 0.2)
                        
                    # 3. Band Gap Bonus (Real Semiconductor)
                    if g > 0.5:
                        r_gap = 5.0 
                    elif g > 0.1:
                        r_gap = 1.0 # Small gap is better than metal
                    else:
                        r_gap = 0.0
                        
                    reward = r_stab + r_gap
                    epoch_valid_count += 1
                    epoch_best_f = form
                    
                    # Log EVERYTHING valid
                    with open("relaxed_all.csv", "a") as f:
                        csv.writer(f).writerow([epoch, form, f"{e:.4f}", f"{g:.4f}", f"{reward:.2f}"])
                        
                    # Save WINNERS (Now defining winner as stable OR gap)
                    if reward > 0.5:
                        with open("final_candidates.csv", "a") as f:
                            csv.writer(f).writerow([form, e, g, reward, epoch])
                        
                    if reward > 2.0:
                        fname = f"{disc_dir}/{form}_{epoch}_gap{g:.2f}.cif"
                        CifWriter(final_s).write_file(fname)

                rewards.append(reward)

                if item["result"] in ["success", "diverged"]:
                     log_sum = torch.stack(item["log_probs"]).sum()
                     kl = F.kl_div(F.log_softmax(item["logits_policy"], dim=-1), F.softmax(item["logits_ref"].detach(), dim=-1), reduction="batchmean")
                     loss = -(reward * log_sum) + (CONFIG["KL_COEF"] * kl)
                     loss_accum += loss

            if loss_accum.requires_grad:
                loss_accum.backward()
                torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), 1.0)
                agent.optimizer.step()
                
            avg_r = np.mean(rewards) if rewards else 0
            
            # --- 5. REPORTING ---
            w_reward += avg_r
            w_valid += epoch_valid_count
            if epoch_valid_count > 0 and rewards:
                if max(rewards) > w_best_r:
                    w_best_r = max(rewards)
                    w_best_f = epoch_best_f
            
            with open("training_log.csv", "a") as f:
                csv.writer(f).writerow([epoch, avg_r, epoch_valid_count, epoch_best_f, time.time()-start_time])
            
            if (epoch + 1) % WINDOW == 0:
                print(f"[E{epoch+1-WINDOW}-{epoch+1}] R={w_reward/WINDOW:.2f} | Valid={w_valid} | Best={w_best_f}")
                agent.save_checkpoint(epoch, avg_r)
                w_reward = 0.0; w_valid = 0; w_best_f = "-"; w_best_r = -10.0

    except KeyboardInterrupt: pass
    finally:
        pool.close(); pool.join()

if __name__ == "__main__":
    main()
