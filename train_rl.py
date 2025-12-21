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
    "BATCH_SIZE": 8,       
    "LR": 1e-5,
    "EPOCHS": 50,
    "KL_COEF": 0.05,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "REPLAY_RATIO": 0.3,    
    "NUM_WORKERS": 1        # Keep safe default
}

# --- WORKER FUNCTION ---
def worker_relax_task(task_data):
    """Running in a separate process with strictly 1 CPU thread."""
    idx, struct = task_data
    torch.set_num_threads(1) 
    try:
        relaxer = Relaxer()
        result = relaxer.relax(struct)
        return (idx, result)
    except Exception as e:
        return (idx, {"is_converged": False, "error": str(e)})

# --- UTILS ---
def get_structure_hash(struct):
    """
    Generates a deterministic hash for structure deduplication.
    Rounds lattice and coords to ensure slight variations match.
    """
    try:
        # 1. Formula
        formula = struct.composition.reduced_formula
        # 2. Lattice (Round to 2 decimals)
        # .parameters returns (a, b, c, alpha, beta, gamma)
        latt = tuple(np.round(struct.lattice.parameters, 2))
        # 3. Coords (Flatten & Round to 2 decimals)
        coords = tuple(np.round(struct.frac_coords.flatten(), 2))
        
        # Combine into immutable tuple
        return (formula, latt, coords)
    except:
        return None

def check_geometry_fast(struct):
    """Fast pre-relax filter."""
    try:
        dm = struct.distance_matrix
        np.fill_diagonal(dm, 10.0)
        return np.min(dm) >= 0.8
    except: return False

def build_structure(A, X, lattice_scale):
    species = [a for a in A if a != 0]
    coords = [x for a, x in zip(A, X) if a != 0]
    if len(species) < 1: return None
    try:
        a = b = c = lattice_scale
        lattice = Lattice.from_parameters(a, b, c, 90, 90, 90)
        return Structure(lattice, species, coords)
    except: return None

# --- AGENT CLASS ---
class PPOAgent_Pipeline:
    def __init__(self):
        print(f"--- Initializing PPO Pipeline (Workers: {CONFIG['NUM_WORKERS']}) ---")
        self.device = CONFIG["DEVICE"]
        self.start_epoch = 0
        self.best_avg_reward = -10.0
        
        # Reward Cache for Deduplication
        self.reward_cache = {} 
        
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
        self.memory = deque(maxlen=3000)
        
        # --- CHECKPOINT RESUME LOGIC ---
        checkpoint_path = os.path.join(ROOT, "checkpoint.pt")
        author_path = os.path.join(PRETRAINED_DIR, "epoch_005500_CLEAN.pt")
        
        if os.path.exists(checkpoint_path):
            print(f"🔄 Resuming from Checkpoint: {os.path.basename(checkpoint_path)}")
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            
            self.policy.load_state_dict(ckpt['policy_state'])
            self.ref_model.load_state_dict(ckpt['ref_state'])
            self.optimizer.load_state_dict(ckpt['optimizer_state'])
            self.start_epoch = ckpt['epoch'] + 1
            self.best_avg_reward = ckpt['best_reward']
            self.memory = ckpt['memory'] 
            # Ideally load cache too, but starting fresh is safe
            if 'reward_cache' in ckpt:
                self.reward_cache = ckpt['reward_cache']
            print(f"   Resuming at Epoch {self.start_epoch}")
            
        else:
            print("⚠️ No checkpoint found. Starting fresh from Author's weights.")
            state = torch.load(author_path, map_location=self.device)
            self.policy.load_state_dict(state, strict=True)
            self.ref_model.load_state_dict(state, strict=True)
        
        self.ref_model.eval()

        self.idx_to_atom = {
            0: 30, 1: 16, 2: 48, 3: 34, 4: 8, 5: 31, 6: 33,
            7: 29, 8: 49, 9: 50, 10: 32, 11: 52, 12: 14
        }

    def save_checkpoint(self, epoch, current_reward):
        """Saves full training state for resume capability."""
        ckpt = {
            'epoch': epoch,
            'policy_state': self.policy.state_dict(),
            'ref_state': self.ref_model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'best_reward': self.best_avg_reward,
            'memory': self.memory,
            'reward_cache': self.reward_cache
        }
        torch.save(ckpt, os.path.join(ROOT, "checkpoint.pt"))
        
        if current_reward > self.best_avg_reward:
            self.best_avg_reward = current_reward
            torch.save(self.policy.state_dict(), os.path.join(ROOT, "best_rl_model_v4.pt"))

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
    
    agent = PPOAgent_Pipeline()
    oracle = Oracle(device=CONFIG["DEVICE"])
    
    disc_dir = os.path.join(ROOT, "rl_discoveries")
    os.makedirs(disc_dir, exist_ok=True)
    
    if agent.start_epoch == 0:
        with open("training_log.csv", "w", newline="") as f:
            # Updated Headers with Diagnostics
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Reward", "Stable_Count", "Top_Formula", "Time_Sec", 
                             "Pct_Filtered", "Pct_Diverged", "Pct_Converged", "Pct_Dedup"])
        with open("final_candidates.csv", "w", newline="") as f:
            csv.writer(f).writerow(["Formula", "Formation_Energy", "Band_Gap", "Reward", "Epoch"])
    
    print(f"\n🚀 STARTING PARALLEL PIPELINE: {CONFIG['EPOCHS']} Epochs")
    
    pool = mp.Pool(processes=CONFIG["NUM_WORKERS"])
    
    try:
        for epoch in range(agent.start_epoch, CONFIG["EPOCHS"]):
            start_time = time.time()
            
            # --- DIAGNOSTIC COUNTERS ---
            count_total = 0
            count_dedup = 0
            count_filtered = 0
            count_diverged = 0
            count_converged = 0
            
            # --- STAGE 1: BATCH GENERATION ---
            batch_data = [] 
            
            for i in range(CONFIG["BATCH_SIZE"]):
                count_total += 1
                
                # Memory Replay
                use_mem = (len(agent.memory) > 20) and (random.random() < CONFIG["REPLAY_RATIO"])
                
                if use_mem:
                    memory_item = random.choice(agent.memory)
                    batch_data.append({
                        "type": "replay", 
                        "struct": memory_item['struct'],
                        "stored_reward": memory_item['reward']
                    })
                else:
                    # Generate New
                    G_raw = random.randint(1, 230)
                    num_atoms = random.randint(2, 6)
                    lattice_guess = random.uniform(3.5, 6.0)
                    
                    inputs = agent.prepare_input(
                        G_raw,
                        [[0.5]*3]*num_atoms,
                        [0]*num_atoms,
                        [0]*num_atoms,
                        [1]*num_atoms
                    )
                
                    # ===================== FIX START =====================
                    # <<< FIX: compute logits BEFORE using them
                    logits_policy = agent.policy(*inputs, is_train=False).squeeze(0)
                    with torch.no_grad():
                        logits_ref = agent.ref_model(*inputs, is_train=False).squeeze(0)
                    # ====================== FIX END ======================
                
                    log_probs = []
                    actions = []
                    X_list = []
                
                    # ---------- Atom sampling ----------
                    for j in range(num_atoms):
                        base = 1 + 5 * j   # <<< FIX (already correct)
                
                        atom_logits = logits_policy[base][:13]
                        atom_dist = torch.distributions.Categorical(logits=atom_logits)
                        atom_action = atom_dist.sample()
                
                        log_probs.append(atom_dist.log_prob(atom_action))
                        actions.append(agent.idx_to_atom.get(atom_action.item(), 6))
                
                    # ---------- Coordinate sampling ----------
                    for j in range(num_atoms):
                        base = 1 + 5 * j   # <<< FIX (already correct)
                
                        x = torch.sigmoid(logits_policy[base + 1][0]).item()
                        y = torch.sigmoid(logits_policy[base + 2][0]).item()
                        z = torch.sigmoid(logits_policy[base + 3][0]).item()
                
                        X_list.append([x, y, z])
                
                    struct = build_structure(actions, X_list, lattice_guess)
                
                    batch_data.append({
                        "type": "gen",
                        "log_probs": log_probs,
                        "logits_policy": logits_policy,
                        "logits_ref": logits_ref,
                        "struct": struct
                    })


            # --- STAGE 2: DEDUPLICATION & FILTERING ---
            relax_tasks = []
            
            for idx, item in enumerate(batch_data):
                if item["type"] == "gen":
                    s = item["struct"]
                    
                    # 1. Structure Validity
                    if s is None:
                        item["result"] = "invalid"
                        count_filtered += 1
                        continue

                    # 2. DEDUPLICATION (Cache Check)
                    shash = get_structure_hash(s)
                    if shash and shash in agent.reward_cache:
                        item["result"] = "cached"
                        item["cached_reward"] = agent.reward_cache[shash]
                        count_dedup += 1
                        continue

                    # 3. Geometry Filter
                    if check_geometry_fast(s):
                        # Valid & New -> Send to Relaxer
                        relax_tasks.append((idx, s))
                        # Note: We assign 'shash' to item so we can cache result later
                        item["shash"] = shash
                    else:
                        item["result"] = "invalid" 
                        count_filtered += 1
                else:
                    item["result"] = "replay" 

            # --- STAGE 3: PARALLEL RELAXATION ---
            if relax_tasks:
                results = pool.map(worker_relax_task, relax_tasks)
                for (idx, res) in results:
                    batch_data[idx]["relax_res"] = res
            
            # --- STAGE 4: BATCHED ORACLE ---
            oracle_tasks = []
            oracle_indices = []
            
            for idx, item in enumerate(batch_data):
                if "relax_res" in item:
                    res = item["relax_res"]
                    if res["is_converged"]:
                        # Converged -> Check Post-Relax Geometry
                        if check_geometry_fast(res["final_structure"]):
                            oracle_tasks.append(res["final_structure"])
                            oracle_indices.append(idx)
                            count_converged += 1
                        else:
                            item["result"] = "collapsed"
                            count_diverged += 1 # Technically geometry fail, but post-relax
                    else:
                        item["result"] = "diverged"
                        count_diverged += 1
            
            if oracle_tasks:
                oracle_preds = oracle.predict_batch(oracle_tasks)
                for i, pred in enumerate(oracle_preds):
                    idx = oracle_indices[i]
                    batch_data[idx]["oracle"] = pred
                    batch_data[idx]["result"] = "success"

            # --- STAGE 5: REWARD, UPDATE & CACHING ---
            rewards = []
            stable_cnt = 0
            best_form = "None"
            
            agent.optimizer.zero_grad()
            loss_accum = torch.tensor(0.0, device=agent.device)
            
            for item in batch_data:
                reward = -5.0 
                
                # A. Handle Cached Result
                if item["result"] == "cached":
                    reward = item["cached_reward"]
                
                # B. Handle Replay
                elif item["result"] == "replay":
                    reward = item["stored_reward"]
                
                # C. Handle New Success
                elif item["result"] == "success":
                    props = item["oracle"]
                    e = props["formation_energy"]
                    g = props["band_gap_scalar"]
                    
                    if e > -50.0: 
                        r_stab = (10 / (1 + np.exp(2 * e))) - 5
                        r_gap = min(g * 5.0, 10.0)
                        reward = r_stab + r_gap
                        
                        # --- CACHE UPDATE ---
                        if "shash" in item and item["shash"]:
                            agent.reward_cache[item["shash"]] = reward

                        if reward > 0.0:
                            stable_cnt += 1
                            final_s = item["relax_res"]["final_structure"]
                            agent.memory.append({'struct': final_s, 'reward': reward})
                            
                            f_str = final_s.composition.reduced_formula
                            best_form = f_str
                            
                            with open("final_candidates.csv", "a", newline="") as f:
                                csv.writer(f).writerow([f_str, e, g, reward, epoch])
                            
                            if stable_cnt <= 5: 
                                fname = f"{disc_dir}/{f_str}_{epoch}.cif"
                                CifWriter(final_s).write_file(fname)

                # D. Failures
                elif item["result"] == "invalid":
                    reward = -10.0
                elif item["result"] == "diverged":
                    reward = -5.0
                elif item["result"] == "collapsed":
                    reward = -10.0
                
                rewards.append(reward)
                
                # Accumulate Loss
                if item["type"] == "gen":
                    log_sum = torch.stack(item["log_probs"]).sum()
                    kl = F.mse_loss(item["logits_policy"], item["logits_ref"].detach())
                    loss = -(reward * log_sum) + (CONFIG["KL_COEF"] * kl)
                    loss_accum += loss

            if loss_accum.requires_grad:
                loss_accum.backward()
                torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), 1.0)
                agent.optimizer.step()

            avg_r = np.mean(rewards)
            epoch_time = time.time() - start_time
            
            # --- LOGGING ---
            # Avoid division by zero
            pct_filt = (count_filtered / CONFIG["BATCH_SIZE"]) * 100
            pct_divg = (count_diverged / CONFIG["BATCH_SIZE"]) * 100
            pct_conv = (count_converged / CONFIG["BATCH_SIZE"]) * 100
            pct_dedup = (count_dedup / CONFIG["BATCH_SIZE"]) * 100
            
            print(f"Epoch {epoch+1}/{CONFIG['EPOCHS']} | R: {avg_r:.2f} | Filt: {int(pct_filt)}% | Divg: {int(pct_divg)}% | Conv: {int(pct_conv)}% | Dup: {int(pct_dedup)}%")
            
            with open("training_log.csv", "a", newline="") as f:
                csv.writer(f).writerow([epoch, avg_r, stable_cnt, best_form, epoch_time, 
                                        pct_filt, pct_divg, pct_conv, pct_dedup])

            agent.save_checkpoint(epoch, avg_r)

    except KeyboardInterrupt:
        print("\n🛑 Training Interrupted. Checkpoint saved.")
    finally:
        pool.close()
        pool.join()
        print("✅ Pipeline Stopped.")

if __name__ == "__main__":
    main()
