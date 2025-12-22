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
    "BATCH_SIZE": 32,       
    "LR": 1e-5,
    "EPOCHS": 150,
    "KL_COEF": 0.05,
    "ENTROPY_COEF": 0.05,   # Added Explicit Entropy Coef
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "REPLAY_RATIO": 0.5,    # INCREASED to 50% for Teacher Mode
    "NUM_WORKERS": 1        # Keep safe default
}

# --- WORKER FUNCTION ---
_relaxer = None

def worker_relax_task(task_data):
    global _relaxer
    torch.set_num_threads(1)

    if _relaxer is None:
        _relaxer = Relaxer()

    idx, struct = task_data
    try:
        result = _relaxer.relax(struct)
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
        # LOOSENED FILTER: 0.8 -> 0.6 to allow slightly messy crystals to survive
        return np.min(dm) >= 0.6
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
            if 'reward_cache' in ckpt:
                self.reward_cache = ckpt['reward_cache']
            print(f"   Resuming at Epoch {self.start_epoch}")
            
        else:
            print("⚠️ No checkpoint found. Starting fresh from Author's weights.")
            state = torch.load(author_path, map_location=self.device)
            self.policy.load_state_dict(state, strict=True)
            self.ref_model.load_state_dict(state, strict=True)
            # INJECT TEACHER KNOWLEDGE ON FRESH START
            self.inject_teacher_knowledge()
        
        self.ref_model.eval()

        self.idx_to_atom = {
            0: 30, 1: 16, 2: 48, 3: 34, 4: 8, 5: 31, 6: 33,
            7: 29, 8: 49, 9: 50, 10: 32, 11: 52, 12: 14
        }

    def inject_teacher_knowledge(self):
        """Hard-codes known semiconductors into memory so the agent knows what to look for."""
        seeds = []
        # Si (Diamond)
        seeds.append(Structure(Lattice.cubic(5.43), ["Si", "Si"], [[0,0,0], [0.25,0.25,0.25]]))
        # GaAs (Zincblende)
        seeds.append(Structure(Lattice.cubic(5.65), ["Ga", "As"], [[0,0,0], [0.25,0.25,0.25]]))
        # MgO (Rock Salt)
        seeds.append(Structure(Lattice.cubic(4.21), ["Mg", "O"], [[0,0,0], [0.5,0.5,0.5]]))
        
        print(f"💉 Injecting {len(seeds)} Teacher Crystals into Memory...")
        for s in seeds:
            for _ in range(50): # Repeat to fill buffer and influence early batches
                self.memory.append({'struct': s, 'reward': 5.0}) # Synthetic High Reward

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

    # Step 1: Add rolling window trackers
    WINDOW = 10

    # Rolling window accumulators
    w_gen = w_filt = w_relx = w_valid = w_target = 0
    w_reward = 0.0
    w_time = 0.0
    # Store all target candidates in this window
    w_candidates = []  # (reward, formula, formation_energy, band_gap)

    # Best target material in window
    w_best = None   # (reward, formula, formation_energy, band_gap)
    

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
                # Increased threshold to 50 to allow teacher memory to work
                use_mem = (len(agent.memory) > 50) and (random.random() < CONFIG["REPLAY_RATIO"])
                
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
                
                    logits_policy = agent.policy(*inputs, is_train=False).squeeze(0)
                    with torch.no_grad():
                        logits_ref = agent.ref_model(*inputs, is_train=False).squeeze(0)
                
                    log_probs = []
                    actions = []
                    X_list = []
                
                    # ---------- Atom & Coordinate Sampling ----------
                    for j in range(num_atoms):
                        base = 1 + 5 * j
                    
                        # 1. Atom Type
                        atom_logits = logits_policy[base][:13]
                        atom_dist = torch.distributions.Categorical(logits=atom_logits)
                        atom_action = atom_dist.sample()
                        log_probs.append(atom_dist.log_prob(atom_action))
                        actions.append(agent.idx_to_atom.get(atom_action.item(), 6))
                    
                        # 2. Wyckoff (Standard)
                        wyckoff_logits = logits_policy[base + 4][:agent.policy.wyck_types]
                        wyckoff_dist = torch.distributions.Categorical(logits=wyckoff_logits)
                        wyckoff_action = wyckoff_dist.sample()
                        log_probs.append(wyckoff_dist.log_prob(wyckoff_action))

                        # 3. Coordinate Sampling (FIXED & MERGED)
                        Kx = agent.policy.Kx
                        
                        def sample_coord(logit_block):
                            # Mixture Weights
                            weights = logit_block[:Kx]
                            dist = torch.distributions.Categorical(logits=weights)
                            mode_idx = dist.sample()
                            log_probs.append(dist.log_prob(mode_idx)) # Learn choice

                            # Mean Value
                            means = logit_block[Kx : 2*Kx]
                            chosen_mu = means[mode_idx]
                            return torch.sigmoid(chosen_mu).item()

                        x = sample_coord(logits_policy[base + 1])
                        y = sample_coord(logits_policy[base + 2])
                        z = sample_coord(logits_policy[base + 3])
                    
                        X_list.append([x, y, z])

                    # Build Structure
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
                    
                    if s is None:
                        item["result"] = "invalid"
                        count_filtered += 1
                        continue

                    # DEDUPLICATION
                    shash = get_structure_hash(s)
                    if shash and shash in agent.reward_cache:
                        item["result"] = "cached"
                        item["cached_reward"] = agent.reward_cache[shash]
                        count_dedup += 1
                        continue

                    # GEOMETRY FILTER
                    if check_geometry_fast(s):
                        relax_tasks.append((idx, s))
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
                            count_diverged += 1
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
                reward = -1.0 # Default penalty (Invalid/Diverged)
                
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

                    # Save for Debug
                    final_s = item["relax_res"]["final_structure"]
                    f_str = final_s.composition.reduced_formula
                    with open("relaxed_all.csv", "a", newline="") as f:
                        csv.writer(f).writerow([epoch, f_str, e, g])

                    # ------------------------------
                    # NEW REWARD LOGIC ("Stepping Stones")
                    # ------------------------------
                    
                    # 1. Stability Base Score (Range -1 to +1)
                    # If stable (e < 0), score is positive.
                    if e < 0:
                        r_stab = 1.0  # BASELINE for any stable crystal (Even metal)
                    else:
                        r_stab = -1.0 # Unstable penalty
                        
                    # 2. Band Gap Bonus (The Jackpot)
                    # If it has a gap > 0.1 eV, massive multiplier
                    if g > 0.1:
                        # Target 1.8 eV, sigma 0.5
                        r_gap = 5.0 * np.exp(-((g - 1.8) ** 2) / (2 * 0.5 ** 2))
                    else:
                        r_gap = 0.0 # No bonus for metals

                    # Total Reward
                    # Stable Metal = 1.0 + 0.0 = 1.0
                    # Stable Semiconductor = 1.0 + 5.0 = 6.0
                    # Unstable = -1.0
                    reward = r_stab + r_gap

                    # --- CACHE UPDATE ---
                    if "shash" in item and item["shash"]:
                        agent.reward_cache[item["shash"]] = reward

                    # Save "Winners" (Stable things)
                    if reward > 0.0:
                        stable_cnt += 1
                        agent.memory.append({'struct': final_s, 'reward': reward})
                        
                        best_form = f_str
                        
                        if (w_best is None) or (reward > w_best[0]):
                            w_best = (reward, f_str, e, g)
                            
                        # Save high-value candidates (Semi-conductors) to CSV
                        if reward > 2.0: 
                            with open("final_candidates.csv", "a", newline="") as f:
                                csv.writer(f).writerow([f_str, e, g, reward, epoch])
                            
                        # Save CIF for any stable crystal (limit 5 per epoch)
                        if stable_cnt <= 5:
                            fname = f"{disc_dir}/{f_str}_{epoch}.cif"
                            CifWriter(final_s).write_file(fname)

                # D. Failures
                elif item["result"] == "invalid":
                    reward = -1.0
                elif item["result"] == "diverged":
                    reward = -1.0
                elif item["result"] == "collapsed":
                    reward = -1.0
                
                rewards.append(reward)
  
                # Accumulate Loss
                if item["type"] == "gen" and item["result"] == "success":
                    log_sum = torch.stack(item["log_probs"]).sum()
                    
                    kl = F.kl_div(
                        F.log_softmax(item["logits_policy"], dim=-1),
                        F.softmax(item["logits_ref"].detach(), dim=-1),
                        reduction="batchmean"
                    )
                    
                    entropy_proxy = -log_sum / len(item["log_probs"])
                    
                    loss = -(reward * log_sum) + (CONFIG["KL_COEF"] * kl) - (CONFIG["ENTROPY_COEF"] * entropy_proxy)
                    loss_accum += loss

            if loss_accum.requires_grad:
                loss_accum.backward()
                torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), 1.0)
                agent.optimizer.step()

            avg_r = np.mean(rewards)
            epoch_time = time.time() - start_time
            
            # --- LOGGING ---
            pct_filt = (count_filtered / CONFIG["BATCH_SIZE"]) * 100
            pct_divg = (count_diverged / CONFIG["BATCH_SIZE"]) * 100
            pct_conv = (count_converged / CONFIG["BATCH_SIZE"]) * 100
            pct_dedup = (count_dedup / CONFIG["BATCH_SIZE"]) * 100
            
            with open("training_log.csv", "a", newline="") as f:
                csv.writer(f).writerow([epoch, avg_r, stable_cnt, best_form, epoch_time, 
                                        pct_filt, pct_divg, pct_conv, pct_dedup])
            
            # ---- WINDOW ACCUMULATION ----
            w_gen += CONFIG["BATCH_SIZE"]
            w_filt += count_filtered
            w_relx += (CONFIG["BATCH_SIZE"] - count_filtered)
            w_valid += count_converged
            w_target += stable_cnt
            w_reward += avg_r
            w_time += epoch_time

            # Step 4: Print ONE line every 10 epochs
            if (epoch + 1) % WINDOW == 0:
                e_end = epoch + 1
                e_start = e_end - WINDOW + 1 
            
                avg_reward = w_reward / WINDOW
                            
                if w_best:
                    # Best in window
                    r, bf, be, bg = w_best
                    best_str = f"⭐ {bf} (E={be:.2f} eV, Bg={bg:.2f} eV)"
                else:
                    best_str = "—"
            
                print(
                    f"[E{e_start}–{e_end}]  "
                    f"gen={w_gen} | filt={w_filt} | relx={w_relx} | "
                    f"valid={w_valid} | target={w_target} | "
                    f"R = {avg_reward:.2f} | {w_time:.1f}sec | {best_str}"
                )
            
                # ---- RESET WINDOW ----
                w_gen = w_filt = w_relx = w_valid = w_target = 0
                w_reward = 0.0
                w_time = 0.0
                w_best = None
                w_candidates = []

            agent.save_checkpoint(epoch, avg_r)

    except KeyboardInterrupt:
        print("\n🛑 Training Interrupted. Checkpoint saved.")
    finally:
        pool.close()
        pool.join()
        print("✅ Pipeline Stopped.")

if __name__ == "__main__":
    main()
