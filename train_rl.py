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
    "LR": 1e-4,             # Fast learning
    "EPOCHS": 150,
    "KL_COEF": 0.05,
    "ENTROPY_COEF": 0.2,    # HIGH ENTROPY: Force exploration
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "REPLAY_RATIO": 0.2,    # Lower replay so it doesn't just copy
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
        result = _relaxer.relax(struct)
        return (idx, result)
    except Exception as e:
        return (idx, {"is_converged": False, "error": str(e)})

# --- UTILS ---
def get_structure_hash(struct):
    try:
        formula = struct.composition.reduced_formula
        latt = tuple(np.round(struct.lattice.parameters, 2))
        coords = tuple(np.round(struct.frac_coords.flatten(), 2))
        return (formula, latt, coords)
    except: return None

def check_geometry_fast(struct):
    try:
        dm = struct.distance_matrix
        np.fill_diagonal(dm, 10.0)
        return np.min(dm) >= 0.6 # Looser filter
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
        self.reward_cache = {} 
        self.memory = deque(maxlen=5000) # Increased memory size
        
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
        
        # Fresh start logic
        checkpoint_path = os.path.join(ROOT, "checkpoint.pt")
        author_path = os.path.join(PRETRAINED_DIR, "epoch_005500_CLEAN.pt")
        
        if os.path.exists(checkpoint_path):
            print(f"🔄 Resuming from Checkpoint...")
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.policy.load_state_dict(ckpt['policy_state'])
            self.ref_model.load_state_dict(ckpt['ref_state'])
            self.optimizer.load_state_dict(ckpt['optimizer_state'])
            self.start_epoch = ckpt['epoch'] + 1
            self.memory = ckpt['memory'] 
            if 'reward_cache' in ckpt: self.reward_cache = ckpt['reward_cache']
        else:
            print("⚠️ Starting Fresh. Injecting Teacher Knowledge...")
            state = torch.load(author_path, map_location=self.device)
            self.policy.load_state_dict(state, strict=True)
            self.ref_model.load_state_dict(state, strict=True)
            self.inject_teacher_knowledge()

        self.ref_model.eval()
        self.idx_to_atom = {
            0: 30, 1: 16, 2: 48, 3: 34, 4: 8, 5: 31, 6: 33,
            7: 29, 8: 49, 9: 50, 10: 32, 11: 52, 12: 14
        }

    def inject_teacher_knowledge(self):
        seeds = []
        # GaAs (Zincblende) - Binary
        seeds.append(Structure(Lattice.cubic(5.65), ["Ga", "As"], [[0,0,0], [0.25,0.25,0.25]]))
        # MgO (Rock Salt) - Binary
        seeds.append(Structure(Lattice.cubic(4.21), ["Mg", "O"], [[0,0,0], [0.5,0.5,0.5]]))
        # ZnS (Zincblende) - Binary
        seeds.append(Structure(Lattice.cubic(5.41), ["Zn", "S"], [[0,0,0], [0.25,0.25,0.25]]))

        print(f"💉 Injecting {len(seeds)} Teacher Crystals...")
        # Inject MORE copies so they survive the flood
        for s in seeds:
            for _ in range(200): 
                self.memory.append({'struct': s, 'reward': 6.0}) 

    def save_checkpoint(self, epoch, current_reward):
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
    os.makedirs(os.path.join(ROOT, "rl_discoveries"), exist_ok=True)
    
    if agent.start_epoch == 0:
        with open("training_log.csv", "w", newline="") as f:
            csv.writer(f).writerow(["Epoch", "Reward", "Stable_Count", "Top_Formula", "Time_Sec", 
                             "Pct_Filtered", "Pct_Diverged", "Pct_Converged", "Pct_Dedup"])
        with open("final_candidates.csv", "w", newline="") as f:
            csv.writer(f).writerow(["Formula", "Formation_Energy", "Band_Gap", "Reward", "Epoch"])

    WINDOW = 10
    w_gen = w_filt = w_relx = w_valid = w_target = 0
    w_reward = 0.0
    w_time = 0.0
    w_best = None
    w_candidates = []

    print(f"\n🚀 STARTING PARALLEL PIPELINE: {CONFIG['EPOCHS']} Epochs")
    
    pool = mp.Pool(processes=CONFIG["NUM_WORKERS"])
    
    try:
        for epoch in range(agent.start_epoch, CONFIG["EPOCHS"]):
            start_time = time.time()
            count_filt = count_divg = count_conv = count_dedup = 0
            
            # --- STAGE 1: BATCH GENERATION ---
            batch_data = [] 
            for i in range(CONFIG["BATCH_SIZE"]):
                
                # REPLAY
                if len(agent.memory) > 100 and random.random() < CONFIG["REPLAY_RATIO"]:
                    mem = random.choice(agent.memory)
                    batch_data.append({"type": "replay", "struct": mem['struct'], "stored_reward": mem['reward']})
                    continue

                # GENERATION
                G_raw = random.randint(1, 230)
                num_atoms = random.randint(2, 6)
                # FIX: Large Lattice to allow complex chemistry
                lattice_guess = random.uniform(4.5, 7.5) 
                
                inputs = agent.prepare_input(G_raw, [[0.5]*3]*num_atoms, [0]*num_atoms, [0]*num_atoms, [1]*num_atoms)
                
                logits_policy = agent.policy(*inputs, is_train=False).squeeze(0)
                with torch.no_grad():
                    logits_ref = agent.ref_model(*inputs, is_train=False).squeeze(0)
                
                log_probs = []
                actions = []
                X_list = []
                
                for j in range(num_atoms):
                    base = 1 + 5 * j
                    
                    # 1. ATOM SAMPLING (MASKING ZINC)
                    atom_logits = logits_policy[base][:13].clone()
                    
                    # MASKING: Penalize Zinc (Index 0) to force exploration
                    # We don't ban it fully (-inf), but make it less likely
                    atom_logits[0] -= 2.0 
                    
                    atom_dist = torch.distributions.Categorical(logits=atom_logits)
                    atom_act = atom_dist.sample()
                    log_probs.append(atom_dist.log_prob(atom_act))
                    actions.append(agent.idx_to_atom.get(atom_act.item(), 6))
                    
                    # 2. Wyckoff
                    w_logits = logits_policy[base + 4][:agent.policy.wyck_types]
                    w_dist = torch.distributions.Categorical(logits=w_logits)
                    w_act = w_dist.sample()
                    log_probs.append(w_dist.log_prob(w_act))

                    # 3. Coordinate Sampling (Fixed)
                    Kx = agent.policy.Kx
                    def sample_c(blk):
                        w = blk[:Kx]
                        dist = torch.distributions.Categorical(logits=w)
                        idx = dist.sample()
                        log_probs.append(dist.log_prob(idx))
                        return torch.sigmoid(blk[Kx : 2*Kx][idx]).item()

                    X_list.append([
                        sample_c(logits_policy[base+1]),
                        sample_c(logits_policy[base+2]),
                        sample_c(logits_policy[base+3])
                    ])
                
                struct = build_structure(actions, X_list, lattice_guess)
                batch_data.append({
                    "type": "gen",
                    "log_probs": log_probs,
                    "logits_policy": logits_policy,
                    "logits_ref": logits_ref,
                    "struct": struct
                })

            # --- STAGE 2: PROCESS ---
            relax_tasks = []
            for idx, item in enumerate(batch_data):
                if item["type"] == "gen":
                    s = item["struct"]
                    if s is None:
                        item["result"] = "invalid"
                        count_filt += 1
                        continue
                        
                    shash = get_structure_hash(s)
                    if shash and shash in agent.reward_cache:
                        item["result"] = "cached"
                        item["cached_reward"] = agent.reward_cache[shash]
                        count_dedup += 1
                        continue
                        
                    if check_geometry_fast(s):
                        relax_tasks.append((idx, s))
                        item["shash"] = shash
                    else:
                        item["result"] = "invalid"
                        count_filt += 1
                else:
                    item["result"] = "replay"

            if relax_tasks:
                results = pool.map(worker_relax_task, relax_tasks)
                for (idx, res) in results:
                    batch_data[idx]["relax_res"] = res

            # --- STAGE 3: ORACLE ---
            oracle_tasks = []
            oracle_map = []
            for idx, item in enumerate(batch_data):
                if "relax_res" in item:
                    if item["relax_res"]["is_converged"]:
                        if check_geometry_fast(item["relax_res"]["final_structure"]):
                            oracle_tasks.append(item["relax_res"]["final_structure"])
                            oracle_map.append(idx)
                            count_conv += 1
                        else:
                            item["result"] = "collapsed"
                            count_divg += 1
                    else:
                        item["result"] = "diverged"
                        count_divg += 1
            
            if oracle_tasks:
                preds = oracle.predict_batch(oracle_tasks)
                for i, pred in enumerate(preds):
                    batch_data[oracle_map[i]].update({"oracle": pred, "result": "success"})

            # --- STAGE 4: REWARD & UPDATE ---
            rewards = []
            loss_accum = torch.tensor(0.0, device=agent.device)
            agent.optimizer.zero_grad()
            
            stable_cnt_batch = 0
            best_form_batch = "-"

            for item in batch_data:
                reward = -0.5 # Default small penalty
                
                if item["result"] == "cached": reward = item["cached_reward"]
                elif item["result"] == "replay": reward = item["stored_reward"]
                
                elif item["result"] == "success":
                    e = item["oracle"]["formation_energy"]
                    g = item["oracle"]["band_gap_scalar"]
                    final_s = item["relax_res"]["final_structure"]
                    f_str = final_s.composition.reduced_formula
                    
                    # --- THE NEW REWARD FUNCTION ---
                    
                    # 1. Base Stability (Allows Metals)
                    # We allow up to 0.1 eV instability to encourage experimentation
                    if e <= 0.1:
                        r_stab = 0.5  # Small reward for creating ANY stable crystal
                    else:
                        r_stab = -0.5
                    
                    # 2. Diversity Bonus (Penalize Pure Elements)
                    # If it's just Zn, or just S, we subtract points.
                    # We want MIXTURES (ZnS, CdSe).
                    if len(final_s.composition.elements) < 2:
                        r_div = -0.5 
                    else:
                        r_div = 0.5  # Bonus for mixing elements
                        
                    # 3. Band Gap Jackpot
                    if g > 0.1:
                        r_gap = 5.0 * np.exp(-((g - 1.8) ** 2) / (2 * 0.5 ** 2))
                    else:
                        r_gap = 0.0

                    # Total Reward Logic:
                    # Pure Zn (Stable) = 0.5 (Stab) - 0.5 (Div) + 0.0 (Gap) = 0.0  (Neutral)
                    # ZnS (Stable Semi) = 0.5 (Stab) + 0.5 (Div) + 5.0 (Gap) = 6.0 (JACKPOT)
                    reward = r_stab + r_div + r_gap

                    if "shash" in item and item["shash"]:
                        agent.reward_cache[item["shash"]] = reward
                    
                    # MEMORY PROTECTION: Only save high-quality stuff
                    if reward > 2.0:
                        agent.memory.append({'struct': final_s, 'reward': reward})
                        stable_cnt_batch += 1
                        best_form_batch = f_str
                        
                        # Save to CSV
                        with open("final_candidates.csv", "a", newline="") as f:
                            csv.writer(f).writerow([f_str, e, g, reward, epoch])
                        
                        # Save CIF
                        fname = f"{disc_dir}/{f_str}_{epoch}.cif"
                        CifWriter(final_s).write_file(fname)
                    
                    # DEBUG LOG
                    with open("relaxed_all.csv", "a", newline="") as f:
                        csv.writer(f).writerow([epoch, f_str, e, g])

                # D. Failures
                elif item["result"] in ["invalid", "diverged", "collapsed"]:
                    reward = -1.0 # Failures are worse than boring Zinc

                rewards.append(reward)

                if item["type"] == "gen" and item["result"] == "success":
                    log_sum = torch.stack(item["log_probs"]).sum()
                    kl = F.kl_div(F.log_softmax(item["logits_policy"], dim=-1), F.softmax(item["logits_ref"].detach(), dim=-1), reduction="batchmean")
                    entropy = -log_sum / len(item["log_probs"])
                    loss = -(reward * log_sum) + (CONFIG["KL_COEF"] * kl) - (CONFIG["ENTROPY_COEF"] * entropy)
                    loss_accum += loss

            if loss_accum.requires_grad:
                loss_accum.backward()
                torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), 1.0)
                agent.optimizer.step()

            avg_r = np.mean(rewards)
            
            # --- LOGGING ---
            with open("training_log.csv", "a", newline="") as f:
                csv.writer(f).writerow([epoch, avg_r, stable_cnt_batch, best_form_batch, time.time()-start_time, 0,0,0,0])

            # Window Tracking
            w_gen += CONFIG["BATCH_SIZE"]; w_valid += count_conv; w_target += stable_cnt_batch; w_reward += avg_r; w_time += (time.time()-start_time)
            if stable_cnt_batch > 0: 
                # Track best candidate in window (approx)
                if w_best is None or reward > 0: w_best = (reward, best_form_batch, 0, 0) # simplified

            if (epoch + 1) % WINDOW == 0:
                print(f"[E{epoch+1-WINDOW}-{epoch+1}] R={w_reward/WINDOW:.2f} | Valid={w_valid} | SemiCond={w_target} | {w_best[1] if w_best else '—'}")
                w_gen=w_valid=w_target=w_reward=w_time=0; w_best=None
                agent.save_checkpoint(epoch, avg_r)

    except KeyboardInterrupt: pass
    finally:
        pool.close(); pool.join()

if __name__ == "__main__":
    main()
