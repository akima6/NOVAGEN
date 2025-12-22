import sys
import types

# ==============================================================================
# 🩹 CRITICAL MONKEY PATCH for PyTorch/TorchData Compatibility
# Fixes "No module named 'torch.utils._import_utils'"
# ==============================================================================
try:
    import torch.utils._import_utils
except ImportError:
    dummy_utils = types.ModuleType("torch.utils._import_utils")
    dummy_utils.dill_available = lambda: False
    sys.modules["torch.utils._import_utils"] = dummy_utils
    import torch.utils
    torch.utils._import_utils = dummy_utils
# ==============================================================================

import os
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
import copy

# --- PATH SETUP ---
ROOT = os.path.dirname(os.path.abspath("__file__"))
sys.path.append(os.path.join(ROOT, "CrystalFormer"))

# --- IMPORTS ---
from crystalformer.src.transformer import make_transformer
from pymatgen.core import Lattice, Structure, Element
from pymatgen.io.cif import CifWriter
from relaxer import Relaxer 
from oracle import Oracle   

# --- CONFIGURATION ---
PRETRAINED_DIR = os.path.join(ROOT, "pretrained_model")
CONFIG = {
    "BATCH_SIZE": 16,       
    "LR": 1e-4,             
    "EPOCHS": 500,          # Longer run for Curriculum
    "KL_COEF": 0.05,
    "ENTROPY_COEF": 0.1,    
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "EPSILON": 0.20,        
    "NUM_WORKERS": 1        
}

# --- ATOMIC RADII LOOKUP (For Smart Lattice) ---
ATOM_RADII = {
    "H": 0.31, "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
    "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02,
    "K": 2.03, "Ca": 1.76, "Sc": 1.70, "Ti": 1.60, "V": 1.53, "Cr": 1.39, "Mn": 1.39, "Fe": 1.32,
    "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22, "Ga": 1.22, "Ge": 1.20, "As": 1.19, "Se": 1.20,
    "Br": 1.20, "Rb": 2.20, "Sr": 1.95, "Y": 1.90, "Zr": 1.75, "Nb": 1.64, "Mo": 1.54, "Tc": 1.47,
    "Ru": 1.46, "Rh": 1.42, "Pd": 1.39, "Ag": 1.45, "Cd": 1.44, "In": 1.42, "Sn": 1.39, "Sb": 1.39,
    "Te": 1.38, "I": 1.39
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

# --- SMART BUILDER ---
def estimate_lattice_parameter(species_list):
    """Guesses a reasonable lattice size based on atomic radii sum."""
    if not species_list: return 4.0
    try:
        radii = [ATOM_RADII.get(str(s), 1.5) for s in species_list]
        avg_r = sum(radii) / len(radii)
        # Empirical heuristic: Box size ~ 4 * average radius for small cells
        guess = 4.0 * avg_r
        return max(3.0, min(guess, 10.0))
    except:
        return 5.0

def build_structure(A, X, forced_lattice=None):
    species = [a for a in A if a != 0]
    coords = [x for a, x in zip(A, X) if a != 0]
    if len(species) < 1: return None
    try:
        if forced_lattice:
            a = b = c = forced_lattice
        else:
            # Smart Guess
            a = b = c = estimate_lattice_parameter(species)
            
        lattice = Lattice.from_parameters(a, b, c, 90, 90, 90)
        return Structure(lattice, species, coords)
    except: return None

# --- AGENT ---
class PPOAgent_Product:
    def __init__(self):
        print(f"--- Initializing Product-Grade Agent (Device: {CONFIG['DEVICE']}) ---")
        self.device = CONFIG["DEVICE"]
        
        # Hall of Fame: Stores (Structure, Formula, Reward)
        self.hall_of_fame = deque(maxlen=50) 
        
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
        
        # Load Weights (Checkpoint or Fresh)
        checkpoint_path = os.path.join(ROOT, "checkpoint.pt")
        author_path = os.path.join(PRETRAINED_DIR, "epoch_005500_CLEAN.pt")
        
        if os.path.exists(checkpoint_path):
            print(f"🔄 Resuming from Checkpoint...")
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.policy.load_state_dict(ckpt['policy_state'])
            self.ref_model.load_state_dict(ckpt['ref_state'])
            self.optimizer.load_state_dict(ckpt['optimizer_state'])
            if 'hall_of_fame' in ckpt: self.hall_of_fame = ckpt['hall_of_fame']
        else:
            print("⚠️ Starting Fresh from Author Weights.")
            state = torch.load(author_path, map_location=self.device)
            self.policy.load_state_dict(state, strict=True)
            self.ref_model.load_state_dict(state, strict=True)

        self.ref_model.eval()

        self.idx_to_atom = {
            0: 30, 1: 16, 2: 48, 3: 34, 4: 8, 5: 31, 6: 33,
            7: 29, 8: 49, 9: 50, 10: 32, 11: 52, 12: 14,
            # Adding common extensions for mutation logic 
            13: 12, 14: 20 
        }
        self.atom_keys = list(self.idx_to_atom.keys())

    def save_checkpoint(self, epoch):
        ckpt = {
            'epoch': epoch,
            'policy_state': self.policy.state_dict(),
            'ref_state': self.ref_model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'hall_of_fame': self.hall_of_fame
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

# --- MAIN ---
def main():
    mp.set_start_method('spawn', force=True)
    agent = PPOAgent_Product()
    print("🔮 Initializing Oracle...")
    oracle = Oracle() # CPU safe, no args
    
    disc_dir = os.path.join(ROOT, "rl_discoveries")
    os.makedirs(disc_dir, exist_ok=True)
    
    # Init Logs
    if not os.path.exists("training_log.csv"):
        with open("training_log.csv", "w") as f:
            csv.writer(f).writerow(["Epoch", "Avg_Reward", "Valid_Count", "Best_Formula", "Mode"])
        with open("final_candidates.csv", "w") as f:
            csv.writer(f).writerow(["Formula", "Formation_Energy", "Band_Gap", "Reward", "Epoch"])
        with open("relaxed_all.csv", "w") as f:
            csv.writer(f).writerow(["Epoch", "Formula", "Energy", "BandGap", "Reward"])

    print(f"\n🚀 STARTING PRODUCT ENGINE: {CONFIG['EPOCHS']} Epochs")
    pool = mp.Pool(processes=CONFIG["NUM_WORKERS"])
    
    try:
        for epoch in range(CONFIG["EPOCHS"]):
            start_time = time.time()
            batch_data = []
            
            # --- CURRICULUM LOGIC ---
            if epoch < 50:
                complexity = 2
                mode = "Binary"
            elif epoch < 150:
                complexity = 3
                mode = "Ternary"
            else:
                complexity = 4
                mode = "Complex"
                
            # --- GENERATION ---
            for i in range(CONFIG["BATCH_SIZE"]):
                
                # FEATURE: EVOLUTIONARY MUTATION (20% chance)
                if len(agent.hall_of_fame) > 5 and random.random() < 0.20:
                    parent = random.choice(agent.hall_of_fame) 
                    p_struct = parent[0]
                    
                    new_species = []
                    for s in p_struct.species:
                        if random.random() < 0.5: 
                            new_s = agent.idx_to_atom.get(random.choice(agent.atom_keys), 6)
                        else:
                            new_s = s
                        new_species.append(new_s)
                    
                    try:
                        mutant_struct = Structure(p_struct.lattice, new_species, p_struct.frac_coords)
                        batch_data.append({"type": "mutation", "struct": mutant_struct, "parent": parent[1]})
                        continue 
                    except: pass 

                # STANDARD GENERATION
                G_raw = random.randint(1, 230)
                num_atoms = random.randint(complexity, complexity+2) 
                
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
                
                struct = build_structure(actions, X_list, forced_lattice=None)
                batch_data.append({
                    "type": "gen",
                    "struct": struct,
                    "log_probs": log_probs,
                    "logits_policy": logits_policy,
                    "logits_ref": logits_ref
                })

            # --- RELAXATION ---
            relax_tasks = []
            for idx, item in enumerate(batch_data):
                if item["struct"]:
                    relax_tasks.append((idx, item["struct"]))
                else:
                    item["result"] = "invalid"

            if relax_tasks:
                results = pool.map(worker_relax_task, relax_tasks)
                for (idx, res) in results:
                    batch_data[idx]["relax_res"] = res

            # --- ORACLE ---
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

            # --- REWARD ---
            agent.optimizer.zero_grad()
            loss_accum = torch.tensor(0.0, device=agent.device)
            rewards = []
            valid_cnt = 0
            best_f = "-"
            
            for item in batch_data:
                reward = -2.0
                
                if item["result"] == "success":
                    final_s = item["relax_res"]["final_structure"]
                    form = final_s.composition.reduced_formula
                    e = item["oracle"]["formation_energy"]
                    g = item["oracle"]["band_gap_scalar"]
                    
                    # PRODUCT-GRADE REWARD
                    if e < 0.2: 
                        r_stab = 1.0
                    else:
                        r_stab = -0.5
                    
                    if g > 0.5: r_gap = 5.0
                    elif g > 0.1: r_gap = 1.0
                    else: r_gap = 0.0
                    
                    if len(final_s.composition.elements) < 2:
                        r_nov = -2.0
                    else:
                        r_nov = 0.5 
                        
                    reward = r_stab + r_gap + r_nov
                    valid_cnt += 1
                    best_f = form
                    
                    with open("relaxed_all.csv", "a") as f:
                        csv.writer(f).writerow([epoch, form, f"{e:.4f}", f"{g:.4f}", f"{reward:.2f}"])
                    
                    if reward > 1.0:
                        agent.hall_of_fame.append((final_s, form, reward))
                        with open("final_candidates.csv", "a") as f:
                            csv.writer(f).writerow([form, e, g, reward, epoch])
                            
                    if reward > 4.0:
                        fname = f"{disc_dir}/{form}_{epoch}.cif"
                        CifWriter(final_s).write_file(fname)

                rewards.append(reward)

                if item["type"] == "gen" and item["result"] in ["success", "diverged"]:
                     log_sum = torch.stack(item["log_probs"]).sum()
                     kl = F.kl_div(F.log_softmax(item["logits_policy"], dim=-1), F.softmax(item["logits_ref"].detach(), dim=-1), reduction="batchmean")
                     loss = -(reward * log_sum) + (CONFIG["KL_COEF"] * kl)
                     loss_accum += loss

            if loss_accum.requires_grad:
                loss_accum.backward()
                torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), 1.0)
                agent.optimizer.step()
                
            avg_r = np.mean(rewards) if rewards else 0
            
            with open("training_log.csv", "a") as f:
                csv.writer(f).writerow([epoch, avg_r, valid_cnt, best_f, mode])
            
            if (epoch+1) % 10 == 0:
                print(f"[E{epoch+1}] Mode={mode} | R={avg_r:.2f} | Valid={valid_cnt} | Best={best_f} | HoF_Size={len(agent.hall_of_fame)}")
                agent.save_checkpoint(epoch)

    except KeyboardInterrupt: pass
    finally:
        pool.close(); pool.join()

if __name__ == "__main__":
    main()
