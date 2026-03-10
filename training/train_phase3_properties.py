import os
import sys
import torch
import warnings
import gc
import math
import time
import re
import numpy as np
from tqdm import tqdm
import logging
import contextlib
import io
import pynvml

# =============================================================================
# 🚑 DRIVERS & SHIMS
# =============================================================================
try:
    sys.modules['nvidia_smi'] = pynvml
    os.environ['PATH'] += os.pathsep + r'C:\Windows\System32'
except Exception:
    pass 

import ase.constraints
sys.modules["ase.filters"] = ase.constraints
if not hasattr(ase.constraints, "ExpCellFilter"):
    if hasattr(ase.constraints, "UnitCellFilter"):
        ase.constraints.ExpCellFilter = ase.constraints.UnitCellFilter

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

loggers_to_mute = ["chgnet", "dgl", "pymatgen", "matgl", "matminer"]
for name in logging.root.manager.loggerDict:
    if any(mute_key in name.lower() for mute_key in loggers_to_mute):
        logger = logging.getLogger(name)
        logger.setLevel(logging.CRITICAL) 
        logger.propagate = False

class QuietBlock:
    def __enter__(self):
        self._suppress = io.StringIO()
        self._redirect_out = contextlib.redirect_stdout(self._suppress)
        self._redirect_err = contextlib.redirect_stderr(self._suppress)
        self._redirect_out.__enter__()
        self._redirect_err.__enter__()
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._redirect_out.__exit__(exc_type, exc_val, exc_tb)
        self._redirect_err.__exit__(exc_type, exc_val, exc_tb)

# ==========================================
# ⚙️ PHASE 3 CONFIGURATION 
# ==========================================
PRODUCT_NAME = "lab_grade_Semiconductor"           
MAX_ATOMS = 40         # Allow larger supercells, relying on our OOM safety net
IDEAL_GAP = 1.5        # 🔹 ADAPTIVE TARGET (Peak Reward at 1.5 eV)    

BATCH_SIZE = 1           
GRAD_ACCUM_STEPS = 16    
STEPS_PER_EPOCH = 20        
EPOCHS = 100                 
LR_BODY = 1e-6             

# 🔹 SNAPPED THE LEASH: Allows the AI to radically change chemistry without heavy punishment
KL_BETA = 0.05               
BASELINE_GAMMA = 0.95        
PPO_EPSILON = 0.2            

CAMPAIGN_ELEMENTS = [
    3, 11, 19, 4, 12, 20, 38, 56, 5, 13, 31, 49,
    6, 14, 32, 50, 7, 15, 33, 51, 8, 16, 34, 52,
    21, 22, 30, 48
]

PROJECT_ROOT = os.getcwd() 
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "core"))
sys.path.append(os.path.join(PROJECT_ROOT, "rewards"))
sys.path.append(os.path.join(PROJECT_ROOT, "CrystalFormer"))

LOG_DIR = os.path.join(PROJECT_ROOT, "training_log")
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[
    logging.FileHandler(os.path.join(LOG_DIR, "train_log_phase3.txt"), mode='a', encoding='utf-8'),
    logging.StreamHandler(sys.stdout)
])

try:
    from generator_service import CrystalGenerator
    from sentinel import CrystalSentinel
    from relaxer import CrystalRelaxer   
    from oracle import Phase3Oracle 
    from reward_phase3 import EngineerRewardEngine
except ImportError as e:
    sys.exit(f"❌ Import Error: {e}")

def compute_discrete_logp(generator, G, XYZ, A, W):
    B, N = A.shape
    G_exp = (G - 1).unsqueeze(1).expand(-1, N)
    M = generator.mult_table[G_exp, W]
    with torch.no_grad():
        h = generator.model(G, XYZ, A, W, M, is_train=False)
        w_logit = h[:, 0::5, :generator.wyck_types][:, :-1, :] 
        a_logit = h[:, 1::5, :generator.atom_types]            
        logp_w = torch.log_softmax(w_logit, dim=-1)
        logp_a = torch.log_softmax(a_logit, dim=-1)
        mask = (A != 0).float()
        w_lp = torch.gather(logp_w, 2, W.unsqueeze(-1)).squeeze(-1) * mask
        a_lp = torch.gather(logp_a, 2, A.unsqueeze(-1)).squeeze(-1) * mask
    return w_lp.sum(dim=1) + a_lp.sum(dim=1)

def run_phase3():
    logging.info(f"🚀 STARTING PHASE 3: ENGINEER (Adaptive Target: {IDEAL_GAP} eV)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir = os.path.join(PROJECT_ROOT, "rl_checkpoints", "phase_3")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    config_path = os.path.join(PROJECT_ROOT, "pretrained_model", "config.yaml")
    phase2_path = os.path.join(PROJECT_ROOT, "pretrained_model", "physicist_epoch_60.pt")
    
    sentinel = CrystalSentinel(device)
    with QuietBlock():
        relaxer = CrystalRelaxer(device=device) 
        oracle = Phase3Oracle()
    reward_engine = EngineerRewardEngine(ideal_gap=IDEAL_GAP)

    # 🔹 CHECKPOINT RESUME MECHANISM
    start_epoch = 1
    load_path = phase2_path 
    
    existing_checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt") and PRODUCT_NAME in f]
    if existing_checkpoints:
        epochs_found = [int(re.search(r"epoch_(\d+).pt", f).group(1)) for f in existing_checkpoints if re.search(r"epoch_(\d+).pt", f)]
        if epochs_found:
            latest = max(epochs_found)
            start_epoch = latest + 1
            load_path = os.path.join(checkpoint_dir, f"{PRODUCT_NAME}_epoch_{latest}.pt")
            logging.info(f"   🔄 Resuming from Phase 3 Checkpoint: Epoch {latest}")

    generator = CrystalGenerator(load_path, config_path, device)
    checkpoint = torch.load(load_path, map_location=device)
    lattice_bias_val = checkpoint.get("lattice_bias_value", 3.0) 
    generator.model.train()
    
    frozen_generator = CrystalGenerator(phase2_path, config_path, device)
    frozen_generator.model.eval()
    for param in frozen_generator.parameters(): param.requires_grad = False

    optimizer = torch.optim.Adam(generator.model.parameters(), lr=LR_BODY)
    if "optimizer_state" in checkpoint and start_epoch > 1: 
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    
    running_baseline = checkpoint.get("running_baseline", 5.0)

    total_steps = STEPS_PER_EPOCH * GRAD_ACCUM_STEPS
    
    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_start_time = time.time()
        epoch_rewards, epoch_gaps = [], []
        
        pbar = tqdm(range(total_steps), desc=f"Ep {epoch}/{EPOCHS}", ncols=110)
        optimizer.zero_grad(set_to_none=True)
        
        for step in pbar:
            outputs = None
            try:
                outputs = generator.generate_with_grads(BATCH_SIZE, allowed_elements=CAMPAIGN_ELEMENTS, temperature=0.7)
                batch_rl_loss = torch.tensor(0.0, device=device)
                valid_items = 0

                for i in range(BATCH_SIZE):
                    struct = outputs["structures"][i]
                    log_prob_old = outputs["log_probs"][i].detach() 
                    G, XYZ, A, W = outputs["G"], outputs["XYZ"], outputs["A"], outputs["W"]
                    
                    if struct is None or len(struct) > MAX_ATOMS or not sentinel.filter([struct])[0].get("valid", False):
                        continue

                    struct.scale_lattice(math.exp(lattice_bias_val) * len(struct))
                    
                    with QuietBlock():
                        with torch.enable_grad():
                            relax_res = relaxer.relax(struct, steps=15)

                    if relax_res["final_structure"] is None: 
                        continue
                        
                    valid_items += 1
                    
                    final_struct = relax_res["final_structure"]
                    energy_pa = relax_res["energy_per_atom"]
                    
                    gap = oracle.evaluate_structure(final_struct)
                    
                    scores = reward_engine.compute_reward(energy_pa, gap)
                    raw_reward = torch.tensor(scores["total_reward"], device=device)
                    
                    epoch_gaps.append(gap)

                    log_probs_frozen = compute_discrete_logp(frozen_generator, G[i:i+1], XYZ[i:i+1], A[i:i+1], W[i:i+1])
                    active_discrete_logp = compute_discrete_logp(generator, G[i:i+1], XYZ[i:i+1], A[i:i+1], W[i:i+1])
                    
                    kl_div = active_discrete_logp.squeeze() - log_probs_frozen.squeeze()
                    penalized_reward = raw_reward - (KL_BETA * kl_div.detach())
                    
                    running_baseline = (BASELINE_GAMMA * running_baseline) + ((1 - BASELINE_GAMMA) * penalized_reward.item())
                    advantage = penalized_reward - running_baseline

                    ratio = torch.exp(outputs["log_probs"][i] - log_prob_old)
                    surr1 = ratio * advantage.detach()
                    surr2 = torch.clamp(ratio, 1.0 - PPO_EPSILON, 1.0 + PPO_EPSILON) * advantage.detach()
                    
                    batch_rl_loss += -torch.min(surr1, surr2)
                    epoch_rewards.append(raw_reward.item())

                if valid_items == 0: 
                    del outputs
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    continue

                total_loss = (batch_rl_loss / valid_items) / GRAD_ACCUM_STEPS
                total_loss.backward()

                del outputs
                if torch.cuda.is_available(): torch.cuda.empty_cache()

                if (step + 1) % GRAD_ACCUM_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(generator.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    
            except Exception as e:
                pbar.write(f"⚠️ Batch Error: {str(e)[:100]}...")
                if torch.cuda.is_available(): 
                    torch.cuda.empty_cache()
                gc.collect()
                continue
                
        mins, secs = int((time.time() - epoch_start_time) // 60), int((time.time() - epoch_start_time) % 60)
        avg_reward = np.mean(epoch_rewards) if epoch_rewards else 0.0
        avg_gap = np.mean(epoch_gaps) if epoch_gaps else 0.0
        
        logging.info(f"📊 Ep {epoch} | Time: {mins:02d}:{secs:02d} | Rwrd: {avg_reward:.2f} | Gap: {avg_gap:.2f}eV")
        
        if epoch % 5 == 0:
            torch.save({
                "model_state": generator.model.state_dict(), 
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "lattice_bias_value": lattice_bias_val,
                "running_baseline": running_baseline
            }, os.path.join(checkpoint_dir, f"{PRODUCT_NAME}_epoch_{epoch}.pt"))

if __name__ == "__main__":
    run_phase3()