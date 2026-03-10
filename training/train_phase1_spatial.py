import os
import sys
import torch
import warnings
import gc
import logging
import time
import numpy as np
from tqdm import tqdm

# =========================
# -------- CONFIG ---------
# =========================
MAX_ATOMS = 20
LR = 1e-4                  
EPOCHS = 100                 
BATCH_SIZE = 2              
GRAD_ACCUM_STEPS = 32       
STEPS_PER_EPOCH = 50        

# 🔹 NEW RL STABILITY CONFIG
KL_BETA = 0.05               # Weight of the KL Divergence penalty
BASELINE_GAMMA = 0.95        # EMA decay for the Advantage Baseline

CAMPAIGN_ELEMENTS = [
    3, 11, 19, 4, 12, 20, 38, 56,
    5, 13, 31, 49, 6, 14, 32, 50,
    7, 15, 33, 51, 8, 16, 34, 52,
    21, 22, 30, 48
]

# =========================
# ----- PATH SETUP --------
# =========================
if "__file__" in locals():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
else:
    SCRIPT_DIR = os.getcwd()

PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "CrystalFormer"))
sys.path.append(os.path.join(PROJECT_ROOT, "core"))
sys.path.append(os.path.join(PROJECT_ROOT, "rewards"))

LOG_DIR = os.path.join(PROJECT_ROOT, "training_log")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "train_log_phase1.txt")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "rl_checkpoints", "phase_1")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

try:
    from generator_service import CrystalGenerator
    from sentinel import CrystalSentinel
    from reward_phase1 import ChemistryRewardEngine
except ImportError as e:
    logging.error(f"❌ Import Error: {e}")
    sys.exit(1)

# =========================
# 🔹 HELPER: KL LOG PROBS
# =========================
# =========================
# 🔹 HELPER: KL LOG PROBS
# =========================
def compute_discrete_logp(generator, G, XYZ, A, W, requires_grad=False):
    """
    Computes the log probabilities of the discrete tokens (Atom & Wyckoff) 
    in a SINGLE pass. If requires_grad=True, it builds the graph for PPO.
    """
    B, N = A.shape
    G_exp = (G - 1).unsqueeze(1).expand(-1, N)
    M = generator.mult_table[G_exp, W]
    
    # Dynamically enable or disable gradients
    context = torch.enable_grad() if requires_grad else torch.no_grad()
    
    with context:
        h = generator.model(G, XYZ, A, W, M, is_train=requires_grad)
        w_logit = h[:, 0::5, :generator.wyck_types][:, :-1, :] # (B, N, wyck_types)
        a_logit = h[:, 1::5, :generator.atom_types]            # (B, N, atom_types)
        
        logp_w = torch.log_softmax(w_logit, dim=-1)
        logp_a = torch.log_softmax(a_logit, dim=-1)
        
        # Mask out padding tokens
        mask = (A != 0).float()
        w_lp = torch.gather(logp_w, 2, W.unsqueeze(-1)).squeeze(-1) * mask
        a_lp = torch.gather(logp_a, 2, A.unsqueeze(-1)).squeeze(-1) * mask
        
    return w_lp.sum(dim=1) + a_lp.sum(dim=1)
# =========================
# 🔹 HELPER: EMPIRICAL SPG
# =========================
def get_empirical_spg_distribution():
    """ Creates a probability distribution favoring stable/common space groups """
    probs = torch.ones(230) * 0.01 # Base low probability for rare groups
    common_sgs = [2, 12, 14, 15, 62, 139, 166, 194, 225, 227]
    for sg in common_sgs:
        probs[sg - 1] = 10.0 # Heavy weight for common groups
    return probs / probs.sum()

# =========================
# ----- TRAIN LOOP --------
# =========================
def run_phase1_chemist():
    logging.info("🚀 STARTING PHASE 1 v2.0: THE CHEMIST (PPO-Anchored Mode)")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config_path = os.path.join(PROJECT_ROOT, "pretrained_model", "config.yaml")
    base_model_path = os.path.join(PROJECT_ROOT, "pretrained_model", "epoch_005500_CLEAN.pt")
    
    # 1. Active Model
    generator = CrystalGenerator(base_model_path, config_path, device)
    generator.model.train()
    optimizer = torch.optim.Adam(generator.parameters(), lr=LR)
    
    # 2. 🔹 Frozen Base Model (The Chemical Anchor)
    frozen_generator = CrystalGenerator(base_model_path, config_path, device)
    frozen_generator.model.eval()
    for param in frozen_generator.parameters():
        param.requires_grad = False
    
    sentinel = CrystalSentinel(device)
    reward_engine = ChemistryRewardEngine()
    spg_distribution = get_empirical_spg_distribution().to(device)
    
    optimizer.zero_grad() 
    
    # 🔹 Running Advantage Baseline
    running_baseline = 0.0 
    
    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        
        epoch_rewards, epoch_kl, epoch_vols = [], [], []
        pbar = tqdm(range(STEPS_PER_EPOCH), desc=f"Epoch {epoch}/{EPOCHS}", file=sys.stdout)
        
        for step in pbar:
            # 🔹 Sample Space Groups Empirically
            G_target = torch.multinomial(spg_distribution, BATCH_SIZE, replacement=True) + 1
            
            # 1. GENERATE WITHOUT GRADS (Saves 95% of VRAM)
            outputs = generator.generate(
                BATCH_SIZE, 
                allowed_elements=CAMPAIGN_ELEMENTS, 
                G=G_target 
            )
            
            structures = outputs["structures"]
            G, XYZ, A, W = outputs["G"], outputs["XYZ"], outputs["A"], outputs["W"]
            
            sentinel_results = sentinel.filter(structures)
            raw_rewards = reward_engine.compute_reward(structures, sentinel_results).to(device)
            
            # 2. Get Frozen logp (No gradients)
            log_probs_frozen = compute_discrete_logp(frozen_generator, G, XYZ, A, W, requires_grad=False)
            
            # 3. Get Active logp (WITH GRADIENTS FOR PPO)
            log_probs_active = compute_discrete_logp(generator, G, XYZ, A, W, requires_grad=True)
            
            # 4. Compute KL Divergence Penalty
            kl_div = log_probs_active - log_probs_frozen
            
            # Penalize the reward if the model deviates too far from base chemistry
            penalized_rewards = raw_rewards - (KL_BETA * kl_div.detach())
            
            # 5. Compute Advantage
            if epoch == 1 and step == 0:
                running_baseline = penalized_rewards.mean().item()
            else:
                running_baseline = (BASELINE_GAMMA * running_baseline) + ((1 - BASELINE_GAMMA) * penalized_rewards.mean().item())
                
            advantage = penalized_rewards - running_baseline

            # 6. Calculate Loss (Policy Gradient with Advantage)
            loss = -(log_probs_active * advantage.detach()).mean() / GRAD_ACCUM_STEPS
            loss.backward()
            
            # Logging Stats
            for i in range(BATCH_SIZE):
                if struct := structures[i]:
                    epoch_vols.append(sentinel_results[i].get("volume_per_atom", 0))
                epoch_rewards.append(raw_rewards[i].item())
                epoch_kl.append(kl_div[i].item())
            
            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(generator.model.parameters(), max_norm=0.5)
                optimizer.step()
                optimizer.zero_grad() 
                
            if step % 5 == 0:
                pbar.set_postfix({
                    "R": f"{np.mean(epoch_rewards[-BATCH_SIZE:]):.2f}", 
                    "Adv": f"{advantage.mean().item():.2f}",
                    "KL": f"{np.mean(epoch_kl[-BATCH_SIZE:]):.2f}"
                })
            
            del outputs, sentinel_results, raw_rewards, kl_div, penalized_rewards, advantage, loss
            torch.cuda.empty_cache()
            gc.collect()

        epoch_duration = time.time() - epoch_start_time
        time_str = f"{int(epoch_duration // 60):02d}:{int(epoch_duration % 60):02d}"

        avg_r = np.mean(epoch_rewards) if epoch_rewards else 0.0
        avg_v = np.mean(epoch_vols) if epoch_vols else 0.0
        avg_kl = np.mean(epoch_kl) if epoch_kl else 0.0
        
        epoch_summary = f"📊 Ep {epoch} | Time: {time_str} | Avg R: {avg_r:.2f} | Avg KL: {avg_kl:.2f} | Avg Vol: {avg_v:.1f} | Baseline: {running_baseline:.2f}"
        logging.info(epoch_summary)
        
        if epoch % 5 == 0:
            save_path = os.path.join(CHECKPOINT_DIR, f"chemist_v2_epoch_{epoch}.pt")
            torch.save({
                "model_state": generator.model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "running_baseline": running_baseline
            }, save_path)
            logging.info(f"   💾 Checkpoint Saved: epoch_{epoch}")

if __name__ == "__main__":
    run_phase1_chemist()