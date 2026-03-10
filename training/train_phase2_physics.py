import os
import sys
import torch
import pynvml
import warnings
import gc
import re
import time
import numpy as np
from tqdm import tqdm
import logging
import contextlib
import io

# =============================================================================
# 🚑 1. CRITICAL WINDOWS GPU SHIM
# =============================================================================
try:
    sys.modules['nvidia_smi'] = pynvml
    os.environ['PATH'] += os.pathsep + r'C:\Windows\System32'
except Exception:
    pass 

warnings.filterwarnings("ignore")
loggers_to_mute = ["chgnet", "dgl", "pymatgen"]
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
#        PHASE 2 CONFIGURATION
# ==========================================
MAX_ATOMS = 40             
BATCH_SIZE = 1             
GRAD_ACCUM_STEPS = 16     
STEPS_PER_EPOCH = 20        
EPOCHS = 60                 

LR_BODY = 1e-6             
LR_BIAS = 5e-4            

KL_BETA = 0.05               
BASELINE_GAMMA = 0.95        

CAMPAIGN_ELEMENTS = [
    3, 11, 19, 4, 12, 20, 38, 56, 5, 13, 31, 49,
    6, 14, 32, 50, 7, 15, 33, 51, 8, 16, 34, 52,
    21, 22, 30, 48
]

if "__file__" in locals():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
else:
    PROJECT_ROOT = os.getcwd()

sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "core"))
sys.path.append(os.path.join(PROJECT_ROOT, "rewards"))
sys.path.append(os.path.join(PROJECT_ROOT, "CrystalFormer"))

LOG_DIR = os.path.join(PROJECT_ROOT, "training_log")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "train_log_phase2.txt")

logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[
    logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
    logging.StreamHandler(sys.stdout)
])

try:
    from generator_service import CrystalGenerator
    from relaxer import CrystalRelaxer 
    from reward_phase2 import PhysicsRewardEngine
except ImportError as e:
    logging.error(f"❌ Import Error: {e}")
    sys.exit(1)

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

def get_empirical_spg_distribution():
    probs = torch.ones(230) * 0.01 
    common_sgs = [2, 12, 14, 15, 62, 139, 166, 194, 225, 227]
    for sg in common_sgs: probs[sg - 1] = 10.0 
    return probs / probs.sum()

def run_phase2():
    logging.info("🚀 STARTING PHASE 2: THE PHYSICIST")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir = os.path.join(PROJECT_ROOT, "rl_checkpoints", "phase_2")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    config_path = os.path.join(PROJECT_ROOT, "pretrained_model", "config.yaml")
    phase1_path = os.path.join(PROJECT_ROOT, "pretrained_model", "chemist_v2_epoch_100.pt")
    
    relaxer = CrystalRelaxer(device=device) 
    reward_engine = PhysicsRewardEngine(device=device)
    spg_distribution = get_empirical_spg_distribution().to(device)

    start_epoch = 1
    load_path = phase1_path
    existing_checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
    if existing_checkpoints:
        epochs_found = [int(re.search(r"epoch_(\d+).pt", f).group(1)) for f in existing_checkpoints if re.search(r"epoch_(\d+).pt", f)]
        if epochs_found:
            latest = max(epochs_found)
            start_epoch = latest + 1
            load_path = os.path.join(checkpoint_dir, f"physicist_epoch_{latest}.pt")
            logging.info(f"   🔄 Resuming from Phase 2 Checkpoint: Epoch {latest}")

    generator = CrystalGenerator(load_path, config_path, device)
    checkpoint = torch.load(load_path, map_location=device)
    
    # 🔹 THE FIX: Standalone Global Lattice Bias!
    # Initializes at 3.0 (exp(3.0) ≈ 20.08 Å³/atom), which is a perfect physical starting point.
    global_lattice_bias = torch.nn.Parameter(torch.tensor(3.0, device=device))
    if "lattice_bias_value" in checkpoint:
        global_lattice_bias.data.fill_(checkpoint["lattice_bias_value"])
        
    generator.model.train()
    
    frozen_generator = CrystalGenerator(phase1_path, config_path, device)
    frozen_generator.model.eval()
    for param in frozen_generator.parameters(): param.requires_grad = False

    # Pass our new standalone parameter to the optimizer
    body_params = [p for n, p in generator.model.named_parameters()]
    optimizer = torch.optim.Adam([
        {'params': body_params, 'lr': LR_BODY},
        {'params': [global_lattice_bias], 'lr': LR_BIAS}
    ])
    
    if start_epoch > 1:
        if "optimizer_state" in checkpoint: optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "running_baseline" in checkpoint: running_baseline = checkpoint["running_baseline"]
        else: running_baseline = -3.0
    else:
        running_baseline = -3.0

    total_steps = STEPS_PER_EPOCH * GRAD_ACCUM_STEPS
    optimizer.zero_grad(set_to_none=True)
    
    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_start_time = time.time()
        epoch_rewards, epoch_energies, epoch_kl = [], [], []
        
        pbar = tqdm(range(total_steps), desc=f"Ep {epoch}/{EPOCHS}", ncols=110)
        
        for step in pbar:
            try:
                G_target = torch.multinomial(spg_distribution, BATCH_SIZE, replacement=True) + 1
                outputs = generator.generate_with_grads(BATCH_SIZE, allowed_elements=CAMPAIGN_ELEMENTS, temperature=0.7, G=G_target)
                
                batch_rl_loss = torch.tensor(0.0, device=device)
                batch_bias_loss = torch.tensor(0.0, device=device)
                valid_items = 0

                for i in range(BATCH_SIZE):
                    struct = outputs["structures"][i]
                    log_prob_active = outputs["log_probs"][i]
                    G, XYZ, A, W = outputs["G"], outputs["XYZ"], outputs["A"], outputs["W"]
                    bias_loss = torch.tensor(0.0, device=device)
                    
                    if struct is None: continue
                    valid_items += 1

                    if len(struct) > MAX_ATOMS:
                        raw_reward = torch.tensor(-10.0, device=device)
                    else:
                        # 🔹 THE FIX: Dynamically inflate the tiny Phase 1 box to a real-world volume!
                        current_vol_per_atom = torch.exp(global_lattice_bias).item()
                        target_volume = current_vol_per_atom * len(struct)
                        struct.scale_lattice(target_volume) # Scales the bounding box perfectly in-place
                        
                        with QuietBlock():
                            relax_result = relaxer.relax(struct, steps=10)

                        if relax_result["converged"]:
                            raw_reward = reward_engine.compute_reward(
                                relax_result["final_structure"], 
                                relax_result["energy_per_atom"]
                            )
                            epoch_energies.append(relax_result["energy_per_atom"])
                            
                            # Train the bias multiplier using MSE
                            target_vol_per_atom = relax_result["relaxed_volume"] / len(relax_result["final_structure"])
                            target_bias = torch.tensor(np.log(target_vol_per_atom), device=device, dtype=torch.float32)
                            bias_loss = torch.nn.functional.mse_loss(global_lattice_bias, target_bias)
                        else:
                            raw_reward = torch.tensor(-5.0, device=device)

                    log_probs_frozen = compute_discrete_logp(frozen_generator, G[i:i+1], XYZ[i:i+1], A[i:i+1], W[i:i+1])
                    active_discrete_logp = compute_discrete_logp(generator, G[i:i+1], XYZ[i:i+1], A[i:i+1], W[i:i+1])
                    
                    kl_div = active_discrete_logp.squeeze() - log_probs_frozen.squeeze()
                    epoch_kl.append(kl_div.item())
                    
                    penalized_reward = raw_reward - (KL_BETA * kl_div.detach())
                    
                    if epoch == 1 and step == 0 and i == 0: running_baseline = penalized_reward.item()
                    else: running_baseline = (BASELINE_GAMMA * running_baseline) + ((1 - BASELINE_GAMMA) * penalized_reward.item())
                        
                    advantage = penalized_reward - running_baseline

                    batch_rl_loss += -(log_prob_active * advantage.detach())
                    batch_bias_loss += bias_loss
                    epoch_rewards.append(raw_reward.item())

                if valid_items == 0: continue

                total_loss = ((batch_rl_loss / valid_items) + (0.1 * (batch_bias_loss / valid_items))) / GRAD_ACCUM_STEPS
                total_loss.backward()

                if (step + 1) % GRAD_ACCUM_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(generator.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    
            except Exception as e:
                pbar.write(f"⚠️ Step Crash: {e}")
                if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    gc.collect()
                continue
                
        epoch_duration = time.time() - epoch_start_time
        minutes, seconds = int(epoch_duration // 60), int(epoch_duration % 60)
        avg_reward = np.mean(epoch_rewards) if epoch_rewards else 0.0
        avg_energy = np.mean(epoch_energies) if epoch_energies else 0.0
        avg_kl = np.mean(epoch_kl) if epoch_kl else 0.0
        
        log_msg = f"📊 Ep {epoch} | Time: {minutes:02d}:{seconds:02d} | Reward: {avg_reward:.3f} | KL: {avg_kl:.2f} | Energy: {avg_energy:.3f} eV | Bias: {global_lattice_bias.item():.4f}"
        logging.info(log_msg)
        
        if epoch % 5 == 0 or epoch == EPOCHS:
            save_path = os.path.join(checkpoint_dir, f"physicist_epoch_{epoch}.pt")
            torch.save({
                "model_state": generator.model.state_dict(), 
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch, 
                "lattice_bias_value": global_lattice_bias.item(),
                "running_baseline": running_baseline
            }, save_path)
            logging.info(f"   💾 Saved Physicist Checkpoint: {os.path.basename(save_path)}")
        
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    run_phase2()