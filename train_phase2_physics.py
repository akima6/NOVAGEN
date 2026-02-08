import os
import sys
import torch
import warnings
import gc
import re
import numpy as np
from tqdm import tqdm
import logging
import contextlib
import io

# ==========================================
# 🔇 SILENCING TOOLS (Keep Console Clean)
# ==========================================
warnings.filterwarnings("ignore")

# Mute specific noisy libraries
loggers_to_mute = ["chgnet", "dgl", "pymatgen"]
for name in logging.root.manager.loggerDict:
    for mute_key in loggers_to_mute:
        if mute_key in name.lower():
            logger = logging.getLogger(name)
            logger.setLevel(logging.CRITICAL) 
            logger.propagate = False

class QuietBlock:
    """Context manager to silence C-level print outputs from physics engines"""
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
# "UNCHAINED MODE" (Creativity + Active Teaching)

MAX_ATOMS =40             
BATCH_SIZE = 2             

GRAD_ACCUM_STEPS = 16       
STEPS_PER_EPOCH = 10        

EPOCHS = 60                 
LR_BODY = 1e-5              
LR_BIAS = 1e-3              
ENTROPY_WEIGHT = 0.05       

CAMPAIGN_ELEMENTS = [
    3, 11, 19, 4, 12, 20, 38, 56, 5, 13, 31, 49,
    6, 14, 32, 50, 7, 15, 33, 51, 8, 16, 34, 52,
    21, 22, 30, 48
]

# ==========================================
#           PATH & IMPORT SETUP
# ==========================================
if "__file__" in locals():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
else:
    BASE_DIR = os.getcwd()

sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "CrystalFormer"))

try:
        from generator_service import CrystalGenerator
        from product_relaxer import CrystalRelaxer
        from reward_phase2 import PhysicsRewardEngine
except ImportError as e:
    sys.exit(f"❌ Import Error: {e}")

# ==========================================
#           TRAINING LOOP
# ==========================================
def run_phase2():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    checkpoint_dir = os.path.join("rl_checkpoints", "phase2")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"🚀 STARTING PHASE 2: THERMODYNAMIC OPTIMIZATION")
    print(f"   Device: {device}")
    print(f"   Engine: CHGNet (GPU-Accelerated)")
    print(f"   Mode:   Unchained (Active Penalty Learning)")

    # 1. LOAD MODEL (Ideally Phase 1 v3)
    config_path = os.path.join(BASE_DIR, "pretrained_model", "config.yaml")
    phase1_path = os.path.join(BASE_DIR, "rl_checkpoints", "phase1_v3", "spatial_v3_epoch_60.pt")
    fallback_path = os.path.join(BASE_DIR, "pretrained_model", "spatial_v3_epoch_60.pt")
    
    if os.path.exists(phase1_path):
        model_path = phase1_path
        print(f"   ✅ Found Phase 1 Model: {os.path.basename(model_path)}")
    elif os.path.exists(fallback_path):
        model_path = fallback_path
        print(f"   ⚠️ Using fallback model: {os.path.basename(model_path)}")
    else:
        sys.exit(f"❌ Critical Error: No model file found.")

    generator = CrystalGenerator(model_path, config_path, device)
    checkpoint = torch.load(model_path, map_location=device)
    if "lattice_bias_value" in checkpoint:
        generator.lattice_bias.data.fill_(checkpoint["lattice_bias_value"])
        print(f"   ✅ Restored Lattice Bias: {checkpoint['lattice_bias_value']:.4f}")

    generator.model.train()

    # 2. OPTIMIZER
    bias_params = [generator.lattice_bias]
    body_params = [p for n, p in generator.model.named_parameters()] 
    optimizer = torch.optim.Adam([
        {'params': body_params, 'lr': LR_BODY},
        {'params': bias_params, 'lr': LR_BIAS}
    ])

    # 3. INITIALIZE COMPONENTS
    print("   Initializing Physics Engine...")
    # Using default init which triggers the 'Smart Relaxer' (CHGNet)
    relaxer = CrystalRelaxer(device=device) 
    reward_engine = PhysicsRewardEngine(device=device)

    # 4. RESUME LOGIC
    start_epoch = 1
    existing_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
    if existing_files:
        epochs_found = [int(re.search(r"epoch_(\d+).pt", f).group(1)) for f in existing_files if re.search(r"epoch_(\d+).pt", f)]
        if epochs_found:
            latest = max(epochs_found)
            print(f"🔄 Resuming from Epoch {latest}...")
            checkpoint = torch.load(os.path.join(checkpoint_dir, f"physics_expert_epoch_{latest}.pt"), map_location=device)
            if "model_state" in checkpoint:
                generator.model.load_state_dict(checkpoint["model_state"])
                optimizer.load_state_dict(checkpoint["optimizer_state"])
                if "lattice_bias_value" in checkpoint:
                    generator.lattice_bias.data.fill_(checkpoint["lattice_bias_value"])
            else:
                generator.model.load_state_dict(checkpoint)
            start_epoch = latest + 1

    # 5. MAIN LOOP
    total_steps = STEPS_PER_EPOCH * GRAD_ACCUM_STEPS
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_rewards = []
        epoch_energies = []
        
        pbar = tqdm(range(total_steps), desc=f"Ep {epoch}/{EPOCHS}", ncols=100)
        
        for step in pbar:
            try:
                # A. GENERATE
                outputs = generator.generate_with_grads(BATCH_SIZE, allowed_elements=CAMPAIGN_ELEMENTS, temperature=0.7)
                struct = outputs["structures"][0]
                log_prob = outputs["log_probs"][0]
                
                if struct is None: continue

                # -----------------------------------------------------
                # 🔧 CHANGE 1: ATOM LIMIT TEACHING (No Silent Skip)
                # -----------------------------------------------------
                # If > 40 atoms, punish heavily so it learns to stop.
                if len(struct) > MAX_ATOMS:
                    penalty = torch.tensor(-10.0, device=device) # Heavy penalty
                    loss = -(log_prob * penalty) / GRAD_ACCUM_STEPS
                    loss.backward()
                    
                    # Clean up
                    del outputs, loss, log_prob, penalty, struct
                    continue

                # -----------------------------------------------------
                # 🔧 CHANGE 2: DENSITY TEACHING (No Silent Skip)
                # -----------------------------------------------------
                vol_per_atom = struct.volume / len(struct)
                # Softened range: 7.0 to 35.0 (allows bigger boxes for complex stuff)
                if vol_per_atom > 35.0 or vol_per_atom < 7.0:
                     penalty = torch.tensor(-5.0, device=device)
                     loss = -(log_prob * penalty) / GRAD_ACCUM_STEPS
                     loss.backward()
                     
                     epoch_rewards.append(-5.0)
                     del outputs, loss, log_prob, penalty, struct
                     continue 
                
                # B. RELAX (Silenced)
                try:
                    with QuietBlock():
                        relax_result = relaxer.relax(struct, steps=25) 
                except Exception:
                    relax_result = {"converged": False, "energy_per_atom": 5.0, "failure_reason": "crash"}

                if relax_result["converged"]:
                    epoch_energies.append(relax_result["energy_per_atom"])

                # C. REWARD & LOSS
                rewards = reward_engine.compute_reward([relax_result])
                base_reward = rewards[0]
                entropy_bonus = ENTROPY_WEIGHT * (-log_prob.detach()) 
                final_reward = base_reward + entropy_bonus

                loss = -(log_prob * final_reward) / GRAD_ACCUM_STEPS
                loss.backward()

                # D. OPTIMIZATION (Accumulated)
                if (step + 1) % GRAD_ACCUM_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(generator.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    
                    del outputs, loss, log_prob, struct
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

                # E. MONITORING
                epoch_rewards.append(base_reward.item())
                if step % 5 == 0:
                    avg_r = sum(epoch_rewards[-10:]) / len(epoch_rewards[-10:]) if epoch_rewards else 0.0
                    avg_e = sum(epoch_energies[-10:]) / len(epoch_energies[-10:]) if epoch_energies else 0.0
                    pbar.set_postfix({"Rwrd": f"{avg_r:.2f}", "eV": f"{avg_e:.2f}"})
            
            except Exception:
                print(f"⚠️ Crash: {e}")  # <--- Add this line
                continue

        # 6. SAVE
        avg_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0.0
        avg_energy = sum(epoch_energies) / len(epoch_energies) if epoch_energies else 0.0
        current_bias = generator.lattice_bias.item()
        
        print(f"📊 Summary Ep {epoch} | Reward: {avg_reward:.3f} | Energy: {avg_energy:.3f} eV | Bias: {current_bias:.4f}")
        
        if epoch % 5 == 0 or epoch == EPOCHS:
            save_path = os.path.join(checkpoint_dir, f"physics_expert_epoch_{epoch}.pt")
            torch.save({
                "model_state": generator.model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "lattice_bias_value": current_bias 
            }, save_path)
            print(f"   💾 Saved: {os.path.basename(save_path)}")

if __name__ == "__main__":
    run_phase2()