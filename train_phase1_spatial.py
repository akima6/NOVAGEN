import os
import sys
import torch
import warnings
import gc
import re
from tqdm import tqdm

# =========================
# -------- CONFIG ---------
# =========================
MAX_ATOMS = 20
LR = 1e-5                   # Slower learning rate
EPOCHS = 60                 
BATCH_SIZE = 4              # Batch size 4
GRAD_ACCUM_STEPS = 16       
STEPS_PER_EPOCH = 50        

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
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
else:
    BASE_DIR = os.getcwd()

sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "CrystalFormer"))

try:
    from generator_service import CrystalGenerator
    from sentinel import CrystalSentinel
    from reward_phase1 import SpatialRewardEngine
except ImportError as e:
    sys.exit(f"❌ Import Error: {e}")

# =========================
# ----- TRAIN LOOP --------
# =========================
def run_phase1_v3():
    print("🚀 STARTING PHASE 1 v3: STABILIZATION")
    print("   Goal: Eliminate Implosions/Gas via Lower LR & Higher Batch Size")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir = os.path.join("rl_checkpoints", "phase1_v3")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    config_path = os.path.join(BASE_DIR, "pretrained_model", "config.yaml")
    base_model_path = os.path.join(BASE_DIR, "pretrained_model", "epoch_005500_CLEAN.pt")
    
    generator = CrystalGenerator(base_model_path, config_path, device)
    generator.model.train()
    optimizer = torch.optim.Adam(generator.parameters(), lr=LR)
    
    sentinel = CrystalSentinel(device)
    reward_engine = SpatialRewardEngine()
    
    optimizer.zero_grad() 
    
    for epoch in range(1, EPOCHS + 1):
        epoch_rewards = []
        epoch_vols = []
        pbar = tqdm(range(STEPS_PER_EPOCH), desc=f"Epoch {epoch}/{EPOCHS}")
        
        for step in pbar:
            # Generate Batch of 4
            outputs = generator.generate_with_grads(BATCH_SIZE, allowed_elements=CAMPAIGN_ELEMENTS)
            
            # Initialize batch accumulator
            batch_loss_tensor = 0 
            
            for i in range(BATCH_SIZE):
                struct = outputs["structures"][i]
                log_prob = outputs["log_probs"][i]
                
                # --- DENSITY CHECK ---
                if struct:
                    vol_per_atom = struct.volume / struct.num_sites
                    epoch_vols.append(vol_per_atom)
                    
                    if vol_per_atom > 50.0:
                        reward = -5.0 # Gas Penalty
                    elif vol_per_atom < 10.0:
                        reward = -5.0 # Implosion Penalty
                    else:
                        sentinel_res = sentinel.filter([struct])
                        reward = reward_engine.compute_reward(sentinel_res)[0].item()
                else:
                    reward = -5.0
                
                epoch_rewards.append(reward)
                
                # Accumulate Loss (DO NOT BACKWARD YET)
                # We average the loss over batch_size and grad_accum_steps
                loss = -(log_prob * reward) / (BATCH_SIZE * GRAD_ACCUM_STEPS)
                batch_loss_tensor += loss

            # --- BACKWARD ONCE PER BATCH ---
            # This backpropagates through the whole graph for all 4 items at once
            if isinstance(batch_loss_tensor, torch.Tensor):
                batch_loss_tensor.backward()
            
            # Optimization Step
            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(generator.model.parameters(), max_norm=0.5)
                optimizer.step()
                optimizer.zero_grad() 
                
            # Monitoring
            if step % 5 == 0:
                avg_vol = sum(epoch_vols[-20:]) / len(epoch_vols[-20:]) if epoch_vols else 0.0
                pbar.set_postfix({
                    "R": f"{sum(epoch_rewards[-BATCH_SIZE:]) / BATCH_SIZE:.2f}", 
                    "Vol": f"{avg_vol:.1f}"
                })
            
            del outputs, batch_loss_tensor
            gc.collect()

        # End of Epoch Stats
        avg_r = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0.0
        avg_v = sum(epoch_vols) / len(epoch_vols) if epoch_vols else 0.0
        
        print(f"📊 Ep {epoch} | Avg Reward: {avg_r:.2f} | Avg Volume: {avg_v:.1f} Å³")
        
        # Save checkpoint
        if epoch % 5 == 0:
            save_path = os.path.join(checkpoint_dir, f"spatial_v3_epoch_{epoch}.pt")
            torch.save({
                "model_state": generator.model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                # Save the learnable lattice bias value explicitly for easier debugging
                "lattice_bias_value": generator.lattice_bias.item()
            }, save_path)
            print(f"   💾 Checkpoint Saved: epoch_{epoch} (Bias: {generator.lattice_bias.item():.4f})")

if __name__ == "__main__":
    run_phase1_v3()
