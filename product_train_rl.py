import os
import sys
import torch
import numpy as np
import warnings
import traceback
import gc  # Garbage Collector

# --- CONFIGURATION (NUCLEAR SAFETY MODE) ---
BATCH_SIZE = 1           # Process 1 crystal at a time
GRAD_ACCUM_STEPS = 1     # Update brain immediately (No backlog)
LR = 1e-5
EPOCHS = 100
VALIDATION_FREQ = 5      # Save more often in case it crashes
MAX_ATOMS = 20           # REJECT large crystals to save RAM

# Fe(26), O(8), S(16), Si(14), N(7)
CAMPAIGN_ELEMENTS = [26, 8, 16, 14, 7]

# PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "CrystalFormer"))
sys.path.append(BASE_DIR)
RL_CHECKPOINT_DIR = os.path.join(BASE_DIR, "rl_checkpoints")
os.makedirs(RL_CHECKPOINT_DIR, exist_ok=True)

warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "1"

try:
    from generator_service import CrystalGenerator
    from sentinel import CrystalSentinel
    from product_oracle import CrystalOracle
    from product_relaxer import CrystalRelaxer
    from product_reward_engine import RewardEngine
except ImportError as e:
    sys.exit(f"Setup Error: {e}")

class ReinforceTrainer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"--- Training on {self.device.upper()} (Single-Shot Mode) ---")

        config_path = os.path.join(BASE_DIR, "pretrained_model", "config.yaml")
        model_path = os.path.join(BASE_DIR, "pretrained_model", "epoch_005500_CLEAN.pt")
        self.generator = CrystalGenerator(model_path, config_path, self.device)
        self.optimizer = torch.optim.Adam(self.generator.model.parameters(), lr=LR)

        print("   Loading Physics Engines...")
        self.oracle = CrystalOracle(device="cpu")
        self.sentinel = CrystalSentinel()
        self.reward_engine = RewardEngine(target_gap_min=0.5, target_gap_max=4.0)
        self.relaxer = CrystalRelaxer(device="cpu")

    def train(self):
        print(f"Starting {EPOCHS} Epochs...")
        print("-" * 60)
        
        # --- NEW: Moving Average Baseline for Batch=1 Stability ---
        moving_baseline = 0.0 
        # ----------------------------------------------------------

        for epoch in range(1, EPOCHS + 1):
            epoch_loss = 0.0
            epoch_reward = 0.0
            valid_batches = 0
            
            self.optimizer.zero_grad(set_to_none=True)
            print(f"\n[Epoch {epoch}] Running...")

            # We process 4 crystals per epoch (you can increase this number now)
            for step in range(4):
                try:
                    # 1. Generate
                    outputs = self.generator.generate_with_grads(BATCH_SIZE, allowed_elements=CAMPAIGN_ELEMENTS)
                    raw_structs = outputs["structures"]
                    log_probs = outputs["log_probs"]

                    if not raw_structs: continue

                    struct = raw_structs[0]
                    
                    # Safety: Reject Large Crystals
                    if len(struct) > MAX_ATOMS:
                        print(f"   Step {step+1}: Skipped (Too Large: {len(struct)} atoms)")
                        del outputs, raw_structs, log_probs
                        continue

                    # 2. Filter & Relax
                    valid_mask, _ = self.sentinel.filter([struct])
                    final_struct = struct
                    
                    if valid_mask[0]:
                        try:
                            # Increased steps slightly for better accuracy
                            res = self.relaxer.relax(struct, steps=50) 
                            final_struct = res["final_structure"]
                        except:
                            valid_mask[0] = False

                    # 3. Reward
                    e_form_preds, bg_preds = self.oracle.predict_batch([final_struct])
                    comps = [final_struct.composition]
                    rewards_tensor, stats = self.reward_engine.compute_reward(
                        valid_mask, e_form_preds, bg_preds, compositions=comps
                    )
                    rewards_tensor = rewards_tensor.to(self.device)
                    
                    current_reward = rewards_tensor.item() # Scalar value

                    # Log
                    formula = struct.composition.reduced_formula
                    if valid_mask[0]:
                        print(f"   Step {step+1}: {formula:<8} | R={current_reward:.2f}")
                    else:
                        print(f"   Step {step+1}: {formula:<8} | Invalid")

                    # --- CRITICAL FIX: ADVANTAGE CALCULATION ---
                    # Old (Broken): advantage = current_reward - current_reward = 0
                    # New (Fixed):  advantage = current_reward - moving_baseline
                    
                    if valid_batches == 0 and epoch == 1:
                        moving_baseline = current_reward # Initialize
                    
                    advantage = current_reward - moving_baseline
                    
                    # Update baseline (Exponential Moving Average)
                    # This allows the AI to learn "Is this better than usual?"
                    moving_baseline = 0.9 * moving_baseline + 0.1 * current_reward
                    # -------------------------------------------

                    # 4. Loss & Update
                    policy_loss = -log_probs.mean() * advantage
                    policy_loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(self.generator.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                    epoch_loss += policy_loss.item()
                    epoch_reward += current_reward
                    valid_batches += 1

                    # Cleanup
                    del outputs, raw_structs, log_probs, policy_loss, rewards_tensor, final_struct
                    torch.cuda.empty_cache()
                    gc.collect() 

                except Exception:
                    continue

            # End Epoch Logs
            if valid_batches > 0:
                avg_rew = epoch_reward / valid_batches
                print(f"✅ [Epoch {epoch}] Avg Reward: {avg_rew:.4f} | Baseline: {moving_baseline:.2f}")
            else:
                print(f"⚠️ [Epoch {epoch}] No valid crystals.")
            
            if epoch % VALIDATION_FREQ == 0:
                self.save_checkpoint()

    def save_checkpoint(self):
        path = os.path.join(RL_CHECKPOINT_DIR, "epoch_100_RL.pt")
        torch.save(self.generator.model.state_dict(), path)
        print(f"💾 Saved: {path}")

if __name__ == "__main__":
    trainer = ReinforceTrainer()
    trainer.train()
