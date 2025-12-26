import os
import sys
import torch
import numpy as np
import warnings
import traceback 

# --- CONFIGURATION ---
BATCH_SIZE = 2          
GRAD_ACCUM_STEPS = 16    
LR = 1e-5                
EPOCHS = 100            
VALIDATION_FREQ = 10    

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
        print(f"--- Training on {self.device.upper()} ---")
        
        config_path = os.path.join(BASE_DIR, "pretrained_model", "config.yaml")
        model_path = os.path.join(BASE_DIR, "pretrained_model", "epoch_005500_CLEAN.pt")
        self.generator = CrystalGenerator(model_path, config_path, self.device)
        self.optimizer = torch.optim.Adam(self.generator.model.parameters(), lr=LR)
        
        self.oracle = CrystalOracle(device="cpu")
        self.sentinel = CrystalSentinel()
        self.reward_engine = RewardEngine(target_gap_min=0.5, target_gap_max=4.0)
        self.relaxer = CrystalRelaxer(device="cpu") 

    def train(self):
        print(f"Starting {EPOCHS} Epochs (Fast Relax Enabled)...")
        print(f"{'Epoch':<6} | {'Avg Reward':<12} | {'Loss':<10}")
        print("-" * 35)
        
        for epoch in range(1, EPOCHS + 1):
            epoch_loss = 0.0
            epoch_reward = 0.0
            self.optimizer.zero_grad()
            valid_batches = 0
            
            for step in range(GRAD_ACCUM_STEPS):
                try:
                    # A. Generate
                    outputs = self.generator.generate_with_grads(
                        BATCH_SIZE, 
                        allowed_elements=CAMPAIGN_ELEMENTS
                    )
                    raw_structs = outputs["structures"]
                    log_probs = outputs["log_probs"] # Shape: [Batch_Size]
                    
                    if not raw_structs: continue

                    # B. Sentinel Filter
                    valid_mask, _ = self.sentinel.filter(raw_structs)
                    
                    # C. Fast Relax
                    processed_structs = []
                    for i, struct in enumerate(raw_structs):
                        if valid_mask[i]:
                            try:
                                res = self.relaxer.relax(struct, steps=25)
                                processed_structs.append(res["final_structure"])
                            except:
                                valid_mask[i] = False
                                processed_structs.append(struct)
                        else:
                            processed_structs.append(struct)

                    # D. Reward
                    e_form_preds, bg_preds = self.oracle.predict_batch(processed_structs)
                    comps = [s.composition for s in processed_structs]
                    
                    rewards_tensor, stats = self.reward_engine.compute_reward(
                        valid_mask, e_form_preds, bg_preds, compositions=comps
                    )
                    rewards_tensor = rewards_tensor.to(self.device)

                    # E. Loss Calculation (SIMPLIFIED)
                    # 1. Calculate Advantage
                    advantage = (rewards_tensor - rewards_tensor.mean()) 
                    
                    # 2. Policy Loss = - log_prob * advantage
                    # Since log_probs is 1D, we just multiply element-wise
                    policy_loss = -torch.mean(log_probs * advantage)
                    
                    loss = policy_loss / GRAD_ACCUM_STEPS
                    
                    loss.backward()
                    
                    epoch_loss += loss.item()
                    epoch_reward += rewards_tensor.mean().item()
                    valid_batches += 1

                except Exception:
                    # PRINT ERROR AND STOP IF CRASH
                    print("\n❌ CRITICAL FAILURE IN BATCH:")
                    traceback.print_exc()
                    return 
            
            # --- LOGGING ---
            if valid_batches > 0:
                torch.nn.utils.clip_grad_norm_(self.generator.model.parameters(), 1.0)
                self.optimizer.step()
                
                avg_rew = epoch_reward / valid_batches
                print(f"{epoch:<6} | {avg_rew:<12.4f} | {epoch_loss:<10.4f}")
            else:
                print(f"{epoch:<6} | {'SKIPPED':<12} | {'0.00':<10}")

            if epoch % VALIDATION_FREQ == 0:
                self.save_checkpoint()
                
    def save_checkpoint(self):
        path = os.path.join(RL_CHECKPOINT_DIR, "epoch_100_RL.pt")
        torch.save(self.generator.model.state_dict(), path)
        sys.stdout.write(f"\r[Saved Checkpoint]   \n") 

if __name__ == "__main__":
    trainer = ReinforceTrainer()
    trainer.train()
