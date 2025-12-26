import os
import sys
import torch
import numpy as np
import warnings
import traceback
import gc  # Garbage Collector

# --- CONFIGURATION (Low Memory Mode) ---
BATCH_SIZE = 2          
GRAD_ACCUM_STEPS = 4    # REDUCED from 16 to 4 to prevent crashes
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
        print(f"--- Training on {self.device.upper()} (Low Memory Mode) ---")
        
        config_path = os.path.join(BASE_DIR, "pretrained_model", "config.yaml")
        model_path = os.path.join(BASE_DIR, "pretrained_model", "epoch_005500_CLEAN.pt")
        self.generator = CrystalGenerator(model_path, config_path, self.device)
        self.optimizer = torch.optim.Adam(self.generator.model.parameters(), lr=LR)
        
        # Load heavy models once
        print("   Loading Physics Engines...")
        self.oracle = CrystalOracle(device="cpu")
        self.sentinel = CrystalSentinel()
        self.reward_engine = RewardEngine(target_gap_min=0.5, target_gap_max=4.0)
        self.relaxer = CrystalRelaxer(device="cpu") 

    def train(self):
        print(f"Starting {EPOCHS} Epochs...")
        print("-" * 60)
        
        for epoch in range(1, EPOCHS + 1):
            epoch_loss = 0.0
            epoch_reward = 0.0
            self.optimizer.zero_grad(set_to_none=True) # Save memory
            valid_batches = 0
            
            print(f"\n[Epoch {epoch}] Processing batches...")

            for step in range(GRAD_ACCUM_STEPS):
                try:
                    # 1. Generate
                    # print(f"   Step {step+1}/{GRAD_ACCUM_STEPS}: Generating...", end="\r")
                    outputs = self.generator.generate_with_grads(
                        BATCH_SIZE, 
                        allowed_elements=CAMPAIGN_ELEMENTS
                    )
                    raw_structs = outputs["structures"]
                    log_probs = outputs["log_probs"]
                    
                    if not raw_structs: 
                        print(f"   Step {step+1}: Failed (Empty Generation)")
                        continue

                    # 2. Sentinel & Relax
                    # print(f"   Step {step+1}/{GRAD_ACCUM_STEPS}: Relaxing...", end="\r")
                    valid_mask, _ = self.sentinel.filter(raw_structs)
                    processed_structs = []
                    
                    for i, struct in enumerate(raw_structs):
                        if valid_mask[i]:
                            try:
                                # Run fast physics
                                res = self.relaxer.relax(struct, steps=25)
                                processed_structs.append(res["final_structure"])
                            except:
                                valid_mask[i] = False
                                processed_structs.append(struct)
                        else:
                            processed_structs.append(struct)

                    # 3. Reward
                    e_form_preds, bg_preds = self.oracle.predict_batch(processed_structs)
                    comps = [s.composition for s in processed_structs]
                    
                    rewards_tensor, stats = self.reward_engine.compute_reward(
                        valid_mask, e_form_preds, bg_preds, compositions=comps
                    )
                    rewards_tensor = rewards_tensor.to(self.device)

                    # Log progress for this specific batch so you see movement
                    if sum(valid_mask) > 0:
                        best_idx = torch.argmax(rewards_tensor).item()
                        best_form = processed_structs[best_idx].composition.reduced_formula
                        print(f"   Step {step+1}: Best={best_form:<8} | R={rewards_tensor[best_idx]:.2f}")
                    else:
                        print(f"   Step {step+1}: No valid crystals.")

                    # 4. Loss
                    advantage = (rewards_tensor - rewards_tensor.mean()) 
                    policy_loss = -torch.mean(log_probs * advantage)
                    loss = policy_loss / GRAD_ACCUM_STEPS
                    
                    loss.backward()
                    
                    epoch_loss += loss.item()
                    epoch_reward += rewards_tensor.mean().item()
                    valid_batches += 1
                    
                    # FREE MEMORY IMMEDIATELY
                    del outputs, raw_structs, processed_structs, log_probs, loss, rewards_tensor
                    torch.cuda.empty_cache()

                except Exception:
                    print(f"\n❌ Crash at Step {step+1}:")
                    traceback.print_exc()
                    continue 
            
            # --- UPDATE ---
            if valid_batches > 0:
                torch.nn.utils.clip_grad_norm_(self.generator.model.parameters(), 1.0)
                self.optimizer.step()
                
                avg_rew = epoch_reward / valid_batches
                print(f"✅ [Epoch {epoch} Done] Avg Reward: {avg_rew:.4f} | Loss: {epoch_loss:.4f}")
            else:
                print(f"⚠️ [Epoch {epoch} Done] No valid batches.")

            # Hard Cleanup between epochs
            gc.collect()
            torch.cuda.empty_cache()

            if epoch % VALIDATION_FREQ == 0:
                self.save_checkpoint()
                
    def save_checkpoint(self):
        path = os.path.join(RL_CHECKPOINT_DIR, "epoch_100_RL.pt")
        torch.save(self.generator.model.state_dict(), path)
        print(f"💾 Checkpoint Saved: {path}")

if __name__ == "__main__":
    trainer = ReinforceTrainer()
    trainer.train()
