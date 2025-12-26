import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import yaml

# --- CONFIGURATION ---
BATCH_SIZE = 2          # Keep small for stability
GRAD_ACCUM_STEPS = 8    # Effective Batch = 16
LR = 1e-5               
EPOCHS = 100            
VALIDATION_FREQ = 10    
ENTROPY_COEF = 0.01     

# UPDATED: Focus on Semiconductor Elements + Oxygen/Sulfur
# Fe(26), O(8), S(16), Si(14), N(7)
CAMPAIGN_ELEMENTS = [26, 8, 16, 14, 7] 

# PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "CrystalFormer"))
sys.path.append(BASE_DIR)

RL_CHECKPOINT_DIR = os.path.join(BASE_DIR, "rl_checkpoints")
os.makedirs(RL_CHECKPOINT_DIR, exist_ok=True)

try:
    from generator_service import CrystalGenerator
    from sentinel import CrystalSentinel
    from product_oracle import CrystalOracle
    from product_relaxer import CrystalRelaxer # <--- NEW IMPORT
    from product_reward_engine import RewardEngine
except ImportError as e:
    print(f"❌ Setup Error: {e}")
    sys.exit(1)

class ReinforceTrainer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Initializing RL Gym on {self.device} (FP32 Mode)...")
        
        # 1. Load Pre-trained Generator
        config_path = os.path.join(BASE_DIR, "pretrained_model", "config.yaml")
        model_path = os.path.join(BASE_DIR, "pretrained_model", "epoch_005500_CLEAN.pt")
        
        print(f"   💎 Loading weights from {os.path.basename(model_path)}...")
        self.generator = CrystalGenerator(model_path, config_path, self.device)
        self.optimizer = torch.optim.Adam(self.generator.model.parameters(), lr=LR)
        
        # 2. Initialize Components
        print("   🔧 Initializing Teachers...")
        self.oracle = CrystalOracle(device="cpu")
        self.sentinel = CrystalSentinel()
        self.reward_engine = RewardEngine(target_gap_min=0.5, target_gap_max=4.0)
        
        # 3. NEW: Initialize Fast Relaxer
        # We keep it on CPU to save GPU memory for the Generator
        print("   ⚛️ Initializing Fast Physics Engine...")
        self.relaxer = CrystalRelaxer(device="cpu") 

    def train(self):
        print(f"⚡ Starting REINFORCE Training with FAST RELAXATION for {EPOCHS} epochs...")
        
        for epoch in range(1, EPOCHS + 1):
            epoch_loss = 0.0
            epoch_reward = 0.0
            self.optimizer.zero_grad()
            
            # --- BATCH LOOP ---
            for step in range(GRAD_ACCUM_STEPS):
                # A. Generate (Raw)
                try:
                    # We need the log_probs to learn
                    outputs = self.generator.generate_with_grads(
                        BATCH_SIZE, 
                        allowed_elements=CAMPAIGN_ELEMENTS
                    )
                    raw_structs = outputs["structures"]
                    log_probs = outputs["log_probs"]
                    
                    if not raw_structs:
                        continue

                    # B. Sentinel Filter (Cheap Check)
                    # Don't waste physics time on exploding crystals
                    valid_mask, _ = self.sentinel.filter(raw_structs)
                    
                    # C. FAST RELAXATION (The New Fix)
                    # We only relax the valid ones to save time.
                    # We modify the structures IN PLACE.
                    
                    processed_structs = []
                    
                    for i, struct in enumerate(raw_structs):
                        if valid_mask[i]:
                            # Run Fast Relax (25 steps only)
                            # This is enough to know if it wants to be stable
                            try:
                                res = self.relaxer.relax(struct, steps=25)
                                processed_structs.append(res["final_structure"])
                            except:
                                # If relax crashes, treat as invalid
                                valid_mask[i] = False
                                processed_structs.append(struct) # Placeholder
                        else:
                            processed_structs.append(struct) # Ignored later

                    # D. Oracle Prediction (On RELAXED structures)
                    # Now the Oracle sees the "True" potential of the crystal
                    e_form_preds, bg_preds = self.oracle.predict_batch(processed_structs)
                    
                    # E. Compute Reward
                    # Pass compositions for the complexity check
                    comps = [s.composition for s in processed_structs]
                    
                    rewards_tensor, stats = self.reward_engine.compute_reward(
                        valid_mask, 
                        e_form_preds, 
                        bg_preds, 
                        compositions=comps
                    )
                    rewards_tensor = rewards_tensor.to(self.device)

                    # F. Loss Calculation (REINFORCE)
                    # Loss = -Reward * log_prob
                    # We want to maximize Reward, so we minimize Negative Reward
                    
                    # Entropy Bonus (Encourage exploration)
                    entropy = -torch.mean(torch.sum(torch.exp(log_probs) * log_probs, dim=1))
                    
                    advantage = (rewards_tensor - rewards_tensor.mean()) 
                    
                    # Policy Loss
                    policy_loss = -torch.mean(log_probs * advantage.unsqueeze(1))
                    
                    loss = policy_loss - (ENTROPY_COEF * entropy)
                    loss = loss / GRAD_ACCUM_STEPS # Normalize
                    
                    loss.backward()
                    
                    epoch_loss += loss.item()
                    epoch_reward += rewards_tensor.mean().item()

                except Exception as e:
                    # Catch OOM or random errors
                    continue
            
            # --- UPDATE WEIGHTS ---
            torch.nn.utils.clip_grad_norm_(self.generator.model.parameters(), 1.0)
            self.optimizer.step()
            
            # --- LOGGING ---
            print(f"[Epoch {epoch}] Avg Reward: {epoch_reward:.2f} | Loss: {epoch_loss:.2f}")
            
            # --- CHECKPOINT ---
            if epoch % VALIDATION_FREQ == 0:
                self.save_checkpoint(epoch)
                
    def save_checkpoint(self, epoch):
        path = os.path.join(RL_CHECKPOINT_DIR, "epoch_100_RL.pt")
        torch.save(self.generator.model.state_dict(), path)
        print(f"💾 Saved Checkpoint: {path}")

if __name__ == "__main__":
    trainer = ReinforceTrainer()
    trainer.train()
