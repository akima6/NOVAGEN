import os
import sys
import torch
import warnings
import itertools
from tqdm import tqdm
import numpy as np

# Silence warnings for clean console output
warnings.filterwarnings("ignore")

# ==========================================
# 1. PATH SETUP (Based on /evaluation/ folder)
# ==========================================
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(EVAL_DIR)

# Add necessary folders to Python path so we can import your custom modules
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "CrystalFormer"))
sys.path.append(os.path.join(PROJECT_ROOT, "core"))
sys.path.append(os.path.join(PROJECT_ROOT, "rewards"))

# Imports from your codebase
try:
    from generator_service import CrystalGenerator
    from sentinel import CrystalSentinel
    from reward_phase1 import ChemistryRewardEngine
    import smact
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

# ==========================================
# 2. CONFIGURATION
# ==========================================
NUM_SAMPLES = 1000          # Total structures to generate per model
BATCH_SIZE = 2             # Keep low to protect RTX 3050 VRAM
TEMPERATURE = 0.7

CAMPAIGN_ELEMENTS = [
    3, 11, 19, 4, 12, 20, 38, 56, 5, 13, 31, 49,
    6, 14, 32, 50, 7, 15, 33, 51, 8, 16, 34, 52,
    21, 22, 30, 48
]

# Model Paths
CONFIG_PATH = os.path.join(PROJECT_ROOT, "pretrained_model", "config.yaml")
MODEL_A_PATH = os.path.join(PROJECT_ROOT, "pretrained_model", "epoch_005500_CLEAN.pt") # Baseline
MODEL_B_PATH = os.path.join(PROJECT_ROOT, "pretrained_model", "chemist_v2_epoch_100.pt") # Finetuned

# ==========================================
# 3. METRICS HELPER (Deep Chemistry Analysis)
# ==========================================
def calculate_chemistry_metrics(struct):
    """Replicates the Phase 1 chemistry rules to get raw percentages."""
    if struct is None:
        return False, 10.0 # Failed

    comp_dict = struct.composition.get_el_amt_dict()
    elements = list(comp_dict.keys())
    
    # 1. Pauling Test
    pauling_pass = False
    if len(elements) > 1:
        enegs = [smact.Element(el).pauling_eneg for el in elements if smact.Element(el).pauling_eneg is not None]
        if enegs and (max(enegs) - min(enegs) > 1.0):
            pauling_pass = True

    # 2. Charge Neutrality Error
    ox_states_list = [smact.Element(el).oxidation_states or [0] for el in elements]
    counts = list(comp_dict.values())
    
    min_abs_charge = float('inf')
    for combo in itertools.product(*ox_states_list):
        net_charge = sum(ox * count for ox, count in zip(combo, counts))
        if abs(net_charge) < min_abs_charge:
            min_abs_charge = abs(net_charge)
            if min_abs_charge == 0:
                break
                
    return pauling_pass, min_abs_charge

# ==========================================
# 4. CORE EVALUATION LOOP
# ==========================================
def run_evaluation(model_path, model_name, device, sentinel, reward_engine):
    print(f"\n{'='*50}")
    print(f"🤖 EVALUATING: {model_name}")
    print(f"{'='*50}")
    
    # Load Model
    generator = CrystalGenerator(model_path, CONFIG_PATH, device)
    generator.model.eval() # Set to evaluation mode!
    
    metrics = {
        "valid_geom": 0,
        "total_vol": 0.0,
        "pauling_passes": 0,
        "total_charge_error": 0.0,
        "jackpots": 0,
        "total_reward": 0.0,
        "valid_count": 0
    }
    
    iterations = NUM_SAMPLES // BATCH_SIZE
    
    with torch.no_grad(): # No gradients needed for evaluation, saves VRAM
        for _ in tqdm(range(iterations), desc=f"Generating {NUM_SAMPLES} structures"):
            outputs = generator.generate(BATCH_SIZE, allowed_elements=CAMPAIGN_ELEMENTS, temperature=TEMPERATURE)
            structures = outputs["structures"]
            
            # 1. Sentinel Pass
            sentinel_results = sentinel.filter(structures)
            
            # 2. Reward Engine Pass
            rewards = reward_engine.compute_reward(structures, sentinel_results)
            
            # 3. Record Metrics
            for i in range(BATCH_SIZE):
                struct = structures[i]
                sent = sentinel_results[i]
                reward = rewards[i].item()
                
                metrics["total_reward"] += reward
                
                if sent.get("valid", False):
                    metrics["valid_geom"] += 1
                    metrics["valid_count"] += 1
                    metrics["total_vol"] += sent.get("volume_per_atom", 0.0)
                    
                    # Chemistry checks
                    p_pass, c_err = calculate_chemistry_metrics(struct)
                    if p_pass: metrics["pauling_passes"] += 1
                    metrics["total_charge_error"] += c_err
                    
                    # Jackpot check (Base rewards usually cap ~7.0, so > 10 means jackpot)
                    if reward > 10.0:
                        metrics["jackpots"] += 1

    # Calculate Averages
    v_count = max(metrics["valid_count"], 1) # Prevent division by zero
    
    print("\n📊 --- RESULTS ---")
    print(f"🛡️ Geometric Survival Rate:  {(metrics['valid_geom']/NUM_SAMPLES)*100:.1f}%")
    print(f"📦 Average Valid Volume:     {metrics['total_vol']/v_count:.2f} Å³/atom")
    print(f"⚡ Pauling Test Pass Rate:   {(metrics['pauling_passes']/v_count)*100:.1f}% (of valid)")
    print(f"⚖️ Average Charge Error:     {metrics['total_charge_error']/v_count:.2f} electrons")
    print(f"🎰 Jackpot Hit Rate:         {(metrics['jackpots']/NUM_SAMPLES)*100:.1f}%")
    print(f"🏆 Average RL Reward:        {metrics['total_reward']/NUM_SAMPLES:.2f}")
    
    # Clean up memory before next model
    del generator
    torch.cuda.empty_cache()

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("🚀 STARTING HEAD-TO-HEAD MODEL EVALUATION")
    
    # Initialize engines once
    sentinel = CrystalSentinel(device)
    reward_engine = ChemistryRewardEngine(target_vol=25.0)
    
    # Run Model A (Baseline)
    if os.path.exists(MODEL_A_PATH):
        run_evaluation(MODEL_A_PATH, "MODEL A (Pre-Trained Baseline)", device, sentinel, reward_engine)
    else:
        print(f"❌ Could not find Model A at: {MODEL_A_PATH}")

    # Run Model B (Fine-Tuned)
    if os.path.exists(MODEL_B_PATH):
        run_evaluation(MODEL_B_PATH, "MODEL B (Phase 1 Chemist)", device, sentinel, reward_engine)
    else:
        print(f"❌ Could not find Model B at: {MODEL_B_PATH}")
        
    print("\n🎯 EVALUATION COMPLETE.")