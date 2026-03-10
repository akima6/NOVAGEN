import os
import sys
import torch
import warnings
import math
from tqdm import tqdm
import numpy as np
import contextlib
import io
import pynvml

# =============================================================================
# 🚑 1. CRITICAL WINDOWS GPU SHIM & SILENCING
# =============================================================================
try:
    sys.modules['nvidia_smi'] = pynvml
    os.environ['PATH'] += os.pathsep + r'C:\Windows\System32'
except Exception:
    pass 

warnings.filterwarnings("ignore")

class QuietBlock:
    """Suppresses CHGNet/MEGNet console spam during evaluation."""
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
# 2. PATH SETUP & IMPORTS
# ==========================================
PROJECT_ROOT = os.getcwd()
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "CrystalFormer"))
sys.path.append(os.path.join(PROJECT_ROOT, "core"))
sys.path.append(os.path.join(PROJECT_ROOT, "rewards"))

try:
    from generator_service import CrystalGenerator
    from sentinel import CrystalSentinel
    from relaxer import CrystalRelaxer
    from oracle import Phase3Oracle
    from reward_phase3 import EngineerRewardEngine
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

# ==========================================
# 3. CONFIGURATION
# ==========================================
NUM_SAMPLES = 200     # Reduced from 500 so you get answers faster    
BATCH_SIZE = 1            
TEMPERATURE = 0.7
MAX_ATOMS = 40

CAMPAIGN_ELEMENTS = [
    3, 11, 19, 4, 12, 20, 38, 56, 5, 13, 31, 49,
    6, 14, 32, 50, 7, 15, 33, 51, 8, 16, 34, 52,
    21, 22, 30, 48
]

CONFIG_PATH = os.path.join(PROJECT_ROOT, "pretrained_model", "config.yaml")
MODEL_PHASE2 = os.path.join(PROJECT_ROOT, "pretrained_model","physicist_epoch_60.pt")
MODEL_PHASE3 = os.path.join(PROJECT_ROOT,  "pretrained_model", "lab_grade_Semiconductor_epoch_100.pt")

# ==========================================
# 4. CORE EVALUATION LOOP
# ==========================================
def run_evaluation(model_path, model_name, device, sentinel, relaxer, oracle, reward_engine):
    print(f"\n{'='*70}")
    print(f"🤖 EVALUATING: {model_name}")
    print(f"{'='*70}")
    
    generator = CrystalGenerator(model_path, CONFIG_PATH, device)
    generator.model.eval() 
    
    checkpoint = torch.load(model_path, map_location=device)
    bias_val = checkpoint.get("lattice_bias_value", 3.0)
    print(f"   📏 Internal Lattice Bias: {bias_val:.4f}\n")
    
    metrics = {
        "valid_geom": 0, "relax_successes": 0, "oracle_successes": 0,
        "total_energy": 0.0, "total_gap": 0.0, "total_reward": 0.0,
        "metals": 0,          # Gap < 0.1 eV
        "narrow_gap": 0,      # Gap 0.1 to 0.8 eV (The Trap Zone)
        "target_gap": 0,      # Gap 0.8 to 2.0 eV (The Holy Grail Zone)
        "insulators": 0       # Gap > 2.0 eV
    }
    
    valid_phys_count = 0
    valid_elec_count = 0
    
    with torch.no_grad(): # Global evaluation mode (Saves VRAM)
        for _ in tqdm(range(NUM_SAMPLES), desc=f"Evaluating", ncols=100):
            outputs = generator.generate(BATCH_SIZE, allowed_elements=CAMPAIGN_ELEMENTS, temperature=TEMPERATURE)
            struct = outputs["structures"][0]
            
            if struct is None or len(struct) > MAX_ATOMS: 
                del outputs
                continue
            
            # --- GEOMETRY CHECK ---
            if not sentinel.filter([struct])[0].get("valid", False):
                del outputs
                continue
                
            metrics["valid_geom"] += 1
            
            # --- RELAXATION (ENERGY) ---
            struct.scale_lattice(math.exp(bias_val) * len(struct))
            
            with QuietBlock():
                with torch.enable_grad():
                    relax_result = relaxer.relax(struct, steps=15)
                
            if not relax_result["converged"] or relax_result["final_structure"] is None:
                del outputs, relax_result
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                continue
                
            metrics["relax_successes"] += 1
            valid_phys_count += 1
            final_struct = relax_result["final_structure"]
            energy_pa = relax_result["energy_per_atom"]
            metrics["total_energy"] += energy_pa
            
            # --- ORACLE (BANDGAP) ---
            gap = oracle.evaluate_structure(final_struct)
            if gap < 0: # Oracle crashed
                del outputs, relax_result, final_struct
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                continue
                
            metrics["oracle_successes"] += 1
            valid_elec_count += 1
            metrics["total_gap"] += gap
            
            # --- REWARD CALCULATION ---
            scores = reward_engine.compute_reward(energy_pa, gap)
            metrics["total_reward"] += scores["total_reward"]
            
            # --- BANDGAP DISTRIBUTION CATEGORIZATION ---
            if gap < 0.1:
                metrics["metals"] += 1
            elif gap < 0.8:
                metrics["narrow_gap"] += 1
            elif gap <= 2.0:
                metrics["target_gap"] += 1
            else:
                metrics["insulators"] += 1

            # Extreme Memory Hygiene
            del outputs, relax_result, final_struct
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Calculate Averages safely
    p_count = max(valid_phys_count, 1)
    e_count = max(valid_elec_count, 1)
    
    print("\n" + "="*70)
    print(f"📊 FINAL METRICS: {model_name}")
    print("="*70)
    
    print("\n⚛️ --- THERMODYNAMIC RETENTION (Did Phase 3 break Phase 2?) ---")
    print(f"✅ Geometric Survival Rate:   {(metrics['valid_geom']/NUM_SAMPLES)*100:.1f}%")
    print(f"✅ Relaxation Success Rate:   {(metrics['relax_successes']/NUM_SAMPLES)*100:.1f}%")
    print(f"🔋 Avg Relaxed Energy:        {metrics['total_energy']/p_count:.3f} eV/atom (Lower is better)")
    
    print("\n⚡ --- ELECTRONIC PERFORMANCE (Did it learn semiconductors?) ---")
    print(f"🔮 Oracle Success Rate:       {(metrics['oracle_successes']/p_count)*100:.1f}%")
    print(f"📈 Average Bandgap:           {metrics['total_gap']/e_count:.3f} eV")
    print(f"💰 Average Phase 3 Reward:    {metrics['total_reward']/e_count:.2f} points")
    
    print("\n📉 --- BANDGAP DISTRIBUTION HISTOGRAM ---")
    print(f"🧲 Metals (< 0.1 eV):         {(metrics['metals']/e_count)*100:.1f}%  [{'█' * int((metrics['metals']/e_count)*20)}]")
    print(f"🔋 Narrow Gap (0.1 - 0.8 eV): {(metrics['narrow_gap']/e_count)*100:.1f}%  [{'█' * int((metrics['narrow_gap']/e_count)*20)}] <-- The Trap Zone")
    print(f"🎯 Target Gap (0.8 - 2.0 eV): {(metrics['target_gap']/e_count)*100:.1f}%  [{'█' * int((metrics['target_gap']/e_count)*20)}] <-- Holy Grail")
    print(f"🧱 Insulators (> 2.0 eV):     {(metrics['insulators']/e_count)*100:.1f}%  [{'█' * int((metrics['insulators']/e_count)*20)}]")
    print("\n\n")
    
    del generator
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("🚀 STARTING PHASE 3 COMPARATIVE BENCHMARK")
    
    sentinel = CrystalSentinel(device)
    relaxer = CrystalRelaxer(device=device)
    with QuietBlock():
        oracle = Phase3Oracle()
    reward_engine = EngineerRewardEngine(ideal_gap=1.5)
    
    if os.path.exists(MODEL_PHASE2):
        run_evaluation(MODEL_PHASE2, "BASELINE: Phase 2 Physicist", device, sentinel, relaxer, oracle, reward_engine)
    else:
        print(f"❌ Model A not found at: {MODEL_PHASE2}")

    if os.path.exists(MODEL_PHASE3):
        run_evaluation(MODEL_PHASE3, "FINE-TUNED: Phase 3 Engineer (Epoch 100)", device, sentinel, relaxer, oracle, reward_engine)
    else:
        print(f"❌ Model B not found at: {MODEL_PHASE3}\n   Check your filename and epoch number!")
        
    print("🎯 BENCHMARK COMPLETE.")