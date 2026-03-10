import os
import sys
import torch
import warnings
import itertools
import math
from tqdm import tqdm
import numpy as np
import contextlib
import io
import pynvml
# =============================================================================
# 🚑 1. CRITICAL WINDOWS GPU SHIM
# =============================================================================
try:
    sys.modules['nvidia_smi'] = pynvml
    os.environ['PATH'] += os.pathsep + r'C:\Windows\System32'
except Exception:
    pass 

# Silence warnings for clean console output
warnings.filterwarnings("ignore")

class QuietBlock:
    """Suppresses CHGNet/ASE console spam during evaluation."""
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
# 1. PATH SETUP
# ==========================================
if "__file__" in locals():
    EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(EVAL_DIR)
else:
    PROJECT_ROOT = os.getcwd()

sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "CrystalFormer"))
sys.path.append(os.path.join(PROJECT_ROOT, "core"))
sys.path.append(os.path.join(PROJECT_ROOT, "rewards"))

try:
    from generator_service import CrystalGenerator
    from sentinel import CrystalSentinel
    from reward_phase1 import ChemistryRewardEngine
    from relaxer import CrystalRelaxer
    from reward_phase2 import PhysicsRewardEngine
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    import smact
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

# ==========================================
# 2. CONFIGURATION
# ==========================================
# Keep this at 50 or 100 to avoid hours of evaluation time!
NUM_SAMPLES = 500        
BATCH_SIZE = 1            
TEMPERATURE = 0.7

CAMPAIGN_ELEMENTS = [
    3, 11, 19, 4, 12, 20, 38, 56, 5, 13, 31, 49,
    6, 14, 32, 50, 7, 15, 33, 51, 8, 16, 34, 52,
    21, 22, 30, 48
]

CONFIG_PATH = os.path.join(PROJECT_ROOT, "pretrained_model", "config.yaml")
MODEL_BASE = os.path.join(PROJECT_ROOT, "pretrained_model", "chemist_v2_epoch_100.pt")
MODEL_PHASE2 = os.path.join(PROJECT_ROOT, "pretrained_model","physicist_epoch_60.pt")

# ==========================================
# 3. CHEMISTRY METRICS (From Phase 1)
# ==========================================
def calculate_chemistry_metrics(struct):
    if struct is None:
        return False, 10.0

    comp_dict = struct.composition.get_el_amt_dict()
    elements = list(comp_dict.keys())
    
    pauling_pass = False
    if len(elements) > 1:
        enegs = [smact.Element(el).pauling_eneg for el in elements if smact.Element(el).pauling_eneg is not None]
        if enegs and (max(enegs) - min(enegs) > 1.0):
            pauling_pass = True

    ox_states_list = [smact.Element(el).oxidation_states or [0] for el in elements]
    counts = list(comp_dict.values())
    
    min_abs_charge = float('inf')
    for combo in itertools.product(*ox_states_list):
        net_charge = sum(ox * count for ox, count in zip(combo, counts))
        if abs(net_charge) < min_abs_charge:
            min_abs_charge = abs(net_charge)
            if min_abs_charge == 0: break
                
    return pauling_pass, min_abs_charge

# ==========================================
# 4. CORE EVALUATION LOOP
# ==========================================
def run_evaluation(model_path, model_name, device, sentinel, chem_rewarder, relaxer, phys_rewarder):
    print(f"\n{'='*60}")
    print(f"🤖 EVALUATING: {model_name}")
    print(f"{'='*60}")
    
    generator = CrystalGenerator(model_path, CONFIG_PATH, device)
    generator.model.eval() 
    
    # Check if this model has a learned Lattice Bias (Phase 2), otherwise use default 3.0
    checkpoint = torch.load(model_path, map_location=device)
    bias_val = checkpoint.get("lattice_bias_value", 3.0)
    print(f"   📏 Using Global Lattice Bias: {bias_val:.4f}")
    
    metrics = {
        "valid_geom": 0, "pauling_passes": 0, "total_charge_error": 0.0,
        "relax_successes": 0, "total_energy": 0.0, "total_e_form": 0.0, 
        "total_e_hull": 0.0, "total_phys_reward": 0.0
    }
    
    valid_chem_count = 0
    valid_phys_count = 0
    
    with torch.no_grad(): # Global evaluation mode (Saves VRAM)
        for _ in tqdm(range(NUM_SAMPLES), desc=f"Evaluating"):
            outputs = generator.generate(BATCH_SIZE, allowed_elements=CAMPAIGN_ELEMENTS, temperature=TEMPERATURE)
            struct = outputs["structures"][0]
            
            if struct is None: continue
            
            # --- PHASE 1: CHEMISTRY CHECKS ---
            sentinel_results = sentinel.filter([struct])
            if sentinel_results[0].get("valid", False):
                metrics["valid_geom"] += 1
                valid_chem_count += 1
                
                p_pass, c_err = calculate_chemistry_metrics(struct)
                if p_pass: metrics["pauling_passes"] += 1
                metrics["total_charge_error"] += c_err
                
            # --- PHASE 2: PHYSICS CHECKS ---
            if len(struct) > 40: continue
            
            # Inflate Box
            target_volume = math.exp(bias_val) * len(struct)
            struct.scale_lattice(target_volume)

            # 🔹 THE FIX: Evaluation Pre-Sweep
            # Catch overlaps out to 0.61A BEFORE the relaxer's guard throws them away
            try:
                dists = struct.distance_matrix.copy()
                to_delete = set()
                for i in range(len(struct)):
                    for j in range(i + 1, len(struct)):
                        if dists[i, j] < 0.61:  
                            to_delete.add(j)
                if to_delete:
                    struct.remove_sites(sorted(list(to_delete), reverse=True))
            except Exception:
                pass
            
            # Relax
            with QuietBlock():
                # 🔹 THE FIX: Temporary Gradient Override
                # Turn the math engine back on JUST for the relaxer
                with torch.enable_grad():
                    relax_result = relaxer.relax(struct, steps=10)
                
            if relax_result["converged"]:
                metrics["relax_successes"] += 1
                valid_phys_count += 1
                final_struct = relax_result["final_structure"]
                energy_pa = relax_result["energy_per_atom"]
                
                metrics["total_energy"] += energy_pa
                
                # Thermodynamics
                struct_elements = set(final_struct.composition.elements)
                if len(struct_elements) <= 4:
                    try:
                        sys_entries = [e for e in phys_rewarder.mp_entries if set(e.composition.elements) <= struct_elements]
                        target_entry = phys_rewarder._generate_corrected_cse(final_struct, energy_pa * len(final_struct))
                        pd = PhaseDiagram(sys_entries + [target_entry])
                        
                        metrics["total_e_form"] += pd.get_form_energy_per_atom(target_entry)
                        metrics["total_e_hull"] += pd.get_e_above_hull(target_entry, allow_negative=True)
                        
                        reward = phys_rewarder.compute_reward(final_struct, energy_pa)
                        metrics["total_phys_reward"] += reward.item()
                    except Exception:
                        pass

    # Calculate Averages safely
    c_count = max(valid_chem_count, 1)
    p_count = max(valid_phys_count, 1)
    
    print("\n🧪 --- PHASE 1: CHEMISTRY RETENTION ---")
    print(f"🛡️ Geometric Survival Rate:  {(metrics['valid_geom']/NUM_SAMPLES)*100:.1f}%")
    print(f"⚡ Pauling Test Pass Rate:   {(metrics['pauling_passes']/c_count)*100:.1f}%")
    print(f"⚖️ Average Charge Error:     {metrics['total_charge_error']/c_count:.2f} electrons")
    
    print("\n⚛️ --- PHASE 2: PHYSICS UPGRADES ---")
    print(f"✅ Relaxation Success Rate:  {(metrics['relax_successes']/NUM_SAMPLES)*100:.1f}%")
    print(f"🔋 Avg Relaxed Energy:       {metrics['total_energy']/p_count:.3f} eV/atom")
    print(f"🔥 Avg Formation Energy:     {metrics['total_e_form']/p_count:.3f} eV/atom")
    print(f"🏔️ Avg E_hull (Stability):   {metrics['total_e_hull']/p_count:.3f} eV/atom")
    print(f"💰 Avg AI Physics Reward:    {metrics['total_phys_reward']/p_count:.3f}")
    
    del generator
    torch.cuda.empty_cache()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("🚀 STARTING FULL PIPELINE BENCHMARK")
    
    sentinel = CrystalSentinel(device)
    chem_rewarder = ChemistryRewardEngine(target_vol=25.0)
    relaxer = CrystalRelaxer(device=device)
    phys_rewarder = PhysicsRewardEngine(device=device)
    
    if os.path.exists(MODEL_BASE):
        run_evaluation(MODEL_BASE, "MODEL A (Phase 1 Baseline)", device, sentinel, chem_rewarder, relaxer, phys_rewarder)
    else:
        print(f"❌ Model A not found at: {MODEL_BASE}")

    if os.path.exists(MODEL_PHASE2):
        run_evaluation(MODEL_PHASE2, "MODEL B (Phase 2 Physicist - Epoch 60)", device, sentinel, chem_rewarder, relaxer, phys_rewarder)
    else:
        print(f"❌ Model B not found at: {MODEL_PHASE2}")
        
    print("\n🎯 BENCHMARK COMPLETE.")