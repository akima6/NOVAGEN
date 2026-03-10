import os
import sys
import torch
import warnings
import math
import gc
import contextlib
import io
import pynvml 
from time import time
import logging

# =============================================================================
# 🚑 CRITICAL DRIVER & COMPATIBILITY SHIMS
# =============================================================================
# 1. GPU Shim: Route 'nvidia_smi' calls to 'pynvml' so CHGNet doesn't crash
sys.modules['nvidia_smi'] = pynvml

# 2. Path Shim: Ensure Windows finds the necessary DLLs
os.environ['PATH'] += os.pathsep + r'C:\Windows\System32'

# 3. ASE Compatibility Fix (Crucial for newer ASE / CHGNet versions)
import ase.constraints
sys.modules["ase.filters"] = ase.constraints
if not hasattr(ase.constraints, "ExpCellFilter"):
    if hasattr(ase.constraints, "UnitCellFilter"):
        ase.constraints.ExpCellFilter = ase.constraints.UnitCellFilter

# =============================================================================
# 🔇 AGGRESSIVE SILENCING SYSTEM
# =============================================================================
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["DGLBACKEND"] = "pytorch"

loggers_to_mute = ["chgnet", "dgl", "pymatgen", "matgl"]
for name in logging.root.manager.loggerDict:
    if any(mute_key in name.lower() for mute_key in loggers_to_mute):
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

# =============================================================================
# 🛠️ SETUP & IMPORTS
# =============================================================================
BASE_DIR = os.getcwd()
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "CrystalFormer"))
sys.path.append(os.path.join(BASE_DIR, "core"))
sys.path.append(os.path.join(BASE_DIR, "rewards"))

try:
    from generator_service import CrystalGenerator
    from sentinel import CrystalSentinel
    from oracle import CrystalOracle     # Ensure this matches your filename
    from relaxer import CrystalRelaxer   # Ensure this matches your filename
except ImportError as e:
    sys.exit(f"❌ Critical Import Error: {e}\nCheck your PYTHONPATH or folder structure.")

# =============================================================================
# ⚙️ CONFIGURATION
# =============================================================================
MODEL_PATH = os.path.join(BASE_DIR, "pretrained_model", "physicist_epoch_60.pt")
CONFIG_PATH = os.path.join(BASE_DIR, "pretrained_model", "config.yaml")

NUM_SAMPLES = 5
# Atomic numbers for Ti (22), O (8), N (7) - A great system for testing bandgaps
TEST_ELEMENTS = [22, 8, 7] 

def test_pipeline():
    print("="*85)
    print("🧪 NOVAGEN FULL PIPELINE DIAGNOSTIC".center(85))
    print("="*85)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   🚀 Hardware Acceleration: {device.upper()}")

    # ---------------------------------------------------------
    # 1. LOAD GENERATOR & EXTRACT LATTICE BIAS
    # ---------------------------------------------------------
    print("\n[1/4] Loading Generator (Phase 2)...")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        return
    
    try:
        generator = CrystalGenerator(MODEL_PATH, CONFIG_PATH, device)
        generator.model.eval()
        
        # Extract the learned Lattice Bias to prevent implosions
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        bias_val = checkpoint.get("lattice_bias_value", 3.0)
        print(f"   ✅ Generator Loaded. Learned Lattice Bias: {bias_val:.4f}")
    except Exception as e:
        print(f"   ❌ Generator Failed: {e}")
        return

    # ---------------------------------------------------------
    # 2. INITIALIZE GATEKEEPERS (Sentinel)
    # ---------------------------------------------------------
    print("\n[2/4] Initializing Sentinel (Geometry Gatekeeper)...")
    try:
        sentinel = CrystalSentinel(device)
        print("   ✅ Sentinel Online.")
    except Exception as e:
        print(f"   ❌ Sentinel Failed: {e}")
        return

    # ---------------------------------------------------------
    # 3. LOAD PHYSICS & ELECTRONIC ENGINES
    # ---------------------------------------------------------
    print("\n[3/4] Initializing Physics Engines...")
    with QuietBlock():
        try:
            relaxer = CrystalRelaxer(device=device) 
            relaxer_status = "✅ Relaxer (CHGNet) Online."
        except Exception as e:
            relaxer_status = f"❌ Relaxer Failed: {e}"
            relaxer = None

        try:
            oracle = CrystalOracle(device="cpu")    
            oracle_status = "✅ Oracle (MEGNet) Online."
        except Exception as e:
            oracle_status = f"❌ Oracle Failed: {e}"
            oracle = None

    print(f"   {relaxer_status}")
    print(f"   {oracle_status}")
    
    # ---------------------------------------------------------
    # 4. RUN TEST BATCH
    # ---------------------------------------------------------
    print(f"\n[4/4] Generating {NUM_SAMPLES} Test Crystals (Ti-O-N System)...")
    print("-" * 85)
    print(f"{'Formula':<12} | {'Sent.':<6} | {'Relaxer':<12} | {'Energy (eV)':<15} | {'Bandgap (eV)'}")
    print("-" * 85)

    start_time = time()

    try:
        with torch.no_grad():
            outputs = generator.generate(NUM_SAMPLES, TEST_ELEMENTS, temperature=0.7)
        
        structures = outputs["structures"]

        for i, struct in enumerate(structures):
            if struct is None:
                print(f"{'Invalid':<12} | {'FAIL':<6} | {'Skipped':<12} | {'N/A':<15} | {'N/A'}")
                continue

            formula_raw = struct.composition.reduced_formula
            energy = "N/A"
            gap = "N/A"
            relax_status = "Pending"

            # A. Sentinel Check
            sent_res = sentinel.filter([struct])[0]
            if not sent_res.get("valid", False):
                print(f"{formula_raw:<12} | {'FAIL':<6} | {'Skipped':<12} | {'N/A':<15} | {'N/A'}")
                continue

            # B. Apply Phase 2 Lattice Bias (Inflate the box)
            target_volume = math.exp(bias_val) * len(struct)
            struct.scale_lattice(target_volume)

            # C. Relax (CHGNet)
            if relaxer is not None:
                with QuietBlock():
                    # Turn on gradients locally for CHGNet FIRE optimizer
                    with torch.enable_grad():
                        relax_res = relaxer.relax(struct, steps=25)
                
                final_struct = relax_res["final_structure"] if relax_res["final_structure"] else struct
                energy = relax_res["energy_per_atom"]
                if relax_res["converged"]:
                    relax_status = "Converged"
                else:
                    relax_status = "Unconverged"
            else:
                final_struct = struct
                relax_status = "NoEngine"

            # D. Predict Properties (MEGNet)
            if oracle is not None and relax_status in ["Converged", "NoEngine", "Unconverged"]:
                with QuietBlock():
                    try:
                        _, gaps = oracle.predict_batch([final_struct])
                        gap = gaps[0]
                    except Exception:
                        gap = "Error"

            # Format outputs cleanly
            e_str = f"{energy:.4f}" if isinstance(energy, float) else str(energy)
            g_str = f"{gap:.4f}" if isinstance(gap, float) else str(gap)
            sent_str = "PASS"

            print(f"{formula_raw:<12} | {sent_str:<6} | {relax_status:<12} | {e_str:<15} | {g_str}")

            # Memory cleanup per loop
            del final_struct
            gc.collect()

    except Exception as e:
        print(f"\n❌ Pipeline Execution Error: {e}")

    print("-" * 85)
    print(f"⏱️  Total Diagnostic Time: {time() - start_time:.2f}s")
    print("="*85)
    print("✅ TEST COMPLETE".center(85))

if __name__ == "__main__":
    test_pipeline()