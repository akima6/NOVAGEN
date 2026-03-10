import os
import sys
import torch
import pandas as pd
import numpy as np
import pynvml 
import warnings
import contextlib
import io
import logging
import time
import math
from tqdm import tqdm
from pymatgen.core import Structure

# ==========================================
# 🚑 DRIVER & PATH SHIMS
# ==========================================
try:
    sys.modules['nvidia_smi'] = pynvml
    os.environ['PATH'] += os.pathsep + r'C:\Windows\System32'
except Exception:
    pass

# ==========================================
# 🚑 ASE COMPATIBILITY PATCH
# ==========================================
import ase.constraints
sys.modules["ase.filters"] = ase.constraints
if not hasattr(ase.constraints, "ExpCellFilter"):
    if hasattr(ase.constraints, "UnitCellFilter"):
        ase.constraints.ExpCellFilter = ase.constraints.UnitCellFilter

# ==========================================
# 🔇 SILENCING SYSTEM
# ==========================================
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    logger.setLevel(logging.CRITICAL)

class QuietBlock:
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
# 🛠️ DYNAMIC PATHS & IMPORTS
# ==========================================
PROJECT_ROOT = os.getcwd()
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "core"))
sys.path.append(os.path.join(PROJECT_ROOT, "CrystalFormer"))
sys.path.append(os.path.join(PROJECT_ROOT, "rewards"))

try:
    from generator_service import CrystalGenerator
    from oracle import Phase3Oracle
    from relaxer import CrystalRelaxer 
except ImportError as e:
    sys.exit(f"❌ Critical Import Error: {e}")

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
N_CRYSTALS = 100  
BATCH_SIZE = 16         
SAVE_EVERY = 25         

MODEL_PATH = os.path.join(PROJECT_ROOT, "pretrained_model", "final_lab_grade_Solar_epoch_100.pt")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "pretrained_model", "config.yaml")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "UNIVERSAL_HARVEST_SEMICONDUCTORS")
CIF_DIR = os.path.join(OUTPUT_DIR, "cif_files")

CAMPAIGN_ELEMENTS = [
    3, 11, 19, 4, 12, 20, 38, 56, 
    5, 13, 31, 49, 6, 14, 32, 50, 
    7, 15, 33, 51, 8, 16, 34, 52,
    21, 22, 30, 48
]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_vram_usage():
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / 1024**2
    except:
        return 0.0

def get_material_type(bandgap):
    if bandgap < 0.05: return "Metal"
    elif bandgap > 3.5: return "Insulator"
    else: return "Semiconductor"

# ==========================================
# 🏭 MAIN FACTORY LOOP
# ==========================================
def run_harvest():
    print("="*80)
    print(f"🏭 FAIL-SAFE FACTORY: HARVESTING {N_CRYSTALS} CRYSTALS".center(80))
    print("="*80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CIF_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "final_harvest_results.csv")

    print(f"🔌 Initializing Main Engines (VRAM Usage: {get_vram_usage():.0f}MB)...")
    try:
        # Load the Model & extract the Lattice Bias for volume scaling
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        lattice_bias_val = checkpoint.get("lattice_bias_value", 3.0)
        print(f"   📏 Internal Lattice Bias Loaded: {lattice_bias_val:.4f}")

        gen = CrystalGenerator(MODEL_PATH, CONFIG_PATH, DEVICE)
        with QuietBlock():
            oracle = Phase3Oracle()   
            relaxer = CrystalRelaxer(device="cuda") # Loaded exactly ONCE!
        print("✅ Engines Ready. Commencing high-speed harvest...")
    except Exception as e:
        sys.exit(f"❌ Engine Init Failed: {e}")

    records = []
    saved_count = 0
    pbar = tqdm(total=N_CRYSTALS, desc="Harvesting", ncols=110)
    
    while saved_count < N_CRYSTALS:
        try:
            # 🧹 MEMORY PURGE
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # A. Batched Generation
            with torch.no_grad():
                out = gen.generate(BATCH_SIZE, allowed_elements=CAMPAIGN_ELEMENTS, temperature=0.7)
            
            structs = out["structures"]
            
            for struct in structs:
                if struct is None: continue

                # 🛑 PRE-FILTER: Avoid "Isolated Atom" Hangs
                if struct.density < 0.8 or struct.volume > 2000 or struct.num_sites > 60:
                    continue 

                # 📏 BUGFIX 1: Scale the lattice BEFORE passing to CHGNet!
                struct.scale_lattice(math.exp(lattice_bias_val) * len(struct))

                # ⏱️ B. Physics Relaxation (Now fast because lattice is scaled)
                res = None
                with QuietBlock():
                    with torch.enable_grad(): # CHGNet needs gradients to compute forces
                        res = relaxer.relax(struct, steps=25)
                
                # Check if relaxation failed
                if not res or not res.get("converged"): 
                    continue
                
                final_struct = res["final_structure"]
                energy = res["energy_per_atom"]

                if energy > 0.0: continue 

                # C. Bandgap
                gap = 0.0
                with QuietBlock():
                    try:
                        gap = oracle.evaluate_structure(final_struct)
                    except: 
                        gap = -1.0

                # 🛑 BUGFIX 2: The Oracle Poison Pill Trap
                if gap < 0.0:
                    continue

                # D. Record and Save
                formula = final_struct.composition.reduced_formula
                try:
                    sg_symbol, sg_number = final_struct.get_space_group_info()
                except:
                    sg_symbol, sg_number = "P1", 1
                
                saved_count += 1
                file_name = f"{formula}_id{saved_count}_E{energy:.2f}.cif"
                final_struct.to(filename=os.path.join(CIF_DIR, file_name))

                records.append({
                    "file_name": file_name,
                    "formula": formula,
                    "energy": round(energy, 4),
                    "bandgap": round(gap, 3),
                    "type": get_material_type(gap),
                    "space_group_symbol": sg_symbol,
                    "space_group_number": sg_number,
                    "num_atoms": final_struct.num_sites
                })
                
                pbar.update(1)
                pbar.set_postfix({
                    "VRAM": f"{get_vram_usage():.0f}MB",
                    "Saved": saved_count
                })

                # E. Periodic Save
                if saved_count % SAVE_EVERY == 0:
                    pd.DataFrame(records).to_csv(csv_path, index=False)
                
                if saved_count >= N_CRYSTALS: break

        except Exception as e:
            # 🚨 BUGFIX 3: LOUD ERROR Catching
            pbar.write(f"\n🚨 CRITICAL LOOP ERROR: {e}")
            time.sleep(2)  
            continue

    pbar.close()
    if records:
        df = pd.DataFrame(records)
        df.sort_values(by="energy", ascending=True).to_csv(csv_path, index=False)
        print(f"\n✅ Success! {saved_count} crystals harvested in {OUTPUT_DIR}")

if __name__ == "__main__":
    run_harvest()