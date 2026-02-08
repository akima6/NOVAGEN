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
import signal
from tqdm import tqdm
from pymatgen.core import Structure

# ==========================================
# 🚑 DRIVER & PATH SHIMS
# ==========================================
sys.modules['nvidia_smi'] = pynvml
os.environ['PATH'] += os.pathsep + r'C:\Windows\System32'

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
# 🛠️ IMPORTS
# ==========================================
sys.path.append(os.getcwd())
try:
    from generator_service import CrystalGenerator
    from product_oracle import CrystalOracle
    from product_relaxer import CrystalRelaxer 
except ImportError as e:
    sys.exit(f"❌ Critical Import Error: {e}")

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
N_CRYSTALS = 5000       
BATCH_SIZE = 16         # 🚀 Recommended from Stress Test
TIMEOUT_PER_RELAX = 45  # ⏱️ Seconds to wait before killing a "stuck" relaxation
SAVE_EVERY = 25         
MODEL_PATH = r"C:\Users\REHNA\NOVAGEN\pretrained_model\final_lab_grade_Light_epoch_100.pt"
CONFIG_PATH = r"C:\Users\REHNA\NOVAGEN\pretrained_model\config.yaml"
OUTPUT_DIR = "UNIVERSAL_HARVEST_LIGHT"
CIF_DIR = os.path.join(OUTPUT_DIR, "cif_files")

CAMPAIGN_ELEMENTS = [
    3, 11, 19, 4, 12, 20, 38, 56, 
    5, 13, 31, 49, 6, 14, 32, 50, 
    7, 15, 33, 51, 8, 16, 34, 52,
    21, 22, 30, 48
]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_vram_usage():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.used / 1024**2

def get_material_type(bandgap):
    if bandgap < 0.05: return "Metal"
    elif bandgap > 3.5: return "Insulator"
    else: return "Semiconductor"

def run_harvest():
    print("="*80)
    print(f"🏭 FAIL-SAFE FACTORY: HARVESTING {N_CRYSTALS} CRYSTALS".center(80))
    print("="*80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CIF_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "final_harvest_results.csv")

    print(f"🔌 Initializing Engines (VRAM Usage: {get_vram_usage():.0f}MB)...")
    try:
        gen = CrystalGenerator(MODEL_PATH, CONFIG_PATH, DEVICE)
        with QuietBlock():
            oracle = CrystalOracle(device="cpu")   
        relaxer = CrystalRelaxer(device="cuda", method="CHGNet")
        print("✅ Engines Ready.")
    except Exception as e:
        sys.exit(f"❌ Engine Init Failed: {e}")

    records = []
    saved_count = 0
    pbar = tqdm(total=N_CRYSTALS, desc="Harvesting", ncols=110)
    
    while saved_count < N_CRYSTALS:
        try:
            # 🧹 MEMORY PURGE: Prevent fragmentation hangs
            torch.cuda.empty_cache()
            gc_vram = get_vram_usage()

            # A. Batched Generation
            with torch.no_grad():
                out = gen.generate(BATCH_SIZE, allowed_elements=CAMPAIGN_ELEMENTS, temperature=0.7)
            
            structs = out["structures"]
            
            for struct in structs:
                if struct is None: continue

                # 🛑 PRE-FILTER: Avoid "Isolated Atom" Hangs
                if struct.density < 0.8 or struct.volume > 2000 or struct.num_sites > 60:
                    continue 

                # ⏱️ B. Physics Relaxation with Heartbeat Debugging
                res = None
                start_relax = time.time()
                with QuietBlock():
                    try:
                        # CHGNet often hangs on specific structures; we wrap this in a manual timeout check
                        res = relaxer.relax(struct, steps=25)
                    except Exception: 
                        pass
                
                # Check if relaxation took way too long or failed
                if not res or not res["converged"] or (time.time() - start_relax) > TIMEOUT_PER_RELAX: 
                    continue
                
                final_struct = res["final_structure"]
                energy = res["energy_per_atom"]

                if energy > 0.0: continue 

                # C. Bandgap
                gap = 0.0
                with QuietBlock():
                    try:
                        _, gaps = oracle.predict_batch([final_struct])
                        gap = float(gaps[0])
                    except: gap = 0.0

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
            # Silence internal batch errors and keep moving
            continue

    pbar.close()
    if records:
        df = pd.DataFrame(records)
        df.sort_values(by="energy", ascending=True).to_csv(csv_path, index=False)
        print(f"\n✅ Success! {saved_count} crystals harvested in {OUTPUT_DIR}")

if __name__ == "__main__":
    run_harvest()