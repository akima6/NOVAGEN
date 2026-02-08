import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings
import contextlib
import io
import logging
import types
from tqdm import tqdm
from pymatgen.core import Structure

# =============================================================================
# 🛑 THE GREAT SILENCER & GPU EMULATOR (Prevents nvidia_smi crashes)
# =============================================================================
# 1. Mock nvidia_smi to satisfy CHGNet
# dummy_nv = types.ModuleType("nvidia_smi")
# dummy_nv.nvmlInit = lambda: None
# dummy_nv.nvmlDeviceGetCount = lambda: 1  
# dummy_nv.nvmlDeviceGetHandleByIndex = lambda i: 0
# class MockMemory:
#     free = 8589934592 # 8 GB
# dummy_nv.nvmlDeviceGetMemoryInfo = lambda h: MockMemory()
# sys.modules["nvidia_smi"] = dummy_nv

import pynvml
sys.modules['nvidia_smi'] = pynvml

# 2. Fix ASE Version Mismatch
import ase.constraints
try:
    import ase.filters
except ImportError:
    sys.modules["ase.filters"] = ase.constraints

# =============================================================================
# 🔇 LOGGING CONTROL
# =============================================================================
warnings.filterwarnings("ignore")
loggers_to_mute = ["chgnet", "dgl", "pymatgen"]
for name in logging.root.manager.loggerDict:
    for mute_key in loggers_to_mute:
        if mute_key in name.lower():
            logger = logging.getLogger(name)
            logger.setLevel(logging.CRITICAL) 
            logger.propagate = False

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
# ⚙️ CONFIGURATION
# ==========================================
# Input: The 815 Realistic Semiconductors
INPUT_CSV = r"C:\Users\REHNA\NOVAGEN\LIGHT_CANDIDATES_FINAL\LIGHT_semiconductors.csv"
INPUT_CIF_DIR = r"C:\Users\REHNA\NOVAGEN\LIGHT_CANDIDATES_FINAL\cif"

# Output: The Final Survivors
OUTPUT_DIR = r"C:\Users\REHNA\NOVAGEN\LIGHT_CANDIDATES_FINAL"
OUTPUT_CIF_DIR = os.path.join(OUTPUT_DIR, "relaxed_cif_files")

# Simulation Settings
RELAX_STEPS = 500  # Deep Clean
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 🛠️ SETUP
# ==========================================
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "CrystalFormer"))

try:
    from product_oracle import CrystalOracle
    from product_relaxer import CrystalRelaxer 
except ImportError as e:
    sys.exit(f"❌ Critical Import Error: {e}")

def get_material_type(bandgap):
    if bandgap < 0.05: return "Metal"
    elif bandgap > 3.5: return "Insulator"
    else: return "Semiconductor"

def run_final_refinement():
    print("="*80)
    print("🔬 FINAL PHYSICS VALIDATION".center(80))
    print("="*80)

    if not os.path.exists(INPUT_CSV):
        sys.exit(f"❌ CSV not found: {INPUT_CSV}")
    
    df = pd.read_csv(INPUT_CSV)
    print(f"   📂 Loaded Dataset: {len(df)} candidates")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_CIF_DIR, exist_ok=True)

    print("   🔌 Initializing Engines...")
    oracle = CrystalOracle(device="cpu")   
    relaxer = CrystalRelaxer(device=DEVICE) 
    print(f"   ✅ Ready (Device: {DEVICE})")

    refined_records = []
    
    # We use a loop with explicit error catching to see if something breaks
    pbar = tqdm(total=len(df), desc="Refining", ncols=100)
    
    for _, row in df.iterrows():
        original_file = row["file_name"]
        cif_path = os.path.join(INPUT_CIF_DIR, original_file)
        
        if not os.path.exists(cif_path):
            pbar.update(1)
            continue

        try:
            # A. Load Structure
            struct = Structure.from_file(cif_path)
            
            # B. Deep Relaxation (500 Steps)
            res = None
            with QuietBlock():
                try:
                    res = relaxer.relax(struct, steps=RELAX_STEPS)
                except Exception:
                    res = None # Hard Fail
            
            # C. Check for Failure or Instability
            if not res or not res["converged"]:
                pbar.update(1)
                continue
                
            # STRICT FILTER: Must have Energy < 0.0
            if res["energy_per_atom"] > 0.0:
                pbar.update(1)
                continue
                
            final_struct = res["final_structure"]
            new_energy = res["energy_per_atom"]

            # D. Re-Calculate Bandgap (Did it turn into a metal?)
            new_gap = 0.0
            with QuietBlock():
                try:
                    _, gaps = oracle.predict_batch([final_struct])
                    new_gap = float(gaps[0])
                except:
                    new_gap = 0.0
            
            # E. Save New File
            formula = final_struct.composition.reduced_formula
            mat_type = get_material_type(new_gap)
            
            # Naming Convention: Formula_ID_Final_E-1.23.cif
            # Robust split to handle various filenames
            base_id = original_file.split("_E")[0] 
            new_filename = f"{base_id}_Final_E{new_energy:.2f}.cif"
            save_path = os.path.join(OUTPUT_CIF_DIR, new_filename)
            
            final_struct.to(filename=save_path)
            
            # F. Save Space Group (Useful for Step 3 later)
            try:
                sg_symbol, sg_num = final_struct.get_space_group_info()
            except:
                sg_symbol, sg_num = "P1", 1

            refined_records.append({
                "file_name": new_filename,
                "original_file": original_file,
                "formula": formula,
                "energy": round(new_energy, 4),
                "bandgap": round(new_gap, 3),
                "type": mat_type,
                "space_group_symbol": sg_symbol,
                "space_group_number": sg_num,
                "num_atoms": final_struct.num_sites
            })
            
        except Exception as e:
            # If a single crystal crashes, we skip it but keep going
            pass
        
        pbar.update(1)

    pbar.close()

    if refined_records:
        new_df = pd.DataFrame(refined_records)
        csv_path = os.path.join(OUTPUT_DIR, "final_relaxed_results.csv")
        new_df.sort_values(by="energy", inplace=True)
        new_df.to_csv(csv_path, index=False)
        
        print("\n" + "="*80)
        print("📊 VALIDATION COMPLETE".center(80))
        print("="*80)
        print(f"   🔹 Input Count:     {len(df)}")
        print(f"   💾 Survived:        {len(refined_records)}")
        print(f"   📂 Output Folder:   {OUTPUT_DIR}")
        print("-" * 80)
        print(new_df[['formula', 'energy', 'bandgap', 'type']].head(5).to_string(index=False))
    else:
        print("\n❌ No crystals survived the relaxation.")

if __name__ == "__main__":
    run_final_refinement()