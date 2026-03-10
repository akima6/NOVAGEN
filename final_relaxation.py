import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings
import contextlib
import io
import logging
import itertools
from tqdm import tqdm
from pymatgen.core import Structure, Composition
import pynvml

# =============================================================================
# 🛑 DRIVER FIX 
# =============================================================================
try:
    sys.modules['nvidia_smi'] = pynvml
    os.environ['PATH'] += os.pathsep + r'C:\Windows\System32'
except Exception:
    pass

# =============================================================================
# 🛑 ASE COMPATIBILITY PATCH 
# =============================================================================
import ase.constraints
sys.modules["ase.filters"] = ase.constraints
if not hasattr(ase.constraints, "ExpCellFilter"):
    if hasattr(ase.constraints, "UnitCellFilter"):
        ase.constraints.ExpCellFilter = ase.constraints.UnitCellFilter

# =============================================================================
# 🔇 SILENCING SYSTEM 
# =============================================================================
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

# =============================================================================
# 🛠️ DYNAMIC PATHS 
# =============================================================================
PROJECT_ROOT = os.getcwd()

sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "core"))
sys.path.append(os.path.join(PROJECT_ROOT, "CrystalFormer"))
sys.path.append(os.path.join(PROJECT_ROOT, "rewards"))

# =============================================================================
# 📂 INPUT / OUTPUT PATHS 
# =============================================================================
INPUT_CSV = os.path.join(
    PROJECT_ROOT,
    "UNIVERSAL_HARVEST_SEMICONDUCTORS",
    "final_harvest_results.csv"
)

INPUT_CIF_DIR = os.path.join(
    PROJECT_ROOT,
    "UNIVERSAL_HARVEST_SEMICONDUCTORS",
    "cif_files"
)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "DEEP_VALIDATION_SEMICONDUCTORS")
OUTPUT_CIF_DIR = os.path.join(OUTPUT_DIR, "relaxed_cif_files")

# =============================================================================
# ⚙️ SIMULATION SETTINGS
# =============================================================================
RELAX_STEPS = 500
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# 🧠 IMPORT ENGINES 
# =============================================================================
try:
    from oracle import Phase3Oracle
    from relaxer import CrystalRelaxer
    import smact
except ImportError as e:
    sys.exit(f"❌ Critical Import Error: {e}")

def get_material_type(bandgap):
    if bandgap < 0.05: return "Metal"
    elif bandgap > 3.5: return "Insulator"
    else: return "Semiconductor"

def passes_chemistry_triage(formula):
    """The Chemistry Triage: Filters out nonsense stoichiometries instantly."""
    try:
        comp = Composition(formula)
        comp_dict = comp.get_el_amt_dict()
        elements = list(comp_dict.keys())
        
        # Rule 1: No Single Elements (We want novel compounds)
        if len(elements) < 2:
            return False
            
        # Rule 2: Charge Neutrality (Oxidation States Must Balance)
        ox_states_list = [smact.Element(el).oxidation_states or [0] for el in elements]
        counts = list(comp_dict.values())
        
        charge_balanced = False
        for combo in itertools.product(*ox_states_list):
            net_charge = sum(ox * count for ox, count in zip(combo, counts))
            if abs(net_charge) == 0:
                charge_balanced = True
                break
                
        if not charge_balanced:
            return False
            
        return True
    except Exception:
        return False # If Pymatgen/SMACT crashes on the formula, it's junk.

def run_final_refinement():
    print("="*80)
    print("🔬 DEEP PHYSICS VALIDATION: SEMICONDUCTORS ONLY".center(80))
    print("="*80)

    if not os.path.exists(INPUT_CSV):
        sys.exit(f"❌ CSV not found: {INPUT_CSV}")
    
    # 1. Load the raw harvest data
    df = pd.read_csv(INPUT_CSV)
    initial_count = len(df)
    
    # 2. Strict Filter: Semiconductors Only
    df = df[df['type'].str.lower() == 'semiconductor']
    semi_count = len(df)
    
    # 3. Basic Deduplication
    df = df.sort_values('energy').drop_duplicates(subset=['formula'], keep='first')
    dedup_count = len(df)
    
    # 4. 🛑 THE CHEMISTRY TRIAGE 🛑
    tqdm.pandas(desc="Chemistry Triage")
    df['chemically_valid'] = df['formula'].progress_apply(passes_chemistry_triage)
    df = df[df['chemically_valid'] == True]
    triage_count = len(df)
    
    print(f"\n   📂 Raw Harvest:         {initial_count} crystals")
    print(f"   🎯 Semiconductors:      {semi_count} crystals")
    print(f"   🧹 Unique Candidates:   {dedup_count} crystals")
    print(f"   🧪 Passed Triage:       {triage_count} crystals ready for 500-step physics")

    if triage_count == 0:
        print("\n❌ No chemically valid semiconductors found to relax.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_CIF_DIR, exist_ok=True)

    print("\n   🔌 Initializing Engines...")
    with QuietBlock():
        oracle = Phase3Oracle()  
        relaxer = CrystalRelaxer(device=DEVICE) 
    print(f"   ✅ Ready (Device: {DEVICE})")

    refined_records = []
    pbar = tqdm(total=len(df), desc="Refining", ncols=100)
    
    for _, row in df.iterrows():
        try:
            # 🧹 MEMORY PURGE
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            original_file = row["file_name"]
            cif_path = os.path.join(INPUT_CIF_DIR, original_file)
            
            if not os.path.exists(cif_path):
                pbar.update(1)
                continue

            # A. Load Structure
            struct = Structure.from_file(cif_path)
            
            # B. Deep Relaxation (500 Steps)
            res = None
            with QuietBlock():
                with torch.enable_grad():
                    try:
                        res = relaxer.relax(struct, steps=RELAX_STEPS)
                    except Exception:
                        res = None
            
            # C. Check for Failure or Instability
            if not res or not res.get("converged"):
                pbar.update(1)
                continue
                
            if res["energy_per_atom"] > 0.0:
                pbar.update(1)
                continue
                
            final_struct = res["final_structure"]
            new_energy = res["energy_per_atom"]

            # D. Re-Calculate Bandgap
            new_gap = 0.0
            with QuietBlock():
                try:
                    new_gap = oracle.evaluate_structure(final_struct)
                except:
                    new_gap = -1.0
            
            if new_gap < 0.0:
                pbar.update(1)
                continue

            # Check if it remained a semiconductor
            mat_type = get_material_type(new_gap)
            if mat_type != "Semiconductor":
                pbar.update(1)
                continue
            
            # E. Save New File
            formula = final_struct.composition.reduced_formula
            base_id = original_file.split("_E")[0] 
            new_filename = f"{base_id}_Final_E{new_energy:.2f}.cif"
            save_path = os.path.join(OUTPUT_CIF_DIR, new_filename)
            
            final_struct.to(filename=save_path)
            
            # F. Save Space Group 
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
            pass
        
        pbar.update(1)

    pbar.close()

    if refined_records:
        new_df = pd.DataFrame(refined_records)
        csv_path = os.path.join(OUTPUT_DIR, "final_relaxed_results.csv")
        new_df.sort_values(by="energy", inplace=True)
        new_df.to_csv(csv_path, index=False)
        
        print("\n" + "="*80)
        print("📊 DEEP VALIDATION COMPLETE".center(80))
        print("="*80)
        print(f"   🔹 Input Unique Semiconductors: {triage_count}")
        print(f"   💾 Survived 500 Steps:          {len(refined_records)}")
        print(f"   📂 Output Folder:               {OUTPUT_DIR}")
        print("-" * 80)
        print(new_df[['formula', 'energy', 'bandgap']].head(10).to_string(index=False))
    else:
        print("\n❌ No crystals survived the deep relaxation.")

if __name__ == "__main__":
    run_final_refinement()