import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
import contextlib

# --- CONFIGURATION ---
TARGET_CANDIDATES = 5000   # Target count
BATCH_SIZE = 50             # Keep small to avoid OOM
MAX_ATTEMPTS = 5000        # Safety limit (Batches, not crystals)

# Fe, O, S, Si, N
CAMPAIGN_ELEMENTS = [26, 8, 16, 14, 7] 

# PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "CrystalFormer"))
sys.path.append(BASE_DIR)
SAVE_DIR = os.path.join(BASE_DIR, "rl_discoveries")
os.makedirs(SAVE_DIR, exist_ok=True)

warnings.filterwarnings("ignore")

# --- SILENCE HELPER ---
# This class mutes the "Sampling" bar from the generator
class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

try:
    from generator_service import CrystalGenerator
    from sentinel import CrystalSentinel
    # We try to import Spglib for polishing
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        HAS_SPGLIB = True
    except ImportError:
        HAS_SPGLIB = False

except ImportError as e:
    sys.exit(f"Setup Error: {e}")

def run_discovery():
    print("==================================================")
    print(f"🚀 STARTING DISCOVERY CAMPAIGN (Silent Mode)")
    print(f"   Target: {TARGET_CANDIDATES} Candidates")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Saving to: {SAVE_DIR}")
    print("==================================================")

    # 1. Initialize
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config_path = os.path.join(BASE_DIR, "pretrained_model", "config.yaml")
    model_path = os.path.join(BASE_DIR, "rl_checkpoints", "epoch_100_RL.pt")
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return

    # Initialize models
    # We mute initialization noise too
    with SuppressOutput():
        generator = CrystalGenerator(model_path, config_path, device)
    
    sentinel = CrystalSentinel(device)
    print(f"   ✅ Engine Loaded. Mining started...")

    # 2. Main Loop
    valid_count = 0
    batch_idx = 0
    all_metadata = []
    
    # The ONLY progress bar you will see
    pbar = tqdm(total=TARGET_CANDIDATES, desc="   Mining", unit="cryst", dynamic_ncols=True)

    while valid_count < TARGET_CANDIDATES and batch_idx < MAX_ATTEMPTS:
        batch_idx += 1
        
        # A. Generate (Silently)
        try:
            with SuppressOutput():
                structures = generator.generate(BATCH_SIZE, temperature=1.0, allowed_elements=CAMPAIGN_ELEMENTS)
        except RuntimeError as e:
            if "out of memory" in str(e):
                # We can print this because we are outside the SuppressOutput block
                pbar.write("   ⚠️ OOM. Clearing Cache...")
                torch.cuda.empty_cache()
                continue
            else:
                raise e

        # B. Filter
        valid_mask, _ = sentinel.filter(structures)
        
        # C. Process Winners
        new_finds = 0
        for i, is_valid in enumerate(valid_mask):
            if is_valid:
                struct = structures[i]
                formula = struct.composition.reduced_formula
                
                # Polish (Symmetry)
                final_struct = struct
                sg_num = 1
                sg_symbol = "P1"
                
                if HAS_SPGLIB:
                    try:
                        sga = SpacegroupAnalyzer(struct, symprec=0.1)
                        refined = sga.get_refined_structure()
                        if refined:
                            final_struct = refined
                            sg_num = sga.get_space_group_number()
                            sg_symbol = sga.get_space_group_symbol()
                    except:
                        pass
                
                # Save
                filename = f"{formula}_{valid_count:05d}.cif"
                save_path = os.path.join(SAVE_DIR, filename)
                final_struct.to(filename=save_path)
                
                all_metadata.append({
                    "File": filename,
                    "Formula": formula,
                    "SpaceGroup": sg_num,
                    "Symbol": sg_symbol,
                    "NumAtoms": len(final_struct)
                })
                
                valid_count += 1
                new_finds += 1
                pbar.update(1)
                
                if valid_count >= TARGET_CANDIDATES:
                    break
        
        # Cleanup
        del structures
        torch.cuda.empty_cache()
        
    pbar.close()
    
    # 3. Report
    if all_metadata:
        df = pd.DataFrame(all_metadata)
        summary_path = os.path.join(SAVE_DIR, "summary_report.csv")
        df.to_csv(summary_path, index=False)
        print("\n" + "="*50)
        print(f"🎉 DONE! Found {valid_count} candidates.")
        print(f"   Symmetric Crystals: {len(df[df['SpaceGroup'] > 1])}")
        print(f"   Report: {summary_path}")
        print("==================================================")
    else:
        print("\n⚠️ Found 0 candidates.")

if __name__ == "__main__":
    run_discovery()
