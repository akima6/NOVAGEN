import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
import time

# --- CONFIGURATION ---
TARGET_CANDIDATES = 10000   # How many valid crystals we want
BATCH_SIZE = 50             # Keep this LOW (50) to prevent OOM
MAX_ATTEMPTS = 500          # Safety limit on batches to prevent infinite loops

# Fe, O, S, Si, N (Matches your training)
CAMPAIGN_ELEMENTS = [26, 8, 16, 14, 7] 

# PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "CrystalFormer"))
sys.path.append(BASE_DIR)
SAVE_DIR = os.path.join(BASE_DIR, "rl_discoveries")
os.makedirs(SAVE_DIR, exist_ok=True)

warnings.filterwarnings("ignore")

try:
    from generator_service import CrystalGenerator
    from sentinel import CrystalSentinel
    # We import Relaxer/Oracle only if needed, but for raw discovery speed 
    # we often skip full relaxation unless you have 24 hours to wait.
    # For this script, we will do Generation + Sentinel + Symmetry Polish.
    # (Full physics relaxation on 10,000 crystals takes ~2 days on Kaggle).
except ImportError as e:
    sys.exit(f"Setup Error: {e}")

def run_discovery():
    print("==================================================")
    print(f"🚀 STARTING DISCOVERY CAMPAIGN (Fixed Batch Size: {BATCH_SIZE})")
    print(f"   Target: {TARGET_CANDIDATES} Valid Candidates")
    print(f"   Saving to: {SAVE_DIR}")
    print("==================================================")

    # 1. Initialize
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the TRAINED model (Epoch 100)
    config_path = os.path.join(BASE_DIR, "pretrained_model", "config.yaml")
    model_path = os.path.join(BASE_DIR, "rl_checkpoints", "epoch_100_RL.pt")
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        print("   Did you finish training?")
        return

    generator = CrystalGenerator(model_path, config_path, device)
    sentinel = CrystalSentinel(device)
    
    # We use Spglib for symmetry (The "Polish" Upgrade)
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        print("   ✅ Symmetry Refinement (Spglib) Enabled.")
    except ImportError:
        print("   ⚠️ Spglib not found. Symmetry polish disabled.")
        SpacegroupAnalyzer = None

    print(f"   ✅ Loaded Generator from Epoch 100.")

    # 2. Main Loop
    valid_count = 0
    batch_idx = 0
    all_metadata = []
    
    pbar = tqdm(total=TARGET_CANDIDATES, desc="   Mining", unit="cryst")

    while valid_count < TARGET_CANDIDATES and batch_idx < MAX_ATTEMPTS:
        batch_idx += 1
        
        # A. Generate
        try:
            # Note: allowed_elements uses your specific campaign list
            structures = generator.generate(BATCH_SIZE, temperature=1.0, allowed_elements=CAMPAIGN_ELEMENTS)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("\n❌ OOM during generation. Clearing cache and retrying...")
                torch.cuda.empty_cache()
                continue
            else:
                raise e

        # B. Filter (Sentinel)
        # Using the new "Smart Radii" check you installed
        valid_mask, _ = sentinel.filter(structures)
        
        # C. Process Winners
        for i, is_valid in enumerate(valid_mask):
            if is_valid:
                struct = structures[i]
                formula = struct.composition.reduced_formula
                
                # --- UPGRADE #5: SYMMETRY POLISH ---
                final_struct = struct
                sg_num = 1
                sg_symbol = "P1"
                
                if SpacegroupAnalyzer:
                    try:
                        sga = SpacegroupAnalyzer(struct, symprec=0.1)
                        refined = sga.get_refined_structure()
                        if refined:
                            final_struct = refined
                            sg_num = sga.get_space_group_number()
                            sg_symbol = sga.get_space_group_symbol()
                    except:
                        pass # Fallback to raw structure
                # -----------------------------------
                
                # Save CIF
                filename = f"{formula}_{valid_count:05d}.cif"
                save_path = os.path.join(SAVE_DIR, filename)
                final_struct.to(filename=save_path)
                
                # Log Data
                all_metadata.append({
                    "File": filename,
                    "Formula": formula,
                    "SpaceGroup": sg_num,
                    "Symbol": sg_symbol,
                    "NumAtoms": len(final_struct),
                    "Volume": final_struct.volume,
                    "Density": final_struct.density
                })
                
                valid_count += 1
                pbar.update(1)
                
                if valid_count >= TARGET_CANDIDATES:
                    break
        
        # D. Cleanup
        torch.cuda.empty_cache()
        
    pbar.close()
    
    # 3. Save Summary
    if all_metadata:
        df = pd.DataFrame(all_metadata)
        summary_path = os.path.join(SAVE_DIR, "summary_report.csv")
        df.to_csv(summary_path, index=False)
        print("\n==================================================")
        print(f"🎉 DONE! Found {valid_count} candidates.")
        print(f"   High Symmetry Found: {len(df[df['SpaceGroup'] > 1])}")
        print(f"   Report saved to: {summary_path}")
        print("==================================================")
    else:
        print("\n⚠️ Found NO valid candidates. Check training quality.")

if __name__ == "__main__":
    run_discovery()
