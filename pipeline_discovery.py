import os
import sys
import torch
import warnings
import pandas as pd
from tqdm import tqdm
import logging
import time

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "CrystalFormer"))
sys.path.append(BASE_DIR)

# Auto-detect the checkpoint (Look for epoch_100, fallback to clean)
CHECKPOINT_PATH = os.path.join(BASE_DIR, "rl_checkpoints", "epoch_100_RL.pt")
if not os.path.exists(CHECKPOINT_PATH):
    print("⚠️ RL Checkpoint not found. Falling back to Pretrained Model.")
    CHECKPOINT_PATH = os.path.join(BASE_DIR, "pretrained_model", "epoch_005500_CLEAN.pt")

BASE_CONFIG_PATH = os.path.join(BASE_DIR, "pretrained_model", "config.yaml")
RESULTS_DIR = os.path.join(BASE_DIR, "rl_discoveries") 

NUM_CANDIDATES = 2000 
CAMPAIGN_ELEMENTS = None  # Full Periodic Table

# --- SILENCE LOGS ---
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")
logging.getLogger("pymatgen").setLevel(logging.CRITICAL)

try:
    from generator_service import CrystalGenerator
    from sentinel import CrystalSentinel
    from product_relaxer import CrystalRelaxer
    from product_oracle import CrystalOracle
except ImportError as e:
    print(f"❌ Setup Error: {e}")
    sys.exit(1)

def main():
    print(f"==================================================")
    print(f"🚀 STARTING DISCOVERY CAMPAIGN (With Auto-Save)")
    print(f"   Target: {NUM_CANDIDATES} Candidates")
    print(f"   Saving to: {RESULTS_DIR}")
    print(f"==================================================")

    # 1. SETUP
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 2. LOAD MODULES
    print("[1/4] Loading Modules...", end="\r")
    try:
        generator = CrystalGenerator(CHECKPOINT_PATH, BASE_CONFIG_PATH, device)
        sentinel = CrystalSentinel()
        relaxer = CrystalRelaxer(device="cpu") 
        oracle = CrystalOracle(device="cpu")
        print(f"[1/4] ✅ Modules Loaded on {device}.          ")
    except Exception as e:
        print(f"\n❌ Initialization Failed: {e}")
        return

    # 3. GENERATE
    print(f"[2/4] Generating {NUM_CANDIDATES} structures...", end="\r")
    try:
        raw_structs = generator.generate(NUM_CANDIDATES, allowed_elements=CAMPAIGN_ELEMENTS)
        print(f"[2/4] ✅ Generated {len(raw_structs)} raw structures.      ")
    except Exception as e:
        print(f"\n❌ Generation Error: {e}")
        return

    # 4. FILTER
    print(f"[3/4] Filtering hallucinations...", end="\r")
    mask, valid_structs = sentinel.filter(raw_structs)
    print(f"[3/4] ✅ Sentinel Passed: {len(valid_structs)} / {NUM_CANDIDATES}")

    if not valid_structs:
        print("❌ No valid structures found.")
        return

    # 5. RELAX, ANALYZE & SAVE
    print(f"[4/4] Processing & Saving Winners...")
    
    results = []
    stats = {"semicon": 0, "metal": 0, "unstable": 0}

    for i, struct in enumerate(tqdm(valid_structs, desc="   Analyzing", unit="cryst")):
        try:
            # A. Relax
            # NOTE: struct is the RAW guess. final_s is the RELAXED crystal.
            res = relaxer.relax(struct)
            final_e = res.get('energy_per_atom', 0.0)
            final_s = res.get('final_structure', struct)
            
            # B. Predict
            props = oracle.predict(final_s)
            gap = props.get('band_gap', 0.0)
            
            # C. Classify
            is_winner = False
            
            if final_e < 0.0:
                if gap > 0.1:
                    stats["semicon"] += 1
                    is_winner = True
                else:
                    stats["metal"] += 1
            else:
                stats["unstable"] += 1

            # D. SAVE IF WINNER
            if is_winner:
                formula = final_s.composition.reduced_formula
                
                # --- UPGRADE 5: SYMMETRY REFINEMENT ---
                # CRITICAL FIX: We must analyze final_s (Relaxed), not struct (Raw)
                final_struct_to_save = final_s 
                sg_number = 1
                
                try:
                    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
                    
                    # 1. Analyze the RELAXED crystal
                    sga = SpacegroupAnalyzer(final_s, symprec=0.1)
                    
                    # 2. Refine it
                    refined_struct = sga.get_refined_structure()
                    
                    # 3. Use refined structure for saving
                    final_struct_to_save = refined_struct
                    sg_number = sga.get_space_group_number()
                    
                except Exception:
                    # Fallback to the relaxed structure if refinement fails
                    final_struct_to_save = final_s
                    sg_number = final_s.get_space_group_info()[1]
                # --------------------------------------

                # Save the POLISHED crystal
                filename = f"{formula}_{i:04d}.cif"
                # CRITICAL FIX: Use RESULTS_DIR, not SAVE_DIR
                save_path = os.path.join(RESULTS_DIR, filename)
                final_struct_to_save.to(filename=save_path)
                
                results.append({
                    "Formula": formula,
                    "Energy": round(final_e, 3),
                    "Band Gap": round(gap, 2),
                    "Space Group": sg_number,
                    "File": filename
                })
                
        except Exception:
            pass

    # --- REPORT ---
    print("\n" + "="*50)
    print("📊 CAMPAIGN REPORT CARD")
    print("="*50)
    print(f"Total Scanned:      {NUM_CANDIDATES}")
    print(f"Valid Candidates:   {len(valid_structs)}")
    print(f"Semiconductors:     {stats['semicon']} (SAVED)")
    print(f"Stable Metals:      {stats['metal']} (Ignored)")
    print("-" * 50)
    
    if results:
        # Save CSV summary
        df = pd.DataFrame(results)
        df = df.sort_values(by="Band Gap", ascending=False)
        csv_path = os.path.join(RESULTS_DIR, "summary_report.csv")
        df.to_csv(csv_path, index=False)
        
        print("\n🏆 TOP DISCOVERIES (Saved to rl_discoveries/):")
        print(df.head(10).to_string(index=False))
        print(f"\n💾 Summary Report: {csv_path}")
        print(f"💾 CIF Files:      {RESULTS_DIR}/*.cif")
    else:
        print("\n⚠️ No new semiconductors found.")
            
    print("\n" + "="*50)

if __name__ == "__main__":
    main()
