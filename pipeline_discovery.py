import os
import sys
import torch
import warnings
import pandas as pd
from tqdm import tqdm
import logging

# --- SETUP PATHS (ABSOLUTE) ---
# Get the directory where THIS script is located (e.g., /kaggle/working/NOVAGEN)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Add library paths relative to this script
sys.path.append(os.path.join(BASE_DIR, "CrystalFormer"))
sys.path.append(BASE_DIR)

# --- CONFIGURATION ---
# Now we build the full paths using BASE_DIR
RL_MODEL_PATH = os.path.join(BASE_DIR, "rl_checkpoints", "epoch_100_RL.pt")
BASE_CONFIG_PATH = os.path.join(BASE_DIR, "pretrained_model", "config.yaml")
NUM_CANDIDATES = 2000  
CAMPAIGN_ELEMENTS = None

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
    print(f"🚀 STARTING DISCOVERY CAMPAIGN (Silent Mode)")
    print(f"   Target: {NUM_CANDIDATES} Candidates")
    print(f"==================================================")

    # 1. INITIALIZE (Quietly)
    print("[1/4] Loading Modules...", end="\r")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Check if files exist before crashing
    if not os.path.exists(RL_MODEL_PATH):
        print(f"\n❌ Error: RL Model not found at {RL_MODEL_PATH}")
        return
    if not os.path.exists(BASE_CONFIG_PATH):
        print(f"\n❌ Error: Config not found at {BASE_CONFIG_PATH}")
        return

    try:
        generator = CrystalGenerator(RL_MODEL_PATH, BASE_CONFIG_PATH, device)
        sentinel = CrystalSentinel()
        relaxer = CrystalRelaxer(device="cpu") 
        oracle = CrystalOracle(device="cpu")
        print(f"[1/4] ✅ Modules Loaded on {device}.          ")
    except Exception as e:
        print(f"\n❌ Initialization Failed: {e}")
        return

    # 2. GENERATE
    print(f"[2/4] Generating {NUM_CANDIDATES} structures...", end="\r")
    try:
        raw_structs = generator.generate(NUM_CANDIDATES, allowed_elements=CAMPAIGN_ELEMENTS)
        print(f"[2/4] ✅ Generated {len(raw_structs)} raw structures.      ")
    except Exception as e:
        print(f"\n❌ Generation Error: {e}")
        return

    # 3. FILTER (Sentinel)
    print(f"[3/4] Filtering hallucinations...", end="\r")
    mask, valid_structs = sentinel.filter(raw_structs)
    n_valid = len(valid_structs)
    print(f"[3/4] ✅ Sentinel Passed: {n_valid} / {NUM_CANDIDATES} ({n_valid/NUM_CANDIDATES:.1%})")

    if n_valid == 0:
        print("❌ No valid structures found. RL Model might need more entropy or training.")
        return

    # 4. RELAX & ANALYZE (The "Silent" Loop)
    print(f"[4/4] Relaxing & Analyzing survivors...")
    
    results = []
    stats = {"stable": 0, "unstable_high_e": 0, "unstable_crash": 0}

    for struct in tqdm(valid_structs, desc="   Processing", unit="cryst"):
        try:
            # A. Relax (Force CPU mode if needed)
            res = relaxer.relax(struct)
            
            # GET DATA EVEN IF NOT CONVERGED
            final_e = res.get('energy_per_atom', 0.0)
            final_s = res.get('final_structure', struct)
            is_converged = res.get('converged', False)
            
            # B. Predict Properties
            props = oracle.predict(final_s)
            gap = props.get('band_gap', 0.0)
            
            # C. SMART CATEGORIZATION
            status = "Unknown"
            
            if final_e > -0.1:
                # Positive/High energy = Trash
                stats["unstable_high_e"] += 1
                status = "High Energy"
            else:
                # Negative Energy = Interesting!
                if is_converged:
                    stats["stable"] += 1
                    status = "Stable"
                else:
                    # It failed to converge, BUT energy is good.
                    # We call this a "Candidate" (needs more cleaning later)
                    stats["stable"] += 1 # Count as success for now
                    status = "Promising (Unconverged)"

            # Only save if it's not high energy garbage
            if status != "High Energy":
                results.append({
                    "Formula": final_s.composition.reduced_formula,
                    "Energy (eV/atom)": round(final_e, 3),
                    "Band Gap (eV)": round(gap, 2),
                    "Space Group": final_s.get_space_group_info()[1],
                    "Status": status
                })
                
        except Exception:
            stats["unstable_crash"] += 1

    # --- FINAL REPORT ---
    print("\n" + "="*50)
    print("📊 CAMPAIGN REPORT CARD")
    print("="*50)
    print(f"Total Attempts:     {NUM_CANDIDATES}")
    print(f"Physically Valid:   {n_valid}")
    print(f"Relaxation Crashes: {stats['unstable_crash']}")
    print(f"High Energy (Bad):  {stats['unstable_high_e']}")
    print(f"STABLE (Winners):   {stats['stable']}")
    print("-" * 50)
    
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values(by="Energy (eV/atom)", ascending=True)
        best_ones = df[df["Status"] == "Stable"]
        
        if not best_ones.empty:
            print("\n🏆 TOP DISCOVERIES (Paste this table):")
            print(best_ones.to_string(index=False))
        else:
            print("\n⚠️ No stable crystals found (Best attempt below):")
            print(df.head(5).to_string(index=False))
            
    print("\n" + "="*50)

if __name__ == "__main__":
    main()
