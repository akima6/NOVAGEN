import os
import sys
import torch
import warnings
import pandas as pd
from tqdm import tqdm
import logging

# --- CONFIGURATION (PRODUCT RUN) ---
# Get the directory where THIS script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "CrystalFormer"))
sys.path.append(BASE_DIR)

RL_MODEL_PATH = os.path.join(BASE_DIR, "rl_checkpoints", "epoch_100_RL.pt")
BASE_CONFIG_PATH = os.path.join(BASE_DIR, "pretrained_model", "config.yaml")

NUM_CANDIDATES = 2000 
CAMPAIGN_ELEMENTS = None  # Full Periodic Table Search

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
    print(f"🚀 STARTING SEMICONDUCTOR DISCOVERY (Silent Mode)")
    print(f"   Target: {NUM_CANDIDATES} Candidates")
    print(f"   Filter: Stable (< 0 eV) AND Band Gap (> 0.1 eV)")
    print(f"==================================================")

    # 1. INITIALIZE
    print("[1/4] Loading Modules...", end="\r")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not os.path.exists(RL_MODEL_PATH):
        print(f"\n❌ Error: RL Model not found at {RL_MODEL_PATH}")
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
        print("❌ No valid structures found.")
        return

    # 4. RELAX & ANALYZE
    print(f"[4/4] Relaxing & Analyzing survivors...")
    
    results = []
    stats = {"stable_semicon": 0, "stable_metal": 0, "unstable": 0, "crash": 0}

    for struct in tqdm(valid_structs, desc="   Processing", unit="cryst"):
        try:
            # A. Relax
            res = relaxer.relax(struct)
            final_e = res.get('energy_per_atom', 0.0)
            final_s = res.get('final_structure', struct)
            
            # B. Predict Properties
            props = oracle.predict(final_s)
            gap = props.get('band_gap', 0.0)
            
            # C. SMART CATEGORIZATION
            status = "Rejected"
            
            if final_e < 0.0: # Must be stable
                if gap > 0.1:
                    # THE GOLDEN STANDARD: Stable + Semiconductor
                    stats["stable_semicon"] += 1
                    status = "✅ Semiconductor"
                else:
                    # It's stable, but it's a metal.
                    stats["stable_metal"] += 1
                    status = "⚠️ Metal (Ignored)"
            else:
                stats["unstable"] += 1
                status = "❌ Unstable"

            # Only save interesting things (Semiconductors)
            if "Semiconductor" in status:
                results.append({
                    "Formula": final_s.composition.reduced_formula,
                    "Energy": round(final_e, 3),
                    "Band Gap": round(gap, 2),
                    "Space Group": final_s.get_space_group_info()[1],
                    "Status": status
                })
                
        except Exception:
            stats["crash"] += 1

    # --- FINAL REPORT ---
    print("\n" + "="*50)
    print("📊 CAMPAIGN REPORT CARD")
    print("="*50)
    print(f"Total Attempts:     {NUM_CANDIDATES}")
    print(f"Physically Valid:   {n_valid}")
    print(f"Metals (Discarded): {stats['stable_metal']}")
    print(f"SEMICONDUCTORS:     {stats['stable_semicon']}")
    print("-" * 50)
    
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values(by="Band Gap", ascending=False)
        print("\n🏆 TOP SEMICONDUCTORS FOUND:")
        print(df.to_string(index=False))
    else:
        print("\n⚠️ No stable semiconductors found in this batch.")
            
    print("\n" + "="*50)

if __name__ == "__main__":
    main()
