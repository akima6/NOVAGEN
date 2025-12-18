import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher

# --- CONFIGURATION ---
ROOT = Path(__file__).parent
DISC_DIR = ROOT / "rl_discoveries"
LOG_FILE = ROOT / "training_log.csv"

def analyze_project():
    print("==============================================================")
    print("   NOVAGEN: FINAL PROJECT ANALYSIS                            ")
    print("==============================================================")

    # 1. VISUALIZE LEARNING (The "Proof" it worked)
    if LOG_FILE.exists():
        try:
            df = pd.read_csv(LOG_FILE)
            plt.figure(figsize=(10, 6))
            
            # Plot Raw Reward
            plt.plot(df['Epoch'], df['Reward'], alpha=0.3, color='gray', label='Batch Reward')
            
            # Plot Smoothed Trend (Rolling Average)
            df['Rolling_Reward'] = df['Reward'].rolling(window=20).mean()
            plt.plot(df['Epoch'], df['Rolling_Reward'], color='blue', linewidth=2, label='Trend (Avg)')
            
            plt.xlabel('Training Epochs')
            plt.ylabel('Crystal Quality (Reward)')
            plt.title('AI Learning Curve: From Random Guessing to Discovery')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plot_file = ROOT / "learning_curve.png"
            plt.savefig(plot_file)
            print(f"📈 Learning Curve saved to '{plot_file.name}'")
            print(f"   (Use this graph in your final report/presentation)")
        except Exception as e:
            print(f"⚠️ Could not plot data: {e}")

    # 2. FILTER & VALIDATE CRYSTALS
    if not DISC_DIR.exists():
        print("❌ No discoveries folder found.")
        return

    cif_files = list(DISC_DIR.glob("*.cif"))
    print(f"\n🔍 Analyzing {len(cif_files)} candidate crystals...")
    
    # We use StructureMatcher to see if crystals are physically identical
    # (even if they look different to the eye, the math might be the same)
    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
    
    unique_crystals = []
    seen_structures = []

    for cif_path in cif_files:
        try:
            struct = Structure.from_file(cif_path)
            formula = struct.composition.reduced_formula
            
            # Parse filename for score (e.g. "ZnS_Ep5_R12.50.cif")
            # This depends on how you named them. If generic, we skip score parsing.
            
            is_duplicate = False
            for seen_s in seen_structures:
                if matcher.fit(struct, seen_s):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_crystals.append((formula, cif_path))
                seen_structures.append(struct)
        except:
            continue

    print(f"💎 UNIQUE DISCOVERIES: {len(unique_crystals)} (after removing duplicates)")
    print("-" * 50)
    print(f"{'Formula':<15} | {'Filename':<30}")
    print("-" * 50)
    
    for formula, path in unique_crystals[:10]: # List top 10
        print(f"{formula:<15} | {path.name:<30}")

    print("-" * 50)
    print("✅ PROJECT COMPLETE.")
    print("   Next Step: Upload these Top 10 .cif files to a viewer (VESTA) to take screenshots.")

if __name__ == "__main__":
    analyze_project()