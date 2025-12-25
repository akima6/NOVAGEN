import sys
import os
import torch
from tqdm import tqdm

# 1. SETUP PATHS
ROOT_DIR = "/kaggle/working/NOVAGEN"
sys.path.append(os.path.abspath(f"{ROOT_DIR}/CrystalFormer"))

# 2. IMPORT OUR MODULES
from generator_service import CrystalGenerator
from sentinel import CrystalSentinel
from product_relaxer import CrystalRelaxer
from product_oracle import CrystalOracle

def run_discovery_pipeline():
    print("==================================================")
    print("🚀 STARTING INORGANIC DISCOVERY PIPELINE V1.0")
    print("==================================================")

    # --- A. INITIALIZATION ---
    print("\n[1/5] 🔌 Initializing Modules...")
    
    # AUTO-DETECT DEVICE (The Fix)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   -> System Hardware Detected: {DEVICE.upper()}")
    
    # 1. Generator
    ckpt_path = f"{ROOT_DIR}/pretrained_model/epoch_005500_CLEAN.pt"
    cfg_path = f"{ROOT_DIR}/CrystalFormer/model/config.yaml"
    
    if not os.path.exists(ckpt_path):
        print(f"❌ Missing Model Weights: {ckpt_path}"); return

    # Pass the detected device
    generator = CrystalGenerator(ckpt_path, cfg_path, device=DEVICE)
    
    # 2. Sentinel (Always CPU)
    sentinel = CrystalSentinel(min_distance=0.6, min_density=0.5)
    
    # 3. Relaxer (Pass detected device)
    relaxer = CrystalRelaxer(device=DEVICE)
    
    # 4. Oracle (Always CPU for safety)
    oracle = CrystalOracle(device="cpu")

    # --- B. CAMPAIGN SETUP ---
    # Hunting for Iron-based semiconductors (Fe-O-S)
    campaign_elements = [26, 8, 16] # Fe, O, S
    num_candidates = 50 # Reduced to 5 for a quick CPU test if needed
    
    print(f"\n[2/5] 🧪 Starting Campaign: Fe-O-S Search ({num_candidates} candidates)")

    # --- C. GENERATION ---
    print("\n[3/5] ⚡ Generating Candidates...")
    raw_structs = generator.generate(
        num_samples=num_candidates, 
        allowed_elements=campaign_elements,
        temperature=1.0
    )
    print(f"   -> Generated {len(raw_structs)} raw structures.")

    # --- D. VALIDATION ---
    print("\n[4/5] 🛡️ Filtering Hallucinations...")
    valid_structs, stats = sentinel.batch_filter(raw_structs)
    
    print(f"   -> {len(valid_structs)} survived. (Rejected: {stats['overlapping']} overlap, {stats['density']} density)")
    
    if len(valid_structs) == 0:
        print("❌ All candidates failed validation. Try increasing num_candidates.")
        return

    # --- E. PHYSICS & SCORING ---
    print("\n[5/5] 🔬 Relaxing & Analyzing...")
    
    results_table = []
    
    # Iterate with progress bar
    for i, struct in enumerate(tqdm(valid_structs, desc="   Processing")):
        # 1. Relax
        relax_res = relaxer.relax(struct)
        
        if not relax_res['converged']:
            continue 
            
        final_s = relax_res['final_structure']
        
        # 2. Oracle
        props = oracle.predict(final_s)
        
        # 3. Log
        results_table.append({
            "formula": final_s.composition.reduced_formula,
            "volume": final_s.volume,
            "E_form": props['formation_energy'],
            "BandGap": props['band_gap']
        })

    # --- F. FINAL REPORT ---
    print("\n==================================================")
    print("🏆 CAMPAIGN RESULTS (Top Candidates)")
    print("==================================================")
    print(f"{'FORMULA':<12} | {'E_FORM':<12} | {'GAP (eV)':<10} | {'STATUS'}")
    print("-" * 50)
    
    # Sort by Stability
    results_table.sort(key=lambda x: x['E_form'])
    
    for r in results_table:
        status = "✅ Stable" if r['E_form'] < -0.1 else "⚠️ Unstable"
        if r['BandGap'] > 0.1: status += " + ⚡ Semi"
        
        print(f"{r['formula']:<12} | {r['E_form']:<12.3f} | {r['BandGap']:<10.3f} | {status}")

if __name__ == "__main__":
    run_discovery_pipeline()
