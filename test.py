# test.py (Fixed Imports)

import sys
import os
import csv
import torch
import numpy as np
import random
from pymatgen.core import Lattice, Structure

# --- PATH SETUP ---
# Adjust this if your CrystalFormer folder is named differently
sys.path.append(os.path.join(os.path.dirname(__file__), "CrystalFormer"))

# --- IMPORTS ---
from crystalformer.extension.transformer import CrystalFormerTransformer
from crystalformer.extension.generator import CrystalFormerGenerator

# Import local modules (Must be in same folder as test.py)
from relaxer import Relaxer
from oracle import Oracle 

# --- CONFIGURATION ---
NUM_SAMPLES = 10         
TEMPERATURE = 1.2        
DEVICE = "cpu"

INDEX_TO_ATOMIC_NUMBER = {
    0: 13, # Al
    1: 14, # Si
    2: 6,  # C
    3: 8,  # O
    4: 32  # Ge
}

def sample_atom_type(logits, temperature=1.0):
    logits = torch.tensor(logits) / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()

def build_structure(L, A, X):
    species = []
    coords = []
    for atomic_num, coord in zip(A, X):
        if atomic_num != 0:
            species.append(atomic_num)
            coords.append(coord)
    
    if not species: return None
    
    lattice = Lattice.from_parameters(
        a=L[0], b=L[1], c=L[2],
        alpha=L[3], beta=L[4], gamma=L[5]
    )
    return Structure(lattice, species, coords)

def main():
    print("--- 1. Initializing Models ---")
    
    # 1. Generator
    model = CrystalFormerTransformer(
        Nf=2, Kx=2, Kl=1, n_max=21, h0_size=0, num_layers=1, num_heads=1,
        key_size=16, model_size=32, embed_size=16, atom_types=5, wyck_types=5,
        dropout_rate=0.0,
    )
    generator = CrystalFormerGenerator(model, device=DEVICE)
    print("   -> Generator Ready")

    # 2. Relaxer
    relaxer = Relaxer()
    print("   -> Relaxer Ready")

    # 3. Oracle
    oracle = Oracle()
    print("   -> Oracle Ready")

    print(f"\n--- 2. Generating & Relaxing {NUM_SAMPLES} Candidates ---")
    stable_candidates = []

    for i in range(NUM_SAMPLES):
        print(f"🔹 Candidate {i+1}/{NUM_SAMPLES}...", end="\r")
        
        # Random inputs
        random_G = random.randint(1, 230)
        
        logits = generator.generate_logits(
            G=random_G, 
            XYZ=[[random.random() for _ in range(3)] for _ in range(2)],
            A=[0, 0], W=[0, 0], M=[1, 1]
        )

        A_list = []
        for j in range(2):
            atom_logits = logits[1 + j][:len(INDEX_TO_ATOMIC_NUMBER)]
            idx = sample_atom_type(atom_logits, TEMPERATURE)
            A_list.append(INDEX_TO_ATOMIC_NUMBER.get(idx, 6))

        X_list = []
        coord_start = 1 + len(A_list)
        for j in range(len(A_list)):
            x = torch.sigmoid(logits[coord_start + 3*j + 0][0]).item()
            y = torch.sigmoid(logits[coord_start + 3*j + 1][0]).item()
            z = torch.sigmoid(logits[coord_start + 3*j + 2][0]).item()
            X_list.append([x, y, z])

        # Build & Relax
        unrelaxed_struct = build_structure([5.0]*3 + [90.0]*3, A_list, X_list)
        if not unrelaxed_struct: continue

        result = relaxer.relax(unrelaxed_struct)

        if result["is_converged"]:
            stable_candidates.append({
                "id": i,
                "structure": result["final_structure"],
                "formula": result["final_structure"].composition.reduced_formula,
                "relax_energy": result["final_energy_per_atom"]
            })
            print(f"   ✅ Stable: {result['final_structure'].composition.reduced_formula}               ")
        else:
            print(f"   ❌ Unstable               ")

    num_stable = len(stable_candidates)
    print(f"\n--- 3. Running Oracle on {num_stable} Stable Crystals ---")
    
    if num_stable > 0:
        structures_to_predict = [cand["structure"] for cand in stable_candidates]
        properties = oracle.predict_properties(structures_to_predict)
        
        final_results = []
        for cand, props in zip(stable_candidates, properties):
            entry = {
                "ID": cand["id"],
                "Formula": cand["formula"],
                "Relaxed_Energy": cand["relax_energy"],
                "Formation_Energy": props.get("formation_energy"),
                "Band_Gap": props.get("band_gap"),
            }
            final_results.append(entry)

        print("\n" + "="*80)
        print(f"{'Formula':<12} | {'Relax E':<10} | {'Form E':<10} | {'Band Gap':<10}")
        print("-" * 80)
        for res in final_results:
            re = f"{res['Relaxed_Energy']:.3f}" if res['Relaxed_Energy'] else "N/A"
            fe = f"{res['Formation_Energy']:.3f}" if res['Formation_Energy'] else "N/A"
            bg = f"{res['Band_Gap']:.3f}" if res['Band_Gap'] else "N/A"
            print(f"{res['Formula']:<12} | {re:<10} | {fe:<10} | {bg:<10}")
        print("="*80)

if __name__ == "__main__":
    main()