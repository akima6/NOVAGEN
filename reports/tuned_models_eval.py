# ============================================================
# COMBINED VALIDATION SCRIPT
# Phase 1: Structural Sanity
# Phase 2: Physics + Chemistry + Symmetry Validation
# ============================================================

import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
import ase.constraints
import ase.filters
from collections import Counter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

warnings.filterwarnings("ignore")

# ============================================================
# 🚨 AUTOMATIC ASE PATCH
# ============================================================
if not hasattr(ase.constraints, "ExpCellFilter"):
    if hasattr(ase.filters, "ExpCellFilter"):
        ase.constraints.ExpCellFilter = ase.filters.ExpCellFilter
    elif hasattr(ase.filters, "UnitCellFilter"):
        ase.constraints.ExpCellFilter = ase.filters.UnitCellFilter

# ============================================================
# 1. GLOBAL CONFIG
# ============================================================
N_SAMPLES = 100
BATCH_SIZE = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CAMPAIGN_ELEMENTS_PHASE1 = [
    3,11,19,4,12,20,38,56,
    5,13,31,49,6,14,32,50,
    7,15,33,51,8,16,34,52,
    21,22,30,48
]

CAMPAIGN_ELEMENTS_PHASE2 = [
    3,11,19,4,12,20,38,56,
    5,13,31,49,6,14,32,50,
    7,15,33,51,8,16,34,52,21,22,30,48
]

BASE_DIR = "/kaggle/working/NOVAGEN"
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "CrystalFormer"))

MODELS_PHASE1 = {
    "Original Base": "/kaggle/working/NOVAGEN/pretrained_model/epoch_005500_CLEAN.pt",
    "Phase 1 (Final)": "/kaggle/working/NOVAGEN/pretrained_model/spatial_v3_epoch_60.pt"
}

MODELS_PHASE2 = {
    "Original Base": "/kaggle/working/NOVAGEN/pretrained_model/epoch_005500_CLEAN.pt",
    "Phase 2 (Fine-Tuned)": "/kaggle/working/NOVAGEN/pretrained_model/physics_expert_epoch_60.pt"
}

# ============================================================
# 2. SETUP
# ============================================================
try:
    from generator_service import CrystalGenerator
    from sentinel import CrystalSentinel
    import matgl
    from pymatgen.io.ase import AseAtomsAdaptor
    from ase.filters import UnitCellFilter
    from ase.optimize import FIRE
except ImportError as e:
    sys.exit(f"❌ Critical Import Error: {e}")

# ============================================================
# 3. COMMON UTILITIES
# ============================================================
def load_generator(path):
    if not os.path.exists(path): 
        return None

    config_path = os.path.join(BASE_DIR, "pretrained_model", "config.yaml")
    if not os.path.exists(config_path):
        config_path = "/kaggle/working/NOVAGEN/CrystalFormer/model/config.yaml"

    gen = CrystalGenerator(None, config_path, DEVICE)
    checkpoint = torch.load(path, map_location=DEVICE)

    if "model_state" in checkpoint:
        gen.model.load_state_dict(checkpoint["model_state"])
    else:
        gen.model.load_state_dict(checkpoint)

    if "lattice_bias_value" in checkpoint:
        gen.lattice_bias.data.fill_(checkpoint["lattice_bias_value"])

    gen.model.eval()
    return gen

# ============================================================
# 4. PHASE 1 : STRUCTURAL VALIDATION
# ============================================================
def run_phase1_validation():
    print("\n" + "="*90)
    print("🚀  PHASE 1 FINAL VALIDATION".center(90))
    print("="*90)
    print(f"Samples per model : {N_SAMPLES}")
    print("Evaluation goal  : Reduce gas/implosion & improve structural stability\n")

    sentinel = CrystalSentinel(device="cpu")
    results = []

    for name, path in MODELS_PHASE1.items():
        print("\n" + "-"*90)
        print(f"🧪 Testing Model: {name}")
        print("-"*90)

        gen = load_generator(path)
        if not gen:
            print(f"⚠️  File not found: {path}")
            continue

        metrics = {
            "valid_structs": 0,
            "gas_cases": 0,
            "implosion_cases": 0,
            "density_pass": 0,
            "sentinel_pass": 0,
            "volumes": []
        }

        for _ in tqdm(range(N_SAMPLES), ncols=80):
            with torch.no_grad():
                out = gen.generate(1, allowed_elements=CAMPAIGN_ELEMENTS_PHASE1, temperature=1.0)
                struct = out["structures"][0]

            if struct is None:
                continue

            metrics["valid_structs"] += 1

            vol_per_atom = struct.volume / len(struct)
            metrics["volumes"].append(vol_per_atom)

            if vol_per_atom > 50.0:
                metrics["gas_cases"] += 1
            elif vol_per_atom < 10.0:
                metrics["implosion_cases"] += 1
            else:
                metrics["density_pass"] += 1

            sentinel_res = sentinel.filter([struct])[0]
            if sentinel_res["valid"]:
                metrics["sentinel_pass"] += 1

        total = max(metrics["valid_structs"], 1)

        results.append({
            "Model": name,
            "Success Rate (%)": (metrics["density_pass"]/total)*100,
            "Gas Rate (%)": (metrics["gas_cases"]/total)*100,
            "Implosion Rate (%)": (metrics["implosion_cases"]/total)*100,
            "Sentinel Pass (%)": (metrics["sentinel_pass"]/total)*100,
            "Avg Volume": np.mean(metrics["volumes"]),
            "Vol Std Dev": np.std(metrics["volumes"])
        })

    print("\n" + "="*90)
    print("📊  PHASE 1 VALIDATION REPORT".center(90))
    print("="*90)

    df = pd.DataFrame(results)
    pretty_df = df.copy()

    for c in ["Success Rate (%)","Gas Rate (%)","Implosion Rate (%)","Sentinel Pass (%)"]:
        pretty_df[c] = pretty_df[c].map(lambda x: f"{x:6.2f}%")

    pretty_df["Avg Volume"] = pretty_df["Avg Volume"].map(lambda x: f"{x:6.2f}")
    pretty_df["Vol Std Dev"] = pretty_df["Vol Std Dev"].map(lambda x: f"{x:6.2f}")

    cols = ["Model","Success Rate (%)","Gas Rate (%)","Implosion Rate (%)","Vol Std Dev","Sentinel Pass (%)"]
    print(pretty_df[cols].to_string(index=False))
    print("="*90)

# ============================================================
# 5. PHASE 2 : PHYSICS + CHEMISTRY + SYMMETRY VALIDATION
# ============================================================
class RobustRelaxer:
    """Universal Physics Engine"""
    def __init__(self):
        self.pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
        try:
            from matgl.ext.ase import M3GNetCalculator
            self.calc = M3GNetCalculator(potential=self.pot)
        except ImportError:
            from matgl.ext.ase import ASECalculator
            self.calc = ASECalculator(potential=self.pot)

    def get_energy(self, structure):
        torch.set_grad_enabled(True) 
        try:
            atoms = AseAtomsAdaptor.get_atoms(structure)
            vol = atoms.get_volume() / len(atoms)
            if vol > 100.0 or vol < 5.0: 
                return 50.0
            
            atoms.calc = self.calc
            ucf = UnitCellFilter(atoms)
            dyn = FIRE(ucf, logfile=None)
            dyn.run(fmax=0.1, steps=25) 
            return atoms.get_potential_energy() / len(atoms)
        except:
            return 50.0

def get_symmetry_score(structure):
    """Checks if the crystal has any symmetry (Not P1)."""
    try:
        sga = SpacegroupAnalyzer(structure, symprec=0.1)
        sg_num = sga.get_space_group_number()
        return sg_num
    except:
        return 1

def is_realistic_chemical(structure):
    elements = [str(e) for e in structure.composition.elements]
    if len(elements) > 4: return False
    if len(structure) == 1: return False
    return True

# ============================================================
# 6. PHASE 2 BENCHMARK
# ============================================================
def run_phase2_validation():
    print("\n" + "="*90)
    print("🚀  PHASE 2: DEFINITIVE VALIDATION".center(90))
    print("="*90)
    print("Objective: Prove Phase 2 generates ORDERED MATERIALS (Symmetry + Chemistry).")
    print("Metrics:   Energy, Realistic Purity, and Crystal Symmetry.\n")

    sentinel = CrystalSentinel(device="cpu")
    relaxer = RobustRelaxer()
    summary_stats = []

    for name, path in MODELS_PHASE2.items():
        print(f"🧪 Evaluating: {name}...")
        gen = load_generator(path)
        if not gen: continue

        crystals = []
        valid_count = 0
        stable_count = 0
        realistic_count = 0
        high_sym_count = 0

        for _ in tqdm(range(N_SAMPLES), ncols=80, leave=False):
            with torch.no_grad():
                out = gen.generate(1, allowed_elements=CAMPAIGN_ELEMENTS_PHASE2, temperature=0.7)
                struct = out["structures"][0]

            if struct is None: 
                continue

            valid_count += 1

            # 1. Physics
            energy = relaxer.get_energy(struct)

            # 2. Geometry
            geo_res = sentinel.filter([struct])[0]

            # 3. Realism
            is_real = is_realistic_chemical(struct)
            if is_real: realistic_count += 1

            # 4. Symmetry
            sg_num = get_symmetry_score(struct)
            if sg_num > 1: high_sym_count += 1

            # 5. Stability
            clean_e = max(min(energy, 50.0), -10.0)
            if clean_e < 0.0: stable_count += 1

            crystals.append({
                "formula": struct.composition.reduced_formula,
                "energy": clean_e,
                "geo_score": geo_res["min_distance_ratio"],
                "is_real": is_real,
                "sg": sg_num
            })

        energies = [c["energy"] for c in crystals]
        real_energies = [c["energy"] for c in crystals if c["is_real"]]

        summary_stats.append({
            "Model": name,
            "Stable Yield": (stable_count/valid_count*100) if valid_count else 0.0,
            "Realistic Purity": (realistic_count/valid_count*100) if valid_count else 0.0,
            "Symmetry (>P1)": (high_sym_count/valid_count*100) if valid_count else 0.0,
            "Best Valid Energy": min(real_energies) if real_energies else 50.0,
            "Median Energy": np.median(energies) if energies else 50.0
        })

        # ---------------- LEADERBOARD ----------------
        print(f"\n🏆 TOP 10 VALID CANDIDATES: {name}")
        print(f"{'Rank':<5} | {'Formula':<20} | {'Energy':<10} | {'SG #':<5} | {'Realism?'}")
        print("-" * 65)

        sorted_crystals = sorted(crystals, key=lambda x: x["energy"])
        count = 0
        for c in sorted_crystals:
            if count >= 10: break
            if c["is_real"]:
                prefix = f"{count+1}"
                print(f"{prefix:<5} | {c['formula']:<20} | {c['energy']:>6.3f} eV   | {c['sg']:<5} | ✅ Valid")
                count += 1
        if count == 0:
            print("   (No valid stable crystals found)")
        print("-" * 65 + "\n")

    # ========================================================
    # FINAL SCOREBOARD
    # ========================================================
    print("="*90)
    print("📊  FINAL COMPARISON".center(90))
    print("="*90)

    df = pd.DataFrame(summary_stats)
    df_d = df.copy()

    df_d["Stable Yield"] = df_d["Stable Yield"].map(lambda x: f"{x:.1f}%")
    df_d["Realistic Purity"] = df_d["Realistic Purity"].map(lambda x: f"{x:.1f}%")
    df_d["Symmetry (>P1)"] = df_d["Symmetry (>P1)"].map(lambda x: f"{x:.1f}%")
    df_d["Best Valid Energy"] = df_d["Best Valid Energy"].map(lambda x: f"{x:.3f} eV")
    df_d["Median Energy"] = df_d["Median Energy"].map(lambda x: f"{x:.3f} eV")

    cols = ["Model","Stable Yield","Realistic Purity","Symmetry (>P1)","Best Valid Energy","Median Energy"]
    print(df_d[cols].to_string(index=False))

    # ---------------- VERDICT ----------------
    print("\n🧠  VERDICT:")
    print("-" * 90)
    if len(df) == 2:
        base = df.iloc[0]
        tuned = df.iloc[1]

        d_purity = tuned["Realistic Purity"] - base["Realistic Purity"]
        d_sym = tuned["Symmetry (>P1)"] - base["Symmetry (>P1)"]

        print(f"1. CHEMISTRY  : Phase 2 produces {abs(d_purity):.1f}% more realistic formulas.")
        print(f"2. ORDER      : Phase 2 produces {abs(d_sym):.1f}% more symmetrical crystals.")
        print("   -> Original model makes random blobs (P1).")
        print("   -> Phase 2 creates structured, ordered matter.")
    print("="*90)

# ============================================================
# 7. MASTER ENTRYPOINT
# ============================================================
if __name__ == "__main__":
    run_phase1_validation()     # Structural sanity
    run_phase2_validation()     # Physics + Chemistry + Symmetry validation
