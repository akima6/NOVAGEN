import sys
import torch
import yaml
import random
from pathlib import Path
from pymatgen.io.cif import CifWriter

ROOT = Path(__file__).parent
CRYSTALFORMER_DIR = ROOT / "CrystalFormer"
PRETRAINED_DIR = ROOT / "pretrained_model"
sys.path.append(str(CRYSTALFORMER_DIR))

from crystalformer.src.transformer import make_transformer
try:
    from crystalformer.extension.generator import CrystalFormerGenerator
except ImportError:
    from crystalformer.extension.generator import CrystalGenerator as CrystalFormerGenerator

from relaxer import Relaxer
from oracle import Oracle

# ---------------- CONFIG ----------------
NUM_ATTEMPTS = 50
SMART_PAIRS = [([30], [16]), ([30], [8]), ([31], [33]), ([48], [34])]
# ----------------------------------------

def initialize_novagen():
    print("==============================================================")
    print("   NOVAGEN: PHYSICALLY INFORMED DISCOVERY (RL-SAFE MODE)      ")
    print("==============================================================")

    with open(PRETRAINED_DIR / "config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    model = make_transformer(
    key=None,
    Nf=cfg.get("Nf", 5),
    Kx=cfg.get("Kx", 16),
    Kl=cfg.get("Kl", 4),
    n_max=cfg.get("n_max", 21),
    h0_size=cfg.get("h0_size", 256),
    num_layers=cfg.get("transformer_layers", 16),
    num_heads=cfg.get("num_heads", 8),
    key_size=cfg.get("key_size", 64),
    model_size=cfg.get("model_size", 512),
    embed_size=cfg.get("embed_size", 32),
    atom_types=cfg.get("atom_types", 119),
    wyck_types=cfg.get("wyck_types", 28),
    dropout_rate=0.0,
    widening_factor=cfg.get("widening_factor", 4)
    )


    state_dict = torch.load(
        PRETRAINED_DIR / "epoch_005500_CLEAN.pt",
        map_location="cpu"
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return (
        CrystalFormerGenerator(model=model, device="cpu"),
        Relaxer(),
        Oracle()
    )

def decode_smart_crystal(logits, num_atoms, cation, anion):
    from pymatgen.core import Lattice, Structure

    species = [cation[0]] * (num_atoms // 2) + [anion[0]] * (num_atoms // 2)
    coords = []

    coord_start = 1 + num_atoms
    for j in range(num_atoms):
        x = torch.sigmoid(logits[coord_start + 3*j + 0][0]).item()
        y = torch.sigmoid(logits[coord_start + 3*j + 1][0]).item()
        z = torch.sigmoid(logits[coord_start + 3*j + 2][0]).item()
        coords.append([x, y, z])

    lattice = Lattice.cubic(5.0)
    return Structure(lattice, species, coords)

def run_novagen():
    agent, relaxer, oracle = initialize_novagen()
    cif_dir = ROOT / "generated_cifs"
    cif_dir.mkdir(exist_ok=True)

    print(f"\n🚀 Launching {NUM_ATTEMPTS} Targeted Attempts...\n")
    print(f"{'Formula':<10} | {'Band Gap':<22} | {'Eform':<8}")
    print("-" * 50)

    for i in range(NUM_ATTEMPTS):
        try:
            cation, anion = random.choice(SMART_PAIRS)
            num_atoms = 4

            logits = agent.generate_logits(
                G=random.randint(1, 230),
                XYZ=[[0.5, 0.5, 0.5]] * num_atoms,
                A=[0] * num_atoms,
                W=[0] * num_atoms,
                M=[1] * num_atoms
            )

            struct = decode_smart_crystal(logits, num_atoms, cation, anion)
            relax = relaxer.relax(struct)
            if not relax["is_converged"]:
                continue

            relaxed = relax["final_structure"]
            formula = relaxed.composition.reduced_formula

            props = oracle.predict_properties([relaxed])[0]

            eform = props["formation_energy"]
            gap_raw = props["band_gap_raw"]
            gap_disp = props["band_gap_display"]

            print(f"{formula:<10} | {gap_disp:<22} | {eform:>6.3f}")

            # -------- RL-SAFE DISCOVERY CRITERIA --------
            if gap_raw > 0.1 and eform < -0.1:
                CifWriter(relaxed).write_file(
                    cif_dir / f"{formula}_{i}.cif"
                )

        except Exception:
            continue

    print("\n🏁 NOVAGEN run complete.")

if __name__ == "__main__":
    run_novagen()
