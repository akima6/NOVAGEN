import os
import sys
import pandas as pd
import warnings
from tqdm import tqdm
import tempfile
import contextlib
import io
import time

# Pymatgen & MP-API Imports
from pymatgen.core import Structure
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.io.vasp.sets import MPRelaxSet
from pymatgen.io.vasp.inputs import Incar, Poscar

try:
    from mp_api.client import MPRester
except ImportError:
    sys.exit("❌ Missing mp-api. Please run: pip install mp-api")

warnings.filterwarnings("ignore")

class QuietBlock:
    def __enter__(self):
        self._suppress = io.StringIO()
        self._redirect_out = contextlib.redirect_stdout(self._suppress)
        self._redirect_err = contextlib.redirect_stderr(self._suppress)
        self._redirect_out.__enter__()
        self._redirect_err.__enter__()
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._redirect_out.__exit__(exc_type, exc_val, exc_tb)
        self._redirect_err.__exit__(exc_type, exc_val, exc_tb)

# =============================================================================
# 📂 PATH SETUP
# =============================================================================
PROJECT_ROOT = os.getcwd()
INPUT_CSV = os.path.join(PROJECT_ROOT, "DEEP_VALIDATION_SEMICONDUCTORS", "final_relaxed_results.csv")
INPUT_CIF_DIR = os.path.join(PROJECT_ROOT, "DEEP_VALIDATION_SEMICONDUCTORS", "relaxed_cif_files")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "DEEP_VALIDATION_SEMICONDUCTORS", "golden_candidates.csv")

# =============================================================================
# 🧮 THERMODYNAMIC COMPATIBILITY ENGINE
# =============================================================================
def generate_corrected_cse(structure, m3gnet_total_energy):
    b = MPRelaxSet(structure)
    with tempfile.TemporaryDirectory() as tmpdirname:
        b.write_input(f"{tmpdirname}/", potcar_spec=True)
        poscar = Poscar.from_file(f"{tmpdirname}/POSCAR")
        incar = Incar.from_file(f"{tmpdirname}/INCAR")
        clean_structure = Structure.from_file(f"{tmpdirname}/POSCAR")

    param = {"hubbards": {}}
    if "LDAUU" in incar:
        param["hubbards"] = dict(zip(poscar.site_symbols, incar["LDAUU"]))
    
    param["is_hubbard"] = incar.get("LDAU", True) and sum(param["hubbards"].values()) > 0
    if param["is_hubbard"]:
        param["run_type"] = "GGA+U"

    cse_d = {
        "structure": clean_structure,
        "energy": m3gnet_total_energy,
        "correction": 0.0,
        "parameters": param,
    }

    cse = ComputedStructureEntry.from_dict(cse_d)
    processed = MaterialsProject2020Compatibility(check_potcar=False).process_entries(
        cse, clean=True
    )
    if not processed:
        return cse 
    return processed[0]

def save_incremental_csv(records):
    df = pd.DataFrame(records)
    tier_mapping = {"Tier 1 (Golden)": 1, "Tier 2 (Metastable)": 2, "Reinvention": 3, "Trash": 4}
    df["tier_rank"] = df["publication_tier"].map(tier_mapping)
    df = df.sort_values(by=["tier_rank", "e_hull"]).drop(columns=["tier_rank"])
    df.to_csv(OUTPUT_CSV, index=False)

# =============================================================================
# 🏆 THE MAIN VALIDATOR
# =============================================================================
def run_golden_validation():
    print("="*80)
    print("🏆 THE GOLDEN CANDIDATE VALIDATOR".center(80))
    print("="*80)

    if not os.path.exists(INPUT_CSV):
        sys.exit(f"❌ Input CSV not found: {INPUT_CSV}\nRun final_relaxation.py first.")

    MP_API_KEY = "eWyH34PzCvMfwZol5jN6CIeCGk4i7j5n"
    df = pd.read_csv(INPUT_CSV)
    
    golden_records = []
    processed_files = set()

    # 🔄 THE RESUME MECHANISM
    if os.path.exists(OUTPUT_CSV):
        try:
            existing_df = pd.read_csv(OUTPUT_CSV)
            processed_files = set(existing_df["file_name"].tolist())
            golden_records = existing_df.to_dict('records')
            print(f"🔄 Found existing save file. Resuming... Skipped {len(processed_files)} already validated crystals.")
        except Exception as e:
            print(f"⚠️ Could not load existing CSV. Starting fresh. ({e})")

    print(f"\n📂 Loaded {len(df)} relaxed semiconductors for deep validation.")

    matcher = StructureMatcher()
    chemsys_cache = {} 

    with MPRester(MP_API_KEY) as mpr:
        pbar = tqdm(total=len(df), desc="Validating", ncols=100)
        pbar.update(len(processed_files))

        for _, row in df.iterrows():
            if row["file_name"] in processed_files:
                continue 

            try:
                cif_path = os.path.join(INPUT_CIF_DIR, row["file_name"])
                if not os.path.exists(cif_path):
                    pbar.update(1)
                    continue

                struct = Structure.from_file(cif_path)
                formula = struct.composition.reduced_formula
                elements = [str(el) for el in struct.composition.elements]
                
                # 🛑 THE FIX: High-Dimensional API Guard
                if len(elements) > 5:
                    pbar.write(f"⏭️ Skipping {formula}: Too many elements ({len(elements)}) for API Phase Diagram.")
                    pbar.update(1)
                    continue

                system_key = "-".join(sorted(elements))
                
                # --- API RETRY LOGIC ---
                max_retries = 3
                system_entries = None
                known_docs = None
                
                for attempt in range(max_retries):
                    try:
                        if system_key not in chemsys_cache:
                            with QuietBlock():
                                system_entries = mpr.get_entries_in_chemsys(elements)
                            chemsys_cache[system_key] = system_entries
                        else:
                            system_entries = chemsys_cache[system_key]
                            
                        with QuietBlock():
                            known_docs = mpr.materials.summary.search(formula=formula, fields=["material_id", "structure"])
                        
                        break 
                        
                    except Exception as e:
                        if attempt < max_retries - 1:
                            wait_time = 10 * (attempt + 1)
                            pbar.write(f"⏳ Server timeout on {formula}. Cooling down for {wait_time}s...")
                            time.sleep(wait_time)
                        else:
                            raise e 
                
                if system_entries is None or known_docs is None:
                    raise Exception("Failed to fetch API data after multiple retries.")

                # --- PILLAR 1: PHASE DIAGRAM & THERMODYNAMICS ---
                total_energy = row["energy"] * struct.num_sites
                target_entry = generate_corrected_cse(struct, total_energy)
                
                pd_diagram = PhaseDiagram(system_entries + [target_entry])
                e_form = pd_diagram.get_form_energy_per_atom(target_entry)
                e_hull = pd_diagram.get_e_above_hull(target_entry, allow_negative=True)

                # --- PILLAR 2: NOVELTY CHECK ---
                is_novel = True
                matched_id = "None"
                for doc in known_docs:
                    if matcher.fit(struct, doc.structure):
                        is_novel = False
                        matched_id = str(doc.material_id)
                        break 

                # --- PILLAR 3: THE GOLDEN TIERING ---
                publication_tier = "Trash"
                if is_novel:
                    if e_hull <= 0.0:
                        publication_tier = "Tier 1 (Golden)"
                    elif e_hull <= 0.05:
                        publication_tier = "Tier 2 (Metastable)"
                else:
                    publication_tier = "Reinvention"

                golden_records.append({
                    "file_name": row["file_name"],
                    "formula": formula,
                    "type": row.get("type", "Semiconductor"),
                    "space_group_symbol": row["space_group_symbol"],
                    "space_group_number": row.get("space_group_number", 1),
                    "num_atoms": row["num_atoms"],
                    "bandgap": row["bandgap"],
                    "ENERGY": row["energy"],
                    "e_form": round(e_form, 4),
                    "e_hull": round(e_hull, 4),
                    "is_novel": is_novel,
                    "matched_mp_id": matched_id,
                    "publication_tier": publication_tier
                })
                
                if len(golden_records) % 10 == 0:
                    save_incremental_csv(golden_records)

            except Exception as e:
                pbar.write(f"⚠️ Failed validating {row.get('formula', 'Unknown')}: {e}")
            
            pbar.update(1)
        pbar.close()

    if golden_records:
        save_incremental_csv(golden_records) 
        golden_df = pd.DataFrame(golden_records)
        tier1_count = len(golden_df[golden_df["publication_tier"] == "Tier 1 (Golden)"])
        tier2_count = len(golden_df[golden_df["publication_tier"] == "Tier 2 (Metastable)"])
        
        print("\n" + "="*80)
        print("🏆 VALIDATION COMPLETE".center(80))
        print("="*80)
        print(f"   🌟 Tier 1 (Golden Novelties): {tier1_count}")
        print(f"   🥈 Tier 2 (Metastables):      {tier2_count}")
        print(f"   💾 Saved to: {OUTPUT_CSV}")
        print("="*80)
    else:
        print("\n❌ No candidates successfully validated.")

if __name__ == "__main__":
    run_golden_validation()