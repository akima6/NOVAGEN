import os
import pickle
from mp_api.client import MPRester
from tqdm import tqdm

MP_API_KEY = "PzCKNI9NZfmGYgFUZlIsejHDNJhu9NiW" #
LOCAL_CACHE_FILE = "mp_thermo_cache_clean.pkl"

def download_clean_database():
    print("📥 Connecting to Materials Project...")
    with MPRester(MP_API_KEY) as mpr:
        docs = mpr.materials.thermo.search(
            thermo_types=["GGA_GGA+U"], 
            fields=["composition", "energy_per_atom"]
        )
    
    clean_entries = []
    for doc in tqdm(docs, desc="Extracting Raw Data"):
        # We store ONLY simple dictionaries to prevent serialization crashes
        clean_entries.append({
            "formula": doc.composition.reduced_formula,
            "energy_per_atom": float(doc.energy_per_atom),
            "elements": [el.symbol for el in doc.composition.elements]
        })

    print(f"\n✅ Extracted {len(clean_entries)} clean entries.")
    save_path = os.path.join("pretrained_model", LOCAL_CACHE_FILE)
    with open(save_path, "wb") as f:
        pickle.dump(clean_entries, f)
    print(f"🚀 Done! Use {LOCAL_CACHE_FILE} in your reward script.")

if __name__ == "__main__":
    download_clean_database()