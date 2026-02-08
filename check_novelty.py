import os
import pandas as pd
from mp_api.client import MPRester
from tqdm import tqdm
import time

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
# 🔑 GET KEY AT: https://next-gen.materialsproject.org/api
MP_API_KEY = "eWyH34PzCvMfwZol5jN6CIeCGk4i7j5n"

INPUT_CSV = r"C:\Users\REHNA\NOVAGEN\final_results\NOVAGEN_MASTER_LIBRARY.csv"
OUTPUT_CSV = r"C:\Users\REHNA\NOVAGEN\final_results\NOVAGEN_NOVELTY_REPORT.csv"

def check_novelty():
    print("="*80)
    print("🕵️  NOVELTY DETECTOR: QUERYING MATERIALS PROJECT".center(80))
    print("="*80)

    if not os.path.exists(INPUT_CSV):
        print(f"❌ Error: Master Library CSV not found at:\n   {INPUT_CSV}")
        return

    # 1. Load your candidates
    df = pd.read_csv(INPUT_CSV)
    
    # We only need to check unique formulas to save time
    unique_formulas = df['formula'].unique().tolist()
    
    print(f"   🔹 Total Candidates:  {len(df)}")
    print(f"   🔹 Unique Formulas:   {len(unique_formulas)}")
    print("   📡 Connecting to Materials Project API...")

    # 2. Initialize Database Connection
    try:
        with MPRester(MP_API_KEY) as mpr:
            print("   ✅ Connection Established.")
            
            known_formulas = set()
            
            # 3. Batch Query (Efficient Method)
            # We ask MP: "Do you have ANY of these formulas?"
            print(f"   🔍 Checking {len(unique_formulas)} formulas against the database...")
            
            # We process in chunks of 1000 to be safe
            chunk_size = 1000
            chunks = [unique_formulas[i:i + chunk_size] for i in range(0, len(unique_formulas), chunk_size)]

            for chunk in tqdm(chunks, desc="Querying Batches"):
                try:
                    # Search for materials where the formula is in our list
                    docs = mpr.summary.search(
                        formula=chunk, 
                        fields=["formula_pretty"]
                    )
                    
                    # Add found formulas to our "Known" set
                    for doc in docs:
                        known_formulas.add(doc.formula_pretty)
                        
                except Exception as e:
                    print(f"\n   ⚠️ Batch Error: {e}")
                    continue

    except Exception as e:
        print(f"\n❌ API Connection Failed. Please check your API Key.\n   Error: {e}")
        return

    # 4. Process Results
    print("\n   📝 Mapping results...")
    
    # Create a mapping dictionary: Formula -> Status
    novelty_map = {}
    for f in unique_formulas:
        if f in known_formulas:
            novelty_map[f] = "Known"
        else:
            novelty_map[f] = "Novel Composition"

    # Apply to the dataframe
    df['Novelty_Status'] = df['formula'].map(novelty_map)

    # 5. Save Report
    df.to_csv(OUTPUT_CSV, index=False)

    # 6. Final Stats
    n_novel = len(df[df['Novelty_Status'] == "Novel Composition"])
    n_known = len(df[df['Novelty_Status'] == "Known"])
    percent_novel = (n_novel / len(df)) * 100

    print("\n" + "="*80)
    print("🏆 DISCOVERY RESULTS".center(80))
    print("="*80)
    print(f"   📚 Known Formulas:      {n_known}")
    print(f"   ✨ NOVEL DISCOVERIES:   {n_novel} ({percent_novel:.1f}%)")
    print("-" * 80)
    print(f"   💾 Report Saved:        {OUTPUT_CSV}")
    
    if n_novel > 0:
        print("\n   👀 Top 5 Potential Discoveries:")
        examples = df[df['Novelty_Status'] == "Novel Composition"]['formula'].head(5).tolist()
        for ex in examples:
            print(f"      - {ex}")

if __name__ == "__main__":
    check_novelty()