import os
import pandas as pd

# Paths
csv_path = r"C:\Users\REHNA\NOVAGEN\UNIVERSAL_HARVEST_SEMICONDUCTORS\final_harvest_results.csv"
cif_folder = r"C:\Users\REHNA\NOVAGEN\UNIVERSAL_HARVEST_SEMICONDUCTORS\cif_files"

print("Loading CSV...")

df = pd.read_csv(csv_path)

# Ensure the column exists
if "file_name" not in df.columns:
    raise ValueError("CSV must contain a 'file_name' column.")

valid_files = set(df["file_name"].astype(str))

print(f"CSV contains {len(valid_files)} valid CIF references.")

deleted = 0
kept = 0

for file in os.listdir(cif_folder):
    if file.endswith(".cif"):
        if file not in valid_files:
            os.remove(os.path.join(cif_folder, file))
            deleted += 1
        else:
            kept += 1

print("\nCleanup Complete")
print(f"Kept CIF files: {kept}")
print(f"Deleted extra CIF files: {deleted}")