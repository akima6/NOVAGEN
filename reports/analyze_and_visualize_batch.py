import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from pymatgen.core import Structure
import warnings

# Suppress pymatgen warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 🔧 DEFAULT CONFIGURATION
# ---------------------------------------------------------
DEFAULT_CIF_DIR = r"C:\Users\REHNA\NOVAGEN\UNIVERSAL_HARVEST_SOLAR\cif_files"
DEFAULT_OUTPUT_IMG = "batch_analysis_map_with_counts.png"

def get_crystal_system(n):
    """Maps Space Group Number to Crystal System."""
    if n <= 2: return "Triclinic"
    if n <= 15: return "Monoclinic"
    if n <= 74: return "Orthorhombic"
    if n <= 142: return "Tetragonal"
    if n <= 167: return "Trigonal"
    if n <= 194: return "Hexagonal"
    return "Cubic"

def parse_batch(cif_dir):
    """Reads all CIFs and extracts both Stats and Features."""
    files = glob.glob(os.path.join(cif_dir, "*.cif"))
    print(f"📂 Found {len(files)} CIF files in: {cif_dir}")
    
    data = []
    print("💎 Extracting Features & Statistics...")
    
    for filepath in tqdm(files):
        try:
            struct = Structure.from_file(filepath)
            
            # --- 1. Statistical Data ---
            sg_num = struct.get_space_group_info()[1]
            sys_name = get_crystal_system(sg_num)
            
            comp = struct.composition
            primary_el = max(comp.items(), key=lambda x: x[1])[0].symbol
            
            # --- 2. Numerical Features for t-SNE ---
            vol_pa = struct.volume / struct.num_sites
            density = struct.density
            avg_z = sum([e.Z * amt for e, amt in comp.items()]) / comp.num_atoms
            avg_eneg = sum([e.X * amt for e, amt in comp.items()]) / comp.num_atoms

            data.append({
                "filename": os.path.basename(filepath),
                "Space Group": sg_num,
                "System": sys_name,
                "Primary Element": primary_el,
                "vol_per_atom": vol_pa,
                "density": density,
                "avg_z": avg_z,
                "avg_eneg": avg_eneg
            })
        except Exception:
            continue
            
    return pd.DataFrame(data)

def generate_visual_report_with_counts(df, output_file):
    if len(df) < 5:
        print("⚠️ Not enough data points for t-SNE visualization.")
        return

    print("🧠 Running t-SNE (Dimensionality Reduction)...")
    
    # 1. Prepare Features
    features = ["vol_per_atom", "density", "avg_z", "avg_eneg"]
    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Run t-SNE
    tsne = TSNE(n_components=2, perplexity=min(30, len(df)-1), random_state=42)
    X_embedded = tsne.fit_transform(X_scaled)
    
    df["x"] = X_embedded[:, 0]
    df["y"] = X_embedded[:, 1]
    
    # 3. Prepare Labels with Counts (The New Part)
    print("🎨 Generating Plot with Counts...")
    fig, axes = plt.subplots(1, 2, figsize=(24, 10)) # Wider figure for legends
    sns.set(style="whitegrid")

    # --- PLOT 1: SYMMETRY ---
    # Calculate counts
    sys_counts = df["System"].value_counts()
    # Create a new column "System_Label" that includes the count, e.g., "Cubic (12)"
    df["System_Label"] = df["System"].apply(lambda x: f"{x} ({sys_counts[x]})")
    
    # Sort legend by count (descending)
    sorted_systems = sys_counts.index.tolist()
    sorted_labels = [f"{s} ({sys_counts[s]})" for s in sorted_systems]

    sns.scatterplot(
        data=df, x="x", y="y", 
        hue="System_Label", hue_order=sorted_labels, # Enforce sorted order
        palette="tab10", s=60, alpha=0.8, ax=axes[0]
    )
    axes[0].set_title(f"Latent Space by Symmetry (Total: {len(df)})", fontsize=16)
    axes[0].legend(title="Crystal System", loc='upper right', bbox_to_anchor=(1.25, 1))

    # --- PLOT 2: CHEMISTRY ---
    # Identify Top 10 Elements
    top_elements = df["Primary Element"].value_counts().nlargest(10).index
    
    # Create Group Column
    def get_el_group(el):
        return el if el in top_elements else "Other"
    
    df["Element Group"] = df["Primary Element"].apply(get_el_group)
    
    # Calculate counts for these groups
    el_counts = df["Element Group"].value_counts()
    df["Element_Label"] = df["Element Group"].apply(lambda x: f"{x} ({el_counts[x]})")
    
    # Sort legend by count
    sorted_els = el_counts.index.tolist()
    sorted_el_labels = [f"{e} ({el_counts[e]})" for e in sorted_els]

    sns.scatterplot(
        data=df, x="x", y="y", 
        hue="Element_Label", hue_order=sorted_el_labels,
        palette="viridis", s=60, alpha=0.8, ax=axes[1]
    )
    axes[1].set_title("Latent Space by Chemistry", fontsize=16)
    axes[1].legend(title="Primary Element", loc='upper right', bbox_to_anchor=(1.25, 1))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"🚀 Visual Report with Counts saved to: {output_file}")

# ---------------------------------------------------------
# 🚀 MAIN EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Crystal Batch with Counts")
    parser.add_argument("--cif", type=str, default=DEFAULT_CIF_DIR, help="Folder containing .cif files")
    parser.add_argument("--out", type=str, default=DEFAULT_OUTPUT_IMG, help="Output filename")
    
    args = parser.parse_args()
    
    # 1. Parse
    df = parse_batch(args.cif)
    
    if not df.empty:
        # 2. Visuals Only (Since text report is now inside the plot)
        generate_visual_report_with_counts(df, args.out)
    else:
        print("❌ No valid crystals found to analyze.")