import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from glob import glob
from tqdm import tqdm
from collections import Counter
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from pymatgen.core import Structure, Composition
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# =============================================================================
# 🔧 CONFIGURATION
# =============================================================================
# Input: The Raw Generation CSV
REPORT_PATH = r"C:\Users\REHNA\NOVAGEN\generated results\combined result\first_generated.csv"

# Input: The folder with the 13,000+ CIFs
CIF_DIR = r"C:\Users\REHNA\NOVAGEN\generated results\combined result\cif" 

# Output: The Dashboard Image
OUTPUT_IMG = r"C:\Users\REHNA\NOVAGEN\generated results\combined result\raw_generation_dashboard.png"

# =============================================================================
# 🧠 HELPER FUNCTIONS
# =============================================================================
CRYSTAL_SYSTEMS = {
    "Triclinic": (1, 2),
    "Monoclinic": (3, 15),
    "Orthorhombic": (16, 74),
    "Tetragonal": (75, 142),
    "Trigonal": (143, 167),
    "Hexagonal": (168, 194),
    "Cubic": (195, 230)
}

def get_crystal_system(sg_num):
    try:
        n = int(sg_num)
        for sys_name, (start, end) in CRYSTAL_SYSTEMS.items():
            if start <= n <= end: return sys_name
    except:
        pass
    return "Unknown"

def analyze_elements(formulas):
    """Parses formulas and counts element usage"""
    counts = Counter()
    for f in formulas:
        try:
            for el in Composition(f).elements:
                counts[str(el)] += 1
        except: pass
    return counts

def get_primary_element(formula):
    try:
        return max(Composition(formula).items(), key=lambda x: x[1])[0].symbol
    except:
        return "Unknown"

# =============================================================================
# 🚀 MAIN DASHBOARD GENERATOR
# =============================================================================
def run_dashboard():
    print(f"\n🎨 GENERATING RAW DISCOVERY DASHBOARD")
    print(f"   Input CSV: {REPORT_PATH}")
    
    if not os.path.exists(REPORT_PATH):
        print("❌ Error: Report CSV not found.")
        return

    # 1. Load Data
    df = pd.read_csv(REPORT_PATH)
    
    # Clean Column Names
    df.columns = [c.strip() for c in df.columns]
    
    # Ensure all required columns exist (fill defaults if missing)
    required_cols = ['file_name', 'formula', 'energy', 'bandgap', 'type', 
                     'space_group_symbol', 'space_group_number', 'num_atoms', 'harvest_mode']
    
    for c in required_cols:
        if c not in df.columns:
            df[c] = 0 if c in ['energy', 'bandgap', 'num_atoms', 'space_group_number'] else "Unknown"

    # Filter for Plotting (Remove extreme outliers for visuals)
    df_clean = df[df['energy'] > -15].copy() 

    print(f"   🔹 Processing {len(df_clean)} raw candidates...")

    # 2. FEATURE EXTRACTION (For t-SNE)
    print("   🔹 Calculating t-SNE Features (Chemistry)...")
    
    features = []
    
    for idx, row in tqdm(df_clean.iterrows(), total=len(df_clean), ncols=100):
        f_feat = {}
        
        # A. Chemical Features (Formula based)
        try:
            comp = Composition(row['formula'])
            f_feat['avg_z'] = sum([e.Z * amt for e, amt in comp.items()]) / comp.num_atoms
            f_feat['avg_eneg'] = sum([e.X * amt for e, amt in comp.items()]) / comp.num_atoms
        except:
            f_feat['avg_z'] = 0
            f_feat['avg_eneg'] = 0

        # Note: We skip CIF loading for t-SNE features here to ensure speed for 13k rows,
        # relying on the robust chemical features + CSV metadata.
        f_feat['density'] = 0 
        f_feat['vol_pa'] = 0
            
        features.append(f_feat)

    df_feat = pd.DataFrame(features)
    
    # t-SNE Setup (Chemical Only for Speed/Robustness)
    use_cols = ['avg_z', 'avg_eneg']

    # 3. RUN t-SNE
    print("   🧠 Running t-SNE dimensionality reduction...")
    X = df_feat[use_cols].values
    X_scaled = StandardScaler().fit_transform(X)
    
    perp = 40 
    # FIX: Removed 'n_iter' to fix the TypeError
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
    X_embedded = tsne.fit_transform(X_scaled)
    
    df_clean['tsne_x'] = X_embedded[:, 0]
    df_clean['tsne_y'] = X_embedded[:, 1]
    
    # 4. PREPARE LABELS
    df_clean['System'] = df_clean['space_group_number'].apply(get_crystal_system)
    
    # Create a label combining Symbol and Number (e.g., "Fm-3m (225)")
    df_clean['SG_Label'] = df_clean['space_group_symbol'].astype(str) + " (" + df_clean['space_group_number'].astype(str) + ")"

    # ================= PLOTTING =================
    print("   🎨 Rendering High-Res Dashboard...")
    
    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.2, 0.8])
    sns.set_style("whitegrid")

    # --- ROW 1: THE GENERATION LANDSCAPE ---
    
    # Plot 1: Energy vs Bandgap (Top Left)
    # Size = num_atoms, Color = harvest_mode
    ax1 = fig.add_subplot(gs[0, 0])
    sns.scatterplot(
        data=df_clean, x='bandgap', y='energy', 
        hue='harvest_mode', size='num_atoms',
        palette='deep', alpha=0.6, sizes=(20, 200), ax=ax1
    )
    ax1.axvspan(1.0, 1.8, color='orange', alpha=0.15, label='Solar Window')
    ax1.set_title('Raw Landscape: Energy vs Bandgap (Size = Atom Count)', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Band Gap (eV)')
    ax1.set_ylabel('Energy (eV/atom)')
    ax1.legend(loc='upper right', fontsize=10, ncol=2)
    ax1.set_xlim(-0.5, 6.0)

    # Plot 2: Harvest Mode vs Type (Top Right)
    ax2 = fig.add_subplot(gs[0, 1])
    if 'harvest_mode' in df_clean.columns:
        sns.countplot(data=df_clean, x='harvest_mode', hue='type', 
                      palette='Set2', ax=ax2)
        ax2.set_title('Yield by Harvest Mode & Material Type', fontsize=16, fontweight='bold')
        ax2.set_ylabel('Count of Candidates')
    else:
        ax2.text(0.5, 0.5, "Missing Harvest Mode", ha='center')

    # --- ROW 2: THE MAP (t-SNE) ---
    
    # Plot 3: t-SNE by Harvest Mode (Middle Left)
    ax3 = fig.add_subplot(gs[1, 0])
    sns.scatterplot(
        data=df_clean, x='tsne_x', y='tsne_y', 
        hue='harvest_mode', palette='bright', 
        s=40, alpha=0.6, ax=ax3
    )
    ax3.set_title('Generative Diversity: Clustered by Harvest Mode', fontsize=16, fontweight='bold')
    ax3.legend(title="Harvest Strategy", loc='upper right')

    # Plot 4: t-SNE by Crystal System (Middle Right)
    ax4 = fig.add_subplot(gs[1, 1])
    sys_counts = df_clean['System'].value_counts()
    top_sys = sys_counts.head(7).index
    df_clean['Sys_Group'] = df_clean['System'].apply(lambda x: x if x in top_sys else "Other")
    
    sns.scatterplot(
        data=df_clean, x='tsne_x', y='tsne_y', 
        hue='Sys_Group', palette='tab10', 
        s=40, alpha=0.7, ax=ax4
    )
    ax4.set_title('Structural Map: Clustered by Crystal System', fontsize=16, fontweight='bold')
    ax4.legend(title="Structure", loc='upper right', ncol=2)

    # --- ROW 3: STATISTICS ---
    
    # Plot 5: Element Usage (Bottom Left)
    ax5 = fig.add_subplot(gs[2, 0])
    elem_counts = analyze_elements(df_clean['formula'])
    df_el = pd.DataFrame.from_dict(elem_counts, orient='index', columns=['Count']).sort_values('Count', ascending=False).head(20)
    sns.barplot(x=df_el.index, y=df_el['Count'], palette="magma", ax=ax5)
    ax5.set_title('Top 20 Elements Used in Generation', fontsize=14)
    ax5.tick_params(axis='x', rotation=45)

    # Plot 6: Space Group Distribution (Bottom Right)
    ax6 = fig.add_subplot(gs[2, 1])
    # Top 10 Specific Space Groups (Symbol + Number)
    top_sgs = df_clean['SG_Label'].value_counts().head(10)
    sns.barplot(x=top_sgs.values, y=top_sgs.index, palette="viridis", ax=ax6)
    ax6.set_title('Top 10 Most Frequent Space Groups', fontsize=14)
    ax6.set_xlabel("Count")

    # Final Polish
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(OUTPUT_IMG), exist_ok=True)
    plt.savefig(OUTPUT_IMG, dpi=300)
    print(f"   🚀 Saved Dashboard: {OUTPUT_IMG}")
    print("\n" + "="*50)
    print("✅ RAW VISUALIZATION COMPLETE")
    print("="*50)

if __name__ == "__main__":
    run_dashboard()