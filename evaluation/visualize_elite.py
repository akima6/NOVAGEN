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

warnings.filterwarnings("ignore")

# =============================================================================
# 🔧 CONFIGURATION
# =============================================================================
# Main Data Source (The Final CSV)
REPORT_PATH = r"C:\Users\REHNA\NOVAGEN\final_results\NOVAGEN_NOVELTY_REPORT.csv"

# Optional: CIF Folder (For calculating Density/Volume for t-SNE)
# If this path is wrong, the script will auto-switch to "Chemistry Only" mode.
CIF_DIR_HINT = r"C:\Users\REHNA\NOVAGEN\final_results\combined cif" 

# Output Image
OUTPUT_IMG = r"C:\Users\REHNA\NOVAGEN\final_results\discovery_dashboard.png"

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
    print(f"\n🎨 GENERATING COMBINED DISCOVERY DASHBOARD")
    print(f"   Input CSV: {REPORT_PATH}")
    
    if not os.path.exists(REPORT_PATH):
        print("❌ Error: Report CSV not found.")
        return

    # 1. Load Data
    df = pd.read_csv(REPORT_PATH)
    
    # Clean Column Names (Strip spaces)
    df.columns = [c.strip() for c in df.columns]
    
    # Filter for Plotting (Remove extreme outliers for better visuals)
    # We keep "Stable" and "Metastable" mostly
    df_clean = df.copy()
    if 'energy' in df_clean.columns:
        df_clean = df_clean[df_clean['energy'] < 0.5] # Remove huge errors

    print(f"   🔹 Processing {len(df_clean)} candidates...")

    # 2. FEATURE EXTRACTION (For t-SNE)
    # We try to load CIFs for Density/Volume. If failed, we use Formula-only features.
    print("   🔹 Calculating t-SNE Features...")
    
    features = []
    
    for idx, row in tqdm(df_clean.iterrows(), total=len(df_clean), ncols=100):
        f_feat = {}
        
        # A. Chemical Features (Always available)
        try:
            comp = Composition(row['formula'])
            f_feat['avg_z'] = sum([e.Z * amt for e, amt in comp.items()]) / comp.num_atoms
            f_feat['avg_eneg'] = sum([e.X * amt for e, amt in comp.items()]) / comp.num_atoms
        except:
            f_feat['avg_z'] = 0
            f_feat['avg_eneg'] = 0

        # B. Structural Features (Try to find CIF)
        cif_name = row.get('File', row.get('file_name', ''))
        cif_full_path = os.path.join(CIF_DIR_HINT, str(cif_name))
        
        if os.path.exists(cif_full_path):
            try:
                s = Structure.from_file(cif_full_path)
                f_feat['density'] = s.density
                f_feat['vol_pa'] = s.volume / s.num_sites
            except:
                f_feat['density'] = 0
                f_feat['vol_pa'] = 0
        else:
            # Fallback if CIF missing
            f_feat['density'] = 0
            f_feat['vol_pa'] = 0
            
        features.append(f_feat)

    df_feat = pd.DataFrame(features)
    
    # Decide which columns to use for t-SNE
    # If density is mostly 0, we drop it to avoid skewing
    use_cols = ['avg_z', 'avg_eneg']
    if df_feat['density'].sum() > 0:
        use_cols += ['density', 'vol_pa']
        print("      (Using Full Structural + Chemical Features)")
    else:
        print("      (CIFs not found - Using Chemical Features Only)")

    # 3. RUN t-SNE
    X = df_feat[use_cols].values
    X_scaled = StandardScaler().fit_transform(X)
    
    perp = min(30, len(df_clean)//5) # Safety for small datasets
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
    X_embedded = tsne.fit_transform(X_scaled)
    
    df_clean['tsne_x'] = X_embedded[:, 0]
    df_clean['tsne_y'] = X_embedded[:, 1]
    
    # 4. PREPARE LABELS
    # Crystal System
    if 'space_group_number' in df_clean.columns:
        df_clean['System'] = df_clean['space_group_number'].apply(get_crystal_system)
    else:
        df_clean['System'] = "Unknown"
        
    # Primary Element
    df_clean['Primary_El'] = df_clean['formula'].apply(get_primary_element)
    
    # Top Elements List
    top_els = df_clean['Primary_El'].value_counts().nlargest(8).index
    df_clean['El_Label'] = df_clean['Primary_El'].apply(lambda x: x if x in top_els else "Other")

    # ================= PLOTTING =================
    print("   🎨 Rendering High-Res Dashboard...")
    
    # Layout: 3 Rows x 2 Columns
    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.2, 0.8])
    sns.set_style("whitegrid")

    # --- ROW 1: THE DISCOVERY LANDSCAPE ---
    
    # Plot 1: Stability vs Bandgap (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    sns.scatterplot(
        data=df_clean, x='bandgap', y='energy', 
        hue='Novelty_Status', style='Origin_Campaign',
        palette={'Known': 'gray', 'Novel Composition': '#D90429'}, # Red for Novel
        alpha=0.7, s=100, ax=ax1
    )
    # Highlight Windows
    ax1.axvspan(1.0, 1.8, color='orange', alpha=0.1, label='Solar Window')
    ax1.axvspan(0.4, 0.9, color='purple', alpha=0.1, label='Thermal Window')
    ax1.invert_yaxis() # Lower energy is better (higher up)
    ax1.set_title('The Discovery Landscape (Stability vs Bandgap)', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Band Gap (eV)')
    ax1.set_ylabel('Energy Above Hull (eV/atom)')
    ax1.legend(loc='upper right', fontsize=10)

    # Plot 2: Campaign Efficiency (Top Right)
    ax2 = fig.add_subplot(gs[0, 1])
    if 'Origin_Campaign' in df_clean.columns:
        sns.countplot(data=df_clean, x='Origin_Campaign', hue='Novelty_Status', 
                      palette={'Known': 'gray', 'Novel Composition': '#D90429'}, ax=ax2)
        ax2.set_title('Discovery Efficiency by Campaign', fontsize=16, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, "Campaign Data Not Available", ha='center')

    # --- ROW 2: THE MAP (t-SNE) ---
    
    # Plot 3: t-SNE by Symmetry (Middle Left)
    ax3 = fig.add_subplot(gs[1, 0])
    # Order legend by count
    sys_counts = df_clean['System'].value_counts()
    df_clean['Sys_Label'] = df_clean['System'].apply(lambda x: f"{x} ({sys_counts[x]})")
    
    sns.scatterplot(
        data=df_clean, x='tsne_x', y='tsne_y', 
        hue='Sys_Label', palette='tab10', 
        s=80, alpha=0.8, ax=ax3
    )
    ax3.set_title('Materials Map: Clustered by Symmetry', fontsize=16, fontweight='bold')
    ax3.legend(title="Crystal System", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Plot 4: t-SNE by Chemistry (Middle Right)
    ax4 = fig.add_subplot(gs[1, 1])
    sns.scatterplot(
        data=df_clean, x='tsne_x', y='tsne_y', 
        hue='El_Label', palette='viridis', 
        s=80, alpha=0.8, ax=ax4
    )
    ax4.set_title('Materials Map: Clustered by Chemistry', fontsize=16, fontweight='bold')
    ax4.legend(title="Primary Element", bbox_to_anchor=(1.05, 1), loc='upper left')

    # --- ROW 3: STATISTICS ---
    
    # Plot 5: Element Usage (Bottom Left)
    ax5 = fig.add_subplot(gs[2, 0])
    elem_counts = analyze_elements(df_clean['formula'])
    df_el = pd.DataFrame.from_dict(elem_counts, orient='index', columns=['Count']).sort_values('Count', ascending=False).head(15)
    sns.barplot(x=df_el.index, y=df_el['Count'], palette="magma", ax=ax5)
    ax5.set_title('Top 15 Elements in Stable Candidates', fontsize=14)
    ax5.tick_params(axis='x', rotation=45)

    # Plot 6: Pie Chart (Bottom Right)
    ax6 = fig.add_subplot(gs[2, 1])
    colors = sns.color_palette("pastel")
    ax6.pie(sys_counts, labels=sys_counts.index, autopct='%1.1f%%', colors=colors, startangle=140)
    ax6.set_title('Distribution of Crystal Systems', fontsize=14)

    # Final Polish
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(OUTPUT_IMG), exist_ok=True)
    plt.savefig(OUTPUT_IMG, dpi=300)
    print(f"   🚀 Saved Dashboard: {OUTPUT_IMG}")
    print("\n" + "="*50)
    print("✅ ANALYSIS COMPLETE")
    print("="*50)

if __name__ == "__main__":
    run_dashboard()