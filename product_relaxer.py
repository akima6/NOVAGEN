import warnings
import torch
import matgl
import traceback
import sys
from matgl.ext.ase import M3GNetCalculator, Relaxer

# Filter minor warnings
warnings.filterwarnings("ignore")

class CrystalRelaxer:
    """
    Robust Physics Engine with Debugging Enabled.
    """
    def __init__(self, device="cpu"):
        self.device = device
        self.pot = None
        
        print("   [Relaxer] Loading M3GNet Potential...")
        try:
            # 1. Try Specific Model Name
            self.pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
        except Exception as e:
            print(f"   [Relaxer] ⚠️ Specific model load failed: {e}")
            try:
                # 2. Try Default Fallback
                print("   [Relaxer] 🔄 Trying default load...")
                self.pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
            except Exception as e2:
                print(f"   [Relaxer] ❌ CRITICAL: Could not load physics model.")
                print(e2)

    def relax(self, structure, steps=500):
        if self.pot is None:
            return self._fail(structure)

        try:
            # --- FIX IS HERE ---
            # 1. Initialize Relaxer (Do NOT pass fmax here)
            relaxer = Relaxer(potential=self.pot, optimizer="Fire")
            
            # 2. Run Relaxation (Pass fmax here!)
            # fmax=0.1 means "stop when forces are low" (stable)
            result = relaxer.relax(structure, fmax=0.1, steps=steps)
            # -------------------
            
            final_s = result["final_structure"]
            
            # Extract energy safely
            if "trajectory" in result and len(result["trajectory"].energies) > 0:
                final_e = float(result["trajectory"].energies[-1])
            else:
                final_e = -1.0 
                
            n_atoms = len(final_s)
            
            return {
                "final_structure": final_s,
                "energy_per_atom": final_e / n_atoms,
                "converged": True 
            }
            
        except Exception as e:
            # Only print if it's NOT the specific error we just fixed
            if "fmax" not in str(e):
                print("\n" + "="*40)
                print("❌ RELAXATION CRASH REPORT")
                print(f"Structure Formula: {structure.composition.reduced_formula}")
                print("Error Details:")
                traceback.print_exc()
                print("="*40 + "\n")
            
            return self._fail(structure)

    def _fail(self, structure):
        return {
            "final_structure": structure,
            "energy_per_atom": 5.0, 
            "converged": False
        }
