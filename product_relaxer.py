import warnings
import torch
import matgl
from matgl.ext.ase import M3GNetCalculator, Relaxer
import traceback

warnings.filterwarnings("ignore")

class CrystalRelaxer:
    def __init__(self, device="cpu"):
        self.device = device
        try:
            # Load model
            self.pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
        except Exception as e:
            print(f"Init Error: {e}")
            self.pot = None
            
    def relax(self, structure, steps=500):
        if self.pot is None:
            print("❌ Error: Potential not loaded.")
            return {"final_structure": structure, "energy_per_atom": 5.0, "converged": False}

        try:
            # Setup
            relaxer = Relaxer(potential=self.pot, optimizer="Fire", fmax=0.1)
            
            # Run
            result = relaxer.relax(structure, steps=steps)
            
            final_s = result["final_structure"]
            final_e = float(result["trajectory"].energies[-1])
            n_atoms = len(final_s)
            
            return {
                "final_structure": final_s,
                "energy_per_atom": final_e / n_atoms,
                "converged": True 
            }
            
        except Exception as e:
            # --- THIS IS THE NEW PART ---
            print("\n❌ PHYSICS CRASH DETECTED:")
            print(traceback.format_exc()) # Print the full error
            # -----------------------------
            
            return {
                "final_structure": structure,
                "energy_per_atom": 5.0, 
                "converged": False
            }
