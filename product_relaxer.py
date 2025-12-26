import warnings
import torch
import matgl
import traceback
import sys
from matgl.ext.ase import M3GNetCalculator, Relaxer

# Filter minor warnings, but keep errors
warnings.filterwarnings("ignore", category=UserWarning)

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
        """
        Runs physics simulation.
        Debug Mode: Prints exact error if it crashes.
        """
        if self.pot is None:
            print("   [Relaxer] ❌ Error: No potential loaded.")
            return self._fail(structure)

        try:
            # 1. Setup Calculator (The Physics Math)
            # Ensure we are not mixing GPU/CPU incorrectly
            calc = M3GNetCalculator(potential=self.pot)
            
            # 2. Setup Optimizer (The Mover)
            # fmax=0.1 is standard loose convergence for RL
            relaxer = Relaxer(potential=self.pot, optimizer="Fire", fmax=0.1)
            
            # 3. Run Simulation
            result = relaxer.relax(structure, steps=steps)
            
            # 4. Extract Results
            final_s = result["final_structure"]
            # Check if trajectory exists
            if "trajectory" in result and len(result["trajectory"].energies) > 0:
                final_e = float(result["trajectory"].energies[-1])
            else:
                # Fallback if trajectory is empty (rare)
                final_e = -1.0 # Placeholder
                
            n_atoms = len(final_s)
            
            return {
                "final_structure": final_s,
                "energy_per_atom": final_e / n_atoms,
                "converged": True 
            }
            
        except Exception as e:
            # --- DEBUGGING OUTPUT ---
            print("\n" + "="*40)
            print("❌ RELAXATION CRASH REPORT")
            print(f"Structure Formula: {structure.composition.reduced_formula}")
            print("Error Details:")
            traceback.print_exc() # This prints the EXACT line that failed
            print("="*40 + "\n")
            # ------------------------
            
            return self._fail(structure)

    def _fail(self, structure):
        """Helper to return a failure object"""
        return {
            "final_structure": structure,
            "energy_per_atom": 5.0, # Penalty for crashing
            "converged": False
        }
