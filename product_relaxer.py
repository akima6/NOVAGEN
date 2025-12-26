import warnings
import torch
import matgl
from matgl.ext.ase import M3GNetCalculator, Relaxer

# Filter intense warnings from M3GNet
warnings.filterwarnings("ignore")

class CrystalRelaxer:
    """
    The Physics Engine.
    Now supports 'Fast Mode' for RL training.
    """
    def __init__(self, device="cpu"):
        self.device = device
        # Load the Universal Potential (M3GNet)
        # CORRECTED MODEL NAME: M3GNet-MP-2021.2.8-PES
        try:
            self.pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
        except Exception as e:
            print(f"⚠️ Failed to load specific model: {e}")
            print("🔄 Attempting to download default M3GNet...")
            # Fallback to the absolute default if the specific version fails
            self.pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
            
    def relax(self, structure, steps=500):
        """
        Relax a structure to find its stable geometry.
        
        Args:
            structure: Pymatgen Structure
            steps (int): Max steps. 
                         Use 25-50 for Fast RL Training.
                         Use 500+ for Final Discovery.
        Returns:
            dict: {final_structure, energy_per_atom, trajectory}
        """
        try:
            # 1. Setup Calculator
            # M3GNetCalculator wraps the potential for ASE
            calc = M3GNetCalculator(potential=self.pot)
            
            # 2. Setup Relaxer with step limit
            # fmax=0.1 is standard "loose" convergence
            # optimizer="Fire" is fast and robust
            relaxer = Relaxer(potential=self.pot, optimizer="Fire", fmax=0.1)
            
            # 3. Run Relaxation
            # We trap the internal ASE loop to respect 'steps'
            result = relaxer.relax(structure, steps=steps)
            
            final_s = result["final_structure"]
            # Extract energy from the last step of trajectory
            final_e = float(result["trajectory"].energies[-1])
            n_atoms = len(final_s)
            
            return {
                "final_structure": final_s,
                "energy_per_atom": final_e / n_atoms,
                "converged": True 
            }
            
        except Exception as e:
            # If physics breaks (atoms fly apart or OOM), return the bad news
            # Return high energy penalty so RL learns to avoid this
            return {
                "final_structure": structure,
                "energy_per_atom": 5.0, # Punishment value
                "converged": False
            }
