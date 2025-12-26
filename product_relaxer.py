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
        try:
            self.pot = matgl.load_model("M3GNet-2021.2.8-PES")
        except Exception:
            # Fallback if download fails
            self.pot = matgl.load_model("M3GNet-2021.2.8-PES")
            
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
            calc = M3GNetCalculator(potential=self.pot)
            
            # 2. Setup Relaxer with step limit
            # fmax=0.1 is standard "loose" convergence
            relaxer = Relaxer(potential=self.pot, optimizer="Fire", fmax=0.1)
            
            # 3. Run Relaxation
            # We trap the internal ASE loop to respect 'steps'
            result = relaxer.relax(structure, steps=steps)
            
            final_s = result["final_structure"]
            final_e = float(result["trajectory"].energies[-1])
            n_atoms = len(final_s)
            
            return {
                "final_structure": final_s,
                "energy_per_atom": final_e / n_atoms,
                "converged": True # If it ran without crash, we count it
            }
            
        except Exception as e:
            # If physics breaks (atoms fly apart), return the bad news
            # Return high energy penalty
            return {
                "final_structure": structure,
                "energy_per_atom": 5.0, # Punishment value
                "converged": False
            }
