import os
import torch
import warnings
import numpy as np

# --- THREAD LOCKDOWN (Critical for RL stability) ---
# Prevents one worker from hogging all CPU cores
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"
torch.set_num_threads(1)

# Force DGL/MatGL to behave
os.environ["DGLBACKEND"] = "pytorch"
warnings.simplefilter("ignore")

from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.filters import UnitCellFilter
from ase.optimize import LBFGS
import matgl
from matgl.ext.ase import M3GNetCalculator

class CrystalRelaxer:
    """
    State-of-the-Art Relaxer using M3GNet.
    Takes a 'rough draft' structure and optimizes its geometry.
    """
    def __init__(self, model_name="M3GNet-MP-2021.2.8-PES", device=None):
        # 1. Setup Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"🔧 Initializing Relaxer on {self.device}...")
        
        # 2. Load Model Once
        try:
            self.potential = matgl.load_model(model_name)
            self.calculator = M3GNetCalculator(potential=self.potential)
            print("   ✅ M3GNet Model Loaded.")
        except Exception as e:
            print(f"   ❌ Failed to load M3GNet: {e}")
            raise e

    def relax(self, structure: Structure, fmax=0.05, steps=200):
        """
        Relax a single structure.
        Returns: {
            "final_structure": Structure, 
            "energy": float (eV/atom), 
            "converged": bool
        }
        """
        # 1. Convert Pymatgen -> ASE
        try:
            atoms = AseAtomsAdaptor.get_atoms(structure)
            atoms.calc = self.calculator
        except Exception as e:
            return {"final_structure": structure, "energy": 0.0, "converged": False, "error": f"Conversion Fail: {e}"}

        # 2. Run Optimization
        # We use a try-except block to catch "Exploding Gradients"
        traj = []
        try:
            # UnitCellFilter allows the lattice box to change shape (critical for learned lattices)
            ucf = UnitCellFilter(atoms)
            
            # LBFGS is the standard optimizer for crystals
            optimizer = LBFGS(ucf, logfile=None)
            
            # Run
            optimizer.run(fmax=fmax, steps=steps)
            
            # Check convergence
            forces = atoms.get_forces()
            max_force = np.sqrt((forces ** 2).sum(axis=1).max())
            converged = max_force <= (fmax * 1.5) # slightly loose tolerance
            
        except Exception as e:
            return {"final_structure": structure, "energy": 0.0, "converged": False, "error": f"Optimization Crash: {e}"}

        # 3. Extract Results
        try:
            final_energy = atoms.get_potential_energy()
            num_atoms = len(atoms)
            energy_per_atom = final_energy / num_atoms
            
            # 4. DEEP CLEAN (Prevent Memory Leaks)
            # We convert back to Pymatgen and discard the ASE object
            final_struct = AseAtomsAdaptor.get_structure(atoms)
            
            # Sanitize: Convert to dict and back to strip attached calculators
            clean_struct = Structure.from_dict(final_struct.as_dict())
            
        except Exception as e:
            return {"final_structure": structure, "energy": 0.0, "converged": False, "error": f"Extraction Fail: {e}"}

        return {
            "final_structure": clean_struct,
            "energy_per_atom": energy_per_atom,
            "converged": converged,
            "error": None
        }

# --- TEST BLOCK ---
if __name__ == "__main__":
    from pymatgen.core import Lattice
    
    # 1. Create a "Rough" Iron crystal (slightly stretched)
    # Iron usually has a lattice of ~2.87 A. We stretch it to 3.0 A.
    rough_s = Structure(Lattice.cubic(3.0), ["Fe", "Fe"], [[0,0,0], [0.5, 0.5, 0.5]])
    
    print(f"📉 Starting Relaxation...")
    print(f"   Initial Volume: {rough_s.volume:.2f} A^3")
    
    relaxer = CrystalRelaxer()
    result = relaxer.relax(rough_s)
    
    if result["converged"]:
        final_s = result["final_structure"]
        print("   ✅ Relaxation Converged!")
        print(f"   Final Volume:   {final_s.volume:.2f} A^3 (Should be smaller)")
        print(f"   Energy:         {result['energy_per_atom']:.4f} eV/atom")
    else:
        print(f"   ❌ Relaxation Failed: {result.get('error')}")
