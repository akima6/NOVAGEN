import os
import torch
import warnings
import numpy as np

# --- THREAD LOCKDOWN ---
# Keeps CPU clear for data transfers
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"
torch.set_num_threads(1)

os.environ["DGLBACKEND"] = "pytorch"
warnings.simplefilter("ignore")

from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.filters import UnitCellFilter
from ase.optimize import LBFGS
import matgl
from matgl.ext.ase import M3GNetCalculator

class CrystalRelaxer:
    def __init__(self, model_name="M3GNet-MP-2021.2.8-PES", device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"🔧 Initializing Relaxer on {self.device}...")
        
        try:
            # Load the potential
            self.potential = matgl.load_model(model_name)
            
            # --- THE FIX: MOVE THE ENTIRE OBJECT TO GPU ---
            # This ensures input graphs are also cast to GPU automatically
            if self.device.type == 'cuda':
                self.potential = self.potential.to(self.device)
                print("   🚀 M3GNet Pipeline moved to GPU.")
            # -----------------------------------------------

            self.calculator = M3GNetCalculator(potential=self.potential)
            print("   ✅ M3GNet Model Loaded.")
        except Exception as e:
            print(f"   ❌ Failed to load M3GNet: {e}")
            raise e

    def relax(self, structure: Structure, fmax=0.05, steps=200):
        try:
            # 1. Pymatgen -> ASE
            atoms = AseAtomsAdaptor.get_atoms(structure)
            
            # 2. Attach Calculator
            atoms.calc = self.calculator
            
            # 3. Optimize
            # ASE runs the loop on CPU, but the Calculator sends data to GPU
            ucf = UnitCellFilter(atoms)
            optimizer = LBFGS(ucf, logfile=None) 
            optimizer.run(fmax=fmax, steps=steps)
            
            # 4. Check Convergence
            forces = atoms.get_forces()
            max_force = np.sqrt((forces ** 2).sum(axis=1).max())
            converged = max_force <= (fmax * 1.5)
            
            # 5. Extract Results
            final_energy = atoms.get_potential_energy()
            num_atoms = len(atoms)
            energy_per_atom = final_energy / num_atoms
            
            final_struct = AseAtomsAdaptor.get_structure(atoms)
            clean_struct = Structure.from_dict(final_struct.as_dict())

            return {
                "final_structure": clean_struct,
                "energy_per_atom": energy_per_atom,
                "converged": converged,
                "error": None
            }
            
        except Exception as e:
            # Catch crashes (like exploding gradients)
            return {
                "final_structure": structure, 
                "energy": 0.0, 
                "converged": False, 
                "error": str(e)
            }
