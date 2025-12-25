import os
import torch
import warnings
import numpy as np
import traceback # <--- Added for detailed error tracing

# --- THREAD LOCKDOWN ---
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
            self.potential = matgl.load_model(model_name)
            
            # FORCE GPU MOVE and Log it
            if self.device.type == 'cuda':
                self.potential = self.potential.to(self.device)
                print("   🚀 Moved Potential to GPU.")
                
                # DEBUG 1: Check where the model actually is
                param = next(self.potential.model.parameters())
                print(f"   [DEBUG] Model Weight Device: {param.device}")

            self.calculator = M3GNetCalculator(potential=self.potential)
            print("   ✅ M3GNet Model Loaded.")
        except Exception as e:
            print(f"   ❌ Failed to load M3GNet: {e}")
            raise e

    def relax(self, structure: Structure, fmax=0.05, steps=200):
        try:
            # 1. Pymatgen -> ASE
            atoms = AseAtomsAdaptor.get_atoms(structure)
            atoms.calc = self.calculator
            
            # DEBUG 2: Trigger a single static calculation BEFORE optimization
            # This helps us see if the crash happens on data transfer
            print("\n   [DEBUG] Testing Static Energy Calculation...")
            try:
                # This forces the calculator to take the atoms (CPU) and feed the model (GPU)
                e_static = atoms.get_potential_energy()
                print(f"   [DEBUG] Static Energy: {e_static:.3f} eV (Success)")
            except Exception as e:
                print(f"   [DEBUG] ❌ Static Calc Failed! The bridge is broken.")
                # We re-raise to be caught by the outer block, but now we know WHERE.
                raise e

            # 3. Optimize
            print("   [DEBUG] Starting LBFGS Loop...")
            ucf = UnitCellFilter(atoms)
            optimizer = LBFGS(ucf, logfile=None) 
            optimizer.run(fmax=fmax, steps=steps)
            
            # 4. Check Convergence
            forces = atoms.get_forces()
            max_force = np.sqrt((forces ** 2).sum(axis=1).max())
            converged = max_force <= (fmax * 1.5)
            
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
            # Detailed Error Logging
            err_msg = str(e)
            print(f"   [DEBUG] CRASH DETAILS: {err_msg}")
            # print(traceback.format_exc()) # Uncomment if you want full huge stack trace
            
            return {
                "final_structure": structure, 
                "energy": 0.0, 
                "converged": False, 
                "error": err_msg
            }
