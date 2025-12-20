import os
import numpy as np
import torch
import warnings

# --- THREAD LOCKDOWN ---
# Crucial for Kaggle: Force single-threaded math so workers don't fight.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"
torch.set_num_threads(1)

# Force DGL to use PyTorch backend
os.environ["DGLBACKEND"] = "pytorch"
warnings.simplefilter("ignore")

from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.filters import UnitCellFilter
from ase.optimize import LBFGS
from ase.geometry import get_distances

import matgl
from matgl.ext.ase import M3GNetCalculator

class Relaxer:
    """
    Production-Grade Relaxer with Early Stopping Hooks.
    """
    def __init__(self):
        try:
            # Load M3GNet on CPU to save VRAM for the Learner
            self.potential = matgl.load_model("M3GNet-MP-2021.2.8-PES")
            self.calculator = M3GNetCalculator(potential=self.potential)
        except Exception as e:
            print(f"Error loading relaxer model: {e}")
            raise e

    def relax(self, structure: Structure, fmax=0.01, steps=100) -> dict:
        try:
            atoms = AseAtomsAdaptor.get_atoms(structure)
        except Exception as e:
            return {"is_converged": False, "error": f"ASE conversion failed: {e}"}

        atoms.calc = self.calculator

        try:
            ucf = UnitCellFilter(atoms)
            optimizer = LBFGS(ucf, logfile=None)
            
            # --- HOOK: Divergence Checker ---
            def safety_hook():
                try:
                    e = atoms.get_potential_energy()
                    f = atoms.get_forces()
                    max_f = np.sqrt((f ** 2).sum(axis=1).max())
                    if e > 0.0: raise StopIteration("Energy Diverged (> 0.0 eV)")
                    if max_f > 50.0: raise StopIteration("Forces Exploded (> 50 eV/A)")
                except StopIteration: raise
                except: pass

            optimizer.attach(safety_hook, interval=1)
            optimizer.run(fmax=fmax, steps=steps)
            
        except StopIteration as e:
             return {"final_structure": structure, "is_converged": False, "error": f"Early Stop: {e}"}
        except Exception as e:
            return {"final_structure": structure, "is_converged": False, "error": f"Crash: {e}"}

        final_structure = AseAtomsAdaptor.get_structure(atoms)
        try:
            forces = atoms.get_forces()
            max_force = np.sqrt((forces ** 2).sum(axis=1).max())
            converged = max_force <= fmax
        except: converged = False

        return {
            "final_structure": final_structure,
            "is_converged": converged,
            "num_steps": optimizer.get_number_of_steps()
        }
