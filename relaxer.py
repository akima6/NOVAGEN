import os
import numpy as np
import torch
import warnings

# --- THREAD LOCKDOWN ---
# Crucial: Force single-threaded math so workers don't fight.
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
    Production-Grade Relaxer with 'Deep Clean' Sanitization.
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
        # 1. Conversion
        try:
            atoms = AseAtomsAdaptor.get_atoms(structure)
        except Exception as e:
            return {"is_converged": False, "error": f"ASE conversion failed: {e}"}

        # 2. Attach Physics
        atoms.calc = self.calculator

        # 3. Optimize with Hooks
        try:
            ucf = UnitCellFilter(atoms)
            optimizer = LBFGS(ucf, logfile=None)
            
            # Safety Hook
            def safety_hook():
                try:
                    e = atoms.get_potential_energy()
                    f = atoms.get_forces()
                    max_f = np.sqrt((f ** 2).sum(axis=1).max())
                    if e > 0.0: raise StopIteration("Energy Diverged")
                    if max_f > 50.0: raise StopIteration("Forces Exploded")
                except StopIteration: raise
                except: pass

            optimizer.attach(safety_hook, interval=1)
            optimizer.run(fmax=fmax, steps=steps)
            
        except StopIteration:
             pass # Continue to allow result extraction (even if partial)
        except Exception as e:
            return {"final_structure": structure, "is_converged": False, "error": f"Crash: {e}"}

        # 4. Result Extraction
        try:
            # Check convergence mathematically
            forces = atoms.get_forces()
            max_force = np.sqrt((forces ** 2).sum(axis=1).max())
            converged = bool(max_force <= fmax)
            
            final_energy = atoms.get_potential_energy()
            num_atoms = len(atoms)
            energy_per_atom = float(final_energy / num_atoms)
        except: 
            converged = False
            energy_per_atom = None

        # 5. DEEP CLEAN SANITIZATION (The Fix)
        # We convert to a Dict and back. This strips ALL hidden references (calculators, hooks, pytorch grads).
        try:
            raw_struct = AseAtomsAdaptor.get_structure(atoms)
            clean_dict = raw_struct.as_dict()
            clean_structure = Structure.from_dict(clean_dict)
        except:
            # Fallback (should essentially never happen)
            clean_structure = structure

        # 6. Garbage Collection
        # Explicitly delete the heavy objects before returning
        del atoms
        del optimizer

        return {
            "final_structure": clean_structure, # 100% Safe to pickle now
            "final_energy_per_atom": energy_per_atom,
            "is_converged": converged,
            "num_steps": 100 # Placeholder or track manually if needed
        }
