import os
import torch
import warnings
import numpy as np
import matgl
from matgl.ext.ase import M3GNetCalculator

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

class CrystalRelaxer:
    def __init__(self, device="cpu"):
        print(f"🔧 Initializing Relaxer (Optimized CPU Mode)...")
        try:
            self.potential = matgl.load_model("M3GNet-MP-2021.2.8-PES")
            self.calculator = M3GNetCalculator(potential=self.potential)
            print("   ✅ M3GNet Model Loaded.")
        except Exception as e:
            print(f"   ❌ Failed to load M3GNet: {e}")
            raise e

    def relax(self, structure: Structure, fmax=0.05, steps=100):
        # 1. Conversion
        try:
            atoms = AseAtomsAdaptor.get_atoms(structure)
        except Exception as e:
            return {"converged": False, "error": f"ASE conversion failed: {e}"}

        # 2. Attach Physics
        atoms.calc = self.calculator

        # 3. Optimize with Safety Hooks
        try:
            ucf = UnitCellFilter(atoms)
            optimizer = LBFGS(ucf, logfile=None)
            
            # THE SAFETY HOOK (Kill Switch)
            def safety_hook():
                try:
                    e = atoms.get_potential_energy()
                    f = atoms.get_forces()
                    max_f = np.sqrt((f ** 2).sum(axis=1).max())
                    
                    if e > 10.0 or max_f > 50.0:
                        raise StopIteration("Unstable")
                except StopIteration: raise
                except: pass

            optimizer.attach(safety_hook, interval=1)
            optimizer.run(fmax=fmax, steps=steps)
            
        except (StopIteration, RuntimeError) as e:
            # --- THE FIX ---
            # If it's our "StopIteration" or Python's "generator raised StopIteration",
            # it means the Kill Switch worked. treat it as "Not Converged", not a crash.
            err_str = str(e)
            if "StopIteration" in err_str or "Unstable" in err_str:
                 return {
                    "final_structure": structure, 
                    "energy_per_atom": 0.0, 
                    "converged": False, 
                    "error": None # No error, just early stop
                }
            # Real crash
            return {"final_structure": structure, "converged": False, "error": f"Crash: {e}"}

        except Exception as e:
            return {"final_structure": structure, "converged": False, "error": f"Crash: {e}"}

        # 4. Result Extraction (Only if it survived)
        try:
            forces = atoms.get_forces()
            max_force = np.sqrt((forces ** 2).sum(axis=1).max())
            converged = bool(max_force <= fmax)
            
            final_energy = atoms.get_potential_energy()
            num_atoms = len(atoms)
            energy_per_atom = float(final_energy / num_atoms)
        except: 
            converged = False
            energy_per_atom = 0.0

        # 5. Cleanup
        try:
            raw_struct = AseAtomsAdaptor.get_structure(atoms)
            clean_structure = Structure.from_dict(raw_struct.as_dict())
        except:
            clean_structure = structure

        del atoms, optimizer

        return {
            "final_structure": clean_structure, 
            "energy_per_atom": energy_per_atom,
            "converged": converged,
            "error": None
        }
