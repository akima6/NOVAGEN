import os
import torch
import warnings
import numpy as np
import matgl
from matgl.ext.ase import M3GNetCalculator

# --- THREAD LOCKDOWN ---
# Keeps the CPU focused on one task at a time to prevent stalling
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
    """
    Production-Grade Relaxer with 'Safety Hooks' and 'Deep Clean'.
    Combines M3GNet accuracy with heuristic safeguards for speed.
    """
    def __init__(self, device="cpu"):
        # We ignore the 'device' arg and force CPU for stability,
        # mirroring the logic in your uploaded prototype.
        print(f"🔧 Initializing Relaxer (Optimized CPU Mode)...")
        
        try:
            # Load M3GNet on CPU to save VRAM for the Generator
            self.potential = matgl.load_model("M3GNet-MP-2021.2.8-PES")
            self.calculator = M3GNetCalculator(potential=self.potential)
            print("   ✅ M3GNet Model Loaded.")
        except Exception as e:
            print(f"   ❌ Failed to load M3GNet: {e}")
            raise e

    def relax(self, structure: Structure, fmax=0.05, steps=100):
        # NOTE: Reduced steps from 200 to 100 (Prototype Logic)
        
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
            
            # --- THE SAFETY HOOK ---
            # This is the secret sauce for speed. 
            # If a crystal is bad, kill it early.
            def safety_hook():
                try:
                    e = atoms.get_potential_energy()
                    f = atoms.get_forces()
                    max_f = np.sqrt((f ** 2).sum(axis=1).max())
                    
                    # Heuristics from your prototype:
                    if e > 10.0: # If energy is massive positive, it's exploding
                        raise StopIteration("Energy Diverged")
                    if max_f > 50.0: # If forces are huge, atoms are crashing
                        raise StopIteration("Forces Exploded")
                        
                except StopIteration: raise
                except: pass

            optimizer.attach(safety_hook, interval=1)
            optimizer.run(fmax=fmax, steps=steps)
            
        except StopIteration:
             pass # Logic captured the explosion, we stop gracefully
        except Exception as e:
            return {"final_structure": structure, "converged": False, "error": f"Crash: {e}"}

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
            energy_per_atom = 0.0

        # 5. DEEP CLEAN SANITIZATION
        # Strips all PyTorch graphs to prevent memory leaks
        try:
            raw_struct = AseAtomsAdaptor.get_structure(atoms)
            clean_dict = raw_struct.as_dict()
            clean_structure = Structure.from_dict(clean_dict)
        except:
            clean_structure = structure

        # 6. Garbage Collection
        del atoms
        del optimizer

        return {
            "final_structure": clean_structure, 
            "energy_per_atom": energy_per_atom,
            "converged": converged,
            "error": None
        }
