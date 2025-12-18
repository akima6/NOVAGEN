# relaxer.py
# (The Physics Engine - Robust Version)

import os
import numpy as np
import torch
import warnings

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
    Relaxer module using MatGL 1.3.0.
    Includes safety checks for atomic overlap.
    """

    def __init__(self):
        # Only print on first load
        try:
            self.potential = matgl.load_model(
                "M3GNet-MP-2021.2.8-PES"
            ).to("cuda")
            
            self.calculator = M3GNetCalculator(
                potential=self.potential,
                device="cuda"
            )

        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

    def relax(self, structure: Structure, fmax=0.01, steps=100) -> dict:
        """
        Relaxes a pymatgen Structure. 
        Returns dictionary with final structure, energy, and convergence status.
        """
        # 1. Convert to ASE
        try:
            atoms = AseAtomsAdaptor.get_atoms(structure)
        except Exception as e:
            return {"is_converged": False, "error": f"ASE conversion failed: {e}"}

        # --- SAFETY CHECK ---
        # If atoms are closer than 0.5 Angstrom, reject immediately.
        # This prevents "explosions" and math errors.
        try:
            # Get distance matrix, exclude self-interaction (0.0)
            dists = get_distances(atoms.get_positions(), cell=atoms.get_cell(), pbc=atoms.get_pbc())
            flat_dists = dists[1].flatten()
            flat_dists = flat_dists[flat_dists > 1e-4] 
            
            if len(flat_dists) > 0 and flat_dists.min() < 0.5:
                return {
                    "final_structure": structure,
                    "final_energy_per_atom": None,
                    "is_converged": False,
                    "error": "Sanity Check Failed: Atoms are too close (<0.5 A)"
                }
        except:
            pass # If check fails for some reason, proceed anyway
        # --------------------

        # 2. Attach Calculator
        atoms.calc = self.calculator

        # 3. Run Optimization
        try:
            # UnitCellFilter allows the box shape/volume to relax
            ucf = UnitCellFilter(atoms)
            # logfile=None keeps your terminal clean during training
            optimizer = LBFGS(ucf, logfile=None)
            optimizer.run(fmax=fmax, steps=steps)
        except Exception as e:
            return {
                "final_structure": structure,
                "final_energy_per_atom": None,
                "is_converged": False,
                "error": f"Optimization crashed: {e}"
            }

        # 4. Get Results
        final_structure = AseAtomsAdaptor.get_structure(atoms)
        
        try:
            final_energy = atoms.get_potential_energy()
            num_atoms = len(atoms)
            energy_per_atom = final_energy / num_atoms
            
            # Check forces to confirm actual convergence
            forces = atoms.get_forces()
            max_force = np.sqrt((forces ** 2).sum(axis=1).max())
            converged = max_force <= fmax
        except:
            energy_per_atom = None
            converged = False

        return {
            "final_structure": final_structure,
            "final_energy_per_atom": energy_per_atom,
            "is_converged": converged,
            "num_steps": optimizer.get_number_of_steps()
        }
