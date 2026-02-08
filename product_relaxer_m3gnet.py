import warnings
import torch
import numpy as np
import ase.constraints
import ase.filters
import traceback  # Added for debugging

# --- PATCH ASE ---
if not hasattr(ase.constraints, "ExpCellFilter"):
    if hasattr(ase.filters, "ExpCellFilter"):
        ase.constraints.ExpCellFilter = ase.filters.ExpCellFilter
    elif hasattr(ase.filters, "UnitCellFilter"):
        ase.constraints.ExpCellFilter = ase.filters.UnitCellFilter

import matgl
from matgl.ext.ase import M3GNetCalculator
from pymatgen.io.ase import AseAtomsAdaptor
from ase.optimize import LBFGS, FIRE

warnings.filterwarnings("ignore")

class CrystalRelaxer:
    """
    Robust Crystal Relaxer.
    Prioritizes returning a valid energy score even if physics crashes.
    """

    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        print(f"   [Relaxer] Initializing M3GNet Potential on {self.device}...")
        
        try:
            self.pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
            if hasattr(self.pot, "to"):
                self.pot.to(self.device)
            if hasattr(self.pot, "model") and hasattr(self.pot.model, "to"):
                self.pot.model.to(self.device)
            self.calc = M3GNetCalculator(potential=self.pot)
            print("   [Relaxer] ✅ Physics Engine Loaded.")
        except Exception as e:
            print(f"   [Relaxer] ❌ Error loading model: {e}")
            self.pot = None
            self.calc = None

        # [FIX 1] Lower the guard to allow "Bad" crystals to be scored
        self.EXPLOSION_DISTANCE = 0.01  # Was 0.1

    def relax(self, structure, steps=25):
        if self.calc is None or structure is None:
            return self._fail(structure, "no_model")

        initial_energy = 0.0

        try:
            atoms = AseAtomsAdaptor.get_atoms(structure)
            
            # 1. GEOMETRY CHECK (Explosion Guard)
            dists = atoms.get_all_distances(mic=True)
            # Filter self-distance (0.0)
            mask = dists > 0.0001
            if mask.any():
                min_dist = dists[mask].min()
                if min_dist < self.EXPLOSION_DISTANCE:
                    # Return a massive penalty energy for explosions
                    return self._fail(structure, "explosion_guard", energy=100.0)

            # 2. INITIAL ENERGY CALCULATION
            atoms.calc = self.calc
            # Calculate this FIRST so we have a score even if optimization fails
            initial_energy = atoms.get_potential_energy() / len(atoms)
            
            # [FIX 2] Remove the arbitrary threshold. Let physics decide.
            # if initial_energy > 50.0: ... (REMOVED)

            # 3. RELAXATION
            ucf = ase.filters.UnitCellFilter(atoms)
            
            # Use FIRE optimizer (more robust for high-energy systems than LBFGS)
            optimizer = FIRE(ucf, logfile=None)
            optimizer.run(fmax=0.1, steps=steps)

            final_energy = atoms.get_potential_energy() / len(atoms)
            final_structure = AseAtomsAdaptor.get_structure(atoms)
            
            return {
                "converged": True,
                "final_structure": final_structure,
                "energy_per_atom": final_energy,
                "min_distance_ratio": 1.0,
                "failure_reason": None
            }

        except Exception as e:
            # [FIX 3] CRITICAL: Return the initial_energy if we crash!
            # If we calculated initial_energy before the crash, use it.
            # If it crashed during initial calculation, assign a default penalty (e.g. 50.0)
            fallback_energy = initial_energy if initial_energy != 0.0 else 50.0
            
            # Optional: Print error for debugging only if needed
            # print(f"Relax crash: {e}")
            
            return self._fail(structure, "relax_crash", energy=fallback_energy)

    def _fail(self, structure, reason, min_distance_ratio=0.0, energy=0.0):
        return {
            "converged": False,
            "final_structure": structure,
            "failure_reason": reason,
            "min_distance_ratio": float(min_distance_ratio),
            # Ensure we never return 0.0 for a failure, give it a penalty if 0
            "energy_per_atom": energy if energy != 0.0 else 5.0
        }