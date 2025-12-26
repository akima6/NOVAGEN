import warnings
import torch
import numpy as np
import matgl
from pymatgen.io.ase import AseAtomsAdaptor
from ase.filters import UnitCellFilter
from ase.optimize import LBFGS
from ase.geometry import get_distances
from matgl.ext.ase import M3GNetCalculator

# Filter harmless warnings
warnings.filterwarnings("ignore")

class CrystalRelaxer:
    """
    Hybrid Relaxer:
    Combines the Robustness of the Old Script (Explosion Guard + Cell Filter)
    with the Compatibility of the New Script (RL API).
    """
    def __init__(self, device="cpu"):
        self.device = device
        print("   [Relaxer] Initializing M3GNet Potential...")
        try:
            # Load the Physics Model
            self.pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
            self.calc = M3GNetCalculator(potential=self.pot)
            print("   [Relaxer] ✅ Physics Engine Loaded.")
        except Exception as e:
            print(f"   [Relaxer] ❌ Failed to load potential: {e}")
            self.pot = None
            self.calc = None

    def relax(self, structure, steps=100):
        """
        Relaxes a structure (Atoms + Lattice).
        Returns dict compatible with RL Trainer.
        """
        # 1. Check if model exists
        if self.calc is None:
            return self._fail(structure, "No Model")

        try:
            # 2. Convert to ASE (Physics Format)
            atoms = AseAtomsAdaptor.get_atoms(structure)
            
            # 3. SAFETY CHECK: EXPLOSION GUARD
            # If atoms are overlapping (< 0.5 Angstrom), reject immediately.
            # This saves huge amounts of memory and prevents crashes.
            try:
                # get_distances returns (dist_matrix, dist_vectors)
                # We want the matrix [0]
                dists = atoms.get_all_distances(mic=True)
                # Filter out self-distances (0.0)
                mask = dists > 0.01
                if mask.any():
                    min_dist = dists[mask].min()
                    if min_dist < 0.5:
                        return self._fail(structure, "Explosion Detected (Atoms too close)")
            except:
                pass # If check fails, risk running the physics anyway

            # 4. Attach Calculator
            atoms.calc = self.calc

            # 5. Setup Optimizer with UnitCellFilter
            # This allows the Box (Lattice) to change shape
            ucf = UnitCellFilter(atoms)
            
            # Use LBFGS (Memory Efficient Optimizer)
            # logfile=None prevents it from printing to console
            optimizer = LBFGS(ucf, logfile=None)
            
            # 6. Run Relaxation
            # fmax=0.1 is "good enough" for RL. 
            # 0.01 is for final paper publication.
            optimizer.run(fmax=0.1, steps=steps)

            # 7. Process Results
            final_structure = AseAtomsAdaptor.get_structure(atoms)
            final_energy = atoms.get_potential_energy() # Total eV
            num_atoms = len(atoms)
            
            # Cleanup to save memory
            del atoms, ucf, optimizer
            
            return {
                "final_structure": final_structure,
                "energy_per_atom": final_energy / num_atoms,
                "converged": True
            }

        except Exception as e:
            # If anything breaks, return failure
            return self._fail(structure, str(e))

    def _fail(self, structure, reason="Unknown"):
        """Helper to return a standardized failure object"""
        return {
            "final_structure": structure,
            "energy_per_atom": 5.0, # High penalty
            "converged": False
        }
