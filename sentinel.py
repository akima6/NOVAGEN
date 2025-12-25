import warnings
import numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer

# Suppress warnings
warnings.filterwarnings("ignore")

class CrystalSentinel:
    """
    The Gatekeeper.
    Filters out structures that are physically impossible or geometrically broken.
    Now includes 'filter' method for RL training loops.
    """
    def __init__(self, 
                 min_distance=0.7, 
                 min_density=0.5, 
                 max_density=25.0, 
                 symmetry_tolerance=0.1,
                 device=None): # Added device=None to be robust if passed
        
        self.min_dist = min_distance       
        self.min_rho = min_density         
        self.max_rho = max_density         
        self.sym_tol = symmetry_tolerance  

    def is_valid(self, structure: Structure) -> tuple:
        """
        Single structure check.
        Returns: (bool, Reason)
        """
        # 0. Check for None (Failed reconstruction)
        if structure is None:
            return False, "Construction Failed"

        # 1. Volume/Density Check
        try:
            if structure.volume < 1.0 or np.isnan(structure.volume):
                 return False, "Collapsed Volume"
            
            rho = structure.density
            if rho < self.min_rho:
                return False, f"Too Fluffy ({rho:.2f} g/cm3)"
            if rho > self.max_rho:
                return False, f"Too Dense ({rho:.2f} g/cm3)"
        except:
             return False, "Density Calculation Error"

        # 2. Geometry Sanity (Interatomic Distances)
        try:
            # We calculate distance matrix. 
            # If any non-diagonal element is < min_dist, it's an overlap.
            dists = structure.distance_matrix
            # Fill diagonal with safe large number to ignore self-distance
            np.fill_diagonal(dists, 10.0) 
            
            min_d = np.min(dists)
            if min_d < self.min_dist:
                return False, f"Atoms Overlapping (min_dist={min_d:.2f} A)"
        except Exception as e:
            return False, f"Geometry Check Failed: {str(e)}"

        return True, "Valid"

    def filter(self, structures):
        """
        RL TRAINER METHOD
        Input: List of Pymatgen Structures (some might be None)
        Output: 
            - mask: List[bool] (True if valid)
            - valid_structures: List[Structure] (Only the survivors)
        """
        mask = []
        valid_structures = []
        
        for struct in structures:
            is_ok, reason = self.is_valid(struct)
            
            mask.append(is_ok)
            if is_ok:
                valid_structures.append(struct)
                
        return mask, valid_structures

# --- TEST BLOCK ---
if __name__ == "__main__":
    from pymatgen.core import Lattice
    print("🛡️ Initializing Sentinel...")
    sentinel = CrystalSentinel()
    
    # Test
    good_s = Structure(Lattice.cubic(2.8), ["Fe", "Fe"], [[0,0,0], [0.5, 0.5, 0.5]])
    mask, valids = sentinel.filter([good_s, None])
    print(f"Filter Test: Mask={mask} (Expected [True, False])")
    print(f"Survivors: {len(valids)}")
