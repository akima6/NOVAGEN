import warnings
import numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer

# Suppress pymatgen warnings about 'electronegativity' missing for dummy atoms
warnings.filterwarnings("ignore")

class CrystalSentinel:
    """
    The Gatekeeper.
    Filters out structures that are physically impossible or geometrically broken
    BEFORE they waste compute time in the physics engine.
    """
    def __init__(self, 
                 min_distance=0.7, 
                 min_density=0.5, 
                 max_density=25.0, 
                 symmetry_tolerance=0.1):
        
        self.min_dist = min_distance       # Angstroms (Nuclear fusion check)
        self.min_rho = min_density         # g/cm3 (Gas check)
        self.max_rho = max_density         # g/cm3 (Black hole check)
        self.sym_tol = symmetry_tolerance  # Angstroms (Symmetry looseness)

    def is_valid(self, structure: Structure, requested_sg=None) -> tuple:
        """
        Runs all checks on a structure.
        Returns: (bool, str) -> (IsValid, Reason)
        """
        # 1. Volume/Density Check
        # Catch NaNs, Infs, or collapsed volumes
        if structure.volume < 1.0 or np.isnan(structure.volume):
             return False, "Collapsed Volume"
        
        rho = structure.density
        if rho < self.min_rho:
            return False, f"Too Fluffy ({rho:.2f} g/cm3)"
        if rho > self.max_rho:
            return False, f"Too Dense ({rho:.2f} g/cm3)"

        # 2. Geometry Sanity (Interatomic Distances)
        # We check the minimum distance in the periodic boundary conditions.
        # This is expensive for huge structures, but cheap for small ones.
        try:
            # Get distance matrix (with periodic images)
            # We only care if ANY distance is too small.
            # dist_matrix returns shape (N, N). Diagonal is 0.
            dists = structure.distance_matrix
            # Mask the diagonal (self-distance is always 0)
            np.fill_diagonal(dists, 10.0) 
            
            min_d = np.min(dists)
            if min_d < self.min_dist:
                return False, f"Atoms Overlapping (min_dist={min_d:.2f} A)"
        except Exception as e:
            return False, f"Geometry Check Failed: {str(e)}"

        # 3. Symmetry Check (Optional Strict Mode)
        # If we asked for Space Group 225, did we actually get it?
        if requested_sg is not None:
            try:
                sga = SpacegroupAnalyzer(structure, symprec=self.sym_tol)
                found_sg = sga.get_space_group_number()
                if found_sg != requested_sg:
                    # In research, we might keep it. In a strict product, we might flag it.
                    # For now, we return True but with a warning in the reason.
                    return True, f"Symmetry Mismatch (Asked {requested_sg}, Got {found_sg})"
            except:
                return False, "Symmetry Analysis Crashed"

        return True, "Valid"

    def batch_filter(self, structures, requested_sgs=None):
        """
        Process a list of structures.
        Returns: valid_list, rejected_stats
        """
        valid_structs = []
        stats = {"valid": 0, "overlapping": 0, "density": 0, "other": 0}
        
        for i, struct in enumerate(structures):
            req_sg = requested_sgs[i] if requested_sgs else None
            is_ok, reason = self.is_valid(struct, req_sg)
            
            if is_ok:
                valid_structs.append(struct)
                stats["valid"] += 1
            else:
                if "Overlapping" in reason: stats["overlapping"] += 1
                elif "Dense" in reason or "Fluffy" in reason: stats["density"] += 1
                else: stats["other"] += 1
                
        return valid_structs, stats

# --- TEST BLOCK ---
if __name__ == "__main__":
    from pymatgen.core import Lattice
    
    print("🛡️ Initializing Sentinel...")
    sentinel = CrystalSentinel()
    
    # Test 1: Good Structure (Iron)
    good_s = Structure(Lattice.cubic(2.8), ["Fe", "Fe"], [[0,0,0], [0.5, 0.5, 0.5]])
    ok, msg = sentinel.is_valid(good_s)
    print(f"Test 1 (Good): {ok} | {msg}")
    
    # Test 2: Bad Structure (Atoms on top of each other)
    bad_s = Structure(Lattice.cubic(3.0), ["Fe", "Fe"], [[0,0,0], [0.01, 0.01, 0.01]])
    ok, msg = sentinel.is_valid(bad_s)
    print(f"Test 2 (Overlapping): {ok} | {msg}")
    
    # Test 3: Bad Density (Hydrogen in a huge box)
    fluffy_s = Structure(Lattice.cubic(10.0), ["H"], [[0,0,0]])
    ok, msg = sentinel.is_valid(fluffy_s)
    print(f"Test 3 (Low Density): {ok} | {msg}")
