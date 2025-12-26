import numpy as np
import torch
from pymatgen.core import Structure

class CrystalSentinel:
    def __init__(self, device="cpu"):
        self.device = device
        
        # Standard Covalent Radii (Angstroms) for common elements
        # Used to detect unphysical overlaps (clumps)
        self.radii = {
            1: 0.31,  # H
            3: 1.28,  # Li
            4: 0.96,  # Be
            5: 0.84,  # B
            6: 0.73,  # C
            7: 0.71,  # N
            8: 0.66,  # O
            9: 0.57,  # F
            11: 1.66, # Na
            12: 1.41, # Mg
            13: 1.21, # Al
            14: 1.11, # Si
            15: 1.07, # P
            16: 1.05, # S
            17: 1.02, # Cl
            19: 2.03, # K
            20: 1.76, # Ca
            26: 1.16, # Fe
            # Default fallback for others: 1.0
        }

    def filter(self, structures):
        """
        Returns a boolean mask: True = Valid, False = Garbage.
        """
        valid_mask = []
        
        for struct in structures:
            if struct is None:
                valid_mask.append(False)
                continue
                
            try:
                # 1. Physics Check: Bond overlap
                if not self._check_overlaps(struct):
                    valid_mask.append(False)
                    continue

                # 2. Density Check (Optional but good)
                # Reject if density is impossibly low (< 0.5 g/cm3) or high (> 20 g/cm3)
                if struct.density < 0.5 or struct.density > 20.0:
                    valid_mask.append(False)
                    continue
                    
                valid_mask.append(True)
                
            except Exception:
                valid_mask.append(False)
                
        return valid_mask, None

    def _check_overlaps(self, struct):
        """
        Rejects structure if any two atoms are too close relative to their radii.
        Threshold: Distance < 0.6 * (Radius A + Radius B)
        """
        # Get atomic numbers
        atomic_numbers = [site.specie.Z for site in struct]
        coords = struct.cart_coords
        
        # Get all distances (matrix)
        dists = struct.distance_matrix
        np.fill_diagonal(dists, 10.0) # Ignore distance to self

        # Check every pair (This is fast enough for small crystals)
        # Note: For huge batches, we would vectorize this, but for <50 atoms loops are fine.
        n = len(struct)
        for i in range(n):
            r1 = self.radii.get(atomic_numbers[i], 1.1) # Default 1.1 if unknown
            for j in range(i + 1, n):
                r2 = self.radii.get(atomic_numbers[j], 1.1)
                
                # Dynamic Threshold
                limit = 0.6 * (r1 + r2)
                
                if dists[i, j] < limit:
                    return False # Atoms colliding
                    
        return True
