import numpy as np
from pymatgen.core import Structure

class CrystalSentinel:
    """
    Fast geometric sanity checker.

    UPDATED STRATEGY: THREE ZONES
    1. Singular (< 0.5): HARD REJECT (Protect the GPU)
    2. Soft Overlap (0.5 - 1.0): PASS (Let Relaxer fix it)
    3. Clean (> 1.0): PASS
    """

    def __init__(self, device="cpu"):
        self.device = device

        # Covalent radii (Å)
        self.radii = {
            1: 0.31,
            3: 1.28, 4: 0.96, 5: 0.84, 6: 0.73, 7: 0.71, 8: 0.66, 9: 0.57,
            11: 1.66, 12: 1.41, 13: 1.21, 14: 1.11, 15: 1.07, 16: 1.05, 17: 1.02,
            19: 2.03, 20: 1.76,
            21: 1.44,
            22: 1.60, 23: 1.53, 24: 1.39, 25: 1.39, 26: 1.32,
            27: 1.26, 28: 1.24, 29: 1.32, 30: 1.22, 31: 1.22, 32: 1.20, 33: 1.19,
            34: 1.20, 35: 1.20, 38: 1.95, 42: 1.90, 48: 1.44, 49: 1.42, 50: 1.39,
            51: 1.39, 52: 1.38, 53: 1.39, 56: 2.15, 74: 1.62
        }

        self.MIN_DENSITY = 0.5
        self.MAX_DENSITY = 20.0
        self.OVERLAP_FACTOR = 0.5

    def filter(self, structures):
        results = []

        for struct in structures:
            result = {
                "valid": False,
                "failure_type": "null_structure",
                "min_distance": 0.0,
                "min_distance_ratio": 0.0,
                "density": 0.0,
                "volume_per_atom": 0.0,   # 🔹 NEW (required by reward)
            }

            if struct is None:
                results.append(result)
                continue

            try:
                density = struct.density
                result["density"] = density

                # 🔹 NEW: explicit volume-per-atom
                result["volume_per_atom"] = struct.volume / len(struct)

                if density < self.MIN_DENSITY or density > self.MAX_DENSITY:
                    result["failure_type"] = "density_violation"
                    results.append(result)
                    continue

                # ---------- Overlap Check ----------
                min_dist, min_ratio = self._compute_min_distance_ratio(struct)
                result["min_distance"] = min_dist
                result["min_distance_ratio"] = min_ratio

                # ZONE 1: SINGULARITY
                if min_ratio < 0.5:
                    result["valid"] = False
                    result["failure_type"] = "overlap_singular"
                    results.append(result)
                    continue

                # ZONE 2: SOFT OVERLAP
                if min_ratio < 1.0:
                    result["valid"] = True
                    result["failure_type"] = "overlap_soft"
                    results.append(result)
                    continue

                # ZONE 3: CLEAN
                result["valid"] = True
                result["failure_type"] = None
                results.append(result)

            except Exception:
                result["failure_type"] = "sentinel_exception"
                results.append(result)

        return results

    def _compute_min_distance_ratio(self, struct):
        atomic_numbers = [site.specie.Z for site in struct]
        dists = struct.distance_matrix.copy()
        np.fill_diagonal(dists, np.inf)

        min_ratio = np.inf
        min_dist = np.inf

        n = len(struct)
        for i in range(n):
            r1 = self.radii.get(atomic_numbers[i], 1.3)
            for j in range(i + 1, n):
                r2 = self.radii.get(atomic_numbers[j], 1.3)
                allowed = self.OVERLAP_FACTOR * (r1 + r2)
                actual = dists[i, j]
                ratio = actual / allowed
                if ratio < min_ratio:
                    min_ratio = ratio
                    min_dist = actual

        if not np.isfinite(min_ratio):
            return 0.0, 0.0

        return float(min_dist), float(min_ratio)
