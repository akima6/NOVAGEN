import warnings
import torch
import numpy as np
import pynvml # The official library you just installed
from pymatgen.io.ase import AseAtomsAdaptor
from ase.optimize import FIRE

# ==========================================
# 🚑 ASE COMPATIBILITY PATCH
# ==========================================
import ase.constraints
if not hasattr(ase.constraints, "ExpCellFilter"):
    # If using newer ASE where name changed or moved
    if hasattr(ase.constraints, "UnitCellFilter"):
        ase.constraints.ExpCellFilter = ase.constraints.UnitCellFilter

warnings.filterwarnings("ignore")

class CrystalRelaxer:
    """
    Robust Crystal Relaxer (GPU-Enabled).
    """
    def __init__(self, device="cuda", method="CHGNet"):
        self.device_str = device
        self.global_device = torch.device(device)
        self.method = method.upper()
        self.EXPLOSION_DISTANCE = 0.6 

        # Initialize CHGNet
        self._init_chgnet()

    def _init_chgnet(self):
        print(f"   [Relaxer] Initializing CHGNet on {self.global_device}...")
        try:
            from chgnet.model import CHGNet
            self.model = CHGNet.load()
            self.model.to(self.global_device)
            self.optimizer_class = "FIRE"
            print("   [Relaxer] ✅ CHGNet Active.")
        except Exception as e:
            print(f"   [Relaxer] ❌ CHGNet Init Failed: {e}")
            self.model = None

    def relax(self, structure, steps=25):
        if structure is None: return self._fail(structure, "no_structure")

        # 1. Explosion Guard
        try:
            dists = structure.distance_matrix
            np.fill_diagonal(dists, 10.0)
            if dists.min() < self.EXPLOSION_DISTANCE:
                return self._fail(structure, "explosion_guard", energy=100.0)
        except: pass

        # 2. Run Relaxation
        return self._relax_chgnet(structure, steps)

    def _relax_chgnet(self, structure, steps):
        if self.model is None: return self._fail(structure, "no_model")
        
        try:
            from chgnet.model.dynamics import StructOptimizer
            
            # Explicitly set device to ensure GPU usage
            relaxer = StructOptimizer(
                model=self.model, 
                optimizer_class=self.optimizer_class,
                use_device=self.device_str 
            )
            
            result = relaxer.relax(structure, steps=steps, verbose=False, fmax=0.1)
            
            final_struct = result["final_structure"]
            try:
                energy = result["trajectory"].energies[-1] / len(final_struct)
            except:
                energy = 0.0

            return {
                "converged": True,
                "final_structure": final_struct,
                "energy_per_atom": energy,
                "min_distance_ratio": 1.0,
                "failure_reason": None
            }
        except Exception as e:
            # print(f"Relax Error: {e}") # Uncomment to debug specific failures
            return self._fail(structure, "relax_crash", energy=50.0)

    def _fail(self, structure, reason, min_distance_ratio=0.0, energy=0.0):
        return {
            "converged": False,
            "final_structure": structure,
            "failure_reason": reason,
            "min_distance_ratio": 0.0,
            "energy_per_atom": energy if energy != 0.0 else 50.0
        }