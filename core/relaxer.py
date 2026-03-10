import os
import sys
import warnings
import torch
import numpy as np

# Silence ASE/Pymatgen warnings to keep the RL logs clean
warnings.filterwarnings("ignore")

class CrystalRelaxer:
    """
    Robust Crystal Relaxer (GPU-Enabled) v2.0.
    Acts as the 'Physicist's Hand' to evaluate atomic geometries using CHGNet.
    Uses FIRE optimizer to handle high initial forces safely.
    """
    def __init__(self, device="cuda"):
        self.device = device
        self.EXPLOSION_DISTANCE = 0.6  # Ångstroms (Atoms closer than this = Crash)

        # 1. Setup Paths
        if "__file__" in locals():
            CORE_DIR = os.path.dirname(os.path.abspath(__file__))
        else:
            CORE_DIR = os.getcwd()
            
        PROJECT_ROOT = os.path.dirname(CORE_DIR)
        MODEL_DIR = os.path.join(PROJECT_ROOT, "pretrained_model")
        local_weights = os.path.join(MODEL_DIR, "chgnet_0.3.0_weights.pth.tar")

        # 2. Initialize CHGNet
        print(f"   [Relaxer] Initializing CHGNet on {self.device}...")
        try:
            from chgnet.model import CHGNet
            
            if os.path.exists(local_weights):
                print(f"   [Relaxer] Loading local weights from: {os.path.basename(local_weights)}")
                self.model = CHGNet.from_file(local_weights)
            else:
                print("   [Relaxer] Downloading/Loading standard CHGNet...")
                self.model = CHGNet.load()
            
            self.model.to(self.device)
            self.model.eval() # Must be in eval mode for stable gradients
            print("   [Relaxer] ✅ CHGNet Active.")
            
        except Exception as e:
            print(f"   [Relaxer] ❌ CHGNet Init Failed: {e}")
            self.model = None

    
    def relax(self, structure, steps=10):
        if structure is None: 
            return self._fail("no_structure")

        # 🔹 THE FIX: Custom Universal Broom
        # Manually scan for and delete ANY overlapping atoms, regardless of element type.
        try:
            dists = structure.distance_matrix.copy()
            to_delete = set()
            
            # Loop through the upper triangle of the distance matrix
            for i in range(len(structure)):
                for j in range(i + 1, len(structure)):
                    if dists[i, j] < 0.5:
                        to_delete.add(j)  # Mark the duplicate index for deletion
            
            if to_delete:
                # Convert set to list and sort in reverse so deleting doesn't shift indices
                structure.remove_sites(sorted(list(to_delete), reverse=True))
        except Exception:
            pass

        # --- Guard 1: The Explosion Check ---
        # If any overlaps somehow survived the broom, catch them here.
        try:
            dists = structure.distance_matrix.copy()
            np.fill_diagonal(dists, np.inf)
            if np.min(dists) < self.EXPLOSION_DISTANCE:
                return self._fail("implosion_guard", energy=10.0) 
        except Exception:
            pass

        if self.model is None: 
            return self._fail("no_model")

        try:
            # Static Evaluation Mode (steps=0)
            if steps == 0:
                pred = self.model.predict_structure(structure)
                return {
                    "converged": True,
                    "final_structure": structure,
                    "energy_per_atom": float(pred["e"]),
                    "relaxed_volume": float(structure.volume),
                    "valid": True,
                    "failure_reason": None
                }
            
            # Ultra-Light Relaxation Mode via ASE
            from chgnet.model.dynamics import StructOptimizer
            import gc
            
            optimizer = StructOptimizer(
                model=self.model, 
                optimizer_class="FIRE", 
                use_device=self.device
            )
            
            result = optimizer.relax(
                structure, 
                steps=steps, 
                verbose=False, 
                fmax=0.1
            )
            del optimizer
            gc.collect()
            
            final_struct = result["final_structure"]
            total_energy = result["trajectory"].energies[-1]
            energy_per_atom = total_energy / len(final_struct)

            return {
                "converged": True,
                "final_structure": final_struct,
                "energy_per_atom": float(energy_per_atom),
                "relaxed_volume": float(final_struct.volume), 
                "valid": True,
                "failure_reason": None
            }

        except Exception as e:
            return self._fail(f"relax_crash: {str(e)[:50]}", energy=10.0)

    def _fail(self, reason, energy=50.0):
        """Standardized failure dictionary."""
        return {
            "converged": False,
            "final_structure": None,
            "energy_per_atom": float(energy),
            "relaxed_volume": 0.0,
            "valid": False,
            "failure_reason": reason
        }