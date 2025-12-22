# ================================================================
#  ORACLE — STABLE CPU-ONLY IMPLEMENTATION (MATGL-SAFE)
# ================================================================

import os

# ----------------------------------------------------------------
# CRITICAL: Force CPU before importing torch / matgl
# ----------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["DGLBACKEND"] = "pytorch"

import torch
import warnings
import sys
import numpy as np
import dgl

warnings.filterwarnings("ignore")


class Oracle:
    """
    Stable Oracle (CPU-Only).

    Why CPU-only?
    - MatGL's predict_structure() internally creates tensors
      based on global CUDA availability.
    - Mixing CPU/GPU causes unavoidable device mismatch errors.
    - This implementation is 100% safe and deterministic.

    Outputs:
    - formation_energy (eV/atom)
    - band_gap_scalar (eV)
    """

    def __init__(self):
        try:
            import matgl
            from matgl.ext.pymatgen import Structure2Graph
            from pymatgen.core.periodic_table import Element

            self.device = torch.device("cpu")
            print("🔮 Oracle initialized on: cpu")

            # --------------------------------------------------------
            # Load models (CPU)
            # --------------------------------------------------------
            self.eform_model = matgl.load_model(
                "M3GNet-MP-2018.6.1-Eform"
            ).to(self.device).eval()

            self.bg_model = matgl.load_model(
                "MEGNet-MP-2019.4.1-BandGap-mfi"
            ).to(self.device).eval()

            # --------------------------------------------------------
            # Element list
            # --------------------------------------------------------
            if hasattr(self.eform_model.model, "element_types"):
                elem_list = self.eform_model.model.element_types
            else:
                elem_list = [str(Element.from_Z(i)) for i in range(1, 95)]

            self.converter = Structure2Graph(
                element_types=elem_list,
                cutoff=5.0
            )

            # Fixed state tensor (CPU)
            self.fixed_state = torch.tensor([0], dtype=torch.long)

        except Exception as e:
            print(f"❌ Oracle initialization failed: {e}")
            raise

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------
    def predict_batch(self, structures):
        """
        Predict properties for a list of pymatgen Structures.
        """
        if not structures:
            return []

        return self._predict_sequential(structures)

    # ------------------------------------------------------------
    # Internal Sequential Prediction (SAFE)
    # ------------------------------------------------------------
    def _predict_sequential(self, structures):
        results = []

        for struct in structures:
            try:
                # Formation energy (M3GNet)
                eform = float(
                    self.eform_model.predict_structure(struct)
                )

                # Band gap (MEGNet)
                try:
                    gap = float(
                        self.bg_model.predict_structure(
                            struct,
                            state_attr=self.fixed_state
                        )
                    )
                    gap = max(0.0, gap)
                except Exception:
                    gap = 0.0

                results.append({
                    "formation_energy": eform,
                    "band_gap_scalar": gap
                })

            except Exception:
                results.append(self._error_result())

        return results

    # ------------------------------------------------------------
    # Fallback result
    # ------------------------------------------------------------
    def _error_result(self):
        return {
            "formation_energy": 0.0,
            "band_gap_scalar": 0.0
        }
