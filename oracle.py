import torch
import warnings
import sys

warnings.filterwarnings("ignore")

class Oracle:
    """
    High-fidelity, RL-safe property oracle for NOVAGEN.

    Uses true property models:
    - Formation energy: M3GNet-MP-2018.6.1-Eform
    - Band gap: MEGNet-MP-2019.4.1-BandGap-mfi

    Design principle:
    - Raw predictions → RL reward
    - Calibrated ranges → human reporting only
    """

    def __init__(self, device="cpu"):
        try:
            import matgl

            self.eform_model = matgl.load_model(
                "M3GNet-MP-2018.6.1-Eform"
            )

            self.bg_model = matgl.load_model(
                "MEGNet-MP-2019.4.1-BandGap-mfi"
            )

            # MEGNet requires Long state attributes
            self.fixed_state = torch.tensor([0], dtype=torch.long)
            self.device = device

        except Exception as e:
            print(f"Oracle initialization failed: {e}")
            sys.exit(1)

    def _calibrate_gap(self, raw_gap: float):
        """
        Calibration used ONLY for reporting (not RL).
        """
        if raw_gap < 0.1:
            return 0.0, "Metal (0.0 eV)"

        ceiling = 1.35 * raw_gap + 0.4
        return ceiling, f"{raw_gap:.2f} – {ceiling:.2f} eV"

    def predict_properties(self, structures):
        """
        Returns a list of dictionaries with:
        - formation_energy  (RL-critical)
        - band_gap_raw      (RL-critical)
        - band_gap_display  (human-facing)
        """
        results = []

        for struct in structures:
            try:
                # --- Formation energy (true property) ---
                eform = float(self.eform_model.predict_structure(struct))

                # --- Band gap (raw, conservative) ---
                raw_gap = float(
                    self.bg_model.predict_structure(
                        struct,
                        state_attr=self.fixed_state
                    )
                )
                raw_gap = max(0.0, raw_gap)

                # --- Reporting-only calibration ---
                ceiling_gap, display_str = self._calibrate_gap(raw_gap)

                results.append({
                    "formation_energy": eform,
                    "band_gap_raw": raw_gap,          # RL reward signal
                    "band_gap_display": display_str,  # user-facing
                    "band_gap_ceiling": ceiling_gap,  # reporting only
                    "error": None
                })

            except Exception as e:
                results.append({
                    "formation_energy": None,
                    "band_gap_raw": 0.0,
                    "band_gap_display": "Error",
                    "band_gap_ceiling": 0.0,
                    "error": str(e)
                })

        return results
