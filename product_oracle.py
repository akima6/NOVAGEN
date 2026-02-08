import warnings
import torch
import os
import numpy as np

# Suppress DGL/MatGL warnings
os.environ["DGLBACKEND"] = "pytorch"
warnings.simplefilter("ignore")

class CrystalOracle:
    """
    Oracle: Band Gap Predictor (MEGNet) ONLY.
    *Optimized for Low RAM (Removed unused Formation Energy model)*
    """

    def __init__(self, device="cpu"):
        # We stick to CPU for the Oracle to leave GPU VRAM for the Generator/Relaxer
        self.device = torch.device("cpu")  
        print(f"🔮 Initializing Oracle on {self.device} (Lite Mode)...")

        try:
            import matgl
            import matgl.data.transformer as mtrans

            # -----------------------------
            # Band Gap Model (MEGNet) - The only one we need
            # -----------------------------
            print("   [1/1] Loading Band Gap Model (MEGNet)...")
            
            # Register safe globals if function exists (new PyTorch safety)
            safe_globals_fn = getattr(torch.serialization, "add_safe_globals", None)
            if safe_globals_fn is not None:
                safe_globals_fn([mtrans.Normalizer])

            # Load MEGNet for Bandgap
            self.model_gap = matgl.load_model("MEGNet-MP-2019.4.1-BandGap-mfi")
            self.model_gap.to(self.device)

            # Precompute fixed state for MEGNet to save time
            self.fixed_state = torch.tensor([0], dtype=torch.long, device=self.device)

            print("   ✅ Oracle System Online (Bandgap Only).")

        except Exception as e:
            print(f"   ❌ Oracle Initialization Failed: {e}")
            raise e

    def predict_batch(self, structures):
        """
        Input: list of Pymatgen Structures
        Output: (Dummy Energy, List of Band Gaps)
        """
        # We return 0.0 for energy because the Reward Engine uses the Relaxer's energy anyway.
        # This saves us from loading the heavy M3GNet model.
        e_forms = [0.0] * len(structures)
        band_gaps = []

        for s in structures:
            if s is None:
                band_gaps.append(0.0)
                continue

            # Band Gap (MEGNet)
            try:
                g_val = self.model_gap.predict_structure(s, state_attr=self.fixed_state)
                g_val = max(0.0, float(g_val))
            except Exception:
                g_val = 0.0

            band_gaps.append(g_val)

        return e_forms, band_gaps