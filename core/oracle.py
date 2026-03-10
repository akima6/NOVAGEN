import os
import torch
import warnings
import contextlib
import io

# Suppress warnings
os.environ["DGLBACKEND"] = "pytorch"
warnings.simplefilter("ignore")

class QuietBlock:
    def __enter__(self):
        self._suppress = io.StringIO()
        self._redirect_out = contextlib.redirect_stdout(self._suppress)
        self._redirect_err = contextlib.redirect_stderr(self._suppress)
        self._redirect_out.__enter__()
        self._redirect_err.__enter__()
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._redirect_out.__exit__(exc_type, exc_val, exc_tb)
        self._redirect_err.__exit__(exc_type, exc_val, exc_tb)

class Phase3Oracle:
    """
    Unified Phase 3 Oracle: Predicts Bandgap (MEGNet) ONLY.
    Runs strictly on CPU to protect GPU VRAM for the Generator and Relaxer.
    """
    def __init__(self):
        self.device = torch.device("cpu")
        print(f"🔮 Initializing Phase 3 Oracle on {self.device}...")

        try:
            # Load Electronic Model (MEGNet)
            import matgl
            import matgl.data.transformer as mtrans
            safe_globals_fn = getattr(torch.serialization, "add_safe_globals", None)
            if safe_globals_fn is not None:
                safe_globals_fn([mtrans.Normalizer])
            
            self.model_gap = matgl.load_model("MEGNet-MP-2019.4.1-BandGap-mfi")
            self.model_gap.to(self.device)
            self.fixed_state = torch.tensor([0], dtype=torch.long, device=self.device)

            print("   ✅ Oracle Online: Electronic engine loaded.")
        except Exception as e:
            raise RuntimeError(f"❌ Oracle Initialization Failed: {e}")

    def evaluate_structure(self, structure):
        import logging
        logger = logging.getLogger(__name__)
        
        if structure is None:
            logger.warning("Oracle: Structure is None")
            return -1.0  # Use -1.0 for failure signal

        with QuietBlock():
            try:
                gap = float(self.model_gap.predict_structure(structure, state_attr=self.fixed_state))
                gap = max(0.0, gap)
            except Exception as e:
                logger.warning(f"Oracle: Gap prediction failed: {e}")
                gap = -1.0  # ← Explicit failure signal

        return gap