import torch
import warnings
import sys
import os
import numpy as np
import dgl

# Force DGL backend
os.environ["DGLBACKEND"] = "pytorch"
warnings.filterwarnings("ignore")


class Oracle:
    """
    Smart Oracle with Auto-Fallback.
    Attempts Batched Inference (Fast).
    If incompatible, falls back to Sequential (Safe).
    """

    def __init__(self, device="cuda"):
        try:
            import matgl
            from matgl.ext.pymatgen import Structure2Graph
            from pymatgen.core import Structure, Lattice
            from pymatgen.core.periodic_table import Element

            self.device = torch.device(device if torch.cuda.is_available() else "cpu")
            print(f"🔮 Oracle initialized on: {self.device}")

            # Load models
            self.eform_model = matgl.load_model("M3GNet-MP-2018.6.1-Eform").to(self.device).eval()
            self.bg_model = matgl.load_model("MEGNet-MP-2019.4.1-BandGap-mfi").to(self.device).eval()

            # Element types
            if hasattr(self.eform_model.model, "element_types"):
                elem_list = self.eform_model.model.element_types
            else:
                elem_list = [str(Element.from_Z(i)) for i in range(1, 95)]

            self.converter = Structure2Graph(element_types=elem_list, cutoff=5.0)
            self.fixed_state = torch.tensor([0], dtype=torch.long, device=self.device)

            # ---- Batching self-test ----
            print("🧪 Running Batching Self-Test...")
            self.use_batching = False
            try:
                dummy = Structure(Lattice.cubic(3.0), ["Si"], [[0, 0, 0]])
                self._predict_batch_impl([dummy, dummy])
                self.use_batching = True
                print("✅ Batching enabled")
            except Exception as e:
                print(f"⚠️ Batching disabled ({e})")

        except Exception as e:
            print(f"❌ Oracle initialization failed: {e}")
            raise

    def predict_batch(self, structures):
        if not structures:
            return []

        if self.use_batching:
            try:
                return self._predict_batch_impl(structures)
            except Exception:
                return self._predict_sequential_impl(structures)
        else:
            return self._predict_sequential_impl(structures)

    def _predict_batch_impl(self, structures):
        graphs, states, valid_idx = [], [], []
        results = [None] * len(structures)

        for i, s in enumerate(structures):
            try:
                g, state, _ = self.converter.get_graph(s)
                graphs.append(g)
                states.append(state)
                valid_idx.append(i)
            except Exception:
                results[i] = self._error_result()

        if not graphs:
            return results

        batched_graph = dgl.batch(graphs).to(self.device)
        batched_state = torch.stack(states).to(self.device)

        with torch.no_grad():
            e_vals = self.eform_model(batched_graph, batched_state).cpu().numpy().flatten()

        ptr = 0
        for idx in valid_idx:
            eform = float(e_vals[ptr])
            try:
                gap = float(self.bg_model.predict_structure(
                    structures[idx], state_attr=self.fixed_state.cpu()
                ))
                gap = max(0.0, gap)
            except Exception:
                gap = 0.0

            results[idx] = {
                "formation_energy": eform,
                "band_gap_scalar": gap,
            }
            ptr += 1

        for i in range(len(results)):
            if results[i] is None:
                results[i] = self._error_result()

        return results

    def _predict_sequential_impl(self, structures):
        results = []
        for struct in structures:
            try:
                eform = float(self.eform_model.predict_structure(struct))
                try:
                    gap = float(self.bg_model.predict_structure(
                        struct, state_attr=self.fixed_state.cpu()
                    ))
                    gap = max(0.0, gap)
                except Exception:
                    gap = 0.0

                results.append({
                    "formation_energy": eform,
                    "band_gap_scalar": gap
                })

            except Exception as e:
                print(f"❌ Sequential Prediction Error: {e}")
                results.append(self._error_result())

        return results

    def _error_result(self):
        return {"formation_energy": 0.0, "band_gap_scalar": 0.0}
