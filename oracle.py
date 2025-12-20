import torch
import warnings
import sys
import os
import dgl
import numpy as np

# Force DGL backend
os.environ["DGLBACKEND"] = "pytorch"
warnings.filterwarnings("ignore")


class Oracle:
    """
    High-Performance, RL-Safe Batched Oracle.
    - Formation energy: batched on GPU (DGL + M3GNet)
    - Band gap: sequential (cheap, stable)
    """

    def __init__(self, device="cuda"):
        try:
            import matgl
            from matgl.ext.pymatgen import Structure2Graph
            from pymatgen.core.periodic_table import Element

            self.device = torch.device(device if torch.cuda.is_available() else "cpu")
            print(f"🔮 Oracle initialized on: {self.device}")

            # Load pretrained models
            self.eform_model = matgl.load_model("M3GNet-MP-2018.6.1-Eform")
            self.bg_model = matgl.load_model("MEGNet-MP-2019.4.1-BandGap-mfi")

            # Move models to device
            self.eform_model.to(self.device).eval()
            self.bg_model.to(self.device).eval()

            # Robust element list (MP elements Z=1..94)
            element_types = [str(Element.from_Z(i)) for i in range(1, 95)]

            # Structure → graph converter
            self.converter = Structure2Graph(
                element_types=element_types,
                cutoff=5.0
            )

            # MEGNet requires a state attribute
            self.fixed_state = torch.tensor([0], dtype=torch.long, device=self.device)

        except Exception as e:
            print(f"❌ Oracle initialization failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    def predict_batch(self, structures):
        """
        Batched oracle inference.
        Input:  list[pymatgen.Structure]
        Output: list[dict] with keys:
            - formation_energy
            - band_gap_scalar
        """

        if not structures:
            return []

        num_structs = len(structures)
        results = [self._error_result() for _ in range(num_structs)]

        graphs = []
        states = []
        index_map = []  # maps batched index → original structure index

        # --- 1. Graph construction (CPU) ---
        for idx, struct in enumerate(structures):
            try:
                g, state, _ = self.converter.get_graph(struct)
                graphs.append(g)
                states.append(state)
                index_map.append(idx)
            except Exception:
                # Leave default error result
                continue

        if not graphs:
            return results

        # --- 2. Batched inference (GPU) ---
        try:
            batched_graph = dgl.batch(graphs).to(self.device)
            batched_state = torch.stack(states).to(self.device)

            with torch.no_grad():
                eform_preds = self.eform_model(batched_graph, batched_state)

            # Convert to numpy
            eform_vals = eform_preds.detach().cpu().numpy().flatten()

        except Exception as e:
            print(f"❌ Batched formation-energy inference failed: {e}")
            return results

        # --- 3. Map formation energies back ---
        for i, struct_idx in enumerate(index_map):
            e_val = float(eform_vals[i])

            # SAFETY CHECK:
            # Depending on MatGL version, this is usually eV/atom.
            # If you ever see absurd magnitudes (|e| > 50),
            # divide by number of atoms here.
            if abs(e_val) > 50:
                try:
                    e_val = e_val / max(1, len(structures[struct_idx]))
                except Exception:
                    pass

            results[struct_idx]["formation_energy"] = e_val

        # --- 4. Band gap (sequential, cheap) ---
        for idx in index_map:
            try:
                gap = float(
                    self.bg_model.predict_structure(
                        structures[idx],
                        state_attr=self.fixed_state
                    )
                )
                results[idx]["band_gap_scalar"] = max(0.0, gap)
            except Exception:
                results[idx]["band_gap_scalar"] = 0.0

        return results

    @staticmethod
    def _error_result():
        return {
            "formation_energy": 0.0,
            "band_gap_scalar": 0.0,
        }
