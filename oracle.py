import torch
import warnings
import sys
import dgl
import numpy as np
import os

# Force DGL backend
os.environ["DGLBACKEND"] = "pytorch"

warnings.filterwarnings("ignore")

class Oracle:
    """
    High-Performance Batched Oracle (GPU).
    """
    def __init__(self, device="cuda"):
        try:
            import matgl
            from matgl.ext.pymatgen import Structure2Graph
            
            self.device = torch.device(device if torch.cuda.is_available() else "cpu")
            print(f"🔮 Oracle initialized on: {self.device}")

            # 1. Load Models
            self.eform_model = matgl.load_model("M3GNet-MP-2018.6.1-Eform")
            self.bg_model = matgl.load_model("MEGNet-MP-2019.4.1-BandGap-mfi")
            
            # 2. Move to GPU
            self.eform_model.to(self.device)
            self.bg_model.to(self.device)
            self.eform_model.eval()
            self.bg_model.eval()

            # 3. ROBUST ELEMENT LIST SETUP (The Fix)
            # Instead of digging into private attributes that change between versions,
            # we use the standard MP element list (H to Pu), which M3GNet supports.
            try:
                # Try the "clean" way if available in this version
                if hasattr(self.eform_model.model, "element_types"):
                    elem_list = self.eform_model.model.element_types
                else:
                    # Fallback: Standard Materials Project elements (Z=1 to 94)
                    # This is safe because M3GNet-MP is trained on this set.
                    from pymatgen.core.periodic_table import Element
                    elem_list = [str(Element.from_Z(i)) for i in range(1, 95)]
            except:
                # Ultimate fallback
                from pymatgen.core.periodic_table import Element
                elem_list = [str(Element.from_Z(i)) for i in range(1, 95)]

            self.converter = Structure2Graph(element_types=elem_list, cutoff=5.0)
            
            # MEGNet specific state
            self.fixed_state = torch.tensor([0], dtype=torch.long, device=self.device)

        except Exception as e:
            print(f"Oracle initialization failed: {e}")
            # Print full trace for debugging if it fails again
            import traceback
            traceback.print_exc()
            sys.exit(1)

    def predict_batch(self, structures):
        """
        Batched prediction for a list of structures.
        """
        if not structures:
            return []
            
        results = []
        valid_indices = []
        graphs = []
        state_attrs = []
        
        # 1. Batch Conversion (CPU side)
        for i, s in enumerate(structures):
            try:
                g, state, _ = self.converter.get_graph(s)
                graphs.append(g)
                state_attrs.append(state)
                valid_indices.append(i)
            except:
                # Placeholder for failed conversion
                results.append(None)
        
        if not graphs:
            return [self._error_result() for _ in structures]

        # 2. Batch Creation & Move to GPU
        try:
            batched_graph = dgl.batch(graphs).to(self.device)
            batched_state = torch.stack(state_attrs).to(self.device)
            
            # 3. Inference (No Grad)
            with torch.no_grad():
                # Formation Energy
                eform_preds = self.eform_model(batched_graph, batched_state)
                
                # Retrieve values
                eform_vals = eform_preds.cpu().numpy().flatten()
                
        except Exception as e:
            print(f"Batch inference failed: {e}")
            return [self._error_result() for _ in structures]

        # 4. Map results back
        res_ptr = 0
        final_output = []
        
        for idx in range(len(structures)):
            if idx in valid_indices:
                e_val = float(eform_vals[res_ptr])
                struct = structures[idx]
                
                # Sequential Gap check (Fast enough for now)
                try:
                    gap = float(self.bg_model.predict_structure(struct, state_attr=self.fixed_state.cpu()))
                    gap = max(0.0, gap)
                except:
                    gap = 0.0
                
                final_output.append({
                    "formation_energy": e_val,
                    "band_gap_scalar": gap,
                })
                res_ptr += 1
            else:
                final_output.append(self._error_result())
                
        return final_output

    def _error_result(self):
        return {
            "formation_energy": 0.0,
            "band_gap_scalar": 0.0,
        }
