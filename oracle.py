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
    If incompatible with installed DGL/MatGL version, falls back to Sequential (Safe).
    """
    def __init__(self, device="cuda"):
        try:
            import matgl
            from matgl.ext.pymatgen import Structure2Graph
            from pymatgen.core import Structure, Lattice
            
            self.device = torch.device(device if torch.cuda.is_available() else "cpu")
            print(f"🔮 Oracle initialized on: {self.device}")

            # 1. Load Models
            self.eform_model = matgl.load_model("M3GNet-MP-2018.6.1-Eform")
            self.bg_model = matgl.load_model("MEGNet-MP-2019.4.1-BandGap-mfi")
            
            self.eform_model.to(self.device)
            self.bg_model.to(self.device)
            self.eform_model.eval()
            self.bg_model.eval()
            
            # Element Converter Setup
            # Robust fallback for finding element types
            try:
                if hasattr(self.eform_model.model, "element_types"):
                    elem_list = self.eform_model.model.element_types
                else:
                    from pymatgen.core.periodic_table import Element
                    elem_list = [str(Element.from_Z(i)) for i in range(1, 95)]
            except:
                from pymatgen.core.periodic_table import Element
                elem_list = [str(Element.from_Z(i)) for i in range(1, 95)]

            self.converter = Structure2Graph(element_types=elem_list, cutoff=5.0)
            self.fixed_state = torch.tensor([0], dtype=torch.long, device=self.device)
            
            # --- SELF-TEST BATCHING CAPABILITY ---
            print("🧪 Running Batching Self-Test...")
            self.use_batching = False
            try:
                # Create 2 Dummy Structures (Simple Cubic)
                dummy_s = Structure(Lattice.cubic(3.0), ["Si"], [[0,0,0]])
                test_batch = [dummy_s, dummy_s]
                
                # Try to run the batch logic
                self._predict_batch_impl(test_batch)
                
                print("✅ Batching Self-Test Passed! Enabling High-Speed GPU Batching.")
                self.use_batching = True
            except Exception as e:
                print(f"⚠️ Batching Self-Test Failed ({e}).")
                print("   -> Falling back to Robust Sequential Mode.")
                self.use_batching = False

        except Exception as e:
            print(f"Oracle initialization failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    def predict_batch(self, structures):
        """
        Public method that chooses the best strategy.
        """
        if not structures: return []
        
        if self.use_batching:
            try:
                return self._predict_batch_impl(structures)
            except:
                # If batching fails during runtime (rare edge case), fallback
                return self._predict_sequential_impl(structures)
        else:
            return self._predict_sequential_impl(structures)

    def _predict_batch_impl(self, structures):
        """
        Internal Batched Implementation (The Fast Way).
        """
        graphs = []
        state_attrs = []
        valid_indices = []
        results = [None] * len(structures)
        
        # 1. Convert to Graphs (CPU)
        for i, s in enumerate(structures):
            try:
                g, state, _ = self.converter.get_graph(s)
                graphs.append(g)
                state_attrs.append(state)
                valid_indices.append(i)
            except:
                results[i] = self._error_result()

        if not graphs: return results

        # 2. Batch & GPU
        batched_graph = dgl.batch(graphs).to(self.device)
        batched_state = torch.stack(state_attrs).to(self.device)
        
        # 3. Inference
        with torch.no_grad():
            preds = self.eform_model(batched_graph, batched_state)
            vals = preds.cpu().numpy().flatten()
            
        # 4. Map Back
        res_ptr = 0
        for idx in valid_indices:
            e_val = float(vals[res_ptr])
            # Sequential Gap (MEGNet is tricky to batch with M3GNet graphs)
            try:
                gap = float(self.bg_model.predict_structure(structures[idx], state_attr=self.fixed_state.cpu()))
                gap = max(0.0, gap)
            except: gap = 0.0
            
            results[idx] = {
                "formation_energy": e_val,
                "band_gap_scalar": gap
            }
            res_ptr += 1
            
        # Fill remaining Nones
        for i in range(len(results)):
            if results[i] is None: results[i] = self._error_result()
            
        return results

    def _predict_sequential_impl(self, structures):
        """
        Internal Sequential Implementation (The Safe Way).
        """
        results = []
        for struct in structures:
            try:
                e_val = float(self.eform_model.predict_structure(struct))
                try:
                    gap = float(self.bg_model.predict_structure(struct, state_attr=self.fixed_state.cpu()))
                    gap = max(0.0, gap)
                except: gap = 0.0
                
                results.append({"formation_energy": e_val, "band_gap_scalar": gap})

            except Exception as e:
                            print(f"❌ Sequential Prediction Error: {e}") # <--- Add this
                            results.append(self._error_result())
                    return results

    def _error_result(self):
        return {"formation_energy": 0.0, "band_gap_scalar": 0.0}
