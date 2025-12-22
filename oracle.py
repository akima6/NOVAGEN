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
    Attempts Batched Inference (Fast/GPU). 
    If incompatible, falls back to Sequential (Safe/CPU) automatically.
    """
    def __init__(self, device="cuda"):
        try:
            import matgl
            from matgl.ext.pymatgen import Structure2Graph
            from pymatgen.core import Structure, Lattice
            from pymatgen.core.periodic_table import Element
            
            # 1. Determine Device
            self.device_str = device if torch.cuda.is_available() else "cpu"
            self.device = torch.device(self.device_str)
            print(f"🔮 Oracle initialized on: {self.device}")

            # 2. Load Models
            self.eform_model = matgl.load_model("M3GNet-MP-2018.6.1-Eform")
            self.bg_model = matgl.load_model("MEGNet-MP-2019.4.1-BandGap-mfi")
            
            # Move to target device initially
            self.eform_model.to(self.device)
            self.bg_model.to(self.device)
            self.eform_model.eval()
            self.bg_model.eval()
            
            # 3. Setup Graph Converter (for batching)
            try:
                if hasattr(self.eform_model.model, "element_types"):
                    elem_list = self.eform_model.model.element_types
                else:
                    elem_list = [str(Element.from_Z(i)) for i in range(1, 95)]
            except:
                elem_list = [str(Element.from_Z(i)) for i in range(1, 95)]

            self.converter = Structure2Graph(element_types=elem_list, cutoff=5.0)
            
            # 4. Run Self-Test
            print("🧪 Running Batching Self-Test...")
            self.use_batching = False
            try:
                # Create Dummy Input
                dummy_s = Structure(Lattice.cubic(3.0), ["Si"], [[0,0,0]])
                test_batch = [dummy_s, dummy_s]
                
                # Try the batched path
                self._predict_batch_impl(test_batch)
                
                print("✅ Batching Self-Test Passed! Enabled High-Speed GPU Batching.")
                self.use_batching = True
                
            except Exception as e:
                print(f"⚠️ Batching Self-Test Failed ({e}).")
                print("   -> 📉 Downgrading models to CPU for Robust Sequential Mode.")
                
                # CRITICAL FIX: Move models to CPU to prevent "Device Mismatch" errors
                self.device = torch.device("cpu")
                self.eform_model.to(self.device)
                self.bg_model.to(self.device)
                self.use_batching = False

            # Create fixed state tensor on the CORRECT final device
            self.fixed_state = torch.tensor([0], dtype=torch.long, device=self.device)

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
                # Runtime fallback if batching crashes later
                return self._predict_sequential_impl(structures)
        else:
            return self._predict_sequential_impl(structures)

    def _predict_batch_impl(self, structures):
        """
        Internal Batched Implementation (GPU).
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
            
        # 4. Map Back (MEGNet runs sequentially even in batch mode usually)
        res_ptr = 0
        for idx in valid_indices:
            e_val = float(vals[res_ptr])
            try:
                # MEGNet prediction
                gap = float(self.bg_model.predict_structure(structures[idx], state_attr=self.fixed_state))
                gap = max(0.0, gap)
            except: gap = 0.0
            
            results[idx] = {"formation_energy": e_val, "band_gap_scalar": gap}
            res_ptr += 1
            
        for i in range(len(results)):
            if results[i] is None: results[i] = self._error_result()
            
        return results

    def _predict_sequential_impl(self, structures):
        """
        Internal Sequential Implementation (CPU Safe Mode).
        """
        results = []
        for struct in structures:
            try:
                # M3GNet Prediction
                e_val = float(self.eform_model.predict_structure(struct))
                
                # MEGNet Prediction
                try:
                    gap = float(self.bg_model.predict_structure(struct, state_attr=self.fixed_state))
                    gap = max(0.0, gap)
                except: gap = 0.0
                
                results.append({"formation_energy": e_val, "band_gap_scalar": gap})
            except Exception as e:
                # Print error only if debugging
                # print(f"Sequential Error: {e}")
                results.append(self._error_result())
        return results

    def _error_result(self):
        return {"formation_energy": 0.0, "band_gap_scalar": 0.0}
