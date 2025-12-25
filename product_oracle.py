import warnings
import torch
import os
import numpy as np

# Suppress DGL/MatGL warnings
os.environ["DGLBACKEND"] = "pytorch"
warnings.simplefilter("ignore")

class CrystalOracle:
    """
    The Inspector.
    Predicts material properties using pre-trained graph neural networks.
    Updated to support RL Training Loop API.
    """
    def __init__(self, device="cpu"):
        # Default to CPU to avoid VRAM fragmentation during RL
        self.device = torch.device(device)
        print(f"🔮 Initializing Oracle on {self.device}...")

        try:
            import matgl
            
            # 1. Load Stability Model (Formation Energy)
            print("   Loading Formation Energy Model...")
            self.model_eform = matgl.load_model("M3GNet-MP-2018.6.1-Eform").to(self.device)
            
            # 2. Load Electronic Model (Band Gap)
            print("   Loading Band Gap Model...")
            self.model_gap = matgl.load_model("MEGNet-MP-2019.4.1-BandGap-mfi").to(self.device)
            
            # 3. Fixed State Input (Required for MEGNet)
            self.fixed_state = torch.tensor([0], dtype=torch.long, device=self.device)
            
            print("   ✅ Oracle System Online.")
            
        except Exception as e:
            print(f"   ❌ Oracle crashed during load: {e}")
            raise e

    def predict_formation_energy(self, structures):
        """
        RL API: Batch predict formation energy.
        Returns: Tensor of shape (batch_size,)
        """
        preds = []
        for s in structures:
            if s is None:
                preds.append(5.0) # Penalty for invalid structures
                continue
            
            try:
                # M3GNet prediction
                val = self.model_eform.predict_structure(s)
                preds.append(float(val))
            except:
                preds.append(5.0) # Penalty if model crashes on weird geometry

        return torch.tensor(preds, device=self.device, dtype=torch.float32)

    def predict_band_gap(self, structures):
        """
        RL API: Batch predict band gap.
        Returns: Tensor of shape (batch_size,)
        """
        preds = []
        for s in structures:
            if s is None:
                preds.append(0.0)
                continue
            
            try:
                # MEGNet prediction
                val = self.model_gap.predict_structure(s, state_attr=self.fixed_state)
                preds.append(max(0.0, float(val)))
            except:
                preds.append(0.0)

        return torch.tensor(preds, device=self.device, dtype=torch.float32)

    def predict(self, structure):
        """
        Legacy/User API: Single structure dictionary return.
        """
        if structure is None: return None
        
        e_form = self.predict_formation_energy([structure])[0].item()
        bg = self.predict_band_gap([structure])[0].item()
        
        return {'formation_energy': e_form, 'band_gap': bg}

# --- TEST BLOCK ---
if __name__ == "__main__":
    from pymatgen.core import Structure, Lattice
    
    # Create a test crystal (NaCl)
    lattice = Lattice.cubic(5.6)
    species = ["Na", "Cl", "Na", "Cl"]
    coords = [[0,0,0], [0.5,0.5,0.5], [0.5,0.5,0], [0,0,0.5]]
    salt = Structure(lattice, species, coords)
    
    print("🔮 Testing Oracle on NaCl...")
    oracle = CrystalOracle(device="cpu")
    
    # Test RL API
    e_tensor = oracle.predict_formation_energy([salt, None])
    print(f"Tensor Output: {e_tensor}")
