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
    Predicts material properties (Stability & Electronic Structure)
    using pre-trained graph neural networks.
    """
    def __init__(self, device="cpu"):
        # We default to CPU for the Oracle to avoid VRAM fragmentation 
        # when running alongside the Generator/Relaxer.
        self.device = torch.device(device)
        print(f"🔮 Initializing Oracle on {self.device}...")

        try:
            import matgl
            
            # 1. Load Stability Model (Formation Energy)
            # This tells us if the crystal can exist in nature.
            print("   Loading Formation Energy Model...")
            self.model_eform = matgl.load_model("M3GNet-MP-2018.6.1-Eform").to(self.device)
            
            # 2. Load Electronic Model (Band Gap)
            # This tells us if it conducts electricity or light.
            print("   Loading Band Gap Model...")
            self.model_gap = matgl.load_model("MEGNet-MP-2019.4.1-BandGap-mfi").to(self.device)
            
            # 3. Fixed State Input (Required for MEGNet)
            # MEGNet expects a global state tensor (Temperature, Pressure, etc.). 
            # We use [0, 0] as the standard reference.
            self.fixed_state = torch.tensor([0, 0], dtype=torch.float32, device=self.device)
            
            print("   ✅ Oracle System Online.")
            
        except Exception as e:
            print(f"   ❌ Oracle crashed during load: {e}")
            raise e

    def predict(self, structure):
        """
        Run inference on a single Pymatgen Structure.
        Returns: {'formation_energy': float, 'band_gap': float}
        """
        if structure is None:
            return None

        result = {}
        
        # A. Predict Formation Energy
        try:
            # matgl models handle the graph conversion internally
            eform = self.model_eform.predict_structure(structure)
            result['formation_energy'] = float(eform)
        except Exception as e:
            print(f"⚠️ E_form failed: {e}")
            result['formation_energy'] = 0.0

        # B. Predict Band Gap
        try:
            # MEGNet needs the state_attr argument
            gap = self.model_gap.predict_structure(structure, state_attr=self.fixed_state)
            result['band_gap'] = max(0.0, float(gap)) # Physics check: Gap cannot be negative
        except Exception as e:
            print(f"⚠️ BandGap failed: {e}")
            result['band_gap'] = 0.0
            
        return result

    def predict_batch(self, structures):
        """
        Helper to predict a list of structures.
        """
        return [self.predict(s) for s in structures]

# --- TEST BLOCK ---
if __name__ == "__main__":
    from pymatgen.core import Structure, Lattice
    
    # Create a test crystal (NaCl - Salt)
    # This is an insulator, so it should have a HIGH band gap (> 4 eV)
    lattice = Lattice.cubic(5.6)
    species = ["Na", "Cl", "Na", "Cl"]
    coords = [[0,0,0], [0.5,0.5,0.5], [0.5,0.5,0], [0,0,0.5]]
    salt = Structure(lattice, species, coords)
    
    print("🔮 Testing Oracle on NaCl (Salt)...")
    oracle = CrystalOracle(device="cpu")
    
    props = oracle.predict(salt)
    
    print("\n📊 PREDICTION RESULTS:")
    print(f"   Formation Energy: {props['formation_energy']:.3f} eV/atom (Should be negative)")
    print(f"   Band Gap:         {props['band_gap']:.3f} eV       (Should be > 3.0 for Salt)")
