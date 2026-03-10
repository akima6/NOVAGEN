import os
import tempfile
import torch
import pickle
import collections
import numpy as np
import warnings

from pymatgen.core import Structure
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.entries.computed_entries import ComputedStructureEntry, ComputedEntry
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.io.vasp.sets import MPRelaxSet
from pymatgen.io.vasp.inputs import Incar, Poscar

warnings.filterwarnings("ignore")

class PhysicsRewardEngine:
    """
    Phase 2: The Physicist (Thermodynamic Optimization Engine) v2.1.
    Upgraded: Includes High-Dimensional Convex Hull Guard to prevent CPU deadlock.
    """
    def __init__(self, device="cuda", cache_path=None):
        self.device = torch.device(device)
        self.min_reward = -5.0
        self.max_reward = 10.0  
        self.mp_entries = [] 

        if cache_path is None:
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cache_path = os.path.join(root, "pretrained_model", "mp_thermo_cache_clean.pkl")

        if os.path.exists(cache_path):
            print(f"🧊 Physicist Jury: Loading Clean Cache from {os.path.basename(cache_path)}...")
            try:
                with open(cache_path, "rb") as f:
                    raw_entries = pickle.load(f)
                
                from pymatgen.core import Composition
                valid_entries = []
                for entry_dict in raw_entries:
                    comp = Composition(entry_dict["formula"])
                    total_energy = entry_dict["energy_per_atom"] * comp.num_atoms
                    
                    entry = ComputedEntry(
                        composition=comp,
                        energy=total_energy,
                        correction=0.0
                    )
                    valid_entries.append(entry)
                
                self.mp_entries = valid_entries
                print(f"   ✅ Loaded {len(self.mp_entries)} reference phases.")
            except Exception as e:
                print(f"   ❌ Critical Cache Failure: {e}")
        else:
            print(f"⚠️ WARNING: {cache_path} not found.")

        self.history = collections.deque(maxlen=100)

    def _generate_corrected_cse(self, structure, m3gnet_energy):
        b = MPRelaxSet(structure)
        with tempfile.TemporaryDirectory() as tmpdirname:
            b.write_input(f"{tmpdirname}/", potcar_spec=True)
            poscar = Poscar.from_file(f"{tmpdirname}/POSCAR")
            incar = Incar.from_file(f"{tmpdirname}/INCAR")
            clean_structure = Structure.from_file(f"{tmpdirname}/POSCAR")

        param = {"hubbards": {}}
        if "LDAUU" in incar:
            param["hubbards"] = dict(zip(poscar.site_symbols, incar["LDAUU"]))
        
        param["is_hubbard"] = incar.get("LDAU", True) and sum(param["hubbards"].values()) > 0
        if param["is_hubbard"]:
            param["run_type"] = "GGA+U"

        cse_d = {
            "structure": clean_structure,
            "energy": m3gnet_energy,
            "correction": 0.0,
            "parameters": param,
        }

        cse = ComputedStructureEntry.from_dict(cse_d)
        processed = MaterialsProject2020Compatibility(check_potcar=False).process_entries(
            cse, clean=True
        )
        if not processed:
            return cse 
        return processed[0]

    def compute_reward(self, relaxed_structure, energy_per_atom):
        if relaxed_structure is None or energy_per_atom is None:
            return torch.tensor(self.min_reward, device=self.device)

        # 🚨 THE FIX: High-Dimensional Convex Hull Guard
        struct_elements = set(relaxed_structure.composition.elements)
        if len(struct_elements) > 4:
            # Reject >4 element alloys to prevent infinite CPU locking
            return torch.tensor(-2.0, device=self.device)

        formula = relaxed_structure.composition.reduced_formula
        repetition_count = self.history.count(formula)
        self.history.append(formula)
        boredom_penalty = -1.0 if repetition_count > 3 else 0.0

        if not self.mp_entries:
            return torch.tensor(max(-5.0, -1.0 * energy_per_atom), device=self.device)

        try:
            system_entries = [
                entry for entry in self.mp_entries 
                if set(entry.composition.elements) <= struct_elements
            ]
            
            total_energy = energy_per_atom * len(relaxed_structure)
            target_entry = self._generate_corrected_cse(relaxed_structure, total_energy)
            
            pd = PhaseDiagram(system_entries + [target_entry])
            
            e_form = pd.get_form_energy_per_atom(target_entry)
            e_hull = pd.get_e_above_hull(target_entry, allow_negative=True)

            r_form = np.clip(-1.0 * e_form, 0.0, 3.0)
            r_hull = -3.0 * e_hull
            jackpot = 5.0 if e_hull <= 0.05 else 0.0
            
            total_reward = r_form + r_hull + jackpot + boredom_penalty
            
            return torch.tensor(total_reward, device=self.device, dtype=torch.float32).clamp(
                self.min_reward, self.max_reward
            )

        except Exception as e:
            return torch.tensor(-1.0, device=self.device)