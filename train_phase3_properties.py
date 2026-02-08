import os
import sys
import torch
import gc
import re
import glob
import numpy as np
import pynvml 
from tqdm import tqdm
import warnings
import math
import logging
import contextlib
import io

# =============================================================================
# 🚑 CRITICAL DRIVER & COMPATIBILITY SHIMS
# =============================================================================
# 1. GPU Shim: Route 'nvidia_smi' calls to 'pynvml'
sys.modules['nvidia_smi'] = pynvml

# 2. Path Shim: Ensure Windows finds the DLL
os.environ['PATH'] += os.pathsep + r'C:\Windows\System32'

# 3. ASE Compatibility Fix
import ase.constraints
sys.modules["ase.filters"] = ase.constraints
if not hasattr(ase.constraints, "ExpCellFilter"):
    if hasattr(ase.constraints, "UnitCellFilter"):
        ase.constraints.ExpCellFilter = ase.constraints.UnitCellFilter

# =============================================================================
# 🔇 AGGRESSIVE SILENCING SYSTEM
# =============================================================================
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

loggers_to_mute = ["chgnet", "dgl", "pymatgen", "matgl"]
for name in logging.root.manager.loggerDict:
    for mute_key in loggers_to_mute:
        if mute_key in name.lower():
            logger = logging.getLogger(name)
            logger.setLevel(logging.CRITICAL) 
            logger.propagate = False

class QuietBlock:
    """Context manager to silence C-level print outputs from physics engines"""
    def __enter__(self):
        self._suppress = io.StringIO()
        self._redirect_out = contextlib.redirect_stdout(self._suppress)
        self._redirect_err = contextlib.redirect_stderr(self._suppress)
        self._redirect_out.__enter__()
        self._redirect_err.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._redirect_out.__exit__(exc_type, exc_val, exc_tb)
        self._redirect_err.__exit__(exc_type, exc_val, exc_tb)

# =============================================================================
# 🏭 CONFIGURATION HUB # 1=Thermal(0.6 eV), 2=Solar(1.5 eV), 3=Light(2.8 eV)
# =============================================================================
FORCE_TARGET = 3          
PRODUCT_NAME = "final_lab_grade_Light" 
BATCH_SIZE = 2             
GRAD_ACCUM_STEPS = 8      
STEPS_PER_EPOCH = 50       
EPOCHS = 100               
LR = 1e-6                  
RELAX_STEPS = 25           

CAMPAIGN_ELEMENTS = [
    3, 11, 19, 4, 12, 20, 38, 56,
    5, 13, 31, 49, 6, 14, 32, 50,
    7, 15, 33, 51, 8, 16, 34, 52,
    21, 22, 30, 48
]

# =============================================================================
# 🛠️ SETUP & IMPORTS
# =============================================================================
BASE_DIR = os.getcwd()
sys.path.append(BASE_DIR)

try:
    from generator_service import CrystalGenerator
    from product_oracle import CrystalOracle
    from product_relaxer import CrystalRelaxer  
    from pymatgen.core.composition import Composition
except ImportError as e:
    sys.exit(f"❌ Critical Import Error: {e}")

# =============================================================================
# 🧠 REWARD ENGINE
# =============================================================================
class LabGradeRewardEngine:
    def __init__(self, target_gap):
        self.target_gap = target_gap

    def check_charge_neutrality(self, formula_str):
        try:
            comp = Composition(formula_str)
            return len(comp.oxy_state_guesses()) > 0
        except:
            return False

    def calculate(self, energy, gap, structure):
        if structure is None: return -10.0, "Explosion"
        if energy > 5.0: return -10.0, "HighEnergy"
        
        vol_per_atom = structure.volume / structure.num_sites
        if vol_per_atom > 75.0: return -5.0, "Gas"

        stability_score = max(0.0, -1.0 * energy)
        formula = structure.composition.reduced_formula
        chem_score = 1.0 if self.check_charge_neutrality(formula) else -1.0

        sigma = 0.3
        gap_error = abs(gap - self.target_gap)
        gap_score = 5.0 * math.exp(-(gap_error**2) / (2 * sigma**2))

        elements = [str(e) for e in structure.composition.elements]
        count = len(elements)
        complex_score = {2: 0.5, 3: 1.5}.get(count, 2.0 if count >= 4 else 0.0)

        total_reward = stability_score + chem_score + gap_score + complex_score
        return total_reward, "Valid"

def get_target_gap():
    return 1.5 if FORCE_TARGET == 2 else (0.6 if FORCE_TARGET == 1 else 2.8)

def find_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir): return None, 1
    files = glob.glob(os.path.join(checkpoint_dir, f"{PRODUCT_NAME}_epoch_*.pt"))
    if not files: return None, 1
    latest = max(files, key=lambda x: int(re.search(r'epoch_(\d+)', x).group(1)))
    epoch = int(re.search(r'epoch_(\d+)', latest).group(1))
    return latest, epoch + 1

# =============================================================================
# 🚀 MAIN TRAINING LOOP
# =============================================================================
def run_lab_training():
    TARGET_GAP = get_target_gap()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    physics_device = "cuda"

    save_checkpoint_dir = os.path.join(BASE_DIR, "rl_checkpoints", "phase3")
    trophy_dir = os.path.join(save_checkpoint_dir, "trophies")
    os.makedirs(save_checkpoint_dir, exist_ok=True)
    os.makedirs(trophy_dir, exist_ok=True)

    print(f"\n🏭 LAB-GRADE FACTORY: {PRODUCT_NAME.upper()}")
    print(f"   🎯 Target Bandgap: {TARGET_GAP} eV")

    latest_pt, start_epoch = find_latest_checkpoint(save_checkpoint_dir)
    model_path = latest_pt if latest_pt else os.path.join(BASE_DIR, "pretrained_model", "physics_expert_epoch_60_CHGNet_final.pt")
    config_path = os.path.join(BASE_DIR, "pretrained_model", "config.yaml")

    generator = CrystalGenerator(model_path, config_path, device)
    generator.model.train() 
    optimizer = torch.optim.Adam(generator.model.parameters(), lr=LR)

    print(f"   🔌 Connecting Engines (Physics on {physics_device})...")
    oracle = CrystalOracle(device="cpu")   
    relaxer = CrystalRelaxer(device=physics_device, method="CHGNet") 
    reward_engine = LabGradeRewardEngine(target_gap=TARGET_GAP)

    moving_baseline = 0.0
    if latest_pt:
        try:
            ckpt = torch.load(latest_pt, map_location=device)
            if "optimizer_state" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            moving_baseline = ckpt.get("baseline", 0.0)
        except: pass

    for epoch in range(start_epoch, EPOCHS + 1):
        current_temp = max(0.6, 1.0 - (0.4 * (epoch / EPOCHS)))
        epoch_rewards, epoch_gaps, hits = [], [], 0

        pbar = tqdm(range(STEPS_PER_EPOCH), desc=f"Ep {epoch}/{EPOCHS}", ncols=100)

        for step in pbar:
            try:
                outputs = generator.generate_with_grads(BATCH_SIZE, CAMPAIGN_ELEMENTS, temperature=current_temp)
                raw_struct = outputs["structures"][0]
                log_probs = outputs["log_probs"]

                if raw_struct:
                    if raw_struct.num_sites > 52: 
                        raw_struct = None
                    # --- 🛑 DENSITY FILTER (Prevents Isolated Atom Hangs) ---
                    elif raw_struct.density < 0.8:
                        raw_struct = None
                    # Scaling logic
                    elif (raw_struct.volume / raw_struct.num_sites) > 35.0:
                        raw_struct.scale_lattice(20.0 * raw_struct.num_sites)

                real_energy, gap, clean_struct = 10.0, 0.0, None
                if raw_struct:
                    with QuietBlock():
                        try:
                            res = relaxer.relax(raw_struct, steps=RELAX_STEPS)
                            if res["converged"]:
                                clean_struct = res["final_structure"]
                                real_energy = res["energy_per_atom"]
                        except: pass

                if clean_struct and real_energy < 0:
                    with QuietBlock():
                        _, gaps = oracle.predict_batch([clean_struct])
                        gap = gaps[0]

                total_reward, reason = reward_engine.calculate(real_energy, gap, clean_struct)
                moving_baseline = 0.95 * moving_baseline + 0.05 * total_reward

                if log_probs[0] is not None:
                    advantage = total_reward - moving_baseline
                    loss = -(log_probs[0] * advantage) / GRAD_ACCUM_STEPS
                    loss.backward()

                    if real_energy < -1.0 and abs(gap - TARGET_GAP) < 0.4:
                        hits += 1
                        formula = clean_struct.composition.reduced_formula
                        fname = f"HIT_ep{epoch}_{formula}_E{real_energy:.2f}_Gap{gap:.2f}.cif"
                        clean_struct.to(filename=os.path.join(trophy_dir, fname))

                if (step + 1) % GRAD_ACCUM_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(generator.model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                
                epoch_rewards.append(total_reward)
                if gap > 0: epoch_gaps.append(gap)
                pbar.set_postfix({"Rwrd": f"{total_reward:.1f}", "Gap": f"{gap:.2f}", "eV": f"{real_energy:.1f}"})
                
                del outputs, raw_struct, clean_struct
                gc.collect()
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"\n⚠️ CRASH REPORT: {e}")
                continue

        avg_r = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0.0
        print(f"📊 Ep {epoch} | Avg Reward: {avg_r:.2f} | Hits: {hits}")

        if epoch % 5 == 0:
            save_path = os.path.join(save_checkpoint_dir, f"{PRODUCT_NAME}_epoch_{epoch}.pt")
            torch.save({
                "model_state": generator.model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "baseline": moving_baseline
            }, save_path)
            print(f"   💾 Saved: {os.path.basename(save_path)}")

if __name__ == "__main__":
    run_lab_training()