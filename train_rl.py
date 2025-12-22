import os
import sys
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import csv
import yaml
from collections import deque
import time

# --- PATH SETUP ---
ROOT = os.path.dirname(os.path.abspath("__file__"))
sys.path.append(os.path.join(ROOT, "CrystalFormer"))

# --- IMPORTS ---
from crystalformer.src.transformer import make_transformer
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter
from relaxer import Relaxer
from oracle import Oracle

# --- CONFIGURATION ---
PRETRAINED_DIR = os.path.join(ROOT, "pretrained_model")
CONFIG = {
    "BATCH_SIZE": 32,
    "LR": 1e-4,
    "EPOCHS": 150,
    "KL_COEF": 0.05,
    "ENTROPY_COEF": 0.12,        # balanced exploration
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "REPLAY_RATIO": 0.4,
    "NUM_WORKERS": 1
}

# --- WORKER FUNCTION ---
_relaxer = None
def worker_relax_task(task_data):
    global _relaxer
    torch.set_num_threads(1)
    if _relaxer is None:
        _relaxer = Relaxer()
    idx, struct = task_data
    try:
        return (idx, _relaxer.relax(struct))
    except Exception as e:
        return (idx, {"is_converged": False, "error": str(e)})

# --- UTILS ---
def get_structure_hash(struct):
    try:
        return (
            struct.composition.reduced_formula,
            tuple(np.round(struct.lattice.parameters, 2)),
            tuple(np.round(struct.frac_coords.flatten(), 2))
        )
    except:
        return None

def check_geometry_fast(struct):
    try:
        dm = struct.distance_matrix
        np.fill_diagonal(dm, 10.0)
        return np.min(dm) >= 0.6
    except:
        return False

def build_structure(A, X, lattice_scale):
    species = [a for a in A if a != 0]
    coords = [x for a, x in zip(A, X) if a != 0]
    if len(species) < 1:
        return None
    try:
        lat = Lattice.from_parameters(lattice_scale, lattice_scale, lattice_scale, 90, 90, 90)
        return Structure(lat, species, coords)
    except:
        return None

# --- AGENT ---
class PPOAgent_Pipeline:
    def __init__(self):
        self.device = CONFIG["DEVICE"]
        self.start_epoch = 0
        self.best_avg_reward = -10.0
        self.reward_cache = {}
        self.memory = deque(maxlen=4000)

        with open(os.path.join(PRETRAINED_DIR, "config.yaml")) as f:
            cfg = yaml.safe_load(f)

        self.policy = make_transformer(
            key=None, **{k: cfg[k] for k in cfg if k != "dropout_rate"},
            dropout_rate=0.0, widening_factor=4
        ).to(self.device)

        self.ref_model = make_transformer(
            key=None, **{k: cfg[k] for k in cfg if k != "dropout_rate"},
            dropout_rate=0.0, widening_factor=4
        ).to(self.device)

        self.optimizer = optim.AdamW(self.policy.parameters(), lr=CONFIG["LR"])

        ckpt_path = os.path.join(ROOT, "checkpoint.pt")
        author_path = os.path.join(PRETRAINED_DIR, "epoch_005500_CLEAN.pt")

        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=self.device)
            self.policy.load_state_dict(ckpt["policy_state"])
            self.ref_model.load_state_dict(ckpt["ref_state"])
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
            self.start_epoch = ckpt["epoch"] + 1
            self.best_avg_reward = ckpt["best_reward"]
            self.memory = ckpt["memory"]
            self.reward_cache = ckpt.get("reward_cache", {})
        else:
            state = torch.load(author_path, map_location=self.device)
            self.policy.load_state_dict(state)
            self.ref_model.load_state_dict(state)
            self.inject_teacher_knowledge()

        self.ref_model.eval()

        self.idx_to_atom = {
            0: 30, 1: 16, 2: 48, 3: 34, 4: 8, 5: 31,
            6: 33, 7: 29, 8: 49, 9: 50, 10: 32, 11: 52, 12: 14
        }

    def inject_teacher_knowledge(self):
        seeds = [
            Structure(Lattice.cubic(5.43), ["Si","Si"], [[0,0,0],[0.25,0.25,0.25]]),
            Structure(Lattice.cubic(5.65), ["Ga","As"], [[0,0,0],[0.25,0.25,0.25]]),
            Structure(Lattice.cubic(4.21), ["Mg","O"], [[0,0,0],[0.5,0.5,0.5]])
        ]
        for s in seeds:
            for _ in range(50):
                self.memory.append({"struct": s, "reward": 6.0})

    def save_checkpoint(self, epoch, avg_reward):
        torch.save({
            "epoch": epoch,
            "policy_state": self.policy.state_dict(),
            "ref_state": self.ref_model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_reward": self.best_avg_reward,
            "memory": self.memory,
            "reward_cache": self.reward_cache
        }, "checkpoint.pt")

        if avg_reward > self.best_avg_reward:
            self.best_avg_reward = avg_reward
            torch.save(self.policy.state_dict(), "best_rl_model_semiconductor.pt")

    def prepare_input(self, G, XYZ, A, W, M):
        return (
            torch.tensor([G], device=self.device),
            torch.tensor([XYZ], device=self.device),
            torch.tensor([A], device=self.device),
            torch.tensor([W], device=self.device),
            torch.tensor([M], device=self.device),
        )

# --- MAIN ---
def main():
    mp.set_start_method("spawn", force=True)
    agent = PPOAgent_Pipeline()
    oracle = Oracle(device=CONFIG["DEVICE"])
    os.makedirs("rl_discoveries", exist_ok=True)

    if agent.start_epoch == 0:
        with open("training_log.csv","w",newline="") as f:
            csv.writer(f).writerow([
                "Epoch","Reward","Stable_Count","Top_Formula","Time",
                "Pct_Filtered","Pct_Diverged","Pct_Converged","Pct_Dedup"
            ])
        with open("final_candidates.csv","w",newline="") as f:
            csv.writer(f).writerow(["Formula","Formation_Energy","Band_Gap","Reward","Epoch"])

    pool = mp.Pool(CONFIG["NUM_WORKERS"])
    WINDOW = 10
    w_reward = 0.0
    w_best = None

    try:
        for epoch in range(agent.start_epoch, CONFIG["EPOCHS"]):
            start = time.time()
            rewards = []
            count_conv = count_divg = count_filt = count_dedup = 0
            batch = []

            for _ in range(CONFIG["BATCH_SIZE"]):
                if agent.memory and random.random() < CONFIG["REPLAY_RATIO"]:
                    m = random.choice(agent.memory)
                    batch.append({"type":"replay","struct":m["struct"],"stored_reward":m["reward"]})
                    continue

                G = random.randint(1,230)
                n = random.randint(2,6)
                lat = random.uniform(4.5,7.5)
                inp = agent.prepare_input(G, [[0.5]*3]*n, [0]*n, [0]*n, [1]*n)
                logits = agent.policy(*inp, is_train=False).squeeze(0)

                log_probs, atoms, coords = [], [], []
                for j in range(n):
                    base = 1 + 5*j
                    a_logits = logits[base][:13]
                    a_dist = torch.distributions.Categorical(logits=a_logits)
                    a = a_dist.sample()
                    log_probs.append(a_dist.log_prob(a))
                    atoms.append(agent.idx_to_atom[a.item()])

                    for k in range(3):
                        blk = logits[base+1+k]
                        Kx = agent.policy.Kx
                        d = torch.distributions.Categorical(logits=blk[:Kx])
                        idx = d.sample()
                        log_probs.append(d.log_prob(idx))
                        coords.append(torch.sigmoid(blk[Kx:2*Kx][idx]).item())

                X = [coords[i:i+3] for i in range(0,len(coords),3)]
                struct = build_structure(atoms,X,lat)
                batch.append({"type":"gen","log_probs":log_probs,"logits":logits,"struct":struct})

            relax_tasks = []
            for i,item in enumerate(batch):
                if item["type"]=="gen":
                    s = item["struct"]
                    if s is None or not check_geometry_fast(s):
                        item["result"]="invalid"; count_filt+=1; continue
                    sh = get_structure_hash(s)
                    if sh in agent.reward_cache:
                        item["result"]="cached"; item["cached_reward"]=agent.reward_cache[sh]; count_dedup+=1; continue
                    item["shash"]=sh
                    relax_tasks.append((i,s))
                else:
                    item["result"]="replay"

            for i,res in pool.map(worker_relax_task, relax_tasks):
                batch[i]["relax"]=res

            oracle_tasks, idxs = [], []
            for i,item in enumerate(batch):
                if "relax" in item and item["relax"]["is_converged"]:
                    fs = item["relax"]["final_structure"]
                    if check_geometry_fast(fs):
                        oracle_tasks.append(fs); idxs.append(i); count_conv+=1
                    else:
                        count_divg+=1

            if oracle_tasks:
                preds = oracle.predict_batch(oracle_tasks)
                for p,i in zip(preds,idxs):
                    batch[i]["oracle"]=p
                    batch[i]["result"]="success"

            agent.optimizer.zero_grad()
            loss = torch.tensor(0.0,device=agent.device)

            for item in batch:
                r = -1.0
                if item["result"]=="cached": r=item["cached_reward"]
                elif item["result"]=="replay": r=item["stored_reward"]
                elif item["result"]=="success":
                    fs = item["relax"]["final_structure"]
                    e = item["oracle"]["formation_energy"]
                    g = item["oracle"]["band_gap_scalar"]

                    ### NEW: semiconductor-forcing reward
                    if len(fs.composition.elements) < 2:
                        r = -0.5
                    else:
                        r_stab = 1.0 if e <= 0.05 else -1.0
                        r_gap = 5.0*np.exp(-((g-1.8)**2)/(2*0.5**2)) if g>0.1 else 0.0
                        r = r_stab + r_gap

                    if r > 2.0:
                        agent.memory.append({"struct":fs,"reward":r})
                        agent.reward_cache[item["shash"]] = r
                        with open("final_candidates.csv","a",newline="") as f:
                            csv.writer(f).writerow([fs.composition.reduced_formula,e,g,r,epoch])
                        CifWriter(fs).write_file(f"rl_discoveries/{fs.composition.reduced_formula}_{epoch}.cif")

                    log_sum = torch.stack(item["log_probs"]).sum()
                    kl = F.kl_div(F.log_softmax(item["logits"],-1),
                                  F.softmax(item["logits"].detach(),-1),
                                  reduction="batchmean")
                    loss += -(r*log_sum) + CONFIG["KL_COEF"]*kl

                rewards.append(r)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.policy.parameters(),1.0)
            agent.optimizer.step()

            avg_r = np.mean(rewards)
            w_reward += avg_r

            if (epoch+1)%WINDOW==0:
                print(f"[E{epoch+1-WINDOW}-{epoch+1}] R={w_reward/WINDOW:.2f}")
                w_reward = 0.0

            agent.save_checkpoint(epoch, avg_r)

    finally:
        pool.close(); pool.join()

if __name__ == "__main__":
    main()
