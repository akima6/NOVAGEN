import torch
import itertools
import smact

class ChemistryRewardEngine:
    """
    Combined Reward Engine v2.0 (Smoothed):
    Evaluates spatial geometry (density, atom spacing) AND chemical viability 
    (Pauling electronegativity, continuous charge neutrality).
    Replaces hard step-functions with smooth exponential and Gaussian curves 
    for stable policy gradient optimization.
    """
    def __init__(
        self,
        reward_clip=(-10.0, 25.0),   
        target_vol=25.0,
        vol_tolerance=15.0,
        max_charge_reward=10.0,      # The peak of the Gaussian "Jackpot" curve
        charge_tolerance=0.5         # How sharply the reward drops as charge imbalances
    ):
        self.reward_min, self.reward_max = reward_clip
        self.target_vol = target_vol
        self.vol_tolerance = vol_tolerance
        self.max_charge_reward = max_charge_reward
        self.charge_tolerance = charge_tolerance

    def compute_reward(self, structures, sentinel_results):
        batch_size = len(sentinel_results)
        rewards = torch.zeros(batch_size)

        for i in range(batch_size):
            struct = structures[i]
            sentinel = sentinel_results[i]

            # ---------------- 1. NULL CHECK ----------------
            if struct is None:
                rewards[i] = self.reward_min
                continue

            ratio = sentinel.get("min_distance_ratio", 0.0)
            vol = sentinel.get("volume_per_atom", None)

            # ---------------- 2. SPATIAL / GEOMETRY GATES ----------------
            if not sentinel.get("valid", False):
                # Smooth degradation based on how close they got to a valid ratio
                # Instead of a flat -5.0, it scales smoothly from -10 (implosion) up to 0.
                rewards[i] = -10.0 * (1.0 - min(ratio, 1.0))
                continue
            
            # ---------------- 3. SMOOTHED PAULING TEST ----------------
            comp_dict = struct.composition.get_el_amt_dict()
            elements = list(comp_dict.keys())
            
            pauling_score = 0.0
            if len(elements) > 1:
                enegs = []
                for el in elements:
                    eneg = smact.Element(el).pauling_eneg
                    if eneg is not None:
                        enegs.append(eneg)
                
                if enegs:
                    eneg_gap = max(enegs) - min(enegs)
                    # Smooth Sigmoid-like curve: Rewards >1.0 gaps, gently penalizes <1.0 gaps.
                    # Eliminates the hard 'continue' cutoff that broke the gradients.
                    pauling_score = 2.0 * torch.tanh(torch.tensor(eneg_gap - 1.0)).item()

            # ---------------- 4. GAUSSIAN CHARGE NEUTRALITY ----------------
            ox_states_list = []
            counts = list(comp_dict.values())
            
            for el in elements:
                states = smact.Element(el).oxidation_states
                if not states:
                    states = [0] 
                ox_states_list.append(states)
            
            min_abs_charge = float('inf')
            for combo in itertools.product(*ox_states_list):
                net_charge = sum(ox * count for ox, count in zip(combo, counts))
                abs_charge = abs(net_charge)
                
                if abs_charge < min_abs_charge:
                    min_abs_charge = abs_charge
                    if min_abs_charge == 0:
                        break 
            
            # 🔹 THE FIX: Gaussian Reward Curve
            # Instead of a linear penalty and a sudden +10 jackpot, we use a bell curve.
            # Perfect neutrality = +10. Slight imbalance (e.g., 0.5) = +3.6. Heavy imbalance = ~0.
            charge_score = self.max_charge_reward * torch.exp(
                torch.tensor(-(min_abs_charge ** 2) / (2 * self.charge_tolerance ** 2))
            ).item()

            # ---------------- 5. DENSITY SHAPING ----------------
            density_bonus = 0.0
            if vol is not None:
                vol_t = torch.tensor(vol, dtype=torch.float32)
                # Your existing Gaussian volume reward was already perfect!
                density_bonus = 3.0 * torch.exp(
                    -((vol_t - self.target_vol) ** 2) / (2 * self.vol_tolerance ** 2)
                ).item()

            # ---------------- 6. BASE SCORE ----------------
            base_score = (
                2.0                              # Base success for passing geometry
                + (2.0 * min(ratio, 1.0))        # Spacing quality bonus
                + density_bonus                  # Solid packing preference
                + pauling_score                  # Smooth electronegativity gap preference
                + charge_score                   # Smooth Gaussian charge jackpot
            )

            rewards[i] = base_score

        return torch.clamp(rewards, self.reward_min, self.reward_max)