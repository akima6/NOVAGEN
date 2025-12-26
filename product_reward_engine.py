import torch
import numpy as np

class RewardEngine:
    def __init__(self, target_gap_min=0.5, target_gap_max=4.0):
        self.target_gap_min = target_gap_min
        self.target_gap_max = target_gap_max

    def compute_reward(self, valid_mask, e_form_preds, bg_preds, compositions=None):
        """
        Calculates reward based on stability, band gap, and complexity.
        """
        batch_size = len(valid_mask)
        rewards = torch.zeros(batch_size)
        
        # Convert predictions to tensor if they aren't already
        if not isinstance(e_form_preds, torch.Tensor):
            e_form_preds = torch.tensor(e_form_preds)
        if not isinstance(bg_preds, torch.Tensor):
            bg_preds = torch.tensor(bg_preds)

        stats = {"valid": 0, "stable": 0, "semicon": 0}

        for i in range(batch_size):
            # 1. VALIDITY CHECK
            if not valid_mask[i]:
                rewards[i] = -5.0  # Heavy penalty for invalid/exploding crystals
                continue
            
            stats["valid"] += 1
            e_form = e_form_preds[i].item()
            bg = bg_preds[i].item()

            # 2. STABILITY REWARD
            # We want Energy < 0.0 eV/atom
            if e_form < 0.0:
                r_stab = 2.0 + abs(e_form)  # Bonus for being deeper in the well
                stats["stable"] += 1
            elif e_form < 0.5:
                r_stab = 0.5  # Small partial credit for "almost stable"
            else:
                r_stab = -2.0 - e_form  # Penalty for instability

            # 3. BAND GAP REWARD (Semiconductor Target)
            if self.target_gap_min <= bg <= self.target_gap_max:
                r_bg = 2.0  # Jackpot (Reduced from 5.0)
                stats["semicon"] += 1
            elif bg > 0.1 and bg < self.target_gap_min:
                r_bg = 0.5  # Partial credit
            elif bg == 0.0:
                r_bg = -1.0 # Penalty for Metals
            else:
                r_bg = -0.5 # Too wide

            # 4. COMPLEXITY BONUS (New Feature)
            # Reward ternary (3) and quaternary (4) elements more than binary (2)
            r_complex = 0.0
            if compositions is not None:
                try:
                    num_elements = len(compositions[i].elements)
                    if num_elements == 3:
                        r_complex = 1.5
                    elif num_elements >= 4:
                        r_complex = 3.0
                except:
                    pass

            # TOTAL REWARD
            rewards[i] = r_stab + r_bg + r_complex

        return rewards, stats
