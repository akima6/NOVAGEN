import numpy as np
import torch

class RewardEngine:
    """
    The 'Teacher' for the RL Agent.
    UPDATED logic to strictly punish metals and simple unary crystals.
    """
    def __init__(self, target_gap_min=0.5, target_gap_max=4.0):
        self.target_gap_min = target_gap_min
        self.target_gap_max = target_gap_max

    def compute_reward(self, validity_mask, formation_energies, band_gaps, compositions=None):
        """
        Input:
            validity_mask: List[bool]
            formation_energies: Tensor (eV/atom)
            band_gaps: Tensor (eV)
            compositions: List[Composition] (Optional, needed to check for Unary)
        """
        rewards = []
        
        # Stats tracking
        n_valid = 0
        n_semicon = 0
        
        # Move tensors to CPU list for easier processing in loop
        e_list = formation_energies.tolist()
        bg_list = band_gaps.tolist()
        
        for i, is_valid in enumerate(validity_mask):
            r_total = 0.0
            
            # --- 1. VALIDITY (The Bouncer) ---
            if not is_valid:
                rewards.append(-5.0) # Punishment for breaking physics
                continue
            
            n_valid += 1
            r_total += 1.0 # Participation trophy for being valid
            
            # --- 2. ELECTRONIC PROPERTY (The Target) ---
            # This is now the MOST important metric.
            bg = bg_list[i]
            
            if bg < 0.1:
                # METAL PENALTY
                # We actively punish metals so the AI learns to avoid them
                r_total -= 3.0 
            elif self.target_gap_min <= bg <= self.target_gap_max:
                # SWEET SPOT BONUS
                # Huge reward to overpower the stability of metals
                r_total += 5.0 
                n_semicon += 1
            elif bg > self.target_gap_max:
                # Insulator (Too wide, but better than metal)
                r_total += 1.0
            else:
                # Narrow gap (0.1 - 0.5)
                r_total += 0.5

            # --- 3. STABILITY (The Constraint) ---
            # We want stable, but not "Metal Stable"
            e_form = e_list[i]
            
            # Clamp energy to avoid infinities
            e_clamped = max(min(e_form, 5.0), -5.0)
            
            if e_clamped < 0.0:
                # Stable: Reward scales with stability, but CAPPED
                # Max stability reward is +2.0 (Lower than Semiconductor bonus)
                # This prevents "Super Stable Metals" from winning just by being stable.
                r_total += 2.0 * np.tanh(-0.5 * e_clamped)
            else:
                # Unstable: Punishment
                r_total -= 2.0 * np.tanh(e_clamped)

            # --- 4. COMPLEXITY CHECK (No Unary) ---
            # If we have composition data, penalize single-element materials
            if compositions is not None and i < len(compositions) and compositions[i] is not None:
                if len(compositions[i].elements) < 2:
                    r_total -= 3.0 # Punishment for being boring (e.g. Pure Fe)

            rewards.append(r_total)

        # Return stats for logging
        stats = {
            "valid_rate": n_valid / (len(rewards) + 1e-6),
            "semicon_rate": n_semicon / (n_valid + 1e-6)
        }
        
        return torch.tensor(rewards), stats
