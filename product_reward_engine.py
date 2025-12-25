import numpy as np
import torch

class RewardEngine:
    """
    The 'Teacher' for the RL Agent.
    Calculates a composite score based on:
    1. Validity (Did you break physics?)
    2. Stability (Is the energy low?)
    3. Electronic Property (Is it a semiconductor?)
    """
    def __init__(self, target_gap_min=0.5, target_gap_max=2.5):
        self.target_gap_min = target_gap_min
        self.target_gap_max = target_gap_max

    def compute_reward(self, validity_mask, formation_energies, band_gaps):
        """
        Input:
            validity_mask: List[bool] (Passed Sentinel?)
            formation_energies: List[float] (eV/atom from Oracle)
            band_gaps: List[float] (eV from Oracle)
        Returns:
            total_rewards: torch.Tensor (Batch of scores)
            stats: Dict (For logging)
        """
        rewards = []
        
        # Stats tracking
        n_valid = 0
        n_stable = 0
        n_semi = 0
        
        for i, is_valid in enumerate(validity_mask):
            r_val = 0.0
            r_stab = 0.0
            r_elec = 0.0
            
            # --- 1. VALIDITY REWARD (The Bouncer) ---
            if not is_valid:
                # Severe punishment for hallucinations (atoms overlapping, etc.)
                r_val = -5.0
            else:
                r_val = +1.0
                n_valid += 1
                
                # --- 2. STABILITY REWARD (The Compass) ---
                e_form = formation_energies[i]
                
                # Sigmoid-like clamping to prevent Oracle exploitation
                # If E < -0.5 (Great): Reward +5.0
                # If E > 0.5 (Bad): Reward -2.0
                # We use a tanh curve centered at 0.0
                
                # Invert energy (Lower is better)
                # We clamp predictions to avoid "Infinity" glitches
                e_clamped = np.clip(e_form, -5.0, 5.0) 
                
                if e_clamped < 0.0:
                    # Positive reward for negative energy (Stable)
                    # Scale: 0.0 -> +5.0
                    r_stab = 5.0 * np.tanh(-1.0 * e_clamped) 
                    if e_clamped < -0.1: n_stable += 1
                else:
                    # Negative reward for positive energy (Unstable)
                    # Scale: 0.0 -> -2.0
                    r_stab = -2.0 * np.tanh(e_clamped)

                # --- 3. ELECTRONIC REWARD (The Target) ---
                bg = band_gaps[i]
                if self.target_gap_min <= bg <= self.target_gap_max:
                    # Solar Sweet Spot Bonus
                    r_elec = +3.0
                    n_semi += 1
                elif bg > self.target_gap_max:
                    # Insulator (Better than metal, but not target)
                    r_elec = +0.5
                else:
                    # Metal (Gap ~ 0)
                    r_elec = 0.0

            total = r_val + r_stab + r_elec
            rewards.append(total)

        return torch.tensor(rewards), {
            "valid_rate": n_valid / len(rewards),
            "stable_rate": n_stable / (n_valid + 1e-6), # Among valid
            "semi_rate": n_semi / (n_valid + 1e-6)     # Among valid
        }
