import torch

class PhysicsRewardEngine:
    """
    Phase 2 Reward Engine.
    Prioritizes Thermodynamic Stability (Energy per Atom).
    """
    def __init__(self, device="cuda"):
        self.device = torch.device(device)
        
        # Clip rewards to prevent exploding gradients
        self.MIN_REWARD = -5.0
        self.MAX_REWARD = 10.0

    def compute_reward(self, relax_results):
        """
        Input: List of dicts from CrystalRelaxer.relax()
        Output: Tensor of rewards on GPU.
        """
        rewards = []
        
        for res in relax_results:
            if res["converged"]:
                # --- CASE A: Physics Success ---
                # Energy is usually negative (e.g., -6.0 eV/atom).
                # We want to MINIMIZE energy, so we MAXIMIZE negative energy.
                # Reward = -1 * Energy.
                # Example: Energy -6.0 => Reward +6.0
                e = res["energy_per_atom"]
                
                # Soft cap to keep rewards in range [-5, 5]
                # Most stable crystals are between -3 and -9 eV/atom
                r = -1.0 * e
                
                # Bonus: If it's very stable (E < -5.0), give a small boost
                if e < -5.0:
                    r += 2.0
                    
                rewards.append(r)
                
            else:
                # --- CASE B: Physics Failure ---
                reason = res.get("failure_reason", "unknown")
                
                if reason == "explosion_guard":
                    # Critical Failure: Model forgot geometry.
                    # Apply the harsh Phase 1 geometric penalty.
                    # ratio is usually < 0.5 here.
                    ratio = res.get("min_distance_ratio", 0.0)
                    penalty = -5.0 + (2.0 * ratio)
                    rewards.append(penalty)
                    
                else:
                    # SCF Failure / Random Math Error
                    # Mild penalty. It's not the model's fault the math crashed.
                    rewards.append(-1.0)

        # Convert list to Tensor
        return torch.tensor(rewards, device=self.device, dtype=torch.float32).clamp(
            self.MIN_REWARD, self.MAX_REWARD
        )
