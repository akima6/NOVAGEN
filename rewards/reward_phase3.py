import torch
import math

class EngineerRewardEngine:
    def __init__(self, ideal_gap=1.5):
        # We target the theoretical optimal semiconductor bandgap
        self.ideal_gap = ideal_gap
        self.max_gap_score = 30.0
        # 🔹 WIDENED THE MOUNTAIN: Gives the AI a massive slope to climb
        self.sigma = 1.0  
        
    def compute_reward(self, energy_pa, gap):
        import logging
        logger = logging.getLogger(__name__)
        
        energy_score = max(0.0, -energy_pa) 
        
        # Check for oracle failure signal
        if gap < 0:  # Failure signal
            logger.warning(f"Reward: Oracle failed (gap={gap})")
            return {
                "total_reward": -10.0,  # Heavy penalty for oracle failure
                "energy_score": 0.0,
                "gap_score": -10.0
            }
        
        # Adaptive Gaussian Reward: The closer to 1.5 eV, the higher the score
        gap_score = self.max_gap_score * math.exp(-((gap - self.ideal_gap)**2) / (2 * self.sigma**2))
        
        return {
            "total_reward": energy_score + gap_score,
            "energy_score": energy_score,
            "gap_score": gap_score
        }