import torch

class SpatialRewardEngine:
    def __init__(
        self,
        reward_clip=(-5.0, 5.0),
        target_vol=25.0,
        vol_tolerance=15.0,
    ):
        self.reward_min, self.reward_max = reward_clip
        self.target_vol = target_vol
        self.vol_tolerance = vol_tolerance

    def compute_reward(self, sentinel_results):
        batch_size = len(sentinel_results)
        rewards = torch.zeros(batch_size)

        for i in range(batch_size):
            sentinel = sentinel_results[i]
            ratio = sentinel.get("min_distance_ratio", 0.0)
            vol = sentinel.get("volume_per_atom", None)

            # ---------------- HARD DENSITY GATE ----------------
            if vol is not None and (vol < 10.0 or vol > 50.0):
                rewards[i] = -5.0
                continue

            # ---------------- INVALID GEOMETRY ----------------
            if not sentinel["valid"]:
                rewards[i] = -5.0 + (2.0 * ratio)
                continue

            # ---------------- DENSITY SHAPING ----------------
            density_bonus = 0.0
            if vol is not None:
                vol_t = torch.tensor(vol, dtype=torch.float32)
                density_bonus = 3.0 * torch.exp(
                    -((vol_t - self.target_vol) ** 2)
                    / (2 * self.vol_tolerance ** 2)
                )

            # ---------------- VALID GEOMETRY ----------------
            rewards[i] = (
                2.0                              # base success
                + (2.0 * min(ratio, 1.0))        # spacing quality
                + density_bonus                  # solid packing preference
            )

        return torch.clamp(rewards, self.reward_min, self.reward_max)
