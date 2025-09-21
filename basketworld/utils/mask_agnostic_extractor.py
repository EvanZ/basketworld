import torch as th
from typing import Dict

from stable_baselines3.common.torch_layers import CombinedExtractor


class MaskAgnosticCombinedExtractor(CombinedExtractor):
    """CombinedExtractor that preserves the observation schema but zeroes 'action_mask'.

    This prevents the policy from learning directly from the mask while keeping
    the Dict observation compatible with SB3's feature extractor.
    """

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:  # type: ignore[override]
        if isinstance(observations, dict) and ("action_mask" in observations):
            obs = dict(observations)
            try:
                obs["action_mask"] = th.zeros_like(observations["action_mask"], device=observations["action_mask"].device, dtype=th.float32)
            except Exception:
                obs["action_mask"] = th.zeros_like(observations["action_mask"], dtype=th.float32)
            return super().forward(obs)
        return super().forward(observations)


