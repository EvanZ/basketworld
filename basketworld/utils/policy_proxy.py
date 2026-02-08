from __future__ import annotations

from typing import Optional

from stable_baselines3 import PPO


class FrozenPolicyProxy:
    """Picklable proxy that loads a PPO policy from a local .zip path on first use.

    Avoids passing non-picklable PPO instances into subprocess workers.
    """

    def __init__(self, policy_path: str, device: str = "cpu"):
        self.policy_path = str(policy_path)
        # Store a string for device to keep picklable
        self.device = str(device)
        self._policy: Optional[PPO] = None

    def _ensure_loaded(self):
        if self._policy is None:
            from basketworld.utils.policies import (
                PassBiasDualCriticPolicy,
                PassBiasMultiInputPolicy,
            )
            from basketworld.policies import SetAttentionDualCriticPolicy, SetAttentionExtractor

            custom_objects = {
                "PassBiasDualCriticPolicy": PassBiasDualCriticPolicy,
                "PassBiasMultiInputPolicy": PassBiasMultiInputPolicy,
                "SetAttentionDualCriticPolicy": SetAttentionDualCriticPolicy,
                "SetAttentionExtractor": SetAttentionExtractor,
            }
            self._policy = PPO.load(
                self.policy_path, device=self.device, custom_objects=custom_objects
            )

    def predict(self, obs, deterministic: bool = False):
        self._ensure_loaded()
        return self._policy.predict(obs, deterministic=deterministic)

    @property
    def policy(self):  # expose underlying PPO for utils that expect `.policy`
        self._ensure_loaded()
        return self._policy
