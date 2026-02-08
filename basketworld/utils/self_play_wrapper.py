from __future__ import annotations

from typing import Optional, Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO

from .action_resolution import (
    IllegalActionStrategy,
    get_policy_action_probabilities,
    resolve_illegal_actions,
)
from basketworld.envs.basketworld_env_v2 import Team


class SelfPlayEnvWrapper(gym.Wrapper):
    """Wrap an env to manage opponent actions and resolve illegal actions.

    - training_strategy: strategy for the current training team
    - opponent_strategy: strategy for the frozen opponent
    - deterministic_opponent: when True, use deterministic choice in sampling
    - action_space: exposes only the training team's actions to the learning policy
    """

    def __init__(
        self,
        env: gym.Env,
        opponent_policy: Any,  # PPO or str path
        training_strategy: IllegalActionStrategy = IllegalActionStrategy.SAMPLE_PROB,
        opponent_strategy: IllegalActionStrategy = IllegalActionStrategy.BEST_PROB,
        deterministic_opponent: bool = False,
    ) -> None:
        super().__init__(env)
        self.opponent_policy = opponent_policy
        self.training_strategy = training_strategy
        self.opponent_strategy = opponent_strategy
        self.deterministic_opponent = bool(deterministic_opponent)
        self._set_team_ids()
        self._configure_action_space()

    def _ensure_opponent_loaded(self):
        # Allow passing a policy path string to avoid pickling full PPO
        if isinstance(self.opponent_policy, str):
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
            self.opponent_policy = PPO.load(
                self.opponent_policy, device="cpu", custom_objects=custom_objects
            )

    def _set_team_ids(self) -> None:
        if self.env.unwrapped.training_team == Team.OFFENSE:
            self.training_player_ids = list(self.env.unwrapped.offense_ids)
            self.opponent_player_ids = list(self.env.unwrapped.defense_ids)
        else:
            self.training_player_ids = list(self.env.unwrapped.defense_ids)
            self.opponent_player_ids = list(self.env.unwrapped.offense_ids)
        # Precompute numpy indices for fast masking
        self._training_player_indices = np.array(self.training_player_ids, dtype=int)
        self._opponent_player_indices = np.array(self.opponent_player_ids, dtype=int)

    def _configure_action_space(self) -> None:
        """Expose a reduced MultiDiscrete action space for the training team."""
        base_space = getattr(self.env, "action_space", None)
        if not isinstance(base_space, spaces.MultiDiscrete):
            raise TypeError(
                "SelfPlayEnvWrapper requires underlying env to use MultiDiscrete action space"
            )
        if len(self.training_player_ids) == 0:
            raise ValueError("Training player list cannot be empty")
        training_nvec = [int(base_space.nvec[i]) for i in self.training_player_ids]
        self.action_space = spaces.MultiDiscrete(training_nvec)

    def reset(self, **kwargs):
        try:
            obs, info = self.env.reset(**kwargs)
            self._last_obs = obs
            self._set_team_ids()
            # Ensure action space matches the (potentially updated) training team
            self._configure_action_space()
            return obs, info
        except Exception as e:
            # Provide detailed context so SubprocVecEnv surfaces useful diagnostics
            raise RuntimeError(
                f"SelfPlayEnvWrapper.reset failed: {type(e).__name__}: {e}"
            ) from e

    def step(self, action):
        # Build opponent observation with flipped role flag for unified policy
        opponent_obs = dict(self._last_obs)
        try:
            # Opponent is offense when training team is defense, and vice versa
            opponent_is_offense = self.env.unwrapped.training_team != Team.OFFENSE
            opponent_obs["role_flag"] = (
                np.array([1.0], dtype=np.float32)
                if opponent_is_offense
                else np.array([-1.0], dtype=np.float32)
            )
        except Exception:
            pass

        # Opponent raw actions and probs
        try:
            self._ensure_opponent_loaded()
            opp_actions_raw, _ = self.opponent_policy.predict(
                opponent_obs, deterministic=self.deterministic_opponent
            )
            action_mask = self._last_obs["action_mask"]
        except Exception as e:
            raise RuntimeError(
                f"SelfPlayEnvWrapper.step opponent predict failed: {type(e).__name__}: {e}"
            ) from e

        # Resolve opponent illegal actions using only their players
        opponent_mask = action_mask[self._opponent_player_indices]
        try:
            opp_probs_full = get_policy_action_probabilities(
                self.opponent_policy, opponent_obs
            )
        except Exception:
            opp_probs_full = None

        opponent_probs = None
        if opp_probs_full is not None:
            try:
                total_players = self.env.unwrapped.n_players
            except Exception:
                total_players = len(opp_probs_full)

            if len(opp_probs_full) == total_players:
                opponent_probs = [opp_probs_full[int(pid)] for pid in self.opponent_player_ids]
            else:
                opponent_probs = opp_probs_full[: len(self.opponent_player_ids)]

        opp_actions = resolve_illegal_actions(
            np.array(opp_actions_raw),
            opponent_mask,
            self.opponent_strategy,
            self.deterministic_opponent,
            opponent_probs,
        )

        action = np.array(action, dtype=int)
        expected_dims = len(self.training_player_ids)
        if action.shape[0] != expected_dims:
            raise ValueError(
                f"Expected {expected_dims} training actions, received shape {action.shape}"
            )
        training_mask = action_mask[self._training_player_indices]
        n_actions = action_mask.shape[1]
        uniform_probs = [
            np.ones(n_actions, dtype=np.float32) for _ in range(expected_dims)
        ]
        training_actions = resolve_illegal_actions(
            action,
            training_mask,
            self.training_strategy,
            False,
            uniform_probs,
        )

        # Build full action vector for the underlying environment
        full_action = np.zeros(self.env.unwrapped.n_players, dtype=int)
        for idx, pid in enumerate(self.training_player_ids):
            full_action[pid] = int(training_actions[idx])
        for idx, pid in enumerate(self.opponent_player_ids):
            full_action[pid] = int(opp_actions[idx])

        try:
            obs, reward, done, truncated, info = self.env.step(full_action)
        except Exception as e:
            raise RuntimeError(
                f"SelfPlayEnvWrapper.step env.step failed: {type(e).__name__}: {e}"
            ) from e
        # (Reverted) do not mutate info here
        self._last_obs = obs
        return obs, reward, done, truncated, info

    # Provide a top-level hook so VecEnv.env_method('set_phi_beta', ...) does not
    # trigger Gymnasium's deprecated attribute forwarding warnings. This forwards
    # directly to the unwrapped environment implementation.
    def set_phi_beta(self, value: float) -> None:  # pragma: no cover - thin shim
        try:
            self.env.unwrapped.set_phi_beta(float(value))
        except Exception:
            pass

    def set_pass_arc_degrees(
        self, value: float
    ) -> None:  # pragma: no cover - thin shim
        try:
            self.env.unwrapped.set_pass_arc_degrees(float(value))
        except Exception:
            pass

    def set_pass_oob_turnover_prob(
        self, value: float
    ) -> None:  # pragma: no cover - thin shim
        try:
            self.env.unwrapped.set_pass_oob_turnover_prob(float(value))
        except Exception:
            pass

    def get_profile_stats(self):  # pragma: no cover - thin shim
        try:
            return self.env.unwrapped.get_profile_stats()
        except Exception:
            return {}

    def reset_profile_stats(self) -> None:  # pragma: no cover - thin shim
        try:
            self.env.unwrapped.reset_profile_stats()
        except Exception:
            pass
