from __future__ import annotations

from typing import Optional, Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from .action_resolution import (
    IllegalActionStrategy,
    get_policy_action_probabilities,
    resolve_illegal_actions,
)
from basketworld.envs.basketworld_env_v2 import Team
from basketworld.utils.policy_loading import load_ppo_for_inference


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
        self._apply_pass_mode_to_policy(self.opponent_policy)

    def _current_pass_mode(self) -> str:
        try:
            return str(getattr(self.env.unwrapped, "pass_mode", "directional")).lower()
        except Exception:
            return "directional"

    def _apply_pass_mode_to_policy(self, policy_obj: Any) -> None:
        if policy_obj is None:
            return
        try:
            mode = self._current_pass_mode()
            policy = getattr(policy_obj, "policy", None)
            if policy is None:
                policy = policy_obj
            if hasattr(policy, "set_pass_mode"):
                policy.set_pass_mode(mode)
        except Exception:
            pass

    def _ensure_opponent_loaded(self):
        # Allow passing a policy path string to avoid pickling full PPO
        if isinstance(self.opponent_policy, str):
            self.opponent_policy = load_ppo_for_inference(
                self.opponent_policy,
                device="cpu",
            )
        self._apply_pass_mode_to_policy(self.opponent_policy)

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

    def _recondition_intent_fields_for_role(
        self, obs_dict: dict, observer_is_offense: bool
    ) -> None:
        """Update role-dependent intent fields after role-flag flips."""
        if not isinstance(obs_dict, dict):
            return
        try:
            env = self.env.unwrapped
        except Exception:
            return
        try:
            fields = env.get_intent_observation_fields(bool(observer_is_offense))
        except Exception:
            fields = {}
        if fields:
            for key, value in fields.items():
                obs_dict[key] = np.array(value, dtype=np.float32, copy=True)
        if "globals" in obs_dict:
            try:
                obs_dict["globals"] = env.patch_globals_with_intent_features(
                    obs_dict["globals"], bool(observer_is_offense)
                )
            except Exception:
                pass

    def reset(self, **kwargs):
        try:
            obs, info = self.env.reset(**kwargs)
            self._last_obs = obs
            self._set_team_ids()
            # Ensure action space matches the (potentially updated) training team
            self._configure_action_space()
            self._apply_pass_mode_to_policy(self.opponent_policy)
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
            self._recondition_intent_fields_for_role(opponent_obs, opponent_is_offense)
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

        opp_actions_raw = np.array(opp_actions_raw, dtype=int).reshape(-1)
        action_len = int(opp_actions_raw.shape[0])
        action_mask = np.asarray(action_mask)

        # Legacy checkpoints may emit full n_players actions while newer set-attention
        # checkpoints emit players_per_side actions. Normalize both to opponent team size.
        if action_len == len(self.opponent_player_ids):
            opp_pred_actions = opp_actions_raw
        elif action_len == int(getattr(self.env.unwrapped, "n_players", action_len)):
            opp_pred_actions = opp_actions_raw[self._opponent_player_indices]
        else:
            opp_pred_actions = opp_actions_raw[: len(self.opponent_player_ids)]

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
            np.array(opp_pred_actions),
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

    def set_intent_null_prob(self, value: float) -> None:  # pragma: no cover - thin shim
        try:
            self.env.unwrapped.set_intent_null_prob(float(value))
        except Exception:
            pass

    def set_intent_visible_to_defense_prob(
        self, value: float
    ) -> None:  # pragma: no cover - thin shim
        try:
            self.env.unwrapped.set_intent_visible_to_defense_prob(float(value))
        except Exception:
            pass

    def set_defense_intent_null_prob(
        self, value: float
    ) -> None:  # pragma: no cover - thin shim
        try:
            self.env.unwrapped.set_defense_intent_null_prob(float(value))
        except Exception:
            pass

    def set_offense_intent_state(
        self,
        intent_index: int,
        *,
        intent_active: bool = True,
        intent_age: int = 0,
        intent_commitment_remaining: Optional[int] = None,
    ) -> None:  # pragma: no cover - thin shim
        try:
            self.env.unwrapped.set_offense_intent_state(
                int(intent_index),
                intent_active=bool(intent_active),
                intent_age=int(intent_age),
                intent_commitment_remaining=intent_commitment_remaining,
            )
            if (
                getattr(self.env.unwrapped, "training_team", None) == Team.OFFENSE
                and isinstance(getattr(self, "_last_obs", None), dict)
            ):
                self._recondition_intent_fields_for_role(self._last_obs, True)
        except Exception:
            pass

    def get_offense_intent_override(self):  # pragma: no cover - thin shim
        try:
            getter = getattr(self.env.unwrapped, "get_offense_intent_override", None)
            if callable(getter):
                return getter()
        except Exception:
            pass
        return None

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
