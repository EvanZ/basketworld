from __future__ import annotations

from typing import Optional, Any

import gymnasium as gym
import numpy as np
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

    def _ensure_opponent_loaded(self):
        # Allow passing a policy path string to avoid pickling full PPO
        if isinstance(self.opponent_policy, str):
            self.opponent_policy = PPO.load(self.opponent_policy, device="cpu")

    def _set_team_ids(self) -> None:
        if self.env.unwrapped.training_team == Team.OFFENSE:
            self.training_player_ids = self.env.unwrapped.offense_ids
            self.opponent_player_ids = self.env.unwrapped.defense_ids
        else:
            self.training_player_ids = self.env.unwrapped.defense_ids
            self.opponent_player_ids = self.env.unwrapped.offense_ids

    def reset(self, **kwargs):
        try:
            obs, info = self.env.reset(**kwargs)
            self._last_obs = obs
            self._set_team_ids()
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
                else np.array([0.0], dtype=np.float32)
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

        # Resolve opponent illegal actions
        try:
            opp_probs = get_policy_action_probabilities(
                self.opponent_policy, opponent_obs
            )
        except Exception:
            opp_probs = None
        opp_actions = resolve_illegal_actions(
            np.array(opp_actions_raw),
            action_mask,
            self.opponent_strategy,
            self.deterministic_opponent,
            opp_probs,
        )

        # Build full action vector
        full_action = np.zeros(self.env.unwrapped.n_players, dtype=int)
        for i in range(self.env.unwrapped.n_players):
            if i in self.training_player_ids:
                full_action[i] = int(action[i])
            else:
                full_action[i] = int(opp_actions[i])

        try:
            obs, reward, done, truncated, info = self.env.step(full_action)
        except Exception as e:
            raise RuntimeError(
                f"SelfPlayEnvWrapper.step env.step failed: {type(e).__name__}: {e}"
            ) from e
        # (Reverted) do not mutate info here
        self._last_obs = obs
        return obs, reward, done, truncated, info
