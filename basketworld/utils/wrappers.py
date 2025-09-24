from __future__ import annotations

import gymnasium as gym
import numpy as np
from basketworld.envs.basketworld_env_v2 import Team


class RewardAggregationWrapper(gym.Wrapper):
    """Aggregate team rewards so the Monitor sees a single reward per step.

    Sums rewards for the team currently being trained.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def step(self, action):  # type: ignore[override]
        obs, rewards, done, truncated, info = self.env.step(action)

        if self.env.unwrapped.training_team == Team.OFFENSE:
            training_player_ids = self.env.unwrapped.offense_ids
        else:
            training_player_ids = self.env.unwrapped.defense_ids

        aggregated_reward = sum(rewards[i] for i in training_player_ids)
        return obs, aggregated_reward, done, truncated, info


class EpisodeStatsWrapper(gym.Wrapper):
    """Collect per-episode stats needed for logging/PPP and expose via info.

    Exposes keys consumed by Monitor(info_keywords=...):
      shot_dunk, shot_2pt, shot_3pt, assisted_dunk, assisted_2pt, assisted_3pt,
      passes, turnover, made_dunk, made_2pt, made_3pt, attempts
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._reset_stats()

    def _reset_stats(self):
        self._passes = 0
        self._shot_dunk = 0.0
        self._shot_2pt = 0.0
        self._shot_3pt = 0.0
        self._asst_dunk = 0.0
        self._asst_2pt = 0.0
        self._asst_3pt = 0.0
        self._turnover = 0.0
        self._made_dunk = 0.0
        self._made_2pt = 0.0
        self._made_3pt = 0.0
        self._attempts = 0.0

    def reset(self, **kwargs):  # type: ignore[override]
        self._reset_stats()
        return self.env.reset(**kwargs)

    def step(self, action):  # type: ignore[override]
        obs, reward, done, truncated, info = self.env.step(action)
        try:
            ar = info.get("action_results", {}) if info else {}
            if ar.get("passes"):
                self._passes += int(len(ar["passes"]))

            if ar.get("shots"):
                shot_res = list(ar["shots"].values())[0]
                shooter_id = int(list(ar["shots"].keys())[0])
                shooter_pos = self.env.unwrapped.positions[shooter_id]
                dist = self.env.unwrapped._hex_distance(
                    shooter_pos, self.env.unwrapped.basket_position
                )
                is_dunk = dist == 0
                is_three = dist >= getattr(
                    self.env.unwrapped, "three_point_distance", 4
                )
                self._attempts = 1.0
                if is_dunk:
                    self._shot_dunk = 1.0
                    if shot_res.get("success"):
                        self._made_dunk = 1.0
                    if shot_res.get("assist_full") and shot_res.get("success"):
                        self._asst_dunk = 1.0
                elif is_three:
                    self._shot_3pt = 1.0
                    if shot_res.get("success"):
                        self._made_3pt = 1.0
                    if shot_res.get("assist_full") and shot_res.get("success"):
                        self._asst_3pt = 1.0
                else:
                    self._shot_2pt = 1.0
                    if shot_res.get("success"):
                        self._made_2pt = 1.0
                    if shot_res.get("assist_full") and shot_res.get("success"):
                        self._asst_2pt = 1.0
            elif ar.get("turnovers"):
                self._turnover = 1.0
        except Exception:
            pass

        if done and info is not None:
            info["shot_dunk"] = self._shot_dunk
            info["shot_2pt"] = self._shot_2pt
            info["shot_3pt"] = self._shot_3pt
            info["assisted_dunk"] = self._asst_dunk
            info["assisted_2pt"] = self._asst_2pt
            info["assisted_3pt"] = self._asst_3pt
            info["passes"] = float(self._passes)
            info["turnover"] = self._turnover
            info["made_dunk"] = self._made_dunk
            info["made_2pt"] = self._made_2pt
            info["made_3pt"] = self._made_3pt
            info["attempts"] = self._attempts
        return obs, reward, done, truncated, info
