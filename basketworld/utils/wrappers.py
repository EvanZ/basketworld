from __future__ import annotations

import gymnasium as gym
import numpy as np
from basketworld.envs.basketworld_env_v2 import Team


class SetObservationWrapper(gym.ObservationWrapper):
    """Expose set-based player tokens + globals while preserving existing obs keys."""

    _TOKEN_DIM = 11
    _GLOBAL_DIM = 3

    def __init__(self, env: gym.Env):
        super().__init__(env)
        n_players = int(self.env.unwrapped.n_players)

        if isinstance(env.observation_space, gym.spaces.Dict):
            spaces_dict = dict(env.observation_space.spaces)
        else:
            spaces_dict = {"obs": env.observation_space}

        spaces_dict["players"] = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_players, self._TOKEN_DIM),
            dtype=np.float32,
        )
        spaces_dict["globals"] = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._GLOBAL_DIM,),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Dict(spaces_dict)

    def observation(self, obs):  # type: ignore[override]
        if isinstance(obs, dict):
            obs_dict = dict(obs)
        else:
            obs_dict = {"obs": obs}

        env = self.env.unwrapped
        n_players = int(env.n_players)
        norm_den = float(max(env.court_width, env.court_height)) or 1.0
        if not getattr(env, "normalize_obs", True):
            norm_den = 1.0

        skills_by_player: dict[int, np.ndarray] = {}
        if hasattr(env, "offense_layup_pct_by_player"):
            for idx, pid in enumerate(env.offense_ids):
                skills_by_player[int(pid)] = np.array(
                    [
                        float(env.offense_layup_pct_by_player[idx]),
                        float(env.offense_three_pt_pct_by_player[idx]),
                        float(env.offense_dunk_pct_by_player[idx]),
                    ],
                    dtype=np.float32,
                )
        else:
            base_layup = float(getattr(env, "layup_pct", 0.0))
            base_three = float(getattr(env, "three_pt_pct", 0.0))
            base_dunk = float(getattr(env, "dunk_pct", 0.0))
            skills = obs_dict.get("skills")
            if skills is not None:
                skills_arr = np.asarray(skills, dtype=np.float32).reshape(-1)
                for idx, pid in enumerate(env.offense_ids):
                    start = idx * 3
                    if start + 2 < len(skills_arr):
                        skills_by_player[int(pid)] = np.array(
                            [
                                base_layup + float(skills_arr[start]),
                                base_three + float(skills_arr[start + 1]),
                                base_dunk + float(skills_arr[start + 2]),
                            ],
                            dtype=np.float32,
                        )

        max_lane_steps = float(getattr(env, "three_second_max_steps", 1) or 1)
        players = np.zeros((n_players, self._TOKEN_DIM), dtype=np.float32)
        offense_ids = set(getattr(env, "offense_ids", []))
        expected_points = {}
        try:
            ep_values = env.calculate_expected_points_all_players()
            for idx, pid in enumerate(env.offense_ids):
                if idx < len(ep_values):
                    expected_points[int(pid)] = float(ep_values[idx])
        except Exception:
            expected_points = {}
        turnover_probs: dict[int, float] = {}
        steal_risks: dict[int, float] = {}
        try:
            ball_holder = getattr(env, "ball_holder", None)
            if ball_holder is not None and ball_holder in offense_ids:
                turnover_prob = float(env.calculate_defender_pressure_turnover_probability())
                turnover_probs[int(ball_holder)] = turnover_prob
                steal_probs = env.calculate_pass_steal_probabilities(ball_holder)
                for offense_id in offense_ids:
                    if offense_id == ball_holder:
                        continue
                    steal_risks[int(offense_id)] = float(steal_probs.get(offense_id, 0.0))
        except Exception:
            turnover_probs = {}
            steal_risks = {}
        for pid in range(n_players):
            q, r = env.positions[pid]
            role = 1.0 if pid in offense_ids else -1.0
            has_ball = 1.0 if env.ball_holder == pid else 0.0
            skill_vec = skills_by_player.get(pid, np.zeros(3, dtype=np.float32))
            if pid in offense_ids:
                lane_steps = env._offensive_lane_steps.get(pid, 0)
            else:
                lane_steps = env._defender_in_key_steps.get(pid, 0)
            lane_steps_norm = float(lane_steps) / max_lane_steps
            if lane_steps_norm > 1.0:
                lane_steps_norm = 1.0
            ep_value = expected_points.get(pid, 0.0) if pid in offense_ids else 0.0
            turnover_prob = turnover_probs.get(pid, 0.0) if pid in offense_ids else 0.0
            steal_risk = steal_risks.get(pid, 0.0) if pid in offense_ids else 0.0

            players[pid] = np.array(
                [
                    float(q) / norm_den,
                    float(r) / norm_den,
                    role,
                    has_ball,
                    float(skill_vec[0]) if skill_vec.size > 0 else 0.0,
                    float(skill_vec[1]) if skill_vec.size > 1 else 0.0,
                    float(skill_vec[2]) if skill_vec.size > 2 else 0.0,
                    lane_steps_norm,
                    ep_value,
                    turnover_prob,
                    steal_risk,
                ],
                dtype=np.float32,
            )

        hoop_q, hoop_r = env.basket_position
        globals_vec = np.array(
            [
                float(env.shot_clock),
                float(hoop_q) / norm_den,
                float(hoop_r) / norm_den,
            ],
            dtype=np.float32,
        )
        obs_dict["players"] = players
        obs_dict["globals"] = globals_vec
        return obs_dict


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
      potential_assisted_dunk, potential_assisted_2pt, potential_assisted_3pt,
      passes, turnover, turnover_pass_oob, turnover_intercepted, turnover_pressure,
      turnover_offensive_lane, defensive_lane_violation, move_rejected_occupied,
      made_dunk, made_2pt, made_3pt, attempts
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
        self._potential_asst_dunk = 0.0
        self._potential_asst_2pt = 0.0
        self._potential_asst_3pt = 0.0
        self._potential_assists = 0.0
        self._turnover = 0.0
        self._turnover_pass_oob = 0.0
        self._turnover_intercepted = 0.0
        self._turnover_pressure = 0.0
        self._turnover_offensive_lane = 0.0
        self._defensive_lane_violation = 0.0
        self._move_rejected_occupied = 0.0
        self._made_dunk = 0.0
        self._made_2pt = 0.0
        self._made_3pt = 0.0
        self._attempts = 0.0
        # Minimal audit
        self._gt_is_three = 0.0
        self._gt_is_dunk = 0.0
        self._gt_points = 0.0
        self._gt_shooter_off = 0.0
        self._gt_shooter_q = 0.0
        self._gt_shooter_r = 0.0
        self._gt_distance = -1.0
        self._basket_q = 0.0
        self._basket_r = 0.0
        # Pressure-adjusted FG% at time of shot (for CSV logging)
        self._shooter_fg_pct = -1.0

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
                potential_assist = bool(shot_res.get("assist_potential")) and not bool(
                    shot_res.get("success")
                )
                if "is_three" in shot_res:
                    is_three = bool(shot_res["is_three"])
                else:
                    is_three = self.env.unwrapped.is_three_point_location(shooter_pos)
                self._attempts = 1.0
                self._gt_is_three = 1.0 if (not is_dunk and is_three) else 0.0
                self._gt_is_dunk = 1.0 if is_dunk else 0.0
                self._gt_points = 2.0 if (is_dunk or not is_three) else 3.0
                # Capture pressure-adjusted FG% from env shot result if present
                try:
                    self._shooter_fg_pct = float(shot_res.get("probability", -1.0))
                except Exception:
                    self._shooter_fg_pct = -1.0
                self._gt_shooter_off = (
                    1.0 if shooter_id in self.env.unwrapped.offense_ids else 0.0
                )
                try:
                    self._gt_shooter_q = float(shooter_pos[0])
                    self._gt_shooter_r = float(shooter_pos[1])
                    self._gt_distance = float(dist)
                    self._basket_q = float(self.env.unwrapped.basket_position[0])
                    self._basket_r = float(self.env.unwrapped.basket_position[1])
                except Exception:
                    self._gt_shooter_q = 0.0
                    self._gt_shooter_r = 0.0
                    self._gt_distance = -1.0
                    try:
                        self._basket_q = float(self.env.unwrapped.basket_position[0])
                        self._basket_r = float(self.env.unwrapped.basket_position[1])
                    except Exception:
                        self._basket_q = 0.0
                        self._basket_r = 0.0
                if is_dunk:
                    self._shot_dunk = 1.0
                    if shot_res.get("success"):
                        self._made_dunk = 1.0
                    if shot_res.get("assist_full") and shot_res.get("success"):
                        self._asst_dunk = 1.0
                    if potential_assist:
                        self._potential_asst_dunk = 1.0
                elif is_three:
                    self._shot_3pt = 1.0
                    if shot_res.get("success"):
                        self._made_3pt = 1.0
                    if shot_res.get("assist_full") and shot_res.get("success"):
                        self._asst_3pt = 1.0
                    if potential_assist:
                        self._potential_asst_3pt = 1.0
                else:
                    self._shot_2pt = 1.0
                    if shot_res.get("success"):
                        self._made_2pt = 1.0
                    if shot_res.get("assist_full") and shot_res.get("success"):
                        self._asst_2pt = 1.0
                    if potential_assist:
                        self._potential_asst_2pt = 1.0
                if potential_assist:
                    self._potential_assists = 1.0
            elif ar.get("turnovers"):
                self._turnover = 1.0
                # Track specific turnover types
                for turnover in ar["turnovers"]:
                    reason = turnover.get("reason", "")
                    if reason == "pass_out_of_bounds":
                        self._turnover_pass_oob = 1.0
                    elif reason == "intercepted":
                        self._turnover_intercepted = 1.0
                    elif reason == "defender_pressure":
                        self._turnover_pressure = 1.0
                    elif reason == "offensive_three_seconds":
                        self._turnover_offensive_lane = 1.0
            
            # Track defensive lane violations (separate from turnovers)
            if ar.get("defensive_lane_violations"):
                self._defensive_lane_violation = float(len(ar["defensive_lane_violations"]))
            if ar.get("moves"):
                for move_res in ar["moves"].values():
                    if (
                        not move_res.get("success", False)
                        and move_res.get("reason") == "occupied_neighbor"
                    ):
                        self._move_rejected_occupied += 1.0
        except Exception:
            pass

        if done and info is not None:
            info["shot_dunk"] = self._shot_dunk
            info["shot_2pt"] = self._shot_2pt
            info["shot_3pt"] = self._shot_3pt
            info["assisted_dunk"] = self._asst_dunk
            info["assisted_2pt"] = self._asst_2pt
            info["assisted_3pt"] = self._asst_3pt
            info["potential_assisted_dunk"] = self._potential_asst_dunk
            info["potential_assisted_2pt"] = self._potential_asst_2pt
            info["potential_assisted_3pt"] = self._potential_asst_3pt
            info["potential_assists"] = self._potential_assists
            info["passes"] = float(self._passes)
            info["turnover"] = self._turnover
            info["turnover_pass_oob"] = self._turnover_pass_oob
            info["turnover_intercepted"] = self._turnover_intercepted
            info["turnover_pressure"] = self._turnover_pressure
            info["turnover_offensive_lane"] = self._turnover_offensive_lane
            info["defensive_lane_violation"] = self._defensive_lane_violation
            info["move_rejected_occupied"] = self._move_rejected_occupied
            info["made_dunk"] = self._made_dunk
            info["made_2pt"] = self._made_2pt
            info["made_3pt"] = self._made_3pt
            info["attempts"] = self._attempts
            info["gt_is_three"] = self._gt_is_three
            info["gt_is_dunk"] = self._gt_is_dunk
            info["gt_points"] = self._gt_points
            info["gt_shooter_off"] = self._gt_shooter_off
            info["gt_shooter_q"] = self._gt_shooter_q
            info["gt_shooter_r"] = self._gt_shooter_r
            info["gt_distance"] = self._gt_distance
            info["basket_q"] = self._basket_q
            info["basket_r"] = self._basket_r
            info["shooter_fg_pct"] = self._shooter_fg_pct
            # (Reverted) keep only the original episode summary fields
        return obs, reward, done, truncated, info


class EnvIndexWrapper(gym.Wrapper):
    """Add environment index to episode info for mixed training metrics filtering.
    
    When using mixed environments (simultaneous offense/defense training),
    AccumulativeMetricsCallback needs to filter episodes by environment index.
    This wrapper adds the env_idx to each episode's info dict.
    """
    
    def __init__(self, env: gym.Env, env_idx: int):
        super().__init__(env)
        self.env_idx = env_idx
    
    def step(self, action):  # type: ignore[override]
        obs, reward, done, truncated, info = self.env.step(action)
        if done and info is not None:
            info["env_idx"] = self.env_idx
        return obs, reward, done, truncated, info


class BetaSetterWrapper(gym.Wrapper):
    """Shim wrapper to expose set_phi_beta at the top level to avoid Gym warnings.

    Gymnasium warns when attribute forwarding traverses wrappers. By defining
    set_phi_beta here, VecEnv.env_method("set_phi_beta", value) will call this
    method directly on the top-most wrapper without deprecated forwarding.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def set_phi_beta(self, value: float) -> None:  # pragma: no cover
        try:
            self.env.unwrapped.set_phi_beta(float(value))
        except Exception:
            pass

    def set_pass_arc_degrees(self, value: float) -> None:  # pragma: no cover
        try:
            self.env.unwrapped.set_pass_arc_degrees(float(value))
        except Exception:
            pass

    def set_pass_oob_turnover_prob(self, value: float) -> None:  # pragma: no cover
        try:
            self.env.unwrapped.set_pass_oob_turnover_prob(float(value))
        except Exception:
            pass

    def set_pass_target_strategy(self, strategy: str) -> None:  # pragma: no cover
        try:
            self.env.unwrapped.set_pass_target_strategy(strategy)
        except Exception:
            pass

    def get_profile_stats(self):  # pragma: no cover
        """Expose profiling stats without deprecated wrapper forwarding."""
        try:
            return self.env.unwrapped.get_profile_stats()
        except Exception:
            return {}

    def reset_profile_stats(self) -> None:  # pragma: no cover
        """Reset profiling stats without deprecated wrapper forwarding."""
        try:
            self.env.unwrapped.reset_profile_stats()
        except Exception:
            pass
