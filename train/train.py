#!/usr/bin/env python3
"""
Main training script for the HexagonBasketballEnv using an alternating
self-play strategy with Proximal Policy Optimization (PPO).

This script implements a self-play loop where two policies, one for offense
and one for defense, are trained against each other in an alternating fashion.
A custom gym.Wrapper is used to manage the opponent's actions during training.
"""
import argparse
import os
from datetime import datetime
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger, HumanOutputFormat
from basketworld.utils.mlflow_logger import MLflowWriter
from basketworld.utils.callbacks import RolloutUpdateTimingCallback

from basketworld.utils.evaluation_helpers import (
    get_outcome_category,
    create_and_log_gif,
)
import imageio
from collections import defaultdict
import re

import mlflow
import sys
import tempfile
import random
from typing import Optional
import torch

import basketworld
from basketworld.envs.basketworld_env_v2 import Team

# --- CPU thread caps to avoid oversubscription in parallel env workers ---
# These defaults can be overridden by user environment.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


# --- GPU Configuration ---
# Check if CUDA is available and configure device
def get_device(device_arg):
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    elif device_arg == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("CUDA requested but not available, falling back to CPU")
            return torch.device("cpu")
    else:
        return torch.device(device_arg)


# Device will be set in main() after parsing args


def linear_schedule(start, end):
    def f(progress_remaining: float):
        return end + (start - end) * progress_remaining

    return f


# --- Custom MLflow Callback ---


class MLflowCallback(BaseCallback):
    """
    A custom callback for logging metrics to MLflow.
    This callback logs the mean reward and episode length periodically.
    """

    def __init__(self, team_name: str, log_freq: int = 2048, verbose=0):
        super(MLflowCallback, self).__init__(verbose)
        self.team_name = team_name
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        # Log metrics periodically to avoid performance overhead
        if self.n_calls % self.log_freq == 0:
            # The ep_info_buffer contains info from the last 100 episodes
            if self.model.ep_info_buffer:
                # Use the current model's timesteps as global step
                global_step = self.model.num_timesteps

                # Calculate the mean reward and length
                ep_rew_mean = np.mean(
                    [ep_info.get("r", 0.0) for ep_info in self.model.ep_info_buffer]
                )
                ep_len_mean = np.mean(
                    [ep_info.get("l", 0.0) for ep_info in self.model.ep_info_buffer]
                )

                # Optional shot/pass metrics if present in episode infos
                def mean_key(key: str, default: float = 0.0):
                    vals = [
                        ep.get(key) for ep in self.model.ep_info_buffer if key in ep
                    ]
                    return float(np.mean(vals)) if vals else default

                shot_dunk_pct = mean_key("shot_dunk")
                shot_2pt_pct = mean_key("shot_2pt")
                shot_3pt_pct = mean_key("shot_3pt")
                asst_dunk_pct = mean_key("assisted_dunk")
                asst_2pt_pct = mean_key("assisted_2pt")
                asst_3pt_pct = mean_key("assisted_3pt")
                passes_avg = mean_key("passes")
                turnover_pct = mean_key("turnover")

                # PPP per episode: (2*made_2pt + 3*made_3pt + 2*made_dunk) / (attempts + turnovers)
                def mean_ppp(default: float = 0.0):
                    numer = []
                    denom = []
                    for ep in self.model.ep_info_buffer:
                        m2 = float(ep.get("made_2pt", 0.0))
                        m3 = float(ep.get("made_3pt", 0.0))
                        md = float(ep.get("made_dunk", 0.0))
                        att = float(ep.get("attempts", 0.0))
                        tov = float(ep.get("turnover", 0.0))
                        n = (2.0 * m2) + (3.0 * m3) + (2.0 * md)
                        d = max(
                            1.0, att + tov
                        )  # avoid div-by-zero; count turnovers as possessions
                        numer.append(n / d)
                        denom.append(1.0)
                    return float(np.mean(numer)) if numer else default

                ppp_avg = mean_ppp()

                # Log to MLflow
                mlflow.log_metric(
                    f"{self.team_name} Mean Episode Reward",
                    ep_rew_mean,
                    step=global_step,
                )
                mlflow.log_metric(
                    f"{self.team_name} Mean Episode Length",
                    ep_len_mean,
                    step=global_step,
                )
                # Entropy coefficient (supports constant or schedule)
                try:
                    total_ts = getattr(self.model, "_total_timesteps", None)
                    if callable(getattr(self.model, "ent_coef", None)):
                        # Approximate current progress remaining consistent with SB3 definition
                        progress_remaining = 1.0
                        if total_ts and total_ts > 0:
                            progress_remaining = max(
                                0.0,
                                min(
                                    1.0,
                                    1.0 - (self.model.num_timesteps / float(total_ts)),
                                ),
                            )
                        current_ent_coef = float(
                            self.model.ent_coef(progress_remaining)
                        )
                    else:
                        current_ent_coef = float(getattr(self.model, "ent_coef", 0.0))
                    mlflow.log_metric(
                        f"{self.team_name} Entropy Coef",
                        current_ent_coef,
                        step=global_step,
                    )
                except Exception:
                    pass
                # Distributions
                mlflow.log_metric(
                    f"{self.team_name} ShotPct Dunk", shot_dunk_pct, step=global_step
                )
                mlflow.log_metric(
                    f"{self.team_name} ShotPct 2PT", shot_2pt_pct, step=global_step
                )
                mlflow.log_metric(
                    f"{self.team_name} ShotPct 3PT", shot_3pt_pct, step=global_step
                )
                mlflow.log_metric(
                    f"{self.team_name} Assist ShotPct Dunk",
                    asst_dunk_pct,
                    step=global_step,
                )
                mlflow.log_metric(
                    f"{self.team_name} Assist ShotPct 2PT",
                    asst_2pt_pct,
                    step=global_step,
                )
                mlflow.log_metric(
                    f"{self.team_name} Assist ShotPct 3PT",
                    asst_3pt_pct,
                    step=global_step,
                )
                mlflow.log_metric(
                    f"{self.team_name} Passes / Episode", passes_avg, step=global_step
                )
                mlflow.log_metric(
                    f"{self.team_name} TurnoverPct", turnover_pct, step=global_step
                )
                mlflow.log_metric(f"{self.team_name} PPP", ppp_avg, step=global_step)
        return True


# --- Entropy Schedule Callback ---


class EntropyScheduleCallback(BaseCallback):
    """Linearly decay entropy coefficient from start to end across the entire run.

    This uses the cumulative `self.model.num_timesteps` and a provided
    `total_planned_timesteps` to compute global progress, then updates
    `self.model.ent_coef` in-place so the change persists across segments.
    """

    def __init__(self, start: float, end: float, total_planned_timesteps: int):
        super().__init__()
        self.start = float(start)
        self.end = float(end)
        self.total = int(max(1, total_planned_timesteps))

    def _on_step(self) -> bool:
        try:
            t = int(getattr(self.model, "num_timesteps", 0))
            progress = min(1.0, max(0.0, t / float(self.total)))
            current = self.end + (self.start - self.end) * (1.0 - progress)
            # Update the algorithm's entropy coefficient
            self.model.ent_coef = float(current)
        except Exception:
            pass
        return True


class EntropyExpScheduleCallback(BaseCallback):
    """Exponential (log-linear) decay of entropy coef with optional per-alternation bump.

    Schedule: ent(progress) = end * (start / end) ** (1 - progress)
    where progress = timesteps / total_planned_timesteps in [0,1].

    Bump: At the start of each alternation segment, call `start_new_alternation()`
    to apply a temporary multiplier to the scheduled entropy for the next
    `bump_updates` PPO updates (one update per rollout segment).
    """

    def __init__(
        self,
        start: float,
        end: float,
        total_planned_timesteps: int,
        bump_updates: int = 0,
        bump_multiplier: float = 1.0,
    ):
        super().__init__()
        self.start = float(max(1e-12, start))
        self.end = float(max(1e-12, end))
        self.total = int(max(1, total_planned_timesteps))
        self.bump_updates = int(max(0, bump_updates))
        self.bump_multiplier = float(max(1.0, bump_multiplier))
        self._bump_updates_remaining = 0

    def start_new_alternation(self):
        self._bump_updates_remaining = self.bump_updates

    def _scheduled_value(self, t: int) -> float:
        progress = min(1.0, max(0.0, t / float(self.total)))
        # Log-linear interpolation between start and end hitting end at progress=1 exactly
        ratio = self.start / self.end
        current = self.end * (ratio ** (1.0 - progress))
        return float(current)

    def _apply_current(self):
        try:
            t = int(getattr(self.model, "num_timesteps", 0))
            current = self._scheduled_value(t)
            if self._bump_updates_remaining > 0:
                current = float(current * self.bump_multiplier)
            self.model.ent_coef = float(current)
        except Exception:
            pass

    def _on_training_start(self) -> None:
        self._apply_current()

    def _on_rollout_start(self) -> None:
        # Ensure ent coef is set before update happens after rollout
        self._apply_current()

    def _on_rollout_end(self) -> None:
        # One rollout finished -> one PPO update will follow; reduce bump if active
        if self._bump_updates_remaining > 0:
            self._bump_updates_remaining -= 1
        # Also refresh current value after potential bump decrement
        self._apply_current()

    def _on_step(self) -> bool:
        # Keep it updated during collection too (harmless)
        self._apply_current()
        return True


# --- Custom Reward Wrapper for Multi-Agent Aggregation ---


class RewardAggregationWrapper(gym.Wrapper):
    """
    A wrapper to aggregate multi-agent rewards for the Monitor wrapper.
    It sums the rewards of the team currently being trained.
    """

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, rewards, done, truncated, info = self.env.step(action)

        # Determine which player IDs belong to the training team
        if self.env.unwrapped.training_team == Team.OFFENSE:
            training_player_ids = self.env.unwrapped.offense_ids
        else:
            training_player_ids = self.env.unwrapped.defense_ids

        # Sum the rewards for only the players on the training team
        aggregated_reward = sum(rewards[i] for i in training_player_ids)

        return obs, aggregated_reward, done, truncated, info


# --- Episode Statistics Wrapper ---
class EpisodeStatsWrapper(gym.Wrapper):
    """Track per-episode shot/pass distributions and expose via info['episode'].

    - shot_dunk/shot_2pt/shot_3pt: 1.0 if that shot type occurred in the episode, else 0.0
    - assisted_dunk/assisted_2pt/assisted_3pt: 1.0 if the made shot in that type was assisted
    - passes: total number of pass attempts in the episode
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
        # For PPP computation: made buckets and attempts
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
            # Pass attempts
            if ar.get("passes"):
                self._passes += int(len(ar["passes"]))

            # Shots happen at episode end in this env
            if ar.get("shots"):
                # Assume single shot per episode
                shot_res = list(ar["shots"].values())[0]
                # Determine distance category using env to avoid stale data
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
                # Episode ends via turnover
                self._turnover = 1.0
        except Exception:
            pass

        if done and info is not None:
            # Expose stats as top-level info keys; Monitor(info_keywords=...) will include them in episode info
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


# --- Custom Environment Wrapper for Self-Play ---


class SelfPlayEnvWrapper(gym.Wrapper):
    """
    A wrapper that manages the opponent's policy in a self-play setup.

    When the learning agent takes a step, this wrapper intercepts the action,
    gets an action from the frozen opponent policy, combines them, and passes
    the full action to the underlying environment.
    """

    def __init__(self, env, opponent_policy, deterministic_opponent: bool = False):
        super().__init__(env)
        self.opponent_policy = opponent_policy
        self.deterministic_opponent = bool(deterministic_opponent)
        self._set_team_ids()

    def _set_team_ids(self):
        """Determine which player IDs belong to the training team and opponent."""
        if self.env.unwrapped.training_team == Team.OFFENSE:
            self.training_player_ids = self.env.unwrapped.offense_ids
            self.opponent_player_ids = self.env.unwrapped.defense_ids
        else:
            self.training_player_ids = self.env.unwrapped.defense_ids
            self.opponent_player_ids = self.env.unwrapped.offense_ids

    def reset(self, **kwargs):
        """Reset the environment and store the initial observation."""
        obs, info = self.env.reset(**kwargs)
        self.last_obs = obs
        return obs, info

    def step(self, action):
        """
        Take a step in the environment.
        'action' comes from the learning agent and is for ALL players.
        We replace the opponent's actions with predictions from the frozen policy.
        """
        # Get action from the frozen opponent policy using the last observation
        # IMPORTANT: Flip the unified role flag for the opponent, so it
        # conditions on the opposite role (defense vs offense). The role flag
        # is appended as the last element of the flat observation vector.
        opponent_obs = self.last_obs
        try:
            # Create a shallow copy of the dict to avoid mutating self.last_obs
            opponent_obs = {
                "obs": np.copy(self.last_obs["obs"]),
                "action_mask": self.last_obs["action_mask"],
            }
            if opponent_obs["obs"].ndim == 1 and opponent_obs["obs"].size > 0:
                role_flag_idx = opponent_obs["obs"].size - 1
                # Flip 1.0 <-> 0.0
                opponent_obs["obs"][role_flag_idx] = (
                    1.0 - opponent_obs["obs"][role_flag_idx]
                )
        except Exception:
            # If anything goes wrong, fall back to original observation
            opponent_obs = self.last_obs

        opponent_action_raw, _ = self.opponent_policy.predict(
            opponent_obs, deterministic=self.deterministic_opponent
        )
        action_mask = self.last_obs["action_mask"]

        # Combine the actions, ensuring the opponent's actions are legal
        full_action = np.zeros(self.env.unwrapped.n_players, dtype=int)
        for i in range(self.env.unwrapped.n_players):
            if i in self.training_player_ids:
                full_action[i] = action[i]
            else:  # This is an opponent player
                predicted_action = opponent_action_raw[i]
                # Enforce the action mask for the opponent
                if action_mask[i][predicted_action] == 1:
                    full_action[i] = predicted_action
                else:
                    # Fallback: pick the first legal action for this player
                    legal_indices = np.where(action_mask[i] == 1)[0]
                    full_action[i] = (
                        int(legal_indices[0]) if len(legal_indices) > 0 else 0
                    )

        # Step the underlying environment with the combined action
        obs, reward, done, truncated, info = self.env.step(full_action)

        # Store the latest observation for the opponent's next decision
        self.last_obs = obs

        return obs, reward, done, truncated, info


class FrozenPolicyProxy:
    """Picklable proxy that loads a PPO policy from a local .zip path on first use.

    This avoids passing non-picklable PPO instances into subprocess workers.
    """

    def __init__(self, policy_path: str, device: torch.device | str = "cpu"):
        self.policy_path = str(policy_path)
        # Store a string for device to keep picklable
        self.device = str(device) if isinstance(device, torch.device) else device
        self._policy = None

    def _ensure_loaded(self):
        if self._policy is None:
            self._policy = PPO.load(self.policy_path, device=self.device)

    def predict(self, obs, deterministic: bool = False):
        self._ensure_loaded()
        return self._policy.predict(obs, deterministic=deterministic)


# --- Main Training Logic ---


def sample_geometric(indices: list[int], beta: float) -> int:
    """Return index sampled with decayed probability (newest highest)."""
    K = len(indices)
    # newest has i = K, oldest i=1
    weights = [(1 - beta) * (beta ** (K - i)) for i in range(1, K + 1)]
    total = sum(weights)
    probs = [w / total for w in weights]
    return random.choices(indices, weights=probs, k=1)[0]


def get_random_policy_from_artifacts(
    client,
    run_id,
    model_prefix,
    tmpdir,
    K: int = 20,
    beta: float = 0.8,
    uniform_eps: float = 0.10,
):
    """Sample an opponent checkpoint using a geometric decay over recent K snapshots.

    Args:
        client: MLflow client
        run_id: experiment run
        team_prefix: "offense" or "defense"
        tmpdir: temp dir to download artifact
        K: reservoir size (keep last K)
        beta: geometric decay factor (0<beta<1)
        uniform_eps: probability of picking uniformly among all snapshots.
    """
    artifact_path = "models"
    all_artifacts = client.list_artifacts(run_id, artifact_path)

    # Extract paths for prefix (e.g., unified)
    team_policies = [
        f.path
        for f in all_artifacts
        if f.path.startswith(f"{artifact_path}/{model_prefix}")
        and f.path.endswith(".zip")
    ]

    if not team_policies:
        return None

    # sort chronologically by alternation number embedded at end _<n>.zip
    def sort_key(p):
        m = re.search(r"_(\d+)\.zip$", p)
        return int(m.group(1)) if m else 0

    team_policies.sort(key=sort_key)

    # keep last K
    recent_pols = team_policies[-K:]

    # with small probability sample uniform over all for coverage
    if random.random() < uniform_eps:
        chosen = random.choice(team_policies)
    else:
        # geometric sampling over recent_pols
        # indices list 0..len-1 correspond to oldest..newest in recent_pols
        idx = sample_geometric(list(range(len(recent_pols))), beta)
        chosen = recent_pols[idx]

    print(f"  - Selected opponent policy: {os.path.basename(chosen)}")
    local_path = client.download_artifacts(run_id, chosen, tmpdir)
    return local_path


# --- Continuation helpers ---


def get_latest_policy_path(client, run_id: str, team_prefix: str) -> Optional[str]:
    artifact_path = "models"
    all_artifacts = client.list_artifacts(run_id, artifact_path)
    candidates = [
        f.path
        for f in all_artifacts
        if f.path.startswith(f"{artifact_path}/{team_prefix}")
        and f.path.endswith(".zip")
    ]
    if not candidates:
        return None

    def sort_key(p):
        m = re.search(r"_(\d+)\.zip$", p)
        return int(m.group(1)) if m else 0

    candidates.sort(key=sort_key)
    return candidates[-1]


def get_latest_unified_policy_path(client, run_id: str) -> Optional[str]:
    """Return latest unified policy artifact path if present."""
    artifact_path = "models"
    all_artifacts = client.list_artifacts(run_id, artifact_path)
    candidates = [
        f.path
        for f in all_artifacts
        if f.path.startswith(f"{artifact_path}/unified") and f.path.endswith(".zip")
    ]
    if not candidates:
        return None

    def sort_key(p):
        m = re.search(r"_(\d+)\.zip$", p)
        return int(m.group(1)) if m else 0

    candidates.sort(key=sort_key)
    return candidates[-1]


def get_max_alternation_index(client, run_id: str) -> int:
    """Return the max alternation index already present in the run (0 if none)."""
    artifact_path = "models"
    all_artifacts = client.list_artifacts(run_id, artifact_path)
    idxs = []
    for f in all_artifacts:
        m = re.search(r"_(\d+)\.zip$", f.path)
        if m:
            idxs.append(int(m.group(1)))
    return max(idxs) if idxs else 0


def setup_environment(args, training_team):
    """Create, configure, and wrap the environment for training."""
    env = basketworld.HexagonBasketballEnv(
        grid_size=args.grid_size,
        players=args.players,
        shot_clock_steps=args.shot_clock,
        min_shot_clock=getattr(args, "min_shot_clock", 10),
        defender_pressure_distance=args.defender_pressure_distance,
        defender_pressure_turnover_chance=args.defender_pressure_turnover_chance,
        three_point_distance=args.three_point_distance,
        layup_pct=args.layup_pct,
        layup_std=getattr(args, "layup_std", 0.0),
        three_pt_pct=args.three_pt_pct,
        three_pt_std=getattr(args, "three_pt_std", 0.0),
        allow_dunks=args.allow_dunks,
        dunk_pct=args.dunk_pct,
        dunk_std=getattr(args, "dunk_std", 0.0),
        shot_pressure_enabled=args.shot_pressure_enabled,
        shot_pressure_max=args.shot_pressure_max,
        shot_pressure_lambda=args.shot_pressure_lambda,
        shot_pressure_arc_degrees=args.shot_pressure_arc_degrees,
        # Reward shaping
        pass_reward=getattr(args, "pass_reward", 0.0),
        turnover_penalty=getattr(args, "turnover_penalty", 0.0),
        made_shot_reward_inside=getattr(args, "made_shot_reward_inside", 2.0),
        made_shot_reward_three=getattr(args, "made_shot_reward_three", 3.0),
        missed_shot_penalty=getattr(args, "missed_shot_penalty", 0.0),
        potential_assist_reward=getattr(args, "potential_assist_reward", 0.1),
        full_assist_bonus=getattr(args, "full_assist_bonus", 0.2),
        assist_window=getattr(args, "assist_window", getattr(args, "assist_window", 2)),
        potential_assist_pct=getattr(args, "potential_assist_pct", 0.10),
        full_assist_bonus_pct=getattr(args, "full_assist_bonus_pct", 0.05),
        enable_profiling=args.enable_env_profiling,
        training_team=training_team,  # Critical for correct rewards
        # Observation controls
        use_egocentric_obs=args.use_egocentric_obs,
        egocentric_rotate_to_hoop=args.egocentric_rotate_to_hoop,
        include_hoop_vector=args.include_hoop_vector,
        normalize_obs=args.normalize_obs,
        mask_occupied_moves=args.mask_occupied_moves,
        illegal_defense_enabled=args.illegal_defense_enabled,
        illegal_defense_max_steps=args.illegal_defense_max_steps,
    )
    # Wrap with episode stats collector then aggregate reward for Monitor/SB3
    env = EpisodeStatsWrapper(env)
    env = RewardAggregationWrapper(env)
    return Monitor(
        env,
        info_keywords=(
            "shot_dunk",
            "shot_2pt",
            "shot_3pt",
            "assisted_dunk",
            "assisted_2pt",
            "assisted_3pt",
            "passes",
            "turnover",
            # Keys required for PPP calculation
            "made_dunk",
            "made_2pt",
            "made_3pt",
            "attempts",
        ),
    )


# ----------------------------------------------------------
# Helper function to create a vectorized self-play environment
# ----------------------------------------------------------


def make_vector_env(
    args,
    training_team: Team,
    opponent_policy,
    num_envs: int,
    deterministic_opponent: bool,
) -> SubprocVecEnv:
    """Return a SubprocVecEnv with `num_envs` copies of the self-play environment.

    Each copy is wrapped with `SelfPlayEnvWrapper` so that the opponent's
    behaviour is provided by the frozen `opponent_policy`.
    """

    def _single_env_factory() -> gym.Env:  # type: ignore[name-defined]
        # We capture the current parameters via default args so that each lambda
        # has its own bound values (important inside list comprehension).
        return SelfPlayEnvWrapper(
            setup_environment(args, training_team),
            opponent_policy=opponent_policy,
            deterministic_opponent=deterministic_opponent,
        )

    # Use subprocesses for parallelism.
    return SubprocVecEnv(
        [_single_env_factory for _ in range(num_envs)], start_method="spawn"
    )


def main(args):
    """Main training function."""

    # --- Set up Device ---
    device = get_device(args.device)
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
    else:
        print("Using CPU")

    # --- Set up MLflow Tracking ---
    # MLflow requires a running server to log artifacts correctly.
    tracking_uri = "http://localhost:5000"
    mlflow.set_tracking_uri(tracking_uri)

    # Set the experiment name. This will create it if it doesn't exist.
    mlflow.set_experiment(args.mlflow_experiment_name)

    try:
        # Check if the server is reachable by trying to get the current experiment
        mlflow.get_experiment_by_name(args.mlflow_experiment_name)
    except mlflow.exceptions.MlflowException as e:
        print(
            f"Could not connect to MLflow tracking server at {tracking_uri}.",
            file=sys.stderr,
        )
        print(
            "Please ensure the MLflow UI server is running in a separate terminal with `mlflow ui`.",
            file=sys.stderr,
        )
        sys.exit(1)

    with mlflow.start_run(run_name=args.mlflow_run_name) as run:
        print("MLflow tracking URI:", mlflow.get_tracking_uri())
        # Log hyperparameters
        mlflow.log_params(vars(args))
        print(f"MLflow Run ID: {run.info.run_id}")

        # --- If continuing from a prior run, copy over prior model artifacts ---
        # This lets us sample frozen policies from the full history in the new run.
        if args.continue_run_id:
            try:
                client = mlflow.tracking.MlflowClient()
                prior = client.list_artifacts(args.continue_run_id, "models")
                current = client.list_artifacts(run.info.run_id, "models")
                current_names = {os.path.basename(f.path) for f in current}

                # Download and re-log any missing prior models into this run's models/ dir
                with tempfile.TemporaryDirectory() as _tmp_copy_dir:
                    for f in prior:
                        if not f.path.endswith(".zip"):
                            continue
                        base = os.path.basename(f.path)
                        if base in current_names:
                            continue
                        local_path = client.download_artifacts(
                            args.continue_run_id, f.path, _tmp_copy_dir
                        )
                        mlflow.log_artifact(local_path, artifact_path="models")
                print("Copied prior models from run", args.continue_run_id)
            except Exception as e:
                print("Warning: failed to copy prior models:", e)

        # --- Define Policy Kwargs ---
        # This allows us to set the network architecture from the command line.
        policy_kwargs = {}
        if args.net_arch is not None:
            policy_kwargs["net_arch"] = args.net_arch

        # The save_path is no longer needed as models are saved to a temp dir
        # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # save_path = os.path.join(args.save_path, f"basketworld_selfplay_{timestamp}")
        # os.makedirs(save_path, exist_ok=True)

        # --- Initialize Base Environment (just for policy creation) ---
        # The model must be created with the same number of parallel envs that will be
        # used later (SB3 stores this value internally).
        temp_env = DummyVecEnv(
            [
                (lambda: setup_environment(args, Team.OFFENSE))
                for _ in range(args.num_envs)
            ]
        )

        # --- Initialize Timing Callbacks ---
        offense_timing_callback = RolloutUpdateTimingCallback()
        defense_timing_callback = RolloutUpdateTimingCallback()

        print("Initializing unified policy...")
        unified_policy = None

        if args.continue_run_id:
            print(f"Continuing from run {args.continue_run_id}...")
            client = mlflow.tracking.MlflowClient()
            with tempfile.TemporaryDirectory() as tmpd:
                uni_art = get_latest_unified_policy_path(client, args.continue_run_id)
                if uni_art:
                    uni_local = client.download_artifacts(
                        args.continue_run_id, uni_art, tmpd
                    )
                    unified_policy = PPO.load(uni_local, env=temp_env, device=device)
                    print(
                        f"  - Loaded latest unified policy: {os.path.basename(uni_art)}"
                    )

        if unified_policy is None:
            # If an entropy schedule is requested, start PPO at the starting coefficient
            initial_ent_coef = args.ent_coef
            if (args.ent_coef_start is not None) or (args.ent_coef_end is not None):
                start = (
                    args.ent_coef_start
                    if args.ent_coef_start is not None
                    else args.ent_coef
                )
                initial_ent_coef = float(start)
            unified_policy = PPO(
                "MultiInputPolicy",
                temp_env,
                verbose=1,
                n_steps=args.n_steps,
                vf_coef=args.vf_coef,
                ent_coef=initial_ent_coef,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                tensorboard_log=None,  # Disable TensorBoard if using MLflow
                policy_kwargs=policy_kwargs,
                device=device,
            )
        temp_env.close()

        # --- Log the actual network architecture used ---
        # This ensures we capture the default if no custom arch is provided.
        actual_net_arch = str(unified_policy.policy.net_arch)
        mlflow.log_param("net_arch_used", actual_net_arch)
        print(f"  - Using network architecture: {actual_net_arch}")

        # --- Alternating Training Loop ---
        # Determine starting alternation index when continuing in-place
        base_alt_idx = 0
        if args.continue_run_id:
            base_alt_idx = get_max_alternation_index(
                mlflow.tracking.MlflowClient(), args.continue_run_id
            )
            print(f"Resuming alternations from index {base_alt_idx + 1}")

        # Create a persistent cache directory for opponent policy files used by workers
        opponent_cache_dir = os.path.join(".opponent_cache", run.info.run_id)
        os.makedirs(opponent_cache_dir, exist_ok=True)

        # Prepare optional entropy scheduler across the whole run
        entropy_callback = None
        if (args.ent_coef_start is not None) or (args.ent_coef_end is not None):
            ent_start = (
                args.ent_coef_start
                if args.ent_coef_start is not None
                else args.ent_coef
            )
            ent_end = args.ent_coef_end if args.ent_coef_end is not None else 0.0
            total_planned_ts = int(
                2
                * args.alternations
                * args.steps_per_alternation
                * args.num_envs
                * args.n_steps
            )
            if getattr(args, "ent_schedule", "linear") == "exp":
                entropy_callback = EntropyExpScheduleCallback(
                    ent_start,
                    ent_end,
                    total_planned_ts,
                    bump_updates=getattr(
                        args, "ent_bump_updates", getattr(args, "ent_bump_rollouts", 0)
                    ),
                    bump_multiplier=getattr(args, "ent_bump_multiplier", 1.0),
                )
            else:
                entropy_callback = EntropyScheduleCallback(
                    ent_start, ent_end, total_planned_ts
                )

        for i in range(args.alternations):
            print("-" * 50)
            global_alt = base_alt_idx + i + 1
            print(f"Alternation {global_alt} (segment {i + 1} / {args.alternations})")
            print("-" * 50)

            # --- Load a random historical opponent for this alternation ---
            print("\nLoading historical opponent policy...")
            opponent_for_offense = get_random_policy_from_artifacts(
                mlflow.tracking.MlflowClient(),
                run.info.run_id,
                "unified",
                opponent_cache_dir,
            )
            if opponent_for_offense is None:
                # Fallback: save current unified policy to a stable path
                fallback_path = os.path.join(opponent_cache_dir, "unified_latest.zip")
                unified_policy.save(fallback_path)
                opponent_for_offense = fallback_path

            # --- 1. Train Offense against frozen Defense ---
            print(f"\nTraining Offense...")
            offense_env = make_vector_env(
                args,
                training_team=Team.OFFENSE,
                opponent_policy=FrozenPolicyProxy(opponent_for_offense, device),
                num_envs=args.num_envs,
                deterministic_opponent=bool(args.deterministic_opponent),
            )
            unified_policy.set_env(offense_env)

            offense_mlflow_callback = MLflowCallback(
                team_name="Offense", log_freq=args.n_steps
            )

            offense_logger = Logger(
                folder=None,
                output_formats=[HumanOutputFormat(sys.stdout), MLflowWriter("Offense")],
            )
            unified_policy.set_logger(offense_logger)

            # Bump entropy at the start of each alternation segment if supported
            if entropy_callback is not None and hasattr(
                entropy_callback, "start_new_alternation"
            ):
                try:
                    entropy_callback.start_new_alternation()
                except Exception:
                    pass

            offense_callbacks = [offense_mlflow_callback, offense_timing_callback]
            if entropy_callback is not None:
                offense_callbacks.append(entropy_callback)
            unified_policy.learn(
                total_timesteps=args.steps_per_alternation
                * args.num_envs
                * args.n_steps,
                reset_num_timesteps=False,
                callback=offense_callbacks,
                progress_bar=True,
            )
            offense_env.close()

            print("\nLoading historical opponent policy...")
            opponent_for_defense = get_random_policy_from_artifacts(
                mlflow.tracking.MlflowClient(),
                run.info.run_id,
                "unified",
                opponent_cache_dir,
            )
            if opponent_for_defense is None:
                fallback_path = os.path.join(opponent_cache_dir, "unified_latest.zip")
                unified_policy.save(fallback_path)
                opponent_for_defense = fallback_path

            # --- 2. Train Defense against frozen Offense ---
            print(f"\nTraining Defense...")
            defense_env = make_vector_env(
                args,
                training_team=Team.DEFENSE,
                opponent_policy=FrozenPolicyProxy(opponent_for_defense, device),
                num_envs=args.num_envs,
                deterministic_opponent=bool(args.deterministic_opponent),
            )
            unified_policy.set_env(defense_env)

            defense_mlflow_callback = MLflowCallback(
                team_name="Defense", log_freq=args.n_steps
            )

            defense_logger = Logger(
                folder=None,
                output_formats=[HumanOutputFormat(sys.stdout), MLflowWriter("Defense")],
            )
            unified_policy.set_logger(defense_logger)

            # Bump entropy again at the start of the defense segment
            if entropy_callback is not None and hasattr(
                entropy_callback, "start_new_alternation"
            ):
                try:
                    entropy_callback.start_new_alternation()
                except Exception:
                    pass

            defense_callbacks = [defense_mlflow_callback, defense_timing_callback]
            if entropy_callback is not None:
                defense_callbacks.append(entropy_callback)
            unified_policy.learn(
                total_timesteps=args.steps_per_alternation
                * args.num_envs
                * args.n_steps,
                reset_num_timesteps=False,
                callback=defense_callbacks,
                progress_bar=True,
            )
            defense_env.close()

            # Save one unified checkpoint per alternation
            with tempfile.TemporaryDirectory() as tmpdir:
                unified_model_path = os.path.join(
                    tmpdir, f"unified_policy_alt_{global_alt}.zip"
                )
                unified_policy.save(unified_model_path)
                mlflow.log_artifact(unified_model_path, artifact_path="models")
            print(f"Logged unified model for alternation {global_alt} to MLflow")

            # --- 3. Run Evaluation Phase ---
            if args.eval_freq > 0 and (i + 1) % args.eval_freq == 0:
                print(f"\n--- Running Evaluation for Alternation {global_alt} ---")

                # Create a renderable environment for evaluation
                eval_env = basketworld.HexagonBasketballEnv(
                    grid_size=args.grid_size,
                    players=args.players,
                    shot_clock_steps=args.shot_clock,
                    min_shot_clock=getattr(args, "min_shot_clock", 10),
                    render_mode="rgb_array",
                    three_point_distance=args.three_point_distance,
                    layup_pct=args.layup_pct,
                    layup_std=getattr(args, "layup_std", 0.0),
                    three_pt_pct=args.three_pt_pct,
                    three_pt_std=getattr(args, "three_pt_std", 0.0),
                    allow_dunks=args.allow_dunks,
                    dunk_pct=args.dunk_pct,
                    dunk_std=getattr(args, "dunk_std", 0.0),
                    shot_pressure_enabled=args.shot_pressure_enabled,
                    shot_pressure_max=args.shot_pressure_max,
                    shot_pressure_lambda=args.shot_pressure_lambda,
                    shot_pressure_arc_degrees=args.shot_pressure_arc_degrees,
                    # Reward shaping
                    pass_reward=getattr(args, "pass_reward", 0.0),
                    turnover_penalty=getattr(args, "turnover_penalty", 0.0),
                    made_shot_reward_inside=getattr(
                        args, "made_shot_reward_inside", 2.0
                    ),
                    made_shot_reward_three=getattr(args, "made_shot_reward_three", 3.0),
                    missed_shot_penalty=getattr(args, "missed_shot_penalty", 0.0),
                    potential_assist_reward=getattr(
                        args, "potential_assist_reward", 0.1
                    ),
                    full_assist_bonus=getattr(args, "full_assist_bonus", 0.2),
                    assist_window=getattr(args, "assist_window", 2),
                    potential_assist_pct=getattr(args, "potential_assist_pct", 0.10),
                    full_assist_bonus_pct=getattr(args, "full_assist_bonus_pct", 0.05),
                    enable_profiling=args.enable_env_profiling,
                    # Observation controls
                    use_egocentric_obs=args.use_egocentric_obs,
                    egocentric_rotate_to_hoop=args.egocentric_rotate_to_hoop,
                    include_hoop_vector=args.include_hoop_vector,
                    normalize_obs=args.normalize_obs,
                    mask_occupied_moves=args.mask_occupied_moves,
                )

                with tempfile.TemporaryDirectory() as temp_dir:
                    for ep_num in range(args.eval_episodes):
                        obs, info = eval_env.reset()
                        done = False
                        episode_frames = []

                        while not done:
                            # Single unified policy chooses all actions
                            full_action, _ = unified_policy.predict(
                                obs, deterministic=True
                            )
                            obs, reward, done, _, info = eval_env.step(full_action)
                            frame = eval_env.render()
                            episode_frames.append(frame)

                        # Post-episode analysis to determine outcome
                        final_info = info
                        action_results = final_info.get("action_results", {})
                        outcome = "Unknown"  # Default outcome

                        if action_results.get("shots"):
                            shooter_id = list(action_results["shots"].keys())[0]
                            shot_result = list(action_results["shots"].values())[0]
                            # Determine 2 or 3 based on position at shot
                            shooter_pos = eval_env.positions[int(shooter_id)]
                            bq, br = eval_env.basket_position
                            dist = (
                                abs(shooter_pos[0] - bq)
                                + abs((shooter_pos[0] + shooter_pos[1]) - (bq + br))
                                + abs(shooter_pos[1] - br)
                            ) // 2
                            is_three = dist >= getattr(
                                eval_env, "three_point_distance", 4
                            )
                            if shot_result["success"]:
                                outcome = "Made 3" if is_three else "Made 2"
                            else:
                                outcome = "Missed 3" if is_three else "Missed 2"
                        elif action_results.get("turnovers"):
                            turnover_reason = action_results["turnovers"][0]["reason"]
                            if turnover_reason == "intercepted":
                                outcome = "Turnover (Intercepted)"
                            elif turnover_reason == "pass_out_of_bounds":
                                outcome = "Turnover (OOB)"
                            elif turnover_reason == "move_out_of_bounds":
                                outcome = "Turnover (OOB)"
                            elif turnover_reason == "defender_pressure":
                                outcome = "Turnover (Pressure)"
                        elif eval_env.unwrapped.shot_clock <= 0:
                            outcome = "Turnover (Shot Clock Violation)"

                        # Define the artifact path for this specific evaluation context
                        artifact_path = f"training_eval/alternation_{global_alt}"
                        create_and_log_gif(
                            frames=episode_frames,
                            episode_num=ep_num,
                            outcome=outcome,
                            temp_dir=temp_dir,
                            artifact_path=artifact_path,
                        )

                eval_env.close()
                print(f"--- Evaluation for Alternation {global_alt} Complete ---")

            # Log environment profiling if enabled
            if args.enable_env_profiling:
                try:
                    prof = offense_env.envs[0].unwrapped.get_profile_stats()
                    for k, v in prof.items():
                        mlflow.log_metric(
                            f"env_prof_{k}_avg_us_offense",
                            v.get("avg_us", 0.0),
                            step=global_alt,
                        )
                    offense_env.envs[0].unwrapped.reset_profile_stats()
                except Exception:
                    pass
                try:
                    prof = defense_env.envs[0].unwrapped.get_profile_stats()
                    for k, v in prof.items():
                        mlflow.log_metric(
                            f"env_prof_{k}_avg_us_defense",
                            v.get("avg_us", 0.0),
                            step=global_alt,
                        )
                    defense_env.envs[0].unwrapped.reset_profile_stats()
                except Exception:
                    pass

            # --- 4. Optional GIF Evaluation ---

        print("\n--- Training Complete ---")

        # --- Log final performance metrics ---
        if offense_timing_callback.rollout_times:
            mean_rollout_offense = np.mean(offense_timing_callback.rollout_times)
            mean_update_offense = np.mean(offense_timing_callback.update_times)
            print(f"Offense Mean Rollout Time: {mean_rollout_offense:.3f} s")
            print(f"Offense Mean Update Time:  {mean_update_offense:.3f} s")
            mlflow.log_param(
                "perf_mean_rollout_sec_offense", f"{mean_rollout_offense:.3f}"
            )
            mlflow.log_param(
                "perf_mean_update_sec_offense", f"{mean_update_offense:.3f}"
            )

        if defense_timing_callback.rollout_times:
            mean_rollout_defense = np.mean(defense_timing_callback.rollout_times)
            mean_update_defense = np.mean(defense_timing_callback.update_times)
            print(f"Defense Mean Rollout Time: {mean_rollout_defense:.3f} s")
            print(f"Defense Mean Update Time:  {mean_update_defense:.3f} s")
            mlflow.log_param(
                "perf_mean_rollout_sec_defense", f"{mean_rollout_defense:.3f}"
            )
            mlflow.log_param(
                "perf_mean_update_sec_defense", f"{mean_update_defense:.3f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO models using self-play.")
    parser.add_argument(
        "--grid-size", type=int, default=12, help="The size of the grid."
    )
    parser.add_argument(
        "--layup-pct", type=float, default=0.60, help="Percentage of layups."
    )
    parser.add_argument(
        "--layup-std",
        type=float,
        default=0.0,
        help="Std dev for per-player layup percentage sampling.",
    )
    parser.add_argument(
        "--three-pt-pct", type=float, default=0.37, help="Percentage of three-pointers."
    )
    parser.add_argument(
        "--three-pt-std",
        type=float,
        default=0.0,
        help="Std dev for per-player three-point percentage sampling.",
    )
    parser.add_argument(
        "--three-point-distance",
        type=int,
        default=4,
        help="Hex distance defining the three-point line.",
    )
    parser.add_argument(
        "--players", type=int, default=2, help="Number of players per side."
    )
    parser.add_argument(
        "--shot-clock", type=int, default=20, help="Steps in the shot clock."
    )
    parser.add_argument(
        "--min-shot-clock",
        dest="min_shot_clock",
        type=int,
        default=10,
        help="Minimum steps for randomly initialized shot clock at reset.",
    )
    parser.add_argument(
        "--alternations",
        type=int,
        default=10,
        help="Number of times to alternate training.",
    )
    parser.add_argument(
        "--steps-per-alternation",
        type=int,
        default=1,
        help="Timesteps to train each policy per alternation.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="PPO hyperparameter: Number of steps to run for each environment per update.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="PPO hyperparameter: Discount factor for future rewards.",
    )
    parser.add_argument(
        "--vf-coef",
        type=float,
        default=0.5,
        help="PPO hyperparameter: Weight for value function loss.",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0,
        help="PPO hyperparameter: Weight for entropy loss.",
    )
    # Optional entropy schedule across entire training
    parser.add_argument(
        "--ent-coef-start",
        type=float,
        default=None,
        help="If set, start entropy coefficient at this value and decay linearly.",
    )
    parser.add_argument(
        "--ent-coef-end",
        type=float,
        default=None,
        help="If set with --ent-coef-start, end entropy coefficient at this value.",
    )
    parser.add_argument(
        "--ent-schedule",
        type=str,
        choices=["linear", "exp"],
        default="linear",
        help="Entropy schedule type when start/end are provided.",
    )
    parser.add_argument(
        "--ent-bump-updates",
        type=int,
        default=0,
        help="If >0 with schedule, number of PPO updates to multiply entropy at start of each segment.",
    )
    parser.add_argument(
        "--ent-bump-rollouts",
        type=int,
        default=0,
        help="Deprecated alias of --ent-bump-updates; counted as updates.",
    )
    parser.add_argument(
        "--ent-bump-multiplier",
        type=float,
        default=1.0,
        help="Multiplier applied to entropy during bump rollouts (>=1.0).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="PPO hyperparameter: Minibatch size."
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2.5e-4,
        help="Learning rate for PPO optimizers.",
    )
    parser.add_argument(
        "--net-arch",
        type=int,
        nargs="+",
        default=None,
        help="The size of the neural network layers (e.g., 128 128). Default is SB3's default.",
    )
    parser.add_argument(
        "--continue-run-id",
        type=str,
        default=None,
        help="If set, load latest offense/defense policies from this MLflow run and continue training. Also appends new artifacts using continued alternation indices.",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=2,
        help="Run evaluation every N alternations. Set to 0 to disable.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of episodes to run for each evaluation.",
    )
    # The --save-path argument is no longer needed
    # parser.add_argument("--save-path", type=str, default="models/", help="Path to save the trained models.")
    parser.add_argument(
        "--defender-pressure-distance",
        type=int,
        default=1,
        help="Distance at which defender pressure is applied.",
    )
    parser.add_argument(
        "--defender-pressure-turnover-chance",
        type=float,
        default=0.05,
        help="Chance of a defender pressure turnover.",
    )
    parser.add_argument(
        "--tensorboard-path",
        type=str,
        default=None,
        help="Path to save TensorBoard logs (set to None if using MLflow).",
    )
    parser.add_argument(
        "--mlflow-experiment-name",
        type=str,
        default="BasketWorld_Training",
        help="Name of the MLflow experiment.",
    )
    parser.add_argument(
        "--mlflow-run-name", type=str, default=None, help="Name of the MLflow run."
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=8,
        help="Number of parallel environments to run for each policy during training.",
    )
    parser.add_argument(
        "--shot-pressure-enabled",
        type=lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        default=True,
        help="Enable defender shot pressure model.",
    )
    parser.add_argument(
        "--shot-pressure-max",
        type=float,
        default=0.5,
        help="Max multiplicative reduction at distance 1 (e.g., 0.5 -> up to -50%).",
    )
    parser.add_argument(
        "--shot-pressure-lambda",
        type=float,
        default=1.0,
        help="Exponential decay rate per hex for shot pressure.",
    )
    parser.add_argument(
        "--shot-pressure-arc-degrees",
        type=float,
        default=60.0,
        help="Arc width centered toward basket for pressure eligibility.",
    )
    parser.add_argument(
        "--enable-env-profiling",
        type=lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        default=False,
        help="Enable timing instrumentation inside the environment and log averages to MLflow after each alternation.",
    )
    parser.add_argument(
        "--spawn-distance",
        type=int,
        default=3,
        help="minimum distance from 3pt line at which players spawn.",
    )
    parser.add_argument(
        "--deterministic-opponent",
        type=lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        default=False,
        help="Use deterministic opponent actions.",
    )
    # Dunk controls
    parser.add_argument(
        "--allow-dunks",
        type=lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        default=False,
        help="Allow players to enter basket hex and enable dunk shots from basket cell.",
    )
    parser.add_argument(
        "--dunk-pct",
        type=float,
        default=0.90,
        help="Probability of a dunk (shot from basket cell).",
    )
    parser.add_argument(
        "--dunk-std",
        type=float,
        default=0.0,
        help="Std dev for per-player dunk percentage sampling.",
    )
    # Observation controls
    parser.add_argument(
        "--use-egocentric-obs",
        type=lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        default=True,
        help="Use egocentric observations centered at the ball handler.",
    )
    parser.add_argument(
        "--egocentric-rotate-to-hoop",
        type=lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        default=True,
        help="Rotate egocentric frame so hoop is aligned to +q axis.",
    )
    parser.add_argument(
        "--include-hoop-vector",
        type=lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        default=True,
        help="Append hoop direction vector to observation.",
    )
    parser.add_argument(
        "--normalize-obs",
        type=lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        default=True,
        help="Normalize relative coordinates to roughly [-1,1].",
    )
    parser.add_argument(
        "--mask-occupied-moves",
        type=lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        default=False,
        help="Disallow moves into currently occupied neighboring hexes.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use for training ('cuda', 'cpu', or 'auto').",
    )
    parser.add_argument(
        "--illegal-defense-enabled",
        type=lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        default=False,
        help="Enable illegal defense mode.",
    )
    parser.add_argument(
        "--illegal-defense-max-steps",
        type=int,
        default=3,
        help="Maximum number of steps to allow illegal defense.",
    )
    # Reward shaping CLI (also logged to MLflow)
    parser.add_argument(
        "--pass-reward",
        dest="pass_reward",
        type=float,
        default=0.0,
        help="Reward for successful pass (team-averaged).",
    )
    parser.add_argument(
        "--turnover-penalty",
        dest="turnover_penalty",
        type=float,
        default=0.0,
        help="Penalty for turnover (team-averaged).",
    )
    parser.add_argument(
        "--made-shot-reward-inside",
        dest="made_shot_reward_inside",
        type=float,
        default=2.0,
        help="Reward for made 2pt (team-averaged).",
    )
    parser.add_argument(
        "--made-shot-reward-three",
        dest="made_shot_reward_three",
        type=float,
        default=3.0,
        help="Reward for made 3pt (team-averaged).",
    )
    parser.add_argument(
        "--missed-shot-penalty",
        dest="missed_shot_penalty",
        type=float,
        default=0.0,
        help="Penalty for missed shot (team-averaged).",
    )
    parser.add_argument(
        "--potential-assist-reward",
        dest="potential_assist_reward",
        type=float,
        default=0.1,
        help="Reward for potential assist within window (team-averaged).",
    )
    parser.add_argument(
        "--full-assist-bonus",
        dest="full_assist_bonus",
        type=float,
        default=0.2,
        help="Additional reward for made shot within assist window (team-averaged).",
    )
    parser.add_argument(
        "--assist-window",
        dest="assist_window",
        type=int,
        default=2,
        help="Steps after pass that count toward assist window.",
    )
    parser.add_argument(
        "--potential-assist-pct",
        dest="potential_assist_pct",
        type=float,
        default=0.10,
        help="Potential assist reward as % of shot reward.",
    )
    parser.add_argument(
        "--full-assist-bonus-pct",
        dest="full_assist_bonus_pct",
        type=float,
        default=0.05,
        help="Full assist bonus as % of shot reward.",
    )
    parser.add_argument(
        "--steal-chance",
        dest="steal_chance",
        type=float,
        default=0.05,
        help="Chance of a steal.",
    )
    args = parser.parse_args()

    main(args)
