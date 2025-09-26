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
from basketworld.utils.callbacks import (
    RolloutUpdateTimingCallback,
    MLflowCallback,
    EntropyScheduleCallback,
    EntropyExpScheduleCallback,
    EpisodeSampleLogger,
)

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
import gc
import time

import basketworld
from basketworld.envs.basketworld_env_v2 import Team
from basketworld.utils.mask_agnostic_extractor import MaskAgnosticCombinedExtractor
from basketworld.utils.self_play_wrapper import SelfPlayEnvWrapper
from basketworld.utils.action_resolution import IllegalActionStrategy
from basketworld.utils.wrappers import RewardAggregationWrapper, EpisodeStatsWrapper
import csv

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


def linear_schedule(start, end):
    def f(progress_remaining: float):
        return end + (start - end) * progress_remaining

    return f


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
            # minimal audit
            "gt_is_three",
            "gt_is_dunk",
            "gt_points",
            "gt_shooter_off",
            "gt_shooter_q",
            "gt_shooter_r",
            "gt_distance",
            "basket_q",
            "basket_r",
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
        base_env = setup_environment(args, training_team)
        return SelfPlayEnvWrapper(
            base_env,
            opponent_policy=opponent_policy,
            training_strategy=IllegalActionStrategy.NOOP,
            opponent_strategy=IllegalActionStrategy.NOOP,
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
        # Allow specifying shared or separate actor/critic network architectures.
        policy_kwargs = {}
        if args.net_arch is not None:
            # Backwards-compat: if a single net_arch is provided, use it as-is
            policy_kwargs["net_arch"] = args.net_arch
        else:
            # Default to separate policy/value arches with defaults [64, 64] each
            pi_arch = getattr(args, "net_arch_pi", [64, 64])
            vf_arch = getattr(args, "net_arch_vf", [64, 64])
            policy_kwargs["net_arch"] = dict(pi=pi_arch, vf=vf_arch)
        # Prevent the policy from learning directly from action_mask
        policy_kwargs["features_extractor_class"] = MaskAgnosticCombinedExtractor

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
                n_epochs=args.n_epochs,
                vf_coef=args.vf_coef,
                ent_coef=initial_ent_coef,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                tensorboard_log=None,  # Disable TensorBoard if using MLflow
                policy_kwargs=policy_kwargs,
                target_kl=args.target_kl,
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
            # Log which opponent checkpoint is used this alternation
            try:
                with tempfile.TemporaryDirectory() as _tmp_note_dir:
                    note_path = os.path.join(
                        _tmp_note_dir, f"opponent_alt_{global_alt}.txt"
                    )
                    with open(note_path, "w") as f:
                        f.write(os.path.basename(str(opponent_for_offense)))
                    mlflow.log_artifact(note_path, artifact_path=f"opponents")
            except Exception:
                pass

            # --- 1. Train Offense against frozen Defense ---
            print(f"\nTraining Offense...")
            offense_env = make_vector_env(
                args,
                training_team=Team.OFFENSE,
                opponent_policy=opponent_for_offense,
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
            offense_callbacks.append(
                EpisodeSampleLogger(
                    team_name="Offense", alternation_id=global_alt, sample_prob=1e-2
                )
            )
            unified_policy.learn(
                total_timesteps=args.steps_per_alternation
                * args.num_envs
                * args.n_steps,
                reset_num_timesteps=False,
                callback=offense_callbacks,
                progress_bar=True,
            )
            offense_env.close()
            try:
                del offense_env
            except Exception:
                pass
            gc.collect()
            time.sleep(0.05)

            # Reuse the same frozen opponent for defense segment to keep distributions comparable
            opponent_for_defense = opponent_for_offense

            # --- 2. Train Defense against frozen Offense ---
            print(f"\nTraining Defense...")
            defense_env = make_vector_env(
                args,
                training_team=Team.DEFENSE,
                opponent_policy=opponent_for_defense,
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
            defense_callbacks.append(
                EpisodeSampleLogger(
                    team_name="Defense", alternation_id=global_alt, sample_prob=1e-2
                )
            )
            unified_policy.learn(
                total_timesteps=args.steps_per_alternation
                * args.num_envs
                * args.n_steps,
                reset_num_timesteps=False,
                callback=defense_callbacks,
                progress_bar=True,
            )
            defense_env.close()
            try:
                del defense_env
            except Exception:
                pass
            gc.collect()
            time.sleep(0.05)

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
        "--target-kl",
        type=float,
        default=0.025,
        help="PPO hyperparameter: Target KL divergence for early stopping.",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=10,
        help="PPO hyperparameter: Number of epochs when optimizing the surrogate.",
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
        "--net-arch-pi",
        type=int,
        nargs="+",
        default=[64, 64],
        help="Actor (policy) MLP hidden sizes, e.g. 64 64. Ignored if --net-arch is set.",
    )
    parser.add_argument(
        "--net-arch-vf",
        type=int,
        nargs="+",
        default=[64, 64],
        help="Critic (value) MLP hidden sizes, e.g. 64 64. Ignored if --net-arch is set.",
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
