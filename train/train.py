#!/usr/bin/env python3
"""
Main training script for the HexagonBasketballEnv using an alternating
self-play strategy with Proximal Policy Optimization (PPO).

This script implements a self-play loop where two policies, one for offense
and one for defense, are trained against each other in an alternating fashion.
A custom gym.Wrapper is used to manage the opponent's actions during training.
"""

# === CRITICAL: Set AWS profile BEFORE any mlflow/boto3 imports ===
# boto3 caches credentials on first import, so we must set the profile early!
import os
import sys
import shlex

# Clear any partial AWS env vars that Cursor/VS Code might have set
# This prevents conflicts with our AWS profile
aws_vars_to_clear = [
    'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_SESSION_TOKEN',
    'AWS_SHARED_CREDENTIALS_FILE', 'AWS_CONFIG_FILE'
]
for var in aws_vars_to_clear:
    if var in os.environ:
        del os.environ[var]

# Use the basketworld AWS profile if available
# Only needed when accessing S3 artifacts (old runs with s3:// URIs)
# If MLflow server uses local storage, this won't affect anything
if os.path.exists(os.path.expanduser('~/.aws/credentials')):
    os.environ['AWS_PROFILE'] = 'basketworld'
    os.environ['AWS_DEFAULT_REGION'] = 'us-west-1'
    
    # Import boto3 early to ensure it uses the profile
    import boto3
    boto3.setup_default_session()
# === END CRITICAL SECTION ===

import argparse
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
    PotentialBetaExpScheduleCallback,
    PassLogitBiasExpScheduleCallback,
    PassCurriculumExpScheduleCallback,
    EpisodeSampleLogger,
)
from basketworld.utils.schedule_state import (
    save_schedule_metadata,
    load_schedule_metadata,
    calculate_continued_total_timesteps,
)
from basketworld.utils.policies import PassBiasMultiInputPolicy, PassBiasDualCriticPolicy

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
from typing import Callable, Optional
import torch
import gc
import time

import basketworld
from basketworld.envs.basketworld_env_v2 import Team
from basketworld.utils.mask_agnostic_extractor import MaskAgnosticCombinedExtractor
from basketworld.utils.self_play_wrapper import SelfPlayEnvWrapper
from basketworld.utils.action_resolution import IllegalActionStrategy
from basketworld.utils.wrappers import (
    RewardAggregationWrapper,
    EpisodeStatsWrapper,
    BetaSetterWrapper,
)
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
    K: int = 10,
    beta: float = 0.8,
    uniform_eps: float = 0.0,
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


def get_opponent_policy_pool_for_envs(
    client,
    run_id,
    model_prefix,
    tmpdir,
    num_envs: int,
    K: int = 10,
    beta: float = 0.7,
    uniform_eps: float = 0.15,
):
    """Sample opponent policies for each environment using geometric distribution.

    Args:
        client: MLflow client
        run_id: experiment run
        model_prefix: "unified", "offense", or "defense"
        tmpdir: temp dir to download artifacts
        num_envs: number of parallel environments (samples this many opponents)
        K: reservoir size (keep last K policies)
        beta: geometric decay factor (0<beta<1, higher = more recent bias)
        uniform_eps: probability of sampling uniformly from ALL history

    Returns:
        List of local paths to policy checkpoints (length = num_envs)
    """
    artifact_path = "models"
    all_artifacts = client.list_artifacts(run_id, artifact_path)

    # Extract paths for prefix
    team_policies = [
        f.path
        for f in all_artifacts
        if f.path.startswith(f"{artifact_path}/{model_prefix}")
        and f.path.endswith(".zip")
    ]

    if not team_policies:
        return []

    # Sort chronologically
    def sort_key(p):
        m = re.search(r"_(\d+)\.zip$", p)
        return int(m.group(1)) if m else 0

    team_policies.sort(key=sort_key)

    # Keep last K policies as the main pool
    recent_pols = team_policies[-K:] if len(team_policies) > K else team_policies

    # Sample one opponent per environment using geometric distribution
    sampled_policies = []
    print(
        f"  - Sampling {num_envs} opponents using geometric distribution (K={K}, beta={beta}, eps={uniform_eps})..."
    )

    for env_idx in range(num_envs):
        # With small probability, sample uniformly from all history for coverage
        if random.random() < uniform_eps and len(team_policies) > len(recent_pols):
            chosen = random.choice(team_policies)
        else:
            # Geometric sampling over recent_pols
            # Higher indices = more recent = higher probability with beta close to 1
            idx = sample_geometric(list(range(len(recent_pols))), beta)
            chosen = recent_pols[idx]

        sampled_policies.append(chosen)

    # Download all unique sampled policies
    unique_policies = list(set(sampled_policies))
    print(
        f"  - {len(unique_policies)} unique policies selected from {len(sampled_policies)} samples"
    )

    policy_paths = {}
    for policy_path in unique_policies:
        local_path = client.download_artifacts(run_id, policy_path, tmpdir)
        policy_paths[policy_path] = local_path
        print(f"    • {os.path.basename(policy_path)}")

    # Return list of local paths in same order as sampled
    return [policy_paths[p] for p in sampled_policies]


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
        defender_pressure_decay_lambda=getattr(args, "defender_pressure_decay_lambda", 1.0),
        base_steal_rate=getattr(args, "base_steal_rate", 0.35),
        steal_perp_decay=getattr(args, "steal_perp_decay", 1.5),
        steal_distance_factor=getattr(args, "steal_distance_factor", 0.08),
        steal_position_weight_min=getattr(args, "steal_position_weight_min", 0.3),
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
        pass_arc_degrees=getattr(args, "pass_arc_start", 60.0),
        pass_oob_turnover_prob=getattr(args, "pass_oob_turnover_prob_start", 1.0),
        spawn_distance=getattr(args, "spawn_distance", 3),
        max_spawn_distance=getattr(args, "max_spawn_distance", None),
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
        # Phi shaping config
        enable_phi_shaping=getattr(args, "enable_phi_shaping", False),
        reward_shaping_gamma=getattr(args, "reward_shaping_gamma", args.gamma),
        phi_beta=getattr(args, "phi_beta_start", 0.0),
        phi_use_ball_handler_only=getattr(args, "phi_use_ball_handler_only", False),
        phi_aggregation_mode=getattr(args, "phi_aggregation_mode", "team_best"),
        phi_blend_weight=getattr(args, "phi_blend_weight", 0.0),
        enable_profiling=args.enable_env_profiling,
        training_team=training_team,  # Critical for correct rewards
        # Observation controls
        use_egocentric_obs=args.use_egocentric_obs,
        egocentric_rotate_to_hoop=args.egocentric_rotate_to_hoop,
        include_hoop_vector=args.include_hoop_vector,
        normalize_obs=args.normalize_obs,
        mask_occupied_moves=args.mask_occupied_moves,
        enable_pass_gating=getattr(args, "enable_pass_gating", True),
        # 3-second violation (shared configuration)
        three_second_lane_width=getattr(args, "three_second_lane_width", 1),
        three_second_max_steps=getattr(args, "three_second_max_steps", 3),
        illegal_defense_enabled=args.illegal_defense_enabled,
        offensive_three_seconds_enabled=getattr(args, "offensive_three_seconds", False),
    )
    # Wrap with episode stats collector then aggregate reward for Monitor/SB3
    env = EpisodeStatsWrapper(env)
    env = RewardAggregationWrapper(env)
    # Put BetaSetterWrapper at the top so env_method('set_phi_beta', ...) hits it directly
    env = BetaSetterWrapper(env)
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
            "turnover_pass_oob",
            "turnover_intercepted",
            "turnover_pressure",
            "turnover_offensive_lane",
            "defensive_lane_violation",
            # Keys required for PPP calculation
            "made_dunk",
            "made_2pt",
            "made_3pt",
            "attempts",
            # Potential-based shaping diagnostics
            "phi_beta",
            "phi_prev",
            "phi_next",
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

    Args:
        opponent_policy: Can be:
            - Single policy path/object: all envs use same opponent
            - List of policy paths: each env gets different opponent (cycled if needed)
    """

    # If opponent_policy is a list, assign different opponents to each environment
    if isinstance(opponent_policy, list):

        def _make_env_with_opponent(env_idx: int, opp_policy_path) -> Callable[[], gym.Env]:  # type: ignore[name-defined]
            """Create a factory function for a single environment with a specific opponent."""
            def _thunk():
                base_env = setup_environment(args, training_team)
                return SelfPlayEnvWrapper(
                    base_env,
                    opponent_policy=opp_policy_path,
                    training_strategy=IllegalActionStrategy.NOOP,
                    opponent_strategy=IllegalActionStrategy.NOOP,
                    deterministic_opponent=deterministic_opponent,
                )
            return _thunk

        # Distribute opponents across environments (cycle if fewer opponents than envs)
        env_fns = [
            _make_env_with_opponent(i, opponent_policy[i % len(opponent_policy)])
            for i in range(num_envs)
        ]
        return SubprocVecEnv(env_fns, start_method="spawn")
    else:
        # Original behavior: all envs use same opponent
        def _make_env() -> Callable[[], gym.Env]:  # type: ignore[name-defined]
            """Create a factory function for a single environment."""
            def _thunk():
                base_env = setup_environment(args, training_team)
                return SelfPlayEnvWrapper(
                    base_env,
                    opponent_policy=opponent_policy,
                    training_strategy=IllegalActionStrategy.NOOP,
                    opponent_strategy=IllegalActionStrategy.NOOP,
                    deterministic_opponent=deterministic_opponent,
                )
            return _thunk

        # Use subprocesses for parallelism.
        env_fns = [_make_env() for _ in range(num_envs)]
        return SubprocVecEnv(env_fns, start_method="spawn")


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
    from basketworld.utils.mlflow_config import setup_mlflow

    try:
        mlflow_config = setup_mlflow(verbose=True)
        tracking_uri = mlflow_config.tracking_uri
    except (ImportError, ValueError) as e:
        print(f"Error setting up MLflow: {e}", file=sys.stderr)
        sys.exit(1)

    # Set the experiment name. This will create it if it doesn't exist.
    mlflow.set_experiment(args.mlflow_experiment_name)

    try:
        # Check if the server is reachable by trying to get the current experiment
        mlflow.get_experiment_by_name(args.mlflow_experiment_name)
    except mlflow.exceptions.MlflowException:
        print(
            f"Could not connect to MLflow tracking server at {tracking_uri}.",
            file=sys.stderr,
        )
        print(
            "Please ensure the MLflow UI server is running in a separate terminal with `mlflow ui`.",
            file=sys.stderr,
        )
        if mlflow_config.use_s3:
            print(
                "When using S3 storage, start the server with: mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root s3://YOUR-BUCKET/mlflow-artifacts",
                file=sys.stderr,
            )
        sys.exit(1)

    with mlflow.start_run(run_name=args.mlflow_run_name) as run:
        print("MLflow tracking URI:", mlflow.get_tracking_uri())
        # Log hyperparameters
        mlflow.log_params(vars(args))
        
        # Log observation encoding version for backward compatibility
        # role_flag encoding: -1/+1 (new) vs 0/1 (old)
        mlflow.log_param("role_flag_offense_value", 1.0)
        mlflow.log_param("role_flag_defense_value", -1.0)
        mlflow.log_param("role_flag_encoding_version", "symmetric")  # "symmetric" vs "legacy"
        
        # Log policy architecture (single vs dual critic)
        mlflow.log_param("use_dual_critic", args.use_dual_critic)
        mlflow.log_param("policy_class", "PassBiasDualCriticPolicy" if args.use_dual_critic else "PassBiasMultiInputPolicy")
        
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
            
            # Choose policy class based on --use-dual-critic flag
            policy_class = PassBiasDualCriticPolicy if args.use_dual_critic else PassBiasMultiInputPolicy
            
            unified_policy = PPO(
                policy_class,
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
        else:
            # If continuing and user requests a restart of the entropy schedule, reset counters and coef
            if getattr(args, "restart_entropy_on_continue", False):
                try:
                    unified_policy.num_timesteps = 0
                except Exception:
                    pass
                try:
                    if (args.ent_coef_start is not None) or (
                        args.ent_coef_end is not None
                    ):
                        ent_start = (
                            args.ent_coef_start
                            if args.ent_coef_start is not None
                            else args.ent_coef
                        )
                        unified_policy.ent_coef = float(ent_start)
                    else:
                        unified_policy.ent_coef = float(args.ent_coef)
                except Exception:
                    pass
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

        # --- Load schedule metadata from previous run if continuing ---
        # Handle backward compatibility with --restart-entropy-on-continue
        if getattr(args, "restart_entropy_on_continue", False):
            print(
                "Warning: --restart-entropy-on-continue is deprecated. Using --continue-schedule-mode=restart"
            )
            schedule_mode = "restart"
        else:
            schedule_mode = getattr(args, "continue_schedule_mode", "extend")
        previous_schedule_meta = None
        if args.continue_run_id and schedule_mode != "restart":
            try:
                client = mlflow.tracking.MlflowClient()
                previous_schedule_meta = load_schedule_metadata(
                    client, args.continue_run_id
                )
                print(f"Loaded schedule metadata from run {args.continue_run_id}")
                print(
                    f"  Previous total timesteps: {previous_schedule_meta.get('total_planned_timesteps', 0)}"
                )
                print(
                    f"  Previous current timesteps: {previous_schedule_meta.get('current_timesteps', 0)}"
                )
            except Exception as e:
                print(f"Warning: Could not load schedule metadata: {e}")
                print("Schedules will be initialized from scratch.")
                previous_schedule_meta = None

        # Prepare optional entropy scheduler across the whole run
        entropy_callback = None
        beta_callback = None

        # Calculate total planned timesteps for this training session
        new_training_timesteps = int(
            2
            * args.alternations
            * args.steps_per_alternation
            * args.num_envs
            * args.n_steps
        )

        # Determine schedule parameters based on continuation mode
        if (
            args.continue_run_id
            and schedule_mode == "extend"
            and previous_schedule_meta
        ):
            # Extend mode: continue the schedule from where it left off
            print(f"Schedule mode: EXTEND - continuing schedules from previous run")

            # Use previous schedule parameters if they exist, otherwise fall back to args
            ent_start = previous_schedule_meta.get(
                "ent_coef_start", args.ent_coef_start
            )
            ent_end = previous_schedule_meta.get(
                "ent_coef_end",
                args.ent_coef_end if args.ent_coef_end is not None else 0.0,
            )
            ent_schedule_type = previous_schedule_meta.get(
                "ent_schedule", getattr(args, "ent_schedule", "linear")
            )

            # Calculate extended total timesteps
            original_total = previous_schedule_meta.get("total_planned_timesteps", 0)
            original_current = previous_schedule_meta.get("current_timesteps", 0)
            total_planned_ts = calculate_continued_total_timesteps(
                original_total,
                original_current,
                args.alternations,
                args.steps_per_alternation,
                args.num_envs,
                args.n_steps,
            )
            print(f"  Extended total timesteps: {total_planned_ts}")
        elif (
            args.continue_run_id
            and schedule_mode == "constant"
            and previous_schedule_meta
        ):
            # Constant mode: use the final schedule values as constants
            print(f"Schedule mode: CONSTANT - using final schedule values as constants")
            # We'll set the callbacks to None and just use constant values
            ent_start = None
            ent_end = None
            ent_schedule_type = "linear"
            total_planned_ts = new_training_timesteps
        else:
            # Fresh start or restart mode
            if args.continue_run_id and schedule_mode == "restart":
                print(f"Schedule mode: RESTART - reinitializing schedules from scratch")
            ent_start = args.ent_coef_start if args.ent_coef_start is not None else None
            ent_end = args.ent_coef_end if args.ent_coef_end is not None else 0.0
            ent_schedule_type = getattr(args, "ent_schedule", "linear")
            total_planned_ts = new_training_timesteps

        # Determine timestep offset for restart mode
        # In restart mode with a continued run, we need to offset by the current timesteps
        # so the schedule starts fresh from 0 instead of using the model's accumulated timesteps
        timestep_offset = 0
        if args.continue_run_id and schedule_mode == "restart":
            timestep_offset = unified_policy.num_timesteps
            print(
                f"  Using timestep offset: {timestep_offset} (current model timesteps)"
            )

        # Create entropy callback if we have a schedule
        if ent_start is not None or (args.ent_coef_start is not None):
            if ent_start is None:
                ent_start = (
                    args.ent_coef
                    if args.ent_coef_start is None
                    else args.ent_coef_start
                )
            if ent_schedule_type == "exp":
                entropy_callback = EntropyExpScheduleCallback(
                    ent_start,
                    ent_end,
                    total_planned_ts,
                    bump_updates=getattr(
                        args, "ent_bump_updates", getattr(args, "ent_bump_rollouts", 0)
                    ),
                    bump_multiplier=getattr(args, "ent_bump_multiplier", 1.0),
                    timestep_offset=timestep_offset,
                )
            else:
                entropy_callback = EntropyScheduleCallback(
                    ent_start,
                    ent_end,
                    total_planned_ts,
                    timestep_offset=timestep_offset,
                )

        # Prepare optional Phi beta scheduler across the whole run
        # Determine phi beta schedule parameters based on continuation mode
        if (
            args.continue_run_id
            and schedule_mode == "extend"
            and previous_schedule_meta
        ):
            # Use previous phi beta schedule if it exists
            phi_beta_start = previous_schedule_meta.get("phi_beta_start")
            phi_beta_end = previous_schedule_meta.get("phi_beta_end")
            phi_beta_schedule_type = previous_schedule_meta.get(
                "phi_beta_schedule", "exp"
            )
            phi_beta_bump_updates = previous_schedule_meta.get("phi_bump_updates", 0)
            phi_beta_bump_multiplier = previous_schedule_meta.get(
                "phi_bump_multiplier", 1.0
            )
            # Use extended total timesteps calculated above
        elif args.continue_run_id and schedule_mode == "constant":
            # In constant mode, disable phi beta schedule
            phi_beta_start = None
            phi_beta_end = None
            phi_beta_schedule_type = "exp"
            phi_beta_bump_updates = 0
            phi_beta_bump_multiplier = 1.0
        else:
            # Fresh start or restart mode
            phi_beta_start = getattr(args, "phi_beta_start", None)
            phi_beta_end = getattr(args, "phi_beta_end", None)
            phi_beta_schedule_type = getattr(args, "phi_beta_schedule", "exp")
            phi_beta_bump_updates = getattr(args, "phi_bump_updates", 0)
            phi_beta_bump_multiplier = getattr(args, "phi_bump_multiplier", 1.0)

        if phi_beta_start is not None or phi_beta_end is not None:
            if phi_beta_start is None:
                phi_beta_start = 0.0
            if phi_beta_end is None:
                phi_beta_end = 0.0
            if phi_beta_schedule_type == "exp":
                beta_callback = PotentialBetaExpScheduleCallback(
                    phi_beta_start,
                    phi_beta_end,
                    total_planned_ts,
                    bump_updates=phi_beta_bump_updates,
                    bump_multiplier=phi_beta_bump_multiplier,
                    timestep_offset=timestep_offset,
                )
        # Optional Pass Logit Bias scheduler across the whole run
        pass_bias_callback = None
        if getattr(args, "pass_logit_bias_enabled", False) and (
            getattr(args, "pass_logit_bias_start", None) is not None
            or getattr(args, "pass_logit_bias_end", None) is not None
        ):
            p_start = (
                args.pass_logit_bias_start
                if getattr(args, "pass_logit_bias_start", None) is not None
                else 0.0
            )
            p_end = (
                args.pass_logit_bias_end
                if getattr(args, "pass_logit_bias_end", None) is not None
                else 0.0
            )
            # Use new_training_timesteps to match the restart logic
            pass_bias_total_ts = int(
                2
                * args.alternations
                * args.steps_per_alternation
                * args.num_envs
                * args.n_steps
            )
            pass_bias_callback = PassLogitBiasExpScheduleCallback(
                p_start, p_end, pass_bias_total_ts, timestep_offset=timestep_offset
            )

        # Optional passing curriculum (arc degrees, OOB turnover probability)
        pass_curriculum_callback = None
        if (
            getattr(args, "pass_arc_start", None) is not None
            or getattr(args, "pass_arc_end", None) is not None
            or getattr(args, "pass_oob_turnover_prob_start", None) is not None
            or getattr(args, "pass_oob_turnover_prob_end", None) is not None
        ):
            arc_start = (
                args.pass_arc_start
                if getattr(args, "pass_arc_start", None) is not None
                else 60.0
            )
            arc_end = (
                args.pass_arc_end
                if getattr(args, "pass_arc_end", None) is not None
                else 60.0
            )
            oob_start = (
                args.pass_oob_turnover_prob_start
                if getattr(args, "pass_oob_turnover_prob_start", None) is not None
                else 1.0
            )
            oob_end = (
                args.pass_oob_turnover_prob_end
                if getattr(args, "pass_oob_turnover_prob_end", None) is not None
                else 1.0
            )
            arc_power = (
                args.pass_arc_power
                if getattr(args, "pass_arc_power", None) is not None
                else 2.0
            )
            oob_power = (
                args.pass_oob_power
                if getattr(args, "pass_oob_power", None) is not None
                else 2.0
            )
            # Use total_planned_ts which is calculated based on schedule mode
            # For pass curriculum, we use the same total timesteps as other schedules
            pass_curriculum_callback = PassCurriculumExpScheduleCallback(
                arc_start,
                arc_end,
                oob_start,
                oob_end,
                total_planned_ts,
                arc_power=arc_power,
                oob_power=oob_power,
                timestep_offset=timestep_offset,
            )

        for i in range(args.alternations):
            print("-" * 50)
            global_alt = base_alt_idx + i + 1
            print(f"Alternation {global_alt} (segment {i + 1} / {args.alternations})")
            print("-" * 50)

            # --- Load opponent(s) for this alternation ---
            if args.per_env_opponent_sampling:
                # Sample opponent for each parallel environment using geometric distribution
                print("\nSampling opponents for each parallel environment...")
                opponent_for_offense = get_opponent_policy_pool_for_envs(
                    mlflow.tracking.MlflowClient(),
                    run.info.run_id,
                    "unified",
                    opponent_cache_dir,
                    num_envs=args.num_envs,
                    K=args.opponent_pool_size,
                    beta=args.opponent_pool_beta,
                    uniform_eps=args.opponent_pool_exploration,
                )
                if not opponent_for_offense:
                    # Fallback: create list with current policy for all envs
                    fallback_path = os.path.join(
                        opponent_cache_dir, "unified_latest.zip"
                    )
                    unified_policy.save(fallback_path)
                    opponent_for_offense = [fallback_path] * args.num_envs
                    print(
                        f"  - Using fallback: all {args.num_envs} envs get current policy"
                    )
                else:
                    print(
                        f"  - Assigned {len(opponent_for_offense)} opponents to {args.num_envs} parallel environments"
                    )
            else:
                # Original behavior: sample one opponent for entire alternation
                print(
                    "\nLoading historical opponent policy (single policy for entire alternation)..."
                )
                opponent_for_offense = get_random_policy_from_artifacts(
                    mlflow.tracking.MlflowClient(),
                    run.info.run_id,
                    "unified",
                    opponent_cache_dir,
                    K=args.opponent_pool_size,
                    beta=args.opponent_pool_beta,
                    uniform_eps=args.opponent_pool_exploration,
                )
                if opponent_for_offense is None:
                    # Fallback: save current unified policy to a stable path
                    fallback_path = os.path.join(
                        opponent_cache_dir, "unified_latest.zip"
                    )
                    unified_policy.save(fallback_path)
                    opponent_for_offense = fallback_path
            # Log which opponent checkpoint(s) used this alternation
            try:
                with tempfile.TemporaryDirectory() as _tmp_note_dir:
                    note_path = os.path.join(
                        _tmp_note_dir, f"opponent_alt_{global_alt}.txt"
                    )
                    with open(note_path, "w") as f:
                        if isinstance(opponent_for_offense, list):
                            # Per-environment sampling mode
                            f.write(
                                f"Per-environment opponent sampling (geometric distribution):\n"
                            )
                            f.write(
                                f"Parameters: K={args.opponent_pool_size}, beta={args.opponent_pool_beta}, eps={args.opponent_pool_exploration}\n"
                            )
                            f.write(f"\nEnvironment-to-Opponent Mapping:\n")

                            # Count occurrences of each policy
                            from collections import Counter

                            policy_counts = Counter(
                                os.path.basename(str(p)) for p in opponent_for_offense
                            )

                            # Log mapping
                            for env_idx, policy_path in enumerate(opponent_for_offense):
                                f.write(
                                    f"  Env {env_idx:2d}: {os.path.basename(str(policy_path))}\n"
                                )

                            # Summary statistics
                            f.write(f"\nSummary:\n")
                            f.write(
                                f"  Total environments: {len(opponent_for_offense)}\n"
                            )
                            f.write(f"  Unique policies: {len(policy_counts)}\n")
                            f.write(f"\nPolicy Usage Counts:\n")
                            for policy_name, count in sorted(
                                policy_counts.items(), key=lambda x: x[1], reverse=True
                            ):
                                f.write(f"  {policy_name}: used by {count} env(s)\n")
                        else:
                            # Single opponent mode
                            f.write(f"Single policy for all environments:\n")
                            f.write(
                                f"  {os.path.basename(str(opponent_for_offense))}\n"
                            )
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
            if beta_callback is not None:
                offense_callbacks.append(beta_callback)
            if pass_bias_callback is not None:
                offense_callbacks.append(pass_bias_callback)
            if pass_curriculum_callback is not None:
                offense_callbacks.append(pass_curriculum_callback)
            offense_callbacks.append(
                EpisodeSampleLogger(
                    team_name="Offense",
                    alternation_id=global_alt,
                    sample_prob=args.episode_sample_prob,
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
            if beta_callback is not None:
                defense_callbacks.append(beta_callback)
            if pass_bias_callback is not None:
                defense_callbacks.append(pass_bias_callback)
            if pass_curriculum_callback is not None:
                defense_callbacks.append(pass_curriculum_callback)
            defense_callbacks.append(
                EpisodeSampleLogger(
                    team_name="Defense",
                    alternation_id=global_alt,
                    sample_prob=args.episode_sample_prob,
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
            # Close defense env and clean up
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
                    spawn_distance=getattr(args, "spawn_distance", 3),
                    max_spawn_distance=getattr(args, "max_spawn_distance", None),
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
                    enable_pass_gating=getattr(args, "enable_pass_gating", True),
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

            # Note: Environment profiling code removed as envs are closed by this point

            # --- 4. Optional GIF Evaluation ---

        print("\n--- Training Complete ---")

        # --- Save schedule metadata for continuation ---
        try:
            save_schedule_metadata(
                ent_coef_start=ent_start if "ent_start" in locals() else None,
                ent_coef_end=ent_end if "ent_end" in locals() else None,
                ent_schedule=(
                    ent_schedule_type if "ent_schedule_type" in locals() else "linear"
                ),
                ent_bump_updates=getattr(args, "ent_bump_updates", 0),
                ent_bump_multiplier=getattr(args, "ent_bump_multiplier", 1.0),
                phi_beta_start=phi_beta_start if "phi_beta_start" in locals() else None,
                phi_beta_end=phi_beta_end if "phi_beta_end" in locals() else None,
                phi_beta_schedule=(
                    phi_beta_schedule_type
                    if "phi_beta_schedule_type" in locals()
                    else "exp"
                ),
                phi_bump_updates=(
                    phi_beta_bump_updates if "phi_beta_bump_updates" in locals() else 0
                ),
                phi_bump_multiplier=(
                    phi_beta_bump_multiplier
                    if "phi_beta_bump_multiplier" in locals()
                    else 1.0
                ),
                pass_logit_bias_start=(
                    getattr(args, "pass_logit_bias_start", None)
                    if getattr(args, "pass_logit_bias_enabled", False)
                    else None
                ),
                pass_logit_bias_end=(
                    getattr(args, "pass_logit_bias_end", None)
                    if getattr(args, "pass_logit_bias_enabled", False)
                    else None
                ),
                pass_arc_start=getattr(args, "pass_arc_start", None),
                pass_arc_end=getattr(args, "pass_arc_end", None),
                pass_oob_turnover_prob_start=getattr(
                    args, "pass_oob_turnover_prob_start", None
                ),
                pass_oob_turnover_prob_end=getattr(
                    args, "pass_oob_turnover_prob_end", None
                ),
                total_planned_timesteps=(
                    total_planned_ts if "total_planned_ts" in locals() else 0
                ),
                current_timesteps=unified_policy.num_timesteps,
            )
            print("Saved schedule metadata for future continuation")
        except Exception as e:
            print(f"Warning: Could not save schedule metadata: {e}")

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
        "--use-dual-critic",
        action="store_true",
        default=False,
        help="Use separate value networks for offense and defense (recommended for zero-sum self-play).",
    )
    parser.add_argument(
        "--continue-run-id",
        type=str,
        default=None,
        help="If set, load latest offense/defense policies from this MLflow run and continue training. Also appends new artifacts using continued alternation indices.",
    )
    parser.add_argument(
        "--continue-schedule-mode",
        type=str,
        choices=["extend", "constant", "restart"],
        default="extend",
        help=(
            "How to handle schedules when continuing training: "
            "'extend' (default) - continue schedules from where they left off, adding more training to the original total; "
            "'constant' - use the final schedule values (where the previous run ended) as constants; "
            "'restart' - restart schedules from scratch using new parameters."
        ),
    )
    parser.add_argument(
        "--restart-entropy-on-continue",
        dest="restart_entropy_on_continue",
        type=lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        default=False,
        help="DEPRECATED: Use --continue-schedule-mode=restart instead. When continuing from a run, reset num_timesteps and reinitialize ent_coef to the schedule start.",
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
        "--defender-pressure-decay-lambda",
        type=float,
        default=1.0,
        help="Exponential decay rate for defender pressure.",
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
        "--use-vec-normalize",
        type=lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        default=False,
        help="(DEPRECATED - no longer used) Previously used VecNormalize wrapper. "
        "Kept for MLflow compatibility.",
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
        help="minimum distance from basket at which players spawn.",
    )
    parser.add_argument(
        "--max-spawn-distance",
        dest="max_spawn_distance",
        type=lambda v: None if v == "" or str(v).lower() == "none" else int(v),
        default=None,
        help="maximum distance from basket at which players spawn (None = unlimited). Use with --spawn-distance for curriculum learning.",
    )
    parser.add_argument(
        "--deterministic-opponent",
        type=lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        default=False,
        help="Use deterministic opponent actions.",
    )
    parser.add_argument(
        "--opponent-pool-size",
        type=int,
        default=10,
        help="Number of recent checkpoints to keep in opponent pool (K parameter).",
    )
    parser.add_argument(
        "--opponent-pool-beta",
        type=float,
        default=0.7,
        help="Geometric decay factor for opponent sampling (0=uniform, 1=most recent only).",
    )
    parser.add_argument(
        "--opponent-pool-exploration",
        type=float,
        default=0.15,
        help="Probability of sampling from ALL history instead of just recent pool (0-1).",
    )
    parser.add_argument(
        "--per-env-opponent-sampling",
        action="store_true",
        help="Sample different opponents for each parallel environment using geometric distribution (prevents forgetting). Each of the --num-envs workers independently samples from last K checkpoints with recency bias. Default: single opponent per alternation.",
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
        "--enable-pass-gating",
        dest="enable_pass_gating",
        type=lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        default=True,
        help="Mask out pass actions that don't have a teammate in the arc. "
        "This prevents learning to avoid passing due to OOB turnovers.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use for training ('cuda', 'cpu', or 'auto').",
    )
    # 3-second violation shared configuration
    parser.add_argument(
        "--three-second-lane-width",
        type=int,
        default=1,
        help="Width of the lane in hexes (shared by offense and defense). 1 = 1 hex on each side of center line.",
    )
    parser.add_argument(
        "--three-second-max-steps",
        type=int,
        default=3,
        help="Maximum steps a player can stay in the lane (shared by offense and defense).",
    )
    # Individual enable flags
    parser.add_argument(
        "--illegal-defense-enabled",
        type=lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        default=False,
        help="Enable illegal defense (defensive 3-second) rule.",
    )
    parser.add_argument(
        "--offensive-three-seconds",
        type=lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        default=False,
        help="Enable offensive 3-second violation rule.",
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
        default=0,
        help="Reward for potential assist within window (team-averaged).",
    )
    parser.add_argument(
        "--full-assist-bonus",
        dest="full_assist_bonus",
        type=float,
        default=0,
        help="Additional reward for made shot within assist window (team-averaged).",
    )
    parser.add_argument(
        "--assist-window",
        dest="assist_window",
        type=int,
        default=3,
        help="Steps after pass that count toward assist window.",
    )
    parser.add_argument(
        "--potential-assist-pct",
        dest="potential_assist_pct",
        type=float,
        default=0,
        help="Potential assist reward as % of shot reward.",
    )
    parser.add_argument(
        "--full-assist-bonus-pct",
        dest="full_assist_bonus_pct",
        type=float,
        default=0,
        help="Full assist bonus as % of shot reward.",
    )
    parser.add_argument(
        "--base-steal-rate",
        dest="base_steal_rate",
        type=float,
        default=0.35,
        help="Base steal rate when defender is directly on pass line.",
    )
    parser.add_argument(
        "--steal-perp-decay",
        dest="steal_perp_decay",
        type=float,
        default=1.5,
        help="Exponential decay rate for steal chance perpendicular to pass line.",
    )
    parser.add_argument(
        "--steal-distance-factor",
        dest="steal_distance_factor",
        type=float,
        default=0.08,
        help="Factor by which pass distance increases steal chance.",
    )
    parser.add_argument(
        "--steal-position-weight-min",
        dest="steal_position_weight_min",
        type=float,
        default=0.3,
        help="Minimum steal weight for defenders near passer (1.0 at receiver). Defenders closer to receiver are more dangerous.",
    )
    parser.add_argument(
        "--episode-sample-prob",
        dest="episode_sample_prob",
        type=float,
        default=1e-2,
        help="Probability of sampling an episode for logging.",
    )
    # Potential-based shaping CLI
    parser.add_argument(
        "--enable-phi-shaping",
        dest="enable_phi_shaping",
        type=lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        default=False,
        help="Enable potential-based reward shaping using best current shot quality.",
    )
    parser.add_argument(
        "--reward-shaping-gamma",
        dest="reward_shaping_gamma",
        type=float,
        default=None,
        help="Discount gamma used inside shaping term (should match PPO gamma).",
    )
    parser.add_argument(
        "--phi-use-ball-handler-only",
        dest="phi_use_ball_handler_only",
        type=lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        default=False,
        help="Use only ball-handler make prob for Phi instead of team best.",
    )
    parser.add_argument(
        "--phi-blend-weight",
        dest="phi_blend_weight",
        type=float,
        default=0.0,
        help="Blend weight w in [0,1] for Phi=(1-w)*aggregate_EP + w*ball_EP (ignored if ball-handler-only).",
    )
    parser.add_argument(
        "--phi-aggregation-mode",
        dest="phi_aggregation_mode",
        type=str,
        choices=[
            "team_best",
            "teammates_best",
            "teammates_avg",
            "team_avg",
            "team_worst",
            "teammates_worst",
        ],
        default="team_best",
        help="How to aggregate teammate EPs: 'team_best' (max including ball), 'teammates_best' (max excluding ball), 'teammates_avg' (mean excluding ball), 'team_avg' (mean including ball), 'team_worst' (min including ball), 'teammates_worst' (min excluding ball).",
    )
    parser.add_argument(
        "--phi-beta-start",
        dest="phi_beta_start",
        type=float,
        default=0.0,
        help="Initial beta multiplier for Phi shaping.",
    )
    parser.add_argument(
        "--phi-beta-end",
        dest="phi_beta_end",
        type=float,
        default=0.0,
        help="Final beta multiplier for Phi shaping (decays to this).",
    )
    parser.add_argument(
        "--phi-bump-updates",
        dest="phi_bump_updates",
        type=int,
        default=0,
        help="Number of PPO updates to bump phi_beta at start of each segment.",
    )
    parser.add_argument(
        "--phi-bump-multiplier",
        dest="phi_bump_multiplier",
        type=float,
        default=1.0,
        help="Multiplier applied to phi_beta during bump updates (>=1.0).",
    )
    # Passing curriculum CLI
    parser.add_argument(
        "--pass-arc-start",
        dest="pass_arc_start",
        type=float,
        default=None,
        help="Initial passing arc degrees (e.g., 120).",
    )
    parser.add_argument(
        "--pass-arc-end",
        dest="pass_arc_end",
        type=float,
        default=None,
        help="Final passing arc degrees (e.g., 60).",
    )
    parser.add_argument(
        "--pass-oob-turnover-prob-start",
        dest="pass_oob_turnover_prob_start",
        type=float,
        default=None,
        help="Initial probability that pass without receiver is OOB turnover (e.g., 0.1).",
    )
    parser.add_argument(
        "--pass-oob-turnover-prob-end",
        dest="pass_oob_turnover_prob_end",
        type=float,
        default=None,
        help="Final OOB turnover probability when no receiver (e.g., 1.0).",
    )
    parser.add_argument(
        "--pass-arc-power",
        dest="pass_arc_power",
        type=float,
        default=2.0,
        help="Power applied to arc curriculum progress for steeper initial decay (default: 2.0, use 1.0 for linear).",
    )
    parser.add_argument(
        "--pass-oob-power",
        dest="pass_oob_power",
        type=float,
        default=2.0,
        help="Power applied to OOB curriculum progress for steeper initial decay (default: 2.0, use 1.0 for linear).",
    )
    parser.add_argument(
        "--pass-logit-bias-enabled",
        dest="pass_logit_bias_enabled",
        type=lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        default=False,
        help="Enable additive pass-logit bias.",
    )
    parser.add_argument(
        "--pass-logit-bias-start",
        dest="pass_logit_bias_start",
        type=float,
        default=None,
        help="Initial additive bias added to PASS action logits (e.g., 0.8).",
    )
    parser.add_argument(
        "--pass-logit-bias-end",
        dest="pass_logit_bias_end",
        type=float,
        default=None,
        help="Final additive bias (0 to disable at end).",
    )
    args = parser.parse_args()
    print(' '.join(shlex.quote(arg) for arg in sys.argv))
    main(args)
