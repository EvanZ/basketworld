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
    os.environ['AWS_PROFILE'] = 'default'
    os.environ['AWS_DEFAULT_REGION'] = 'us-west-1'
    
    # Import boto3 early to ensure it uses the profile
    import boto3
    boto3.setup_default_session()
# === END CRITICAL SECTION ===

from datetime import datetime
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger, HumanOutputFormat
from basketworld.utils.mlflow_logger import MLflowWriter
from basketworld.utils.callbacks import MLflowCallback
from basketworld.utils.schedule_state import (
    save_schedule_metadata,
    load_schedule_metadata,
    calculate_continued_total_timesteps,
)
from basketworld.utils.policies import PassBiasMultiInputPolicy, PassBiasDualCriticPolicy
from basketworld.policies import SetAttentionDualCriticPolicy, SetAttentionExtractor

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
    EnvIndexWrapper,
)
import csv
try:
    from train.train_utils import (
        get_device,
        linear_schedule,
        get_steps_for_alternation,
        calculate_total_timesteps_with_schedule,
        sample_geometric,
        resolve_phi_beta_schedule,
        resolve_spa_schedule,
    )
    from train.config import get_args
    from train.env_factory import (
        setup_environment,
        make_vector_env,
        make_mixed_vector_env,
        make_policy_init_env,
    )
    from train.policy_utils import (
        get_random_policy_from_artifacts,
        get_opponent_policy_pool_for_envs,
        get_latest_policy_path,
        get_latest_unified_policy_path,
        get_max_alternation_index,
        transfer_critic_weights,
    )
    from train.profiling import log_vecenv_profile_stats
    from train.eval import run_evaluation
    from train.callbacks import (
        build_timing_callbacks,
        build_entropy_callback,
        build_beta_callback,
        build_pass_bias_callback,
        build_pass_curriculum_callback,
        build_mixed_callbacks,
        build_mixed_logger,
        log_opponent_mapping,
    )
except ImportError:
    from train_utils import (
        get_device,
        linear_schedule,
        get_steps_for_alternation,
    calculate_total_timesteps_with_schedule,
    sample_geometric,
    resolve_phi_beta_schedule,
    resolve_spa_schedule,
)
    from policy_utils import (
        get_random_policy_from_artifacts,
        get_opponent_policy_pool_for_envs,
        get_latest_policy_path,
        get_latest_unified_policy_path,
        get_max_alternation_index,
        transfer_critic_weights,
    )
    from config import get_args
    from profiling import log_vecenv_profile_stats
    from eval import run_evaluation
    from env_factory import (
        setup_environment,
        make_vector_env,
        make_mixed_vector_env,
        make_policy_init_env,
    )
    from callbacks import (
        build_timing_callbacks,
        build_entropy_callback,
        build_beta_callback,
        build_pass_bias_callback,
        build_pass_curriculum_callback,
        build_mixed_callbacks,
        build_mixed_logger,
        log_opponent_mapping,
    )

# --- CPU thread caps to avoid oversubscription in parallel env workers ---
# These defaults can be overridden by user environment.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(max(4, (os.cpu_count() or 16 // 2)))
except RuntimeError:
    # If parallel work has already started (e.g., in a spawned process), skip changing interop threads.
    pass
torch.__config__.show()


def main(args):
    """Main training function."""
    
    # --- Auto-enable dual critic if dual policy is requested ---
    # Dual policy implies dual critic (separate actors need separate critics)
    if args.use_dual_policy and not args.use_dual_critic:
        print(f"[Config] Auto-enabling --use-dual-critic due to --use-dual-policy")
        args.use_dual_critic = True
    
    # --- Auto-enable dual critic if transfer learning is requested ---
    if args.init_critic_from_run is not None:
        if not args.use_dual_critic:
            print(f"[Config] Auto-enabling --use-dual-critic due to --init-critic-from-run")
            args.use_dual_critic = True

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
        
        if getattr(args, "use_set_obs", False) and not args.use_dual_critic:
            print("[Warning] Set-attention policy requires dual critics; enabling use_dual_critic.")
            args.use_dual_critic = True

        # Log policy architecture (single vs dual critic, single vs dual policy)
        mlflow.log_param("use_dual_critic", args.use_dual_critic)
        mlflow.log_param("use_dual_policy", args.use_dual_policy)
        if getattr(args, "use_set_obs", False):
            policy_class_name = "SetAttentionDualCriticPolicy"
        else:
            policy_class_name = "PassBiasDualCriticPolicy" if args.use_dual_critic else "PassBiasMultiInputPolicy"
        if args.use_dual_policy:
            policy_class_name += " (dual_policy=True)"
        mlflow.log_param("policy_class", policy_class_name)
        if getattr(args, "use_set_obs", False):
            mlflow.log_param("set_embed_dim", getattr(args, "set_embed_dim", 64))
            mlflow.log_param("set_heads", getattr(args, "set_heads", 4))
            mlflow.log_param("set_token_mlp_dim", getattr(args, "set_token_mlp_dim", 64))
            mlflow.log_param("set_cls_tokens", getattr(args, "set_cls_tokens", 2))
            mlflow.log_param("set_token_activation", getattr(args, "set_token_activation", "relu"))
            mlflow.log_param("set_head_activation", getattr(args, "set_head_activation", "tanh"))
        
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
        use_set_obs = getattr(args, "use_set_obs", False)
        if use_set_obs:
            # Set-attention policy expects tokens and does its own extraction.
            if args.net_arch is not None:
                print("[Info] Using --net-arch for set-attention head MLP.")
            else:
                print("[Info] Using --net-arch-pi/vf for set-attention head MLP.")
            policy_kwargs["embed_dim"] = int(getattr(args, "set_embed_dim", 64))
            policy_kwargs["n_heads"] = int(getattr(args, "set_heads", 4))
            policy_kwargs["token_mlp_dim"] = int(getattr(args, "set_token_mlp_dim", 64))
            policy_kwargs["num_cls_tokens"] = int(getattr(args, "set_cls_tokens", 2))
            policy_kwargs["token_activation"] = str(getattr(args, "set_token_activation", "relu"))
            policy_kwargs["head_activation"] = str(getattr(args, "set_head_activation", "tanh"))
            print(
                "Set-attention config:",
                f"embed_dim={policy_kwargs['embed_dim']}",
                f"n_heads={policy_kwargs['n_heads']}",
                f"token_mlp_dim={policy_kwargs['token_mlp_dim']}",
                f"num_cls_tokens={policy_kwargs['num_cls_tokens']}",
                f"token_activation={policy_kwargs['token_activation']}",
                f"head_activation={policy_kwargs['head_activation']}",
            )
        else:
            # Prevent the policy from learning directly from action_mask
            policy_kwargs["features_extractor_class"] = MaskAgnosticCombinedExtractor
        
        # Enable dual policy (separate action networks for offense/defense) if requested
        if args.use_dual_policy:
            policy_kwargs["use_dual_policy"] = True

        # The save_path is no longer needed as models are saved to a temp dir
        # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # save_path = os.path.join(args.save_path, f"basketworld_selfplay_{timestamp}")
        # os.makedirs(save_path, exist_ok=True)

        # --- Initialize Base Environment (just for policy creation) ---
        # The model must be created with the same number of parallel envs that will be
        # used later (SB3 stores this value internally).
        temp_env = DummyVecEnv([lambda: make_policy_init_env(args) for _ in range(args.num_envs)])

        # --- Initialize Timing Callbacks ---
        offense_timing_callback, defense_timing_callback = build_timing_callbacks()

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
                    custom_objects = {
                        "PassBiasMultiInputPolicy": PassBiasMultiInputPolicy,
                        "PassBiasDualCriticPolicy": PassBiasDualCriticPolicy,
                        "SetAttentionDualCriticPolicy": SetAttentionDualCriticPolicy,
                        "SetAttentionExtractor": SetAttentionExtractor,
                    }
                    unified_policy = PPO.load(
                        uni_local, env=temp_env, device=device, custom_objects=custom_objects
                    )
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
            
            # Choose policy class based on flags
            if getattr(args, "use_set_obs", False):
                policy_class = SetAttentionDualCriticPolicy
            else:
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
            
            # --- Transfer critic weights if requested ---
            transfer_critic_weights(args, unified_policy)
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

        # Determine effective steps-per-alternation schedule parameters
        # Check if continuing from a previous run with spa schedule
        if (
            args.continue_run_id
            and schedule_mode == "extend"
            and previous_schedule_meta
            and "spa_start" in previous_schedule_meta
        ):
            # Use previous spa schedule if it exists
            spa_start = previous_schedule_meta.get("spa_start", args.steps_per_alternation)
            spa_end = previous_schedule_meta.get(
                "spa_end",
                args.steps_per_alternation_end
                if args.steps_per_alternation_end is not None
                else args.steps_per_alternation,
            )
            spa_schedule = previous_schedule_meta.get(
                "spa_schedule", args.steps_per_alternation_schedule
            )
            print(f"  Using previous SPA schedule: {spa_start} → {spa_end} ({spa_schedule})")
        else:
            # Fresh start or no previous spa schedule
            spa_start = args.steps_per_alternation
            spa_end = (
                args.steps_per_alternation_end
                if args.steps_per_alternation_end is not None
                else args.steps_per_alternation
            )
            spa_schedule = args.steps_per_alternation_schedule

        # Calculate total planned timesteps for this training session
        new_training_timesteps = calculate_total_timesteps_with_schedule(
            args.alternations,
            spa_start,
            spa_end,
            spa_schedule,
            args.num_envs,
            args.n_steps,
        )

        # Print and log SPA schedule info if not constant
        if spa_start != spa_end:
            print(f"\nSteps-per-alternation schedule: {spa_start} → {spa_end} ({spa_schedule})")
            print(f"Total timesteps: {new_training_timesteps:,}\n")
            
            # Generate full schedule table
            schedule_lines = []
            schedule_lines.append(f"Steps-per-alternation schedule: {spa_start} → {spa_end} ({spa_schedule})")
            schedule_lines.append(f"Total alternations: {args.alternations}")
            schedule_lines.append(f"Total timesteps: {new_training_timesteps:,}")
            schedule_lines.append("")
            schedule_lines.append("| Alternation | Steps |")
            schedule_lines.append("|-------------|-------|")
            
            for i in range(args.alternations):
                steps = get_steps_for_alternation(i, args.alternations, spa_start, spa_end, spa_schedule)
                schedule_lines.append(f"| {i + 1:11d} | {steps:5d} |")
            
            schedule_table = "\n".join(schedule_lines)
            
            # Print table to console (compact format for many alternations)
            if args.alternations <= 20:
                print(schedule_table)
            else:
                # Print abbreviated version for long schedules
                print("| Alternation | Steps |")
                print("|-------------|-------|")
                # First 5
                for i in range(min(5, args.alternations)):
                    steps = get_steps_for_alternation(i, args.alternations, spa_start, spa_end, spa_schedule)
                    print(f"| {i + 1:11d} | {steps:5d} |")
                print("|     ...     |  ...  |")
                # Last 5
                for i in range(max(5, args.alternations - 5), args.alternations):
                    steps = get_steps_for_alternation(i, args.alternations, spa_start, spa_end, spa_schedule)
                    print(f"| {i + 1:11d} | {steps:5d} |")
                print(f"\n(Full schedule logged to MLflow artifacts)")
            
            # Log schedule to MLflow as artifact
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    schedule_file = os.path.join(tmpdir, "spa_schedule.md")
                    with open(schedule_file, 'w') as f:
                        f.write(schedule_table)
                    mlflow.log_artifact(schedule_file, artifact_path="schedules")
                print("Logged SPA schedule to MLflow: schedules/spa_schedule.md")
            except Exception as e:
                print(f"Warning: Could not log SPA schedule to MLflow: {e}")

        # Log total planned timesteps as a parameter (useful for web UI)
        mlflow.log_param("total_timesteps_planned", new_training_timesteps)

        spa_start, spa_end, spa_schedule = resolve_spa_schedule(
            args, schedule_mode, previous_schedule_meta, base_alt_idx
        )
        # Determine schedule parameters based on continuation mode (entropy)
        if (
            args.continue_run_id
            and schedule_mode == "extend"
            and previous_schedule_meta
        ):
            print(f"Schedule mode: EXTEND - continuing schedules from previous run")
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

            original_total = previous_schedule_meta.get("total_planned_timesteps", 0)
            original_current = previous_schedule_meta.get("current_timesteps", 0)
            total_planned_ts = calculate_continued_total_timesteps(
                original_total,
                original_current,
                args.alternations,
                spa_start,
                spa_end,
                spa_schedule,
                args.num_envs,
                args.n_steps,
            )
            print(f"  Extended total timesteps: {total_planned_ts}")
        elif (
            args.continue_run_id
            and schedule_mode == "constant"
            and previous_schedule_meta
        ):
            print(f"Schedule mode: CONSTANT - using final schedule values as constants")
            ent_start = None
            ent_end = None
            ent_schedule_type = "linear"
            total_planned_ts = new_training_timesteps
        else:
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
        entropy_callback = build_entropy_callback(
            args,
            total_planned_ts,
            timestep_offset,
        )

        (
            phi_beta_start,
            phi_beta_end,
            phi_beta_schedule_type,
            phi_beta_bump_updates,
            phi_beta_bump_multiplier,
            total_planned_ts,
        ) = resolve_phi_beta_schedule(args, previous_schedule_meta, new_training_timesteps, schedule_mode)

        if phi_beta_start is not None or phi_beta_end is not None:
            if phi_beta_start is None:
                phi_beta_start = 0.0
            if phi_beta_end is None:
                phi_beta_end = 0.0
            beta_callback = build_beta_callback(
                args,
                total_planned_ts,
                timestep_offset,
            )
        # Optional Pass Logit Bias scheduler across the whole run
        pass_bias_callback = build_pass_bias_callback(
            args,
            total_planned_ts,
            timestep_offset,
        )

        # Optional passing curriculum (arc degrees, OOB turnover probability)
        pass_curriculum_callback = build_pass_curriculum_callback(
            args,
            total_planned_ts,
            timestep_offset,
        )

        for i in range(args.alternations):
            print("-" * 50)
            global_alt = base_alt_idx + i + 1
            # Calculate steps for this alternation based on schedule
            current_spa = get_steps_for_alternation(
                i, args.alternations, spa_start, spa_end, spa_schedule
            )
            print(f"Alternation {global_alt} (segment {i + 1} / {args.alternations})")
            if spa_start != spa_end:
                print(f"  Steps per alternation: {current_spa} (scheduled {spa_start} → {spa_end})")
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
                    args.num_envs,
                    args.opponent_pool_size,
                    args.opponent_pool_beta,
                    args.opponent_pool_exploration,
                    True,
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
            # Log opponent selection to stdout for quick sanity checks
            try:
                if isinstance(opponent_for_offense, list):
                    from collections import Counter
                    counts = Counter(
                        os.path.basename(str(p)) for p in opponent_for_offense
                    )
                    print("Opponent selection (per-env):")
                    for name, count in counts.most_common():
                        print(f"  {name}: {count} env(s)")
                else:
                    print(
                        f"Opponent selection: {os.path.basename(str(opponent_for_offense))}"
                    )
            except Exception:
                pass
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

            # --- Simultaneous Mixed Training (Offense + Defense Together) ---
            print(f"\nTraining Mixed (Offense + Defense simultaneously)...")
            
            # Create mixed environment pool (first half offense, second half defense)
            mixed_env = make_mixed_vector_env(
                args,
                opponent_policy=opponent_for_offense,
                num_envs=args.num_envs,
                deterministic_opponent=bool(args.deterministic_opponent),
            )
            unified_policy.set_env(mixed_env)

            # Use mixed logger (stdout + MLflow) for PPO metrics
            unified_policy.set_logger(build_mixed_logger())

            # Bump entropy at the start of each alternation
            if entropy_callback is not None and hasattr(
                entropy_callback, "start_new_alternation"
            ):
                try:
                    entropy_callback.start_new_alternation()
                except Exception:
                    pass

            # Combine callbacks for mixed training
            mixed_callbacks = build_mixed_callbacks(
                args,
                global_alt,
                offense_timing_callback,
                entropy_callback,
                beta_callback,
                pass_bias_callback,
                pass_curriculum_callback,
            )
            
            # Single learn() call trains both offense and defense together
            # PPO collects and mixes data from all environments
            unified_policy.learn(
                total_timesteps=current_spa * args.num_envs * args.n_steps,
                reset_num_timesteps=False,
                callback=mixed_callbacks,
                progress_bar=True,
            )
            
            # Collect profiling stats before closing the environment
            if args.enable_env_profiling:
                log_vecenv_profile_stats(
                    mixed_env,
                    prefix="Mixed/profile",
                    step=global_alt,
                )
            
            # Close mixed env and clean up
            mixed_env.close()
            try:
                del mixed_env
            except Exception:
                pass
            gc.collect()
            time.sleep(0.05)


            # Save unified checkpoint per alternation
            print(f"\nSaving unified checkpoint for alternation {global_alt}...")
            with tempfile.TemporaryDirectory() as tmpdir:
                unified_model_path = os.path.join(
                    tmpdir, f"unified_iter_{global_alt}.zip"
                )
                unified_policy.save(unified_model_path)
                mlflow.log_artifact(unified_model_path, artifact_path="models")
            print(f"Logged unified model for alternation {global_alt} to MLflow")

            # --- 3. Run Evaluation Phase ---
            if args.eval_freq > 0 and (i + 1) % args.eval_freq == 0:
                run_evaluation(args, unified_policy, global_alt)

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
                spa_start=spa_start if "spa_start" in locals() else None,
                spa_end=spa_end if "spa_end" in locals() else None,
                spa_schedule=spa_schedule if "spa_schedule" in locals() else "linear",
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
    args = get_args()
    print(' '.join(shlex.quote(arg) for arg in sys.argv))
    main(args)
