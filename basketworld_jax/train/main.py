from __future__ import annotations

import argparse
import json
from copy import copy
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter_ns
from typing import Any
import sys

import numpy as np

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from basketworld.utils.mlflow_config import setup_mlflow
from basketworld.utils.play_names import (
    PLAY_NAME_POOL_VERSION,
    build_model_codename,
    build_play_name_artifact_payload,
)
from basketworld_jax.checkpoints import (
    build_checkpoint_paths,
    build_checkpoint_payload,
    load_checkpoint,
    save_checkpoint,
)
from basketworld_jax.config import TRAIN_FROZEN_VALUES
from basketworld_jax.env import (
    MULTI_POSSESSION_SCHEMA_VERSION,
    PASS_ACTION_END,
    PASS_ACTION_START,
    token_observation_dims,
    build_action_masks_batch,
    build_policy_intent_context_batch,
    build_policy_observation_batch,
    reset_batch_minimal,
    sample_state_batch,
)
from basketworld_jax.models import (
    ActorCriticSpec,
    build_actor_critic_spec,
    init_actor_critic_params,
)
from basketworld_jax.intent import (
    build_intent_discriminator_spec,
    build_intent_discriminator_update_runner,
    build_intent_policy_sensitivity_runner,
    build_intent_sample_dump,
    build_intent_step_features_from_rollout,
    compute_intent_beta,
    compute_normalized_intent_bonus,
    init_bonus_stats,
    init_intent_discriminator_params,
    update_bonus_stats,
)
from basketworld_jax.optim import init_optimizer_state
from basketworld_jax.train.cli import (
    build_parser,
    build_progress,
    ensure_jax_available,
    to_builtin,
    write_json,
)
from basketworld_jax.train.control import (
    acknowledge_control_request,
    read_control_request,
    write_training_status,
)
from basketworld_jax.train.types import (
    TrainerConfig,
    build_ppo_batch,
    build_selector_batch,
    build_trajectory_training_masks,
    concatenate_selector_batches,
    concatenate_ppo_batches,
    limit_selector_batch_samples,
)
from basketworld_jax.train.runtime import (
    benchmark_compiled_rollout,
    benchmark_update_runner,
    block_until_ready_tree,
    build_compiled_deploy_eval_runner,
    build_compiled_eval_runner,
    build_compiled_frozen_opponent_eval_runner,
    build_compiled_frozen_opponent_rollout_runner,
    build_compiled_grouped_opponent_eval_runner,
    build_compiled_grouped_opponent_rollout_runner,
    build_compiled_historical_match_eval_runner,
    HISTORICAL_MATCH_DIAGNOSTIC_KEYS,
    HISTORICAL_MATCH_FLOAT_DIAGNOSTIC_KEYS,
    build_compiled_rollout_runner,
    build_jitted_actor_critic_runner,
    build_jitted_ppo_update_runner,
    build_jitted_selector_update_runner,
    concatenate_rollout_outputs,
    serialize_eval_trace,
    summarize_deploy_eval_outputs,
    summarize_episode_events,
    summarize_intent_metrics,
    summarize_lane_violation_metrics,
    summarize_learner_rebound_metrics,
    summarize_ppo_eligible_episode_metrics,
    summarize_ppo_eligible_reward_component_metrics,
    summarize_reward_by_intent_metrics,
    summarize_shot_type_metrics,
    summarize_training_step,
    training_player_ids_from_static,
)


TRAINING_ROLES = ("offense", "defense")
TRAIN_LOOP_SUMMARY_ARTIFACT_DIR = "results"
TRAIN_LOOP_SUMMARY_ARTIFACT_NAME = "train_loop_summary.json"
TRAIN_LOOP_SUMMARY_ARTIFACT_PATH = (
    f"{TRAIN_LOOP_SUMMARY_ARTIFACT_DIR}/{TRAIN_LOOP_SUMMARY_ARTIFACT_NAME}"
)
HISTORICAL_EVAL_ARTIFACT_NAME = "historical_opponent_evaluation.json"
HISTORICAL_EVAL_ARTIFACT_PATH = (
    f"{TRAIN_LOOP_SUMMARY_ARTIFACT_DIR}/{HISTORICAL_EVAL_ARTIFACT_NAME}"
)
HISTORICAL_EVAL_UPDATE_ARTIFACT_DIR = (
    f"{TRAIN_LOOP_SUMMARY_ARTIFACT_DIR}/historical_opponent_evaluations"
)
JAX_ALLOWED_ENV_OVERRIDE_KEYS = frozenset(
    {
        # These structural/geometry/physics settings are valid for fresh runs.
        # Checkpoint/resume compatibility is still enforced by saved frozen config.
        "players",
        "court_rows",
        "court_cols",
        "shot_clock",
        "min_shot_clock",
        "three_point_distance",
        "three_point_short_distance",
        "three_pt_extra_hex_decay",
        "shot_pressure_enabled",
        "shot_pressure_max",
        "shot_pressure_lambda",
        "shot_pressure_arc_degrees",
        "defender_pressure_distance",
        "defender_pressure_turnover_chance",
        "defender_pressure_decay_lambda",
        "base_steal_rate",
        "steal_perp_decay",
        "steal_distance_factor",
        "steal_position_weight_min",
        "pass_interception_model",
        "pass_passer_pressure_weight",
        "pass_receiver_pressure_weight",
        "pass_lob_lane_multiplier",
        "pass_lob_receiver_distance",
        "pass_speed",
        "defender_reaction_time",
        "defender_speed",
        "defender_reach_radius",
        "reaction_softness",
        "base_passer_risk",
        "passer_pressure_decay",
        "base_receiver_risk",
        "receiver_alignment_min",
        "receiver_alignment_width",
        "max_receiver_hazard",
        "lane_weight",
        "spawn_distance",
        "max_spawn_distance",
        "defender_spawn_distance",
        "defender_guard_distance",
        "assist_window",
        "mask_occupied_moves",
        "enable_pass_gating",
        "allow_dunks",
        "layup_pct",
        "three_pt_pct",
        "dunk_pct",
        "illegal_defense_enabled",
        "offensive_three_seconds",
        "reward_shaping_gamma",
        "enable_phi_shaping",
        "phi_beta_start",
        "phi_beta_end",
        "phi_beta_warmup_updates",
        "phi_beta_ramp_updates",
        "phi_blend_weight",
        "phi_aggregation_mode",
        "phi_use_ball_handler_only",
        "enable_intent_learning",
        "enable_defense_intent_learning",
        "intent_diversity_enabled",
        "intent_selector_enabled",
        "intent_selector_hidden_dim",
        "intent_selector_alpha_start",
        "intent_selector_alpha_end",
        "intent_selector_alpha_warmup_steps",
        "intent_selector_alpha_ramp_steps",
        "intent_selector_eps_start",
        "intent_selector_eps_end",
        "intent_selector_eps_warmup_steps",
        "intent_selector_eps_ramp_steps",
        "intent_selector_learning_rate",
        "intent_selector_entropy_coef",
        "intent_selector_usage_reg_coef",
        "intent_selector_value_coef",
        "intent_selector_train_every_rollouts",
        "intent_selector_max_samples_per_update",
        "intent_selector_multiselect_enabled",
        "intent_selector_min_play_steps",
        "num_intents",
        "intent_commitment_steps",
        "intent_null_prob",
        "defense_intent_null_prob",
        "intent_visible_to_defense_prob",
        "start_template_enabled",
        "start_template_library",
        "start_template_prob",
        "start_template_jitter_scale",
        "start_template_mirror_prob",
        "start_template_strict",
        "enable_rebounds",
        "rebound_table_model_dir",
        "rebound_target_temperature",
        "rebound_target_uniform_mix",
        "rebound_winner_distance_weight",
        "rebound_basket_position_weight",
        "rebound_winner_temperature",
        "rebound_skill_std",
        "rebound_skill_sampling_mode",
        "rebound_skill_high",
        "rebound_skill_low",
        "rebound_skill_weight",
        "rebound_contest_mode",
        "rebound_contest_radius",
        "rebound_obs_top_n_targets",
        "offensive_rebound_shot_clock_reset",
        "rebound_terminal_reward_mode",
        "enable_rebound_reward_redistribution",
        "offensive_rebound_reward_advance",
        "rebound_reward_once_per_possession",
        "rebound_counterfactual_positioning_enabled",
        "enable_multi_possession",
        "multi_possession_limit",
        "multi_possession_limit_start",
        "multi_possession_limit_end",
        "multi_possession_limit_ramp_updates",
        "multi_possession_overtime_round_cap",
        "multi_possession_reward_mode",
        "score_potential_scale",
        "game_winner_reward",
        "multi_possession_aux_rewards_enabled",
        "multi_possession_use_inbounds",
        "made_basket_restart_mode",
        "inbound_deadline_steps",
        "check_deadline_steps",
        "check_setup_steps",
    }
)
JAX_ENV_MLFLOW_PARAM_KEYS = (
    "training_team",
    "players",
    "court_rows",
    "court_cols",
    "shot_clock",
    "min_shot_clock",
    "allow_dunks",
    "layup_pct",
    "three_pt_pct",
    "dunk_pct",
    "layup_std",
    "three_pt_std",
    "dunk_std",
    "three_point_distance",
    "three_point_short_distance",
    "three_pt_extra_hex_decay",
    "shot_pressure_enabled",
    "shot_pressure_max",
    "shot_pressure_lambda",
    "shot_pressure_arc_degrees",
    "defender_pressure_distance",
    "defender_pressure_turnover_chance",
    "defender_pressure_decay_lambda",
    "base_steal_rate",
    "steal_perp_decay",
    "steal_distance_factor",
    "steal_position_weight_min",
    "pass_interception_model",
    "pass_passer_pressure_weight",
    "pass_receiver_pressure_weight",
    "pass_lob_lane_multiplier",
    "pass_lob_receiver_distance",
    "pass_speed",
    "defender_reaction_time",
    "defender_speed",
    "defender_reach_radius",
    "reaction_softness",
    "base_passer_risk",
    "passer_pressure_decay",
    "base_receiver_risk",
    "receiver_alignment_min",
    "receiver_alignment_width",
    "max_receiver_hazard",
    "lane_weight",
    "spawn_distance",
    "max_spawn_distance",
    "defender_spawn_distance",
    "defender_guard_distance",
    "assist_window",
    "mask_occupied_moves",
    "enable_pass_gating",
    "pass_mode",
    "use_set_obs",
    "illegal_defense_enabled",
    "offensive_three_seconds",
    "three_second_lane_width",
    "three_second_lane_height",
    "three_second_max_steps",
    "violation_reward",
    "include_hoop_vector",
    "enable_phi_shaping",
    "reward_shaping_gamma",
    "phi_beta_start",
    "phi_beta_end",
    "phi_beta_warmup_updates",
    "phi_beta_ramp_updates",
    "phi_blend_weight",
    "phi_aggregation_mode",
    "phi_use_ball_handler_only",
    "pass_reward",
    "potential_assist_pct",
    "full_assist_bonus_pct",
    "start_template_enabled",
    "start_template_library",
    "start_template_prob",
    "start_template_jitter_scale",
    "start_template_mirror_prob",
    "start_template_strict",
    "enable_intent_learning",
    "enable_defense_intent_learning",
    "num_intents",
    "intent_commitment_steps",
    "intent_null_prob",
    "defense_intent_null_prob",
    "intent_visible_to_defense_prob",
    "enable_rebounds",
    "rebound_table_model_dir",
    "rebound_target_temperature",
    "rebound_target_uniform_mix",
    "rebound_winner_distance_weight",
    "rebound_basket_position_weight",
    "rebound_winner_temperature",
    "rebound_skill_std",
    "rebound_skill_sampling_mode",
    "rebound_skill_high",
    "rebound_skill_low",
    "rebound_skill_weight",
    "rebound_contest_mode",
    "rebound_contest_radius",
    "rebound_obs_top_n_targets",
    "offensive_rebound_shot_clock_reset",
    "rebound_terminal_reward_mode",
    "enable_rebound_reward_redistribution",
    "offensive_rebound_reward_advance",
    "rebound_reward_once_per_possession",
    "rebound_counterfactual_positioning_enabled",
    "enable_multi_possession",
    "multi_possession_limit",
    "multi_possession_limit_start",
    "multi_possession_limit_end",
    "multi_possession_limit_ramp_updates",
    "multi_possession_overtime_round_cap",
    "multi_possession_reward_mode",
    "score_potential_scale",
    "game_winner_reward",
    "multi_possession_aux_rewards_enabled",
    "multi_possession_use_inbounds",
    "made_basket_restart_mode",
    "inbound_deadline_steps",
    "check_deadline_steps",
    "check_setup_steps",
)


def _reject_legacy_opponent_flag(argv: list[str]) -> None:
    if "--per-env-opponent-sampling" in argv:
        raise SystemExit(
            "Use --grouped-opponent-sampling for JAX grouped opponent sampling."
        )


def _suppress_legacy_opponent_help(parser) -> None:
    action = parser._option_string_actions.get("--per-env-opponent-sampling")
    if action is not None:
        action.help = argparse.SUPPRESS


def build_train_parser():
    parser = build_parser("JAX trainer: reduced actor-critic + compiled rollout path.")
    _suppress_legacy_opponent_help(parser)
    parser.set_defaults(**TRAIN_FROZEN_VALUES)
    parser.add_argument(
        "--kernel-batch-size",
        type=int,
        default=256,
        help="Number of env states packed into one JAX rollout batch.",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=3,
        help="Number of untimed warm iterations before scaffold timing.",
    )
    parser.add_argument(
        "--benchmark-iters",
        type=int,
        default=50,
        help="Number of timed iterations for scaffold timing.",
    )
    parser.add_argument(
        "--sample-reset-seed",
        type=int,
        default=0,
        help="Base reset seed used when sampling representative env snapshots.",
    )
    parser.add_argument(
        "--policy-hidden-dims",
        type=int,
        nargs="+",
        default=[128, 128],
        help="Hidden layer widths for the reduced flat actor-critic.",
    )
    parser.add_argument(
        "--policy-model",
        choices=("mlp", "attention"),
        default="mlp",
        help="JAX policy architecture. 'attention' uses packed player tokens plus globals.",
    )
    parser.add_argument(
        "--action-head-mode",
        choices=("flat", "pointer_targeted"),
        default="flat",
        help=(
            "JAX action distribution head. 'pointer_targeted' factorizes pass "
            "selection into action type plus teammate target slot."
        ),
    )
    parser.add_argument(
        "--attention-embed-dim",
        type=int,
        default=64,
        help="Token embedding dimension for --policy-model attention.",
    )
    parser.add_argument(
        "--attention-num-heads",
        type=int,
        default=4,
        help="Number of self-attention heads for --policy-model attention.",
    )
    parser.add_argument(
        "--attention-token-mlp-dim",
        type=int,
        default=64,
        help="Hidden width of the shared token MLP for --policy-model attention.",
    )
    parser.add_argument(
        "--attention-cls-tokens",
        type=int,
        default=2,
        help="Number of learned CLS tokens for --policy-model attention.",
    )
    parser.add_argument(
        "--attention-pi-head-hidden-dims",
        type=int,
        nargs="*",
        default=[],
        help="Post-attention policy head MLP hidden widths. Empty means direct linear head.",
    )
    parser.add_argument(
        "--attention-vf-head-hidden-dims",
        type=int,
        nargs="*",
        default=[],
        help="Post-attention value head MLP hidden widths. Empty means direct linear head.",
    )
    parser.add_argument(
        "--attention-head-activation",
        choices=("tanh", "relu", "gelu", "silu", "swish"),
        default="tanh",
        help="Activation used between post-attention PI/VF head MLP layers.",
    )
    parser.add_argument(
        "--intent-embedding-enabled",
        action="store_true",
        help=(
            "Condition the JAX attention policy on the active runtime intent ID. "
            "Requires an intent runtime to be enabled."
        ),
    )
    parser.add_argument(
        "--intent-embedding-dim",
        type=int,
        default=16,
        help="Embedding dimension for runtime intent conditioning in the JAX attention policy.",
    )
    parser.add_argument(
        "--intent-conditioning-scale",
        type=float,
        default=1.0,
        help=(
            "Multiplier applied to the learned intent-conditioning delta. "
            "Values above one strengthen initial and learned policy sensitivity."
        ),
    )
    parser.add_argument(
        "--intent-sample-dump-size",
        type=int,
        default=2048,
        help=(
            "Maximum active-offense intent samples to include in each JAX "
            "discriminator sample dump when --disc-eval-batch-output is enabled."
        ),
    )
    parser.add_argument(
        "--intent-diversity-warmup-updates",
        type=int,
        default=None,
        help=(
            "JAX-only update-count warmup for the intent-diversity bonus. "
            "The discriminator still pretrains during this period."
        ),
    )
    parser.add_argument(
        "--intent-diversity-ramp-updates",
        type=int,
        default=None,
        help=(
            "JAX-only update-count ramp from zero diversity beta to target after "
            "--intent-diversity-warmup-updates. Defaults to one update when "
            "warmup updates are set and this is omitted."
        ),
    )
    parser.add_argument(
        "--task-reward-scale-warmup-updates",
        type=int,
        default=None,
        help=(
            "JAX-only update-count warmup for task reward scaling. When set, "
            "this takes precedence over --task-reward-scale-warmup-steps."
        ),
    )
    parser.add_argument(
        "--task-reward-scale-ramp-updates",
        type=int,
        default=None,
        help=(
            "JAX-only update-count ramp for task reward scaling. When set, "
            "this takes precedence over --task-reward-scale-ramp-steps."
        ),
    )
    parser.add_argument(
        "--phi-beta-warmup-updates",
        type=int,
        default=0,
        help="JAX-only PPO updates to hold phi shaping at --phi-beta-start.",
    )
    parser.add_argument(
        "--phi-beta-ramp-updates",
        type=int,
        default=1,
        help="JAX-only PPO updates to ramp phi shaping beta from start to end.",
    )
    parser.add_argument(
        "--intent-selector-alpha-warmup-updates",
        type=int,
        default=0,
        help="PPO updates to wait before ramping selector alpha.",
    )
    parser.add_argument(
        "--intent-selector-alpha-ramp-updates",
        type=int,
        default=1,
        help="PPO updates over which selector alpha ramps from start to end.",
    )
    parser.add_argument(
        "--intent-selector-eps-warmup-updates",
        type=int,
        default=0,
        help="PPO updates to wait before ramping selector epsilon floor.",
    )
    parser.add_argument(
        "--intent-selector-eps-ramp-updates",
        type=int,
        default=1,
        help="PPO updates over which selector epsilon ramps from start to end.",
    )
    parser.add_argument(
        "--intent-selector-learning-rate",
        type=float,
        default=None,
        help=(
            "Optional selector PPO optimizer learning rate. Defaults to "
            "--learning-rate when omitted."
        ),
    )
    parser.add_argument(
        "--policy-seed",
        type=int,
        default=0,
        help="Random seed used for policy init and rollout randomness.",
    )
    parser.add_argument(
        "--rollout-horizon",
        type=int,
        default=64,
        help="Rollout horizon per PPO update.",
    )
    parser.add_argument(
        "--single-episode-rollouts",
        action="store_true",
        help=(
            "Do not reset completed envs inside a JAX training rollout. "
            "Post-terminal slots are masked out of PPO, making each env "
            "contribute at most one possession per update."
        ),
    )
    parser.add_argument(
        "--ppo-completed-episodes-only",
        action="store_true",
        help=(
            "Train PPO only on transitions from episodes that start and finish "
            "inside the collected rollout. Transitions are weighted so each "
            "completed episode has equal total PPO weight."
        ),
    )
    parser.add_argument(
        "--num-updates",
        type=int,
        default=500,
        help="Number of PPO update cycles to run in train-loop mode.",
    )
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ppo-clip-range", type=float, default=0.2)
    parser.add_argument(
        "--policy-update-epochs",
        type=int,
        default=1,
        help="Number of PPO update epochs per rollout.",
    )
    parser.add_argument(
        "--ppo-minibatches",
        type=int,
        default=1,
        help=(
            "Number of shuffled PPO minibatches per update epoch. "
            "Set 1 to keep full-batch update behavior."
        ),
    )
    parser.add_argument(
        "--run-train-loop",
        action="store_true",
        help="Run the multi-update train loop instead of scaffold timing.",
    )
    parser.add_argument(
        "--log-every-updates",
        type=int,
        default=10,
        help="How often to append scalar train-history entries.",
    )
    parser.add_argument(
        "--eval-every-updates",
        type=int,
        default=50,
        help="How often to run deterministic eval rollouts. Set <=0 to disable.",
    )
    parser.add_argument(
        "--eval-horizon",
        type=int,
        default=64,
        help="Deterministic eval rollout horizon.",
    )
    parser.add_argument(
        "--eval-deploy-every-updates",
        type=int,
        default=0,
        help=(
            "How often to run fixed-seed same-policy argmax-vs-argmax deploy eval. "
            "Set <=0 to disable."
        ),
    )
    parser.add_argument(
        "--eval-deploy-batches",
        type=int,
        default=20,
        help="Number of kernel-sized batches per deploy evaluation event.",
    )
    parser.add_argument(
        "--eval-deploy-horizon",
        type=int,
        default=128,
        help="Maximum episode steps for deploy evaluation.",
    )
    parser.add_argument(
        "--eval-deploy-seed",
        type=int,
        default=2000000,
        help="Fixed root seed reused for every deploy evaluation event.",
    )
    parser.add_argument(
        "--historical-eval-updates",
        type=str,
        default="",
        help=(
            "Comma-separated update milestones for additive historical-opponent "
            "evaluation, for example '100,2000,5000'. Each milestone forces a "
            "checkpoint and evaluates against every reached milestone, including itself."
        ),
    )
    parser.add_argument(
        "--historical-eval-episodes",
        type=int,
        default=200,
        help=(
            "Episodes per historical matchup. Must be even so candidate Team A and "
            "Team B assignments are balanced."
        ),
    )
    parser.add_argument(
        "--historical-eval-horizon",
        type=int,
        default=1024,
        help="Maximum environment steps per historical-evaluation episode.",
    )
    parser.add_argument(
        "--historical-eval-seed",
        type=int,
        default=3000000,
        help="Fixed root seed used for reproducible historical-opponent matchups.",
    )
    parser.add_argument(
        "--max-eval-dumps",
        type=int,
        default=4,
        help="Maximum number of eval trajectory dumps to keep in JSON output.",
    )
    parser.add_argument(
        "--eval-trajectory-env-index",
        type=int,
        default=0,
        help="Which env index from the eval batch to serialize.",
    )
    parser.add_argument(
        "--log-mlflow",
        action="store_true",
        help="Log params and scalar metrics to MLflow.",
    )
    parser.add_argument(
        "--mlflow-metric-profile",
        choices=("core", "full"),
        default="core",
        help=(
            "MLflow chart volume. 'core' logs a compact set of aggregate train "
            "and evaluation diagnostics while keeping internal summaries and "
            "artifacts unchanged; 'full' logs every scalar."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="",
        help=(
            "Optional local directory for persistent periodic and final checkpoints. "
            "If omitted and --log-mlflow is enabled, checkpoints are staged "
            "temporarily and uploaded to MLflow only."
        ),
    )
    parser.add_argument(
        "--checkpoint-every-updates",
        type=int,
        default=0,
        help=(
            "Save a numbered checkpoint every N updates in fixed mode, or use "
            "this as the capped late-training interval in logarithmic mode. "
            "Final update is always saved when checkpoint publishing is enabled."
        ),
    )
    parser.add_argument(
        "--checkpoint-schedule",
        choices=("fixed", "log"),
        default="fixed",
        help=(
            "Periodic checkpoint cadence. 'fixed' preserves the legacy modulo "
            "behavior. 'log' starts from --checkpoint-log-initial-updates and "
            "grows logarithmically until capped by --checkpoint-every-updates."
        ),
    )
    parser.add_argument(
        "--checkpoint-log-initial-updates",
        type=int,
        default=1,
        help=(
            "Initial periodic checkpoint interval for --checkpoint-schedule log. "
            "Ignored in fixed mode."
        ),
    )
    parser.add_argument(
        "--checkpoint-log-ramp-updates",
        type=int,
        default=0,
        help=(
            "Update index by which the logarithmic checkpoint interval reaches "
            "--checkpoint-every-updates. 0 means --num-updates."
        ),
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default="",
        help="Resume train-loop state from a saved JAX checkpoint.",
    )
    parser.add_argument(
        "--resume-reset-env-state",
        action="store_true",
        help=(
            "When resuming, restore model/optimizer/RNG/update state but reset transient "
            "batched env rows. This is the default for --continue-run-id unless "
            "--continue-preserve-env-state is supplied."
        ),
    )
    parser.add_argument(
        "--resume-reset-intent-discriminator-state",
        action="store_true",
        help=(
            "When resuming, reset the auxiliary intent discriminator params, optimizer, "
            "and bonus normalization stats. This is the default for --continue-run-id unless "
            "--continue-preserve-intent-discriminator-state is supplied."
        ),
    )
    parser.add_argument(
        "--continue-preserve-env-state",
        action="store_true",
        help=(
            "With --continue-run-id, restore the checkpoint's batched env rows and RNG "
            "instead of the default reset."
        ),
    )
    parser.add_argument(
        "--continue-preserve-intent-discriminator-state",
        action="store_true",
        help=(
            "With --continue-run-id, restore the checkpoint's auxiliary intent "
            "discriminator params, optimizer, and bonus stats instead of the default reset."
        ),
    )
    parser.add_argument(
        "--continue-artifact",
        type=str,
        default="",
        help=(
            "Optional checkpoint artifact path/name for --continue-run-id. "
            "Defaults to the tagged/latest artifact, or the basename of "
            "--resume-checkpoint when provided."
        ),
    )
    parser.add_argument(
        "--continue-opponent-pool-size",
        type=int,
        default=-1,
        help=(
            "Number of recent checkpoint artifacts from --continue-run-id to "
            "seed into the opponent pool. -1 uses --opponent-pool-size; 0 disables seeding."
        ),
    )
    parser.add_argument(
        "--continue-cache-dir",
        type=str,
        default="artifacts/mlflow_checkpoints",
        help="Persistent local cache directory for MLflow continuation checkpoint downloads.",
    )
    parser.add_argument(
        "--control-file",
        type=str,
        default="",
        help=(
            "Optional JSON control file checked at PPO update boundaries. "
            "Supported actions are checkpoint and pause."
        ),
    )
    parser.add_argument(
        "--training-status-file",
        type=str,
        default="",
        help="Optional JSON status file updated atomically during the train loop.",
    )
    parser.add_argument(
        "--frozen-opponent-checkpoint",
        type=str,
        default="",
        help="Optional local JAX checkpoint directory to use as the frozen opponent.",
    )
    parser.add_argument(
        "--frozen-opponent-run-id",
        type=str,
        default="",
        help="Optional MLflow run id whose latest JAX checkpoint should be used as the frozen opponent.",
    )
    parser.add_argument(
        "--frozen-opponent-artifact",
        type=str,
        default="",
        help="Optional MLflow artifact path/name for --frozen-opponent-run-id. Defaults to the tagged/latest JAX checkpoint.",
    )
    parser.add_argument(
        "--disable-opponent-pool",
        action="store_true",
        help="Keep a provided frozen opponent fixed instead of resampling from saved JAX checkpoints.",
    )
    parser.add_argument(
        "--grouped-opponent-sampling",
        action="store_true",
        help=(
            "Sample multiple frozen opponent checkpoints per JAX rollout batch "
            "and assign contiguous env-row groups to each opponent."
        ),
    )
    parser.add_argument(
        "--opponent-group-count",
        type=int,
        default=8,
        help=(
            "Maximum number of sampled opponent checkpoint groups per JAX batch "
            "when --grouped-opponent-sampling is enabled."
        ),
    )
    parser.add_argument(
        "--opponent-deterministic-episode-prob",
        type=float,
        default=0.0,
        help=(
            "Probability that a frozen opponent episode uses deterministic argmax "
            "actions instead of sampled actions. The choice is held fixed until "
            "that env row resets. Used as a constant fallback when no schedule "
            "start/end is provided."
        ),
    )
    parser.add_argument(
        "--opponent-deterministic-episode-prob-start",
        type=float,
        default=None,
        help=(
            "Optional starting probability for a linear schedule controlling "
            "frozen-opponent deterministic argmax episodes."
        ),
    )
    parser.add_argument(
        "--opponent-deterministic-episode-prob-end",
        type=float,
        default=None,
        help=(
            "Optional ending probability for a linear schedule controlling "
            "frozen-opponent deterministic argmax episodes."
        ),
    )
    parser.add_argument(
        "--opponent-deterministic-episode-prob-warmup-updates",
        type=int,
        default=0,
        help="Updates to hold the deterministic-opponent schedule at the start value.",
    )
    parser.add_argument(
        "--opponent-deterministic-episode-prob-ramp-updates",
        type=int,
        default=1,
        help="Updates over which to linearly ramp deterministic-opponent probability.",
    )
    parser.add_argument(
        "--enable-rebounds",
        action="store_true",
        help=(
            "Enable half-court offensive rebound continuation after missed shots. "
            "Requires --rebound-table-model-dir."
        ),
    )
    parser.add_argument(
        "--enable-multi-possession",
        action="store_true",
        help=(
            "Run a fixed-length half-court game made of multiple possessions. "
            "Opening possession is sampled 50/50; start templates are disabled."
        ),
    )
    parser.add_argument(
        "--multi-possession-limit",
        type=int,
        default=25,
        help=(
            "Completed offensive possessions per team in a multi-possession game "
            "(25 means 25 for each team, 50 combined)."
        ),
    )
    parser.add_argument(
        "--multi-possession-limit-start",
        type=int,
        default=None,
        help=(
            "Optional starting per-team possession limit for the episode-length "
            "curriculum. Omit to keep fixed-length training."
        ),
    )
    parser.add_argument(
        "--multi-possession-limit-end",
        type=int,
        default=None,
        help=(
            "Optional final per-team possession limit for the episode-length "
            "curriculum. Defaults to --multi-possession-limit."
        ),
    )
    parser.add_argument(
        "--multi-possession-limit-ramp-updates",
        type=int,
        default=0,
        help=(
            "Training updates over which to linearly ramp the integer per-team "
            "possession limit. Update 1 uses the start value."
        ),
    )
    parser.add_argument(
        "--multi-possession-overtime-round-cap",
        type=int,
        default=0,
        help=(
            "Maximum paired overtime rounds before a tied game ends as a tie. "
            "Zero derives the cap from that episode's per-team possession limit."
        ),
    )
    parser.add_argument(
        "--inbound-deadline-steps",
        type=int,
        default=5,
        help=(
            "Simulation steps allowed to release a baseline inbound pass; "
            "one current environment step represents one countdown second."
        ),
    )
    parser.add_argument(
        "--check-deadline-steps",
        type=int,
        default=5,
        help="Simulation steps allowed for the receiving team to collect a check ball.",
    )
    parser.add_argument(
        "--check-setup-steps",
        type=int,
        default=0,
        help=(
            "Movement-only setup steps before the receiving team's check pickup "
            "clock begins; zero preserves the immediate-check behavior."
        ),
    )
    parser.add_argument(
        "--made-basket-restart-mode",
        choices=("baseline_inbound", "check", "direct_handoff"),
        default="baseline_inbound",
        help=(
            "Restart after made baskets with the legacy baseline inbound, a "
            "protected top-of-key check, or an immediate random handoff."
        ),
    )
    parser.add_argument(
        "--multi-possession-use-inbounds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use dead-ball inbound plays in multi-possession games. Disable "
            "for the no-inbounds ablation, which gives a random receiving "
            "player the ball in place."
        ),
    )
    parser.add_argument(
        "--multi-possession-reward-mode",
        choices=("win_loss", "point_differential", "scoring_events"),
        default="win_loss",
        help=(
            "Multi-possession objective: terminal +/-1/0 outcome, terminal "
            "team-A point differential, or immediate actual scoring events."
        ),
    )
    parser.add_argument(
        "--score-potential-scale",
        type=float,
        default=1.0,
        help="Scale for the fixed-team score-difference potential in multi-possession games.",
    )
    parser.add_argument(
        "--game-winner-reward",
        type=float,
        default=0.0,
        help=(
            "Zero-sum terminal bonus awarded once to the winning stable team "
            "after regulation or paired overtime."
        ),
    )
    parser.add_argument(
        "--multi-possession-aux-rewards-enabled",
        action="store_true",
        help=(
            "Explicitly add the legacy pass/assist/violation/EP/rebound reward "
            "bundle to the selected multi-possession game objective."
        ),
    )
    parser.add_argument(
        "--rebound-table-model-dir",
        type=str,
        default="",
        help="Path to a fitted rebound target table artifact directory.",
    )
    parser.add_argument(
        "--rebound-target-temperature",
        type=float,
        default=1.0,
        help="Temperature applied when sampling rebound landing target cells.",
    )
    parser.add_argument(
        "--rebound-target-uniform-mix",
        type=float,
        default=0.0,
        help="Uniform mixture weight for rebound target sampling, in [0, 1].",
    )
    parser.add_argument(
        "--rebound-winner-distance-weight",
        type=float,
        default=1.0,
        help="Distance penalty weight for choosing the rebound winner from the target cell.",
    )
    parser.add_argument(
        "--rebound-basket-position-weight",
        type=float,
        default=0.0,
        help=(
            "Penalty weight for rebounders farther from the basket than the sampled "
            "rebound target. Zero preserves distance-only winner logits."
        ),
    )
    parser.add_argument(
        "--rebound-winner-temperature",
        type=float,
        default=1.0,
        help="Temperature applied when sampling the rebound winner from distance logits.",
    )
    parser.add_argument(
        "--rebound-skill-std",
        type=float,
        default=0.0,
        help="Standard deviation for per-player rebound skill sampled at episode reset.",
    )
    parser.add_argument(
        "--rebound-skill-sampling-mode",
        type=str,
        default="gaussian",
        choices=("gaussian", "one_high_per_team"),
        help="How per-episode rebound skills are sampled. one_high_per_team picks one high-skill player per team and assigns low skill to teammates.",
    )
    parser.add_argument(
        "--rebound-skill-high",
        type=float,
        default=1.0,
        help="High rebound skill assigned to each team's specialist when rebound-skill-sampling-mode is one_high_per_team.",
    )
    parser.add_argument(
        "--rebound-skill-low",
        type=float,
        default=-0.25,
        help="Teammate rebound skill assigned when rebound-skill-sampling-mode is one_high_per_team.",
    )
    parser.add_argument(
        "--rebound-skill-weight",
        type=float,
        default=0.0,
        help="Skill-to-distance offset weight used in rebound winner logits.",
    )
    parser.add_argument(
        "--rebound-contest-mode",
        type=str,
        default="global_contest",
        choices=("global_contest", "local_contest"),
        help="Rebound winner contest mode: global_contest uses all players; local_contest masks to players within rebound_contest_radius of the target.",
    )
    parser.add_argument(
        "--rebound-contest-radius",
        type=int,
        default=1,
        help="Fixed hex radius around the rebound target for local_contest eligibility.",
    )
    parser.add_argument(
        "--rebound-obs-top-n-targets",
        type=int,
        default=0,
        help="Observation-only top-N rebound target approximation. Zero keeps exact full-table observation features.",
    )
    parser.add_argument(
        "--rebound-win-prob-features",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add each player's conditional rebound-win probability and the global "
            "conditional offensive-rebound probability to policy observations. "
            "Disabled by default to preserve the current observation schema."
        ),
    )
    parser.add_argument(
        "--rebound-target-observation-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Expose expected rebound-target coordinates, target entropy, each player\x27s "
            "distance to the expected target, and the rebound-specialist marker. "
            "Disable for the skill-only rebound-observation ablation."
        ),
    )
    parser.add_argument(
        "--offensive-rebound-shot-clock-reset",
        type=int,
        default=14,
        help=(
            "Shot-clock value used after an offensive rebound when the current clock "
            "is lower than this value."
        ),
    )
    parser.add_argument(
        "--enable-rebound-reward-redistribution",
        action="store_true",
        help=(
            "Advance part of the offense's eventual possession reward at an offensive "
            "rebound, then settle the same amount at possession end."
        ),
    )
    parser.add_argument(
        "--offensive-rebound-reward-advance",
        type=float,
        default=0.4,
        help="Offense reward advanced at a qualifying offensive rebound.",
    )
    parser.add_argument(
        "--rebound-reward-once-per-possession",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Advance rebound reward only on the first offensive rebound in a possession. "
            "Use --no-rebound-reward-once-per-possession to advance it on every ORB."
        ),
    )
    parser.add_argument(
        "--rebound-critic-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable an auxiliary rebound-outcome critic and a masked PPO positioning "
            "loss on actual rebound events; ordinary environment rewards are unchanged."
        ),
    )
    parser.add_argument(
        "--rebound-critic-policy-coef",
        type=float,
        default=0.05,
        help="Weight for the masked rebound-positioning PPO auxiliary loss.",
    )
    parser.add_argument(
        "--rebound-critic-value-coef",
        type=float,
        default=0.5,
        help="Weight for the rebound-outcome critic regression loss.",
    )
    parser.add_argument(
        "--rebound-counterfactual-positioning-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add an auxiliary PPO positioning loss on rebound events using each "
            "mover's counterfactual team softmax-mass gain versus its pre-move cell."
        ),
    )
    parser.add_argument(
        "--rebound-counterfactual-positioning-coef",
        type=float,
        default=0.02,
        help="Weight for the per-player rebound counterfactual PPO auxiliary loss.",
    )
    parser.add_argument(
        "--dump-config-schema", action="store_true", help=argparse.SUPPRESS
    )
    return parser


def parse_args(argv=None):
    argv_list = list(sys.argv[1:] if argv is None else argv)
    _reject_legacy_opponent_flag(argv_list)
    parser = build_train_parser()
    args = parser.parse_args(argv_list)
    if bool(getattr(args, "use_set_obs", False)):
        args.policy_model = "attention"
    if str(args.policy_model) == "attention":
        args.use_set_obs = True
    return args


def _config_category(dest: str) -> str:
    prefixes = (
        ("intent_", "Intent learning"),
        ("rebound_", "Rebounding"),
        ("pass_", "Passing"),
        ("opponent_", "Opponent sampling"),
        ("historical_eval_", "Historical evaluation"),
        ("eval_", "Evaluation"),
        ("mlflow_", "MLflow"),
        ("multi_possession_", "Multi-possession"),
        ("check_", "Restarts and checking"),
        ("inbound_", "Restarts and checking"),
        ("shot_", "Shooting"),
        ("three_", "Shooting"),
        ("defender_", "Defense"),
        ("illegal_defense", "Defense"),
        ("ppo_", "PPO"),
        ("policy_", "Policy"),
        ("attention_", "Policy"),
        ("checkpoint_", "Runtime and checkpoints"),
        ("resume_", "Runtime and checkpoints"),
    )
    for prefix, category in prefixes:
        if dest.startswith(prefix):
            return category
    return "Environment and training"


def build_train_config_schema(args) -> list[dict[str, Any]]:
    parser = build_train_parser()
    app_internal = {
        "run_train_loop",
        "dump_config_schema",
        "checkpoint_dir",
        "control_file",
        "training_status_file",
        "resume_checkpoint",
        "mlflow_resume_run_id",
    }
    fields: list[dict[str, Any]] = []
    for action in parser._actions:
        if not action.option_strings or action.dest in {"help", "dump_config_schema"}:
            continue
        if action.help is argparse.SUPPRESS:
            continue
        is_bool_action = isinstance(
            action,
            (
                argparse._StoreTrueAction,
                argparse._StoreFalseAction,
                argparse.BooleanOptionalAction,
            ),
        )
        effective_value = getattr(args, action.dest, action.default)
        lambda_bool = False
        if not is_bool_action and action.type is not None:
            try:
                lambda_bool = isinstance(action.type("true"), bool) and isinstance(
                    action.type("false"), bool
                )
            except (TypeError, ValueError):
                lambda_bool = False
        is_bool = is_bool_action or lambda_bool
        boolean_mode = None
        if isinstance(action, argparse.BooleanOptionalAction):
            boolean_mode = "boolean_optional"
        elif isinstance(action, argparse._StoreTrueAction):
            boolean_mode = "store_true"
        elif isinstance(action, argparse._StoreFalseAction):
            boolean_mode = "store_false"
        elif lambda_bool:
            boolean_mode = "value"
        is_list = action.nargs in {"+", "*"} or (
            isinstance(action.nargs, int) and action.nargs > 0
        )
        inferred_scalar_type = getattr(action.type, "__name__", None)
        if inferred_scalar_type == "<lambda>":
            sample = effective_value if effective_value is not None else action.default
            if sample is None:
                try:
                    sample = action.type("1")
                except (TypeError, ValueError):
                    sample = ""
            inferred_scalar_type = type(sample).__name__
        value_type = "bool" if is_bool else "list" if is_list else inferred_scalar_type
        if not value_type:
            value_type = (
                type(action.default).__name__ if action.default is not None else "str"
            )
        frozen = (
            action.dest in TRAIN_FROZEN_VALUES
            and action.dest not in JAX_ALLOWED_ENV_OVERRIDE_KEYS
        )
        fields.append(
            {
                "name": action.dest,
                "option_strings": list(action.option_strings),
                "type": value_type,
                "boolean_mode": boolean_mode,
                "item_type": (
                    getattr(action.type, "__name__", "str") if is_list else None
                ),
                "nargs": action.nargs,
                "default": to_builtin(action.default),
                "effective_value": to_builtin(effective_value),
                "choices": (
                    to_builtin(list(action.choices))
                    if action.choices is not None
                    else None
                ),
                "help": str(action.help or ""),
                "category": _config_category(action.dest),
                "editable": action.dest not in app_internal and not frozen,
                "internal": action.dest in app_internal,
                "frozen": frozen,
            }
        )
    return fields


def _values_match(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        return np.isclose(float(actual), float(expected), atol=1e-8, rtol=0.0)
    return actual == expected


def validate_train_args(args) -> None:
    mismatches: list[str] = []
    for key, expected in TRAIN_FROZEN_VALUES.items():
        if key in JAX_ALLOWED_ENV_OVERRIDE_KEYS:
            continue
        if (
            key == "use_set_obs"
            and str(getattr(args, "policy_model", "mlp")) == "attention"
        ):
            continue
        actual = getattr(args, key)
        if not _values_match(actual, expected):
            mismatches.append(f"{key}={actual!r} expected {expected!r}")
    if mismatches:
        raise SystemExit(
            "JAX trainer uses a frozen reduced structural config. Unsupported overrides: "
            + ", ".join(mismatches)
        )
    ppo_minibatches = int(getattr(args, "ppo_minibatches", 1))
    if ppo_minibatches < 1:
        raise SystemExit("--ppo-minibatches must be >= 1.")
    for key in ("eval_deploy_every_updates", "eval_deploy_batches", "eval_deploy_seed"):
        if int(getattr(args, key, 0)) < 0:
            raise SystemExit(f"--{key.replace('_', '-')} must be >= 0.")
    if int(getattr(args, "eval_deploy_horizon", 128)) < 1:
        raise SystemExit("--eval-deploy-horizon must be >= 1.")
    historical_updates = _historical_eval_updates(args)
    if historical_updates:
        historical_episodes = int(getattr(args, "historical_eval_episodes", 0))
        if historical_episodes < 2 or historical_episodes % 2:
            raise SystemExit(
                "--historical-eval-episodes must be an even integer of at least 2."
            )
        if int(getattr(args, "historical_eval_horizon", 0)) < 1:
            raise SystemExit("--historical-eval-horizon must be >= 1.")
        if not str(getattr(args, "checkpoint_dir", "") or "").strip() and not bool(
            getattr(args, "log_mlflow", False)
        ):
            raise SystemExit(
                "--historical-eval-updates requires --checkpoint-dir or --log-mlflow "
                "so milestone policies can be pinned."
            )
    role_multiplier = (
        len(TRAINING_ROLES) if bool(getattr(args, "run_train_loop", False)) else 1
    )
    ppo_sample_count = (
        int(getattr(args, "kernel_batch_size"))
        * int(getattr(args, "rollout_horizon"))
        * role_multiplier
    )
    if ppo_sample_count % ppo_minibatches != 0:
        raise SystemExit(
            "--ppo-minibatches must evenly divide the PPO batch size. "
            f"ppo_batch_size={ppo_sample_count}, ppo_minibatches={ppo_minibatches}."
        )
    for key in (
        "players",
        "court_rows",
        "court_cols",
        "shot_clock",
        "min_shot_clock",
        "spawn_distance",
    ):
        value = getattr(args, key, None)
        if value is not None and int(value) < 1:
            raise SystemExit(f"--{key.replace('_', '-')} must be >= 1.")
    for key in ("defender_spawn_distance", "defender_guard_distance", "assist_window"):
        value = getattr(args, key, None)
        if value is not None and int(value) < 0:
            raise SystemExit(f"--{key.replace('_', '-')} must be >= 0.")
    max_spawn_distance = getattr(args, "max_spawn_distance", None)
    if max_spawn_distance is not None and int(max_spawn_distance) < 1:
        raise SystemExit("--max-spawn-distance must be >= 1 when set.")
    if int(getattr(args, "min_shot_clock", 1)) > int(getattr(args, "shot_clock", 1)):
        raise SystemExit("--min-shot-clock must be <= --shot-clock.")
    for key in (
        "three_point_distance",
        "three_pt_extra_hex_decay",
        "shot_pressure_lambda",
        "defender_pressure_decay_lambda",
        "steal_perp_decay",
        "steal_distance_factor",
        "pass_passer_pressure_weight",
        "pass_receiver_pressure_weight",
        "pass_lob_receiver_distance",
        "pass_speed",
        "defender_reaction_time",
        "defender_speed",
        "defender_reach_radius",
        "reaction_softness",
        "passer_pressure_decay",
        "receiver_alignment_width",
    ):
        value = getattr(args, key, None)
        if value is not None and float(value) < 0.0:
            raise SystemExit(f"--{key.replace('_', '-')} must be >= 0.")
    three_point_short_distance = getattr(args, "three_point_short_distance", None)
    if (
        three_point_short_distance is not None
        and float(three_point_short_distance) < 0.0
    ):
        raise SystemExit("--three-point-short-distance must be >= 0 when set.")
    for key in (
        "shot_pressure_max",
        "defender_pressure_turnover_chance",
        "base_steal_rate",
        "steal_position_weight_min",
        "pass_lob_lane_multiplier",
        "base_passer_risk",
        "base_receiver_risk",
        "receiver_alignment_min",
        "max_receiver_hazard",
        "lane_weight",
    ):
        value = getattr(args, key, None)
        if value is not None and (float(value) < 0.0 or float(value) > 1.0):
            raise SystemExit(f"--{key.replace('_', '-')} must be in [0, 1].")
    pass_interception_model = (
        str(getattr(args, "pass_interception_model", "line") or "line").strip().lower()
    )
    if pass_interception_model not in {
        "line",
        "lob_aware",
        "lob-aware",
        "lob",
        "reaction",
        "speed",
        "speed_based",
        "speed-based",
    }:
        raise SystemExit(
            "--pass-interception-model must be one of: line, lob_aware, reaction."
        )
    shot_pressure_arc_degrees = float(getattr(args, "shot_pressure_arc_degrees", 0.0))
    if shot_pressure_arc_degrees <= 0.0 or shot_pressure_arc_degrees > 360.0:
        raise SystemExit("--shot-pressure-arc-degrees must be in (0, 360].")
    if int(getattr(args, "num_intents", 8)) < 1:
        raise SystemExit("--num-intents must be >= 1.")
    if int(getattr(args, "intent_commitment_steps", 4)) < 1:
        raise SystemExit("--intent-commitment-steps must be >= 1.")
    for key in (
        "opponent_deterministic_episode_prob",
        "opponent_deterministic_episode_prob_start",
        "opponent_deterministic_episode_prob_end",
    ):
        raw_value = getattr(args, key, None)
        if raw_value is None:
            continue
        value = float(raw_value)
        if value < 0.0 or value > 1.0:
            raise SystemExit(f"--{key.replace('_', '-')} must be in [0, 1].")
    for key in (
        "opponent_deterministic_episode_prob_warmup_updates",
        "opponent_deterministic_episode_prob_ramp_updates",
    ):
        if int(getattr(args, key, 0)) < 0:
            raise SystemExit(f"--{key.replace('_', '-')} must be >= 0.")
    if bool(getattr(args, "enable_rebounds", False)):
        rebound_table_model_dir = str(
            getattr(args, "rebound_table_model_dir", "") or ""
        ).strip()
        if not rebound_table_model_dir:
            raise SystemExit("--enable-rebounds requires --rebound-table-model-dir.")
        if not Path(rebound_table_model_dir).exists():
            raise SystemExit(
                f"--rebound-table-model-dir does not exist: {rebound_table_model_dir}"
            )
    if bool(getattr(args, "enable_multi_possession", False)):
        if int(getattr(args, "multi_possession_limit", 25)) < 1:
            raise SystemExit("--multi-possession-limit must be >= 1.")
        for key in ("multi_possession_limit_start", "multi_possession_limit_end"):
            value = getattr(args, key, None)
            if value is not None and int(value) < 1:
                raise SystemExit(f"--{key.replace('_', '-')} must be >= 1.")
        if int(getattr(args, "multi_possession_limit_ramp_updates", 0)) < 0:
            raise SystemExit("--multi-possession-limit-ramp-updates must be >= 0.")
        if int(getattr(args, "multi_possession_overtime_round_cap", 0)) < 0:
            raise SystemExit("--multi-possession-overtime-round-cap must be >= 0.")
        if int(getattr(args, "inbound_deadline_steps", 5)) < 1:
            raise SystemExit("--inbound-deadline-steps must be >= 1.")
        if int(getattr(args, "check_deadline_steps", 5)) < 1:
            raise SystemExit("--check-deadline-steps must be >= 1.")
        if int(getattr(args, "check_setup_steps", 0)) < 0:
            raise SystemExit("--check-setup-steps must be >= 0.")
        if bool(getattr(args, "start_template_enabled", False)):
            raise SystemExit(
                "--start-template-enabled is incompatible with --enable-multi-possession."
            )
        if bool(getattr(args, "single_episode_rollouts", False)):
            raise SystemExit(
                "--single-episode-rollouts is incompatible with continuous multi-possession training."
            )
        if bool(getattr(args, "ppo_completed_episodes_only", False)):
            raise SystemExit(
                "--ppo-completed-episodes-only is incompatible with continuous multi-possession training."
            )
        reward_mode = str(
            getattr(args, "multi_possession_reward_mode", "win_loss") or "win_loss"
        )
        if reward_mode not in {"win_loss", "point_differential", "scoring_events"}:
            raise SystemExit(
                "--multi-possession-reward-mode must be 'win_loss', "
                "'point_differential', or 'scoring_events'."
            )
        if float(getattr(args, "score_potential_scale", 1.0)) < 0.0:
            raise SystemExit("--score-potential-scale must be >= 0.")
        if float(getattr(args, "game_winner_reward", 0.0)) < 0.0:
            raise SystemExit("--game-winner-reward must be >= 0.")
        if reward_mode == "scoring_events":
            if bool(getattr(args, "enable_phi_shaping", False)):
                raise SystemExit(
                    "--enable-phi-shaping is incompatible with "
                    "--multi-possession-reward-mode scoring_events."
                )
            if bool(getattr(args, "multi_possession_aux_rewards_enabled", False)):
                raise SystemExit(
                    "--multi-possession-aux-rewards-enabled is incompatible with "
                    "--multi-possession-reward-mode scoring_events."
                )
        elif bool(getattr(args, "enable_phi_shaping", False)):
            if bool(getattr(args, "phi_use_ball_handler_only", False)):
                raise SystemExit(
                    "--phi-use-ball-handler-only is incompatible with multi-possession score potential shaping."
                )
            if float(getattr(args, "phi_blend_weight", 0.0)) != 0.0:
                raise SystemExit(
                    "--phi-blend-weight is incompatible with multi-possession score potential shaping."
                )
    else:
        if bool(getattr(args, "multi_possession_aux_rewards_enabled", False)):
            raise SystemExit(
                "--multi-possession-aux-rewards-enabled requires --enable-multi-possession."
            )
        if str(getattr(args, "multi_possession_reward_mode", "win_loss")) != "win_loss":
            raise SystemExit(
                "--multi-possession-reward-mode requires --enable-multi-possession."
            )
    for key in ("rebound_target_temperature", "rebound_winner_temperature"):
        value = float(getattr(args, key, 1.0))
        if value <= 0.0:
            raise SystemExit(f"--{key.replace('_', '-')} must be > 0.")
    rebound_target_uniform_mix = float(getattr(args, "rebound_target_uniform_mix", 0.0))
    if rebound_target_uniform_mix < 0.0 or rebound_target_uniform_mix > 1.0:
        raise SystemExit("--rebound-target-uniform-mix must be in [0, 1].")
    if float(getattr(args, "rebound_winner_distance_weight", 1.0)) < 0.0:
        raise SystemExit("--rebound-winner-distance-weight must be >= 0.")
    if float(getattr(args, "rebound_basket_position_weight", 0.0)) < 0.0:
        raise SystemExit("--rebound-basket-position-weight must be >= 0.")
    if float(getattr(args, "rebound_skill_std", 0.0)) < 0.0:
        raise SystemExit("--rebound-skill-std must be >= 0.")
    rebound_skill_sampling_mode = (
        str(getattr(args, "rebound_skill_sampling_mode", "gaussian") or "gaussian")
        .strip()
        .lower()
    )
    if rebound_skill_sampling_mode not in {"gaussian", "one_high_per_team"}:
        raise SystemExit(
            "--rebound-skill-sampling-mode must be 'gaussian' or 'one_high_per_team'."
        )
    if float(getattr(args, "rebound_skill_weight", 0.0)) < 0.0:
        raise SystemExit("--rebound-skill-weight must be >= 0.")
    rebound_contest_mode = (
        str(getattr(args, "rebound_contest_mode", "global_contest") or "global_contest")
        .strip()
        .lower()
    )
    if rebound_contest_mode not in {"global_contest", "local_contest"}:
        raise SystemExit(
            "--rebound-contest-mode must be 'global_contest' or 'local_contest'."
        )
    rebound_contest_radius = int(getattr(args, "rebound_contest_radius", 1))
    if rebound_contest_radius < 0:
        raise SystemExit("--rebound-contest-radius must be >= 0.")
    if int(getattr(args, "rebound_obs_top_n_targets", 0)) < 0:
        raise SystemExit("--rebound-obs-top-n-targets must be >= 0.")
    if int(getattr(args, "offensive_rebound_shot_clock_reset", 14)) < 1:
        raise SystemExit("--offensive-rebound-shot-clock-reset must be >= 1.")
    if float(getattr(args, "offensive_rebound_reward_advance", 0.4)) < 0.0:
        raise SystemExit("--offensive-rebound-reward-advance must be >= 0.")
    if bool(getattr(args, "enable_rebound_reward_redistribution", False)) and not bool(
        getattr(args, "enable_rebounds", False)
    ):
        raise SystemExit(
            "--enable-rebound-reward-redistribution requires --enable-rebounds."
        )
    if bool(getattr(args, "rebound_critic_enabled", False)) and not bool(
        getattr(args, "enable_rebounds", False)
    ):
        raise SystemExit("--rebound-critic-enabled requires --enable-rebounds.")
    if bool(
        getattr(args, "rebound_counterfactual_positioning_enabled", False)
    ) and not bool(getattr(args, "enable_rebounds", False)):
        raise SystemExit(
            "--rebound-counterfactual-positioning-enabled requires --enable-rebounds."
        )
    if float(getattr(args, "rebound_counterfactual_positioning_coef", 0.0)) < 0.0:
        raise SystemExit("--rebound-counterfactual-positioning-coef must be >= 0.")
    for key in ("rebound_critic_policy_coef", "rebound_critic_value_coef"):
        if float(getattr(args, key, 0.0)) < 0.0:
            raise SystemExit(f"--{key.replace('_', '-')} must be >= 0.")
    rebound_terminal_reward_mode = str(
        getattr(args, "rebound_terminal_reward_mode", "actual_points")
        or "actual_points"
    )
    if rebound_terminal_reward_mode not in {
        "actual_points",
        "last_shot_ep_on_defensive_rebound",
        "last_shot_ep",
    }:
        raise SystemExit(
            "--rebound-terminal-reward-mode must be 'actual_points', 'last_shot_ep_on_defensive_rebound', or 'last_shot_ep'."
        )
    for key in (
        "intent_null_prob",
        "defense_intent_null_prob",
        "intent_visible_to_defense_prob",
    ):
        value = float(getattr(args, key, 0.0))
        if value < 0.0 or value > 1.0:
            raise SystemExit(f"--{key.replace('_', '-')} must be in [0, 1].")
    if bool(getattr(args, "intent_embedding_enabled", False)):
        if str(getattr(args, "policy_model", "mlp")) != "attention":
            raise SystemExit(
                "--intent-embedding-enabled requires --policy-model attention."
            )
        if not (
            bool(getattr(args, "enable_intent_learning", False))
            or bool(getattr(args, "enable_defense_intent_learning", False))
        ):
            raise SystemExit(
                "--intent-embedding-enabled requires --enable-intent-learning or "
                "--enable-defense-intent-learning."
            )
    if int(getattr(args, "intent_embedding_dim", 16)) < 1:
        raise SystemExit("--intent-embedding-dim must be >= 1.")
    if float(getattr(args, "intent_conditioning_scale", 1.0)) < 0.0:
        raise SystemExit("--intent-conditioning-scale must be >= 0.")
    for key in ("ent_coef", "ent_coef_start", "ent_coef_end"):
        value = getattr(args, key, None)
        if value is not None and float(value) < 0.0:
            raise SystemExit(f"--{key.replace('_', '-')} must be >= 0.")
    ent_schedule = str(getattr(args, "ent_schedule", "linear")).lower()
    if ent_schedule not in {"linear", "exp"}:
        raise SystemExit("--ent-schedule must be one of: linear, exp.")
    entropy_decay_updates = getattr(args, "entropy_decay_updates", None)
    if entropy_decay_updates is not None and int(entropy_decay_updates) < 1:
        raise SystemExit("--entropy-decay-updates must be >= 1 when provided.")
    for key in ("task_reward_scale_start", "task_reward_scale_end"):
        value = getattr(args, key, None)
        if value is not None and float(value) < 0.0:
            raise SystemExit(f"--{key.replace('_', '-')} must be >= 0.")
    for key in ("task_reward_scale_warmup_updates", "task_reward_scale_ramp_updates"):
        value = getattr(args, key, None)
        if value is not None and int(value) < 0:
            raise SystemExit(f"--{key.replace('_', '-')} must be >= 0.")
    for key in ("phi_beta_start", "phi_beta_end"):
        value = getattr(args, key, None)
        if value is not None and float(value) < 0.0:
            raise SystemExit(f"--{key.replace('_', '-')} must be >= 0.")
    for key in ("phi_beta_warmup_updates", "phi_beta_ramp_updates"):
        if int(getattr(args, key, 0)) < 0:
            raise SystemExit(f"--{key.replace('_', '-')} must be >= 0.")
    phi_blend_weight = float(getattr(args, "phi_blend_weight", 0.0))
    if phi_blend_weight < 0.0 or phi_blend_weight > 1.0:
        raise SystemExit("--phi-blend-weight must be in [0, 1].")
    reward_shaping_gamma = getattr(args, "reward_shaping_gamma", None)
    if reward_shaping_gamma is not None and float(reward_shaping_gamma) < 0.0:
        raise SystemExit("--reward-shaping-gamma must be >= 0.")
    if bool(getattr(args, "intent_selector_enabled", False)):
        if str(getattr(args, "policy_model", "mlp")) != "attention":
            raise SystemExit(
                "--intent-selector-enabled requires --policy-model attention."
            )
        if not bool(getattr(args, "enable_intent_learning", False)):
            raise SystemExit(
                "--intent-selector-enabled requires --enable-intent-learning."
            )
        if not bool(getattr(args, "intent_embedding_enabled", False)):
            raise SystemExit(
                "--intent-selector-enabled requires --intent-embedding-enabled."
            )
    if int(getattr(args, "intent_selector_hidden_dim", 64)) < 1:
        raise SystemExit("--intent-selector-hidden-dim must be >= 1.")
    for key in (
        "intent_selector_alpha_start",
        "intent_selector_alpha_end",
        "intent_selector_eps_start",
        "intent_selector_eps_end",
    ):
        value = float(getattr(args, key, 0.0))
        if value < 0.0 or value > 1.0:
            raise SystemExit(f"--{key.replace('_', '-')} must be in [0, 1].")
    for key in (
        "intent_selector_alpha_warmup_updates",
        "intent_selector_alpha_ramp_updates",
        "intent_selector_eps_warmup_updates",
        "intent_selector_eps_ramp_updates",
        "intent_selector_train_every_rollouts",
        "intent_selector_max_samples_per_update",
        "intent_selector_min_play_steps",
    ):
        if int(getattr(args, key, 0)) < 0:
            raise SystemExit(f"--{key.replace('_', '-')} must be >= 0.")
    if int(getattr(args, "intent_selector_train_every_rollouts", 1)) < 1:
        raise SystemExit("--intent-selector-train-every-rollouts must be >= 1.")
    if int(getattr(args, "intent_selector_min_play_steps", 3)) < 1:
        raise SystemExit("--intent-selector-min-play-steps must be >= 1.")
    selector_learning_rate = getattr(args, "intent_selector_learning_rate", None)
    if selector_learning_rate is not None and float(selector_learning_rate) <= 0.0:
        raise SystemExit("--intent-selector-learning-rate must be > 0.")
    if int(getattr(args, "intent_sample_dump_size", 2048)) < 0:
        raise SystemExit("--intent-sample-dump-size must be >= 0.")
    if bool(getattr(args, "intent_diversity_enabled", False)):
        if not bool(getattr(args, "run_train_loop", False)):
            raise SystemExit(
                "--intent-diversity-enabled is supported only with --run-train-loop."
            )
        if not bool(getattr(args, "enable_intent_learning", False)):
            raise SystemExit(
                "--intent-diversity-enabled requires --enable-intent-learning."
            )
        if not bool(getattr(args, "intent_embedding_enabled", False)):
            raise SystemExit(
                "--intent-diversity-enabled requires --intent-embedding-enabled."
            )
        encoder_type = str(getattr(args, "intent_disc_encoder_type", "mlp_mean"))
        if encoder_type not in {"mlp_mean", "set_step"}:
            raise SystemExit(
                "JAX intent discriminator supports --intent-disc-encoder-type "
                "mlp_mean or set_step."
            )
        if encoder_type == "set_step":
            if str(getattr(args, "policy_model", "mlp")) != "attention":
                raise SystemExit(
                    f"--intent-disc-encoder-type {encoder_type} requires --policy-model attention."
                )
            hidden_dim = int(getattr(args, "intent_disc_hidden_dim", 128))
            num_heads = int(getattr(args, "attention_num_heads", 4))
            if hidden_dim % num_heads != 0:
                raise SystemExit(
                    "--intent-disc-hidden-dim must be divisible by "
                    f"--attention-num-heads for {encoder_type}."
                )
        if not bool(getattr(args, "intent_disc_current_policy_only", True)):
            raise SystemExit(
                "JAX intent discriminator currently requires --intent-disc-current-policy-only true."
            )
        if int(getattr(args, "intent_disc_batch_size", 256)) < 1:
            raise SystemExit("--intent-disc-batch-size must be >= 1.")
        if int(getattr(args, "intent_disc_updates_per_rollout", 2)) < 1:
            raise SystemExit("--intent-disc-updates-per-rollout must be >= 1.")
        disc_dropout = float(getattr(args, "intent_disc_dropout", 0.1))
        if disc_dropout < 0.0 or disc_dropout >= 1.0:
            raise SystemExit("--intent-disc-dropout must be in [0, 1).")
        holdout_fraction = float(
            getattr(args, "intent_disc_eval_holdout_fraction", 0.25)
        )
        if holdout_fraction < 0.0 or holdout_fraction > 1.0:
            raise SystemExit("--intent-disc-eval-holdout-fraction must be in [0, 1].")
        if getattr(args, "intent_diversity_warmup_updates", None) is not None:
            if int(getattr(args, "intent_diversity_warmup_updates")) < 0:
                raise SystemExit("--intent-diversity-warmup-updates must be >= 0.")
        if getattr(args, "intent_diversity_ramp_updates", None) is not None:
            if int(getattr(args, "intent_diversity_ramp_updates")) < 0:
                raise SystemExit("--intent-diversity-ramp-updates must be >= 0.")


def _multi_possession_final_limit(args) -> int:
    end = getattr(args, "multi_possession_limit_end", None)
    return max(
        1,
        int(getattr(args, "multi_possession_limit", 25) if end is None else end),
    )


def _multi_possession_start_limit(args) -> int:
    start = getattr(args, "multi_possession_limit_start", None)
    return max(
        1,
        int(_multi_possession_final_limit(args) if start is None else start),
    )


def _multi_possession_curriculum_enabled(args) -> bool:
    return (
        bool(getattr(args, "enable_multi_possession", False))
        and _multi_possession_start_limit(args) != _multi_possession_final_limit(args)
        and int(getattr(args, "multi_possession_limit_ramp_updates", 0)) > 0
    )


def _jax_env_config_from_args(args) -> dict[str, Any]:
    config = {
        key: to_builtin(getattr(args, key))
        for key in JAX_ENV_MLFLOW_PARAM_KEYS
        if hasattr(args, key)
    }
    # The final curriculum value is the canonical fixed limit used by native
    # evaluation and by tooling that does not run the training schedule.
    config["multi_possession_limit"] = _multi_possession_final_limit(args)
    config["multi_possession_schema_version"] = (
        MULTI_POSSESSION_SCHEMA_VERSION
        if bool(getattr(args, "enable_multi_possession", False))
        else 1
    )
    return config


_RESUME_ENV_CONFIG_ADDITIVE_DEFAULTS = {
    "rebound_skill_sampling_mode": "gaussian",
    "rebound_skill_high": 1.0,
    "rebound_skill_low": -0.25,
    "enable_rebound_reward_redistribution": False,
    "offensive_rebound_reward_advance": 0.4,
    "rebound_reward_once_per_possession": True,
    "rebound_counterfactual_positioning_enabled": False,
    "enable_multi_possession": False,
    "multi_possession_limit": 25,
    "multi_possession_limit_start": None,
    "multi_possession_limit_end": None,
    "multi_possession_limit_ramp_updates": 0,
    "multi_possession_overtime_round_cap": 0,
    "multi_possession_reward_mode": "win_loss",
    "score_potential_scale": 1.0,
    "game_winner_reward": 0.0,
    "multi_possession_aux_rewards_enabled": False,
    "multi_possession_use_inbounds": True,
    "made_basket_restart_mode": "baseline_inbound",
    "multi_possession_schema_version": 1,
    "inbound_deadline_steps": 5,
    "check_deadline_steps": 5,
    "check_setup_steps": 0,
}


def _compatible_env_config_for_resume(
    actual: dict[str, Any], expected: dict[str, Any]
) -> dict[str, Any]:
    out = dict(actual or {})
    for key in _RESUME_ENV_CONFIG_ADDITIVE_DEFAULTS:
        if key not in out and key in expected:
            if key == "made_basket_restart_mode":
                out[key] = (
                    "baseline_inbound"
                    if bool(out.get("multi_possession_use_inbounds", True))
                    else "direct_handoff"
                )
            else:
                out[key] = expected[key]
    return out


def build_trainer_config(args) -> TrainerConfig:
    return TrainerConfig(
        kernel_batch_size=int(args.kernel_batch_size),
        rollout_horizon=int(args.rollout_horizon),
        num_updates=int(args.num_updates),
        gamma=float(args.gamma),
        gae_lambda=float(args.gae_lambda),
        ppo_clip_range=float(args.ppo_clip_range),
        value_coef=float(args.vf_coef),
        entropy_coef=float(args.ent_coef),
        learning_rate=float(args.learning_rate),
        policy_update_epochs=int(args.policy_update_epochs),
        ppo_minibatches=int(args.ppo_minibatches),
        single_episode_rollouts=bool(getattr(args, "single_episode_rollouts", False)),
        ppo_completed_episodes_only=bool(
            getattr(args, "ppo_completed_episodes_only", False)
        ),
        rebound_critic_policy_coef=(
            float(getattr(args, "rebound_critic_policy_coef", 0.0))
            if bool(getattr(args, "rebound_critic_enabled", False))
            else 0.0
        ),
        rebound_critic_value_coef=(
            float(getattr(args, "rebound_critic_value_coef", 0.0))
            if bool(getattr(args, "rebound_critic_enabled", False))
            else 0.0
        ),
        rebound_counterfactual_positioning_coef=(
            float(getattr(args, "rebound_counterfactual_positioning_coef", 0.0))
            if bool(getattr(args, "rebound_counterfactual_positioning_enabled", False))
            else 0.0
        ),
    )


def _selector_learning_rate_for_args(args, trainer_config: TrainerConfig) -> float:
    override = getattr(args, "intent_selector_learning_rate", None)
    if override is None:
        return float(trainer_config.learning_rate)
    return float(override)


def _checkpoint_interval_for_update(args, update_index: int) -> int:
    max_interval = int(getattr(args, "checkpoint_every_updates", 0) or 0)
    if max_interval <= 0:
        return 0

    schedule = (
        str(getattr(args, "checkpoint_schedule", "fixed") or "fixed").strip().lower()
    )
    if schedule != "log":
        return max_interval

    min_interval = int(getattr(args, "checkpoint_log_initial_updates", 1) or 1)
    min_interval = max(1, min(min_interval, max_interval))
    if max_interval <= min_interval:
        return max_interval

    ramp_updates = int(getattr(args, "checkpoint_log_ramp_updates", 0) or 0)
    if ramp_updates <= 0:
        ramp_updates = int(getattr(args, "num_updates", max_interval) or max_interval)
    ramp_updates = max(1, ramp_updates)

    update = max(1, int(update_index))
    progress = float(np.log1p(update)) / float(np.log1p(ramp_updates))
    progress = max(0.0, min(1.0, progress))
    interval = int(round(min_interval + (max_interval - min_interval) * progress))
    return max(min_interval, min(max_interval, interval))


def _periodic_checkpoint_updates(args) -> set[int]:
    max_update = int(getattr(args, "num_updates", 0) or 0)
    max_interval = int(getattr(args, "checkpoint_every_updates", 0) or 0)
    if max_update <= 0 or max_interval <= 0:
        return set()

    schedule = (
        str(getattr(args, "checkpoint_schedule", "fixed") or "fixed").strip().lower()
    )
    if schedule != "log":
        return {
            update for update in range(1, max_update + 1) if update % max_interval == 0
        }

    due_updates: set[int] = set()
    min_interval = int(getattr(args, "checkpoint_log_initial_updates", 1) or 1)
    min_interval = max(1, min(min_interval, max_interval))
    next_due = min_interval
    while next_due <= max_update:
        due_updates.add(int(next_due))
        interval = max(1, _checkpoint_interval_for_update(args, next_due))
        next_due += interval
    return due_updates


def _historical_eval_updates(args) -> tuple[int, ...]:
    """Parse the user-configured, explicitly pinned eval milestones."""
    raw = str(getattr(args, "historical_eval_updates", "") or "").strip()
    if not raw:
        return ()
    tokens = [item.strip() for item in raw.split(",")]
    if not all(tokens):
        raise SystemExit(
            "--historical-eval-updates must be a comma-separated list of positive updates."
        )
    try:
        updates = tuple(int(item) for item in tokens)
    except ValueError as exc:
        raise SystemExit(
            "--historical-eval-updates must contain integer update indices."
        ) from exc
    if any(update <= 0 for update in updates):
        raise SystemExit("--historical-eval-updates entries must be positive.")
    if tuple(sorted(set(updates))) != updates:
        raise SystemExit(
            "--historical-eval-updates must be strictly increasing with no duplicates."
        )
    if updates[-1] > int(getattr(args, "num_updates", 0) or 0):
        raise SystemExit(
            "--historical-eval-updates cannot include an update after --num-updates."
        )
    return updates


def _checkpoint_trainer_config_from_args(
    trainer_config: TrainerConfig,
    args,
    *,
    spec: ActorCriticSpec | None = None,
) -> dict[str, Any]:
    """Persist runtime-relevant train settings without widening TrainerConfig."""
    config = {
        str(key): to_builtin(value) for key, value in asdict(trainer_config).items()
    }
    selector_fields = {
        "eval_deploy_every_updates": int(getattr(args, "eval_deploy_every_updates", 0)),
        "eval_deploy_batches": int(getattr(args, "eval_deploy_batches", 20)),
        "eval_deploy_horizon": int(getattr(args, "eval_deploy_horizon", 128)),
        "eval_deploy_seed": int(getattr(args, "eval_deploy_seed", 2000000)),
        "historical_eval_updates": str(
            getattr(args, "historical_eval_updates", "") or ""
        ),
        "historical_eval_episodes": int(getattr(args, "historical_eval_episodes", 200)),
        "historical_eval_horizon": int(getattr(args, "historical_eval_horizon", 1024)),
        "historical_eval_seed": int(getattr(args, "historical_eval_seed", 3000000)),
        "enable_intent_learning": bool(getattr(args, "enable_intent_learning", False)),
        "enable_defense_intent_learning": bool(
            getattr(args, "enable_defense_intent_learning", False)
        ),
        "num_intents": int(
            getattr(args, "num_intents", getattr(spec, "num_intents", 8))
        ),
        "intent_commitment_steps": int(getattr(args, "intent_commitment_steps", 4)),
        "intent_null_prob": float(getattr(args, "intent_null_prob", 0.2)),
        "defense_intent_null_prob": float(
            getattr(args, "defense_intent_null_prob", 1.0)
        ),
        "intent_visible_to_defense_prob": float(
            getattr(args, "intent_visible_to_defense_prob", 0.0)
        ),
        "intent_selector_enabled": bool(
            getattr(
                args,
                "intent_selector_enabled",
                getattr(spec, "intent_selector_enabled", False),
            )
        ),
        "intent_selector_mode": str(
            getattr(args, "intent_selector_mode", "integrated")
        ),
        "intent_selector_hidden_dim": int(
            getattr(
                args,
                "intent_selector_hidden_dim",
                getattr(spec, "intent_selector_hidden_dim", 64),
            )
        ),
        "intent_selector_learning_rate": _selector_learning_rate_for_args(
            args,
            trainer_config,
        ),
        "intent_selector_alpha_start": float(
            getattr(args, "intent_selector_alpha_start", 0.0)
        ),
        "intent_selector_alpha_end": float(
            getattr(args, "intent_selector_alpha_end", 1.0)
        ),
        "intent_selector_alpha_warmup_updates": int(
            getattr(args, "intent_selector_alpha_warmup_updates", 0)
        ),
        "intent_selector_alpha_ramp_updates": int(
            getattr(args, "intent_selector_alpha_ramp_updates", 1)
        ),
        "intent_selector_eps_start": float(
            getattr(args, "intent_selector_eps_start", 0.0)
        ),
        "intent_selector_eps_end": float(getattr(args, "intent_selector_eps_end", 0.0)),
        "intent_selector_eps_warmup_updates": int(
            getattr(args, "intent_selector_eps_warmup_updates", 0)
        ),
        "intent_selector_eps_ramp_updates": int(
            getattr(args, "intent_selector_eps_ramp_updates", 1)
        ),
        "intent_selector_entropy_coef": float(
            getattr(args, "intent_selector_entropy_coef", 0.01)
        ),
        "intent_selector_usage_reg_coef": float(
            getattr(args, "intent_selector_usage_reg_coef", 0.01)
        ),
        "intent_selector_value_coef": float(
            getattr(args, "intent_selector_value_coef", 0.5)
        ),
        "intent_selector_train_every_rollouts": int(
            getattr(args, "intent_selector_train_every_rollouts", 1)
        ),
        "intent_selector_max_samples_per_update": int(
            getattr(args, "intent_selector_max_samples_per_update", 0)
        ),
        "intent_selector_multiselect_enabled": bool(
            getattr(args, "intent_selector_multiselect_enabled", False)
        ),
        "intent_selector_min_play_steps": int(
            getattr(args, "intent_selector_min_play_steps", 3)
        ),
        "opponent_deterministic_episode_prob": float(
            getattr(args, "opponent_deterministic_episode_prob", 0.0)
        ),
        "opponent_deterministic_episode_prob_start": (
            None
            if getattr(args, "opponent_deterministic_episode_prob_start", None) is None
            else float(getattr(args, "opponent_deterministic_episode_prob_start"))
        ),
        "opponent_deterministic_episode_prob_end": (
            None
            if getattr(args, "opponent_deterministic_episode_prob_end", None) is None
            else float(getattr(args, "opponent_deterministic_episode_prob_end"))
        ),
        "opponent_deterministic_episode_prob_warmup_updates": int(
            getattr(args, "opponent_deterministic_episode_prob_warmup_updates", 0)
        ),
        "opponent_deterministic_episode_prob_ramp_updates": int(
            getattr(args, "opponent_deterministic_episode_prob_ramp_updates", 1)
        ),
        "enable_rebounds": bool(getattr(args, "enable_rebounds", False)),
        "rebound_table_model_dir": str(
            getattr(args, "rebound_table_model_dir", "") or ""
        ),
        "rebound_target_temperature": float(
            getattr(args, "rebound_target_temperature", 1.0)
        ),
        "rebound_target_uniform_mix": float(
            getattr(args, "rebound_target_uniform_mix", 0.0)
        ),
        "rebound_winner_distance_weight": float(
            getattr(args, "rebound_winner_distance_weight", 1.0)
        ),
        "rebound_basket_position_weight": float(
            getattr(args, "rebound_basket_position_weight", 0.0)
        ),
        "rebound_winner_temperature": float(
            getattr(args, "rebound_winner_temperature", 1.0)
        ),
        "rebound_skill_std": float(getattr(args, "rebound_skill_std", 0.0)),
        "rebound_skill_sampling_mode": str(
            getattr(args, "rebound_skill_sampling_mode", "gaussian") or "gaussian"
        ),
        "rebound_skill_high": float(getattr(args, "rebound_skill_high", 1.0)),
        "rebound_skill_low": float(getattr(args, "rebound_skill_low", -0.25)),
        "rebound_skill_weight": float(getattr(args, "rebound_skill_weight", 0.0)),
        "rebound_contest_mode": str(
            getattr(args, "rebound_contest_mode", "global_contest") or "global_contest"
        ),
        "rebound_contest_radius": int(getattr(args, "rebound_contest_radius", 1)),
        "rebound_obs_top_n_targets": int(getattr(args, "rebound_obs_top_n_targets", 0)),
        "rebound_win_prob_features": bool(
            getattr(args, "rebound_win_prob_features", False)
        ),
        "rebound_target_observation_features": bool(
            getattr(args, "rebound_target_observation_features", True)
        ),
        "offensive_rebound_shot_clock_reset": int(
            getattr(args, "offensive_rebound_shot_clock_reset", 14)
        ),
        "rebound_terminal_reward_mode": str(
            getattr(args, "rebound_terminal_reward_mode", "actual_points")
            or "actual_points"
        ),
        "enable_rebound_reward_redistribution": bool(
            getattr(args, "enable_rebound_reward_redistribution", False)
        ),
        "offensive_rebound_reward_advance": float(
            getattr(args, "offensive_rebound_reward_advance", 0.4)
        ),
        "rebound_reward_once_per_possession": bool(
            getattr(args, "rebound_reward_once_per_possession", True)
        ),
        "rebound_counterfactual_positioning_enabled": bool(
            getattr(args, "rebound_counterfactual_positioning_enabled", False)
        ),
    }
    config.update({key: to_builtin(value) for key, value in selector_fields.items()})
    return config


def _policy_model_type(args) -> str:
    return str(getattr(args, "policy_model", "mlp")).lower()


def _build_policy_spec(
    args, static, flat_obs_np: np.ndarray, action_masks_np: np.ndarray
) -> ActorCriticSpec:
    model_type = _policy_model_type(args)
    rebound_win_prob_features = bool(getattr(args, "rebound_win_prob_features", False))
    rebound_target_observation_features = bool(
        getattr(args, "rebound_target_observation_features", True)
    )
    multi_possession_features = bool(getattr(args, "enable_multi_possession", False))
    token_dim, global_dim = token_observation_dims(
        rebound_win_prob_features,
        rebound_target_observation_features,
        multi_possession_features,
    )
    return build_actor_critic_spec(
        flat_obs_np,
        action_masks_np,
        hidden_dims=args.policy_hidden_dims,
        model_type=model_type,
        token_player_count=(
            int(static.role_encoding.shape[0]) if model_type == "attention" else 0
        ),
        token_dim=token_dim if model_type == "attention" else 0,
        global_dim=global_dim if model_type == "attention" else 0,
        attention_embed_dim=int(getattr(args, "attention_embed_dim", 64)),
        attention_num_heads=int(getattr(args, "attention_num_heads", 4)),
        attention_token_mlp_dim=int(getattr(args, "attention_token_mlp_dim", 64)),
        attention_num_cls_tokens=int(getattr(args, "attention_cls_tokens", 2)),
        attention_pi_head_hidden_dims=tuple(
            int(v) for v in getattr(args, "attention_pi_head_hidden_dims", [])
        ),
        attention_vf_head_hidden_dims=tuple(
            int(v) for v in getattr(args, "attention_vf_head_hidden_dims", [])
        ),
        attention_head_activation=str(
            getattr(args, "attention_head_activation", "tanh")
        ),
        action_head_mode=str(getattr(args, "action_head_mode", "flat")),
        pass_action_start=int(PASS_ACTION_START),
        pass_action_end=int(PASS_ACTION_END),
        intent_embedding_enabled=bool(getattr(args, "intent_embedding_enabled", False)),
        intent_embedding_dim=int(getattr(args, "intent_embedding_dim", 16)),
        intent_conditioning_scale=float(
            getattr(args, "intent_conditioning_scale", 1.0)
        ),
        num_intents=int(getattr(args, "num_intents", 8)),
        intent_selector_enabled=bool(getattr(args, "intent_selector_enabled", False)),
        intent_selector_hidden_dim=int(getattr(args, "intent_selector_hidden_dim", 64)),
        rebound_win_prob_features=rebound_win_prob_features,
        rebound_target_observation_features=rebound_target_observation_features,
        rebound_critic_enabled=bool(getattr(args, "rebound_critic_enabled", False)),
        multi_possession_features=multi_possession_features,
        observation_schema_version=(
            MULTI_POSSESSION_SCHEMA_VERSION if multi_possession_features else 1
        ),
    )


def _normalize_policy_spec_dict(raw_spec: dict[str, Any]) -> dict[str, Any]:
    return asdict(ActorCriticSpec(**dict(raw_spec)))


def _uses_grouped_opponent_sampling(args) -> bool:
    return bool(getattr(args, "grouped_opponent_sampling", False))


def _args_for_training_role(args, role: str):
    role_args = copy(args)
    role_args.training_team = str(role)
    return role_args


def _remaining_eval_count(
    *, start_update: int, num_updates: int, eval_every_updates: int
) -> int:
    if int(eval_every_updates) <= 0 or int(start_update) >= int(num_updates):
        return 0
    remaining = 0
    for update_idx in range(int(start_update) + 1, int(num_updates) + 1):
        if update_idx == int(num_updates) or update_idx % int(eval_every_updates) == 0:
            remaining += 1
    return remaining


def _restore_like_template(restored, template):
    if isinstance(template, dict):
        if not isinstance(restored, dict):
            return restored
        return {
            key: _restore_like_template(restored[key], value)
            for key, value in template.items()
        }
    if isinstance(template, tuple) and hasattr(template, "_fields"):
        if isinstance(restored, dict):
            return type(template)(
                **{
                    field: _restore_like_template(
                        restored.get(field, getattr(template, field)),
                        getattr(template, field),
                    )
                    for field in template._fields
                }
            )
        if isinstance(restored, (tuple, list)):
            restored_by_field = {
                field: item
                for item, field in zip(restored, template._fields, strict=False)
            }
            return type(template)(
                **{
                    field: _restore_like_template(
                        restored_by_field.get(field, getattr(template, field)),
                        getattr(template, field),
                    )
                    for field in template._fields
                }
            )
        return restored
    if isinstance(template, tuple):
        if isinstance(restored, (tuple, list)):
            return type(template)(
                _restore_like_template(item, tmpl)
                for item, tmpl in zip(restored, template, strict=False)
            )
        return restored
    if isinstance(template, list):
        if isinstance(restored, list):
            return [
                _restore_like_template(item, tmpl)
                for item, tmpl in zip(restored, template, strict=False)
            ]
        return restored
    return restored


def _validate_resume_checkpoint_payload(
    payload: dict[str, Any],
    *,
    trainer_config: TrainerConfig,
    spec: ActorCriticSpec,
    args,
) -> None:
    expected_trainer_config = asdict(trainer_config)
    actual_trainer_config = dict(payload.get("trainer_config", {}))
    compatible_keys = [
        "kernel_batch_size",
        "rollout_horizon",
        "gamma",
        "gae_lambda",
        "ppo_clip_range",
        "value_coef",
        "entropy_coef",
        "learning_rate",
        "policy_update_epochs",
    ]
    for key in compatible_keys:
        if actual_trainer_config.get(key) != expected_trainer_config[key]:
            raise SystemExit(f"Resume checkpoint trainer_config mismatch for {key!r}.")

    expected_policy_spec = asdict(spec)
    if (
        _normalize_policy_spec_dict(payload.get("policy_spec", {}))
        != expected_policy_spec
    ):
        raise SystemExit(
            "Resume checkpoint policy_spec does not match the current JAX run."
        )

    expected_frozen = {
        key: to_builtin(getattr(args, key)) for key in TRAIN_FROZEN_VALUES
    }
    if dict(payload.get("frozen_config", {})) != expected_frozen:
        raise SystemExit(
            "Resume checkpoint frozen_config does not match the current JAX run."
        )
    if "env_config" in payload:
        expected_env_config = _jax_env_config_from_args(args)
        actual_env_config = _compatible_env_config_for_resume(
            dict(payload.get("env_config", {})),
            expected_env_config,
        )
        if actual_env_config != expected_env_config:
            raise SystemExit(
                "Resume checkpoint env_config does not match the current JAX run."
            )
    if bool(getattr(args, "intent_diversity_enabled", False)):
        if "intent_discriminator_state" not in payload:
            raise SystemExit(
                "Resume checkpoint does not contain JAX intent discriminator state."
            )


def _save_training_checkpoint(
    *,
    checkpoint_dir: str | None,
    update_index: int,
    trainer_config: TrainerConfig,
    spec: ActorCriticSpec,
    args,
    params,
    opt_state,
    current_state,
    eval_initial_state,
    base_key,
    eval_trajectories: list[dict[str, Any]],
    last_metrics: dict[str, Any] | None,
    opponent_info: dict[str, Any] | None,
    opponent_pool_state: dict[str, Any] | None = None,
    selector_opt_state=None,
    intent_discriminator_state: dict[str, Any] | None = None,
    play_name_metadata: dict[str, Any] | None = None,
) -> tuple[str | None, str]:
    payload = build_checkpoint_payload(
        update_index=int(update_index),
        trainer_config=_checkpoint_trainer_config_from_args(
            trainer_config,
            args,
            spec=spec,
        ),
        policy_spec=asdict(spec),
        frozen_config={
            key: to_builtin(getattr(args, key)) for key in TRAIN_FROZEN_VALUES
        },
        env_config=_jax_env_config_from_args(args),
        params=params,
        opt_state=opt_state,
        selector_opt_state=selector_opt_state,
        current_state=current_state,
        eval_initial_state=eval_initial_state,
        base_key=base_key,
        eval_trajectories=eval_trajectories,
        last_metrics=last_metrics,
        opponent_info=opponent_info,
        opponent_pool_state=opponent_pool_state,
        intent_discriminator_state=intent_discriminator_state,
        play_name_metadata=play_name_metadata,
    )
    if checkpoint_dir is None:
        raise ValueError(
            "checkpoint_dir must not be None when saving a persistent local checkpoint."
        )
    numbered_path, latest_path = build_checkpoint_paths(
        checkpoint_dir,
        update_index=int(update_index),
    )
    save_checkpoint(numbered_path, payload)
    save_checkpoint(latest_path, payload)
    return str(latest_path), str(numbered_path)


def _maybe_start_mlflow_run(args, *, mode: str):
    if not bool(getattr(args, "log_mlflow", False)):
        return None, nullcontext()

    import mlflow

    setup_mlflow(verbose=False)
    resume_run_id = str(getattr(args, "mlflow_resume_run_id", "") or "").strip()
    if resume_run_id:
        return mlflow, mlflow.start_run(run_id=resume_run_id)

    mlflow.set_experiment(str(args.mlflow_experiment_name))
    run_name = args.mlflow_run_name
    if not run_name:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"jax-train-{mode}-{timestamp}"
    context = mlflow.start_run(run_name=run_name)
    return mlflow, context


def _training_play_name_seed_key(args, mlflow, checkpoint_dir: str | None) -> str:
    if mlflow is not None:
        active_run = mlflow.active_run()
        run_id = str(
            getattr(getattr(active_run, "info", None), "run_id", "") or ""
        ).strip()
        if run_id:
            return run_id
    for value in (
        getattr(args, "mlflow_run_name", None),
        checkpoint_dir,
    ):
        text = str(value or "").strip()
        if text:
            return text
    return "jax-train"


def _build_training_play_name_metadata(
    *,
    args,
    mlflow,
    checkpoint_dir: str | None,
) -> dict[str, Any]:
    seed_key = _training_play_name_seed_key(args, mlflow, checkpoint_dir)
    payload = build_play_name_artifact_payload(
        seed_key,
        int(getattr(args, "num_intents", 0) or 0),
    )
    payload["model_codename"] = build_model_codename(seed_key)
    payload["backend"] = "jax"
    return payload


def _log_mlflow_play_name_metadata(mlflow, play_name_metadata: dict[str, Any]) -> None:
    try:
        mlflow.log_param("jax/play_name_pool_version", int(PLAY_NAME_POOL_VERSION))
        model_codename = str(play_name_metadata.get("model_codename", "")).strip()
        if model_codename:
            mlflow.set_tag("model_codename", model_codename)
        with TemporaryDirectory(prefix="basketworld_jax_play_names_") as tmpdir:
            path = Path(tmpdir) / "play_names.json"
            path.write_text(
                json.dumps(play_name_metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            mlflow.log_artifact(str(path), artifact_path="metadata")
    except Exception as exc:
        print(f"[play_names] Failed to log JAX play name mapping artifact: {exc}")


def _log_mlflow_start_template_library(mlflow, args) -> None:
    if not bool(getattr(args, "start_template_enabled", False)):
        return
    source_path = getattr(args, "start_template_library", None)
    if not source_path:
        return
    try:
        from basketworld.utils.start_templates import load_start_template_library

        library = load_start_template_library(
            source_path,
            players_per_side=int(getattr(args, "players", 3) or 3),
        )
        with TemporaryDirectory(prefix="basketworld_jax_start_templates_") as tmpdir:
            artifact_name = "start_template_library.json"
            artifact_path = Path(tmpdir) / artifact_name
            artifact_path.write_text(
                json.dumps(library, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            mlflow.log_artifact(str(artifact_path), artifact_path="metadata")
        mlflow.log_param(
            "start_template_library_artifact_path",
            f"metadata/{artifact_name}",
        )
        mlflow.log_param(
            "start_template_library_template_count",
            int(len(library.get("templates", []) or [])),
        )
    except Exception as exc:
        print(f"[start_templates] Failed to log JAX template library artifact: {exc}")


def _is_jax_checkpoint_artifact(path: str) -> bool:
    name = Path(path).name
    return (
        name == "latest"
        or name == "phase_a_latest"
        or name.startswith("update_")
        or name.startswith("phase_a_update_")
    )


def _checkpoint_artifact_sort_key(path: str) -> tuple[int, int, str]:
    name = Path(path).name
    if name == "latest" or name == "phase_a_latest":
        return (1, 10**12, path)
    for prefix in ("update_", "phase_a_update_"):
        if name.startswith(prefix):
            try:
                return (1, int(name.removeprefix(prefix)), path)
            except ValueError:
                break
    return (2, 0, path)


def _resolve_mlflow_checkpoint_artifact(
    client, run_id: str, artifact_hint: str | None
) -> str:
    artifacts = client.list_artifacts(run_id, "models")
    choices = [
        item.path for item in artifacts if _is_jax_checkpoint_artifact(str(item.path))
    ]
    if not choices:
        raise SystemExit(
            f"No JAX checkpoint artifacts found under models/ for MLflow run {run_id!r}."
        )

    hint = str(artifact_hint or "").strip()
    if hint:
        for choice in choices:
            if choice == hint or choice.endswith(hint):
                return choice
        raise SystemExit(
            f"JAX checkpoint artifact {hint!r} was not found in MLflow run {run_id!r}."
        )

    tags = dict(
        getattr(getattr(client.get_run(run_id), "data", None), "tags", {}) or {}
    )
    tagged = str(tags.get("jax_latest_checkpoint_artifact", "")).strip()
    if tagged and tagged in choices:
        return tagged

    return sorted(choices, key=_checkpoint_artifact_sort_key)[-1]


def _continue_artifact_hint_from_args(args) -> str:
    hint = str(getattr(args, "continue_artifact", "") or "").strip()
    if hint:
        return hint
    resume_checkpoint = str(getattr(args, "resume_checkpoint", "") or "").strip()
    if resume_checkpoint:
        name = Path(resume_checkpoint).name
        if _is_jax_checkpoint_artifact(name):
            return f"models/{name}"
    return ""


def _numbered_mlflow_checkpoint_artifacts(client, run_id: str) -> list[str]:
    artifacts = client.list_artifacts(run_id, "models")
    choices = []
    for item in artifacts:
        path = str(item.path)
        name = Path(path).name
        if name.startswith("update_") or name.startswith("phase_a_update_"):
            choices.append(path)
    return sorted(choices, key=_checkpoint_artifact_sort_key)


def _download_mlflow_checkpoint_artifact(
    client,
    *,
    run_id: str,
    artifact_path: str,
    cache_dir: str,
) -> Path:
    cache_root = Path(cache_dir or "artifacts/mlflow_checkpoints").expanduser()
    run_cache = cache_root / str(run_id)
    run_cache.mkdir(parents=True, exist_ok=True)
    return Path(client.download_artifacts(run_id, artifact_path, str(run_cache)))


def _prepare_continuation_checkpoint(args) -> dict[str, Any] | None:
    run_id = str(getattr(args, "continue_run_id", "") or "").strip()
    if not run_id:
        return None

    import mlflow

    setup_mlflow(verbose=False)
    client = mlflow.tracking.MlflowClient()
    artifact_path = _resolve_mlflow_checkpoint_artifact(
        client,
        run_id,
        _continue_artifact_hint_from_args(args),
    )
    local_path = ""
    if not str(getattr(args, "resume_checkpoint", "") or "").strip():
        local_path = str(
            _download_mlflow_checkpoint_artifact(
                client,
                run_id=run_id,
                artifact_path=artifact_path,
                cache_dir=str(getattr(args, "continue_cache_dir", "") or ""),
            )
        )
    return {
        "run_id": run_id,
        "artifact_path": artifact_path,
        "local_path": local_path,
    }


def _load_continuation_opponent_candidates(
    args,
    *,
    jax,
    spec: ActorCriticSpec,
    resume_artifact_path: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    run_id = str(getattr(args, "continue_run_id", "") or "").strip()
    if not run_id:
        return [], None

    seed_count = int(getattr(args, "continue_opponent_pool_size", -1))
    if seed_count < 0:
        seed_count = int(getattr(args, "opponent_pool_size", 10))
    if seed_count <= 0:
        return [], {
            "run_id": run_id,
            "seeded_count": 0,
            "available_count": 0,
            "artifacts": [],
        }

    import mlflow

    setup_mlflow(verbose=False)
    client = mlflow.tracking.MlflowClient()
    boundary_artifact = _resolve_mlflow_checkpoint_artifact(
        client,
        run_id,
        str(resume_artifact_path or "").strip()
        or _continue_artifact_hint_from_args(args),
    )
    choices = _numbered_mlflow_checkpoint_artifacts(client, run_id)
    boundary_key = _checkpoint_artifact_sort_key(boundary_artifact)
    eligible = [
        path for path in choices if _checkpoint_artifact_sort_key(path) <= boundary_key
    ]
    if not eligible and boundary_artifact in choices:
        eligible = [boundary_artifact]
    selected = eligible[-seed_count:]

    candidates: list[dict[str, Any]] = []
    expected_spec = asdict(spec)
    for artifact_path in selected:
        local_path = _download_mlflow_checkpoint_artifact(
            client,
            run_id=run_id,
            artifact_path=artifact_path,
            cache_dir=str(getattr(args, "continue_cache_dir", "") or ""),
        )
        payload = load_checkpoint(local_path)
        if _normalize_policy_spec_dict(payload.get("policy_spec", {})) != expected_spec:
            raise SystemExit(
                f"Continuation opponent checkpoint {artifact_path!r} policy_spec does not match the current run."
            )
        candidates.append(
            {
                "params": jax.device_put(payload["params"]),
                "info": {
                    "source": "mlflow_continue",
                    "run_id": run_id,
                    "artifact_path": artifact_path,
                    "checkpoint_path": str(local_path),
                    "update_index": int(payload.get("update_index", 0)),
                    "candidate_kind": "continued_pool",
                },
            }
        )

    return candidates, {
        "run_id": run_id,
        "boundary_artifact_path": boundary_artifact,
        "seeded_count": int(len(candidates)),
        "available_count": int(len(eligible)),
        "requested_count": int(seed_count),
        "artifacts": list(selected),
    }


def _load_frozen_opponent_payload(
    args,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    checkpoint_path = str(getattr(args, "frozen_opponent_checkpoint", "") or "").strip()
    run_id = str(getattr(args, "frozen_opponent_run_id", "") or "").strip()
    artifact_hint = str(getattr(args, "frozen_opponent_artifact", "") or "").strip()

    if checkpoint_path and run_id:
        raise SystemExit(
            "Use either --frozen-opponent-checkpoint or --frozen-opponent-run-id, not both."
        )
    if artifact_hint and not run_id:
        raise SystemExit(
            "--frozen-opponent-artifact requires --frozen-opponent-run-id."
        )
    if checkpoint_path:
        payload = load_checkpoint(checkpoint_path)
        return payload, {
            "source": "checkpoint",
            "checkpoint_path": str(Path(checkpoint_path)),
            "update_index": int(payload.get("update_index", 0)),
        }
    if not run_id:
        return None, None

    import mlflow

    setup_mlflow(verbose=False)
    client = mlflow.tracking.MlflowClient()
    artifact_path = _resolve_mlflow_checkpoint_artifact(client, run_id, artifact_hint)
    with TemporaryDirectory(prefix="basketworld_jax_opponent_") as tmpdir:
        local_path = client.download_artifacts(run_id, artifact_path, tmpdir)
        payload = load_checkpoint(local_path)
    return payload, {
        "source": "mlflow",
        "run_id": run_id,
        "artifact_path": artifact_path,
        "update_index": int(payload.get("update_index", 0)),
    }


def _add_opponent_candidate(
    candidates: list[dict[str, Any]],
    *,
    params,
    info: dict[str, Any],
) -> None:
    candidates.append(
        {
            "params": params,
            "info": dict(info),
        }
    )


def _sample_geometric_candidate_index(
    count: int, beta: float, rng: np.random.Generator
) -> int:
    if count <= 1:
        return 0
    beta = float(beta)
    if beta >= 1.0:
        return count - 1
    beta = max(beta, 0.0)
    weights = np.asarray(
        [(1.0 - beta) * (beta ** (count - idx)) for idx in range(1, count + 1)],
        dtype=np.float64,
    )
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        return count - 1
    probs = weights / total
    return int(rng.choice(np.arange(count), p=probs))


def _sample_opponent_candidate(
    candidates: list[dict[str, Any]],
    *,
    pool_size: int,
    beta: float,
    exploration: float,
    rng: np.random.Generator,
) -> dict[str, Any] | None:
    idx = _sample_opponent_candidate_index(
        candidates,
        pool_size=pool_size,
        beta=beta,
        exploration=exploration,
        rng=rng,
    )
    if idx is None:
        return None
    return candidates[idx]


def _sample_opponent_candidate_index(
    candidates: list[dict[str, Any]],
    *,
    pool_size: int,
    beta: float,
    exploration: float,
    rng: np.random.Generator,
) -> int | None:
    if not candidates:
        return None
    recent_count = max(1, min(int(pool_size), len(candidates)))
    recent_start = len(candidates) - recent_count
    exploration = float(np.clip(float(exploration), 0.0, 1.0))
    if recent_start > 0 and float(rng.random()) < exploration:
        return int(rng.integers(0, len(candidates)))
    chosen_idx = _sample_geometric_candidate_index(recent_count, float(beta), rng)
    return int(recent_start + chosen_idx)


def _select_opponent_from_pool(
    candidates: list[dict[str, Any]],
    *,
    args,
    rng: np.random.Generator,
):
    chosen = _sample_opponent_candidate(
        candidates,
        pool_size=int(getattr(args, "opponent_pool_size", 10)),
        beta=float(getattr(args, "opponent_pool_beta", 0.7)),
        exploration=float(getattr(args, "opponent_pool_exploration", 0.0)),
        rng=rng,
    )
    if chosen is None:
        return None, None
    return chosen["params"], dict(chosen["info"])


def _opponent_candidate_probabilities(
    candidates: list[dict[str, Any]],
    *,
    args,
) -> np.ndarray:
    """Return the configured sampling distribution over stable candidate ids."""
    count = len(candidates)
    if count <= 0:
        return np.zeros((0,), dtype=np.float32)
    recent_count = max(1, min(int(getattr(args, "opponent_pool_size", 10)), count))
    recent_start = count - recent_count
    beta = float(np.clip(float(getattr(args, "opponent_pool_beta", 0.7)), 0.0, 1.0))
    if recent_count == 1 or beta >= 1.0:
        recent_probs = np.zeros((recent_count,), dtype=np.float64)
        recent_probs[-1] = 1.0
    else:
        weights = np.asarray(
            [
                (1.0 - beta) * (beta ** (recent_count - idx))
                for idx in range(1, recent_count + 1)
            ],
            dtype=np.float64,
        )
        recent_probs = weights / max(float(weights.sum()), 1.0e-12)
    probs = np.zeros((count,), dtype=np.float64)
    probs[recent_start:] = recent_probs
    exploration = float(
        np.clip(float(getattr(args, "opponent_pool_exploration", 0.0)), 0.0, 1.0)
    )
    if exploration > 0.0 and recent_start > 0:
        probs = ((1.0 - exploration) * probs) + (
            exploration * np.full((count,), 1.0 / float(count), dtype=np.float64)
        )
    return (probs / max(float(probs.sum()), 1.0e-12)).astype(np.float32)


def _stack_opponent_candidate_params(candidates, *, jax, jnp):
    if not candidates:
        return None
    return jax.tree_util.tree_map(
        lambda *leaves: jnp.stack(leaves, axis=0),
        *[candidate["params"] for candidate in candidates],
    )


def _unstack_opponent_candidate_params(stacked_params, count: int, *, jax):
    if stacked_params is None or int(count) <= 0:
        return []
    return [
        jax.tree_util.tree_map(lambda leaf, idx=idx: leaf[idx], stacked_params)
        for idx in range(int(count))
    ]


def _build_opponent_pool_checkpoint_state(
    candidates,
    assignments,
    deterministic_modes,
    *,
    opponent_rng,
    enabled: bool,
    jax,
    jnp,
):
    return {
        "enabled": bool(enabled),
        "candidate_infos": [dict(candidate["info"]) for candidate in candidates],
        "candidate_params": _stack_opponent_candidate_params(
            candidates,
            jax=jax,
            jnp=jnp,
        ),
        "assignments": dict(assignments),
        "deterministic_modes": dict(deterministic_modes),
        "rng_state": dict(opponent_rng.bit_generator.state),
    }


def _effective_opponent_group_count(args, *, candidate_count: int) -> int:
    requested = max(1, int(getattr(args, "opponent_group_count", 8)))
    batch_size = max(1, int(getattr(args, "kernel_batch_size")))
    max_groups = max(1, min(requested, int(candidate_count), batch_size))
    for group_count in range(max_groups, 0, -1):
        if batch_size % group_count == 0:
            return int(group_count)
    return 1


def _stack_opponent_params(params_by_group: list[Any], *, jax, jnp):
    return jax.tree_util.tree_map(
        lambda *leaves: jnp.stack(leaves, axis=0),
        *params_by_group,
    )


def _select_grouped_opponents_from_pool(
    candidates: list[dict[str, Any]],
    *,
    args,
    rng: np.random.Generator,
    jax,
    jnp,
):
    if not candidates:
        return None, None
    group_count = _effective_opponent_group_count(args, candidate_count=len(candidates))
    chosen_indices = [
        _sample_opponent_candidate_index(
            candidates,
            pool_size=int(getattr(args, "opponent_pool_size", 10)),
            beta=float(getattr(args, "opponent_pool_beta", 0.7)),
            exploration=float(getattr(args, "opponent_pool_exploration", 0.0)),
            rng=rng,
        )
        for _ in range(group_count)
    ]
    chosen_indices = [int(idx) for idx in chosen_indices if idx is not None]
    if not chosen_indices:
        return None, None
    chosen_candidates = [candidates[idx] for idx in chosen_indices]
    grouped_params = _stack_opponent_params(
        [candidate["params"] for candidate in chosen_candidates],
        jax=jax,
        jnp=jnp,
    )
    groups = []
    update_indices = []
    for group_idx, candidate in zip(chosen_indices, chosen_candidates, strict=True):
        info = dict(candidate["info"])
        update_index = int(info.get("update_index", 0))
        update_indices.append(update_index)
        groups.append(
            {
                "candidate_index": int(group_idx),
                "source": str(info.get("source", "unknown")),
                "candidate_kind": str(info.get("candidate_kind", "unknown")),
                "update_index": update_index,
            }
        )
    return grouped_params, {
        "source": "grouped_pool",
        "group_count": int(len(chosen_candidates)),
        "batch_group_size": int(
            int(getattr(args, "kernel_batch_size")) // len(chosen_candidates)
        ),
        "candidate_count": int(len(candidates)),
        "unique_update_count": int(len(set(update_indices))),
        "latest_update_index": int(max(update_indices)) if update_indices else 0,
        "groups": groups,
    }


def _log_mlflow_params(
    mlflow, args, trainer_config: TrainerConfig, spec: ActorCriticSpec
) -> None:
    if hasattr(mlflow, "log_dict"):
        mlflow.log_dict(
            {key: to_builtin(value) for key, value in vars(args).items()},
            "metadata/resolved_training_config.json",
        )
    params = {
        "jax/script": "basketworld_jax/train/main.py",
        "jax/mode": "train_loop" if bool(args.run_train_loop) else "scaffold",
        "jax/kernel_batch_size": int(args.kernel_batch_size),
        "jax/rollout_horizon": int(args.rollout_horizon),
        "jax/single_episode_rollouts": bool(
            getattr(args, "single_episode_rollouts", False)
        ),
        "jax/ppo_completed_episodes_only": bool(
            getattr(args, "ppo_completed_episodes_only", False)
        ),
        "jax/num_updates": int(args.num_updates),
        "jax/policy_update_epochs": int(args.policy_update_epochs),
        "jax/ppo_minibatches": int(args.ppo_minibatches),
        "jax/log_every_updates": int(args.log_every_updates),
        "jax/eval_every_updates": int(args.eval_every_updates),
        "jax/eval_horizon": int(args.eval_horizon),
        "jax/eval_deploy_every_updates": int(args.eval_deploy_every_updates),
        "jax/historical_eval_updates": str(
            getattr(args, "historical_eval_updates", "") or ""
        ),
        "jax/historical_eval_episodes": int(
            getattr(args, "historical_eval_episodes", 200)
        ),
        "jax/historical_eval_horizon": int(
            getattr(args, "historical_eval_horizon", 1024)
        ),
        "jax/historical_eval_seed": int(getattr(args, "historical_eval_seed", 3000000)),
        "jax/eval_deploy_batches": int(args.eval_deploy_batches),
        "jax/eval_deploy_horizon": int(args.eval_deploy_horizon),
        "jax/eval_deploy_seed": int(args.eval_deploy_seed),
        "jax/eval_deploy_episode_count": int(args.eval_deploy_batches)
        * int(args.kernel_batch_size),
        "jax/mlflow_metric_profile": str(
            getattr(args, "mlflow_metric_profile", "core")
        ),
        "jax/learning_rate": float(trainer_config.learning_rate),
        "jax/gamma": float(trainer_config.gamma),
        "jax/gae_lambda": float(trainer_config.gae_lambda),
        "jax/ppo_clip_range": float(trainer_config.ppo_clip_range),
        "jax/value_coef": float(trainer_config.value_coef),
        "jax/entropy_coef": float(trainer_config.entropy_coef),
        "jax/ent_coef_start": (
            ""
            if getattr(args, "ent_coef_start", None) is None
            else float(getattr(args, "ent_coef_start"))
        ),
        "jax/ent_coef_end": (
            ""
            if getattr(args, "ent_coef_end", None) is None
            else float(getattr(args, "ent_coef_end"))
        ),
        "jax/ent_schedule": str(getattr(args, "ent_schedule", "linear")),
        "jax/entropy_decay_updates": (
            -1
            if getattr(args, "entropy_decay_updates", None) is None
            else int(getattr(args, "entropy_decay_updates"))
        ),
        "jax/policy_model": str(spec.model_type),
        "jax/action_head_mode": str(spec.action_head_mode),
        "jax/policy_hidden_dims": ",".join(str(v) for v in spec.hidden_dims),
        "jax/flat_obs_dim": int(spec.flat_obs_dim),
        "jax/training_player_count": int(spec.training_player_count),
        "jax/action_dim_per_player": int(spec.action_dim_per_player),
        "jax/token_player_count": int(spec.token_player_count),
        "jax/token_dim": int(spec.token_dim),
        "jax/global_dim": int(spec.global_dim),
        "jax/attention_embed_dim": int(spec.attention_embed_dim),
        "jax/attention_num_heads": int(spec.attention_num_heads),
        "jax/attention_token_mlp_dim": int(spec.attention_token_mlp_dim),
        "jax/attention_cls_tokens": int(spec.attention_num_cls_tokens),
        "jax/attention_pi_head_hidden_dims": ",".join(
            str(v) for v in spec.attention_pi_head_hidden_dims
        ),
        "jax/attention_vf_head_hidden_dims": ",".join(
            str(v) for v in spec.attention_vf_head_hidden_dims
        ),
        "jax/attention_head_activation": str(spec.attention_head_activation),
        "jax/pass_action_start": int(spec.pass_action_start),
        "jax/pass_action_end": int(spec.pass_action_end),
        "jax/intent_embedding_enabled": bool(spec.intent_embedding_enabled),
        "jax/intent_embedding_dim": int(spec.intent_embedding_dim),
        "jax/intent_conditioning_scale": float(spec.intent_conditioning_scale),
        "jax/num_intents": int(spec.num_intents),
        "jax/intent_selector_enabled": bool(spec.intent_selector_enabled),
        "jax/intent_selector_hidden_dim": int(spec.intent_selector_hidden_dim),
        "jax/intent_selector_learning_rate": _selector_learning_rate_for_args(
            args,
            trainer_config,
        ),
        "jax/intent_selector_alpha_start": float(
            getattr(args, "intent_selector_alpha_start", 0.0)
        ),
        "jax/intent_selector_alpha_end": float(
            getattr(args, "intent_selector_alpha_end", 1.0)
        ),
        "jax/intent_selector_alpha_warmup_updates": int(
            getattr(args, "intent_selector_alpha_warmup_updates", 0)
        ),
        "jax/intent_selector_alpha_ramp_updates": int(
            getattr(args, "intent_selector_alpha_ramp_updates", 1)
        ),
        "jax/intent_selector_eps_start": float(
            getattr(args, "intent_selector_eps_start", 0.0)
        ),
        "jax/intent_selector_eps_end": float(
            getattr(args, "intent_selector_eps_end", 0.0)
        ),
        "jax/intent_selector_eps_warmup_updates": int(
            getattr(args, "intent_selector_eps_warmup_updates", 0)
        ),
        "jax/intent_selector_eps_ramp_updates": int(
            getattr(args, "intent_selector_eps_ramp_updates", 1)
        ),
        "jax/intent_selector_entropy_coef": float(
            getattr(args, "intent_selector_entropy_coef", 0.01)
        ),
        "jax/intent_selector_usage_reg_coef": float(
            getattr(args, "intent_selector_usage_reg_coef", 0.01)
        ),
        "jax/intent_selector_value_coef": float(
            getattr(args, "intent_selector_value_coef", 0.5)
        ),
        "jax/intent_selector_train_every_rollouts": int(
            getattr(args, "intent_selector_train_every_rollouts", 1)
        ),
        "jax/intent_selector_max_samples_per_update": int(
            getattr(args, "intent_selector_max_samples_per_update", 0)
        ),
        "jax/intent_selector_multiselect_enabled": bool(
            getattr(args, "intent_selector_multiselect_enabled", False)
        ),
        "jax/intent_selector_min_play_steps": int(
            getattr(args, "intent_selector_min_play_steps", 3)
        ),
        "jax/rebound_skill_std": float(getattr(args, "rebound_skill_std", 0.0)),
        "jax/rebound_skill_sampling_mode": str(
            getattr(args, "rebound_skill_sampling_mode", "gaussian") or "gaussian"
        ),
        "jax/rebound_skill_high": float(getattr(args, "rebound_skill_high", 1.0)),
        "jax/rebound_skill_low": float(getattr(args, "rebound_skill_low", -0.25)),
        "jax/rebound_skill_weight": float(getattr(args, "rebound_skill_weight", 0.0)),
        "jax/rebound_basket_position_weight": float(
            getattr(args, "rebound_basket_position_weight", 0.0)
        ),
        "jax/rebound_contest_mode": str(
            getattr(args, "rebound_contest_mode", "global_contest") or "global_contest"
        ),
        "jax/rebound_contest_radius": int(getattr(args, "rebound_contest_radius", 1)),
        "jax/rebound_obs_top_n_targets": int(
            getattr(args, "rebound_obs_top_n_targets", 0)
        ),
        "jax/rebound_win_prob_features": bool(
            getattr(args, "rebound_win_prob_features", False)
        ),
        "jax/rebound_target_observation_features": bool(
            getattr(args, "rebound_target_observation_features", True)
        ),
        "jax/rebound_terminal_reward_mode": str(
            getattr(args, "rebound_terminal_reward_mode", "actual_points")
            or "actual_points"
        ),
        "jax/enable_rebound_reward_redistribution": bool(
            getattr(args, "enable_rebound_reward_redistribution", False)
        ),
        "jax/offensive_rebound_reward_advance": float(
            getattr(args, "offensive_rebound_reward_advance", 0.4)
        ),
        "jax/rebound_reward_once_per_possession": bool(
            getattr(args, "rebound_reward_once_per_possession", True)
        ),
        "jax/rebound_critic_enabled": bool(
            getattr(args, "rebound_critic_enabled", False)
        ),
        "jax/rebound_critic_policy_coef": float(
            trainer_config.rebound_critic_policy_coef
        ),
        "jax/rebound_critic_value_coef": float(
            trainer_config.rebound_critic_value_coef
        ),
        "jax/rebound_counterfactual_positioning_enabled": bool(
            getattr(args, "rebound_counterfactual_positioning_enabled", False)
        ),
        "jax/rebound_counterfactual_positioning_coef": float(
            trainer_config.rebound_counterfactual_positioning_coef
        ),
        "jax/task_reward_scale_start": (
            ""
            if getattr(args, "task_reward_scale_start", None) is None
            else float(getattr(args, "task_reward_scale_start"))
        ),
        "jax/task_reward_scale_end": (
            ""
            if getattr(args, "task_reward_scale_end", None) is None
            else float(getattr(args, "task_reward_scale_end"))
        ),
        "jax/task_reward_scale_warmup_updates": (
            -1
            if getattr(args, "task_reward_scale_warmup_updates", None) is None
            else int(getattr(args, "task_reward_scale_warmup_updates"))
        ),
        "jax/task_reward_scale_ramp_updates": (
            -1
            if getattr(args, "task_reward_scale_ramp_updates", None) is None
            else int(getattr(args, "task_reward_scale_ramp_updates"))
        ),
        "jax/task_reward_scale_warmup_steps": int(
            getattr(args, "task_reward_scale_warmup_steps", 0)
        ),
        "jax/task_reward_scale_ramp_steps": int(
            getattr(args, "task_reward_scale_ramp_steps", 1)
        ),
        "jax/multi_possession_limit_start": _multi_possession_start_limit(args),
        "jax/multi_possession_limit_end": _multi_possession_final_limit(args),
        "jax/multi_possession_limit_ramp_updates": int(
            getattr(args, "multi_possession_limit_ramp_updates", 0)
        ),
        "jax/multi_possession_limit_curriculum_enabled": (
            _multi_possession_curriculum_enabled(args)
        ),
        "jax/multi_possession_overtime_round_cap": int(
            getattr(args, "multi_possession_overtime_round_cap", 0)
        ),
        "jax/pass_mode": str(getattr(args, "pass_mode")),
        "jax/use_set_obs": bool(getattr(args, "use_set_obs")),
        "jax/training_team": str(getattr(args, "training_team")),
        "jax/checkpoint_every_updates": int(args.checkpoint_every_updates),
        "jax/checkpoint_schedule": str(getattr(args, "checkpoint_schedule", "fixed")),
        "jax/checkpoint_log_initial_updates": int(
            getattr(args, "checkpoint_log_initial_updates", 1)
        ),
        "jax/checkpoint_log_ramp_updates": int(
            getattr(args, "checkpoint_log_ramp_updates", 0)
        ),
        "jax/resume_reset_env_state": bool(
            getattr(args, "resume_reset_env_state", False)
            or (
                str(getattr(args, "continue_run_id", "") or "").strip()
                and not bool(getattr(args, "continue_preserve_env_state", False))
            )
        ),
        "jax/resume_reset_intent_discriminator_state": bool(
            getattr(args, "resume_reset_intent_discriminator_state", False)
            or (
                str(getattr(args, "continue_run_id", "") or "").strip()
                and not bool(
                    getattr(args, "continue_preserve_intent_discriminator_state", False)
                )
            )
        ),
        "jax/continue_preserve_env_state": bool(
            getattr(args, "continue_preserve_env_state", False)
        ),
        "jax/continue_preserve_intent_discriminator_state": bool(
            getattr(args, "continue_preserve_intent_discriminator_state", False)
        ),
        "jax/continue_run_id": str(getattr(args, "continue_run_id", "") or ""),
        "jax/continue_artifact": str(getattr(args, "continue_artifact", "") or ""),
        "jax/continue_opponent_pool_size": int(
            getattr(args, "continue_opponent_pool_size", -1)
        ),
        "jax/continue_cache_dir": str(getattr(args, "continue_cache_dir", "") or ""),
        "jax/frozen_opponent_checkpoint": str(
            getattr(args, "frozen_opponent_checkpoint", "") or ""
        ),
        "jax/frozen_opponent_run_id": str(
            getattr(args, "frozen_opponent_run_id", "") or ""
        ),
        "jax/frozen_opponent_artifact": str(
            getattr(args, "frozen_opponent_artifact", "") or ""
        ),
        "jax/opponent_pool_enabled": not bool(
            getattr(args, "disable_opponent_pool", False)
        ),
        "jax/opponent_pool_size": int(getattr(args, "opponent_pool_size", 10)),
        "jax/opponent_pool_beta": float(getattr(args, "opponent_pool_beta", 0.7)),
        "jax/opponent_pool_exploration": float(
            getattr(args, "opponent_pool_exploration", 0.0)
        ),
        "jax/opponent_deterministic_episode_prob": float(
            getattr(args, "opponent_deterministic_episode_prob", 0.0)
        ),
        "jax/opponent_deterministic_episode_prob_start": (
            None
            if getattr(args, "opponent_deterministic_episode_prob_start", None) is None
            else float(getattr(args, "opponent_deterministic_episode_prob_start"))
        ),
        "jax/opponent_deterministic_episode_prob_end": (
            None
            if getattr(args, "opponent_deterministic_episode_prob_end", None) is None
            else float(getattr(args, "opponent_deterministic_episode_prob_end"))
        ),
        "jax/opponent_deterministic_episode_prob_warmup_updates": int(
            getattr(args, "opponent_deterministic_episode_prob_warmup_updates", 0)
        ),
        "jax/opponent_deterministic_episode_prob_ramp_updates": int(
            getattr(args, "opponent_deterministic_episode_prob_ramp_updates", 1)
        ),
        "jax/grouped_opponent_sampling": _uses_grouped_opponent_sampling(args),
        "jax/opponent_group_count": int(getattr(args, "opponent_group_count", 8)),
        "jax/intent_diversity_enabled": bool(
            getattr(args, "intent_diversity_enabled", False)
        ),
        "jax/intent_diversity_beta_target": float(
            getattr(args, "intent_diversity_beta_target", 0.05)
        ),
        "jax/intent_diversity_warmup_updates": (
            -1
            if getattr(args, "intent_diversity_warmup_updates", None) is None
            else int(getattr(args, "intent_diversity_warmup_updates"))
        ),
        "jax/intent_diversity_ramp_updates": (
            -1
            if getattr(args, "intent_diversity_ramp_updates", None) is None
            else int(getattr(args, "intent_diversity_ramp_updates"))
        ),
        "jax/intent_diversity_warmup_steps": int(
            getattr(args, "intent_diversity_warmup_steps", 1_000_000)
        ),
        "jax/intent_diversity_ramp_steps": int(
            getattr(args, "intent_diversity_ramp_steps", 1_000_000)
        ),
        "jax/intent_diversity_clip": float(getattr(args, "intent_diversity_clip", 2.0)),
        "jax/intent_disc_lr": float(getattr(args, "intent_disc_lr", 3e-4)),
        "jax/intent_disc_batch_size": int(getattr(args, "intent_disc_batch_size", 256)),
        "jax/intent_disc_updates_per_rollout": int(
            getattr(args, "intent_disc_updates_per_rollout", 2)
        ),
        "jax/intent_disc_hidden_dim": int(getattr(args, "intent_disc_hidden_dim", 128)),
        "jax/intent_disc_encoder_type": str(
            getattr(args, "intent_disc_encoder_type", "mlp_mean")
        ),
        "jax/intent_disc_dropout": float(getattr(args, "intent_disc_dropout", 0.1)),
        "jax/intent_disc_eval_holdout_fraction": float(
            getattr(args, "intent_disc_eval_holdout_fraction", 0.25)
        ),
        "jax/intent_sample_dump_size": int(
            getattr(args, "intent_sample_dump_size", 2048)
        ),
        "jax/disc_eval_batch_output": bool(
            getattr(args, "disc_eval_batch_output", False)
        ),
        "jax/intent_policy_sensitivity_enabled": bool(
            getattr(args, "intent_policy_sensitivity_enabled", True)
        ),
        "jax/intent_policy_sensitivity_sample_states": int(
            getattr(args, "intent_policy_sensitivity_sample_states", 32)
        ),
        "jax/intent_policy_sensitivity_log_every_rollouts": int(
            getattr(args, "intent_policy_sensitivity_log_every_rollouts", 4)
        ),
    }
    for key, value in _jax_env_config_from_args(args).items():
        params[f"jax/env/{key}"] = value
    mlflow.log_params(params)


def _log_mlflow_metrics(
    mlflow, metrics: dict[str, Any], *, step: int, prefix: str
) -> None:
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            mlflow.log_metric(f"{prefix}/{key}", float(value), step=int(step))


_CORE_TRAIN_METRICS = frozenset(
    {
        # Throughput and timing.
        "update_index",
        "steps_per_update",
        "train_loop_steps_per_sec",
        "train_loop_active_steps_per_sec",
        "end_to_end_steps_per_sec",
        "active_end_to_end_steps_per_sec",
        "rollout_states_per_sec",
        "ppo_update_optimizer_samples_per_sec",
        "train_loop_latency_ms",
        "end_to_end_latency_ms",
        "rollout_latency_ms",
        "update_latency_ms",
        "rollout_time_pct",
        "ppo_update_time_pct",
        # PPO health.
        "total_loss",
        "policy_loss",
        "value_loss",
        "entropy_bonus",
        "entropy_coef",
        "approx_kl",
        "clip_fraction",
        "grad_norm",
        "advantage_std",
        "mean_reward",
        "mean_return",
        "mean_value",
        "game_reward_mean",
        "value_explained_variance_mean",
        "value_bias_mean",
        "value_rmse",
        # Episode and possession outcomes.
        "completed_episode_count",
        "completed_possession_count",
        "mean_completed_episode_length",
        "mean_live_steps_per_completed_possession",
        "shots_per_completed_episode",
        "mean_assists_per_completed_episode",
        "mean_completed_passes_per_completed_episode",
        "mean_pass_attempts_per_completed_episode",
        "mean_learner_turnovers_per_completed_episode",
        "mean_opponent_turnovers_per_completed_episode",
        "mean_turnovers_per_completed_episode",
        "mean_offensive_three_seconds_per_completed_episode",
        "mean_defensive_lane_violations_per_completed_episode",
        "offensive_three_seconds_rate_per_step",
        "defensive_lane_violation_rate_per_step",
        "check_pickup_rate",
        "check_mean_pickup_steps",
        "check_violation_count",
        "clearance_elapsed_steps_mean",
        "offense_learner_points_per_completed_episode",
        "offense_opponent_points_per_completed_episode",
        "defense_learner_points_per_completed_episode",
        "defense_opponent_points_per_completed_episode",
        "offense_learner_mean_reward",
        "offense_opponent_mean_reward",
        "defense_learner_mean_reward",
        "defense_opponent_mean_reward",
        # Spatial diagnostics.
        "spatial_live_step_count",
        "mean_live_all_player_pair_distance",
        "mean_live_offense_teammate_pair_distance",
        "mean_live_defense_teammate_pair_distance",
        "mean_live_boundary_player_fraction",
        "mean_live_corner_player_fraction",
        "mean_live_team_centroid_distance",
        "mean_live_offense_nearest_defender_distance",
        "mean_live_ball_handler_nearest_defender_distance",
        "mean_live_unguarded_offense_fraction",
        "live_ball_handler_pressure_rate",
        # Curricula and frozen-opponent schedule.
        "task_reward_scale",
        "multi_possession_limit_active",
        "episode_possession_limit_mean",
        "episode_overtime_round_cap_mean",
        "opponent_update_index",
        "opponent_unique_update_count",
        "opponent_pool_candidate_count",
        "opponent_deterministic_episode_rate",
        # Intent learner and selector summaries. Per-intent/action breakdowns are
        # intentionally excluded from core because MLflow charts each one.
        "intent_disc_loss",
        "intent_disc_entropy",
        "intent_disc_auc_ovr_macro_trainbatch",
        "intent_disc_auc_ovr_macro_holdout",
        "intent_disc_top1_acc_trainbatch",
        "intent_disc_top1_acc_holdout",
        "intent_disc_holdout_size",
        "intent_bonus_beta",
        "intent_bonus_raw_mean",
        "intent_bonus_shaping_per_step_mean",
        "intent_intrinsic_advantage_magnitude_fraction",
        "intent_intrinsic_segment_mean_steps",
        "intent_policy_sensitivity_tv_mean",
        "intent_policy_sensitivity_tv_p95",
        "intent_policy_sensitivity_argmax_disagreement",
        "selector_alpha",
        "selector_eps",
        "selector_usage_rate",
        "selector_applied_rate",
        "selector_entropy",
        "selector_max_prob",
        "selector_segment_mean_steps",
        "selector_train_loss",
        "selector_train_approx_kl",
        # Rebound learning summaries.
        "rebound_learner_offensive_rebound_rate",
        "rebound_learner_defensive_rebound_rate",
        "rebound_learner_softmax_win_rate_offense",
        "rebound_learner_softmax_win_rate_defense",
        "rebound_aux_policy_loss",
        "rebound_aux_value_loss",
    }
)


_CORE_DEPLOY_METRICS = frozenset(
    {
        "update_index",
        "action_argmax_offense",
        "action_argmax_defense",
        "same_policy_both_sides",
        "episode_count",
        "completed_episode_count",
        "completion_rate",
        "mean_completed_episode_length",
        "completed_possession_count",
        "mean_live_steps_per_completed_possession",
        "mean_offense_score",
        "mean_defense_score",
        "mean_score_margin",
        "mean_game_reward_per_episode",
        "mean_offense_reward_per_episode",
        "mean_defense_reward_per_episode",
        "mean_winner_reward_per_episode",
        "shot_make_rate",
        "shot_dunk_share",
        "shot_two_share",
        "shot_three_share",
        "shots_per_episode",
        "pass_completion_rate",
        "passes_per_episode",
        "assists_per_episode",
        "turnovers_per_episode",
        "offensive_rebound_rate",
        "defensive_rebound_rate",
        "check_pickup_rate",
        "check_mean_pickup_steps",
        "mean_live_all_player_pair_distance",
        "mean_live_offense_teammate_pair_distance",
        "mean_live_defense_teammate_pair_distance",
        "mean_live_boundary_player_fraction",
        "mean_live_corner_player_fraction",
        "mean_live_team_centroid_distance",
        "mean_live_offense_nearest_defender_distance",
        "mean_live_ball_handler_nearest_defender_distance",
        "mean_live_unguarded_offense_fraction",
        "live_ball_handler_pressure_rate",
    }
)


def _filter_mlflow_train_metrics(
    metrics: dict[str, Any],
    *,
    profile: str = "core",
) -> dict[str, Any]:
    if profile == "full":
        return dict(metrics)
    if profile != "core":
        raise ValueError(f"Unknown MLflow metric profile: {profile}")
    return {key: value for key, value in metrics.items() if key in _CORE_TRAIN_METRICS}


def _filter_mlflow_deploy_metrics(
    metrics: dict[str, Any],
    *,
    profile: str = "core",
) -> dict[str, Any]:
    if profile == "full":
        return dict(metrics)
    if profile != "core":
        raise ValueError(f"Unknown MLflow metric profile: {profile}")
    return {key: value for key, value in metrics.items() if key in _CORE_DEPLOY_METRICS}


def _log_mlflow_checkpoint_artifacts(
    mlflow,
    *,
    numbered_checkpoint_path: str,
    update_index: int,
) -> str:
    checkpoint_dir = Path(numbered_checkpoint_path)
    artifact_path = f"models/{checkpoint_dir.name}"
    mlflow.log_artifacts(str(checkpoint_dir), artifact_path=artifact_path)
    mlflow.set_tag("model_backend", "jax")
    mlflow.set_tag("jax_checkpoint_format", "orbax_v2")
    mlflow.set_tag("jax_latest_checkpoint_artifact", artifact_path)
    mlflow.set_tag("jax_latest_checkpoint_update", str(int(update_index)))
    return artifact_path


def _log_mlflow_intent_sample_artifact(
    mlflow,
    *,
    sample_payload: dict[str, np.ndarray],
    update_index: int,
) -> str | None:
    if not sample_payload:
        return None
    with TemporaryDirectory(prefix="basketworld_jax_intent_samples_") as tmpdir:
        path = Path(tmpdir) / f"intent_samples_update_{int(update_index):07d}.npz"
        np.savez_compressed(path, **sample_payload)
        artifact_path = f"intent_samples/update_{int(update_index):07d}"
        mlflow.log_artifact(str(path), artifact_path=artifact_path)
        return f"{artifact_path}/{path.name}"


def _historical_episode_rows(
    *,
    final_state,
    diagnostics: dict[str, Any] | None,
    candidate_fixed_team: str,
) -> list[dict[str, Any]]:
    """Convert compact device-side counters into JSON-ready episode records."""
    episode_ended = np.asarray(final_state.episode_ended, dtype=bool)
    team_a_score = np.asarray(final_state.team_a_score, dtype=np.int64)
    team_b_score = np.asarray(final_state.team_b_score, dtype=np.int64)
    batch_size = int(episode_ended.size)
    # Keep the pure host-side summarizer usable with the small synthetic final
    # states in unit tests and with legacy checkpoints that predate these
    # multi-possession state fields.
    starting_offense_team = np.asarray(
        getattr(final_state, "starting_offense_team", np.zeros(batch_size)),
        dtype=np.int8,
    )
    completed_possessions = np.asarray(
        getattr(final_state, "completed_possessions", np.zeros(batch_size)),
        dtype=np.int32,
    )
    team_a_completed_possessions = np.asarray(
        getattr(
            final_state,
            "team_a_completed_possessions",
            np.where(
                starting_offense_team == 0,
                (completed_possessions + 1) // 2,
                completed_possessions // 2,
            ),
        ),
        dtype=np.int32,
    )
    team_b_completed_possessions = np.asarray(
        getattr(
            final_state,
            "team_b_completed_possessions",
            completed_possessions - team_a_completed_possessions,
        ),
        dtype=np.int32,
    )
    step_count = np.asarray(
        getattr(final_state, "step_count", np.zeros(batch_size)),
        dtype=np.int32,
    )
    diagnostic_arrays = {
        key: np.asarray(value, dtype=np.float64)
        for key, value in (diagnostics or {}).items()
    }

    rows = []
    candidate_is_team_a = candidate_fixed_team == "team_a"
    for episode_index in range(batch_size):
        candidate_score = int(
            team_a_score[episode_index]
            if candidate_is_team_a
            else team_b_score[episode_index]
        )
        opponent_score = int(
            team_b_score[episode_index]
            if candidate_is_team_a
            else team_a_score[episode_index]
        )
        candidate_started_with_possession = bool(
            int(starting_offense_team[episode_index])
            == (0 if candidate_is_team_a else 1)
        )
        if not bool(episode_ended[episode_index]):
            outcome = "truncated"
        elif candidate_score > opponent_score:
            outcome = "candidate_win"
        elif candidate_score < opponent_score:
            outcome = "candidate_loss"
        else:
            outcome = "tie"
        row = {
            "episode_index": int(episode_index),
            "candidate_fixed_team": candidate_fixed_team,
            "candidate_started_with_possession": candidate_started_with_possession,
            "starting_offense_team": (
                "team_a" if int(starting_offense_team[episode_index]) == 0 else "team_b"
            ),
            "completed": bool(episode_ended[episode_index]),
            "termination_kind": (
                "completed"
                if bool(episode_ended[episode_index])
                else "horizon_truncated"
            ),
            "candidate_score": candidate_score,
            "opponent_score": opponent_score,
            "candidate_point_differential": candidate_score - opponent_score,
            "outcome": outcome,
            "completed_possessions": int(completed_possessions[episode_index]),
            "candidate_completed_possessions": int(
                team_a_completed_possessions[episode_index]
                if candidate_is_team_a
                else team_b_completed_possessions[episode_index]
            ),
            "opponent_completed_possessions": int(
                team_b_completed_possessions[episode_index]
                if candidate_is_team_a
                else team_a_completed_possessions[episode_index]
            ),
            "environment_steps": int(step_count[episode_index]),
        }
        for key in HISTORICAL_MATCH_DIAGNOSTIC_KEYS:
            values = diagnostic_arrays.get(key)
            if values is None:
                row[key] = 0.0 if key in HISTORICAL_MATCH_FLOAT_DIAGNOSTIC_KEYS else 0
            elif key in HISTORICAL_MATCH_FLOAT_DIAGNOSTIC_KEYS:
                row[key] = float(values[episode_index])
            else:
                row[key] = int(round(float(values[episode_index])))
        rows.append(row)
    return rows


def _historical_outcome_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed_rows = [row for row in rows if row["completed"]]
    completed_count = len(completed_rows)
    wins = sum(row["outcome"] == "candidate_win" for row in completed_rows)
    ties = sum(row["outcome"] == "tie" for row in completed_rows)
    losses = sum(row["outcome"] == "candidate_loss" for row in completed_rows)

    def _mean(key: str) -> float | None:
        return (
            float(np.mean([row[key] for row in completed_rows]))
            if completed_rows
            else None
        )

    return {
        "episode_count": len(rows),
        "completed_episode_count": completed_count,
        "candidate_wins": int(wins),
        "candidate_ties": int(ties),
        "candidate_losses": int(losses),
        "candidate_win_rate": (
            float(wins / completed_count) if completed_count else None
        ),
        "candidate_tie_rate": (
            float(ties / completed_count) if completed_count else None
        ),
        "candidate_loss_rate": (
            float(losses / completed_count) if completed_count else None
        ),
        "mean_candidate_score": _mean("candidate_score"),
        "mean_opponent_score": _mean("opponent_score"),
        "mean_candidate_point_differential": _mean("candidate_point_differential"),
    }


def _summarize_historical_match(
    *,
    candidate_update: int,
    opponent_update: int,
    team_a_candidate_final_state,
    team_b_candidate_final_state,
    team_a_candidate_diagnostics: dict[str, Any] | None = None,
    team_b_candidate_diagnostics: dict[str, Any] | None = None,
    seed: int,
    horizon: int,
) -> dict[str, Any]:
    """Summarize balanced match results and compact gameplay diagnostics."""
    team_a_rows = _historical_episode_rows(
        final_state=team_a_candidate_final_state,
        diagnostics=team_a_candidate_diagnostics,
        candidate_fixed_team="team_a",
    )
    team_b_rows = _historical_episode_rows(
        final_state=team_b_candidate_final_state,
        diagnostics=team_b_candidate_diagnostics,
        candidate_fixed_team="team_b",
    )
    episode_rows = [*team_a_rows, *team_b_rows]
    completed_rows = [row for row in episode_rows if row["completed"]]
    outcome_summary = _historical_outcome_summary(episode_rows)
    episode_count = len(episode_rows)
    completed_count = len(completed_rows)

    scoreline_histogram: dict[str, int] = {}
    point_differential_histogram: dict[str, int] = {}
    for row in completed_rows:
        scoreline = f"{row['candidate_score']}-{row['opponent_score']}"
        scoreline_histogram[scoreline] = scoreline_histogram.get(scoreline, 0) + 1
        differential = str(int(row["candidate_point_differential"]))
        point_differential_histogram[differential] = (
            point_differential_histogram.get(differential, 0) + 1
        )

    team_assignment_summaries = {
        "candidate_team_a": _historical_outcome_summary(team_a_rows),
        "candidate_team_b": _historical_outcome_summary(team_b_rows),
    }
    starting_possession_summaries = {
        "candidate_started_with_possession": _historical_outcome_summary(
            [row for row in episode_rows if row["candidate_started_with_possession"]]
        ),
        "opponent_started_with_possession": _historical_outcome_summary(
            [
                row
                for row in episode_rows
                if not row["candidate_started_with_possession"]
            ]
        ),
    }

    def _total(key: str) -> float | int:
        total = sum(row[key] for row in episode_rows)
        return (
            float(total)
            if key in HISTORICAL_MATCH_FLOAT_DIAGNOSTIC_KEYS
            else int(total)
        )

    def _rate(numerator: float, denominator: float) -> float | None:
        return float(numerator / denominator) if denominator else None

    candidate_score_total = int(sum(row["candidate_score"] for row in completed_rows))
    opponent_score_total = int(sum(row["opponent_score"] for row in completed_rows))
    completed_possessions = _total("completed_possessions")
    candidate_completed_possessions = _total("candidate_completed_possessions")
    opponent_completed_possessions = _total("opponent_completed_possessions")
    scalar_diagnostics: dict[str, Any] = {
        "mean_candidate_score": outcome_summary["mean_candidate_score"],
        "mean_opponent_score": outcome_summary["mean_opponent_score"],
        "mean_combined_score": (
            float((candidate_score_total + opponent_score_total) / completed_count)
            if completed_count
            else None
        ),
        "zero_zero_tie_count": int(
            sum(
                row["candidate_score"] == 0 and row["opponent_score"] == 0
                for row in completed_rows
            )
        ),
        "mean_steps_per_episode": _rate(_total("active_steps"), episode_count),
        "mean_live_steps_per_completed_possession": _rate(
            _total("completed_possession_live_steps"),
            completed_possessions,
        ),
        "completed_possession_count": completed_possessions,
        "candidate_points_per_completed_possession": _rate(
            candidate_score_total,
            candidate_completed_possessions,
        ),
        "opponent_points_per_completed_possession": _rate(
            opponent_score_total,
            opponent_completed_possessions,
        ),
        "candidate_completed_possession_count": candidate_completed_possessions,
        "opponent_completed_possession_count": opponent_completed_possessions,
        "check_opportunity_count": _total("check_opportunities"),
        "check_setup_opportunity_count": _total("check_setup_opportunities"),
        "check_setup_steps_total": _total("check_setup_steps"),
        "check_setup_mean_executed_steps": _rate(
            _total("check_setup_steps"),
            _total("check_setup_opportunities"),
        ),
        "check_pickup_count": _total("check_pickups"),
        "check_violation_count": _total("check_violations"),
        "check_pickup_rate": _rate(
            _total("check_pickups"),
            _total("check_opportunities"),
        ),
        "check_mean_pickup_steps": _rate(
            _total("check_pickup_steps"),
            _total("check_pickups"),
        ),
    }
    scalar_diagnostics["zero_zero_tie_rate"] = _rate(
        scalar_diagnostics["zero_zero_tie_count"],
        completed_count,
    )
    for key in HISTORICAL_MATCH_DIAGNOSTIC_KEYS:
        total = _total(key)
        scalar_diagnostics[f"{key}_total"] = total
        scalar_diagnostics[f"{key}_per_completed_episode"] = _rate(
            total,
            completed_count,
        )
    spatial_live_steps = scalar_diagnostics["spatial_live_steps_total"]
    scalar_diagnostics.update(
        {
            "mean_live_all_player_pair_distance": _rate(
                scalar_diagnostics["spatial_all_player_pair_distance_total"],
                spatial_live_steps,
            ),
            "candidate_mean_live_teammate_pair_distance": _rate(
                scalar_diagnostics["candidate_spatial_teammate_pair_distance_total"],
                spatial_live_steps,
            ),
            "opponent_mean_live_teammate_pair_distance": _rate(
                scalar_diagnostics["opponent_spatial_teammate_pair_distance_total"],
                spatial_live_steps,
            ),
            "mean_live_boundary_player_fraction": _rate(
                scalar_diagnostics["spatial_boundary_player_fraction_total"],
                spatial_live_steps,
            ),
            "mean_live_corner_player_fraction": _rate(
                scalar_diagnostics["spatial_corner_player_fraction_total"],
                spatial_live_steps,
            ),
            "mean_live_ball_handler_nearest_defender_distance": _rate(
                scalar_diagnostics[
                    "spatial_ball_handler_nearest_defender_distance_total"
                ],
                scalar_diagnostics["spatial_ball_handler_samples_total"],
            ),
            "live_ball_handler_pressure_rate": _rate(
                scalar_diagnostics["spatial_ball_handler_pressured_total"],
                scalar_diagnostics["spatial_ball_handler_samples_total"],
            ),
            "mean_live_offense_nearest_defender_distance": _rate(
                scalar_diagnostics["spatial_offense_nearest_defender_distance_total"],
                spatial_live_steps,
            ),
            "mean_live_unguarded_offense_fraction": _rate(
                scalar_diagnostics["spatial_unguarded_offense_fraction_total"],
                spatial_live_steps,
            ),
            "mean_live_team_centroid_distance": _rate(
                scalar_diagnostics["spatial_team_centroid_distance_total"],
                spatial_live_steps,
            ),
        }
    )
    for side in ("candidate", "opponent"):
        pass_attempts = scalar_diagnostics[f"{side}_pass_attempts_total"]
        completed_passes = scalar_diagnostics[f"{side}_completed_passes_total"]
        shot_attempts = scalar_diagnostics[f"{side}_shot_attempts_total"]
        clearance_events = scalar_diagnostics[f"{side}_clearance_events_total"]
        inbound_attempts = scalar_diagnostics[f"{side}_inbound_pass_attempts_total"]
        scalar_diagnostics[f"{side}_pass_completion_rate"] = _rate(
            completed_passes,
            pass_attempts,
        )
        scalar_diagnostics[f"{side}_assist_per_completed_pass_rate"] = _rate(
            scalar_diagnostics[f"{side}_assists_total"],
            completed_passes,
        )
        scalar_diagnostics[f"{side}_shot_make_rate"] = _rate(
            scalar_diagnostics[f"{side}_shot_makes_total"],
            shot_attempts,
        )
        scalar_diagnostics[f"{side}_rebound_win_rate"] = _rate(
            scalar_diagnostics[f"{side}_rebounds_total"],
            scalar_diagnostics["rebound_attempts_total"],
        )
        scalar_diagnostics[f"{side}_inbound_pass_completion_rate"] = _rate(
            scalar_diagnostics[f"{side}_inbound_completed_passes_total"],
            inbound_attempts,
        )
        scalar_diagnostics[f"{side}_mean_clearance_steps"] = _rate(
            scalar_diagnostics[f"{side}_clearance_elapsed_steps_total"],
            clearance_events,
        )
    for label, assignment_summary in team_assignment_summaries.items():
        scalar_diagnostics[f"{label}_win_rate"] = assignment_summary[
            "candidate_win_rate"
        ]
        scalar_diagnostics[f"{label}_tie_rate"] = assignment_summary[
            "candidate_tie_rate"
        ]
        scalar_diagnostics[f"{label}_mean_candidate_score"] = assignment_summary[
            "mean_candidate_score"
        ]
    for label, starting_summary in starting_possession_summaries.items():
        scalar_diagnostics[f"{label}_win_rate"] = starting_summary["candidate_win_rate"]
        scalar_diagnostics[f"{label}_mean_candidate_score"] = starting_summary[
            "mean_candidate_score"
        ]

    return {
        "candidate_update": int(candidate_update),
        "opponent_update": int(opponent_update),
        "self_match": bool(candidate_update == opponent_update),
        "episode_count": episode_count,
        "episodes_candidate_team_a": len(team_a_rows),
        "episodes_candidate_team_b": len(team_b_rows),
        "completed_episode_count": completed_count,
        "truncated_episode_count": int(episode_count - completed_count),
        "completion_rate": _rate(completed_count, episode_count) or 0.0,
        **outcome_summary,
        **scalar_diagnostics,
        "scoreline_histogram": scoreline_histogram,
        "point_differential_histogram": point_differential_histogram,
        "team_assignment_summaries": team_assignment_summaries,
        "starting_possession_summaries": starting_possession_summaries,
        "episode_diagnostics": episode_rows,
        "seed": int(seed),
        "horizon": int(horizon),
        "action_mode": "deterministic",
    }


def _run_historical_match_evaluation(
    *,
    runner,
    statics: dict[str, Any],
    candidate_params,
    opponent_params,
    candidate_update: int,
    opponent_update: int,
    args,
    jax,
    jnp,
) -> dict[str, Any]:
    """Run one reproducible, balanced policy-checkpoint matchup."""
    episode_count = int(getattr(args, "historical_eval_episodes", 200))
    episodes_per_assignment = episode_count // 2
    root_key = jax.random.PRNGKey(int(getattr(args, "historical_eval_seed", 3_000_000)))
    root_key = jax.random.fold_in(root_key, int(candidate_update))
    root_key = jax.random.fold_in(root_key, int(opponent_update))
    reset_key = jax.random.fold_in(root_key, 0)
    rollout_key = jax.random.fold_in(root_key, 1)
    reset_keys = jax.random.split(reset_key, episodes_per_assignment)
    # The two assignments begin from the same seeded game rows. Only policy
    # ownership changes, removing incidental Team A/B start bias.
    team_a_initial_state = reset_batch_minimal(statics["offense"], reset_keys, jax, jnp)
    team_b_initial_state = reset_batch_minimal(statics["defense"], reset_keys, jax, jnp)
    team_a_final_state, team_a_diagnostics = runner(
        statics["offense"],
        team_a_initial_state,
        candidate_params,
        opponent_params,
        jax.random.fold_in(rollout_key, 0),
        int(getattr(args, "historical_eval_horizon", 1024)),
        bool(getattr(args, "intent_selector_multiselect_enabled", False)),
        int(getattr(args, "intent_selector_min_play_steps", 3)),
    )
    team_b_final_state, team_b_diagnostics = runner(
        statics["defense"],
        team_b_initial_state,
        candidate_params,
        opponent_params,
        jax.random.fold_in(rollout_key, 1),
        int(getattr(args, "historical_eval_horizon", 1024)),
        bool(getattr(args, "intent_selector_multiselect_enabled", False)),
        int(getattr(args, "intent_selector_min_play_steps", 3)),
    )
    block_until_ready_tree(
        (
            team_a_final_state,
            team_a_diagnostics,
            team_b_final_state,
            team_b_diagnostics,
        )
    )
    return _summarize_historical_match(
        candidate_update=candidate_update,
        opponent_update=opponent_update,
        team_a_candidate_final_state=team_a_final_state,
        team_b_candidate_final_state=team_b_final_state,
        team_a_candidate_diagnostics=team_a_diagnostics,
        team_b_candidate_diagnostics=team_b_diagnostics,
        seed=int(getattr(args, "historical_eval_seed", 3_000_000)),
        horizon=int(getattr(args, "historical_eval_horizon", 1024)),
    )


def _historical_eval_artifact_payload(
    *,
    milestone_updates: tuple[int, ...],
    milestone_records: dict[int, dict[str, Any]],
    match_history: list[dict[str, Any]],
    update_artifact_paths: dict[int, str] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "milestone_updates": [int(update) for update in milestone_updates],
        "milestone_checkpoints": {
            str(int(update)): dict(record)
            for update, record in sorted(milestone_records.items())
        },
        "matches": [dict(record) for record in match_history],
        "evaluation_update_artifacts": {
            str(int(update)): str(path)
            for update, path in sorted((update_artifact_paths or {}).items())
        },
    }


def _persist_historical_eval_artifact(
    *,
    checkpoint_dir: str,
    mlflow,
    payload: dict[str, Any],
) -> str | None:
    local_path = None
    if checkpoint_dir:
        local_path = Path(checkpoint_dir) / HISTORICAL_EVAL_ARTIFACT_NAME
        write_json(local_path, payload)
    if mlflow is not None:
        if local_path is not None:
            mlflow.log_artifact(
                str(local_path), artifact_path=TRAIN_LOOP_SUMMARY_ARTIFACT_DIR
            )
        else:
            with TemporaryDirectory(
                prefix="basketworld_jax_historical_eval_"
            ) as tmpdir:
                path = Path(tmpdir) / HISTORICAL_EVAL_ARTIFACT_NAME
                write_json(path, payload)
                mlflow.log_artifact(
                    str(path), artifact_path=TRAIN_LOOP_SUMMARY_ARTIFACT_DIR
                )
        return HISTORICAL_EVAL_ARTIFACT_PATH
    return str(local_path) if local_path is not None else None


def _historical_eval_update_artifact_path(candidate_update: int) -> str:
    return (
        f"{HISTORICAL_EVAL_UPDATE_ARTIFACT_DIR}/"
        f"update_{int(candidate_update):07d}.json"
    )


def _historical_eval_update_artifact_payload(
    *,
    candidate_update: int,
    milestone_records: dict[int, dict[str, Any]],
    match_history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create the standalone historical-evaluation record for one update."""
    candidate_update = int(candidate_update)
    matches = [
        dict(record)
        for record in match_history
        if int(record.get("candidate_update", -1)) == candidate_update
    ]
    opponent_updates = sorted({int(record["opponent_update"]) for record in matches})
    return {
        "schema_version": 2,
        "candidate_update": candidate_update,
        "candidate_checkpoint": dict(milestone_records[candidate_update]),
        "evaluated_opponent_updates": opponent_updates,
        "opponent_checkpoints": {
            str(update): dict(milestone_records[update]) for update in opponent_updates
        },
        "matches": matches,
    }


def _persist_historical_eval_update_artifact(
    *,
    checkpoint_dir: str,
    mlflow,
    candidate_update: int,
    payload: dict[str, Any],
) -> str:
    """Persist one candidate-update matchup record locally and/or in MLflow."""
    artifact_path = _historical_eval_update_artifact_path(candidate_update)
    filename = Path(artifact_path).name
    local_path = None
    if checkpoint_dir:
        local_path = Path(checkpoint_dir) / "historical_opponent_evaluations" / filename
        write_json(local_path, payload)
    if mlflow is not None:
        if local_path is not None:
            mlflow.log_artifact(
                str(local_path),
                artifact_path=HISTORICAL_EVAL_UPDATE_ARTIFACT_DIR,
            )
        else:
            with TemporaryDirectory(
                prefix="basketworld_jax_historical_eval_update_"
            ) as tmpdir:
                path = Path(tmpdir) / filename
                write_json(path, payload)
                mlflow.log_artifact(
                    str(path),
                    artifact_path=HISTORICAL_EVAL_UPDATE_ARTIFACT_DIR,
                )
        return artifact_path
    return str(local_path)


def _historical_checkpoint_root(
    *,
    checkpoint_dir: str,
    resume_checkpoint: str,
) -> Path | None:
    """Find the directory containing numbered checkpoint directories."""
    if checkpoint_dir:
        return Path(checkpoint_dir)
    if resume_checkpoint:
        return Path(resume_checkpoint).parent
    return None


def _load_historical_eval_artifact(
    *,
    checkpoint_root: Path | None,
    continuation_run_id: str,
    continue_cache_dir: str,
) -> dict[str, Any] | None:
    """Restore the cumulative matchup record when a run is continued."""
    local_path = (
        checkpoint_root / HISTORICAL_EVAL_ARTIFACT_NAME
        if checkpoint_root is not None
        else None
    )
    if local_path is not None and local_path.is_file():
        with local_path.open("r", encoding="utf-8") as handle:
            return dict(json.load(handle))
    if not continuation_run_id:
        return None

    import mlflow

    setup_mlflow(verbose=False)
    client = mlflow.tracking.MlflowClient()
    try:
        downloaded = _download_mlflow_checkpoint_artifact(
            client,
            run_id=continuation_run_id,
            artifact_path=HISTORICAL_EVAL_ARTIFACT_PATH,
            cache_dir=continue_cache_dir,
        )
    except Exception:
        return None
    if not downloaded.is_file():
        return None
    with downloaded.open("r", encoding="utf-8") as handle:
        return dict(json.load(handle))


def _restore_historical_milestone_params(
    *,
    milestone_updates: tuple[int, ...],
    completed_updates: int,
    checkpoint_root: Path | None,
    continuation_run_id: str,
    continue_cache_dir: str,
    spec: ActorCriticSpec,
    jax,
) -> tuple[dict[int, Any], dict[int, dict[str, Any]]]:
    """Load every reached milestone policy so later matches remain comparable."""
    reached_updates = [
        update for update in milestone_updates if update <= completed_updates
    ]
    if not reached_updates:
        return {}, {}

    client = None
    restored_params: dict[int, Any] = {}
    records: dict[int, dict[str, Any]] = {}
    for update in reached_updates:
        checkpoint_path = (
            checkpoint_root / f"update_{int(update):07d}"
            if checkpoint_root is not None
            else None
        )
        artifact_path = ""
        if checkpoint_path is None or not checkpoint_path.exists():
            if not continuation_run_id:
                raise SystemExit(
                    "Cannot restore historical-evaluation milestone "
                    f"update {update}: checkpoint is unavailable locally. "
                    "Use --checkpoint-dir or --continue-run-id."
                )
            if client is None:
                import mlflow

                setup_mlflow(verbose=False)
                client = mlflow.tracking.MlflowClient()
            artifact_path = f"models/update_{int(update):07d}"
            try:
                checkpoint_path = _download_mlflow_checkpoint_artifact(
                    client,
                    run_id=continuation_run_id,
                    artifact_path=artifact_path,
                    cache_dir=continue_cache_dir,
                )
            except Exception as exc:
                raise SystemExit(
                    "Cannot restore historical-evaluation milestone "
                    f"update {update} from MLflow artifact {artifact_path!r}."
                ) from exc
        payload = load_checkpoint(checkpoint_path)
        if _normalize_policy_spec_dict(payload.get("policy_spec", {})) != asdict(spec):
            raise SystemExit(
                "Historical-evaluation milestone policy_spec does not match "
                f"the current JAX run (update {update})."
            )
        restored_params[int(update)] = jax.device_put(payload["params"])
        records[int(update)] = {
            "update_index": int(update),
            "checkpoint_path": str(checkpoint_path),
            "artifact_path": artifact_path or None,
            "pinned": True,
            "restored_on_resume": True,
        }
    return restored_params, records


def _build_train_loop_summary_payload(result: dict[str, Any]) -> dict[str, Any]:
    """Build a compact MLflow run summary without per-update trace payloads."""
    train_history = result.get("train_history")
    eval_trajectories = result.get("eval_trajectories")
    deploy_eval_history = result.get("deploy_eval_history")
    historical_eval_history = result.get("historical_eval_history")
    summary = {
        key: value
        for key, value in result.items()
        if key
        not in {
            "train_history",
            "eval_trajectories",
            "deploy_eval_history",
            "historical_eval_history",
        }
    }
    summary["train_history_count"] = (
        len(train_history) if isinstance(train_history, list) else 0
    )
    summary["eval_trajectory_count"] = (
        len(eval_trajectories) if isinstance(eval_trajectories, list) else 0
    )
    summary["deploy_eval_history_count"] = (
        len(deploy_eval_history) if isinstance(deploy_eval_history, list) else 0
    )
    summary["historical_eval_match_count"] = (
        len(historical_eval_history) if isinstance(historical_eval_history, list) else 0
    )
    return summary


def _log_mlflow_train_loop_summary(mlflow, result: dict[str, Any]) -> str:
    with TemporaryDirectory(prefix="basketworld_jax_train_summary_") as tmpdir:
        path = Path(tmpdir) / TRAIN_LOOP_SUMMARY_ARTIFACT_NAME
        write_json(path, _build_train_loop_summary_payload(result))
        mlflow.log_artifact(str(path), artifact_path=TRAIN_LOOP_SUMMARY_ARTIFACT_DIR)
    mlflow.set_tag("jax_train_loop_summary_artifact", TRAIN_LOOP_SUMMARY_ARTIFACT_PATH)
    return TRAIN_LOOP_SUMMARY_ARTIFACT_PATH


def _save_local_intent_sample_artifact(
    *,
    checkpoint_dir: str,
    sample_payload: dict[str, np.ndarray],
    update_index: int,
) -> str | None:
    if not sample_payload:
        return None
    root = Path(checkpoint_dir) / "intent_samples"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"intent_samples_update_{int(update_index):07d}.npz"
    np.savez_compressed(path, **sample_payload)
    return str(path)


def _format_summary_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        magnitude = abs(float(value))
        if magnitude >= 1000.0:
            return f"{float(value):,.2f}"
        if magnitude >= 1.0:
            return f"{float(value):.4f}"
        return f"{float(value):.6f}"
    return str(value)


def _safe_metric_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if float(denominator) > 0.0 else 0.0


def _masked_value_diagnostics(values, returns, mask) -> dict[str, float]:
    values_np = np.asarray(values, dtype=np.float32).reshape(-1)
    returns_np = np.asarray(returns, dtype=np.float32).reshape(-1)
    mask_np = np.asarray(mask, dtype=np.float32).reshape(-1)
    empty = {
        "sample_count": 0.0,
        "value_mean": 0.0,
        "return_mean": 0.0,
        "value_bias_mean": 0.0,
        "value_mae": 0.0,
        "value_rmse": 0.0,
        "explained_variance": 0.0,
    }
    if values_np.size != returns_np.size or values_np.size != mask_np.size:
        return dict(empty)
    weight_sum = float(mask_np.sum())
    if weight_sum <= 0.0:
        return dict(empty)
    error = values_np - returns_np
    value_mean = float((values_np * mask_np).sum() / weight_sum)
    return_mean = float((returns_np * mask_np).sum() / weight_sum)
    value_bias_mean = float((error * mask_np).sum() / weight_sum)
    value_mae = float((np.abs(error) * mask_np).sum() / weight_sum)
    value_rmse = float(np.sqrt((np.square(error) * mask_np).sum() / weight_sum))
    centered_returns = returns_np - return_mean
    return_var = float((np.square(centered_returns) * mask_np).sum() / weight_sum)
    error_var = float((np.square(error) * mask_np).sum() / weight_sum)
    explained_variance = (
        float(1.0 - (error_var / return_var)) if return_var > 1.0e-8 else 0.0
    )
    return {
        "sample_count": weight_sum,
        "value_mean": value_mean,
        "return_mean": return_mean,
        "value_bias_mean": value_bias_mean,
        "value_mae": value_mae,
        "value_rmse": value_rmse,
        "explained_variance": explained_variance,
    }


def _summarize_combined_value_diagnostics(metrics: dict[str, Any]) -> dict[str, float]:
    offense_samples = _metric_float(metrics, "offense_value_sample_count")
    defense_samples = _metric_float(metrics, "defense_value_sample_count")
    total_samples = offense_samples + defense_samples
    if total_samples <= 0.0:
        return {
            "value_sample_count": 0.0,
            "value_bias_mean": 0.0,
            "value_mae": 0.0,
            "value_rmse": 0.0,
            "value_explained_variance_mean": 0.0,
            "independent_role_value_sum_mean": 0.0,
            "independent_role_value_sum_abs_mean": 0.0,
            "independent_role_return_sum_mean": 0.0,
            "independent_role_return_sum_abs_mean": 0.0,
        }
    offense_bias = _metric_float(metrics, "offense_value_bias_mean")
    defense_bias = _metric_float(metrics, "defense_value_bias_mean")
    offense_mae = _metric_float(metrics, "offense_value_mae")
    defense_mae = _metric_float(metrics, "defense_value_mae")
    offense_rmse_sq = _metric_float(metrics, "offense_value_rmse") ** 2
    defense_rmse_sq = _metric_float(metrics, "defense_value_rmse") ** 2
    offense_ev = _metric_float(metrics, "offense_value_explained_variance")
    defense_ev = _metric_float(metrics, "defense_value_explained_variance")
    offense_value_mean = _metric_float(metrics, "offense_value_mean")
    defense_value_mean = _metric_float(metrics, "defense_value_mean")
    offense_return_mean = _metric_float(metrics, "offense_return_mean")
    defense_return_mean = _metric_float(metrics, "defense_return_mean")
    independent_value_sum = offense_value_mean + defense_value_mean
    independent_return_sum = offense_return_mean + defense_return_mean
    return {
        "value_sample_count": total_samples,
        "value_bias_mean": (
            (offense_bias * offense_samples) + (defense_bias * defense_samples)
        )
        / total_samples,
        "value_mae": ((offense_mae * offense_samples) + (defense_mae * defense_samples))
        / total_samples,
        "value_rmse": float(
            np.sqrt(
                (
                    (offense_rmse_sq * offense_samples)
                    + (defense_rmse_sq * defense_samples)
                )
                / total_samples
            )
        ),
        "value_explained_variance_mean": (
            (offense_ev * offense_samples) + (defense_ev * defense_samples)
        )
        / total_samples,
        "independent_role_value_sum_mean": independent_value_sum,
        "independent_role_value_sum_abs_mean": abs(independent_value_sum),
        "independent_role_return_sum_mean": independent_return_sum,
        "independent_role_return_sum_abs_mean": abs(independent_return_sum),
    }


_CUMULATIVE_EPISODE_USAGE_KEYS = (
    "active_step_count",
    "ppo_used_active_step_count",
    "ppo_unused_active_step_count",
    "completed_episode_count",
    "completed_active_step_count",
    "ppo_used_completed_episode_count",
    "ppo_used_completed_active_step_count",
    "ppo_unused_completed_episode_count",
    "ppo_unused_completed_active_step_count",
)


def _metric_float(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key, 0.0)
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _init_cumulative_episode_usage(
    last_metrics: dict[str, Any] | None,
) -> dict[str, float]:
    metrics = dict(last_metrics or {})
    return {
        key: _metric_float(metrics, f"cumulative_{key}")
        for key in _CUMULATIVE_EPISODE_USAGE_KEYS
    }


def _add_episode_usage_metrics(
    metrics: dict[str, Any],
    cumulative: dict[str, float],
) -> None:
    active_step_count = _metric_float(metrics, "rollout_active_step_count")
    ppo_used_active_step_count = _metric_float(metrics, "ppo_active_sample_count")
    completed_episode_count = _metric_float(metrics, "completed_episode_count")
    completed_active_step_count = _metric_float(metrics, "completed_active_step_count")
    ppo_used_completed_episode_count = sum(
        _metric_float(metrics, f"{role}_ppo_eligible_completed_episode_count")
        for role in TRAINING_ROLES
    )
    ppo_used_completed_active_step_count = sum(
        _metric_float(metrics, f"{role}_ppo_eligible_completed_active_step_count")
        for role in TRAINING_ROLES
    )
    update_values = {
        "active_step_count": active_step_count,
        "ppo_used_active_step_count": ppo_used_active_step_count,
        "ppo_unused_active_step_count": max(
            0.0,
            active_step_count - ppo_used_active_step_count,
        ),
        "completed_episode_count": completed_episode_count,
        "completed_active_step_count": completed_active_step_count,
        "ppo_used_completed_episode_count": ppo_used_completed_episode_count,
        "ppo_used_completed_active_step_count": ppo_used_completed_active_step_count,
        "ppo_unused_completed_episode_count": max(
            0.0,
            completed_episode_count - ppo_used_completed_episode_count,
        ),
        "ppo_unused_completed_active_step_count": max(
            0.0,
            completed_active_step_count - ppo_used_completed_active_step_count,
        ),
    }

    metrics.update(update_values)
    for key, value in update_values.items():
        cumulative[key] = float(cumulative.get(key, 0.0) + float(value))
        metrics[f"cumulative_{key}"] = float(cumulative[key])


def _linear_update_schedule(
    update_index: int,
    *,
    start: float,
    end: float,
    warmup_updates: int,
    ramp_updates: int,
) -> float:
    if int(update_index) < int(warmup_updates):
        return float(start)
    if int(ramp_updates) <= 0:
        return float(end)
    progress = min(
        1.0,
        max(0.0, (int(update_index) - int(warmup_updates)) / float(int(ramp_updates))),
    )
    return float(start) + ((float(end) - float(start)) * float(progress))


def _linear_position_schedule(
    position: int,
    *,
    start: float,
    end: float,
    warmup: int,
    ramp: int,
) -> float:
    if int(position) < int(warmup):
        return float(start)
    if int(ramp) <= 0:
        return float(end)
    progress = min(
        1.0,
        max(0.0, (int(position) - int(warmup)) / float(int(ramp))),
    )
    return float(start) + ((float(end) - float(start)) * float(progress))


def _multi_possession_limit_for_update(args, update_index: int) -> int:
    """Return the curriculum target for episodes beginning in this update.

    Update one uses the configured start value. Existing episodes retain the
    episode-scoped value stored in KernelState and therefore never change
    length midway through a game.
    """
    end = _multi_possession_final_limit(args)
    start = _multi_possession_start_limit(args)
    if not _multi_possession_curriculum_enabled(args):
        return end
    ramp_updates = max(
        0,
        int(getattr(args, "multi_possession_limit_ramp_updates", 0)),
    )
    if ramp_updates <= 0:
        return end
    progress = min(
        1.0,
        max(0.0, (int(update_index) - 1) / float(ramp_updates)),
    )
    interpolated = float(start) + ((float(end) - float(start)) * progress)
    return max(1, int(np.floor(interpolated + 0.5)))


def _entropy_coef_for_update(args, update_index: int) -> float:
    start_raw = getattr(args, "ent_coef_start", None)
    end_raw = getattr(args, "ent_coef_end", None)
    if start_raw is None and end_raw is None:
        return float(getattr(args, "ent_coef", 0.0))
    start = float(getattr(args, "ent_coef", 0.0) if start_raw is None else start_raw)
    end = float(start if end_raw is None else end_raw)
    decay_update = getattr(args, "entropy_decay_updates", None)
    explicit_decay_horizon = decay_update is not None
    if decay_update is None:
        decay_update = int(getattr(args, "num_updates", 1))
    total_updates = max(1, int(decay_update) - 1)
    progress = (
        1.0
        if explicit_decay_horizon and int(decay_update) == 1
        else min(
            1.0,
            max(0.0, (int(update_index) - 1) / float(total_updates)),
        )
    )
    schedule = str(getattr(args, "ent_schedule", "linear")).lower()
    if schedule == "exp":
        start_pos = max(float(start), 1.0e-12)
        end_pos = max(float(end), 1.0e-12)
        ratio = start_pos / end_pos
        return float(end_pos * (ratio ** (1.0 - progress)))
    return float(start + ((end - start) * progress))


def _task_reward_scale_for_update(args, update_index: int) -> float:
    start_raw = getattr(args, "task_reward_scale_start", None)
    end_raw = getattr(args, "task_reward_scale_end", None)
    if start_raw is None and end_raw is None:
        return 1.0
    start = 1.0 if start_raw is None else float(start_raw)
    end = start if end_raw is None else float(end_raw)
    warmup_updates = getattr(args, "task_reward_scale_warmup_updates", None)
    ramp_updates = getattr(args, "task_reward_scale_ramp_updates", None)
    if warmup_updates is not None or ramp_updates is not None:
        return _linear_position_schedule(
            int(update_index),
            start=start,
            end=end,
            warmup=int(0 if warmup_updates is None else warmup_updates),
            ramp=int(1 if ramp_updates is None else ramp_updates),
        )

    steps_per_update = (
        int(getattr(args, "kernel_batch_size"))
        * int(getattr(args, "rollout_horizon"))
        * len(TRAINING_ROLES)
    )
    completed_steps = max(0, int(update_index) - 1) * int(steps_per_update)
    return _linear_position_schedule(
        completed_steps,
        start=start,
        end=end,
        warmup=int(getattr(args, "task_reward_scale_warmup_steps", 0)),
        ramp=int(getattr(args, "task_reward_scale_ramp_steps", 1)),
    )


def _phi_beta_for_update(args, update_index: int) -> float:
    if not bool(getattr(args, "enable_phi_shaping", False)):
        return 0.0
    start_raw = getattr(args, "phi_beta_start", None)
    end_raw = getattr(args, "phi_beta_end", None)
    start = 0.0 if start_raw is None else float(start_raw)
    end = start if end_raw is None else float(end_raw)
    return _linear_update_schedule(
        update_index,
        start=start,
        end=end,
        warmup_updates=int(getattr(args, "phi_beta_warmup_updates", 0)),
        ramp_updates=int(getattr(args, "phi_beta_ramp_updates", 1)),
    )


def _static_with_phi_beta(static, phi_beta: float, jnp):
    return static._replace(
        phi_beta=jnp.asarray(float(phi_beta), dtype=jnp.float32),
    )


def _static_with_training_schedules(
    static,
    *,
    phi_beta: float,
    possession_limit: int,
    jnp,
):
    return static._replace(
        phi_beta=jnp.asarray(float(phi_beta), dtype=jnp.float32),
        multi_possession_limit=jnp.asarray(
            int(possession_limit),
            dtype=static.multi_possession_limit.dtype,
        ),
    )


def _state_with_episode_game_length(
    state, *, possession_limit: int, overtime_cap: int, jnp
):
    effective_limit = max(1, int(possession_limit))
    effective_cap = max(
        1, int(overtime_cap) if int(overtime_cap) > 0 else effective_limit
    )
    return state._replace(
        episode_possession_limit=jnp.full_like(
            state.episode_possession_limit,
            effective_limit,
        ),
        episode_overtime_round_cap=jnp.full_like(
            state.episode_overtime_round_cap,
            effective_cap,
        ),
    )


def _apply_task_reward_scale_to_rollout(rollout, scale: float, jnp):
    scale_t = jnp.asarray(float(scale), dtype=jnp.float32)
    return rollout._replace(
        trajectory=rollout.trajectory._replace(
            rewards=rollout.trajectory.rewards * scale_t,
        )
    )


def _rollout_phi_reward_component(rollout, static, task_reward_scale: float, jnp):
    role_signs = jnp.where(
        static.role_encoding.astype(jnp.float32) > 0.0,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray(-1.0, dtype=jnp.float32),
    )
    training_sign_sum = jnp.sum(
        static.training_player_mask.astype(jnp.float32) * role_signs
    )
    static_scale = jnp.asarray(static.task_reward_scale, dtype=jnp.float32)
    schedule_scale = jnp.asarray(float(task_reward_scale), dtype=jnp.float32)
    return (
        rollout.trajectory.phi_r_shape.astype(jnp.float32)
        * training_sign_sum
        * static_scale
        * schedule_scale
    )


def _build_reward_component_arrays(
    rollout, static, task_reward_scale: float, jnp
) -> dict[str, Any]:
    phi_reward = _rollout_phi_reward_component(rollout, static, task_reward_scale, jnp)
    task_reward = rollout.trajectory.rewards.astype(jnp.float32) - phi_reward
    schedule_scale = jnp.asarray(float(task_reward_scale), dtype=jnp.float32)
    static_scale = jnp.asarray(static.task_reward_scale, dtype=jnp.float32)
    return {
        "task_reward": task_reward,
        "phi_reward": phi_reward,
        "game_reward": (
            rollout.trajectory.game_rewards.astype(jnp.float32)
            * static_scale
            * schedule_scale
        ),
        "winner_reward": (
            rollout.trajectory.winner_rewards.astype(jnp.float32)
            * static_scale
            * schedule_scale
        ),
        "auxiliary_reward": (
            rollout.trajectory.auxiliary_rewards.astype(jnp.float32)
            * static_scale
            * schedule_scale
        ),
        "intent_bonus": jnp.zeros_like(task_reward, dtype=jnp.float32),
    }


def _summarize_intent_intrinsic_advantages(
    ppo_batches,
    role_rollouts,
    *,
    jax,
) -> dict[str, float]:
    """Describe intrinsic PPO credit and the boundaries that terminate it."""
    task_parts = []
    intent_parts = []
    active_parts = []
    continuation_count = 0
    boundary_counts = {
        "selector": 0,
        "possession": 0,
        "game": 0,
        "intent": 0,
        "role_or_gate": 0,
    }
    for ppo_batch, rollout in zip(ppo_batches, role_rollouts, strict=True):
        task_parts.append(
            np.asarray(jax.device_get(ppo_batch.task_advantages), dtype=np.float32)
        )
        intent_parts.append(
            np.asarray(jax.device_get(ppo_batch.intent_advantages), dtype=np.float32)
        )
        active_flat = (
            np.asarray(jax.device_get(ppo_batch.intent_active_mask), dtype=np.float32)
            > 0.5
        )
        active_parts.append(active_flat)

        trajectory = rollout.trajectory
        active = active_flat.reshape(trajectory.policy_intent_gate.shape)
        labels = np.asarray(
            jax.device_get(trajectory.policy_intent_index), dtype=np.int32
        )
        roles = np.asarray(jax.device_get(trajectory.training_role), dtype=np.float32)
        next_active = np.concatenate(
            [active[1:], np.zeros_like(active[:1], dtype=bool)],
            axis=0,
        )
        next_labels = np.concatenate([labels[1:], labels[-1:]], axis=0)
        next_roles = np.concatenate([roles[1:], roles[-1:]], axis=0)
        next_selector = np.concatenate(
            [
                np.asarray(jax.device_get(trajectory.selector_applied[1:]), dtype=bool),
                np.ones_like(
                    np.asarray(jax.device_get(trajectory.selector_applied[-1:])),
                    dtype=bool,
                ),
            ],
            axis=0,
        )
        possession_end = np.asarray(
            jax.device_get(trajectory.possession_ended),
            dtype=bool,
        )
        game_end = np.asarray(jax.device_get(trajectory.dones), dtype=bool)
        continuation = (
            active
            & next_active
            & (labels == next_labels)
            & (roles == next_roles)
            & (~next_selector)
            & (~possession_end)
            & (~game_end)
        )
        continuation_count += int(np.sum(continuation))
        boundary_counts["selector"] += int(np.sum(active & next_selector))
        boundary_counts["possession"] += int(np.sum(active & possession_end))
        boundary_counts["game"] += int(np.sum(active & game_end))
        boundary_counts["intent"] += int(np.sum(active & (labels != next_labels)))
        boundary_counts["role_or_gate"] += int(
            np.sum(active & ((~next_active) | (roles != next_roles)))
        )

    task = (
        np.concatenate(task_parts) if task_parts else np.zeros((0,), dtype=np.float32)
    )
    intent = (
        np.concatenate(intent_parts)
        if intent_parts
        else np.zeros((0,), dtype=np.float32)
    )
    active = (
        np.concatenate(active_parts) if active_parts else np.zeros((0,), dtype=bool)
    )
    active_task = task[active]
    active_intent = intent[active]
    active_count = int(active_intent.size)
    task_abs = float(np.sum(np.abs(active_task)))
    intent_abs = float(np.sum(np.abs(active_intent)))
    metrics = {
        "intent_intrinsic_advantage_active_count": float(active_count),
        "intent_intrinsic_advantage_mean": (
            float(np.mean(active_intent)) if active_count else 0.0
        ),
        "intent_intrinsic_advantage_std": (
            float(np.std(active_intent)) if active_count else 0.0
        ),
        "intent_intrinsic_advantage_abs_mean": (
            float(np.mean(np.abs(active_intent))) if active_count else 0.0
        ),
        "intent_intrinsic_advantage_magnitude_fraction": (
            intent_abs / max(task_abs + intent_abs, 1.0e-12)
        ),
        "intent_intrinsic_continuation_count": float(continuation_count),
        "intent_intrinsic_boundary_count": float(active_count - continuation_count),
        "intent_intrinsic_segment_mean_steps": (
            float(active_count) / float(max(active_count - continuation_count, 1))
        ),
    }
    for name, count in boundary_counts.items():
        metrics[f"intent_intrinsic_boundary_{name}_count"] = float(count)
    return metrics


def _summarize_intent_behavior_by_label(
    rollout,
    training_mask,
    *,
    num_intents: int,
    jax,
) -> dict[str, float]:
    """Log rollout events by latent so behavior collapse is visible directly."""
    trajectory = rollout.trajectory
    labels = np.asarray(
        jax.device_get(trajectory.policy_intent_index),
        dtype=np.int32,
    )
    active = (
        (
            np.asarray(jax.device_get(trajectory.policy_intent_gate), dtype=np.float32)
            > 0.5
        )
        & (np.asarray(jax.device_get(trajectory.training_role), dtype=np.float32) > 0.0)
        & (np.asarray(jax.device_get(training_mask), dtype=np.float32) > 0.5)
    )
    events = {
        "pass_attempt_rate": trajectory.pass_attempts,
        "completed_pass_rate": trajectory.completed_passes,
        "turnover_rate": trajectory.turnovers,
        "shot_attempt_rate": trajectory.shot_attempts,
        "shot_make_rate": trajectory.shot_makes,
        "shot_two_rate": trajectory.shot_twos,
        "shot_three_rate": trajectory.shot_threes,
    }
    event_arrays = {
        name: np.asarray(jax.device_get(value), dtype=np.float32)
        for name, value in events.items()
    }
    actions = np.asarray(jax.device_get(trajectory.actions), dtype=np.int32)
    total_active = max(int(np.sum(active)), 1)
    metrics: dict[str, float] = {}
    for intent_idx in range(int(num_intents)):
        intent_mask = active & (labels == intent_idx)
        count = int(np.sum(intent_mask))
        denom = max(count, 1)
        metrics[f"intent_behavior_active_count_by_intent/{intent_idx}"] = float(count)
        metrics[f"intent_behavior_active_share_by_intent/{intent_idx}"] = float(
            count
        ) / float(total_active)
        for name, values in event_arrays.items():
            metrics[f"intent_behavior_{name}_by_intent/{intent_idx}"] = float(
                np.sum(values * intent_mask)
            ) / float(denom)
        selected_actions = actions[intent_mask]
        action_count = max(int(selected_actions.size), 1)
        for action_idx in range(int(trajectory.action_mask.shape[-1])):
            metrics[
                f"intent_behavior_action_prob_by_intent/{intent_idx}/action_{action_idx}"
            ] = float(np.sum(selected_actions == action_idx)) / float(action_count)
    return metrics


def _opponent_deterministic_episode_prob_for_update(args, update_index: int) -> float:
    start_raw = getattr(args, "opponent_deterministic_episode_prob_start", None)
    end_raw = getattr(args, "opponent_deterministic_episode_prob_end", None)
    base = float(getattr(args, "opponent_deterministic_episode_prob", 0.0))
    if start_raw is None and end_raw is None:
        return base
    start = base if start_raw is None else float(start_raw)
    end = start if end_raw is None else float(end_raw)
    return _linear_update_schedule(
        update_index,
        start=start,
        end=end,
        warmup_updates=int(
            getattr(args, "opponent_deterministic_episode_prob_warmup_updates", 0)
        ),
        ramp_updates=int(
            getattr(args, "opponent_deterministic_episode_prob_ramp_updates", 1)
        ),
    )


def _opponent_deterministic_episode_prob_is_scheduled(args) -> bool:
    return (
        getattr(args, "opponent_deterministic_episode_prob_start", None) is not None
        or getattr(args, "opponent_deterministic_episode_prob_end", None) is not None
    )


def _selector_schedules_for_update(args, update_index: int) -> tuple[float, float]:
    if not bool(getattr(args, "intent_selector_enabled", False)):
        return 0.0, 0.0
    alpha = _linear_update_schedule(
        update_index,
        start=float(getattr(args, "intent_selector_alpha_start", 0.0)),
        end=float(getattr(args, "intent_selector_alpha_end", 1.0)),
        warmup_updates=int(getattr(args, "intent_selector_alpha_warmup_updates", 0)),
        ramp_updates=int(getattr(args, "intent_selector_alpha_ramp_updates", 1)),
    )
    eps = _linear_update_schedule(
        update_index,
        start=float(getattr(args, "intent_selector_eps_start", 0.0)),
        end=float(getattr(args, "intent_selector_eps_end", 0.0)),
        warmup_updates=int(getattr(args, "intent_selector_eps_warmup_updates", 0)),
        ramp_updates=int(getattr(args, "intent_selector_eps_ramp_updates", 1)),
    )
    return alpha, eps


def summarize_selector_metrics(
    rollout, *, num_intents: int, alpha: float, eps: float
) -> dict[str, Any]:
    selector_used = np.asarray(rollout.trajectory.selector_used, dtype=bool)
    selector_applied = np.asarray(rollout.trajectory.selector_applied, dtype=bool)
    selector_fallback_used = np.asarray(
        rollout.trajectory.selector_fallback_used, dtype=bool
    )
    selector_boundary_episode_start = np.asarray(
        rollout.trajectory.selector_boundary_episode_start,
        dtype=bool,
    )
    selector_boundary_possession_start = np.asarray(
        rollout.trajectory.selector_boundary_possession_start,
        dtype=bool,
    )
    selector_boundary_commitment_timeout = np.asarray(
        rollout.trajectory.selector_boundary_commitment_timeout,
        dtype=bool,
    )
    selector_boundary_completed_pass = np.asarray(
        rollout.trajectory.selector_boundary_completed_pass,
        dtype=bool,
    )
    selector_boundary_offensive_rebound = np.asarray(
        rollout.trajectory.selector_boundary_offensive_rebound,
        dtype=bool,
    )
    selector_intent_index = np.asarray(
        rollout.trajectory.selector_intent_index, dtype=np.int32
    )
    selector_entropy = np.asarray(rollout.trajectory.selector_entropy, dtype=np.float32)
    selector_max_prob = np.asarray(
        rollout.trajectory.selector_max_prob, dtype=np.float32
    )
    selector_value = np.asarray(rollout.trajectory.selector_value, dtype=np.float32)
    selector_log_prob = np.asarray(
        rollout.trajectory.selector_old_log_prob, dtype=np.float32
    )
    active_mask = np.asarray(rollout.trajectory.active_mask, dtype=np.float32) > 0.5
    dones = np.asarray(rollout.trajectory.dones, dtype=bool)
    used_count = int(selector_used.sum())
    total_steps = int(selector_used.size)
    metrics: dict[str, Any] = {
        "selector_alpha": float(alpha),
        "selector_eps": float(eps),
        "selector_used_count": int(used_count),
        "selector_usage_rate": _safe_metric_ratio(used_count, total_steps),
        "selector_applied_count": int(selector_applied.sum()),
        "selector_applied_rate": _safe_metric_ratio(
            int(selector_applied.sum()), total_steps
        ),
        "selector_fallback_count": int(selector_fallback_used.sum()),
        "selector_fallback_rate": _safe_metric_ratio(
            int(selector_fallback_used.sum()), total_steps
        ),
        "selector_boundary_episode_start_count": int(
            selector_boundary_episode_start.sum()
        ),
        "selector_boundary_possession_start_count": int(
            selector_boundary_possession_start.sum()
        ),
        "selector_boundary_possession_start_rate": _safe_metric_ratio(
            int(selector_boundary_possession_start.sum()),
            int(selector_applied.sum()),
        ),
        "selector_boundary_commitment_timeout_count": int(
            selector_boundary_commitment_timeout.sum()
        ),
        "selector_boundary_commitment_timeout_rate": _safe_metric_ratio(
            int(selector_boundary_commitment_timeout.sum()),
            int(selector_applied.sum()),
        ),
        "selector_boundary_completed_pass_count": int(
            selector_boundary_completed_pass.sum()
        ),
        "selector_boundary_completed_pass_rate": _safe_metric_ratio(
            int(selector_boundary_completed_pass.sum()),
            int(selector_applied.sum()),
        ),
        "selector_boundary_offensive_rebound_count": int(
            selector_boundary_offensive_rebound.sum()
        ),
        "selector_boundary_offensive_rebound_rate": _safe_metric_ratio(
            int(selector_boundary_offensive_rebound.sum()),
            int(selector_applied.sum()),
        ),
    }
    if used_count > 0:
        used_entropy = selector_entropy[selector_used]
        used_max_prob = selector_max_prob[selector_used]
        used_value = selector_value[selector_used]
        used_log_prob = selector_log_prob[selector_used]
        metrics.update(
            {
                "selector_entropy": float(np.mean(used_entropy)),
                "selector_max_prob": float(np.mean(used_max_prob)),
                "selector_value_mean": float(np.mean(used_value)),
                "selector_old_log_prob_mean": float(np.mean(used_log_prob)),
            }
        )
        selected = selector_intent_index[selector_used]
        for intent_idx in range(int(num_intents)):
            metrics[f"selector_usage_by_intent/{intent_idx}"] = float(
                np.mean(selected == int(intent_idx))
            )
    else:
        metrics.update(
            {
                "selector_entropy": 0.0,
                "selector_max_prob": 0.0,
                "selector_value_mean": 0.0,
                "selector_old_log_prob_mean": 0.0,
            }
        )
        for intent_idx in range(int(num_intents)):
            metrics[f"selector_usage_by_intent/{intent_idx}"] = 0.0

    segment_step_sums = np.zeros((int(num_intents),), dtype=np.float64)
    segment_counts = np.zeros((int(num_intents),), dtype=np.int64)
    segment_lengths: list[int] = []
    if selector_used.ndim == 2:
        horizon, batch_size = selector_used.shape
        for env_idx in range(batch_size):
            current_intent: int | None = None
            current_length = 0
            for step_idx in range(horizon):
                if not bool(active_mask[step_idx, env_idx]):
                    if current_intent is not None:
                        segment_step_sums[current_intent] += float(current_length)
                        segment_counts[current_intent] += 1
                        segment_lengths.append(int(current_length))
                        current_intent = None
                        current_length = 0
                    continue

                if bool(selector_used[step_idx, env_idx]):
                    if current_intent is not None:
                        segment_step_sums[current_intent] += float(current_length)
                        segment_counts[current_intent] += 1
                        segment_lengths.append(int(current_length))
                    selected_intent = int(selector_intent_index[step_idx, env_idx])
                    current_intent = (
                        selected_intent
                        if 0 <= selected_intent < int(num_intents)
                        else None
                    )
                    current_length = 0

                if current_intent is not None:
                    current_length += 1

                if bool(dones[step_idx, env_idx]) and current_intent is not None:
                    segment_step_sums[current_intent] += float(current_length)
                    segment_counts[current_intent] += 1
                    segment_lengths.append(int(current_length))
                    current_intent = None
                    current_length = 0

            if current_intent is not None:
                segment_step_sums[current_intent] += float(current_length)
                segment_counts[current_intent] += 1
                segment_lengths.append(int(current_length))

    metrics["selector_segment_count"] = int(segment_counts.sum())
    metrics["selector_segment_mean_steps"] = (
        float(np.mean(np.asarray(segment_lengths, dtype=np.float32)))
        if segment_lengths
        else 0.0
    )
    for intent_idx in range(int(num_intents)):
        count = int(segment_counts[intent_idx])
        metrics[f"selector_segment_count_by_intent/{intent_idx}"] = count
        metrics[f"selector_segment_mean_steps_by_intent/{intent_idx}"] = (
            float(segment_step_sums[intent_idx] / float(count)) if count > 0 else 0.0
        )
    return metrics


def _summarize_role_rollout_metrics(
    role: str,
    rollout,
    *,
    num_intents: int,
    trainer_config: TrainerConfig | None = None,
    jax=None,
    jnp=None,
) -> dict[str, Any]:
    rewards = np.asarray(rollout.trajectory.rewards, dtype=np.float32)
    dones = np.asarray(rollout.trajectory.dones, dtype=np.float32)
    terminal_steps = np.asarray(
        rollout.trajectory.terminal_episode_steps, dtype=np.int32
    )
    offense_score_delta = np.asarray(
        rollout.trajectory.offense_score_delta, dtype=np.float32
    )
    defense_score_delta = np.asarray(
        rollout.trajectory.defense_score_delta, dtype=np.float32
    )
    active_mask = np.asarray(rollout.trajectory.active_mask, dtype=np.float32)
    active_bool = active_mask > 0.5
    training_role = np.asarray(rollout.trajectory.training_role, dtype=np.float32)
    episode_start = np.asarray(rollout.trajectory.episode_start, dtype=np.float32)
    active_count = float(active_mask.sum())
    total_count = int(active_mask.size)

    def _active_sum(values: np.ndarray) -> float:
        return float((np.asarray(values, dtype=np.float32) * active_mask).sum())

    def _active_mean(values: np.ndarray) -> float:
        if active_count <= 0.0:
            return 0.0
        return float(
            (np.asarray(values, dtype=np.float32) * active_mask).sum() / active_count
        )

    terminal_mask = (terminal_steps > 0) & active_bool
    completed_episodes = int(terminal_mask.sum())
    completed_episode_steps = int(
        (terminal_steps * terminal_mask.astype(np.int32)).sum()
    )
    learner_reward_total = _active_sum(rewards)
    learner_reward_mean = _active_mean(rewards)
    opponent_reward_total = -learner_reward_total
    opponent_reward_mean = -learner_reward_mean
    offense_points_total = _active_sum(offense_score_delta)
    defense_points_total = _active_sum(defense_score_delta)
    rebound_reward_advances = np.asarray(
        rollout.trajectory.rebound_reward_advances, dtype=np.float32
    )
    rebound_reward_settlements = np.asarray(
        rollout.trajectory.rebound_reward_settlements, dtype=np.float32
    )
    rebound_reward_advance_total = _active_sum(rebound_reward_advances)
    rebound_reward_settlement_total = _active_sum(rebound_reward_settlements)
    if role == "offense":
        learner_points_total = offense_points_total
        opponent_points_total = defense_points_total
    else:
        learner_points_total = defense_points_total
        opponent_points_total = offense_points_total

    metrics = {
        f"{role}_mean_reward": learner_reward_mean,
        f"{role}_learner_mean_reward": learner_reward_mean,
        f"{role}_opponent_mean_reward": opponent_reward_mean,
        f"{role}_learner_reward_total": learner_reward_total,
        f"{role}_opponent_reward_total": opponent_reward_total,
        f"{role}_game_reward_mean": _active_mean(rollout.trajectory.game_rewards),
        f"{role}_game_reward_total": _active_sum(rollout.trajectory.game_rewards),
        f"{role}_winner_reward_mean": _active_mean(rollout.trajectory.winner_rewards),
        f"{role}_winner_reward_total": _active_sum(rollout.trajectory.winner_rewards),
        f"{role}_auxiliary_reward_mean": _active_mean(
            rollout.trajectory.auxiliary_rewards
        ),
        f"{role}_auxiliary_reward_total": _active_sum(
            rollout.trajectory.auxiliary_rewards
        ),
        f"{role}_team_a_score_delta_total": _active_sum(
            rollout.trajectory.team_a_score_delta
        ),
        f"{role}_team_b_score_delta_total": _active_sum(
            rollout.trajectory.team_b_score_delta
        ),
        f"{role}_done_rate": _active_mean(dones),
        f"{role}_active_step_count": int(active_count),
        f"{role}_active_step_fraction": _safe_metric_ratio(active_count, total_count),
        f"{role}_learner_offense_step_fraction": _active_mean(training_role > 0.0),
        f"{role}_learner_defense_step_fraction": _active_mean(training_role < 0.0),
        f"{role}_starting_offense_count": _active_sum(
            (episode_start > 0.5) & (training_role > 0.0)
        ),
        f"{role}_starting_defense_count": _active_sum(
            (episode_start > 0.5) & (training_role < 0.0)
        ),
        f"{role}_completed_episodes": int(completed_episodes),
        f"{role}_completed_episode_count": int(completed_episodes),
        f"{role}_completed_active_step_count": int(completed_episode_steps),
        f"{role}_offense_points_total": offense_points_total,
        f"{role}_defense_points_total": defense_points_total,
        f"{role}_offense_points_per_completed_episode": _safe_metric_ratio(
            offense_points_total,
            completed_episodes,
        ),
        f"{role}_defense_points_per_completed_episode": _safe_metric_ratio(
            defense_points_total,
            completed_episodes,
        ),
        f"{role}_learner_points_total": learner_points_total,
        f"{role}_opponent_points_total": opponent_points_total,
        f"{role}_rebound_reward_advance_count": int(
            _active_sum(rebound_reward_advances > 0.0)
        ),
        f"{role}_rebound_reward_advance_total": rebound_reward_advance_total,
        f"{role}_rebound_reward_settlement_total": rebound_reward_settlement_total,
        f"{role}_rebound_reward_net_total": (
            rebound_reward_advance_total + rebound_reward_settlement_total
        ),
        f"{role}_learner_points_per_completed_episode": _safe_metric_ratio(
            learner_points_total,
            completed_episodes,
        ),
        f"{role}_opponent_points_per_completed_episode": _safe_metric_ratio(
            opponent_points_total,
            completed_episodes,
        ),
        f"{role}_phi_r_shape_mean": _active_mean(rollout.trajectory.phi_r_shape),
        f"{role}_phi_r_shape_abs_mean": _active_mean(
            np.abs(np.asarray(rollout.trajectory.phi_r_shape, dtype=np.float32))
        ),
        f"{role}_phi_prev_mean": _active_mean(rollout.trajectory.phi_prev),
        f"{role}_phi_next_mean": _active_mean(rollout.trajectory.phi_next),
        f"{role}_phi_beta_mean": _active_mean(rollout.trajectory.phi_beta),
    }
    metrics.update(
        summarize_shot_type_metrics(
            f"{role}_all",
            shot_attempts=rollout.trajectory.shot_attempts,
            shot_makes=rollout.trajectory.shot_makes,
            shot_dunks=rollout.trajectory.shot_dunks,
            shot_twos=rollout.trajectory.shot_twos,
            shot_threes=rollout.trajectory.shot_threes,
        )
    )
    include_learner_shot_metrics = role != "defense"
    include_opponent_shot_metrics = role != "offense"
    if include_learner_shot_metrics:
        metrics.update(
            summarize_shot_type_metrics(
                f"{role}_learner",
                shot_attempts=rollout.trajectory.learner_shot_attempts,
                shot_makes=rollout.trajectory.learner_shot_makes,
                shot_dunks=rollout.trajectory.learner_shot_dunks,
                shot_twos=rollout.trajectory.learner_shot_twos,
                shot_threes=rollout.trajectory.learner_shot_threes,
            )
        )
    if include_opponent_shot_metrics:
        metrics.update(
            summarize_shot_type_metrics(
                f"{role}_opponent",
                shot_attempts=rollout.trajectory.opponent_shot_attempts,
                shot_makes=rollout.trajectory.opponent_shot_makes,
                shot_dunks=rollout.trajectory.opponent_shot_dunks,
                shot_twos=rollout.trajectory.opponent_shot_twos,
                shot_threes=rollout.trajectory.opponent_shot_threes,
            )
        )
    metrics.update(
        summarize_intent_metrics(
            f"{role}_offense",
            intent_index=rollout.trajectory.intent_index,
            intent_active=rollout.trajectory.intent_active,
            intent_age=rollout.trajectory.intent_age,
            intent_commitment_remaining=rollout.trajectory.intent_commitment_remaining,
            intent_visible_to_defense=rollout.trajectory.intent_visible_to_defense,
        )
    )
    metrics.update(
        summarize_intent_metrics(
            f"{role}_defense",
            intent_index=rollout.trajectory.defense_intent_index,
            intent_active=rollout.trajectory.defense_intent_active,
            intent_age=rollout.trajectory.defense_intent_age,
            intent_commitment_remaining=rollout.trajectory.defense_intent_commitment_remaining,
        )
    )
    metrics.update(
        summarize_reward_by_intent_metrics(
            f"{role}_intent",
            rollout.trajectory,
            num_intents=num_intents,
        )
    )
    if trainer_config is not None and jax is not None and jnp is not None:
        training_mask, _, _ = build_trajectory_training_masks(
            rollout.trajectory,
            trainer_config,
            jax,
            jnp,
        )
        # A multi-possession learner controls one fixed team through both
        # offensive and defensive possessions.  Keep both phases in PPO, but
        # do not let the role-labelled diagnostics mix them: an "offense"
        # reward must contain only the learner's live offensive timesteps and
        # a "defense" reward only its live defensive timesteps.
        role_value = 1.0 if role == "offense" else -1.0
        metric_role_mask = (
            np.asarray(rollout.trajectory.training_role, dtype=np.float32) == role_value
        ).astype(np.float32)
        role_ppo_batch = build_ppo_batch(rollout, trainer_config, jax, jnp)
        values_shape = np.asarray(rollout.trajectory.values).shape
        value_diag = _masked_value_diagnostics(
            rollout.trajectory.values,
            np.asarray(role_ppo_batch.returns, dtype=np.float32).reshape(values_shape),
            training_mask,
        )
        metrics.update(
            {
                f"{role}_value_sample_count": value_diag["sample_count"],
                f"{role}_value_mean": value_diag["value_mean"],
                f"{role}_return_mean": value_diag["return_mean"],
                f"{role}_value_bias_mean": value_diag["value_bias_mean"],
                f"{role}_value_mae": value_diag["value_mae"],
                f"{role}_value_rmse": value_diag["value_rmse"],
                f"{role}_value_explained_variance": value_diag["explained_variance"],
            }
        )
        metrics.update(
            summarize_ppo_eligible_episode_metrics(
                f"{role}_ppo_eligible",
                rollout.trajectory,
                training_mask,
                metric_mask=metric_role_mask,
                completion_mask=training_mask,
                include_learner_shots=include_learner_shot_metrics,
                include_opponent_shots=include_opponent_shot_metrics,
                role=role,
            )
        )
        metrics.update(
            summarize_reward_by_intent_metrics(
                f"{role}_ppo_eligible_intent",
                rollout.trajectory,
                num_intents=num_intents,
                training_mask=training_mask,
            )
        )
    return metrics


def _print_checkpoint_summary(
    *,
    update_index: int,
    last_metrics: dict[str, Any] | None,
    latest_checkpoint_path: str | None,
    latest_checkpoint_artifact_path: str | None,
) -> None:
    metrics = dict(last_metrics or {})
    rows = [
        ("update_index", int(update_index)),
        ("multi_possession_limit_active", metrics.get("multi_possession_limit_active")),
        ("episode_possession_limit_mean", metrics.get("episode_possession_limit_mean")),
        (
            "episode_overtime_round_cap_mean",
            metrics.get("episode_overtime_round_cap_mean"),
        ),
        ("steps_per_update", metrics.get("steps_per_update")),
        ("rollout_active_step_fraction", metrics.get("rollout_active_step_fraction")),
        ("ppo_active_sample_fraction", metrics.get("ppo_active_sample_fraction")),
        ("ppo_loss_weight_sum", metrics.get("ppo_loss_weight_sum")),
        ("end_to_end_steps_per_sec", metrics.get("end_to_end_steps_per_sec")),
        (
            "active_end_to_end_steps_per_sec",
            metrics.get("active_end_to_end_steps_per_sec"),
        ),
        ("rollout_states_per_sec", metrics.get("rollout_states_per_sec")),
        (
            "ppo_update_rollout_samples_per_sec",
            metrics.get("ppo_update_rollout_samples_per_sec"),
        ),
        (
            "ppo_update_optimizer_samples_per_sec",
            metrics.get("ppo_update_optimizer_samples_per_sec"),
        ),
        ("end_to_end_latency_ms", metrics.get("end_to_end_latency_ms")),
        ("rollout_latency_ms", metrics.get("rollout_latency_ms")),
        ("update_latency_ms", metrics.get("update_latency_ms")),
        ("rollout_time_pct", metrics.get("rollout_time_pct")),
        ("ppo_update_time_pct", metrics.get("ppo_update_time_pct")),
        ("ppo_update_epochs", metrics.get("ppo_update_epochs")),
        ("ppo_update_minibatches", metrics.get("ppo_update_minibatches")),
        ("ppo_update_minibatch_size", metrics.get("ppo_update_minibatch_size")),
        ("phi_beta", metrics.get("phi_beta")),
        ("phi_r_shape_mean", metrics.get("phi_r_shape_mean")),
        ("phi_r_shape_abs_mean", metrics.get("phi_r_shape_abs_mean")),
        ("offense_phi_r_shape_mean", metrics.get("offense_phi_r_shape_mean")),
        ("defense_phi_r_shape_mean", metrics.get("defense_phi_r_shape_mean")),
        ("completed_episodes", metrics.get("completed_episodes")),
        ("mean_completed_episode_length", metrics.get("mean_completed_episode_length")),
        ("cumulative_active_step_count", metrics.get("cumulative_active_step_count")),
        (
            "cumulative_ppo_used_active_step_count",
            metrics.get("cumulative_ppo_used_active_step_count"),
        ),
        (
            "cumulative_ppo_unused_active_step_count",
            metrics.get("cumulative_ppo_unused_active_step_count"),
        ),
        (
            "cumulative_completed_episode_count",
            metrics.get("cumulative_completed_episode_count"),
        ),
        (
            "cumulative_completed_active_step_count",
            metrics.get("cumulative_completed_active_step_count"),
        ),
        (
            "cumulative_ppo_used_completed_episode_count",
            metrics.get("cumulative_ppo_used_completed_episode_count"),
        ),
        (
            "cumulative_ppo_unused_completed_episode_count",
            metrics.get("cumulative_ppo_unused_completed_episode_count"),
        ),
        (
            "cumulative_ppo_used_completed_active_step_count",
            metrics.get("cumulative_ppo_used_completed_active_step_count"),
        ),
        (
            "cumulative_ppo_unused_completed_active_step_count",
            metrics.get("cumulative_ppo_unused_completed_active_step_count"),
        ),
        (
            "offense_ppo_eligible_completed_episodes",
            metrics.get("offense_ppo_eligible_completed_episodes"),
        ),
        (
            "offense_ppo_eligible_mean_completed_episode_length",
            metrics.get("offense_ppo_eligible_mean_completed_episode_length"),
        ),
        (
            "offense_ppo_eligible_reward_per_completed_episode",
            metrics.get("offense_ppo_eligible_reward_per_completed_episode"),
        ),
        (
            "offense_ppo_eligible_reward_per_step",
            metrics.get("offense_ppo_eligible_reward_per_step"),
        ),
        (
            "offense_ppo_eligible_task_reward_per_completed_episode",
            metrics.get("offense_ppo_eligible_task_reward_per_completed_episode"),
        ),
        (
            "offense_ppo_eligible_phi_reward_per_completed_episode",
            metrics.get("offense_ppo_eligible_phi_reward_per_completed_episode"),
        ),
        (
            "offense_ppo_eligible_intent_bonus_per_completed_episode",
            metrics.get("offense_ppo_eligible_intent_bonus_per_completed_episode"),
        ),
        (
            "offense_ppo_eligible_intent_bonus_abs_share_of_reward",
            metrics.get("offense_ppo_eligible_intent_bonus_abs_share_of_reward"),
        ),
        (
            "offense_ppo_eligible_terminal_shot_share",
            metrics.get("offense_ppo_eligible_terminal_shot_share"),
        ),
        (
            "offense_ppo_eligible_terminal_turnover_share",
            metrics.get("offense_ppo_eligible_terminal_turnover_share"),
        ),
        (
            "defense_ppo_eligible_completed_episodes",
            metrics.get("defense_ppo_eligible_completed_episodes"),
        ),
        (
            "defense_ppo_eligible_mean_completed_episode_length",
            metrics.get("defense_ppo_eligible_mean_completed_episode_length"),
        ),
        (
            "defense_ppo_eligible_reward_per_completed_episode",
            metrics.get("defense_ppo_eligible_reward_per_completed_episode"),
        ),
        (
            "defense_ppo_eligible_reward_per_step",
            metrics.get("defense_ppo_eligible_reward_per_step"),
        ),
        (
            "mean_pass_attempts_per_completed_episode",
            metrics.get("mean_pass_attempts_per_completed_episode"),
        ),
        (
            "mean_completed_passes_per_completed_episode",
            metrics.get("mean_completed_passes_per_completed_episode"),
        ),
        (
            "mean_assists_per_completed_episode",
            metrics.get("mean_assists_per_completed_episode"),
        ),
        (
            "mean_turnovers_per_completed_episode",
            metrics.get("mean_turnovers_per_completed_episode"),
        ),
        (
            "mean_learner_turnovers_per_completed_episode",
            metrics.get("mean_learner_turnovers_per_completed_episode"),
        ),
        (
            "mean_opponent_turnovers_per_completed_episode",
            metrics.get("mean_opponent_turnovers_per_completed_episode"),
        ),
        (
            "mean_turnovers_reason_intercepted_per_completed_episode",
            metrics.get("mean_turnovers_reason_intercepted_per_completed_episode"),
        ),
        (
            "mean_turnovers_reason_defender_pressure_per_completed_episode",
            metrics.get(
                "mean_turnovers_reason_defender_pressure_per_completed_episode"
            ),
        ),
        (
            "mean_turnovers_reason_move_out_of_bounds_per_completed_episode",
            metrics.get(
                "mean_turnovers_reason_move_out_of_bounds_per_completed_episode"
            ),
        ),
        (
            "mean_turnovers_reason_shot_clock_per_completed_episode",
            metrics.get("mean_turnovers_reason_shot_clock_per_completed_episode"),
        ),
        (
            "mean_3_second_violations_per_completed_episode",
            metrics.get("mean_3_second_violations_per_completed_episode"),
        ),
        (
            "three_second_violation_rate_per_step",
            metrics.get("three_second_violation_rate_per_step"),
        ),
        (
            "mean_defensive_lane_violations_per_completed_episode",
            metrics.get("mean_defensive_lane_violations_per_completed_episode"),
        ),
        ("learner_shot_dunk_share", metrics.get("learner_shot_dunk_share")),
        ("learner_shot_two_share", metrics.get("learner_shot_two_share")),
        ("learner_shot_three_share", metrics.get("learner_shot_three_share")),
        ("opponent_shot_dunk_share", metrics.get("opponent_shot_dunk_share")),
        ("opponent_shot_two_share", metrics.get("opponent_shot_two_share")),
        ("opponent_shot_three_share", metrics.get("opponent_shot_three_share")),
        ("offense_intent_active_rate", metrics.get("offense_intent_active_rate")),
        ("defense_intent_active_rate", metrics.get("defense_intent_active_rate")),
        ("total_offensive_three_seconds", metrics.get("total_offensive_three_seconds")),
        (
            "total_defensive_lane_violations",
            metrics.get("total_defensive_lane_violations"),
        ),
        ("approx_kl", metrics.get("approx_kl")),
        ("clip_fraction", metrics.get("clip_fraction")),
        ("mean_abs_log_ratio", metrics.get("mean_abs_log_ratio")),
        ("max_abs_log_ratio", metrics.get("max_abs_log_ratio")),
        ("entropy_coef", metrics.get("entropy_coef")),
        ("entropy_bonus", metrics.get("entropy_bonus")),
        ("policy_loss", metrics.get("policy_loss")),
        ("value_loss", metrics.get("value_loss")),
        ("value_sample_count", metrics.get("value_sample_count")),
        ("offense_value_bias_mean", metrics.get("offense_value_bias_mean")),
        ("defense_value_bias_mean", metrics.get("defense_value_bias_mean")),
        ("value_bias_mean", metrics.get("value_bias_mean")),
        ("offense_value_mae", metrics.get("offense_value_mae")),
        ("defense_value_mae", metrics.get("defense_value_mae")),
        (
            "offense_value_explained_variance",
            metrics.get("offense_value_explained_variance"),
        ),
        (
            "defense_value_explained_variance",
            metrics.get("defense_value_explained_variance"),
        ),
        ("value_explained_variance_mean", metrics.get("value_explained_variance_mean")),
        (
            "independent_role_value_sum_mean",
            metrics.get("independent_role_value_sum_mean"),
        ),
        (
            "independent_role_value_sum_abs_mean",
            metrics.get("independent_role_value_sum_abs_mean"),
        ),
        (
            "independent_role_return_sum_mean",
            metrics.get("independent_role_return_sum_mean"),
        ),
        (
            "independent_role_return_sum_abs_mean",
            metrics.get("independent_role_return_sum_abs_mean"),
        ),
        ("total_loss", metrics.get("total_loss")),
        ("grad_norm", metrics.get("grad_norm")),
        ("mean_reward", metrics.get("mean_reward")),
        ("offense_learner_mean_reward", metrics.get("offense_learner_mean_reward")),
        ("defense_learner_mean_reward", metrics.get("defense_learner_mean_reward")),
        ("offense_opponent_mean_reward", metrics.get("offense_opponent_mean_reward")),
        ("defense_opponent_mean_reward", metrics.get("defense_opponent_mean_reward")),
        (
            "offense_learner_points_per_completed_episode",
            metrics.get("offense_learner_points_per_completed_episode"),
        ),
        (
            "defense_opponent_points_per_completed_episode",
            metrics.get("defense_opponent_points_per_completed_episode"),
        ),
        ("mean_return", metrics.get("mean_return")),
        ("done_rate", metrics.get("done_rate")),
        ("opponent_update_index", metrics.get("opponent_update_index")),
        ("opponent_source", metrics.get("opponent_source")),
        ("opponent_group_count", metrics.get("opponent_group_count")),
        ("opponent_unique_update_count", metrics.get("opponent_unique_update_count")),
        (
            "opponent_deterministic_episode_prob",
            metrics.get("opponent_deterministic_episode_prob"),
        ),
        (
            "opponent_deterministic_episode_rate",
            metrics.get("opponent_deterministic_episode_rate"),
        ),
        ("task_reward_scale", metrics.get("task_reward_scale")),
        ("intent_disc_active_count", metrics.get("intent_disc_active_count")),
        ("intent_disc_loss", metrics.get("intent_disc_loss")),
        (
            "intent_disc_top1_acc_trainbatch",
            metrics.get("intent_disc_top1_acc_trainbatch"),
        ),
        ("intent_disc_top1_acc_holdout", metrics.get("intent_disc_top1_acc_holdout")),
        (
            "intent_disc_auc_ovr_macro_trainbatch",
            metrics.get("intent_disc_auc_ovr_macro_trainbatch"),
        ),
        (
            "intent_disc_auc_ovr_macro_holdout",
            metrics.get("intent_disc_auc_ovr_macro_holdout"),
        ),
        ("intent_bonus_beta", metrics.get("intent_bonus_beta")),
        ("intent_bonus_raw_mean", metrics.get("intent_bonus_raw_mean")),
        (
            "intent_bonus_shaping_per_step_mean",
            metrics.get("intent_bonus_shaping_per_step_mean"),
        ),
        ("selector_alpha", metrics.get("selector_alpha")),
        ("selector_eps", metrics.get("selector_eps")),
        ("selector_used_count", metrics.get("selector_used_count")),
        ("selector_usage_rate", metrics.get("selector_usage_rate")),
        ("selector_applied_count", metrics.get("selector_applied_count")),
        ("selector_fallback_count", metrics.get("selector_fallback_count")),
        (
            "selector_boundary_possession_start_count",
            metrics.get("selector_boundary_possession_start_count"),
        ),
        (
            "selector_boundary_commitment_timeout_count",
            metrics.get("selector_boundary_commitment_timeout_count"),
        ),
        (
            "selector_boundary_completed_pass_count",
            metrics.get("selector_boundary_completed_pass_count"),
        ),
        (
            "selector_boundary_offensive_rebound_count",
            metrics.get("selector_boundary_offensive_rebound_count"),
        ),
        ("selector_entropy", metrics.get("selector_entropy")),
        ("selector_max_prob", metrics.get("selector_max_prob")),
        ("selector_train_sample_count", metrics.get("selector_train_sample_count")),
        (
            "selector_train_pending_rollout_count",
            metrics.get("selector_train_pending_rollout_count"),
        ),
        ("selector_train_loss", metrics.get("selector_train_loss")),
        ("selector_train_approx_kl", metrics.get("selector_train_approx_kl")),
        ("selector_train_clip_fraction", metrics.get("selector_train_clip_fraction")),
        ("selector_train_grad_norm", metrics.get("selector_train_grad_norm")),
        ("checkpoint_path", latest_checkpoint_path),
        ("checkpoint_artifact", latest_checkpoint_artifact_path),
    ]
    field_width = max(len(field) for field, _ in rows)
    print("\nJAX trainer checkpoint summary")
    print(f"{'metric':<{field_width}}  value")
    print(f"{'-' * field_width}  {'-' * 40}")
    for field, value in rows:
        print(f"{field:<{field_width}}  {_format_summary_value(value)}")


def run_training_loop(args) -> dict[str, Any]:
    validate_train_args(args)
    historical_milestone_updates = _historical_eval_updates(args)
    jax, jnp = ensure_jax_available("basketworld_jax/train/main.py")
    final_possession_limit = _multi_possession_final_limit(args)
    initial_possession_limit = _multi_possession_limit_for_update(args, 1)
    role_args = {role: _args_for_training_role(args, role) for role in TRAINING_ROLES}
    for role in TRAINING_ROLES:
        role_args[role].multi_possession_limit = final_possession_limit
    statics = {
        role: sample_state_batch(role_args[role], xp=jnp)[0] for role in TRAINING_ROLES
    }
    initial_training_statics = {
        role: statics[role]._replace(
            multi_possession_limit=jnp.asarray(
                initial_possession_limit,
                dtype=statics[role].multi_possession_limit.dtype,
            )
        )
        for role in TRAINING_ROLES
    }
    static = statics["offense"]
    base_key = jax.random.PRNGKey(int(args.policy_seed))
    reset_seed_key, eval_reset_seed_key, base_key = jax.random.split(base_key, 3)
    role_reset_keys = jax.random.split(reset_seed_key, len(TRAINING_ROLES))
    role_eval_reset_keys = jax.random.split(eval_reset_seed_key, len(TRAINING_ROLES))
    current_states = {}
    eval_initial_states = {}
    for role, reset_key, eval_key in zip(
        TRAINING_ROLES, role_reset_keys, role_eval_reset_keys, strict=True
    ):
        initial_reset_keys = jax.random.split(reset_key, int(args.kernel_batch_size))
        current_states[role] = reset_batch_minimal(
            initial_training_statics[role],
            initial_reset_keys,
            jax,
            jnp,
        )
        eval_reset_keys = jax.random.split(eval_key, int(args.kernel_batch_size))
        eval_initial_states[role] = reset_batch_minimal(
            statics[role], eval_reset_keys, jax, jnp
        )

    training_player_ids_by_role = {
        role: training_player_ids_from_static(statics[role]) for role in TRAINING_ROLES
    }
    training_player_ids = training_player_ids_by_role["offense"]
    training_player_ids_jnp = jnp.asarray(training_player_ids, dtype=jnp.int32)
    flat_obs = build_policy_observation_batch(
        static,
        current_states["offense"],
        jnp,
        model_type=_policy_model_type(args),
        rebound_win_prob_features=bool(
            getattr(args, "rebound_win_prob_features", False)
        ),
        rebound_target_observation_features=bool(
            getattr(args, "rebound_target_observation_features", True)
        ),
        multi_possession_features=bool(getattr(args, "enable_multi_possession", False)),
    )
    action_masks = build_action_masks_batch(static, current_states["offense"], jnp)[
        :, training_player_ids_jnp, :
    ]
    flat_obs_np = np.asarray(jax.device_get(flat_obs), dtype=np.float32)
    action_masks_np = np.asarray(jax.device_get(action_masks), dtype=np.int8)
    spec = _build_policy_spec(args, static, flat_obs_np, action_masks_np)
    trainer_config = build_trainer_config(args)
    rollout_runner = build_compiled_rollout_runner(jax, jnp, spec)
    deploy_eval_runner = build_compiled_deploy_eval_runner(jax, jnp, spec)
    eval_runner = build_compiled_eval_runner(jax, jnp, spec)
    frozen_rollout_runner = build_compiled_frozen_opponent_rollout_runner(
        jax, jnp, spec
    )
    frozen_eval_runner = build_compiled_frozen_opponent_eval_runner(jax, jnp, spec)
    grouped_rollout_runner = build_compiled_grouped_opponent_rollout_runner(
        jax, jnp, spec
    )
    grouped_eval_runner = build_compiled_grouped_opponent_eval_runner(jax, jnp, spec)
    historical_match_eval_runner = (
        build_compiled_historical_match_eval_runner(jax, jnp, spec)
        if historical_milestone_updates
        else None
    )
    update_runner, optimizer_transform = build_jitted_ppo_update_runner(
        jax, jnp, spec, trainer_config
    )
    selector_optimizer_transform = None
    if bool(getattr(args, "intent_selector_enabled", False)):
        selector_update_runner, selector_optimizer_transform = (
            build_jitted_selector_update_runner(
                jax,
                jnp,
                spec,
                trainer_config,
                selector_value_coef=float(
                    getattr(args, "intent_selector_value_coef", 0.5)
                ),
                selector_entropy_coef=float(
                    getattr(args, "intent_selector_entropy_coef", 0.01)
                ),
                selector_usage_reg_coef=float(
                    getattr(args, "intent_selector_usage_reg_coef", 0.01)
                ),
                selector_learning_rate=_selector_learning_rate_for_args(
                    args, trainer_config
                ),
            )
        )
    else:
        selector_update_runner = None
    intent_disc_enabled = bool(getattr(args, "intent_diversity_enabled", False))
    intent_disc_spec = (
        build_intent_discriminator_spec(args, spec) if intent_disc_enabled else None
    )
    if intent_disc_spec is not None:
        intent_disc_runner, intent_disc_transform = (
            build_intent_discriminator_update_runner(
                jax,
                jnp,
                intent_disc_spec,
            )
        )
        initial_intent_disc_params = init_intent_discriminator_params(
            jax,
            jnp,
            intent_disc_spec,
            seed=int(args.policy_seed) + 7_001,
        )
        initial_intent_disc_opt_state = intent_disc_transform.init(
            initial_intent_disc_params
        )
        intent_policy_sensitivity_runner = (
            build_intent_policy_sensitivity_runner(
                jax,
                jnp,
                spec,
                sample_count=int(
                    getattr(args, "intent_policy_sensitivity_sample_states", 32)
                ),
            )
            if bool(getattr(args, "intent_policy_sensitivity_enabled", True))
            else None
        )
    else:
        intent_disc_runner = None
        initial_intent_disc_params = None
        initial_intent_disc_opt_state = None
        intent_policy_sensitivity_runner = None
    checkpoint_dir = str(args.checkpoint_dir).strip()
    resume_checkpoint = str(args.resume_checkpoint).strip()
    continuation_checkpoint_info = _prepare_continuation_checkpoint(args)
    if continuation_checkpoint_info is not None and not resume_checkpoint:
        resume_checkpoint = str(
            continuation_checkpoint_info.get("local_path", "") or ""
        ).strip()
    if continuation_checkpoint_info is not None and not resume_checkpoint:
        raise SystemExit("--continue-run-id did not resolve a local resume checkpoint.")
    latest_checkpoint_path: str | None = None
    latest_checkpoint_artifact_path: str | None = None
    frozen_opponent_payload, frozen_opponent_info = _load_frozen_opponent_payload(args)
    opponent_params = None
    grouped_opponent_params = None
    active_opponent_info = None
    opponent_candidates: list[dict[str, Any]] = []
    intent_sample_artifacts: list[str] = []
    opponent_rng = np.random.default_rng(int(args.policy_seed) + 90_001)
    opponent_pool_enabled = not bool(getattr(args, "disable_opponent_pool", False))
    grouped_opponent_sampling_enabled = (
        opponent_pool_enabled and _uses_grouped_opponent_sampling(args)
    )
    if frozen_opponent_payload is not None:
        if _normalize_policy_spec_dict(
            frozen_opponent_payload.get("policy_spec", {})
        ) != asdict(spec):
            raise SystemExit(
                "Frozen opponent policy_spec does not match the current JAX trainer policy_spec."
            )
        opponent_params = jax.device_put(frozen_opponent_payload["params"])
        active_opponent_info = dict(frozen_opponent_info or {})
        _add_opponent_candidate(
            opponent_candidates,
            params=opponent_params,
            info={
                **active_opponent_info,
                "candidate_kind": "bootstrap",
            },
        )
        if grouped_opponent_sampling_enabled:
            grouped_opponent_params, active_opponent_info = (
                _select_grouped_opponents_from_pool(
                    opponent_candidates,
                    args=args,
                    rng=opponent_rng,
                    jax=jax,
                    jnp=jnp,
                )
            )
            opponent_params = None

    pinned_opponent_pool_enabled = bool(getattr(args, "enable_multi_possession", False))
    initial_assignment = 0 if opponent_candidates else -1
    opponent_assignments = {
        role: jnp.full(
            (int(args.kernel_batch_size),),
            initial_assignment,
            dtype=jnp.int32,
        )
        for role in TRAINING_ROLES
    }
    initial_mode_prob = float(getattr(args, "opponent_deterministic_episode_prob", 0.0))
    opponent_deterministic_modes = {
        role: jnp.asarray(
            opponent_rng.random(int(args.kernel_batch_size)) < initial_mode_prob,
            dtype=jnp.bool_,
        )
        for role in TRAINING_ROLES
    }
    if pinned_opponent_pool_enabled and opponent_candidates:
        opponent_params = None
        grouped_opponent_params = None
        active_opponent_info = {
            "source": "pinned_pool",
            "candidate_count": len(opponent_candidates),
        }

    initial_params = init_actor_critic_params(
        jax,
        jnp,
        spec,
        seed=int(args.policy_seed),
    )
    initial_opt_state = init_optimizer_state(optimizer_transform, initial_params)
    initial_selector_opt_state = (
        init_optimizer_state(selector_optimizer_transform, initial_params)
        if selector_optimizer_transform is not None
        else None
    )

    reset_resume_env_state = False
    if resume_checkpoint:
        checkpoint_payload = load_checkpoint(resume_checkpoint)
        _validate_resume_checkpoint_payload(
            checkpoint_payload,
            trainer_config=trainer_config,
            spec=spec,
            args=args,
        )
        completed_updates = int(checkpoint_payload["update_index"])
        if completed_updates >= int(args.num_updates):
            raise SystemExit(
                "Resume checkpoint already reached or exceeded --num-updates; increase --num-updates to continue."
            )
        params = jax.device_put(checkpoint_payload["params"])
        opt_state = jax.device_put(
            _restore_like_template(checkpoint_payload["opt_state"], initial_opt_state)
        )
        if initial_selector_opt_state is not None:
            restored_selector_opt_state = checkpoint_payload.get("selector_opt_state")
            selector_opt_state = jax.device_put(
                _restore_like_template(
                    restored_selector_opt_state,
                    initial_selector_opt_state,
                )
                if restored_selector_opt_state is not None
                else initial_selector_opt_state
            )
        else:
            selector_opt_state = None
        continuation_preserves_intent_disc = bool(
            getattr(args, "continue_preserve_intent_discriminator_state", False)
        )
        reset_resume_intent_disc = bool(
            getattr(args, "resume_reset_intent_discriminator_state", False)
        ) or (
            continuation_checkpoint_info is not None
            and not continuation_preserves_intent_disc
        )
        if intent_disc_enabled:
            restored_disc = dict(
                checkpoint_payload.get("intent_discriminator_state", {}) or {}
            )
            if reset_resume_intent_disc:
                intent_disc_params = initial_intent_disc_params
                intent_disc_opt_state = initial_intent_disc_opt_state
                intent_bonus_stats = init_bonus_stats()
                print(
                    "[resume] Reset auxiliary intent discriminator state for continuation."
                )
            elif (
                restored_disc.get("params") is not None
                and restored_disc.get("opt_state") is not None
            ):
                intent_disc_params = jax.device_put(
                    _restore_like_template(
                        restored_disc["params"], initial_intent_disc_params
                    )
                )
                intent_disc_opt_state = jax.device_put(
                    _restore_like_template(
                        restored_disc["opt_state"], initial_intent_disc_opt_state
                    )
                )
                intent_bonus_stats = dict(
                    restored_disc.get("bonus_stats", {}) or init_bonus_stats()
                )
                if (
                    continuation_checkpoint_info is not None
                    and continuation_preserves_intent_disc
                ):
                    print(
                        "[resume] Preserved auxiliary intent discriminator state for continuation."
                    )
            else:
                intent_disc_params = initial_intent_disc_params
                intent_disc_opt_state = initial_intent_disc_opt_state
                intent_bonus_stats = init_bonus_stats()
        else:
            intent_disc_params = None
            intent_disc_opt_state = None
            intent_bonus_stats = init_bonus_stats()
        continuation_preserves_env_state = bool(
            getattr(args, "continue_preserve_env_state", False)
        )
        reset_resume_env_state = bool(
            getattr(args, "resume_reset_env_state", False)
        ) or (
            continuation_checkpoint_info is not None
            and not continuation_preserves_env_state
        )
        if reset_resume_env_state:
            print(
                "[resume] Reset transient JAX env state and RNG; restored policy, "
                f"optimizer, and update index from {resume_checkpoint}."
            )
        else:
            restored_current_state = checkpoint_payload["current_state"]
            restored_eval_initial_state = checkpoint_payload["eval_initial_state"]
            if not isinstance(restored_current_state, dict) or not isinstance(
                restored_eval_initial_state, dict
            ):
                raise SystemExit(
                    "Resume checkpoint does not contain mixed-role JAX train state."
                )
            current_states = {
                role: jax.device_put(
                    _restore_like_template(
                        restored_current_state[role], current_states[role]
                    )
                )
                for role in TRAINING_ROLES
            }
            eval_initial_states = {
                role: jax.device_put(
                    _restore_like_template(
                        restored_eval_initial_state[role], eval_initial_states[role]
                    )
                )
                for role in TRAINING_ROLES
            }
            if (
                continuation_checkpoint_info is not None
                and continuation_preserves_env_state
            ):
                print(
                    "[resume] Preserved checkpoint transient JAX env state and RNG for continuation."
                )
        if reset_resume_env_state:
            base_key = jax.device_put(jax.random.fold_in(base_key, completed_updates))
        else:
            base_key = jax.device_put(checkpoint_payload["base_key"])
        restored_opponent_pool = checkpoint_payload.get("opponent_pool_state")
        if (
            pinned_opponent_pool_enabled
            and not reset_resume_env_state
            and restored_opponent_pool is not None
        ):
            restored_infos = list(
                restored_opponent_pool.get("candidate_infos", []) or []
            )
            restored_params = _unstack_opponent_candidate_params(
                restored_opponent_pool.get("candidate_params"),
                len(restored_infos),
                jax=jax,
            )
            opponent_candidates = [
                {"params": jax.device_put(candidate_params), "info": dict(info)}
                for candidate_params, info in zip(
                    restored_params,
                    restored_infos,
                    strict=True,
                )
            ]
            restored_assignments = dict(
                restored_opponent_pool.get("assignments", {}) or {}
            )
            restored_modes = dict(
                restored_opponent_pool.get("deterministic_modes", {}) or {}
            )
            opponent_assignments = {
                role: jax.device_put(restored_assignments[role])
                for role in TRAINING_ROLES
            }
            opponent_deterministic_modes = {
                role: jax.device_put(restored_modes[role]) for role in TRAINING_ROLES
            }
            restored_rng_state = restored_opponent_pool.get("rng_state")
            if restored_rng_state:
                opponent_rng.bit_generator.state = dict(restored_rng_state)
            opponent_params = None
            grouped_opponent_params = None
        train_history = []
        deploy_eval_history = []
        eval_trajectories = list(checkpoint_payload.get("eval_trajectories", []))
        last_metrics = checkpoint_payload.get("last_metrics")
    else:
        completed_updates = 0
        params = initial_params
        opt_state = initial_opt_state
        selector_opt_state = initial_selector_opt_state
        intent_disc_params = initial_intent_disc_params
        intent_disc_opt_state = initial_intent_disc_opt_state
        intent_bonus_stats = init_bonus_stats()
        train_history = []
        deploy_eval_history = []
        eval_trajectories = []
        last_metrics = None

    if reset_resume_env_state:
        resume_possession_limit = _multi_possession_limit_for_update(
            args,
            completed_updates + 1,
        )
        configured_overtime_cap = int(
            getattr(args, "multi_possession_overtime_round_cap", 0)
        )
        current_states = {
            role: _state_with_episode_game_length(
                current_states[role],
                possession_limit=resume_possession_limit,
                overtime_cap=configured_overtime_cap,
                jnp=jnp,
            )
            for role in TRAINING_ROLES
        }

    historical_checkpoint_root = _historical_checkpoint_root(
        checkpoint_dir=checkpoint_dir,
        resume_checkpoint=resume_checkpoint,
    )
    restored_historical_artifact = _load_historical_eval_artifact(
        checkpoint_root=historical_checkpoint_root,
        continuation_run_id=str(getattr(args, "continue_run_id", "") or "").strip(),
        continue_cache_dir=str(getattr(args, "continue_cache_dir", "") or ""),
    )
    historical_eval_history = list(
        (restored_historical_artifact or {}).get("matches", []) or []
    )
    historical_eval_update_artifact_paths = {
        int(update): str(path)
        for update, path in dict(
            (restored_historical_artifact or {}).get(
                "evaluation_update_artifacts",
                {},
            )
            or {}
        ).items()
    }
    (
        historical_milestone_params,
        historical_milestone_records,
    ) = _restore_historical_milestone_params(
        milestone_updates=historical_milestone_updates,
        completed_updates=completed_updates,
        checkpoint_root=historical_checkpoint_root,
        continuation_run_id=str(getattr(args, "continue_run_id", "") or "").strip(),
        continue_cache_dir=str(getattr(args, "continue_cache_dir", "") or ""),
        spec=spec,
        jax=jax,
    )
    historical_eval_artifact_path = (
        HISTORICAL_EVAL_ARTIFACT_PATH
        if historical_eval_history
        and str(getattr(args, "continue_run_id", "") or "").strip()
        else None
    )

    continuation_pool_info = None
    if (
        str(getattr(args, "continue_run_id", "") or "").strip()
        and opponent_pool_enabled
    ):
        continuation_candidates, continuation_pool_info = (
            _load_continuation_opponent_candidates(
                args,
                jax=jax,
                spec=spec,
                resume_artifact_path=(
                    str(continuation_checkpoint_info.get("artifact_path", "") or "")
                    if continuation_checkpoint_info is not None
                    else ""
                ),
            )
        )
        for candidate in continuation_candidates:
            _add_opponent_candidate(
                opponent_candidates,
                params=candidate["params"],
                info=dict(candidate["info"]),
            )
        if continuation_candidates:
            if pinned_opponent_pool_enabled:
                opponent_params = None
                grouped_opponent_params = None
                active_opponent_info = {
                    "source": "pinned_pool",
                    "candidate_count": len(opponent_candidates),
                }
            elif grouped_opponent_sampling_enabled:
                grouped_opponent_params, active_opponent_info = (
                    _select_grouped_opponents_from_pool(
                        opponent_candidates,
                        args=args,
                        rng=opponent_rng,
                        jax=jax,
                        jnp=jnp,
                    )
                )
                opponent_params = None
            else:
                opponent_params, active_opponent_info = _select_opponent_from_pool(
                    opponent_candidates,
                    args=args,
                    rng=opponent_rng,
                )
                grouped_opponent_params = None
            print(
                "[continue] Seeded opponent pool with "
                f"{len(continuation_candidates)} checkpoint(s) from MLflow run "
                f"{continuation_pool_info.get('run_id') if continuation_pool_info else ''}."
            )

    cumulative_episode_usage = _init_cumulative_episode_usage(last_metrics)

    mlflow, mlflow_context = _maybe_start_mlflow_run(args, mode="train")

    with mlflow_context:
        active_mlflow_run_id = None
        if mlflow is not None and mlflow.active_run() is not None:
            active_mlflow_run_id = str(mlflow.active_run().info.run_id)
            mlflow.set_tag("basketworld.training_state", "running")
        write_training_status(
            getattr(args, "training_status_file", ""),
            state="running",
            completed_updates=completed_updates,
            target_updates=int(args.num_updates),
            mlflow_run_id=active_mlflow_run_id,
            checkpoint_path=latest_checkpoint_path,
        )
        play_name_metadata = _build_training_play_name_metadata(
            args=args,
            mlflow=mlflow,
            checkpoint_dir=checkpoint_dir,
        )
        if mlflow is not None:
            _log_mlflow_params(mlflow, args, trainer_config, spec)
            _log_mlflow_play_name_metadata(mlflow, play_name_metadata)
            _log_mlflow_start_template_library(mlflow, args)

        expected_evals = _remaining_eval_count(
            start_update=completed_updates,
            num_updates=int(args.num_updates),
            eval_every_updates=int(args.eval_every_updates),
        )
        expected_deploy_evals = (
            _remaining_eval_count(
                start_update=completed_updates,
                num_updates=int(args.num_updates),
                eval_every_updates=int(args.eval_deploy_every_updates),
            )
            if int(args.eval_deploy_batches) > 0
            else 0
        )
        expected_historical_matches = sum(
            milestone_position
            for milestone_position, milestone_update in enumerate(
                historical_milestone_updates,
                start=1,
            )
            if milestone_update > completed_updates
        )
        progress = build_progress(
            total=(
                (int(args.num_updates) - completed_updates)
                + expected_evals
                + expected_deploy_evals
                + expected_historical_matches
            ),
            desc="jax_train:loop",
            disable=bool(args.no_progress),
            unit="event",
        )
        pending_selector_batches = []
        periodic_checkpoint_updates = _periodic_checkpoint_updates(args)
        historical_milestone_set = set(historical_milestone_updates)
        pause_requested = False
        last_completed_update = int(completed_updates)

        for update_idx in range(completed_updates + 1, int(args.num_updates) + 1):
            loop_start_ns = perf_counter_ns()
            base_key, update_key, *rollout_keys = jax.random.split(
                base_key, len(TRAINING_ROLES) + 2
            )
            entropy_coef = _entropy_coef_for_update(args, update_idx)
            task_reward_scale = _task_reward_scale_for_update(args, update_idx)
            phi_beta = _phi_beta_for_update(args, update_idx)
            active_possession_limit = _multi_possession_limit_for_update(
                args,
                update_idx,
            )
            active_statics = {
                role: _static_with_training_schedules(
                    statics[role],
                    phi_beta=phi_beta,
                    possession_limit=active_possession_limit,
                    jnp=jnp,
                )
                for role in TRAINING_ROLES
            }
            # Evaluation remains comparable across training updates by using
            # the final configured game length, never the partial curriculum.
            eval_statics = {
                role: _static_with_phi_beta(statics[role], phi_beta, jnp)
                for role in TRAINING_ROLES
            }
            selector_alpha, selector_eps = _selector_schedules_for_update(
                args, update_idx
            )
            opponent_deterministic_episode_prob = (
                _opponent_deterministic_episode_prob_for_update(
                    args,
                    update_idx,
                )
            )
            pinned_candidate_params = (
                _stack_opponent_candidate_params(
                    opponent_candidates,
                    jax=jax,
                    jnp=jnp,
                )
                if pinned_opponent_pool_enabled and opponent_candidates
                else None
            )
            pinned_candidate_probs = (
                jnp.asarray(
                    _opponent_candidate_probabilities(
                        opponent_candidates,
                        args=args,
                    ),
                    dtype=jnp.float32,
                )
                if pinned_candidate_params is not None
                else None
            )
            selector_multiselect_enabled = bool(
                getattr(args, "intent_selector_multiselect_enabled", False)
            )
            selector_min_play_steps = int(
                getattr(args, "intent_selector_min_play_steps", 3)
            )
            single_episode_rollout = bool(
                getattr(args, "single_episode_rollouts", False)
            )
            rollout_start_ns = perf_counter_ns()
            role_rollouts = {}
            for role, rollout_key in zip(TRAINING_ROLES, rollout_keys, strict=True):
                if pinned_candidate_params is not None:
                    role_rollouts[role] = frozen_rollout_runner(
                        active_statics[role],
                        current_states[role],
                        params,
                        pinned_candidate_params,
                        opponent_assignments[role],
                        opponent_deterministic_modes[role],
                        pinned_candidate_probs,
                        rollout_key,
                        int(args.rollout_horizon),
                        selector_alpha,
                        selector_eps,
                        selector_multiselect_enabled,
                        selector_min_play_steps,
                        single_episode_rollout,
                        float(opponent_deterministic_episode_prob),
                    )
                elif grouped_opponent_params is not None:
                    role_rollouts[role] = grouped_rollout_runner(
                        active_statics[role],
                        current_states[role],
                        params,
                        grouped_opponent_params,
                        rollout_key,
                        int(args.rollout_horizon),
                        int(active_opponent_info["group_count"]),
                        selector_alpha,
                        selector_eps,
                        selector_multiselect_enabled,
                        selector_min_play_steps,
                        single_episode_rollout,
                        float(opponent_deterministic_episode_prob),
                    )
                elif opponent_params is None:
                    role_rollouts[role] = rollout_runner(
                        active_statics[role],
                        current_states[role],
                        params,
                        rollout_key,
                        int(args.rollout_horizon),
                        selector_alpha,
                        selector_eps,
                        selector_multiselect_enabled,
                        selector_min_play_steps,
                        single_episode_rollout,
                    )
                else:
                    single_candidate_params = jax.tree_util.tree_map(
                        lambda leaf: leaf[None, ...],
                        opponent_params,
                    )
                    role_rollouts[role] = frozen_rollout_runner(
                        active_statics[role],
                        current_states[role],
                        params,
                        single_candidate_params,
                        jnp.zeros_like(opponent_assignments[role]),
                        opponent_deterministic_modes[role],
                        jnp.ones((1,), dtype=jnp.float32),
                        rollout_key,
                        int(args.rollout_horizon),
                        selector_alpha,
                        selector_eps,
                        selector_multiselect_enabled,
                        selector_min_play_steps,
                        single_episode_rollout,
                        float(opponent_deterministic_episode_prob),
                    )
            block_until_ready_tree(role_rollouts)
            if pinned_candidate_params is not None:
                opponent_assignments = {
                    role: role_rollouts[role].final_opponent_assignment
                    for role in TRAINING_ROLES
                }
            if pinned_candidate_params is not None or opponent_params is not None:
                opponent_deterministic_modes = {
                    role: role_rollouts[role].final_opponent_deterministic_episode
                    for role in TRAINING_ROLES
                }
            rollout_elapsed_ns = perf_counter_ns() - rollout_start_ns
            role_rollouts = {
                role: _apply_task_reward_scale_to_rollout(
                    rollout,
                    task_reward_scale,
                    jnp,
                )
                for role, rollout in role_rollouts.items()
            }
            role_reward_components = {
                role: _build_reward_component_arrays(
                    role_rollouts[role],
                    active_statics[role],
                    task_reward_scale,
                    jnp,
                )
                for role in TRAINING_ROLES
            }

            if bool(getattr(args, "intent_selector_enabled", False)):
                for role in TRAINING_ROLES:
                    selector_batch = build_selector_batch(
                        role_rollouts[role],
                        trainer_config,
                        jax,
                        jnp,
                    )
                    pending_selector_batches.append(
                        limit_selector_batch_samples(
                            selector_batch,
                            jnp,
                            max_samples=int(
                                getattr(
                                    args,
                                    "intent_selector_max_samples_per_update",
                                    0,
                                )
                            ),
                        )
                    )

            intent_disc_metrics: dict[str, Any] = {}
            latest_intent_sample_payload = None
            if (
                intent_disc_enabled
                and intent_disc_spec is not None
                and intent_disc_runner is not None
            ):
                global_step = (
                    int(update_idx)
                    * int(args.kernel_batch_size)
                    * int(args.rollout_horizon)
                    * len(TRAINING_ROLES)
                )
                intent_beta = compute_intent_beta(
                    global_step=global_step,
                    spec=intent_disc_spec,
                    update_index=update_idx,
                )
                intent_disc_metrics = {
                    "intent_bonus_beta": float(intent_beta),
                    "intent_disc_skipped_warmup": 0.0,
                    "intent_bonus_skipped_warmup": (
                        1.0 if float(intent_beta) <= 0.0 else 0.0
                    ),
                }
                sensitivity_every = max(
                    1,
                    int(
                        getattr(
                            args,
                            "intent_policy_sensitivity_log_every_rollouts",
                            4,
                        )
                    ),
                )
                should_measure_sensitivity = (
                    intent_policy_sensitivity_runner is not None
                    and int(update_idx) % sensitivity_every == 0
                )
                intent_rollout = None
                intent_training_mask = None
                if should_measure_sensitivity:
                    intent_rollout = concatenate_rollout_outputs(
                        [role_rollouts[role] for role in TRAINING_ROLES],
                        jnp,
                    )
                    intent_training_mask, _, _ = build_trajectory_training_masks(
                        intent_rollout.trajectory,
                        trainer_config,
                        jax,
                        jnp,
                    )
                    sensitivity_active_mask = (
                        (intent_rollout.trajectory.policy_intent_gate > 0.5)
                        & (intent_rollout.trajectory.training_role > 0.0)
                        & (intent_training_mask > 0.5)
                    )
                    raw_sensitivity_metrics = intent_policy_sensitivity_runner(
                        params,
                        intent_rollout.trajectory.flat_obs,
                        intent_rollout.trajectory.action_mask,
                        sensitivity_active_mask,
                    )
                    block_until_ready_tree(raw_sensitivity_metrics)
                    intent_disc_metrics.update(
                        {
                            key: float(np.asarray(value))
                            for key, value in raw_sensitivity_metrics.items()
                        }
                    )
                # Pretrain the state-only q(z | s_(t+1)) even while the policy bonus is in
                # warmup.  At beta == 0 the computed bonus is exactly zero, but
                # the classifier is ready to provide a useful signal when the
                # bonus schedule starts.
                if intent_disc_enabled:
                    if intent_rollout is None:
                        intent_rollout = concatenate_rollout_outputs(
                            [role_rollouts[role] for role in TRAINING_ROLES],
                            jnp,
                        )
                    if intent_training_mask is None:
                        intent_training_mask, _, _ = build_trajectory_training_masks(
                            intent_rollout.trajectory,
                            trainer_config,
                            jax,
                            jnp,
                        )
                    (
                        intent_features,
                        intent_labels,
                        intent_active_mask,
                        intent_segment_ids,
                        intent_age,
                    ) = build_intent_step_features_from_rollout(
                        intent_rollout,
                        intent_disc_spec,
                        jnp,
                        training_mask=intent_training_mask,
                    )
                    params_key, update_key = jax.random.split(update_key)
                    (
                        intent_disc_params,
                        intent_disc_opt_state,
                        raw_disc_metrics,
                        raw_intent_bonus,
                    ) = intent_disc_runner(
                        intent_disc_params,
                        intent_disc_opt_state,
                        intent_features,
                        intent_labels,
                        intent_active_mask,
                        intent_segment_ids,
                        intent_age,
                        params_key,
                    )
                    block_until_ready_tree(
                        (
                            intent_disc_params,
                            intent_disc_opt_state,
                            raw_disc_metrics,
                            raw_intent_bonus,
                        )
                    )
                    raw_bonus_np = np.asarray(
                        jax.device_get(raw_intent_bonus), dtype=np.float32
                    )
                    active_mask_np = np.asarray(
                        jax.device_get(intent_active_mask), dtype=bool
                    )
                    active_raw_bonus = raw_bonus_np[active_mask_np]
                    intent_bonus_stats = update_bonus_stats(
                        intent_bonus_stats, active_raw_bonus
                    )
                    intent_bonus = compute_normalized_intent_bonus(
                        raw_intent_bonus,
                        intent_active_mask,
                        intent_bonus_stats,
                        beta=intent_beta,
                        clip=float(intent_disc_spec.bonus_clip),
                        jnp=jnp,
                    )
                    batch_offset = 0
                    for role in TRAINING_ROLES:
                        role_batch_size = int(
                            role_rollouts[role].trajectory.rewards.shape[1]
                        )
                        role_bonus = intent_bonus[
                            :, batch_offset : batch_offset + role_batch_size
                        ]
                        role_reward_components[role] = {
                            **role_reward_components[role],
                            "intent_bonus": role_bonus.astype(jnp.float32),
                        }
                        batch_offset += role_batch_size
                    norm_bonus_np = np.asarray(
                        jax.device_get(intent_bonus), dtype=np.float32
                    )
                    active_norm_bonus = norm_bonus_np[active_mask_np]
                    intent_disc_metrics.update(
                        {
                            key: float(np.asarray(value))
                            for key, value in raw_disc_metrics.items()
                        }
                    )
                    intent_disc_metrics.update(
                        {
                            "intent_bonus_stats_count": float(
                                intent_bonus_stats["count"]
                            ),
                            "intent_bonus_stats_mean": float(
                                intent_bonus_stats["mean"]
                            ),
                            "intent_bonus_stats_std": float(
                                np.sqrt(max(float(intent_bonus_stats["var"]), 1.0e-12))
                            ),
                            "intent_bonus_raw_mean": (
                                float(np.mean(active_raw_bonus))
                                if active_raw_bonus.size
                                else 0.0
                            ),
                            "intent_bonus_raw_std": (
                                float(np.std(active_raw_bonus))
                                if active_raw_bonus.size
                                else 0.0
                            ),
                            "intent_bonus_shaping_per_step_mean": (
                                float(np.mean(active_norm_bonus))
                                if active_norm_bonus.size
                                else 0.0
                            ),
                            "intent_bonus_shaping_per_step_std": (
                                float(np.std(active_norm_bonus))
                                if active_norm_bonus.size
                                else 0.0
                            ),
                            "intent_bonus_active_sample_count": int(
                                active_mask_np.sum()
                            ),
                        }
                    )
                    if bool(getattr(args, "disc_eval_batch_output", False)):
                        latest_intent_sample_payload = build_intent_sample_dump(
                            params=intent_disc_params,
                            features=intent_features,
                            labels=intent_labels,
                            active_mask=intent_active_mask,
                            segment_ids=intent_segment_ids,
                            intent_age=intent_age,
                            bonus=intent_bonus,
                            rollout=intent_rollout,
                            spec=intent_disc_spec,
                            jax=jax,
                            jnp=jnp,
                            update_index=update_idx,
                            max_samples=int(
                                getattr(args, "intent_sample_dump_size", 2048)
                            ),
                        )

            role_ppo_batches = [
                build_ppo_batch(
                    role_rollouts[role],
                    trainer_config,
                    jax,
                    jnp,
                    intent_bonus=role_reward_components[role]["intent_bonus"],
                )
                for role in TRAINING_ROLES
            ]
            should_log_intent_rollout_diagnostics = (
                int(update_idx) == int(args.num_updates)
                or int(update_idx) % max(1, int(args.log_every_updates)) == 0
            )
            if intent_disc_enabled and should_log_intent_rollout_diagnostics:
                intent_disc_metrics.update(
                    _summarize_intent_intrinsic_advantages(
                        role_ppo_batches,
                        [role_rollouts[role] for role in TRAINING_ROLES],
                        jax=jax,
                    )
                )
                if intent_rollout is None:
                    intent_rollout = concatenate_rollout_outputs(
                        [role_rollouts[role] for role in TRAINING_ROLES],
                        jnp,
                    )
                if intent_training_mask is None:
                    intent_training_mask, _, _ = build_trajectory_training_masks(
                        intent_rollout.trajectory,
                        trainer_config,
                        jax,
                        jnp,
                    )
                intent_disc_metrics.update(
                    _summarize_intent_behavior_by_label(
                        intent_rollout,
                        intent_training_mask,
                        num_intents=int(args.num_intents),
                        jax=jax,
                    )
                )
            if latest_intent_sample_payload is not None:
                for key, value in intent_disc_metrics.items():
                    if key.startswith(
                        (
                            "intent_policy_sensitivity_",
                            "intent_intrinsic_",
                        )
                    ):
                        latest_intent_sample_payload[f"metric_{key}"] = np.asarray(
                            [float(value)],
                            dtype=np.float32,
                        )
            ppo_batch = concatenate_ppo_batches(role_ppo_batches, jnp)
            rollout_out = concatenate_rollout_outputs(
                [role_rollouts[role] for role in TRAINING_ROLES],
                jnp,
            )
            update_start_ns = perf_counter_ns()
            update_key, ppo_update_key, selector_update_key = jax.random.split(
                update_key, 3
            )
            params, opt_state, update_metrics = update_runner(
                params,
                opt_state,
                ppo_batch,
                ppo_update_key,
                entropy_coef,
            )
            selector_update_metrics: dict[str, Any] = {}
            if (
                selector_update_runner is not None
                and update_idx
                % int(getattr(args, "intent_selector_train_every_rollouts", 1))
                == 0
                and pending_selector_batches
            ):
                selector_train_batch = concatenate_selector_batches(
                    pending_selector_batches, jnp
                )
                selector_train_batch = limit_selector_batch_samples(
                    selector_train_batch,
                    jnp,
                    max_samples=int(
                        getattr(args, "intent_selector_max_samples_per_update", 0)
                    ),
                )
                selector_sample_count = float(
                    np.asarray(
                        jax.device_get(jnp.sum(selector_train_batch.active_mask))
                    )
                )
                if selector_sample_count > 0.0:
                    if selector_opt_state is None:
                        raise RuntimeError(
                            "Selector optimizer state is missing for selector PPO update."
                        )
                    params, selector_opt_state, raw_selector_metrics = (
                        selector_update_runner(
                            params,
                            selector_opt_state,
                            selector_train_batch,
                            selector_update_key,
                            selector_eps,
                        )
                    )
                    selector_update_metrics = {
                        key: float(np.asarray(value))
                        for key, value in raw_selector_metrics.items()
                    }
                    selector_update_metrics["selector_train_skipped_empty"] = 0.0
                else:
                    selector_update_metrics = {
                        "selector_train_skipped_empty": 1.0,
                        "selector_train_sample_count": 0.0,
                    }
                selector_update_metrics["selector_train_skipped_cadence"] = 0.0
                selector_update_metrics["selector_train_pending_rollout_count"] = float(
                    len(pending_selector_batches)
                )
                pending_selector_batches.clear()
            elif bool(getattr(args, "intent_selector_enabled", False)):
                selector_update_metrics = {
                    "selector_train_skipped_cadence": 1.0,
                    "selector_train_sample_count": 0.0,
                    "selector_train_pending_rollout_count": float(
                        len(pending_selector_batches)
                    ),
                }
            block_until_ready_tree(
                (
                    params,
                    opt_state,
                    selector_opt_state,
                    update_metrics,
                    selector_update_metrics,
                )
            )
            update_elapsed_ns = perf_counter_ns() - update_start_ns
            if single_episode_rollout:
                base_key, reset_block_key = jax.random.split(base_key)
                role_reset_keys = jax.random.split(reset_block_key, len(TRAINING_ROLES))
                current_states = {}
                for role, role_reset_key in zip(
                    TRAINING_ROLES, role_reset_keys, strict=True
                ):
                    reset_keys = jax.random.split(
                        role_reset_key, int(args.kernel_batch_size)
                    )
                    current_states[role] = reset_batch_minimal(
                        active_statics[role], reset_keys, jax, jnp
                    )
                block_until_ready_tree(current_states)
            else:
                current_states = {
                    role: role_rollouts[role].final_state for role in TRAINING_ROLES
                }

            last_metrics = summarize_training_step(
                rollout_out,
                ppo_batch,
                {
                    key: float(np.asarray(value))
                    for key, value in update_metrics.items()
                },
                rollout_elapsed_ns,
                update_elapsed_ns,
                batch_size=int(args.kernel_batch_size) * len(TRAINING_ROLES),
                horizon=int(args.rollout_horizon),
                update_index=update_idx,
                policy_update_epochs=int(args.policy_update_epochs),
                ppo_minibatches=int(args.ppo_minibatches),
            )
            last_metrics.update(
                summarize_learner_rebound_metrics(
                    role_rollouts["offense"],
                    role_rollouts["defense"],
                    include_opponent_mode_split=(
                        grouped_opponent_params is not None
                        or opponent_params is not None
                    ),
                )
            )
            for role in TRAINING_ROLES:
                last_metrics.update(
                    _summarize_role_rollout_metrics(
                        role,
                        role_rollouts[role],
                        num_intents=int(args.num_intents),
                        trainer_config=trainer_config,
                        jax=jax,
                        jnp=jnp,
                    )
                )
                role_training_mask, _, _ = build_trajectory_training_masks(
                    role_rollouts[role].trajectory,
                    trainer_config,
                    jax,
                    jnp,
                )
                metric_role_value = 1.0 if role == "offense" else -1.0
                metric_role_mask = (
                    np.asarray(
                        role_rollouts[role].trajectory.training_role,
                        dtype=np.float32,
                    )
                    == metric_role_value
                ).astype(np.float32)
                role_components = role_reward_components[role]
                last_metrics.update(
                    summarize_ppo_eligible_reward_component_metrics(
                        f"{role}_ppo_eligible",
                        role_rollouts[role].trajectory,
                        role_training_mask,
                        task_rewards=role_components["task_reward"],
                        phi_rewards=role_components["phi_reward"],
                        intent_bonus_rewards=role_components["intent_bonus"],
                        metric_mask=metric_role_mask,
                        completion_mask=role_training_mask,
                    )
                )
            last_metrics.update(_summarize_combined_value_diagnostics(last_metrics))
            if bool(getattr(args, "intent_selector_enabled", False)):
                last_metrics.update(
                    summarize_selector_metrics(
                        rollout_out,
                        num_intents=int(args.num_intents),
                        alpha=selector_alpha,
                        eps=selector_eps,
                    )
                )
                last_metrics.update(selector_update_metrics)
            _add_episode_usage_metrics(last_metrics, cumulative_episode_usage)
            episode_possession_limits = np.concatenate(
                [
                    np.asarray(
                        jax.device_get(current_states[role].episode_possession_limit),
                        dtype=np.int32,
                    ).reshape(-1)
                    for role in TRAINING_ROLES
                ]
            )
            episode_overtime_caps = np.concatenate(
                [
                    np.asarray(
                        jax.device_get(current_states[role].episode_overtime_round_cap),
                        dtype=np.int32,
                    ).reshape(-1)
                    for role in TRAINING_ROLES
                ]
            )
            last_metrics["multi_possession_limit_active"] = int(active_possession_limit)
            last_metrics["multi_possession_limit_curriculum_enabled"] = float(
                _multi_possession_curriculum_enabled(args)
            )
            last_metrics["episode_possession_limit_mean"] = float(
                np.mean(episode_possession_limits)
            )
            last_metrics["episode_possession_limit_min"] = int(
                np.min(episode_possession_limits)
            )
            last_metrics["episode_possession_limit_max"] = int(
                np.max(episode_possession_limits)
            )
            last_metrics["episode_overtime_round_cap_mean"] = float(
                np.mean(episode_overtime_caps)
            )
            last_metrics["task_reward_scale"] = float(task_reward_scale)
            last_metrics["task_reward_scale_is_scheduled"] = float(
                getattr(args, "task_reward_scale_start", None) is not None
                or getattr(args, "task_reward_scale_end", None) is not None
            )
            last_metrics["opponent_deterministic_episode_prob"] = float(
                opponent_deterministic_episode_prob
            )
            last_metrics["opponent_deterministic_episode_prob_is_scheduled"] = float(
                _opponent_deterministic_episode_prob_is_scheduled(args)
            )
            last_metrics["phi_beta"] = float(phi_beta)
            last_metrics["phi_beta_is_scheduled"] = float(
                bool(getattr(args, "enable_phi_shaping", False))
                and (
                    float(getattr(args, "phi_beta_start", 0.0) or 0.0)
                    != float(getattr(args, "phi_beta_end", 0.0) or 0.0)
                    or int(getattr(args, "phi_beta_warmup_updates", 0)) > 0
                    or int(getattr(args, "phi_beta_ramp_updates", 1)) > 1
                )
            )
            if intent_disc_metrics:
                last_metrics.update(intent_disc_metrics)
            if active_opponent_info is not None:
                last_metrics["opponent_update_index"] = int(
                    active_opponent_info.get(
                        "latest_update_index",
                        active_opponent_info.get("update_index", 0),
                    )
                )
                last_metrics["opponent_source"] = str(
                    active_opponent_info.get("source", "unknown")
                )
                last_metrics["opponent_group_count"] = int(
                    active_opponent_info.get("group_count", 1)
                )
                last_metrics["opponent_unique_update_count"] = int(
                    active_opponent_info.get("unique_update_count", 1)
                )
            else:
                last_metrics["opponent_update_index"] = -1
                last_metrics["opponent_source"] = "legal_random"
                last_metrics["opponent_group_count"] = 0
                last_metrics["opponent_unique_update_count"] = 0
            if pinned_opponent_pool_enabled:
                assignment_values = np.concatenate(
                    [
                        np.asarray(
                            jax.device_get(opponent_assignments[role]),
                            dtype=np.int32,
                        )
                        for role in TRAINING_ROLES
                    ]
                )
                assigned = assignment_values[assignment_values >= 0]
                last_metrics["opponent_pool_candidate_count"] = int(
                    len(opponent_candidates)
                )
                last_metrics["opponent_assigned_game_count"] = int(assigned.size)
                last_metrics["opponent_legal_random_game_count"] = int(
                    np.sum(assignment_values < 0)
                )
                last_metrics["opponent_active_candidate_count"] = int(
                    np.unique(assigned).size
                )
            loop_elapsed_ns = perf_counter_ns() - loop_start_ns
            loop_elapsed_sec = max(loop_elapsed_ns / 1e9, 1e-12)
            loop_steps = (
                int(args.kernel_batch_size)
                * int(args.rollout_horizon)
                * len(TRAINING_ROLES)
            )
            last_metrics["train_loop_elapsed_sec"] = float(loop_elapsed_sec)
            last_metrics["train_loop_latency_ms"] = float(loop_elapsed_ns / 1e6)
            last_metrics["train_loop_steps_per_sec"] = float(
                loop_steps / loop_elapsed_sec
            )
            last_metrics["train_loop_active_steps_per_sec"] = float(
                float(last_metrics.get("rollout_active_step_count", 0.0))
                / loop_elapsed_sec
            )
            last_metrics["train_loop_overhead_sec"] = float(
                max(
                    0.0,
                    loop_elapsed_sec
                    - float(last_metrics.get("end_to_end_elapsed_sec", 0.0)),
                )
            )

            should_log_history = (
                update_idx == 1
                or update_idx == int(args.num_updates)
                or (
                    int(args.log_every_updates) > 0
                    and update_idx % int(args.log_every_updates) == 0
                )
            )
            if should_log_history:
                train_history.append(last_metrics)
                if mlflow is not None:
                    _log_mlflow_metrics(
                        mlflow,
                        _filter_mlflow_train_metrics(
                            last_metrics,
                            profile=str(getattr(args, "mlflow_metric_profile", "core")),
                        ),
                        step=update_idx,
                        prefix="jax/train",
                    )

            progress.update(1)
            progress.set_postfix_str(
                (
                    f"train:{update_idx}"
                    f" sps:{float(last_metrics['end_to_end_steps_per_sec']):.0f}"
                    f" active:{float(last_metrics['active_end_to_end_steps_per_sec']):.0f}"
                    f" rollout:{float(last_metrics['rollout_time_pct']):.0f}%"
                    f" update:{float(last_metrics['ppo_update_time_pct']):.0f}%"
                ),
                refresh=False,
            )

            should_eval = int(args.eval_every_updates) > 0 and (
                update_idx == int(args.num_updates)
                or update_idx % int(args.eval_every_updates) == 0
            )
            if should_eval:
                eval_key = jax.random.PRNGKey(
                    int(args.policy_seed) + 1_000_000 + update_idx
                )
                role_eval_keys = jax.random.split(eval_key, len(TRAINING_ROLES))
                eval_outputs = {}
                for role, role_eval_key in zip(
                    TRAINING_ROLES, role_eval_keys, strict=True
                ):
                    if grouped_opponent_params is not None:
                        eval_outputs[role] = grouped_eval_runner(
                            eval_statics[role],
                            eval_initial_states[role],
                            params,
                            grouped_opponent_params,
                            role_eval_key,
                            int(args.eval_horizon),
                            int(active_opponent_info["group_count"]),
                        )
                    elif opponent_params is None:
                        eval_outputs[role] = eval_runner(
                            eval_statics[role],
                            eval_initial_states[role],
                            params,
                            role_eval_key,
                            int(args.eval_horizon),
                        )
                    else:
                        eval_outputs[role] = frozen_eval_runner(
                            eval_statics[role],
                            eval_initial_states[role],
                            params,
                            opponent_params,
                            role_eval_key,
                            int(args.eval_horizon),
                        )
                block_until_ready_tree(eval_outputs)
                if len(eval_trajectories) < int(args.max_eval_dumps):
                    env_index = min(
                        max(0, int(args.eval_trajectory_env_index)),
                        int(args.kernel_batch_size) - 1,
                    )
                    for role in TRAINING_ROLES:
                        if len(eval_trajectories) >= int(args.max_eval_dumps):
                            break
                        final_eval_state, eval_trace = eval_outputs[role]
                        serialized = serialize_eval_trace(
                            eval_trace,
                            final_eval_state,
                            env_index=env_index,
                            update_index=update_idx,
                        )
                        serialized["training_role"] = role
                        eval_trajectories.append(serialized)
                if mlflow is not None:
                    for role in TRAINING_ROLES:
                        final_eval_state, eval_trace = eval_outputs[role]
                        eval_episode_metrics = summarize_episode_events(
                            eval_trace.dones,
                            eval_trace.terminal_episode_steps,
                            eval_trace.pass_attempts,
                            eval_trace.completed_passes,
                            eval_trace.assists,
                            eval_trace.turnovers,
                        )
                        eval_metrics = {
                            "update_index": update_idx,
                            "mean_final_offense_score": float(
                                np.asarray(final_eval_state.offense_score).mean()
                            ),
                            "mean_final_defense_score": float(
                                np.asarray(final_eval_state.defense_score).mean()
                            ),
                            "mean_final_score_margin": float(
                                np.asarray(
                                    final_eval_state.offense_score
                                    - final_eval_state.defense_score
                                ).mean()
                            ),
                            "mean_done_rate": float(
                                np.asarray(eval_trace.dones).mean()
                            ),
                            "mean_reward": float(np.asarray(eval_trace.rewards).mean()),
                        }
                        eval_metrics.update(eval_episode_metrics)
                        eval_metrics.update(
                            summarize_lane_violation_metrics(
                                terminal_episode_steps=eval_trace.terminal_episode_steps,
                                offensive_three_seconds=eval_trace.offensive_three_seconds,
                                defensive_lane_violations=eval_trace.defensive_lane_violations,
                            )
                        )
                        eval_rebound_attempts = float(
                            np.asarray(
                                eval_trace.rebound_attempts, dtype=np.float32
                            ).sum()
                        )
                        eval_offensive_rebounds = float(
                            np.asarray(
                                eval_trace.offensive_rebounds, dtype=np.float32
                            ).sum()
                        )
                        eval_defensive_rebounds = float(
                            np.asarray(
                                eval_trace.defensive_rebounds, dtype=np.float32
                            ).sum()
                        )
                        eval_rebound_global_contests = float(
                            np.asarray(
                                eval_trace.rebound_global_contests, dtype=np.float32
                            ).sum()
                        )
                        eval_metrics.update(
                            {
                                "rebound_attempts": int(eval_rebound_attempts),
                                "offensive_rebounds": int(eval_offensive_rebounds),
                                "defensive_rebounds": int(eval_defensive_rebounds),
                                "offensive_rebound_rate": float(
                                    eval_offensive_rebounds
                                    / max(1.0, eval_rebound_attempts)
                                ),
                                "defensive_rebound_rate": float(
                                    eval_defensive_rebounds
                                    / max(1.0, eval_rebound_attempts)
                                ),
                                "rebound_global_contest_count": int(
                                    eval_rebound_global_contests
                                ),
                                "rebound_global_contest_rate": float(
                                    eval_rebound_global_contests
                                    / max(1.0, eval_rebound_attempts)
                                ),
                            }
                        )
                        eval_metrics.update(
                            summarize_shot_type_metrics(
                                "all",
                                shot_attempts=eval_trace.shot_attempts,
                                shot_makes=eval_trace.shot_makes,
                                shot_dunks=eval_trace.shot_dunks,
                                shot_twos=eval_trace.shot_twos,
                                shot_threes=eval_trace.shot_threes,
                            )
                        )
                        eval_metrics.update(
                            summarize_shot_type_metrics(
                                "learner",
                                shot_attempts=eval_trace.learner_shot_attempts,
                                shot_makes=eval_trace.learner_shot_makes,
                                shot_dunks=eval_trace.learner_shot_dunks,
                                shot_twos=eval_trace.learner_shot_twos,
                                shot_threes=eval_trace.learner_shot_threes,
                            )
                        )
                        eval_metrics.update(
                            summarize_shot_type_metrics(
                                "opponent",
                                shot_attempts=eval_trace.opponent_shot_attempts,
                                shot_makes=eval_trace.opponent_shot_makes,
                                shot_dunks=eval_trace.opponent_shot_dunks,
                                shot_twos=eval_trace.opponent_shot_twos,
                                shot_threes=eval_trace.opponent_shot_threes,
                            )
                        )
                        eval_metrics.update(
                            summarize_intent_metrics(
                                "offense",
                                intent_index=eval_trace.intent_index,
                                intent_active=eval_trace.intent_active,
                                intent_age=eval_trace.intent_age,
                                intent_commitment_remaining=eval_trace.intent_commitment_remaining,
                                intent_visible_to_defense=eval_trace.intent_visible_to_defense,
                            )
                        )
                        eval_metrics.update(
                            summarize_intent_metrics(
                                "defense",
                                intent_index=eval_trace.defense_intent_index,
                                intent_active=eval_trace.defense_intent_active,
                                intent_age=eval_trace.defense_intent_age,
                                intent_commitment_remaining=eval_trace.defense_intent_commitment_remaining,
                            )
                        )
                        _log_mlflow_metrics(
                            mlflow,
                            eval_metrics,
                            step=update_idx,
                            prefix=f"jax/eval_{role}",
                        )
                progress.update(1)
                progress.set_postfix_str(f"eval:{update_idx}", refresh=False)

            should_deploy_eval = (
                int(args.eval_deploy_every_updates) > 0
                and int(args.eval_deploy_batches) > 0
                and (
                    update_idx == int(args.num_updates)
                    or update_idx % int(args.eval_deploy_every_updates) == 0
                )
            )
            if should_deploy_eval:
                deploy_outputs = []
                deploy_seed_key = jax.random.PRNGKey(int(args.eval_deploy_seed))
                for deploy_batch_index in range(int(args.eval_deploy_batches)):
                    reset_key = jax.random.fold_in(
                        deploy_seed_key,
                        (2 * deploy_batch_index),
                    )
                    deploy_rollout_key = jax.random.fold_in(
                        deploy_seed_key,
                        (2 * deploy_batch_index) + 1,
                    )
                    deploy_reset_keys = jax.random.split(
                        reset_key,
                        int(args.kernel_batch_size),
                    )
                    deploy_initial_state = reset_batch_minimal(
                        eval_statics["offense"],
                        deploy_reset_keys,
                        jax,
                        jnp,
                    )
                    deploy_output = deploy_eval_runner(
                        eval_statics["offense"],
                        deploy_initial_state,
                        params,
                        deploy_rollout_key,
                        int(args.eval_deploy_horizon),
                        selector_multiselect_enabled,
                        selector_min_play_steps,
                    )
                    block_until_ready_tree(deploy_output)
                    deploy_outputs.append(deploy_output)
                deploy_metrics = summarize_deploy_eval_outputs(
                    deploy_outputs,
                    batch_size=int(args.kernel_batch_size),
                )
                deploy_metrics.update(
                    {
                        "update_index": int(update_idx),
                        "horizon": int(args.eval_deploy_horizon),
                        "seed": int(args.eval_deploy_seed),
                        "action_argmax_offense": 1,
                        "action_argmax_defense": 1,
                        "same_policy_both_sides": 1,
                    }
                )
                deploy_eval_history.append(deploy_metrics)
                if mlflow is not None:
                    _log_mlflow_metrics(
                        mlflow,
                        _filter_mlflow_deploy_metrics(
                            deploy_metrics,
                            profile=str(getattr(args, "mlflow_metric_profile", "core")),
                        ),
                        step=update_idx,
                        prefix="jax/eval_deploy",
                    )
                progress.update(1)
                progress.set_postfix_str(f"deploy_eval:{update_idx}", refresh=False)

            last_completed_update = int(update_idx)
            control_request = read_control_request(getattr(args, "control_file", ""))
            force_control_checkpoint = control_request is not None
            checkpoint_enabled = bool(checkpoint_dir) or mlflow is not None
            is_historical_milestone = int(update_idx) in historical_milestone_set
            should_checkpoint = checkpoint_enabled and (
                update_idx == int(args.num_updates)
                or int(update_idx) in periodic_checkpoint_updates
                or is_historical_milestone
                or force_control_checkpoint
            )
            if should_checkpoint:
                saved_candidate_info = None
                if opponent_pool_enabled and pinned_opponent_pool_enabled:
                    _add_opponent_candidate(
                        opponent_candidates,
                        params=params,
                        info={
                            "source": "self_checkpoint",
                            "update_index": int(update_idx),
                            "candidate_kind": "self_checkpoint",
                        },
                    )
                opponent_pool_checkpoint_state = (
                    _build_opponent_pool_checkpoint_state(
                        opponent_candidates,
                        opponent_assignments,
                        opponent_deterministic_modes,
                        opponent_rng=opponent_rng,
                        enabled=pinned_opponent_pool_enabled,
                        jax=jax,
                        jnp=jnp,
                    )
                    if pinned_opponent_pool_enabled
                    else None
                )
                intent_discriminator_state = None
                if intent_disc_enabled and intent_disc_spec is not None:
                    intent_discriminator_state = {
                        "enabled": True,
                        "config": intent_disc_spec.asdict(),
                        "params": intent_disc_params,
                        "opt_state": intent_disc_opt_state,
                        "bonus_stats": dict(intent_bonus_stats),
                    }
                if checkpoint_dir:
                    latest_checkpoint_path, numbered_checkpoint_path = (
                        _save_training_checkpoint(
                            checkpoint_dir=checkpoint_dir,
                            update_index=update_idx,
                            trainer_config=trainer_config,
                            spec=spec,
                            args=args,
                            params=params,
                            opt_state=opt_state,
                            selector_opt_state=selector_opt_state,
                            current_state=current_states,
                            eval_initial_state=eval_initial_states,
                            base_key=base_key,
                            eval_trajectories=eval_trajectories,
                            last_metrics=last_metrics,
                            opponent_info=active_opponent_info,
                            opponent_pool_state=opponent_pool_checkpoint_state,
                            intent_discriminator_state=intent_discriminator_state,
                            play_name_metadata=play_name_metadata,
                        )
                    )
                    saved_candidate_info = {
                        "source": "local_checkpoint",
                        "checkpoint_path": str(numbered_checkpoint_path),
                        "latest_checkpoint_path": str(latest_checkpoint_path),
                        "update_index": int(update_idx),
                    }
                    if mlflow is not None:
                        latest_checkpoint_artifact_path = (
                            _log_mlflow_checkpoint_artifacts(
                                mlflow,
                                numbered_checkpoint_path=numbered_checkpoint_path,
                                update_index=update_idx,
                            )
                        )
                        if latest_intent_sample_payload is not None:
                            artifact = _log_mlflow_intent_sample_artifact(
                                mlflow,
                                sample_payload=latest_intent_sample_payload,
                                update_index=update_idx,
                            )
                            if artifact is not None:
                                intent_sample_artifacts.append(artifact)
                        saved_candidate_info.update(
                            {
                                "source": "mlflow",
                                "artifact_path": latest_checkpoint_artifact_path,
                            }
                        )
                    elif latest_intent_sample_payload is not None:
                        artifact = _save_local_intent_sample_artifact(
                            checkpoint_dir=checkpoint_dir,
                            sample_payload=latest_intent_sample_payload,
                            update_index=update_idx,
                        )
                        if artifact is not None:
                            intent_sample_artifacts.append(artifact)
                elif mlflow is not None:
                    with TemporaryDirectory(
                        prefix="basketworld_jax_ckpt_"
                    ) as staging_dir:
                        latest_checkpoint_path, numbered_checkpoint_path = (
                            _save_training_checkpoint(
                                checkpoint_dir=staging_dir,
                                update_index=update_idx,
                                trainer_config=trainer_config,
                                spec=spec,
                                args=args,
                                params=params,
                                opt_state=opt_state,
                                selector_opt_state=selector_opt_state,
                                current_state=current_states,
                                eval_initial_state=eval_initial_states,
                                base_key=base_key,
                                eval_trajectories=eval_trajectories,
                                last_metrics=last_metrics,
                                opponent_info=active_opponent_info,
                                opponent_pool_state=opponent_pool_checkpoint_state,
                                intent_discriminator_state=intent_discriminator_state,
                                play_name_metadata=play_name_metadata,
                            )
                        )
                        latest_checkpoint_artifact_path = (
                            _log_mlflow_checkpoint_artifacts(
                                mlflow,
                                numbered_checkpoint_path=numbered_checkpoint_path,
                                update_index=update_idx,
                            )
                        )
                        if latest_intent_sample_payload is not None:
                            artifact = _log_mlflow_intent_sample_artifact(
                                mlflow,
                                sample_payload=latest_intent_sample_payload,
                                update_index=update_idx,
                            )
                            if artifact is not None:
                                intent_sample_artifacts.append(artifact)
                        saved_candidate_info = {
                            "source": "mlflow",
                            "artifact_path": latest_checkpoint_artifact_path,
                            "update_index": int(update_idx),
                        }
                    latest_checkpoint_path = None
                if opponent_pool_enabled and saved_candidate_info is not None:
                    if pinned_opponent_pool_enabled:
                        opponent_candidates[-1]["info"] = {
                            **saved_candidate_info,
                            "candidate_kind": "self_checkpoint",
                        }
                        opponent_params = None
                        grouped_opponent_params = None
                        active_opponent_info = {
                            "source": "pinned_pool",
                            "candidate_count": len(opponent_candidates),
                        }
                    else:
                        _add_opponent_candidate(
                            opponent_candidates,
                            params=params,
                            info={
                                **saved_candidate_info,
                                "candidate_kind": "self_checkpoint",
                            },
                        )
                    if (
                        not pinned_opponent_pool_enabled
                        and grouped_opponent_sampling_enabled
                    ):
                        grouped_opponent_params, active_opponent_info = (
                            _select_grouped_opponents_from_pool(
                                opponent_candidates,
                                args=args,
                                rng=opponent_rng,
                                jax=jax,
                                jnp=jnp,
                            )
                        )
                        opponent_params = None
                    elif not pinned_opponent_pool_enabled:
                        opponent_params, active_opponent_info = (
                            _select_opponent_from_pool(
                                opponent_candidates,
                                args=args,
                                rng=opponent_rng,
                            )
                        )
                        grouped_opponent_params = None
                if is_historical_milestone:
                    if (
                        saved_candidate_info is None
                        or historical_match_eval_runner is None
                    ):
                        raise RuntimeError(
                            "Historical-opponent evaluation requires a persisted "
                            "milestone checkpoint and compiled match runner."
                        )
                    historical_milestone_params[int(update_idx)] = params
                    historical_milestone_records[int(update_idx)] = {
                        **saved_candidate_info,
                        "update_index": int(update_idx),
                        "pinned": True,
                        "forced_checkpoint": bool(
                            update_idx != int(args.num_updates)
                            and int(update_idx) not in periodic_checkpoint_updates
                        ),
                    }
                    for opponent_update in sorted(historical_milestone_params):
                        match = _run_historical_match_evaluation(
                            runner=historical_match_eval_runner,
                            statics=eval_statics,
                            candidate_params=params,
                            opponent_params=historical_milestone_params[
                                opponent_update
                            ],
                            candidate_update=int(update_idx),
                            opponent_update=int(opponent_update),
                            args=args,
                            jax=jax,
                            jnp=jnp,
                        )
                        historical_eval_history.append(match)
                        progress.update(1)
                        progress.set_postfix_str(
                            f"historical_eval:{update_idx}v{opponent_update}",
                            refresh=False,
                        )
                    historical_eval_update_artifact_paths[int(update_idx)] = (
                        _persist_historical_eval_update_artifact(
                            checkpoint_dir=checkpoint_dir,
                            mlflow=mlflow,
                            candidate_update=update_idx,
                            payload=_historical_eval_update_artifact_payload(
                                candidate_update=update_idx,
                                milestone_records=historical_milestone_records,
                                match_history=historical_eval_history,
                            ),
                        )
                    )
                    historical_eval_artifact_path = _persist_historical_eval_artifact(
                        checkpoint_dir=checkpoint_dir,
                        mlflow=mlflow,
                        payload=_historical_eval_artifact_payload(
                            milestone_updates=historical_milestone_updates,
                            milestone_records=historical_milestone_records,
                            match_history=historical_eval_history,
                            update_artifact_paths=historical_eval_update_artifact_paths,
                        ),
                    )
                _print_checkpoint_summary(
                    update_index=update_idx,
                    last_metrics=last_metrics,
                    latest_checkpoint_path=latest_checkpoint_path,
                    latest_checkpoint_artifact_path=latest_checkpoint_artifact_path,
                )

            if control_request is not None:
                acknowledge_control_request(
                    getattr(args, "control_file", ""),
                    control_request,
                    update_index=update_idx,
                    checkpoint_path=latest_checkpoint_path,
                )
                pause_requested = control_request[
                    "action"
                ] == "pause" and update_idx < int(args.num_updates)

            write_training_status(
                getattr(args, "training_status_file", ""),
                state="paused" if pause_requested else "running",
                completed_updates=last_completed_update,
                target_updates=int(args.num_updates),
                mlflow_run_id=active_mlflow_run_id,
                checkpoint_path=latest_checkpoint_path,
                metrics=last_metrics,
            )
            if pause_requested:
                if mlflow is not None:
                    mlflow.set_tag("basketworld.training_state", "paused")
                    mlflow.set_tag(
                        "basketworld.paused_at_update", str(last_completed_update)
                    )
                break

        progress.close()

        result = {
            "script": "basketworld_jax/train/main.py",
            "status": "paused" if pause_requested else "train_loop",
            "completed_updates": int(last_completed_update),
            "resumed_from_checkpoint": resume_checkpoint or None,
            "resume_reset_env_state": bool(
                getattr(args, "resume_reset_env_state", False)
                or (
                    continuation_checkpoint_info is not None
                    and not bool(getattr(args, "continue_preserve_env_state", False))
                )
            ),
            "resume_reset_intent_discriminator_state": bool(
                getattr(args, "resume_reset_intent_discriminator_state", False)
                or (
                    continuation_checkpoint_info is not None
                    and not bool(
                        getattr(
                            args, "continue_preserve_intent_discriminator_state", False
                        )
                    )
                )
            ),
            "trainer_config": _checkpoint_trainer_config_from_args(
                trainer_config,
                args,
                spec=spec,
            ),
            "frozen_config": {
                key: to_builtin(getattr(args, key)) for key in TRAIN_FROZEN_VALUES
            },
            "env_config": _jax_env_config_from_args(args),
            "policy_spec": asdict(spec),
            "training_player_ids": {
                role: [int(v) for v in ids.tolist()]
                for role, ids in training_player_ids_by_role.items()
            },
            "train_history": train_history,
            "deploy_eval_history": deploy_eval_history,
            "historical_eval_milestones": list(historical_milestone_updates),
            "historical_eval_milestone_checkpoints": {
                str(int(update)): dict(record)
                for update, record in sorted(historical_milestone_records.items())
            },
            "historical_eval_update_artifacts": {
                str(int(update)): str(path)
                for update, path in sorted(
                    historical_eval_update_artifact_paths.items()
                )
            },
            "historical_eval_history": historical_eval_history,
            "historical_eval_artifact_path": historical_eval_artifact_path,
            "eval_trajectories": eval_trajectories,
            "final_metrics": last_metrics,
            "latest_checkpoint_path": latest_checkpoint_path,
            "latest_checkpoint_artifact_path": latest_checkpoint_artifact_path,
            "intent_discriminator_config": (
                intent_disc_spec.asdict() if intent_disc_spec is not None else None
            ),
            "play_name_metadata": play_name_metadata,
            "play_name_map": {
                str(int(item["intent_index"])): str(item["play_name"])
                for item in play_name_metadata.get("play_names", [])
                if isinstance(item, dict)
            },
            "intent_sample_artifacts": intent_sample_artifacts,
            "active_opponent": active_opponent_info,
            "opponent_pool_size": len(opponent_candidates),
            "continuation_checkpoint": continuation_checkpoint_info,
            "continuation_opponent_pool": continuation_pool_info,
            "summary_artifact_path": (
                TRAIN_LOOP_SUMMARY_ARTIFACT_PATH if mlflow is not None else None
            ),
            "next_step": "run a longer learnability check and inspect eval trajectories for behavior changes",
        }
        if mlflow is not None:
            _log_mlflow_train_loop_summary(mlflow, result)
            if not pause_requested:
                mlflow.set_tag("basketworld.training_state", "completed")
        if not pause_requested:
            write_training_status(
                getattr(args, "training_status_file", ""),
                state="completed",
                completed_updates=last_completed_update,
                target_updates=int(args.num_updates),
                mlflow_run_id=active_mlflow_run_id,
                checkpoint_path=latest_checkpoint_path,
                metrics=last_metrics,
            )
        return result


def run_train_scaffold(args) -> dict[str, Any]:
    validate_train_args(args)
    jax, jnp = ensure_jax_available("basketworld_jax/train/main.py")
    static, state = sample_state_batch(args, xp=jnp)
    training_player_ids = training_player_ids_from_static(static)
    training_player_ids_jnp = jnp.asarray(training_player_ids, dtype=jnp.int32)

    flat_obs = build_policy_observation_batch(
        static,
        state,
        jnp,
        model_type=_policy_model_type(args),
        rebound_win_prob_features=bool(
            getattr(args, "rebound_win_prob_features", False)
        ),
        rebound_target_observation_features=bool(
            getattr(args, "rebound_target_observation_features", True)
        ),
        multi_possession_features=bool(getattr(args, "enable_multi_possession", False)),
    )
    policy_intent_context = build_policy_intent_context_batch(static, state, jnp)
    action_masks = build_action_masks_batch(static, state, jnp)[
        :, training_player_ids_jnp, :
    ]
    flat_obs_np = np.asarray(jax.device_get(flat_obs), dtype=np.float32)
    action_masks_np = np.asarray(jax.device_get(action_masks), dtype=np.int8)
    spec = _build_policy_spec(args, static, flat_obs_np, action_masks_np)
    params = init_actor_critic_params(
        jax,
        jnp,
        spec,
        seed=int(args.policy_seed),
    )
    trainer_config = build_trainer_config(args)
    update_runner, optimizer_transform = build_jitted_ppo_update_runner(
        jax, jnp, spec, trainer_config
    )
    opt_state = init_optimizer_state(optimizer_transform, params)
    runner = build_jitted_actor_critic_runner(jax, jnp, spec)
    rollout_runner = build_compiled_rollout_runner(jax, jnp, spec)

    total_iters = 3 * (int(args.warmup_iters) + int(args.benchmark_iters))
    progress = build_progress(
        total=total_iters,
        desc="jax_train:actor_critic",
        disable=bool(args.no_progress),
        unit="iter",
    )

    sample_key = jax.random.PRNGKey(int(args.policy_seed) + 11)
    final_out = None
    for idx in range(int(args.warmup_iters)):
        sample_key = jax.random.fold_in(sample_key, idx)
        final_out = runner(
            params, flat_obs, action_masks, policy_intent_context, sample_key
        )
        jax.block_until_ready(final_out["values"])
        progress.update(1)
        progress.set_postfix_str("forward_warmup", refresh=False)

    timed_ns = 0
    for idx in range(int(args.benchmark_iters)):
        sample_key = jax.random.fold_in(sample_key, idx + 10_000)
        start_ns = perf_counter_ns()
        final_out = runner(
            params, flat_obs, action_masks, policy_intent_context, sample_key
        )
        jax.block_until_ready(final_out["values"])
        timed_ns += perf_counter_ns() - start_ns
        progress.update(1)
        progress.set_postfix_str("forward_benchmark", refresh=False)

    rollout_key = jax.random.PRNGKey(int(args.policy_seed) + 101)
    if int(args.warmup_iters) > 0:
        benchmark_compiled_rollout(
            jax,
            rollout_runner,
            static,
            state,
            params,
            rollout_key,
            batch_size=int(args.kernel_batch_size),
            horizon=int(args.rollout_horizon),
            iterations=int(args.warmup_iters),
            progress=progress,
        )
    rollout_metrics, rollout_out = benchmark_compiled_rollout(
        jax,
        rollout_runner,
        static,
        state,
        params,
        rollout_key,
        batch_size=int(args.kernel_batch_size),
        horizon=int(args.rollout_horizon),
        iterations=int(args.benchmark_iters),
        progress=progress,
    )

    total_states = int(args.kernel_batch_size) * int(args.benchmark_iters)
    total_seconds = max(timed_ns / 1e9, 1e-12)
    ppo_batch = build_ppo_batch(rollout_out, trainer_config, jax, jnp)
    update_key = jax.random.PRNGKey(int(args.policy_seed) + 202)
    if int(args.warmup_iters) > 0:
        _, _, _ = benchmark_update_runner(
            jax,
            update_runner,
            params,
            opt_state,
            ppo_batch,
            update_key,
            iterations=int(args.warmup_iters),
            progress=progress,
        )
    update_metrics, updated_params, updated_opt_state = benchmark_update_runner(
        jax,
        update_runner,
        params,
        opt_state,
        ppo_batch,
        jax.random.fold_in(update_key, 50_000),
        iterations=int(args.benchmark_iters),
        progress=progress,
    )
    del updated_params, updated_opt_state
    progress.close()

    result = {
        "script": "basketworld_jax/train/main.py",
        "status": "trajectory_and_update_scaffold",
        "trainer_config": _checkpoint_trainer_config_from_args(
            trainer_config,
            args,
            spec=spec,
        ),
        "frozen_config": {
            key: to_builtin(getattr(args, key)) for key in TRAIN_FROZEN_VALUES
        },
        "env_config": _jax_env_config_from_args(args),
        "policy_spec": asdict(spec),
        "steps_per_update": int(args.kernel_batch_size) * int(args.rollout_horizon),
        "actor_critic_forward_states_per_sec": float(total_states / total_seconds),
        "actor_critic_mean_batch_latency_ms": float(
            (timed_ns / 1e6) / max(1, int(args.benchmark_iters))
        ),
        "rollout_trajectory_states_per_sec": float(rollout_metrics["states_per_sec"]),
        "rollout_mean_latency_ms": float(rollout_metrics["mean_rollout_latency_ms"]),
        "ppo_update_updates_per_sec": float(update_metrics["updates_per_sec"]),
        "ppo_update_mean_latency_ms": float(update_metrics["mean_update_latency_ms"]),
        "end_to_end_steps_per_sec": float(
            (int(args.kernel_batch_size) * int(args.rollout_horizon))
            / max(
                (
                    float(rollout_metrics["mean_rollout_latency_ms"])
                    + float(update_metrics["mean_update_latency_ms"])
                )
                / 1000.0,
                1e-12,
            )
        ),
        "ppo_update_final_metrics": update_metrics["final_metrics"],
        "trajectory_spec": {
            "flat_obs_shape": list(flat_obs_np.shape),
            "action_mask_shape": list(action_masks_np.shape),
            "action_shape": [
                int(args.kernel_batch_size),
                int(spec.training_player_count),
            ],
            "full_action_shape": [
                int(args.kernel_batch_size),
                int(static.role_encoding.shape[0]),
            ],
            "value_shape": [int(args.kernel_batch_size)],
            "log_prob_shape": [
                int(args.kernel_batch_size),
                int(spec.training_player_count),
            ],
            "rollout_horizon": int(args.rollout_horizon),
            "trajectory_flat_obs_shape": list(
                np.asarray(rollout_out.trajectory.flat_obs).shape
            ),
            "trajectory_policy_intent_index_shape": list(
                np.asarray(rollout_out.trajectory.policy_intent_index).shape
            ),
            "trajectory_policy_intent_gate_shape": list(
                np.asarray(rollout_out.trajectory.policy_intent_gate).shape
            ),
            "trajectory_action_mask_shape": list(
                np.asarray(rollout_out.trajectory.action_mask).shape
            ),
            "trajectory_actions_shape": list(
                np.asarray(rollout_out.trajectory.actions).shape
            ),
            "trajectory_full_actions_shape": list(
                np.asarray(rollout_out.trajectory.full_actions).shape
            ),
            "trajectory_log_prob_shape": list(
                np.asarray(rollout_out.trajectory.selected_log_probs).shape
            ),
            "trajectory_values_shape": list(
                np.asarray(rollout_out.trajectory.values).shape
            ),
            "trajectory_rewards_shape": list(
                np.asarray(rollout_out.trajectory.rewards).shape
            ),
            "trajectory_dones_shape": list(
                np.asarray(rollout_out.trajectory.dones).shape
            ),
            "trajectory_pass_attempts_shape": list(
                np.asarray(rollout_out.trajectory.pass_attempts).shape
            ),
            "trajectory_completed_passes_shape": list(
                np.asarray(rollout_out.trajectory.completed_passes).shape
            ),
            "trajectory_assists_shape": list(
                np.asarray(rollout_out.trajectory.assists).shape
            ),
            "trajectory_turnovers_shape": list(
                np.asarray(rollout_out.trajectory.turnovers).shape
            ),
            "trajectory_offensive_three_seconds_shape": list(
                np.asarray(rollout_out.trajectory.offensive_three_seconds).shape
            ),
            "trajectory_defensive_lane_violations_shape": list(
                np.asarray(rollout_out.trajectory.defensive_lane_violations).shape
            ),
            "trajectory_terminal_episode_steps_shape": list(
                np.asarray(rollout_out.trajectory.terminal_episode_steps).shape
            ),
            "trajectory_offense_score_delta_shape": list(
                np.asarray(rollout_out.trajectory.offense_score_delta).shape
            ),
            "trajectory_defense_score_delta_shape": list(
                np.asarray(rollout_out.trajectory.defense_score_delta).shape
            ),
            "bootstrap_values_shape": list(
                np.asarray(rollout_out.bootstrap_values).shape
            ),
            "ppo_batch_flat_obs_shape": list(np.asarray(ppo_batch.flat_obs).shape),
            "ppo_batch_policy_intent_index_shape": list(
                np.asarray(ppo_batch.policy_intent_index).shape
            ),
            "ppo_batch_policy_intent_gate_shape": list(
                np.asarray(ppo_batch.policy_intent_gate).shape
            ),
            "ppo_batch_action_mask_shape": list(
                np.asarray(ppo_batch.action_mask).shape
            ),
            "ppo_batch_actions_shape": list(np.asarray(ppo_batch.actions).shape),
            "ppo_batch_old_log_probs_shape": list(
                np.asarray(ppo_batch.old_selected_log_probs).shape
            ),
            "ppo_batch_advantages_shape": list(np.asarray(ppo_batch.advantages).shape),
            "ppo_batch_returns_shape": list(np.asarray(ppo_batch.returns).shape),
        },
        "training_player_ids": [int(v) for v in training_player_ids.tolist()],
        "action_preview": (
            np.asarray(final_out["sampled_actions"][:3], dtype=np.int32)
            if final_out is not None
            else None
        ),
        "value_preview": (
            np.asarray(final_out["values"][:3], dtype=np.float32)
            if final_out is not None
            else None
        ),
        "selected_log_prob_preview": (
            np.asarray(final_out["selected_log_probs"][:3], dtype=np.float32)
            if final_out is not None
            else None
        ),
        "next_step": "measure short multi-update training behavior and add eval trajectory dumps",
    }
    return result


def main(argv=None):
    args = parse_args(argv)
    if bool(getattr(args, "dump_config_schema", False)):
        print(json.dumps(build_train_config_schema(args), sort_keys=True))
        return
    if bool(args.run_train_loop):
        result = run_training_loop(args)
    else:
        result = run_train_scaffold(args)

    if bool(args.run_train_loop):
        print("JAX trainer loop")
        print(f"policy_spec: {result['policy_spec']}")
        print(f"logged_train_entries: {len(result['train_history'])}")
        if result["final_metrics"] is not None:
            print(f"final_metrics: {result['final_metrics']}")
        print(f"eval_trajectory_dumps: {len(result['eval_trajectories'])}")
    else:
        print("JAX trainer scaffold")
        print(f"policy_spec: {result['policy_spec']}")
        print(
            "actor_critic_forward:"
            f" states_per_sec={result['actor_critic_forward_states_per_sec']:.2f}"
            f" mean_batch_latency_ms={result['actor_critic_mean_batch_latency_ms']:.4f}"
        )
        print(
            "compiled_rollout_trajectory:"
            f" states_per_sec={result['rollout_trajectory_states_per_sec']:.2f}"
            f" mean_rollout_latency_ms={result['rollout_mean_latency_ms']:.4f}"
        )
        print(
            "end_to_end:"
            f" steps_per_update={result['steps_per_update']}"
            f" steps_per_sec={result['end_to_end_steps_per_sec']:.2f}"
        )
        print(
            "ppo_update:"
            f" updates_per_sec={result['ppo_update_updates_per_sec']:.2f}"
            f" mean_update_latency_ms={result['ppo_update_mean_latency_ms']:.4f}"
        )
        print(f"trajectory_spec: {result['trajectory_spec']}")


if __name__ == "__main__":
    main()
