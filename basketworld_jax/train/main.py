from __future__ import annotations

import argparse
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
from basketworld_jax.checkpoints import (
    build_checkpoint_paths,
    build_checkpoint_payload,
    load_checkpoint,
    save_checkpoint,
)
from basketworld_jax.config import TRAIN_FROZEN_VALUES
from basketworld_jax.env import (
    build_action_masks_batch,
    build_flat_observation_batch,
    reset_batch_minimal,
    sample_state_batch,
)
from basketworld_jax.models import (
    ActorCriticSpec,
    build_actor_critic_spec,
    init_actor_critic_params,
)
from basketworld_jax.optim import init_optimizer_state
from basketworld_jax.train.cli import (
    build_parser,
    build_progress,
    ensure_jax_available,
    to_builtin,
    write_json,
)
from basketworld_jax.train.types import (
    TrainerConfig,
    build_ppo_batch,
    concatenate_ppo_batches,
)
from basketworld_jax.train.runtime import (
    benchmark_compiled_rollout,
    benchmark_update_runner,
    block_until_ready_tree,
    build_compiled_eval_runner,
    build_compiled_frozen_opponent_eval_runner,
    build_compiled_frozen_opponent_rollout_runner,
    build_compiled_grouped_opponent_eval_runner,
    build_compiled_grouped_opponent_rollout_runner,
    build_compiled_rollout_runner,
    build_jitted_actor_critic_runner,
    build_jitted_ppo_update_runner,
    concatenate_rollout_outputs,
    serialize_eval_trace,
    summarize_episode_events,
    summarize_training_step,
    training_player_ids_from_static,
)


TRAINING_ROLES = ("offense", "defense")
JAX_ALLOWED_ENV_OVERRIDE_KEYS = frozenset(
    {
        "layup_pct",
        "three_pt_pct",
        "dunk_pct",
    }
)
JAX_ENV_MLFLOW_PARAM_KEYS = (
    "training_team",
    "players",
    "court_rows",
    "court_cols",
    "shot_clock",
    "min_shot_clock",
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
    "include_hoop_vector",
    "enable_phi_shaping",
    "phi_aggregation_mode",
    "phi_use_ball_handler_only",
    "pass_reward",
    "potential_assist_pct",
    "full_assist_bonus_pct",
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


def parse_args(argv=None):
    argv_list = list(sys.argv[1:] if argv is None else argv)
    _reject_legacy_opponent_flag(argv_list)
    parser = build_parser(
        "JAX trainer: reduced actor-critic + compiled rollout path."
    )
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
        help="Number of full-batch PPO update epochs per rollout.",
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
            "Save a numbered checkpoint every N updates. Final update is always "
            "saved when checkpoint publishing is enabled."
        ),
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default="",
        help="Resume train-loop state from a saved JAX checkpoint.",
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
    return parser.parse_args(argv_list)


def _values_match(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        return np.isclose(float(actual), float(expected), atol=1e-8, rtol=0.0)
    return actual == expected


def validate_train_args(args) -> None:
    mismatches: list[str] = []
    for key, expected in TRAIN_FROZEN_VALUES.items():
        if key in JAX_ALLOWED_ENV_OVERRIDE_KEYS:
            continue
        actual = getattr(args, key)
        if not _values_match(actual, expected):
            mismatches.append(f"{key}={actual!r} expected {expected!r}")
    if mismatches:
        raise SystemExit(
            "JAX trainer uses a frozen reduced structural config. Unsupported overrides: "
            + ", ".join(mismatches)
        )


def _jax_env_config_from_args(args) -> dict[str, Any]:
    return {
        key: to_builtin(getattr(args, key))
        for key in JAX_ENV_MLFLOW_PARAM_KEYS
        if hasattr(args, key)
    }


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
    )


def _uses_grouped_opponent_sampling(args) -> bool:
    return bool(getattr(args, "grouped_opponent_sampling", False))


def _args_for_training_role(args, role: str):
    role_args = copy(args)
    role_args.training_team = str(role)
    return role_args


def _remaining_eval_count(*, start_update: int, num_updates: int, eval_every_updates: int) -> int:
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
                    field: _restore_like_template(restored[field], getattr(template, field))
                    for field in template._fields
                }
            )
        if isinstance(restored, (tuple, list)):
            return type(template)(
                *[
                    _restore_like_template(item, getattr(template, field))
                    for item, field in zip(restored, template._fields, strict=False)
                ]
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
    if dict(payload.get("policy_spec", {})) != expected_policy_spec:
        raise SystemExit("Resume checkpoint policy_spec does not match the current JAX run.")

    expected_frozen = {
        key: to_builtin(getattr(args, key))
        for key in TRAIN_FROZEN_VALUES
    }
    if dict(payload.get("frozen_config", {})) != expected_frozen:
        raise SystemExit("Resume checkpoint frozen_config does not match the current JAX run.")
    if "env_config" in payload and dict(payload.get("env_config", {})) != _jax_env_config_from_args(args):
        raise SystemExit("Resume checkpoint env_config does not match the current JAX run.")


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
) -> tuple[str | None, str]:
    payload = build_checkpoint_payload(
        update_index=int(update_index),
        trainer_config=asdict(trainer_config),
        policy_spec=asdict(spec),
        frozen_config={
            key: to_builtin(getattr(args, key))
            for key in TRAIN_FROZEN_VALUES
        },
        env_config=_jax_env_config_from_args(args),
        params=params,
        opt_state=opt_state,
        current_state=current_state,
        eval_initial_state=eval_initial_state,
        base_key=base_key,
        eval_trajectories=eval_trajectories,
        last_metrics=last_metrics,
        opponent_info=opponent_info,
    )
    if checkpoint_dir is None:
        raise ValueError("checkpoint_dir must not be None when saving a persistent local checkpoint.")
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
    mlflow.set_experiment(str(args.mlflow_experiment_name))
    run_name = args.mlflow_run_name
    if not run_name:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"jax-train-{mode}-{timestamp}"
    context = mlflow.start_run(run_name=run_name)
    return mlflow, context


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


def _resolve_mlflow_checkpoint_artifact(client, run_id: str, artifact_hint: str | None) -> str:
    artifacts = client.list_artifacts(run_id, "models")
    choices = [item.path for item in artifacts if _is_jax_checkpoint_artifact(str(item.path))]
    if not choices:
        raise SystemExit(f"No JAX checkpoint artifacts found under models/ for MLflow run {run_id!r}.")

    hint = str(artifact_hint or "").strip()
    if hint:
        for choice in choices:
            if choice == hint or choice.endswith(hint):
                return choice
        raise SystemExit(f"JAX checkpoint artifact {hint!r} was not found in MLflow run {run_id!r}.")

    tags = dict(getattr(getattr(client.get_run(run_id), "data", None), "tags", {}) or {})
    tagged = str(tags.get("jax_latest_checkpoint_artifact", "")).strip()
    if tagged and tagged in choices:
        return tagged

    return sorted(choices, key=_checkpoint_artifact_sort_key)[-1]


def _load_frozen_opponent_payload(args) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    checkpoint_path = str(getattr(args, "frozen_opponent_checkpoint", "") or "").strip()
    run_id = str(getattr(args, "frozen_opponent_run_id", "") or "").strip()
    artifact_hint = str(getattr(args, "frozen_opponent_artifact", "") or "").strip()

    if checkpoint_path and run_id:
        raise SystemExit("Use either --frozen-opponent-checkpoint or --frozen-opponent-run-id, not both.")
    if artifact_hint and not run_id:
        raise SystemExit("--frozen-opponent-artifact requires --frozen-opponent-run-id.")
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


def _sample_geometric_candidate_index(count: int, beta: float, rng: np.random.Generator) -> int:
    if count <= 1:
        return 0
    beta = float(beta)
    if beta >= 1.0:
        return count - 1
    beta = max(beta, 0.0)
    weights = np.asarray(
        [
            (1.0 - beta) * (beta ** (count - idx))
            for idx in range(1, count + 1)
        ],
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
        "batch_group_size": int(int(getattr(args, "kernel_batch_size")) // len(chosen_candidates)),
        "candidate_count": int(len(candidates)),
        "unique_update_count": int(len(set(update_indices))),
        "latest_update_index": int(max(update_indices)) if update_indices else 0,
        "groups": groups,
    }


def _log_mlflow_params(mlflow, args, trainer_config: TrainerConfig, spec: ActorCriticSpec) -> None:
    params = {
        "jax/script": "basketworld_jax/train/main.py",
        "jax/mode": "train_loop" if bool(args.run_train_loop) else "scaffold",
        "jax/kernel_batch_size": int(args.kernel_batch_size),
        "jax/rollout_horizon": int(args.rollout_horizon),
        "jax/num_updates": int(args.num_updates),
        "jax/policy_update_epochs": int(args.policy_update_epochs),
        "jax/log_every_updates": int(args.log_every_updates),
        "jax/eval_every_updates": int(args.eval_every_updates),
        "jax/eval_horizon": int(args.eval_horizon),
        "jax/learning_rate": float(trainer_config.learning_rate),
        "jax/gamma": float(trainer_config.gamma),
        "jax/gae_lambda": float(trainer_config.gae_lambda),
        "jax/ppo_clip_range": float(trainer_config.ppo_clip_range),
        "jax/value_coef": float(trainer_config.value_coef),
        "jax/entropy_coef": float(trainer_config.entropy_coef),
        "jax/policy_hidden_dims": ",".join(str(v) for v in spec.hidden_dims),
        "jax/flat_obs_dim": int(spec.flat_obs_dim),
        "jax/training_player_count": int(spec.training_player_count),
        "jax/action_dim_per_player": int(spec.action_dim_per_player),
        "jax/pass_mode": str(getattr(args, "pass_mode")),
        "jax/use_set_obs": bool(getattr(args, "use_set_obs")),
        "jax/training_team": str(getattr(args, "training_team")),
        "jax/checkpoint_every_updates": int(args.checkpoint_every_updates),
        "jax/frozen_opponent_checkpoint": str(getattr(args, "frozen_opponent_checkpoint", "") or ""),
        "jax/frozen_opponent_run_id": str(getattr(args, "frozen_opponent_run_id", "") or ""),
        "jax/frozen_opponent_artifact": str(getattr(args, "frozen_opponent_artifact", "") or ""),
        "jax/opponent_pool_enabled": not bool(getattr(args, "disable_opponent_pool", False)),
        "jax/opponent_pool_size": int(getattr(args, "opponent_pool_size", 10)),
        "jax/opponent_pool_beta": float(getattr(args, "opponent_pool_beta", 0.7)),
        "jax/opponent_pool_exploration": float(getattr(args, "opponent_pool_exploration", 0.0)),
        "jax/grouped_opponent_sampling": _uses_grouped_opponent_sampling(args),
        "jax/opponent_group_count": int(getattr(args, "opponent_group_count", 8)),
    }
    for key, value in _jax_env_config_from_args(args).items():
        params[f"jax/env/{key}"] = value
    mlflow.log_params(params)


def _log_mlflow_metrics(mlflow, metrics: dict[str, Any], *, step: int, prefix: str) -> None:
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            mlflow.log_metric(f"{prefix}/{key}", float(value), step=int(step))


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


def _summarize_role_rollout_metrics(role: str, rollout) -> dict[str, Any]:
    rewards = np.asarray(rollout.trajectory.rewards, dtype=np.float32)
    dones = np.asarray(rollout.trajectory.dones, dtype=np.float32)
    terminal_steps = np.asarray(rollout.trajectory.terminal_episode_steps, dtype=np.int32)
    offense_score_delta = np.asarray(rollout.trajectory.offense_score_delta, dtype=np.float32)
    defense_score_delta = np.asarray(rollout.trajectory.defense_score_delta, dtype=np.float32)

    completed_episodes = int((terminal_steps > 0).sum())
    learner_reward_total = float(rewards.sum())
    learner_reward_mean = float(rewards.mean())
    opponent_reward_total = -learner_reward_total
    opponent_reward_mean = -learner_reward_mean
    offense_points_total = float(offense_score_delta.sum())
    defense_points_total = float(defense_score_delta.sum())
    if role == "offense":
        learner_points_total = offense_points_total
        opponent_points_total = defense_points_total
    else:
        learner_points_total = defense_points_total
        opponent_points_total = offense_points_total

    return {
        f"{role}_mean_reward": learner_reward_mean,
        f"{role}_learner_mean_reward": learner_reward_mean,
        f"{role}_opponent_mean_reward": opponent_reward_mean,
        f"{role}_learner_reward_total": learner_reward_total,
        f"{role}_opponent_reward_total": opponent_reward_total,
        f"{role}_learner_reward_per_completed_episode": _safe_metric_ratio(
            learner_reward_total,
            completed_episodes,
        ),
        f"{role}_opponent_reward_per_completed_episode": _safe_metric_ratio(
            opponent_reward_total,
            completed_episodes,
        ),
        f"{role}_done_rate": float(dones.mean()),
        f"{role}_completed_episodes": int(completed_episodes),
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
        f"{role}_learner_points_per_completed_episode": _safe_metric_ratio(
            learner_points_total,
            completed_episodes,
        ),
        f"{role}_opponent_points_per_completed_episode": _safe_metric_ratio(
            opponent_points_total,
            completed_episodes,
        ),
    }


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
        ("steps_per_update", metrics.get("steps_per_update")),
        ("end_to_end_steps_per_sec", metrics.get("end_to_end_steps_per_sec")),
        ("rollout_states_per_sec", metrics.get("rollout_states_per_sec")),
        ("completed_episodes", metrics.get("completed_episodes")),
        ("mean_completed_episode_length", metrics.get("mean_completed_episode_length")),
        ("mean_pass_attempts_per_completed_episode", metrics.get("mean_pass_attempts_per_completed_episode")),
        ("mean_completed_passes_per_completed_episode", metrics.get("mean_completed_passes_per_completed_episode")),
        ("mean_assists_per_completed_episode", metrics.get("mean_assists_per_completed_episode")),
        ("mean_turnovers_per_completed_episode", metrics.get("mean_turnovers_per_completed_episode")),
        ("approx_kl", metrics.get("approx_kl")),
        ("clip_fraction", metrics.get("clip_fraction")),
        ("mean_abs_log_ratio", metrics.get("mean_abs_log_ratio")),
        ("max_abs_log_ratio", metrics.get("max_abs_log_ratio")),
        ("entropy_bonus", metrics.get("entropy_bonus")),
        ("policy_loss", metrics.get("policy_loss")),
        ("value_loss", metrics.get("value_loss")),
        ("total_loss", metrics.get("total_loss")),
        ("grad_norm", metrics.get("grad_norm")),
        ("mean_reward", metrics.get("mean_reward")),
        ("offense_learner_mean_reward", metrics.get("offense_learner_mean_reward")),
        ("defense_learner_mean_reward", metrics.get("defense_learner_mean_reward")),
        ("offense_opponent_mean_reward", metrics.get("offense_opponent_mean_reward")),
        ("defense_opponent_mean_reward", metrics.get("defense_opponent_mean_reward")),
        ("offense_learner_points_per_completed_episode", metrics.get("offense_learner_points_per_completed_episode")),
        ("defense_opponent_points_per_completed_episode", metrics.get("defense_opponent_points_per_completed_episode")),
        ("mean_return", metrics.get("mean_return")),
        ("done_rate", metrics.get("done_rate")),
        ("opponent_update_index", metrics.get("opponent_update_index")),
        ("opponent_source", metrics.get("opponent_source")),
        ("opponent_group_count", metrics.get("opponent_group_count")),
        ("opponent_unique_update_count", metrics.get("opponent_unique_update_count")),
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
    jax, jnp = ensure_jax_available("basketworld_jax/train/main.py")
    role_args = {
        role: _args_for_training_role(args, role)
        for role in TRAINING_ROLES
    }
    statics = {
        role: sample_state_batch(role_args[role], xp=jnp)[0]
        for role in TRAINING_ROLES
    }
    static = statics["offense"]
    base_key = jax.random.PRNGKey(int(args.policy_seed))
    reset_seed_key, eval_reset_seed_key, base_key = jax.random.split(base_key, 3)
    role_reset_keys = jax.random.split(reset_seed_key, len(TRAINING_ROLES))
    role_eval_reset_keys = jax.random.split(eval_reset_seed_key, len(TRAINING_ROLES))
    current_states = {}
    eval_initial_states = {}
    for role, reset_key, eval_key in zip(TRAINING_ROLES, role_reset_keys, role_eval_reset_keys, strict=True):
        initial_reset_keys = jax.random.split(reset_key, int(args.kernel_batch_size))
        current_states[role] = reset_batch_minimal(statics[role], initial_reset_keys, jax, jnp)
        eval_reset_keys = jax.random.split(eval_key, int(args.kernel_batch_size))
        eval_initial_states[role] = reset_batch_minimal(statics[role], eval_reset_keys, jax, jnp)

    training_player_ids_by_role = {
        role: training_player_ids_from_static(statics[role])
        for role in TRAINING_ROLES
    }
    training_player_ids = training_player_ids_by_role["offense"]
    training_player_ids_jnp = jnp.asarray(training_player_ids, dtype=jnp.int32)
    flat_obs = build_flat_observation_batch(static, current_states["offense"], jnp)
    action_masks = build_action_masks_batch(static, current_states["offense"], jnp)[:, training_player_ids_jnp, :]
    flat_obs_np = np.asarray(jax.device_get(flat_obs), dtype=np.float32)
    action_masks_np = np.asarray(jax.device_get(action_masks), dtype=np.int8)
    spec = build_actor_critic_spec(
        flat_obs_np,
        action_masks_np,
        hidden_dims=args.policy_hidden_dims,
    )
    trainer_config = build_trainer_config(args)
    rollout_runner = build_compiled_rollout_runner(jax, jnp, spec)
    eval_runner = build_compiled_eval_runner(jax, jnp, spec)
    frozen_rollout_runner = build_compiled_frozen_opponent_rollout_runner(jax, jnp, spec)
    frozen_eval_runner = build_compiled_frozen_opponent_eval_runner(jax, jnp, spec)
    grouped_rollout_runner = build_compiled_grouped_opponent_rollout_runner(jax, jnp, spec)
    grouped_eval_runner = build_compiled_grouped_opponent_eval_runner(jax, jnp, spec)
    update_runner, optimizer_transform = build_jitted_ppo_update_runner(jax, jnp, spec, trainer_config)
    checkpoint_dir = str(args.checkpoint_dir).strip()
    resume_checkpoint = str(args.resume_checkpoint).strip()
    latest_checkpoint_path: str | None = None
    latest_checkpoint_artifact_path: str | None = None
    frozen_opponent_payload, frozen_opponent_info = _load_frozen_opponent_payload(args)
    opponent_params = None
    grouped_opponent_params = None
    active_opponent_info = None
    opponent_candidates: list[dict[str, Any]] = []
    opponent_rng = np.random.default_rng(int(args.policy_seed) + 90_001)
    opponent_pool_enabled = not bool(getattr(args, "disable_opponent_pool", False))
    grouped_opponent_sampling_enabled = (
        opponent_pool_enabled
        and _uses_grouped_opponent_sampling(args)
    )
    if frozen_opponent_payload is not None:
        if dict(frozen_opponent_payload.get("policy_spec", {})) != asdict(spec):
            raise SystemExit("Frozen opponent policy_spec does not match the current JAX trainer policy_spec.")
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
            grouped_opponent_params, active_opponent_info = _select_grouped_opponents_from_pool(
                opponent_candidates,
                args=args,
                rng=opponent_rng,
                jax=jax,
                jnp=jnp,
            )
            opponent_params = None

    initial_params = init_actor_critic_params(
        jax,
        jnp,
        spec,
        seed=int(args.policy_seed),
    )
    initial_opt_state = init_optimizer_state(optimizer_transform, initial_params)

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
        restored_current_state = checkpoint_payload["current_state"]
        restored_eval_initial_state = checkpoint_payload["eval_initial_state"]
        if not isinstance(restored_current_state, dict) or not isinstance(restored_eval_initial_state, dict):
            raise SystemExit("Resume checkpoint does not contain mixed-role JAX train state.")
        current_states = {
            role: jax.device_put(
                _restore_like_template(restored_current_state[role], current_states[role])
            )
            for role in TRAINING_ROLES
        }
        eval_initial_states = {
            role: jax.device_put(
                _restore_like_template(restored_eval_initial_state[role], eval_initial_states[role])
            )
            for role in TRAINING_ROLES
        }
        base_key = jax.device_put(checkpoint_payload["base_key"])
        train_history = []
        eval_trajectories = list(checkpoint_payload.get("eval_trajectories", []))
        last_metrics = checkpoint_payload.get("last_metrics")
    else:
        completed_updates = 0
        params = initial_params
        opt_state = initial_opt_state
        train_history = []
        eval_trajectories = []
        last_metrics = None

    mlflow, mlflow_context = _maybe_start_mlflow_run(args, mode="train")

    with mlflow_context:
        if mlflow is not None:
            _log_mlflow_params(mlflow, args, trainer_config, spec)

        expected_evals = _remaining_eval_count(
            start_update=completed_updates,
            num_updates=int(args.num_updates),
            eval_every_updates=int(args.eval_every_updates),
        )
        progress = build_progress(
            total=(int(args.num_updates) - completed_updates) + expected_evals,
            desc="jax_train:loop",
            disable=bool(args.no_progress),
            unit="event",
        )

        for update_idx in range(completed_updates + 1, int(args.num_updates) + 1):
            base_key, *rollout_keys = jax.random.split(base_key, len(TRAINING_ROLES) + 1)
            rollout_start_ns = perf_counter_ns()
            role_rollouts = {}
            for role, rollout_key in zip(TRAINING_ROLES, rollout_keys, strict=True):
                if grouped_opponent_params is not None:
                    role_rollouts[role] = grouped_rollout_runner(
                        statics[role],
                        current_states[role],
                        params,
                        grouped_opponent_params,
                        rollout_key,
                        int(args.rollout_horizon),
                        int(active_opponent_info["group_count"]),
                    )
                elif opponent_params is None:
                    role_rollouts[role] = rollout_runner(
                        statics[role],
                        current_states[role],
                        params,
                        rollout_key,
                        int(args.rollout_horizon),
                    )
                else:
                    role_rollouts[role] = frozen_rollout_runner(
                        statics[role],
                        current_states[role],
                        params,
                        opponent_params,
                        rollout_key,
                        int(args.rollout_horizon),
                    )
            block_until_ready_tree(role_rollouts)
            rollout_elapsed_ns = perf_counter_ns() - rollout_start_ns

            role_ppo_batches = [
                build_ppo_batch(role_rollouts[role], trainer_config, jax, jnp)
                for role in TRAINING_ROLES
            ]
            ppo_batch = concatenate_ppo_batches(role_ppo_batches, jnp)
            rollout_out = concatenate_rollout_outputs(
                [role_rollouts[role] for role in TRAINING_ROLES],
                jnp,
            )
            update_start_ns = perf_counter_ns()
            params, opt_state, update_metrics = update_runner(params, opt_state, ppo_batch)
            block_until_ready_tree((params, opt_state, update_metrics))
            update_elapsed_ns = perf_counter_ns() - update_start_ns
            current_states = {
                role: role_rollouts[role].final_state
                for role in TRAINING_ROLES
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
            )
            for role in TRAINING_ROLES:
                last_metrics.update(_summarize_role_rollout_metrics(role, role_rollouts[role]))
            if active_opponent_info is not None:
                last_metrics["opponent_update_index"] = int(
                    active_opponent_info.get(
                        "latest_update_index",
                        active_opponent_info.get("update_index", 0),
                    )
                )
                last_metrics["opponent_source"] = str(active_opponent_info.get("source", "unknown"))
                last_metrics["opponent_group_count"] = int(active_opponent_info.get("group_count", 1))
                last_metrics["opponent_unique_update_count"] = int(
                    active_opponent_info.get("unique_update_count", 1)
                )
            else:
                last_metrics["opponent_update_index"] = -1
                last_metrics["opponent_source"] = "legal_random"
                last_metrics["opponent_group_count"] = 0
                last_metrics["opponent_unique_update_count"] = 0

            should_log_history = (
                update_idx == 1
                or update_idx == int(args.num_updates)
                or (int(args.log_every_updates) > 0 and update_idx % int(args.log_every_updates) == 0)
            )
            if should_log_history:
                train_history.append(last_metrics)
                if mlflow is not None:
                    _log_mlflow_metrics(
                        mlflow,
                        last_metrics,
                        step=update_idx,
                        prefix="jax/train",
                    )

            progress.update(1)
            progress.set_postfix_str(
                (
                    f"train:{update_idx}"
                    f" sps:{float(last_metrics['end_to_end_steps_per_sec']):.0f}"
                ),
                refresh=False,
            )

            should_eval = int(args.eval_every_updates) > 0 and (
                update_idx == int(args.num_updates)
                or update_idx % int(args.eval_every_updates) == 0
            )
            if should_eval:
                eval_key = jax.random.PRNGKey(int(args.policy_seed) + 1_000_000 + update_idx)
                role_eval_keys = jax.random.split(eval_key, len(TRAINING_ROLES))
                eval_outputs = {}
                for role, role_eval_key in zip(TRAINING_ROLES, role_eval_keys, strict=True):
                    if grouped_opponent_params is not None:
                        eval_outputs[role] = grouped_eval_runner(
                            statics[role],
                            eval_initial_states[role],
                            params,
                            grouped_opponent_params,
                            role_eval_key,
                            int(args.eval_horizon),
                            int(active_opponent_info["group_count"]),
                        )
                    elif opponent_params is None:
                        eval_outputs[role] = eval_runner(
                            statics[role],
                            eval_initial_states[role],
                            params,
                            role_eval_key,
                            int(args.eval_horizon),
                        )
                    else:
                        eval_outputs[role] = frozen_eval_runner(
                            statics[role],
                            eval_initial_states[role],
                            params,
                            opponent_params,
                            role_eval_key,
                            int(args.eval_horizon),
                        )
                block_until_ready_tree(eval_outputs)
                if len(eval_trajectories) < int(args.max_eval_dumps):
                    env_index = min(max(0, int(args.eval_trajectory_env_index)), int(args.kernel_batch_size) - 1)
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
                            "mean_final_offense_score": float(np.asarray(final_eval_state.offense_score).mean()),
                            "mean_final_defense_score": float(np.asarray(final_eval_state.defense_score).mean()),
                            "mean_final_score_margin": float(
                                np.asarray(final_eval_state.offense_score - final_eval_state.defense_score).mean()
                            ),
                            "mean_done_rate": float(np.asarray(eval_trace.dones).mean()),
                            "mean_reward": float(np.asarray(eval_trace.rewards).mean()),
                        }
                        eval_metrics.update(eval_episode_metrics)
                        _log_mlflow_metrics(
                            mlflow,
                            eval_metrics,
                            step=update_idx,
                            prefix=f"jax/eval_{role}",
                        )
                progress.update(1)
                progress.set_postfix_str(f"eval:{update_idx}", refresh=False)

            checkpoint_enabled = bool(checkpoint_dir) or mlflow is not None
            should_checkpoint = checkpoint_enabled and (
                update_idx == int(args.num_updates)
                or (
                    int(args.checkpoint_every_updates) > 0
                    and update_idx % int(args.checkpoint_every_updates) == 0
                )
            )
            if should_checkpoint:
                saved_candidate_info = None
                if checkpoint_dir:
                    latest_checkpoint_path, numbered_checkpoint_path = _save_training_checkpoint(
                        checkpoint_dir=checkpoint_dir,
                        update_index=update_idx,
                        trainer_config=trainer_config,
                        spec=spec,
                        args=args,
                        params=params,
                        opt_state=opt_state,
                        current_state=current_states,
                        eval_initial_state=eval_initial_states,
                        base_key=base_key,
                        eval_trajectories=eval_trajectories,
                        last_metrics=last_metrics,
                        opponent_info=active_opponent_info,
                    )
                    saved_candidate_info = {
                        "source": "local_checkpoint",
                        "checkpoint_path": str(numbered_checkpoint_path),
                        "latest_checkpoint_path": str(latest_checkpoint_path),
                        "update_index": int(update_idx),
                    }
                    if mlflow is not None:
                        latest_checkpoint_artifact_path = _log_mlflow_checkpoint_artifacts(
                            mlflow,
                            numbered_checkpoint_path=numbered_checkpoint_path,
                            update_index=update_idx,
                        )
                        saved_candidate_info.update(
                            {
                                "source": "mlflow",
                                "artifact_path": latest_checkpoint_artifact_path,
                            }
                        )
                elif mlflow is not None:
                    with TemporaryDirectory(prefix="basketworld_jax_ckpt_") as staging_dir:
                        latest_checkpoint_path, numbered_checkpoint_path = _save_training_checkpoint(
                            checkpoint_dir=staging_dir,
                            update_index=update_idx,
                            trainer_config=trainer_config,
                            spec=spec,
                            args=args,
                            params=params,
                            opt_state=opt_state,
                            current_state=current_states,
                            eval_initial_state=eval_initial_states,
                            base_key=base_key,
                            eval_trajectories=eval_trajectories,
                            last_metrics=last_metrics,
                            opponent_info=active_opponent_info,
                        )
                        latest_checkpoint_artifact_path = _log_mlflow_checkpoint_artifacts(
                            mlflow,
                            numbered_checkpoint_path=numbered_checkpoint_path,
                            update_index=update_idx,
                        )
                        saved_candidate_info = {
                            "source": "mlflow",
                            "artifact_path": latest_checkpoint_artifact_path,
                            "update_index": int(update_idx),
                        }
                    latest_checkpoint_path = None
                if opponent_pool_enabled and saved_candidate_info is not None:
                    _add_opponent_candidate(
                        opponent_candidates,
                        params=params,
                        info={
                            **saved_candidate_info,
                            "candidate_kind": "self_checkpoint",
                        },
                    )
                    if grouped_opponent_sampling_enabled:
                        grouped_opponent_params, active_opponent_info = _select_grouped_opponents_from_pool(
                            opponent_candidates,
                            args=args,
                            rng=opponent_rng,
                            jax=jax,
                            jnp=jnp,
                        )
                        opponent_params = None
                    else:
                        opponent_params, active_opponent_info = _select_opponent_from_pool(
                            opponent_candidates,
                            args=args,
                            rng=opponent_rng,
                        )
                        grouped_opponent_params = None
                _print_checkpoint_summary(
                    update_index=update_idx,
                    last_metrics=last_metrics,
                    latest_checkpoint_path=latest_checkpoint_path,
                    latest_checkpoint_artifact_path=latest_checkpoint_artifact_path,
                )

        progress.close()

        result = {
            "script": "basketworld_jax/train/main.py",
            "status": "train_loop",
            "resumed_from_checkpoint": resume_checkpoint or None,
            "trainer_config": asdict(trainer_config),
            "frozen_config": {
                key: to_builtin(getattr(args, key))
                for key in TRAIN_FROZEN_VALUES
            },
            "env_config": _jax_env_config_from_args(args),
            "policy_spec": asdict(spec),
            "training_player_ids": {
                role: [int(v) for v in ids.tolist()]
                for role, ids in training_player_ids_by_role.items()
            },
            "train_history": train_history,
            "eval_trajectories": eval_trajectories,
            "final_metrics": last_metrics,
            "latest_checkpoint_path": latest_checkpoint_path,
            "latest_checkpoint_artifact_path": latest_checkpoint_artifact_path,
            "active_opponent": active_opponent_info,
            "opponent_pool_size": len(opponent_candidates),
            "next_step": "run a longer learnability check and inspect eval trajectories for behavior changes",
        }
        return result


def run_train_scaffold(args) -> dict[str, Any]:
    validate_train_args(args)
    jax, jnp = ensure_jax_available("basketworld_jax/train/main.py")
    static, state = sample_state_batch(args, xp=jnp)
    training_player_ids = training_player_ids_from_static(static)
    training_player_ids_jnp = jnp.asarray(training_player_ids, dtype=jnp.int32)

    flat_obs = build_flat_observation_batch(static, state, jnp)
    action_masks = build_action_masks_batch(static, state, jnp)[:, training_player_ids_jnp, :]
    flat_obs_np = np.asarray(jax.device_get(flat_obs), dtype=np.float32)
    action_masks_np = np.asarray(jax.device_get(action_masks), dtype=np.int8)
    spec = build_actor_critic_spec(
        flat_obs_np,
        action_masks_np,
        hidden_dims=args.policy_hidden_dims,
    )
    params = init_actor_critic_params(
        jax,
        jnp,
        spec,
        seed=int(args.policy_seed),
    )
    trainer_config = build_trainer_config(args)
    update_runner, optimizer_transform = build_jitted_ppo_update_runner(jax, jnp, spec, trainer_config)
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
        final_out = runner(params, flat_obs, action_masks, sample_key)
        jax.block_until_ready(final_out["values"])
        progress.update(1)
        progress.set_postfix_str("forward_warmup", refresh=False)

    timed_ns = 0
    for idx in range(int(args.benchmark_iters)):
        sample_key = jax.random.fold_in(sample_key, idx + 10_000)
        start_ns = perf_counter_ns()
        final_out = runner(params, flat_obs, action_masks, sample_key)
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
    if int(args.warmup_iters) > 0:
        _, _, _ = benchmark_update_runner(
            jax,
            update_runner,
            params,
            opt_state,
            ppo_batch,
            iterations=int(args.warmup_iters),
            progress=progress,
        )
    update_metrics, updated_params, updated_opt_state = benchmark_update_runner(
        jax,
        update_runner,
        params,
        opt_state,
        ppo_batch,
        iterations=int(args.benchmark_iters),
        progress=progress,
    )
    del updated_params, updated_opt_state
    progress.close()

    result = {
        "script": "basketworld_jax/train/main.py",
        "status": "trajectory_and_update_scaffold",
        "trainer_config": asdict(trainer_config),
        "frozen_config": {
            key: to_builtin(getattr(args, key))
            for key in TRAIN_FROZEN_VALUES
        },
        "env_config": _jax_env_config_from_args(args),
        "policy_spec": asdict(spec),
        "steps_per_update": int(args.kernel_batch_size) * int(args.rollout_horizon),
        "actor_critic_forward_states_per_sec": float(total_states / total_seconds),
        "actor_critic_mean_batch_latency_ms": float((timed_ns / 1e6) / max(1, int(args.benchmark_iters))),
        "rollout_trajectory_states_per_sec": float(rollout_metrics["states_per_sec"]),
        "rollout_mean_latency_ms": float(rollout_metrics["mean_rollout_latency_ms"]),
        "ppo_update_updates_per_sec": float(update_metrics["updates_per_sec"]),
        "ppo_update_mean_latency_ms": float(update_metrics["mean_update_latency_ms"]),
        "end_to_end_steps_per_sec": float(
            (int(args.kernel_batch_size) * int(args.rollout_horizon))
            / max(
                (float(rollout_metrics["mean_rollout_latency_ms"]) + float(update_metrics["mean_update_latency_ms"]))
                / 1000.0,
                1e-12,
            )
        ),
        "ppo_update_final_metrics": update_metrics["final_metrics"],
            "trajectory_spec": {
                "flat_obs_shape": list(flat_obs_np.shape),
                "action_mask_shape": list(action_masks_np.shape),
                "action_shape": [int(args.kernel_batch_size), int(spec.training_player_count)],
                "full_action_shape": [int(args.kernel_batch_size), int(static.role_encoding.shape[0])],
                "value_shape": [int(args.kernel_batch_size)],
                "log_prob_shape": [int(args.kernel_batch_size), int(spec.training_player_count)],
                "rollout_horizon": int(args.rollout_horizon),
                "trajectory_flat_obs_shape": list(np.asarray(rollout_out.trajectory.flat_obs).shape),
                "trajectory_action_mask_shape": list(np.asarray(rollout_out.trajectory.action_mask).shape),
                "trajectory_actions_shape": list(np.asarray(rollout_out.trajectory.actions).shape),
                "trajectory_full_actions_shape": list(np.asarray(rollout_out.trajectory.full_actions).shape),
                "trajectory_log_prob_shape": list(np.asarray(rollout_out.trajectory.selected_log_probs).shape),
                "trajectory_values_shape": list(np.asarray(rollout_out.trajectory.values).shape),
                "trajectory_rewards_shape": list(np.asarray(rollout_out.trajectory.rewards).shape),
                "trajectory_dones_shape": list(np.asarray(rollout_out.trajectory.dones).shape),
                "trajectory_pass_attempts_shape": list(np.asarray(rollout_out.trajectory.pass_attempts).shape),
                "trajectory_completed_passes_shape": list(np.asarray(rollout_out.trajectory.completed_passes).shape),
                "trajectory_assists_shape": list(np.asarray(rollout_out.trajectory.assists).shape),
                "trajectory_turnovers_shape": list(np.asarray(rollout_out.trajectory.turnovers).shape),
                "trajectory_terminal_episode_steps_shape": list(
                    np.asarray(rollout_out.trajectory.terminal_episode_steps).shape
                ),
                "trajectory_offense_score_delta_shape": list(
                    np.asarray(rollout_out.trajectory.offense_score_delta).shape
                ),
                "trajectory_defense_score_delta_shape": list(
                    np.asarray(rollout_out.trajectory.defense_score_delta).shape
                ),
                "bootstrap_values_shape": list(np.asarray(rollout_out.bootstrap_values).shape),
                "ppo_batch_flat_obs_shape": list(np.asarray(ppo_batch.flat_obs).shape),
                "ppo_batch_action_mask_shape": list(np.asarray(ppo_batch.action_mask).shape),
            "ppo_batch_actions_shape": list(np.asarray(ppo_batch.actions).shape),
            "ppo_batch_old_log_probs_shape": list(np.asarray(ppo_batch.old_selected_log_probs).shape),
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

    if args.output_json:
        write_json(args.output_json, result)
        print(f"wrote_json: {args.output_json}")


if __name__ == "__main__":
    main()
