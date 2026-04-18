from __future__ import annotations

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
)
from basketworld_jax.train.runtime import (
    benchmark_compiled_rollout,
    benchmark_update_runner,
    block_until_ready_tree,
    build_compiled_eval_runner,
    build_compiled_rollout_runner,
    build_jitted_actor_critic_runner,
    build_jitted_ppo_update_runner,
    serialize_eval_trace,
    summarize_episode_events,
    summarize_training_step,
    training_player_ids_from_static,
)


def parse_args(argv=None):
    parser = build_parser(
        "JAX trainer: reduced actor-critic + compiled rollout path."
    )
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
    return parser.parse_args(argv)


def _values_match(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        return np.isclose(float(actual), float(expected), atol=1e-8, rtol=0.0)
    return actual == expected


def validate_train_args(args) -> None:
    mismatches: list[str] = []
    for key, expected in TRAIN_FROZEN_VALUES.items():
        actual = getattr(args, key)
        if not _values_match(actual, expected):
            mismatches.append(f"{key}={actual!r} expected {expected!r}")
    if mismatches:
        raise SystemExit(
            "JAX trainer uses a frozen reduced config. Unsupported overrides: "
            + ", ".join(mismatches)
        )


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
) -> tuple[str | None, str]:
    payload = build_checkpoint_payload(
        update_index=int(update_index),
        trainer_config=asdict(trainer_config),
        policy_spec=asdict(spec),
        frozen_config={
            key: to_builtin(getattr(args, key))
            for key in TRAIN_FROZEN_VALUES
        },
        params=params,
        opt_state=opt_state,
        current_state=current_state,
        eval_initial_state=eval_initial_state,
        base_key=base_key,
        eval_trajectories=eval_trajectories,
        last_metrics=last_metrics,
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


def _log_mlflow_params(mlflow, args, trainer_config: TrainerConfig, spec: ActorCriticSpec) -> None:
    params = {
            "jax_phase_a/script": "basketworld_jax/train/main.py",
        "jax_phase_a/mode": "train_loop" if bool(args.run_train_loop) else "scaffold",
        "jax_phase_a/kernel_batch_size": int(args.kernel_batch_size),
        "jax_phase_a/rollout_horizon": int(args.rollout_horizon),
        "jax_phase_a/num_updates": int(args.num_updates),
        "jax_phase_a/policy_update_epochs": int(args.policy_update_epochs),
        "jax_phase_a/log_every_updates": int(args.log_every_updates),
        "jax_phase_a/eval_every_updates": int(args.eval_every_updates),
        "jax_phase_a/eval_horizon": int(args.eval_horizon),
        "jax_phase_a/learning_rate": float(trainer_config.learning_rate),
        "jax_phase_a/gamma": float(trainer_config.gamma),
        "jax_phase_a/gae_lambda": float(trainer_config.gae_lambda),
        "jax_phase_a/ppo_clip_range": float(trainer_config.ppo_clip_range),
        "jax_phase_a/value_coef": float(trainer_config.value_coef),
        "jax_phase_a/entropy_coef": float(trainer_config.entropy_coef),
        "jax_phase_a/policy_hidden_dims": ",".join(str(v) for v in spec.hidden_dims),
        "jax_phase_a/flat_obs_dim": int(spec.flat_obs_dim),
        "jax_phase_a/training_player_count": int(spec.training_player_count),
        "jax_phase_a/action_dim_per_player": int(spec.action_dim_per_player),
        "jax_phase_a/pass_mode": str(getattr(args, "pass_mode")),
        "jax_phase_a/use_set_obs": bool(getattr(args, "use_set_obs")),
        "jax_phase_a/training_team": str(getattr(args, "training_team")),
        "jax_phase_a/checkpoint_every_updates": int(args.checkpoint_every_updates),
    }
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
    mlflow.set_tag("model_backend", "jax_phase_a")
    mlflow.set_tag("jax_phase_a_checkpoint_format", "orbax_phase_a_v2")
    mlflow.set_tag("jax_phase_a_latest_checkpoint_artifact", artifact_path)
    mlflow.set_tag("jax_phase_a_latest_checkpoint_update", str(int(update_index)))
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
        ("entropy_bonus", metrics.get("entropy_bonus")),
        ("policy_loss", metrics.get("policy_loss")),
        ("value_loss", metrics.get("value_loss")),
        ("total_loss", metrics.get("total_loss")),
        ("grad_norm", metrics.get("grad_norm")),
        ("mean_reward", metrics.get("mean_reward")),
        ("mean_return", metrics.get("mean_return")),
        ("done_rate", metrics.get("done_rate")),
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
    static, _ = sample_state_batch(args, xp=jnp)
    base_key = jax.random.PRNGKey(int(args.policy_seed))
    reset_seed_key, eval_reset_seed_key, base_key = jax.random.split(base_key, 3)
    initial_reset_keys = jax.random.split(reset_seed_key, int(args.kernel_batch_size))
    current_state = reset_batch_minimal(static, initial_reset_keys, jax, jnp)
    eval_reset_keys = jax.random.split(eval_reset_seed_key, int(args.kernel_batch_size))
    eval_initial_state = reset_batch_minimal(static, eval_reset_keys, jax, jnp)

    training_player_ids = training_player_ids_from_static(static)
    training_player_ids_jnp = jnp.asarray(training_player_ids, dtype=jnp.int32)
    flat_obs = build_flat_observation_batch(static, current_state, jnp)
    action_masks = build_action_masks_batch(static, current_state, jnp)[:, training_player_ids_jnp, :]
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
    update_runner, optimizer_transform = build_jitted_ppo_update_runner(jax, jnp, spec, trainer_config)
    checkpoint_dir = str(args.checkpoint_dir).strip()
    resume_checkpoint = str(args.resume_checkpoint).strip()
    latest_checkpoint_path: str | None = None
    latest_checkpoint_artifact_path: str | None = None

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
        current_state = jax.device_put(
            _restore_like_template(checkpoint_payload["current_state"], current_state)
        )
        eval_initial_state = jax.device_put(
            _restore_like_template(checkpoint_payload["eval_initial_state"], eval_initial_state)
        )
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
            base_key, rollout_key = jax.random.split(base_key)
            rollout_start_ns = perf_counter_ns()
            rollout_out = rollout_runner(
                static,
                current_state,
                params,
                rollout_key,
                int(args.rollout_horizon),
            )
            block_until_ready_tree(rollout_out)
            rollout_elapsed_ns = perf_counter_ns() - rollout_start_ns

            ppo_batch = build_ppo_batch(rollout_out, trainer_config, jax, jnp)
            update_start_ns = perf_counter_ns()
            params, opt_state, update_metrics = update_runner(params, opt_state, ppo_batch)
            block_until_ready_tree((params, opt_state, update_metrics))
            update_elapsed_ns = perf_counter_ns() - update_start_ns
            current_state = rollout_out.final_state

            last_metrics = summarize_training_step(
                rollout_out,
                ppo_batch,
                {
                    key: float(np.asarray(value))
                    for key, value in update_metrics.items()
                },
                rollout_elapsed_ns,
                update_elapsed_ns,
                batch_size=int(args.kernel_batch_size),
                horizon=int(args.rollout_horizon),
                update_index=update_idx,
            )

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
                        prefix="jax_phase_a/train",
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
                final_eval_state, eval_trace = eval_runner(
                    static,
                    eval_initial_state,
                    params,
                    eval_key,
                    int(args.eval_horizon),
                )
                block_until_ready_tree((final_eval_state, eval_trace))
                if len(eval_trajectories) < int(args.max_eval_dumps):
                    env_index = min(max(0, int(args.eval_trajectory_env_index)), int(args.kernel_batch_size) - 1)
                    eval_trajectories.append(
                        serialize_eval_trace(
                            eval_trace,
                            final_eval_state,
                            env_index=env_index,
                            update_index=update_idx,
                        )
                    )
                if mlflow is not None:
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
                        prefix="jax_phase_a/eval",
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
                if checkpoint_dir:
                    latest_checkpoint_path, numbered_checkpoint_path = _save_training_checkpoint(
                        checkpoint_dir=checkpoint_dir,
                        update_index=update_idx,
                        trainer_config=trainer_config,
                        spec=spec,
                        args=args,
                        params=params,
                        opt_state=opt_state,
                        current_state=current_state,
                        eval_initial_state=eval_initial_state,
                        base_key=base_key,
                        eval_trajectories=eval_trajectories,
                        last_metrics=last_metrics,
                    )
                    if mlflow is not None:
                        latest_checkpoint_artifact_path = _log_mlflow_checkpoint_artifacts(
                            mlflow,
                            numbered_checkpoint_path=numbered_checkpoint_path,
                            update_index=update_idx,
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
                            current_state=current_state,
                            eval_initial_state=eval_initial_state,
                            base_key=base_key,
                            eval_trajectories=eval_trajectories,
                            last_metrics=last_metrics,
                        )
                        latest_checkpoint_artifact_path = _log_mlflow_checkpoint_artifacts(
                            mlflow,
                            numbered_checkpoint_path=numbered_checkpoint_path,
                            update_index=update_idx,
                        )
                    latest_checkpoint_path = None
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
            "policy_spec": asdict(spec),
            "training_player_ids": [int(v) for v in training_player_ids.tolist()],
            "train_history": train_history,
            "eval_trajectories": eval_trajectories,
            "final_metrics": last_metrics,
            "latest_checkpoint_path": latest_checkpoint_path,
            "latest_checkpoint_artifact_path": latest_checkpoint_artifact_path,
            "next_step": "run a longer learnability check and inspect eval trajectories for behavior changes",
        }
        if mlflow is not None and last_metrics is not None:
            _log_mlflow_metrics(
                mlflow,
                last_metrics,
                step=int(args.num_updates),
                prefix="jax_phase_a/final",
            )
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
