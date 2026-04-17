from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter_ns
from typing import Any, NamedTuple
from contextlib import nullcontext
import sys

import numpy as np

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

try:
    from benchmarks.common import (
        build_benchmark_parser,
        build_progress,
        ensure_jax_available,
        to_builtin,
        write_json,
    )
    from benchmarks.jax_phase_a_optim import (
        build_adam_transform,
        global_norm,
        init_optimizer_state,
        optimizer_update,
    )
    from benchmarks.jax_kernel import (
        _assemble_full_actions_jax,
        _replace_done_states,
        _resolve_team_player_ids,
        _sample_uniform_legal_actions_jax,
        _step_batch_minimal_impl,
        build_aggregated_reward_batch,
        build_action_masks_batch,
        build_phase_a_flat_observation_batch,
        reset_batch_minimal,
        sample_state_batch,
    )
    from benchmarks.jax_phase_a_policy import (
        PhaseAActorCriticSpec,
        apply_action_mask,
        actor_critic_forward,
        build_phase_a_actor_critic_spec,
        init_phase_a_actor_critic_params,
        run_actor_critic,
    )
    from benchmarks.sbx_phase_a import PHASE_A_FROZEN_VALUES
    from basketworld.utils.mlflow_config import setup_mlflow
except ImportError:  # pragma: no cover - direct script execution path
    from common import build_benchmark_parser, build_progress, ensure_jax_available, to_builtin, write_json  # type: ignore[no-redef]
    from jax_phase_a_optim import (  # type: ignore[no-redef]
        build_adam_transform,
        global_norm,
        init_optimizer_state,
        optimizer_update,
    )
    from jax_kernel import (  # type: ignore[no-redef]
        _assemble_full_actions_jax,
        _replace_done_states,
        _resolve_team_player_ids,
        _sample_uniform_legal_actions_jax,
        _step_batch_minimal_impl,
        build_aggregated_reward_batch,
        build_action_masks_batch,
        build_phase_a_flat_observation_batch,
        reset_batch_minimal,
        sample_state_batch,
    )
    from jax_phase_a_policy import (  # type: ignore[no-redef]
        PhaseAActorCriticSpec,
        apply_action_mask,
        actor_critic_forward,
        build_phase_a_actor_critic_spec,
        init_phase_a_actor_critic_params,
        run_actor_critic,
    )
    from sbx_phase_a import PHASE_A_FROZEN_VALUES  # type: ignore[no-redef]
    from basketworld.utils.mlflow_config import setup_mlflow  # type: ignore[no-redef]


PHASE_A_TRAIN_FROZEN_VALUES: dict[str, Any] = {
    **PHASE_A_FROZEN_VALUES,
    "pass_mode": "pointer_targeted",
    "use_set_obs": False,
    "training_team": "offense",
    "enable_phi_shaping": False,
    "illegal_defense_enabled": False,
    "offensive_three_seconds": False,
    "include_hoop_vector": True,
    "phi_aggregation_mode": "team_best",
    "phi_use_ball_handler_only": False,
}


@dataclass(frozen=True)
class PhaseATrainerConfig:
    kernel_batch_size: int
    rollout_horizon: int
    num_updates: int
    gamma: float
    gae_lambda: float
    ppo_clip_range: float
    value_coef: float
    entropy_coef: float
    learning_rate: float
    policy_update_epochs: int


class PhaseATrajectoryBatch(NamedTuple):
    flat_obs: Any
    action_mask: Any
    actions: Any
    selected_log_probs: Any
    values: Any
    rewards: Any
    dones: Any


class PhaseARolloutOutput(NamedTuple):
    trajectory: PhaseATrajectoryBatch
    final_state: Any
    bootstrap_values: Any
    final_flat_obs: Any
    final_action_mask: Any


class PhaseAPPOBatch(NamedTuple):
    flat_obs: Any
    action_mask: Any
    actions: Any
    old_selected_log_probs: Any
    old_values: Any
    advantages: Any
    returns: Any


class PhaseAEvalTrace(NamedTuple):
    positions: Any
    ball_holder: Any
    shot_clock: Any
    full_actions: Any
    rewards: Any
    dones: Any
    offense_score: Any
    defense_score: Any


def parse_args(argv=None):
    parser = build_benchmark_parser(
        "Phase A JAX trainer scaffold: separate actor-critic + compiled rollout-ready input path."
    )
    parser.set_defaults(**PHASE_A_TRAIN_FROZEN_VALUES)
    parser.set_defaults(mode="throughput", runner="sequential")
    parser.add_argument(
        "--kernel-batch-size",
        type=int,
        default=256,
        help="Number of sampled env states to pack into one actor-critic forward pass.",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=3,
        help="Number of warm actor-critic iterations to run before timing.",
    )
    parser.add_argument(
        "--benchmark-iters",
        type=int,
        default=50,
        help="Number of timed actor-critic iterations to run.",
    )
    parser.add_argument(
        "--sample-reset-seed",
        type=int,
        default=0,
        help="Base reset seed used when sampling env snapshots into the JAX batch.",
    )
    parser.add_argument(
        "--policy-hidden-dims",
        type=int,
        nargs="+",
        default=[128, 128],
        help="Hidden layer widths for the reduced flat JAX actor-critic.",
    )
    parser.add_argument(
        "--policy-seed",
        type=int,
        default=0,
        help="Random seed used to initialize the JAX actor-critic parameters.",
    )
    parser.add_argument(
        "--rollout-horizon",
        type=int,
        default=64,
        help="Planned rollout horizon for the future PPO scan path.",
    )
    parser.add_argument(
        "--num-updates",
        type=int,
        default=500,
        help="Planned PPO update count for the future trainer loop.",
    )
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ppo-clip-range", type=float, default=0.2)
    parser.add_argument(
        "--policy-update-epochs",
        type=int,
        default=1,
        help="Number of full-batch PPO update epochs per rollout in this minimal trainer.",
    )
    parser.add_argument(
        "--run-train-loop",
        action="store_true",
        help="Run a short multi-update training loop instead of only the scaffold benchmarks.",
    )
    parser.add_argument(
        "--log-every-updates",
        type=int,
        default=10,
        help="How often to record scalar training history entries.",
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
        help="Maximum number of eval trajectory dumps to keep in the JSON output.",
    )
    parser.add_argument(
        "--eval-trajectory-env-index",
        type=int,
        default=0,
        help="Which env index from the eval batch to serialize into the JSON trajectory dump.",
    )
    parser.add_argument(
        "--log-mlflow",
        action="store_true",
        help="Log params and scalar metrics for the JAX-only trainer to MLflow.",
    )
    return parser.parse_args(argv)


def _values_match(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        return np.isclose(float(actual), float(expected), atol=1e-8, rtol=0.0)
    return actual == expected


def validate_phase_a_train_args(args) -> None:
    mismatches: list[str] = []
    for key, expected in PHASE_A_TRAIN_FROZEN_VALUES.items():
        actual = getattr(args, key)
        if not _values_match(actual, expected):
            mismatches.append(f"{key}={actual!r} expected {expected!r}")
    if mismatches:
        raise SystemExit(
            "Phase A JAX trainer scaffold uses a frozen reduced config. Unsupported overrides: "
            + ", ".join(mismatches)
        )


def build_trainer_config(args) -> PhaseATrainerConfig:
    return PhaseATrainerConfig(
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


def _training_player_ids_from_static(static) -> np.ndarray:
    mask = np.asarray(static.training_player_mask, dtype=np.float32)
    return np.flatnonzero(mask > 0.5).astype(np.int32)


def _build_jitted_actor_critic_runner(jax, jnp, spec: PhaseAActorCriticSpec):
    @jax.jit
    def _runner(params, flat_obs, action_mask, sample_key):
        return run_actor_critic(
            params,
            flat_obs,
            action_mask,
            spec,
            sample_key,
            jax,
            jnp,
        )

    return _runner


def _build_compiled_phase_a_rollout_runner(jax, jnp, spec: PhaseAActorCriticSpec):
    def _runner(static, initial_state, params, rollout_key, horizon: int):
        training_ids, opponent_ids = _resolve_team_player_ids(static, jax, jnp)
        n_players = int(static.role_encoding.shape[0])

        def _scan_step(carry, _):
            state, key = carry
            key, policy_key, opponent_key, env_key, reset_key = jax.random.split(key, 5)
            flat_obs = build_phase_a_flat_observation_batch(static, state, jnp)
            full_action_mask = build_action_masks_batch(static, state, jnp)
            training_action_mask = full_action_mask[:, training_ids, :]
            opponent_action_mask = full_action_mask[:, opponent_ids, :]

            policy_out = run_actor_critic(
                params,
                flat_obs,
                training_action_mask,
                spec,
                policy_key,
                jax,
                jnp,
            )
            opponent_actions = _sample_uniform_legal_actions_jax(
                opponent_action_mask,
                opponent_key,
                jax,
                jnp,
            )
            full_actions = _assemble_full_actions_jax(
                policy_out["sampled_actions"],
                opponent_actions,
                training_ids,
                opponent_ids,
                n_players,
                jnp,
            )
            env_keys = jax.random.split(env_key, initial_state.positions.shape[0])
            env_out = _step_batch_minimal_impl(
                static,
                state,
                full_actions,
                env_keys,
                jax,
                jnp,
            )
            reset_keys = jax.random.split(reset_key, initial_state.positions.shape[0])
            reset_state = reset_batch_minimal(static, reset_keys, jax, jnp)
            next_state = _replace_done_states(env_out.state, reset_state, env_out.done, jnp)
            aggregated_reward = build_aggregated_reward_batch(static, env_out.rewards, jnp)
            transition = PhaseATrajectoryBatch(
                flat_obs=flat_obs,
                action_mask=training_action_mask,
                actions=policy_out["sampled_actions"],
                selected_log_probs=policy_out["selected_log_probs"],
                values=policy_out["values"],
                rewards=aggregated_reward,
                dones=env_out.done.astype(jnp.int8),
            )
            return (next_state, key), transition

        (final_state, final_key), trajectory = jax.lax.scan(
            _scan_step,
            (initial_state, rollout_key),
            xs=None,
            length=int(horizon),
        )
        final_flat_obs = build_phase_a_flat_observation_batch(static, final_state, jnp)
        final_action_mask = build_action_masks_batch(static, final_state, jnp)[:, training_ids, :]
        bootstrap_values = actor_critic_forward(params, final_flat_obs, jnp)["values"]
        del final_key  # explicit discard so the return stays PPO-focused
        return PhaseARolloutOutput(
            trajectory=trajectory,
            final_state=final_state,
            bootstrap_values=bootstrap_values,
            final_flat_obs=final_flat_obs,
            final_action_mask=final_action_mask,
        )

    return jax.jit(_runner, static_argnums=(4,))


def _build_compiled_phase_a_eval_runner(jax, jnp, spec: PhaseAActorCriticSpec):
    def _runner(static, initial_state, params, rollout_key, horizon: int):
        training_ids, opponent_ids = _resolve_team_player_ids(static, jax, jnp)
        n_players = int(static.role_encoding.shape[0])

        def _scan_step(carry, _):
            state, key = carry
            key, opponent_key, env_key = jax.random.split(key, 3)
            full_action_mask = build_action_masks_batch(static, state, jnp)
            training_action_mask = full_action_mask[:, training_ids, :]
            opponent_action_mask = full_action_mask[:, opponent_ids, :]

            forward_out = actor_critic_forward(
                params,
                build_phase_a_flat_observation_batch(static, state, jnp),
                jnp,
            )
            masked_out = apply_action_mask(
                forward_out["flat_policy_logits"],
                training_action_mask,
                spec,
                jax,
                jnp,
            )
            opponent_actions = _sample_uniform_legal_actions_jax(
                opponent_action_mask,
                opponent_key,
                jax,
                jnp,
            )
            full_actions = _assemble_full_actions_jax(
                masked_out["deterministic_actions"],
                opponent_actions,
                training_ids,
                opponent_ids,
                n_players,
                jnp,
            )
            env_keys = jax.random.split(env_key, initial_state.positions.shape[0])
            env_out = _step_batch_minimal_impl(
                static,
                state,
                full_actions,
                env_keys,
                jax,
                jnp,
            )
            trace = PhaseAEvalTrace(
                positions=state.positions,
                ball_holder=state.ball_holder,
                shot_clock=state.shot_clock,
                full_actions=full_actions,
                rewards=build_aggregated_reward_batch(static, env_out.rewards, jnp),
                dones=env_out.done.astype(jnp.int8),
                offense_score=env_out.state.offense_score,
                defense_score=env_out.state.defense_score,
            )
            return (env_out.state, key), trace

        (final_state, _), trace = jax.lax.scan(
            _scan_step,
            (initial_state, rollout_key),
            xs=None,
            length=int(horizon),
        )
        return final_state, trace

    return jax.jit(_runner, static_argnums=(4,))


def compute_gae_and_returns(rewards, values, dones, bootstrap_values, *, gamma: float, gae_lambda: float, jax, jnp):
    gamma_t = jnp.asarray(gamma, dtype=jnp.float32)
    gae_lambda_t = jnp.asarray(gae_lambda, dtype=jnp.float32)
    next_values = jnp.concatenate([values[1:], bootstrap_values[None, :]], axis=0)
    not_done = 1.0 - dones.astype(jnp.float32)
    deltas = rewards + (gamma_t * next_values * not_done) - values

    def _scan_step(carry, scan_inputs):
        delta_t, not_done_t = scan_inputs
        advantage = delta_t + (gamma_t * gae_lambda_t * not_done_t * carry)
        return advantage, advantage

    _, advantages_rev = jax.lax.scan(
        _scan_step,
        jnp.zeros_like(bootstrap_values, dtype=jnp.float32),
        (deltas[::-1], not_done[::-1]),
    )
    advantages = advantages_rev[::-1]
    returns = advantages + values
    return advantages, returns


def build_ppo_batch(rollout: PhaseARolloutOutput, trainer_config: PhaseATrainerConfig, jax, jnp) -> PhaseAPPOBatch:
    advantages, returns = compute_gae_and_returns(
        rollout.trajectory.rewards,
        rollout.trajectory.values,
        rollout.trajectory.dones,
        rollout.bootstrap_values,
        gamma=float(trainer_config.gamma),
        gae_lambda=float(trainer_config.gae_lambda),
        jax=jax,
        jnp=jnp,
    )
    flat_advantages = advantages.reshape(-1)
    adv_mean = jnp.mean(flat_advantages)
    adv_std = jnp.std(flat_advantages)
    normalized_advantages = (advantages - adv_mean) / jnp.maximum(adv_std, 1.0e-8)
    return PhaseAPPOBatch(
        flat_obs=rollout.trajectory.flat_obs.reshape(
            -1,
            int(rollout.trajectory.flat_obs.shape[-1]),
        ),
        action_mask=rollout.trajectory.action_mask.reshape(
            -1,
            int(rollout.trajectory.action_mask.shape[-2]),
            int(rollout.trajectory.action_mask.shape[-1]),
        ),
        actions=rollout.trajectory.actions.reshape(
            -1,
            int(rollout.trajectory.actions.shape[-1]),
        ),
        old_selected_log_probs=rollout.trajectory.selected_log_probs.reshape(
            -1,
            int(rollout.trajectory.selected_log_probs.shape[-1]),
        ),
        old_values=rollout.trajectory.values.reshape(-1),
        advantages=normalized_advantages.reshape(-1),
        returns=returns.reshape(-1),
    )


def _build_jitted_ppo_update_runner(jax, jnp, spec: PhaseAActorCriticSpec, trainer_config: PhaseATrainerConfig):
    import optax

    tree_util = jax.tree_util
    clip_range = jnp.asarray(trainer_config.ppo_clip_range, dtype=jnp.float32)
    value_coef = jnp.asarray(trainer_config.value_coef, dtype=jnp.float32)
    entropy_coef = jnp.asarray(trainer_config.entropy_coef, dtype=jnp.float32)
    epochs = int(trainer_config.policy_update_epochs)
    transform = build_adam_transform(
        optax,
        learning_rate=float(trainer_config.learning_rate),
    )

    def _loss_fn(params, batch: PhaseAPPOBatch):
        forward_out = actor_critic_forward(params, batch.flat_obs, jnp)
        masked_out = apply_action_mask(
            forward_out["flat_policy_logits"],
            batch.action_mask,
            spec,
            jax,
            jnp,
        )
        new_selected_log_probs = jnp.take_along_axis(
            masked_out["log_probs"],
            batch.actions[..., None],
            axis=-1,
        )[..., 0]
        old_log_prob_state = jnp.sum(batch.old_selected_log_probs, axis=-1)
        new_log_prob_state = jnp.sum(new_selected_log_probs, axis=-1)
        log_ratio = new_log_prob_state - old_log_prob_state
        ratio = jnp.exp(log_ratio)
        clipped_ratio = jnp.clip(ratio, 1.0 - clip_range, 1.0 + clip_range)
        policy_loss = -jnp.mean(
            jnp.minimum(
                ratio * batch.advantages,
                clipped_ratio * batch.advantages,
            )
        )
        value_loss = jnp.mean(jnp.square(forward_out["values"] - batch.returns))
        entropy_bonus = jnp.mean(jnp.mean(masked_out["entropy"], axis=-1))
        approx_kl = jnp.mean(batch.old_selected_log_probs - new_selected_log_probs)
        clip_fraction = jnp.mean((jnp.abs(ratio - 1.0) > clip_range).astype(jnp.float32))
        total_loss = policy_loss + (value_coef * value_loss) - (entropy_coef * entropy_bonus)
        metrics = {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_bonus": entropy_bonus,
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
        }
        return total_loss, metrics

    def _single_epoch(params, opt_state, batch):
        (loss, metrics), grads = jax.value_and_grad(_loss_fn, has_aux=True)(params, batch)
        grad_norm = global_norm(grads, optax)
        new_params, new_opt_state = optimizer_update(
            params,
            grads,
            opt_state,
            transform=transform,
            optax=optax,
        )
        metrics = {
            **metrics,
            "grad_norm": grad_norm,
            "total_loss": loss,
        }
        return new_params, new_opt_state, metrics

    @jax.jit
    def _runner(params, opt_state, batch):
        def _epoch_step(carry, _):
            epoch_params, epoch_opt_state = carry
            next_params, next_opt_state, metrics = _single_epoch(epoch_params, epoch_opt_state, batch)
            return (next_params, next_opt_state), metrics

        (next_params, next_opt_state), metrics = jax.lax.scan(
            _epoch_step,
            (params, opt_state),
            xs=None,
            length=epochs,
        )
        final_metrics = {name: values[-1] for name, values in metrics.items()}
        return next_params, next_opt_state, final_metrics

    return _runner, transform


def _block_until_ready_tree(value):
    if isinstance(value, dict):
        for item in value.values():
            _block_until_ready_tree(item)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _block_until_ready_tree(item)
        return
    blocker = getattr(value, "block_until_ready", None)
    if callable(blocker):
        blocker()


def _benchmark_compiled_rollout(jax, runner, static, state, params, rollout_key, *, batch_size: int, horizon: int, iterations: int, progress=None):
    final_out = None
    timed_ns = 0
    for idx in range(int(iterations)):
        iter_key = jax.random.fold_in(rollout_key, idx)
        start_ns = perf_counter_ns()
        final_out = runner(static, state, params, iter_key, int(horizon))
        _block_until_ready_tree(final_out)
        timed_ns += perf_counter_ns() - start_ns
        if progress is not None:
            progress.update(1)
            progress.set_postfix_str("rollout", refresh=False)
    total_states = int(batch_size) * int(horizon) * int(iterations)
    total_seconds = max(timed_ns / 1e9, 1e-12)
    return {
        "states_per_sec": float(total_states / total_seconds),
        "mean_rollout_latency_ms": float((timed_ns / 1e6) / max(1, int(iterations))),
        "total_states": int(total_states),
        "elapsed_sec": float(total_seconds),
    }, final_out


def _benchmark_update_runner(jax, runner, params, opt_state, batch, *, iterations: int, progress=None):
    timed_ns = 0
    final_params = params
    final_opt_state = opt_state
    final_metrics = None
    for _ in range(int(iterations)):
        start_ns = perf_counter_ns()
        final_params, final_opt_state, final_metrics = runner(final_params, final_opt_state, batch)
        _block_until_ready_tree((final_params, final_opt_state, final_metrics))
        timed_ns += perf_counter_ns() - start_ns
        if progress is not None:
            progress.update(1)
            progress.set_postfix_str("update", refresh=False)
    total_seconds = max(timed_ns / 1e9, 1e-12)
    return {
        "updates_per_sec": float(int(iterations) / total_seconds),
        "mean_update_latency_ms": float((timed_ns / 1e6) / max(1, int(iterations))),
        "elapsed_sec": float(total_seconds),
        "final_metrics": {
            key: float(np.asarray(value))
            for key, value in (final_metrics or {}).items()
        },
    }, final_params, final_opt_state


def _summarize_training_step(
    rollout_out: PhaseARolloutOutput,
    ppo_batch: PhaseAPPOBatch,
    update_metrics: dict[str, float],
    rollout_elapsed_ns: int,
    update_elapsed_ns: int,
    *,
    batch_size: int,
    horizon: int,
    update_index: int,
) -> dict[str, Any]:
    total_states = int(batch_size) * int(horizon)
    rollout_sec = max(rollout_elapsed_ns / 1e9, 1e-12)
    update_sec = max(update_elapsed_ns / 1e9, 1e-12)
    end_to_end_sec = max((rollout_elapsed_ns + update_elapsed_ns) / 1e9, 1e-12)
    reward_mean = float(np.asarray(rollout_out.trajectory.rewards).mean())
    done_rate = float(np.asarray(rollout_out.trajectory.dones).mean())
    advantage_std = float(np.asarray(ppo_batch.advantages).std())
    return_mean = float(np.asarray(ppo_batch.returns).mean())
    value_mean = float(np.asarray(rollout_out.trajectory.values).mean())
    summary = {
        "update_index": int(update_index),
        "steps_per_update": int(total_states),
        "rollout_states_per_sec": float(total_states / rollout_sec),
        "end_to_end_steps_per_sec": float(total_states / end_to_end_sec),
        "rollout_latency_ms": float(rollout_elapsed_ns / 1e6),
        "update_steps_per_sec": float(1.0 / update_sec),
        "update_latency_ms": float(update_elapsed_ns / 1e6),
        "mean_reward": reward_mean,
        "done_rate": done_rate,
        "mean_return": return_mean,
        "mean_value": value_mean,
        "advantage_std": advantage_std,
    }
    summary.update({key: float(value) for key, value in update_metrics.items()})
    return summary


def _serialize_eval_trace(
    trace: PhaseAEvalTrace,
    final_state,
    *,
    env_index: int,
    update_index: int,
) -> dict[str, Any]:
    positions = np.asarray(trace.positions)
    ball_holder = np.asarray(trace.ball_holder)
    shot_clock = np.asarray(trace.shot_clock)
    full_actions = np.asarray(trace.full_actions)
    rewards = np.asarray(trace.rewards)
    dones = np.asarray(trace.dones)
    offense_score = np.asarray(trace.offense_score)
    defense_score = np.asarray(trace.defense_score)
    final_offense = np.asarray(final_state.offense_score)
    final_defense = np.asarray(final_state.defense_score)
    return {
        "update_index": int(update_index),
        "env_index": int(env_index),
        "trajectory_length": int(positions.shape[0]),
        "positions": positions[:, env_index].astype(np.int32),
        "ball_holder": ball_holder[:, env_index].astype(np.int32),
        "shot_clock": shot_clock[:, env_index].astype(np.int32),
        "full_actions": full_actions[:, env_index].astype(np.int32),
        "rewards": rewards[:, env_index].astype(np.float32),
        "dones": dones[:, env_index].astype(np.int8),
        "offense_score": offense_score[:, env_index].astype(np.float32),
        "defense_score": defense_score[:, env_index].astype(np.float32),
        "final_offense_score": float(final_offense[env_index]),
        "final_defense_score": float(final_defense[env_index]),
    }


def _maybe_start_mlflow_run(args, *, mode: str):
    if not bool(getattr(args, "log_mlflow", False)):
        return None, nullcontext()

    import mlflow

    setup_mlflow(verbose=False)
    mlflow.set_experiment(str(args.mlflow_experiment_name))
    run_name = args.mlflow_run_name
    if not run_name:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"jax-phase-a-{mode}-{timestamp}"
    context = mlflow.start_run(run_name=run_name)
    return mlflow, context


def _log_mlflow_params(mlflow, args, trainer_config: PhaseATrainerConfig, spec: PhaseAActorCriticSpec) -> None:
    params = {
        "jax_phase_a/script": "benchmarks/jax_phase_a_train.py",
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
    }
    mlflow.log_params(params)


def _log_mlflow_metrics(mlflow, metrics: dict[str, Any], *, step: int, prefix: str) -> None:
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            mlflow.log_metric(f"{prefix}/{key}", float(value), step=int(step))


def run_phase_a_training_loop(args) -> dict[str, Any]:
    validate_phase_a_train_args(args)
    jax, jnp = ensure_jax_available("benchmarks/jax_phase_a_train.py")
    static, _ = sample_state_batch(args, xp=jnp)
    base_key = jax.random.PRNGKey(int(args.policy_seed))
    reset_seed_key, eval_reset_seed_key, base_key = jax.random.split(base_key, 3)
    initial_reset_keys = jax.random.split(reset_seed_key, int(args.kernel_batch_size))
    current_state = reset_batch_minimal(static, initial_reset_keys, jax, jnp)
    eval_reset_keys = jax.random.split(eval_reset_seed_key, int(args.kernel_batch_size))
    eval_initial_state = reset_batch_minimal(static, eval_reset_keys, jax, jnp)

    training_player_ids = _training_player_ids_from_static(static)
    training_player_ids_jnp = jnp.asarray(training_player_ids, dtype=jnp.int32)
    flat_obs = build_phase_a_flat_observation_batch(static, current_state, jnp)
    action_masks = build_action_masks_batch(static, current_state, jnp)[:, training_player_ids_jnp, :]
    flat_obs_np = np.asarray(jax.device_get(flat_obs), dtype=np.float32)
    action_masks_np = np.asarray(jax.device_get(action_masks), dtype=np.int8)
    spec = build_phase_a_actor_critic_spec(
        flat_obs_np,
        action_masks_np,
        hidden_dims=args.policy_hidden_dims,
    )
    params = init_phase_a_actor_critic_params(
        jax,
        jnp,
        spec,
        seed=int(args.policy_seed),
    )

    trainer_config = build_trainer_config(args)
    rollout_runner = _build_compiled_phase_a_rollout_runner(jax, jnp, spec)
    eval_runner = _build_compiled_phase_a_eval_runner(jax, jnp, spec)
    update_runner, optimizer_transform = _build_jitted_ppo_update_runner(jax, jnp, spec, trainer_config)
    opt_state = init_optimizer_state(optimizer_transform, params)
    mlflow, mlflow_context = _maybe_start_mlflow_run(args, mode="train")

    with mlflow_context:
        if mlflow is not None:
            _log_mlflow_params(mlflow, args, trainer_config, spec)

        expected_evals = 0
        if int(args.eval_every_updates) > 0:
            expected_evals = int(args.num_updates) // int(args.eval_every_updates)
            if int(args.num_updates) % int(args.eval_every_updates) != 0:
                expected_evals += 1
        progress = build_progress(
            total=int(args.num_updates) + expected_evals,
            desc="phase_a_train:loop",
            disable=bool(args.no_progress),
            unit="step",
        )

        train_history: list[dict[str, Any]] = []
        eval_trajectories: list[dict[str, Any]] = []
        last_metrics: dict[str, Any] | None = None

        for update_idx in range(1, int(args.num_updates) + 1):
            base_key, rollout_key = jax.random.split(base_key)
            rollout_start_ns = perf_counter_ns()
            rollout_out = rollout_runner(
                static,
                current_state,
                params,
                rollout_key,
                int(args.rollout_horizon),
            )
            _block_until_ready_tree(rollout_out)
            rollout_elapsed_ns = perf_counter_ns() - rollout_start_ns

            ppo_batch = build_ppo_batch(rollout_out, trainer_config, jax, jnp)
            update_start_ns = perf_counter_ns()
            params, opt_state, update_metrics = update_runner(params, opt_state, ppo_batch)
            _block_until_ready_tree((params, opt_state, update_metrics))
            update_elapsed_ns = perf_counter_ns() - update_start_ns
            current_state = rollout_out.final_state

            last_metrics = _summarize_training_step(
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
            progress.set_postfix_str(f"train:{update_idx}", refresh=False)

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
                _block_until_ready_tree((final_eval_state, eval_trace))
                if len(eval_trajectories) < int(args.max_eval_dumps):
                    env_index = min(max(0, int(args.eval_trajectory_env_index)), int(args.kernel_batch_size) - 1)
                    eval_trajectories.append(
                        _serialize_eval_trace(
                            eval_trace,
                            final_eval_state,
                            env_index=env_index,
                            update_index=update_idx,
                        )
                    )
                if mlflow is not None:
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
                    _log_mlflow_metrics(
                        mlflow,
                        eval_metrics,
                        step=update_idx,
                        prefix="jax_phase_a/eval",
                    )
                progress.update(1)
                progress.set_postfix_str(f"eval:{update_idx}", refresh=False)

        progress.close()

        result = {
            "script": "benchmarks/jax_phase_a_train.py",
            "status": "train_loop",
            "trainer_config": asdict(trainer_config),
            "frozen_config": {
                key: to_builtin(getattr(args, key))
                for key in PHASE_A_TRAIN_FROZEN_VALUES
            },
            "policy_spec": asdict(spec),
            "training_player_ids": [int(v) for v in training_player_ids.tolist()],
            "train_history": train_history,
            "eval_trajectories": eval_trajectories,
            "final_metrics": last_metrics,
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


def run_phase_a_train_scaffold(args) -> dict[str, Any]:
    validate_phase_a_train_args(args)
    jax, jnp = ensure_jax_available("benchmarks/jax_phase_a_train.py")
    static, state = sample_state_batch(args, xp=jnp)
    training_player_ids = _training_player_ids_from_static(static)
    training_player_ids_jnp = jnp.asarray(training_player_ids, dtype=jnp.int32)

    flat_obs = build_phase_a_flat_observation_batch(static, state, jnp)
    action_masks = build_action_masks_batch(static, state, jnp)[:, training_player_ids_jnp, :]
    flat_obs_np = np.asarray(jax.device_get(flat_obs), dtype=np.float32)
    action_masks_np = np.asarray(jax.device_get(action_masks), dtype=np.int8)
    spec = build_phase_a_actor_critic_spec(
        flat_obs_np,
        action_masks_np,
        hidden_dims=args.policy_hidden_dims,
    )
    params = init_phase_a_actor_critic_params(
        jax,
        jnp,
        spec,
        seed=int(args.policy_seed),
    )
    update_runner, optimizer_transform = _build_jitted_ppo_update_runner(jax, jnp, spec, build_trainer_config(args))
    opt_state = init_optimizer_state(optimizer_transform, params)
    runner = _build_jitted_actor_critic_runner(jax, jnp, spec)
    rollout_runner = _build_compiled_phase_a_rollout_runner(jax, jnp, spec)

    total_iters = 3 * (int(args.warmup_iters) + int(args.benchmark_iters))
    progress = build_progress(
        total=total_iters,
        desc="phase_a_train:actor_critic",
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
        _benchmark_compiled_rollout(
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
    rollout_metrics, rollout_out = _benchmark_compiled_rollout(
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

    trainer_config = build_trainer_config(args)
    total_states = int(args.kernel_batch_size) * int(args.benchmark_iters)
    total_seconds = max(timed_ns / 1e9, 1e-12)
    ppo_batch = build_ppo_batch(rollout_out, trainer_config, jax, jnp)
    if int(args.warmup_iters) > 0:
        _, _, _ = _benchmark_update_runner(
            jax,
            update_runner,
            params,
            opt_state,
            ppo_batch,
            iterations=int(args.warmup_iters),
            progress=progress,
        )
    update_metrics, updated_params, updated_opt_state = _benchmark_update_runner(
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
        "script": "benchmarks/jax_phase_a_train.py",
        "status": "trajectory_and_update_scaffold",
        "trainer_config": asdict(trainer_config),
        "frozen_config": {
            key: to_builtin(getattr(args, key))
            for key in PHASE_A_TRAIN_FROZEN_VALUES
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
            "value_shape": [int(args.kernel_batch_size)],
            "log_prob_shape": [int(args.kernel_batch_size), int(spec.training_player_count)],
            "rollout_horizon": int(args.rollout_horizon),
            "trajectory_flat_obs_shape": list(np.asarray(rollout_out.trajectory.flat_obs).shape),
            "trajectory_action_mask_shape": list(np.asarray(rollout_out.trajectory.action_mask).shape),
            "trajectory_actions_shape": list(np.asarray(rollout_out.trajectory.actions).shape),
            "trajectory_log_prob_shape": list(np.asarray(rollout_out.trajectory.selected_log_probs).shape),
            "trajectory_values_shape": list(np.asarray(rollout_out.trajectory.values).shape),
            "trajectory_rewards_shape": list(np.asarray(rollout_out.trajectory.rewards).shape),
            "trajectory_dones_shape": list(np.asarray(rollout_out.trajectory.dones).shape),
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
        result = run_phase_a_training_loop(args)
    else:
        result = run_phase_a_train_scaffold(args)

    if bool(args.run_train_loop):
        print("Phase A JAX trainer loop")
        print(f"policy_spec: {result['policy_spec']}")
        print(f"logged_train_entries: {len(result['train_history'])}")
        if result["final_metrics"] is not None:
            print(f"final_metrics: {result['final_metrics']}")
        print(f"eval_trajectory_dumps: {len(result['eval_trajectories'])}")
    else:
        print("Phase A JAX trainer scaffold")
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
