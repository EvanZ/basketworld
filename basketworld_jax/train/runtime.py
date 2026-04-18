from __future__ import annotations

from time import perf_counter_ns
from typing import Any

import numpy as np

from basketworld_jax.env import (
    assemble_full_actions_jax,
    build_action_masks_batch,
    build_aggregated_reward_batch,
    build_flat_observation_batch,
    replace_done_states,
    reset_batch_minimal,
    resolve_team_player_ids,
    sample_uniform_legal_actions_jax,
    step_batch_minimal,
)
from basketworld_jax.models import (
    ActorCriticSpec,
    actor_critic_forward,
    apply_action_mask,
    run_actor_critic,
)
from basketworld_jax.optim import build_adam_transform, global_norm, optimizer_update
from basketworld_jax.train.types import (
    EvalTrace,
    PPOBatch,
    RolloutOutput,
    TrajectoryBatch,
    TrainerConfig,
)


def training_player_ids_from_static(static) -> np.ndarray:
    mask = np.asarray(static.training_player_mask, dtype=np.float32)
    return np.flatnonzero(mask > 0.5).astype(np.int32)


def build_jitted_actor_critic_runner(jax, jnp, spec: ActorCriticSpec):
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


def build_compiled_rollout_runner(jax, jnp, spec: ActorCriticSpec):
    def _runner(static, initial_state, params, rollout_key, horizon: int):
        training_ids, opponent_ids = resolve_team_player_ids(static, jax, jnp)
        n_players = int(static.role_encoding.shape[0])

        def _scan_step(carry, _):
            state, key = carry
            key, policy_key, opponent_key, env_key, reset_key = jax.random.split(key, 5)
            flat_obs = build_flat_observation_batch(static, state, jnp)
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
            opponent_actions = sample_uniform_legal_actions_jax(
                opponent_action_mask,
                opponent_key,
                jax,
                jnp,
            )
            full_actions = assemble_full_actions_jax(
                policy_out["sampled_actions"],
                opponent_actions,
                training_ids,
                opponent_ids,
                n_players,
                jnp,
            )
            env_keys = jax.random.split(env_key, initial_state.positions.shape[0])
            env_out = step_batch_minimal(
                static,
                state,
                full_actions,
                env_keys,
                jax,
                jnp,
            )
            reset_keys = jax.random.split(reset_key, initial_state.positions.shape[0])
            reset_state = reset_batch_minimal(static, reset_keys, jax, jnp)
            next_state = replace_done_states(env_out.state, reset_state, env_out.done, jnp)
            aggregated_reward = build_aggregated_reward_batch(static, env_out.rewards, jnp)
            transition = TrajectoryBatch(
                flat_obs=flat_obs,
                action_mask=training_action_mask,
                actions=policy_out["sampled_actions"],
                full_actions=full_actions,
                selected_log_probs=policy_out["selected_log_probs"],
                values=policy_out["values"],
                rewards=aggregated_reward,
                dones=env_out.done.astype(jnp.int8),
                pass_attempts=env_out.pass_attempt.astype(jnp.int8),
                completed_passes=env_out.completed_pass.astype(jnp.int8),
                assists=env_out.assist.astype(jnp.int8),
                turnovers=env_out.turnover.astype(jnp.int8),
                terminal_episode_steps=env_out.terminal_episode_steps.astype(jnp.int32),
            )
            return (next_state, key), transition

        (final_state, _), trajectory = jax.lax.scan(
            _scan_step,
            (initial_state, rollout_key),
            xs=None,
            length=int(horizon),
        )
        final_flat_obs = build_flat_observation_batch(static, final_state, jnp)
        final_action_mask = build_action_masks_batch(static, final_state, jnp)[:, training_ids, :]
        bootstrap_values = actor_critic_forward(params, final_flat_obs, spec, jnp)["values"]
        return RolloutOutput(
            trajectory=trajectory,
            final_state=final_state,
            bootstrap_values=bootstrap_values,
            final_flat_obs=final_flat_obs,
            final_action_mask=final_action_mask,
        )

    return jax.jit(_runner, static_argnums=(4,))


def build_compiled_eval_runner(jax, jnp, spec: ActorCriticSpec):
    def _runner(static, initial_state, params, rollout_key, horizon: int):
        training_ids, opponent_ids = resolve_team_player_ids(static, jax, jnp)
        n_players = int(static.role_encoding.shape[0])

        def _scan_step(carry, _):
            state, key = carry
            key, opponent_key, env_key = jax.random.split(key, 3)
            full_action_mask = build_action_masks_batch(static, state, jnp)
            training_action_mask = full_action_mask[:, training_ids, :]
            opponent_action_mask = full_action_mask[:, opponent_ids, :]

            forward_out = actor_critic_forward(
                params,
                build_flat_observation_batch(static, state, jnp),
                spec,
                jnp,
            )
            masked_out = apply_action_mask(
                forward_out["flat_policy_logits"],
                training_action_mask,
                spec,
                jax,
                jnp,
            )
            opponent_actions = sample_uniform_legal_actions_jax(
                opponent_action_mask,
                opponent_key,
                jax,
                jnp,
            )
            full_actions = assemble_full_actions_jax(
                masked_out["deterministic_actions"],
                opponent_actions,
                training_ids,
                opponent_ids,
                n_players,
                jnp,
            )
            env_keys = jax.random.split(env_key, initial_state.positions.shape[0])
            env_out = step_batch_minimal(
                static,
                state,
                full_actions,
                env_keys,
                jax,
                jnp,
            )
            trace = EvalTrace(
                positions=state.positions,
                ball_holder=state.ball_holder,
                shot_clock=state.shot_clock,
                full_actions=full_actions,
                rewards=build_aggregated_reward_batch(static, env_out.rewards, jnp),
                dones=env_out.done.astype(jnp.int8),
                pass_attempts=env_out.pass_attempt.astype(jnp.int8),
                completed_passes=env_out.completed_pass.astype(jnp.int8),
                assists=env_out.assist.astype(jnp.int8),
                turnovers=env_out.turnover.astype(jnp.int8),
                terminal_episode_steps=env_out.terminal_episode_steps.astype(jnp.int32),
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


def build_jitted_ppo_update_runner(jax, jnp, spec: ActorCriticSpec, trainer_config: TrainerConfig):
    import optax

    clip_range = jnp.asarray(trainer_config.ppo_clip_range, dtype=jnp.float32)
    value_coef = jnp.asarray(trainer_config.value_coef, dtype=jnp.float32)
    entropy_coef = jnp.asarray(trainer_config.entropy_coef, dtype=jnp.float32)
    epochs = int(trainer_config.policy_update_epochs)
    transform = build_adam_transform(
        optax,
        learning_rate=float(trainer_config.learning_rate),
    )

    def _loss_fn(params, batch: PPOBatch):
        forward_out = actor_critic_forward(params, batch.flat_obs, spec, jnp)
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


def block_until_ready_tree(value):
    if isinstance(value, dict):
        for item in value.values():
            block_until_ready_tree(item)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            block_until_ready_tree(item)
        return
    blocker = getattr(value, "block_until_ready", None)
    if callable(blocker):
        blocker()


def benchmark_compiled_rollout(jax, runner, static, state, params, rollout_key, *, batch_size: int, horizon: int, iterations: int, progress=None):
    final_out = None
    timed_ns = 0
    for idx in range(int(iterations)):
        iter_key = jax.random.fold_in(rollout_key, idx)
        start_ns = perf_counter_ns()
        final_out = runner(static, state, params, iter_key, int(horizon))
        block_until_ready_tree(final_out)
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


def benchmark_update_runner(jax, runner, params, opt_state, batch, *, iterations: int, progress=None):
    timed_ns = 0
    final_params = params
    final_opt_state = opt_state
    final_metrics = None
    for _ in range(int(iterations)):
        start_ns = perf_counter_ns()
        final_params, final_opt_state, final_metrics = runner(final_params, final_opt_state, batch)
        block_until_ready_tree((final_params, final_opt_state, final_metrics))
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


def summarize_episode_events(
    dones,
    terminal_episode_steps,
    pass_attempts,
    completed_passes,
    assists,
    turnovers,
) -> dict[str, float]:
    done_arr = np.asarray(dones, dtype=np.float32)
    terminal_steps_arr = np.asarray(terminal_episode_steps, dtype=np.int32)
    pass_attempts_arr = np.asarray(pass_attempts, dtype=np.float32)
    completed_passes_arr = np.asarray(completed_passes, dtype=np.float32)
    assists_arr = np.asarray(assists, dtype=np.float32)
    turnovers_arr = np.asarray(turnovers, dtype=np.float32)

    completed_episodes = int(done_arr.sum())
    completed_episode_steps = int(terminal_steps_arr.sum())
    denom = float(completed_episodes) if completed_episodes > 0 else 0.0

    def _mean_per_episode(total: float) -> float:
        return float(total / denom) if denom > 0.0 else 0.0

    total_pass_attempts = float(pass_attempts_arr.sum())
    total_completed_passes = float(completed_passes_arr.sum())
    total_assists = float(assists_arr.sum())
    total_turnovers = float(turnovers_arr.sum())

    return {
        "completed_episodes": int(completed_episodes),
        "completed_episode_steps": int(completed_episode_steps),
        "mean_completed_episode_length": (
            float(completed_episode_steps / denom) if denom > 0.0 else 0.0
        ),
        "total_pass_attempts": total_pass_attempts,
        "total_completed_passes": total_completed_passes,
        "total_assists": total_assists,
        "total_turnovers": total_turnovers,
        "mean_pass_attempts_per_completed_episode": _mean_per_episode(total_pass_attempts),
        "mean_completed_passes_per_completed_episode": _mean_per_episode(total_completed_passes),
        "mean_assists_per_completed_episode": _mean_per_episode(total_assists),
        "mean_turnovers_per_completed_episode": _mean_per_episode(total_turnovers),
    }


def summarize_training_step(
    rollout_out: RolloutOutput,
    ppo_batch: PPOBatch,
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
    episode_metrics = summarize_episode_events(
        rollout_out.trajectory.dones,
        rollout_out.trajectory.terminal_episode_steps,
        rollout_out.trajectory.pass_attempts,
        rollout_out.trajectory.completed_passes,
        rollout_out.trajectory.assists,
        rollout_out.trajectory.turnovers,
    )
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
    summary.update(episode_metrics)
    summary.update({key: float(value) for key, value in update_metrics.items()})
    return summary


def serialize_eval_trace(
    trace: EvalTrace,
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
    pass_attempts = np.asarray(trace.pass_attempts)
    completed_passes = np.asarray(trace.completed_passes)
    assists = np.asarray(trace.assists)
    turnovers = np.asarray(trace.turnovers)
    terminal_episode_steps = np.asarray(trace.terminal_episode_steps)
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
        "pass_attempts": pass_attempts[:, env_index].astype(np.int8),
        "completed_passes": completed_passes[:, env_index].astype(np.int8),
        "assists": assists[:, env_index].astype(np.int8),
        "turnovers": turnovers[:, env_index].astype(np.int8),
        "terminal_episode_steps": terminal_episode_steps[:, env_index].astype(np.int32),
        "offense_score": offense_score[:, env_index].astype(np.float32),
        "defense_score": defense_score[:, env_index].astype(np.float32),
        "final_offense_score": float(final_offense[env_index]),
        "final_defense_score": float(final_defense[env_index]),
    }
