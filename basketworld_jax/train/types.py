from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple, Sequence


@dataclass(frozen=True)
class TrainerConfig:
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
    ppo_minibatches: int = 1
    single_episode_rollouts: bool = False
    ppo_completed_episodes_only: bool = False


class TrajectoryBatch(NamedTuple):
    active_mask: Any
    episode_start: Any
    flat_obs: Any
    policy_intent_index: Any
    policy_intent_gate: Any
    action_mask: Any
    actions: Any
    full_actions: Any
    selected_log_probs: Any
    values: Any
    rewards: Any
    dones: Any
    phi_r_shape: Any
    phi_prev: Any
    phi_next: Any
    phi_beta: Any
    pass_attempts: Any
    completed_passes: Any
    assists: Any
    turnovers: Any
    learner_turnovers: Any
    opponent_turnovers: Any
    turnover_pass_out_of_bounds: Any
    turnover_intercepted: Any
    turnover_defender_pressure: Any
    turnover_move_out_of_bounds: Any
    turnover_shot_clock: Any
    turnover_offensive_three_seconds: Any
    shot_attempts: Any
    shot_makes: Any
    shot_dunks: Any
    shot_twos: Any
    shot_threes: Any
    learner_shot_attempts: Any
    learner_shot_makes: Any
    learner_shot_dunks: Any
    learner_shot_twos: Any
    learner_shot_threes: Any
    opponent_shot_attempts: Any
    opponent_shot_makes: Any
    opponent_shot_dunks: Any
    opponent_shot_twos: Any
    opponent_shot_threes: Any
    intent_index: Any
    intent_active: Any
    intent_age: Any
    intent_commitment_remaining: Any
    intent_visible_to_defense: Any
    defense_intent_index: Any
    defense_intent_active: Any
    defense_intent_age: Any
    defense_intent_commitment_remaining: Any
    selector_used: Any
    selector_applied: Any
    selector_fallback_used: Any
    selector_boundary_episode_start: Any
    selector_boundary_commitment_timeout: Any
    selector_boundary_completed_pass: Any
    selector_intent_index: Any
    selector_old_log_prob: Any
    selector_value: Any
    selector_entropy: Any
    selector_max_prob: Any
    offensive_three_seconds: Any
    defensive_lane_violations: Any
    terminal_episode_steps: Any
    offense_score_delta: Any
    defense_score_delta: Any


class RolloutOutput(NamedTuple):
    trajectory: TrajectoryBatch
    final_state: Any
    bootstrap_values: Any
    final_selector_values: Any
    final_flat_obs: Any
    final_action_mask: Any


class PPOBatch(NamedTuple):
    flat_obs: Any
    policy_intent_index: Any
    policy_intent_gate: Any
    action_mask: Any
    actions: Any
    old_selected_log_probs: Any
    old_values: Any
    advantages: Any
    returns: Any
    active_mask: Any
    loss_weights: Any
    loss_denominator: Any


class SelectorBatch(NamedTuple):
    flat_obs: Any
    chosen_intents: Any
    old_log_probs: Any
    old_values: Any
    advantages: Any
    returns: Any
    active_mask: Any


class EvalTrace(NamedTuple):
    positions: Any
    ball_holder: Any
    shot_clock: Any
    full_actions: Any
    rewards: Any
    dones: Any
    pass_attempts: Any
    completed_passes: Any
    assists: Any
    turnovers: Any
    shot_attempts: Any
    shot_makes: Any
    shot_dunks: Any
    shot_twos: Any
    shot_threes: Any
    learner_shot_attempts: Any
    learner_shot_makes: Any
    learner_shot_dunks: Any
    learner_shot_twos: Any
    learner_shot_threes: Any
    opponent_shot_attempts: Any
    opponent_shot_makes: Any
    opponent_shot_dunks: Any
    opponent_shot_twos: Any
    opponent_shot_threes: Any
    intent_index: Any
    intent_active: Any
    intent_age: Any
    intent_commitment_remaining: Any
    intent_visible_to_defense: Any
    defense_intent_index: Any
    defense_intent_active: Any
    defense_intent_age: Any
    defense_intent_commitment_remaining: Any
    offensive_three_seconds: Any
    defensive_lane_violations: Any
    terminal_episode_steps: Any
    offense_score: Any
    defense_score: Any


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


def build_trajectory_training_masks(trajectory: TrajectoryBatch, trainer_config: TrainerConfig, jax, jnp):
    active_mask = trajectory.active_mask.astype(jnp.float32)
    if not bool(getattr(trainer_config, "ppo_completed_episodes_only", False)):
        return active_mask, active_mask, jnp.sum(active_mask).astype(jnp.float32)

    active_bool = active_mask > 0.5
    starts = trajectory.episode_start.astype(jnp.bool_) & active_bool
    dones = trajectory.dones.astype(jnp.bool_) & active_bool

    def _forward_started(carry, start_t):
        next_carry = carry | start_t
        return next_carry, next_carry

    _, has_started = jax.lax.scan(
        _forward_started,
        jnp.zeros_like(starts[0], dtype=jnp.bool_),
        starts,
    )

    def _reverse_complete(carry, scan_inputs):
        done_t, start_t = scan_inputs
        complete_t = carry | done_t
        prev_carry = jnp.where(start_t, jnp.zeros_like(complete_t), complete_t)
        return prev_carry, complete_t

    _, complete_rev = jax.lax.scan(
        _reverse_complete,
        jnp.zeros_like(dones[0], dtype=jnp.bool_),
        (dones[::-1], starts[::-1]),
    )
    completed_episode_mask = active_bool & has_started & complete_rev[::-1]

    completed_episode_count = jnp.sum((dones & completed_episode_mask).astype(jnp.float32))
    loss_weights = jnp.where(
        completed_episode_mask,
        jnp.ones_like(active_mask),
        jnp.zeros_like(active_mask),
    )
    return (
        completed_episode_mask.astype(jnp.float32),
        loss_weights.astype(jnp.float32),
        completed_episode_count.astype(jnp.float32),
    )


def build_ppo_batch(rollout: RolloutOutput, trainer_config: TrainerConfig, jax, jnp) -> PPOBatch:
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
    active_mask, loss_weights, loss_denominator = build_trajectory_training_masks(
        rollout.trajectory,
        trainer_config,
        jax,
        jnp,
    )
    flat_advantages = advantages.reshape(-1)
    flat_active_mask = active_mask.reshape(-1).astype(jnp.float32)
    flat_loss_weights = loss_weights.reshape(-1).astype(jnp.float32)
    adv_norm_den = jnp.maximum(jnp.sum(flat_active_mask), 1.0)
    adv_mean = jnp.sum(flat_advantages * flat_active_mask) / adv_norm_den
    adv_var = jnp.sum(jnp.square(flat_advantages - adv_mean) * flat_active_mask) / adv_norm_den
    normalized_advantages = (advantages - adv_mean) / jnp.sqrt(jnp.maximum(adv_var, 1.0e-8))
    normalized_advantages = jnp.where(
        active_mask.astype(jnp.bool_),
        normalized_advantages,
        jnp.zeros_like(normalized_advantages),
    )
    return PPOBatch(
        flat_obs=rollout.trajectory.flat_obs.reshape(
            -1,
            int(rollout.trajectory.flat_obs.shape[-1]),
        ),
        policy_intent_index=rollout.trajectory.policy_intent_index.reshape(-1),
        policy_intent_gate=rollout.trajectory.policy_intent_gate.reshape(-1),
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
        active_mask=flat_active_mask,
        loss_weights=flat_loss_weights,
        loss_denominator=jnp.full_like(flat_loss_weights, loss_denominator.astype(jnp.float32)),
    )


def compute_discounted_returns(rewards, dones, bootstrap_values, *, gamma: float, jax, jnp):
    gamma_t = jnp.asarray(gamma, dtype=jnp.float32)

    def _scan_step(carry, scan_inputs):
        reward_t, done_t = scan_inputs
        next_return = reward_t + (gamma_t * carry * (1.0 - done_t.astype(jnp.float32)))
        return next_return, next_return

    _, returns_rev = jax.lax.scan(
        _scan_step,
        bootstrap_values.astype(jnp.float32),
        (rewards[::-1], dones[::-1]),
    )
    return returns_rev[::-1]


def compute_selector_segment_returns(rollout: RolloutOutput, trainer_config: TrainerConfig, jax, jnp):
    rewards = rollout.trajectory.rewards.astype(jnp.float32)
    dones = rollout.trajectory.dones.astype(jnp.float32)
    selector_used = rollout.trajectory.selector_used.astype(jnp.bool_)
    selector_values = rollout.trajectory.selector_value.astype(jnp.float32)
    gamma_t = jnp.asarray(float(trainer_config.gamma), dtype=jnp.float32)
    next_selector_used = jnp.concatenate(
        [
            selector_used[1:],
            jnp.zeros_like(selector_used[:1], dtype=jnp.bool_),
        ],
        axis=0,
    )
    next_selector_values = jnp.concatenate(
        [
            selector_values[1:],
            rollout.final_selector_values[None, :].astype(jnp.float32),
        ],
        axis=0,
    )

    def _scan_step(carry, scan_inputs):
        reward_t, done_t, next_selector_t, next_value_t = scan_inputs
        continuation = jnp.where(next_selector_t, next_value_t, carry)
        return_t = reward_t + (gamma_t * continuation * (1.0 - done_t.astype(jnp.float32)))
        return return_t, return_t

    _, returns_rev = jax.lax.scan(
        _scan_step,
        rollout.final_selector_values.astype(jnp.float32),
        (rewards[::-1], dones[::-1], next_selector_used[::-1], next_selector_values[::-1]),
    )
    return returns_rev[::-1]


def build_selector_batch(
    rollout: RolloutOutput,
    trainer_config: TrainerConfig,
    jax,
    jnp,
    *,
    max_samples_per_update: int = 0,
) -> SelectorBatch:
    selector_returns = compute_selector_segment_returns(
        rollout,
        trainer_config,
        jax=jax,
        jnp=jnp,
    )
    training_mask, _, _ = build_trajectory_training_masks(
        rollout.trajectory,
        trainer_config,
        jax,
        jnp,
    )
    flat_mask = (
        rollout.trajectory.selector_used.reshape(-1).astype(jnp.bool_)
        & (training_mask.reshape(-1).astype(jnp.float32) > 0.5)
    )
    max_samples = int(max_samples_per_update)
    if max_samples > 0:
        flat_order = jnp.cumsum(flat_mask.astype(jnp.int32))
        flat_mask = flat_mask & (flat_order <= max_samples)

    flat_returns = selector_returns.reshape(-1)
    flat_old_values = rollout.trajectory.selector_value.reshape(-1)
    raw_advantages = flat_returns - flat_old_values
    mask_f = flat_mask.astype(jnp.float32)
    sample_count = jnp.maximum(jnp.sum(mask_f), 1.0)
    adv_mean = jnp.sum(raw_advantages * mask_f) / sample_count
    adv_var = jnp.sum(jnp.square(raw_advantages - adv_mean) * mask_f) / sample_count
    normalized_advantages = (raw_advantages - adv_mean) / jnp.sqrt(jnp.maximum(adv_var, 1.0e-8))
    normalized_advantages = jnp.where(flat_mask, normalized_advantages, jnp.zeros_like(normalized_advantages))

    return SelectorBatch(
        flat_obs=rollout.trajectory.flat_obs.reshape(
            -1,
            int(rollout.trajectory.flat_obs.shape[-1]),
        ),
        chosen_intents=rollout.trajectory.selector_intent_index.reshape(-1).astype(jnp.int32),
        old_log_probs=rollout.trajectory.selector_old_log_prob.reshape(-1).astype(jnp.float32),
        old_values=flat_old_values.astype(jnp.float32),
        advantages=normalized_advantages.astype(jnp.float32),
        returns=flat_returns.astype(jnp.float32),
        active_mask=flat_mask.astype(jnp.float32),
    )


def concatenate_ppo_batches(batches: Sequence[PPOBatch], jnp) -> PPOBatch:
    if not batches:
        raise ValueError("At least one PPO batch is required.")
    loss_denominator = jnp.sum(
        jnp.stack([jnp.max(batch.loss_denominator) for batch in batches]).astype(jnp.float32)
    )
    concatenated = {
        field: jnp.concatenate([getattr(batch, field) for batch in batches], axis=0)
        for field in PPOBatch._fields
        if field != "loss_denominator"
    }
    concatenated["loss_denominator"] = jnp.full_like(
        concatenated["loss_weights"],
        loss_denominator,
    )
    return PPOBatch(
        **concatenated
    )


def concatenate_selector_batches(batches: Sequence[SelectorBatch], jnp) -> SelectorBatch:
    if not batches:
        raise ValueError("At least one selector batch is required.")
    return SelectorBatch(
        *(
            jnp.concatenate([getattr(batch, field) for batch in batches], axis=0)
            for field in SelectorBatch._fields
        )
    )


def limit_selector_batch_samples(batch: SelectorBatch, jnp, *, max_samples: int) -> SelectorBatch:
    max_count = int(max_samples)
    if max_count <= 0:
        return batch
    active_mask = batch.active_mask.astype(jnp.bool_)
    active_count = jnp.sum(active_mask.astype(jnp.int32))
    valid_count = jnp.minimum(active_count, jnp.asarray(max_count, dtype=jnp.int32))
    indices = jnp.nonzero(active_mask, size=max_count, fill_value=0)[0]
    valid_rows = jnp.arange(max_count, dtype=jnp.int32) < valid_count

    def _take_active(field):
        selected = jnp.take(field, indices, axis=0)
        mask_shape = (max_count,) + (1,) * max(0, int(selected.ndim) - 1)
        return jnp.where(valid_rows.reshape(mask_shape), selected, jnp.zeros_like(selected))

    return SelectorBatch(
        flat_obs=_take_active(batch.flat_obs),
        chosen_intents=_take_active(batch.chosen_intents).astype(jnp.int32),
        old_log_probs=_take_active(batch.old_log_probs).astype(jnp.float32),
        old_values=_take_active(batch.old_values).astype(jnp.float32),
        advantages=_take_active(batch.advantages).astype(jnp.float32),
        returns=_take_active(batch.returns).astype(jnp.float32),
        active_mask=valid_rows.astype(jnp.float32),
    )
