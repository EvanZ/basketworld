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


class TrajectoryBatch(NamedTuple):
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
    flat_advantages = advantages.reshape(-1)
    adv_mean = jnp.mean(flat_advantages)
    adv_std = jnp.std(flat_advantages)
    normalized_advantages = (advantages - adv_mean) / jnp.maximum(adv_std, 1.0e-8)
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
    flat_mask = rollout.trajectory.selector_used.reshape(-1).astype(jnp.bool_)
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
    return PPOBatch(
        *(
            jnp.concatenate([getattr(batch, field) for batch in batches], axis=0)
            for field in PPOBatch._fields
        )
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
    active_order = jnp.cumsum(active_mask.astype(jnp.int32))
    limited_mask = active_mask & (active_order <= max_count)
    limited_mask_f = limited_mask.astype(jnp.float32)
    return SelectorBatch(
        flat_obs=batch.flat_obs,
        chosen_intents=batch.chosen_intents,
        old_log_probs=batch.old_log_probs,
        old_values=batch.old_values,
        advantages=jnp.where(limited_mask, batch.advantages, jnp.zeros_like(batch.advantages)),
        returns=batch.returns,
        active_mask=limited_mask_f,
    )
