from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple


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


class TrajectoryBatch(NamedTuple):
    flat_obs: Any
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
    terminal_episode_steps: Any


class RolloutOutput(NamedTuple):
    trajectory: TrajectoryBatch
    final_state: Any
    bootstrap_values: Any
    final_flat_obs: Any
    final_action_mask: Any


class PPOBatch(NamedTuple):
    flat_obs: Any
    action_mask: Any
    actions: Any
    old_selected_log_probs: Any
    old_values: Any
    advantages: Any
    returns: Any


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
