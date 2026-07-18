from __future__ import annotations

from time import perf_counter_ns
from typing import Any, Sequence

import numpy as np

from basketworld_jax.env import (
    SHOT_TYPE_DUNK,
    SHOT_TYPE_THREE,
    SHOT_TYPE_TWO,
    assemble_full_actions_jax,
    build_action_masks_batch,
    build_aggregated_reward_batch,
    build_policy_intent_context_batch,
    build_policy_intent_context_batch_with_role_flag,
    build_policy_observation_batch,
    build_policy_observation_batch_with_role_flag,
    replace_done_states,
    reset_batch_minimal,
    resolve_team_player_ids,
    sample_uniform_legal_actions_jax,
    set_offense_intent_state_batch,
    step_batch_minimal,
)
from basketworld_jax.env.minimal import (
    TURNOVER_REASON_DEFENDER_PRESSURE,
    TURNOVER_REASON_INTERCEPTED,
    TURNOVER_REASON_MOVE_OUT_OF_BOUNDS,
    TURNOVER_REASON_OFFENSIVE_THREE_SECONDS,
    TURNOVER_REASON_PASS_OUT_OF_BOUNDS,
    TURNOVER_REASON_SHOT_CLOCK,
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
    SelectorBatch,
    TrajectoryBatch,
    TrainerConfig,
)


_SELECTOR_UPDATE_PARAM_NAMES = frozenset(
    {
        "intent_selector_head_0",
        "intent_selector_head_out",
        "intent_selector_value_head_0",
        "intent_selector_value_head_out",
    }
)


def _tree_path_key_names(path) -> tuple[str, ...]:
    names = []
    for item in path:
        key = getattr(item, "key", item)
        names.append(str(key))
    return tuple(names)


def _is_selector_update_param_path(path) -> bool:
    return any(name in _SELECTOR_UPDATE_PARAM_NAMES for name in _tree_path_key_names(path))


def _selector_tree_map(jax, selector_fn, fallback_fn, tree, *rest):
    def _map(path, value, *others):
        if _is_selector_update_param_path(path):
            return selector_fn(value, *others)
        return fallback_fn(value, *others)

    return jax.tree_util.tree_map_with_path(_map, tree, *rest)


def _mask_selector_update_grads(grads, sample_count, jax, jnp):
    def _keep_selector_grad(grad):
        return jnp.where(sample_count > 0.0, grad, jnp.zeros_like(grad))

    return _selector_tree_map(
        jax,
        _keep_selector_grad,
        lambda grad: jnp.zeros_like(grad),
        grads,
    )


def _merge_selector_update_params(original_params, candidate_params, jax):
    return _selector_tree_map(
        jax,
        lambda original, candidate: candidate,
        lambda original, candidate: original,
        original_params,
        candidate_params,
    )


def _apply_selector_update_param_scope(original_params, candidate_params, sample_count, jax, jnp):
    selector_only_params = _merge_selector_update_params(original_params, candidate_params, jax)
    return jax.tree_util.tree_map(
        lambda original, candidate: jnp.where(sample_count > 0.0, candidate, original),
        original_params,
        selector_only_params,
    )


def training_player_ids_from_static(static) -> np.ndarray:
    mask = np.asarray(static.training_player_mask, dtype=np.float32)
    return np.flatnonzero(mask > 0.5).astype(np.int32)


def _concatenate_namedtuple_batch(items, jnp, *, axis: int):
    if not items:
        raise ValueError("At least one item is required.")
    return type(items[0])(
        *(
            jnp.concatenate([getattr(item, field) for item in items], axis=axis)
            for field in items[0]._fields
        )
    )


def concatenate_rollout_outputs(rollouts: Sequence[RolloutOutput], jnp) -> RolloutOutput:
    if not rollouts:
        raise ValueError("At least one rollout output is required.")
    return RolloutOutput(
        trajectory=_concatenate_namedtuple_batch(
            [rollout.trajectory for rollout in rollouts],
            jnp,
            axis=1,
        ),
        final_state=_concatenate_namedtuple_batch(
            [rollout.final_state for rollout in rollouts],
            jnp,
            axis=0,
        ),
        bootstrap_values=jnp.concatenate(
            [rollout.bootstrap_values for rollout in rollouts],
            axis=0,
        ),
        final_selector_values=jnp.concatenate(
            [rollout.final_selector_values for rollout in rollouts],
            axis=0,
        ),
        final_flat_obs=jnp.concatenate(
            [rollout.final_flat_obs for rollout in rollouts],
            axis=0,
        ),
        final_action_mask=jnp.concatenate(
            [rollout.final_action_mask for rollout in rollouts],
            axis=0,
        ),
    )


def build_jitted_actor_critic_runner(jax, jnp, spec: ActorCriticSpec):
    @jax.jit
    def _runner(params, flat_obs, action_mask, intent_context, sample_key):
        return run_actor_critic(
            params,
            flat_obs,
            action_mask,
            spec,
            sample_key,
            jax,
            jnp,
            intent_context=intent_context,
        )

    return _runner


def _build_shot_type_transition_metrics(static, env_out, jnp) -> dict[str, Any]:
    shot_attempt = env_out.shot_attempt.astype(jnp.int8)
    shot_attempt_bool = shot_attempt.astype(jnp.bool_)
    shot_make = env_out.shot_success.astype(jnp.int8)
    shot_dunk = (shot_attempt_bool & (env_out.shot_type == SHOT_TYPE_DUNK)).astype(jnp.int8)
    shot_three = (shot_attempt_bool & (env_out.shot_type == SHOT_TYPE_THREE)).astype(jnp.int8)
    shot_two = (shot_attempt_bool & (env_out.shot_type == SHOT_TYPE_TWO)).astype(jnp.int8)

    safe_shooter = jnp.clip(env_out.shot_shooter, 0, int(static.role_encoding.shape[0]) - 1)
    learner_shot = shot_attempt_bool & (static.training_player_mask[safe_shooter] > 0.5)
    opponent_shot = shot_attempt_bool & (~learner_shot)
    return {
        "shot_attempts": shot_attempt,
        "shot_makes": shot_make,
        "shot_dunks": shot_dunk,
        "shot_twos": shot_two,
        "shot_threes": shot_three,
        "learner_shot_attempts": learner_shot.astype(jnp.int8),
        "learner_shot_makes": (learner_shot & shot_make.astype(jnp.bool_)).astype(jnp.int8),
        "learner_shot_dunks": (learner_shot & shot_dunk.astype(jnp.bool_)).astype(jnp.int8),
        "learner_shot_twos": (learner_shot & shot_two.astype(jnp.bool_)).astype(jnp.int8),
        "learner_shot_threes": (learner_shot & shot_three.astype(jnp.bool_)).astype(jnp.int8),
        "opponent_shot_attempts": opponent_shot.astype(jnp.int8),
        "opponent_shot_makes": (opponent_shot & shot_make.astype(jnp.bool_)).astype(jnp.int8),
        "opponent_shot_dunks": (opponent_shot & shot_dunk.astype(jnp.bool_)).astype(jnp.int8),
        "opponent_shot_twos": (opponent_shot & shot_two.astype(jnp.bool_)).astype(jnp.int8),
        "opponent_shot_threes": (opponent_shot & shot_three.astype(jnp.bool_)).astype(jnp.int8),
    }


def _build_rebound_transition_metrics(env_out, jnp) -> dict[str, Any]:
    return {
        "rebound_attempts": env_out.rebound_attempt.astype(jnp.int8),
        "offensive_rebounds": env_out.offensive_rebound.astype(jnp.int8),
        "defensive_rebounds": env_out.defensive_rebound.astype(jnp.int8),
        "rebound_target_cells": env_out.rebound_target_cell.astype(jnp.int32),
        "rebound_winners": env_out.rebound_winner.astype(jnp.int32),
        "rebound_global_contests": env_out.rebound_global_contest.astype(jnp.int8),
        "shot_clock_reset_14": env_out.shot_clock_reset_14.astype(jnp.int8),
        "rebound_reward_advances": env_out.rebound_reward_advance.astype(jnp.float32),
        "rebound_reward_settlements": env_out.rebound_reward_settlement.astype(jnp.float32),
    }


def _mask_rebound_transition_metrics(metrics: dict[str, Any], active_step, jnp) -> dict[str, Any]:
    masked: dict[str, Any] = {}
    for key, value in metrics.items():
        fallback = -1 if key in {"rebound_target_cells", "rebound_winners"} else 0
        masked[key] = jnp.where(active_step, value, fallback)
    return masked


def _build_turnover_transition_metrics(static, env_out, jnp) -> dict[str, Any]:
    turnover = env_out.turnover.astype(jnp.int8)
    turnover_bool = turnover.astype(jnp.bool_)
    safe_player = jnp.clip(env_out.turnover_player, 0, int(static.role_encoding.shape[0]) - 1)
    valid_player = env_out.turnover_player >= 0
    learner_turnover = turnover_bool & valid_player & (static.training_player_mask[safe_player] > 0.5)
    opponent_turnover = turnover_bool & valid_player & (~learner_turnover)

    def _reason_flag(reason: int):
        return (turnover_bool & (env_out.turnover_reason == int(reason))).astype(jnp.int8)

    return {
        "learner_turnovers": learner_turnover.astype(jnp.int8),
        "opponent_turnovers": opponent_turnover.astype(jnp.int8),
        "turnover_pass_out_of_bounds": _reason_flag(TURNOVER_REASON_PASS_OUT_OF_BOUNDS),
        "turnover_intercepted": _reason_flag(TURNOVER_REASON_INTERCEPTED),
        "turnover_defender_pressure": _reason_flag(TURNOVER_REASON_DEFENDER_PRESSURE),
        "turnover_move_out_of_bounds": _reason_flag(TURNOVER_REASON_MOVE_OUT_OF_BOUNDS),
        "turnover_shot_clock": _reason_flag(TURNOVER_REASON_SHOT_CLOCK),
        "turnover_offensive_three_seconds": _reason_flag(TURNOVER_REASON_OFFENSIVE_THREE_SECONDS),
    }


def _build_intent_transition_metrics(state) -> dict[str, Any]:
    return {
        "intent_index": state.intent_index.astype(state.intent_index.dtype),
        "intent_active": state.intent_active.astype(state.intent_active.dtype),
        "intent_age": state.intent_age.astype(state.intent_age.dtype),
        "intent_commitment_remaining": state.intent_commitment_remaining.astype(
            state.intent_commitment_remaining.dtype
        ),
        "intent_visible_to_defense": state.intent_visible_to_defense.astype(
            state.intent_visible_to_defense.dtype
        ),
        "defense_intent_index": state.defense_intent_index.astype(state.defense_intent_index.dtype),
        "defense_intent_active": state.defense_intent_active.astype(state.defense_intent_active.dtype),
        "defense_intent_age": state.defense_intent_age.astype(state.defense_intent_age.dtype),
        "defense_intent_commitment_remaining": state.defense_intent_commitment_remaining.astype(
            state.defense_intent_commitment_remaining.dtype
        ),
    }


def _zero_selector_transition_metrics(state, jnp) -> dict[str, Any]:
    batch_shape = state.intent_index.shape
    return {
        "selector_used": jnp.zeros(batch_shape, dtype=jnp.int8),
        "selector_applied": jnp.zeros(batch_shape, dtype=jnp.int8),
        "selector_fallback_used": jnp.zeros(batch_shape, dtype=jnp.int8),
        "selector_boundary_episode_start": jnp.zeros(batch_shape, dtype=jnp.int8),
        "selector_boundary_commitment_timeout": jnp.zeros(batch_shape, dtype=jnp.int8),
        "selector_boundary_completed_pass": jnp.zeros(batch_shape, dtype=jnp.int8),
        "selector_intent_index": jnp.full(batch_shape, -1, dtype=jnp.int32),
        "selector_old_log_prob": jnp.zeros(batch_shape, dtype=jnp.float32),
        "selector_value": jnp.zeros(batch_shape, dtype=jnp.float32),
        "selector_entropy": jnp.zeros(batch_shape, dtype=jnp.float32),
        "selector_max_prob": jnp.zeros(batch_shape, dtype=jnp.float32),
    }


def _where_state(mask, selected_state, fallback_state, jnp):
    replaced = []
    for selected_value, fallback_value in zip(selected_state, fallback_state):
        if getattr(selected_value, "ndim", 0) <= 1:
            replaced.append(jnp.where(mask, selected_value, fallback_value))
        else:
            expand_shape = (mask.shape[0],) + (1,) * (selected_value.ndim - 1)
            replaced.append(jnp.where(mask.reshape(expand_shape), selected_value, fallback_value))
    return type(fallback_state)(*replaced)


def _selector_segment_application_masks(
    state,
    *,
    alpha_used,
    multiselect_enabled,
    completed_pass_boundary,
    selector_min_play_steps,
    jnp,
):
    active = state.intent_active.astype(jnp.bool_)
    episode_start = active & (state.intent_age == 0)
    commitment_timeout = (
        multiselect_enabled
        & active
        & (state.intent_age > 0)
        & (state.intent_commitment_remaining <= 0)
    )
    completed_pass = (
        multiselect_enabled
        & active
        & jnp.asarray(completed_pass_boundary).astype(jnp.bool_)
        & (state.intent_age >= jnp.asarray(selector_min_play_steps, dtype=jnp.int32))
    )
    eligible = episode_start | commitment_timeout | completed_pass
    used = eligible & alpha_used
    # Exploration already happens through epsilon-mixed selector probabilities. A random
    # segment fallback makes multiselect active during warmup and mostly random when
    # selector alpha is below 1.0, which is not the intended learned-selector behavior.
    fallback_used = jnp.zeros_like(used, dtype=jnp.bool_)
    applied = used
    return episode_start, commitment_timeout, completed_pass, used, applied, fallback_used


def _maybe_apply_selector_segment_start(
    static,
    state,
    params,
    flat_obs,
    selector_key,
    selector_alpha,
    selector_eps,
    selector_multiselect_enabled,
    completed_pass_boundary,
    selector_min_play_steps,
    jax,
    jnp,
    spec: ActorCriticSpec,
):
    metrics = _zero_selector_transition_metrics(state, jnp)
    if not bool(spec.intent_selector_enabled):
        return state, metrics

    alpha = jnp.clip(jnp.asarray(selector_alpha, dtype=jnp.float32), 0.0, 1.0)
    multiselect_enabled = jnp.asarray(selector_multiselect_enabled).astype(jnp.bool_)
    should_run = (
        static.enable_intent_learning.astype(jnp.bool_)
        & (static.training_role_flag > 0.0)
        & (alpha > 0.0)
    )

    def _disabled(_):
        return state, metrics

    def _enabled(_):
        batch_size = int(state.intent_index.shape[0])
        neutral_context = {
            "intent_index": jnp.zeros((batch_size,), dtype=jnp.int32),
            "intent_gate": jnp.zeros((batch_size,), dtype=jnp.float32),
        }
        selector_out = actor_critic_forward(
            params,
            flat_obs,
            spec,
            jnp,
            intent_context=neutral_context,
        )
        logits = selector_out["selector_logits"]
        raw_probs = jax.nn.softmax(logits, axis=-1)
        eps = jnp.clip(jnp.asarray(selector_eps, dtype=jnp.float32), 0.0, 1.0)
        uniform = jnp.full_like(raw_probs, 1.0 / float(max(1, int(spec.num_intents))))
        probs = ((1.0 - eps) * raw_probs) + (eps * uniform)
        log_probs = jnp.log(jnp.maximum(probs, 1.0e-8))
        sample_key, alpha_key = jax.random.split(selector_key, 2)
        sampled_intent = jax.random.categorical(sample_key, log_probs, axis=-1).astype(jnp.int32)
        sampled_log_prob = jnp.take_along_axis(
            log_probs,
            sampled_intent[:, None],
            axis=-1,
        )[:, 0]
        entropy = -jnp.sum(probs * log_probs, axis=-1)
        max_prob = jnp.max(probs, axis=-1)
        alpha_used = jax.random.uniform(alpha_key, shape=(batch_size,)) < alpha
        (
            episode_start,
            commitment_timeout,
            completed_pass,
            used,
            applied,
            fallback_used,
        ) = _selector_segment_application_masks(
            state,
            alpha_used=alpha_used,
            multiselect_enabled=multiselect_enabled,
            completed_pass_boundary=completed_pass_boundary,
            selector_min_play_steps=selector_min_play_steps,
            jnp=jnp,
        )
        selected_state = set_offense_intent_state_batch(
            static,
            state,
            sampled_intent,
            jnp.ones((batch_size,), dtype=jnp.int8),
            jnp,
        )
        next_state = _where_state(applied, selected_state, state, jnp)
        return next_state, {
            "selector_used": used.astype(jnp.int8),
            "selector_applied": applied.astype(jnp.int8),
            "selector_fallback_used": fallback_used.astype(jnp.int8),
            "selector_boundary_episode_start": (applied & episode_start).astype(jnp.int8),
            "selector_boundary_commitment_timeout": (applied & commitment_timeout).astype(jnp.int8),
            "selector_boundary_completed_pass": (applied & completed_pass & (~commitment_timeout)).astype(jnp.int8),
            "selector_intent_index": jnp.where(used, sampled_intent, jnp.asarray(-1, dtype=jnp.int32)),
            "selector_old_log_prob": jnp.where(used, sampled_log_prob, jnp.asarray(0.0, dtype=jnp.float32)),
            "selector_value": jnp.where(used, selector_out["selector_values"], jnp.asarray(0.0, dtype=jnp.float32)),
            "selector_entropy": jnp.where(used, entropy, jnp.asarray(0.0, dtype=jnp.float32)),
            "selector_max_prob": jnp.where(used, max_prob, jnp.asarray(0.0, dtype=jnp.float32)),
        }

    return jax.lax.cond(should_run, _enabled, _disabled, operand=None)


def _compute_final_selector_values(params, flat_obs, spec: ActorCriticSpec, jnp):
    if not bool(spec.intent_selector_enabled):
        return jnp.zeros((flat_obs.shape[0],), dtype=jnp.float32)
    batch_size = int(flat_obs.shape[0])
    neutral_context = {
        "intent_index": jnp.zeros((batch_size,), dtype=jnp.int32),
        "intent_gate": jnp.zeros((batch_size,), dtype=jnp.float32),
    }
    return actor_critic_forward(
        params,
        flat_obs,
        spec,
        jnp,
        intent_context=neutral_context,
    )["selector_values"]


def _mask_step_metrics(metrics: dict[str, Any], active_mask, jnp) -> dict[str, Any]:
    return {
        key: jnp.where(active_mask, value, jnp.zeros_like(value))
        for key, value in metrics.items()
    }


def build_compiled_rollout_runner(jax, jnp, spec: ActorCriticSpec):
    def _runner(
        static,
        initial_state,
        params,
        rollout_key,
        horizon: int,
        selector_alpha=0.0,
        selector_eps=0.0,
        selector_multiselect_enabled=False,
        selector_min_play_steps=3,
        single_episode_rollout=False,
    ):
        training_ids, opponent_ids = resolve_team_player_ids(static, jax, jnp)
        n_players = int(static.role_encoding.shape[0])

        def _scan_step(carry, _):
            state, key, completed_pass_boundary = carry
            key, selector_key, policy_key, opponent_key, env_key, reset_key = jax.random.split(key, 6)
            active_step = (~state.episode_ended.astype(jnp.bool_))
            flat_obs = build_policy_observation_batch(
                static,
                state,
                jnp,
                model_type=spec.model_type,
            )
            policy_state, selector_metrics = _maybe_apply_selector_segment_start(
                static,
                state,
                params,
                flat_obs,
                selector_key,
                selector_alpha,
                selector_eps,
                selector_multiselect_enabled,
                completed_pass_boundary,
                selector_min_play_steps,
                jax,
                jnp,
                spec,
            )
            policy_state = _where_state(active_step, policy_state, state, jnp)
            selector_metrics = _mask_step_metrics(selector_metrics, active_step, jnp)
            policy_intent_context = build_policy_intent_context_batch(static, policy_state, jnp)
            full_action_mask = build_action_masks_batch(static, policy_state, jnp)
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
                intent_context=policy_intent_context,
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
                policy_state,
                full_actions,
                env_keys,
                jax,
                jnp,
            )
            reset_keys = jax.random.split(reset_key, initial_state.positions.shape[0])
            reset_state = reset_batch_minimal(static, reset_keys, jax, jnp)
            reset_done = env_out.done & (~jnp.asarray(single_episode_rollout).astype(jnp.bool_))
            next_state = replace_done_states(env_out.state, reset_state, reset_done, jnp)
            aggregated_reward = build_aggregated_reward_batch(static, env_out.rewards, jnp)
            shot_metrics = _build_shot_type_transition_metrics(static, env_out, jnp)
            turnover_metrics = _build_turnover_transition_metrics(static, env_out, jnp)
            shot_metrics = _mask_step_metrics(shot_metrics, active_step, jnp)
            rebound_metrics = _mask_rebound_transition_metrics(
                _build_rebound_transition_metrics(env_out, jnp),
                active_step,
                jnp,
            )
            turnover_metrics = _mask_step_metrics(turnover_metrics, active_step, jnp)
            intent_metrics = _mask_step_metrics(
                _build_intent_transition_metrics(policy_state),
                active_step,
                jnp,
            )
            masked_reward = jnp.where(active_step, aggregated_reward, jnp.zeros_like(aggregated_reward))
            masked_done = jnp.where(
                active_step,
                env_out.done.astype(jnp.int8),
                jnp.zeros_like(env_out.done.astype(jnp.int8)),
            )
            transition = TrajectoryBatch(
                active_mask=active_step.astype(jnp.float32),
                episode_start=(active_step & (policy_state.step_count == 0)).astype(jnp.int8),
                flat_obs=flat_obs,
                policy_intent_index=policy_intent_context["intent_index"],
                policy_intent_gate=policy_intent_context["intent_gate"],
                action_mask=training_action_mask,
                actions=policy_out["sampled_actions"],
                full_actions=full_actions,
                opponent_deterministic_episode=jnp.zeros_like(active_step, dtype=jnp.float32),
                selected_log_probs=policy_out["selected_log_probs"],
                values=policy_out["values"],
                rewards=masked_reward,
                dones=masked_done,
                phi_r_shape=jnp.where(active_step, env_out.phi_r_shape.astype(jnp.float32), 0.0),
                phi_prev=jnp.where(active_step, env_out.phi_prev.astype(jnp.float32), 0.0),
                phi_next=jnp.where(active_step, env_out.phi_next.astype(jnp.float32), 0.0),
                phi_beta=jnp.where(active_step, env_out.phi_beta.astype(jnp.float32), 0.0),
                pass_attempts=jnp.where(active_step, env_out.pass_attempt.astype(jnp.int8), 0),
                completed_passes=jnp.where(active_step, env_out.completed_pass.astype(jnp.int8), 0),
                assists=jnp.where(active_step, env_out.assist.astype(jnp.int8), 0),
                turnovers=jnp.where(active_step, env_out.turnover.astype(jnp.int8), 0),
                **turnover_metrics,
                **shot_metrics,
                **rebound_metrics,
                **intent_metrics,
                **selector_metrics,
                offensive_three_seconds=jnp.where(
                    active_step,
                    env_out.offensive_three_seconds.astype(jnp.int8),
                    0,
                ),
                defensive_lane_violations=jnp.where(
                    active_step,
                    env_out.defensive_lane_violation.astype(jnp.int8),
                    0,
                ),
                terminal_episode_steps=jnp.where(
                    active_step,
                    env_out.terminal_episode_steps.astype(jnp.int32),
                    0,
                ),
                offense_score_delta=jnp.where(
                    active_step,
                    (env_out.state.offense_score - policy_state.offense_score).astype(jnp.float32),
                    0.0,
                ),
                defense_score_delta=jnp.where(
                    active_step,
                    (env_out.state.defense_score - policy_state.defense_score).astype(jnp.float32),
                    0.0,
                ),
            )
            next_completed_pass_boundary = (
                active_step
                & (
                    env_out.completed_pass.astype(jnp.bool_)
                    | env_out.offensive_rebound.astype(jnp.bool_)
                )
                & (~env_out.done.astype(jnp.bool_))
            )
            return (next_state, key, next_completed_pass_boundary), transition

        initial_completed_pass_boundary = jnp.zeros(
            (int(initial_state.positions.shape[0]),),
            dtype=jnp.bool_,
        )
        (final_state, _, _), trajectory = jax.lax.scan(
            _scan_step,
            (initial_state, rollout_key, initial_completed_pass_boundary),
            xs=None,
            length=int(horizon),
        )
        final_flat_obs = build_policy_observation_batch(
            static,
            final_state,
            jnp,
            model_type=spec.model_type,
        )
        final_intent_context = build_policy_intent_context_batch(static, final_state, jnp)
        final_action_mask = build_action_masks_batch(static, final_state, jnp)[:, training_ids, :]
        final_forward = actor_critic_forward(
            params,
            final_flat_obs,
            spec,
            jnp,
            intent_context=final_intent_context,
        )
        bootstrap_values = final_forward["values"]
        final_selector_values = _compute_final_selector_values(
            params,
            final_flat_obs,
            spec,
            jnp,
        )
        return RolloutOutput(
            trajectory=trajectory,
            final_state=final_state,
            bootstrap_values=bootstrap_values,
            final_selector_values=final_selector_values,
            final_flat_obs=final_flat_obs,
            final_action_mask=final_action_mask,
        )

    return jax.jit(_runner, static_argnums=(4,))


def build_compiled_frozen_opponent_rollout_runner(jax, jnp, spec: ActorCriticSpec):
    def _runner(
        static,
        initial_state,
        params,
        opponent_params,
        rollout_key,
        horizon: int,
        selector_alpha=0.0,
        selector_eps=0.0,
        selector_multiselect_enabled=False,
        selector_min_play_steps=3,
        single_episode_rollout=False,
        opponent_deterministic_episode_prob=0.0,
    ):
        training_ids, opponent_ids = resolve_team_player_ids(static, jax, jnp)
        n_players = int(static.role_encoding.shape[0])
        batch_size = int(initial_state.positions.shape[0])
        opponent_deterministic_episode_prob = jnp.clip(
            jnp.asarray(opponent_deterministic_episode_prob, dtype=jnp.float32),
            0.0,
            1.0,
        )

        def _scan_step(carry, _):
            state, key, completed_pass_boundary, opponent_deterministic_episode = carry
            key, selector_key, policy_key, opponent_key, env_key, reset_key, opponent_det_key = jax.random.split(key, 7)
            active_step = (~state.episode_ended.astype(jnp.bool_))
            flat_obs = build_policy_observation_batch(
                static,
                state,
                jnp,
                model_type=spec.model_type,
            )
            policy_state, selector_metrics = _maybe_apply_selector_segment_start(
                static,
                state,
                params,
                flat_obs,
                selector_key,
                selector_alpha,
                selector_eps,
                selector_multiselect_enabled,
                completed_pass_boundary,
                selector_min_play_steps,
                jax,
                jnp,
                spec,
            )
            policy_state = _where_state(active_step, policy_state, state, jnp)
            selector_metrics = _mask_step_metrics(selector_metrics, active_step, jnp)
            opponent_flat_obs = build_policy_observation_batch_with_role_flag(
                static,
                policy_state,
                -static.training_role_flag,
                jnp,
                model_type=spec.model_type,
            )
            policy_intent_context = build_policy_intent_context_batch(static, policy_state, jnp)
            opponent_intent_context = build_policy_intent_context_batch_with_role_flag(
                static,
                policy_state,
                -static.training_role_flag,
                jnp,
            )
            full_action_mask = build_action_masks_batch(static, policy_state, jnp)
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
                intent_context=policy_intent_context,
            )
            opponent_out = run_actor_critic(
                opponent_params,
                opponent_flat_obs,
                opponent_action_mask,
                spec,
                opponent_key,
                jax,
                jnp,
                intent_context=opponent_intent_context,
            )
            opponent_actions = jnp.where(
                opponent_deterministic_episode[:, None],
                opponent_out["deterministic_actions"],
                opponent_out["sampled_actions"],
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
                policy_state,
                full_actions,
                env_keys,
                jax,
                jnp,
            )
            reset_keys = jax.random.split(reset_key, initial_state.positions.shape[0])
            reset_state = reset_batch_minimal(static, reset_keys, jax, jnp)
            reset_done = env_out.done & (~jnp.asarray(single_episode_rollout).astype(jnp.bool_))
            next_state = replace_done_states(env_out.state, reset_state, reset_done, jnp)
            next_opponent_deterministic_episode_sample = jax.random.bernoulli(
                opponent_det_key,
                opponent_deterministic_episode_prob,
                (batch_size,),
            )
            next_opponent_deterministic_episode = jnp.where(
                reset_done,
                next_opponent_deterministic_episode_sample,
                opponent_deterministic_episode,
            )
            aggregated_reward = build_aggregated_reward_batch(static, env_out.rewards, jnp)
            shot_metrics = _build_shot_type_transition_metrics(static, env_out, jnp)
            turnover_metrics = _build_turnover_transition_metrics(static, env_out, jnp)
            shot_metrics = _mask_step_metrics(shot_metrics, active_step, jnp)
            rebound_metrics = _mask_rebound_transition_metrics(
                _build_rebound_transition_metrics(env_out, jnp),
                active_step,
                jnp,
            )
            turnover_metrics = _mask_step_metrics(turnover_metrics, active_step, jnp)
            intent_metrics = _mask_step_metrics(
                _build_intent_transition_metrics(policy_state),
                active_step,
                jnp,
            )
            masked_reward = jnp.where(active_step, aggregated_reward, jnp.zeros_like(aggregated_reward))
            masked_done = jnp.where(
                active_step,
                env_out.done.astype(jnp.int8),
                jnp.zeros_like(env_out.done.astype(jnp.int8)),
            )
            transition = TrajectoryBatch(
                active_mask=active_step.astype(jnp.float32),
                episode_start=(active_step & (policy_state.step_count == 0)).astype(jnp.int8),
                flat_obs=flat_obs,
                policy_intent_index=policy_intent_context["intent_index"],
                policy_intent_gate=policy_intent_context["intent_gate"],
                action_mask=training_action_mask,
                actions=policy_out["sampled_actions"],
                full_actions=full_actions,
                opponent_deterministic_episode=jnp.where(
                    active_step,
                    opponent_deterministic_episode.astype(jnp.float32),
                    0.0,
                ),
                selected_log_probs=policy_out["selected_log_probs"],
                values=policy_out["values"],
                rewards=masked_reward,
                dones=masked_done,
                phi_r_shape=jnp.where(active_step, env_out.phi_r_shape.astype(jnp.float32), 0.0),
                phi_prev=jnp.where(active_step, env_out.phi_prev.astype(jnp.float32), 0.0),
                phi_next=jnp.where(active_step, env_out.phi_next.astype(jnp.float32), 0.0),
                phi_beta=jnp.where(active_step, env_out.phi_beta.astype(jnp.float32), 0.0),
                pass_attempts=jnp.where(active_step, env_out.pass_attempt.astype(jnp.int8), 0),
                completed_passes=jnp.where(active_step, env_out.completed_pass.astype(jnp.int8), 0),
                assists=jnp.where(active_step, env_out.assist.astype(jnp.int8), 0),
                turnovers=jnp.where(active_step, env_out.turnover.astype(jnp.int8), 0),
                **turnover_metrics,
                **shot_metrics,
                **rebound_metrics,
                **intent_metrics,
                **selector_metrics,
                offensive_three_seconds=jnp.where(
                    active_step,
                    env_out.offensive_three_seconds.astype(jnp.int8),
                    0,
                ),
                defensive_lane_violations=jnp.where(
                    active_step,
                    env_out.defensive_lane_violation.astype(jnp.int8),
                    0,
                ),
                terminal_episode_steps=jnp.where(
                    active_step,
                    env_out.terminal_episode_steps.astype(jnp.int32),
                    0,
                ),
                offense_score_delta=jnp.where(
                    active_step,
                    (env_out.state.offense_score - policy_state.offense_score).astype(jnp.float32),
                    0.0,
                ),
                defense_score_delta=jnp.where(
                    active_step,
                    (env_out.state.defense_score - policy_state.defense_score).astype(jnp.float32),
                    0.0,
                ),
            )
            next_completed_pass_boundary = (
                active_step
                & (
                    env_out.completed_pass.astype(jnp.bool_)
                    | env_out.offensive_rebound.astype(jnp.bool_)
                )
                & (~env_out.done.astype(jnp.bool_))
            )
            return (
                next_state,
                key,
                next_completed_pass_boundary,
                next_opponent_deterministic_episode,
            ), transition

        scan_key, opponent_det_init_key = jax.random.split(rollout_key)
        initial_completed_pass_boundary = jnp.zeros(
            (batch_size,),
            dtype=jnp.bool_,
        )
        initial_opponent_deterministic_episode = jax.random.bernoulli(
            opponent_det_init_key,
            opponent_deterministic_episode_prob,
            (batch_size,),
        )
        (final_state, _, _, _), trajectory = jax.lax.scan(
            _scan_step,
            (
                initial_state,
                scan_key,
                initial_completed_pass_boundary,
                initial_opponent_deterministic_episode,
            ),
            xs=None,
            length=int(horizon),
        )
        final_flat_obs = build_policy_observation_batch(
            static,
            final_state,
            jnp,
            model_type=spec.model_type,
        )
        final_intent_context = build_policy_intent_context_batch(static, final_state, jnp)
        final_action_mask = build_action_masks_batch(static, final_state, jnp)[:, training_ids, :]
        final_forward = actor_critic_forward(
            params,
            final_flat_obs,
            spec,
            jnp,
            intent_context=final_intent_context,
        )
        bootstrap_values = final_forward["values"]
        final_selector_values = _compute_final_selector_values(
            params,
            final_flat_obs,
            spec,
            jnp,
        )
        return RolloutOutput(
            trajectory=trajectory,
            final_state=final_state,
            bootstrap_values=bootstrap_values,
            final_selector_values=final_selector_values,
            final_flat_obs=final_flat_obs,
            final_action_mask=final_action_mask,
        )

    return jax.jit(_runner, static_argnums=(5,))


def build_compiled_grouped_opponent_rollout_runner(jax, jnp, spec: ActorCriticSpec):
    def _runner(
        static,
        initial_state,
        params,
        opponent_params_by_group,
        rollout_key,
        horizon: int,
        opponent_group_count: int,
        selector_alpha=0.0,
        selector_eps=0.0,
        selector_multiselect_enabled=False,
        selector_min_play_steps=3,
        single_episode_rollout=False,
        opponent_deterministic_episode_prob=0.0,
    ):
        training_ids, opponent_ids = resolve_team_player_ids(static, jax, jnp)
        n_players = int(static.role_encoding.shape[0])
        group_count = int(opponent_group_count)
        batch_size = int(initial_state.positions.shape[0])
        group_size = batch_size // group_count
        opponent_deterministic_episode_prob = jnp.clip(
            jnp.asarray(opponent_deterministic_episode_prob, dtype=jnp.float32),
            0.0,
            1.0,
        )

        def _sample_grouped_opponent_actions(
            opponent_flat_obs,
            opponent_action_mask,
            opponent_intent_context,
            key,
        ):
            grouped_obs = opponent_flat_obs.reshape(
                group_count,
                group_size,
                int(opponent_flat_obs.shape[-1]),
            )
            grouped_mask = opponent_action_mask.reshape(
                group_count,
                group_size,
                int(opponent_action_mask.shape[-2]),
                int(opponent_action_mask.shape[-1]),
            )
            grouped_intent_context = {
                name: value.reshape(group_count, group_size)
                for name, value in opponent_intent_context.items()
            }
            group_keys = jax.random.split(key, group_count)

            def _run_group(group_params, group_obs, group_mask, group_intent_context, group_key):
                group_out = run_actor_critic(
                    group_params,
                    group_obs,
                    group_mask,
                    spec,
                    group_key,
                    jax,
                    jnp,
                    intent_context=group_intent_context,
                )
                return group_out["sampled_actions"], group_out["deterministic_actions"]

            grouped_sampled_actions, grouped_deterministic_actions = jax.vmap(_run_group)(
                opponent_params_by_group,
                grouped_obs,
                grouped_mask,
                grouped_intent_context,
                group_keys,
            )
            return (
                grouped_sampled_actions.reshape(
                    batch_size,
                    int(spec.training_player_count),
                ),
                grouped_deterministic_actions.reshape(
                    batch_size,
                    int(spec.training_player_count),
                ),
            )

        def _scan_step(carry, _):
            state, key, completed_pass_boundary, opponent_deterministic_episode = carry
            key, selector_key, policy_key, opponent_key, env_key, reset_key, opponent_det_key = jax.random.split(key, 7)
            active_step = (~state.episode_ended.astype(jnp.bool_))
            flat_obs = build_policy_observation_batch(
                static,
                state,
                jnp,
                model_type=spec.model_type,
            )
            policy_state, selector_metrics = _maybe_apply_selector_segment_start(
                static,
                state,
                params,
                flat_obs,
                selector_key,
                selector_alpha,
                selector_eps,
                selector_multiselect_enabled,
                completed_pass_boundary,
                selector_min_play_steps,
                jax,
                jnp,
                spec,
            )
            policy_state = _where_state(active_step, policy_state, state, jnp)
            selector_metrics = _mask_step_metrics(selector_metrics, active_step, jnp)
            opponent_flat_obs = build_policy_observation_batch_with_role_flag(
                static,
                policy_state,
                -static.training_role_flag,
                jnp,
                model_type=spec.model_type,
            )
            policy_intent_context = build_policy_intent_context_batch(static, policy_state, jnp)
            opponent_intent_context = build_policy_intent_context_batch_with_role_flag(
                static,
                policy_state,
                -static.training_role_flag,
                jnp,
            )
            full_action_mask = build_action_masks_batch(static, policy_state, jnp)
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
                intent_context=policy_intent_context,
            )
            sampled_opponent_actions, deterministic_opponent_actions = _sample_grouped_opponent_actions(
                opponent_flat_obs,
                opponent_action_mask,
                opponent_intent_context,
                opponent_key,
            )
            opponent_actions = jnp.where(
                opponent_deterministic_episode[:, None],
                deterministic_opponent_actions,
                sampled_opponent_actions,
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
                policy_state,
                full_actions,
                env_keys,
                jax,
                jnp,
            )
            reset_keys = jax.random.split(reset_key, initial_state.positions.shape[0])
            reset_state = reset_batch_minimal(static, reset_keys, jax, jnp)
            reset_done = env_out.done & (~jnp.asarray(single_episode_rollout).astype(jnp.bool_))
            next_state = replace_done_states(env_out.state, reset_state, reset_done, jnp)
            next_opponent_deterministic_episode_sample = jax.random.bernoulli(
                opponent_det_key,
                opponent_deterministic_episode_prob,
                (batch_size,),
            )
            next_opponent_deterministic_episode = jnp.where(
                reset_done,
                next_opponent_deterministic_episode_sample,
                opponent_deterministic_episode,
            )
            aggregated_reward = build_aggregated_reward_batch(static, env_out.rewards, jnp)
            shot_metrics = _build_shot_type_transition_metrics(static, env_out, jnp)
            turnover_metrics = _build_turnover_transition_metrics(static, env_out, jnp)
            shot_metrics = _mask_step_metrics(shot_metrics, active_step, jnp)
            rebound_metrics = _mask_rebound_transition_metrics(
                _build_rebound_transition_metrics(env_out, jnp),
                active_step,
                jnp,
            )
            turnover_metrics = _mask_step_metrics(turnover_metrics, active_step, jnp)
            intent_metrics = _mask_step_metrics(
                _build_intent_transition_metrics(policy_state),
                active_step,
                jnp,
            )
            masked_reward = jnp.where(active_step, aggregated_reward, jnp.zeros_like(aggregated_reward))
            masked_done = jnp.where(
                active_step,
                env_out.done.astype(jnp.int8),
                jnp.zeros_like(env_out.done.astype(jnp.int8)),
            )
            transition = TrajectoryBatch(
                active_mask=active_step.astype(jnp.float32),
                episode_start=(active_step & (policy_state.step_count == 0)).astype(jnp.int8),
                flat_obs=flat_obs,
                policy_intent_index=policy_intent_context["intent_index"],
                policy_intent_gate=policy_intent_context["intent_gate"],
                action_mask=training_action_mask,
                actions=policy_out["sampled_actions"],
                full_actions=full_actions,
                opponent_deterministic_episode=jnp.where(
                    active_step,
                    opponent_deterministic_episode.astype(jnp.float32),
                    0.0,
                ),
                selected_log_probs=policy_out["selected_log_probs"],
                values=policy_out["values"],
                rewards=masked_reward,
                dones=masked_done,
                phi_r_shape=jnp.where(active_step, env_out.phi_r_shape.astype(jnp.float32), 0.0),
                phi_prev=jnp.where(active_step, env_out.phi_prev.astype(jnp.float32), 0.0),
                phi_next=jnp.where(active_step, env_out.phi_next.astype(jnp.float32), 0.0),
                phi_beta=jnp.where(active_step, env_out.phi_beta.astype(jnp.float32), 0.0),
                pass_attempts=jnp.where(active_step, env_out.pass_attempt.astype(jnp.int8), 0),
                completed_passes=jnp.where(active_step, env_out.completed_pass.astype(jnp.int8), 0),
                assists=jnp.where(active_step, env_out.assist.astype(jnp.int8), 0),
                turnovers=jnp.where(active_step, env_out.turnover.astype(jnp.int8), 0),
                **turnover_metrics,
                **shot_metrics,
                **rebound_metrics,
                **intent_metrics,
                **selector_metrics,
                offensive_three_seconds=jnp.where(
                    active_step,
                    env_out.offensive_three_seconds.astype(jnp.int8),
                    0,
                ),
                defensive_lane_violations=jnp.where(
                    active_step,
                    env_out.defensive_lane_violation.astype(jnp.int8),
                    0,
                ),
                terminal_episode_steps=jnp.where(
                    active_step,
                    env_out.terminal_episode_steps.astype(jnp.int32),
                    0,
                ),
                offense_score_delta=jnp.where(
                    active_step,
                    (env_out.state.offense_score - policy_state.offense_score).astype(jnp.float32),
                    0.0,
                ),
                defense_score_delta=jnp.where(
                    active_step,
                    (env_out.state.defense_score - policy_state.defense_score).astype(jnp.float32),
                    0.0,
                ),
            )
            next_completed_pass_boundary = (
                active_step
                & (
                    env_out.completed_pass.astype(jnp.bool_)
                    | env_out.offensive_rebound.astype(jnp.bool_)
                )
                & (~env_out.done.astype(jnp.bool_))
            )
            return (
                next_state,
                key,
                next_completed_pass_boundary,
                next_opponent_deterministic_episode,
            ), transition

        scan_key, opponent_det_init_key = jax.random.split(rollout_key)
        initial_completed_pass_boundary = jnp.zeros(
            (batch_size,),
            dtype=jnp.bool_,
        )
        initial_opponent_deterministic_episode = jax.random.bernoulli(
            opponent_det_init_key,
            opponent_deterministic_episode_prob,
            (batch_size,),
        )
        (final_state, _, _, _), trajectory = jax.lax.scan(
            _scan_step,
            (
                initial_state,
                scan_key,
                initial_completed_pass_boundary,
                initial_opponent_deterministic_episode,
            ),
            xs=None,
            length=int(horizon),
        )
        final_flat_obs = build_policy_observation_batch(
            static,
            final_state,
            jnp,
            model_type=spec.model_type,
        )
        final_intent_context = build_policy_intent_context_batch(static, final_state, jnp)
        final_action_mask = build_action_masks_batch(static, final_state, jnp)[:, training_ids, :]
        final_forward = actor_critic_forward(
            params,
            final_flat_obs,
            spec,
            jnp,
            intent_context=final_intent_context,
        )
        bootstrap_values = final_forward["values"]
        final_selector_values = _compute_final_selector_values(
            params,
            final_flat_obs,
            spec,
            jnp,
        )
        return RolloutOutput(
            trajectory=trajectory,
            final_state=final_state,
            bootstrap_values=bootstrap_values,
            final_selector_values=final_selector_values,
            final_flat_obs=final_flat_obs,
            final_action_mask=final_action_mask,
        )

    return jax.jit(_runner, static_argnums=(5, 6))


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
            policy_intent_context = build_policy_intent_context_batch(static, state, jnp)

            forward_out = actor_critic_forward(
                params,
                build_policy_observation_batch(
                    static,
                    state,
                    jnp,
                    model_type=spec.model_type,
                ),
                spec,
                jnp,
                intent_context=policy_intent_context,
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
            shot_metrics = _build_shot_type_transition_metrics(static, env_out, jnp)
            rebound_metrics = _build_rebound_transition_metrics(env_out, jnp)
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
                **shot_metrics,
                **rebound_metrics,
                **_build_intent_transition_metrics(state),
                offensive_three_seconds=env_out.offensive_three_seconds.astype(jnp.int8),
                defensive_lane_violations=env_out.defensive_lane_violation.astype(jnp.int8),
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


def build_compiled_frozen_opponent_eval_runner(jax, jnp, spec: ActorCriticSpec):
    def _runner(static, initial_state, params, opponent_params, rollout_key, horizon: int):
        training_ids, opponent_ids = resolve_team_player_ids(static, jax, jnp)
        n_players = int(static.role_encoding.shape[0])

        def _scan_step(carry, _):
            state, key = carry
            key, opponent_key, env_key = jax.random.split(key, 3)
            full_action_mask = build_action_masks_batch(static, state, jnp)
            training_action_mask = full_action_mask[:, training_ids, :]
            opponent_action_mask = full_action_mask[:, opponent_ids, :]
            policy_intent_context = build_policy_intent_context_batch(static, state, jnp)
            opponent_intent_context = build_policy_intent_context_batch_with_role_flag(
                static,
                state,
                -static.training_role_flag,
                jnp,
            )

            forward_out = actor_critic_forward(
                params,
                build_policy_observation_batch(
                    static,
                    state,
                    jnp,
                    model_type=spec.model_type,
                ),
                spec,
                jnp,
                intent_context=policy_intent_context,
            )
            masked_out = apply_action_mask(
                forward_out["flat_policy_logits"],
                training_action_mask,
                spec,
                jax,
                jnp,
            )
            opponent_out = run_actor_critic(
                opponent_params,
                build_policy_observation_batch_with_role_flag(
                    static,
                    state,
                    -static.training_role_flag,
                    jnp,
                    model_type=spec.model_type,
                ),
                opponent_action_mask,
                spec,
                opponent_key,
                jax,
                jnp,
                intent_context=opponent_intent_context,
            )
            full_actions = assemble_full_actions_jax(
                masked_out["deterministic_actions"],
                opponent_out["sampled_actions"],
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
            shot_metrics = _build_shot_type_transition_metrics(static, env_out, jnp)
            rebound_metrics = _build_rebound_transition_metrics(env_out, jnp)
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
                **shot_metrics,
                **rebound_metrics,
                **_build_intent_transition_metrics(state),
                offensive_three_seconds=env_out.offensive_three_seconds.astype(jnp.int8),
                defensive_lane_violations=env_out.defensive_lane_violation.astype(jnp.int8),
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

    return jax.jit(_runner, static_argnums=(5,))


def build_compiled_grouped_opponent_eval_runner(jax, jnp, spec: ActorCriticSpec):
    def _runner(
        static,
        initial_state,
        params,
        opponent_params_by_group,
        rollout_key,
        horizon: int,
        opponent_group_count: int,
    ):
        training_ids, opponent_ids = resolve_team_player_ids(static, jax, jnp)
        n_players = int(static.role_encoding.shape[0])
        group_count = int(opponent_group_count)
        batch_size = int(initial_state.positions.shape[0])
        group_size = batch_size // group_count

        def _sample_grouped_opponent_actions(
            opponent_flat_obs,
            opponent_action_mask,
            opponent_intent_context,
            key,
        ):
            grouped_obs = opponent_flat_obs.reshape(
                group_count,
                group_size,
                int(opponent_flat_obs.shape[-1]),
            )
            grouped_mask = opponent_action_mask.reshape(
                group_count,
                group_size,
                int(opponent_action_mask.shape[-2]),
                int(opponent_action_mask.shape[-1]),
            )
            grouped_intent_context = {
                name: value.reshape(group_count, group_size)
                for name, value in opponent_intent_context.items()
            }
            group_keys = jax.random.split(key, group_count)

            def _run_group(group_params, group_obs, group_mask, group_intent_context, group_key):
                group_out = run_actor_critic(
                    group_params,
                    group_obs,
                    group_mask,
                    spec,
                    group_key,
                    jax,
                    jnp,
                    intent_context=group_intent_context,
                )
                return group_out["sampled_actions"]

            grouped_actions = jax.vmap(_run_group)(
                opponent_params_by_group,
                grouped_obs,
                grouped_mask,
                grouped_intent_context,
                group_keys,
            )
            return grouped_actions.reshape(
                batch_size,
                int(spec.training_player_count),
            )

        def _scan_step(carry, _):
            state, key = carry
            key, opponent_key, env_key = jax.random.split(key, 3)
            full_action_mask = build_action_masks_batch(static, state, jnp)
            training_action_mask = full_action_mask[:, training_ids, :]
            opponent_action_mask = full_action_mask[:, opponent_ids, :]
            policy_intent_context = build_policy_intent_context_batch(static, state, jnp)
            opponent_intent_context = build_policy_intent_context_batch_with_role_flag(
                static,
                state,
                -static.training_role_flag,
                jnp,
            )

            forward_out = actor_critic_forward(
                params,
                build_policy_observation_batch(
                    static,
                    state,
                    jnp,
                    model_type=spec.model_type,
                ),
                spec,
                jnp,
                intent_context=policy_intent_context,
            )
            masked_out = apply_action_mask(
                forward_out["flat_policy_logits"],
                training_action_mask,
                spec,
                jax,
                jnp,
            )
            opponent_actions = _sample_grouped_opponent_actions(
                build_policy_observation_batch_with_role_flag(
                    static,
                    state,
                    -static.training_role_flag,
                    jnp,
                    model_type=spec.model_type,
                ),
                opponent_action_mask,
                opponent_intent_context,
                opponent_key,
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
            shot_metrics = _build_shot_type_transition_metrics(static, env_out, jnp)
            rebound_metrics = _build_rebound_transition_metrics(env_out, jnp)
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
                **shot_metrics,
                **rebound_metrics,
                **_build_intent_transition_metrics(state),
                offensive_three_seconds=env_out.offensive_three_seconds.astype(jnp.int8),
                defensive_lane_violations=env_out.defensive_lane_violation.astype(jnp.int8),
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

    return jax.jit(_runner, static_argnums=(5, 6))


def build_jitted_ppo_update_runner(jax, jnp, spec: ActorCriticSpec, trainer_config: TrainerConfig):
    import optax

    clip_range = jnp.asarray(trainer_config.ppo_clip_range, dtype=jnp.float32)
    value_coef = jnp.asarray(trainer_config.value_coef, dtype=jnp.float32)
    default_entropy_coef = jnp.asarray(trainer_config.entropy_coef, dtype=jnp.float32)
    epochs = int(trainer_config.policy_update_epochs)
    configured_minibatches = max(1, int(getattr(trainer_config, "ppo_minibatches", 1)))
    transform = build_adam_transform(
        optax,
        learning_rate=float(trainer_config.learning_rate),
    )

    def _loss_fn(params, batch: PPOBatch, entropy_coef):
        forward_out = actor_critic_forward(
            params,
            batch.flat_obs,
            spec,
            jnp,
            intent_context={
                "intent_index": batch.policy_intent_index,
                "intent_gate": batch.policy_intent_gate,
            },
        )
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
        active_mask = batch.active_mask.astype(jnp.float32)
        loss_weights = batch.loss_weights.astype(jnp.float32)
        loss_denominator = batch.loss_denominator.astype(jnp.float32)
        weight_den = jnp.maximum(jnp.max(loss_denominator), 1.0)

        def _masked_mean(values):
            return jnp.sum(values * loss_weights) / weight_den

        policy_loss = -_masked_mean(
            jnp.minimum(
                ratio * batch.advantages,
                clipped_ratio * batch.advantages,
            )
        )
        value_loss = _masked_mean(jnp.square(forward_out["values"] - batch.returns))
        entropy_bonus = _masked_mean(jnp.mean(masked_out["entropy"], axis=-1))
        approx_kl = _masked_mean((ratio - 1.0) - log_ratio)
        clip_fraction = _masked_mean((jnp.abs(ratio - 1.0) > clip_range).astype(jnp.float32))
        mean_abs_log_ratio = _masked_mean(jnp.abs(log_ratio))
        max_abs_log_ratio = jnp.max(
            jnp.where(active_mask > 0.0, jnp.abs(log_ratio), jnp.zeros_like(log_ratio))
        )
        total_loss = policy_loss + (value_coef * value_loss) - (entropy_coef * entropy_bonus)
        metrics = {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_bonus": entropy_bonus,
            "entropy_coef": entropy_coef,
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
            "mean_abs_log_ratio": mean_abs_log_ratio,
            "max_abs_log_ratio": max_abs_log_ratio,
            "ppo_active_sample_count": jnp.sum(active_mask),
            "ppo_active_sample_fraction": jnp.mean(active_mask),
            "ppo_loss_weight_sum": jnp.sum(loss_weights),
            "ppo_loss_denominator": weight_den,
        }
        return total_loss, metrics

    def _single_epoch(params, opt_state, batch, entropy_coef):
        (_, _), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
            params,
            batch,
            entropy_coef,
        )
        grad_norm = global_norm(grads, optax)
        new_params, new_opt_state = optimizer_update(
            params,
            grads,
            opt_state,
            transform=transform,
            optax=optax,
        )
        post_update_loss, metrics = _loss_fn(new_params, batch, entropy_coef)
        metrics = {
            **metrics,
            "grad_norm": grad_norm,
            "total_loss": post_update_loss,
        }
        return new_params, new_opt_state, metrics

    def _mean_metrics(metric_steps):
        return {
            name: jnp.mean(values)
            for name, values in metric_steps.items()
        }

    def _take_minibatch(batch: PPOBatch, indices):
        return PPOBatch(
            *(
                jnp.take(getattr(batch, field), indices, axis=0)
                for field in PPOBatch._fields
            )
        )

    def _full_batch_runner(params, opt_state, batch, update_key, entropy_coef):
        del update_key

        def _epoch_step(carry, _):
            epoch_params, epoch_opt_state = carry
            next_params, next_opt_state, metrics = _single_epoch(
                epoch_params,
                epoch_opt_state,
                batch,
                entropy_coef,
            )
            return (next_params, next_opt_state), metrics

        (next_params, next_opt_state), metrics = jax.lax.scan(
            _epoch_step,
            (params, opt_state),
            xs=None,
            length=epochs,
        )
        final_metrics = {name: values[-1] for name, values in metrics.items()}
        return next_params, next_opt_state, final_metrics

    def _build_minibatch_runner(batch_size: int, minibatches: int):
        if minibatches <= 1:
            return _full_batch_runner
        if minibatches > batch_size:
            raise ValueError(
                "PPO minibatch count must not exceed the compiled PPO batch size: "
                f"batch_size={batch_size}, ppo_minibatches={minibatches}."
            )
        if batch_size % minibatches != 0:
            raise ValueError(
                "PPO minibatch count must evenly divide the compiled PPO batch size: "
                f"batch_size={batch_size}, ppo_minibatches={minibatches}."
            )
        minibatch_count = int(minibatches)
        minibatch_size = batch_size // minibatch_count

        def _minibatch_runner(params, opt_state, batch, update_key, entropy_coef):
            def _epoch_step(carry, epoch_index):
                epoch_params, epoch_opt_state, epoch_key = carry
                epoch_key = jax.random.fold_in(epoch_key, epoch_index)
                permutation = jax.random.permutation(
                    epoch_key,
                    jnp.arange(batch_size, dtype=jnp.int32),
                )
                minibatch_indices = permutation.reshape(minibatch_count, minibatch_size)

                def _minibatch_step(mini_carry, indices):
                    mini_params, mini_opt_state = mini_carry
                    minibatch = _take_minibatch(batch, indices)
                    next_params, next_opt_state, metrics = _single_epoch(
                        mini_params,
                        mini_opt_state,
                        minibatch,
                        entropy_coef,
                    )
                    return (next_params, next_opt_state), metrics

                (next_params, next_opt_state), minibatch_metrics = jax.lax.scan(
                    _minibatch_step,
                    (epoch_params, epoch_opt_state),
                    minibatch_indices,
                )
                return (next_params, next_opt_state, epoch_key), _mean_metrics(minibatch_metrics)

            (next_params, next_opt_state, _), epoch_metrics = jax.lax.scan(
                _epoch_step,
                (params, opt_state, update_key),
                jnp.arange(epochs, dtype=jnp.int32),
            )
            final_metrics = {name: values[-1] for name, values in epoch_metrics.items()}
            return next_params, next_opt_state, final_metrics

        return _minibatch_runner

    compiled_batch_size: int | None = None
    compiled_runner = None

    def _runner(params, opt_state, batch, update_key, entropy_coef=None):
        nonlocal compiled_batch_size, compiled_runner
        batch_size = int(batch.flat_obs.shape[0])
        entropy_coef_t = (
            default_entropy_coef
            if entropy_coef is None
            else jnp.asarray(float(entropy_coef), dtype=jnp.float32)
        )
        if compiled_batch_size is None:
            compiled_batch_size = batch_size
            compiled_runner = jax.jit(
                _build_minibatch_runner(batch_size, configured_minibatches)
            )
        elif compiled_batch_size != batch_size:
            raise ValueError(
                "PPO update runner was called with a different batch size than it was compiled for: "
                f"expected {compiled_batch_size}, got {batch_size}."
            )
        return compiled_runner(params, opt_state, batch, update_key, entropy_coef_t)

    return _runner, transform


def build_jitted_selector_update_runner(
    jax,
    jnp,
    spec: ActorCriticSpec,
    trainer_config: TrainerConfig,
    *,
    selector_value_coef: float,
    selector_entropy_coef: float,
    selector_usage_reg_coef: float,
    selector_learning_rate: float | None = None,
):
    import optax

    clip_range = jnp.asarray(trainer_config.ppo_clip_range, dtype=jnp.float32)
    value_coef = jnp.asarray(selector_value_coef, dtype=jnp.float32)
    entropy_coef = jnp.asarray(selector_entropy_coef, dtype=jnp.float32)
    usage_reg_coef = jnp.asarray(selector_usage_reg_coef, dtype=jnp.float32)
    transform = build_adam_transform(
        optax,
        learning_rate=(
            float(trainer_config.learning_rate)
            if selector_learning_rate is None
            else float(selector_learning_rate)
        ),
    )

    def _masked_mean(values, mask):
        return jnp.sum(values * mask) / jnp.maximum(jnp.sum(mask), 1.0)

    def _loss_fn(params, batch: SelectorBatch, selector_eps):
        batch_size = int(batch.flat_obs.shape[0])
        neutral_context = {
            "intent_index": jnp.zeros((batch_size,), dtype=jnp.int32),
            "intent_gate": jnp.zeros((batch_size,), dtype=jnp.float32),
        }
        forward_out = actor_critic_forward(
            params,
            batch.flat_obs,
            spec,
            jnp,
            intent_context=neutral_context,
        )
        raw_probs = jax.nn.softmax(forward_out["selector_logits"], axis=-1)
        eps = jnp.clip(jnp.asarray(selector_eps, dtype=jnp.float32), 0.0, 1.0)
        uniform = jnp.full_like(raw_probs, 1.0 / float(max(1, int(spec.num_intents))))
        probs = ((1.0 - eps) * raw_probs) + (eps * uniform)
        log_probs = jnp.log(jnp.maximum(probs, 1.0e-8))
        safe_intents = jnp.clip(batch.chosen_intents, 0, int(spec.num_intents) - 1)
        new_log_probs = jnp.take_along_axis(
            log_probs,
            safe_intents[:, None],
            axis=-1,
        )[:, 0]
        mask = batch.active_mask.astype(jnp.float32)
        log_ratio = new_log_probs - batch.old_log_probs
        ratio = jnp.exp(log_ratio)
        clipped_ratio = jnp.clip(ratio, 1.0 - clip_range, 1.0 + clip_range)
        policy_loss = -_masked_mean(
            jnp.minimum(ratio * batch.advantages, clipped_ratio * batch.advantages),
            mask,
        )
        values = forward_out["selector_values"]
        value_loss = _masked_mean(jnp.square(values - batch.returns), mask)
        entropy = _masked_mean(-jnp.sum(probs * log_probs, axis=-1), mask)
        sample_count = jnp.maximum(jnp.sum(mask), 1.0)
        mean_raw_probs = jnp.sum(raw_probs * mask[:, None], axis=0) / sample_count
        uniform_prob = jnp.asarray(1.0 / float(max(1, int(spec.num_intents))), dtype=jnp.float32)
        usage_kl_uniform = jnp.sum(
            mean_raw_probs
            * (jnp.log(jnp.maximum(mean_raw_probs, 1.0e-8)) - jnp.log(uniform_prob))
        )
        approx_kl = _masked_mean((ratio - 1.0) - log_ratio, mask)
        clip_fraction = _masked_mean((jnp.abs(ratio - 1.0) > clip_range).astype(jnp.float32), mask)
        total_loss = (
            policy_loss
            + (value_coef * value_loss)
            - (entropy_coef * entropy)
            + (usage_reg_coef * usage_kl_uniform)
        )
        metrics = {
            "selector_train_loss": total_loss,
            "selector_train_policy_loss": policy_loss,
            "selector_train_value_loss": value_loss,
            "selector_train_entropy": entropy,
            "selector_train_usage_kl_uniform": usage_kl_uniform,
            "selector_train_approx_kl": approx_kl,
            "selector_train_clip_fraction": clip_fraction,
            "selector_train_sample_count": jnp.sum(mask),
            "selector_train_return_mean": _masked_mean(batch.returns, mask),
            "selector_train_advantage_mean": _masked_mean(batch.advantages, mask),
            "selector_train_value_mean": _masked_mean(values, mask),
        }
        for intent_idx in range(int(spec.num_intents)):
            metrics[f"selector_train_usage_by_intent/{intent_idx}"] = _masked_mean(
                (safe_intents == int(intent_idx)).astype(jnp.float32),
                mask,
            )
            metrics[f"selector_train_prob_by_intent/{intent_idx}"] = mean_raw_probs[int(intent_idx)]
        return total_loss, metrics

    @jax.jit
    def _runner(params, opt_state, batch: SelectorBatch, update_key, selector_eps):
        del update_key
        (_, loss_metrics), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
            params,
            batch,
            selector_eps,
        )
        sample_count = loss_metrics["selector_train_sample_count"]
        effective_grads = _mask_selector_update_grads(grads, sample_count, jax, jnp)
        grad_norm = global_norm(effective_grads, optax)
        candidate_params, new_opt_state = optimizer_update(
            params,
            effective_grads,
            opt_state,
            transform=transform,
            optax=optax,
        )
        new_params = _apply_selector_update_param_scope(
            params,
            candidate_params,
            sample_count,
            jax,
            jnp,
        )
        post_update_loss, metrics = _loss_fn(new_params, batch, selector_eps)
        metrics = {
            **metrics,
            "selector_train_loss": post_update_loss,
            "selector_train_grad_norm": jnp.where(
                sample_count > 0.0,
                grad_norm,
                jnp.asarray(0.0, dtype=jnp.float32),
            ),
        }
        return new_params, new_opt_state, metrics

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


def benchmark_update_runner(jax, runner, params, opt_state, batch, update_key, *, iterations: int, progress=None):
    timed_ns = 0
    final_params = params
    final_opt_state = opt_state
    final_metrics = None
    for idx in range(int(iterations)):
        iter_key = jax.random.fold_in(update_key, idx)
        start_ns = perf_counter_ns()
        final_params, final_opt_state, final_metrics = runner(
            final_params,
            final_opt_state,
            batch,
            iter_key,
        )
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
    shot_attempts=None,
) -> dict[str, float]:
    done_arr = np.asarray(dones, dtype=np.float32)
    terminal_steps_arr = np.asarray(terminal_episode_steps, dtype=np.int32)
    pass_attempts_arr = np.asarray(pass_attempts, dtype=np.float32)
    completed_passes_arr = np.asarray(completed_passes, dtype=np.float32)
    assists_arr = np.asarray(assists, dtype=np.float32)
    turnovers_arr = np.asarray(turnovers, dtype=np.float32)
    shot_attempts_arr = (
        np.asarray(shot_attempts, dtype=np.float32)
        if shot_attempts is not None
        else None
    )

    completed_episodes = int((terminal_steps_arr > 0).sum())
    completed_episode_steps = int(terminal_steps_arr.sum())
    denom = float(completed_episodes) if completed_episodes > 0 else 0.0

    def _mean_per_episode(total: float) -> float:
        return float(total / denom) if denom > 0.0 else 0.0

    total_pass_attempts = float(pass_attempts_arr.sum())
    total_completed_passes = float(completed_passes_arr.sum())
    total_assists = float(assists_arr.sum())
    total_turnovers = float(turnovers_arr.sum())
    total_shot_attempts = float(shot_attempts_arr.sum()) if shot_attempts_arr is not None else None

    metrics = {
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
    if total_shot_attempts is not None:
        metrics["total_shot_attempts"] = total_shot_attempts
        metrics["shots_per_completed_episode"] = _mean_per_episode(total_shot_attempts)
    return metrics


def summarize_shot_type_metrics(
    prefix: str,
    *,
    shot_attempts,
    shot_makes,
    shot_dunks,
    shot_twos,
    shot_threes,
    completed_episodes: int | float | None = None,
) -> dict[str, float]:
    attempts = float(np.asarray(shot_attempts, dtype=np.float32).sum())
    makes = float(np.asarray(shot_makes, dtype=np.float32).sum())
    dunks = float(np.asarray(shot_dunks, dtype=np.float32).sum())
    twos = float(np.asarray(shot_twos, dtype=np.float32).sum())
    threes = float(np.asarray(shot_threes, dtype=np.float32).sum())
    safe_attempts = attempts if attempts > 0.0 else 0.0

    def _rate(value: float) -> float:
        return float(value / safe_attempts) if safe_attempts > 0.0 else 0.0

    completed_denom = float(completed_episodes or 0.0)

    def _per_completed_episode(value: float) -> float:
        return float(value / completed_denom) if completed_denom > 0.0 else 0.0

    metrics = {
        f"{prefix}_shot_attempts": attempts,
        f"{prefix}_shot_makes": makes,
        f"{prefix}_shot_make_rate": _rate(makes),
        f"{prefix}_shot_dunk_attempts": dunks,
        f"{prefix}_shot_two_attempts": twos,
        f"{prefix}_shot_three_attempts": threes,
        f"{prefix}_shot_dunk_share": _rate(dunks),
        f"{prefix}_shot_two_share": _rate(twos),
        f"{prefix}_shot_three_share": _rate(threes),
    }
    if completed_episodes is not None:
        metrics.update(
            {
                f"{prefix}_shot_makes_per_completed_episode": _per_completed_episode(makes),
                f"{prefix}_shot_dunks_per_completed_episode": _per_completed_episode(dunks),
                f"{prefix}_shot_twos_per_completed_episode": _per_completed_episode(twos),
                f"{prefix}_shot_threes_per_completed_episode": _per_completed_episode(threes),
            }
        )
        if prefix != "all" and not prefix.endswith("_all"):
            metrics[f"{prefix}_shots_per_completed_episode"] = _per_completed_episode(attempts)
    return metrics


def summarize_turnover_diagnostics(
    *,
    terminal_episode_steps,
    learner_turnovers,
    opponent_turnovers,
    turnover_pass_out_of_bounds,
    turnover_intercepted,
    turnover_defender_pressure,
    turnover_move_out_of_bounds,
    turnover_shot_clock,
    turnover_offensive_three_seconds,
) -> dict[str, float]:
    terminal_steps_arr = np.asarray(terminal_episode_steps, dtype=np.int32)
    completed_episodes = int((terminal_steps_arr > 0).sum())
    denom = float(completed_episodes) if completed_episodes > 0 else 0.0

    def _total(values) -> float:
        return float(np.asarray(values, dtype=np.float32).sum())

    def _mean_per_episode(total: float) -> float:
        return float(total / denom) if denom > 0.0 else 0.0

    metrics: dict[str, float] = {}
    role_items = {
        "learner": _total(learner_turnovers),
        "opponent": _total(opponent_turnovers),
    }
    for name, total in role_items.items():
        metrics[f"total_{name}_turnovers"] = total
        metrics[f"mean_{name}_turnovers_per_completed_episode"] = _mean_per_episode(total)

    reason_items = {
        "pass_out_of_bounds": _total(turnover_pass_out_of_bounds),
        "intercepted": _total(turnover_intercepted),
        "defender_pressure": _total(turnover_defender_pressure),
        "move_out_of_bounds": _total(turnover_move_out_of_bounds),
        "shot_clock": _total(turnover_shot_clock),
        "offensive_three_seconds": _total(turnover_offensive_three_seconds),
    }
    for reason, total in reason_items.items():
        metrics[f"total_turnovers_reason_{reason}"] = total
        metrics[f"mean_turnovers_reason_{reason}_per_completed_episode"] = _mean_per_episode(total)
    return metrics


def summarize_ppo_eligible_episode_metrics(
    prefix: str,
    trajectory: TrajectoryBatch,
    training_mask,
    *,
    include_learner_shots: bool = True,
    include_opponent_shots: bool = True,
    role: str | None = None,
) -> dict[str, float]:
    """Summarize only the rollout steps that are eligible for PPO loss updates."""
    mask = np.asarray(training_mask, dtype=np.float32)
    terminal_steps = np.asarray(trajectory.terminal_episode_steps, dtype=np.int32)
    terminal_mask = ((terminal_steps > 0) & (mask > 0.5)).astype(np.float32)
    completed_episodes = int(terminal_mask.sum())
    completed_denom = float(completed_episodes) if completed_episodes > 0 else 0.0
    step_count = float(mask.sum())

    def _masked_total(values, event_mask) -> float:
        return float((np.asarray(values, dtype=np.float32) * event_mask).sum())

    def _per_completed_episode(total: float) -> float:
        return float(total / completed_denom) if completed_denom > 0.0 else 0.0

    def _per_step(total: float) -> float:
        return float(total / step_count) if step_count > 0.0 else 0.0

    reward_total = _masked_total(trajectory.rewards, mask)
    offense_points_total = _masked_total(trajectory.offense_score_delta, mask)
    defense_points_total = _masked_total(trajectory.defense_score_delta, mask)
    completed_episode_steps = _masked_total(terminal_steps, terminal_mask)
    metrics: dict[str, float] = {
        f"{prefix}_active_step_count": step_count,
        f"{prefix}_completed_episodes": float(completed_episodes),
        f"{prefix}_completed_episode_steps": completed_episode_steps,
        f"{prefix}_completed_episode_count": float(completed_episodes),
        f"{prefix}_completed_active_step_count": completed_episode_steps,
        f"{prefix}_mean_completed_episode_length": _per_completed_episode(completed_episode_steps),
        f"{prefix}_reward_total": reward_total,
        f"{prefix}_reward_per_step": _per_step(reward_total),
        f"{prefix}_reward_per_completed_episode": _per_completed_episode(reward_total),
        f"{prefix}_offense_points_total": offense_points_total,
        f"{prefix}_offense_points_per_step": _per_step(offense_points_total),
        f"{prefix}_offense_points_per_completed_episode": _per_completed_episode(
            offense_points_total
        ),
        f"{prefix}_defense_points_total": defense_points_total,
        f"{prefix}_defense_points_per_step": _per_step(defense_points_total),
        f"{prefix}_defense_points_per_completed_episode": _per_completed_episode(
            defense_points_total
        ),
    }
    if role == "offense":
        learner_points_total = offense_points_total
        opponent_points_total = defense_points_total
    elif role == "defense":
        learner_points_total = defense_points_total
        opponent_points_total = offense_points_total
    else:
        learner_points_total = None
        opponent_points_total = None
    if learner_points_total is not None and opponent_points_total is not None:
        metrics.update(
            {
                f"{prefix}_learner_points_total": learner_points_total,
                f"{prefix}_learner_points_per_step": _per_step(learner_points_total),
                f"{prefix}_learner_points_per_completed_episode": _per_completed_episode(
                    learner_points_total
                ),
                f"{prefix}_opponent_points_total": opponent_points_total,
                f"{prefix}_opponent_points_per_step": _per_step(opponent_points_total),
                f"{prefix}_opponent_points_per_completed_episode": _per_completed_episode(
                    opponent_points_total
                ),
            }
        )

    step_event_items = {
        "pass_attempts": trajectory.pass_attempts,
        "completed_passes": trajectory.completed_passes,
        "assists": trajectory.assists,
        "turnovers": trajectory.turnovers,
        "learner_turnovers": trajectory.learner_turnovers,
        "opponent_turnovers": trajectory.opponent_turnovers,
        "offensive_three_seconds": trajectory.offensive_three_seconds,
        "defensive_lane_violations": trajectory.defensive_lane_violations,
        "shot_attempts": trajectory.shot_attempts,
        "shot_makes": trajectory.shot_makes,
        "shot_dunks": trajectory.shot_dunks,
        "shot_twos": trajectory.shot_twos,
        "shot_threes": trajectory.shot_threes,
    }
    if include_learner_shots:
        step_event_items.update(
            {
                "learner_shot_attempts": trajectory.learner_shot_attempts,
                "learner_shot_makes": trajectory.learner_shot_makes,
                "learner_shot_dunks": trajectory.learner_shot_dunks,
                "learner_shot_twos": trajectory.learner_shot_twos,
                "learner_shot_threes": trajectory.learner_shot_threes,
            }
        )
    if include_opponent_shots:
        step_event_items.update(
            {
                "opponent_shot_attempts": trajectory.opponent_shot_attempts,
                "opponent_shot_makes": trajectory.opponent_shot_makes,
                "opponent_shot_dunks": trajectory.opponent_shot_dunks,
                "opponent_shot_twos": trajectory.opponent_shot_twos,
                "opponent_shot_threes": trajectory.opponent_shot_threes,
            }
        )
    shot_event_names = {
        "shot_attempts",
        "shot_makes",
        "shot_dunks",
        "shot_twos",
        "shot_threes",
        "learner_shot_attempts",
        "learner_shot_makes",
        "learner_shot_dunks",
        "learner_shot_twos",
        "learner_shot_threes",
        "opponent_shot_attempts",
        "opponent_shot_makes",
        "opponent_shot_dunks",
        "opponent_shot_twos",
        "opponent_shot_threes",
    }
    for name, values in step_event_items.items():
        total = _masked_total(values, mask)
        metrics[f"{prefix}_{name}_total"] = total
        metrics[f"{prefix}_{name}_per_step"] = _per_step(total)
        if name.endswith("shot_attempts"):
            alias = name.replace("shot_attempts", "shots")
            metrics[f"{prefix}_{alias}_per_completed_episode"] = _per_completed_episode(total)
        else:
            metrics[f"{prefix}_{name}_per_completed_episode"] = _per_completed_episode(total)

    terminal_reason_items = {
        "shot": trajectory.shot_attempts,
        "turnover": trajectory.turnovers,
        "learner_turnover": trajectory.learner_turnovers,
        "opponent_turnover": trajectory.opponent_turnovers,
        "turnover_pass_out_of_bounds": trajectory.turnover_pass_out_of_bounds,
        "turnover_intercepted": trajectory.turnover_intercepted,
        "turnover_defender_pressure": trajectory.turnover_defender_pressure,
        "turnover_move_out_of_bounds": trajectory.turnover_move_out_of_bounds,
        "turnover_shot_clock": trajectory.turnover_shot_clock,
        "turnover_offensive_three_seconds": trajectory.turnover_offensive_three_seconds,
        "offensive_three_seconds": trajectory.offensive_three_seconds,
        "defensive_lane_violation": trajectory.defensive_lane_violations,
        "shot_make": trajectory.shot_makes,
        "shot_dunk": trajectory.shot_dunks,
        "shot_two": trajectory.shot_twos,
        "shot_three": trajectory.shot_threes,
    }
    for name, values in terminal_reason_items.items():
        total = _masked_total(values, terminal_mask)
        metrics[f"{prefix}_terminal_{name}_episodes"] = total
        metrics[f"{prefix}_terminal_{name}_share"] = _per_completed_episode(total)

    return metrics


def summarize_ppo_eligible_reward_component_metrics(
    prefix: str,
    trajectory: TrajectoryBatch,
    training_mask,
    *,
    task_rewards,
    phi_rewards,
    intent_bonus_rewards,
) -> dict[str, float]:
    """Summarize PPO-eligible reward components on the same denominator as total reward."""
    mask = np.asarray(training_mask, dtype=np.float32)
    terminal_steps = np.asarray(trajectory.terminal_episode_steps, dtype=np.int32)
    terminal_mask = ((terminal_steps > 0) & (mask > 0.5)).astype(np.float32)
    completed_episodes = int(terminal_mask.sum())
    completed_denom = float(completed_episodes) if completed_episodes > 0 else 0.0
    step_count = float(mask.sum())

    def _masked_total(values) -> float:
        return float((np.asarray(values, dtype=np.float32) * mask).sum())

    def _per_completed_episode(total: float) -> float:
        return float(total / completed_denom) if completed_denom > 0.0 else 0.0

    def _per_step(total: float) -> float:
        return float(total / step_count) if step_count > 0.0 else 0.0

    component_items = {
        "task_reward": task_rewards,
        "phi_reward": phi_rewards,
        "intent_bonus": intent_bonus_rewards,
    }
    metrics: dict[str, float] = {}
    for name, values in component_items.items():
        total = _masked_total(values)
        metrics[f"{prefix}_{name}_total"] = total
        metrics[f"{prefix}_{name}_per_step"] = _per_step(total)
        metrics[f"{prefix}_{name}_per_completed_episode"] = _per_completed_episode(total)

    composed = np.asarray(trajectory.rewards, dtype=np.float32)
    residual = composed - (
        np.asarray(task_rewards, dtype=np.float32)
        + np.asarray(phi_rewards, dtype=np.float32)
        + np.asarray(intent_bonus_rewards, dtype=np.float32)
    )
    residual_total = _masked_total(residual)
    composed_total = _masked_total(composed)
    intent_total = metrics[f"{prefix}_intent_bonus_total"]
    phi_total = metrics[f"{prefix}_phi_reward_total"]
    metrics[f"{prefix}_reward_component_residual_total"] = residual_total
    metrics[f"{prefix}_reward_component_residual_per_step"] = _per_step(residual_total)
    metrics[f"{prefix}_reward_component_residual_per_completed_episode"] = (
        _per_completed_episode(residual_total)
    )
    metrics[f"{prefix}_intent_bonus_abs_share_of_reward"] = (
        float(abs(intent_total) / max(abs(composed_total), 1.0e-8))
    )
    metrics[f"{prefix}_phi_reward_abs_share_of_reward"] = (
        float(abs(phi_total) / max(abs(composed_total), 1.0e-8))
    )
    return metrics


def summarize_reward_by_intent_metrics(
    prefix: str,
    trajectory: TrajectoryBatch,
    *,
    num_intents: int,
    training_mask=None,
) -> dict[str, float]:
    """Attribute completed episode reward to the intent active at episode start."""
    rewards = np.asarray(trajectory.rewards, dtype=np.float32)
    intent_index = np.asarray(trajectory.policy_intent_index, dtype=np.int32)
    episode_start = np.asarray(trajectory.episode_start, dtype=np.float32) > 0.5
    terminal_steps = np.asarray(trajectory.terminal_episode_steps, dtype=np.int32)
    active_mask = np.asarray(trajectory.active_mask, dtype=np.float32) > 0.5
    if training_mask is not None:
        active_mask = active_mask & (np.asarray(training_mask, dtype=np.float32) > 0.5)

    if rewards.ndim != 2:
        raise ValueError("trajectory rewards must have shape [time, batch].")

    intent_count = max(1, int(num_intents))
    reward_totals = np.zeros((intent_count,), dtype=np.float64)
    episode_counts = np.zeros((intent_count,), dtype=np.float64)
    time_steps, batch_size = rewards.shape
    for env_idx in range(batch_size):
        current_intent = int(np.clip(intent_index[0, env_idx], 0, intent_count - 1))
        episode_reward = 0.0
        for step_idx in range(time_steps):
            if not bool(active_mask[step_idx, env_idx]):
                continue
            if bool(episode_start[step_idx, env_idx]):
                current_intent = int(np.clip(intent_index[step_idx, env_idx], 0, intent_count - 1))
                episode_reward = 0.0
            episode_reward += float(rewards[step_idx, env_idx])
            if int(terminal_steps[step_idx, env_idx]) > 0:
                reward_totals[current_intent] += episode_reward
                episode_counts[current_intent] += 1.0
                episode_reward = 0.0

    metrics: dict[str, float] = {}
    for intent_idx in range(intent_count):
        count = float(episode_counts[intent_idx])
        total = float(reward_totals[intent_idx])
        metrics[f"{prefix}_completed_episodes_by_intent/{intent_idx}"] = count
        metrics[f"{prefix}_reward_per_completed_episode_by_intent/{intent_idx}"] = (
            float(total / count) if count > 0.0 else 0.0
        )
    return metrics


def summarize_lane_violation_metrics(
    *,
    terminal_episode_steps,
    offensive_three_seconds,
    defensive_lane_violations,
) -> dict[str, float]:
    terminal_steps_arr = np.asarray(terminal_episode_steps, dtype=np.int32)
    completed_episodes = int((terminal_steps_arr > 0).sum())
    denom = float(completed_episodes) if completed_episodes > 0 else 0.0
    offensive_total = float(np.asarray(offensive_three_seconds, dtype=np.float32).sum())
    defensive_total = float(np.asarray(defensive_lane_violations, dtype=np.float32).sum())
    step_count = float(np.asarray(offensive_three_seconds, dtype=np.float32).size)

    def _per_episode(total: float) -> float:
        return float(total / denom) if denom > 0.0 else 0.0

    def _per_step(total: float) -> float:
        return float(total / step_count) if step_count > 0.0 else 0.0

    return {
        "total_offensive_three_seconds": offensive_total,
        "total_3_second_violations": offensive_total,
        "mean_offensive_three_seconds_per_completed_episode": _per_episode(offensive_total),
        "mean_3_second_violations_per_completed_episode": _per_episode(offensive_total),
        "offensive_three_seconds_rate_per_step": _per_step(offensive_total),
        "three_second_violation_rate_per_step": _per_step(offensive_total),
        "total_defensive_lane_violations": defensive_total,
        "mean_defensive_lane_violations_per_completed_episode": _per_episode(defensive_total),
        "defensive_lane_violation_rate_per_step": _per_step(defensive_total),
    }


def summarize_intent_metrics(
    prefix: str,
    *,
    intent_index,
    intent_active,
    intent_age,
    intent_commitment_remaining,
    intent_visible_to_defense=None,
) -> dict[str, float]:
    index_arr = np.asarray(intent_index, dtype=np.int32)
    active_arr = np.asarray(intent_active, dtype=np.float32)
    age_arr = np.asarray(intent_age, dtype=np.float32)
    remaining_arr = np.asarray(intent_commitment_remaining, dtype=np.float32)
    active_count = float(active_arr.sum())
    total_count = float(active_arr.size)
    active_bool = active_arr > 0.5
    metrics = {
        f"{prefix}_intent_active_rate": float(active_count / total_count) if total_count > 0.0 else 0.0,
        f"{prefix}_intent_active_count": active_count,
        f"{prefix}_intent_mean_age": float(age_arr[active_bool].mean()) if np.any(active_bool) else 0.0,
        f"{prefix}_intent_mean_commitment_remaining": (
            float(remaining_arr[active_bool].mean()) if np.any(active_bool) else 0.0
        ),
        f"{prefix}_intent_mean_index": (
            float(index_arr[active_bool].mean()) if np.any(active_bool) else 0.0
        ),
    }
    if intent_visible_to_defense is not None:
        visible_arr = np.asarray(intent_visible_to_defense, dtype=np.float32)
        metrics[f"{prefix}_intent_visible_to_defense_rate"] = (
            float(visible_arr.mean()) if visible_arr.size else 0.0
        )
    if np.any(active_bool):
        active_indices, counts = np.unique(index_arr[active_bool], return_counts=True)
        for raw_index, raw_count in zip(active_indices, counts, strict=True):
            idx = int(raw_index)
            count = float(raw_count)
            metrics[f"{prefix}_intent_usage_count/{idx}"] = count
            metrics[f"{prefix}_intent_usage_share/{idx}"] = float(count / active_count) if active_count > 0.0 else 0.0
    return metrics


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
    policy_update_epochs: int = 1,
    ppo_minibatches: int = 1,
) -> dict[str, Any]:
    total_states = int(batch_size) * int(horizon)
    ppo_batch_size = int(ppo_batch.flat_obs.shape[0])
    rollout_sec = max(rollout_elapsed_ns / 1e9, 1e-12)
    update_sec = max(update_elapsed_ns / 1e9, 1e-12)
    end_to_end_sec = max((rollout_elapsed_ns + update_elapsed_ns) / 1e9, 1e-12)
    rollout_time_fraction = float(rollout_sec / end_to_end_sec)
    update_time_fraction = float(update_sec / end_to_end_sec)
    optimizer_sample_count = int(ppo_batch_size * max(1, int(policy_update_epochs)))
    minibatch_count = max(1, int(ppo_minibatches))
    minibatch_size = int(ppo_batch_size // minibatch_count) if ppo_batch_size % minibatch_count == 0 else 0
    rollout_active = np.asarray(rollout_out.trajectory.active_mask, dtype=np.float32)
    ppo_active = np.asarray(ppo_batch.active_mask, dtype=np.float32)
    ppo_loss_weights = np.asarray(ppo_batch.loss_weights, dtype=np.float32)
    ppo_loss_denominator = np.asarray(ppo_batch.loss_denominator, dtype=np.float32)
    active_count = float(rollout_active.sum())
    active_ppo_count = float(ppo_active.sum())
    ppo_loss_weight_sum = float(ppo_loss_weights.sum())
    ppo_loss_denominator_value = float(ppo_loss_denominator.max()) if ppo_loss_denominator.size else 0.0

    def _active_mean(values, mask) -> float:
        denom = float(mask.sum())
        if denom <= 0.0:
            return 0.0
        return float((np.asarray(values, dtype=np.float32) * mask).sum() / denom)

    reward_mean = _active_mean(rollout_out.trajectory.rewards, rollout_active)
    phi_r_shape_mean = _active_mean(rollout_out.trajectory.phi_r_shape, rollout_active)
    phi_r_shape_abs_mean = _active_mean(
        np.abs(np.asarray(rollout_out.trajectory.phi_r_shape, dtype=np.float32)),
        rollout_active,
    )
    phi_prev_mean = _active_mean(rollout_out.trajectory.phi_prev, rollout_active)
    phi_next_mean = _active_mean(rollout_out.trajectory.phi_next, rollout_active)
    phi_beta_mean = _active_mean(rollout_out.trajectory.phi_beta, rollout_active)
    done_rate = _active_mean(rollout_out.trajectory.dones, rollout_active)
    opponent_deterministic_episode_rate = _active_mean(
        rollout_out.trajectory.opponent_deterministic_episode,
        rollout_active,
    )
    advantage_std = (
        float(np.asarray(ppo_batch.advantages)[ppo_active.astype(bool)].std())
        if active_ppo_count > 0.0
        else 0.0
    )
    return_mean = _active_mean(ppo_batch.returns, ppo_active)
    value_mean = _active_mean(rollout_out.trajectory.values, rollout_active)
    episode_metrics = summarize_episode_events(
        rollout_out.trajectory.dones,
        rollout_out.trajectory.terminal_episode_steps,
        rollout_out.trajectory.pass_attempts,
        rollout_out.trajectory.completed_passes,
        rollout_out.trajectory.assists,
        rollout_out.trajectory.turnovers,
        rollout_out.trajectory.shot_attempts,
    )
    summary = {
        "update_index": int(update_index),
        "steps_per_update": int(total_states),
        "ppo_batch_size": int(ppo_batch_size),
        "ppo_active_sample_count": int(active_ppo_count),
        "ppo_active_sample_fraction": float(active_ppo_count / max(1, int(ppo_batch_size))),
        "ppo_loss_weight_sum": float(ppo_loss_weight_sum),
        "ppo_loss_denominator": float(ppo_loss_denominator_value),
        "rollout_active_step_count": int(active_count),
        "rollout_active_step_fraction": float(active_count / max(1, int(total_states))),
        "ppo_update_epochs": int(max(1, int(policy_update_epochs))),
        "ppo_update_minibatches": int(minibatch_count),
        "ppo_update_minibatch_size": int(minibatch_size),
        "ppo_update_optimizer_samples": int(optimizer_sample_count),
        "end_to_end_latency_ms": float((rollout_elapsed_ns + update_elapsed_ns) / 1e6),
        "rollout_elapsed_sec": float(rollout_sec),
        "ppo_update_elapsed_sec": float(update_sec),
        "end_to_end_elapsed_sec": float(end_to_end_sec),
        "rollout_time_fraction": rollout_time_fraction,
        "ppo_update_time_fraction": update_time_fraction,
        "rollout_time_pct": float(100.0 * rollout_time_fraction),
        "ppo_update_time_pct": float(100.0 * update_time_fraction),
        "rollout_states_per_sec": float(total_states / rollout_sec),
        "end_to_end_steps_per_sec": float(total_states / end_to_end_sec),
        "active_rollout_steps_per_sec": float(active_count / rollout_sec),
        "active_end_to_end_steps_per_sec": float(active_count / end_to_end_sec),
        "rollout_latency_ms": float(rollout_elapsed_ns / 1e6),
        "update_steps_per_sec": float(1.0 / update_sec),
        "update_latency_ms": float(update_elapsed_ns / 1e6),
        "ppo_update_rollout_samples_per_sec": float(ppo_batch_size / update_sec),
        "ppo_update_optimizer_samples_per_sec": float(optimizer_sample_count / update_sec),
        "mean_reward": reward_mean,
        "phi_r_shape_mean": phi_r_shape_mean,
        "phi_r_shape_abs_mean": phi_r_shape_abs_mean,
        "phi_prev_mean": phi_prev_mean,
        "phi_next_mean": phi_next_mean,
        "phi_beta_mean": phi_beta_mean,
        "done_rate": done_rate,
        "opponent_deterministic_episode_rate": opponent_deterministic_episode_rate,
        "mean_return": return_mean,
        "mean_value": value_mean,
        "advantage_std": advantage_std,
    }
    summary.update(episode_metrics)
    summary["completed_episode_count"] = int(episode_metrics["completed_episodes"])
    summary["completed_active_step_count"] = int(episode_metrics["completed_episode_steps"])
    summary.update(
        summarize_lane_violation_metrics(
            terminal_episode_steps=rollout_out.trajectory.terminal_episode_steps,
            offensive_three_seconds=rollout_out.trajectory.offensive_three_seconds,
            defensive_lane_violations=rollout_out.trajectory.defensive_lane_violations,
        )
    )
    rebound_attempts = float(np.asarray(rollout_out.trajectory.rebound_attempts, dtype=np.float32).sum())
    offensive_rebounds = float(np.asarray(rollout_out.trajectory.offensive_rebounds, dtype=np.float32).sum())
    defensive_rebounds = float(np.asarray(rollout_out.trajectory.defensive_rebounds, dtype=np.float32).sum())
    rebound_global_contests = float(np.asarray(rollout_out.trajectory.rebound_global_contests, dtype=np.float32).sum())
    shot_clock_resets = float(np.asarray(rollout_out.trajectory.shot_clock_reset_14, dtype=np.float32).sum())
    rebound_reward_advances = float(
        np.asarray(rollout_out.trajectory.rebound_reward_advances, dtype=np.float32).sum()
    )
    rebound_reward_settlements = float(
        np.asarray(rollout_out.trajectory.rebound_reward_settlements, dtype=np.float32).sum()
    )
    rebound_reward_advance_count = int(
        (np.asarray(rollout_out.trajectory.rebound_reward_advances, dtype=np.float32) > 0.0).sum()
    )
    summary.update(
        {
            "rebound_attempts": int(rebound_attempts),
            "offensive_rebounds": int(offensive_rebounds),
            "defensive_rebounds": int(defensive_rebounds),
            "offensive_rebound_rate": float(offensive_rebounds / max(1.0, rebound_attempts)),
            "defensive_rebound_rate": float(defensive_rebounds / max(1.0, rebound_attempts)),
            "rebound_global_contest_count": int(rebound_global_contests),
            "rebound_global_contest_rate": float(rebound_global_contests / max(1.0, rebound_attempts)),
            "shot_clock_reset_14_count": int(shot_clock_resets),
            "rebound_reward_advance_count": rebound_reward_advance_count,
            "rebound_reward_advance_total": rebound_reward_advances,
            "rebound_reward_settlement_total": rebound_reward_settlements,
            "rebound_reward_net_total": rebound_reward_advances + rebound_reward_settlements,
        }
    )
    summary.update(
        summarize_turnover_diagnostics(
            terminal_episode_steps=rollout_out.trajectory.terminal_episode_steps,
            learner_turnovers=rollout_out.trajectory.learner_turnovers,
            opponent_turnovers=rollout_out.trajectory.opponent_turnovers,
            turnover_pass_out_of_bounds=rollout_out.trajectory.turnover_pass_out_of_bounds,
            turnover_intercepted=rollout_out.trajectory.turnover_intercepted,
            turnover_defender_pressure=rollout_out.trajectory.turnover_defender_pressure,
            turnover_move_out_of_bounds=rollout_out.trajectory.turnover_move_out_of_bounds,
            turnover_shot_clock=rollout_out.trajectory.turnover_shot_clock,
            turnover_offensive_three_seconds=rollout_out.trajectory.turnover_offensive_three_seconds,
        )
    )
    summary.update(
        summarize_shot_type_metrics(
            "all",
            shot_attempts=rollout_out.trajectory.shot_attempts,
            shot_makes=rollout_out.trajectory.shot_makes,
            shot_dunks=rollout_out.trajectory.shot_dunks,
            shot_twos=rollout_out.trajectory.shot_twos,
            shot_threes=rollout_out.trajectory.shot_threes,
            completed_episodes=episode_metrics["completed_episodes"],
        )
    )
    summary.update(
        summarize_shot_type_metrics(
            "learner",
            shot_attempts=rollout_out.trajectory.learner_shot_attempts,
            shot_makes=rollout_out.trajectory.learner_shot_makes,
            shot_dunks=rollout_out.trajectory.learner_shot_dunks,
            shot_twos=rollout_out.trajectory.learner_shot_twos,
            shot_threes=rollout_out.trajectory.learner_shot_threes,
            completed_episodes=episode_metrics["completed_episodes"],
        )
    )
    summary.update(
        summarize_shot_type_metrics(
            "opponent",
            shot_attempts=rollout_out.trajectory.opponent_shot_attempts,
            shot_makes=rollout_out.trajectory.opponent_shot_makes,
            shot_dunks=rollout_out.trajectory.opponent_shot_dunks,
            shot_twos=rollout_out.trajectory.opponent_shot_twos,
            shot_threes=rollout_out.trajectory.opponent_shot_threes,
            completed_episodes=episode_metrics["completed_episodes"],
        )
    )
    summary.update(
        summarize_intent_metrics(
            "offense",
            intent_index=rollout_out.trajectory.intent_index,
            intent_active=rollout_out.trajectory.intent_active,
            intent_age=rollout_out.trajectory.intent_age,
            intent_commitment_remaining=rollout_out.trajectory.intent_commitment_remaining,
            intent_visible_to_defense=rollout_out.trajectory.intent_visible_to_defense,
        )
    )
    summary.update(
        summarize_intent_metrics(
            "defense",
            intent_index=rollout_out.trajectory.defense_intent_index,
            intent_active=rollout_out.trajectory.defense_intent_active,
            intent_age=rollout_out.trajectory.defense_intent_age,
            intent_commitment_remaining=rollout_out.trajectory.defense_intent_commitment_remaining,
        )
    )
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
    shot_attempts = np.asarray(trace.shot_attempts)
    shot_makes = np.asarray(trace.shot_makes)
    shot_dunks = np.asarray(trace.shot_dunks)
    shot_twos = np.asarray(trace.shot_twos)
    shot_threes = np.asarray(trace.shot_threes)
    rebound_attempts = np.asarray(trace.rebound_attempts)
    offensive_rebounds = np.asarray(trace.offensive_rebounds)
    defensive_rebounds = np.asarray(trace.defensive_rebounds)
    rebound_target_cells = np.asarray(trace.rebound_target_cells)
    rebound_winners = np.asarray(trace.rebound_winners)
    rebound_global_contests = np.asarray(trace.rebound_global_contests)
    shot_clock_reset_14 = np.asarray(trace.shot_clock_reset_14)
    intent_index = np.asarray(trace.intent_index)
    intent_active = np.asarray(trace.intent_active)
    intent_age = np.asarray(trace.intent_age)
    intent_commitment_remaining = np.asarray(trace.intent_commitment_remaining)
    intent_visible_to_defense = np.asarray(trace.intent_visible_to_defense)
    defense_intent_index = np.asarray(trace.defense_intent_index)
    defense_intent_active = np.asarray(trace.defense_intent_active)
    defense_intent_age = np.asarray(trace.defense_intent_age)
    defense_intent_commitment_remaining = np.asarray(trace.defense_intent_commitment_remaining)
    offensive_three_seconds = np.asarray(trace.offensive_three_seconds)
    defensive_lane_violations = np.asarray(trace.defensive_lane_violations)
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
        "shot_attempts": shot_attempts[:, env_index].astype(np.int8),
        "shot_makes": shot_makes[:, env_index].astype(np.int8),
        "shot_dunks": shot_dunks[:, env_index].astype(np.int8),
        "shot_twos": shot_twos[:, env_index].astype(np.int8),
        "shot_threes": shot_threes[:, env_index].astype(np.int8),
        "rebound_attempts": rebound_attempts[:, env_index].astype(np.int8),
        "offensive_rebounds": offensive_rebounds[:, env_index].astype(np.int8),
        "defensive_rebounds": defensive_rebounds[:, env_index].astype(np.int8),
        "rebound_target_cells": rebound_target_cells[:, env_index].astype(np.int32),
        "rebound_winners": rebound_winners[:, env_index].astype(np.int32),
        "rebound_global_contests": rebound_global_contests[:, env_index].astype(np.int8),
        "shot_clock_reset_14": shot_clock_reset_14[:, env_index].astype(np.int8),
        "intent_index": intent_index[:, env_index].astype(np.int32),
        "intent_active": intent_active[:, env_index].astype(np.int8),
        "intent_age": intent_age[:, env_index].astype(np.int32),
        "intent_commitment_remaining": intent_commitment_remaining[:, env_index].astype(np.int32),
        "intent_visible_to_defense": intent_visible_to_defense[:, env_index].astype(np.int8),
        "defense_intent_index": defense_intent_index[:, env_index].astype(np.int32),
        "defense_intent_active": defense_intent_active[:, env_index].astype(np.int8),
        "defense_intent_age": defense_intent_age[:, env_index].astype(np.int32),
        "defense_intent_commitment_remaining": defense_intent_commitment_remaining[:, env_index].astype(np.int32),
        "offensive_three_seconds": offensive_three_seconds[:, env_index].astype(np.int8),
        "defensive_lane_violations": defensive_lane_violations[:, env_index].astype(np.int8),
        "terminal_episode_steps": terminal_episode_steps[:, env_index].astype(np.int32),
        "offense_score": offense_score[:, env_index].astype(np.float32),
        "defense_score": defense_score[:, env_index].astype(np.float32),
        "final_offense_score": float(final_offense[env_index]),
        "final_defense_score": float(final_defense[env_index]),
    }
