from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

import basketworld
from basketworld.envs.basketworld_env_v2 import ActionType, Team
from basketworld_jax.checkpoints import load_checkpoint
from basketworld_jax.env.minimal import (
    MOVE_ACTION_END,
    MOVE_ACTION_START,
    PASS_ACTION_END,
    PASS_ACTION_START,
    SHOT_TYPE_DUNK,
    SHOT_TYPE_THREE,
    TURNOVER_REASON_DEFENDER_PRESSURE,
    TURNOVER_REASON_INTERCEPTED,
    TURNOVER_REASON_MOVE_OUT_OF_BOUNDS,
    TURNOVER_REASON_OFFENSIVE_THREE_SECONDS,
    TURNOVER_REASON_PASS_OUT_OF_BOUNDS,
    TURNOVER_REASON_SHOT_CLOCK,
    assemble_full_actions_jax,
    build_action_masks_batch,
    build_kernel_static_from_env,
    build_policy_intent_context_batch_with_role_flag,
    build_policy_observation_batch_with_role_flag,
    reset_batch_minimal,
    set_offense_intent_state_batch,
    step_batch_minimal,
)
from basketworld_jax.inference import is_checkpoint_path
from basketworld_jax.models import ActorCriticSpec, actor_critic_forward, apply_action_mask
from basketworld_jax.train.cli import ensure_jax_available


def can_run_native_jax_evaluation(
    *,
    unified_policy_path: str,
    opponent_policy_path: str | None,
    custom_setup: dict | None,
    randomize_offense_permutation: bool,
) -> bool:
    if custom_setup:
        return False
    if bool(randomize_offense_permutation):
        return False
    if not is_checkpoint_path(unified_policy_path):
        return False
    return opponent_policy_path is None or is_checkpoint_path(opponent_policy_path)


def _load_checkpoint_params(path: str | Path, jax) -> tuple[dict[str, Any], Any, ActorCriticSpec]:
    payload = load_checkpoint(path)
    params = jax.device_put(payload["params"])
    spec = ActorCriticSpec(**dict(payload["policy_spec"]))
    return payload, params, spec


_SELECTOR_MODE_LEARNED_SAMPLE = 0
_SELECTOR_MODE_BEST_INTENT = 1
_SELECTOR_MODE_UNIFORM_RANDOM = 2
_SELECTOR_METADATA_PRIORITY_KEYS = {
    "intent_selector_enabled",
    "intent_selector_mode",
    "intent_selector_multiselect_enabled",
    "intent_selector_min_play_steps",
    "intent_selector_hidden_dim",
    "intent_selector_value_coef",
}


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _coerce_int(value: Any, default: int = 0) -> int:
    if value is None:
        return int(default)
    try:
        return int(value)
    except Exception:
        return int(default)


def _coerce_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _param(training_params: dict[str, Any], key: str, default: Any = None) -> Any:
    if key in training_params:
        return training_params[key]
    jax_key = f"jax/{key}"
    if jax_key in training_params:
        return training_params[jax_key]
    jax_env_key = f"jax/env/{key}"
    if jax_env_key in training_params:
        return training_params[jax_env_key]
    return default


_JAX_STATIC_ONLY_ENV_KEYS = {
    "enable_rebounds",
    "rebound_table_model_dir",
    "rebound_target_temperature",
    "rebound_target_uniform_mix",
    "rebound_winner_distance_weight",
    "rebound_winner_temperature",
    "offensive_rebound_shot_clock_reset",
    "rebound_terminal_reward_mode",
}

_JAX_STATIC_ONLY_ENV_DEFAULTS = {
    "enable_rebounds": False,
    "rebound_table_model_dir": "",
    "rebound_target_temperature": 1.0,
    "rebound_target_uniform_mix": 0.0,
    "rebound_winner_distance_weight": 1.0,
    "rebound_winner_temperature": 1.0,
    "offensive_rebound_shot_clock_reset": 14,
    "rebound_terminal_reward_mode": "actual_points",
}

_JAX_STATIC_ONLY_ENV_CASTS = {
    "enable_rebounds": "bool",
    "rebound_table_model_dir": "str",
    "rebound_target_temperature": "float",
    "rebound_target_uniform_mix": "float",
    "rebound_winner_distance_weight": "float",
    "rebound_winner_temperature": "float",
    "offensive_rebound_shot_clock_reset": "int",
    "rebound_terminal_reward_mode": "str",
}


def _coerce_runtime_static_value(key: str, value: Any) -> Any:
    kind = _JAX_STATIC_ONLY_ENV_CASTS.get(key)
    if kind == "bool":
        return _coerce_bool(value)
    if kind == "float":
        return _coerce_float(value)
    if kind == "int":
        return _coerce_int(value)
    if kind == "str":
        return str(value)
    return value


def _jax_static_env_params_from_payload(*payloads: dict[str, Any]) -> dict[str, Any]:
    """Extract JAX kernel-only env attrs from checkpoint metadata."""
    out: dict[str, Any] = {}
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        sources = (
            payload.get("frozen_config"),
            payload.get("env_config"),
            payload.get("trainer_config"),
        )
        if any(isinstance(source, dict) for source in sources[:2]):
            for key, value in _JAX_STATIC_ONLY_ENV_DEFAULTS.items():
                out.setdefault(key, value)
        for source in sources:
            if not isinstance(source, dict):
                continue
            for key in _JAX_STATIC_ONLY_ENV_KEYS:
                if key in source and source[key] not in (None, ""):
                    out[key] = _coerce_runtime_static_value(key, source[key])
    return out


def _split_native_env_params(optional_params: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    params = dict(optional_params or {})
    env_kwargs = {
        key: value for key, value in params.items() if key not in _JAX_STATIC_ONLY_ENV_KEYS
    }
    static_overrides = {
        key: _coerce_runtime_static_value(key, value)
        for key, value in params.items()
        if key in _JAX_STATIC_ONLY_ENV_KEYS
    }
    return env_kwargs, static_overrides


def _native_eval_horizon(env, training_params: dict[str, Any] | None, payload: dict[str, Any]) -> int:
    shot_clock_steps = int(getattr(env, "shot_clock_steps", 24))
    params = dict(training_params or {})
    payload_config = payload.get("trainer_config") if isinstance(payload, dict) else None
    if isinstance(payload_config, dict):
        payload_params = dict(payload_config)
        payload_params.update(params)
        params = payload_params
    configured = _coerce_int(_param(params, "eval_horizon", 0), 0)
    horizon = max(shot_clock_steps + 2, configured)
    if bool(getattr(env, "enable_rebounds", False)):
        reset_steps = max(1, int(getattr(env, "offensive_rebound_shot_clock_reset", 14)))
        horizon = max(horizon, shot_clock_steps + 3 * reset_steps + 2)
    return int(horizon)


def _selector_training_params_from_checkpoint(
    payload: dict[str, Any],
    training_params: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge session params with checkpoint metadata, preferring checkpoint selector flags.

    MLflow/session extraction can populate default False values when a run is
    incomplete or when a checkpoint is loaded outside its original run. Static
    selector flags are architectural/runtime facts of the checkpoint, so native
    eval should resolve them from checkpoint metadata before falling back.
    """
    params: dict[str, Any] = {}
    trainer_config = payload.get("trainer_config")
    if isinstance(trainer_config, dict):
        params.update(trainer_config)
    if isinstance(training_params, dict):
        params.update(training_params)
    if isinstance(trainer_config, dict):
        for key in _SELECTOR_METADATA_PRIORITY_KEYS:
            if key in trainer_config:
                params[key] = trainer_config[key]
    policy_spec = payload.get("policy_spec")
    if isinstance(policy_spec, dict) and "intent_selector_enabled" in policy_spec:
        policy_selector_enabled = _coerce_bool(
            policy_spec.get("intent_selector_enabled"),
            default=False,
        )
        if policy_selector_enabled:
            params["intent_selector_enabled"] = True
            selector_mode = str(params.get("intent_selector_mode") or "").strip().lower()
            if not selector_mode or selector_mode == "callback":
                params["intent_selector_mode"] = "integrated"
            else:
                params.setdefault("intent_selector_mode", "integrated")
        else:
            params.setdefault("intent_selector_enabled", False)
    return params


def _normalize_selector_mode_code(mode: str | None) -> int:
    value = str(mode or "learned_sample").strip().lower()
    if value in {"best_intent", "best", "argmax", "greedy"}:
        return _SELECTOR_MODE_BEST_INTENT
    if value in {"uniform_random", "uniform", "random"}:
        return _SELECTOR_MODE_UNIFORM_RANDOM
    return _SELECTOR_MODE_LEARNED_SAMPLE


def _selector_mode_label(mode_code: int) -> str:
    if int(mode_code) == _SELECTOR_MODE_BEST_INTENT:
        return "best_intent"
    if int(mode_code) == _SELECTOR_MODE_UNIFORM_RANDOM:
        return "uniform_random"
    return "learned_sample"


def _linear_eval_schedule(
    training_params: dict[str, Any],
    prefix: str,
    *,
    update_index: int,
    default_start: float,
    default_end: float,
) -> float:
    start = _coerce_float(_param(training_params, f"{prefix}_start", default_start), default_start)
    end = _coerce_float(_param(training_params, f"{prefix}_end", default_end), default_end)
    warmup = _coerce_int(
        _param(
            training_params,
            f"{prefix}_warmup_steps",
            _param(training_params, f"{prefix}_warmup_updates", 0),
        ),
        0,
    )
    ramp = _coerce_int(
        _param(
            training_params,
            f"{prefix}_ramp_steps",
            _param(training_params, f"{prefix}_ramp_updates", 1),
        ),
        1,
    )
    if int(update_index) < warmup:
        return float(start)
    if ramp <= 0:
        return float(end)
    progress = min(1.0, max(0.0, (int(update_index) - warmup) / float(ramp)))
    return float(start + progress * (end - start))


def _selector_eval_settings(
    *,
    payload: dict[str, Any],
    spec: ActorCriticSpec,
    training_params: dict[str, Any] | None,
    intent_selection_mode: str | None,
) -> dict[str, Any]:
    params = _selector_training_params_from_checkpoint(payload, training_params)
    update_index = int(payload.get("update_index", 0) or 0)
    mode_code = _normalize_selector_mode_code(intent_selection_mode)
    spec_selector_enabled = bool(spec.intent_selector_enabled)
    config_selector_enabled = _coerce_bool(
        _param(params, "intent_selector_enabled", spec_selector_enabled),
        spec_selector_enabled,
    )
    selector_enabled = spec_selector_enabled and config_selector_enabled
    selector_mode = str(_param(params, "intent_selector_mode", "integrated") or "integrated").strip().lower()
    selector_enabled = selector_enabled and selector_mode == "integrated"
    disabled_reason = None
    if not spec_selector_enabled:
        disabled_reason = "policy_spec_selector_disabled"
    elif not config_selector_enabled:
        disabled_reason = "config_selector_disabled"
    elif selector_mode != "integrated":
        disabled_reason = f"selector_mode_{selector_mode}"
    eps = _linear_eval_schedule(
        params,
        "intent_selector_eps",
        update_index=update_index,
        default_start=0.0,
        default_end=0.0,
    )
    training_alpha = _linear_eval_schedule(
        params,
        "intent_selector_alpha",
        update_index=update_index,
        default_start=0.0,
        default_end=1.0,
    )
    return {
        "enabled": bool(selector_enabled),
        "spec_enabled": bool(spec_selector_enabled),
        "config_enabled": bool(config_selector_enabled),
        "selector_mode": str(selector_mode),
        "disabled_reason": disabled_reason,
        "mode_code": int(mode_code),
        "mode_label": _selector_mode_label(mode_code),
        "eps": float(np.clip(eps, 0.0, 1.0)),
        # Eval "learned_sample" should sample the learned selector distribution,
        # not the training exploration mix. Uniform behavior is explicit via
        # intent_selection_mode="uniform_random".
        "eval_sampling_eps": 0.0,
        "training_alpha": float(np.clip(training_alpha, 0.0, 1.0)),
        "eval_alpha": 1.0 if selector_enabled and mode_code != _SELECTOR_MODE_UNIFORM_RANDOM else 0.0,
        "multiselect_enabled": _coerce_bool(
            _param(params, "intent_selector_multiselect_enabled", False),
            False,
        ),
        "min_play_steps": max(
            1,
            _coerce_int(_param(params, "intent_selector_min_play_steps", 3), 3),
        ),
        "update_index": int(update_index),
    }


def _build_native_eval_runner(jax, jnp, spec: ActorCriticSpec):
    def _runner(
        static,
        initial_state,
        offense_params,
        defense_params,
        eval_key,
        role_flag_offense,
        role_flag_defense,
        horizon: int,
        offense_deterministic: bool,
        defense_deterministic: bool,
        selector_enabled: bool,
        selector_eps: float,
        selector_multiselect_enabled: bool,
        selector_min_play_steps: int,
        selector_mode_code: int,
    ):
        n_players = int(static.role_encoding.shape[0])
        offense_ids = static.offense_ids.astype(jnp.int32)
        defense_ids = static.defense_ids.astype(jnp.int32)

        def _team_actions(params, flat_obs, action_mask, intent_context, key, deterministic: bool):
            forward_out = actor_critic_forward(
                params,
                flat_obs,
                spec,
                jnp,
                intent_context=intent_context,
            )
            masked_out = apply_action_mask(
                forward_out["flat_policy_logits"],
                action_mask,
                spec,
                jax,
                jnp,
            )
            if deterministic:
                return masked_out["deterministic_actions"]
            return jax.random.categorical(
                key,
                masked_out["masked_logits"],
                axis=-1,
            ).astype(jnp.int32)

        def _zero_selector_trace(state):
            batch_shape = state.intent_index.shape
            batch_size = int(batch_shape[0])
            return {
                "selector_applied": jnp.zeros(batch_shape, dtype=jnp.int8),
                "selector_used": jnp.zeros(batch_shape, dtype=jnp.int8),
                "selector_uniform_used": jnp.zeros(batch_shape, dtype=jnp.int8),
                "selector_boundary_episode_start": jnp.zeros(batch_shape, dtype=jnp.int8),
                "selector_boundary_commitment_timeout": jnp.zeros(batch_shape, dtype=jnp.int8),
                "selector_boundary_completed_pass": jnp.zeros(batch_shape, dtype=jnp.int8),
                "selector_intent_index": jnp.full(batch_shape, -1, dtype=jnp.int32),
                "selector_raw_probs": jnp.zeros(
                    (batch_size, int(spec.num_intents)),
                    dtype=jnp.float32,
                ),
                "selector_raw_argmax": jnp.full(batch_shape, -1, dtype=jnp.int32),
                "selector_raw_max_prob": jnp.zeros(batch_shape, dtype=jnp.float32),
            }

        def _where_state(mask, selected_state, fallback_state):
            replaced = []
            for selected_value, fallback_value in zip(selected_state, fallback_state):
                if getattr(selected_value, "ndim", 0) <= 1:
                    replaced.append(jnp.where(mask, selected_value, fallback_value))
                else:
                    expand_shape = (mask.shape[0],) + (1,) * (selected_value.ndim - 1)
                    replaced.append(jnp.where(mask.reshape(expand_shape), selected_value, fallback_value))
            return type(fallback_state)(*replaced)

        def _maybe_apply_selector_segment_start(state, selector_obs, selector_key, completed_pass_boundary):
            metrics = _zero_selector_trace(state)
            if not bool(selector_enabled) or not bool(spec.intent_selector_enabled):
                return state, metrics

            batch_size = int(state.intent_index.shape[0])
            neutral_context = {
                "intent_index": jnp.zeros((batch_size,), dtype=jnp.int32),
                "intent_gate": jnp.zeros((batch_size,), dtype=jnp.float32),
            }
            selector_out = actor_critic_forward(
                offense_params,
                selector_obs,
                spec,
                jnp,
                intent_context=neutral_context,
            )
            logits = selector_out["selector_logits"]
            raw_probs = jax.nn.softmax(logits, axis=-1)
            eps = jnp.clip(jnp.asarray(selector_eps, dtype=jnp.float32), 0.0, 1.0)
            uniform_probs = jnp.full_like(raw_probs, 1.0 / float(max(1, int(spec.num_intents))))
            probs = ((1.0 - eps) * raw_probs) + (eps * uniform_probs)
            log_probs = jnp.log(jnp.maximum(probs, 1.0e-8))
            sample_key, uniform_key = jax.random.split(selector_key, 2)

            if int(selector_mode_code) == _SELECTOR_MODE_BEST_INTENT:
                chosen_intent = jnp.argmax(logits, axis=-1).astype(jnp.int32)
                selector_used = jnp.ones((batch_size,), dtype=jnp.bool_)
                uniform_used = jnp.zeros((batch_size,), dtype=jnp.bool_)
            elif int(selector_mode_code) == _SELECTOR_MODE_UNIFORM_RANDOM:
                chosen_intent = jax.random.randint(
                    uniform_key,
                    shape=(batch_size,),
                    minval=0,
                    maxval=int(spec.num_intents),
                    dtype=jnp.int32,
                )
                selector_used = jnp.zeros((batch_size,), dtype=jnp.bool_)
                uniform_used = jnp.ones((batch_size,), dtype=jnp.bool_)
            else:
                chosen_intent = jax.random.categorical(sample_key, log_probs, axis=-1).astype(jnp.int32)
                selector_used = jnp.ones((batch_size,), dtype=jnp.bool_)
                uniform_used = jnp.zeros((batch_size,), dtype=jnp.bool_)

            active = state.intent_active.astype(jnp.bool_)
            episode_start = active & (state.intent_age == 0)
            multiselect_enabled = jnp.asarray(selector_multiselect_enabled).astype(jnp.bool_)
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
            eligible = (
                static.enable_intent_learning.astype(jnp.bool_)
                & (episode_start | commitment_timeout | completed_pass)
            )
            selected_state = set_offense_intent_state_batch(
                static,
                state,
                chosen_intent,
                jnp.ones((batch_size,), dtype=jnp.int8),
                jnp,
            )
            next_state = _where_state(eligible, selected_state, state)
            return next_state, {
                "selector_applied": eligible.astype(jnp.int8),
                "selector_used": (eligible & selector_used).astype(jnp.int8),
                "selector_uniform_used": (eligible & uniform_used).astype(jnp.int8),
                "selector_boundary_episode_start": (eligible & episode_start).astype(jnp.int8),
                "selector_boundary_commitment_timeout": (eligible & commitment_timeout).astype(jnp.int8),
                "selector_boundary_completed_pass": (
                    eligible & completed_pass & (~commitment_timeout)
                ).astype(jnp.int8),
                "selector_intent_index": jnp.where(eligible, chosen_intent, jnp.asarray(-1, dtype=jnp.int32)),
                "selector_raw_probs": jnp.where(
                    eligible[:, None],
                    raw_probs.astype(jnp.float32),
                    jnp.zeros_like(raw_probs, dtype=jnp.float32),
                ),
                "selector_raw_argmax": jnp.where(
                    eligible,
                    jnp.argmax(raw_probs, axis=-1).astype(jnp.int32),
                    jnp.asarray(-1, dtype=jnp.int32),
                ),
                "selector_raw_max_prob": jnp.where(
                    eligible,
                    jnp.max(raw_probs, axis=-1).astype(jnp.float32),
                    jnp.asarray(0.0, dtype=jnp.float32),
                ),
            }

        def _scan_step(carry, _):
            state, key, completed_pass_boundary = carry
            key, selector_key, offense_key, defense_key, env_key = jax.random.split(key, 5)
            selector_obs = build_policy_observation_batch_with_role_flag(
                static,
                state,
                role_flag_offense,
                jnp,
                model_type=spec.model_type,
            )
            policy_state, selector_trace = _maybe_apply_selector_segment_start(
                state,
                selector_obs,
                selector_key,
                completed_pass_boundary,
            )
            full_action_mask = build_action_masks_batch(static, policy_state, jnp)
            offense_mask = full_action_mask[:, offense_ids, :]
            defense_mask = full_action_mask[:, defense_ids, :]
            offense_obs = build_policy_observation_batch_with_role_flag(
                static,
                policy_state,
                role_flag_offense,
                jnp,
                model_type=spec.model_type,
            )
            defense_obs = build_policy_observation_batch_with_role_flag(
                static,
                policy_state,
                role_flag_defense,
                jnp,
                model_type=spec.model_type,
            )
            offense_intent_context = build_policy_intent_context_batch_with_role_flag(
                static,
                policy_state,
                role_flag_offense,
                jnp,
            )
            defense_intent_context = build_policy_intent_context_batch_with_role_flag(
                static,
                policy_state,
                role_flag_defense,
                jnp,
            )
            offense_actions = _team_actions(
                offense_params,
                offense_obs,
                offense_mask,
                offense_intent_context,
                offense_key,
                offense_deterministic,
            )
            defense_actions = _team_actions(
                defense_params,
                defense_obs,
                defense_mask,
                defense_intent_context,
                defense_key,
                defense_deterministic,
            )
            full_actions = assemble_full_actions_jax(
                offense_actions,
                defense_actions,
                offense_ids,
                defense_ids,
                n_players,
                jnp,
            )
            env_keys = jax.random.split(env_key, initial_state.positions.shape[0])
            env_out = step_batch_minimal(static, policy_state, full_actions, env_keys, jax, jnp)
            trace = {
                "full_actions": full_actions.astype(jnp.int32),
                "ball_holder": policy_state.ball_holder.astype(jnp.int32),
                "positions": env_out.state.positions.astype(jnp.int32),
                "done": env_out.done.astype(jnp.int8),
                "terminal_episode_steps": env_out.terminal_episode_steps.astype(jnp.int32),
                "offense_rewards": jnp.sum(env_out.rewards[:, offense_ids], axis=1),
                "defense_rewards": jnp.sum(env_out.rewards[:, defense_ids], axis=1),
                "offense_score_delta": (
                    env_out.state.offense_score - policy_state.offense_score
                ).astype(jnp.float32),
                "defense_score_delta": (
                    env_out.state.defense_score - policy_state.defense_score
                ).astype(jnp.float32),
                "pass_attempts": env_out.pass_attempt.astype(jnp.int8),
                "pass_passer": env_out.pass_passer.astype(jnp.int32),
                "pass_receiver": env_out.pass_receiver.astype(jnp.int32),
                "completed_passes": env_out.completed_pass.astype(jnp.int8),
                "assists": env_out.assist.astype(jnp.int8),
                "turnovers": env_out.turnover.astype(jnp.int8),
                "shot_attempt": env_out.shot_attempt.astype(jnp.int8),
                "shot_success": env_out.shot_success.astype(jnp.int8),
                "shot_shooter": env_out.shot_shooter.astype(jnp.int32),
                "shot_value": env_out.shot_value.astype(jnp.float32),
                "shot_expected_points": env_out.shot_expected_points.astype(jnp.float32),
                "shot_distance": env_out.shot_distance.astype(jnp.float32),
                "shot_type": env_out.shot_type.astype(jnp.int32),
                "shot_q": env_out.shot_q.astype(jnp.int32),
                "shot_r": env_out.shot_r.astype(jnp.int32),
                "potential_assist": env_out.potential_assist.astype(jnp.int8),
                "assist_passer": env_out.assist_passer.astype(jnp.int32),
                "turnover_player": env_out.turnover_player.astype(jnp.int32),
                "turnover_reason": env_out.turnover_reason.astype(jnp.int32),
                "offensive_three_seconds": env_out.offensive_three_seconds.astype(jnp.int8),
                "defensive_lane_violation": env_out.defensive_lane_violation.astype(jnp.int8),
                "defensive_lane_violation_player": env_out.defensive_lane_violation_player.astype(jnp.int32),
                "rebound_attempt": env_out.rebound_attempt.astype(jnp.int8),
                "offensive_rebound": env_out.offensive_rebound.astype(jnp.int8),
                "defensive_rebound": env_out.defensive_rebound.astype(jnp.int8),
                "rebound_target_cell": env_out.rebound_target_cell.astype(jnp.int32),
                "rebound_winner": env_out.rebound_winner.astype(jnp.int32),
                "intent_index": policy_state.intent_index.astype(jnp.int32),
                "intent_active": policy_state.intent_active.astype(jnp.int8),
                "intent_age": policy_state.intent_age.astype(jnp.int32),
                "intent_commitment_remaining": policy_state.intent_commitment_remaining.astype(jnp.int32),
                "intent_visible_to_defense": policy_state.intent_visible_to_defense.astype(jnp.int8),
                "defense_intent_index": policy_state.defense_intent_index.astype(jnp.int32),
                "defense_intent_active": policy_state.defense_intent_active.astype(jnp.int8),
                "defense_intent_age": policy_state.defense_intent_age.astype(jnp.int32),
                "defense_intent_commitment_remaining": policy_state.defense_intent_commitment_remaining.astype(jnp.int32),
                **selector_trace,
            }
            next_completed_pass_boundary = (
                env_out.completed_pass.astype(jnp.bool_)
                & (~env_out.done.astype(jnp.bool_))
            )
            return (env_out.state, key, next_completed_pass_boundary), trace

        initial_completed_pass_boundary = jnp.zeros(
            (int(initial_state.positions.shape[0]),),
            dtype=jnp.bool_,
        )
        (_, _, _), trace = jax.lax.scan(
            _scan_step,
            (initial_state, eval_key, initial_completed_pass_boundary),
            xs=None,
            length=int(horizon),
        )
        return trace

    return jax.jit(_runner, static_argnums=(7, 8, 9, 10, 12, 13, 14))


def _episode_stats_from_trace(trace: dict[str, np.ndarray], *, take: int, horizon: int) -> dict[str, np.ndarray]:
    terminal_steps = np.max(np.asarray(trace["terminal_episode_steps"])[:, :take], axis=0)
    completed = terminal_steps > 0
    steps = np.where(completed, terminal_steps, int(horizon)).astype(np.int32)
    return {
        "steps": steps,
        "completed": completed.astype(np.int8),
        "offense_rewards": np.asarray(trace["offense_rewards"])[:, :take].sum(axis=0),
        "defense_rewards": np.asarray(trace["defense_rewards"])[:, :take].sum(axis=0),
        "offense_points": np.asarray(trace["offense_score_delta"])[:, :take].sum(axis=0),
        "defense_points": np.asarray(trace["defense_score_delta"])[:, :take].sum(axis=0),
        "pass_attempts": np.asarray(trace["pass_attempts"])[:, :take].sum(axis=0),
        "completed_passes": np.asarray(trace["completed_passes"])[:, :take].sum(axis=0),
        "assists": np.asarray(trace["assists"])[:, :take].sum(axis=0),
        "turnovers": np.asarray(trace["turnovers"])[:, :take].sum(axis=0),
        "rebound_attempts": np.asarray(trace["rebound_attempt"])[:, :take].sum(axis=0),
        "offensive_rebounds": np.asarray(trace["offensive_rebound"])[:, :take].sum(axis=0),
        "defensive_rebounds": np.asarray(trace["defensive_rebound"])[:, :take].sum(axis=0),
    }


def _mean(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float32)
    return float(arr.mean()) if arr.size else 0.0


def _sum(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float32)
    return float(arr.sum()) if arr.size else 0.0


def _init_player_stats(n_players: int) -> dict[int, dict[str, Any]]:
    return {
        pid: {
            "shots": 0,
            "makes": 0,
            "shot_types": {"dunk": [0, 0], "two": [0, 0], "three": [0, 0]},
            "assist_full_by_type": {"dunk": 0, "two": 0, "three": 0},
            "assists": 0,
            "potential_assists": 0,
            "turnovers": 0,
            "points": 0.0,
            "offensive_rebounds": 0,
            "rebound_chances": 0,
            "episodes": 0,
            "steps": 0,
            "shot_chart": {},
            "unassisted": {"dunk": 0, "two": 0, "three": 0},
        }
        for pid in range(int(n_players))
    }


def _init_aggregate_stats() -> dict[str, Any]:
    return {
        "shots": 0,
        "makes": 0,
        "shot_types": {"dunk": [0, 0], "two": [0, 0], "three": [0, 0]},
        "assist_full_by_type": {"dunk": 0, "two": 0, "three": 0},
        "assists": 0,
        "potential_assists": 0,
        "turnovers": 0,
        "points": 0.0,
        "offensive_rebounds": 0,
        "rebound_chances": 0,
        "episodes": 0,
        "steps": 0,
        "shot_chart": {},
        "unassisted": {"dunk": 0, "two": 0, "three": 0},
    }


def _init_eval_diagnostics() -> dict[str, Any]:
    return {
        "intent_selection_counts": {},
        "intent_inactive_count": 0,
        "defense_intent_selection_counts": {},
        "defense_intent_inactive_count": 0,
        "selector": {
            "applied_count": 0,
            "used_count": 0,
            "uniform_count": 0,
            "boundary_episode_start_count": 0,
            "boundary_commitment_timeout_count": 0,
            "boundary_completed_pass_count": 0,
            "selection_counts": {},
            "episode_start_selection_counts": {},
        },
        "turnover_reasons": {},
        "assist_links": {},
        "assist_links_by_type": {"dunk": {}, "two": {}, "three": {}},
        "potential_assist_links": {},
        "potential_assist_links_by_type": {"dunk": {}, "two": {}, "three": {}},
        "pass_links": {},
        "completed_pass_links": {},
        "shot_attempts_by_player": {},
        "made_shots_by_player": {},
        "action_mix": {
            "noop": 0,
            "move": 0,
            "shoot": 0,
            "pass": 0,
            "other": 0,
            "total": 0,
        },
        "holder_action_mix": {
            "noop": 0,
            "move": 0,
            "shoot": 0,
            "pass": 0,
            "other": 0,
            "total": 0,
        },
        "reward_breakdown": {
            "total_reward": 0.0,
            "expected_points": 0.0,
            "pass_reward": 0.0,
            "violation_reward": 0.0,
            "assist_potential": 0.0,
            "assist_full_bonus": 0.0,
            "phi_shaping": 0.0,
            "unexplained": 0.0,
        },
        "rebounds": {
            "attempts": 0,
            "offensive": 0,
            "defensive": 0,
            "by_player_offensive": {},
        },
    }


def _merge_aggregate_stats(dest: dict[str, Any] | None, src: dict[str, Any] | None) -> dict[str, Any]:
    if dest is None:
        dest = _init_aggregate_stats()
    if not src:
        return dest

    for key in ("shots", "makes", "assists", "potential_assists", "turnovers", "episodes", "steps", "offensive_rebounds", "rebound_chances"):
        dest[key] = int(dest.get(key, 0) or 0) + int(src.get(key, 0) or 0)
    dest["points"] = float(dest.get("points", 0.0) or 0.0) + float(src.get("points", 0.0) or 0.0)

    for shot_type in ("dunk", "two", "three"):
        src_pair = (src.get("shot_types") or {}).get(shot_type, [0, 0])
        dst_pair = dest["shot_types"].setdefault(shot_type, [0, 0])
        dst_pair[0] += int(src_pair[0] if isinstance(src_pair, (list, tuple)) else 0)
        dst_pair[1] += int(src_pair[1] if isinstance(src_pair, (list, tuple)) else 0)
        dest["assist_full_by_type"][shot_type] = int(dest["assist_full_by_type"].get(shot_type, 0)) + int(
            (src.get("assist_full_by_type") or {}).get(shot_type, 0) or 0
        )
        dest["unassisted"][shot_type] = int(dest["unassisted"].get(shot_type, 0)) + int(
            (src.get("unassisted") or {}).get(shot_type, 0) or 0
        )

    for loc, vals in (src.get("shot_chart") or {}).items():
        dst_pair = dest["shot_chart"].setdefault(str(loc), [0, 0])
        dst_pair[0] += int(vals[0] if isinstance(vals, (list, tuple)) and len(vals) > 0 else 0)
        dst_pair[1] += int(vals[1] if isinstance(vals, (list, tuple)) and len(vals) > 1 else 0)
    return dest


def _accumulate_team_stats_from_players(
    dest: dict[str, Any] | None,
    player_stats: dict[int, dict[str, Any]],
    team_ids: list[int],
    *,
    episodes: int,
    steps: int,
) -> dict[str, Any]:
    merged = _init_aggregate_stats()
    merged["episodes"] = int(episodes)
    merged["steps"] = int(steps)
    max_rebound_chances = 0
    for pid in team_ids:
        entry = player_stats.get(int(pid))
        if entry:
            max_rebound_chances = max(max_rebound_chances, int(entry.get("rebound_chances", 0) or 0))
            merged = _merge_aggregate_stats(merged, entry)
    merged["rebound_chances"] = max_rebound_chances
    return _merge_aggregate_stats(dest, merged)


def _shot_type_label(code: int) -> str:
    if int(code) == int(SHOT_TYPE_DUNK):
        return "dunk"
    if int(code) == int(SHOT_TYPE_THREE):
        return "three"
    return "two"


def _turnover_reason_label(code: int) -> str:
    labels = {
        int(TURNOVER_REASON_PASS_OUT_OF_BOUNDS): "pass_out_of_bounds",
        int(TURNOVER_REASON_INTERCEPTED): "intercepted",
        int(TURNOVER_REASON_DEFENDER_PRESSURE): "defender_pressure",
        int(TURNOVER_REASON_MOVE_OUT_OF_BOUNDS): "move_out_of_bounds",
        int(TURNOVER_REASON_SHOT_CLOCK): "shot_clock_violation",
        int(TURNOVER_REASON_OFFENSIVE_THREE_SECONDS): "offensive_three_seconds",
    }
    return labels.get(int(code), "unknown")


def _action_bucket(action_id: int) -> str:
    aid = int(action_id)
    if aid == int(ActionType.NOOP.value):
        return "noop"
    if int(MOVE_ACTION_START) <= aid < int(MOVE_ACTION_END):
        return "move"
    if aid == int(ActionType.SHOOT.value):
        return "shoot"
    if int(PASS_ACTION_START) <= aid < int(PASS_ACTION_END):
        return "pass"
    return "other"


def _record_action_mix(eval_diagnostics: dict[str, Any], actions: np.ndarray, user_team_ids: list[int]) -> None:
    action_mix = eval_diagnostics["action_mix"]
    for pid in user_team_ids:
        bucket = _action_bucket(int(actions[int(pid)]))
        action_mix[bucket] = int(action_mix.get(bucket, 0)) + 1
        action_mix["total"] = int(action_mix.get("total", 0)) + 1


def _record_holder_action_mix(
    eval_diagnostics: dict[str, Any],
    actions: np.ndarray,
    ball_holder: int,
    user_team_ids_set: set[int],
) -> None:
    holder_id = int(ball_holder)
    if holder_id not in user_team_ids_set:
        return
    holder_mix = eval_diagnostics["holder_action_mix"]
    bucket = _action_bucket(int(actions[holder_id]))
    holder_mix[bucket] = int(holder_mix.get(bucket, 0)) + 1
    holder_mix["total"] = int(holder_mix.get("total", 0)) + 1


def _record_shot_event(
    *,
    stats: dict[int, dict[str, Any]],
    shot_accumulator: dict[str, list[int]] | None,
    shooter_id: int,
    success: bool,
    shot_value: float,
    shot_type: str,
    q: int,
    r: int,
    assist_full: bool,
) -> None:
    entry = stats.get(int(shooter_id))
    if entry is None:
        return
    entry["shots"] += 1
    entry["makes"] += int(bool(success))
    entry["points"] += float(shot_value) if success else 0.0
    pair = entry["shot_types"].setdefault(shot_type, [0, 0])
    pair[0] += 1
    pair[1] += int(bool(success))
    loc = f"{int(q)},{int(r)}"
    chart_pair = entry["shot_chart"].setdefault(loc, [0, 0])
    chart_pair[0] += 1
    chart_pair[1] += int(bool(success))
    if shot_accumulator is not None:
        shot_pair = shot_accumulator.setdefault(loc, [0, 0])
        shot_pair[0] += 1
        shot_pair[1] += int(bool(success))
    if success:
        if assist_full:
            entry["assist_full_by_type"][shot_type] = int(entry["assist_full_by_type"].get(shot_type, 0)) + 1
        else:
            entry["unassisted"][shot_type] = int(entry["unassisted"].get(shot_type, 0)) + 1


def _record_rebound_heatmap_event(
    *,
    rebound_accumulator: dict[str, Any],
    shot_q: int,
    shot_r: int,
    target_cell: int,
    winner_id: int,
    offensive: bool,
    positions: np.ndarray,
    cell_coords: np.ndarray,
) -> None:
    source_key = f"{int(shot_q)},{int(shot_r)}"
    bucket = rebound_accumulator.setdefault(
        source_key,
        {
            "total": 0,
            "targets": {},
            "target_offensive": {},
            "rebounders": {},
            "rebounder_offensive": {},
        },
    )
    bucket["total"] = int(bucket.get("total", 0)) + 1

    target_idx = int(target_cell)
    if 0 <= target_idx < int(cell_coords.shape[0]):
        target_coord = cell_coords[target_idx]
        target_key = f"{int(target_coord[0])},{int(target_coord[1])}"
        targets = bucket.setdefault("targets", {})
        targets[target_key] = int(targets.get(target_key, 0)) + 1
        if bool(offensive):
            target_offensive = bucket.setdefault("target_offensive", {})
            target_offensive[target_key] = int(target_offensive.get(target_key, 0)) + 1

    winner = int(winner_id)
    if 0 <= winner < int(positions.shape[0]):
        winner_pos = positions[winner]
        winner_key = f"{int(winner_pos[0])},{int(winner_pos[1])}"
        rebounders = bucket.setdefault("rebounders", {})
        rebounders[winner_key] = int(rebounders.get(winner_key, 0)) + 1
        if bool(offensive):
            rebounder_offensive = bucket.setdefault("rebounder_offensive", {})
            rebounder_offensive[winner_key] = int(rebounder_offensive.get(winner_key, 0)) + 1


def _record_shot_diagnostics(
    eval_diagnostics: dict[str, Any],
    *,
    shooter_id: int,
    success: bool,
    user_team_ids_set: set[int],
) -> None:
    if int(shooter_id) not in user_team_ids_set:
        return
    key = str(int(shooter_id))
    attempts = eval_diagnostics.setdefault("shot_attempts_by_player", {})
    attempts[key] = int(attempts.get(key, 0)) + 1
    if success:
        makes = eval_diagnostics.setdefault("made_shots_by_player", {})
        makes[key] = int(makes.get(key, 0)) + 1


def _record_pass_link_diagnostics(
    eval_diagnostics: dict[str, Any],
    *,
    passer_id: int,
    receiver_id: int,
    completed: bool,
    user_team_ids_set: set[int],
) -> None:
    if int(passer_id) < 0 or int(receiver_id) < 0:
        return
    if int(passer_id) not in user_team_ids_set or int(receiver_id) not in user_team_ids_set:
        return
    link_key = f"{int(passer_id)}->{int(receiver_id)}"
    pass_links = eval_diagnostics.setdefault("pass_links", {})
    pass_links[link_key] = int(pass_links.get(link_key, 0)) + 1
    if completed:
        completed_links = eval_diagnostics.setdefault("completed_pass_links", {})
        completed_links[link_key] = int(completed_links.get(link_key, 0)) + 1


def _record_assist_event(
    *,
    stats: dict[int, dict[str, Any]],
    eval_diagnostics: dict[str, Any],
    passer_id: int,
    shooter_id: int,
    shot_type: str,
    potential: bool,
    full: bool,
    user_team_ids_set: set[int],
) -> None:
    if int(passer_id) < 0:
        return
    entry = stats.get(int(passer_id))
    if entry is not None:
        entry["potential_assists"] += int(bool(potential))
        entry["assists"] += int(bool(full))
    if int(passer_id) == int(shooter_id):
        return
    if int(passer_id) not in user_team_ids_set or int(shooter_id) not in user_team_ids_set:
        return
    link_key = f"{int(passer_id)}->{int(shooter_id)}"
    if full:
        links = eval_diagnostics["assist_links"]
        links[link_key] = int(links.get(link_key, 0)) + 1
        by_type = eval_diagnostics["assist_links_by_type"].setdefault(shot_type, {})
        by_type[link_key] = int(by_type.get(link_key, 0)) + 1
    if potential and not full:
        links = eval_diagnostics["potential_assist_links"]
        links[link_key] = int(links.get(link_key, 0)) + 1
        by_type = eval_diagnostics["potential_assist_links_by_type"].setdefault(shot_type, {})
        by_type[link_key] = int(by_type.get(link_key, 0)) + 1


def _record_turnover_event(
    *,
    stats: dict[int, dict[str, Any]],
    eval_diagnostics: dict[str, Any],
    player_id: int,
    reason_code: int,
    user_team_ids_set: set[int],
    count_for_user_team: bool | None = None,
) -> str:
    reason = _turnover_reason_label(int(reason_code))
    entry = stats.get(int(player_id))
    if entry is not None:
        entry["turnovers"] += 1
    should_count = bool(count_for_user_team) if count_for_user_team is not None else int(player_id) in user_team_ids_set
    if should_count:
        reasons = eval_diagnostics["turnover_reasons"]
        reasons[reason] = int(reasons.get(reason, 0)) + 1
    return reason


def run_native_jax_evaluation(
    *,
    num_episodes: int,
    player_deterministic: bool,
    opponent_deterministic: bool,
    required_params: dict,
    optional_params: dict,
    unified_policy_path: str,
    opponent_policy_path: str | None,
    user_team_name: str,
    role_flag_offense: float,
    role_flag_defense: float,
    training_params: dict | None = None,
    eval_seed: int | None = None,
    intent_selection_mode: str = "learned_sample",
    progress_callback=None,
) -> dict[str, Any]:
    jax, jnp = ensure_jax_available("basketworld_jax/eval/native.py")
    unified_payload, unified_params, spec = _load_checkpoint_params(unified_policy_path, jax)
    if opponent_policy_path:
        opponent_payload, opponent_params, opponent_spec = _load_checkpoint_params(opponent_policy_path, jax)
        if opponent_spec != spec:
            raise ValueError("JAX opponent checkpoint policy_spec does not match unified checkpoint.")
    else:
        opponent_payload = unified_payload
        opponent_params = unified_params

    user_team = Team.OFFENSE if str(user_team_name) == "OFFENSE" else Team.DEFENSE
    if user_team == Team.OFFENSE:
        offense_params = unified_params
        defense_params = opponent_params
        offense_deterministic = bool(player_deterministic)
        defense_deterministic = bool(opponent_deterministic)
    else:
        offense_params = opponent_params
        defense_params = unified_params
        offense_deterministic = bool(opponent_deterministic)
        defense_deterministic = bool(player_deterministic)
    selector_payload = unified_payload if user_team == Team.OFFENSE else opponent_payload
    selector_settings = _selector_eval_settings(
        payload=selector_payload,
        spec=spec,
        training_params=training_params,
        intent_selection_mode=intent_selection_mode,
    )

    env_kwargs, static_overrides = _split_native_env_params(optional_params)
    static_params = _jax_static_env_params_from_payload(unified_payload, opponent_payload)
    static_params.update(static_overrides)
    env = basketworld.HexagonBasketballEnv(
        **required_params,
        **env_kwargs,
        render_mode=None,
    )
    for key, value in static_params.items():
        setattr(env, key, value)
    static = build_kernel_static_from_env(env, jnp)
    horizon = _native_eval_horizon(env, training_params, unified_payload)
    configured_batch_size = int(
        dict(unified_payload.get("trainer_config", {})).get("kernel_batch_size", 4096)
    )
    batch_size = max(1, min(int(num_episodes), configured_batch_size))
    runner = _build_native_eval_runner(jax, jnp, spec)

    n_players = int(getattr(env, "n_players", 0))
    offense_ids = [int(pid) for pid in getattr(env, "offense_ids", [])]
    defense_ids = [int(pid) for pid in getattr(env, "defense_ids", [])]
    user_team_ids = offense_ids if user_team == Team.OFFENSE else defense_ids
    user_team_ids_set = set(user_team_ids)
    offense_sign = 1.0 if user_team == Team.OFFENSE else -1.0
    pass_reward = float(getattr(env, "pass_reward", 0.0) or 0.0)
    violation_reward = float(getattr(env, "violation_reward", 0.0) or 0.0)
    potential_assist_pct = float(getattr(env, "potential_assist_pct", 0.0) or 0.0)
    full_assist_bonus_pct = float(getattr(env, "full_assist_bonus_pct", 0.0) or 0.0)
    shot_clock_steps = int(getattr(env, "shot_clock_steps", horizon))
    three_point_distance = float(getattr(env, "three_point_distance", 4.0))

    results: list[dict[str, Any]] = []
    shot_accumulator: dict[str, list[int]] = {}
    rebound_accumulator: dict[str, Any] = {}
    cell_coords_np = np.asarray(jax.device_get(static.cell_coords), dtype=np.int32)
    per_player_stats = _init_player_stats(n_players)
    per_intent_stats: dict[str, dict[str, Any]] = {}
    eval_diagnostics = _init_eval_diagnostics()
    all_steps: list[int] = []
    all_completed: list[int] = []
    all_offense_rewards: list[float] = []
    all_defense_rewards: list[float] = []
    all_offense_points: list[float] = []
    all_defense_points: list[float] = []
    all_pass_attempts: list[float] = []
    all_completed_passes: list[float] = []
    all_assists: list[float] = []
    all_turnovers: list[float] = []
    all_rebound_attempts: list[float] = []
    all_offensive_rebounds: list[float] = []
    all_defensive_rebounds: list[float] = []

    start = perf_counter()
    completed_episodes = 0
    if eval_seed is None:
        eval_seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
    key = jax.random.PRNGKey(int(eval_seed))
    while completed_episodes < int(num_episodes):
        take = min(batch_size, int(num_episodes) - completed_episodes)
        key, reset_key, eval_key = jax.random.split(key, 3)
        reset_keys = jax.random.split(reset_key, batch_size)
        initial_state = reset_batch_minimal(static, reset_keys, jax, jnp)
        trace_device = runner(
            static,
            initial_state,
            offense_params,
            defense_params,
            eval_key,
            jnp.asarray(float(role_flag_offense), dtype=jnp.float32),
            jnp.asarray(float(role_flag_defense), dtype=jnp.float32),
            int(horizon),
            bool(offense_deterministic),
            bool(defense_deterministic),
            bool(selector_settings["enabled"]),
            float(selector_settings["eval_sampling_eps"]),
            bool(selector_settings["multiselect_enabled"]),
            int(selector_settings["min_play_steps"]),
            int(selector_settings["mode_code"]),
        )
        trace = jax.device_get(trace_device)
        stats = _episode_stats_from_trace(trace, take=take, horizon=horizon)
        for idx in range(take):
            episode_num = completed_episodes + idx + 1
            step_count = int(stats["steps"][idx])
            active_steps = max(0, min(step_count, int(horizon)))
            offense_reward = float(stats["offense_rewards"][idx])
            defense_reward = float(stats["defense_rewards"][idx])
            user_reward = offense_reward if user_team == Team.OFFENSE else defense_reward
            episode_player_stats = _init_player_stats(n_players)
            shots_payload: dict[str, dict[str, Any]] = {}
            turnovers_payload: list[dict[str, Any]] = []
            defensive_lane_violations_payload: list[dict[str, Any]] = []
            rebounds_payload: list[dict[str, Any]] = []
            episode_shots: dict[str, list[int]] = {}
            selector_start_applied = (
                bool(int(trace["selector_applied"][0, idx]))
                and bool(int(trace["selector_boundary_episode_start"][0, idx]))
            ) if int(horizon) > 0 else False
            if selector_start_applied:
                episode_intent_index = int(trace["selector_intent_index"][0, idx])
                episode_intent_active = episode_intent_index >= 0
            else:
                episode_intent_active = bool(int(trace["intent_active"][0, idx])) if int(horizon) > 0 else False
                episode_intent_index = int(trace["intent_index"][0, idx]) if episode_intent_active else None
            episode_intent_key = str(episode_intent_index) if episode_intent_active else "none"
            episode_intent_visible_to_defense = (
                bool(int(trace["intent_visible_to_defense"][0, idx])) if int(horizon) > 0 else False
            )
            if episode_intent_active:
                intent_counts = eval_diagnostics.setdefault("intent_selection_counts", {})
                intent_counts[episode_intent_key] = int(intent_counts.get(episode_intent_key, 0)) + 1
            else:
                eval_diagnostics["intent_inactive_count"] = int(eval_diagnostics.get("intent_inactive_count", 0)) + 1

            episode_defense_intent_active = (
                bool(int(trace["defense_intent_active"][0, idx])) if int(horizon) > 0 else False
            )
            episode_defense_intent_index = (
                int(trace["defense_intent_index"][0, idx]) if episode_defense_intent_active else None
            )
            if episode_defense_intent_active:
                defense_counts = eval_diagnostics.setdefault("defense_intent_selection_counts", {})
                defense_key = str(episode_defense_intent_index)
                defense_counts[defense_key] = int(defense_counts.get(defense_key, 0)) + 1
            else:
                eval_diagnostics["defense_intent_inactive_count"] = (
                    int(eval_diagnostics.get("defense_intent_inactive_count", 0)) + 1
                )

            selector_diag = eval_diagnostics.setdefault("selector", {})
            selector_applied = np.asarray(trace["selector_applied"])[:active_steps, idx].astype(bool)
            selector_used = np.asarray(trace["selector_used"])[:active_steps, idx].astype(bool)
            selector_uniform = np.asarray(trace["selector_uniform_used"])[:active_steps, idx].astype(bool)
            selector_diag["applied_count"] = int(selector_diag.get("applied_count", 0)) + int(selector_applied.sum())
            selector_diag["used_count"] = int(selector_diag.get("used_count", 0)) + int(selector_used.sum())
            selector_diag["uniform_count"] = int(selector_diag.get("uniform_count", 0)) + int(selector_uniform.sum())
            selector_diag["boundary_episode_start_count"] = int(
                selector_diag.get("boundary_episode_start_count", 0)
            ) + int(np.asarray(trace["selector_boundary_episode_start"])[:active_steps, idx].sum())
            selector_diag["boundary_commitment_timeout_count"] = int(
                selector_diag.get("boundary_commitment_timeout_count", 0)
            ) + int(np.asarray(trace["selector_boundary_commitment_timeout"])[:active_steps, idx].sum())
            selector_diag["boundary_completed_pass_count"] = int(
                selector_diag.get("boundary_completed_pass_count", 0)
            ) + int(np.asarray(trace["selector_boundary_completed_pass"])[:active_steps, idx].sum())
            selector_counts = selector_diag.setdefault("selection_counts", {})
            selector_start_counts = selector_diag.setdefault("episode_start_selection_counts", {})
            selector_intents = np.asarray(trace["selector_intent_index"])[:active_steps, idx]
            selector_episode_start = np.asarray(trace["selector_boundary_episode_start"])[:active_steps, idx].astype(bool)
            selector_episode_start_applied = selector_applied & selector_episode_start
            for selected_intent in selector_intents[selector_applied]:
                selected_key = str(int(selected_intent))
                selector_counts[selected_key] = int(selector_counts.get(selected_key, 0)) + 1
            for selected_intent in selector_intents[selector_episode_start_applied]:
                selected_key = str(int(selected_intent))
                selector_start_counts[selected_key] = int(selector_start_counts.get(selected_key, 0)) + 1
            if selector_episode_start_applied.any():
                raw_probs = np.asarray(trace["selector_raw_probs"])[:active_steps, idx]
                raw_argmax = np.asarray(trace["selector_raw_argmax"])[:active_steps, idx]
                raw_max_prob = np.asarray(trace["selector_raw_max_prob"])[:active_steps, idx]
                start_raw_probs = raw_probs[selector_episode_start_applied]
                selector_diag["episode_start_raw_prob_count"] = int(
                    selector_diag.get("episode_start_raw_prob_count", 0)
                ) + int(start_raw_probs.shape[0])
                prev_sums = np.asarray(
                    selector_diag.get(
                        "episode_start_raw_prob_sums",
                        [0.0] * int(spec.num_intents),
                    ),
                    dtype=np.float64,
                )
                selector_diag["episode_start_raw_prob_sums"] = (
                    prev_sums + start_raw_probs.astype(np.float64).sum(axis=0)
                ).tolist()
                selector_diag["episode_start_raw_max_prob_sum"] = float(
                    selector_diag.get("episode_start_raw_max_prob_sum", 0.0)
                ) + float(raw_max_prob[selector_episode_start_applied].sum())
                argmax_counts = selector_diag.setdefault("episode_start_argmax_counts", {})
                for argmax_intent in raw_argmax[selector_episode_start_applied]:
                    argmax_key = str(int(argmax_intent))
                    argmax_counts[argmax_key] = int(argmax_counts.get(argmax_key, 0)) + 1

            for pid in per_player_stats:
                per_player_stats[pid]["episodes"] += 1
                per_player_stats[pid]["steps"] += step_count

            for t in range(active_steps):
                _record_action_mix(eval_diagnostics, trace["full_actions"][t, idx], user_team_ids)
                _record_holder_action_mix(
                    eval_diagnostics,
                    trace["full_actions"][t, idx],
                    int(trace["ball_holder"][t, idx]),
                    user_team_ids_set,
                )
                if int(trace["pass_attempts"][t, idx]):
                    _record_pass_link_diagnostics(
                        eval_diagnostics,
                        passer_id=int(trace["pass_passer"][t, idx]),
                        receiver_id=int(trace["pass_receiver"][t, idx]),
                        completed=bool(int(trace["completed_passes"][t, idx])),
                        user_team_ids_set=user_team_ids_set,
                    )

                if int(trace["shot_attempt"][t, idx]):
                    shooter_id = int(trace["shot_shooter"][t, idx])
                    shot_success = bool(int(trace["shot_success"][t, idx]))
                    shot_value = float(trace["shot_value"][t, idx])
                    shot_type = _shot_type_label(int(trace["shot_type"][t, idx]))
                    q = int(trace["shot_q"][t, idx])
                    r = int(trace["shot_r"][t, idx])
                    assist_full = bool(int(trace["assists"][t, idx]))
                    potential_assist = bool(int(trace["potential_assist"][t, idx]))
                    passer_id = int(trace["assist_passer"][t, idx])
                    _record_shot_event(
                        stats=per_player_stats,
                        shot_accumulator=shot_accumulator,
                        shooter_id=shooter_id,
                        success=shot_success,
                        shot_value=shot_value,
                        shot_type=shot_type,
                        q=q,
                        r=r,
                        assist_full=assist_full,
                    )
                    _record_shot_diagnostics(
                        eval_diagnostics,
                        shooter_id=shooter_id,
                        success=shot_success,
                        user_team_ids_set=user_team_ids_set,
                    )
                    _record_shot_event(
                        stats=episode_player_stats,
                        shot_accumulator=None,
                        shooter_id=shooter_id,
                        success=shot_success,
                        shot_value=shot_value,
                        shot_type=shot_type,
                        q=q,
                        r=r,
                        assist_full=assist_full,
                    )
                    if potential_assist or assist_full:
                        _record_assist_event(
                            stats=per_player_stats,
                            eval_diagnostics=eval_diagnostics,
                            passer_id=passer_id,
                            shooter_id=shooter_id,
                            shot_type=shot_type,
                            potential=potential_assist,
                            full=assist_full,
                            user_team_ids_set=user_team_ids_set,
                        )
                        if passer_id in episode_player_stats:
                            episode_player_stats[passer_id]["potential_assists"] += int(potential_assist)
                            episode_player_stats[passer_id]["assists"] += int(assist_full)

                    loc = f"{q},{r}"
                    episode_shot_pair = episode_shots.setdefault(loc, [0, 0])
                    episode_shot_pair[0] += 1
                    episode_shot_pair[1] += int(shot_success)
                    shots_payload[str(shooter_id)] = {
                        "success": shot_success,
                        "distance": float(trace["shot_distance"][t, idx]),
                        "is_three": shot_type == "three",
                        "expected_points": float(trace["shot_expected_points"][t, idx]),
                        "shot_value": shot_value,
                        "assist_full": assist_full,
                        "assist_potential": potential_assist,
                        "assist_passer_id": passer_id if passer_id >= 0 else None,
                    }

                if int(trace["turnovers"][t, idx]):
                    turnover_player = int(trace["turnover_player"][t, idx])
                    turnover_reason_code = int(trace["turnover_reason"][t, idx])
                    count_for_user_team = turnover_player in user_team_ids_set
                    if turnover_player < 0 and turnover_reason_code == int(TURNOVER_REASON_SHOT_CLOCK):
                        count_for_user_team = user_team == Team.OFFENSE
                    reason = _record_turnover_event(
                        stats=per_player_stats,
                        eval_diagnostics=eval_diagnostics,
                        player_id=turnover_player,
                        reason_code=turnover_reason_code,
                        user_team_ids_set=user_team_ids_set,
                        count_for_user_team=count_for_user_team,
                    )
                    if turnover_player >= 0:
                        if turnover_player in episode_player_stats:
                            episode_player_stats[turnover_player]["turnovers"] += 1
                    turnovers_payload.append(
                        {
                            "player_id": turnover_player if turnover_player >= 0 else None,
                            "reason": reason,
                        }
                    )
                if int(trace["defensive_lane_violation"][t, idx]):
                    defender_id = int(trace["defensive_lane_violation_player"][t, idx])
                    defensive_lane_violations_payload.append(
                        {
                            "player_id": defender_id,
                            "reason": "illegal_defense",
                        }
                    )
                if int(trace["rebound_attempt"][t, idx]):
                    rebound_offensive = bool(int(trace["offensive_rebound"][t, idx]))
                    rebound_defensive = bool(int(trace["defensive_rebound"][t, idx]))
                    rebound_winner = int(trace["rebound_winner"][t, idx])
                    rebound_target_cell = int(trace["rebound_target_cell"][t, idx])
                    rebound_diag = eval_diagnostics.setdefault(
                        "rebounds",
                        {"attempts": 0, "offensive": 0, "defensive": 0, "by_player_offensive": {}},
                    )
                    rebound_diag["attempts"] = int(rebound_diag.get("attempts", 0)) + 1
                    rebound_diag["offensive"] = int(rebound_diag.get("offensive", 0)) + int(rebound_offensive)
                    rebound_diag["defensive"] = int(rebound_diag.get("defensive", 0)) + int(rebound_defensive)
                    for stats_target in (per_player_stats, episode_player_stats):
                        for offense_pid in offense_ids:
                            if offense_pid in stats_target:
                                stats_target[offense_pid]["rebound_chances"] = int(stats_target[offense_pid].get("rebound_chances", 0)) + 1
                        if rebound_offensive and rebound_winner in stats_target:
                            stats_target[rebound_winner]["offensive_rebounds"] = int(
                                stats_target[rebound_winner].get("offensive_rebounds", 0)
                            ) + 1
                    if rebound_offensive and rebound_winner >= 0:
                        by_player = rebound_diag.setdefault("by_player_offensive", {})
                        rebound_key = str(int(rebound_winner))
                        by_player[rebound_key] = int(by_player.get(rebound_key, 0)) + 1
                    _record_rebound_heatmap_event(
                        rebound_accumulator=rebound_accumulator,
                        shot_q=int(trace["shot_q"][t, idx]),
                        shot_r=int(trace["shot_r"][t, idx]),
                        target_cell=rebound_target_cell,
                        winner_id=rebound_winner,
                        offensive=rebound_offensive,
                        positions=np.asarray(trace["positions"][t, idx], dtype=np.int32),
                        cell_coords=cell_coords_np,
                    )
                    rebounds_payload.append(
                        {
                            "attempt": True,
                            "offensive": rebound_offensive,
                            "defensive": rebound_defensive,
                            "winner": rebound_winner if rebound_winner >= 0 else None,
                            "winner_team": "OFFENSE" if rebound_offensive else ("DEFENSE" if rebound_defensive else None),
                            "target_cell_index": rebound_target_cell if rebound_target_cell >= 0 else None,
                        }
                    )

            completed_passes = float(np.asarray(trace["completed_passes"])[:active_steps, idx].sum())
            defensive_lane_violations = float(
                np.asarray(trace["defensive_lane_violation"])[:active_steps, idx].sum()
            )
            shot_attempts = np.asarray(trace["shot_attempt"])[:active_steps, idx].astype(bool)
            shot_success_flags = np.asarray(trace["shot_success"])[:active_steps, idx].astype(bool)
            shot_values = np.asarray(trace["shot_value"])[:active_steps, idx].astype(np.float32)
            shot_expected_points = np.asarray(trace["shot_expected_points"])[:active_steps, idx]
            defensive_rebound_flags = np.asarray(trace["defensive_rebound"])[:active_steps, idx].astype(bool)
            potential_flags = np.asarray(trace["potential_assist"])[:active_steps, idx].astype(bool)
            assist_flags = np.asarray(trace["assists"])[:active_steps, idx].astype(bool)
            if bool(getattr(env, "enable_rebounds", False)):
                rebound_reward_mode = str(
                    getattr(env, "rebound_terminal_reward_mode", "actual_points") or "actual_points"
                )
                scored_points = shot_values * shot_success_flags.astype(np.float32)
                if rebound_reward_mode == "last_shot_ep":
                    shot_reward_component = np.where(
                        shot_attempts & (shot_success_flags | defensive_rebound_flags),
                        shot_expected_points,
                        0.0,
                    )
                elif rebound_reward_mode == "last_shot_ep_on_defensive_rebound":
                    shot_reward_component = np.where(
                        shot_attempts & shot_success_flags,
                        scored_points,
                        np.where(
                            shot_attempts & defensive_rebound_flags,
                            shot_expected_points,
                            0.0,
                        ),
                    )
                else:
                    shot_reward_component = np.where(shot_attempts, scored_points, 0.0)
            else:
                shot_reward_component = shot_expected_points
            expected_amt = offense_sign * float(shot_reward_component.sum())
            pass_amt = offense_sign * pass_reward * completed_passes
            violation_amt = offense_sign * violation_reward * defensive_lane_violations
            potential_amt = offense_sign * potential_assist_pct * float(shot_expected_points[potential_flags].sum())
            full_amt = offense_sign * full_assist_bonus_pct * float(shot_expected_points[assist_flags].sum())
            known_reward = expected_amt + pass_amt + violation_amt + potential_amt + full_amt
            reward_breakdown = eval_diagnostics["reward_breakdown"]
            reward_breakdown["total_reward"] += user_reward
            reward_breakdown["expected_points"] += expected_amt
            reward_breakdown["pass_reward"] += pass_amt
            reward_breakdown["violation_reward"] += violation_amt
            reward_breakdown["assist_potential"] += potential_amt
            reward_breakdown["assist_full_bonus"] += full_amt
            reward_breakdown["unexplained"] += user_reward - known_reward

            per_intent_stats[episode_intent_key] = _accumulate_team_stats_from_players(
                per_intent_stats.get(episode_intent_key),
                episode_player_stats,
                user_team_ids,
                episodes=1,
                steps=step_count,
            )

            results.append(
                {
                    "episode": int(episode_num),
                    "intent_index": episode_intent_index,
                    "intent_active": episode_intent_active,
                    "intent_visible_to_defense": episode_intent_visible_to_defense,
                    "defense_intent_index": episode_defense_intent_index,
                    "defense_intent_active": episode_defense_intent_active,
                    "steps": step_count,
                    "episode_rewards": {
                        "offense": offense_reward,
                        "defense": defense_reward,
                    },
                    "outcome_info": {
                        "shots": shots_payload,
                        "turnovers": turnovers_payload,
                        "defensive_lane_violations": defensive_lane_violations_payload,
                        "rebounds": rebounds_payload,
                        "rebound": rebounds_payload[-1] if rebounds_payload else None,
                        "shot_clock": shot_clock_steps,
                        "three_point_distance": three_point_distance,
                    },
                    "shot_counts": episode_shots,
                }
            )
        all_steps.extend([int(v) for v in stats["steps"].tolist()])
        all_completed.extend([int(v) for v in stats["completed"].tolist()])
        all_offense_rewards.extend([float(v) for v in stats["offense_rewards"].tolist()])
        all_defense_rewards.extend([float(v) for v in stats["defense_rewards"].tolist()])
        all_offense_points.extend([float(v) for v in stats["offense_points"].tolist()])
        all_defense_points.extend([float(v) for v in stats["defense_points"].tolist()])
        all_pass_attempts.extend([float(v) for v in stats["pass_attempts"].tolist()])
        all_completed_passes.extend([float(v) for v in stats["completed_passes"].tolist()])
        all_assists.extend([float(v) for v in stats["assists"].tolist()])
        all_turnovers.extend([float(v) for v in stats["turnovers"].tolist()])
        all_rebound_attempts.extend([float(v) for v in stats["rebound_attempts"].tolist()])
        all_offensive_rebounds.extend([float(v) for v in stats["offensive_rebounds"].tolist()])
        all_defensive_rebounds.extend([float(v) for v in stats["defensive_rebounds"].tolist()])
        completed_episodes += take
        if progress_callback is not None:
            progress_callback(completed_episodes, int(num_episodes))

    elapsed = max(perf_counter() - start, 1.0e-12)
    shot_type_attempts = {
        shot_type: int(
            sum(
                int((stats.get("shot_types") or {}).get(shot_type, [0, 0])[0])
                for stats in per_player_stats.values()
            )
        )
        for shot_type in ("dunk", "two", "three")
    }
    total_shot_attempts = int(sum(shot_type_attempts.values()))
    shot_type_shares = {
        shot_type: (
            float(count / total_shot_attempts)
            if total_shot_attempts > 0
            else 0.0
        )
        for shot_type, count in shot_type_attempts.items()
    }
    intent_active_episodes = int(sum(int(v) for v in eval_diagnostics["intent_selection_counts"].values()))
    intent_inactive_episodes = int(eval_diagnostics.get("intent_inactive_count", 0))
    defense_intent_active_episodes = int(
        sum(int(v) for v in eval_diagnostics["defense_intent_selection_counts"].values())
    )
    defense_intent_inactive_episodes = int(eval_diagnostics.get("defense_intent_inactive_count", 0))
    selector_diag = dict(eval_diagnostics.get("selector") or {})
    selector_applied_count = int(selector_diag.get("applied_count", 0) or 0)
    selector_used_count = int(selector_diag.get("used_count", 0) or 0)
    selector_uniform_count = int(selector_diag.get("uniform_count", 0) or 0)
    selector_start_raw_count = int(selector_diag.get("episode_start_raw_prob_count", 0) or 0)
    selector_start_raw_sums = list(selector_diag.get("episode_start_raw_prob_sums") or [])
    if selector_start_raw_count > 0 and selector_start_raw_sums:
        selector_start_mean_raw_probs = [
            float(value) / float(selector_start_raw_count)
            for value in selector_start_raw_sums
        ]
        selector_start_mean_raw_max_prob = float(
            selector_diag.get("episode_start_raw_max_prob_sum", 0.0) or 0.0
        ) / float(selector_start_raw_count)
    else:
        selector_start_mean_raw_probs = []
        selector_start_mean_raw_max_prob = 0.0
    selector_diag["episode_start_mean_raw_probs"] = selector_start_mean_raw_probs
    selector_diag["episode_start_mean_raw_max_prob"] = float(selector_start_mean_raw_max_prob)
    summary = {
        "backend": "jax",
        "mode": "native_compiled",
        "num_episodes": int(num_episodes),
        "eval_seed": int(eval_seed),
        "allow_dunks": bool(getattr(env, "allow_dunks", False)),
        "batch_size": int(batch_size),
        "horizon": int(horizon),
        "elapsed_sec": float(elapsed),
        "episodes_per_sec": float(int(num_episodes) / elapsed),
        "states_per_sec": float((int(num_episodes) * int(horizon)) / elapsed),
        "completed_episodes": int(np.sum(np.asarray(all_completed, dtype=np.int32))),
        "completion_rate": _mean(all_completed),
        "mean_steps": _mean(all_steps),
        "offense_reward_per_episode": _mean(all_offense_rewards),
        "defense_reward_per_episode": _mean(all_defense_rewards),
        "offense_points_per_episode": _mean(all_offense_points),
        "defense_points_per_episode": _mean(all_defense_points),
        "score_margin_per_episode": _mean(np.asarray(all_offense_points) - np.asarray(all_defense_points)),
        "pass_attempts_per_episode": _mean(all_pass_attempts),
        "completed_passes_per_episode": _mean(all_completed_passes),
        "assists_per_episode": _mean(all_assists),
        "turnovers_per_episode": _mean(all_turnovers),
        "rebound_attempts_per_episode": _mean(all_rebound_attempts),
        "offensive_rebounds_per_episode": _mean(all_offensive_rebounds),
        "defensive_rebounds_per_episode": _mean(all_defensive_rebounds),
        "total_offense_points": _sum(all_offense_points),
        "total_defense_points": _sum(all_defense_points),
        "total_rebound_attempts": int(_sum(all_rebound_attempts)),
        "total_offensive_rebounds": int(_sum(all_offensive_rebounds)),
        "total_defensive_rebounds": int(_sum(all_defensive_rebounds)),
        "offensive_rebounds_by_player": dict(
            (eval_diagnostics.get("rebounds") or {}).get("by_player_offensive", {}) or {}
        ),
        "total_shot_attempts": int(total_shot_attempts),
        "total_shot_dunk_attempts": int(shot_type_attempts["dunk"]),
        "total_shot_two_attempts": int(shot_type_attempts["two"]),
        "total_shot_three_attempts": int(shot_type_attempts["three"]),
        "shot_dunk_share": float(shot_type_shares["dunk"]),
        "shot_two_share": float(shot_type_shares["two"]),
        "shot_three_share": float(shot_type_shares["three"]),
        "intent_active_episodes": int(intent_active_episodes),
        "intent_inactive_episodes": int(intent_inactive_episodes),
        "intent_active_rate": float(intent_active_episodes / int(num_episodes)) if int(num_episodes) else 0.0,
        "defense_intent_active_episodes": int(defense_intent_active_episodes),
        "defense_intent_inactive_episodes": int(defense_intent_inactive_episodes),
        "defense_intent_active_rate": (
            float(defense_intent_active_episodes / int(num_episodes)) if int(num_episodes) else 0.0
        ),
        "selector_enabled": bool(selector_settings["enabled"]),
        "selector_spec_enabled": bool(selector_settings["spec_enabled"]),
        "selector_config_enabled": bool(selector_settings["config_enabled"]),
        "selector_config_mode": str(selector_settings["selector_mode"]),
        "selector_disabled_reason": selector_settings["disabled_reason"],
        "selector_selection_mode": str(selector_settings["mode_label"]),
        "selector_training_alpha": float(selector_settings["training_alpha"]),
        "selector_eval_alpha": float(selector_settings["eval_alpha"]),
        "selector_eps": float(selector_settings["eps"]),
        "selector_eval_sampling_eps": float(selector_settings["eval_sampling_eps"]),
        "selector_multiselect_enabled": bool(selector_settings["multiselect_enabled"]),
        "selector_min_play_steps": int(selector_settings["min_play_steps"]),
        "selector_applied_count": int(selector_applied_count),
        "selector_used_count": int(selector_used_count),
        "selector_uniform_count": int(selector_uniform_count),
        "selector_applied_rate_per_timestep": (
            float(selector_applied_count / max(1, int(num_episodes) * int(horizon)))
        ),
        "selector_used_rate_per_applied": (
            float(selector_used_count / selector_applied_count) if selector_applied_count else 0.0
        ),
        "selector_boundary_episode_start_count": int(
            selector_diag.get("boundary_episode_start_count", 0) or 0
        ),
        "selector_boundary_commitment_timeout_count": int(
            selector_diag.get("boundary_commitment_timeout_count", 0) or 0
        ),
        "selector_boundary_completed_pass_count": int(
            selector_diag.get("boundary_completed_pass_count", 0) or 0
        ),
        "selector_episode_start_raw_prob_count": int(selector_start_raw_count),
        "selector_episode_start_mean_raw_probs": selector_start_mean_raw_probs,
        "selector_episode_start_mean_raw_max_prob": float(selector_start_mean_raw_max_prob),
    }
    return {
        "results": results,
        "shot_accumulator": shot_accumulator,
        "rebound_accumulator": rebound_accumulator,
        "per_player_stats": per_player_stats,
        "per_intent_stats": per_intent_stats,
        "eval_diagnostics": {
            **eval_diagnostics,
            "jax_native_summary": summary,
        },
    }
