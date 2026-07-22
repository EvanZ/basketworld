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
    REBOUND_SKILL_SAMPLING_ONE_HIGH_PER_TEAM,
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
}

_JAX_STATIC_ONLY_ENV_DEFAULTS = {
    "enable_rebounds": False,
    "rebound_table_model_dir": "",
    "rebound_target_temperature": 1.0,
    "rebound_target_uniform_mix": 0.0,
    "rebound_winner_distance_weight": 1.0,
    "rebound_basket_position_weight": 0.0,
    "rebound_winner_temperature": 1.0,
    "rebound_skill_std": 0.0,
    "rebound_skill_sampling_mode": "gaussian",
    "rebound_skill_high": 1.0,
    "rebound_skill_low": -0.25,
    "rebound_skill_weight": 0.0,
    "rebound_contest_mode": "global_contest",
    "rebound_contest_radius": 1,
    "rebound_obs_top_n_targets": 0,
    "offensive_rebound_shot_clock_reset": 14,
    "rebound_terminal_reward_mode": "actual_points",
    "enable_rebound_reward_redistribution": False,
    "offensive_rebound_reward_advance": 0.4,
    "rebound_reward_once_per_possession": True,
}

_JAX_STATIC_ONLY_ENV_CASTS = {
    "enable_rebounds": "bool",
    "rebound_table_model_dir": "str",
    "rebound_target_temperature": "float",
    "rebound_target_uniform_mix": "float",
    "rebound_winner_distance_weight": "float",
    "rebound_basket_position_weight": "float",
    "rebound_winner_temperature": "float",
    "rebound_skill_std": "float",
    "rebound_skill_sampling_mode": "str",
    "rebound_skill_high": "float",
    "rebound_skill_low": "float",
    "rebound_skill_weight": "float",
    "rebound_contest_mode": "str",
    "rebound_contest_radius": "int",
    "rebound_obs_top_n_targets": "int",
    "offensive_rebound_shot_clock_reset": "int",
    "rebound_terminal_reward_mode": "str",
    "enable_rebound_reward_redistribution": "bool",
    "offensive_rebound_reward_advance": "float",
    "rebound_reward_once_per_possession": "bool",
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


def _infer_attention_observation_dims(
    current_dim: int,
    *,
    expected_dim: int,
    token_count: int,
    token_dim: int,
    global_dim: int,
) -> tuple[int, int] | None:
    if token_count <= 0:
        return None
    max_token_dim = max(1, (int(current_dim) - 1) // int(token_count))
    candidates: list[tuple[tuple[int, int, int], int, int]] = []
    for current_token_dim in range(1, max_token_dim + 1):
        current_global_dim = int(current_dim) - 1 - (int(token_count) * int(current_token_dim))
        if current_global_dim < 0:
            continue
        if current_dim >= expected_dim:
            if current_token_dim < token_dim or current_global_dim < global_dim:
                continue
            # Prefer interpreting extras as a small number of added globals, then
            # added per-token fields. This handles 10*18+8+1 -> 10*17+7+1.
            score = (current_global_dim - global_dim, current_token_dim - token_dim, -current_token_dim)
        else:
            if current_token_dim > token_dim or current_global_dim > global_dim:
                continue
            # When padding for older checkpoints, preserve as much current
            # structure as possible.
            score = (token_dim - current_token_dim, global_dim - current_global_dim, -current_token_dim)
        candidates.append((score, current_token_dim, current_global_dim))
    if not candidates:
        return None
    _score, current_token_dim, current_global_dim = min(candidates, key=lambda item: item[0])
    return int(current_token_dim), int(current_global_dim)


def _adapt_attention_observation_to_spec(flat_obs, spec, jnp):
    expected_dim = int(spec.flat_obs_dim)
    current_dim = int(flat_obs.shape[-1])
    token_count = int(spec.token_player_count)
    token_dim = int(spec.token_dim)
    global_dim = int(spec.global_dim)
    dims = _infer_attention_observation_dims(
        current_dim,
        expected_dim=expected_dim,
        token_count=token_count,
        token_dim=token_dim,
        global_dim=global_dim,
    )
    if dims is None:
        raise ValueError(f"Cannot adapt attention observation dim {current_dim} to checkpoint dim {expected_dim}.")
    current_token_dim, current_global_dim = dims
    if current_token_dim <= 0 or current_global_dim < 0:
        raise ValueError(f"Cannot adapt attention observation dim {current_dim} to checkpoint dim {expected_dim}.")

    players = flat_obs[:, : token_count * current_token_dim].reshape(
        flat_obs.shape[0], token_count, current_token_dim
    )
    if current_token_dim >= token_dim:
        players = players[:, :, :token_dim]
    else:
        pad = jnp.zeros((flat_obs.shape[0], token_count, token_dim - current_token_dim), dtype=flat_obs.dtype)
        players = jnp.concatenate([players, pad], axis=-1)

    global_start = token_count * current_token_dim
    globals_vec = flat_obs[:, global_start : global_start + current_global_dim]
    if current_global_dim >= global_dim:
        globals_vec = globals_vec[:, :global_dim]
    else:
        pad = jnp.zeros((flat_obs.shape[0], global_dim - current_global_dim), dtype=flat_obs.dtype)
        globals_vec = jnp.concatenate([globals_vec, pad], axis=-1)

    role_start = global_start + current_global_dim
    role_flag = flat_obs[:, role_start : role_start + 1]
    if role_flag.shape[-1] < 1:
        role_flag = jnp.zeros((flat_obs.shape[0], 1), dtype=flat_obs.dtype)

    return jnp.concatenate(
        [players.reshape(flat_obs.shape[0], token_count * token_dim), globals_vec, role_flag],
        axis=1,
    ).astype(jnp.float32)


def _adapt_policy_observation_to_spec(flat_obs, static, spec: ActorCriticSpec, jnp):
    """Pack observations to the checkpoint's saved spec.

    Newer runtime code may expose extra features. Older checkpoints unpack the
    flat tensor by saved token/global dimensions, so passing the longer tensor
    silently shifts globals/role flags. Trim/pad additive attention features by
    token/global group rather than blindly slicing the flat vector.
    """
    expected_dim = int(spec.flat_obs_dim)
    current_dim = int(flat_obs.shape[-1])
    if current_dim == expected_dim:
        return flat_obs

    if str(spec.model_type) == "attention":
        return _adapt_attention_observation_to_spec(flat_obs, spec, jnp)

    if current_dim < expected_dim:
        raise ValueError(f"Policy observation dim {current_dim} is smaller than checkpoint dim {expected_dim}.")

    # Flat obs legacy compatibility for additive per-player rebound features
    # inserted immediately before rebound globals, role flag, and offense skill deltas.
    n_players = int(static.role_encoding.shape[0])
    offense_count = int(static.offense_ids.shape[0])
    tail_dim = 4 + 1 + (3 * offense_count)
    extra_dim = current_dim - expected_dim
    if (
        n_players > 0
        and extra_dim > 0
        and extra_dim % n_players == 0
        and extra_dim <= (2 * n_players)
        and current_dim > (extra_dim + tail_dim)
    ):
        remove_start = current_dim - tail_dim - extra_dim
        return jnp.concatenate(
            [flat_obs[:, :remove_start], flat_obs[:, remove_start + extra_dim :]],
            axis=1,
        ).astype(jnp.float32)

    return flat_obs[:, :expected_dim].astype(jnp.float32)


def _sample_constrained_rebound_skills_for_eval(
    static,
    sampling: dict[str, Any],
    batch_size: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    if not isinstance(sampling, dict):
        return None
    if str(sampling.get("mode") or "constrained_gaussian") != "constrained_gaussian":
        return None
    n_players = int(static.role_encoding.shape[0])
    if n_players <= 0:
        return None
    offense_ids = np.asarray(static.offense_ids, dtype=np.int32).reshape(-1)
    defense_ids = np.asarray(static.defense_ids, dtype=np.int32).reshape(-1)
    std = max(1.0e-8, float(sampling.get("std", 1.0)))
    target_edge = float(sampling.get("target_edge", 0.0))
    tolerance = max(0.0, float(sampling.get("tolerance", 0.25)))
    max_attempts = max(1, int(sampling.get("max_attempts", 5000)))

    out = np.zeros((int(batch_size), n_players), dtype=np.float32)
    for row in range(int(batch_size)):
        best_values = None
        best_error = np.inf
        for _attempt in range(max_attempts):
            values = rng.normal(loc=0.0, scale=std, size=n_players).astype(np.float32)
            offense_sum = float(np.sum(values[offense_ids])) if offense_ids.size else 0.0
            defense_sum = float(np.sum(values[defense_ids])) if defense_ids.size else 0.0
            error = abs((offense_sum - defense_sum) - target_edge)
            if error < best_error:
                best_error = error
                best_values = values
            if error <= tolerance:
                best_values = values
                break
        if best_values is not None:
            out[row] = best_values
    return out


def _apply_native_custom_setup(
    static,
    state,
    custom_setup: dict | None,
    batch_size: int,
    jnp,
    rng_seed: int | None = None,
):
    if not custom_setup:
        return state
    updates: dict[str, Any] = {}
    if custom_setup.get("initial_positions") is not None:
        positions = np.asarray(custom_setup.get("initial_positions"), dtype=np.int32)
        positions = positions.reshape(int(static.role_encoding.shape[0]), 2)
        updates["positions"] = jnp.asarray(
            np.broadcast_to(positions[None, ...], (int(batch_size),) + positions.shape),
            dtype=jnp.int32,
        )
    if custom_setup.get("ball_holder") is not None:
        updates["ball_holder"] = jnp.full(
            (int(batch_size),),
            int(custom_setup.get("ball_holder")),
            dtype=jnp.int32,
        )
    if (custom_setup.get("shooting_mode") or "random") == "fixed" and custom_setup.get("offense_skills"):
        skills = custom_setup.get("offense_skills") or {}
        offense_ids = np.asarray(static.offense_ids, dtype=np.int32)
        for key, field_name in (("layup", "layup_pct"), ("three_pt", "three_pt_pct"), ("dunk", "dunk_pct")):
            values = np.asarray(skills.get(key), dtype=np.float32).reshape(-1)
            if values.size != offense_ids.size:
                continue
            current = np.asarray(getattr(state, field_name), dtype=np.float32).copy()
            current[:, offense_ids] = values[None, :]
            updates[field_name] = jnp.asarray(current, dtype=jnp.float32)
    if custom_setup.get("rebound_skills") is not None:
        rebound_values = np.asarray(custom_setup.get("rebound_skills"), dtype=np.float32).reshape(-1)
        if rebound_values.size == int(static.role_encoding.shape[0]):
            broadcast_values = np.broadcast_to(rebound_values[None, :], (int(batch_size), rebound_values.size))
            try:
                skill_sampling_mode_id = int(np.asarray(static.rebound_skill_sampling_mode).reshape(-1)[0])
            except Exception:
                skill_sampling_mode_id = 0
            if skill_sampling_mode_id == int(REBOUND_SKILL_SAMPLING_ONE_HIGH_PER_TEAM):
                specialist_values = (broadcast_values > 0.0).astype(np.float32)
            else:
                specialist_values = np.zeros_like(broadcast_values, dtype=np.float32)
            updates["rebound_skill"] = jnp.asarray(broadcast_values, dtype=jnp.float32)
            updates["rebound_skill_specialist"] = jnp.asarray(specialist_values, dtype=jnp.float32)
    elif custom_setup.get("rebound_skill_sampling") is not None:
        rng = np.random.default_rng(rng_seed)
        sampled_values = _sample_constrained_rebound_skills_for_eval(
            static,
            custom_setup.get("rebound_skill_sampling") or {},
            int(batch_size),
            rng,
        )
        if sampled_values is not None and sampled_values.shape == (int(batch_size), int(static.role_encoding.shape[0])):
            updates["rebound_skill"] = jnp.asarray(sampled_values, dtype=jnp.float32)
            updates["rebound_skill_specialist"] = jnp.zeros_like(updates["rebound_skill"], dtype=jnp.float32)
    return state._replace(**updates) if updates else state


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
                    continue
                if key == "rebound_contest_radius":
                    for old_key in ("rebound_contest_initial_radius",):
                        if old_key in source and source[old_key] not in (None, ""):
                            out[key] = _coerce_runtime_static_value(key, source[old_key])
                            break
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


def _task_reward_scale_for_eval(
    training_params: dict[str, Any] | None,
    payload: dict[str, Any],
) -> float:
    params: dict[str, Any] = {}
    params.update(dict(payload.get("env_config", {}) or {}))
    params.update(dict(payload.get("trainer_config", {}) or {}))
    params.update(dict(training_params or {}))
    update_index = int(payload.get("update_index", 0) or 0)

    def _optional_float(value):
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        try:
            return float(value)
        except Exception:
            return None

    start_raw = _optional_float(_param(params, "task_reward_scale_start", None))
    end_raw = _optional_float(_param(params, "task_reward_scale_end", None))
    if start_raw is None and end_raw is None:
        return 1.0
    start = 1.0 if start_raw is None else float(start_raw)
    end = start if end_raw is None else float(end_raw)
    warmup_updates = _coerce_int(
        _param(params, "task_reward_scale_warmup_updates", -1),
        -1,
    )
    ramp_updates = _coerce_int(
        _param(params, "task_reward_scale_ramp_updates", -1),
        -1,
    )
    if warmup_updates >= 0 or ramp_updates >= 0:
        position = int(update_index)
        warmup = max(0, warmup_updates)
        ramp = max(0, ramp_updates)
    else:
        kernel_batch_size = max(
            1,
            _coerce_int(_param(params, "kernel_batch_size", 1), 1),
        )
        rollout_horizon = max(
            1,
            _coerce_int(_param(params, "rollout_horizon", 1), 1),
        )
        position = max(0, int(update_index) - 1) * kernel_batch_size * rollout_horizon * 2
        warmup = max(
            0,
            _coerce_int(_param(params, "task_reward_scale_warmup_steps", 0), 0),
        )
        ramp = max(
            0,
            _coerce_int(_param(params, "task_reward_scale_ramp_steps", 1), 1),
        )
    if position < warmup:
        return float(start)
    if ramp <= 0:
        return float(end)
    progress = min(1.0, max(0.0, (position - warmup) / float(ramp)))
    return float(start + progress * (end - start))


def _phi_beta_for_eval(
    training_params: dict[str, Any] | None,
    payload: dict[str, Any],
    *,
    default: float,
) -> float:
    params: dict[str, Any] = {}
    params.update(dict(payload.get("env_config", {}) or {}))
    params.update(dict(payload.get("trainer_config", {}) or {}))
    params.update(dict(training_params or {}))
    enabled = _coerce_bool(_param(params, "enable_phi_shaping", default > 0.0), default > 0.0)
    if not enabled:
        return 0.0

    def _optional_float(value):
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        try:
            return float(value)
        except Exception:
            return None

    start_raw = _optional_float(_param(params, "phi_beta_start", None))
    end_raw = _optional_float(_param(params, "phi_beta_end", None))
    if start_raw is None and end_raw is None:
        return max(0.0, float(default))
    start = max(0.0, 0.0 if start_raw is None else float(start_raw))
    end = max(0.0, start if end_raw is None else float(end_raw))
    update_index = int(payload.get("update_index", 0) or 0)
    warmup = max(0, _coerce_int(_param(params, "phi_beta_warmup_updates", 0), 0))
    ramp = max(0, _coerce_int(_param(params, "phi_beta_ramp_updates", 1), 1))
    if update_index < warmup:
        return float(start)
    if ramp <= 0:
        return float(end)
    progress = min(1.0, max(0.0, (update_index - warmup) / float(ramp)))
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

        def _team_policy_step(params, flat_obs, action_mask, intent_context, key, deterministic: bool):
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
                actions = masked_out["deterministic_actions"]
            else:
                actions = jax.random.categorical(
                    key,
                    masked_out["masked_logits"],
                    axis=-1,
                ).astype(jnp.int32)
            return actions, forward_out["values"].astype(jnp.float32)

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
            selector_obs = _adapt_policy_observation_to_spec(selector_obs, static, spec, jnp)
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
            offense_obs = _adapt_policy_observation_to_spec(offense_obs, static, spec, jnp)
            defense_obs = build_policy_observation_batch_with_role_flag(
                static,
                policy_state,
                role_flag_defense,
                jnp,
                model_type=spec.model_type,
            )
            defense_obs = _adapt_policy_observation_to_spec(defense_obs, static, spec, jnp)
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
            offense_actions, offense_values = _team_policy_step(
                offense_params,
                offense_obs,
                offense_mask,
                offense_intent_context,
                offense_key,
                offense_deterministic,
            )
            defense_actions, defense_values = _team_policy_step(
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
                "next_ball_holder": env_out.state.ball_holder.astype(jnp.int32),
                "positions": env_out.state.positions.astype(jnp.int32),
                "done": env_out.done.astype(jnp.int8),
                "terminal_episode_steps": env_out.terminal_episode_steps.astype(jnp.int32),
                "active": (~policy_state.episode_ended.astype(jnp.bool_)).astype(jnp.int8),
                "offense_values": offense_values,
                "defense_values": defense_values,
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
                "steal_player": env_out.steal_player.astype(jnp.int32),
                "offensive_three_seconds": env_out.offensive_three_seconds.astype(jnp.int8),
                "defensive_lane_violation": env_out.defensive_lane_violation.astype(jnp.int8),
                "defensive_lane_violation_player": env_out.defensive_lane_violation_player.astype(jnp.int32),
                "rebound_attempt": env_out.rebound_attempt.astype(jnp.int8),
                "offensive_rebound": env_out.offensive_rebound.astype(jnp.int8),
                "defensive_rebound": env_out.defensive_rebound.astype(jnp.int8),
                "rebound_target_cell": env_out.rebound_target_cell.astype(jnp.int32),
                "rebound_winner": env_out.rebound_winner.astype(jnp.int32),
                "rebound_global_contest": env_out.rebound_global_contest.astype(jnp.int8),
                "rebound_skill": policy_state.rebound_skill.astype(jnp.float32),
                "rebound_skill_specialist": policy_state.rebound_skill_specialist.astype(jnp.float32),
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
        "rebound_global_contests": np.asarray(trace["rebound_global_contest"])[:, :take].sum(axis=0),
    }


def _empty_value_diagnostics() -> dict[str, float | int]:
    return {
        "sample_count": 0,
        "completed_episode_count": 0,
        "offense_value_sum": 0.0,
        "defense_value_sum": 0.0,
        "value_sum_sum": 0.0,
        "value_sum_abs_sum": 0.0,
        "offense_return_sum": 0.0,
        "defense_return_sum": 0.0,
        "return_sum_sum": 0.0,
        "return_sum_abs_sum": 0.0,
        "offense_error_sum": 0.0,
        "defense_error_sum": 0.0,
        "offense_abs_error_sum": 0.0,
        "defense_abs_error_sum": 0.0,
    }


def _merge_value_diagnostics(dest: dict[str, float | int], src: dict[str, float | int]) -> None:
    for key, value in src.items():
        if key.endswith("_count") or key == "sample_count":
            dest[key] = int(dest.get(key, 0) or 0) + int(value or 0)
        else:
            dest[key] = float(dest.get(key, 0.0) or 0.0) + float(value or 0.0)


def _trace_values_array(trace: dict[str, np.ndarray], key: str, take: int) -> np.ndarray:
    arr = np.asarray(trace[key])[:, :take].astype(np.float64)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr


def _discounted_returns(rewards: np.ndarray, done: np.ndarray, gamma: float) -> np.ndarray:
    rewards = np.asarray(rewards, dtype=np.float64)
    done = np.asarray(done, dtype=bool)
    returns = np.zeros_like(rewards, dtype=np.float64)
    carry = np.zeros((rewards.shape[1],), dtype=np.float64)
    for t in range(rewards.shape[0] - 1, -1, -1):
        carry = rewards[t] + float(gamma) * carry * (~done[t]).astype(np.float64)
        returns[t] = carry
    return returns


def _post_orb_continuation_diagnostics_from_trace(
    trace: dict[str, np.ndarray],
    *,
    env_index: int,
    gamma: float,
) -> dict[str, float | int]:
    result: dict[str, float | int] = {
        "post_orb_samples": 0,
        "post_orb_points_sum": 0.0,
        "post_orb_value_samples": 0,
        "post_orb_consensus_value_sum": 0.0,
        "post_orb_offense_value_sum": 0.0,
        "post_orb_defense_value_sum": 0.0,
        "post_orb_shaped_return_samples": 0,
        "post_orb_consensus_shaped_return_sum": 0.0,
        "post_orb_offense_shaped_return_sum": 0.0,
        "post_orb_defense_shaped_return_sum": 0.0,
    }
    active = np.asarray(trace["active"])[:, env_index].astype(bool)
    terminal_steps = np.asarray(trace["terminal_episode_steps"])[:, env_index]
    if not bool(np.any(terminal_steps > 0)):
        return result

    offensive_rebounds = np.asarray(trace["offensive_rebound"])[:, env_index].astype(bool)
    orb_indices = np.flatnonzero(offensive_rebounds & active)
    value_indices = orb_indices + 1
    valid = value_indices < active.shape[0]
    if np.any(valid):
        clipped_indices = np.minimum(value_indices, active.shape[0] - 1)
        valid &= active[clipped_indices]
    orb_indices = orb_indices[valid]
    value_indices = value_indices[valid]
    if value_indices.size == 0:
        return result

    def _scalar_trace(key: str) -> np.ndarray:
        values = np.asarray(trace[key])[:, env_index].astype(np.float64)
        if values.ndim == 2 and values.shape[-1] == 1:
            values = values[:, 0]
        return values

    scored_points = (
        _scalar_trace("shot_value")
        * _scalar_trace("shot_success").astype(bool).astype(np.float64)
    )
    points_sum = float(
        sum(float(scored_points[int(orb_t) + 1 :].sum()) for orb_t in orb_indices)
    )
    offense_values = _scalar_trace("offense_values")[value_indices]
    defense_values = _scalar_trace("defense_values")[value_indices]
    consensus_values = 0.5 * (offense_values - defense_values)

    done = np.asarray(trace["done"])[:, env_index].astype(bool)[:, None]
    offense_reward_key = (
        "offense_training_rewards"
        if "offense_training_rewards" in trace
        else "offense_rewards"
    )
    defense_reward_key = (
        "defense_training_rewards"
        if "defense_training_rewards" in trace
        else "defense_rewards"
    )
    offense_rewards = _scalar_trace(offense_reward_key)[:, None]
    defense_rewards = _scalar_trace(defense_reward_key)[:, None]
    offense_returns = _discounted_returns(offense_rewards, done, gamma)[:, 0][value_indices]
    defense_returns = _discounted_returns(defense_rewards, done, gamma)[:, 0][value_indices]
    consensus_returns = 0.5 * (offense_returns - defense_returns)
    sample_count = int(value_indices.size)
    result.update(
        {
            "post_orb_samples": sample_count,
            "post_orb_points_sum": points_sum,
            "post_orb_value_samples": sample_count,
            "post_orb_consensus_value_sum": float(consensus_values.sum()),
            "post_orb_offense_value_sum": float(offense_values.sum()),
            "post_orb_defense_value_sum": float(defense_values.sum()),
            "post_orb_shaped_return_samples": sample_count,
            "post_orb_consensus_shaped_return_sum": float(consensus_returns.sum()),
            "post_orb_offense_shaped_return_sum": float(offense_returns.sum()),
            "post_orb_defense_shaped_return_sum": float(defense_returns.sum()),
        }
    )
    return result


def _value_diagnostics_from_trace(
    trace: dict[str, np.ndarray],
    *,
    take: int,
    gamma: float,
) -> dict[str, float | int]:
    active = np.asarray(trace["active"])[:, :take].astype(bool)
    done = np.asarray(trace["done"])[:, :take].astype(bool)
    completed = (np.max(np.asarray(trace["terminal_episode_steps"])[:, :take], axis=0) > 0)
    mask = active & completed.reshape((1, -1))
    if not np.any(mask):
        return _empty_value_diagnostics()

    offense_values = _trace_values_array(trace, "offense_values", take)
    defense_values = _trace_values_array(trace, "defense_values", take)
    offense_rewards = np.asarray(
        trace.get("offense_training_rewards", trace["offense_rewards"])
    )[:, :take].astype(np.float64) * active
    defense_rewards = np.asarray(
        trace.get("defense_training_rewards", trace["defense_rewards"])
    )[:, :take].astype(np.float64) * active
    offense_returns = _discounted_returns(offense_rewards, done, gamma)
    defense_returns = _discounted_returns(defense_rewards, done, gamma)

    offense_error = offense_values - offense_returns
    defense_error = defense_values - defense_returns
    value_sum = offense_values + defense_values
    return_sum = offense_returns + defense_returns
    return {
        "sample_count": int(mask.sum()),
        "completed_episode_count": int(completed.sum()),
        "offense_value_sum": float(offense_values[mask].sum()),
        "defense_value_sum": float(defense_values[mask].sum()),
        "value_sum_sum": float(value_sum[mask].sum()),
        "value_sum_abs_sum": float(np.abs(value_sum[mask]).sum()),
        "offense_return_sum": float(offense_returns[mask].sum()),
        "defense_return_sum": float(defense_returns[mask].sum()),
        "return_sum_sum": float(return_sum[mask].sum()),
        "return_sum_abs_sum": float(np.abs(return_sum[mask]).sum()),
        "offense_error_sum": float(offense_error[mask].sum()),
        "defense_error_sum": float(defense_error[mask].sum()),
        "offense_abs_error_sum": float(np.abs(offense_error[mask]).sum()),
        "defense_abs_error_sum": float(np.abs(defense_error[mask]).sum()),
    }


def _finalize_value_diagnostics(accum: dict[str, float | int], *, gamma: float) -> dict[str, float | int]:
    count = int(accum.get("sample_count", 0) or 0)

    def mean(key: str) -> float:
        return float(accum.get(key, 0.0) or 0.0) / float(count) if count > 0 else 0.0

    return {
        "discount_gamma": float(gamma),
        "sample_count": count,
        "completed_episode_count": int(accum.get("completed_episode_count", 0) or 0),
        "offense_value_mean": mean("offense_value_sum"),
        "defense_value_mean": mean("defense_value_sum"),
        "value_sum_mean": mean("value_sum_sum"),
        "value_sum_abs_mean": mean("value_sum_abs_sum"),
        "offense_return_mean": mean("offense_return_sum"),
        "defense_return_mean": mean("defense_return_sum"),
        "return_sum_mean": mean("return_sum_sum"),
        "return_sum_abs_mean": mean("return_sum_abs_sum"),
        "offense_value_bias_mean": mean("offense_error_sum"),
        "defense_value_bias_mean": mean("defense_error_sum"),
        "offense_value_mae": mean("offense_abs_error_sum"),
        "defense_value_mae": mean("defense_abs_error_sum"),
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
            "steals": 0,
            "points": 0.0,
            "offensive_rebounds": 0,
            "defensive_rebounds": 0,
            "rebound_chances": 0,
            "rebound_target_distance_sum": 0.0,
            "rebound_target_distance_count": 0,
            "rebound_win_probability_sum": 0.0,
            "rebound_win_probability_count": 0,
            "rebound_skill_sum": 0.0,
            "rebound_skill_count": 0,
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
        "steals": 0,
        "points": 0.0,
        "offensive_rebounds": 0,
        "defensive_rebounds": 0,
        "rebound_chances": 0,
        "rebound_target_distance_sum": 0.0,
        "rebound_target_distance_count": 0,
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
        "initial_ball_holder_counts": {},
        "pass_attempts_by_passer": {},
        "completed_passes_by_passer": {},
        "pass_attempts_by_receiver": {},
        "completed_passes_by_receiver": {},
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
            "by_player_defensive": {},
            "target_distance_sum_offense": 0.0,
            "target_distance_sum_defense": 0.0,
            "target_distance_count": 0,
            "post_orb_samples": 0,
            "post_orb_points_sum": 0.0,
            "post_orb_value_samples": 0,
            "post_orb_consensus_value_sum": 0.0,
            "post_orb_offense_value_sum": 0.0,
            "post_orb_defense_value_sum": 0.0,
            "post_orb_shaped_return_samples": 0,
            "post_orb_consensus_shaped_return_sum": 0.0,
            "post_orb_offense_shaped_return_sum": 0.0,
            "post_orb_defense_shaped_return_sum": 0.0,
        },
    }


def _merge_aggregate_stats(dest: dict[str, Any] | None, src: dict[str, Any] | None) -> dict[str, Any]:
    if dest is None:
        dest = _init_aggregate_stats()
    if not src:
        return dest

    for key in ("shots", "makes", "assists", "potential_assists", "turnovers", "steals", "episodes", "steps", "offensive_rebounds", "defensive_rebounds", "rebound_chances", "rebound_target_distance_count"):
        dest[key] = int(dest.get(key, 0) or 0) + int(src.get(key, 0) or 0)
    dest["points"] = float(dest.get("points", 0.0) or 0.0) + float(src.get("points", 0.0) or 0.0)
    dest["rebound_target_distance_sum"] = float(
        dest.get("rebound_target_distance_sum", 0.0) or 0.0
    ) + float(src.get("rebound_target_distance_sum", 0.0) or 0.0)

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


def _hex_distance_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    arr = np.asarray(a, dtype=np.int32)
    target = np.asarray(b, dtype=np.int32)
    dq = arr[..., 0] - int(target[0])
    dr = arr[..., 1] - int(target[1])
    return np.maximum(np.maximum(np.abs(dq), np.abs(dr)), np.abs(dq + dr)).astype(np.float32)


def _mean_team_distance_to_target_np(
    positions: np.ndarray,
    team_ids: list[int],
    target_coord: np.ndarray,
) -> float | None:
    valid_ids = [int(pid) for pid in team_ids if 0 <= int(pid) < int(positions.shape[0])]
    if not valid_ids:
        return None
    distances = _hex_distance_np(positions[np.asarray(valid_ids, dtype=np.int32)], target_coord)
    return float(np.mean(distances))


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
    eligible_mask: np.ndarray | None = None,
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
            "targets_by_player": {},
            "target_offensive_by_player": {},
            "rebounders_by_player": {},
            "rebounder_offensive_by_player": {},
            "target_chances_by_player": {},
            "rebounder_chances_by_player": {},
        },
    )
    bucket["total"] = int(bucket.get("total", 0)) + 1

    winner = int(winner_id)
    winner_valid = 0 <= winner < int(positions.shape[0])
    winner_player_key = str(winner)

    target_idx = int(target_cell)
    target_key = None
    if 0 <= target_idx < int(cell_coords.shape[0]):
        target_coord = cell_coords[target_idx]
        target_key = f"{int(target_coord[0])},{int(target_coord[1])}"
        targets = bucket.setdefault("targets", {})
        targets[target_key] = int(targets.get(target_key, 0)) + 1
        if bool(offensive):
            target_offensive = bucket.setdefault("target_offensive", {})
            target_offensive[target_key] = int(target_offensive.get(target_key, 0)) + 1
        if winner_valid:
            targets_by_player = bucket.setdefault("targets_by_player", {})
            player_targets = targets_by_player.setdefault(winner_player_key, {})
            player_targets[target_key] = int(player_targets.get(target_key, 0)) + 1
            if bool(offensive):
                target_offensive_by_player = bucket.setdefault("target_offensive_by_player", {})
                player_target_offensive = target_offensive_by_player.setdefault(winner_player_key, {})
                player_target_offensive[target_key] = int(player_target_offensive.get(target_key, 0)) + 1

    if eligible_mask is not None:
        eligible_arr = np.asarray(eligible_mask, dtype=bool).reshape(-1)
        max_players = min(int(positions.shape[0]), int(eligible_arr.shape[0]))
        target_chances_by_player = bucket.setdefault("target_chances_by_player", {})
        rebounder_chances_by_player = bucket.setdefault("rebounder_chances_by_player", {})
        for player_id in range(max_players):
            if not bool(eligible_arr[player_id]):
                continue
            player_key = str(int(player_id))
            if target_key is not None:
                player_target_chances = target_chances_by_player.setdefault(player_key, {})
                player_target_chances[target_key] = int(player_target_chances.get(target_key, 0)) + 1
            player_pos = positions[player_id]
            player_pos_key = f"{int(player_pos[0])},{int(player_pos[1])}"
            player_rebounder_chances = rebounder_chances_by_player.setdefault(player_key, {})
            player_rebounder_chances[player_pos_key] = int(player_rebounder_chances.get(player_pos_key, 0)) + 1

    if winner_valid:
        winner_pos = positions[winner]
        winner_key = f"{int(winner_pos[0])},{int(winner_pos[1])}"
        rebounders = bucket.setdefault("rebounders", {})
        rebounders[winner_key] = int(rebounders.get(winner_key, 0)) + 1
        rebounders_by_player = bucket.setdefault("rebounders_by_player", {})
        player_rebounders = rebounders_by_player.setdefault(winner_player_key, {})
        player_rebounders[winner_key] = int(player_rebounders.get(winner_key, 0)) + 1
        if bool(offensive):
            rebounder_offensive = bucket.setdefault("rebounder_offensive", {})
            rebounder_offensive[winner_key] = int(rebounder_offensive.get(winner_key, 0)) + 1
            rebounder_offensive_by_player = bucket.setdefault("rebounder_offensive_by_player", {})
            player_rebounder_offensive = rebounder_offensive_by_player.setdefault(winner_player_key, {})
            player_rebounder_offensive[winner_key] = int(player_rebounder_offensive.get(winner_key, 0)) + 1


def _increment_position_count(counts: dict[str, int], positions: np.ndarray, player_id: int) -> None:
    pid = int(player_id)
    if pid < 0 or pid >= int(positions.shape[0]):
        return
    pos = positions[pid]
    if np.asarray(pos).shape[0] < 2:
        return
    key = f"{int(pos[0])},{int(pos[1])}"
    counts[key] = int(counts.get(key, 0)) + 1


def _record_positioning_heatmap_event(
    *,
    positioning_accumulator: dict[str, Any],
    shot_q: int,
    shot_r: int,
    shooter_id: int,
    positions: np.ndarray,
    offense_ids: list[int],
    defense_ids: list[int],
) -> None:
    source_key = f"{int(shot_q)},{int(shot_r)}"
    bucket = positioning_accumulator.setdefault(
        source_key,
        {
            "total": 0,
            "offense": {},
            "offense_non_shooter": {},
            "defense": {},
            "shooter": {},
        },
    )
    bucket["total"] = int(bucket.get("total", 0)) + 1
    offense_counts = bucket.setdefault("offense", {})
    offense_non_shooter_counts = bucket.setdefault("offense_non_shooter", {})
    defense_counts = bucket.setdefault("defense", {})
    shooter_counts = bucket.setdefault("shooter", {})
    shooter = int(shooter_id)
    for raw_pid in offense_ids:
        pid = int(raw_pid)
        _increment_position_count(offense_counts, positions, pid)
        if pid == shooter:
            _increment_position_count(shooter_counts, positions, pid)
        else:
            _increment_position_count(offense_non_shooter_counts, positions, pid)
    for raw_pid in defense_ids:
        _increment_position_count(defense_counts, positions, int(raw_pid))


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
    passer_key = str(int(passer_id))
    receiver_key = str(int(receiver_id))
    attempts_by_passer = eval_diagnostics.setdefault("pass_attempts_by_passer", {})
    attempts_by_passer[passer_key] = int(attempts_by_passer.get(passer_key, 0)) + 1
    attempts_by_receiver = eval_diagnostics.setdefault("pass_attempts_by_receiver", {})
    attempts_by_receiver[receiver_key] = int(attempts_by_receiver.get(receiver_key, 0)) + 1
    link_key = f"{int(passer_id)}->{int(receiver_id)}"
    pass_links = eval_diagnostics.setdefault("pass_links", {})
    pass_links[link_key] = int(pass_links.get(link_key, 0)) + 1
    if completed:
        completed_by_passer = eval_diagnostics.setdefault("completed_passes_by_passer", {})
        completed_by_passer[passer_key] = int(completed_by_passer.get(passer_key, 0)) + 1
        completed_by_receiver = eval_diagnostics.setdefault("completed_passes_by_receiver", {})
        completed_by_receiver[receiver_key] = int(completed_by_receiver.get(receiver_key, 0)) + 1
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
    custom_setup: dict | None = None,
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
    base_phi_beta = float(
        np.asarray(jax.device_get(static.phi_beta), dtype=np.float32).reshape(-1)[0]
    )
    eval_phi_beta = _phi_beta_for_eval(
        training_params,
        unified_payload,
        default=base_phi_beta,
    )
    static = static._replace(phi_beta=jnp.asarray(eval_phi_beta, dtype=jnp.float32))
    trainer_config = dict(unified_payload.get("trainer_config", {}) or {})
    scheduled_task_reward_scale = _task_reward_scale_for_eval(
        training_params,
        unified_payload,
    )
    static_task_reward_scale = float(
        np.asarray(jax.device_get(static.task_reward_scale), dtype=np.float32).reshape(-1)[0]
    )
    eval_task_reward_scale = float(
        static_task_reward_scale * scheduled_task_reward_scale
    )
    horizon = _native_eval_horizon(env, training_params, unified_payload)
    value_discount_gamma = max(
        0.0,
        _coerce_float(_param(training_params or {}, "gamma", trainer_config.get("gamma", 0.99)), 0.99),
    )
    configured_batch_size = int(trainer_config.get("kernel_batch_size", 4096))
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
    positioning_accumulator: dict[str, Any] = {}
    cell_coords_np = np.asarray(jax.device_get(static.cell_coords), dtype=np.int32)
    basket_position_np = np.asarray(jax.device_get(static.basket_position), dtype=np.int32)
    eval_rebound_contest_mode = str(
        getattr(env, "rebound_contest_mode", "global_contest") or "global_contest"
    ).strip().lower().replace("-", "_").replace(" ", "_")
    if eval_rebound_contest_mode not in {"local_contest", "local"}:
        eval_rebound_contest_mode = "global_contest"
    else:
        eval_rebound_contest_mode = "local_contest"
    eval_rebound_contest_radius = max(0, int(getattr(env, "rebound_contest_radius", 1)))

    def _static_float_for_eval(name: str, default: float) -> float:
        try:
            arr = np.asarray(jax.device_get(getattr(static, name)))
            if arr.size == 0:
                return float(default)
            return float(arr.reshape(-1)[0])
        except Exception:
            return float(default)

    eval_rebound_winner_distance_weight = max(
        0.0,
        _static_float_for_eval("rebound_winner_distance_weight", 1.0),
    )
    eval_rebound_basket_position_weight = max(
        0.0,
        _static_float_for_eval("rebound_basket_position_weight", 0.0),
    )
    eval_rebound_winner_temperature = max(
        1.0e-6,
        _static_float_for_eval("rebound_winner_temperature", 1.0),
    )
    eval_rebound_skill_weight = max(
        0.0,
        _static_float_for_eval("rebound_skill_weight", 0.0),
    )
    offense_ids_np = np.asarray(offense_ids, dtype=np.int32)
    defense_ids_np = np.asarray(defense_ids, dtype=np.int32)
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
    all_rebound_global_contests: list[float] = []
    value_diagnostics_accum = _empty_value_diagnostics()

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
        reset_seed_parts = np.asarray(jax.device_get(reset_key), dtype=np.uint32).reshape(-1)
        custom_setup_seed = int(reset_seed_parts[0]) ^ (int(reset_seed_parts[-1]) << 1)
        initial_state = _apply_native_custom_setup(
            static,
            initial_state,
            custom_setup,
            batch_size,
            jnp,
            rng_seed=custom_setup_seed,
        )
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
        trace = dict(jax.device_get(trace_device))
        trace["offense_training_rewards"] = (
            np.asarray(trace["offense_rewards"], dtype=np.float32)
            * np.float32(eval_task_reward_scale)
        )
        trace["defense_training_rewards"] = (
            np.asarray(trace["defense_rewards"], dtype=np.float32)
            * np.float32(eval_task_reward_scale)
        )
        stats = _episode_stats_from_trace(trace, take=take, horizon=horizon)
        _merge_value_diagnostics(
            value_diagnostics_accum,
            _value_diagnostics_from_trace(trace, take=take, gamma=value_discount_gamma),
        )
        for idx in range(take):
            episode_num = completed_episodes + idx + 1
            step_count = int(stats["steps"][idx])
            active_steps = max(0, min(step_count, int(horizon)))
            offense_reward = float(stats["offense_rewards"][idx])
            defense_reward = float(stats["defense_rewards"][idx])
            user_reward = offense_reward if user_team == Team.OFFENSE else defense_reward
            initial_holder = int(trace["ball_holder"][0, idx]) if int(horizon) > 0 else -1
            if initial_holder >= 0:
                initial_counts = eval_diagnostics.setdefault("initial_ball_holder_counts", {})
                initial_key = str(initial_holder)
                initial_counts[initial_key] = int(initial_counts.get(initial_key, 0)) + 1
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
                    _record_positioning_heatmap_event(
                        positioning_accumulator=positioning_accumulator,
                        shot_q=q,
                        shot_r=r,
                        shooter_id=shooter_id,
                        positions=np.asarray(trace["positions"][t, idx], dtype=np.int32),
                        offense_ids=offense_ids,
                        defense_ids=defense_ids,
                    )
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
                    if turnover_reason_code == int(TURNOVER_REASON_INTERCEPTED):
                        steal_player = int(trace.get("steal_player", trace["next_ball_holder"])[t, idx])
                        if steal_player < 0:
                            steal_player = int(trace["next_ball_holder"][t, idx])
                        if steal_player >= 0:
                            if steal_player in per_player_stats:
                                per_player_stats[steal_player]["steals"] += 1
                            if steal_player in episode_player_stats:
                                episode_player_stats[steal_player]["steals"] += 1
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
                        {
                            "attempts": 0,
                            "offensive": 0,
                            "defensive": 0,
                            "by_player_offensive": {},
                            "by_player_defensive": {},
                            "target_distance_sum_offense": 0.0,
                            "target_distance_sum_defense": 0.0,
                            "target_distance_count": 0,
                            "post_orb_samples": 0,
                            "post_orb_points_sum": 0.0,
                            "post_orb_value_samples": 0,
                            "post_orb_consensus_value_sum": 0.0,
                            "post_orb_offense_value_sum": 0.0,
                            "post_orb_defense_value_sum": 0.0,
                            "post_orb_shaped_return_samples": 0,
                            "post_orb_consensus_shaped_return_sum": 0.0,
                            "post_orb_offense_shaped_return_sum": 0.0,
                            "post_orb_defense_shaped_return_sum": 0.0,
                            "eligibility": {
                                "attempts": 0,
                                "local_attempts": 0,
                                "global_attempts": 0,
                                "fallback_global_attempts": 0,
                                "local_mixed_attempts": 0,
                                "local_offense_only_attempts": 0,
                                "local_defense_only_attempts": 0,
                                "eligible_players_sum": 0,
                                "eligible_offense_sum": 0,
                                "eligible_defense_sum": 0,
                                "eligible_skill_sum": 0.0,
                                "eligible_offense_skill_sum": 0.0,
                                "eligible_defense_skill_sum": 0.0,
                                "eligible_offense_target_logit_sum": 0.0,
                                "eligible_defense_target_logit_sum": 0.0,
                                "eligible_offense_basket_logit_sum": 0.0,
                                "eligible_defense_basket_logit_sum": 0.0,
                                "eligible_offense_skill_logit_sum": 0.0,
                                "eligible_defense_skill_logit_sum": 0.0,
                                "eligible_offense_total_logit_sum": 0.0,
                                "eligible_defense_total_logit_sum": 0.0,
                                "winner_prob_attempts": 0,
                                "offense_winner_prob_sum": 0.0,
                                "defense_winner_prob_sum": 0.0,
                                "offensive_rebounds": 0,
                                "defensive_rebounds": 0,
                            },
                        },
                    )
                    rebound_diag["attempts"] = int(rebound_diag.get("attempts", 0)) + 1
                    rebound_diag["offensive"] = int(rebound_diag.get("offensive", 0)) + int(rebound_offensive)
                    rebound_diag["defensive"] = int(rebound_diag.get("defensive", 0)) + int(rebound_defensive)
                    rebound_positions = np.asarray(trace["positions"][t, idx], dtype=np.int32)
                    offense_target_distance = None
                    defense_target_distance = None
                    player_target_distances = None
                    eligible_mask_for_chances = None
                    winner_probabilities_for_chances = None
                    rebound_skills_for_chances = None
                    target_idx = int(rebound_target_cell)
                    if 0 <= target_idx < int(cell_coords_np.shape[0]):
                        target_coord = cell_coords_np[target_idx]
                        player_target_distances = _hex_distance_np(rebound_positions, target_coord)
                        offense_target_distance = _mean_team_distance_to_target_np(
                            rebound_positions,
                            offense_ids,
                            target_coord,
                        )
                        defense_target_distance = _mean_team_distance_to_target_np(
                            rebound_positions,
                            defense_ids,
                            target_coord,
                        )
                    if offense_target_distance is not None and defense_target_distance is not None:
                        rebound_diag["target_distance_sum_offense"] = float(
                            rebound_diag.get("target_distance_sum_offense", 0.0) or 0.0
                        ) + float(offense_target_distance)
                        rebound_diag["target_distance_sum_defense"] = float(
                            rebound_diag.get("target_distance_sum_defense", 0.0) or 0.0
                        ) + float(defense_target_distance)
                        rebound_diag["target_distance_count"] = int(rebound_diag.get("target_distance_count", 0) or 0) + 1
                    if player_target_distances is not None:
                        eligibility_diag = rebound_diag.setdefault("eligibility", {})
                        eligibility_diag["attempts"] = int(eligibility_diag.get("attempts", 0) or 0) + 1
                        local_candidate_mask = player_target_distances <= float(eval_rebound_contest_radius)
                        local_found = bool(np.any(local_candidate_mask))
                        use_local_contest = eval_rebound_contest_mode == "local_contest" and local_found
                        if use_local_contest:
                            eligible_mask = np.asarray(local_candidate_mask, dtype=bool)
                            eligibility_diag["local_attempts"] = int(eligibility_diag.get("local_attempts", 0) or 0) + 1
                        else:
                            eligible_mask = np.ones((n_players,), dtype=bool)
                            eligibility_diag["global_attempts"] = int(eligibility_diag.get("global_attempts", 0) or 0) + 1
                            if eval_rebound_contest_mode == "local_contest":
                                eligibility_diag["fallback_global_attempts"] = int(
                                    eligibility_diag.get("fallback_global_attempts", 0) or 0
                                ) + 1
                        offense_eligible = int(np.sum(eligible_mask[offense_ids_np])) if offense_ids_np.size else 0
                        defense_eligible = int(np.sum(eligible_mask[defense_ids_np])) if defense_ids_np.size else 0
                        rebound_skills = np.asarray(trace["rebound_skill"][t, idx], dtype=np.float32)
                        rebound_skills_for_chances = rebound_skills
                        eligible_skill_sum = float(np.sum(rebound_skills[eligible_mask])) if rebound_skills.size else 0.0
                        eligible_offense_skill_sum = (
                            float(np.sum(rebound_skills[offense_ids_np][eligible_mask[offense_ids_np]]))
                            if rebound_skills.size and offense_ids_np.size
                            else 0.0
                        )
                        eligible_defense_skill_sum = (
                            float(np.sum(rebound_skills[defense_ids_np][eligible_mask[defense_ids_np]]))
                            if rebound_skills.size and defense_ids_np.size
                            else 0.0
                        )
                        eligibility_diag["eligible_players_sum"] = int(
                            eligibility_diag.get("eligible_players_sum", 0) or 0
                        ) + int(np.sum(eligible_mask))
                        eligibility_diag["eligible_offense_sum"] = int(
                            eligibility_diag.get("eligible_offense_sum", 0) or 0
                        ) + offense_eligible
                        eligibility_diag["eligible_defense_sum"] = int(
                            eligibility_diag.get("eligible_defense_sum", 0) or 0
                        ) + defense_eligible
                        eligibility_diag["eligible_skill_sum"] = float(
                            eligibility_diag.get("eligible_skill_sum", 0.0) or 0.0
                        ) + eligible_skill_sum
                        eligibility_diag["eligible_offense_skill_sum"] = float(
                            eligibility_diag.get("eligible_offense_skill_sum", 0.0) or 0.0
                        ) + eligible_offense_skill_sum
                        eligibility_diag["eligible_defense_skill_sum"] = float(
                            eligibility_diag.get("eligible_defense_skill_sum", 0.0) or 0.0
                        ) + eligible_defense_skill_sum
                        eligibility_diag["offensive_rebounds"] = int(
                            eligibility_diag.get("offensive_rebounds", 0) or 0
                        ) + int(rebound_offensive)
                        eligibility_diag["defensive_rebounds"] = int(
                            eligibility_diag.get("defensive_rebounds", 0) or 0
                        ) + int(rebound_defensive)
                        eligible_mask_for_chances = eligible_mask
                        if rebound_skills.size and np.any(eligible_mask):
                            player_basket_distances = _hex_distance_np(
                                rebound_positions,
                                basket_position_np,
                            ).astype(np.float32)
                            target_basket_distance = float(
                                _hex_distance_np(
                                    np.asarray(target_coord, dtype=np.int32).reshape(1, 2),
                                    basket_position_np,
                                )[0]
                            )
                            basket_position_penalties = np.maximum(
                                0.0,
                                player_basket_distances - target_basket_distance,
                            ).astype(np.float32)
                            target_logit_terms = (
                                -eval_rebound_winner_distance_weight * player_target_distances.astype(np.float32)
                            ) / eval_rebound_winner_temperature
                            basket_logit_terms = (
                                -eval_rebound_basket_position_weight * basket_position_penalties
                            ) / eval_rebound_winner_temperature
                            skill_logit_terms = (
                                eval_rebound_winner_distance_weight
                                * eval_rebound_skill_weight
                                * rebound_skills.astype(np.float32)
                            ) / eval_rebound_winner_temperature
                            logits = target_logit_terms + basket_logit_terms + skill_logit_terms
                            offense_eligible_mask = np.zeros((n_players,), dtype=bool)
                            defense_eligible_mask = np.zeros((n_players,), dtype=bool)
                            if offense_ids_np.size:
                                offense_eligible_mask[offense_ids_np] = eligible_mask[offense_ids_np]
                            if defense_ids_np.size:
                                defense_eligible_mask[defense_ids_np] = eligible_mask[defense_ids_np]
                            for role_name, role_mask in (
                                ("offense", offense_eligible_mask),
                                ("defense", defense_eligible_mask),
                            ):
                                if not np.any(role_mask):
                                    continue
                                eligibility_diag[f"eligible_{role_name}_target_logit_sum"] = float(
                                    eligibility_diag.get(f"eligible_{role_name}_target_logit_sum", 0.0) or 0.0
                                ) + float(np.sum(target_logit_terms[role_mask]))
                                eligibility_diag[f"eligible_{role_name}_basket_logit_sum"] = float(
                                    eligibility_diag.get(f"eligible_{role_name}_basket_logit_sum", 0.0) or 0.0
                                ) + float(np.sum(basket_logit_terms[role_mask]))
                                eligibility_diag[f"eligible_{role_name}_skill_logit_sum"] = float(
                                    eligibility_diag.get(f"eligible_{role_name}_skill_logit_sum", 0.0) or 0.0
                                ) + float(np.sum(skill_logit_terms[role_mask]))
                                eligibility_diag[f"eligible_{role_name}_total_logit_sum"] = float(
                                    eligibility_diag.get(f"eligible_{role_name}_total_logit_sum", 0.0) or 0.0
                                ) + float(np.sum(logits[role_mask]))
                            eligible_logits = logits[eligible_mask]
                            eligible_logits = eligible_logits - float(np.max(eligible_logits))
                            eligible_exp = np.exp(eligible_logits)
                            denom = float(np.sum(eligible_exp))
                            winner_probs = np.zeros((n_players,), dtype=np.float32)
                            if denom > 0.0 and np.isfinite(denom):
                                winner_probs[eligible_mask] = eligible_exp / denom
                                valid_offense_ids = offense_ids_np[
                                    (offense_ids_np >= 0) & (offense_ids_np < int(winner_probs.shape[0]))
                                ]
                                valid_defense_ids = defense_ids_np[
                                    (defense_ids_np >= 0) & (defense_ids_np < int(winner_probs.shape[0]))
                                ]
                                eligibility_diag["winner_prob_attempts"] = int(
                                    eligibility_diag.get("winner_prob_attempts", 0) or 0
                                ) + 1
                                eligibility_diag["offense_winner_prob_sum"] = float(
                                    eligibility_diag.get("offense_winner_prob_sum", 0.0) or 0.0
                                ) + float(np.sum(winner_probs[valid_offense_ids]))
                                eligibility_diag["defense_winner_prob_sum"] = float(
                                    eligibility_diag.get("defense_winner_prob_sum", 0.0) or 0.0
                                ) + float(np.sum(winner_probs[valid_defense_ids]))
                            winner_probabilities_for_chances = winner_probs
                        if use_local_contest:
                            if offense_eligible > 0 and defense_eligible > 0:
                                eligibility_diag["local_mixed_attempts"] = int(
                                    eligibility_diag.get("local_mixed_attempts", 0) or 0
                                ) + 1
                            elif offense_eligible > 0:
                                eligibility_diag["local_offense_only_attempts"] = int(
                                    eligibility_diag.get("local_offense_only_attempts", 0) or 0
                                ) + 1
                            elif defense_eligible > 0:
                                eligibility_diag["local_defense_only_attempts"] = int(
                                    eligibility_diag.get("local_defense_only_attempts", 0) or 0
                                ) + 1
                    for stats_target in (per_player_stats, episode_player_stats):
                        for pid in range(n_players):
                            if pid in stats_target:
                                if (
                                    eligible_mask_for_chances is not None
                                    and pid < int(eligible_mask_for_chances.shape[0])
                                    and bool(eligible_mask_for_chances[pid])
                                ):
                                    stats_target[pid]["rebound_chances"] = int(stats_target[pid].get("rebound_chances", 0)) + 1
                                    if (
                                        winner_probabilities_for_chances is not None
                                        and pid < int(winner_probabilities_for_chances.shape[0])
                                    ):
                                        stats_target[pid]["rebound_win_probability_sum"] = float(
                                            stats_target[pid].get("rebound_win_probability_sum", 0.0) or 0.0
                                        ) + float(winner_probabilities_for_chances[pid])
                                        stats_target[pid]["rebound_win_probability_count"] = int(
                                            stats_target[pid].get("rebound_win_probability_count", 0) or 0
                                        ) + 1
                                    if (
                                        rebound_skills_for_chances is not None
                                        and pid < int(rebound_skills_for_chances.shape[0])
                                    ):
                                        stats_target[pid]["rebound_skill_sum"] = float(
                                            stats_target[pid].get("rebound_skill_sum", 0.0) or 0.0
                                        ) + float(rebound_skills_for_chances[pid])
                                        stats_target[pid]["rebound_skill_count"] = int(
                                            stats_target[pid].get("rebound_skill_count", 0) or 0
                                        ) + 1
                                if (
                                    player_target_distances is not None
                                    and eligible_mask_for_chances is not None
                                    and pid < int(player_target_distances.shape[0])
                                    and pid < int(eligible_mask_for_chances.shape[0])
                                    and bool(eligible_mask_for_chances[pid])
                                ):
                                    stats_target[pid]["rebound_target_distance_sum"] = float(
                                        stats_target[pid].get("rebound_target_distance_sum", 0.0) or 0.0
                                    ) + float(player_target_distances[pid])
                                    stats_target[pid]["rebound_target_distance_count"] = int(
                                        stats_target[pid].get("rebound_target_distance_count", 0) or 0
                                    ) + 1
                        if rebound_winner in stats_target:
                            if rebound_offensive:
                                stats_target[rebound_winner]["offensive_rebounds"] = int(
                                    stats_target[rebound_winner].get("offensive_rebounds", 0)
                                ) + 1
                            elif rebound_defensive:
                                stats_target[rebound_winner]["defensive_rebounds"] = int(
                                    stats_target[rebound_winner].get("defensive_rebounds", 0)
                                ) + 1
                    if rebound_winner >= 0:
                        bucket_name = "by_player_offensive" if rebound_offensive else "by_player_defensive"
                        by_player = rebound_diag.setdefault(bucket_name, {})
                        rebound_key = str(int(rebound_winner))
                        by_player[rebound_key] = int(by_player.get(rebound_key, 0)) + 1
                    _record_rebound_heatmap_event(
                        rebound_accumulator=rebound_accumulator,
                        shot_q=int(trace["shot_q"][t, idx]),
                        shot_r=int(trace["shot_r"][t, idx]),
                        target_cell=rebound_target_cell,
                        winner_id=rebound_winner,
                        offensive=rebound_offensive,
                        positions=rebound_positions,
                        cell_coords=cell_coords_np,
                        eligible_mask=eligible_mask_for_chances,
                    )
                    rebounds_payload.append(
                        {
                            "attempt": True,
                            "offensive": rebound_offensive,
                            "defensive": rebound_defensive,
                            "winner": rebound_winner if rebound_winner >= 0 else None,
                            "winner_team": "OFFENSE" if rebound_offensive else ("DEFENSE" if rebound_defensive else None),
                            "target_cell_index": rebound_target_cell if rebound_target_cell >= 0 else None,
                            "offense_avg_distance_to_target": offense_target_distance,
                            "defense_avg_distance_to_target": defense_target_distance,
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
            offensive_rebound_flags = np.asarray(trace["offensive_rebound"])[:active_steps, idx].astype(bool)
            potential_flags = np.asarray(trace["potential_assist"])[:active_steps, idx].astype(bool)
            assist_flags = np.asarray(trace["assists"])[:active_steps, idx].astype(bool)
            post_orb_diagnostics = _post_orb_continuation_diagnostics_from_trace(
                trace,
                env_index=idx,
                gamma=value_discount_gamma,
            )
            if int(post_orb_diagnostics["post_orb_samples"]) > 0:
                rebound_diag = eval_diagnostics.setdefault("rebounds", {})
                for key_name, value in post_orb_diagnostics.items():
                    rebound_diag[key_name] = (
                        rebound_diag.get(key_name, 0) or 0
                    ) + value
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
        all_rebound_global_contests.extend([float(v) for v in stats["rebound_global_contests"].tolist()])
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
    rebound_diag_final = eval_diagnostics.get("rebounds") or {}
    rebound_eligibility = dict(rebound_diag_final.get("eligibility", {}) or {})
    rebound_target_distance_count = int(rebound_diag_final.get("target_distance_count", 0) or 0)
    post_orb_samples = int(rebound_diag_final.get("post_orb_samples", 0) or 0)
    post_orb_points_sum = float(rebound_diag_final.get("post_orb_points_sum", 0.0) or 0.0)
    post_orb_value_samples = int(rebound_diag_final.get("post_orb_value_samples", 0) or 0)
    post_orb_consensus_value_sum = float(
        rebound_diag_final.get("post_orb_consensus_value_sum", 0.0) or 0.0
    )
    post_orb_offense_value_sum = float(rebound_diag_final.get("post_orb_offense_value_sum", 0.0) or 0.0)
    post_orb_defense_value_sum = float(rebound_diag_final.get("post_orb_defense_value_sum", 0.0) or 0.0)
    post_orb_shaped_return_samples = int(
        rebound_diag_final.get("post_orb_shaped_return_samples", 0) or 0
    )
    post_orb_consensus_shaped_return_sum = float(
        rebound_diag_final.get("post_orb_consensus_shaped_return_sum", 0.0) or 0.0
    )
    post_orb_offense_shaped_return_sum = float(
        rebound_diag_final.get("post_orb_offense_shaped_return_sum", 0.0) or 0.0
    )
    post_orb_defense_shaped_return_sum = float(
        rebound_diag_final.get("post_orb_defense_shaped_return_sum", 0.0) or 0.0
    )
    total_rebound_attempts = _sum(all_rebound_attempts)
    total_rebound_global_contests = _sum(all_rebound_global_contests)
    value_diagnostics = _finalize_value_diagnostics(
        value_diagnostics_accum,
        gamma=value_discount_gamma,
    )
    value_diagnostics["task_reward_scale"] = float(eval_task_reward_scale)
    value_diagnostics["phi_beta"] = float(eval_phi_beta)
    value_diagnostics["includes_training_intent_bonus"] = False

    def _static_scalar(name: str, default: Any) -> Any:
        try:
            arr = np.asarray(jax.device_get(getattr(static, name)))
            if arr.size == 0:
                return default
            return arr.reshape(-1)[0].item()
        except Exception:
            return default

    resolved_rebound_contest_mode_id = int(_static_scalar("rebound_contest_mode", 0))
    resolved_rebound_skill_sampling_mode_id = int(_static_scalar("rebound_skill_sampling_mode", 0))
    custom_rebound_skill_values: list[float] = []
    if custom_setup and custom_setup.get("rebound_skills") is not None:
        try:
            custom_rebound_skill_values = [
                float(v)
                for v in np.asarray(custom_setup.get("rebound_skills"), dtype=np.float32).reshape(-1).tolist()
            ]
        except Exception:
            custom_rebound_skill_values = []
    custom_rebound_skill_sampling = (custom_setup or {}).get("rebound_skill_sampling") if custom_setup else None
    if not isinstance(custom_rebound_skill_sampling, dict):
        custom_rebound_skill_sampling = {}
    static_offense_ids = np.asarray(jax.device_get(static.offense_ids), dtype=np.int32).reshape(-1)
    static_defense_ids = np.asarray(jax.device_get(static.defense_ids), dtype=np.int32).reshape(-1)
    static_role_encoding = np.asarray(jax.device_get(static.role_encoding), dtype=np.float32).reshape(-1)
    resolved_rebound_params = {
        "enable_rebounds": bool(int(_static_scalar("enable_rebounds", 0))),
        "offense_player_count": int(static_offense_ids.size),
        "defense_player_count": int(static_defense_ids.size),
        "positive_role_count": int((static_role_encoding > 0.0).sum()),
        "negative_role_count": int((static_role_encoding < 0.0).sum()),
        "rebound_target_temperature": float(_static_scalar("rebound_target_temperature", 1.0)),
        "rebound_target_uniform_mix": float(_static_scalar("rebound_target_uniform_mix", 0.0)),
        "rebound_winner_distance_weight": float(_static_scalar("rebound_winner_distance_weight", 1.0)),
        "rebound_basket_position_weight": float(_static_scalar("rebound_basket_position_weight", 0.0)),
        "rebound_winner_temperature": float(_static_scalar("rebound_winner_temperature", 1.0)),
        "rebound_skill_std": float(_static_scalar("rebound_skill_std", 0.0)),
        "rebound_skill_sampling_mode": (
            "one_high_per_team"
            if resolved_rebound_skill_sampling_mode_id == REBOUND_SKILL_SAMPLING_ONE_HIGH_PER_TEAM
            else "gaussian"
        ),
        "rebound_skill_sampling_mode_id": int(resolved_rebound_skill_sampling_mode_id),
        "rebound_skill_high": float(_static_scalar("rebound_skill_high", 1.0)),
        "rebound_skill_low": float(_static_scalar("rebound_skill_low", -0.25)),
        "rebound_skill_weight": float(_static_scalar("rebound_skill_weight", 0.0)),
        "custom_rebound_skills_applied": bool(custom_rebound_skill_values),
        "custom_rebound_skill_values": custom_rebound_skill_values,
        "custom_rebound_skill_positive_count": int(sum(1 for value in custom_rebound_skill_values if value > 0.0)),
        "custom_rebound_skill_sum": float(sum(custom_rebound_skill_values)) if custom_rebound_skill_values else 0.0,
        "custom_rebound_skill_sampling_applied": bool(custom_rebound_skill_sampling) and not bool(custom_rebound_skill_values),
        "custom_rebound_skill_sampling_mode": str(custom_rebound_skill_sampling.get("mode", "") or ""),
        "custom_rebound_skill_sampling_std": float(custom_rebound_skill_sampling.get("std", 0.0) or 0.0),
        "custom_rebound_skill_sampling_target_edge": float(custom_rebound_skill_sampling.get("target_edge", 0.0) or 0.0),
        "custom_rebound_skill_sampling_tolerance": float(custom_rebound_skill_sampling.get("tolerance", 0.0) or 0.0),
        "custom_rebound_skill_sampling_max_attempts": int(custom_rebound_skill_sampling.get("max_attempts", 0) or 0),
        "rebound_contest_mode": "local_contest" if resolved_rebound_contest_mode_id == 1 else "global_contest",
        "rebound_contest_mode_id": int(resolved_rebound_contest_mode_id),
        "rebound_contest_radius": int(_static_scalar("rebound_contest_radius", 1)),
    }
    print(
        "[Evaluation] Value diagnostics: "
        f"samples={int(value_diagnostics.get('sample_count', 0) or 0)} "
        f"episodes={int(value_diagnostics.get('completed_episode_count', 0) or 0)} "
        f"Vo={float(value_diagnostics.get('offense_value_mean', 0.0) or 0.0):.3f} "
        f"Vd={float(value_diagnostics.get('defense_value_mean', 0.0) or 0.0):.3f} "
        f"Vo+Vd={float(value_diagnostics.get('value_sum_mean', 0.0) or 0.0):.3f} "
        f"Ro={float(value_diagnostics.get('offense_return_mean', 0.0) or 0.0):.3f} "
        f"Rd={float(value_diagnostics.get('defense_return_mean', 0.0) or 0.0):.3f}"
    )
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
        "total_rebound_attempts": int(total_rebound_attempts),
        "total_offensive_rebounds": int(_sum(all_offensive_rebounds)),
        "total_defensive_rebounds": int(_sum(all_defensive_rebounds)),
        "post_orb_sample_count": int(post_orb_samples),
        "post_orb_points_total": float(post_orb_points_sum),
        "post_orb_points_per_sample": (
            float(post_orb_points_sum / post_orb_samples) if post_orb_samples > 0 else 0.0
        ),
        "post_orb_value_sample_count": int(post_orb_value_samples),
        "post_orb_consensus_value_total": float(post_orb_consensus_value_sum),
        "post_orb_consensus_value_per_sample": (
            float(post_orb_consensus_value_sum / post_orb_value_samples)
            if post_orb_value_samples > 0
            else 0.0
        ),
        "post_orb_offense_value_per_sample": (
            float(post_orb_offense_value_sum / post_orb_value_samples)
            if post_orb_value_samples > 0
            else 0.0
        ),
        "post_orb_defense_value_per_sample": (
            float(post_orb_defense_value_sum / post_orb_value_samples)
            if post_orb_value_samples > 0
            else 0.0
        ),
        "post_orb_shaped_return_sample_count": int(post_orb_shaped_return_samples),
        "post_orb_consensus_shaped_return_total": float(
            post_orb_consensus_shaped_return_sum
        ),
        "post_orb_consensus_shaped_return_per_sample": (
            float(post_orb_consensus_shaped_return_sum / post_orb_shaped_return_samples)
            if post_orb_shaped_return_samples > 0
            else 0.0
        ),
        "post_orb_offense_shaped_return_per_sample": (
            float(post_orb_offense_shaped_return_sum / post_orb_shaped_return_samples)
            if post_orb_shaped_return_samples > 0
            else 0.0
        ),
        "post_orb_defense_shaped_return_per_sample": (
            float(post_orb_defense_shaped_return_sum / post_orb_shaped_return_samples)
            if post_orb_shaped_return_samples > 0
            else 0.0
        ),
        "post_orb_critic_minus_shaped_return_per_sample": (
            float(
                (post_orb_consensus_value_sum / post_orb_value_samples)
                - (post_orb_consensus_shaped_return_sum / post_orb_shaped_return_samples)
            )
            if post_orb_value_samples > 0 and post_orb_shaped_return_samples > 0
            else 0.0
        ),
        "post_orb_task_reward_scale": float(eval_task_reward_scale),
        "post_orb_phi_beta": float(eval_phi_beta),
        "post_orb_shaped_return_includes_training_intent_bonus": False,
        "total_rebound_global_contests": int(total_rebound_global_contests),
        "rebound_global_contest_rate": (
            float(total_rebound_global_contests / total_rebound_attempts) if total_rebound_attempts > 0 else 0.0
        ),
        "rebound_eligibility": rebound_eligibility,
        "resolved_rebound_params": resolved_rebound_params,
        "value_diagnostics": value_diagnostics,
        "rebound_target_distance_count": rebound_target_distance_count,
        "avg_offense_rebound_target_distance": (
            float(rebound_diag_final.get("target_distance_sum_offense", 0.0) or 0.0) / rebound_target_distance_count
            if rebound_target_distance_count > 0
            else 0.0
        ),
        "avg_defense_rebound_target_distance": (
            float(rebound_diag_final.get("target_distance_sum_defense", 0.0) or 0.0) / rebound_target_distance_count
            if rebound_target_distance_count > 0
            else 0.0
        ),
        "offensive_rebounds_by_player": dict(
            rebound_diag_final.get("by_player_offensive", {}) or {}
        ),
        "defensive_rebounds_by_player": dict(
            rebound_diag_final.get("by_player_defensive", {}) or {}
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
        "positioning_accumulator": positioning_accumulator,
        "per_player_stats": per_player_stats,
        "per_intent_stats": per_intent_stats,
        "eval_diagnostics": {
            **eval_diagnostics,
            "value_diagnostics": value_diagnostics,
            "jax_native_summary": summary,
        },
    }
