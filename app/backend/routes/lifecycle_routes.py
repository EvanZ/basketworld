import copy
import os
import tempfile
from pathlib import Path

import mlflow
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder

from basketworld.utils.mlflow_params import (
    get_mlflow_params,
    get_mlflow_phi_shaping_params,
    get_mlflow_start_template_library,
    get_mlflow_training_params,
)
from basketworld.utils.mlflow_config import setup_mlflow
from basketworld.domain import ActionType, Team
from basketworld.utils.start_templates import load_start_template_library, resolve_start_template

from app.backend.inference_adapters import (
    get_policy_backend_kind,
    get_policy_capabilities,
    get_policy_metadata,
    load_inference_policy,
)
from app.backend.observations import (
    validate_policy_observation_schema,
)
from app.backend.rebound_preview import compute_rebound_preview
from app.backend.policies import (
    _compute_param_counts_from_policy,
    get_unified_policy_path,
    list_policies_from_run,
)
from app.backend.env_access import env_view
from app.backend.schemas import (
    ActionRequest,
    InitGameRequest,
    ListPoliciesRequest,
    ReboundPreviewRequest,
    SetMultiPossessionLimitRequest,
    StartSelfPlayRequest,
    TemplateBootstrapRequest,
)
from app.backend.state import (
    _capture_turn_start_snapshot,
    _role_flag_value_for_team,
    game_state,
    get_full_game_state,
    get_ui_game_state,
)
from fastapi.encoders import jsonable_encoder


router = APIRouter()


_JAX_RUNTIME_STATIC_ENV_KEYS = {
    # Kernel and runtime fields retained in backend session metadata.
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
    "enable_multi_possession",
    "multi_possession_limit",
    "multi_possession_limit_start",
    "multi_possession_limit_end",
    "multi_possession_limit_ramp_updates",
    "multi_possession_overtime_round_cap",
    "multi_possession_reward_mode",
    "score_potential_scale",
    "game_winner_reward",
    "multi_possession_aux_rewards_enabled",
    "multi_possession_use_inbounds",
    "made_basket_restart_mode",
    "multi_possession_schema_version",
    "inbound_deadline_steps",
    "check_deadline_steps",
    "check_setup_steps",
    "check_clearance",
    "random_check_location",
}


_REBOUND_RUNTIME_STICKY_ENV_KEYS = {
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
    "enable_rebound_reward_redistribution",
    "offensive_rebound_reward_advance",
    "rebound_reward_once_per_possession",
}


def _split_jax_env_and_wrapper_params(optional_params: dict) -> tuple[dict, dict]:
    """Split JAX config from wrapper metadata without inspecting the legacy env."""
    from basketworld_jax.config import JaxEnvironmentConfig

    valid_env_keys = set(JaxEnvironmentConfig.__dataclass_fields__)
    valid_env_keys.update(
        {
            "court_height",
            "court_width",
            "players_per_side",
            "shot_clock_steps",
            "offensive_three_seconds",
            "phi_beta_start",
            "pass_oob_turnover_prob_start",
        }
    )
    valid_env_keys.update(_JAX_RUNTIME_STATIC_ENV_KEYS)
    env_kwargs = {key: value for key, value in optional_params.items() if key in valid_env_keys}
    wrapper_kwargs = {
        key: value for key, value in optional_params.items() if key not in valid_env_keys
    }
    return env_kwargs, wrapper_kwargs


def _str_to_bool(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


_JAX_MLFLOW_ENV_PARAM_CASTS = {
    "training_team": str,
    "players": int,
    "court_rows": int,
    "court_cols": int,
    "shot_clock": int,
    "min_shot_clock": int,
    "allow_dunks": _str_to_bool,
    "layup_pct": float,
    "three_pt_pct": float,
    "dunk_pct": float,
    "layup_std": float,
    "three_pt_std": float,
    "dunk_std": float,
    "three_point_distance": float,
    "three_point_short_distance": float,
    "three_pt_extra_hex_decay": float,
    "shot_pressure_enabled": _str_to_bool,
    "shot_pressure_max": float,
    "shot_pressure_lambda": float,
    "shot_pressure_arc_degrees": float,
    "defender_pressure_distance": int,
    "defender_pressure_turnover_chance": float,
    "defender_pressure_decay_lambda": float,
    "pressure_skill_mean": float,
    "pressure_skill_std": float,
    "shot_contest_skill_mean": float,
    "shot_contest_skill_std": float,
    "ball_handling_skill_mean": float,
    "ball_handling_skill_std": float,
    "pressure_skill_effect_weight": float,
    "shot_contest_skill_effect_weight": float,
    "base_steal_rate": float,
    "steal_perp_decay": float,
    "steal_distance_factor": float,
    "steal_position_weight_min": float,
    "pass_interception_model": str,
    "pass_passer_pressure_weight": float,
    "pass_receiver_pressure_weight": float,
    "pass_lob_lane_multiplier": float,
    "pass_lob_receiver_distance": float,
    "pass_speed": float,
    "defender_reaction_time": float,
    "defender_speed": float,
    "defender_reach_radius": float,
    "reaction_softness": float,
    "base_passer_risk": float,
    "passer_pressure_decay": float,
    "base_receiver_risk": float,
    "receiver_alignment_min": float,
    "receiver_alignment_width": float,
    "max_receiver_hazard": float,
    "lane_weight": float,
    "steal_skill_mean": float,
    "steal_skill_std": float,
    "passing_skill_mean": float,
    "passing_skill_std": float,
    "steal_skill_effect_weight": float,
    "spawn_distance": int,
    "max_spawn_distance": int,
    "defender_spawn_distance": int,
    "defender_guard_distance": int,
    "assist_window": int,
    "mask_occupied_moves": _str_to_bool,
    "enable_pass_gating": _str_to_bool,
    "pass_mode": str,
    "use_set_obs": _str_to_bool,
    "illegal_defense_enabled": _str_to_bool,
    "offensive_three_seconds": _str_to_bool,
    "three_second_lane_width": int,
    "three_second_lane_height": int,
    "three_second_max_steps": int,
    "violation_reward": float,
    "include_hoop_vector": _str_to_bool,
    "enable_phi_shaping": _str_to_bool,
    "reward_shaping_gamma": float,
    "phi_beta": float,
    "phi_beta_start": float,
    "phi_beta_end": float,
    "phi_beta_warmup_updates": int,
    "phi_beta_ramp_updates": int,
    "phi_blend_weight": float,
    "phi_aggregation_mode": str,
    "phi_use_ball_handler_only": _str_to_bool,
    "pass_reward": float,
    "potential_assist_pct": float,
    "full_assist_bonus_pct": float,
    "start_template_enabled": _str_to_bool,
    "start_template_library": str,
    "start_template_prob": float,
    "start_template_jitter_scale": float,
    "start_template_mirror_prob": float,
    "start_template_strict": _str_to_bool,
    "enable_intent_learning": _str_to_bool,
    "enable_defense_intent_learning": _str_to_bool,
    "num_intents": int,
    "intent_commitment_steps": int,
    "intent_null_prob": float,
    "defense_intent_null_prob": float,
    "intent_visible_to_defense_prob": float,
    "enable_rebounds": _str_to_bool,
    "rebound_table_model_dir": str,
    "rebound_target_temperature": float,
    "rebound_target_uniform_mix": float,
    "rebound_winner_distance_weight": float,
    "rebound_basket_position_weight": float,
    "rebound_winner_temperature": float,
    "rebound_skill_std": float,
    "rebound_skill_sampling_mode": str,
    "rebound_skill_high": float,
    "rebound_skill_low": float,
    "rebound_skill_weight": float,
    "rebound_contest_mode": str,
    "rebound_contest_radius": int,
    "rebound_obs_top_n_targets": int,
    "offensive_rebound_shot_clock_reset": int,
    "rebound_terminal_reward_mode": str,
    "enable_rebound_reward_redistribution": _str_to_bool,
    "offensive_rebound_reward_advance": float,
    "rebound_reward_once_per_possession": _str_to_bool,
    "enable_multi_possession": _str_to_bool,
    "multi_possession_limit": int,
    "multi_possession_limit_start": int,
    "multi_possession_limit_end": int,
    "multi_possession_limit_ramp_updates": int,
    "multi_possession_overtime_round_cap": int,
    "multi_possession_reward_mode": str,
    "score_potential_scale": float,
    "game_winner_reward": float,
    "multi_possession_aux_rewards_enabled": _str_to_bool,
    "multi_possession_use_inbounds": _str_to_bool,
    "made_basket_restart_mode": str,
    "multi_possession_schema_version": int,
    "inbound_deadline_steps": int,
    "check_deadline_steps": int,
    "check_setup_steps": int,
    "check_clearance": int,
    "random_check_location": _str_to_bool,
}


def _coerce_jax_mlflow_env_param(raw_value, cast):
    if raw_value is None:
        return None
    raw_str = str(raw_value).strip()
    if raw_str == "" or raw_str.lower() in {"none", "null"}:
        return None
    if cast is int:
        return int(float(raw_str))
    return cast(raw_value)


def _overlay_jax_mlflow_env_params(optional_params: dict, mlflow_params: dict) -> dict:
    merged = dict(optional_params or {})
    for key, cast in _JAX_MLFLOW_ENV_PARAM_CASTS.items():
        names = (f"jax/env/{key}",)
        if key == "offensive_three_seconds":
            names = (f"jax/env/{key}", "jax/env/offensive_three_seconds_enabled")
        elif key == "rebound_contest_radius":
            names = (
                "jax/env/rebound_contest_radius",
                "jax/rebound_contest_radius",
                "jax/env/rebound_contest_initial_radius",
                "jax/rebound_contest_initial_radius",
            )
        elif key in {
            "rebound_skill_std",
            "rebound_skill_sampling_mode",
            "rebound_skill_high",
            "rebound_skill_low",
            "rebound_skill_weight",
            "rebound_basket_position_weight",
            "rebound_contest_mode",
            "rebound_obs_top_n_targets",
            "rebound_terminal_reward_mode",
            "enable_rebound_reward_redistribution",
            "offensive_rebound_reward_advance",
            "rebound_reward_once_per_possession",
        }:
            names = (f"jax/env/{key}", f"jax/{key}")
        for name in names:
            if name not in mlflow_params:
                continue
            value = _coerce_jax_mlflow_env_param(mlflow_params.get(name), cast)
            if key == "training_team" and isinstance(value, str):
                merged[key] = Team.DEFENSE if value.strip().lower() == "defense" else Team.OFFENSE
            elif key == "offensive_three_seconds":
                merged["offensive_three_seconds_enabled"] = bool(value)
            else:
                merged[key] = value
            break
    return _normalize_jax_phi_runtime_env_params(merged)


def _normalize_jax_phi_runtime_env_params(
    config: dict,
    *,
    current_phi_beta: float | None = None,
) -> dict:
    """Map JAX phi schedule metadata into the env's runtime phi fields.

    The trainer persists phi schedule knobs (`phi_beta_start/end`) for
    reproducibility, but the live env/runtime expects the active field
    `phi_beta`. For checkpoints, prefer the checkpoint's recorded current beta;
    otherwise fall back to the target end beta and then start beta.
    """
    merged = dict(config or {})
    if "phi_beta" not in merged or merged.get("phi_beta") in ("", None):
        beta_value = current_phi_beta
        if beta_value is None:
            for beta_key in ("phi_beta_end", "phi_beta_start"):
                raw = merged.get(beta_key)
                if raw in ("", None):
                    continue
                try:
                    beta_value = float(raw)
                    break
                except Exception:
                    continue
        if beta_value is not None:
            merged["phi_beta"] = float(beta_value)
    return merged


def _coerce_jax_mlflow_training_value(raw_value):
    if raw_value is None:
        return None
    raw_str = str(raw_value).strip()
    if raw_str == "":
        return raw_value
    raw_lower = raw_str.lower()
    if raw_lower in {"none", "null"}:
        return None
    if raw_lower in {"true", "false"}:
        return raw_lower == "true"
    try:
        if all(token not in raw_str for token in (".", "e", "E")):
            return int(raw_str)
    except Exception:
        pass
    try:
        return float(raw_str)
    except Exception:
        return raw_value


def _jax_param_is_missing(value) -> bool:
    return value is None or value == ""


def _jax_setdefault_param(params: dict, key: str, value) -> None:
    if _jax_param_is_missing(value):
        return
    if key not in params or _jax_param_is_missing(params.get(key)):
        params[key] = value


def _jax_int_param(params: dict, key: str, default: int | None = None) -> int | None:
    value = params.get(key, default)
    if _jax_param_is_missing(value):
        return default
    try:
        return int(float(value))
    except Exception:
        return default


def _populate_jax_training_ui_aliases(training_params: dict) -> dict:
    """Populate UI aliases from JAX-native MLflow/trainer params."""
    merged = dict(training_params or {})

    kernel_batch_size = _jax_int_param(merged, "kernel_batch_size")
    rollout_horizon = _jax_int_param(merged, "rollout_horizon")
    ppo_minibatches = max(1, _jax_int_param(merged, "ppo_minibatches", 1) or 1)
    role_multiplier = 2 if str(merged.get("mode", "train_loop")).lower() == "train_loop" else 1
    steps_per_update = None
    if kernel_batch_size is not None and rollout_horizon is not None:
        steps_per_update = int(kernel_batch_size * rollout_horizon * role_multiplier)
        _jax_setdefault_param(merged, "num_envs", kernel_batch_size)
        _jax_setdefault_param(merged, "n_steps", rollout_horizon)
        _jax_setdefault_param(merged, "steps_per_update", steps_per_update)
        _jax_setdefault_param(merged, "ppo_batch_size", steps_per_update)
        minibatch_size = (
            steps_per_update // ppo_minibatches
            if steps_per_update % ppo_minibatches == 0
            else steps_per_update
        )
        _jax_setdefault_param(merged, "batch_size", minibatch_size)

    alias_pairs = {
        "n_epochs": "policy_update_epochs",
        "clip_range": "ppo_clip_range",
        "vf_coef": "value_coef",
        "ent_coef": "entropy_coef",
    }
    for ui_key, jax_key in alias_pairs.items():
        _jax_setdefault_param(merged, ui_key, merged.get(jax_key))

    has_model_config = any(
        not _jax_param_is_missing(merged.get(key))
        for key in (
            "policy_model",
            "model_type",
            "attention_embed_dim",
            "policy_hidden_dims",
        )
    )

    if "policy_class" not in merged and has_model_config:
        model = str(merged.get("policy_model", "") or merged.get("model_type", "") or "").strip()
        if not model:
            model = "attention" if merged.get("attention_embed_dim") else "mlp"
        merged["policy_class"] = f"JAX {model} actor-critic"

    if has_model_config:
        _jax_setdefault_param(merged, "use_dual_critic", bool(merged.get("use_dual_critic", False)))
    if "net_arch_used" not in merged and has_model_config:
        if merged.get("attention_embed_dim"):
            pi = merged.get("attention_pi_head_hidden_dims")
            vf = merged.get("attention_vf_head_hidden_dims")
            parts = [
                f"attention embed={merged.get('attention_embed_dim')}",
                f"heads={merged.get('attention_num_heads', 'N/A')}",
                f"token_mlp={merged.get('attention_token_mlp_dim', 'N/A')}",
            ]
            if pi:
                parts.append(f"pi=[{pi}]")
            if vf:
                parts.append(f"vf=[{vf}]")
            merged["net_arch_used"] = ", ".join(parts)
        elif merged.get("policy_hidden_dims"):
            merged["net_arch_used"] = str(merged.get("policy_hidden_dims"))

    if steps_per_update is not None:
        for prefix in (
            "intent_selector_alpha",
            "intent_selector_eps",
            "task_reward_scale",
            "intent_diversity",
        ):
            for phase in ("warmup", "ramp"):
                update_key = f"{prefix}_{phase}_updates"
                steps_key = f"{prefix}_{phase}_steps"
                updates = _jax_int_param(merged, update_key)
                if updates is None or updates < 0:
                    continue
                _jax_setdefault_param(merged, steps_key, int(updates * steps_per_update))
    return merged


def _overlay_jax_mlflow_training_params(training_params: dict, mlflow_params: dict) -> dict:
    """Expose JAX MLflow params as normal unprefixed training params for the UI/runtime."""
    merged = dict(training_params or {})
    for raw_key, raw_value in dict(mlflow_params or {}).items():
        if not str(raw_key).startswith("jax/"):
            continue
        key = str(raw_key)[len("jax/") :]
        if not key or key.startswith("env/"):
            continue
        merged[key] = _coerce_jax_mlflow_training_value(raw_value)
    for raw_key, raw_value in dict(mlflow_params or {}).items():
        if not str(raw_key).startswith("jax/env/"):
            continue
        key = str(raw_key)[len("jax/env/") :]
        if key in {
            "enable_phi_shaping",
            "reward_shaping_gamma",
            "phi_beta_start",
            "phi_beta_end",
            "phi_beta_warmup_updates",
            "phi_beta_ramp_updates",
            "phi_blend_weight",
            "phi_aggregation_mode",
            "phi_use_ball_handler_only",
            "start_template_enabled",
            "start_template_library",
            "start_template_prob",
            "start_template_jitter_scale",
            "start_template_mirror_prob",
            "start_template_strict",
        }:
            merged[key] = _coerce_jax_mlflow_training_value(raw_value)
    if bool(merged.get("intent_selector_enabled", False)):
        merged.setdefault("intent_selector_mode", "integrated")
    return _populate_jax_training_ui_aliases(merged)


def _load_start_template_library_for_run(
    mlflow_client: mlflow.tracking.MlflowClient, run_id: str
) -> dict | None:
    """Best-effort load of a persisted start-template library artifact."""
    try:
        return get_mlflow_start_template_library(mlflow_client, run_id)
    except Exception as exc:
        print(f"[start_templates] Failed to load template library for run {run_id}: {exc}")
        return None


def _load_start_template_library_from_training_params(
    mlflow_training_params: dict,
) -> tuple[dict | None, str | None]:
    """Best-effort fallback for local/dev runs when the MLflow artifact is absent."""
    source_path = str(mlflow_training_params.get("start_template_library") or "").strip()
    if not source_path:
        return None, None

    expanded = os.path.expandvars(os.path.expanduser(source_path))
    candidates = [Path(expanded)]
    if not candidates[0].is_absolute():
        repo_root = Path(basketworld.__file__).resolve().parents[1]
        candidates.extend([Path.cwd() / expanded, repo_root / expanded])

    players_per_side = int(mlflow_training_params.get("players") or 3)
    for candidate in candidates:
        try:
            if not candidate.exists():
                continue
            library = load_start_template_library(
                candidate,
                players_per_side=players_per_side,
            )
            return library, str(candidate)
        except Exception as exc:
            print(f"[start_templates] Failed to load local template library {candidate}: {exc}")
    return None, None


_SESSION_START_TEMPLATE_SOURCES = {"local_file", "file_upload", "session_editor"}


def _resolve_start_template_library_for_init(
    *,
    mlflow_client: mlflow.tracking.MlflowClient,
    run_id: str,
    mlflow_training_params: dict,
    previous_library: dict | None,
    previous_source: str | None,
    previous_path: str | None,
) -> tuple[dict | None, str | None, str | None]:
    """Prefer a run artifact, but keep manually loaded session templates across New Game."""
    run_library = _load_start_template_library_for_run(mlflow_client, run_id)
    if run_library is not None:
        return (
            run_library,
            "mlflow_artifact",
            str(
                mlflow_training_params.get(
                    "start_template_library_artifact_path",
                    "metadata/start_template_library.json",
                )
            ),
        )

    local_library, local_path = _load_start_template_library_from_training_params(
        mlflow_training_params
    )
    if local_library is not None:
        return local_library, "local_file", local_path

    source = str(previous_source or "").strip()
    if previous_library is not None and source in _SESSION_START_TEMPLATE_SOURCES:
        return copy.deepcopy(previous_library), source, previous_path
    return None, None, None


def _jax_local_env_config_from_metadata(policy_obj) -> tuple[dict, dict, dict, dict | None]:
    metadata = get_policy_metadata(policy_obj) or {}
    frozen_config = dict(metadata.get("frozen_config", {}) or {})
    env_config = dict(metadata.get("env_config", {}) or {})
    config = env_config or frozen_config
    if not config:
        raise HTTPException(
            status_code=400,
            detail="JAX checkpoint is missing env_config/frozen_config metadata.",
        )
    training_team = config.get("training_team")
    if isinstance(training_team, str):
        config["training_team"] = (
            Team.DEFENSE if training_team.strip().lower() == "defense" else Team.OFFENSE
        )
    if "offensive_three_seconds" in config and "offensive_three_seconds_enabled" not in config:
        config["offensive_three_seconds_enabled"] = bool(config.pop("offensive_three_seconds"))
    if "allow_dunks" not in config:
        # Historical JAX checkpoints were trained with the trainer default
        # allow_dunks=True but did not persist that key. Preserve JAX behavior
        # instead of falling back to the base env constructor default False.
        config["allow_dunks"] = True
    trainer_config = dict(metadata.get("trainer_config", {}) or {})
    policy_spec = dict(metadata.get("policy_spec", {}) or {})
    last_metrics = dict(metadata.get("last_metrics", {}) or {})
    current_phi_beta = None
    for beta_metric in ("phi_beta", "phi_beta_mean", "offense_phi_beta_mean"):
        if beta_metric not in last_metrics:
            continue
        try:
            current_phi_beta = float(last_metrics[beta_metric])
            break
        except Exception:
            continue
    config = _normalize_jax_phi_runtime_env_params(
        config,
        current_phi_beta=current_phi_beta,
    )
    if bool(policy_spec.get("intent_selector_enabled", False)):
        trainer_config.setdefault("intent_selector_enabled", True)
        trainer_config.setdefault("intent_selector_mode", "integrated")
        current_alpha = float(last_metrics.get("selector_alpha", 1.0) or 0.0)
        current_eps = float(last_metrics.get("selector_eps", 0.0) or 0.0)
        trainer_config.setdefault("intent_selector_alpha_start", current_alpha)
        trainer_config.setdefault("intent_selector_alpha_end", current_alpha)
        trainer_config.setdefault("intent_selector_alpha_warmup_steps", 0)
        trainer_config.setdefault("intent_selector_alpha_ramp_steps", 0)
        trainer_config.setdefault("intent_selector_eps_start", current_eps)
        trainer_config.setdefault("intent_selector_eps_end", current_eps)
        trainer_config.setdefault("intent_selector_eps_warmup_steps", 0)
        trainer_config.setdefault("intent_selector_eps_ramp_steps", 0)
        trainer_config.setdefault("intent_selector_value_coef", 0.5)
        trainer_config.setdefault("intent_selector_multiselect_enabled", True)
        trainer_config.setdefault("intent_selector_min_play_steps", 3)
    trainer_config = _populate_jax_training_ui_aliases(trainer_config)
    required_params: dict = {}
    optional_params = dict(config)
    phi_params = {
        "enable_phi_shaping": bool(config.get("enable_phi_shaping", False)),
        "phi_beta": float(config.get("phi_beta", 0.0) or 0.0),
        "reward_shaping_gamma": float(config.get("reward_shaping_gamma", 1.0) or 1.0),
        "phi_aggregation_mode": str(config.get("phi_aggregation_mode", "team_best")),
        "phi_use_ball_handler_only": bool(config.get("phi_use_ball_handler_only", False)),
        "phi_blend_weight": float(config.get("phi_blend_weight", 0.0) or 0.0),
    }
    return required_params, optional_params, trainer_config, phi_params


def _run_game_mode_config(client, run_id: str) -> dict:
    """Read the lightweight game-format settings logged with a training run."""
    try:
        params = dict(client.get_run(run_id).data.params or {})
    except Exception as exc:
        print(f"[list_policies] Failed to read run params for {run_id}: {exc}")
        params = {}
    enable_multi_possession = _str_to_bool(params.get(
            "jax/env/enable_multi_possession",
            params.get("enable_multi_possession", False),
        ))
    episode_end_mode = str(params.get(
        "jax/env/episode_end_mode",
        params.get("episode_end_mode", "possessions"),
    ) or "possessions").strip().lower()
    if episode_end_mode not in {"possessions", "game_clock"}:
        episode_end_mode = "possessions"
    def _positive_int(name: str, default: int) -> int:
        try:
            return max(1, int(params.get(f"jax/env/{name}", params.get(name, default))))
        except (TypeError, ValueError):
            return default
    return {
        "enable_multi_possession": enable_multi_possession,
        "episode_end_mode": episode_end_mode,
        "game_clock_seconds": _positive_int("game_clock_seconds", 300),
        "overtime_seconds": _positive_int("overtime_seconds", 120),
    }


def _run_uses_multi_possession(client, run_id: str) -> bool:
    return bool(_run_game_mode_config(client, run_id)["enable_multi_possession"])


def _validate_requested_episode_end_mode(
    checkpoint_env_config: dict,
    requested_mode: str | None,
) -> None:
    if requested_mode is None:
        return
    checkpoint_mode = str(
        checkpoint_env_config.get("episode_end_mode", "possessions") or "possessions"
    ).strip().lower()
    if requested_mode != checkpoint_mode:
        raise HTTPException(
            status_code=400,
            detail=(
                "Checkpoint episode mode mismatch: the selected "
                f"{requested_mode!r} mode cannot load a {checkpoint_mode!r} policy. "
                "Choose a checkpoint trained for the selected mode."
            ),
        )


@router.post("/api/list_policies")
def list_policies(request: ListPoliciesRequest):
    """Return available policy filenames and lightweight run setup metadata."""
    try:
        try:
            setup_mlflow(verbose=False)
        except Exception as setup_err:
            print(f"[list_policies] MLflow setup warning: {setup_err}")

        client = mlflow.tracking.MlflowClient()
        mode_config = _run_game_mode_config(client, request.run_id)
        try:
            unified_paths = list_policies_from_run(client, request.run_id)
        except Exception as e:
            print(f"[list_policies] Failed to list artifacts for run {request.run_id}: {e}")
            return {"unified": [], **mode_config}
        if not unified_paths:
            return {"unified": [], **mode_config}
        return {
            "unified": [os.path.basename(p) for p in unified_paths],
            **mode_config,
        }
    except Exception as e:
        import traceback

        print(f"Error listing policies: {e}")
        traceback.print_exc()
        return {"unified": [], "enable_multi_possession": False}


@router.post("/api/init_game")
async def init_game(request: InitGameRequest):
    """Initializes a new game from an MLflow run."""
    global game_state

    from basketworld.utils.mlflow_config import setup_mlflow

    try:
        setup_mlflow()
    except Exception as e:
        print(f"Warning: MLflow setup failed: {e}")

    mlflow_client = mlflow.tracking.MlflowClient()

    run_name = request.run_name
    run_id = str(request.run_id or "").strip()
    unified_policy_name = request.unified_policy_name
    opponent_unified_policy_name = request.opponent_unified_policy_name
    if not run_id:
        raise HTTPException(status_code=400, detail="init_game requires run_id.")

    previous_run_id = getattr(game_state, "run_id", None)
    previous_unified_policy_key = getattr(game_state, "unified_policy_key", None)
    previous_opponent_policy_key = getattr(game_state, "opponent_unified_policy_key", None)
    previous_env_optional_params = copy.deepcopy(getattr(game_state, "env_optional_params", None) or {})
    previous_runtime_env = getattr(getattr(game_state, "jax_runtime", None), "environment", None)
    if previous_runtime_env is None:
        previous_runtime_env = getattr(game_state, "env", None)

    previous_start_template_library = copy.deepcopy(
        getattr(game_state, "mlflow_start_template_library", None)
    )
    previous_start_template_source = getattr(game_state, "start_template_library_source", None)
    previous_start_template_path = getattr(game_state, "start_template_library_path", None)

    with tempfile.TemporaryDirectory() as tmpdir:
        required_params: dict
        optional_params: dict
        mlflow_phi_params: dict
        mlflow_training_params: dict
        start_template_library: dict | None
        start_template_library_source: str | None = None
        start_template_library_path: str | None = None

        try:
            unified_path = get_unified_policy_path(mlflow_client, run_id, unified_policy_name)
        except Exception as e:
            msg = str(e)
            if "RESOURCE_DOES_NOT_EXIST" in msg:
                raise HTTPException(status_code=404, detail=f"Unified policy not found for run {run_id}")
            raise HTTPException(status_code=500, detail=f"Failed to download unified policy: {e}")
        opponent_unified_path = None
        if opponent_unified_policy_name:
            try:
                opponent_unified_path = get_unified_policy_path(mlflow_client, run_id, opponent_unified_policy_name)
            except Exception as e:
                msg = str(e)
                if "RESOURCE_DOES_NOT_EXIST" in msg:
                    raise HTTPException(status_code=404, detail=f"Opponent policy not found for run {run_id}")
                raise HTTPException(status_code=500, detail=f"Failed to download opponent policy: {e}")

        game_state.unified_policy = load_inference_policy(unified_path, device="cpu")
        game_state.offense_policy = None
        game_state.defense_policy = (
            load_inference_policy(opponent_unified_path, device="cpu")
            if opponent_unified_path
            else None
        )

        (
            required_params,
            optional_params,
            mlflow_training_params,
            mlflow_phi_params,
        ) = _jax_local_env_config_from_metadata(game_state.unified_policy)
        run_params = {}
        try:
            run_params = dict(mlflow_client.get_run(run_id).data.params or {})
            mlflow_training_params = _overlay_jax_mlflow_training_params(
                mlflow_training_params, run_params
            )
            mlflow_phi_params = get_mlflow_phi_shaping_params(mlflow_client, run_id)
        except Exception as exc:
            print(f"[init_game] Warning: failed to apply JAX MLflow params: {exc}")
        optional_params = _overlay_jax_mlflow_env_params(optional_params, run_params)
        (
            start_template_library,
            start_template_library_source,
            start_template_library_path,
        ) = _resolve_start_template_library_for_init(
            mlflow_client=mlflow_client,
            run_id=run_id,
            mlflow_training_params=mlflow_training_params,
            previous_library=previous_start_template_library,
            previous_source=previous_start_template_source,
            previous_path=previous_start_template_path,
        )
        if start_template_library is not None:
            optional_params["start_template_library"] = start_template_library
        else:
            optional_params.pop("start_template_library", None)

        # Extract role_flag encoding for backward compatibility (not passed to env)
        game_state.role_flag_offense = optional_params.pop("role_flag_offense_value", 1.0)
        game_state.role_flag_defense = optional_params.pop("role_flag_defense_value", -1.0)
        optional_params.pop("role_flag_encoding_version", None)

        # Apply request overrides for optional parameters
        _validate_requested_episode_end_mode(optional_params, request.episode_end_mode)
        if request.spawn_distance is not None:
            optional_params["spawn_distance"] = request.spawn_distance
        if request.defender_spawn_distance is not None:
            optional_params["defender_spawn_distance"] = request.defender_spawn_distance
        if request.allow_dunks is not None:
            optional_params["allow_dunks"] = request.allow_dunks
        if request.dunk_pct is not None:
            optional_params["dunk_pct"] = request.dunk_pct
        if request.multi_possession_limit is not None:
            # This is a runtime episode-length choice. It does not alter the
            # loaded policy; it only controls how many completed possessions
            # the dev game plays before ending.
            optional_params["multi_possession_limit"] = request.multi_possession_limit
        if request.episode_end_mode is not None:
            optional_params["episode_end_mode"] = request.episode_end_mode
        if request.game_clock_seconds is not None:
            optional_params["game_clock_seconds"] = request.game_clock_seconds
        if request.overtime_seconds is not None:
            optional_params["overtime_seconds"] = request.overtime_seconds

        mlflow_default_env_optional_params = copy.deepcopy(optional_params)

        target_unified_policy_key = os.path.basename(unified_path)
        target_opponent_policy_key = os.path.basename(opponent_unified_path) if opponent_unified_path else None
        same_loaded_policy = (
            str(previous_run_id or "") == run_id
            and previous_unified_policy_key == target_unified_policy_key
            and previous_opponent_policy_key == target_opponent_policy_key
        )
        if same_loaded_policy:
            for key in _REBOUND_RUNTIME_STICKY_ENV_KEYS:
                if previous_runtime_env is not None and hasattr(previous_runtime_env, key):
                    optional_params[key] = copy.deepcopy(getattr(previous_runtime_env, key))
                elif key in previous_env_optional_params:
                    optional_params[key] = copy.deepcopy(previous_env_optional_params[key])

        game_state.unified_policy_key = target_unified_policy_key
        game_state.opponent_unified_policy_key = target_opponent_policy_key
        game_state.unified_policy_backend = get_policy_backend_kind(game_state.unified_policy)
        game_state.defense_policy_backend = get_policy_backend_kind(game_state.defense_policy)
        game_state.unified_policy_capabilities = get_policy_capabilities(game_state.unified_policy)
        game_state.defense_policy_capabilities = get_policy_capabilities(game_state.defense_policy)

        env_optional_params, _wrapper_only_params = _split_jax_env_and_wrapper_params(
            optional_params
        )
        resolved_user_team = Team[request.user_team_name.upper()]
        game_state.user_team = resolved_user_team

        from app.backend.jax_dev_runtime import JaxDevRuntime

        game_state.jax_runtime = JaxDevRuntime(
            required_params=required_params,
            env_params=env_optional_params,
            unified_policy=game_state.unified_policy,
            opponent_policy=game_state.defense_policy,
            user_team=resolved_user_team,
            role_flag_offense=game_state.role_flag_offense,
            role_flag_defense=game_state.role_flag_defense,
        )
        resolved_user_team = game_state.jax_runtime.user_team
        game_state.user_team = resolved_user_team
        game_state.jax_runtime.reset()
        game_state.env = game_state.jax_runtime.environment
        game_state.obs = game_state.jax_runtime.observation_dict(
            observer_is_offense=resolved_user_team != Team.DEFENSE
        )

        def _apply_policy_pass_mode(policy_obj, mode_value: str) -> None:
            if policy_obj is None:
                return
            if hasattr(policy_obj, "set_pass_mode"):
                policy_obj.set_pass_mode(mode_value)
                return
            policy = getattr(policy_obj, "policy", None)
            if policy is None:
                return
            if hasattr(policy, "set_pass_mode"):
                try:
                    policy.set_pass_mode(mode_value)
                except Exception:
                    pass

        env_read = env_view(game_state.env)
        current_pass_mode = str(env_read.pass_mode or "directional")
        _apply_policy_pass_mode(game_state.unified_policy, current_pass_mode)
        _apply_policy_pass_mode(game_state.defense_policy, current_pass_mode)

        try:
            game_state.obs = validate_policy_observation_schema(
                game_state.unified_policy,
                game_state.env,
                game_state.obs,
                policy_label="unified_policy",
            )
            _ = validate_policy_observation_schema(
                game_state.defense_policy,
                game_state.env,
                game_state.obs,
                policy_label="opponent_policy",
            )
        except ValueError as schema_err:
            raise HTTPException(
                status_code=400,
                detail=f"Observation schema mismatch during init_game: {schema_err}",
            )
        game_state.prev_obs = None

        game_state.env_required_params = copy.deepcopy(required_params)
        active_env_optional_params = copy.deepcopy(env_optional_params)
        for key in _JAX_RUNTIME_STATIC_ENV_KEYS:
            if key in optional_params:
                active_env_optional_params[key] = copy.deepcopy(optional_params[key])
        game_state.env_optional_params = active_env_optional_params
        game_state.mlflow_env_optional_defaults = copy.deepcopy(mlflow_default_env_optional_params)
        game_state.unified_policy_path = unified_path
        game_state.opponent_policy_path = opponent_unified_path

        sampled_skills = {
            "layup": list(env_read.offense_layup_pct_by_player or []),
            "three_pt": list(env_read.offense_three_pt_pct_by_player or []),
            "dunk": list(env_read.offense_dunk_pct_by_player or []),
        }
        game_state.replay_offense_skills = copy.deepcopy(sampled_skills)
        game_state.sampled_offense_skills = copy.deepcopy(sampled_skills)

        _capture_turn_start_snapshot()

        game_state.user_team = resolved_user_team
        game_state.run_id = run_id
        game_state.run_name = run_name or run_id
        game_state.mlflow_phi_shaping_params = mlflow_phi_params
        game_state.mlflow_training_params = mlflow_training_params
        game_state.counterfactual_snapshot = None
        game_state.frames = []
        game_state.reward_history = []
        game_state.episode_rewards = {"offense": 0.0, "defense": 0.0}
        game_state.actions_log = []
        game_state.episode_states = []
        game_state.replay_session_id = None
        game_state.phi_log = []
        game_state.playable_session = None
        game_state.selector_segment_index = 0
        game_state.selector_last_boundary_reason = None

        try:
            frame = game_state.env.render()
            if frame is not None:
                game_state.frames.append(frame)
        except Exception:
            pass

        initial_state = get_ui_game_state()
        game_state.episode_states.append(initial_state)

        # Record initial phi entry (step 0) for Rewards tab calculations
        try:
            env = env_view(game_state.env)
            offense_ids = list(env.offense_ids or [])
            ball_holder_id = (
                int(env.ball_holder)
                if env.ball_holder is not None
                else (offense_ids[0] if offense_ids else -1)
            )

            ep_by_player: list[float] = []
            for pid in range(int(env.n_players or 0)):
                pos = env.positions[pid]
                dist = env._hex_distance(pos, env.basket_position)
                is_three = env.is_three_point_location(pos)
                if getattr(env, "allow_dunks", False) and dist == 0:
                    shot_value = 2.0
                else:
                    shot_value = 3.0 if is_three else 2.0
                p = float(env._calculate_shot_probability(pid, dist))
                ep_by_player.append(float(shot_value * p))

            phi_next = 0.0
            team_best_ep = 0.0
            ball_handler_ep = 0.0
            if offense_ids and ep_by_player and ball_holder_id >= 0:
                ball_ep = ep_by_player[ball_holder_id] if ball_holder_id < len(ep_by_player) else 0.0
                teammate_eps = [
                    ep_by_player[int(pid)] for pid in offense_ids if int(pid) != ball_holder_id and int(pid) < len(ep_by_player)
                ]
                if teammate_eps:
                    team_best_ep = max(max(teammate_eps), ball_ep)
                    ball_handler_ep = ball_ep
                    blend_weight = float(getattr(env, "phi_blend_weight", 0.0))
                    phi_next = (1.0 - blend_weight) * float(team_best_ep) + blend_weight * float(ball_ep)
                else:
                    phi_next = float(ball_ep)
                    team_best_ep = float(ball_ep)
                    ball_handler_ep = float(ball_ep)

            initial_phi_entry = {
                "step": 0,
                "phi_prev": 0.0,
                "phi_next": float(phi_next),
                "phi_beta": float(getattr(env, "phi_beta", 0.0)),
                "phi_r_shape": 0.0,
                "ball_handler": ball_holder_id,
                "offense_ids": offense_ids,
                "defense_ids": list(getattr(env, "defense_ids", [])),
                "shot_clock": int(getattr(env, "shot_clock", -1)),
                "ep_by_player": ep_by_player,
                "team_best_ep": float(team_best_ep),
                "ball_handler_ep": float(ball_handler_ep),
                "is_terminal": False,
            }
            game_state.phi_log.append(initial_phi_entry)
        except Exception as e:
            print(f"[init_game] Failed to record initial phi entry: {e}")

    counts = _compute_param_counts_from_policy(game_state.unified_policy)
    if counts:
        mlflow_training_params["param_counts"] = counts
    game_state.mlflow_training_params = mlflow_training_params
    game_state.mlflow_start_template_library = copy.deepcopy(start_template_library)
    if start_template_library is not None:
        game_state.start_template_library_source = start_template_library_source
        game_state.start_template_library_path = start_template_library_path
    else:
        game_state.start_template_library_source = None
        game_state.start_template_library_path = None

    return {
        "status": "success",
        "state": get_ui_game_state(),
        "seed": int(np.random.randint(0, 2**31 - 1)),
    }


@router.post("/api/template_bootstrap")
async def template_bootstrap(request: TemplateBootstrapRequest | None = None):
    """Initialize a JAX sandbox state for template authoring."""
    request = request or TemplateBootstrapRequest()
    user_team_name = str(request.user_team_name or "OFFENSE").strip().upper()
    if user_team_name not in {"OFFENSE", "DEFENSE"}:
        raise HTTPException(status_code=400, detail="user_team_name must be OFFENSE or DEFENSE")

    source_run_id = str(request.run_id or "").strip() or None
    required_params = {"players": int(max(1, request.players_per_side or 3))}
    env_optional_params: dict = {}
    training_params: dict = {}
    policy = None
    policy_path = None
    run_name = "Template Sandbox"

    if source_run_id:
        setup_mlflow(verbose=False)
        client = mlflow.tracking.MlflowClient()
        policy_path = get_unified_policy_path(client, source_run_id, None)
        policy = load_inference_policy(policy_path, device="cpu")
        required_params, env_optional_params, training_params, _ = (
            _jax_local_env_config_from_metadata(policy)
        )
        run_params = dict(client.get_run(source_run_id).data.params or {})
        env_optional_params = _overlay_jax_mlflow_env_params(
            env_optional_params, run_params
        )
        training_params = _overlay_jax_mlflow_training_params(
            training_params, run_params
        )
        game_state.mlflow_start_template_library = _load_start_template_library_for_run(
            client, source_run_id
        )
        run_name = f"Template Sandbox ({source_run_id})"
    else:
        game_state.mlflow_start_template_library = None

    if request.allow_dunks is not None:
        env_optional_params["allow_dunks"] = bool(request.allow_dunks)

    from app.backend.jax_dev_runtime import JaxDevRuntime

    game_state.unified_policy = policy
    game_state.offense_policy = None
    game_state.defense_policy = None
    game_state.unified_policy_key = os.path.basename(policy_path) if policy_path else None
    game_state.opponent_unified_policy_key = None
    game_state.unified_policy_backend = "jax" if policy is not None else None
    game_state.defense_policy_backend = None
    game_state.unified_policy_capabilities = get_policy_capabilities(policy)
    game_state.defense_policy_capabilities = None
    game_state.unified_policy_path = policy_path
    game_state.opponent_policy_path = None
    game_state.user_team = Team[user_team_name]
    game_state.jax_runtime = JaxDevRuntime(
        required_params=required_params,
        env_params=env_optional_params,
        unified_policy=policy,
        opponent_policy=None,
        user_team=game_state.user_team,
    )
    game_state.jax_runtime.reset()
    game_state.env = game_state.jax_runtime.environment
    game_state.obs = game_state.jax_runtime.observation_dict(
        observer_is_offense=game_state.jax_runtime.user_team_is_a
    )
    game_state.user_team = game_state.jax_runtime.user_team
    game_state.prev_obs = None
    game_state.env_required_params = copy.deepcopy(required_params)
    game_state.env_optional_params = copy.deepcopy(env_optional_params)
    game_state.mlflow_env_optional_defaults = copy.deepcopy(env_optional_params)
    game_state.run_id = source_run_id
    game_state.run_name = run_name
    game_state.mlflow_training_params = training_params
    game_state.mlflow_phi_shaping_params = None
    game_state.counterfactual_snapshot = None
    game_state.frames = []
    game_state.reward_history = []
    game_state.episode_rewards = {"offense": 0.0, "defense": 0.0}
    game_state.actions_log = []
    game_state.episode_states = []
    game_state.replay_session_id = None
    game_state.phi_log = []
    game_state.playable_session = None
    game_state.selector_segment_index = 0
    game_state.selector_last_boundary_reason = None
    _capture_turn_start_snapshot()
    initial_state = get_ui_game_state()
    game_state.episode_states.append(initial_state)
    return {
        "status": "success",
        "state": initial_state,
        "mode": "template_sandbox",
        "seed": int(np.random.randint(0, 2**31 - 1)),
    }


def _coerce_action_index(raw_action) -> int | None:
    """Convert incoming action representation to an action index when possible."""
    if isinstance(raw_action, str):
        token = raw_action.strip().upper()
        if token in ActionType.__members__:
            return int(ActionType[token].value)
    try:
        return int(raw_action)
    except Exception:
        return None


def _normalize_action_overrides(raw_actions, n_players: int) -> tuple[dict[int, int], dict[int, dict]]:
    """Normalize payloads into numeric action overrides and optional per-player metadata."""
    overrides: dict[int, int] = {}
    meta: dict[int, dict] = {}
    if isinstance(raw_actions, dict):
        for k, v in raw_actions.items():
            try:
                idx = int(k)
            except Exception:
                continue
            if idx < 0 or idx >= n_players:
                continue
            if isinstance(v, dict):
                action_type = str(v.get("type", "")).upper()
                if action_type == "PASS" and "target" in v:
                    action_idx = int(ActionType.PASS_E.value)
                    try:
                        target_id = int(v.get("target"))
                    except Exception:
                        target_id = v.get("target")
                    overrides[idx] = action_idx
                    meta[idx] = {"type": "PASS", "target": target_id}
                    continue
                if "action" in v:
                    v = v["action"]
                elif "selected_action" in v:
                    v = v["selected_action"]
            action_idx = _coerce_action_index(v)
            if action_idx is None:
                continue
            overrides[idx] = action_idx
    elif isinstance(raw_actions, (list, tuple, np.ndarray)):
        for idx, v in enumerate(raw_actions):
            if idx >= n_players:
                break
            action_idx = _coerce_action_index(v)
            if action_idx is None:
                continue
            overrides[idx] = action_idx
    return overrides, meta


@router.post("/api/step")
def step(request: ActionRequest):
    """Advance the active JAX runtime by one interactive step."""
    runtime = getattr(game_state, "jax_runtime", None)
    if runtime is None:
        raise HTTPException(status_code=400, detail="Game not initialized with a JAX checkpoint.")
    try:
        return jsonable_encoder(
            runtime.step(request, game_state),
            custom_encoder={
                np.ndarray: lambda value: value.tolist(),
                np.generic: lambda value: value.item(),
            },
        )
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"JAX runtime step failed: {err}") from err


@router.post("/api/action")
def action(request: ActionRequest):
    # Backward-compatible alias
    return step(request)


@router.post("/api/rebound_preview")
def rebound_preview(request: ReboundPreviewRequest | None = None):
    """Read-only rebound target/winner preview for the live holder or a terminal miss."""
    request = request or ReboundPreviewRequest()
    try:
        return jsonable_encoder(compute_rebound_preview(game_state, request))
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))
    except Exception as err:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Rebound preview failed: {err}")


@router.post("/api/prepare_fast_self_play")
def prepare_fast_self_play():
    """Begin non-blocking JAX Fast Mode preparation for the current session."""
    runtime = getattr(game_state, "jax_runtime", None)
    if runtime is None:
        return {"ready": True, "warming": False, "error": None}
    try:
        return runtime.prepare_fast_mode()
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Fast Mode preparation failed: {err}")


@router.post("/api/set_multi_possession_limit")
def set_multi_possession_limit(request: SetMultiPossessionLimitRequest):
    """Restart a continuous dev game at a new cap without reloading its policy."""
    runtime = getattr(game_state, "jax_runtime", None)
    if runtime is None or not runtime.multi_possession_enabled:
        raise HTTPException(status_code=400, detail="The possession limit is available only for multi-possession games.")
    try:
        runtime.set_multi_possession_limit(request.multi_possession_limit)
        if (
            request.made_basket_restart_mode is not None
            or request.check_deadline_steps is not None
            or request.check_setup_steps is not None
        ):
            runtime.set_made_basket_restart(
                request.made_basket_restart_mode
                or str(
                    runtime.env_params.get(
                        "made_basket_restart_mode",
                        "baseline_inbound",
                    )
                ),
                request.check_deadline_steps
                or int(runtime.env_params.get("check_deadline_steps", 5)),
                (
                    request.check_setup_steps
                    if request.check_setup_steps is not None
                    else int(runtime.env_params.get("check_setup_steps", 0))
                ),
            )
        runtime.reset()
        game_state.env = runtime.environment
        game_state.user_team = runtime.user_team
        game_state.obs = runtime.observation_dict(observer_is_offense=runtime.user_team_is_a)
        game_state.prev_obs = None
        game_state.self_play_active = False
        game_state.counterfactual_snapshot = None
        game_state.frames = []
        game_state.reward_history = []
        game_state.episode_rewards = {"offense": 0.0, "defense": 0.0}
        game_state.actions_log = []
        game_state.episode_states = []
        game_state.replay_session_id = None
        game_state.phi_log = []
        game_state.selector_segment_index = 0
        game_state.selector_last_boundary_reason = None
        runtime._capture_turn_start(game_state)
        state = runtime.get_full_game_state(
            game_state,
            include_policy_probs=True,
            include_action_values=True,
            include_state_values=True,
        )
        game_state.episode_states.append(dict(state))
        return {
            "status": "success",
            "state": state,
            "fast_kernel": runtime.fast_kernel_status(),
        }
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    except Exception as err:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Could not restart multi-possession game: {err}") from err


@router.post("/api/start_self_play")
def start_self_play(request: StartSelfPlayRequest | None = None):
    """Start self-play on the active JAX runtime."""
    runtime = getattr(game_state, "jax_runtime", None)
    if runtime is None:
        raise HTTPException(status_code=400, detail="Game not initialized with a JAX checkpoint.")
    try:
        return runtime.start_self_play(request, game_state)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Failed to start JAX self-play: {err}") from err


@router.post("/api/reset_turn_state")
def reset_turn_state():
    """Restore the active JAX runtime to the beginning of the turn."""
    runtime = getattr(game_state, "jax_runtime", None)
    if runtime is None:
        raise HTTPException(status_code=400, detail="Game not initialized with a JAX checkpoint.")
    try:
        return runtime.reset_turn_state(game_state)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Failed to reset JAX turn state: {err}") from err
