import copy
import logging
import math
import os
from typing import List

from fastapi import APIRouter, HTTPException
import mlflow
import numpy as np
from stable_baselines3 import PPO
from basketworld.utils.policies import PassBiasDualCriticPolicy, PassBiasMultiInputPolicy
from basketworld.policies import SetAttentionDualCriticPolicy, SetAttentionExtractor

from app.backend.schemas import (
    BatchUpdatePositionRequest,
    SetBallHolderRequest,
    SetIntentStateRequest,
    SetOffenseSkillsRequest,
    SetPassLogitBiasRequest,
    SetPressureParamsRequest,
    SetPassTargetStrategyRequest,
    SwapPoliciesRequest,
    UpdatePositionRequest,
    UpdateShotClockRequest,
)
from app.backend.state import get_ui_game_state, game_state
from app.backend.policies import _compute_param_counts_from_policy, get_unified_policy_path
from basketworld.envs.basketworld_env_v2 import Team


router = APIRouter()
logger = logging.getLogger(__name__)


def _rebuild_cached_obs() -> None:
    """Rebuild game_state.obs from env while preserving current viewer role."""
    if not game_state.env:
        return
    env = game_state.env
    role_value = (
        float(np.asarray(game_state.obs.get("role_flag"), dtype=np.float32).reshape(-1)[0])
        if game_state.obs is not None and game_state.obs.get("role_flag") is not None
        else (1.0 if env.training_team == Team.OFFENSE else -1.0)
    )
    observer_is_offense = bool(role_value > 0.0)
    if hasattr(env, "_build_observation_dict"):
        game_state.obs = env._build_observation_dict(observer_is_offense)
        game_state.obs["role_flag"] = np.array([role_value], dtype=np.float32)
    else:
        game_state.obs = {
            "obs": env._get_observation(),
            "action_mask": env._get_action_masks(),
            "role_flag": np.array([role_value], dtype=np.float32),
            "skills": env._get_offense_skills_array(),
        }


@router.post("/api/batch_update_player_positions")
def batch_update_player_positions(req: BatchUpdatePositionRequest):
    """Updates positions for multiple players at once."""
    if not game_state.env or game_state.obs is None:
        raise HTTPException(status_code=400, detail="Game not initialized.")
    if game_state.env.episode_ended:
        raise HTTPException(status_code=400, detail="Cannot move players after episode has ended.")

    updates = req.updates or []
    if not updates:
        raise HTTPException(status_code=400, detail="No position updates provided.")

    for upd in updates:
        pid = upd.player_id
        new_pos = (upd.q, upd.r)
        if pid < 0 or pid >= game_state.env.n_players:
            raise HTTPException(status_code=400, detail=f"Invalid player ID: {pid}")
        if not game_state.env._is_valid_position(*new_pos):
            raise HTTPException(status_code=400, detail=f"Position {new_pos} is out of bounds.")
        for i, pos in enumerate(game_state.env.positions):
            if i != pid and pos == new_pos:
                raise HTTPException(status_code=400, detail=f"Position {new_pos} is occupied by Player {i}.")
        game_state.env.positions[pid] = new_pos

    _rebuild_cached_obs()

    updated_state = get_ui_game_state()
    if game_state.episode_states:
        game_state.episode_states[-1] = updated_state
    return {
        "status": "success",
        "state": updated_state,
    }


@router.post("/api/update_player_position")
def update_player_position(req: UpdatePositionRequest):
    """Updates a single player's position during an ongoing episode."""
    if not game_state.env or game_state.obs is None:
        raise HTTPException(status_code=400, detail="Game not initialized.")
    if game_state.env.episode_ended:
        raise HTTPException(status_code=400, detail="Cannot move players after episode has ended.")

    pid = req.player_id
    new_pos = (req.q, req.r)

    if pid < 0 or pid >= game_state.env.n_players:
        raise HTTPException(status_code=400, detail=f"Invalid player ID: {pid}")
    if not game_state.env._is_valid_position(*new_pos):
        raise HTTPException(status_code=400, detail=f"Position {new_pos} is out of bounds.")
    for i, pos in enumerate(game_state.env.positions):
        if i != pid and pos == new_pos:
            raise HTTPException(status_code=400, detail=f"Position {new_pos} is occupied by Player {i}.")

    game_state.env.positions[pid] = new_pos

    _rebuild_cached_obs()

    updated_state = get_ui_game_state()
    if game_state.episode_states:
        game_state.episode_states[-1] = updated_state
    return {
        "status": "success",
        "state": updated_state,
    }


@router.post("/api/update_shot_clock")
def update_shot_clock(req: UpdateShotClockRequest):
    """Adjust the shot clock by a delta (see UpdateShotClockRequest)."""
    if not game_state.env:
        raise HTTPException(status_code=400, detail="Game not initialized.")
    try:
        delta = int(req.delta)
        current = int(getattr(game_state.env, "shot_clock", 0))
        max_val = int(getattr(game_state.env, "shot_clock_steps", current))
        new_val = current + delta
        if max_val > 0:
            new_val = max(0, min(max_val, new_val))
        else:
            new_val = max(0, new_val)
        game_state.env.shot_clock = int(new_val)
        return {
            "status": "success",
            "shot_clock": int(game_state.env.shot_clock),
            "state": get_ui_game_state(),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/api/set_shot_clock")
def set_shot_clock(req: UpdateShotClockRequest):
    """Backwards-compatible alias for update_shot_clock (expects delta)."""
    return update_shot_clock(req)


@router.post("/api/set_ball_holder")
def set_ball_holder(req: SetBallHolderRequest):
    """Manually set the ball holder during a live game (offense only)."""
    if not game_state.env:
        raise HTTPException(status_code=400, detail="Game not initialized.")

    env = game_state.env
    if req.player_id not in env.offense_ids:
        raise HTTPException(status_code=400, detail="Ball holder must be an offensive player.")

    env.ball_holder = int(req.player_id)
    try:
        _rebuild_cached_obs()
        updated_state = get_ui_game_state()
        if game_state.episode_states:
            game_state.episode_states[-1] = updated_state
        return {"status": "success", "state": updated_state}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to set ball holder: {e}")


@router.post("/api/set_intent_state")
def set_intent_state(req: SetIntentStateRequest):
    """Override the live offense intent state for the current possession."""
    if not game_state.env or game_state.obs is None:
        raise HTTPException(status_code=400, detail="Game not initialized.")
    if game_state.env.episode_ended:
        raise HTTPException(status_code=400, detail="Cannot edit intent after episode has ended.")

    env = game_state.env
    if not bool(getattr(env, "enable_intent_learning", False)):
        raise HTTPException(
            status_code=400,
            detail="Offense intent learning is not enabled for this environment.",
        )

    try:
        num_intents = max(1, int(getattr(env, "num_intents", 1)))
        max_age = max(0, int(getattr(env, "intent_commitment_steps", 0)))

        active = bool(req.active)
        intent_index = max(0, min(num_intents - 1, int(req.intent_index)))
        intent_age = max(0, min(max_age, int(req.intent_age)))

        if not active:
            intent_index = 0
            intent_age = 0
            intent_commitment_remaining = 0
        else:
            intent_commitment_remaining = max(0, max_age - intent_age)

        env.intent_active = active
        env.intent_index = int(intent_index)
        env.intent_age = int(intent_age)
        env.intent_commitment_remaining = int(intent_commitment_remaining)

        _rebuild_cached_obs()

        updated_state = get_ui_game_state()
        if game_state.episode_states:
            game_state.episode_states[-1] = updated_state
        return {"status": "success", "state": updated_state}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to set intent state: {e}")


@router.post("/api/offense_skills")
def set_offense_skills(req: SetOffenseSkillsRequest):
    """Override or reset the per-offensive-player shooting percentages for the current episode."""
    if not game_state.env:
        raise HTTPException(status_code=400, detail="Game not initialized.")

    env = game_state.env
    count = env.players_per_side

    def _normalize(values: List[float] | None, name: str) -> List[float]:
        if values is None:
            raise HTTPException(status_code=400, detail=f"Missing {name} values.")
        if len(values) != count:
            raise HTTPException(
                status_code=400,
                detail=f"{name} must include {count} values (one per offensive player).",
            )
        normalized: List[float] = []
        for v in values:
            try:
                val = float(v)
            except Exception:
                raise HTTPException(status_code=400, detail=f"Invalid {name} value: {v}")
            normalized.append(val)
        return normalized

    try:
        if req.reset_to_sampled:
            env.offense_layup_pct_by_player = copy.deepcopy(game_state.sampled_offense_skills.get("layup"))
            env.offense_three_pt_pct_by_player = copy.deepcopy(game_state.sampled_offense_skills.get("three_pt"))
            env.offense_dunk_pct_by_player = copy.deepcopy(game_state.sampled_offense_skills.get("dunk"))
        else:
            if not req.skills:
                raise HTTPException(status_code=400, detail="Missing skills payload.")
            layup = _normalize(req.skills.layup, "layup")
            three_pt = _normalize(req.skills.three_pt, "three_pt")
            dunk = _normalize(req.skills.dunk, "dunk")

            env.offense_layup_pct_by_player = layup
            env.offense_three_pt_pct_by_player = three_pt
            env.offense_dunk_pct_by_player = dunk

        return {
            "status": "success",
            "state": get_ui_game_state(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to set offense skills: {e}")


@router.post("/api/update_pass_target_strategy")
def set_pass_target_strategy(req: SetPassTargetStrategyRequest):
    """Update pass target strategy (admin)."""
    if not game_state.env:
        raise HTTPException(status_code=400, detail="Game not initialized.")
    try:
        strategy = req.strategy
        game_state.env.pass_target_strategy = strategy
        return {"status": "success", "pass_target_strategy": strategy}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update pass target strategy: {e}")


@router.post("/api/set_pass_logit_bias")
def set_pass_logit_bias(req: SetPassLogitBiasRequest):
    """Update pass logit bias for the active policies."""
    if game_state.unified_policy is None:
        raise HTTPException(status_code=400, detail="Game not initialized.")
    try:
        bias = float(req.bias) if req.bias is not None else 0.0
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid pass logit bias: {req.bias}")

    def _apply(policy):
        policy_obj = getattr(policy, "policy", None)
        if policy_obj is None:
            return
        if hasattr(policy_obj, "set_pass_logit_bias"):
            policy_obj.set_pass_logit_bias(bias)
        else:
            try:
                setattr(policy_obj, "pass_logit_bias", float(bias))
            except Exception:
                pass

    _apply(game_state.unified_policy)
    if game_state.defense_policy is not None:
        _apply(game_state.defense_policy)

    return {
        "status": "success",
        "state": get_ui_game_state(),
    }


def _set_pressure_params_impl(
    req: SetPressureParamsRequest, forced_scope: str | None = None
):
    """Internal implementation for pressure/interception parameter updates."""
    if not game_state.env:
        raise HTTPException(status_code=400, detail="Game not initialized.")
    env = game_state.env

    mlflow_defaults = (
        game_state.mlflow_env_optional_defaults
        or game_state.env_optional_params
        or {}
    )
    all_param_keys = {
        "three_pt_extra_hex_decay",
        "shot_pressure_enabled",
        "shot_pressure_max",
        "shot_pressure_lambda",
        "shot_pressure_arc_degrees",
        "base_steal_rate",
        "steal_perp_decay",
        "steal_distance_factor",
        "steal_position_weight_min",
        "defender_pressure_distance",
        "defender_pressure_turnover_chance",
        "defender_pressure_decay_lambda",
    }
    scoped_param_keys = {
        "all": set(all_param_keys),
        "shot_pressure": {
            "three_pt_extra_hex_decay",
            "shot_pressure_enabled",
            "shot_pressure_max",
            "shot_pressure_lambda",
            "shot_pressure_arc_degrees",
        },
        "pass_interception": {
            "base_steal_rate",
            "steal_perp_decay",
            "steal_distance_factor",
            "steal_position_weight_min",
        },
        "defender_pressure": {
            "defender_pressure_distance",
            "defender_pressure_turnover_chance",
            "defender_pressure_decay_lambda",
        },
    }
    requested_scope = (
        str(forced_scope).strip().lower()
        if forced_scope is not None
        else str(req.scope or req.reset_group or "").strip().lower()
    )
    if requested_scope and requested_scope not in scoped_param_keys:
        raise HTTPException(status_code=400, detail=f"Unknown scope: {requested_scope}")
    active_scope = requested_scope or ""

    if req.reset_to_mlflow_defaults:
        def _default_value(key: str, fallback):
            val = mlflow_defaults.get(key, fallback)
            return fallback if val is None else val

        default_payload = {
            "three_pt_extra_hex_decay": _default_value(
                "three_pt_extra_hex_decay", getattr(env, "three_pt_extra_hex_decay", 0.05)
            ),
            "shot_pressure_enabled": _default_value(
                "shot_pressure_enabled", getattr(env, "shot_pressure_enabled", True)
            ),
            "shot_pressure_max": _default_value(
                "shot_pressure_max", getattr(env, "shot_pressure_max", 0.5)
            ),
            "shot_pressure_lambda": _default_value(
                "shot_pressure_lambda", getattr(env, "shot_pressure_lambda", 1.0)
            ),
            "shot_pressure_arc_degrees": _default_value(
                "shot_pressure_arc_degrees", getattr(env, "shot_pressure_arc_degrees", 60.0)
            ),
            "base_steal_rate": _default_value(
                "base_steal_rate", getattr(env, "base_steal_rate", 0.35)
            ),
            "steal_perp_decay": _default_value(
                "steal_perp_decay", getattr(env, "steal_perp_decay", 1.5)
            ),
            "steal_distance_factor": _default_value(
                "steal_distance_factor", getattr(env, "steal_distance_factor", 0.08)
            ),
            "steal_position_weight_min": _default_value(
                "steal_position_weight_min", getattr(env, "steal_position_weight_min", 0.3)
            ),
            "defender_pressure_distance": _default_value(
                "defender_pressure_distance", getattr(env, "defender_pressure_distance", 1)
            ),
            "defender_pressure_turnover_chance": _default_value(
                "defender_pressure_turnover_chance",
                getattr(env, "defender_pressure_turnover_chance", 0.05),
            ),
            "defender_pressure_decay_lambda": _default_value(
                "defender_pressure_decay_lambda",
                getattr(env, "defender_pressure_decay_lambda", 1.0),
            ),
        }
        if active_scope:
            selected_keys = set(scoped_param_keys[active_scope])
        elif req.reset_keys is not None:
            selected_keys = {str(k) for k in req.reset_keys}
        else:
            raise HTTPException(
                status_code=400,
                detail="reset_to_mlflow_defaults requires scope/reset_group or reset_keys",
            )

        unknown_keys = sorted(selected_keys - set(all_param_keys))
        if unknown_keys:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown reset_keys: {', '.join(unknown_keys)}",
            )
        payload = {k: v for k, v in default_payload.items() if k in selected_keys}
    else:
        payload = {
            "three_pt_extra_hex_decay": req.three_pt_extra_hex_decay,
            "shot_pressure_enabled": req.shot_pressure_enabled,
            "shot_pressure_max": req.shot_pressure_max,
            "shot_pressure_lambda": req.shot_pressure_lambda,
            "shot_pressure_arc_degrees": req.shot_pressure_arc_degrees,
            "base_steal_rate": req.base_steal_rate,
            "steal_perp_decay": req.steal_perp_decay,
            "steal_distance_factor": req.steal_distance_factor,
            "steal_position_weight_min": req.steal_position_weight_min,
            "defender_pressure_distance": req.defender_pressure_distance,
            "defender_pressure_turnover_chance": req.defender_pressure_turnover_chance,
            "defender_pressure_decay_lambda": req.defender_pressure_decay_lambda,
        }
        payload = {k: v for k, v in payload.items() if v is not None}
        if active_scope and active_scope != "all":
            allowed = scoped_param_keys[active_scope]
            payload = {k: v for k, v in payload.items() if k in allowed}

    if not payload:
        return {
            "status": "no_change",
            "updated_keys": [],
            "applied_scope": active_scope or "all",
            "state": get_ui_game_state(),
        }

    def _as_bool(v, key: str) -> bool:
        if isinstance(v, bool):
            return v
        if v in (0, 1):
            return bool(v)
        raise HTTPException(status_code=400, detail=f"{key} must be a boolean.")

    def _as_float(v, key: str) -> float:
        try:
            return float(v)
        except Exception:
            raise HTTPException(status_code=400, detail=f"{key} must be a number.")

    def _as_int(v, key: str) -> int:
        try:
            return int(v)
        except Exception:
            raise HTTPException(status_code=400, detail=f"{key} must be an integer.")

    def _validate_range(v: float, key: str, lo: float, hi: float) -> float:
        if v < lo or v > hi:
            raise HTTPException(
                status_code=400,
                detail=f"{key} must be between {lo} and {hi}.",
            )
        return v

    def _validate_min(v: float, key: str, lo: float) -> float:
        if v < lo:
            raise HTTPException(
                status_code=400,
                detail=f"{key} must be >= {lo}.",
            )
        return v

    normalized = {}
    if "three_pt_extra_hex_decay" in payload:
        normalized["three_pt_extra_hex_decay"] = _validate_range(
            _as_float(payload["three_pt_extra_hex_decay"], "three_pt_extra_hex_decay"),
            "three_pt_extra_hex_decay",
            0.0,
            1.0,
        )

    if "shot_pressure_enabled" in payload:
        normalized["shot_pressure_enabled"] = _as_bool(
            payload["shot_pressure_enabled"], "shot_pressure_enabled"
        )
    if "shot_pressure_max" in payload:
        normalized["shot_pressure_max"] = _validate_range(
            _as_float(payload["shot_pressure_max"], "shot_pressure_max"),
            "shot_pressure_max",
            0.0,
            1.0,
        )
    if "shot_pressure_lambda" in payload:
        normalized["shot_pressure_lambda"] = _validate_min(
            _as_float(payload["shot_pressure_lambda"], "shot_pressure_lambda"),
            "shot_pressure_lambda",
            0.0,
        )
    if "shot_pressure_arc_degrees" in payload:
        normalized["shot_pressure_arc_degrees"] = _validate_range(
            _as_float(payload["shot_pressure_arc_degrees"], "shot_pressure_arc_degrees"),
            "shot_pressure_arc_degrees",
            0.0,
            360.0,
        )

    if "base_steal_rate" in payload:
        normalized["base_steal_rate"] = _validate_range(
            _as_float(payload["base_steal_rate"], "base_steal_rate"),
            "base_steal_rate",
            0.0,
            1.0,
        )
    if "steal_perp_decay" in payload:
        normalized["steal_perp_decay"] = _validate_min(
            _as_float(payload["steal_perp_decay"], "steal_perp_decay"),
            "steal_perp_decay",
            0.0,
        )
    if "steal_distance_factor" in payload:
        normalized["steal_distance_factor"] = _validate_min(
            _as_float(payload["steal_distance_factor"], "steal_distance_factor"),
            "steal_distance_factor",
            0.0,
        )
    if "steal_position_weight_min" in payload:
        normalized["steal_position_weight_min"] = _validate_range(
            _as_float(payload["steal_position_weight_min"], "steal_position_weight_min"),
            "steal_position_weight_min",
            0.0,
            1.0,
        )

    if "defender_pressure_distance" in payload:
        normalized["defender_pressure_distance"] = _as_int(
            payload["defender_pressure_distance"], "defender_pressure_distance"
        )
        if normalized["defender_pressure_distance"] < 0:
            raise HTTPException(
                status_code=400,
                detail="defender_pressure_distance must be >= 0.",
            )
    if "defender_pressure_turnover_chance" in payload:
        normalized["defender_pressure_turnover_chance"] = _validate_range(
            _as_float(
                payload["defender_pressure_turnover_chance"],
                "defender_pressure_turnover_chance",
            ),
            "defender_pressure_turnover_chance",
            0.0,
            1.0,
        )
    if "defender_pressure_decay_lambda" in payload:
        normalized["defender_pressure_decay_lambda"] = _validate_min(
            _as_float(
                payload["defender_pressure_decay_lambda"],
                "defender_pressure_decay_lambda",
            ),
            "defender_pressure_decay_lambda",
            0.0,
        )

    try:
        for key, val in normalized.items():
            setattr(env, key, val)
        if "shot_pressure_arc_degrees" in normalized:
            env.shot_pressure_arc_rad = math.radians(
                float(normalized["shot_pressure_arc_degrees"])
            )

        if game_state.env_optional_params is None:
            game_state.env_optional_params = {}
        game_state.env_optional_params.update(normalized)

        _rebuild_cached_obs()

        updated_state = get_ui_game_state()
        if game_state.episode_states:
            game_state.episode_states[-1] = updated_state
        return {
            "status": "success",
            "updated_keys": sorted(normalized.keys()),
            "applied_scope": active_scope or "all",
            "state": updated_state,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update pressure/interception parameters: {e}",
        )


@router.post("/api/set_pressure_params")
def set_pressure_params(req: SetPressureParamsRequest):
    """Backward-compatible catchall endpoint for pressure/interception settings."""
    return _set_pressure_params_impl(req)


@router.post("/api/set_shot_pressure_params")
def set_shot_pressure_params(req: SetPressureParamsRequest):
    """Update only shot-pressure and shot-distance-decay parameters."""
    return _set_pressure_params_impl(req, forced_scope="shot_pressure")


@router.post("/api/set_pass_interception_params")
def set_pass_interception_params(req: SetPressureParamsRequest):
    """Update only pass-interception parameters."""
    return _set_pressure_params_impl(req, forced_scope="pass_interception")


@router.post("/api/set_defender_pressure_params")
def set_defender_pressure_params(req: SetPressureParamsRequest):
    """Update only defender turnover-pressure parameters."""
    return _set_pressure_params_impl(req, forced_scope="defender_pressure")


@router.post("/api/swap_policies")
def swap_policies(req: SwapPoliciesRequest):
    """Swap the active PPO policies without resetting the environment."""
    if not game_state.env or game_state.obs is None:
        raise HTTPException(status_code=400, detail="Game not initialized.")
    if not game_state.run_id:
        raise HTTPException(status_code=400, detail="No MLflow run associated with current game.")

    requested_user_policy = req.user_policy_name
    requested_opponent_policy = req.opponent_policy_name

    if requested_user_policy is None and requested_opponent_policy is None:
        raise HTTPException(status_code=400, detail="No policy requested for swap.")

    client = mlflow.tracking.MlflowClient()
    custom_objects = {
        "PassBiasDualCriticPolicy": PassBiasDualCriticPolicy,
        "PassBiasMultiInputPolicy": PassBiasMultiInputPolicy,
        "SetAttentionDualCriticPolicy": SetAttentionDualCriticPolicy,
        "SetAttentionExtractor": SetAttentionExtractor,
    }

    def _apply_pass_mode(policy_obj) -> None:
        policy = getattr(policy_obj, "policy", None)
        if policy is None:
            return
        mode_value = str(getattr(game_state.env, "pass_mode", "directional"))
        if hasattr(policy, "set_pass_mode"):
            try:
                policy.set_pass_mode(mode_value)
            except Exception:
                pass

    policies_changed = False

    if requested_user_policy is not None and requested_user_policy != game_state.unified_policy_key:
        try:
            user_path = get_unified_policy_path(client, game_state.run_id, requested_user_policy)
            game_state.unified_policy = PPO.load(user_path, custom_objects=custom_objects)
            _apply_pass_mode(game_state.unified_policy)
            game_state.unified_policy_key = os.path.basename(user_path)
            game_state.unified_policy_path = user_path
            try:
                counts = _compute_param_counts_from_policy(game_state.unified_policy)
                if counts:
                    if game_state.mlflow_training_params is None:
                        game_state.mlflow_training_params = {}
                    game_state.mlflow_training_params["param_counts"] = counts
            except Exception:
                pass
            policies_changed = True
        except Exception as e:
            logger.exception("swap_policies: failed loading user policy %s", requested_user_policy)
            raise HTTPException(status_code=500, detail=f"Failed to load user policy '{requested_user_policy}': {e}")

    if requested_opponent_policy is not None:
        if requested_opponent_policy == "":
            if game_state.defense_policy is not None or game_state.opponent_unified_policy_key is not None:
                game_state.defense_policy = None
                game_state.opponent_unified_policy_key = None
                game_state.opponent_policy_path = None
                policies_changed = True
        elif requested_opponent_policy != game_state.opponent_unified_policy_key:
            try:
                opp_path = get_unified_policy_path(client, game_state.run_id, requested_opponent_policy)
                game_state.defense_policy = PPO.load(opp_path, custom_objects=custom_objects)
                _apply_pass_mode(game_state.defense_policy)
                game_state.opponent_unified_policy_key = os.path.basename(opp_path)
                game_state.opponent_policy_path = opp_path
                try:
                    counts = _compute_param_counts_from_policy(game_state.unified_policy)
                    if counts:
                        if game_state.mlflow_training_params is None:
                            game_state.mlflow_training_params = {}
                        game_state.mlflow_training_params["param_counts"] = counts
                except Exception:
                    pass
                policies_changed = True
            except Exception as e:
                logger.exception("swap_policies: failed loading opponent policy %s", requested_opponent_policy)
                raise HTTPException(status_code=500, detail=f"Failed to load opponent policy '{requested_opponent_policy}': {e}")

    if not policies_changed:
        return {
            "status": "no_change",
            "state": get_ui_game_state(),
        }

    updated_state = get_ui_game_state()
    if game_state.episode_states:
        game_state.episode_states[-1] = updated_state

    logger.info(
        "swap_policies success: user=%s opponent=%s",
        game_state.unified_policy_key,
        game_state.opponent_unified_policy_key,
    )
    return {"status": "success", "state": updated_state}
