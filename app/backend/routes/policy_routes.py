from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder

from app.backend.observations import (
    _compute_q_values_for_player,
    _compute_state_values_from_obs,
    compute_policy_probabilities,
)
from app.backend.schemas import SetPhiParamsRequest
from app.backend.state import game_state


router = APIRouter()


@router.get("/api/phi_params")
def get_phi_params():
    """Get current phi shaping parameters."""
    if not game_state.env:
        return {
            "enable_phi_shaping": False,
            "phi_beta": 0.0,
            "reward_shaping_gamma": 1.0,
            "phi_use_ball_handler_only": False,
            "phi_blend_weight": 0.0,
            "phi_aggregation_mode": "team_best",
        }
    env = game_state.env
    return {
        "enable_phi_shaping": bool(getattr(env, "enable_phi_shaping", False)),
        "phi_beta": float(getattr(env, "phi_beta", 0.0)),
        "reward_shaping_gamma": float(getattr(env, "reward_shaping_gamma", 1.0)),
        "phi_use_ball_handler_only": bool(
            getattr(env, "phi_use_ball_handler_only", False)
        ),
        "phi_blend_weight": float(getattr(env, "phi_blend_weight", 0.0)),
        "phi_aggregation_mode": str(getattr(env, "phi_aggregation_mode", "team_best")),
    }


@router.post("/api/phi_params")
def set_phi_params(req: SetPhiParamsRequest):
    """Update phi shaping parameters on the live environment."""
    if not game_state.env:
        raise HTTPException(status_code=400, detail="Game not initialized")
    try:
        env = game_state.env
        if req.enable_phi_shaping is not None:
            env.enable_phi_shaping = bool(req.enable_phi_shaping)
        if req.phi_beta is not None:
            env.phi_beta = float(req.phi_beta)
        if req.reward_shaping_gamma is not None:
            env.reward_shaping_gamma = float(req.reward_shaping_gamma)
        if req.phi_use_ball_handler_only is not None:
            env.phi_use_ball_handler_only = bool(req.phi_use_ball_handler_only)
        if req.phi_blend_weight is not None:
            try:
                env.phi_blend_weight = float(max(0.0, min(1.0, req.phi_blend_weight)))
            except Exception:
                pass
        if req.phi_aggregation_mode is not None:
            valid_modes = ["team_best", "teammates_best", "teammates_avg", "team_avg"]
            if str(req.phi_aggregation_mode) in valid_modes:
                env.phi_aggregation_mode = str(req.phi_aggregation_mode)
        return {"status": "success", "params": get_phi_params()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to set phi params: {e}")


@router.get("/api/phi_log")
def get_phi_log():
    if not game_state.env:
        raise HTTPException(status_code=400, detail="Game not initialized")
    return {"phi_log": list(game_state.phi_log)}


@router.get("/api/policy_probabilities")
def get_policy_probabilities():
    """Get action probabilities from the policy for the user's team."""
    if not game_state.env or not game_state.user_team:
        raise HTTPException(status_code=400, detail="Game not initialized.")
    try:
        response = compute_policy_probabilities()
        if response is None:
            raise HTTPException(
                status_code=500, detail="Failed to compute policy probabilities"
            )
        return jsonable_encoder(response)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get policy probabilities: {e}"
        )


@router.get("/api/debug/action_masks")
def get_action_masks_debug():
    """Inspect raw action masks from the environment."""
    if not game_state.env or game_state.obs is None:
        raise HTTPException(status_code=400, detail="Game not initialized.")
    try:
        action_mask = game_state.obs["action_mask"]
        debug_info = {
            "enable_pass_gating": getattr(game_state.env, "enable_pass_gating", None),
            "pass_arc_degrees": getattr(game_state.env, "pass_arc_degrees", None),
            "ball_holder": (
                int(game_state.env.ball_holder)
                if game_state.env.ball_holder is not None
                else None
            ),
            "positions": [(int(q), int(r)) for q, r in game_state.env.positions],
            "offense_ids": list(game_state.env.offense_ids),
            "defense_ids": list(game_state.env.defense_ids),
            "action_masks": {},
        }
        action_names = [
            "NOOP",
            "MOVE_E",
            "MOVE_NE",
            "MOVE_NW",
            "MOVE_W",
            "MOVE_SW",
            "MOVE_SE",
            "SHOOT",
            "PASS_E",
            "PASS_NE",
            "PASS_NW",
            "PASS_W",
            "PASS_SW",
            "PASS_SE",
        ]
        for player_id in range(game_state.env.n_players):
            mask = action_mask[player_id].tolist()
            debug_info["action_masks"][player_id] = {
                "mask": mask,
                "legal_actions": [action_names[i] for i, m in enumerate(mask) if m == 1],
                "pass_mask": mask[8:14],
                "num_legal_passes": sum(mask[8:14]),
            }
        if debug_info["enable_pass_gating"] and debug_info["ball_holder"] is not None:
            ball_holder = debug_info["ball_holder"]
            debug_info["pass_gating_debug"] = {}
            for dir_idx in range(6):
                has_teammate = game_state.env._has_teammate_in_pass_arc(
                    ball_holder, dir_idx
                )
                debug_info["pass_gating_debug"][f"direction_{dir_idx}"] = {
                    "has_teammate_in_arc": has_teammate,
                    "pass_action": action_names[8 + dir_idx],
                    "is_legal": bool(action_mask[ball_holder][8 + dir_idx] == 1),
                }
        return jsonable_encoder(debug_info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get action masks: {e}")


@router.get("/api/action_values/{player_id}")
def get_action_values(player_id: int):
    """Calculate one-step Q-values for all actions for a player."""
    if not game_state.env or game_state.obs is None:
        raise HTTPException(status_code=400, detail="Game not initialized.")
    if game_state.env.episode_ended:
        return jsonable_encoder({})
    action_values = _compute_q_values_for_player(player_id, game_state)
    return jsonable_encoder(action_values)


@router.get("/api/state_values")
def get_state_values():
    """Get value function estimates for the pre-step state (offense/defense)."""
    if not game_state.env or game_state.obs is None:
        raise HTTPException(status_code=400, detail="Game not initialized")
    if game_state.env.episode_ended:
        return {"offensive_value": 0.0, "defensive_value": 0.0}
    try:
        obs_to_use = game_state.prev_obs if game_state.prev_obs is not None else game_state.obs
        state_values = _compute_state_values_from_obs(obs_to_use)
        game_state.prev_obs = None
        if state_values:
            return state_values
        return {"offensive_value": 0.0, "defensive_value": 0.0}
    except Exception as e:
        return {
            "offensive_value": 0.0,
            "defensive_value": 0.0,
            "error": str(e),
        }


@router.get("/api/shot_probability/{player_id}")
def get_shot_probability(player_id: int):
    """Get base and pressure-adjusted shot probability for a player."""
    if game_state.env is None:
        raise HTTPException(status_code=400, detail="Game not initialized")
    try:
        player_pos = game_state.env.positions[player_id]
        basket_pos = game_state.env.basket_position
        distance = game_state.env._hex_distance(player_pos, basket_pos)
        d0 = 1
        d1 = max(game_state.env.three_point_distance, d0 + 1)
        p0 = game_state.env.layup_pct
        p1 = game_state.env.three_pt_pct
        if distance <= d0:
            base_prob = p0
        else:
            t = (distance - d0) / (d1 - d0)
            base_prob = p0 + (p1 - p0) * t
        final_prob = game_state.env._calculate_shot_probability(player_id, distance)
        return {
            "player_id": player_id,
            "shot_probability": float(base_prob),
            "shot_probability_final": float(final_prob),
            "distance": int(distance),
        }
    except Exception as e:
        return {"player_id": player_id, "shot_probability": 0.0, "error": str(e)}

