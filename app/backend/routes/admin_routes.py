import copy
import logging
import os

from fastapi import APIRouter, HTTPException
import mlflow
from stable_baselines3 import PPO
from basketworld.utils.policies import PassBiasDualCriticPolicy, PassBiasMultiInputPolicy
from basketworld.policies import SetAttentionDualCriticPolicy, SetAttentionExtractor

from app.backend.schemas import (
    BatchUpdatePositionRequest,
    SetBallHolderRequest,
    SetOffenseSkillsRequest,
    SetPassTargetStrategyRequest,
    SwapPoliciesRequest,
    UpdatePositionRequest,
    UpdateShotClockRequest,
)
from app.backend.state import get_full_game_state, game_state
from app.backend.policies import _compute_param_counts_from_policy, get_unified_policy_path
from basketworld.envs.basketworld_env_v2 import Team


router = APIRouter()
logger = logging.getLogger(__name__)


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

    obs_vec = game_state.env._get_observation()
    action_mask = game_state.env._get_action_masks()
    game_state.obs = {
        "obs": obs_vec,
        "action_mask": action_mask,
        "role_flag": game_state.obs.get("role_flag"),
        "skills": game_state.obs.get("skills"),
    }

    return {
        "status": "success",
        "state": get_full_game_state(
            include_policy_probs=True,
            include_action_values=True,
            include_state_values=True,
        ),
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

    obs_vec = game_state.env._get_observation()
    action_mask = game_state.env._get_action_masks()
    game_state.obs = {
        "obs": obs_vec,
        "action_mask": action_mask,
        "role_flag": game_state.obs.get("role_flag"),
        "skills": game_state.obs.get("skills"),
    }

    return {
        "status": "success",
        "state": get_full_game_state(
            include_policy_probs=True,
            include_action_values=True,
            include_state_values=True,
        ),
    }


@router.post("/api/update_shot_clock")
def update_shot_clock(req: UpdateShotClockRequest):
    """Manually set the shot clock to a specific value."""
    if not game_state.env:
        raise HTTPException(status_code=400, detail="Game not initialized.")
    try:
        new_val = int(req.shot_clock)
        game_state.env.shot_clock = new_val
        return {
            "status": "success",
            "shot_clock": int(game_state.env.shot_clock),
            "state": get_full_game_state(),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


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
        obs_vec = env._get_observation()
        action_mask = env._get_action_masks()
        game_state.obs = {
            "obs": obs_vec,
            "action_mask": action_mask,
            "role_flag": game_state.obs.get("role_flag"),
            "skills": game_state.obs.get("skills"),
        }
        return {"status": "success", "state": get_full_game_state()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to set ball holder: {e}")


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
            layup = _normalize(req.layup, "layup")
            three_pt = _normalize(req.three_pt, "three_pt")
            dunk = _normalize(req.dunk, "dunk")

            env.offense_layup_pct_by_player = layup
            env.offense_three_pt_pct_by_player = three_pt
            env.offense_dunk_pct_by_player = dunk

        return {
            "status": "success",
            "state": get_full_game_state(include_policy_probs=True),
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

    policies_changed = False

    if requested_user_policy is not None and requested_user_policy != game_state.unified_policy_key:
        try:
            user_path = get_unified_policy_path(client, game_state.run_id, requested_user_policy)
            game_state.unified_policy = PPO.load(user_path, custom_objects=custom_objects)
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
            "state": get_full_game_state(
                include_policy_probs=True,
                include_action_values=True,
                include_state_values=True,
            ),
        }

    updated_state = get_full_game_state(
        include_policy_probs=True,
        include_action_values=True,
        include_state_values=True,
    )
    if game_state.episode_states:
        game_state.episode_states[-1] = updated_state

    logger.info(
        "swap_policies success: user=%s opponent=%s",
        game_state.unified_policy_key,
        game_state.opponent_unified_policy_key,
    )
    return {"status": "success", "state": updated_state}
