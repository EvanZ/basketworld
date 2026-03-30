from __future__ import annotations

from typing import Any

import numpy as np
from basketworld.envs.basketworld_env_v2 import ActionType, Team
from basketworld.utils.action_resolution import IllegalActionStrategy

from app.backend.env_access import env_view
from app.backend.observations import _predict_policy_actions
from app.backend.selector_runtime import (
    apply_rollout_segment_start,
    maybe_apply_rollout_multisegment_boundary,
)


def strategy_from_determinism(deterministic: bool) -> IllegalActionStrategy:
    return (
        IllegalActionStrategy.BEST_PROB
        if bool(deterministic)
        else IllegalActionStrategy.SAMPLE_PROB
    )


def predict_joint_policy_actions(
    *,
    unified_policy: Any,
    opponent_policy: Any,
    obs: dict | None,
    env: Any,
    player_deterministic: bool,
    opponent_deterministic: bool,
    role_flag_offense: float,
    role_flag_defense: float,
) -> dict[str, Any]:
    env_read = env_view(env)
    n_players = int(env_read.n_players or 0)
    num_actions = len(ActionType)
    resolved_unified, unified_probs = _predict_policy_actions(
        unified_policy,
        obs,
        env,
        deterministic=bool(player_deterministic),
        strategy=strategy_from_determinism(bool(player_deterministic)),
        role_flag_offense=float(role_flag_offense),
        role_flag_defense=float(role_flag_defense),
    )
    if resolved_unified is None:
        resolved_unified = np.zeros(n_players, dtype=int)
        unified_probs = [
            np.zeros(num_actions, dtype=np.float32)
            for _ in range(n_players)
        ]

    resolved_opponent = None
    opponent_probs = None
    if opponent_policy is not None:
        resolved_opponent, opponent_probs = _predict_policy_actions(
            opponent_policy,
            obs,
            env,
            deterministic=bool(opponent_deterministic),
            strategy=strategy_from_determinism(bool(opponent_deterministic)),
            role_flag_offense=float(role_flag_offense),
            role_flag_defense=float(role_flag_defense),
        )
    return {
        "resolved_unified": resolved_unified,
        "unified_probs": unified_probs,
        "resolved_opponent": resolved_opponent,
        "opponent_probs": opponent_probs,
    }


def combine_team_actions(
    *,
    env: Any,
    user_team: Team,
    resolved_unified: np.ndarray,
    resolved_opponent: np.ndarray | None,
) -> np.ndarray:
    env_read = env_view(env)
    n_players = int(env_read.n_players or 0)
    offense_ids = list(env_read.offense_ids or [])
    defense_ids = list(env_read.defense_ids or [])
    full_action = np.zeros(n_players, dtype=int)
    if user_team == Team.OFFENSE:
        full_action[offense_ids] = resolved_unified[offense_ids]
        if resolved_opponent is None:
            full_action[defense_ids] = resolved_unified[defense_ids]
        else:
            full_action[defense_ids] = resolved_opponent[defense_ids]
    else:
        full_action[defense_ids] = resolved_unified[defense_ids]
        if resolved_opponent is None:
            full_action[offense_ids] = resolved_unified[offense_ids]
        else:
            full_action[offense_ids] = resolved_opponent[offense_ids]
    return full_action


def initialize_rollout_selector_episode(
    *,
    env: Any,
    obs: dict | None,
    training_params: dict | None,
    unified_policy: Any,
    opponent_policy: Any,
    user_team: Team,
    role_flag_offense: float,
    selector_intent_selection_mode: str | None = None,
) -> dict[str, Any]:
    result = apply_rollout_segment_start(
        env,
        obs,
        training_params=training_params,
        unified_policy=unified_policy,
        opponent_policy=opponent_policy,
        user_team=user_team,
        role_flag_offense=float(role_flag_offense),
        allow_uniform_fallback=False,
        selection_mode=selector_intent_selection_mode,
    )
    return {
        "obs": result.get("obs", obs),
        "selector_segment_index": 0,
        "selector_last_boundary_reason": None,
        "used_selector": bool(result.get("used_selector", False)),
    }


def apply_post_step_rollout_updates(
    *,
    env: Any,
    next_obs: dict | None,
    info: dict | None,
    done: bool,
    training_params: dict | None,
    unified_policy: Any,
    opponent_policy: Any,
    user_team: Team,
    role_flag_offense: float,
    role_flag_defense: float,
    selector_segment_index: int,
    selector_intent_selection_mode: str | None = None,
) -> dict[str, Any]:
    boundary = maybe_apply_rollout_multisegment_boundary(
        env,
        next_obs,
        info=info,
        done=bool(done),
        training_params=training_params,
        unified_policy=unified_policy,
        opponent_policy=opponent_policy,
        user_team=user_team,
        role_flag_offense=float(role_flag_offense),
        selector_segment_index=int(selector_segment_index),
        selection_mode=selector_intent_selection_mode,
    )
    obs = boundary.get("obs", next_obs)
    if obs is not None:
        role_value = role_flag_offense if user_team == Team.OFFENSE else role_flag_defense
        obs["role_flag"] = np.array([float(role_value)], dtype=np.float32)
    return {
        "obs": obs,
        "selector_segment_index": int(boundary.get("selector_segment_index", selector_segment_index)),
        "selector_last_boundary_reason": boundary.get("reason"),
        "selector_used_selector": bool(boundary.get("used_selector", False)),
        "selector_start_source": boundary.get("start_source"),
    }
