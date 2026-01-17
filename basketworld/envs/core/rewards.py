from __future__ import annotations

from typing import Dict, Tuple, Optional

import numpy as np


def calculate_expected_points_for_player(env, player_id: int) -> float:
    """Calculate expected points for a single player (pressure-adjusted)."""
    player_pos = env.positions[player_id]
    dist = env._hex_distance(player_pos, env.basket_position)
    if env.allow_dunks and dist == 0:
        shot_value = 2.0
    else:
        shot_value = 3.0 if env._is_three_point_hex(tuple(player_pos)) else 2.0
    p_make = float(env._calculate_shot_probability(player_id, dist))
    return float(shot_value * p_make)


def phi_shot_quality(env) -> float:
    """
    Potential function Phi(s): team's current best expected points (pressure-adjusted).
    Respects env.phi_use_ball_handler_only and aggregation/blend settings.
    """
    if env.ball_holder is None:
        return 0.0

    team_ids = (
        env.offense_ids if (env.ball_holder in env.offense_ids) else env.defense_ids
    )

    if getattr(env, "phi_use_ball_handler_only", False):
        return calculate_expected_points_for_player(env, int(env.ball_holder))

    ball_ep = calculate_expected_points_for_player(env, int(env.ball_holder))
    ball_holder_id = int(env.ball_holder)
    mode = getattr(env, "phi_aggregation_mode", "team_best")

    if mode == "team_avg":
        eps = [calculate_expected_points_for_player(env, int(pid)) for pid in team_ids]
        return float(sum(eps) / max(1, len(eps)))

    teammate_eps = [
        calculate_expected_points_for_player(env, int(pid))
        for pid in team_ids
        if pid != ball_holder_id
    ]
    if not teammate_eps:
        return float(ball_ep)

    if mode == "teammates_best":
        teammate_aggregate = max(teammate_eps)
    elif mode == "teammates_avg":
        teammate_aggregate = sum(teammate_eps) / len(teammate_eps)
    elif mode == "teammates_worst":
        teammate_aggregate = min(teammate_eps)
    elif mode == "team_worst":
        teammate_aggregate = min(min(teammate_eps), ball_ep)
    else:
        teammate_aggregate = max(max(teammate_eps), ball_ep)

    w = float(max(0.0, min(1.0, getattr(env, "phi_blend_weight", 0.0))))
    blended = (1.0 - w) * float(teammate_aggregate) + w * float(ball_ep)
    return float(blended)


def phi_ep_breakdown(env) -> Tuple[float, float]:
    """Return (team_best_ep, ball_handler_ep) for current possession team."""
    if env.ball_holder is None:
        return 0.0, 0.0
    team_ids = (
        env.offense_ids
        if (env.ball_holder in env.offense_ids)
        else env.defense_ids
    )
    team_best = 0.0
    ball_ep = 0.0
    for pid in team_ids:
        ep = calculate_expected_points_for_player(env, pid)
        if pid == env.ball_holder:
            ball_ep = ep
        if ep > team_best:
            team_best = ep
    return float(team_best), float(ball_ep)


def calculate_expected_points_all_players(env) -> np.ndarray:
    """Expected points for each offensive player (pressure-adjusted)."""
    eps = np.zeros(env.players_per_side, dtype=np.float32)
    for idx, player_id in enumerate(env.offense_ids):
        eps[idx] = env._calculate_expected_points_for_player(player_id)
    return eps


def check_termination_and_rewards(env, action_results: Dict) -> Tuple[bool, np.ndarray]:
    """Compute done flag and rewards, mirroring HexagonBasketballEnv._check_termination_and_rewards."""
    rewards = np.zeros(env.n_players)
    done = False
    pass_reward = env.pass_reward
    violation_reward = env.violation_reward
    potential_assist_reward = env.potential_assist_reward
    full_assist_bonus = env.full_assist_bonus

    for _, pass_result in action_results.get("passes", {}).items():
        if pass_result.get("success"):
            rewards[env.offense_ids] += pass_reward / env.players_per_side
            rewards[env.defense_ids] -= pass_reward / env.players_per_side

    if action_results.get("turnovers"):
        done = True

    if action_results.get("defensive_lane_violations") and not action_results.get("shots"):
        done = True
        rewards[env.offense_ids] += violation_reward / env.players_per_side
        rewards[env.defense_ids] -= violation_reward / env.players_per_side

    for player_id, shot_result in action_results.get("shots", {}).items():
        done = True
        shooter_pos = env.positions[player_id]
        is_three_point = env._is_three_point_hex(tuple(shooter_pos))

        shot_expected_points = calculate_expected_points_for_player(env, int(player_id))
        shot_result["expected_points"] = float(shot_expected_points)
        rewards[env.offense_ids] += shot_expected_points / env.players_per_side
        rewards[env.defense_ids] -= shot_expected_points / env.players_per_side

        assist_potential = False
        assist_full = False
        assist_passer = None
        if env._assist_candidate is not None and env._assist_candidate.get("recipient_id") == int(player_id):
            if env.step_count <= int(env._assist_candidate.get("expires_at_step", -1)):
                assist_potential = True
                assist_passer = int(env._assist_candidate["passer_id"])
                base_for_pct = shot_expected_points
                potential_assist_amt = (
                    max(0.0, float(env.potential_assist_pct) * float(base_for_pct))
                    if hasattr(env, "potential_assist_pct") and env.potential_assist_pct is not None
                    else potential_assist_reward
                )
                rewards[env.offense_ids] += potential_assist_amt / env.players_per_side
                rewards[env.defense_ids] -= potential_assist_amt / env.players_per_side
                if shot_result["success"]:
                    assist_full = True
                    full_bonus_amt = (
                        max(0.0, float(env.full_assist_bonus_pct) * float(shot_expected_points))
                        if hasattr(env, "full_assist_bonus_pct") and env.full_assist_bonus_pct is not None
                        else full_assist_bonus
                    )
                    rewards[env.offense_ids] += full_bonus_amt / env.players_per_side
                    rewards[env.defense_ids] -= full_bonus_amt / env.players_per_side
        shot_result["assist_potential"] = bool(assist_potential)
        shot_result["assist_full"] = bool(assist_full)
        shot_result["is_three"] = bool(is_three_point)
        if assist_passer is not None:
            shot_result["assist_passer_id"] = assist_passer
        env._assist_candidate = None

    return done, rewards
