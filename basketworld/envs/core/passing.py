from __future__ import annotations

import math
from typing import Dict, Tuple, List, Union, Optional


def calculate_steal_probability_for_pass(env, passer_pos: Tuple[int, int], recv_pos: Tuple[int, int], passer_id: int):
    """
    Shared helper for steal probability computation. Mirrors env._calculate_steal_probability_for_pass.
    """
    pass_distance = env._hex_distance(passer_pos, recv_pos)

    forward_defenders = []
    dir_q = recv_pos[0] - passer_pos[0]
    dir_r = recv_pos[1] - passer_pos[1]
    dir_x, dir_y = env._axial_to_cartesian(dir_q, dir_r)
    dir_norm = math.hypot(dir_x, dir_y) or 1.0

    for did in env.defense_ids:
        defender_pos = env.positions[did]
        if defender_pos == passer_pos or defender_pos == recv_pos:
            continue

        dx, dy = env._axial_to_cartesian(defender_pos[0] - passer_pos[0], defender_pos[1] - passer_pos[1])
        dot = dx * dir_x + dy * dir_y
        if dot < 0:
            continue
        forward_defenders.append(did)

    defender_contributions = []
    for did in forward_defenders:
        defender_pos = env.positions[did]
        perp_distance = env._point_to_line_distance(defender_pos, passer_pos, recv_pos)
        position_t = env._get_position_on_line(defender_pos, passer_pos, recv_pos)
        position_weight = env.steal_position_weight_min + (1.0 - env.steal_position_weight_min) * position_t
        steal_contrib = (
            env.base_steal_rate
            * math.exp(-env.steal_perp_decay * perp_distance)
            * (1.0 + env.steal_distance_factor * pass_distance)
            * position_weight
        )
        steal_contrib = max(0.0, min(1.0, steal_contrib))
        defender_contributions.append((did, steal_contrib, perp_distance, position_t))

    total_steal_prob = 0.0
    if defender_contributions:
        complement_product = 1.0
        for _, steal_contrib, _, _ in defender_contributions:
            complement_product *= (1.0 - steal_contrib)
        total_steal_prob = 1.0 - complement_product

    return total_steal_prob, defender_contributions


def calculate_pass_steal_probabilities(env, passer_id: int) -> Dict[int, float]:
    """Public wrapper to compute steal probabilities for passes to each teammate."""
    if env.ball_holder != passer_id:
        return {}

    passer_pos = env.positions[passer_id]
    team_ids = env.offense_ids if passer_id in env.offense_ids else env.defense_ids

    steal_probs: Dict[int, float] = {}
    for teammate_id in team_ids:
        if teammate_id == passer_id:
            continue
        recv_pos = env.positions[teammate_id]
        total_steal_prob, _ = calculate_steal_probability_for_pass(env, passer_pos, recv_pos, passer_id)
        steal_probs[teammate_id] = total_steal_prob
    return steal_probs


def has_teammate_in_pass_arc(env, passer_id: int, direction_idx: int) -> bool:
    """Return True if any teammate is within the configured pass arc for the given direction."""
    passer_pos = env.positions[passer_id]
    dir_dq, dir_dr = env.hex_directions[direction_idx]

    dir_x, dir_y = env._axial_to_cartesian(dir_dq, dir_dr)
    dir_norm = math.hypot(dir_x, dir_y) or 1.0
    half_angle_rad = math.radians(max(1.0, min(360.0, env.pass_arc_degrees))) / 2.0
    cos_threshold = math.cos(half_angle_rad) - env._PASS_ARC_COS_EPS

    team_ids = env.offense_ids if passer_id in env.offense_ids else env.defense_ids
    for pid in team_ids:
        if pid == passer_id:
            continue
        tq, tr = env.positions[pid]
        vx, vy = env._axial_to_cartesian(tq - passer_pos[0], tr - passer_pos[1])
        vnorm = math.hypot(vx, vy)
        if vnorm == 0:
            continue
        cosang = (vx * dir_x + vy * dir_y) / (vnorm * dir_norm)
        if cosang >= cos_threshold:
            return True
    return False


def attempt_pass(env, passer_id: int, direction_idx: int, results: Dict) -> None:
    """
    Arc-based passing with line-of-sight steal mechanics.
    """
    passer_pos = env.positions[passer_id]
    dir_dq, dir_dr = env.hex_directions[direction_idx]

    dir_x, dir_y = env._axial_to_cartesian(dir_dq, dir_dr)
    dir_norm = math.hypot(dir_x, dir_y) or 1.0
    half_angle_rad = (
        math.radians(max(1.0, min(360.0, getattr(env, "pass_arc_degrees", 60.0)))) / 2.0
    )
    cos_threshold = math.cos(half_angle_rad) - env._PASS_ARC_COS_EPS

    def in_arc(to_q: int, to_r: int) -> bool:
        vx, vy = env._axial_to_cartesian(to_q - passer_pos[0], to_r - passer_pos[1])
        vnorm = math.hypot(vx, vy)
        if vnorm == 0:
            return False
        cosang = (vx * dir_x + vy * dir_y) / (vnorm * dir_norm)
        return cosang >= cos_threshold

    def in_defender_arc(to_q: int, to_r: int) -> bool:
        vx, vy = env._axial_to_cartesian(to_q - passer_pos[0], to_r - passer_pos[1])
        vnorm = math.hypot(vx, vy)
        if vnorm == 0:
            return False
        cosang = (vx * dir_x + vy * dir_y) / (vnorm * dir_norm)
        return cosang >= 0.0

    team_ids = env.offense_ids if passer_id in env.offense_ids else env.defense_ids
    recv_id = None
    recv_dist = None
    recv_value = None
    strategy = getattr(env, "pass_target_strategy", "nearest")
    strategy = strategy if strategy in ("nearest", "best_ev") else "nearest"
    candidate_data: List[Dict[str, Union[int, float]]] = []
    for pid in team_ids:
        if pid == passer_id:
            continue
        tq, tr = env.positions[pid]
        if not in_arc(tq, tr):
            continue
        d = env._hex_distance(passer_pos, (tq, tr))
        if strategy == "nearest":
            if recv_id is None or d < recv_dist:
                recv_id = pid
                recv_dist = d
        else:
            steal_prob, _ = env._calculate_steal_probability_for_pass(passer_pos, (tq, tr), passer_id)
            ep = float(env._calculate_expected_points_for_player(pid))
            value = (1.0 - float(steal_prob)) * ep
            candidate_data.append(
                {
                    "pid": pid,
                    "distance": float(d),
                    "steal_prob": float(steal_prob),
                    "ep": ep,
                    "value": value,
                }
            )
    if strategy == "best_ev" and candidate_data:
        candidate_data.sort(
            key=lambda c: (
                -c.get("value", -1e9),
                c.get("distance", float("inf")),
                c.get("pid", 0),
            )
        )
        top = candidate_data[0]
        recv_id = int(top["pid"])
        recv_dist = float(top["distance"])
        recv_value = float(top.get("value", 0.0))

    if recv_id is None:
        if getattr(env, "pass_oob_turnover_prob", 0.0) > 0.0:
            if env._rng.random() < float(env.pass_oob_turnover_prob):
                env._turnover_to_defense(passer_id)
                results["turnovers"].append(
                    {
                        "player_id": passer_id,
                        "reason": "pass_out_of_bounds",
                        "turnover_pos": tuple(env.positions[passer_id]),
                    }
                )
                return
        return

    recv_pos = env.positions[recv_id]
    pass_distance = env._hex_distance(passer_pos, recv_pos)

    forward_defenders = []
    for did in env.defense_ids:
        if did == passer_id or did == recv_id:
            continue
        def_pos = env.positions[did]
        if not in_defender_arc(def_pos[0], def_pos[1]):
            continue
        if env._is_between_points(def_pos, passer_pos, recv_pos):
            forward_defenders.append(did)

    total_steal_prob = 0.0
    defender_contributions: List[Tuple[int, float, float, float]] = []
    if forward_defenders:
        def_pos_list = [env.positions[d] for d in forward_defenders]
        perp_dists = [env._point_to_line_distance(p, passer_pos, recv_pos) for p in def_pos_list]
        pos_on_line = [env._get_position_on_line(p, passer_pos, recv_pos) for p in def_pos_list]
        for idx, did in enumerate(forward_defenders):
            perp_dist = perp_dists[idx]
            pos_t = pos_on_line[idx]
            position_weight = env.steal_position_weight_min + (1.0 - env.steal_position_weight_min) * pos_t
            steal_contrib = (
                env.base_steal_rate
                * math.exp(-env.steal_perp_decay * perp_dist)
                * (1.0 + env.steal_distance_factor * pass_distance)
                * position_weight
            )
            steal_contrib = max(0.0, min(1.0, steal_contrib))
            defender_contributions.append((did, steal_contrib, perp_dist, pos_t))

        complement_product = 1.0
        for _, steal_contrib, _, _ in defender_contributions:
            complement_product *= (1.0 - steal_contrib)
        total_steal_prob = 1.0 - complement_product

        if env._rng.random() < total_steal_prob:
            max_contrib_tuple = max(defender_contributions, key=lambda x: x[1])
            stealing_defender = int(max_contrib_tuple[0])
            env.ball_holder = stealing_defender
            results["turnovers"].append(
                {
                    "player_id": passer_id,
                    "reason": "steal",
                    "stolen_by": stealing_defender,
                    "turnover_pos": tuple(env.positions[passer_id]),
                    "pass_target": recv_id,
                    "pass_distance": pass_distance,
                    "total_steal_prob": total_steal_prob,
                    "defenders_evaluated": [
                        {
                            "id": did,
                            "steal_contribution": contrib,
                            "perp_distance": perp_dist,
                            "position_on_line": pos_t,
                        }
                        for did, contrib, perp_dist, pos_t in defender_contributions
                    ],
                }
            )
            return

    env.ball_holder = recv_id
    results["passes"][passer_id] = {
        "success": True,
        "target": recv_id,
        "pass_distance": pass_distance,
        "target_strategy": strategy,
        "target_value": recv_value,
        "total_steal_prob": total_steal_prob,
        "defenders_evaluated": [
            {
                "id": did,
                "steal_contribution": contrib,
                "perp_distance": perp_dist,
                "position_on_line": pos_t,
            }
            for did, contrib, perp_dist, pos_t in defender_contributions
        ],
    }
    env._assist_candidate = {
        "passer_id": int(passer_id),
        "recipient_id": int(recv_id),
        "expires_at_step": int(env.step_count + env.assist_window),
    }
