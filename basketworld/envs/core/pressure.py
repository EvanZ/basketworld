from __future__ import annotations

import math
from typing import Dict, List


def calculate_defender_pressure_info(env) -> List[Dict]:
    """
    Calculate defender pressure information for the current offensive ball handler.
    Returns list of dicts with defender_id, distance, turnover_prob, cos_angle.
    """
    if env.ball_holder is None:
        return []
    if env.ball_holder not in env.offense_ids:
        return []

    distances = env._get_player_distances(env.ball_holder, env.defense_ids)

    base_pos = env.positions[env.ball_holder]
    to_basket_q = env.basket_position[0] - base_pos[0]
    to_basket_r = env.basket_position[1] - base_pos[1]
    basket_x, basket_y = env._axial_to_cartesian(to_basket_q, to_basket_r)
    basket_mag = math.hypot(basket_x, basket_y)

    info: List[Dict] = []
    for i, def_id in enumerate(env.defense_ids):
        distance = distances[i]
        target_pos = env.positions[def_id]
        to_target_q = target_pos[0] - base_pos[0]
        to_target_r = target_pos[1] - base_pos[1]
        target_x, target_y = env._axial_to_cartesian(to_target_q, to_target_r)
        target_mag = math.hypot(target_x, target_y)

        if target_mag == 0 or basket_mag == 0:
            cos_angle = 0.0
        else:
            dot = (basket_x * target_x) + (basket_y * target_y)
            cos_angle = dot / (basket_mag * target_mag)

        if distance <= env.defender_pressure_distance and cos_angle >= 0:
            turnover_prob = env.defender_pressure_turnover_chance * math.exp(
                -env.defender_pressure_decay_lambda * max(0, distance - 1)
            )
            info.append(
                {
                    "defender_id": int(def_id),
                    "distance": int(distance),
                    "turnover_prob": float(turnover_prob),
                    "cos_angle": float(cos_angle),
                }
            )
    return info


def calculate_defender_pressure_turnover_probability(env) -> float:
    """Compound turnover probability from all nearby defenders for current ball handler."""
    defender_pressure_info = calculate_defender_pressure_info(env)
    if not defender_pressure_info:
        return 0.0

    complement_product = 1.0
    for pressure in defender_pressure_info:
        complement_product *= 1.0 - pressure["turnover_prob"]
    return 1.0 - complement_product
