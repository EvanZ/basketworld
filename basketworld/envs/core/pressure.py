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
    angles = env._get_player_angles(env.ball_holder, env.defense_ids)

    info: List[Dict] = []
    for i, def_id in enumerate(env.defense_ids):
        distance = distances[i]
        cos_angle = angles[i]
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
