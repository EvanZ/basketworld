from __future__ import annotations

import math
from typing import Dict, Optional, Tuple


def calculate_shot_probability(env, shooter_id: int, distance: int) -> float:
    """Calculate shot make probability with defender pressure applied."""
    d0 = 1
    d1 = max(float(env.three_point_distance), float(d0 + 1))

    if shooter_id in env.offense_ids:
        idx = int(shooter_id)
        p0 = float(env.offense_layup_pct_by_player[idx])
        p1 = float(env.offense_three_pt_pct_by_player[idx])
        dunk_p = float(env.offense_dunk_pct_by_player[idx])
    else:
        p0 = float(env.layup_pct)
        p1 = float(env.three_pt_pct)
        dunk_p = float(env.dunk_pct)

    if env.allow_dunks and distance == 0:
        prob = dunk_p
    elif distance <= d0:
        prob = p0
    else:
        t = (distance - d0) / (d1 - d0)
        prob = p0 + (p1 - p0) * t

    if env.shot_pressure_enabled and shooter_id is not None:
        shooter_pos = env.positions[shooter_id]
        pressure_mult = compute_shot_pressure_multiplier(env, shooter_id, shooter_pos, distance)
        prob *= pressure_mult

    prob = max(0.01, min(0.99, prob))
    return float(prob)


def compute_shot_pressure_multiplier(
    env,
    shooter_id: Optional[int],
    shooter_pos: Tuple[int, int],
    distance_to_basket: int,
) -> float:
    """Compute multiplicative reduction to shot probability due to nearest defender."""
    if not env.shot_pressure_enabled or shooter_id is None:
        return 1.0

    dir_q = env.basket_position[0] - shooter_pos[0]
    dir_r = env.basket_position[1] - shooter_pos[1]
    dir_x, dir_y = env._axial_to_cartesian(dir_q, dir_r)
    dir_norm = math.hypot(dir_x, dir_y) or 1.0
    cos_threshold = math.cos(env.shot_pressure_arc_rad / 2.0)

    opp_ids = env.defense_ids if shooter_id in env.offense_ids else env.offense_ids
    best_pressure_reduction: Optional[float] = None

    for did in opp_ids:
        dq = env.positions[did][0] - shooter_pos[0]
        dr = env.positions[did][1] - shooter_pos[1]
        vx, vy = env._axial_to_cartesian(dq, dr)
        if vx == 0 and vy == 0:
            continue
        vnorm = math.hypot(vx, vy)
        if vnorm == 0:
            continue
        cosang = (vx * dir_x + vy * dir_y) / (vnorm * dir_norm)
        in_arc = cosang >= cos_threshold
        d_def = env._hex_distance(shooter_pos, env.positions[did])
        if in_arc and d_def <= distance_to_basket:
            angle_factor = (
                (cosang - cos_threshold) / (1.0 - cos_threshold) if cos_threshold < 1.0 else 1.0
            )
            exponent_arg = d_def - 1
            distance_reduction = env.shot_pressure_max * math.exp(
                -env.shot_pressure_lambda * exponent_arg
            )
            pressure_reduction = distance_reduction * (angle_factor ** 2)

            if best_pressure_reduction is None or pressure_reduction > best_pressure_reduction:
                best_pressure_reduction = pressure_reduction

    if best_pressure_reduction is None:
        return 1.0

    return max(0.0, 1.0 - best_pressure_reduction)


def attempt_shot(env, shooter_id: int) -> Dict:
    """Perform a shot attempt matching env._attempt_shot behavior."""
    shooter_pos = env.positions[shooter_id]
    basket_pos = env.basket_position
    distance = env._hex_distance(shooter_pos, basket_pos)

    shot_success_prob = calculate_shot_probability(env, shooter_id, distance)

    d0 = 1
    d1 = max(env.three_point_distance, d0 + 1)
    if shooter_id in env.offense_ids:
        idx = int(shooter_id)
        dunk_p = env.offense_dunk_pct_by_player[idx]
        layup_p = env.offense_layup_pct_by_player[idx]
        three_p = env.offense_three_pt_pct_by_player[idx]
    else:
        dunk_p = env.dunk_pct
        layup_p = env.layup_pct
        three_p = env.three_pt_pct
    if env.allow_dunks and distance == 0:
        base_prob = dunk_p
    elif distance <= d0:
        base_prob = layup_p
    else:
        t = (distance - d0) / (d1 - d0)
        base_prob = layup_p + (three_p - layup_p) * t
    base_prob = max(0.01, min(0.99, base_prob))

    pressure_mult = compute_shot_pressure_multiplier(env, shooter_id, shooter_pos, distance)

    rng_u = env._rng.random()
    shot_made = rng_u < shot_success_prob
    if not shot_made:
        env.ball_holder = None

    is_three = env._is_three_point_hex(tuple(shooter_pos))

    return {
        "success": bool(shot_made),
        "distance": int(distance),
        "probability": float(shot_success_prob),
        "rng": float(rng_u),
        "base_probability": float(base_prob),
        "pressure_multiplier": float(pressure_mult),
        "is_three": bool(is_three),
    }
