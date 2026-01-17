from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np


def get_player_distances(env, base_id: int, target_ids: List[int]) -> np.ndarray:
    """Return hex distances from `base_id` to each ID in `target_ids`."""
    if not env.positions or base_id >= len(env.positions):
        return np.zeros(0, dtype=np.float32)

    base_pos = env.positions[base_id]
    distances = [
        float(env._hex_distance(base_pos, env.positions[target_id])) for target_id in target_ids
    ]
    return np.array(distances, dtype=np.float32)


def get_player_angles(env, base_id: int, target_ids: List[int]) -> np.ndarray:
    """Return cosine angles between base→target and base→basket for each target."""
    import math

    if not env.positions or base_id >= len(env.positions):
        return np.zeros(0, dtype=np.float32)

    base_pos = env.positions[base_id]
    to_basket_q = env.basket_position[0] - base_pos[0]
    to_basket_r = env.basket_position[1] - base_pos[1]
    basket_mag_sq = to_basket_q ** 2 + to_basket_r ** 2 + to_basket_q * to_basket_r
    basket_mag = math.sqrt(max(0.0, basket_mag_sq))

    angles: List[float] = []
    for target_id in target_ids:
        target_pos = env.positions[target_id]
        to_target_q = target_pos[0] - base_pos[0]
        to_target_r = target_pos[1] - base_pos[1]
        target_mag_sq = to_target_q ** 2 + to_target_r ** 2 + to_target_q * to_target_r
        target_mag = math.sqrt(max(0.0, target_mag_sq))

        if target_mag == 0 or basket_mag == 0:
            cos_angle = 0.0
        else:
            dot = (
                to_basket_q * to_target_q
                + to_basket_r * to_target_r
                + to_basket_q * to_target_r
            )
            cos_angle = dot / (basket_mag * target_mag)
            cos_angle = max(-1.0, min(1.0, cos_angle))
        angles.append(float(cos_angle))

    return np.array(angles, dtype=np.float32)


def _collect_pairwise_features(
    base_ids: List[int],
    target_ids: List[int],
    getter: Callable[[int, List[int]], np.ndarray],
) -> np.ndarray:
    values: List[float] = []
    if not target_ids:
        return np.array([], dtype=np.float32)

    for base_id in base_ids:
        feature_vec = getter(base_id, target_ids)
        values.extend(feature_vec.tolist())
    return np.array(values, dtype=np.float32)


def _collect_teammate_features(
    team_ids: List[int],
    getter: Callable[[int, List[int]], np.ndarray],
) -> np.ndarray:
    """Collect features from the first teammate to their squad-mates."""
    if len(team_ids) <= 1:
        return np.array([], dtype=np.float32)
    return _collect_pairwise_features([team_ids[0]], team_ids[1:], getter)


def calculate_offense_defense_distances(env) -> np.ndarray:
    return _collect_pairwise_features(
        env.offense_ids,
        env.defense_ids,
        lambda base, targets: get_player_distances(env, base, targets),
    )


def calculate_offense_defense_angles(env) -> np.ndarray:
    return _collect_pairwise_features(
        env.offense_ids,
        env.defense_ids,
        lambda base, targets: get_player_angles(env, base, targets),
    )


def calculate_teammate_distances(env) -> np.ndarray:
    offense_distances = _collect_teammate_features(
        env.offense_ids, lambda base, targets: get_player_distances(env, base, targets)
    )
    defense_distances = _collect_teammate_features(
        env.defense_ids, lambda base, targets: get_player_distances(env, base, targets)
    )
    if offense_distances.size or defense_distances.size:
        return np.concatenate((offense_distances, defense_distances))
    return np.array([], dtype=np.float32)


def calculate_teammate_angles(env) -> np.ndarray:
    offense_angles = _collect_teammate_features(
        env.offense_ids, lambda base, targets: get_player_angles(env, base, targets)
    )
    defense_angles = _collect_teammate_features(
        env.defense_ids, lambda base, targets: get_player_angles(env, base, targets)
    )
    if offense_angles.size or defense_angles.size:
        return np.concatenate((offense_angles, defense_angles))
    return np.array([], dtype=np.float32)


def build_observation(env) -> np.ndarray:
    """Construct the full observation vector (absolute, not egocentric)."""
    obs: List[float] = []

    norm_den: float = float(max(env.court_width, env.court_height)) or 1.0
    if not env.normalize_obs:
        norm_den = 1.0

    for q, r in env.positions:
        obs.extend([q / norm_den, r / norm_den])

    ball_holder_one_hot = np.zeros(env.n_players, dtype=np.float32)
    if env.ball_holder is not None:
        ball_holder_one_hot[env.ball_holder] = 1.0
    obs.extend(ball_holder_one_hot.tolist())

    obs.append(float(env.shot_clock))

    for pid in range(env.n_players):
        if pid in env.offense_ids:
            obs.append(1.0)
        else:
            obs.append(-1.0)

    if env.ball_holder is not None:
        ball_handler_q, ball_handler_r = env.positions[env.ball_holder]
        obs.extend([ball_handler_q / norm_den, ball_handler_r / norm_den])
    else:
        obs.extend([env.basket_position[0] / norm_den, env.basket_position[1] / norm_den])

    if env.include_hoop_vector:
        hoop_q, hoop_r = env.basket_position
        obs.extend([hoop_q / norm_den, hoop_r / norm_den])

    distances = calculate_offense_defense_distances(env)
    if env.normalize_obs:
        distances = distances / norm_den
    obs.extend(distances.tolist())

    angles = calculate_offense_defense_angles(env)
    obs.extend(angles.tolist())

    teammate_distances = calculate_teammate_distances(env)
    if env.normalize_obs:
        teammate_distances = teammate_distances / norm_den
    obs.extend(teammate_distances.tolist())

    teammate_angles = calculate_teammate_angles(env)
    obs.extend(teammate_angles.tolist())

    for pid in range(env.n_players):
        if pid in env.offense_ids:
            lane_steps = env._offensive_lane_steps.get(pid, 0)
        else:
            lane_steps = env._defender_in_key_steps.get(pid, 0)
        obs.append(float(lane_steps))

    ep_values = env.calculate_expected_points_all_players()
    obs.extend(ep_values.tolist())

    turnover_probs = np.zeros(env.players_per_side, dtype=np.float32)
    if env.ball_holder is not None and env.ball_holder in env.offense_ids:
        ball_holder_idx = env.offense_ids.index(env.ball_holder)
        turnover_prob = env.calculate_defender_pressure_turnover_probability()
        turnover_probs[ball_holder_idx] = float(turnover_prob)
    obs.extend(turnover_probs.tolist())

    steal_risks = np.zeros(env.players_per_side, dtype=np.float32)
    if env.ball_holder is not None and env.ball_holder in env.offense_ids:
        steal_probs_dict = env.calculate_pass_steal_probabilities(env.ball_holder)
        for offense_id in env.offense_ids:
            if offense_id != env.ball_holder:
                offense_idx = env.offense_ids.index(offense_id)
                steal_prob = steal_probs_dict.get(offense_id, 0.0)
                steal_risks[offense_idx] = float(steal_prob)
    obs.extend(steal_risks.tolist())

    return np.array(obs, dtype=np.float32)
