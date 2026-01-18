import numpy as np

from basketworld.envs.basketworld_env_v2 import HexagonBasketballEnv
from basketworld.envs.core import observations as obs_core


def make_env(players=4, grid_size=6):
    env = HexagonBasketballEnv(players=players, grid_size=grid_size, render_mode=None)
    env.reset(seed=0)
    return env


def test_offense_defense_distances_and_angles_shapes():
    env = make_env(players=4)
    dists = obs_core.calculate_offense_defense_distances(env)
    angles = obs_core.calculate_offense_defense_angles(env)
    expected = len(env.offense_ids) * len(env.defense_ids)
    assert dists.size == expected
    assert angles.size == expected
    assert np.all(np.isfinite(angles))


def test_teammate_features_shapes():
    env = make_env(players=4)
    teammate_dists = obs_core.calculate_teammate_distances(env)
    teammate_angles = obs_core.calculate_teammate_angles(env)
    per_team_pairs = max(0, len(env.offense_ids) * (len(env.offense_ids) - 1) // 2)
    per_team_ordered = len(env.offense_ids) * (len(env.offense_ids) - 1)
    expected_dists = 2 * per_team_pairs
    expected_angles = 2 * per_team_ordered
    assert teammate_dists.size == expected_dists
    assert teammate_angles.size == expected_angles
    assert np.all(np.isfinite(teammate_angles))


def test_build_observation_contains_expected_lengths():
    env = make_env(players=4)
    obs = obs_core.build_observation(env)
    # Rough length check: 2 coords per player (8) + ball one-hot (4) + shot clock (1)
    # + team ids (4) + ball handler coords (2) + hoop (2) + off/def dists (4)
    # + angles (4) + teammate dists (2) + teammate angles (4) + lane steps (4)
    # + EP per offense (2) + turnover probs (2) + steal risks (2)
    expected_min_len = (
        8 + 4 + 1 + 4 + 2 + 2 + 4 + 4 + 2 + 4 + 4 + 2 + 2 + 2
    )
    assert len(obs) >= expected_min_len
    assert np.all(np.isfinite(obs))
