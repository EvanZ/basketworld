import numpy as np

from basketworld.envs.basketworld_env_v2 import ActionType, HexagonBasketballEnv
from basketworld.envs.core import observations as obs_core
from basketworld.utils.wrappers import MirrorObservationWrapper, SetObservationWrapper


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
    # + pressure exposure (1)
    # + team ids (4) + ball handler coords (2) + hoop (2) + off/def dists (4)
    # + angles (4) + teammate dists (2) + teammate angles (4) + lane steps (4)
    # + EP per offense (2) + turnover probs (2) + steal risks (2)
    expected_min_len = (
        8 + 4 + 1 + 1 + 4 + 2 + 2 + 4 + 4 + 2 + 4 + 4 + 2 + 2 + 2
    )
    assert len(obs) >= expected_min_len
    assert np.all(np.isfinite(obs))


def test_mirror_wrapper_preserves_non_coordinate_features_and_masks():
    base_env = SetObservationWrapper(
        HexagonBasketballEnv(players=4, grid_size=6, render_mode=None)
    )
    mirror_env = MirrorObservationWrapper(
        SetObservationWrapper(HexagonBasketballEnv(players=4, grid_size=6, render_mode=None)),
        mirror_prob=1.0,
        seed=123,
    )
    base_obs, _ = base_env.reset(seed=123)
    mirror_obs, _ = mirror_env.reset(seed=123)

    base_players = np.asarray(base_obs["players"], dtype=np.float64)
    mirror_players = np.asarray(mirror_obs["players"], dtype=np.float64)

    env = base_env.unwrapped
    norm_den = float(max(env.court_width, env.court_height)) or 1.0
    if not getattr(env, "normalize_obs", True):
        norm_den = 1.0
    basket_q, basket_r = env.basket_position

    q = base_players[:, 0] * norm_den
    r = base_players[:, 1] * norm_den
    q_rel = q - basket_q
    r_rel = r - basket_r
    q_m = q_rel + r_rel
    r_m = -r_rel

    expected = base_players.copy()
    expected[:, 0] = (basket_q + q_m) / norm_den
    expected[:, 1] = (basket_r + r_m) / norm_den

    assert np.allclose(expected[:, :2], mirror_players[:, :2], atol=1e-6)
    assert np.allclose(base_players[:, 2:], mirror_players[:, 2:], atol=1e-6)

    mapping = list(range(len(ActionType)))
    mapping[ActionType.MOVE_NE.value] = ActionType.MOVE_SE.value
    mapping[ActionType.MOVE_SE.value] = ActionType.MOVE_NE.value
    mapping[ActionType.MOVE_NW.value] = ActionType.MOVE_SW.value
    mapping[ActionType.MOVE_SW.value] = ActionType.MOVE_NW.value
    mapping[ActionType.PASS_NE.value] = ActionType.PASS_SE.value
    mapping[ActionType.PASS_SE.value] = ActionType.PASS_NE.value
    mapping[ActionType.PASS_NW.value] = ActionType.PASS_SW.value
    mapping[ActionType.PASS_SW.value] = ActionType.PASS_NW.value

    base_mask = np.asarray(base_obs["action_mask"], dtype=np.int32)
    mirror_mask = np.asarray(mirror_obs["action_mask"], dtype=np.int32)
    remapped = np.zeros_like(mirror_mask)
    for original_idx, mirror_idx in enumerate(mapping):
        if mirror_idx < mirror_mask.shape[-1]:
            remapped[..., original_idx] = mirror_mask[..., mirror_idx]
    assert np.array_equal(base_mask, remapped)
