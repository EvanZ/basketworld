import numpy as np

from basketworld.envs.basketworld_env_v2 import HexagonBasketballEnv


def test_rewards_expected_points_for_shot():
    env = HexagonBasketballEnv(
        players=1,
        render_mode=None,
        allow_dunks=False,
        shot_pressure_enabled=False,
    )
    env.reset(seed=0)
    shooter = env.offense_ids[0]
    env.ball_holder = shooter
    env.positions[shooter] = env.basket_position  # distance 0 but dunks disabled -> layup prob

    expected = env._calculate_expected_points_for_player(shooter)
    action_results = {"shots": {shooter: {"success": False}}}

    done, rewards = env._check_termination_and_rewards(action_results)
    assert done is True
    assert np.isclose(rewards[env.offense_ids].sum(), expected)
    assert np.isclose(rewards[env.defense_ids].sum(), -expected)
