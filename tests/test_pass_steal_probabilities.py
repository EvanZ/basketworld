import numpy as np

from basketworld.envs.basketworld_env_v2 import HexagonBasketballEnv


def test_pass_steal_probabilities_shape():
    env = HexagonBasketballEnv(players=2, render_mode=None)
    env.reset(seed=0)
    env.ball_holder = env.offense_ids[0]
    env.positions[env.offense_ids[0]] = env.basket_position
    env.positions[env.offense_ids[1]] = (env.basket_position[0] + 1, env.basket_position[1])

    probs = env.calculate_pass_steal_probabilities(env.ball_holder)
    assert set(probs.keys()) == {env.offense_ids[1]}
    val = probs[env.offense_ids[1]]
    assert 0.0 <= val <= 1.0
