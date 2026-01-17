from basketworld.envs.basketworld_env_v2 import HexagonBasketballEnv


def test_initial_positions_respect_spawn_distance():
    env = HexagonBasketballEnv(players=3, spawn_distance=3, render_mode=None)
    env.reset(seed=42)

    assert len(env.positions) == env.n_players
    assert len(set(env.positions)) == env.n_players

    basket = env.basket_position
    for pos in env.positions:
        assert pos != basket
        assert env._is_valid_position(*pos)

    for pid in env.offense_ids:
        dist = env._hex_distance(env.positions[pid], basket)
        assert dist >= env.spawn_distance
