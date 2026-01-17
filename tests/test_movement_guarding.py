from basketworld.envs.basketworld_env_v2 import HexagonBasketballEnv


def test_defender_guarding_offense_distance():
    env = HexagonBasketballEnv(players=2, defender_guard_distance=1, render_mode=None)
    env.reset(seed=0)
    # Place offense at basket, defense adjacent
    env.positions[env.offense_ids[0]] = env.basket_position
    dq, dr = env.hex_directions[0]
    env.positions[env.defense_ids[0]] = (env.basket_position[0] + dq, env.basket_position[1] + dr)

    assert env._defender_is_guarding_offense(env.defense_ids[0]) is True

    # Move defense further away
    env.positions[env.defense_ids[0]] = (env.basket_position[0] + 3 * dq, env.basket_position[1] + 3 * dr)
    assert env._defender_is_guarding_offense(env.defense_ids[0]) is False
