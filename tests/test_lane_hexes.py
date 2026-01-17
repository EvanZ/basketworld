from basketworld.envs.basketworld_env_v2 import HexagonBasketballEnv


def test_lane_hexes_non_empty_and_symmetric():
    env = HexagonBasketballEnv(players=3, render_mode=None, three_second_lane_width=1, three_second_lane_height=3)
    env.reset(seed=0)
    off_lane = env._calculate_offensive_lane_hexes()
    def_lane = env._calculate_defensive_lane_hexes()
    assert off_lane == def_lane
    assert len(off_lane) > 0
    # Basket itself should be in lane
    assert env.basket_position in off_lane
