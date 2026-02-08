import pytest

from basketworld.envs.basketworld_env_v2 import HexagonBasketballEnv


def _find_ball_holder_with_symmetric_neighbors(env):
    basket_q, basket_r = env.basket_position
    for q, r in env._valid_axial:
        if r != basket_r:
            continue
        if q <= basket_q:
            continue
        nw = (q, r - 1)
        sw = (q - 1, r + 1)
        if nw in env._valid_axial and sw in env._valid_axial:
            return (q, r), nw, sw
    return None, None, None


def test_defender_pressure_probability_symmetric():
    env = HexagonBasketballEnv(players_per_side=1, render_mode=None, defender_pressure_turnover_chance=0.025)
    env.reset(seed=0)

    ball_pos, nw_pos, sw_pos = _find_ball_holder_with_symmetric_neighbors(env)
    assert ball_pos is not None

    off_id = env.offense_ids[0]
    def_id = env.defense_ids[0]
    env.positions[off_id] = ball_pos
    env.ball_holder = off_id

    env.positions[def_id] = nw_pos
    prob_nw = env.calculate_defender_pressure_turnover_probability()

    env.positions[def_id] = sw_pos
    prob_sw = env.calculate_defender_pressure_turnover_probability()

    assert prob_nw == pytest.approx(0.025)
    assert prob_sw == pytest.approx(0.025)
    assert prob_nw == pytest.approx(prob_sw)
