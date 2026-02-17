import pytest
import math

from basketworld.envs.basketworld_env_v2 import HexagonBasketballEnv


def test_three_point_hex_uses_distance_decay_without_pressure():
    env = HexagonBasketballEnv(
        players=3,
        render_mode=None,
        shot_pressure_enabled=False,
        allow_dunks=False,
        three_point_distance=4.25,
        three_point_short_distance=3.0,
    )
    env.reset(seed=0)

    shooter = env.offense_ids[1]
    env.ball_holder = shooter

    env.offense_layup_pct_by_player[1] = 0.595
    env.offense_three_pt_pct_by_player[1] = 0.36

    # Pick a genuine three-point location for this court geometry.
    three_hexes = sorted(
        list(getattr(env, "_three_point_hexes", [])),
        key=lambda pos: env._hex_distance(pos, env.basket_position),
        reverse=True,
    )
    assert three_hexes, "Expected at least one three-point hex on the court."
    shooter_pos = None
    for pos in three_hexes:
        if tuple(pos) != tuple(env.basket_position):
            shooter_pos = tuple(pos)
            break
    assert shooter_pos is not None

    env.positions[shooter] = shooter_pos
    distance = env._hex_distance(shooter_pos, env.basket_position)
    assert env._is_three_point_hex(shooter_pos)

    base_prob = env._calculate_base_shot_probability(shooter, distance)
    final_prob = env._calculate_shot_probability(shooter, distance)

    d0 = 1.0
    d1 = float(env.three_point_distance) + 1.0
    t = (float(distance) - d0) / (d1 - d0)
    t = max(0.0, min(1.0, t))
    expected = 0.595 + (0.36 - 0.595) * t
    if distance > d1:
        extra_hexes = max(0, int(distance) - int(math.floor(d1)))
        expected -= float(env.three_pt_extra_hex_decay) * float(extra_hexes)

    assert base_prob == pytest.approx(expected, abs=1e-9)
    assert final_prob == pytest.approx(expected, abs=1e-9)
