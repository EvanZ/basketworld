import numpy as np

from basketworld.envs.basketworld_env_v2 import HexagonBasketballEnv


def test_shot_attempt_logs_fields_and_matches_internal_calcs():
    env = HexagonBasketballEnv(players=2, render_mode=None, allow_dunks=False)
    env.reset(seed=0)
    shooter = env.offense_ids[0]
    env.ball_holder = shooter
    env.positions[shooter] = env.basket_position  # distance = 0
    env._rng = np.random.default_rng(123)  # deterministic shot RNG

    distance = env._hex_distance(env.positions[shooter], env.basket_position)
    expected_prob = env._calculate_shot_probability(shooter, distance)

    idx = shooter
    base_prob = float(env.offense_layup_pct_by_player[idx])
    base_prob = max(0.01, min(0.99, base_prob))
    expected_pressure = env._compute_shot_pressure_multiplier(shooter, env.positions[shooter], distance)

    res = env._attempt_shot(shooter)

    assert set(res.keys()) >= {
        "success",
        "distance",
        "probability",
        "rng",
        "base_probability",
        "pressure_multiplier",
        "is_three",
    }
    assert res["distance"] == distance
    assert res["probability"] == expected_prob
    assert res["base_probability"] == base_prob
    assert res["pressure_multiplier"] == expected_pressure
    assert res["is_three"] is False  # at basket
    assert 0.0 <= res["rng"] <= 1.0

    # With deterministic RNG, success should match probability comparison
    assert res["success"] == (res["rng"] < res["probability"])
