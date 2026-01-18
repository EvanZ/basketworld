import numpy as np

from basketworld.envs.basketworld_env_v2 import HexagonBasketballEnv, ActionType


def test_offensive_three_seconds_violation_triggers_turnover():
    env = HexagonBasketballEnv(
        players=2,
        offensive_three_seconds_enabled=True,
        three_second_max_steps=1,
        render_mode=None,
    )
    env.reset(seed=0)
    # Place offense in lane and ensure ball_holder is defense to trigger violation quickly
    env.positions[env.offense_ids[0]] = env.basket_position
    env.ball_holder = env.defense_ids[0]

    # Force offense to stay NOOP so they remain in lane
    actions = [ActionType.NOOP.value for _ in range(env.n_players)]

    # First step increments counter
    env._process_simultaneous_actions(np.array(actions, dtype=int))
    # Second step should trigger violation
    res = env._process_simultaneous_actions(np.array(actions, dtype=int))
    assert any(t.get("reason") == "offensive_three_seconds" for t in res.get("turnovers", []))
