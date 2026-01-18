import numpy as np

from basketworld.envs.basketworld_env_v2 import HexagonBasketballEnv, ActionType


def test_out_of_bounds_move_triggers_turnover_for_ball_holder():
    env = HexagonBasketballEnv(players=2, allow_dunks=False, render_mode=None)
    env.reset(seed=0)
    # Force ball holder near edge
    env.positions[env.ball_holder] = (0, 0)
    env.positions[1 - env.ball_holder] = (1, 0)

    # Try to move out of bounds (e.g., west from (0,0))
    actions = np.array([ActionType.MOVE_W.value if pid == env.ball_holder else ActionType.NOOP.value for pid in range(env.n_players)], dtype=int)

    res = env._process_simultaneous_actions(actions)
    assert any(t.get("reason") == "move_out_of_bounds" for t in res.get("turnovers", []))
