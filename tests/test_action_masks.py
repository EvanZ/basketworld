import numpy as np

from basketworld.envs.basketworld_env_v2 import HexagonBasketballEnv, ActionType


def test_action_masks_respect_ball_holder_and_moves():
    env = HexagonBasketballEnv(players=3, render_mode=None, mask_occupied_moves=False)
    env.reset(seed=123)
    masks = env._get_action_masks()
    assert masks.shape == (env.n_players, len(ActionType))
    for i in range(env.n_players):
        if i != env.ball_holder:
            # Non-ball holders cannot shoot or pass
            assert np.all(masks[i, ActionType.SHOOT.value :] == 0)
        else:
            # Ball holder can shoot/pass unless gated
            assert masks[i, ActionType.SHOOT.value] in (0, 1)


def test_action_masks_with_occupied_moves():
    env = HexagonBasketballEnv(players=2, render_mode=None, mask_occupied_moves=True)
    env.reset(seed=1)
    # Force both players adjacent to block a move
    env.positions[0] = env.basket_position
    env.positions[1] = (env.basket_position[0] + env.hex_directions[0][0], env.basket_position[1] + env.hex_directions[0][1])
    masks = env._get_action_masks()
    blocked_idx = ActionType.MOVE_E.value
    assert masks[0, blocked_idx] == 0  # move into occupied hex blocked
