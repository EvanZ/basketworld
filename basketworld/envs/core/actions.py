from __future__ import annotations

import numpy as np

from basketworld.envs.basketworld_env_v2 import ActionType


def build_action_masks(
    n_players: int,
    positions,
    ball_holder: int | None,
    move_mask_by_cell,
    hex_directions,
    mask_occupied_moves: bool,
    enable_pass_gating: bool,
    has_teammate_in_pass_arc,
) -> np.ndarray:
    """
    Generate legal action masks for each player.
    Mirrors HexagonBasketballEnv._get_action_masks behavior.
    """
    masks = np.ones((n_players, len(ActionType)), dtype=np.int8)

    for i in range(n_players):
        if ball_holder is None or i != ball_holder:
            masks[i, slice(ActionType.SHOOT.value, ActionType.PASS_SE.value + 1)] = masks[
                i, slice(ActionType.SHOOT.value, ActionType.PASS_SE.value + 1)
            ]
            masks[i, ActionType.SHOOT.value : ActionType.PASS_SE.value + 1] = 0

        cell = positions[i]
        move_mask = move_mask_by_cell.get(cell)
        if move_mask is not None:
            for dir_idx in range(6):
                masks[i, ActionType.MOVE_E.value + dir_idx] = move_mask[dir_idx]

    if mask_occupied_moves:
        occupied = set(positions)
        for i in range(n_players):
            curr_q, curr_r = positions[i]
            for dir_idx in range(6):
                action_idx = ActionType.MOVE_E.value + dir_idx
                if masks[i, action_idx] == 0:
                    continue
                dq, dr = hex_directions[dir_idx]
                nbr = (curr_q + dq, curr_r + dr)
                if nbr in occupied:
                    masks[i, action_idx] = 0

    if enable_pass_gating and ball_holder is not None:
        for dir_idx in range(6):
            pass_action_idx = ActionType.PASS_E.value + dir_idx
            if not has_teammate_in_pass_arc(ball_holder, dir_idx):
                masks[ball_holder, pass_action_idx] = 0

    return masks
