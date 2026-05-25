from __future__ import annotations

import numpy as np

from benchmarks.sbx_phase_a_rollout_torch import (
    _assemble_full_actions,
    _minimal_rollout_blockers,
    _prepare_policy_inputs,
)


def test_assemble_full_actions_scatter_matches_team_ids():
    training_actions = np.array([[1, 2], [3, 4]], dtype=np.int64)
    opponent_actions = np.array([[5, 6], [7, 8]], dtype=np.int64)
    training_ids = np.array([0, 2], dtype=np.int64)
    opponent_ids = np.array([1, 3], dtype=np.int64)

    full = _assemble_full_actions(
        training_actions,
        opponent_actions,
        training_ids,
        opponent_ids,
        n_players=4,
    )

    np.testing.assert_array_equal(
        full,
        np.array([[1, 5, 2, 6], [3, 7, 4, 8]], dtype=np.int64),
    )


def test_prepare_policy_inputs_flattens_obs_and_extracts_masks():
    obs_batch = [
        {
            "obs": np.array([1.0, 2.0], dtype=np.float32),
            "role_flag": np.array([1.0], dtype=np.float32),
            "skills": np.array([0.1, 0.2, 0.3], dtype=np.float32),
            "action_mask": np.array(
                [[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]],
                dtype=np.int8,
            ),
        },
        {
            "obs": np.array([3.0, 4.0], dtype=np.float32),
            "role_flag": np.array([1.0], dtype=np.float32),
            "skills": np.array([0.4, 0.5, 0.6], dtype=np.float32),
            "action_mask": np.array(
                [[0, 1, 0], [1, 1, 0], [0, 1, 1], [1, 0, 0]],
                dtype=np.int8,
            ),
        },
    ]

    flat_obs, training_masks, full_masks = _prepare_policy_inputs(
        obs_batch,
        training_player_ids=np.array([0, 2], dtype=np.int64),
    )

    np.testing.assert_array_equal(
        flat_obs,
        np.array(
            [
                [1.0, 2.0, 1.0, 0.1, 0.2, 0.3],
                [3.0, 4.0, 1.0, 0.4, 0.5, 0.6],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        training_masks,
        np.array(
            [
                [[1, 0, 1], [1, 1, 0]],
                [[0, 1, 0], [0, 1, 1]],
            ],
            dtype=np.int8,
        ),
    )
    assert full_masks.shape == (2, 4, 3)


def test_minimal_rollout_blockers_flag_disabled_scope_features():
    class Args:
        enable_phi_shaping = True
        illegal_defense_enabled = False
        offensive_three_seconds = True

    blockers = _minimal_rollout_blockers(Args())

    assert blockers == ["phi_shaping", "offensive_three_seconds"]
