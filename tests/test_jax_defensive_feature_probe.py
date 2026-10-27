from __future__ import annotations

import numpy as np

from basketworld_jax.eval.defensive_feature_probe import (
    classification_metrics,
    compute_defensive_relational_features,
    make_state_splits,
    stable_action_mask,
)


def test_state_splits_are_reproducible_and_keep_one_label_per_state() -> None:
    state_ids = [f"state-{index}" for index in range(20)]

    first = make_state_splits(
        state_ids,
        seed=9,
        train_fraction=0.7,
        validation_fraction=0.15,
    )
    second = make_state_splits(
        state_ids,
        seed=9,
        train_fraction=0.7,
        validation_fraction=0.15,
    )

    assert first.tolist() == second.tolist()
    assert {value: int(np.sum(first == value)) for value in set(first)} == {
        "train": 14,
        "validation": 3,
        "test": 3,
    }


def test_relational_features_encode_direction_threat_and_responsibility() -> None:
    positions = np.asarray(
        [
            [
                [0, 0],
                [2, 0],
                [4, 0],
                [0, 4],
                [4, 4],
                [2, 1],
                [4, 1],
                [0, 3],
                [4, 3],
                [5, 0],
            ]
        ],
        dtype=np.int32,
    )
    expected_shot_values = np.zeros((1, 10), dtype=np.float32)
    expected_shot_values[0, :5] = [0.2, 0.8, 0.4, 2.4, 1.1]

    directional, threat = compute_defensive_relational_features(
        positions=positions,
        ball_holder=np.asarray([1]),
        offense_ids=np.asarray([[0, 1, 2, 3, 4]]),
        defense_ids=np.asarray([[5, 6, 7, 8, 9]]),
        expected_shot_values=expected_shot_values,
        court_norm_den=8.0,
    )

    assert directional.shape == (1, 5, 6)
    assert threat.shape == (1, 5, 3)
    np.testing.assert_allclose(directional[0, 0, :2], [0.0, -0.125])
    np.testing.assert_allclose(directional[0, 0, 2:4], [0.0, -0.125])
    assert directional[0, 0, 4:].tolist() == [1.0, 1.0]
    assert threat[0, 0, 0] == np.float32(0.8 / 3.0)
    assert threat[0, :, 1].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert threat[0, :, 2].tolist() == [0.0, 0.0, 1.0, 0.0, 0.0]


def test_stability_requires_a_second_completed_artifact() -> None:
    states = ["a", "b", "c"]
    source = {"a": (1, 2), "b": (3, 4), "c": (5, 6)}

    unavailable, available = stable_action_mask(states, source, [])
    stable, stable_available = stable_action_mask(
        states,
        source,
        [{"a": (1, 2), "b": (3, 0), "c": (5, 6)}],
    )

    assert available is False
    assert unavailable.tolist() == [False, False, False]
    assert stable_available is True
    assert stable.tolist() == [True, False, True]


def test_classification_metrics_bootstrap_by_state() -> None:
    labels = np.asarray([[1, 2], [0, 5]], dtype=np.int32)
    probabilities = np.full((2, 2, 7), 0.01, dtype=np.float64)
    probabilities[0, 0, 1] = 0.94
    probabilities[0, 1, 3] = 0.94
    probabilities[1, 0, 0] = 0.94
    probabilities[1, 1, 5] = 0.94
    probabilities /= probabilities.sum(axis=-1, keepdims=True)

    metrics = classification_metrics(
        labels,
        probabilities,
        bootstrap_samples=100,
        bootstrap_seed=5,
    )

    assert metrics["exact_player_action_accuracy"]["estimate"] == 0.75
    assert metrics["exact_player_action_accuracy"]["state_count"] == 2
    assert metrics["exact_player_action_accuracy"]["decision_count"] == 4
    assert metrics["exact_joint_action_accuracy"]["estimate"] == 0.5
    assert metrics["move_vs_noop_accuracy"]["estimate"] == 1.0
    assert metrics["exact_move_direction_accuracy"]["estimate"] == 2.0 / 3.0
    assert metrics["three_bucket_move_direction_accuracy"]["estimate"] == 1.0
    assert metrics["legal_action_macro_roc_auc"]["class_count"] == 4
    assert metrics["legal_action_macro_pr_auc"]["class_count"] == 4


def test_auc_metrics_condition_on_legal_actions_and_reward_correct_ranking() -> None:
    labels = np.asarray([[0], [1], [0], [1]], dtype=np.int32)
    probabilities = np.zeros((4, 1, 7), dtype=np.float64)
    probabilities[:, 0, :2] = np.asarray(
        [[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9]]
    )
    action_masks = np.zeros_like(probabilities, dtype=bool)
    action_masks[..., :2] = True

    metrics = classification_metrics(
        labels,
        probabilities,
        action_masks=action_masks,
        bootstrap_samples=100,
        bootstrap_seed=11,
    )

    assert metrics["legal_action_macro_roc_auc"]["estimate"] == 1.0
    assert metrics["legal_action_macro_roc_auc"]["class_count"] == 2
    assert metrics["legal_action_macro_pr_auc"]["estimate"] == 1.0
    assert metrics["move_vs_noop_roc_auc"]["estimate"] == 1.0
    assert metrics["move_vs_noop_pr_auc"]["estimate"] == 1.0
