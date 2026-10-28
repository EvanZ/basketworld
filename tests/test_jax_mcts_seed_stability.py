from __future__ import annotations

from copy import deepcopy

import pytest

from basketworld_jax.eval.mcts_seed_stability import (
    summarize_search_seed_stability,
)


def _action_stats(action: list[int]) -> list[dict[str, object]]:
    runner = list(action)
    runner[0] = (runner[0] + 1) % 7
    return [
        {
            "action": action,
            "visits": 80,
            "mean_value": 0.5,
            "prior": 0.5,
        },
        {
            "action": runner,
            "visits": 48,
            "mean_value": 0.4,
            "prior": 0.5,
        },
    ]


def _payload(seed_offset: int, actions: list[list[int]]) -> dict[str, object]:
    provenance = {
        "evaluated_checkpoint_update_index": 10,
        "opponent_checkpoint_update_index": 10,
        "evaluated_policy_spec": {"model_type": "attention"},
        "opponent_policy_spec": {"model_type": "attention"},
        "evaluated_stable_team": "team_a",
        "opponent_stable_team": "team_b",
        "evaluated_action_mode": "deterministic",
        "opponent_action_mode": "deterministic",
        "resolved_environment_config": {"players_per_team": 2},
        "value_target": "team_a_minus_team_b_points",
    }
    return {
        "protocol": {
            "primary_search_depth": 5,
            "search_depths": [5],
            "simulations_per_action": 128,
        },
        "provenance": provenance,
        "aggregate": {"primary_search_depth": 5},
        "records": [
            {
                "sample": {
                    "state_id": f"state-{state_index}",
                    "evaluated_role": "defense",
                    "policy_action": [0, 0],
                },
                "depth_results": {
                    "5": {
                        "search_seed": seed_offset + state_index,
                        "search": {
                            "selected_action": action,
                            "selected_value": 0.5,
                            "simulations": 128,
                            "action_stats": _action_stats(action),
                        },
                    }
                },
            }
            for state_index, action in enumerate(actions)
        ],
    }


def test_seed_stability_reports_player_joint_and_margin_metrics() -> None:
    payloads = [
        _payload(100, [[0, 1], [2, 3]]),
        _payload(200, [[0, 1], [2, 4]]),
        _payload(300, [[0, 1], [5, 4]]),
    ]

    summary = summarize_search_seed_stability(
        payloads,
        bootstrap_samples=100,
        bootstrap_seed=9,
    )

    assert summary["run_count"] == 3
    assert summary["state_count"] == 2
    assert summary["controlled_player_count"] == 2
    assert summary["pairwise_player_action_agreement"]["estimate"] == pytest.approx(
        2.0 / 3.0
    )
    assert summary["pairwise_joint_action_agreement"]["estimate"] == 0.5
    assert summary["all_seed_player_action_consensus"]["estimate"] == 0.5
    assert summary["all_seed_joint_action_consensus"]["estimate"] == 0.5
    assert summary["mean_modal_player_action_agreement"]["estimate"] == pytest.approx(
        5.0 / 6.0
    )
    assert summary["search_confidence"]["selected_visit_share"][
        "estimate"
    ] == pytest.approx(80.0 / 128.0)
    assert summary["search_confidence"]["selected_vs_runner_up_visit_margin_fraction"][
        "estimate"
    ] == pytest.approx(32.0 / 128.0)
    assert summary["search_confidence"]["selected_vs_runner_up_mean_value_margin"][
        "estimate"
    ] == pytest.approx(0.1)


def test_seed_stability_rejects_reused_search_seed() -> None:
    first = _payload(100, [[0, 1], [2, 3]])
    second = deepcopy(first)

    with pytest.raises(ValueError, match="distinct search seed"):
        summarize_search_seed_stability(
            [first, second],
            bootstrap_samples=10,
        )


def test_seed_stability_rejects_protocol_drift() -> None:
    first = _payload(100, [[0, 1], [2, 3]])
    second = _payload(200, [[0, 1], [2, 3]])
    second["protocol"]["simulations_per_action"] = 256  # type: ignore[index]

    with pytest.raises(ValueError, match="different search protocol"):
        summarize_search_seed_stability(
            [first, second],
            bootstrap_samples=10,
        )
