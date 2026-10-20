from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pytest

from basketworld_jax.eval.native import _episode_stats_from_trace
from basketworld_jax.train.main import _summarize_historical_match


@pytest.mark.parametrize("safety", [True, False], ids=["safety", "horizon"])
def test_historical_cutoff_preserves_clock_and_overtime_without_counting_a_win(safety):
    completed = SimpleNamespace(
        episode_ended=np.array([True]),
        episode_truncated=np.array([False]),
        team_a_score=np.array([6]),
        team_b_score=np.array([4]),
        overtime_round=np.array([1]),
        game_clock_remaining=np.array([0]),
        elapsed_game_clock_seconds=np.array([420]),
    )
    cutoff = SimpleNamespace(
        episode_ended=np.array([safety]),
        episode_truncated=np.array([safety]),
        team_a_score=np.array([0]),
        team_b_score=np.array([20]),
        overtime_round=np.array([2]),
        game_clock_remaining=np.array([70]),
        elapsed_game_clock_seconds=np.array([470]),
    )
    summary = _summarize_historical_match(
        candidate_update=2,
        opponent_update=1,
        team_a_candidate_final_state=completed,
        team_b_candidate_final_state=cutoff,
        # The runtime's completed-game-only counters are zero at cutoffs.
        team_b_candidate_diagnostics={
            "terminal_overtime_periods": np.array([0]),
            "terminal_game_clock_seconds_remaining": np.array([0]),
        },
        seed=49,
        horizon=1000,
    )
    assert summary["completed_episode_count"] == 1
    assert summary["candidate_wins"] == 1
    assert summary["candidate_win_rate"] == 1.0
    assert summary["safety_truncated_episode_count"] == int(safety)
    assert summary["horizon_truncated_episode_count"] == int(not safety)
    row = summary["episode_diagnostics"][1]
    assert row["outcome"] == "truncated"
    assert row["terminal_overtime_periods"] == 2
    assert row["terminal_game_clock_seconds_remaining"] == 70
    truncated = summary["clock_metrics_truncated_games"]
    assert truncated["elapsed_game_clock_seconds"] == 470
    assert truncated["overtime_game_count"] == 1
    assert truncated["mean_overtime_periods"] == 2
    assert truncated["mean_terminal_game_clock_seconds_remaining"] == 70
    finished = summary["clock_metrics_completed_games"]
    assert finished["game_count"] == 1
    assert finished["mean_overtime_periods"] == 1
    assert finished["mean_terminal_game_clock_seconds_remaining"] == 0
    assert finished["denominators"]["pace"] == (
        "elapsed game-clock minutes, including overtime and check pickup, "
        "excluding check setup and inbounds"
    )


def test_native_cutoff_reads_last_active_state_not_terminal_counters_or_padding():
    # Three games: completed at step 1, safety cutoff at 2, horizon cutoff at 3.
    # Unrelated gameplay counters are zero; padded state values are deliberately
    # different so selecting the last scan row for every game would fail.
    trace = defaultdict(lambda: np.zeros((3, 3)))
    trace.update(
        done=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]]),
        truncated=np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
        terminal_episode_steps=np.array([[1, 0, 0], [0, 2, 0], [0, 0, 0]]),
        overtime_round=np.array([[1, 2, 1], [9, 2, 2], [9, 9, 2]]),
        game_clock_remaining=np.array([[0, 71, 1], [99, 70, 120], [99, 99, 119]]),
        terminal_overtime_periods=np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]),
        terminal_game_clock_seconds_remaining=np.zeros((3, 3)),
        active_offense_ids=np.zeros((3, 3, 1), dtype=int),
        pass_interception_risks=np.zeros((3, 3, 1)),
    )
    stats = _episode_stats_from_trace(trace, take=3, horizon=3)
    np.testing.assert_array_equal(stats["completed"], [1, 0, 0])
    np.testing.assert_array_equal(stats["safety_truncated"], [0, 1, 0])
    np.testing.assert_array_equal(stats["horizon_truncated"], [0, 0, 1])
    np.testing.assert_array_equal(stats["terminal_overtime_periods"], [1, 2, 2])
    np.testing.assert_array_equal(
        stats["terminal_game_clock_seconds_remaining"], [0, 70, 119]
    )
