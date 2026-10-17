from __future__ import annotations

from typing import Any, Sequence


def zero_safe_ratio(numerator: float, denominator: float) -> float:
    """Return a stable zero when an evaluation denominator is empty."""
    return float(numerator / denominator) if denominator > 0.0 else 0.0


def summarize_clock_metrics(
    *,
    game_count: int,
    side_a_points: float,
    side_b_points: float,
    side_a_completed_offensive_possessions: int,
    side_b_completed_offensive_possessions: int,
    elapsed_game_clock_seconds: float,
    completed_possession_game_clock_seconds: float,
    regulation_completion_count: int,
    overtime_game_count: int,
    overtime_periods_total: int,
    terminal_game_clock_seconds_total: float,
    regulation_segment_side_a_points: Sequence[float] = (0.0, 0.0, 0.0, 0.0),
    regulation_segment_side_b_points: Sequence[float] = (0.0, 0.0, 0.0, 0.0),
    overtime_side_a_points: float = 0.0,
    overtime_side_b_points: float = 0.0,
    completed_possessions_for_per_game: int | None = None,
) -> dict[str, Any]:
    """Apply the shared training/evaluation clock diagnostic denominators."""
    total_possessions = int(
        side_a_completed_offensive_possessions
        + side_b_completed_offensive_possessions
    )
    per_game_possessions = (
        total_possessions
        if completed_possessions_for_per_game is None
        else int(completed_possessions_for_per_game)
    )
    elapsed_minutes = float(elapsed_game_clock_seconds) / 60.0
    side_a_segments = [float(value) for value in regulation_segment_side_a_points]
    side_b_segments = [float(value) for value in regulation_segment_side_b_points]
    if len(side_a_segments) != 4 or len(side_b_segments) != 4:
        raise ValueError("Clock summaries require exactly four regulation segments.")
    return {
        "game_count": int(game_count),
        "elapsed_game_clock_seconds": float(elapsed_game_clock_seconds),
        "completed_possessions_per_game": zero_safe_ratio(
            per_game_possessions, game_count
        ),
        "mean_completed_possession_game_clock_seconds": zero_safe_ratio(
            completed_possession_game_clock_seconds, total_possessions
        ),
        "regulation_completion_count": int(regulation_completion_count),
        "overtime_game_count": int(overtime_game_count),
        "mean_overtime_periods": zero_safe_ratio(
            overtime_periods_total, overtime_game_count
        ),
        "mean_terminal_game_clock_seconds_remaining": zero_safe_ratio(
            terminal_game_clock_seconds_total, game_count
        ),
        "side_a": {
            "points": float(side_a_points),
            "completed_offensive_possessions": int(
                side_a_completed_offensive_possessions
            ),
            "offensive_efficiency_points_per_possession": zero_safe_ratio(
                side_a_points, side_a_completed_offensive_possessions
            ),
            "pace_completed_offensive_possessions_per_game_minute": zero_safe_ratio(
                side_a_completed_offensive_possessions, elapsed_minutes
            ),
        },
        "side_b": {
            "points": float(side_b_points),
            "completed_offensive_possessions": int(
                side_b_completed_offensive_possessions
            ),
            "offensive_efficiency_points_per_possession": zero_safe_ratio(
                side_b_points, side_b_completed_offensive_possessions
            ),
            "pace_completed_offensive_possessions_per_game_minute": zero_safe_ratio(
                side_b_completed_offensive_possessions, elapsed_minutes
            ),
        },
        "regulation_segments": [
            {
                "segment": index + 1,
                "side_a_points": side_a_segments[index],
                "side_b_points": side_b_segments[index],
                "side_a_point_differential": (
                    side_a_segments[index] - side_b_segments[index]
                ),
            }
            for index in range(4)
        ],
        "overtime": {
            "side_a_points": float(overtime_side_a_points),
            "side_b_points": float(overtime_side_b_points),
            "side_a_point_differential": float(
                overtime_side_a_points - overtime_side_b_points
            ),
        },
        "denominators": {
            "offensive_efficiency": "completed offensive possessions",
            "pace": "elapsed game-clock minutes, including overtime and stopped phases",
            "possession_duration": "completed possessions, measured in elapsed game-clock seconds",
            "per_game": "games in this completion group",
            "zero_denominator_value": 0.0,
        },
    }
