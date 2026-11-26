"""Versioned basketball scoring summaries; rewards deliberately live elsewhere.

Game rates use terminal scoreboards from completed games. Training PPP uses
observed score deltas and completed possessions, including ongoing games.
Missing denominators produce None (a chart gap), never a fictitious zero.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

import numpy as np

SCORING_METRIC_SCHEMA_VERSION = 2


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else None


def summarize_learner_games(rows):
    completed = [row for row in rows if row["completed"]]
    n = len(completed)
    scored = sum(row["learner_score"] for row in completed)
    allowed = sum(row["opponent_score"] for row in completed)
    wins = sum(row["learner_score"] > row["opponent_score"] for row in completed)
    ties = sum(row["learner_score"] == row["opponent_score"] for row in completed)
    metrics = {
        "scoring_metric_schema_version": SCORING_METRIC_SCHEMA_VERSION,
        "completed_game_count": n,
        "truncated_game_count": sum(row.get("truncated", False) for row in rows),
        "learner_completed_game_points_scored": scored,
        "learner_completed_game_points_allowed": allowed,
        "learner_completed_game_wins": wins,
        "learner_completed_game_ties": ties,
        "learner_points_scored_per_completed_game": ratio(scored, n),
        "learner_points_allowed_per_completed_game": ratio(allowed, n),
        "learner_score_margin_per_completed_game": ratio(scored - allowed, n),
        "learner_completed_game_win_rate": ratio(wins, n),
        "learner_completed_game_win_score": ratio(wins + 0.5 * ties, n),
    }
    splits = {
        "learner_roster_assignment_a": lambda r: r["learner_roster_slot"] == "a",
        "learner_roster_assignment_b": lambda r: r["learner_roster_slot"] == "b",
        "learner_started_with_possession": lambda r: r[
            "learner_started_with_possession"
        ],
        "opponent_started_with_possession": lambda r: not r[
            "learner_started_with_possession"
        ],
    }
    for name, predicate in splits.items():
        group = [row for row in completed if predicate(row)]
        metrics[f"{name}_completed_game_count"] = len(group)
        metrics[f"{name}_score_margin_per_completed_game"] = ratio(
            sum(r["learner_score"] - r["opponent_score"] for r in group), len(group)
        )
        metrics[f"{name}_win_score"] = ratio(
            sum(
                (
                    1
                    if r["learner_score"] > r["opponent_score"]
                    else 0.5 if r["learner_score"] == r["opponent_score"] else 0
                )
                for r in group
            ),
            len(group),
        )
    for name, a, b in (
        (
            "roster_assignment",
            "learner_roster_assignment_a",
            "learner_roster_assignment_b",
        ),
        (
            "opening_possession",
            "learner_started_with_possession",
            "opponent_started_with_possession",
        ),
    ):
        left, right = (metrics[f"{k}_score_margin_per_completed_game"] for k in (a, b))
        metrics[f"{name}_score_margin_delta"] = (
            left - right if left is not None and right is not None else None
        )
    return metrics


def summarize_training_scoring(role_rollouts):
    rows = []
    scored = allowed = offensive = defensive = 0.0
    for role, rollout in role_rollouts.items():
        t = rollout.trajectory
        active = np.asarray(t.active_mask) > 0.5
        completed = (
            active
            & (np.asarray(t.terminal_episode_steps) > 0)
            & ~np.asarray(t.truncated, bool)
        )
        truncated = active & np.asarray(t.truncated, bool)
        a, b = np.asarray(t.team_a_score_delta), np.asarray(t.team_b_score_delta)
        learner, opponent = (a, b) if role == "offense" else (b, a)
        scored += float(learner[active].sum())
        allowed += float(opponent[active].sum())
        ended = active & np.asarray(t.possession_ended, bool)
        offensive += int((ended & (np.asarray(t.training_role) > 0)).sum())
        defensive += int((ended & (np.asarray(t.training_role) < 0)).sum())
        terminal = completed | truncated
        # Explicit terminal diagnostics survive rollouts that begin mid-game.
        for own, other, starter, done in zip(
            np.asarray(t.terminal_learner_score)[terminal],
            np.asarray(t.terminal_opponent_score)[terminal],
            np.asarray(t.learner_started_with_possession)[terminal],
            completed[terminal],
            strict=True,
        ):
            rows.append(
                dict(
                    learner_score=float(own),
                    opponent_score=float(other),
                    learner_started_with_possession=bool(starter),
                    learner_roster_slot="a" if role == "offense" else "b",
                    completed=bool(done),
                    truncated=not bool(done),
                )
            )
    metrics = summarize_learner_games(rows)
    metrics.update(
        {
            "learner_points_scored_observed": scored,
            "learner_points_allowed_observed": allowed,
            "learner_completed_offensive_possessions_observed": offensive,
            "learner_completed_defensive_possessions_observed": defensive,
            "learner_points_scored_per_completed_offensive_possession": ratio(
                scored, offensive
            ),
            "learner_points_allowed_per_completed_defensive_possession": ratio(
                allowed, defensive
            ),
        }
    )
    return metrics


def summarize_self_play_scoring(
    a, b, a_possessions, b_possessions, completed, truncated
):
    points = a + b
    return {
        "scoring_metric_schema_version": SCORING_METRIC_SCHEMA_VERSION,
        "completed_game_count": completed,
        "truncated_game_count": truncated,
        "self_play_completed_game_points": points,
        "self_play_completed_offensive_possessions": a_possessions + b_possessions,
        "self_play_points_per_team_per_completed_game": ratio(points, 2 * completed),
        "self_play_combined_points_per_completed_game": ratio(points, completed),
        "self_play_points_per_completed_offensive_possession": ratio(
            points, a_possessions + b_possessions
        ),
        "self_play_roster_slot_a_points_per_completed_game": ratio(a, completed),
        "self_play_roster_slot_b_points_per_completed_game": ratio(b, completed),
        "roster_assignment_score_margin_per_completed_game": ratio(a - b, completed),
        "self_play_roster_slot_a_points_per_completed_offensive_possession": ratio(
            a, a_possessions
        ),
        "self_play_roster_slot_b_points_per_completed_offensive_possession": ratio(
            b, b_possessions
        ),
        "self_play_roster_slot_a_completed_offensive_possessions": a_possessions,
        "self_play_roster_slot_b_completed_offensive_possessions": b_possessions,
    }


def retired_scoring_metric(key):
    return (
        key.endswith("_points_per_completed_episode")
        or key.endswith("_offensive_efficiency")
        or key
        in {
            "mean_offense_score",
            "mean_defense_score",
            "mean_score_margin",
            "mean_final_offense_score",
            "mean_final_defense_score",
            "mean_final_score_margin",
        }
    )


def artifact_scoring_metric(key):
    return bool(
        re.search(
            r"(^|_)(team_[ab]_score_delta_total|(?:offense|defense|learner|opponent)_points_(?:total|per_step))$",
            key,
        )
        or re.search(
            r"(?:regulation_segment_[1-4]|overtime)_(?:team_[ab]_points|(?:team_a_)?point_differential)$",
            key,
        )
        or key.startswith(("clock_team_a_", "clock_team_b_"))
    )


def scoring_diagnostic_name(key):
    key = re.sub(
        r"^(offense|defense)_",
        lambda m: (
            "learner_roster_slot_a_" if m[1] == "offense" else "learner_roster_slot_b_"
        ),
        key,
    )
    key = key.replace(
        "team_a_score_delta_total", "roster_slot_a_points_observed"
    ).replace("team_b_score_delta_total", "roster_slot_b_points_observed")
    key = key.replace("team_a", "roster_slot_a").replace("team_b", "roster_slot_b")
    key = key.replace("_offense_points_", "_roster_slot_a_points_").replace(
        "_defense_points_", "_roster_slot_b_points_"
    )
    return key


def separate_scoring_diagnostics(metrics: Mapping[str, Any]):
    out = {}
    diagnostics = dict(metrics.get("scoring_diagnostics") or {})
    for key, value in metrics.items():
        if retired_scoring_metric(key):
            continue
        if artifact_scoring_metric(key):
            diagnostics[scoring_diagnostic_name(key)] = value
        else:
            out[key] = value
    if diagnostics:
        out["scoring_diagnostics"] = diagnostics
    return out


def read_native_scoring_summary(summary):
    """Compatibility adapter for immutable pre-v2 artifacts; never rewrite them."""
    out = dict(summary)
    for side in ("user", "opponent"):
        new = f"{side}_points_per_completed_offensive_possession"
        if new not in out and f"{side}_points_per_possession" in out:
            out[new] = out[f"{side}_points_per_possession"]
    return out


def read_historical_scoring_summary(summary):
    """Resolve explicit PPP names without mutating an old match artifact."""
    out = dict(summary)
    for side in ("candidate", "opponent"):
        new = f"{side}_points_per_completed_offensive_possession"
        old = f"{side}_points_per_completed_possession"
        if new not in out and old in out:
            out[new] = out[old]
    return out
