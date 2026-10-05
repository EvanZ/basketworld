from __future__ import annotations

import csv
import json
import math

import pytest

from analytics.historical_eval_elo import main
from basketworld.utils.historical_eval_elo import (
    build_deterministic_elo_ladder,
    expected_score,
)


def _match(
    candidate: int,
    opponent: int,
    *,
    wins: int,
    ties: int,
    losses: int,
    point_differential: float = 0.0,
    action_mode: str = "deterministic",
    truncated: int = 0,
) -> dict:
    return {
        "candidate_update": candidate,
        "opponent_update": opponent,
        "self_match": candidate == opponent,
        "action_mode": action_mode,
        "completed_episode_count": wins + ties + losses,
        "truncated_episode_count": truncated,
        "candidate_wins": wins,
        "candidate_ties": ties,
        "candidate_losses": losses,
        "mean_candidate_point_differential": point_differential,
        "seed": 3_000_000,
        "horizon": 1024,
    }


def test_batch_elo_fits_completed_outcomes_and_tracks_both_sides():
    payload = {
        "milestone_updates": [100, 500],
        "matches": [
            _match(
                500,
                100,
                wins=120,
                ties=40,
                losses=40,
                point_differential=1.25,
            )
        ],
    }

    result = build_deterministic_elo_ladder(payload)

    # Aggregate score is (120 + 0.5 * 40) / 200 = 0.7. The fitted Elo gap is
    # therefore about 147 points, with slight shrinkage from the weak prior.
    by_update = {row["update"]: row for row in result["rankings"]}
    assert by_update[500]["elo_rating"] == pytest.approx(1073.55, abs=0.02)
    assert by_update[100]["elo_rating"] == pytest.approx(926.45, abs=0.02)
    assert by_update[500]["rating_standard_error"] == pytest.approx(13.40, abs=0.02)
    assert by_update[500]["wins"] == 120
    assert by_update[500]["ties"] == 40
    assert by_update[500]["losses"] == 40
    assert by_update[100]["wins"] == 40
    assert by_update[100]["ties"] == 40
    assert by_update[100]["losses"] == 120
    assert by_update[500]["mean_point_differential"] == pytest.approx(1.25)
    assert by_update[100]["mean_point_differential"] == pytest.approx(-1.25)
    processed = result["processed_matches"][0]
    assert processed["candidate_score"] == pytest.approx(0.7)
    assert processed["fitted_candidate_score"] == pytest.approx(0.7, abs=0.001)


def test_chronological_elo_remains_available_for_comparison():
    result = build_deterministic_elo_ladder(
        {
            "milestone_updates": [100, 500],
            "matches": [_match(500, 100, wins=120, ties=40, losses=40)],
        },
        method="chronological",
        k_factor=32.0,
    )

    by_update = {row["update"]: row for row in result["rankings"]}
    assert by_update[500]["elo_rating"] == pytest.approx(1006.4)
    assert by_update[100]["elo_rating"] == pytest.approx(993.6)
    assert by_update[500]["rating_standard_error"] is None
    assert result["config"]["method"] == "chronological"


def test_rating_anchor_is_a_uniform_display_shift():
    payload = {
        "milestone_updates": [100, 500],
        "matches": [_match(500, 100, wins=120, ties=40, losses=40)],
    }
    unanchored = build_deterministic_elo_ladder(payload)
    anchored = build_deterministic_elo_ladder(
        payload,
        anchor_update=100,
        anchor_rating=400.0,
    )

    before = {row["update"]: row for row in unanchored["rankings"]}
    after = {row["update"]: row for row in anchored["rankings"]}
    shift = 400.0 - before[100]["elo_rating"]
    assert after[100]["elo_rating"] == pytest.approx(400.0)
    assert after[500]["elo_rating"] == pytest.approx(
        before[500]["elo_rating"] + shift
    )
    assert (
        after[500]["elo_rating"] - after[100]["elo_rating"]
    ) == pytest.approx(before[500]["elo_rating"] - before[100]["elo_rating"])
    assert after[500]["rating_standard_error"] == pytest.approx(
        before[500]["rating_standard_error"]
    )
    assert after[500]["rating_ci95_low"] == pytest.approx(
        before[500]["rating_ci95_low"] + shift
    )
    assert anchored["processed_matches"][0][
        "fitted_candidate_score"
    ] == pytest.approx(unanchored["processed_matches"][0]["fitted_candidate_score"])
    assert anchored["config"]["anchor_update"] == 100
    assert anchored["config"]["anchor_rating"] == 400.0
    assert anchored["config"]["rating_offset"] == pytest.approx(shift)


def test_deterministic_elo_filters_and_reports_unusable_records():
    valid = _match(500, 100, wins=1, ties=0, losses=1)
    duplicate = dict(valid)
    inconsistent = _match(1000, 500, wins=1, ties=0, losses=1)
    inconsistent["completed_episode_count"] = 3
    truncated_only = _match(1500, 1000, wins=0, ties=0, losses=0, truncated=2)
    payload = {
        "milestone_updates": [100, 500, 1000, 1500],
        "matches": [
            _match(100, 100, wins=0, ties=2, losses=0),
            _match(500, 100, wins=1, ties=0, losses=1, action_mode="sampled"),
            valid,
            duplicate,
            inconsistent,
            truncated_only,
            {"candidate_update": "bad"},
        ],
    }

    result = build_deterministic_elo_ladder(payload)

    assert result["summary"] == {
        "input_record_count": 7,
        "accepted_matchup_count": 1,
        "accepted_completed_game_count": 2,
        "skipped_record_count": 6,
        "skip_reasons": {
            "duplicate_record": 1,
            "inconsistent_outcome_counts": 1,
            "malformed_updates": 1,
            "non_deterministic": 1,
            "self_match": 1,
            "truncated_only": 1,
        },
        "ranked_checkpoint_count": 4,
        "connected_component_count": 3,
        "ratings_globally_comparable": False,
    }


def test_deterministic_elo_is_invariant_to_input_match_order():
    matches = [
        _match(1000, 100, wins=60, ties=20, losses=20),
        _match(500, 100, wins=40, ties=20, losses=40),
        _match(1000, 500, wins=30, ties=40, losses=30),
    ]
    forward = build_deterministic_elo_ladder(
        {"milestone_updates": [100, 500, 1000], "matches": matches}
    )
    reverse = build_deterministic_elo_ladder(
        {"milestone_updates": [100, 500, 1000], "matches": list(reversed(matches))}
    )

    assert forward == reverse
    assert [
        (row["candidate_update"], row["opponent_update"])
        for row in forward["processed_matches"]
    ] == [(500, 100), (1000, 100), (1000, 500)]


def test_tied_aggregate_match_leaves_equal_ratings_unchanged():
    result = build_deterministic_elo_ladder(
        {
            "milestone_updates": [1, 2],
            "matches": [_match(2, 1, wins=25, ties=50, losses=25)],
        }
    )

    assert [row["elo_rating"] for row in result["rankings"]] == [1000.0, 1000.0]
    assert expected_score(1000.0, 1000.0) == pytest.approx(0.5)


def test_local_cli_writes_deterministic_json_and_csv(tmp_path, capsys):
    input_path = tmp_path / "historical_opponent_evaluation.json"
    output_dir = tmp_path / "out"
    input_path.write_text(
        json.dumps(
            {
                "milestone_updates": [100, 500],
                "matches": [_match(500, 100, wins=2, ties=0, losses=0)],
            }
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "--input-json",
                str(input_path),
                "--output-dir",
                str(output_dir),
                "--initial-rating",
                "1200",
                "--prior-stddev",
                "600",
                "--anchor-update",
                "100",
                "--anchor-rating",
                "400",
            ]
        )
        == 0
    )

    json_path = output_dir / "deterministic_elo.json"
    csv_path = output_dir / "deterministic_elo.csv"
    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["config"]["initial_rating"] == 1200.0
    assert saved["config"]["method"] == "batch"
    assert saved["config"]["k_factor"] is None
    assert saved["config"]["prior_stddev"] == 600.0
    assert saved["config"]["anchor_update"] == 100
    assert saved["config"]["anchor_rating"] == 400.0
    assert saved["source"]["kind"] == "local_json"
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["update"]) for row in rows] == [500, 100]
    assert float(rows[1]["elo_rating"]) == pytest.approx(400.0)
    assert "Rank  Update" in capsys.readouterr().out


@pytest.mark.parametrize("bad_k", [0.0, -1.0, math.inf])
def test_deterministic_elo_rejects_invalid_k_factor(bad_k):
    with pytest.raises(ValueError):
        build_deterministic_elo_ladder(
            {"matches": []}, method="chronological", k_factor=bad_k
        )


@pytest.mark.parametrize("bad_prior", [0.0, -1.0, math.inf])
def test_batch_elo_rejects_invalid_prior_stddev(bad_prior):
    with pytest.raises(ValueError):
        build_deterministic_elo_ladder({"matches": []}, prior_stddev=bad_prior)


def test_rating_anchor_requires_both_options_and_a_known_checkpoint():
    payload = {
        "milestone_updates": [100, 500],
        "matches": [_match(500, 100, wins=1, ties=0, losses=1)],
    }
    with pytest.raises(ValueError, match="provided together"):
        build_deterministic_elo_ladder(payload, anchor_update=100)
    with pytest.raises(ValueError, match="provided together"):
        build_deterministic_elo_ladder(payload, anchor_rating=400.0)
    with pytest.raises(ValueError, match="not present"):
        build_deterministic_elo_ladder(
            payload,
            anchor_update=999,
            anchor_rating=400.0,
        )
