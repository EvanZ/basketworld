from __future__ import annotations

import json

from app.training_backend import historical_elo


def _match(
    candidate: int,
    opponent: int,
    *,
    wins: int,
    losses: int,
    truncated: int = 0,
    point_differential: float = 0.0,
    self_match: bool = False,
) -> dict:
    return {
        "action_mode": "deterministic",
        "candidate_update": candidate,
        "opponent_update": opponent,
        "candidate_wins": wins,
        "candidate_ties": 0,
        "candidate_losses": losses,
        "completed_episode_count": wins + losses,
        "truncated_episode_count": truncated,
        "mean_candidate_point_differential": point_differential,
        "seed": 3_000_000,
        "horizon": 2048,
        "self_match": self_match,
    }


def _write_payload(path, matches):
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "milestone_updates": [100, 250, 500],
                "milestone_checkpoints": {
                    "100": {},
                    "250": {},
                    "500": {},
                },
                "matches": matches,
            }
        ),
        encoding="utf-8",
    )


def test_summary_anchors_u100_and_excludes_self_and_future_updates(tmp_path):
    artifact = tmp_path / historical_elo.HISTORICAL_EVAL_FILENAME
    _write_payload(
        artifact,
        [
            _match(100, 100, wins=10, losses=10, self_match=True),
            _match(
                250,
                100,
                wins=15,
                losses=5,
                truncated=2,
                point_differential=4.0,
            ),
        ],
    )
    historical_elo.clear_historical_elo_cache()

    result = historical_elo.historical_elo_summary(artifact)

    assert result["status"] == "ready"
    assert [point["update"] for point in result["points"]] == [100, 250]
    assert result["points"][0]["elo_rating"] == 400.0
    assert result["latest"]["update"] == 250
    assert result["next_evaluation_update"] == 500
    assert result["summary"]["accepted_matchup_count"] == 1
    assert result["summary"]["truncated_game_count"] == 2
    assert "2 truncated evaluation episodes" in result["warnings"][0]


def test_summary_cache_reuses_fit_until_artifact_changes(tmp_path, monkeypatch):
    artifact = tmp_path / historical_elo.HISTORICAL_EVAL_FILENAME
    matches = [_match(250, 100, wins=12, losses=8)]
    _write_payload(artifact, matches)
    historical_elo.clear_historical_elo_cache()
    real_build = historical_elo.build_deterministic_elo_ladder
    calls = 0

    def counted_build(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_build(*args, **kwargs)

    monkeypatch.setattr(
        historical_elo,
        "build_deterministic_elo_ladder",
        counted_build,
    )

    first = historical_elo.historical_elo_summary(artifact)
    second = historical_elo.historical_elo_summary(artifact)
    assert calls == 1
    assert first == second

    matches.append(_match(500, 250, wins=11, losses=9))
    _write_payload(artifact, matches)
    updated = historical_elo.historical_elo_summary(artifact)
    assert calls == 2
    assert updated["latest"]["update"] == 500


def test_missing_artifact_returns_waiting_state(tmp_path):
    result = historical_elo.historical_elo_summary(
        tmp_path / historical_elo.HISTORICAL_EVAL_FILENAME
    )

    assert result["status"] == "waiting"
    assert result["points"] == []
    assert result["latest"] is None
