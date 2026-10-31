from __future__ import annotations

import json

import pytest

import basketworld_jax.eval.policy_crossplay as policy_crossplay
from basketworld_jax.eval.policy_crossplay import (
    _normalize_episode,
    summarize_crossplay,
    validate_crossplay_pair,
    validate_protocol,
)


def _episode(*, user_score: float, opponent_score: float, starting_team: str):
    return {
        "completed": True,
        "truncated": False,
        "termination_kind": "completed",
        "game": {
            "user_score": user_score,
            "opponent_score": opponent_score,
            "user_completed_possessions": 10,
            "opponent_completed_possessions": 10,
            "starting_offense_team": starting_team,
        },
        "outcome_info": {"turnovers": [], "defensive_lane_violations": []},
    }


def test_protocol_requires_complete_four_game_blocks():
    validate_protocol(episodes=1000, batch_size=64, bootstrap_samples=2000)
    with pytest.raises(ValueError, match="multiple of four"):
        validate_protocol(episodes=1002, batch_size=64, bootstrap_samples=2000)
    with pytest.raises(ValueError, match="even"):
        validate_protocol(episodes=1000, batch_size=63, bootstrap_samples=2000)


def test_summary_bootstraps_four_game_matched_scenarios():
    rows = []
    for assignment in ("ar_team_a", "ar_team_b"):
        for index, (start, ar_score, independent_score) in enumerate(
            (("team_a", 10, 8), ("team_b", 9, 9))
        ):
            rows.append(
                _normalize_episode(
                    _episode(
                        user_score=ar_score,
                        opponent_score=independent_score,
                        starting_team=start,
                    ),
                    assignment=assignment,
                    assignment_episode_index=index,
                )
            )

    summary = summarize_crossplay(rows, bootstrap_seed=7, bootstrap_samples=100)

    assert summary["overall"]["games"] == 4
    assert summary["overall"]["autoregressive_wins"] == 2
    assert summary["overall"]["ties"] == 2
    matched = summary["matched_scenarios"]
    assert matched["complete_four_game_block_count"] == 1
    assert matched["autoregressive_win_score"] == {
        "mean": 0.75,
        "ci95_low": 0.75,
        "ci95_high": 0.75,
    }
    assert matched["autoregressive_margin"]["mean"] == 1.0


def test_crossplay_pair_rejects_same_trained_mode(tmp_path):
    def write_checkpoint(name: str, mode: str, saved_at: str):
        path = tmp_path / name
        path.mkdir()
        (path / "metadata.json").write_text(
            json.dumps(
                {
                    "checkpoint_version": 2,
                    "saved_at": saved_at,
                    "update_index": 1,
                    "trainer_config": {},
                    "policy_spec": {
                        "team_action_mode": mode,
                        "training_player_count": 4,
                    },
                    "frozen_config": {},
                }
            ),
            encoding="utf-8",
        )
        return path

    first = write_checkpoint("first", "autoregressive", "2026-10-29T01:00:00Z")
    second = write_checkpoint("second", "autoregressive", "2026-10-29T02:00:00Z")

    with pytest.raises(ValueError, match="was not trained with.*independent"):
        validate_crossplay_pair(first, second)


def test_crossplay_pair_accepts_distinct_trained_policies(tmp_path):
    def write_checkpoint(name: str, mode: str, saved_at: str):
        path = tmp_path / name
        path.mkdir()
        (path / "metadata.json").write_text(
            json.dumps(
                {
                    "checkpoint_version": 2,
                    "saved_at": saved_at,
                    "update_index": 1,
                    "trainer_config": {},
                    "policy_spec": {
                        "team_action_mode": mode,
                        "training_player_count": 4,
                    },
                    "frozen_config": {},
                }
            ),
            encoding="utf-8",
        )
        return path

    ar_path = write_checkpoint("ar", "autoregressive", "2026-10-29T01:00:00Z")
    independent_path = write_checkpoint(
        "independent", "independent", "2026-10-29T02:00:00Z"
    )

    pair = validate_crossplay_pair(ar_path, independent_path)

    assert pair["autoregressive"]["path"] == str(ar_path.resolve())
    assert pair["independent"]["path"] == str(independent_path.resolve())
    assert (
        pair["autoregressive"]["metadata_sha256"]
        != pair["independent"]["metadata_sha256"]
    )


def test_runner_uses_two_trained_checkpoints_without_mode_override(
    tmp_path, monkeypatch
):
    ar_path = tmp_path / "trained-ar"
    independent_path = tmp_path / "trained-independent"
    checkpoints = {
        "autoregressive": {
            "path": str(ar_path),
            "metadata_sha256": "ar-digest",
            "metadata": {
                "policy_spec": {
                    "team_action_mode": "autoregressive",
                    "training_player_count": 4,
                }
            },
        },
        "independent": {
            "path": str(independent_path),
            "metadata_sha256": "independent-digest",
            "metadata": {
                "policy_spec": {
                    "team_action_mode": "independent",
                    "training_player_count": 4,
                }
            },
        },
    }
    monkeypatch.setattr(
        policy_crossplay,
        "validate_crossplay_pair",
        lambda *_args: checkpoints,
    )
    calls = []

    def fake_native(**kwargs):
        calls.append(kwargs)
        episodes = [
            _episode(user_score=10, opponent_score=8, starting_team="team_a"),
            _episode(user_score=9, opponent_score=9, starting_team="team_b"),
        ]
        kwargs["batch_result_callback"](episodes, 2, 2)
        return {
            "results": episodes,
            "per_player_stats": {
                player_id: {
                    "shots": 1,
                    "makes": 1,
                    "assists": 1,
                    "potential_assists": 1,
                    "turnovers": 1,
                    "steals": 0,
                    "points": 2,
                    "offensive_rebounds": 0,
                    "defensive_rebounds": 0,
                }
                for player_id in range(8)
            },
            "eval_diagnostics": {
                "jax_native_summary": {
                    "mean_live_ball_handler_nearest_defender_distance": 1.0,
                    "live_ball_handler_pressure_rate": 0.5,
                    "mean_live_offense_nearest_defender_distance": 1.0,
                    "mean_live_unguarded_offense_fraction": 0.1,
                    "mean_live_team_centroid_distance": 2.0,
                }
            },
        }

    monkeypatch.setattr(
        policy_crossplay, "run_native_jax_evaluation", fake_native
    )

    result = policy_crossplay.run_policy_crossplay(
        autoregressive_checkpoint=ar_path,
        independent_checkpoint=independent_path,
        output_dir=tmp_path / "output",
        episodes=4,
        batch_size=2,
        bootstrap_samples=10,
        printer=lambda _message: None,
    )

    assert [call["user_team_name"] for call in calls] == ["OFFENSE", "DEFENSE"]
    assert all(call["unified_policy_path"] == str(ar_path) for call in calls)
    assert all(
        call["opponent_policy_path"] == str(independent_path) for call in calls
    )
    assert all("team_action_mode" not in call for call in calls)
    assert result["summary_payload"]["matched_scenarios"][
        "complete_four_game_block_count"
    ] == 1
    assert (tmp_path / "output" / "games.jsonl").is_file()
