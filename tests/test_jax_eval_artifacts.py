import gzip
import hashlib
import json

import pytest

from basketworld_jax.eval.artifacts import (
    compact_existing_historical_artifact,
    compact_historical_payload,
    write_historical_artifact,
)
from app.training_backend.historical_elo import historical_elo_summary


def payload():
    return {"schema_version": 3, "milestone_updates": [100, 250], "matches": [{
        "candidate_update": 250, "opponent_update": 100, "action_mode": "deterministic",
        "candidate_wins": 7, "candidate_ties": 1, "candidate_losses": 2,
        "completed_episode_count": 10, "truncated_episode_count": 1,
        "mean_candidate_point_differential": 3.0, "seed": 49, "horizon": 2048,
        "clock_metrics_completed_games": {"game_count": 10},
        "clock_metrics_truncated_games": {"mean_terminal_game_clock_seconds_remaining": 70},
        "environment_contract": {"episode_end_mode": "game_clock"},
        "scoreline_histogram": {"7:2": 1},
        "clock_events": [{"type": "check_pickup", "elapsed": 4}] * 500,
        "episode_diagnostics": [{"outcome": "truncated", "remaining": 70}],
    }]}


def test_summary_and_gzip_preserve_aggregates_and_full_diagnostics(tmp_path):
    original = payload()
    path = tmp_path / "update_0000250.json"
    artifacts = write_historical_artifact(path, original, save_details=True)
    saved = json.loads(path.read_text())
    expected = compact_historical_payload(original)
    assert saved["matches"] == expected["matches"]
    assert saved["matches"][0]["clock_metrics_truncated_games"] == original["matches"][0]["clock_metrics_truncated_games"]
    assert "clock_events" in original["matches"][0], "Input must not be mutated"
    with gzip.open(tmp_path / saved["full_results_artifact"], "rt") as handle:
        assert json.load(handle) == original
    assert artifacts == [path, tmp_path / saved["full_results_artifact"]]
    assert path.stat().st_size < len(json.dumps(original)) / 5


def test_existing_migration_is_lossless_idempotent_and_preserves_elo(tmp_path):
    path = tmp_path / "historical_opponent_evaluation.json"
    raw = (json.dumps(payload(), indent=2) + "\n").encode()
    path.write_bytes(raw)
    before = historical_elo_summary(path)
    result = compact_existing_historical_artifact(path)
    assert result["changed"]
    saved = json.loads(path.read_text())
    assert saved["original_sha256"] == hashlib.sha256(raw).hexdigest()
    with gzip.open(result["archive"], "rb") as handle:
        assert handle.read() == raw
    after = historical_elo_summary(path)
    for field in ("latest", "points", "matchup_win_scores", "summary", "warnings"):
        assert after[field] == before[field]
    assert compact_existing_historical_artifact(path)["changed"] is False


def test_corrupt_archive_prevents_replacement(tmp_path):
    path = tmp_path / "historical_opponent_evaluation.json"
    raw = json.dumps(payload()).encode()
    path.write_bytes(raw)
    checksum = hashlib.sha256(raw).hexdigest()
    archive = path.with_name(f"{path.name}.original-{checksum[:16]}.gz")
    with gzip.open(archive, "wb") as handle:
        handle.write(b"wrong original")
    with pytest.raises(ValueError, match="verification failed"):
        compact_existing_historical_artifact(path)
    assert path.read_bytes() == raw


def test_cumulative_index_does_not_duplicate_details(tmp_path):
    path = tmp_path / "historical_opponent_evaluation.json"
    assert write_historical_artifact(path, payload(), save_details=False) == [path]
    assert json.loads(path.read_text()) == compact_historical_payload(payload())
    assert not list(tmp_path.glob("*.gz"))


def test_trainer_persists_and_logs_both_update_artifacts_and_resumes_compact(tmp_path):
    from basketworld_jax.train.main import (
        _load_historical_eval_artifact,
        _persist_historical_eval_update_artifact,
    )
    class Mlflow:
        def __init__(self): self.logged = []
        def log_artifact(self, path, artifact_path): self.logged.append((path, artifact_path))
    mlflow = Mlflow()
    _persist_historical_eval_update_artifact(
        checkpoint_dir=str(tmp_path), mlflow=mlflow, candidate_update=250, payload=payload(),
    )
    assert len(mlflow.logged) == 2
    assert mlflow.logged[0][0].endswith(".full.json.gz")
    assert mlflow.logged[1][0].endswith(".json")
    legacy = tmp_path / "historical_opponent_evaluation.json"
    legacy.write_text(json.dumps(payload()))
    restored = _load_historical_eval_artifact(checkpoint_root=tmp_path, continuation_run_id="", continue_cache_dir="")
    assert restored == compact_historical_payload(payload())
