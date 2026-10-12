from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import HTTPException

from app.backend.routes import skill_ratings_routes


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _report_with_artifacts(tmp_path: Path, *, recorded: dict | None) -> tuple[Path, dict]:
    raw_root = tmp_path / "artifacts" / "fixed_pool_studies" / "study-a"
    pool_path = raw_root / "pool_manifest.json"
    schedule_path = raw_root / "schedule_manifest.json"
    _write_json(pool_path, {"players": [{"player_id": "one"}], "pool_seed": 11})
    _write_json(schedule_path, {"games": [{"game_id": "one"}], "schedule_seed": 12})
    report_path = tmp_path / "artifacts" / "fixed_pool_reports" / "study-a" / "checkpoint_report.json"
    report = {
        "checkpoint": {"path": str(tmp_path / "checkpoint")},
        "action_mode": "direct",
        "execution": {
            "requested_batch_size": 64,
            **({"reproducibility": recorded} if recorded is not None else {}),
        },
        "raw_artifact_dependencies": {
            "root": str(raw_root),
            "pool_manifest": str(pool_path),
            "schedule_manifest": str(schedule_path),
        },
    }
    _write_json(report_path, report)
    return report_path, report


def test_report_reproducibility_preserves_all_recorded_seeds(tmp_path, monkeypatch):
    monkeypatch.setattr(skill_ratings_routes, "REPO_ROOT", tmp_path)
    report_path, report = _report_with_artifacts(
        tmp_path,
        recorded={
            "eval_seed": 13,
            "split_seed": 14,
            "bootstrap_seed": 15,
            "bootstrap_samples": 200,
        },
    )

    reproducibility = skill_ratings_routes._report_reproducibility(report_path, report)

    assert reproducibility is not None
    assert reproducibility["seeds"] == {
        "pool_seed": 11,
        "schedule_seed": 12,
        "evaluation_seed": 13,
        "split_seed": 14,
        "bootstrap_seed": 15,
    }
    assert reproducibility["missing_seed_keys"] == []


def test_extension_rejects_legacy_reports_with_unrecorded_seeds(tmp_path, monkeypatch):
    monkeypatch.setattr(skill_ratings_routes, "REPO_ROOT", tmp_path)
    report_path, _ = _report_with_artifacts(tmp_path, recorded=None)

    with pytest.raises(HTTPException, match="did not record"):
        skill_ratings_routes.extend_skill_ratings_study(
            skill_ratings_routes.SkillRatingsExtensionRequest(
                report_path=str(report_path),
                additional_games=10,
            )
        )
