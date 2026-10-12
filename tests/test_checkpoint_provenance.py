from __future__ import annotations

from basketworld_jax.checkpoints import build_checkpoint_payload
from basketworld_jax.provenance import (
    compare_source_provenance,
    resume_config_snapshot,
)


def test_checkpoint_payload_keeps_compact_source_and_resolved_config_metadata():
    provenance = {
        "schema_version": 1,
        "available": True,
        "commit_sha": "abc123",
        "worktree_dirty": True,
        "tracked_diff_sha256": "digest",
    }
    payload = build_checkpoint_payload(
        update_index=1,
        trainer_config={},
        policy_spec={},
        frozen_config={},
        params={},
        opt_state={},
        current_state={},
        eval_initial_state={},
        base_key=None,
        eval_trajectories=[],
        last_metrics=None,
        source_provenance=provenance,
        resolved_training_config={"policy_model": "attention"},
    )

    assert payload["source_provenance"] == provenance
    assert payload["resolved_training_config"] == {"policy_model": "attention"}


def test_provenance_comparison_requires_confirmation_for_changed_dirty_diff():
    checkpoint = {
        "available": True,
        "commit_sha": "abc123",
        "worktree_dirty": True,
        "tracked_diff_sha256": "before",
    }
    current = {**checkpoint, "tracked_diff_sha256": "after"}

    comparison = compare_source_provenance(checkpoint, current)

    assert comparison["status"] == "mismatch"
    assert comparison["requires_confirmation"] is True
    assert comparison["differences"] == ["tracked worktree diff differs"]


def test_resume_config_snapshot_excludes_target_and_runtime_paths():
    snapshot = resume_config_snapshot(
        {
            "policy_model": "attention",
            "num_updates": 31_000,
            "resume_checkpoint": "/tmp/checkpoint",
            "training_status_file": "/tmp/status.json",
        }
    )

    assert snapshot == {"policy_model": "attention"}
