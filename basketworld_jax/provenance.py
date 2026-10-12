"""Small, dependency-free source provenance helpers for durable checkpoints.

The fingerprint intentionally covers only tracked Git changes.  It is an
identity aid for a checkpoint, not a claim that a matching checkout also has
matching Python, JAX, CUDA, or other runtime dependencies.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import subprocess
from typing import Any


PROVENANCE_SCHEMA_VERSION = 1

# These values describe the invocation or storage location rather than the
# policy/environment being restored.  ``num_updates`` is deliberately allowed
# to change for a completed-run continuation.
RESUME_CONFIG_EXCLUDED_KEYS = frozenset(
    {
        "checkpoint_dir",
        "continue_artifact",
        "continue_cache_dir",
        "continue_opponent_pool_size",
        "continue_preserve_env_state",
        "continue_preserve_intent_discriminator_state",
        "continue_run_id",
        "control_file",
        "dump_config_schema",
        "historical_eval_updates",
        "mlflow_resume_run_id",
        "num_updates",
        "resume_checkpoint",
        "training_status_file",
    }
)


def _git(repo_root: Path, *args: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), *args],
        stderr=subprocess.STDOUT,
        timeout=10,
    )


def capture_source_provenance(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Return a compact, reproducible identity for the tracked checkout.

    No diff contents are retained: when the tracked worktree is dirty, only a
    SHA-256 of ``git diff HEAD`` is stored.  Failures are represented in the
    returned record so callers can surface them and require deliberate action
    instead of silently claiming an exact source match.
    """

    root = Path(repo_root or Path.cwd()).expanduser().resolve()
    try:
        commit_sha = _git(root, "rev-parse", "HEAD").decode("ascii").strip()
        diff = _git(root, "diff", "--no-ext-diff", "--binary", "HEAD", "--")
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return {
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "available": False,
            "repository_root": str(root),
            "error": f"Unable to inspect Git source provenance: {exc}",
        }

    dirty = bool(diff)
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "available": True,
        "repository_root": str(root),
        "commit_sha": commit_sha,
        "worktree_dirty": dirty,
        "tracked_diff_sha256": sha256(diff).hexdigest() if dirty else None,
    }


def compare_source_provenance(
    checkpoint: dict[str, Any] | None,
    current: dict[str, Any] | None,
) -> dict[str, Any]:
    """Describe whether a checkpoint source identity matches this checkout."""

    saved = dict(checkpoint or {})
    observed = dict(current or {})
    if not saved:
        return {
            "status": "legacy",
            "exact": False,
            "requires_confirmation": True,
            "message": (
                "This checkpoint predates source provenance, so its checkout "
                "cannot be verified. Explicit confirmation is required."
            ),
            "checkpoint": saved,
            "current": observed,
        }
    if not saved.get("available", False):
        return {
            "status": "unavailable",
            "exact": False,
            "requires_confirmation": True,
            "message": (
                "The checkpoint's source provenance was unavailable when it was "
                "written. Explicit confirmation is required."
            ),
            "checkpoint": saved,
            "current": observed,
        }
    if not observed.get("available", False):
        return {
            "status": "unavailable",
            "exact": False,
            "requires_confirmation": True,
            "message": (
                "The current checkout's source provenance cannot be inspected. "
                "Explicit confirmation is required."
            ),
            "checkpoint": saved,
            "current": observed,
        }

    differences: list[str] = []
    if saved.get("commit_sha") != observed.get("commit_sha"):
        differences.append("Git commit differs")
    if bool(saved.get("worktree_dirty")) != bool(observed.get("worktree_dirty")):
        differences.append("tracked worktree cleanliness differs")
    elif bool(saved.get("worktree_dirty")) and (
        saved.get("tracked_diff_sha256") != observed.get("tracked_diff_sha256")
    ):
        differences.append("tracked worktree diff differs")

    if not differences:
        return {
            "status": "exact",
            "exact": True,
            "requires_confirmation": False,
            "message": "Checkpoint source matches the current tracked checkout.",
            "checkpoint": saved,
            "current": observed,
        }
    return {
        "status": "mismatch",
        "exact": False,
        "requires_confirmation": True,
        "message": (
            "Checkpoint source does not exactly match the current checkout: "
            + "; ".join(differences)
            + ". Explicit confirmation is required."
        ),
        "differences": differences,
        "checkpoint": saved,
        "current": observed,
    }


def resume_config_snapshot(values: dict[str, Any]) -> dict[str, Any]:
    """Return the resolved settings that must agree before state restore."""

    return {
        str(key): value
        for key, value in sorted(values.items())
        if str(key) not in RESUME_CONFIG_EXCLUDED_KEYS
    }
