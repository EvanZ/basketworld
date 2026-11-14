"""Small historical-evaluation summaries with lossless compressed detail."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from typing import Any


DETAIL_FIELDS = frozenset({"clock_events", "episode_diagnostics"})


def compact_historical_match(record: dict[str, Any]) -> dict[str, Any]:
    """Keep every aggregate, rule, seed and outcome; separate only raw rows."""
    return {key: value for key, value in record.items() if key not in DETAIL_FIELDS}


def compact_historical_payload(payload: dict[str, Any]) -> dict[str, Any]:
    matches = payload.get("matches")
    if not isinstance(matches, list) or any(not isinstance(row, dict) for row in matches):
        raise ValueError("Historical evaluation matches must be a list of objects.")
    return {**payload, "matches": [compact_historical_match(row) for row in matches]}


def _atomic_json(path: Path, payload: dict[str, Any], *, compressed: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        opener = gzip.open if compressed else open
        options = {"compresslevel": 1} if compressed else {}
        with opener(temporary_path, "wt", encoding="utf-8", **options) as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":") if compressed else None,
                      indent=None if compressed else 2)
            handle.write("\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_historical_artifact(path: str | Path, payload: dict[str, Any], *, save_details: bool) -> list[Path]:
    """Publish details first, then atomically expose their small summary."""
    path = Path(path)
    summary = compact_historical_payload(payload)
    artifacts = []
    if save_details and any(DETAIL_FIELDS.intersection(row) for row in payload["matches"]):
        detail_path = path.with_suffix(".full.json.gz")
        _atomic_json(detail_path, payload, compressed=True)
        summary["full_results_artifact"] = detail_path.name
        artifacts.append(detail_path)
    _atomic_json(path, summary)
    return [path, *artifacts]


def compact_existing_historical_artifact(path: str | Path) -> dict[str, Any]:
    """Migrate a legacy file, preserving its exact bytes in a verified gzip.

    Refuse replacement if a live writer changed the source during migration.
    Existing compressed originals are never overwritten.
    """
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Refusing to replace a symlink.")
    before = path.stat()
    signature = (before.st_ino, before.st_size, before.st_mtime_ns)
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    summary = compact_historical_payload(payload)
    if not any(DETAIL_FIELDS.intersection(row) for row in payload["matches"]):
        return {"path": str(path), "changed": False, "before_bytes": before.st_size}
    del payload
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    checksum = digest.hexdigest()
    archive = path.with_name(f"{path.name}.original-{checksum[:16]}.gz")
    if not archive.exists():
        with path.open("rb") as source, archive.open("xb") as destination:
            with gzip.GzipFile(fileobj=destination, mode="wb", compresslevel=1, mtime=0) as compressed:
                shutil.copyfileobj(source, compressed, length=1024 * 1024)
    with gzip.open(archive, "rb") as handle:
        archived_checksum = hashlib.file_digest(handle, "sha256").hexdigest()
    if archived_checksum != checksum:
        raise ValueError(f"Original archive verification failed: {archive}")
    after = path.stat()
    if (after.st_ino, after.st_size, after.st_mtime_ns) != signature:
        raise RuntimeError(f"Source changed during compaction; left it untouched: {path}")
    summary["full_results_artifact"] = archive.name
    summary["original_sha256"] = checksum
    _atomic_json(path, summary)
    return {"path": str(path), "changed": True, "before_bytes": before.st_size,
            "after_bytes": path.stat().st_size, "archive_bytes": archive.stat().st_size,
            "archive": str(archive), "original_sha256": checksum}
