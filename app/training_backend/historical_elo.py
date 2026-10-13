from __future__ import annotations

import copy
import json
from pathlib import Path
from threading import Lock
from typing import Any

from basketworld.utils.historical_eval_elo import build_deterministic_elo_ladder


HISTORICAL_EVAL_FILENAME = "historical_opponent_evaluation.json"
DEFAULT_ANCHOR_UPDATE = 100
DEFAULT_ANCHOR_RATING = 400.0

_cache_lock = Lock()
_summary_cache: dict[str, tuple[tuple[int, int], dict[str, Any]]] = {}


def clear_historical_elo_cache() -> None:
    """Clear cached summaries. Intended for tests and explicit invalidation."""
    with _cache_lock:
        _summary_cache.clear()


def _artifact_signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return int(stat.st_mtime_ns), int(stat.st_size)


def _waiting_summary() -> dict[str, Any]:
    return {
        "status": "waiting",
        "anchor_update": DEFAULT_ANCHOR_UPDATE,
        "anchor_rating": DEFAULT_ANCHOR_RATING,
        "latest": None,
        "points": [],
        "next_evaluation_update": None,
        "warnings": [],
        "summary": {
            "accepted_matchup_count": 0,
            "completed_game_count": 0,
            "truncated_game_count": 0,
            "connected_component_count": 0,
        },
        "source_signature": None,
    }


def _strict_milestone_updates(payload: dict[str, Any]) -> list[int]:
    updates: set[int] = set()
    for raw in payload.get("milestone_updates", []) or []:
        try:
            update = int(raw)
        except (TypeError, ValueError):
            continue
        if update > 0:
            updates.add(update)
    return sorted(updates)


def _scheduled_rule_configurations(payload: dict[str, Any]) -> set[tuple[int, int]]:
    """Return explicit shared rule configurations recorded for matchups."""
    configurations: set[tuple[int, int]] = set()
    for record in payload.get("matches", []) or []:
        if not isinstance(record, dict):
            continue
        try:
            configurations.add(
                (
                    int(record["evaluation_multi_possession_limit"]),
                    int(record["evaluation_check_clearance"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            # Older artifacts did not persist a scheduled rule context.
            continue
    return configurations


def _compact_summary(
    payload: dict[str, Any],
    *,
    source_path: Path,
    signature: tuple[int, int],
) -> dict[str, Any]:
    milestone_updates = _strict_milestone_updates(payload)
    known_updates = set(milestone_updates)
    for record in payload.get("matches", []) or []:
        if not isinstance(record, dict):
            continue
        for key in ("candidate_update", "opponent_update"):
            try:
                known_updates.add(int(record[key]))
            except (KeyError, TypeError, ValueError):
                continue

    anchor_update = (
        DEFAULT_ANCHOR_UPDATE if DEFAULT_ANCHOR_UPDATE in known_updates else None
    )
    result = build_deterministic_elo_ladder(
        payload,
        anchor_update=anchor_update,
        anchor_rating=(DEFAULT_ANCHOR_RATING if anchor_update is not None else None),
        source={"kind": "trainer_local_json", "path": str(source_path)},
    )

    evaluated = [
        dict(row)
        for row in result.get("rankings", [])
        if int(row.get("completed_games", 0) or 0) > 0
    ]
    rank_by_update = {
        int(row["update"]): rank
        for rank, row in enumerate(
            sorted(evaluated, key=lambda item: (-float(item["elo_rating"]), int(item["update"]))),
            start=1,
        )
    }
    points: list[dict[str, Any]] = []
    previous_rating: float | None = None
    for row in sorted(evaluated, key=lambda item: int(item["update"])):
        rating = float(row["elo_rating"])
        point = {
            "update": int(row["update"]),
            "elo_rating": rating,
            "elo_delta": (
                None if previous_rating is None else rating - previous_rating
            ),
            "rank": rank_by_update[int(row["update"])],
            "rating_ci95_low": row.get("rating_ci95_low"),
            "rating_ci95_high": row.get("rating_ci95_high"),
            "completed_games": int(row.get("completed_games", 0) or 0),
            "wins": int(row.get("wins", 0) or 0),
            "ties": int(row.get("ties", 0) or 0),
            "losses": int(row.get("losses", 0) or 0),
            "win_score": row.get("win_score"),
            "mean_point_differential": row.get("mean_point_differential"),
            "connected_component": int(row.get("connected_component", 0) or 0),
        }
        points.append(point)
        previous_rating = rating

    latest = points[-1] if points else None
    latest_update = int(latest["update"]) if latest else 0
    next_evaluation_update = next(
        (update for update in milestone_updates if update > latest_update),
        None,
    )
    processed = result.get("processed_matches", []) or []
    truncated_games = sum(
        int(match.get("truncated_games", 0) or 0) for match in processed
    )
    evaluated_components = {
        int(point["connected_component"])
        for point in points
        if int(point["connected_component"]) > 0
    }
    warnings: list[str] = []
    if truncated_games:
        warnings.append(
            f"{truncated_games} truncated evaluation episodes were excluded from Elo."
        )
    if len(evaluated_components) > 1:
        warnings.append(
            "Evaluated checkpoints span disconnected matchup components; compare ratings only within a component."
        )
    if len(_scheduled_rule_configurations(payload)) > 1:
        warnings.append(
            "Historical Elo spans multiple scheduled evaluation rule configurations; compare checkpoints within the same configuration."
        )
    if result.get("fit", {}).get("success") is False:
        warnings.append("The Elo optimizer did not report a successful fit.")

    result_summary = result.get("summary", {}) or {}
    return {
        "status": "ready" if points else "waiting",
        "anchor_update": anchor_update,
        "anchor_rating": (
            DEFAULT_ANCHOR_RATING if anchor_update is not None else None
        ),
        "latest": latest,
        "points": points,
        "next_evaluation_update": next_evaluation_update,
        "warnings": warnings,
        "summary": {
            "accepted_matchup_count": int(
                result_summary.get("accepted_matchup_count", 0) or 0
            ),
            "completed_game_count": int(
                result_summary.get("accepted_completed_game_count", 0) or 0
            ),
            "truncated_game_count": int(truncated_games),
            "connected_component_count": len(evaluated_components),
        },
        "source_signature": {
            "modified_ns": signature[0],
            "size_bytes": signature[1],
        },
    }


def historical_elo_summary(artifact_path: str | Path) -> dict[str, Any]:
    """Return a cached compact Elo summary for a trainer artifact."""
    path = Path(artifact_path).resolve()
    if not path.is_file():
        return _waiting_summary()
    signature = _artifact_signature(path)
    cache_key = str(path)
    with _cache_lock:
        cached = _summary_cache.get(cache_key)
        if cached is not None and cached[0] == signature:
            return copy.deepcopy(cached[1])

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Historical evaluation artifact must contain a JSON object.")
    summary = _compact_summary(payload, source_path=path, signature=signature)
    with _cache_lock:
        _summary_cache[cache_key] = (signature, summary)
    return copy.deepcopy(summary)
