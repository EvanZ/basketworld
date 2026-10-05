"""Offline Elo summaries for JAX historical-opponent evaluation artifacts."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import minimize


DEFAULT_INITIAL_RATING = 1000.0
DEFAULT_K_FACTOR = 32.0
DEFAULT_PRIOR_STDDEV = 800.0
DEFAULT_METHOD = "batch"
DETERMINISTIC_ACTION_MODE = "deterministic"
ELO_LOGISTIC_SCALE = math.log(10.0) / 400.0


def expected_score(rating_a: float, rating_b: float) -> float:
    """Return the standard Elo expected score for competitor A."""
    return 1.0 / (1.0 + (10.0 ** ((float(rating_b) - float(rating_a)) / 400.0)))


def _strict_int(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("boolean is not an integer field")
    parsed = int(value)
    if isinstance(value, float) and not value.is_integer():
        raise ValueError("non-integral value")
    if isinstance(value, str) and str(parsed) != value.strip():
        raise ValueError("non-canonical integer string")
    return parsed


def _finite_float(value: Any) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite float")
    return parsed


def _known_updates(payload: dict[str, Any]) -> set[int]:
    updates: set[int] = set()
    for value in payload.get("milestone_updates", []) or []:
        try:
            updates.add(_strict_int(value))
        except (TypeError, ValueError):
            continue
    checkpoints = payload.get("milestone_checkpoints")
    if isinstance(checkpoints, dict):
        for key in checkpoints:
            try:
                updates.add(_strict_int(key))
            except (TypeError, ValueError):
                continue
    return updates


def _normalize_match(record: Any) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(record, dict):
        return None, "malformed_record"
    try:
        candidate_update = _strict_int(record["candidate_update"])
        opponent_update = _strict_int(record["opponent_update"])
    except (KeyError, TypeError, ValueError):
        return None, "malformed_updates"

    if str(record.get("action_mode", "")).strip().lower() != DETERMINISTIC_ACTION_MODE:
        return None, "non_deterministic"
    if candidate_update == opponent_update or bool(record.get("self_match", False)):
        return None, "self_match"

    try:
        completed = _strict_int(record["completed_episode_count"])
        wins = _strict_int(record["candidate_wins"])
        ties = _strict_int(record["candidate_ties"])
        losses = _strict_int(record["candidate_losses"])
        truncated = _strict_int(record.get("truncated_episode_count", 0))
    except (KeyError, TypeError, ValueError):
        return None, "malformed_outcomes"
    if min(completed, wins, ties, losses, truncated) < 0:
        return None, "malformed_outcomes"
    if completed == 0:
        return None, "truncated_only" if truncated > 0 else "no_completed_games"
    if wins + ties + losses != completed:
        return None, "inconsistent_outcome_counts"

    try:
        mean_point_differential = _finite_float(
            record["mean_candidate_point_differential"]
        )
    except (KeyError, TypeError, ValueError):
        return None, "malformed_point_differential"

    normalized = {
        "candidate_update": candidate_update,
        "opponent_update": opponent_update,
        "completed_games": completed,
        "candidate_wins": wins,
        "candidate_ties": ties,
        "candidate_losses": losses,
        "truncated_games": truncated,
        "mean_candidate_point_differential": mean_point_differential,
        "seed": record.get("seed"),
        "horizon": record.get("horizon"),
    }
    normalized["candidate_score"] = (wins + (0.5 * ties)) / completed
    return normalized, None


def _match_sort_key(match: dict[str, Any]) -> tuple[Any, ...]:
    return (
        match["candidate_update"],
        match["opponent_update"],
        str(match.get("seed", "")),
        str(match.get("horizon", "")),
        match["candidate_wins"],
        match["candidate_ties"],
        match["candidate_losses"],
        match["mean_candidate_point_differential"],
    )


def _fingerprint(match: dict[str, Any]) -> str:
    return json.dumps(match, sort_keys=True, separators=(",", ":"), default=str)


def _connected_components(
    updates: Iterable[int], matches: Iterable[dict[str, Any]]
) -> dict[int, int]:
    adjacency: dict[int, set[int]] = {int(update): set() for update in updates}
    for match in matches:
        candidate = int(match["candidate_update"])
        opponent = int(match["opponent_update"])
        adjacency.setdefault(candidate, set()).add(opponent)
        adjacency.setdefault(opponent, set()).add(candidate)

    components: dict[int, int] = {}
    for component_id, start in enumerate(sorted(adjacency), start=1):
        if start in components:
            continue
        queue: deque[int] = deque([start])
        components[start] = component_id
        while queue:
            current = queue.popleft()
            for neighbor in sorted(adjacency[current]):
                if neighbor not in components:
                    components[neighbor] = component_id
                    queue.append(neighbor)
    return components


def _filter_matches(
    payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], set[int], Counter[str], int]:
    raw_matches = payload.get("matches", [])
    if not isinstance(raw_matches, list):
        raise ValueError("Historical evaluation payload field 'matches' must be a list.")

    known_updates = _known_updates(payload)
    skip_reasons: Counter[str] = Counter()
    accepted: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    for record in raw_matches:
        normalized, reason = _normalize_match(record)
        if normalized is None:
            skip_reasons[str(reason or "unknown")] += 1
            continue
        known_updates.update(
            (normalized["candidate_update"], normalized["opponent_update"])
        )
        fingerprint = _fingerprint(normalized)
        if fingerprint in fingerprints:
            skip_reasons["duplicate_record"] += 1
            continue
        fingerprints.add(fingerprint)
        accepted.append(normalized)
    accepted.sort(key=_match_sort_key)
    return accepted, known_updates, skip_reasons, len(raw_matches)


def _accumulate_checkpoint_stats(
    updates: Iterable[int], matches: Iterable[dict[str, Any]]
) -> dict[int, dict[str, float | int]]:
    stats: dict[int, dict[str, float | int]] = defaultdict(
        lambda: {
            "matchup_count": 0,
            "completed_games": 0,
            "wins": 0,
            "ties": 0,
            "losses": 0,
            "point_differential_sum": 0.0,
        }
    )
    for update in updates:
        stats[int(update)]
    for match in matches:
        candidate = int(match["candidate_update"])
        opponent = int(match["opponent_update"])
        completed = int(match["completed_games"])
        point_differential_sum = (
            float(match["mean_candidate_point_differential"]) * completed
        )
        candidate_stats = stats[candidate]
        opponent_stats = stats[opponent]
        candidate_stats["matchup_count"] += 1
        opponent_stats["matchup_count"] += 1
        candidate_stats["completed_games"] += completed
        opponent_stats["completed_games"] += completed
        candidate_stats["wins"] += int(match["candidate_wins"])
        candidate_stats["ties"] += int(match["candidate_ties"])
        candidate_stats["losses"] += int(match["candidate_losses"])
        opponent_stats["wins"] += int(match["candidate_losses"])
        opponent_stats["ties"] += int(match["candidate_ties"])
        opponent_stats["losses"] += int(match["candidate_wins"])
        candidate_stats["point_differential_sum"] += point_differential_sum
        opponent_stats["point_differential_sum"] -= point_differential_sum
    return stats


def _fit_batch_ratings(
    updates: list[int],
    matches: list[dict[str, Any]],
    *,
    initial_rating: float,
    prior_stddev: float,
) -> tuple[dict[int, float], dict[int, float], list[dict[str, Any]], dict[str, Any]]:
    """Fit regularized Bradley-Terry ratings to aggregate W/L/T counts."""
    if not updates:
        return {}, {}, [], {
            "success": True,
            "iterations": 0,
            "objective": 0.0,
            "message": "No checkpoints found.",
        }
    update_to_index = {update: index for index, update in enumerate(updates)}
    candidate_indices = np.asarray(
        [update_to_index[int(match["candidate_update"])] for match in matches],
        dtype=np.int64,
    )
    opponent_indices = np.asarray(
        [update_to_index[int(match["opponent_update"])] for match in matches],
        dtype=np.int64,
    )
    games = np.asarray([int(match["completed_games"]) for match in matches], dtype=float)
    score_points = np.asarray(
        [
            int(match["candidate_wins"]) + (0.5 * int(match["candidate_ties"]))
            for match in matches
        ],
        dtype=float,
    )
    prior_precision = 1.0 / (prior_stddev * prior_stddev)

    def objective(offsets: np.ndarray) -> tuple[float, np.ndarray]:
        if matches:
            logits = ELO_LOGISTIC_SCALE * (
                offsets[candidate_indices] - offsets[opponent_indices]
            )
            loss = float(
                np.sum(games * np.logaddexp(0.0, logits) - score_points * logits)
            )
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            residuals = games * probabilities - score_points
            gradient = np.zeros_like(offsets)
            np.add.at(
                gradient,
                candidate_indices,
                ELO_LOGISTIC_SCALE * residuals,
            )
            np.add.at(
                gradient,
                opponent_indices,
                -ELO_LOGISTIC_SCALE * residuals,
            )
        else:
            loss = 0.0
            gradient = np.zeros_like(offsets)
        loss += 0.5 * prior_precision * float(np.dot(offsets, offsets))
        gradient += prior_precision * offsets
        return loss, gradient

    fit = minimize(
        objective,
        np.zeros(len(updates), dtype=float),
        method="L-BFGS-B",
        jac=True,
        options={"ftol": 1.0e-12, "gtol": 1.0e-9, "maxiter": 10_000},
    )
    if not bool(fit.success):
        raise RuntimeError(f"Batch Elo fit failed: {fit.message}")
    offsets = np.asarray(fit.x, dtype=float)
    # Numerical optimization can leave a tiny common offset even though the
    # symmetric prior anchors the population. Center it exactly for stable output.
    offsets -= float(np.mean(offsets))
    ratings = {
        update: float(initial_rating + offsets[index])
        for index, update in enumerate(updates)
    }

    hessian = np.eye(len(updates), dtype=float) * prior_precision
    if matches:
        logits = ELO_LOGISTIC_SCALE * (
            offsets[candidate_indices] - offsets[opponent_indices]
        )
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        weights = (
            games
            * probabilities
            * (1.0 - probabilities)
            * (ELO_LOGISTIC_SCALE**2)
        )
        for candidate_index, opponent_index, weight in zip(
            candidate_indices, opponent_indices, weights
        ):
            hessian[candidate_index, candidate_index] += weight
            hessian[opponent_index, opponent_index] += weight
            hessian[candidate_index, opponent_index] -= weight
            hessian[opponent_index, candidate_index] -= weight
    covariance = np.linalg.inv(hessian)
    # Ratings are reported after fixing their population mean exactly. Project
    # the posterior covariance into that same zero-sum subspace so intervals
    # describe relative ladder strength rather than uncertainty in a common,
    # behaviorally meaningless offset.
    projection = np.eye(len(updates), dtype=float) - (
        np.ones((len(updates), len(updates)), dtype=float) / float(len(updates))
    )
    covariance = projection @ covariance @ projection.T
    standard_errors = {
        update: float(math.sqrt(max(0.0, covariance[index, index])))
        for index, update in enumerate(updates)
    }
    processed = []
    for match in matches:
        candidate = int(match["candidate_update"])
        opponent = int(match["opponent_update"])
        fitted_score = expected_score(ratings[candidate], ratings[opponent])
        processed.append(
            {
                **match,
                "fitted_candidate_score": fitted_score,
                "score_residual": float(match["candidate_score"]) - fitted_score,
            }
        )
    diagnostics = {
        "success": True,
        "iterations": int(getattr(fit, "nit", 0)),
        "objective": float(fit.fun),
        "gradient_max_abs": float(np.max(np.abs(fit.jac))) if len(fit.jac) else 0.0,
        "message": str(fit.message),
    }
    return ratings, standard_errors, processed, diagnostics


def _fit_chronological_ratings(
    updates: list[int],
    matches: list[dict[str, Any]],
    *,
    initial_rating: float,
    k_factor: float,
) -> tuple[dict[int, float], dict[int, None], list[dict[str, Any]], dict[str, Any]]:
    ratings = {update: float(initial_rating) for update in updates}
    processed: list[dict[str, Any]] = []
    for match in matches:
        candidate = int(match["candidate_update"])
        opponent = int(match["opponent_update"])
        rating_candidate_before = ratings[candidate]
        rating_opponent_before = ratings[opponent]
        expected_candidate = expected_score(
            rating_candidate_before, rating_opponent_before
        )
        delta = k_factor * (float(match["candidate_score"]) - expected_candidate)
        ratings[candidate] = rating_candidate_before + delta
        ratings[opponent] = rating_opponent_before - delta
        processed.append(
            {
                **match,
                "expected_candidate_score": expected_candidate,
                "elo_delta_candidate": delta,
                "candidate_rating_before": rating_candidate_before,
                "candidate_rating_after": ratings[candidate],
                "opponent_rating_before": rating_opponent_before,
                "opponent_rating_after": ratings[opponent],
            }
        )
    return ratings, {update: None for update in updates}, processed, {
        "success": True,
        "iterations": len(matches),
        "objective": None,
        "message": "Chronological aggregate Elo updates.",
    }


def build_deterministic_elo_ladder(
    payload: dict[str, Any],
    *,
    method: str = DEFAULT_METHOD,
    initial_rating: float = DEFAULT_INITIAL_RATING,
    k_factor: float = DEFAULT_K_FACTOR,
    prior_stddev: float = DEFAULT_PRIOR_STDDEV,
    anchor_update: int | None = None,
    anchor_rating: float | None = None,
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic checkpoint ladder from historical evaluations.

    The default ``batch`` method fits a regularized Bradley-Terry model to every
    completed W/L/T outcome. Ties contribute half a win and half a loss. The
    legacy-compatible ``chronological`` method treats each aggregate evaluation
    record as one Elo result and updates ratings using ``k_factor``.
    """
    if not isinstance(payload, dict):
        raise TypeError("Historical evaluation payload must be a JSON object.")
    method = str(method).strip().lower()
    if method not in {"batch", "chronological"}:
        raise ValueError("method must be 'batch' or 'chronological'.")
    initial_rating = _finite_float(initial_rating)
    k_factor = _finite_float(k_factor)
    prior_stddev = _finite_float(prior_stddev)
    if k_factor <= 0.0:
        raise ValueError("k_factor must be greater than zero.")
    if prior_stddev <= 0.0:
        raise ValueError("prior_stddev must be greater than zero.")
    if (anchor_update is None) != (anchor_rating is None):
        raise ValueError("anchor_update and anchor_rating must be provided together.")
    if anchor_update is not None:
        anchor_update = _strict_int(anchor_update)
        anchor_rating = _finite_float(anchor_rating)

    accepted, known_updates, skip_reasons, input_record_count = _filter_matches(payload)
    updates = sorted(known_updates)
    stats = _accumulate_checkpoint_stats(updates, accepted)
    if method == "batch":
        ratings, standard_errors, processed_matches, fit_diagnostics = (
            _fit_batch_ratings(
                updates,
                accepted,
                initial_rating=initial_rating,
                prior_stddev=prior_stddev,
            )
        )
    else:
        ratings, standard_errors, processed_matches, fit_diagnostics = (
            _fit_chronological_ratings(
                updates,
                accepted,
                initial_rating=initial_rating,
                k_factor=k_factor,
            )
        )

    rating_offset = 0.0
    if anchor_update is not None:
        if anchor_update not in ratings:
            raise ValueError(
                f"anchor_update {anchor_update} is not present in the rating pool."
            )
        rating_offset = float(anchor_rating) - ratings[anchor_update]
        ratings = {
            update: float(rating + rating_offset)
            for update, rating in ratings.items()
        }

    components = _connected_components(ratings, accepted)
    ordered_updates = sorted(ratings, key=lambda update: (-ratings[update], update))
    rankings: list[dict[str, Any]] = []
    for rank, update in enumerate(ordered_updates, start=1):
        update_stats = stats[update]
        completed_games = int(update_stats["completed_games"])
        win_score = (
            (
                float(update_stats["wins"])
                + (0.5 * float(update_stats["ties"]))
            )
            / completed_games
            if completed_games
            else None
        )
        mean_point_differential = (
            float(update_stats["point_differential_sum"]) / completed_games
            if completed_games
            else None
        )
        standard_error = standard_errors.get(update)
        rankings.append(
            {
                "rank": rank,
                "update": int(update),
                "elo_rating": float(ratings[update]),
                "rating_change": float(ratings[update] - initial_rating),
                "rating_standard_error": standard_error,
                "rating_ci95_low": (
                    float(ratings[update] - (1.96 * standard_error))
                    if standard_error is not None
                    else None
                ),
                "rating_ci95_high": (
                    float(ratings[update] + (1.96 * standard_error))
                    if standard_error is not None
                    else None
                ),
                "matchup_count": int(update_stats["matchup_count"]),
                "completed_games": completed_games,
                "wins": int(update_stats["wins"]),
                "ties": int(update_stats["ties"]),
                "losses": int(update_stats["losses"]),
                "win_score": win_score,
                "mean_point_differential": mean_point_differential,
                "connected_component": int(components.get(update, 0)),
            }
        )

    component_count = len(set(components.values())) if components else 0
    return {
        "schema_version": 2,
        "source": dict(source or {}),
        "config": {
            "action_mode": DETERMINISTIC_ACTION_MODE,
            "method": method,
            "initial_rating": initial_rating,
            "k_factor": k_factor if method == "chronological" else None,
            "prior_stddev": prior_stddev if method == "batch" else None,
            "anchor_update": anchor_update,
            "anchor_rating": anchor_rating,
            "rating_offset": rating_offset,
            "match_unit": (
                "completed_episode_outcomes"
                if method == "batch"
                else "paired_aggregate_evaluation"
            ),
            "stable_order": "candidate_update, opponent_update, seed, horizon, outcomes",
            "tie_score": 0.5,
        },
        "fit": fit_diagnostics,
        "summary": {
            "input_record_count": input_record_count,
            "accepted_matchup_count": len(accepted),
            "accepted_completed_game_count": int(
                sum(int(match["completed_games"]) for match in accepted)
            ),
            "skipped_record_count": int(sum(skip_reasons.values())),
            "skip_reasons": dict(sorted(skip_reasons.items())),
            "ranked_checkpoint_count": len(rankings),
            "connected_component_count": component_count,
            "ratings_globally_comparable": component_count <= 1,
        },
        "rankings": rankings,
        "processed_matches": processed_matches,
    }


RANKING_CSV_FIELDS = (
    "rank",
    "update",
    "elo_rating",
    "rating_change",
    "rating_standard_error",
    "rating_ci95_low",
    "rating_ci95_high",
    "matchup_count",
    "completed_games",
    "wins",
    "ties",
    "losses",
    "win_score",
    "mean_point_differential",
    "connected_component",
)


def write_elo_outputs(
    result: dict[str, Any],
    output_dir: str | Path,
    *,
    stem: str = "deterministic_elo",
) -> tuple[Path, Path]:
    """Write deterministic JSON and CSV ladder artifacts."""
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    json_path = target_dir / f"{stem}.json"
    csv_path = target_dir / f"{stem}.csv"
    json_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RANKING_CSV_FIELDS)
        writer.writeheader()
        for row in result.get("rankings", []):
            writer.writerow({key: row.get(key) for key in RANKING_CSV_FIELDS})
    return json_path, csv_path
