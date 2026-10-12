"""Offline, paired-group-safe models for fixed-player-pool studies.

This module consumes only the raw artifacts written by ``fixed_pool_results``.
It intentionally knows nothing about live evaluation or MLflow, which makes
reports reproducible and prevents a fitted estimate from changing a study's
immutable outcomes or possession ledger.
"""

from __future__ import annotations

from collections import defaultdict
import itertools
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from basketworld_jax.eval.fixed_pool_results import (
    FIXED_POOL_EVENT_LEDGER_SCHEMA_VERSION,
    FIXED_POOL_STUDY_SCHEMA_VERSION,
    load_fixed_pool_study_artifacts,
)


FACTOR_FEATURE_SCHEMA_VERSION = 1
CONVENTIONAL_THREE_FACTOR_FEATURE_SCHEMA_VERSION = 1
_COLLINEAR_CONDITION_THRESHOLD = 1.0e8
# The grid is deliberately declared as part of the analysis, rather than tuned
# interactively.  It spans unregularized least squares through materially
# shrunk estimates on the standardized/player-incidence designs used below.
RIDGE_ALPHA_CANDIDATES = (0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
RIDGE_CV_FOLDS = 5


class FixedPoolModelError(ValueError):
    """A raw study cannot support the requested offline comparison or model."""


def fixed_pool_study_budget(skill_count: int) -> dict[str, int]:
    """Return the skill-scaled initial study budget recorded by issue #57."""
    if int(skill_count) < 1:
        raise FixedPoolModelError("A fixed-pool model needs at least one active skill.")
    seeded_players = max(10, 8 * int(skill_count))
    game_appearances_per_player = 100
    games = math.ceil(seeded_players * game_appearances_per_player / 10)
    return {
        "effective_skill_count": int(skill_count),
        "seeded_players": int(seeded_players),
        "game_appearances_per_player": int(game_appearances_per_player),
        "games": int(games),
    }


def _pearson_correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    """Return Pearson's r, or ``None`` when it is not defined."""
    if first.size < 2 or second.size < 2:
        return None
    first_centered = first - np.mean(first)
    second_centered = second - np.mean(second)
    denominator = math.sqrt(
        float(np.dot(first_centered, first_centered))
        * float(np.dot(second_centered, second_centered))
    )
    if denominator == 0.0:
        return None
    return float(np.dot(first_centered, second_centered) / denominator)


def _average_tie_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks without adding a SciPy dependency."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def skill_apm_agreement(
    direct_player_report: Mapping[str, Mapping[str, Any]],
    *,
    top_k: int = 10,
) -> dict[str, Any]:
    """Summarize agreement between direct skill ratings and diagnostic APM.

    APM is deliberately a comparator rather than the target of the direct model.
    These metrics make that comparison explicit without conflating it with either
    model's per-game prediction error.
    """
    comparable = []
    for player_id, player in sorted(direct_player_report.items()):
        skill_value = player.get("net_rating_points_vs_replacement")
        apm_value = player.get("apm_diagnostic_value")
        if skill_value is None or apm_value is None:
            continue
        skill_float = float(skill_value)
        apm_float = float(apm_value)
        if math.isfinite(skill_float) and math.isfinite(apm_float):
            comparable.append((str(player_id), skill_float, apm_float))

    player_count = len(comparable)
    overlap_k = min(max(int(top_k), 0), player_count)
    if player_count == 0:
        return {
            "sample_count": 0,
            "rating_pearson_correlation": None,
            "ranking_spearman_correlation": None,
            "overlap_k": overlap_k,
            "top_k_overlap_count": 0,
            "bottom_k_overlap_count": 0,
        }

    skill_values = np.asarray([row[1] for row in comparable], dtype=np.float64)
    apm_values = np.asarray([row[2] for row in comparable], dtype=np.float64)
    skill_ranks = _average_tie_ranks(skill_values)
    apm_ranks = _average_tie_ranks(apm_values)
    top_skill = {
        player_id
        for player_id, _, _ in sorted(
            comparable, key=lambda row: (-row[1], row[0])
        )[:overlap_k]
    }
    top_apm = {
        player_id
        for player_id, _, _ in sorted(
            comparable, key=lambda row: (-row[2], row[0])
        )[:overlap_k]
    }
    bottom_skill = {
        player_id
        for player_id, _, _ in sorted(comparable, key=lambda row: (row[1], row[0]))[
            :overlap_k
        ]
    }
    bottom_apm = {
        player_id
        for player_id, _, _ in sorted(comparable, key=lambda row: (row[2], row[0]))[
            :overlap_k
        ]
    }
    return {
        "sample_count": player_count,
        "rating_pearson_correlation": _pearson_correlation(skill_values, apm_values),
        "ranking_spearman_correlation": _pearson_correlation(skill_ranks, apm_ranks),
        "overlap_k": overlap_k,
        "top_k_overlap_count": len(top_skill & top_apm),
        "bottom_k_overlap_count": len(bottom_skill & bottom_apm),
    }


def _as_study(value: str | Path | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(value, (str, Path)):
        return load_fixed_pool_study_artifacts(value)
    if not isinstance(value, Mapping):
        raise FixedPoolModelError("Study input must be an artifact path or mapping.")
    return value


def _episode_action_mode(study: Mapping[str, Any]) -> str:
    episodes = study.get("episodes", [])
    modes = {str(row.get("action_mode")) for row in episodes if row.get("action_mode")}
    if len(modes) != 1:
        raise FixedPoolModelError(
            "Every fitted study must contain exactly one action mode; direct and "
            "sampled games may not be pooled."
        )
    return next(iter(modes))


def assert_studies_schema_compatible(studies: Sequence[Mapping[str, Any]]) -> None:
    """Refuse silent comparison across raw-artifact schemas or skill schemas."""
    if not studies:
        raise FixedPoolModelError("At least one fixed-pool study is required.")
    references: dict[str, Any] | None = None
    for raw_study in studies:
        study = _as_study(raw_study)
        manifest = study.get("manifest")
        if not isinstance(manifest, Mapping):
            raise FixedPoolModelError("Study is missing its versioned manifest.")
        episodes = study.get("episodes")
        if not isinstance(episodes, list) or not episodes:
            raise FixedPoolModelError("Study has no episode table rows.")
        checkpoint_schemas = {
            tuple(row.get("checkpoint_observation_schema_versions") or [])
            for row in episodes
        }
        environment_schemas = {
            row.get("environment_observation_schema_version") for row in episodes
        }
        if len(checkpoint_schemas) != 1 or len(environment_schemas) != 1:
            raise FixedPoolModelError(
                "Refusing a study with unreported mixed observation schemas."
            )
        observed = {
            "study_schema_version": int(manifest.get("schema_version", -1)),
            "event_ledger_schema_version": int(
                manifest.get("tables", {})
                .get("possession_events", {})
                .get("schema_version", -1)
            ),
            "skill_schema_hash": manifest.get("skill_schema_hash"),
            "checkpoint_observation_schema_versions": next(iter(checkpoint_schemas)),
            "environment_observation_schema_version": next(iter(environment_schemas)),
        }
        if observed["study_schema_version"] != FIXED_POOL_STUDY_SCHEMA_VERSION:
            raise FixedPoolModelError(
                "Unsupported fixed-pool study schema; migrate the raw artifact first."
            )
        if (
            observed["event_ledger_schema_version"]
            != FIXED_POOL_EVENT_LEDGER_SCHEMA_VERSION
        ):
            raise FixedPoolModelError(
                "Unsupported possession ledger schema; migrate the raw artifact first."
            )
        if references is None:
            references = observed
        elif observed != references:
            mismatches = [
                key
                for key, value in observed.items()
                if references is not None and references.get(key) != value
            ]
            raise FixedPoolModelError(
                "Refusing an unreported cross-schema comparison: "
                + ", ".join(mismatches)
                + "."
            )


def paired_group_split(
    pair_ids: Sequence[str],
    *,
    validation_fraction: float = 0.2,
    seed: int = 0,
) -> dict[str, Any]:
    """Split complete sampling groups without train/validation leakage.

    The legacy name remains for compatibility with schema-v2 paired studies.
    Schema-v3 independent schedules contribute one game per group.
    """
    normalized = np.asarray([str(value) for value in pair_ids], dtype=object)
    unique_pair_ids = np.asarray(sorted(set(normalized.tolist())), dtype=object)
    if unique_pair_ids.size == 0:
        raise FixedPoolModelError("Cannot split an empty sampling-group collection.")
    if not 0.0 < float(validation_fraction) < 1.0:
        raise FixedPoolModelError("validation_fraction must be between zero and one.")
    if unique_pair_ids.size < 2:
        train_indices = np.arange(normalized.size, dtype=np.int32)
        validation_indices = np.empty(0, dtype=np.int32)
        return {
            "train_indices": train_indices,
            "validation_indices": validation_indices,
            "train_pair_ids": unique_pair_ids.tolist(),
            "validation_pair_ids": [],
            "validation_available": False,
            "warning": "fewer_than_two_sampling_groups",
        }
    rng = np.random.default_rng(int(seed))
    shuffled = rng.permutation(unique_pair_ids)
    validation_count = min(
        unique_pair_ids.size - 1,
        max(1, int(round(unique_pair_ids.size * float(validation_fraction)))),
    )
    validation_pairs = set(shuffled[:validation_count].tolist())
    validation_mask = np.asarray(
        [pair_id in validation_pairs for pair_id in normalized], dtype=bool
    )
    train_indices = np.flatnonzero(~validation_mask).astype(np.int32)
    validation_indices = np.flatnonzero(validation_mask).astype(np.int32)
    train_pair_ids = sorted(set(normalized[train_indices].tolist()))
    validation_pair_ids = sorted(set(normalized[validation_indices].tolist()))
    if set(train_pair_ids) & set(validation_pair_ids):
        raise FixedPoolModelError("Sampling-group leakage in train/validation split.")
    return {
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "train_pair_ids": train_pair_ids,
        "validation_pair_ids": validation_pair_ids,
        "validation_available": True,
        "warning": None,
    }


def paired_group_kfold(
    pair_ids: Sequence[str], *, n_splits: int = RIDGE_CV_FOLDS, seed: int = 0
) -> dict[str, Any]:
    """Create deterministic, leakage-safe folds of whole sampling groups.

    Schema-v2 pairs share one group and stay together.  Schema-v3 independent
    schedules contribute one game per group. ``seed`` only chooses a stable
    fold assignment; repeating an analysis with the same raw study and seed
    produces byte-for-byte equivalent fold membership.
    """
    normalized = np.asarray([str(value) for value in pair_ids], dtype=object)
    unique_pair_ids = np.asarray(sorted(set(normalized.tolist())), dtype=object)
    requested_fold_count = int(n_splits)
    if requested_fold_count < 2:
        raise FixedPoolModelError("Ridge cross-validation requires at least two folds.")
    if unique_pair_ids.size < 2:
        return {
            "requested_fold_count": requested_fold_count,
            "fold_count": 0,
            "folds": [],
            "available": False,
            "warning": "fewer_than_two_sampling_groups",
        }
    fold_count = min(requested_fold_count, int(unique_pair_ids.size))
    shuffled = np.random.default_rng(int(seed)).permutation(unique_pair_ids)
    folds = []
    for fold_index, validation_pairs_array in enumerate(np.array_split(shuffled, fold_count)):
        validation_pairs = set(validation_pairs_array.tolist())
        validation_mask = np.asarray(
            [pair_id in validation_pairs for pair_id in normalized], dtype=bool
        )
        train_indices = np.flatnonzero(~validation_mask).astype(np.int32)
        validation_indices = np.flatnonzero(validation_mask).astype(np.int32)
        if train_indices.size == 0 or validation_indices.size == 0:
            raise FixedPoolModelError("Invalid empty ridge cross-validation fold.")
        folds.append(
            {
                "fold": int(fold_index),
                "train_indices": train_indices,
                "validation_indices": validation_indices,
                "train_pair_ids": sorted(set(normalized[train_indices].tolist())),
                "validation_pair_ids": sorted(
                    set(normalized[validation_indices].tolist())
                ),
            }
        )
    return {
        "requested_fold_count": requested_fold_count,
        "fold_count": fold_count,
        "folds": folds,
        "available": True,
        "warning": (
            None if fold_count == requested_fold_count else "fewer_than_requested_sampling_groups"
        ),
    }


def _manual_ridge_selection(alpha: float) -> dict[str, Any]:
    """Represent an explicit legacy override in the same report shape as CV."""
    if float(alpha) < 0.0:
        raise FixedPoolModelError("ridge alpha must be non-negative.")
    return {
        "method": "manual_override",
        "selected_alpha": float(alpha),
        "candidate_alphas": [float(alpha)],
        "requested_fold_count": 0,
        "fold_count": 0,
        "available": False,
        "warning": "manual_ridge_alpha_override",
        "candidates": [],
    }


def _select_grouped_ridge_alpha(
    group_ids: Sequence[str],
    target: Sequence[float],
    *,
    predict_fold,
    n_splits: int = RIDGE_CV_FOLDS,
    seed: int = 0,
    candidates: Sequence[float] = RIDGE_ALPHA_CANDIDATES,
) -> dict[str, Any]:
    """Select alpha by deterministic grouped K-fold prediction error.

    ``predict_fold(train_indices, validation_indices, alpha)`` must return one
    prediction per validation row.  This small abstraction keeps tuning tied
    to the exact preprocessing used by each model family.
    """
    y = np.asarray(target, dtype=np.float64)
    if y.ndim != 1 or y.shape[0] != len(group_ids):
        raise FixedPoolModelError("Ridge CV target and group IDs have incompatible rows.")
    if not np.all(np.isfinite(y)):
        raise FixedPoolModelError("Ridge CV target must contain only finite values.")
    fold_plan = paired_group_kfold(group_ids, n_splits=n_splits, seed=seed)
    normalized_candidates = tuple(sorted({float(value) for value in candidates}))
    if not normalized_candidates or normalized_candidates[0] < 0.0:
        raise FixedPoolModelError("Ridge CV candidates must be non-negative.")
    if not fold_plan["available"]:
        # There is no unbiased tuning signal with a single group.  OLS is the
        # transparent fallback, explicitly marked so it is never mistaken for
        # a data-selected regularization strength.
        return {
            "method": "grouped_k_fold_cross_validation",
            "selected_alpha": 0.0,
            "candidate_alphas": list(normalized_candidates),
            "requested_fold_count": int(fold_plan["requested_fold_count"]),
            "fold_count": 0,
            "available": False,
            "warning": fold_plan["warning"],
            "selection_metric": "mean_held_out_squared_error",
            "candidates": [],
        }
    candidate_reports = []
    for alpha in normalized_candidates:
        squared_error_sum = 0.0
        observation_count = 0
        fold_scores = []
        valid = True
        for fold in fold_plan["folds"]:
            validation_indices = np.asarray(fold["validation_indices"], dtype=np.int32)
            prediction = predict_fold(
                np.asarray(fold["train_indices"], dtype=np.int32),
                validation_indices,
                float(alpha),
            )
            if prediction is None:
                valid = False
                break
            prediction = np.asarray(prediction, dtype=np.float64)
            if prediction.shape != (validation_indices.size,) or not np.all(
                np.isfinite(prediction)
            ):
                valid = False
                break
            residual = y[validation_indices] - prediction
            squared_error = float(np.dot(residual, residual))
            squared_error_sum += squared_error
            observation_count += int(validation_indices.size)
            fold_scores.append(
                {
                    "fold": int(fold["fold"]),
                    "sample_count": int(validation_indices.size),
                    "mean_squared_error": float(
                        squared_error / max(1, validation_indices.size)
                    ),
                }
            )
        if valid and observation_count:
            mean_squared_error = squared_error_sum / observation_count
            candidate_reports.append(
                {
                    "alpha": float(alpha),
                    "mean_held_out_squared_error": float(mean_squared_error),
                    "mean_held_out_rmse": float(math.sqrt(mean_squared_error)),
                    "fold_scores": fold_scores,
                }
            )
    if not candidate_reports:
        raise FixedPoolModelError("No ridge alpha candidate could be cross-validated.")
    # Prefer stronger shrinkage only when two candidates score identically;
    # otherwise selection is strictly by out-of-fold error.
    selected = min(
        candidate_reports,
        key=lambda item: (item["mean_held_out_squared_error"], -item["alpha"]),
    )
    return {
        "method": "grouped_k_fold_cross_validation",
        "selected_alpha": float(selected["alpha"]),
        "candidate_alphas": list(normalized_candidates),
        "requested_fold_count": int(fold_plan["requested_fold_count"]),
        "fold_count": int(fold_plan["fold_count"]),
        "available": True,
        "warning": fold_plan["warning"],
        "selection_metric": "mean_held_out_squared_error",
        "candidates": candidate_reports,
    }


def _matrix_diagnostics(
    matrix: np.ndarray, *, structural_nullity: int = 0
) -> dict[str, Any]:
    if matrix.ndim != 2:
        raise FixedPoolModelError("Design matrix must have two dimensions.")
    structural_nullity = max(0, min(int(structural_nullity), int(matrix.shape[1])))
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return {
            "rank": 0,
            "column_count": int(matrix.shape[1]),
            "row_count": int(matrix.shape[0]),
            "condition_number": None,
            "condition_number_is_infinite": False,
            "condition_number_basis": "identified_subspace",
            "structural_nullity": structural_nullity,
            "unidentified_nullity": max(
                0, int(matrix.shape[1]) - structural_nullity
            ),
            "rank_deficient": True,
            "materially_collinear": True,
        }
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    rank = int(np.linalg.matrix_rank(matrix))
    expected_rank = min(
        int(matrix.shape[0]), int(matrix.shape[1]) - structural_nullity
    )
    unidentified_nullity = max(0, expected_rank - rank)
    largest = float(singular_values[0]) if singular_values.size else 0.0
    if unidentified_nullity:
        condition_number = math.inf
    else:
        # Exclude only the known gauge direction.  For example, every signed
        # five-versus-five player row sums to zero, so absolute player value is
        # intentionally unidentified while all values versus replacement are
        # estimable.
        smallest_identified = (
            float(singular_values[rank - 1]) if rank > 0 else 0.0
        )
        condition_number = (
            float(largest / smallest_identified)
            if smallest_identified > np.finfo(np.float64).eps
            else math.inf
        )
    return {
        "rank": rank,
        "column_count": int(matrix.shape[1]),
        "row_count": int(matrix.shape[0]),
        # JSON reports cannot represent IEEE infinity.  Preserve the useful
        # diagnostic explicitly instead of failing after an otherwise valid
        # study has written its immutable raw artifacts.
        "condition_number": (
            float(condition_number) if math.isfinite(condition_number) else None
        ),
        "condition_number_is_infinite": bool(not math.isfinite(condition_number)),
        "condition_number_basis": "identified_subspace",
        "structural_nullity": structural_nullity,
        "unidentified_nullity": int(unidentified_nullity),
        "rank_deficient": bool(unidentified_nullity > 0),
        "materially_collinear": bool(
            unidentified_nullity > 0
            or condition_number > _COLLINEAR_CONDITION_THRESHOLD
        ),
    }


def _ridge_fit(
    matrix: np.ndarray,
    target: np.ndarray,
    *,
    alpha: float,
    penalize: np.ndarray | None = None,
) -> np.ndarray:
    if float(alpha) < 0.0:
        raise FixedPoolModelError("ridge alpha must be non-negative.")
    if matrix.shape[0] != target.shape[0]:
        raise FixedPoolModelError("Design matrix and target have incompatible rows.")
    if matrix.shape[0] == 0:
        raise FixedPoolModelError("Cannot fit a model with no observations.")
    penalty = np.ones(matrix.shape[1], dtype=np.float64)
    if penalize is not None:
        penalty = np.asarray(penalize, dtype=np.float64)
        if penalty.shape != (matrix.shape[1],):
            raise FixedPoolModelError("Invalid ridge penalty mask.")
    gram = matrix.T @ matrix + float(alpha) * np.diag(penalty)
    rhs = matrix.T @ target
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(gram) @ rhs


def build_signed_player_design_matrix(
    episodes: Sequence[Mapping[str, Any]],
    *,
    player_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build signed Team-A-minus-Team-B player and known-design columns."""
    if not episodes:
        raise FixedPoolModelError("Cannot build a design matrix from zero games.")
    if player_ids is None:
        player_ids = sorted(
            {
                str(player_id)
                for row in episodes
                for key in ("team_a_player_ids", "team_b_player_ids")
                for player_id in row.get(key, [])
            }
        )
    player_ids = tuple(str(player_id) for player_id in player_ids)
    index = {player_id: position for position, player_id in enumerate(player_ids)}
    player_matrix = np.zeros((len(episodes), len(player_ids)), dtype=np.float64)
    starting_side = np.zeros(len(episodes), dtype=np.float64)
    overtime_round = np.zeros(len(episodes), dtype=np.float64)
    for row_index, row in enumerate(episodes):
        for player_id in row.get("team_a_player_ids", []):
            if str(player_id) not in index:
                raise FixedPoolModelError(f"Unknown Team A player {player_id!r}.")
            player_matrix[row_index, index[str(player_id)]] += 1.0
        for player_id in row.get("team_b_player_ids", []):
            if str(player_id) not in index:
                raise FixedPoolModelError(f"Unknown Team B player {player_id!r}.")
            player_matrix[row_index, index[str(player_id)]] -= 1.0
        starting_team = row.get("starting_team")
        if starting_team == "team_a":
            starting_side[row_index] = 1.0
        elif starting_team == "team_b":
            starting_side[row_index] = -1.0
        else:
            raise FixedPoolModelError("Episode is missing a Team A/B starting side.")
        overtime_round[row_index] = float(row.get("overtime_round", 0) or 0)
    design_matrix = np.column_stack(
        [
            player_matrix,
            np.ones(len(episodes), dtype=np.float64),
            starting_side,
            overtime_round,
        ]
    )
    return {
        "matrix": design_matrix,
        "player_matrix": player_matrix,
        "player_ids": list(player_ids),
        "design_effect_names": ["intercept", "team_a_starting_side", "overtime_round"],
        "diagnostics": _matrix_diagnostics(design_matrix, structural_nullity=1),
    }


def _fit_adjusted_plus_minus(
    episodes: Sequence[Mapping[str, Any]],
    target: Sequence[float],
    *,
    player_ids: Sequence[str],
    ridge_alpha: float,
) -> dict[str, Any]:
    design = build_signed_player_design_matrix(episodes, player_ids=player_ids)
    matrix = np.asarray(design["matrix"], dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if not np.all(np.isfinite(y)):
        raise FixedPoolModelError("Model target must contain only finite values.")
    penalty = np.ones(matrix.shape[1], dtype=np.float64)
    # Intercept, starting side, and overtime are observed design effects rather
    # than latent player values.  Leave the intercept unpenalized only.
    penalty[len(player_ids)] = 0.0
    coefficients = _ridge_fit(matrix, y, alpha=ridge_alpha, penalize=penalty)
    raw_player_values = coefficients[: len(player_ids)]
    replacement_value = (
        float(np.mean(raw_player_values)) if raw_player_values.size else 0.0
    )
    player_values = raw_player_values - replacement_value
    predicted = matrix @ coefficients
    residual = y - predicted
    return {
        "player_ids": list(player_ids),
        "coefficients": coefficients,
        "player_values": player_values,
        "replacement_player_value": replacement_value,
        "design_effects": {
            name: float(coefficients[len(player_ids) + offset])
            for offset, name in enumerate(design["design_effect_names"])
        },
        "predicted": predicted,
        "residual": residual,
        "fit": {
            **design["diagnostics"],
            "ridge_alpha": float(ridge_alpha),
            "rmse": float(np.sqrt(np.mean(np.square(residual)))),
            "r_squared": _r_squared(y, predicted),
        },
    }


def _select_adjusted_plus_minus_ridge_alpha(
    episodes: Sequence[Mapping[str, Any]],
    target: Sequence[float],
    *,
    player_ids: Sequence[str],
    ridge_alpha: float | None,
    cv_folds: int = RIDGE_CV_FOLDS,
    cv_seed: int = 0,
) -> dict[str, Any]:
    """Tune the APM penalty without splitting a sampling group."""
    if ridge_alpha is not None:
        return _manual_ridge_selection(float(ridge_alpha))
    pair_ids = [str(episode["pair_id"]) for episode in episodes]
    target_array = np.asarray(target, dtype=np.float64)

    def _predict_fold(
        train_indices: np.ndarray, validation_indices: np.ndarray, alpha: float
    ) -> np.ndarray:
        fit = _fit_adjusted_plus_minus(
            [episodes[int(index)] for index in train_indices],
            target_array[train_indices],
            player_ids=player_ids,
            ridge_alpha=alpha,
        )
        validation_matrix = build_signed_player_design_matrix(
            [episodes[int(index)] for index in validation_indices],
            player_ids=player_ids,
        )["matrix"]
        return np.asarray(validation_matrix @ fit["coefficients"], dtype=np.float64)

    return _select_grouped_ridge_alpha(
        pair_ids,
        target_array,
        predict_fold=_predict_fold,
        n_splits=cv_folds,
        seed=cv_seed,
    )


def _r_squared(target: np.ndarray, predicted: np.ndarray) -> float | None:
    if target.size < 2:
        return None
    centered = target - float(np.mean(target))
    total = float(np.dot(centered, centered))
    if total <= np.finfo(np.float64).eps:
        return None
    return float(1.0 - np.dot(target - predicted, target - predicted) / total)


def _paired_bootstrap_intervals(
    episodes: Sequence[Mapping[str, Any]],
    target: Sequence[float],
    *,
    player_ids: Sequence[str],
    ridge_alpha: float,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    if int(bootstrap_samples) < 1:
        return {"samples": 0, "intervals": {}, "successful_samples": 0}
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for index, episode in enumerate(episodes):
        grouped_indices[str(episode["pair_id"])].append(index)
    pair_ids = np.asarray(sorted(grouped_indices), dtype=object)
    rng = np.random.default_rng(int(seed))
    samples: list[np.ndarray] = []
    target_array = np.asarray(target, dtype=np.float64)
    for _ in range(int(bootstrap_samples)):
        drawn_pair_ids = rng.choice(pair_ids, size=pair_ids.size, replace=True)
        indices = [
            index
            for pair_id in drawn_pair_ids
            for index in grouped_indices[str(pair_id)]
        ]
        sample_episodes = [episodes[index] for index in indices]
        try:
            fitted = _fit_adjusted_plus_minus(
                sample_episodes,
                target_array[indices],
                player_ids=player_ids,
                ridge_alpha=ridge_alpha,
            )
        except (FixedPoolModelError, np.linalg.LinAlgError):
            continue
        samples.append(np.asarray(fitted["player_values"], dtype=np.float64))
    if not samples:
        return {
            "samples": int(bootstrap_samples),
            "intervals": {},
            "successful_samples": 0,
        }
    values = np.asarray(samples, dtype=np.float64)
    return {
        "samples": int(bootstrap_samples),
        "successful_samples": int(values.shape[0]),
        "resampled_unit": "paired_match_id",
        "intervals": {
            player_id: {
                "p05": float(np.percentile(values[:, index], 5)),
                "p95": float(np.percentile(values[:, index], 95)),
            }
            for index, player_id in enumerate(player_ids)
        },
    }


def _pool_skill_matrix(
    study: Mapping[str, Any], player_ids: Sequence[str]
) -> dict[str, Any]:
    pool_manifest = study.get("pool_manifest")
    if not isinstance(pool_manifest, Mapping):
        raise FixedPoolModelError("Study is missing a pool manifest.")
    descriptors = pool_manifest.get("skill_descriptors")
    players = pool_manifest.get("players")
    if not isinstance(descriptors, list) or not isinstance(players, list):
        raise FixedPoolModelError("Pool manifest lacks descriptors or players.")
    descriptor_keys = [
        str(descriptor.get("key"))
        for descriptor in descriptors
        if isinstance(descriptor, Mapping) and descriptor.get("key") is not None
    ]
    if not descriptor_keys:
        raise FixedPoolModelError("Pool manifest has no declared skills.")
    by_player = {
        str(player.get("player_id")): player.get("skills", {})
        for player in players
        if isinstance(player, Mapping)
    }
    values = np.empty((len(player_ids), len(descriptor_keys)), dtype=np.float64)
    for row_index, player_id in enumerate(player_ids):
        skills = by_player.get(str(player_id))
        if not isinstance(skills, Mapping):
            raise FixedPoolModelError(f"Pool manifest lacks skills for {player_id!r}.")
        for column_index, skill_key in enumerate(descriptor_keys):
            try:
                values[row_index, column_index] = float(skills[skill_key])
            except (KeyError, TypeError, ValueError) as exc:
                raise FixedPoolModelError(
                    f"Pool player {player_id!r} lacks declared skill {skill_key!r}."
                ) from exc
    return {"keys": descriptor_keys, "values": values}


def _fit_skill_ridge(
    train_values: np.ndarray,
    train_target: np.ndarray,
    *,
    alpha: float,
    predict_values: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit standardized skill ridge and optionally predict a second roster slice."""
    means = np.mean(train_values, axis=0)
    stds = np.std(train_values, axis=0, ddof=0)
    included = np.flatnonzero(stds > 1.0e-12).astype(np.int32)
    if included.size == 0:
        prediction_rows = train_values if predict_values is None else predict_values
        return {
            "means": means,
            "stds": stds,
            "included": included,
            "coefficients": np.asarray([float(np.mean(train_target))], dtype=np.float64),
            "predicted": np.full(
                prediction_rows.shape[0], float(np.mean(train_target)), dtype=np.float64
            ),
        }
    standardized = (train_values[:, included] - means[included]) / stds[included]
    design = np.column_stack(
        [np.ones(len(train_values), dtype=np.float64), standardized]
    )
    penalty = np.ones(design.shape[1], dtype=np.float64)
    penalty[0] = 0.0
    coefficients = _ridge_fit(design, train_target, alpha=alpha, penalize=penalty)
    prediction_rows = train_values if predict_values is None else predict_values
    prediction_design = np.column_stack(
        [
            np.ones(len(prediction_rows), dtype=np.float64),
            (prediction_rows[:, included] - means[included]) / stds[included],
        ]
    )
    return {
        "means": means,
        "stds": stds,
        "included": included,
        "coefficients": coefficients,
        "predicted": prediction_design @ coefficients,
    }


def fit_skill_regression(
    study: Mapping[str, Any],
    *,
    player_ids: Sequence[str],
    player_values: Sequence[float],
    ridge_alpha: float | None = None,
    ridge_cv_folds: int = RIDGE_CV_FOLDS,
    ridge_cv_seed: int = 0,
) -> dict[str, Any]:
    """Regress adjusted player value on declared pool skills with tuned ridge."""
    skill_matrix = _pool_skill_matrix(study, player_ids)
    keys = skill_matrix["keys"]
    values = np.asarray(skill_matrix["values"], dtype=np.float64)
    y = np.asarray(player_values, dtype=np.float64)
    if y.shape != (len(player_ids),):
        raise FixedPoolModelError("Player-value vector does not match the pool roster.")
    overall = _fit_skill_ridge(values, y, alpha=0.0)
    included_indices = [int(index) for index in overall["included"]]
    excluded = [
        {"skill": keys[index], "reason": "zero_variance"}
        for index in range(len(keys))
        if index not in included_indices
    ]
    if not included_indices:
        return {
            "coefficients": {},
            "predicted_player_value": {
                player_id: float(overall["predicted"][index])
                for index, player_id in enumerate(player_ids)
            },
            "residual_player_value": {
                player_id: float(y[index] - overall["predicted"][index])
                for index, player_id in enumerate(player_ids)
            },
            "excluded_skills": excluded,
            "fit": {
                "rank": 0,
                "condition_number": None,
                "r_squared": None,
                "warning": "all_declared_skills_zero_variance",
            },
        }
    if ridge_alpha is None:
        def _predict_fold(
            train_indices: np.ndarray, validation_indices: np.ndarray, alpha: float
        ) -> np.ndarray:
            return np.asarray(
                _fit_skill_ridge(
                    values[train_indices],
                    y[train_indices],
                    alpha=alpha,
                    predict_values=values[validation_indices],
                )["predicted"],
                dtype=np.float64,
            )

        ridge_selection = _select_grouped_ridge_alpha(
            [str(player_id) for player_id in player_ids],
            y,
            predict_fold=_predict_fold,
            n_splits=ridge_cv_folds,
            seed=ridge_cv_seed,
        )
    else:
        ridge_selection = _manual_ridge_selection(float(ridge_alpha))
    selected_alpha = float(ridge_selection["selected_alpha"])
    fitted = _fit_skill_ridge(values, y, alpha=selected_alpha)
    means = np.asarray(fitted["means"], dtype=np.float64)
    stds = np.asarray(fitted["stds"], dtype=np.float64)
    coefficients = np.asarray(fitted["coefficients"], dtype=np.float64)
    predicted = np.asarray(fitted["predicted"], dtype=np.float64)
    standardized = (values[:, included_indices] - means[included_indices]) / stds[
        included_indices
    ]
    matrix = np.column_stack([np.ones(len(player_ids), dtype=np.float64), standardized])
    fit = _matrix_diagnostics(matrix)
    fit.update(
        {
            "ridge_alpha": selected_alpha,
            "ridge_selection": ridge_selection,
            "r_squared": _r_squared(y, predicted),
            "standardization_population": "study_pool",
        }
    )
    return {
        "intercept": float(coefficients[0]),
        "coefficients": {
            keys[source_index]: {
                "net_rating_points_per_standard_deviation": float(
                    coefficients[output_index + 1]
                ),
                "pool_mean": float(means[source_index]),
                "pool_standard_deviation": float(stds[source_index]),
            }
            for output_index, source_index in enumerate(included_indices)
        },
        "excluded_skills": excluded,
        "predicted_player_value": {
            player_id: float(predicted[index])
            for index, player_id in enumerate(player_ids)
        },
        "residual_player_value": {
            player_id: float(y[index] - predicted[index])
            for index, player_id in enumerate(player_ids)
        },
        "fit": fit,
    }


def _direct_skill_feature_matrix(
    episodes: Sequence[Mapping[str, Any]],
    *,
    player_ids: Sequence[str],
    skill_values: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build game rows from known Team-A-minus-Team-B profile skills.

    A profile has no identity-level attribute outside ``skill_values``.  The
    feature for a game is therefore the sum of Team A's seeded skills minus
    the sum of Team B's seeded skills, plus observed starting-side/overtime
    controls.  This is the direct structural model; it deliberately does not
    use an estimated player +/- as an intermediate target.
    """
    index = {str(player_id): position for position, player_id in enumerate(player_ids)}
    differences = np.zeros((len(episodes), skill_values.shape[1]), dtype=np.float64)
    starting_side = np.zeros(len(episodes), dtype=np.float64)
    overtime_round = np.zeros(len(episodes), dtype=np.float64)
    for row_index, episode in enumerate(episodes):
        for player_id in episode.get("team_a_player_ids", []):
            try:
                differences[row_index] += skill_values[index[str(player_id)]]
            except KeyError as exc:
                raise FixedPoolModelError(
                    f"Team A references unknown pooled player {player_id!r}."
                ) from exc
        for player_id in episode.get("team_b_player_ids", []):
            try:
                differences[row_index] -= skill_values[index[str(player_id)]]
            except KeyError as exc:
                raise FixedPoolModelError(
                    f"Team B references unknown pooled player {player_id!r}."
                ) from exc
        starting_team = episode.get("starting_team")
        if starting_team == "team_a":
            starting_side[row_index] = 1.0
        elif starting_team == "team_b":
            starting_side[row_index] = -1.0
        else:
            raise FixedPoolModelError("Episode is missing a Team A/B starting side.")
        overtime_round[row_index] = float(episode.get("overtime_round", 0) or 0)
    return {
        "skill_differences": differences,
        "starting_side": starting_side,
        "overtime_round": overtime_round,
    }


def _fit_direct_skill_rating_model(
    episodes: Sequence[Mapping[str, Any]],
    target: Sequence[float],
    *,
    player_ids: Sequence[str],
    skill_keys: Sequence[str],
    skill_values: np.ndarray,
    ridge_alpha: float,
) -> dict[str, Any]:
    """Fit game outcome directly from seeded profile skills and controls."""
    y = np.asarray(target, dtype=np.float64)
    if y.shape != (len(episodes),) or not np.all(np.isfinite(y)):
        raise FixedPoolModelError("Direct skill target must be finite and match episodes.")
    features = _direct_skill_feature_matrix(
        episodes, player_ids=player_ids, skill_values=skill_values
    )
    differences = features["skill_differences"]
    means = np.mean(differences, axis=0)
    scales = np.std(differences, axis=0)
    included = np.flatnonzero(scales > 1.0e-12).astype(np.int32)
    standardized = (differences[:, included] - means[included]) / scales[included]
    design = np.column_stack(
        [
            np.ones(len(episodes), dtype=np.float64),
            standardized,
            features["starting_side"],
            features["overtime_round"],
        ]
    )
    penalty = np.concatenate(
        [np.zeros(1, dtype=np.float64), np.ones(included.size, dtype=np.float64), np.zeros(2, dtype=np.float64)]
    )
    coefficients = _ridge_fit(design, y, alpha=ridge_alpha, penalize=penalty)
    raw_skill_coefficients = np.zeros(len(skill_keys), dtype=np.float64)
    raw_skill_coefficients[included] = coefficients[1 : 1 + included.size] / scales[
        included
    ]
    intercept = float(coefficients[0] - np.dot(raw_skill_coefficients, means))
    predicted = (
        intercept
        + differences @ raw_skill_coefficients
        + features["starting_side"] * coefficients[-2]
        + features["overtime_round"] * coefficients[-1]
    )
    profile_mean = np.mean(skill_values, axis=0)
    player_values = (skill_values - profile_mean) @ raw_skill_coefficients
    return {
        "player_values": np.asarray(player_values, dtype=np.float64),
        "predicted": np.asarray(predicted, dtype=np.float64),
        "coefficients": raw_skill_coefficients,
        "intercept": intercept,
        "design_effects": {
            "team_a_starting_side": float(coefficients[-2]),
            "overtime_round": float(coefficients[-1]),
        },
        "included": included,
        "means": means,
        "scales": scales,
        "fit": {
            **_matrix_diagnostics(design),
            "ridge_alpha": float(ridge_alpha),
            "rmse": float(np.sqrt(np.mean(np.square(y - predicted)))),
            "r_squared": _r_squared(y, predicted),
            "target": "team_a_minus_team_b_net_rating",
            "feature_definition": "sum(team_a_player_skills) - sum(team_b_player_skills)",
        },
    }


def _select_direct_skill_ridge_alpha(
    episodes: Sequence[Mapping[str, Any]],
    target: Sequence[float],
    *,
    player_ids: Sequence[str],
    skill_keys: Sequence[str],
    skill_values: np.ndarray,
    ridge_alpha: float | None,
    cv_folds: int = RIDGE_CV_FOLDS,
    cv_seed: int = 0,
) -> dict[str, Any]:
    """Tune the direct skill-to-outcome model inside paired training games."""
    if ridge_alpha is not None:
        return _manual_ridge_selection(float(ridge_alpha))
    target_array = np.asarray(target, dtype=np.float64)

    def _predict_fold(
        train_indices: np.ndarray, validation_indices: np.ndarray, alpha: float
    ) -> np.ndarray:
        fit = _fit_direct_skill_rating_model(
            [episodes[int(index)] for index in train_indices],
            target_array[train_indices],
            player_ids=player_ids,
            skill_keys=skill_keys,
            skill_values=skill_values,
            ridge_alpha=alpha,
        )
        validation_features = _direct_skill_feature_matrix(
            [episodes[int(index)] for index in validation_indices],
            player_ids=player_ids,
            skill_values=skill_values,
        )
        return np.asarray(
            fit["intercept"]
            + validation_features["skill_differences"] @ fit["coefficients"]
            + validation_features["starting_side"]
            * fit["design_effects"]["team_a_starting_side"]
            + validation_features["overtime_round"]
            * fit["design_effects"]["overtime_round"],
            dtype=np.float64,
        )

    return _select_grouped_ridge_alpha(
        [str(episode["pair_id"]) for episode in episodes],
        target_array,
        predict_fold=_predict_fold,
        n_splits=cv_folds,
        seed=cv_seed,
    )


def _paired_bootstrap_direct_skill_intervals(
    episodes: Sequence[Mapping[str, Any]],
    target: Sequence[float],
    *,
    player_ids: Sequence[str],
    skill_keys: Sequence[str],
    skill_values: np.ndarray,
    ridge_alpha: float,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Sampling-group bootstrap intervals for directly estimated profile values."""
    if int(bootstrap_samples) < 1:
        return {"samples": 0, "intervals": {}, "successful_samples": 0}
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for index, episode in enumerate(episodes):
        grouped_indices[str(episode["pair_id"])].append(index)
    pair_ids = np.asarray(sorted(grouped_indices), dtype=object)
    rng = np.random.default_rng(int(seed))
    target_array = np.asarray(target, dtype=np.float64)
    samples: list[np.ndarray] = []
    for _ in range(int(bootstrap_samples)):
        chosen = rng.choice(pair_ids, size=pair_ids.size, replace=True)
        indices = [index for pair_id in chosen for index in grouped_indices[str(pair_id)]]
        try:
            fitted = _fit_direct_skill_rating_model(
                [episodes[index] for index in indices],
                target_array[indices],
                player_ids=player_ids,
                skill_keys=skill_keys,
                skill_values=skill_values,
                ridge_alpha=ridge_alpha,
            )
        except (FixedPoolModelError, np.linalg.LinAlgError):
            continue
        samples.append(np.asarray(fitted["player_values"], dtype=np.float64))
    if not samples:
        return {"samples": int(bootstrap_samples), "intervals": {}, "successful_samples": 0}
    sample_array = np.stack(samples, axis=0)
    return {
        "samples": int(bootstrap_samples),
        "successful_samples": int(sample_array.shape[0]),
        "intervals": {
            player_id: {
                "p05": float(np.quantile(sample_array[:, index], 0.05)),
                "p95": float(np.quantile(sample_array[:, index], 0.95)),
            }
            for index, player_id in enumerate(player_ids)
        },
    }


def derive_candidate_factor_features(
    episodes: Sequence[Mapping[str, Any]],
    possession_events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Derive versioned Team-A-minus-Team-B possession factor outcomes."""
    by_game: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: {
            "team_a": defaultdict(float),
            "team_b": defaultdict(float),
        }
    )
    for event in possession_events:
        game_id = str(event.get("schedule_game_id", ""))
        team = event.get("actor_team")
        if team not in {"team_a", "team_b"}:
            continue
        bucket = by_game[game_id][str(team)]
        event_type = event.get("event_type")
        if event_type == "shot":
            bucket["shot_expected_value"] += float(
                event.get("shot_expected_points") or 0.0
            )
            if event.get("shot_success"):
                bucket["shot_conversion"] += float(event.get("shot_value") or 0.0)
        elif event_type == "turnover":
            bucket["ball_security"] -= 1.0
        elif event_type == "rebound" and event.get("rebound_offensive"):
            bucket["second_chance_creation"] += 1.0
        elif event_type == "pass" and event.get("pass_completed"):
            bucket["possession_organization"] += 1.0
    factor_names = (
        "shot_expected_value",
        "shot_conversion",
        "ball_security",
        "second_chance_creation",
        "possession_organization",
    )
    factor_targets: dict[str, list[float | None]] = {name: [] for name in factor_names}
    for episode in episodes:
        game_id = str(episode["schedule_game_id"])
        team_a_possessions = int(episode.get("team_a_completed_possessions") or 0)
        team_b_possessions = int(episode.get("team_b_completed_possessions") or 0)
        for factor_name in factor_names:
            if team_a_possessions <= 0 or team_b_possessions <= 0:
                factor_targets[factor_name].append(None)
                continue
            counts = by_game[game_id]
            team_a_rate = counts["team_a"][factor_name] / team_a_possessions
            team_b_rate = counts["team_b"][factor_name] / team_b_possessions
            factor_targets[factor_name].append(100.0 * (team_a_rate - team_b_rate))
    return {
        "schema_version": FACTOR_FEATURE_SCHEMA_VERSION,
        "factor_names": list(factor_names),
        "targets": factor_targets,
    }


def derive_conventional_three_factor_features(
    episodes: Sequence[Mapping[str, Any]],
    possession_events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Derive per-game eFG%, TOV%, and ORB% differences from the event ledger.

    The simulator does not yet model fouls or free throws.  Consequently the
    conventional Four Factors' free-throw-rate term is explicitly unavailable,
    rather than silently represented as a zero rate.  The returned targets are
    Team A minus Team B *percentage points*; a positive turnover-rate estimate
    is therefore bad for Team A, while positive eFG% and ORB% estimates are
    good for Team A.
    """
    by_game: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: {
            "team_a": defaultdict(float),
            "team_b": defaultdict(float),
        }
    )
    for event in possession_events:
        game_id = str(event.get("schedule_game_id", ""))
        team = event.get("actor_team")
        if team not in {"team_a", "team_b"}:
            continue
        bucket = by_game[game_id][str(team)]
        event_type = event.get("event_type")
        if event_type == "shot":
            bucket["field_goal_attempts"] += 1.0
            if bool(event.get("shot_success")):
                bucket["field_goals_made"] += 1.0
                if float(event.get("shot_value") or 0.0) == 3.0:
                    bucket["three_point_field_goals_made"] += 1.0
        elif event_type == "turnover":
            bucket["turnovers"] += 1.0
        elif event_type == "rebound":
            if bool(event.get("rebound_offensive")):
                bucket["offensive_rebounds"] += 1.0
            elif bool(event.get("rebound_defensive")):
                bucket["defensive_rebounds"] += 1.0

    def _efg(stats: Mapping[str, float]) -> float | None:
        attempts = stats["field_goal_attempts"]
        if attempts <= 0.0:
            return None
        return (
            stats["field_goals_made"]
            + 0.5 * stats["three_point_field_goals_made"]
        ) / attempts

    def _tov_pct(stats: Mapping[str, float]) -> float | None:
        # Standard TOV% includes 0.44 * FTA.  With no free-throw events in
        # this schema, its observable form is turnovers / (FGA + turnovers).
        denominator = stats["field_goal_attempts"] + stats["turnovers"]
        if denominator <= 0.0:
            return None
        return stats["turnovers"] / denominator

    def _orb_pct(offense: Mapping[str, float], defense: Mapping[str, float]) -> float | None:
        denominator = offense["offensive_rebounds"] + defense["defensive_rebounds"]
        if denominator <= 0.0:
            return None
        return offense["offensive_rebounds"] / denominator

    factor_names = ("effective_field_goal_pct", "turnover_pct", "offensive_rebound_pct")
    targets: dict[str, list[float | None]] = {name: [] for name in factor_names}
    team_rates: dict[str, dict[str, list[float | None]]] = {
        "team_a": {name: [] for name in factor_names},
        "team_b": {name: [] for name in factor_names},
    }
    aggregate: dict[str, defaultdict[str, float]] = {
        "team_a": defaultdict(float),
        "team_b": defaultdict(float),
    }
    for game in by_game.values():
        for team in ("team_a", "team_b"):
            for name, value in game[team].items():
                aggregate[team][name] += value

    for episode in episodes:
        game = by_game[str(episode["schedule_game_id"])]
        team_a = game["team_a"]
        team_b = game["team_b"]
        efg_a, efg_b = _efg(team_a), _efg(team_b)
        tov_a, tov_b = _tov_pct(team_a), _tov_pct(team_b)
        orb_a, orb_b = _orb_pct(team_a, team_b), _orb_pct(team_b, team_a)
        team_rates["team_a"]["effective_field_goal_pct"].append(
            None if efg_a is None else 100.0 * efg_a
        )
        team_rates["team_b"]["effective_field_goal_pct"].append(
            None if efg_b is None else 100.0 * efg_b
        )
        team_rates["team_a"]["turnover_pct"].append(
            None if tov_a is None else 100.0 * tov_a
        )
        team_rates["team_b"]["turnover_pct"].append(
            None if tov_b is None else 100.0 * tov_b
        )
        team_rates["team_a"]["offensive_rebound_pct"].append(
            None if orb_a is None else 100.0 * orb_a
        )
        team_rates["team_b"]["offensive_rebound_pct"].append(
            None if orb_b is None else 100.0 * orb_b
        )
        targets["effective_field_goal_pct"].append(
            None if efg_a is None or efg_b is None else 100.0 * (efg_a - efg_b)
        )
        targets["turnover_pct"].append(
            None if tov_a is None or tov_b is None else 100.0 * (tov_a - tov_b)
        )
        targets["offensive_rebound_pct"].append(
            None if orb_a is None or orb_b is None else 100.0 * (orb_a - orb_b)
        )

    aggregate_rates = {
        "team_a": {
            "effective_field_goal_pct": _efg(aggregate["team_a"]),
            "turnover_pct": _tov_pct(aggregate["team_a"]),
            "offensive_rebound_pct": _orb_pct(aggregate["team_a"], aggregate["team_b"]),
        },
        "team_b": {
            "effective_field_goal_pct": _efg(aggregate["team_b"]),
            "turnover_pct": _tov_pct(aggregate["team_b"]),
            "offensive_rebound_pct": _orb_pct(aggregate["team_b"], aggregate["team_a"]),
        },
    }
    return {
        "schema_version": CONVENTIONAL_THREE_FACTOR_FEATURE_SCHEMA_VERSION,
        "factor_names": list(factor_names),
        "targets": targets,
        "team_rates_percentage_points": team_rates,
        "units": "team_a_minus_team_b_percentage_points",
        "definitions": {
            "effective_field_goal_pct": "(FGM + 0.5 * 3PM) / FGA",
            "turnover_pct": "TOV / (FGA + TOV); free throws are unmodeled",
            "offensive_rebound_pct": "ORB / (ORB + opponent DRB)",
        },
        "aggregate_rates": aggregate_rates,
        "unavailable_factors": {
            "free_throw_rate": {
                "status": "unavailable",
                "reason": "environment_has_no_foul_or_free_throw_events",
                "definition": "FTA / FGA",
            }
        },
    }


def _fit_conventional_three_factor_scenario_weights(
    feature_set: Mapping[str, Any],
    *,
    point_margin: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    episodes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Reproduce the two-sided, one-standard-deviation weighting method.

    This is the three-factor analogue of the methodology in Evan Zamir's 2010
    Four Factors regression: ordinary least squares on separate own/opponent
    terms, then a mean-reference team whose individual terms are moved one
    training-set standard deviation in the direction that improves predicted
    point margin.  The resulting six scenario deltas are grouped by factor and
    normalized to sum to 100 percent.
    """
    factor_names = [str(name) for name in feature_set["factor_names"]]
    team_rates = feature_set["team_rates_percentage_points"]
    term_specs = [
        (f"own_{name}", "team_a", name) for name in factor_names
    ] + [(f"opponent_{name}", "team_b", name) for name in factor_names]
    usable_train = [
        int(index)
        for index in train_indices
        if all(
            team_rates[team][factor][int(index)] is not None
            for _, team, factor in term_specs
        )
    ]
    usable_validation = [
        int(index)
        for index in validation_indices
        if all(
            team_rates[team][factor][int(index)] is not None
            for _, team, factor in term_specs
        )
    ]
    base = {
        "methodology": (
            "ordinary_least_squares with separate own and opponent terms; "
            "one training-standard-deviation favorable scenarios about the "
            "mean-reference team; factor shares normalize the six deltas"
        ),
        "target": "team_a_minus_team_b_margin",
        "feature_units": "percentage_points",
        "factor_definitions": feature_set.get("definitions", {}),
        "training_pair_ids": sorted(
            {str(episodes[index]["pair_id"]) for index in usable_train}
        ),
        "validation": {
            "available": len(usable_validation) >= 2,
            "pair_ids": sorted(
                {str(episodes[index]["pair_id"]) for index in usable_validation}
            ),
            "sample_count": len(usable_validation),
            "rmse": None,
            "r_squared": None,
        },
    }
    if len(usable_train) < len(term_specs) + 1:
        return {
            **base,
            "status": "excluded",
            "reason": "insufficient_complete_factor_observations",
        }
    features = np.asarray(
        [
            [team_rates[team][factor][index] for _, team, factor in term_specs]
            for index in range(len(point_margin))
        ],
        dtype=np.float64,
    )
    train_features = features[usable_train]
    means = np.mean(train_features, axis=0)
    standard_deviations = np.std(train_features, axis=0)
    if np.any(standard_deviations <= 1.0e-12):
        return {**base, "status": "excluded", "reason": "zero_variance_factor"}
    design = np.column_stack(
        [np.ones(len(usable_train), dtype=np.float64), train_features]
    )
    train_target = point_margin[usable_train]
    coefficients = _ridge_fit(
        design,
        train_target,
        alpha=0.0,
        penalize=np.zeros(design.shape[1], dtype=np.float64),
    )
    train_prediction = design @ coefficients
    reference_prediction = float(coefficients[0] + np.dot(coefficients[1:], means))
    scenarios: dict[str, Any] = {}
    grouped_deltas: dict[str, float] = defaultdict(float)
    for index, (term_name, team, factor_name) in enumerate(term_specs):
        coefficient = float(coefficients[index + 1])
        standard_deviation = float(standard_deviations[index])
        favorable_change = (
            standard_deviation
            if coefficient > 0.0
            else -standard_deviation
            if coefficient < 0.0
            else 0.0
        )
        margin_delta = coefficient * favorable_change
        grouped_deltas[factor_name] += margin_delta
        scenarios[term_name] = {
            "team_side": "own" if team == "team_a" else "opponent",
            "factor": factor_name,
            "coefficient_margin_points_per_percentage_point": coefficient,
            "mean_percentage_points": float(means[index]),
            "standard_deviation_percentage_points": standard_deviation,
            "favorable_change_percentage_points": favorable_change,
            "scenario_percentage_points": float(means[index] + favorable_change),
            "reference_predicted_margin": reference_prediction,
            "scenario_predicted_margin": float(reference_prediction + margin_delta),
            "margin_delta": float(margin_delta),
        }
    total_delta = float(sum(grouped_deltas.values()))
    factor_weights = {
        factor_name: {
            "combined_margin_delta": float(grouped_deltas[factor_name]),
            "percent_of_total_margin_delta": (
                100.0 * float(grouped_deltas[factor_name]) / total_delta
                if total_delta > 1.0e-12
                else None
            ),
        }
        for factor_name in factor_names
    }
    report = {
        **base,
        "status": "fit",
        "fit": {
            **_matrix_diagnostics(design),
            "estimator": "ordinary_least_squares",
            "rmse": float(np.sqrt(np.mean(np.square(train_target - train_prediction)))),
            "r_squared": _r_squared(train_target, train_prediction),
        },
        "reference_team": {
            "predicted_margin": reference_prediction,
            "term_means_percentage_points": {
                term_name: float(mean)
                for (term_name, _, _), mean in zip(term_specs, means)
            },
        },
        "one_standard_deviation_scenarios": scenarios,
        "factor_weights": factor_weights,
        "total_margin_delta": total_delta,
    }
    if usable_validation:
        validation_design = np.column_stack(
            [
                np.ones(len(usable_validation), dtype=np.float64),
                features[usable_validation],
            ]
        )
        validation_target = point_margin[usable_validation]
        validation_prediction = validation_design @ coefficients
        report["validation"]["rmse"] = float(
            np.sqrt(np.mean(np.square(validation_target - validation_prediction)))
        )
        report["validation"]["r_squared"] = _r_squared(
            validation_target, validation_prediction
        )
    return report


def _fit_conventional_three_factor_outcome_model(
    feature_set: Mapping[str, Any],
    *,
    outcome_target: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    episodes: Sequence[Mapping[str, Any]],
    ridge_alpha: float | None,
) -> dict[str, Any]:
    """Fit the joint conventional-factor-to-net-rating relationship.

    Features are standardized on the training pairs only for the ridge fit,
    then converted back to net-rating points per percentage point for reporting.
    This preserves a directly interpretable unit without letting the held-out
    pair group influence the feature scaling or coefficients.
    """
    if ridge_alpha is None:
        ridge_selection = _select_conventional_three_factor_ridge_alpha(
            feature_set,
            outcome_target=outcome_target,
            train_indices=train_indices,
            episodes=episodes,
        )
        report = _fit_conventional_three_factor_outcome_model(
            feature_set,
            outcome_target=outcome_target,
            train_indices=train_indices,
            validation_indices=validation_indices,
            episodes=episodes,
            ridge_alpha=float(ridge_selection["selected_alpha"]),
        )
        report["ridge_selection"] = ridge_selection
        return report
    factor_names = [str(name) for name in feature_set["factor_names"]]
    values = feature_set["targets"]
    usable_train = [
        int(index)
        for index in train_indices
        if all(values[name][int(index)] is not None for name in factor_names)
    ]
    usable_validation = [
        int(index)
        for index in validation_indices
        if all(values[name][int(index)] is not None for name in factor_names)
    ]
    base = {
        "target": "team_a_minus_team_b_net_rating",
        "feature_units": feature_set.get("units"),
        "feature_definitions": feature_set.get("definitions", {}),
        "train_pair_ids": sorted(
            {str(episodes[index]["pair_id"]) for index in usable_train}
        ),
        "validation": {
            "pair_ids": sorted(
                {str(episodes[index]["pair_id"]) for index in usable_validation}
            ),
            "sample_count": len(usable_validation),
            "available": len(usable_validation) >= 2,
            "rmse": None,
            "r_squared": None,
        },
    }
    if len(usable_train) < len(factor_names) + 1:
        return {
            **base,
            "status": "excluded",
            "reason": "insufficient_complete_factor_observations",
        }
    train_features = np.asarray(
        [[values[name][index] for name in factor_names] for index in usable_train],
        dtype=np.float64,
    )
    train_target = outcome_target[usable_train]
    means = np.mean(train_features, axis=0)
    scales = np.std(train_features, axis=0)
    usable_columns = scales > 1.0e-12
    if not np.any(usable_columns):
        return {**base, "status": "excluded", "reason": "zero_variance_factors"}
    included_names = [
        name for name, included in zip(factor_names, usable_columns) if included
    ]
    excluded_names = [
        name for name, included in zip(factor_names, usable_columns) if not included
    ]
    standardized = (train_features[:, usable_columns] - means[usable_columns]) / scales[
        usable_columns
    ]
    design = np.column_stack([np.ones(len(usable_train), dtype=np.float64), standardized])
    coefficients = _ridge_fit(
        design,
        train_target,
        alpha=ridge_alpha,
        penalize=np.concatenate(
            [np.zeros(1, dtype=np.float64), np.ones(len(included_names), dtype=np.float64)]
        ),
    )
    point_coefficients = coefficients[1:] / scales[usable_columns]
    raw_intercept = float(
        coefficients[0] - np.dot(point_coefficients, means[usable_columns])
    )
    train_prediction = design @ coefficients
    report = {
        **base,
        "status": "fit",
        "ridge_alpha": float(ridge_alpha),
        "intercept_net_rating_points": raw_intercept,
        "coefficients": {
            name: {
                "net_rating_points_per_percentage_point": float(point),
                "net_rating_points_per_train_standard_deviation": float(standard),
                "train_mean_percentage_points": float(mean),
                "train_standard_deviation_percentage_points": float(scale),
            }
            for name, point, standard, mean, scale in zip(
                included_names,
                point_coefficients,
                coefficients[1:],
                means[usable_columns],
                scales[usable_columns],
            )
        },
        "excluded_factors": excluded_names,
        "train": {
            "sample_count": len(usable_train),
            "rmse": float(np.sqrt(np.mean(np.square(train_target - train_prediction)))),
            "r_squared": _r_squared(train_target, train_prediction),
            "design": _matrix_diagnostics(design),
        },
    }
    if usable_validation:
        validation_features = np.asarray(
            [
                [values[name][index] for name in factor_names]
                for index in usable_validation
            ],
            dtype=np.float64,
        )[:, usable_columns]
        validation_prediction = raw_intercept + validation_features @ point_coefficients
        validation_target = outcome_target[usable_validation]
        report["validation"]["rmse"] = float(
            np.sqrt(np.mean(np.square(validation_target - validation_prediction)))
        )
        report["validation"]["r_squared"] = _r_squared(
            validation_target, validation_prediction
        )
    return report


def _select_conventional_three_factor_ridge_alpha(
    feature_set: Mapping[str, Any],
    *,
    outcome_target: np.ndarray,
    train_indices: np.ndarray,
    episodes: Sequence[Mapping[str, Any]],
    cv_folds: int = RIDGE_CV_FOLDS,
    cv_seed: int = 0,
) -> dict[str, Any]:
    """Tune the joint three-factor outcome model within the outer train set."""
    selected_train = np.asarray(train_indices, dtype=np.int32)
    fold_plan = paired_group_kfold(
        [str(episodes[int(index)]["pair_id"]) for index in selected_train],
        n_splits=cv_folds,
        seed=cv_seed,
    )
    base = {
        "method": "grouped_k_fold_cross_validation",
        "candidate_alphas": list(RIDGE_ALPHA_CANDIDATES),
        "requested_fold_count": int(fold_plan["requested_fold_count"]),
        "fold_count": int(fold_plan["fold_count"]),
        "selection_metric": "mean_held_out_squared_error",
    }
    if not fold_plan["available"]:
        return {
            **base,
            "selected_alpha": 0.0,
            "available": False,
            "warning": fold_plan["warning"],
            "candidates": [],
        }
    candidate_reports = []
    for alpha in RIDGE_ALPHA_CANDIDATES:
        squared_error_sum = 0.0
        observation_count = 0
        fold_scores = []
        valid = True
        for fold in fold_plan["folds"]:
            inner_train = selected_train[np.asarray(fold["train_indices"], dtype=np.int32)]
            inner_validation = selected_train[
                np.asarray(fold["validation_indices"], dtype=np.int32)
            ]
            report = _fit_conventional_three_factor_outcome_model(
                feature_set,
                outcome_target=outcome_target,
                train_indices=inner_train,
                validation_indices=inner_validation,
                episodes=episodes,
                ridge_alpha=float(alpha),
            )
            validation = report.get("validation", {})
            sample_count = int(validation.get("sample_count") or 0)
            rmse = validation.get("rmse")
            if report.get("status") != "fit" or sample_count < 1 or rmse is None:
                valid = False
                break
            squared_error_sum += float(rmse) ** 2 * sample_count
            observation_count += sample_count
            fold_scores.append(
                {
                    "fold": int(fold["fold"]),
                    "sample_count": sample_count,
                    "mean_squared_error": float(rmse) ** 2,
                }
            )
        if valid and observation_count:
            mean_squared_error = squared_error_sum / observation_count
            candidate_reports.append(
                {
                    "alpha": float(alpha),
                    "mean_held_out_squared_error": float(mean_squared_error),
                    "mean_held_out_rmse": float(math.sqrt(mean_squared_error)),
                    "fold_scores": fold_scores,
                }
            )
    if not candidate_reports:
        return {
            **base,
            "selected_alpha": 0.0,
            "available": False,
            "warning": "no_candidate_with_complete_factor_folds",
            "candidates": [],
        }
    selected = min(
        candidate_reports,
        key=lambda item: (item["mean_held_out_squared_error"], -item["alpha"]),
    )
    return {
        **base,
        "selected_alpha": float(selected["alpha"]),
        "available": True,
        "warning": fold_plan["warning"],
        "candidates": candidate_reports,
    }


def _conventional_three_factor_outcome_importance(
    feature_set: Mapping[str, Any],
    *,
    outcome_target: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    episodes: Sequence[Mapping[str, Any]],
    ridge_alpha: float,
) -> dict[str, Any]:
    """Allocate held-out outcome-model fit among the conventional factors.

    A factor's importance is its Shapley value: its average marginal held-out
    R-squared improvement across every ordering of the available factors.
    Unlike raw rates or regression slopes, the normalized contributions add to
    100 percent of the three-factor model's explained improvement.
    """
    factor_names = [str(name) for name in feature_set["factor_names"]]
    values = feature_set["targets"]
    usable_train = [
        int(index)
        for index in train_indices
        if all(values[name][int(index)] is not None for name in factor_names)
    ]
    usable_validation = [
        int(index)
        for index in validation_indices
        if all(values[name][int(index)] is not None for name in factor_names)
    ]
    base = {
        "method": (
            "Shapley allocation of held-out R-squared improvement across all "
            "factor subsets"
        ),
        "validation_pair_ids": sorted(
            {str(episodes[index]["pair_id"]) for index in usable_validation}
        ),
        "sample_count": len(usable_validation),
    }
    if len(usable_train) < len(factor_names) + 1 or len(usable_validation) < 2:
        return {
            **base,
            "status": "excluded",
            "reason": "insufficient_complete_factor_observations",
        }
    features = np.asarray(
        [[values[name][index] for name in factor_names] for index in range(len(values[factor_names[0]]))],
        dtype=np.float64,
    )
    train_target = outcome_target[usable_train]
    validation_target = outcome_target[usable_validation]

    def _held_out_r_squared(columns: tuple[int, ...]) -> float | None:
        if not columns:
            baseline = np.full(
                len(usable_validation), float(np.mean(train_target)), dtype=np.float64
            )
            return _r_squared(validation_target, baseline)
        train_features = features[usable_train][:, columns]
        means = np.mean(train_features, axis=0)
        scales = np.std(train_features, axis=0)
        if np.any(scales <= 1.0e-12):
            return None
        standardized = (train_features - means) / scales
        design = np.column_stack(
            [np.ones(len(usable_train), dtype=np.float64), standardized]
        )
        coefficients = _ridge_fit(
            design,
            train_target,
            alpha=ridge_alpha,
            penalize=np.concatenate(
                [np.zeros(1, dtype=np.float64), np.ones(len(columns), dtype=np.float64)]
            ),
        )
        validation_design = np.column_stack(
            [
                np.ones(len(usable_validation), dtype=np.float64),
                (features[usable_validation][:, columns] - means) / scales,
            ]
        )
        return _r_squared(validation_target, validation_design @ coefficients)

    subset_scores: dict[tuple[int, ...], float] = {}
    for size in range(len(factor_names) + 1):
        for subset in itertools.combinations(range(len(factor_names)), size):
            score = _held_out_r_squared(subset)
            if score is None:
                return {
                    **base,
                    "status": "excluded",
                    "reason": "zero_variance_factor_subset",
                }
            subset_scores[subset] = score
    contributions = np.zeros(len(factor_names), dtype=np.float64)
    factor_count = len(factor_names)
    for factor_index in range(factor_count):
        other_indices = [
            index for index in range(factor_count) if index != factor_index
        ]
        for size in range(factor_count):
            for subset in itertools.combinations(other_indices, size):
                subset = tuple(sorted(subset))
                with_factor = tuple(sorted((*subset, factor_index)))
                weight = (
                    math.factorial(size)
                    * math.factorial(factor_count - size - 1)
                    / math.factorial(factor_count)
                )
                contributions[factor_index] += weight * (
                    subset_scores[with_factor] - subset_scores[subset]
                )
    baseline_score = subset_scores[()]
    full_score = subset_scores[tuple(range(factor_count))]
    improvement = full_score - baseline_score
    normalized = (
        100.0 * contributions / improvement
        if abs(improvement) > 1.0e-12
        else np.full(factor_count, np.nan, dtype=np.float64)
    )
    return {
        **base,
        "status": "fit",
        "baseline_r_squared": float(baseline_score),
        "full_model_r_squared": float(full_score),
        "explained_r_squared_improvement": float(improvement),
        "factor_contributions": {
            name: {
                "held_out_r_squared_improvement": float(contribution),
                "percent_of_explained_improvement": (
                    float(percent) if math.isfinite(percent) else None
                ),
            }
            for name, contribution, percent in zip(
                factor_names, contributions, normalized
            )
        },
        "subset_r_squared": {
            "+".join(factor_names[index] for index in subset) or "baseline": float(
                score
            )
            for subset, score in subset_scores.items()
        },
    }


def _factor_models(
    episodes: Sequence[Mapping[str, Any]],
    *,
    feature_set: Mapping[str, Any],
    player_ids: Sequence[str],
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    outcome_target: np.ndarray,
    ridge_alpha: float | None,
    bootstrap_samples: int,
    bootstrap_seed: int,
    ridge_cv_folds: int = RIDGE_CV_FOLDS,
    ridge_cv_seed: int = 0,
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    train_episodes = [episodes[index] for index in train_indices]
    for offset, factor_name in enumerate(feature_set["factor_names"]):
        values = feature_set["targets"][factor_name]
        usable_train = [
            index for index in train_indices if values[int(index)] is not None
        ]
        if len(usable_train) < 2:
            reports[factor_name] = {
                "status": "excluded",
                "reason": "insufficient_factor_observations",
            }
            continue
        target = np.asarray([values[index] for index in usable_train], dtype=np.float64)
        if float(np.std(target)) <= 1.0e-12:
            reports[factor_name] = {
                "status": "excluded",
                "reason": "zero_variance_factor",
            }
            continue
        factor_episodes = [episodes[index] for index in usable_train]
        ridge_selection = _select_adjusted_plus_minus_ridge_alpha(
            factor_episodes,
            target,
            player_ids=player_ids,
            ridge_alpha=ridge_alpha,
            cv_folds=ridge_cv_folds,
            cv_seed=int(ridge_cv_seed) + offset,
        )
        selected_alpha = float(ridge_selection["selected_alpha"])
        fitted = _fit_adjusted_plus_minus(
            factor_episodes,
            target,
            player_ids=player_ids,
            ridge_alpha=selected_alpha,
        )
        fitted["fit"]["ridge_selection"] = ridge_selection
        bootstrap = _paired_bootstrap_intervals(
            factor_episodes,
            target,
            player_ids=player_ids,
            ridge_alpha=selected_alpha,
            bootstrap_samples=bootstrap_samples,
            seed=int(bootstrap_seed) + offset,
        )
        bootstrap["ridge_alpha_conditioning"] = {
            "selected_alpha": selected_alpha,
            "selection_repeated_within_resample": False,
        }
        validation_factor_indices = [
            index for index in validation_indices if values[int(index)] is not None
        ]
        outcome_association: dict[str, Any] = {
            "validation_pair_ids": sorted(
                {str(episodes[index]["pair_id"]) for index in validation_factor_indices}
            ),
            "outcome_correlation": None,
            "sample_count": len(validation_factor_indices),
        }
        if len(validation_factor_indices) >= 2:
            validation_design = build_signed_player_design_matrix(
                [episodes[index] for index in validation_factor_indices],
                player_ids=player_ids,
            )["matrix"]
            factor_prediction = validation_design @ fitted["coefficients"]
            target_outcome = outcome_target[validation_factor_indices]
            if np.std(factor_prediction) > 1.0e-12 and np.std(target_outcome) > 1.0e-12:
                outcome_association["outcome_correlation"] = float(
                    np.corrcoef(factor_prediction, target_outcome)[0, 1]
                )
        reports[factor_name] = {
            "status": "fit",
            "factor_feature_schema_version": feature_set["schema_version"],
            "player_factor_impact": {
                player_id: float(fitted["player_values"][index])
                for index, player_id in enumerate(player_ids)
            },
            "fit": fitted["fit"],
            "bootstrap": bootstrap,
            "outcome_validation": outcome_association,
        }
    report = {
        "factor_feature_schema_version": feature_set["schema_version"],
        "factor_estimation_pair_ids": sorted(
            {str(episode["pair_id"]) for episode in train_episodes}
        ),
        "outcome_validation_pair_ids": sorted(
            {str(episodes[index]["pair_id"]) for index in validation_indices}
        ),
        "models": reports,
    }
    for key in ("units", "definitions", "aggregate_rates", "unavailable_factors"):
        if key in feature_set:
            report[key] = feature_set[key]
    return report


def fit_fixed_pool_study_models(
    study: str | Path | Mapping[str, Any],
    *,
    ridge_alpha: float | None = None,
    validation_fraction: float = 0.2,
    split_seed: int = 0,
    bootstrap_samples: int = 200,
    bootstrap_seed: int = 1,
) -> dict[str, Any]:
    """Fit APM, skill, and factor reports for one immutable action-mode study."""
    normalized_study = _as_study(study)
    assert_studies_schema_compatible([normalized_study])
    action_mode = _episode_action_mode(normalized_study)
    episodes = list(normalized_study["episodes"])
    included = [
        row
        for row in episodes
        if bool(row.get("analysis_default_included"))
        and row.get("team_a_minus_team_b_net_rating") is not None
    ]
    excluded = [row for row in episodes if row not in included]
    if not included:
        raise FixedPoolModelError(
            "No completed, nonzero-possession games are eligible for fitting."
        )
    player_ids = sorted(
        {
            str(player_id)
            for row in episodes
            for team_key in ("team_a_player_ids", "team_b_player_ids")
            for player_id in row.get(team_key, [])
        }
    )
    pool_manifest = normalized_study["pool_manifest"]
    skill_count = len(pool_manifest.get("skill_descriptors", []))
    budget = fixed_pool_study_budget(skill_count)
    pair_ids = [str(row["pair_id"]) for row in included]
    split = paired_group_split(
        pair_ids, validation_fraction=validation_fraction, seed=split_seed
    )
    target = np.asarray(
        [float(row["team_a_minus_team_b_net_rating"]) for row in included],
        dtype=np.float64,
    )
    point_margin = np.asarray(
        [
            float(
                row.get(
                    "team_a_minus_team_b_margin",
                    row["team_a_minus_team_b_net_rating"],
                )
            )
            for row in included
        ],
        dtype=np.float64,
    )
    train_indices = np.asarray(split["train_indices"], dtype=np.int32)
    train_episodes = [included[index] for index in train_indices]
    train_target = target[train_indices]
    # Skills are the full profile state, so this is the primary rating model:
    # it learns outcome effects directly from skills and games, instead of
    # first estimating one unconstrained coefficient per player identity.
    skill_matrix = _pool_skill_matrix(normalized_study, player_ids)
    skill_keys = [str(key) for key in skill_matrix["keys"]]
    skill_values = np.asarray(skill_matrix["values"], dtype=np.float64)
    direct_ridge_selection = _select_direct_skill_ridge_alpha(
        train_episodes,
        train_target,
        player_ids=player_ids,
        skill_keys=skill_keys,
        skill_values=skill_values,
        ridge_alpha=ridge_alpha,
        cv_folds=RIDGE_CV_FOLDS,
        cv_seed=split_seed,
    )
    direct_ridge_alpha = float(direct_ridge_selection["selected_alpha"])
    direct_train_fit = _fit_direct_skill_rating_model(
        train_episodes,
        train_target,
        player_ids=player_ids,
        skill_keys=skill_keys,
        skill_values=skill_values,
        ridge_alpha=direct_ridge_alpha,
    )
    direct_full_fit = _fit_direct_skill_rating_model(
        included,
        target,
        player_ids=player_ids,
        skill_keys=skill_keys,
        skill_values=skill_values,
        ridge_alpha=direct_ridge_alpha,
    )
    direct_full_fit["fit"]["ridge_selection"] = direct_ridge_selection
    validation_indices = np.asarray(split["validation_indices"], dtype=np.int32)
    direct_validation_report: dict[str, Any] = {
        "pair_ids": list(split["validation_pair_ids"]),
        "available": bool(split["validation_available"]),
        "sample_count": int(validation_indices.size),
        "rmse": None,
        "r_squared": None,
    }
    if validation_indices.size:
        direct_validation_features = _direct_skill_feature_matrix(
            [included[index] for index in validation_indices],
            player_ids=player_ids,
            skill_values=skill_values,
        )
        direct_prediction = (
            direct_train_fit["intercept"]
            + direct_validation_features["skill_differences"]
            @ direct_train_fit["coefficients"]
            + direct_validation_features["starting_side"]
            * direct_train_fit["design_effects"]["team_a_starting_side"]
            + direct_validation_features["overtime_round"]
            * direct_train_fit["design_effects"]["overtime_round"]
        )
        direct_validation_target = target[validation_indices]
        direct_validation_report["rmse"] = float(
            np.sqrt(np.mean(np.square(direct_validation_target - direct_prediction)))
        )
        direct_validation_report["r_squared"] = _r_squared(
            direct_validation_target, direct_prediction
        )
    direct_bootstrap = _paired_bootstrap_direct_skill_intervals(
        train_episodes,
        train_target,
        player_ids=player_ids,
        skill_keys=skill_keys,
        skill_values=skill_values,
        ridge_alpha=direct_ridge_alpha,
        bootstrap_samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    direct_bootstrap["ridge_alpha_conditioning"] = {
        "selected_alpha": direct_ridge_alpha,
        "selection_repeated_within_resample": False,
        "note": "Intervals condition on the outer-training five-fold CV selection.",
    }
    apm_ridge_selection = _select_adjusted_plus_minus_ridge_alpha(
        train_episodes,
        train_target,
        player_ids=player_ids,
        ridge_alpha=ridge_alpha,
        cv_folds=RIDGE_CV_FOLDS,
        cv_seed=split_seed,
    )
    apm_ridge_alpha = float(apm_ridge_selection["selected_alpha"])
    train_fit = _fit_adjusted_plus_minus(
        train_episodes,
        train_target,
        player_ids=player_ids,
        ridge_alpha=apm_ridge_alpha,
    )
    # The selected penalty is chosen on the outer-training pairs only, then
    # applied to the full study to produce the published player ratings.
    full_fit = _fit_adjusted_plus_minus(
        included, target, player_ids=player_ids, ridge_alpha=apm_ridge_alpha
    )
    full_fit["fit"]["ridge_selection"] = apm_ridge_selection
    validation_report: dict[str, Any] = {
        "pair_ids": list(split["validation_pair_ids"]),
        "available": bool(split["validation_available"]),
        "sample_count": int(validation_indices.size),
        "rmse": None,
        "r_squared": None,
    }
    if validation_indices.size:
        validation_design = build_signed_player_design_matrix(
            [included[index] for index in validation_indices], player_ids=player_ids
        )["matrix"]
        prediction = validation_design @ train_fit["coefficients"]
        validation_target = target[validation_indices]
        validation_report["rmse"] = float(
            np.sqrt(np.mean(np.square(validation_target - prediction)))
        )
        validation_report["r_squared"] = _r_squared(validation_target, prediction)
    bootstrap = _paired_bootstrap_intervals(
        train_episodes,
        train_target,
        player_ids=player_ids,
        ridge_alpha=apm_ridge_alpha,
        bootstrap_samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    bootstrap["ridge_alpha_conditioning"] = {
        "selected_alpha": apm_ridge_alpha,
        "selection_repeated_within_resample": False,
        "note": "Intervals condition on the outer-training five-fold CV selection.",
    }
    skill_model = fit_skill_regression(
        normalized_study,
        player_ids=player_ids,
        player_values=full_fit["player_values"],
        ridge_alpha=ridge_alpha,
        ridge_cv_folds=RIDGE_CV_FOLDS,
        ridge_cv_seed=split_seed,
    )
    factor_model = _factor_models(
        included,
        feature_set=derive_candidate_factor_features(
            included, normalized_study.get("possession_events", [])
        ),
        player_ids=player_ids,
        train_indices=train_indices,
        validation_indices=validation_indices,
        outcome_target=target,
        ridge_alpha=ridge_alpha,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        ridge_cv_folds=RIDGE_CV_FOLDS,
        ridge_cv_seed=split_seed,
    )
    conventional_three_factor_features = derive_conventional_three_factor_features(
        included, normalized_study.get("possession_events", [])
    )
    conventional_three_factor_model = _factor_models(
        included,
        feature_set=conventional_three_factor_features,
        player_ids=player_ids,
        train_indices=train_indices,
        validation_indices=validation_indices,
        outcome_target=target,
        ridge_alpha=ridge_alpha,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        ridge_cv_folds=RIDGE_CV_FOLDS,
        ridge_cv_seed=split_seed,
    )
    conventional_three_factor_model["outcome_model"] = (
        _fit_conventional_three_factor_outcome_model(
            conventional_three_factor_features,
            outcome_target=target,
            train_indices=train_indices,
            validation_indices=validation_indices,
            episodes=included,
            ridge_alpha=ridge_alpha,
        )
    )
    conventional_three_factor_model["scenario_weights"] = (
        _fit_conventional_three_factor_scenario_weights(
            conventional_three_factor_features,
            point_margin=point_margin,
            train_indices=train_indices,
            validation_indices=validation_indices,
            episodes=included,
        )
    )
    player_report = {}
    for index, player_id in enumerate(player_ids):
        player_report[player_id] = {
            "net_rating_points_vs_replacement": float(full_fit["player_values"][index]),
            "bootstrap_interval": bootstrap["intervals"].get(player_id),
            "predicted_skill_value": skill_model["predicted_player_value"].get(
                player_id
            ),
            "residual_skill_value": skill_model["residual_player_value"].get(player_id),
        }
    direct_player_report = {}
    profile_means = np.mean(skill_values, axis=0)
    profile_stds = np.std(skill_values, axis=0, ddof=0)
    for index, player_id in enumerate(player_ids):
        direct_value = float(direct_full_fit["player_values"][index])
        apm_value = float(full_fit["player_values"][index])
        direct_player_report[player_id] = {
            "net_rating_points_vs_replacement": direct_value,
            "bootstrap_interval": direct_bootstrap["intervals"].get(player_id),
            "apm_diagnostic_value": apm_value,
            "apm_minus_skill_rating": apm_value - direct_value,
        }
    direct_skill_report = {
        "model": "direct_skill_to_game_outcome_ridge",
        "rating_interpretation": (
            "Profile value relative to the average seeded player, predicted directly "
            "from its complete declared skill vector."
        ),
        "target": "team_a_minus_team_b_net_rating",
        "replacement_anchor": "mean_seeded_skill_profile",
        "design_effects": direct_full_fit["design_effects"],
        "fit": direct_full_fit["fit"],
        "validation": direct_validation_report,
        "train_pair_ids": list(split["train_pair_ids"]),
        "bootstrap": direct_bootstrap,
        "coefficients": {
            skill_key: {
                "net_rating_points_per_skill_unit": float(
                    direct_full_fit["coefficients"][skill_index]
                ),
                "net_rating_points_per_pool_standard_deviation": float(
                    direct_full_fit["coefficients"][skill_index]
                    * profile_stds[skill_index]
                ),
                "pool_mean": float(profile_means[skill_index]),
                "pool_standard_deviation": float(profile_stds[skill_index]),
            }
            for skill_index, skill_key in enumerate(skill_keys)
        },
        "players": direct_player_report,
        "apm_agreement": skill_apm_agreement(direct_player_report),
    }
    unique_sampling_group_count = len(set(pair_ids))
    actual_possessions = int(
        sum(
            int(row.get("team_a_completed_possessions") or 0)
            + int(row.get("team_b_completed_possessions") or 0)
            for row in included
        )
    )
    sizing = {
        **budget,
        "actual_seeded_players": len(player_ids),
        "actual_sampling_groups": unique_sampling_group_count,
        "actual_games": len(included),
        "actual_possessions": actual_possessions,
        "player_to_skill_ratio": (
            float(len(player_ids) / skill_count) if skill_count else None
        ),
        "under_sized": bool(
            len(player_ids) < budget["seeded_players"]
            or len(included) < budget["games"]
        ),
    }
    warnings = []
    if sizing["under_sized"]:
        warnings.append("under_sized_relative_to_skill_scaled_budget")
    if full_fit["fit"]["rank_deficient"]:
        warnings.append("player_design_rank_deficient")
    if full_fit["fit"]["materially_collinear"]:
        warnings.append("player_design_materially_collinear")
    if not split["validation_available"]:
        warnings.append("no_disjoint_validation_sampling_group")
    return {
        "study_id": normalized_study["manifest"].get("study_id"),
        "action_mode": action_mode,
        "schema_compatibility": {
            "compatible": True,
            "warnings": [],
            "skill_schema_hash": normalized_study["manifest"].get("skill_schema_hash"),
            "checkpoint_observation_schema_versions": included[0].get(
                "checkpoint_observation_schema_versions"
            ),
            "environment_observation_schema_version": included[0].get(
                "environment_observation_schema_version"
            ),
        },
        "sizing": sizing,
        "analysis_rows": {
            "included_game_count": len(included),
            "excluded_game_count": len(excluded),
            "excluded_by_reason": {
                str(row.get("analysis_exclusion_reason") or "unspecified"): sum(
                    1
                    for candidate in excluded
                    if str(candidate.get("analysis_exclusion_reason") or "unspecified")
                    == str(row.get("analysis_exclusion_reason") or "unspecified")
                )
                for row in excluded
            },
        },
        "adjusted_plus_minus": {
            "target": "team_a_minus_team_b_net_rating",
            "replacement_anchor": "mean_fitted_player_value",
            "design_effects": full_fit["design_effects"],
            "fit": full_fit["fit"],
            "validation": validation_report,
            "train_pair_ids": list(split["train_pair_ids"]),
            "bootstrap": bootstrap,
            "players": player_report,
        },
        "direct_skill_ratings": direct_skill_report,
        "skill_regression": skill_model,
        "factor_models": factor_model,
        "conventional_three_factor_models": conventional_three_factor_model,
        "warnings": warnings,
    }


def fit_fixed_pool_studies(
    studies: Sequence[str | Path | Mapping[str, Any]], **kwargs: Any
) -> dict[str, Any]:
    """Fit a separate report for every direct/sampled study supplied."""
    normalized = [_as_study(study) for study in studies]
    assert_studies_schema_compatible(normalized)
    reports: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for study in normalized:
        report = fit_fixed_pool_study_models(study, **kwargs)
        reports[report["action_mode"]].append(report)
    return {
        "comparison_schema_compatible": True,
        "reports_by_action_mode": dict(reports),
        "action_modes_are_separate": True,
    }
