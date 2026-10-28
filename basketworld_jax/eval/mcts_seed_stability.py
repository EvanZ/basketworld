"""Same-state, multi-seed stability analysis for completed JAX MCTS studies."""

from __future__ import annotations

from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from basketworld_jax.provenance import capture_source_provenance


SEED_STABILITY_SCHEMA_VERSION = 1
_PROVENANCE_IDENTITY_FIELDS = (
    "evaluated_checkpoint_update_index",
    "opponent_checkpoint_update_index",
    "evaluated_policy_spec",
    "opponent_policy_spec",
    "evaluated_stable_team",
    "opponent_stable_team",
    "evaluated_action_mode",
    "opponent_action_mode",
    "resolved_environment_config",
    "value_target",
)


def _load_payload(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {source}.")
    if not payload.get("records"):
        raise ValueError(f"MCTS artifact contains no completed records: {source}.")
    return payload


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bootstrap_mean(
    values: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not array.size:
        raise ValueError("Bootstrap values must be a non-empty vector.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Bootstrap values must be finite.")
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, array.size, size=(int(samples), array.size))
    estimates = np.mean(array[draws], axis=1)
    return {
        "estimate": float(np.mean(array)),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
        "state_count": int(array.size),
    }


def _primary_depth(payload: Mapping[str, Any], explicit_depth: int | None) -> int:
    depth = int(
        explicit_depth
        if explicit_depth is not None
        else payload.get("aggregate", {}).get(
            "primary_search_depth",
            payload.get("protocol", {}).get("primary_search_depth", 0),
        )
    )
    if depth < 1:
        raise ValueError("A positive MCTS search depth is required.")
    return depth


def _ordered_records(
    payload: Mapping[str, Any], *, depth: int
) -> tuple[list[str], list[Mapping[str, Any]]]:
    state_ids: list[str] = []
    records: list[Mapping[str, Any]] = []
    for raw_record in payload.get("records", []):
        record = dict(raw_record)
        state_id = str(record.get("sample", {}).get("state_id", ""))
        depth_result = dict(record.get("depth_results", {})).get(str(int(depth)))
        if not state_id or not isinstance(depth_result, Mapping):
            raise ValueError("Every record must contain a state ID and depth result.")
        selected_action = list(
            depth_result.get("search", {}).get("selected_action", [])
        )
        if not selected_action:
            raise ValueError(f"State {state_id!r} has no selected MCTS action.")
        state_ids.append(state_id)
        records.append(record)
    if len(set(state_ids)) != len(state_ids):
        raise ValueError("MCTS artifact contains duplicate state IDs.")
    return state_ids, records


def _validate_artifacts(
    payloads: Sequence[Mapping[str, Any]], *, depth: int
) -> tuple[list[str], list[list[Mapping[str, Any]]]]:
    reference = payloads[0]
    reference_protocol = dict(reference.get("protocol", {}))
    reference_provenance = dict(reference.get("provenance", {}))
    reference_ids, reference_records = _ordered_records(reference, depth=depth)
    record_sets = [reference_records]
    for artifact_index, candidate in enumerate(payloads[1:], start=2):
        if dict(candidate.get("protocol", {})) != reference_protocol:
            raise ValueError(
                f"MCTS artifact {artifact_index} uses a different search protocol."
            )
        candidate_provenance = dict(candidate.get("provenance", {}))
        mismatches = [
            field
            for field in _PROVENANCE_IDENTITY_FIELDS
            if candidate_provenance.get(field) != reference_provenance.get(field)
        ]
        if mismatches:
            raise ValueError(
                f"MCTS artifact {artifact_index} policy/environment provenance "
                "differs for: " + ", ".join(mismatches)
            )
        candidate_ids, candidate_records = _ordered_records(candidate, depth=depth)
        if candidate_ids != reference_ids:
            raise ValueError(
                "Search-seed stability artifacts must contain identical ordered states."
            )
        for state_id, reference_record, candidate_record in zip(
            reference_ids,
            reference_records,
            candidate_records,
            strict=True,
        ):
            reference_sample = dict(reference_record.get("sample", {}))
            candidate_sample = dict(candidate_record.get("sample", {}))
            for field in ("evaluated_role", "policy_action"):
                if candidate_sample.get(field) != reference_sample.get(field):
                    raise ValueError(
                        f"State {state_id!r} differs in sample field {field!r}."
                    )
        record_sets.append(candidate_records)

    for state_index, state_id in enumerate(reference_ids):
        search_seeds = [
            int(records[state_index]["depth_results"][str(depth)]["search_seed"])
            for records in record_sets
        ]
        if len(set(search_seeds)) != len(search_seeds):
            raise ValueError(
                f"State {state_id!r} does not have a distinct search seed in each "
                "artifact."
            )
    return reference_ids, record_sets


def _search_margin(depth_result: Mapping[str, Any]) -> dict[str, Any]:
    search = dict(depth_result.get("search", {}))
    selected_action = tuple(int(value) for value in search.get("selected_action", []))
    action_stats = [dict(row) for row in search.get("action_stats", [])]
    if not selected_action or not action_stats:
        raise ValueError("Search result is missing selected-action statistics.")
    selected_rows = [
        row
        for row in action_stats
        if tuple(int(value) for value in row.get("action", [])) == selected_action
    ]
    if len(selected_rows) != 1:
        raise ValueError(
            "Selected action is missing or duplicated in action statistics."
        )
    selected = selected_rows[0]
    remaining = [row for row in action_stats if row is not selected]
    runner_up = (
        max(
            remaining,
            key=lambda row: (
                int(row.get("visits", 0)),
                float(row.get("mean_value", 0.0)),
                tuple(int(value) for value in row.get("action", [])),
            ),
        )
        if remaining
        else None
    )
    expected_selected = max(
        action_stats,
        key=lambda row: (
            int(row.get("visits", 0)),
            float(row.get("mean_value", 0.0)),
            tuple(int(value) for value in row.get("action", [])),
        ),
    )
    if selected is not expected_selected:
        raise ValueError("Selected action does not match the most-visited action.")
    total_visits = sum(int(row.get("visits", 0)) for row in action_stats)
    if total_visits <= 0:
        raise ValueError("Search action statistics contain no root visits.")
    selected_visits = int(selected.get("visits", 0))
    selected_value = float(selected.get("mean_value", 0.0))
    return {
        "selected_visit_share": float(selected_visits / total_visits),
        "selected_vs_runner_up_visit_margin_fraction": (
            float((selected_visits - int(runner_up.get("visits", 0))) / total_visits)
            if runner_up is not None
            else 1.0
        ),
        "selected_vs_runner_up_mean_value_margin": (
            float(selected_value - float(runner_up.get("mean_value", 0.0)))
            if runner_up is not None
            else None
        ),
        "root_action_count": len(action_stats),
    }


def summarize_search_seed_stability(
    payloads: Sequence[Mapping[str, Any]],
    *,
    depth: int | None = None,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 17,
) -> dict[str, Any]:
    """Summarize action convergence while only the MCTS search seed varies."""

    if len(payloads) < 2:
        raise ValueError("Search-seed stability requires at least two artifacts.")
    if int(bootstrap_samples) < 1:
        raise ValueError("bootstrap_samples must be positive.")
    selected_depth = _primary_depth(payloads[0], depth)
    state_ids, record_sets = _validate_artifacts(payloads, depth=selected_depth)
    run_count = len(record_sets)
    pair_indices = tuple(combinations(range(run_count), 2))

    actions = np.asarray(
        [
            [
                records[state_index]["depth_results"][str(selected_depth)]["search"][
                    "selected_action"
                ]
                for state_index in range(len(state_ids))
            ]
            for records in record_sets
        ],
        dtype=np.int32,
    )
    if actions.ndim != 3:
        raise ValueError("Selected actions must form run x state x player arrays.")
    player_count = int(actions.shape[2])
    player_pairwise_by_state = []
    joint_pairwise_by_state = []
    move_pairwise_by_state = []
    for state_index in range(len(state_ids)):
        player_matches = []
        joint_matches = []
        move_matches = []
        for left_index, right_index in pair_indices:
            left = actions[left_index, state_index]
            right = actions[right_index, state_index]
            exact = left == right
            player_matches.extend(exact.tolist())
            joint_matches.append(bool(np.all(exact)))
            move_matches.extend(((left > 0) == (right > 0)).tolist())
        player_pairwise_by_state.append(float(np.mean(player_matches)))
        joint_pairwise_by_state.append(float(np.mean(joint_matches)))
        move_pairwise_by_state.append(float(np.mean(move_matches)))

    player_consensus = np.all(actions == actions[0:1], axis=0)
    joint_consensus = np.all(player_consensus, axis=1)
    player_consensus_by_state = np.mean(player_consensus, axis=1)
    modal_agreement = np.zeros((len(state_ids), player_count), dtype=np.float64)
    for state_index in range(len(state_ids)):
        for player_index in range(player_count):
            counts = np.bincount(actions[:, state_index, player_index])
            modal_agreement[state_index, player_index] = float(
                np.max(counts) / run_count
            )
    modal_agreement_by_state = np.mean(modal_agreement, axis=1)

    state_rows = []
    margin_rows = []
    for state_index, state_id in enumerate(state_ids):
        per_run = []
        for run_index, records in enumerate(record_sets):
            depth_result = records[state_index]["depth_results"][str(selected_depth)]
            margin = _search_margin(depth_result)
            per_run.append(
                {
                    "run_index": int(run_index),
                    "search_seed": int(depth_result["search_seed"]),
                    "selected_action": actions[run_index, state_index].tolist(),
                    **margin,
                }
            )
            margin_rows.append((state_index, margin))
        state_rows.append(
            {
                "state_id": state_id,
                "evaluated_role": str(
                    record_sets[0][state_index]
                    .get("sample", {})
                    .get("evaluated_role", "unknown")
                ),
                "pairwise_player_action_agreement": player_pairwise_by_state[
                    state_index
                ],
                "pairwise_joint_action_agreement": joint_pairwise_by_state[state_index],
                "all_seed_player_consensus": player_consensus[state_index].tolist(),
                "all_seed_joint_action_consensus": bool(joint_consensus[state_index]),
                "mean_modal_player_agreement": modal_agreement_by_state[state_index],
                "runs": per_run,
            }
        )

    def aggregate_margin(name: str, offset: int) -> dict[str, Any]:
        per_state = []
        for state_index in range(len(state_ids)):
            values = [
                float(margin[name])
                for row_state_index, margin in margin_rows
                if row_state_index == state_index and margin[name] is not None
            ]
            if not values:
                raise ValueError(f"No {name} values are available for one state.")
            per_state.append(float(np.mean(values)))
        return _bootstrap_mean(
            per_state,
            samples=int(bootstrap_samples),
            seed=int(bootstrap_seed) + offset,
        )

    role_counts = {
        role: sum(row["evaluated_role"] == role for row in state_rows)
        for role in ("offense", "defense")
        if any(row["evaluated_role"] == role for row in state_rows)
    }
    return {
        "run_count": int(run_count),
        "state_count": len(state_ids),
        "controlled_player_count": player_count,
        "controlled_player_decision_count": int(len(state_ids) * player_count),
        "search_depth": int(selected_depth),
        "simulations_per_action": int(
            payloads[0].get("protocol", {}).get("simulations_per_action", 0)
        ),
        "role_counts": role_counts,
        "pairwise_player_action_agreement": _bootstrap_mean(
            player_pairwise_by_state,
            samples=int(bootstrap_samples),
            seed=int(bootstrap_seed) + 1,
        ),
        "pairwise_joint_action_agreement": _bootstrap_mean(
            joint_pairwise_by_state,
            samples=int(bootstrap_samples),
            seed=int(bootstrap_seed) + 2,
        ),
        "pairwise_move_vs_noop_agreement": _bootstrap_mean(
            move_pairwise_by_state,
            samples=int(bootstrap_samples),
            seed=int(bootstrap_seed) + 3,
        ),
        "all_seed_player_action_consensus": _bootstrap_mean(
            player_consensus_by_state,
            samples=int(bootstrap_samples),
            seed=int(bootstrap_seed) + 4,
        ),
        "all_seed_joint_action_consensus": _bootstrap_mean(
            joint_consensus.astype(np.float64),
            samples=int(bootstrap_samples),
            seed=int(bootstrap_seed) + 5,
        ),
        "mean_modal_player_action_agreement": _bootstrap_mean(
            modal_agreement_by_state,
            samples=int(bootstrap_samples),
            seed=int(bootstrap_seed) + 6,
        ),
        "search_confidence": {
            "selected_visit_share": aggregate_margin("selected_visit_share", 7),
            "selected_vs_runner_up_visit_margin_fraction": aggregate_margin(
                "selected_vs_runner_up_visit_margin_fraction", 8
            ),
            "selected_vs_runner_up_mean_value_margin": aggregate_margin(
                "selected_vs_runner_up_mean_value_margin", 9
            ),
        },
        "states": state_rows,
    }


def run_search_seed_stability_study(
    result_paths: Sequence[str | Path],
    *,
    output_dir: str | Path,
    depth: int | None = None,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 17,
) -> dict[str, Any]:
    """Load completed same-state searches and write one immutable analysis."""

    sources = [Path(path).expanduser().resolve() for path in result_paths]
    payloads = [_load_payload(path) for path in sources]
    summary = summarize_search_seed_stability(
        payloads,
        depth=depth,
        bootstrap_samples=int(bootstrap_samples),
        bootstrap_seed=int(bootstrap_seed),
    )
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Output directory already exists: {destination}")
    payload = {
        "schema_version": SEED_STABILITY_SCHEMA_VERSION,
        "study": "same_state_mcts_search_seed_stability",
        "inputs": [
            {
                "path": str(path),
                "sha256": _sha256_file(path),
            }
            for path in sources
        ],
        "configuration": {
            "depth": int(summary["search_depth"]),
            "bootstrap_samples": int(bootstrap_samples),
            "bootstrap_seed": int(bootstrap_seed),
        },
        "summary": summary,
        "analysis_provenance": capture_source_provenance(),
    }
    destination.mkdir(parents=True, exist_ok=False)
    result_path = destination / "mcts_search_seed_stability.json"
    result_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"results": str(result_path), "payload": payload}


__all__ = [
    "SEED_STABILITY_SCHEMA_VERSION",
    "run_search_seed_stability_study",
    "summarize_search_seed_stability",
]
