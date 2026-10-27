"""Study orchestration for JAX MCTS policy-regret analysis."""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

from basketworld_jax.eval.mcts_regret import (
    MCTS_REGRET_SCHEMA_VERSION,
    MctsRegretConfig,
    RegretRecord,
    compare_actions_with_paired_rollouts,
    evaluate_receding_horizon_return,
    joint_action_probability,
    joint_action_agreement_payload,
    paired_comparison_payload,
    paired_bootstrap_mean_interval,
    run_joint_action_mcts,
    search_result_payload,
    summarize_regret_records,
    top_factorized_joint_actions,
)
from basketworld_jax.eval.mcts_regret_jax import (
    JaxCounterfactualRuntime,
    SampledDecision,
    load_persisted_sample_states,
    persist_sampled_decisions,
    sample_policy_decisions,
)


_SAMPLE_REPLAY_PROVENANCE_FIELDS = (
    "evaluated_checkpoint_path",
    "evaluated_checkpoint_update_index",
    "evaluated_policy_spec",
    "evaluated_stable_team",
    "evaluated_action_mode",
    "opponent_checkpoint_path",
    "opponent_checkpoint_update_index",
    "opponent_policy_spec",
    "opponent_stable_team",
    "opponent_action_mode",
    "resolved_environment_config",
    "value_target",
)


def _sampled_decision_from_metadata(
    metadata: Mapping[str, Any], state: Any
) -> SampledDecision:
    return SampledDecision(
        episode_index=int(metadata["episode_index"]),
        episode_seed=int(metadata["episode_seed"]),
        policy_seed=int(metadata["policy_seed"]),
        selection_seed=int(metadata["selection_seed"]),
        environment_step=int(metadata["environment_step"]),
        eligible_decision_index=int(metadata["eligible_decision_index"]),
        eligible_decision_count=int(metadata["eligible_decision_count"]),
        state_id=str(metadata["state_id"]),
        evaluated_role=str(metadata["evaluated_role"]),
        shot_clock=int(metadata["shot_clock"]),
        game_clock_remaining=int(metadata["game_clock_remaining"]),
        team_a_score=float(metadata["team_a_score"]),
        team_b_score=float(metadata["team_b_score"]),
        ball_holder=int(metadata["ball_holder"]),
        legal_action_counts=tuple(
            int(value) for value in metadata["legal_action_counts"]
        ),
        policy_action=tuple(int(value) for value in metadata["policy_action"]),
        state=state,
    )


def _study_samples(
    runtime: JaxCounterfactualRuntime,
    *,
    episode_count: int,
    seed: int,
    max_episode_steps: int,
    sample_metadata_path: str | Path | None,
    evaluated_role: str | None,
) -> tuple[list[SampledDecision], dict[str, Any]]:
    if evaluated_role not in {None, "offense", "defense"}:
        raise ValueError("evaluated_role must be offense, defense, or None.")
    if sample_metadata_path is None:
        samples, sampling_summary = sample_policy_decisions(
            runtime,
            episode_count=int(episode_count),
            seed=int(seed),
            max_episode_steps=int(max_episode_steps),
        )
    else:
        source_path = Path(sample_metadata_path).expanduser().resolve()
        metadata_rows, states, source_payload = load_persisted_sample_states(
            source_path
        )
        source_provenance = dict(source_payload.get("provenance", {}))
        runtime_provenance = dict(runtime.provenance)
        mismatches = [
            field
            for field in _SAMPLE_REPLAY_PROVENANCE_FIELDS
            if json.dumps(source_provenance.get(field), sort_keys=True)
            != json.dumps(runtime_provenance.get(field), sort_keys=True)
        ]
        if mismatches:
            raise ValueError(
                "Persisted sample provenance does not match the active runtime: "
                + ", ".join(mismatches)
            )
        samples = [
            _sampled_decision_from_metadata(metadata, state)
            for metadata, state in zip(metadata_rows, states, strict=True)
        ]
        sampling_summary = dict(source_payload.get("sampling_summary", {}))
        sampling_summary.update(
            {
                "selection_method": "reused-persisted-policy-decisions",
                "source_sample_count": len(samples),
                "source_sample_metadata": str(source_path),
            }
        )

    if evaluated_role is not None:
        samples = [
            sample for sample in samples if sample.evaluated_role == evaluated_role
        ]
    role_counts = {
        role: sum(sample.evaluated_role == role for sample in samples)
        for role in ("offense", "defense")
        if any(sample.evaluated_role == role for sample in samples)
    }
    sampling_summary.update(
        {
            "sample_count": len(samples),
            "role_counts": role_counts,
            "evaluated_role_filter": evaluated_role,
        }
    )
    return samples, sampling_summary


def _seed_tuple(sequence: np.random.SeedSequence, count: int) -> tuple[int, ...]:
    return tuple(
        int(value) for value in sequence.generate_state(int(count), dtype=np.uint32)
    )


def _state_planning_seed(base_seed: int, state_key: str) -> int:
    digest = hashlib.blake2b(digest_size=4)
    digest.update(str(int(base_seed)).encode("ascii"))
    digest.update(b":")
    digest.update(str(state_key).encode("utf-8"))
    return int.from_bytes(digest.digest(), byteorder="little", signed=False)


def _paired_gain_payload(
    candidate_returns: Sequence[float],
    baseline_returns: Sequence[float],
    *,
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    candidate = np.asarray(candidate_returns, dtype=np.float64)
    baseline = np.asarray(baseline_returns, dtype=np.float64)
    if candidate.shape != baseline.shape or candidate.ndim != 1 or not candidate.size:
        raise ValueError("Paired return arrays must be non-empty and equally shaped.")
    gains = candidate - baseline
    low, high = paired_bootstrap_mean_interval(
        gains,
        seed=int(bootstrap_seed),
        samples=int(bootstrap_samples),
    )
    return {
        "samples": gains.tolist(),
        "mean": float(gains.mean()),
        "median": float(np.median(gains)),
        "positive_fraction": float(np.mean(gains > 0.0)),
        "mean_ci95_low": float(low),
        "mean_ci95_high": float(high),
    }


def _state_scalar_summary(
    values: Sequence[float],
    *,
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not array.size:
        raise ValueError("State-level values must be a non-empty vector.")
    low, high = paired_bootstrap_mean_interval(
        array,
        seed=int(bootstrap_seed),
        samples=int(bootstrap_samples),
    )
    return {
        "state_count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p90": float(np.quantile(array, 0.90)),
        "positive_fraction": float(np.mean(array > 0.0)),
        "mean_ci95_low": float(low),
        "mean_ci95_high": float(high),
    }


def analyze_full_possession_planning(
    runtime: JaxCounterfactualRuntime,
    sample: SampledDecision,
    *,
    protocol: MctsRegretConfig,
    root_planner_action: tuple[int, ...],
    paired_rollout_seeds: Sequence[int],
    policy_returns: Sequence[float],
    root_planner_returns: Sequence[float],
    planning_seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    """Evaluate depth-primary MCTS replanning through possession end."""

    cache: dict[str, tuple[tuple[int, ...], dict[str, Any]]] = {}
    root_key = runtime.state_key(sample.state)
    cache[root_key] = (
        tuple(int(value) for value in root_planner_action),
        {"source": "root_only_primary_search"},
    )
    planner_search_runtime = 0.0
    planner_search_count = 0
    forced_policy_decisions = 0

    def select_action(state: Any, _decision_index: int) -> tuple[int, ...]:
        nonlocal planner_search_runtime, planner_search_count, forced_policy_decisions
        state_key = runtime.state_key(state)
        cached = cache.get(state_key)
        if cached is not None:
            return cached[0]
        if hasattr(runtime, "is_eligible_live_decision"):
            eligible, reason = runtime.is_eligible_live_decision(state)
            if not eligible:
                if reason != "forced_joint_action":
                    raise RuntimeError(
                        "Full-possession planner reached an ineligible state "
                        f"before possession termination: {reason}."
                    )
                action = tuple(runtime.policy_decision(state).deterministic_action)
                forced_policy_decisions += 1
                cache[state_key] = (
                    action,
                    {"source": "forced_policy_action", "reason": reason},
                )
                return action
        search_seed = _state_planning_seed(int(planning_seed), state_key)
        started = perf_counter()
        search = run_joint_action_mcts(
            state,
            state_key=runtime.state_key,
            propose_actions=runtime.propose_actions,
            transition=runtime.transition,
            rollout_value=runtime.rollout_value,
            simulations=int(protocol.simulations_per_action),
            max_depth=int(protocol.primary_search_depth),
            seed=int(search_seed),
            config=protocol,
        )
        planner_search_runtime += float(perf_counter() - started)
        planner_search_count += 1
        action = tuple(search.selected_action)
        cache[state_key] = (
            action,
            {
                "source": "receding_horizon_search",
                "search_seed": int(search_seed),
                "search": search_result_payload(search),
            },
        )
        return action

    full_returns: list[float] = []
    rollout_rows: list[dict[str, Any]] = []
    started = perf_counter()
    for rollout_seed in paired_rollout_seeds:
        rollout = evaluate_receding_horizon_return(
            sample.state,
            select_action=select_action,
            transition=runtime.transition,
            seed=int(rollout_seed),
            max_steps=int(runtime.max_rollout_steps),
            discount=float(protocol.discount),
        )
        full_returns.append(float(rollout.value))
        rollout_rows.append(
            {
                "seed": int(rollout_seed),
                "value": float(rollout.value),
                "decision_count": int(rollout.decision_count),
                "terminated": bool(rollout.terminated),
                "actions": [list(action) for action in rollout.actions],
            }
        )
    total_runtime = float(perf_counter() - started)
    full_array = np.asarray(full_returns, dtype=np.float64)
    return {
        "mode": "full_possession_receding_horizon",
        "search_depth": int(protocol.primary_search_depth),
        "simulations_per_action_selection": int(protocol.simulations_per_action),
        "planning_seed": int(planning_seed),
        "value": float(full_array.mean()),
        "returns": full_returns,
        "terminated_rollout_count": int(
            sum(bool(row["terminated"]) for row in rollout_rows)
        ),
        "step_limited_rollout_count": int(
            sum(not bool(row["terminated"]) for row in rollout_rows)
        ),
        "planner_search_count": int(planner_search_count),
        "forced_policy_decision_count": int(forced_policy_decisions),
        "unique_planned_state_count": len(cache),
        "planner_search_runtime_seconds": float(planner_search_runtime),
        "total_runtime_seconds": total_runtime,
        "full_minus_policy": _paired_gain_payload(
            full_returns,
            policy_returns,
            bootstrap_seed=int(planning_seed),
            bootstrap_samples=int(bootstrap_samples),
        ),
        "full_minus_root_only": _paired_gain_payload(
            full_returns,
            root_planner_returns,
            bootstrap_seed=int(planning_seed) + 1,
            bootstrap_samples=int(bootstrap_samples),
        ),
        "rollouts": rollout_rows,
        "planned_states": {
            key: details for key, (_action, details) in sorted(cache.items())
        },
    }


def analyze_sample_root_actions(
    runtime: JaxCounterfactualRuntime,
    sample: SampledDecision,
    *,
    protocol: MctsRegretConfig,
    seed: int,
    paired_rollout_count: int,
    bootstrap_samples: int = 2000,
    include_full_possession: bool = False,
    paired_return_cache: dict[tuple[int, ...], tuple[float, ...]] | None = None,
) -> dict[str, Any]:
    """Run each requested depth and compare its action with the policy action."""

    if int(paired_rollout_count) < 1:
        raise ValueError("paired_rollout_count must be positive.")
    policy_decision = runtime.policy_decision(sample.state)
    policy_action = tuple(int(value) for value in sample.policy_action)
    for player_index, action in enumerate(policy_action):
        if not policy_decision.legal_mask[player_index, action]:
            raise ValueError(
                f"Sample {sample.state_id} contains an illegal policy action."
            )
    policy_probability = joint_action_probability(
        policy_action,
        policy_decision.probabilities,
        policy_decision.legal_mask,
    )
    top_policy_actions = top_factorized_joint_actions(
        policy_decision.probabilities,
        policy_decision.legal_mask,
        count=10,
    )
    top_policy_ranks = {
        proposal.action: rank
        for rank, proposal in enumerate(top_policy_actions, start=1)
    }

    root_sequence = np.random.SeedSequence(int(seed))
    depth_sequences = root_sequence.spawn(len(protocol.search_depths) + 2)
    search_sequences = depth_sequences[: len(protocol.search_depths)]
    paired_seeds = _seed_tuple(depth_sequences[-2], int(paired_rollout_count))
    planning_seed = int(depth_sequences[-1].generate_state(1, dtype=np.uint32)[0])
    depth_results: dict[str, Any] = {}
    return_cache = paired_return_cache if paired_return_cache is not None else {}
    for depth, depth_sequence in zip(
        protocol.search_depths,
        search_sequences,
        strict=True,
    ):
        search_seed = int(depth_sequence.generate_state(1, dtype=np.uint32)[0])
        started = perf_counter()
        search = run_joint_action_mcts(
            sample.state,
            state_key=runtime.state_key,
            propose_actions=runtime.propose_actions,
            transition=runtime.transition,
            rollout_value=runtime.rollout_value,
            simulations=int(protocol.simulations_per_action),
            max_depth=int(depth),
            seed=search_seed,
            config=protocol,
        )
        search_runtime_seconds = float(perf_counter() - started)
        comparison = compare_actions_with_paired_rollouts(
            sample.state,
            policy_action=policy_action,
            planner_action=search.selected_action,
            transition=runtime.transition,
            rollout_value=runtime.rollout_value,
            rollout_seeds=paired_seeds,
            discount=float(protocol.discount),
            bootstrap_seed=search_seed,
            bootstrap_samples=int(bootstrap_samples),
            precomputed_policy_returns=return_cache.get(policy_action),
            precomputed_planner_returns=return_cache.get(tuple(search.selected_action)),
        )
        return_cache.setdefault(policy_action, tuple(comparison.policy_returns))
        return_cache.setdefault(
            tuple(search.selected_action),
            tuple(comparison.planner_returns),
        )
        planner_probability = joint_action_probability(
            search.selected_action,
            policy_decision.probabilities,
            policy_decision.legal_mask,
        )
        agreement = joint_action_agreement_payload(
            policy_action,
            search.selected_action,
        )
        planner_rank = top_policy_ranks.get(tuple(search.selected_action))
        depth_results[str(depth)] = {
            "depth": int(depth),
            "search_seed": int(search_seed),
            "search_runtime_seconds": search_runtime_seconds,
            "simulations_per_action_selection": int(protocol.simulations_per_action),
            "planner_action_policy_probability": float(planner_probability),
            "planner_action_policy_rank_top_10": planner_rank,
            "planner_action_within_policy_top_k": {
                str(k): bool(planner_rank is not None and planner_rank <= k)
                for k in (1, 3, 5, 10)
            },
            **agreement,
            "search": search_result_payload(search),
            "paired_comparison": paired_comparison_payload(comparison),
        }

    result = {
        "schema_version": MCTS_REGRET_SCHEMA_VERSION,
        "sample": sample.metadata(),
        "policy_action": list(policy_action),
        "policy_action_probability": float(policy_probability),
        "paired_rollout_count": int(paired_rollout_count),
        "paired_rollout_seeds": list(paired_seeds),
        "depth_results": depth_results,
    }
    if include_full_possession:
        primary = depth_results[str(protocol.primary_search_depth)]
        paired = primary["paired_comparison"]
        result["full_possession"] = analyze_full_possession_planning(
            runtime,
            sample,
            protocol=protocol,
            root_planner_action=tuple(primary["search"]["selected_action"]),
            paired_rollout_seeds=paired_seeds,
            policy_returns=paired["policy_returns"],
            root_planner_returns=paired["planner_returns"],
            planning_seed=planning_seed,
            bootstrap_samples=int(bootstrap_samples),
        )
    return result


def summarize_root_action_results(
    records: Sequence[Mapping[str, Any]],
    *,
    protocol: MctsRegretConfig,
    bootstrap_seed: int,
    bootstrap_samples: int = 2000,
) -> dict[str, Any]:
    if not records:
        raise ValueError("At least one analyzed state is required.")
    by_depth: dict[str, Any] = {}
    for offset, depth in enumerate(protocol.search_depths):
        depth_key = str(depth)
        regret_records: list[RegretRecord] = []
        role_records: dict[str, list[RegretRecord]] = {"offense": [], "defense": []}
        runtimes: list[float] = []
        player_agreements: list[bool] = []
        family_agreements: list[bool] = []
        pass_target_agreements: list[bool] = []
        for record in records:
            sample = dict(record["sample"])
            depth_result = dict(record["depth_results"][depth_key])
            paired = dict(depth_result["paired_comparison"])
            regret = RegretRecord(
                state_id=str(sample["state_id"]),
                policy_value=float(paired["policy_value"]),
                planner_value=float(paired["planner_value"]),
                policy_action=tuple(int(value) for value in paired["policy_action"]),
                planner_action=tuple(int(value) for value in paired["planner_action"]),
            )
            regret_records.append(regret)
            role = str(sample["evaluated_role"])
            role_records.setdefault(role, []).append(regret)
            runtimes.append(float(depth_result["search_runtime_seconds"]))
            player_agreements.extend(
                bool(value) for value in depth_result["player_action_agreement"]
            )
            family_agreements.extend(
                bool(value) for value in depth_result["player_action_family_agreement"]
            )
            pass_target_agreements.extend(
                bool(value)
                for value in depth_result["player_pass_target_agreement"]
                if value is not None
            )
        summary = summarize_regret_records(
            regret_records,
            bootstrap_seed=int(bootstrap_seed) + offset,
            bootstrap_samples=int(bootstrap_samples),
        )
        summary["search_runtime_seconds_total"] = float(sum(runtimes))
        summary["search_runtime_seconds_mean"] = float(np.mean(runtimes))
        summary["controlled_player_action_agreement_rate"] = float(
            np.mean(player_agreements)
        )
        summary["controlled_player_action_family_agreement_rate"] = float(
            np.mean(family_agreements)
        )
        summary["pass_target_comparison_count"] = len(pass_target_agreements)
        summary["pass_target_agreement_rate"] = (
            None
            if not pass_target_agreements
            else float(np.mean(pass_target_agreements))
        )
        summary["by_role"] = {
            role: summarize_regret_records(
                role_values,
                bootstrap_seed=int(bootstrap_seed) + 100 + offset,
                bootstrap_samples=int(bootstrap_samples),
            )
            for role, role_values in sorted(role_records.items())
            if role_values
        }
        by_depth[depth_key] = summary

    primary_key = str(protocol.primary_search_depth)
    depth_one = by_depth.get("1")
    primary = by_depth[primary_key]
    result = {
        "schema_version": MCTS_REGRET_SCHEMA_VERSION,
        "state_count": len(records),
        "primary_search_depth": int(protocol.primary_search_depth),
        "by_depth": by_depth,
        "primary_regret_mean": float(primary["regret_mean"]),
        "primary_exact_action_agreement_rate": float(
            primary["exact_action_agreement_rate"]
        ),
        "primary_minus_depth_one_regret_mean": (
            None
            if depth_one is None
            else float(primary["regret_mean"]) - float(depth_one["regret_mean"])
        ),
    }
    if depth_one is not None:
        deeper_gains = [
            float(
                record["depth_results"][primary_key]["paired_comparison"][
                    "planner_value"
                ]
            )
            - float(record["depth_results"]["1"]["paired_comparison"]["planner_value"])
            for record in records
        ]
        result["primary_planner_gain_over_depth_one"] = _state_scalar_summary(
            deeper_gains,
            bootstrap_seed=int(bootstrap_seed) + 400,
            bootstrap_samples=int(bootstrap_samples),
        )
    full_flags = ["full_possession" in record for record in records]
    if any(full_flags) and not all(full_flags):
        raise ValueError("Full-possession results are missing for some states.")
    if all(full_flags):
        full_values = [float(record["full_possession"]["value"]) for record in records]
        policy_gains = [
            float(record["full_possession"]["full_minus_policy"]["mean"])
            for record in records
        ]
        root_gains = [
            float(record["full_possession"]["full_minus_root_only"]["mean"])
            for record in records
        ]
        roles = sorted({str(record["sample"]["evaluated_role"]) for record in records})
        result["full_possession"] = {
            "value": _state_scalar_summary(
                full_values,
                bootstrap_seed=int(bootstrap_seed) + 500,
                bootstrap_samples=int(bootstrap_samples),
            ),
            "gain_over_policy": _state_scalar_summary(
                policy_gains,
                bootstrap_seed=int(bootstrap_seed) + 501,
                bootstrap_samples=int(bootstrap_samples),
            ),
            "additional_gain_over_root_only": _state_scalar_summary(
                root_gains,
                bootstrap_seed=int(bootstrap_seed) + 502,
                bootstrap_samples=int(bootstrap_samples),
            ),
            "planner_search_count_total": int(
                sum(
                    int(record["full_possession"]["planner_search_count"])
                    for record in records
                )
            ),
            "planner_search_runtime_seconds_total": float(
                sum(
                    float(record["full_possession"]["planner_search_runtime_seconds"])
                    for record in records
                )
            ),
            "step_limited_rollout_count": int(
                sum(
                    int(record["full_possession"]["step_limited_rollout_count"])
                    for record in records
                )
            ),
            "by_role": {
                role: {
                    "gain_over_policy": _state_scalar_summary(
                        [
                            float(
                                record["full_possession"]["full_minus_policy"]["mean"]
                            )
                            for record in records
                            if str(record["sample"]["evaluated_role"]) == role
                        ],
                        bootstrap_seed=int(bootstrap_seed) + 510 + index,
                        bootstrap_samples=int(bootstrap_samples),
                    ),
                    "additional_gain_over_root_only": _state_scalar_summary(
                        [
                            float(
                                record["full_possession"]["full_minus_root_only"][
                                    "mean"
                                ]
                            )
                            for record in records
                            if str(record["sample"]["evaluated_role"]) == role
                        ],
                        bootstrap_seed=int(bootstrap_seed) + 520 + index,
                        bootstrap_samples=int(bootstrap_samples),
                    ),
                }
                for index, role in enumerate(roles)
            },
        }
    return result


def _running_primary_regret_point(
    records: Sequence[Mapping[str, Any]],
    *,
    protocol: MctsRegretConfig,
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    """Summarize the primary-depth estimate while a study is running."""

    primary_key = str(protocol.primary_search_depth)

    def summarize(
        selected: Sequence[Mapping[str, Any]], *, seed: int
    ) -> dict[str, Any]:
        regrets = [
            float(
                record["depth_results"][primary_key]["paired_comparison"]["regret_mean"]
            )
            for record in selected
        ]
        low, high = paired_bootstrap_mean_interval(
            regrets,
            seed=int(seed),
            samples=int(bootstrap_samples),
        )
        agreements = [
            bool(record["depth_results"][primary_key]["exact_action_agreement"])
            for record in selected
        ]
        return {
            "state_count": len(regrets),
            "regret_mean": float(np.mean(regrets)),
            "regret_mean_ci95_low": float(low),
            "regret_mean_ci95_high": float(high),
            "regret_positive_fraction": float(np.mean(np.asarray(regrets) > 0.0)),
            "exact_action_agreement_rate": float(np.mean(agreements)),
        }

    point = summarize(records, seed=int(bootstrap_seed))
    point["by_role"] = {
        role: summarize(
            [
                record
                for record in records
                if str(record["sample"]["evaluated_role"]) == role
            ],
            seed=int(bootstrap_seed) + 100,
        )
        for role in ("offense", "defense")
        if any(str(record["sample"]["evaluated_role"]) == role for record in records)
    }
    return point


def _write_regret_trajectory(
    destination: Path,
    *,
    trajectory: Sequence[Mapping[str, Any]],
    total_state_count: int,
    primary_search_depth: int,
    status: str,
    aggregate: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically update the live trajectory so readers never see partial JSON."""

    path = destination / "regret_trajectory.json"
    temporary_path = destination / ".regret_trajectory.json.tmp"
    payload: dict[str, Any] = {
        "schema_version": MCTS_REGRET_SCHEMA_VERSION,
        "status": str(status),
        "primary_search_depth": int(primary_search_depth),
        "completed_state_count": len(trajectory),
        "total_state_count": int(total_state_count),
        "trajectory": list(trajectory),
    }
    if aggregate is not None:
        payload["final_aggregate"] = dict(aggregate)
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)
    return path


def summarize_budget_convergence(
    records_by_budget: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    primary_search_depth: int,
    bootstrap_seed: int,
    bootstrap_samples: int = 2000,
) -> dict[str, Any]:
    budgets = tuple(sorted(int(value) for value in records_by_budget))
    if len(budgets) < 2:
        raise ValueError("Budget convergence requires at least two budgets.")
    depth_key = str(int(primary_search_depth))
    reference_ids: list[str] | None = None
    comparisons: list[dict[str, Any]] = []
    for comparison_index, (lower, upper) in enumerate(
        zip(budgets[:-1], budgets[1:], strict=True)
    ):
        lower_records = list(records_by_budget[lower])
        upper_records = list(records_by_budget[upper])
        lower_ids = [str(record["sample"]["state_id"]) for record in lower_records]
        upper_ids = [str(record["sample"]["state_id"]) for record in upper_records]
        if lower_ids != upper_ids:
            raise ValueError(
                "Budget convergence records must contain identical ordered states."
            )
        if reference_ids is None:
            reference_ids = lower_ids
        elif reference_ids != lower_ids:
            raise ValueError("Budget convergence state order changed between budgets.")
        action_matches: list[bool] = []
        regret_deltas: list[float] = []
        rows: list[dict[str, Any]] = []
        for lower_record, upper_record in zip(
            lower_records,
            upper_records,
            strict=True,
        ):
            lower_depth = lower_record["depth_results"][depth_key]
            upper_depth = upper_record["depth_results"][depth_key]
            lower_action = tuple(lower_depth["search"]["selected_action"])
            upper_action = tuple(upper_depth["search"]["selected_action"])
            lower_regret = float(lower_depth["paired_comparison"]["regret_mean"])
            upper_regret = float(upper_depth["paired_comparison"]["regret_mean"])
            matched = lower_action == upper_action
            delta = upper_regret - lower_regret
            action_matches.append(matched)
            regret_deltas.append(delta)
            rows.append(
                {
                    "state_id": str(lower_record["sample"]["state_id"]),
                    "lower_action": list(lower_action),
                    "upper_action": list(upper_action),
                    "action_stable": bool(matched),
                    "lower_regret": lower_regret,
                    "upper_regret": upper_regret,
                    "regret_delta": delta,
                }
            )
        delta_summary = _state_scalar_summary(
            regret_deltas,
            bootstrap_seed=int(bootstrap_seed) + comparison_index,
            bootstrap_samples=int(bootstrap_samples),
        )
        delta_summary["mean_absolute"] = float(
            np.mean(np.abs(np.asarray(regret_deltas, dtype=np.float64)))
        )
        comparisons.append(
            {
                "lower_budget": int(lower),
                "upper_budget": int(upper),
                "state_count": len(rows),
                "selected_action_stability_rate": float(np.mean(action_matches)),
                "regret_delta": delta_summary,
                "states": rows,
            }
        )
    requested_pair = next(
        (
            comparison
            for comparison in comparisons
            if comparison["lower_budget"] == 512 and comparison["upper_budget"] == 1024
        ),
        comparisons[-1],
    )
    return {
        "budgets": list(budgets),
        "primary_search_depth": int(primary_search_depth),
        "comparisons": comparisons,
        "main_budget_evidence": requested_pair,
        "main_budget_evidence_pair": [
            int(requested_pair["lower_budget"]),
            int(requested_pair["upper_budget"]),
        ],
        "automatic_budget_decision": None,
        "automatic_budget_decision_reason": (
            "No stability threshold is imposed by the evaluator; inspect action "
            "stability and paired regret deltas before retaining the lower budget."
        ),
    }


def run_budget_calibration_study(
    runtime: JaxCounterfactualRuntime,
    *,
    output_dir: str | Path,
    episode_count: int = 50,
    seed: int,
    protocol: MctsRegretConfig | None = None,
    paired_rollout_count: int = 128,
    max_episode_steps: int = 10_000,
    bootstrap_samples: int = 2000,
    progress_callback=None,
) -> dict[str, Any]:
    """Run all configured budgets over one immutable sampled-state set."""

    settings = protocol or MctsRegretConfig()
    samples, sampling_summary = sample_policy_decisions(
        runtime,
        episode_count=int(episode_count),
        seed=int(seed),
        max_episode_steps=int(max_episode_steps),
    )
    if not samples:
        raise RuntimeError("No eligible policy decisions were sampled.")
    destination = Path(output_dir)
    state_paths = persist_sampled_decisions(
        destination,
        samples,
        provenance=runtime.provenance,
        sampling_summary=sampling_summary,
        jax=runtime.jax,
    )
    records_by_budget: dict[int, list[dict[str, Any]]] = {}
    budget_payloads: dict[str, Any] = {}
    paired_return_caches: list[dict[tuple[int, ...], tuple[float, ...]]] = [
        {} for _sample in samples
    ]
    for budget_index, budget in enumerate(settings.calibration_budgets, start=1):
        budget_protocol = replace(
            settings,
            search_depths=(int(settings.primary_search_depth),),
            simulations_per_action=int(budget),
        )
        sample_sequences = np.random.SeedSequence(int(seed)).spawn(len(samples))
        records: list[dict[str, Any]] = []
        for sample_index, (sample, sequence, return_cache) in enumerate(
            zip(samples, sample_sequences, paired_return_caches, strict=True),
            start=1,
        ):
            sample_seed = int(sequence.generate_state(1, dtype=np.uint32)[0])
            records.append(
                analyze_sample_root_actions(
                    runtime,
                    sample,
                    protocol=budget_protocol,
                    seed=sample_seed,
                    paired_rollout_count=int(paired_rollout_count),
                    bootstrap_samples=int(bootstrap_samples),
                    include_full_possession=False,
                    paired_return_cache=return_cache,
                )
            )
            if progress_callback is not None:
                progress_callback(
                    budget_index,
                    len(settings.calibration_budgets),
                    sample_index,
                    len(samples),
                )
        records_by_budget[int(budget)] = records
        budget_payloads[str(int(budget))] = {
            "protocol": asdict(budget_protocol),
            "aggregate": summarize_root_action_results(
                records,
                protocol=budget_protocol,
                bootstrap_seed=int(seed) + int(budget),
                bootstrap_samples=int(bootstrap_samples),
            ),
            "records": records,
        }
    convergence = summarize_budget_convergence(
        records_by_budget,
        primary_search_depth=int(settings.primary_search_depth),
        bootstrap_seed=int(seed),
        bootstrap_samples=int(bootstrap_samples),
    )
    payload = {
        "schema_version": MCTS_REGRET_SCHEMA_VERSION,
        "mode": "search_budget_calibration",
        "sampling_summary": sampling_summary,
        "provenance": dict(runtime.provenance),
        "base_protocol": asdict(settings),
        "convergence": convergence,
        "budgets": budget_payloads,
        "sample_artifacts": state_paths,
    }
    results_path = destination / "budget_calibration.json"
    results_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "output_dir": str(destination),
        "state_archive": state_paths["state_archive"],
        "sample_metadata": state_paths["metadata"],
        "results": str(results_path),
        "convergence": convergence,
    }


def run_root_only_study(
    runtime: JaxCounterfactualRuntime,
    *,
    output_dir: str | Path,
    episode_count: int,
    seed: int,
    protocol: MctsRegretConfig | None = None,
    paired_rollout_count: int = 128,
    max_episode_steps: int = 10_000,
    bootstrap_samples: int = 2000,
    include_full_possession: bool = False,
    sample_metadata_path: str | Path | None = None,
    evaluated_role: str | None = None,
    progress_callback=None,
) -> dict[str, Any]:
    """Sample states, run MCTS comparisons, and persist immutable artifacts."""

    settings = protocol or MctsRegretConfig()
    samples, sampling_summary = _study_samples(
        runtime,
        episode_count=int(episode_count),
        seed=int(seed),
        max_episode_steps=int(max_episode_steps),
        sample_metadata_path=sample_metadata_path,
        evaluated_role=evaluated_role,
    )
    if not samples:
        raise RuntimeError("No eligible policy decisions were sampled.")
    destination = Path(output_dir)
    state_paths = persist_sampled_decisions(
        destination,
        samples,
        provenance=runtime.provenance,
        sampling_summary=sampling_summary,
        jax=runtime.jax,
    )
    study_sequences = np.random.SeedSequence(int(seed)).spawn(len(samples))
    records: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    trajectory_path = destination / "regret_trajectory.json"
    for index, (sample, sequence) in enumerate(
        zip(samples, study_sequences, strict=True),
        start=1,
    ):
        sample_seed = int(sequence.generate_state(1, dtype=np.uint32)[0])
        records.append(
            analyze_sample_root_actions(
                runtime,
                sample,
                protocol=settings,
                seed=sample_seed,
                paired_rollout_count=int(paired_rollout_count),
                bootstrap_samples=int(bootstrap_samples),
                include_full_possession=bool(include_full_possession),
            )
        )
        trajectory.append(
            _running_primary_regret_point(
                records,
                protocol=settings,
                bootstrap_seed=int(seed),
                bootstrap_samples=int(bootstrap_samples),
            )
        )
        trajectory_path = _write_regret_trajectory(
            destination,
            trajectory=trajectory,
            total_state_count=len(samples),
            primary_search_depth=int(settings.primary_search_depth),
            status="running",
        )
        if progress_callback is not None:
            progress_callback(index, len(samples))

    aggregate = summarize_root_action_results(
        records,
        protocol=settings,
        bootstrap_seed=int(seed),
        bootstrap_samples=int(bootstrap_samples),
    )
    trajectory_path = _write_regret_trajectory(
        destination,
        trajectory=trajectory,
        total_state_count=len(samples),
        primary_search_depth=int(settings.primary_search_depth),
        status="complete",
        aggregate=aggregate,
    )
    payload = {
        "schema_version": MCTS_REGRET_SCHEMA_VERSION,
        "mode": (
            "root_and_full_possession" if include_full_possession else "root_only"
        ),
        "protocol": asdict(settings),
        "sampling_summary": sampling_summary,
        "provenance": dict(runtime.provenance),
        "aggregate": aggregate,
        "records": records,
        "sample_artifacts": state_paths,
        "trajectory_artifact": str(trajectory_path),
    }
    results_path = destination / (
        "policy_regret_study.json"
        if include_full_possession
        else "root_action_comparison.json"
    )
    results_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "output_dir": str(destination),
        "state_archive": state_paths["state_archive"],
        "sample_metadata": state_paths["metadata"],
        "trajectory": str(trajectory_path),
        "results": str(results_path),
        "aggregate": aggregate,
    }
