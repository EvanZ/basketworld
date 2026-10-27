"""Study orchestration for JAX MCTS policy-regret analysis."""

from __future__ import annotations

from dataclasses import asdict
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
    joint_action_probability,
    paired_comparison_payload,
    run_joint_action_mcts,
    search_result_payload,
    summarize_regret_records,
)
from basketworld_jax.eval.mcts_regret_jax import (
    JaxCounterfactualRuntime,
    SampledDecision,
    persist_sampled_decisions,
    sample_policy_decisions,
)


def _seed_tuple(sequence: np.random.SeedSequence, count: int) -> tuple[int, ...]:
    return tuple(
        int(value) for value in sequence.generate_state(int(count), dtype=np.uint32)
    )


def analyze_sample_root_actions(
    runtime: JaxCounterfactualRuntime,
    sample: SampledDecision,
    *,
    protocol: MctsRegretConfig,
    seed: int,
    paired_rollout_count: int,
    bootstrap_samples: int = 2000,
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

    root_sequence = np.random.SeedSequence(int(seed))
    depth_sequences = root_sequence.spawn(len(protocol.search_depths) + 1)
    paired_seeds = _seed_tuple(depth_sequences[-1], int(paired_rollout_count))
    depth_results: dict[str, Any] = {}
    for depth, depth_sequence in zip(
        protocol.search_depths,
        depth_sequences[:-1],
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
        )
        planner_probability = joint_action_probability(
            search.selected_action,
            policy_decision.probabilities,
            policy_decision.legal_mask,
        )
        player_agreement = [
            bool(policy == planner)
            for policy, planner in zip(
                policy_action,
                search.selected_action,
                strict=True,
            )
        ]
        depth_results[str(depth)] = {
            "depth": int(depth),
            "search_seed": int(search_seed),
            "search_runtime_seconds": search_runtime_seconds,
            "simulations_per_action_selection": int(protocol.simulations_per_action),
            "planner_action_policy_probability": float(planner_probability),
            "exact_action_agreement": bool(all(player_agreement)),
            "player_action_agreement": player_agreement,
            "search": search_result_payload(search),
            "paired_comparison": paired_comparison_payload(comparison),
        }

    return {
        "schema_version": MCTS_REGRET_SCHEMA_VERSION,
        "sample": sample.metadata(),
        "policy_action": list(policy_action),
        "policy_action_probability": float(policy_probability),
        "paired_rollout_count": int(paired_rollout_count),
        "paired_rollout_seeds": list(paired_seeds),
        "depth_results": depth_results,
    }


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
        summary = summarize_regret_records(
            regret_records,
            bootstrap_seed=int(bootstrap_seed) + offset,
            bootstrap_samples=int(bootstrap_samples),
        )
        summary["search_runtime_seconds_total"] = float(sum(runtimes))
        summary["search_runtime_seconds_mean"] = float(np.mean(runtimes))
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
    return {
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
    progress_callback=None,
) -> dict[str, Any]:
    """Sample states, run root-only MCTS, and persist immutable artifacts."""

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
    study_sequences = np.random.SeedSequence(int(seed)).spawn(len(samples))
    records: list[dict[str, Any]] = []
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
            )
        )
        if progress_callback is not None:
            progress_callback(index, len(samples))

    aggregate = summarize_root_action_results(
        records,
        protocol=settings,
        bootstrap_seed=int(seed),
        bootstrap_samples=int(bootstrap_samples),
    )
    payload = {
        "schema_version": MCTS_REGRET_SCHEMA_VERSION,
        "mode": "root_only",
        "protocol": asdict(settings),
        "sampling_summary": sampling_summary,
        "provenance": dict(runtime.provenance),
        "aggregate": aggregate,
        "records": records,
        "sample_artifacts": state_paths,
    }
    results_path = destination / "root_action_comparison.json"
    results_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "output_dir": str(destination),
        "state_archive": state_paths["state_archive"],
        "sample_metadata": state_paths["metadata"],
        "results": str(results_path),
        "aggregate": aggregate,
    }
