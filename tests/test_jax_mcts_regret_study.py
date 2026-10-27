from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from basketworld_jax.eval.mcts_regret import (
    MctsRegretConfig,
    MctsTransition,
    sample_factorized_joint_actions,
)
from basketworld_jax.eval.mcts_regret_jax import PolicyDecision, SampledDecision
from basketworld_jax.eval.mcts_regret_study import (
    analyze_sample_root_actions,
    summarize_budget_convergence,
    summarize_root_action_results,
)


class _BanditRuntime:
    max_rollout_steps = 4

    def policy_decision(self, state):
        assert state == "root"
        return PolicyDecision(
            probabilities=np.asarray([[0.8, 0.2]], dtype=np.float64),
            legal_mask=np.asarray([[True, True]]),
            deterministic_action=(0,),
        )

    def state_key(self, state):
        return state

    def propose_actions(self, state, rng, count, existing):
        decision = self.policy_decision(state)
        return sample_factorized_joint_actions(
            decision.probabilities,
            decision.legal_mask,
            count=count,
            rng=rng,
            excluded=existing,
        )

    def transition(self, state, action, rng):
        assert state == "root"
        return MctsTransition(
            state="done",
            reward=float(action[0]) + float(rng.normal(scale=0.01)),
            terminal=True,
        )

    def rollout_value(self, state, rng):
        raise AssertionError("The bandit transition is terminal.")


class _TwoStepRuntime(_BanditRuntime):
    def policy_decision(self, state):
        assert state in {"root", "middle"}
        return PolicyDecision(
            probabilities=np.asarray([[0.8, 0.2]], dtype=np.float64),
            legal_mask=np.asarray([[True, True]]),
            deterministic_action=(0,),
        )

    def transition(self, state, action, rng):
        assert state in {"root", "middle"}
        return MctsTransition(
            state="middle" if state == "root" else "done",
            reward=float(action[0]),
            terminal=state == "middle",
        )

    def rollout_value(self, state, rng):
        assert state == "middle"
        return 0.0

    def is_eligible_live_decision(self, state):
        return True, "eligible"


def _sample(state_id: str, role: str = "offense") -> SampledDecision:
    return SampledDecision(
        episode_index=0,
        episode_seed=1,
        policy_seed=2,
        selection_seed=3,
        environment_step=4,
        eligible_decision_index=2,
        eligible_decision_count=6,
        state_id=state_id,
        evaluated_role=role,  # type: ignore[arg-type]
        shot_clock=20,
        game_clock_remaining=0,
        team_a_score=0.0,
        team_b_score=0.0,
        ball_holder=0,
        legal_action_counts=(2,),
        policy_action=(0,),
        state="root",
    )


def test_root_analysis_reports_paired_regret_for_each_depth() -> None:
    protocol = MctsRegretConfig(
        search_depths=(1, 3),
        primary_search_depth=3,
        simulations_per_action=64,
        max_joint_actions_per_node=2,
    )

    result = analyze_sample_root_actions(
        _BanditRuntime(),  # type: ignore[arg-type]
        _sample("state-a"),
        protocol=protocol,
        seed=17,
        paired_rollout_count=16,
        bootstrap_samples=100,
    )

    assert result["policy_action"] == [0]
    assert result["policy_action_probability"] == pytest.approx(0.8)
    for depth in ("1", "3"):
        row = result["depth_results"][depth]
        assert row["search"]["selected_action"] == [1]
        assert row["exact_action_agreement"] is False
        assert row["planner_action_policy_probability"] == pytest.approx(0.2)
        assert row["planner_action_policy_rank_top_10"] == 2
        assert row["planner_action_within_policy_top_k"] == {
            "1": False,
            "3": True,
            "5": True,
            "10": True,
        }
        assert row["paired_comparison"]["regret_mean"] == pytest.approx(1.0)
        assert len(row["paired_comparison"]["regret_samples"]) == 16


def test_root_summary_splits_roles_and_compares_primary_with_depth_one() -> None:
    protocol = MctsRegretConfig(
        search_depths=(1, 3),
        primary_search_depth=3,
        simulations_per_action=64,
        max_joint_actions_per_node=2,
    )
    records = [
        analyze_sample_root_actions(
            _BanditRuntime(),  # type: ignore[arg-type]
            _sample("state-a", "offense"),
            protocol=protocol,
            seed=21,
            paired_rollout_count=8,
            bootstrap_samples=50,
        ),
        analyze_sample_root_actions(
            _BanditRuntime(),  # type: ignore[arg-type]
            _sample("state-b", "defense"),
            protocol=protocol,
            seed=22,
            paired_rollout_count=8,
            bootstrap_samples=50,
        ),
    ]

    summary = summarize_root_action_results(
        records,
        protocol=protocol,
        bootstrap_seed=9,
        bootstrap_samples=100,
    )

    assert summary["state_count"] == 2
    assert set(summary["by_depth"]["3"]["by_role"]) == {"offense", "defense"}
    assert summary["primary_regret_mean"] == pytest.approx(1.0)
    assert summary["primary_minus_depth_one_regret_mean"] == pytest.approx(0.0)
    assert summary["primary_planner_gain_over_depth_one"]["mean"] == pytest.approx(0.0)


def test_full_possession_mode_replans_after_the_root_action() -> None:
    protocol = MctsRegretConfig(
        search_depths=(1, 3),
        primary_search_depth=3,
        simulations_per_action=64,
        max_joint_actions_per_node=2,
    )

    result = analyze_sample_root_actions(
        _TwoStepRuntime(),  # type: ignore[arg-type]
        _sample("state-a"),
        protocol=protocol,
        seed=31,
        paired_rollout_count=8,
        bootstrap_samples=50,
        include_full_possession=True,
    )

    primary = result["depth_results"]["3"]["paired_comparison"]
    full = result["full_possession"]
    assert primary["planner_value"] == pytest.approx(1.0)
    assert full["value"] == pytest.approx(2.0)
    assert full["full_minus_root_only"]["mean"] == pytest.approx(1.0)
    assert full["full_minus_policy"]["mean"] == pytest.approx(2.0)
    assert full["planner_search_count"] == 1
    assert full["step_limited_rollout_count"] == 0
    assert all(row["actions"] == [[1], [1]] for row in full["rollouts"])

    summary = summarize_root_action_results(
        [result],
        protocol=protocol,
        bootstrap_seed=4,
        bootstrap_samples=50,
    )
    assert summary["full_possession"]["gain_over_policy"]["mean"] == pytest.approx(2.0)
    assert summary["full_possession"]["additional_gain_over_root_only"][
        "mean"
    ] == pytest.approx(1.0)


def test_budget_convergence_compares_identical_states_and_high_budgets() -> None:
    protocol = MctsRegretConfig(
        search_depths=(1, 3),
        primary_search_depth=3,
        simulations_per_action=64,
        max_joint_actions_per_node=2,
    )
    base = [
        analyze_sample_root_actions(
            _BanditRuntime(),  # type: ignore[arg-type]
            _sample("state-a", "offense"),
            protocol=protocol,
            seed=41,
            paired_rollout_count=4,
            bootstrap_samples=20,
        ),
        analyze_sample_root_actions(
            _BanditRuntime(),  # type: ignore[arg-type]
            _sample("state-b", "defense"),
            protocol=protocol,
            seed=42,
            paired_rollout_count=4,
            bootstrap_samples=20,
        ),
    ]
    lower = deepcopy(base)
    upper = deepcopy(base)
    lower[0]["depth_results"]["3"]["search"]["selected_action"] = [0]
    lower[0]["depth_results"]["3"]["paired_comparison"]["regret_mean"] = 0.0

    summary = summarize_budget_convergence(
        {512: lower, 1024: upper},
        primary_search_depth=3,
        bootstrap_seed=7,
        bootstrap_samples=50,
    )

    evidence = summary["main_budget_evidence"]
    assert summary["main_budget_evidence_pair"] == [512, 1024]
    assert evidence["selected_action_stability_rate"] == pytest.approx(0.5)
    assert evidence["regret_delta"]["mean"] == pytest.approx(0.5)
    assert evidence["regret_delta"]["mean_absolute"] == pytest.approx(0.5)
    assert summary["automatic_budget_decision"] is None
