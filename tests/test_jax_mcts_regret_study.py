from __future__ import annotations

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
    summarize_root_action_results,
)


class _BanditRuntime:
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
