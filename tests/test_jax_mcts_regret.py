from __future__ import annotations

import re

import numpy as np
import pytest

from basketworld_jax.eval.mcts_regret import (
    DEFAULT_CALIBRATION_BUDGETS,
    DEFAULT_SEARCH_DEPTHS,
    JointActionProposal,
    MctsRegretConfig,
    MctsTransition,
    RegretRecord,
    calibration_protocol_payload,
    compare_actions_with_paired_rollouts,
    evaluate_receding_horizon_return,
    joint_action_probability,
    joint_action_agreement_payload,
    run_joint_action_mcts,
    sample_factorized_joint_actions,
    search_result_payload,
    summarize_regret_records,
    top_factorized_joint_actions,
)


def test_default_protocol_matches_issue_76_plan() -> None:
    payload = calibration_protocol_payload()

    assert payload["search_depths"] == list(DEFAULT_SEARCH_DEPTHS) == [1, 3, 5]
    assert payload["primary_search_depth"] == 5
    assert payload["simulations_per_action"] == 512
    assert payload["calibration_budgets"] == list(DEFAULT_CALIBRATION_BUDGETS)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"search_depths": (3, 1)}, "unique and increasing"),
        ({"primary_search_depth": 2}, "present in search_depths"),
        ({"simulations_per_action": 0}, "positive"),
        ({"progressive_widening_alpha": 0.0}, "in (0, 1]"),
        ({"discount": 1.1}, "in [0, 1]"),
    ],
)
def test_protocol_rejects_invalid_settings(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        MctsRegretConfig(**kwargs)


def test_joint_action_sampling_respects_every_player_mask_and_deduplicates() -> None:
    probabilities = np.asarray(
        [
            [0.10, 0.90, 0.00],
            [0.80, 0.20, 0.00],
        ]
    )
    legal_mask = np.asarray(
        [
            [True, True, False],
            [True, True, False],
        ]
    )

    proposals = sample_factorized_joint_actions(
        probabilities,
        legal_mask,
        count=4,
        rng=np.random.default_rng(7),
    )

    assert proposals[0].action == (1, 0)
    assert len({proposal.action for proposal in proposals}) == len(proposals) == 4
    assert all(action_index < 2 for item in proposals for action_index in item.action)
    assert proposals[0].prior == pytest.approx(0.9 * 0.8)
    assert joint_action_probability((0, 1), probabilities, legal_mask) == pytest.approx(
        0.1 * 0.2
    )

    excluded = sample_factorized_joint_actions(
        probabilities,
        legal_mask,
        count=4,
        rng=np.random.default_rng(7),
        excluded={(1, 0)},
    )
    assert all(proposal.action != (1, 0) for proposal in excluded)


def test_joint_action_agreement_separates_family_and_pass_target() -> None:
    payload = joint_action_agreement_payload(
        (1, 8, 7, 0),
        (2, 9, 7, 0),
    )

    assert payload["exact_action_agreement"] is False
    assert payload["player_action_agreement"] == [False, False, True, True]
    assert payload["player_action_family_agreement"] == [True, True, True, True]
    assert payload["player_pass_target_agreement"] == [None, False, None, None]


def test_exact_top_joint_actions_do_not_enumerate_irrelevant_masked_actions() -> None:
    probabilities = np.asarray([[0.6, 0.4, 10.0], [0.7, 0.3, 10.0]])
    legal_mask = np.asarray([[True, True, False], [True, True, False]])

    top = top_factorized_joint_actions(probabilities, legal_mask, count=4)

    assert [item.action for item in top] == [(0, 0), (1, 0), (0, 1), (1, 1)]
    assert [item.prior for item in top] == pytest.approx([0.42, 0.28, 0.18, 0.12])


def _fixed_proposals(actions: tuple[int, ...]):
    def propose(_state, _rng, count, existing):
        return tuple(
            JointActionProposal((action,), 1.0 / len(actions))
            for action in actions
            if (action,) not in existing
        )[:count]

    return propose


def test_one_step_mcts_selects_the_higher_value_joint_action() -> None:
    result = run_joint_action_mcts(
        "root",
        state_key=lambda state: state,
        propose_actions=_fixed_proposals((0, 1)),
        transition=lambda state, action, rng: MctsTransition(
            state=state,
            reward=float(action[0]),
            terminal=True,
        ),
        rollout_value=lambda state, rng: 0.0,
        simulations=100,
        max_depth=1,
        seed=3,
        config=MctsRegretConfig(
            exploration_c=1.0,
            progressive_widening_c=2.0,
            max_joint_actions_per_node=2,
        ),
    )

    assert result.selected_action == (1,)
    assert result.selected_value == pytest.approx(1.0)
    assert sum(item.visits for item in result.action_stats) == 100
    assert search_result_payload(result)["selected_action"] == [1]


def test_deeper_search_can_find_delayed_possession_value() -> None:
    def propose(state, _rng, count, existing):
        actions = (0, 1) if state == "root" else (0,)
        return tuple(
            JointActionProposal((action,), 1.0 / len(actions))
            for action in actions
            if (action,) not in existing
        )[:count]

    def transition(state, action, _rng):
        if state == "root" and action == (0,):
            return MctsTransition("done", reward=0.25, terminal=True)
        if state == "root" and action == (1,):
            return MctsTransition("delayed", reward=0.0, terminal=False)
        return MctsTransition("done", reward=1.0, terminal=True)

    config = MctsRegretConfig(
        exploration_c=1.0,
        progressive_widening_c=2.0,
        max_joint_actions_per_node=2,
    )
    shallow = run_joint_action_mcts(
        "root",
        state_key=lambda state: state,
        propose_actions=propose,
        transition=transition,
        rollout_value=lambda state, rng: 0.0,
        simulations=100,
        max_depth=1,
        seed=11,
        config=config,
    )
    deep = run_joint_action_mcts(
        "root",
        state_key=lambda state: state,
        propose_actions=propose,
        transition=transition,
        rollout_value=lambda state, rng: 0.0,
        simulations=100,
        max_depth=2,
        seed=11,
        config=config,
    )

    assert shallow.selected_action == (0,)
    assert deep.selected_action == (1,)
    assert deep.selected_value > shallow.selected_value


def test_stochastic_search_is_reproducible_from_seed() -> None:
    def search():
        return run_joint_action_mcts(
            "root",
            state_key=lambda state: state,
            propose_actions=_fixed_proposals((0, 1)),
            transition=lambda state, action, rng: MctsTransition(
                state=state,
                reward=float(action[0]) + float(rng.normal(scale=0.1)),
                terminal=True,
            ),
            rollout_value=lambda state, rng: 0.0,
            simulations=64,
            max_depth=1,
            seed=19,
            config=MctsRegretConfig(max_joint_actions_per_node=2),
        )

    assert search() == search()


def test_policy_and_planner_actions_use_common_random_numbers() -> None:
    def transition(state, action, rng):
        shared_noise = float(rng.normal())
        return MctsTransition(
            state=(state, shared_noise),
            reward=float(action[0]) + shared_noise,
            terminal=False,
        )

    def rollout(state, rng):
        return float(state[1]) + float(rng.normal())

    comparison = compare_actions_with_paired_rollouts(
        "root",
        policy_action=(0,),
        planner_action=(1,),
        transition=transition,
        rollout_value=rollout,
        rollout_seeds=(3, 5, 7, 11),
        bootstrap_samples=100,
    )

    # Both actions see the same transition and continuation noise for each
    # seed, so their paired difference contains only the action effect.
    assert comparison.regret_samples == pytest.approx((1.0, 1.0, 1.0, 1.0))
    assert comparison.regret_mean == pytest.approx(1.0)
    assert comparison.regret_ci95_low == pytest.approx(1.0)
    assert comparison.regret_ci95_high == pytest.approx(1.0)


def test_paired_comparison_reuses_precomputed_action_returns() -> None:
    comparison = compare_actions_with_paired_rollouts(
        "root",
        policy_action=(0,),
        planner_action=(1,),
        transition=lambda *_args: (_ for _ in ()).throw(
            AssertionError("precomputed returns should avoid transitions")
        ),
        rollout_value=lambda *_args: (_ for _ in ()).throw(
            AssertionError("precomputed returns should avoid rollouts")
        ),
        rollout_seeds=(3, 5, 7),
        precomputed_policy_returns=(0.0, 1.0, 2.0),
        precomputed_planner_returns=(1.0, 2.0, 3.0),
        bootstrap_samples=50,
    )

    assert comparison.policy_returns == (0.0, 1.0, 2.0)
    assert comparison.planner_returns == (1.0, 2.0, 3.0)
    assert comparison.regret_samples == (1.0, 1.0, 1.0)


def test_paired_comparison_rejects_misaligned_precomputed_returns() -> None:
    with pytest.raises(ValueError, match="must align with rollout_seeds"):
        compare_actions_with_paired_rollouts(
            "root",
            policy_action=(0,),
            planner_action=(1,),
            transition=lambda *_args: MctsTransition("done", 0.0, True),
            rollout_value=lambda *_args: 0.0,
            rollout_seeds=(3, 5),
            precomputed_policy_returns=(0.0,),
            bootstrap_samples=10,
        )


def test_receding_horizon_planner_selects_each_action_until_possession_end() -> None:
    selected_states = []

    def select_action(state, decision_index):
        selected_states.append((state, decision_index))
        return (1,)

    def transition(state, action, rng):
        next_state = state + 1
        return MctsTransition(
            state=next_state,
            reward=float(action[0]),
            terminal=next_state == 3,
        )

    result = evaluate_receding_horizon_return(
        0,
        select_action=select_action,
        transition=transition,
        seed=5,
        max_steps=10,
        discount=0.5,
    )

    assert result.terminated
    assert result.decision_count == 3
    assert result.actions == ((1,), (1,), (1,))
    assert result.value == pytest.approx(1.0 + 0.5 + 0.25)
    assert selected_states == [(0, 0), (1, 1), (2, 2)]


def test_receding_horizon_reports_step_limit_without_inventing_termination() -> None:
    result = evaluate_receding_horizon_return(
        "state",
        select_action=lambda state, index: (0,),
        transition=lambda state, action, rng: MctsTransition(
            state=state,
            reward=0.0,
            terminal=False,
        ),
        seed=7,
        max_steps=2,
    )

    assert not result.terminated
    assert result.decision_count == 2


def test_regret_summary_reports_action_agreement_and_paired_uncertainty() -> None:
    records = [
        RegretRecord("a", 0.0, 1.0, (0, 1), (1, 1)),
        RegretRecord("b", 0.5, 0.5, (1, 0), (1, 0)),
        RegretRecord("c", 1.0, 1.5, (0, 0), (0, 1)),
    ]

    summary = summarize_regret_records(
        records,
        bootstrap_seed=5,
        bootstrap_samples=500,
    )

    assert summary["state_count"] == 3
    assert summary["exact_action_agreement_rate"] == pytest.approx(1.0 / 3.0)
    assert summary["regret_mean"] == pytest.approx(0.5)
    assert summary["regret_median"] == pytest.approx(0.5)
    assert summary["regret_positive_fraction"] == pytest.approx(2.0 / 3.0)
    assert summary["regret_mean_ci95_low"] <= summary["regret_mean"]
    assert summary["regret_mean_ci95_high"] >= summary["regret_mean"]
