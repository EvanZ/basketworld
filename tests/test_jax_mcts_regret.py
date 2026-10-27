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
    joint_action_probability,
    run_joint_action_mcts,
    sample_factorized_joint_actions,
    search_result_payload,
    summarize_regret_records,
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
