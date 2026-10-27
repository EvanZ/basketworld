from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from basketworld_jax.config import JaxEnvironmentConfig
from basketworld_jax.env.minimal import (
    TEAM_A,
    build_action_masks_batch,
    build_kernel_static_from_config,
    build_policy_observation_batch_with_role_flag,
    reset_batch_minimal,
)
from basketworld_jax.eval.mcts_regret_jax import (
    JaxCounterfactualRuntime,
    kernel_state_key,
    load_persisted_sample_states,
    meaningful_joint_choice,
    persist_sampled_decisions,
    sample_episode_decision,
    sample_policy_decisions,
)
from basketworld_jax.models import build_actor_critic_spec, init_actor_critic_params


def _runtime(
    *,
    evaluated_action_mode: str = "deterministic",
    opponent_action_mode: str = "deterministic",
    evaluated_seed: int = 5,
    opponent_seed: int = 5,
) -> tuple[JaxCounterfactualRuntime, object]:
    config = JaxEnvironmentConfig(
        grid_size=9,
        court_rows=9,
        court_cols=8,
        players=2,
        min_shot_clock=10,
        shot_clock=10,
    )
    static = build_kernel_static_from_config(config, jnp)
    state = reset_batch_minimal(
        static,
        jax.random.PRNGKey(3)[None, :],
        jax,
        jnp,
    )
    role_flag = jnp.ones((1,), dtype=jnp.float32)
    observations = build_policy_observation_batch_with_role_flag(
        static,
        state,
        role_flag,
        jnp,
        model_type="mlp",
    )
    masks = build_action_masks_batch(static, state, jnp)[:, static.offense_ids, :]
    spec = build_actor_critic_spec(
        np.asarray(observations),
        np.asarray(masks),
        (16,),
    )
    evaluated_params = init_actor_critic_params(
        jax,
        jnp,
        spec,
        seed=int(evaluated_seed),
    )
    opponent_params = init_actor_critic_params(
        jax,
        jnp,
        spec,
        seed=int(opponent_seed),
    )
    runtime = JaxCounterfactualRuntime(
        jax=jax,
        jnp=jnp,
        static=static,
        config=config,
        evaluated_params=evaluated_params,
        evaluated_spec=spec,
        opponent_params=opponent_params,
        opponent_spec=spec,
        evaluated_action_mode=evaluated_action_mode,  # type: ignore[arg-type]
        opponent_action_mode=opponent_action_mode,  # type: ignore[arg-type]
        max_rollout_steps=64,
    )
    return runtime, state


def test_meaningful_joint_choice_requires_a_non_forced_player() -> None:
    assert not meaningful_joint_choice(np.asarray([[1, 0], [0, 1]]))
    assert meaningful_joint_choice(np.asarray([[1, 1], [0, 1]]))
    assert not meaningful_joint_choice(np.asarray([[0, 0], [0, 1]]))


def test_runtime_exposes_complete_legal_joint_policy_decision() -> None:
    runtime, state = _runtime()

    decision = runtime.policy_decision(state)

    assert decision.probabilities.shape == decision.legal_mask.shape == (2, 14)
    assert len(decision.deterministic_action) == 2
    for player_index, action in enumerate(decision.deterministic_action):
        assert decision.legal_mask[player_index, action]
        assert decision.probabilities[player_index].sum() == pytest.approx(1.0)
    assert runtime.is_eligible_live_decision(state) == (True, "eligible")


def test_runtime_state_key_covers_all_kernel_state_fields() -> None:
    runtime, state = _runtime()

    original = kernel_state_key(state, jax=jax)
    changed = kernel_state_key(
        state._replace(shot_clock=state.shot_clock - 1),
        jax=jax,
    )

    assert original == runtime.state_key(state)
    assert changed != original


def test_runtime_forced_transition_uses_real_kernel_and_is_seeded() -> None:
    runtime, state = _runtime()
    action = runtime.policy_decision(state).deterministic_action

    first = runtime.transition(state, action, np.random.default_rng(17))
    second = runtime.transition(state, action, np.random.default_rng(17))

    assert first.reward == second.reward
    assert first.terminal == second.terminal
    assert runtime.state_key(first.state) == runtime.state_key(second.state)
    assert int(np.asarray(first.state.step_count)[0]) == 1


def test_sampled_policy_modes_are_seeded_and_legal() -> None:
    runtime, state = _runtime(
        evaluated_action_mode="sampled",
        opponent_action_mode="sampled",
        opponent_seed=7,
    )

    first = runtime.policy_step(state, np.random.default_rng(41))
    second = runtime.policy_step(state, np.random.default_rng(41))
    _, evaluated_mask, _, _ = runtime.jax.device_get(
        runtime._compiled_evaluated_policy(state)
    )
    _, opponent_mask, _, _ = runtime.jax.device_get(
        runtime._compiled_opponent_policy(state)
    )

    assert first[1:] == second[1:]
    assert runtime.state_key(first[0]) == runtime.state_key(second[0])
    assert all(
        evaluated_mask[0, player_index, action]
        for player_index, action in enumerate(first[1])
    )
    assert all(
        opponent_mask[0, player_index, action]
        for player_index, action in enumerate(first[2])
    )


def test_fixed_deterministic_opponent_is_not_changed_by_evaluated_sampling() -> None:
    runtime, state = _runtime(
        evaluated_action_mode="sampled",
        opponent_action_mode="deterministic",
        evaluated_seed=5,
        opponent_seed=17,
    )
    _, _, _, expected_actions = runtime.jax.device_get(
        runtime._compiled_opponent_policy(state)
    )
    expected = tuple(int(value) for value in np.asarray(expected_actions)[0])

    observed = [
        runtime.policy_step(state, np.random.default_rng(seed))[2]
        for seed in (3, 11, 29)
    ]

    assert observed == [expected, expected, expected]


def test_runtime_treats_environment_truncation_as_terminal() -> None:
    runtime, state = _runtime()
    action = runtime.policy_decision(state).deterministic_action
    runtime._compiled_forced_step = lambda *_args: SimpleNamespace(
        state=state,
        possession_ended=jnp.asarray([False]),
        done=jnp.asarray([False]),
        truncated=jnp.asarray([True]),
        team_a_score_delta=jnp.asarray([1.0]),
        team_b_score_delta=jnp.asarray([3.0]),
    )

    transition = runtime.transition(state, action, np.random.default_rng(13))

    assert transition.terminal is True
    assert transition.reward == pytest.approx(-2.0)


def test_runtime_rejects_wrong_joint_action_width() -> None:
    runtime, state = _runtime()

    with pytest.raises(ValueError, match="2-player joint action"):
        runtime.transition(state, (0,), np.random.default_rng(1))


def test_runtime_value_is_always_oriented_to_evaluated_stable_team_a() -> None:
    runtime, _ = _runtime()

    assert runtime._score_margin_delta(
        SimpleNamespace(
            team_a_score_delta=jnp.asarray([2.0]),
            team_b_score_delta=jnp.asarray([0.0]),
        )
    ) == pytest.approx(2.0)
    assert runtime._score_margin_delta(
        SimpleNamespace(
            team_a_score_delta=jnp.asarray([0.0]),
            team_b_score_delta=jnp.asarray([3.0]),
        )
    ) == pytest.approx(-3.0)


def test_runtime_reset_is_reproducible() -> None:
    runtime, _ = _runtime()

    assert runtime.state_key(runtime.reset(29)) == runtime.state_key(runtime.reset(29))
    assert int(np.asarray(runtime.reset(29).offense_team)[0]) == int(TEAM_A)


def test_episode_sampler_persists_exact_replay_state(tmp_path) -> None:
    runtime, _ = _runtime()

    result = sample_episode_decision(
        runtime,
        episode_index=0,
        episode_seed=101,
        policy_seed=102,
        selection_seed=103,
        max_episode_steps=64,
    )

    assert result.completed
    assert result.sample is not None
    assert result.eligible_decisions >= 1
    assert result.sample.eligible_decision_count == result.eligible_decisions
    assert result.sample.evaluated_role == "offense"
    assert result.sample.state_id == runtime.state_key(result.sample.state)

    paths = persist_sampled_decisions(
        tmp_path / "study-001",
        [result.sample],
        provenance={"test": True},
        sampling_summary={"sample_count": 1},
        jax=jax,
    )
    records, states, payload = load_persisted_sample_states(paths["metadata"])

    assert records[0]["state_id"] == result.sample.state_id
    assert runtime.state_key(states[0]) == result.sample.state_id
    assert payload["provenance"] == {"test": True}


def test_sample_artifacts_refuse_to_overwrite_existing_study(tmp_path) -> None:
    runtime, _ = _runtime()
    result = sample_episode_decision(
        runtime,
        episode_index=0,
        episode_seed=201,
        policy_seed=202,
        selection_seed=203,
        max_episode_steps=64,
    )
    assert result.sample is not None
    destination = tmp_path / "study-001"
    persist_sampled_decisions(
        destination,
        [result.sample],
        provenance={},
        sampling_summary={},
        jax=jax,
    )

    with pytest.raises(FileExistsError):
        persist_sampled_decisions(
            destination,
            [result.sample],
            provenance={},
            sampling_summary={},
            jax=jax,
        )


def test_compiled_batch_sampler_returns_one_replayable_state_per_episode() -> None:
    runtime, _ = _runtime()

    samples, summary = sample_policy_decisions(
        runtime,
        episode_count=2,
        seed=301,
        max_episode_steps=64,
    )

    assert summary["episode_count"] == 2
    assert summary["completed_episode_count"] == 2
    assert summary["sample_count"] == 2
    assert summary["selection_method"].startswith("compiled-")
    assert len(samples) == 2
    assert {sample.episode_index for sample in samples} == {0, 1}
    assert all(sample.eligible_decision_count >= 1 for sample in samples)
    assert all(sample.state_id == runtime.state_key(sample.state) for sample in samples)


def test_compiled_batch_sampling_is_reproducible_from_seed() -> None:
    runtime, _ = _runtime(
        evaluated_action_mode="sampled",
        opponent_action_mode="sampled",
        opponent_seed=7,
    )

    first_samples, first_summary = sample_policy_decisions(
        runtime,
        episode_count=2,
        seed=401,
        max_episode_steps=64,
    )
    second_samples, second_summary = sample_policy_decisions(
        runtime,
        episode_count=2,
        seed=401,
        max_episode_steps=64,
    )

    assert first_summary == second_summary
    assert [sample.metadata() for sample in first_samples] == [
        sample.metadata() for sample in second_samples
    ]
    assert [sample.state_id for sample in first_samples] == [
        runtime.state_key(sample.state) for sample in second_samples
    ]
