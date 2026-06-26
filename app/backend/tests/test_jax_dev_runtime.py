from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from app.backend import state as backend_state
from app.backend.jax_dev_runtime import JaxDevRuntime, _adapt_policy_observation_to_spec
from app.backend.routes import admin_routes
from app.backend.routes import lifecycle_routes
from app.backend.schemas import (
    ActionRequest,
    ApplyStartTemplateRequest,
    PlaybookAnalysisRequest,
    ReplayCounterfactualRequest,
    SetIntentStateRequest,
    SetOffenseSkillsRequest,
    StartSelfPlayRequest,
    UpdatePositionRequest,
)
from app.backend.state import GameState
from basketworld.envs.basketworld_env_v2 import Team
from basketworld_jax.env.minimal import SHOT_TYPE_DUNK


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


@dataclass(frozen=True)
class _FakeSpec:
    model_type: str = "mlp"
    attention_num_cls_tokens: int = 0
    num_intents: int = 8
    intent_selector_enabled: bool = False


@dataclass(frozen=True)
class _FakeAttentionSpec:
    model_type: str = "attention"
    flat_obs_dim: int = (10 * 17) + 7 + 1
    token_player_count: int = 10
    token_dim: int = 17
    global_dim: int = 7


def test_jax_dev_runtime_adapts_attention_observation_with_extra_token_and_global_features():
    spec = _FakeAttentionSpec()
    current_token_dim = 18
    current_global_dim = 8
    token_count = int(spec.token_player_count)
    current_dim = (token_count * current_token_dim) + current_global_dim + 1
    flat = jnp.arange(current_dim, dtype=jnp.float32)[None, :]

    adapted = np.asarray(_adapt_policy_observation_to_spec(flat, SimpleNamespace(), spec, jnp))

    assert adapted.shape == (1, int(spec.flat_obs_dim))
    current_players = np.asarray(flat[:, : token_count * current_token_dim]).reshape(1, token_count, current_token_dim)
    expected_players = current_players[:, :, : int(spec.token_dim)].reshape(1, token_count * int(spec.token_dim))
    np.testing.assert_allclose(adapted[:, : token_count * int(spec.token_dim)], expected_players)
    adapted_global_start = token_count * int(spec.token_dim)
    current_global_start = token_count * current_token_dim
    np.testing.assert_allclose(
        adapted[:, adapted_global_start : adapted_global_start + int(spec.global_dim)],
        np.asarray(flat[:, current_global_start : current_global_start + int(spec.global_dim)]),
    )
    np.testing.assert_allclose(adapted[:, -1], np.asarray(flat[:, current_global_start + current_global_dim]))


class _FakeRawJaxModel:
    metadata = {"policy_spec": {"model_type": "mlp"}}

    def __init__(self, action_bias: int | None = None, metadata: dict | None = None):
        self.jax = jax
        self.jnp = jnp
        self.params = {}
        self.spec = _FakeSpec()
        self.metadata = dict(metadata or self.metadata)
        self._sample_key = jax.random.PRNGKey(123)
        self.action_bias = action_bias

    def _masked_runner(self, params, flat_obs, team_action_mask, intent_context):
        legal = team_action_mask.astype(jnp.float32)
        denom = jnp.maximum(jnp.sum(legal, axis=-1, keepdims=True), 1.0)
        base_probs = legal / denom
        masked_logits = jnp.where(legal > 0, jnp.log(jnp.maximum(base_probs, 1.0e-8)), -1.0e9)
        if self.action_bias is not None:
            bias = jax.nn.one_hot(int(self.action_bias), team_action_mask.shape[-1]) * 4.0
            masked_logits = jnp.where(legal > 0, masked_logits + bias, -1.0e9)
        probs = jax.nn.softmax(masked_logits, axis=-1)
        return {
            "probs": probs,
            "masked_logits": masked_logits,
            "deterministic_actions": jnp.argmax(masked_logits, axis=-1).astype(jnp.int32),
            "values": jnp.full((flat_obs.shape[0],), 0.5, dtype=jnp.float32),
            "attention_weights": None,
        }


class _ExplodingPythonEnv:
    episode_ended = False

    def step(self, actions):
        raise AssertionError("legacy Python env step should not be called for JAX sessions")


def _make_runtime(
    *,
    rng_seed: int | None = None,
    reset_seed: int | None = 17,
    env_params: dict | None = None,
) -> JaxDevRuntime:
    policy = _FakeRawJaxModel()
    runtime_env_params = {
        "allow_dunks": True,
        "pass_mode": "pointer_targeted",
        "training_team": Team.OFFENSE,
    }
    runtime_env_params.update(dict(env_params or {}))
    runtime = JaxDevRuntime(
        required_params={"players": 3},
        env_params=runtime_env_params,
        unified_policy=policy,
        opponent_policy=policy,
        user_team=Team.OFFENSE,
        rng_seed=rng_seed,
    )
    if reset_seed is not None:
        runtime.reset(seed=reset_seed)
    return runtime


def test_state_snapshot_dispatches_to_jax_runtime(monkeypatch):
    class DummyRuntime:
        def __init__(self):
            self.kwargs = None

        def get_full_game_state(self, game_state, **kwargs):
            self.kwargs = dict(kwargs)
            return {"source": "jax-runtime"}

    fresh = GameState()
    runtime = DummyRuntime()
    fresh.jax_runtime = runtime
    monkeypatch.setattr(backend_state, "game_state", fresh)

    payload = backend_state.get_full_game_state(
        include_policy_probs=True,
        include_action_values=True,
        include_state_values=True,
    )

    assert payload == {"source": "jax-runtime"}
    assert runtime.kwargs == {
        "include_policy_probs": True,
        "include_action_values": True,
        "include_state_values": True,
    }


def test_step_route_dispatches_to_jax_runtime(monkeypatch):
    class DummyRuntime:
        def __init__(self):
            self.called = False

        def step(self, request, game_state):
            self.called = True
            return {"status": "success", "source": "jax-runtime"}

    fresh = GameState()
    fresh.jax_runtime = DummyRuntime()
    fresh.env = _ExplodingPythonEnv()
    monkeypatch.setattr(backend_state, "game_state", fresh)
    monkeypatch.setattr(lifecycle_routes, "game_state", fresh)

    body = lifecycle_routes.step(ActionRequest(actions={}))

    assert body == {"status": "success", "source": "jax-runtime"}
    assert fresh.jax_runtime.called is True


def test_jax_dev_runtime_step_does_not_call_python_env_step(monkeypatch):
    runtime = _make_runtime()
    game_state = GameState()
    game_state.jax_runtime = runtime
    game_state.env = runtime.display_env
    game_state.unified_policy = runtime.unified_policy
    game_state.defense_policy = runtime.opponent_policy
    game_state.user_team = Team.OFFENSE
    game_state.obs = runtime.observation_dict()

    runtime.display_env.step = _ExplodingPythonEnv().step
    choose_calls = 0
    original_choose = runtime._choose_joint_policy_actions

    def spy_choose_joint_policy_actions(*, player_deterministic, opponent_deterministic):
        nonlocal choose_calls
        choose_calls += 1
        return original_choose(
            player_deterministic=player_deterministic,
            opponent_deterministic=opponent_deterministic,
        )

    monkeypatch.setattr(runtime, "_choose_joint_policy_actions", spy_choose_joint_policy_actions)

    before_positions = list(runtime.positions)
    body = runtime.step(
        ActionRequest(
            actions={},
            player_deterministic=True,
            opponent_deterministic=True,
        ),
        game_state,
    )

    assert body["status"] == "success"
    assert runtime.last_step_output is not None
    assert isinstance(body["state"]["positions"], list)
    assert len(body["state"]["positions"]) == runtime.n_players
    assert game_state.actions_log
    assert np.asarray(runtime.positions).shape == np.asarray(before_positions).shape
    assert choose_calls == 2
    assert {
        str(pid): probs
        for pid, probs in runtime._last_policy_probs.items()
    } == body["state"]["policy_probabilities"]


def test_jax_dev_runtime_replace_policies_refreshes_policy_outputs():
    runtime = _make_runtime()
    game_state = GameState()
    game_state.jax_runtime = runtime
    game_state.env = runtime.display_env
    game_state.unified_policy = runtime.unified_policy
    game_state.defense_policy = runtime.opponent_policy
    game_state.user_team = Team.OFFENSE
    game_state.obs = runtime.observation_dict()

    _, old_probs = runtime._choose_joint_policy_actions(
        player_deterministic=True,
        opponent_deterministic=True,
    )
    runtime._last_attention_payload = {"stale": True}
    runtime._last_selector_transition = {"stale": True}
    runtime._playbook_batch_runner_cache = {"spec": runtime.raw_model.spec, "runner": object()}

    new_policy = _FakeRawJaxModel(action_bias=0)
    runtime.replace_policies(
        unified_policy=new_policy,
        opponent_policy=new_policy,
        game_state=game_state,
    )

    assert runtime.unified_policy is new_policy
    assert runtime.opponent_policy is new_policy
    assert runtime.raw_model is new_policy
    assert runtime._last_policy_probs is None
    assert runtime._last_attention_payload is None
    assert runtime._last_selector_transition is None
    assert runtime._playbook_batch_runner_cache is None
    assert game_state.env is runtime.display_env
    assert game_state.obs is not None

    _, new_probs = runtime._choose_joint_policy_actions(
        player_deterministic=True,
        opponent_deterministic=True,
    )
    first_player = runtime.offense_ids[0]
    assert new_probs[first_player][0] > old_probs[first_player][0]



def test_jax_dev_runtime_replace_policies_refreshes_jax_static_env_from_metadata():
    runtime = _make_runtime(
        env_params={
            "rebound_target_temperature": 0.25,
            "rebound_winner_temperature": 0.25,
        }
    )
    game_state = GameState()
    game_state.jax_runtime = runtime
    game_state.env = runtime.display_env
    game_state.unified_policy = runtime.unified_policy
    game_state.defense_policy = runtime.opponent_policy
    game_state.user_team = Team.OFFENSE
    game_state.obs = runtime.observation_dict()

    new_policy = _FakeRawJaxModel(
        metadata={
            "policy_spec": {"model_type": "mlp"},
            "env_config": {
                "enable_rebounds": False,
                "rebound_target_temperature": 0.75,
                "rebound_winner_temperature": 0.5,
                "offensive_rebound_shot_clock_reset": 13,
            },
        }
    )
    runtime.replace_policies(
        unified_policy=new_policy,
        opponent_policy=new_policy,
        game_state=game_state,
    )

    assert runtime.env_params["enable_rebounds"] is False
    assert runtime.display_env.enable_rebounds is False
    assert float(np.asarray(runtime.static.rebound_target_temperature)) == pytest.approx(0.75)
    assert float(np.asarray(runtime.static.rebound_winner_temperature)) == pytest.approx(0.5)
    assert int(np.asarray(runtime.static.offensive_rebound_shot_clock_reset)) == 13

    runtime.env_params["enable_rebounds"] = True
    runtime.display_env.enable_rebounds = True
    non_rebound_policy = _FakeRawJaxModel(
        metadata={
            "policy_spec": {"model_type": "mlp"},
            "env_config": {},
        }
    )
    runtime.replace_policies(
        unified_policy=non_rebound_policy,
        opponent_policy=non_rebound_policy,
        game_state=game_state,
    )

    assert runtime.env_params["enable_rebounds"] is False
    assert runtime.display_env.enable_rebounds is False
    assert int(np.asarray(runtime.static.enable_rebounds)) == 0


def test_jax_dev_runtime_self_play_respects_requested_template_seed():
    runtime = _make_runtime()
    game_state = GameState()
    game_state.jax_runtime = runtime
    game_state.env = runtime.display_env
    game_state.unified_policy = runtime.unified_policy
    game_state.defense_policy = runtime.opponent_policy
    game_state.user_team = Team.OFFENSE

    body = runtime.start_self_play(
        StartSelfPlayRequest(template_seed=4242),
        game_state,
    )

    assert body["status"] == "success"
    assert body["seed"] == 4242
    assert game_state.replay_seed == 4242


def test_jax_dev_runtime_self_play_without_template_preserves_current_board():
    runtime = _make_runtime()
    game_state = GameState()
    game_state.jax_runtime = runtime
    game_state.env = runtime.display_env
    game_state.unified_policy = runtime.unified_policy
    game_state.defense_policy = runtime.opponent_policy
    game_state.user_team = Team.OFFENSE

    cells = [
        tuple(int(v) for v in cell)
        for cell in sorted(runtime.display_env._move_mask_by_cell.keys())
        if tuple(cell) != tuple(runtime.display_env.basket_position)
    ][: runtime.n_players]
    runtime.apply_resolved_start_template(
        {
            "initial_positions": cells,
            "ball_holder": 1,
            "shot_clock": 18,
        },
        game_state,
    )

    body = runtime.start_self_play(StartSelfPlayRequest(), game_state)

    assert body["status"] == "success"
    assert body["start_template"] is None
    assert runtime.positions == cells
    assert runtime.ball_holder == 1
    assert runtime.shot_clock == 18
    assert game_state.replay_initial_positions == cells
    assert game_state.replay_ball_holder == 1
    assert game_state.replay_shot_clock == 18


def _make_selector_runtime_and_state():
    runtime = _make_runtime(
        env_params={
            "enable_intent_learning": True,
            "num_intents": 8,
            "intent_commitment_steps": 4,
        }
    )
    runtime.raw_model.spec = _FakeSpec(intent_selector_enabled=True, num_intents=8)
    game_state = GameState()
    game_state.jax_runtime = runtime
    game_state.env = runtime.display_env
    game_state.unified_policy = runtime.unified_policy
    game_state.defense_policy = runtime.opponent_policy
    game_state.user_team = Team.OFFENSE
    game_state.self_play_active = True
    game_state.mlflow_training_params = {
        "intent_selector_enabled": True,
        "intent_selector_multiselect_enabled": True,
        "intent_selector_alpha_end": 1.0,
        "intent_selector_eps_end": 0.0,
        "intent_selector_min_play_steps": 3,
    }
    game_state.obs = runtime.observation_dict()
    return runtime, game_state


def _install_jax_runtime_session(monkeypatch, runtime: JaxDevRuntime | None = None):
    runtime = runtime or _make_runtime(
        env_params={
            "enable_intent_learning": True,
            "num_intents": 8,
            "intent_commitment_steps": 4,
        }
    )
    fresh = GameState()
    fresh.jax_runtime = runtime
    fresh.env = runtime.display_env
    fresh.unified_policy = runtime.unified_policy
    fresh.defense_policy = runtime.opponent_policy
    fresh.user_team = Team.OFFENSE
    fresh.obs = runtime.observation_dict()
    fresh.episode_rewards = {"offense": 0.0, "defense": 0.0}
    fresh.mlflow_training_params = {
        "intent_selector_enabled": True,
        "intent_selector_multiselect_enabled": True,
        "intent_selector_alpha_end": 1.0,
        "intent_selector_eps_end": 0.0,
        "intent_selector_min_play_steps": 3,
    }
    monkeypatch.setattr(backend_state, "game_state", fresh)
    monkeypatch.setattr(admin_routes, "game_state", fresh)
    monkeypatch.setattr(lifecycle_routes, "game_state", fresh)
    return runtime, fresh


def test_counterfactual_snapshot_restores_jax_runtime_state(monkeypatch):
    runtime, game_state = _install_jax_runtime_session(monkeypatch)
    runtime.set_offense_intent_state(
        active=True,
        intent_index=2,
        intent_age=1,
        intent_commitment_remaining=3,
        game_state=game_state,
    )
    baseline_positions = list(runtime.positions)
    baseline_ball_holder = runtime.ball_holder
    baseline_shot_clock = runtime.shot_clock
    baseline_intent = int(runtime.display_env.intent_index)

    capture_body = admin_routes.capture_counterfactual_snapshot_route()

    assert capture_body["status"] == "success"
    assert capture_body["snapshot"]["available"] is True
    assert capture_body["state"]["counterfactual_snapshot_available"] is True
    assert capture_body["state"]["counterfactual_snapshot_shot_clock"] == baseline_shot_clock
    assert capture_body["state"]["counterfactual_snapshot_ball_holder"] == baseline_ball_holder
    assert capture_body["state"]["counterfactual_snapshot_intent_index"] == baseline_intent

    runtime.set_offense_intent_state(
        active=True,
        intent_index=6,
        intent_age=3,
        intent_commitment_remaining=1,
        game_state=game_state,
    )
    runtime.state = runtime.state._replace(
        positions=jnp.asarray([[(9, 9)] * runtime.n_players], dtype=jnp.int32),
        ball_holder=jnp.asarray([1], dtype=jnp.int32),
        shot_clock=jnp.asarray([3], dtype=jnp.int32),
    )
    runtime._sync_display_env()
    game_state.env = runtime.display_env
    game_state.obs = runtime.observation_dict()

    backend_state.restore_counterfactual_snapshot()

    assert game_state.env is runtime.display_env
    assert runtime.positions == baseline_positions
    assert runtime.ball_holder == baseline_ball_holder
    assert runtime.shot_clock == baseline_shot_clock
    assert int(runtime.display_env.intent_index) == baseline_intent
    assert game_state.obs is not None


def test_replay_counterfactual_snapshot_route_uses_jax_runtime_snapshot(monkeypatch):
    runtime, _game_state = _install_jax_runtime_session(monkeypatch)
    backend_state.capture_counterfactual_snapshot()

    first = admin_routes.replay_counterfactual_snapshot_route(
        ReplayCounterfactualRequest(
            player_deterministic=True,
            opponent_deterministic=True,
            max_steps=3,
        )
    )
    backend_state.restore_counterfactual_snapshot()
    second = admin_routes.replay_counterfactual_snapshot_route(
        ReplayCounterfactualRequest(
            player_deterministic=True,
            opponent_deterministic=True,
            max_steps=3,
        )
    )

    assert first["status"] == "success"
    assert second["status"] == "success"
    assert first["steps_taken"] == second["steps_taken"]
    assert first["state"]["positions"] == second["state"]["positions"]
    assert first["state"]["shot_clock"] == second["state"]["shot_clock"]
    assert runtime.last_step_output is not None


def test_playbook_analysis_route_forces_jax_runtime_intent(monkeypatch):
    runtime, game_state = _install_jax_runtime_session(monkeypatch)
    backend_state.capture_counterfactual_snapshot()
    calls: list[int] = []
    original = runtime.set_offense_intent_state

    def spy_set_offense_intent_state(**kwargs):
        calls.append(int(kwargs["intent_index"]))
        return original(**kwargs)

    monkeypatch.setattr(runtime, "set_offense_intent_state", spy_set_offense_intent_state)

    body = admin_routes.playbook_analysis_route(
        PlaybookAnalysisRequest(
            intent_indices=[1],
            num_rollouts=1,
            max_steps=1,
            use_snapshot=True,
            player_deterministic=True,
            opponent_deterministic=True,
        )
    )

    assert body["status"] == "success"
    assert body["used_parallel"] is False
    assert calls == [1]
    assert body["panels"][0]["intent_index"] == 1
    assert game_state.env is runtime.display_env


def test_jax_dev_runtime_self_play_reselects_after_commitment_timeout(monkeypatch):
    runtime, game_state = _make_selector_runtime_and_state()
    runtime.set_offense_intent_state(
        active=True,
        intent_index=2,
        intent_age=4,
        intent_commitment_remaining=0,
        game_state=game_state,
    )
    monkeypatch.setattr(
        runtime,
        "_sample_selector_intent",
        lambda _game_state: {
            "intent_index": 5,
            "used_selector": True,
            "alpha": 1.0,
            "eps": 0.0,
            "value": 0.25,
        },
    )
    monkeypatch.setattr(runtime, "_selector_preferences", lambda _game_state: None)

    body = runtime.step(
        ActionRequest(actions={}, player_deterministic=True, opponent_deterministic=True),
        game_state,
    )

    assert body["status"] == "success"
    assert body["selector_transition"]["reason"] == "commitment_timeout"
    assert body["selector_transition"]["intent_index"] == 5
    assert body["state"]["intent_index_current"] == 5
    assert body["state"]["intent_age"] == 1
    assert body["state"]["intent_commitment_remaining"] == 3
    assert body["state"]["selector_last_boundary_reason"] == "commitment_timeout"
    assert game_state.selector_segment_index == 1


def test_jax_dev_runtime_reselection_can_sample_same_intent(monkeypatch):
    runtime, game_state = _make_selector_runtime_and_state()
    runtime.set_offense_intent_state(
        active=True,
        intent_index=2,
        intent_age=4,
        intent_commitment_remaining=0,
        game_state=game_state,
    )
    monkeypatch.setattr(
        runtime,
        "_sample_selector_intent",
        lambda _game_state: {
            "intent_index": 2,
            "used_selector": True,
            "alpha": 1.0,
            "eps": 0.0,
            "value": 0.25,
        },
    )
    monkeypatch.setattr(runtime, "_selector_preferences", lambda _game_state: None)

    body = runtime.step(
        ActionRequest(actions={}, player_deterministic=True, opponent_deterministic=True),
        game_state,
    )

    assert body["status"] == "success"
    assert body["selector_transition"]["reason"] == "commitment_timeout"
    assert body["selector_transition"]["previous_intent_index"] == 2
    assert body["selector_transition"]["intent_index"] == 2
    assert body["selector_transition"]["changed_intent"] is False
    assert body["state"]["intent_index_current"] == 2
    assert body["state"]["selector_last_boundary_reason"] == "commitment_timeout"
    assert game_state.selector_segment_index == 1


def test_jax_dev_runtime_forces_learned_selector_for_interactive_sampling(monkeypatch):
    runtime, game_state = _make_selector_runtime_and_state()
    game_state.mlflow_training_params.update(
        {
            "intent_selector_alpha_start": 0.0,
            "intent_selector_alpha_end": 0.0,
            "intent_selector_eps_end": 0.0,
        }
    )
    monkeypatch.setattr(
        runtime,
        "_selector_distribution",
        lambda _game_state: {
            "mixed_probs_device": jnp.asarray([0.0, 1.0, 0.0, 0.0], dtype=jnp.float32),
            "alpha": runtime._selector_alpha_eps(_game_state)[0],
            "eps": runtime._selector_alpha_eps(_game_state)[1],
            "value": 0.0,
            "num_intents": 4,
        },
    )

    selection = runtime._sample_selector_intent(game_state)
    debug = runtime._selector_debug_payload(game_state)

    assert selection["used_selector"] is True
    assert selection["intent_index"] == 1
    assert selection["alpha"] == pytest.approx(1.0)
    assert debug["alpha_current"] == pytest.approx(1.0)
    assert debug["training_alpha_current"] == pytest.approx(0.0)
    assert debug["force_learned_runtime"] is True


def test_jax_dev_runtime_selector_multiselect_prefers_checkpoint_metadata():
    runtime = _make_runtime(
        env_params={
            "enable_intent_learning": True,
            "num_intents": 8,
            "intent_commitment_steps": 4,
        }
    )
    runtime.raw_model.spec = _FakeSpec(intent_selector_enabled=True, num_intents=8)
    runtime.raw_model.metadata = {
        "policy_spec": {
            "model_type": "mlp",
            "intent_selector_enabled": True,
        },
        "trainer_config": {
            "intent_selector_enabled": True,
            "intent_selector_multiselect_enabled": True,
            "intent_selector_min_play_steps": 4,
            "intent_selector_alpha_end": 1.0,
            "intent_selector_eps_end": 0.0,
        },
    }
    game_state = GameState()
    game_state.jax_runtime = runtime
    game_state.env = runtime.display_env
    game_state.unified_policy = runtime.unified_policy
    game_state.user_team = Team.OFFENSE
    game_state.mlflow_training_params = {
        "intent_selector_enabled": True,
        # Simulates a default produced by training-param extraction when the
        # checkpoint metadata carries the real static selector setting.
        "intent_selector_multiselect_enabled": False,
        "intent_selector_min_play_steps": 3,
    }

    debug = runtime._selector_debug_payload(game_state)

    assert debug["runtime_enabled"] is True
    assert debug["multiselect_enabled"] is True
    assert debug["min_play_steps"] == 4


def test_jax_dev_runtime_action_results_preserve_assisted_shot_metadata():
    runtime = _make_runtime()
    prev_state = runtime.state
    out = SimpleNamespace(
        state=prev_state,
        shot_attempt=np.asarray([1], dtype=np.int8),
        shot_shooter=np.asarray([1], dtype=np.int32),
        shot_value=np.asarray([2.0], dtype=np.float32),
        shot_expected_points=np.asarray([1.4], dtype=np.float32),
        shot_distance=np.asarray([0.0], dtype=np.float32),
        shot_success=np.asarray([1], dtype=np.int8),
        shot_type=np.asarray([SHOT_TYPE_DUNK], dtype=np.int32),
        assist=np.asarray([1], dtype=np.int8),
        potential_assist=np.asarray([1], dtype=np.int8),
        assist_passer=np.asarray([0], dtype=np.int32),
        pass_attempt=np.asarray([0], dtype=np.int8),
        turnover=np.asarray([0], dtype=np.int8),
        offensive_three_seconds=np.asarray([0], dtype=np.int8),
        defensive_lane_violation=np.asarray([0], dtype=np.int8),
    )

    results = runtime._action_results_from_step(prev_state, out)

    shot = results["shots"]["1"]
    assert shot["success"] is True
    assert shot["distance"] == 0
    assert shot["assist_full"] is True
    assert shot["assist_potential"] is True
    assert shot["assist_passer_id"] == 0


def test_jax_dev_runtime_turn_step_does_not_reselect_at_episode_start(monkeypatch):
    runtime, game_state = _make_selector_runtime_and_state()
    game_state.self_play_active = False
    runtime.set_offense_intent_state(
        active=True,
        intent_index=2,
        intent_age=0,
        intent_commitment_remaining=4,
        game_state=game_state,
    )
    monkeypatch.setattr(
        runtime,
        "_sample_selector_intent",
        lambda _game_state: pytest.fail("episode start should not sample during step"),
    )
    monkeypatch.setattr(runtime, "_selector_preferences", lambda _game_state: None)

    body = runtime.step(
        ActionRequest(actions={}, player_deterministic=True, opponent_deterministic=True),
        game_state,
    )

    assert body["status"] == "success"
    assert body["selector_transition"] is None
    assert body["state"]["intent_index_current"] == 2
    assert body["state"]["intent_age"] == 1
    assert body["state"]["intent_commitment_remaining"] == 3
    assert body["state"]["selector_last_boundary_reason"] is None
    assert game_state.selector_segment_index == 0


def test_jax_dev_runtime_turn_step_reselects_after_commitment_timeout(monkeypatch):
    runtime, game_state = _make_selector_runtime_and_state()
    game_state.self_play_active = False
    runtime.set_offense_intent_state(
        active=True,
        intent_index=2,
        intent_age=4,
        intent_commitment_remaining=0,
        game_state=game_state,
    )
    monkeypatch.setattr(
        runtime,
        "_sample_selector_intent",
        lambda _game_state: {
            "intent_index": 4,
            "used_selector": True,
            "alpha": 1.0,
            "eps": 0.0,
            "value": 0.5,
        },
    )
    monkeypatch.setattr(runtime, "_selector_preferences", lambda _game_state: None)

    body = runtime.step(
        ActionRequest(actions={}, player_deterministic=True, opponent_deterministic=True),
        game_state,
    )

    assert body["status"] == "success"
    assert body["selector_transition"]["reason"] == "commitment_timeout"
    assert body["selector_transition"]["intent_index"] == 4
    assert body["state"]["intent_index_current"] == 4
    assert body["state"]["selector_last_boundary_reason"] == "commitment_timeout"


def test_jax_dev_runtime_reselects_after_natural_commitment_expiry(monkeypatch):
    runtime, game_state = _make_selector_runtime_and_state()
    runtime.set_offense_intent_state(
        active=True,
        intent_index=2,
        intent_age=3,
        intent_commitment_remaining=1,
        game_state=game_state,
    )
    selections = iter(
        [
            {
                "intent_index": 4,
                "used_selector": True,
                "alpha": 1.0,
                "eps": 0.0,
                "value": 0.5,
            }
        ]
    )
    monkeypatch.setattr(runtime, "_sample_selector_intent", lambda _game_state: next(selections))
    monkeypatch.setattr(runtime, "_selector_preferences", lambda _game_state: None)

    first = runtime.step(
        ActionRequest(actions={}, player_deterministic=True, opponent_deterministic=True),
        game_state,
    )
    second = runtime.step(
        ActionRequest(actions={}, player_deterministic=True, opponent_deterministic=True),
        game_state,
    )

    assert first["status"] == "success"
    assert first["selector_transition"] is None
    assert first["state"]["intent_active_current"] is True
    assert first["state"]["intent_commitment_remaining"] == 0
    assert second["status"] == "success"
    assert second["selector_transition"]["reason"] == "commitment_timeout"
    assert second["selector_transition"]["intent_index"] == 4
    assert second["state"]["intent_index_current"] == 4
    assert second["state"]["selector_last_boundary_reason"] == "commitment_timeout"


def test_jax_dev_runtime_self_play_ignores_completed_pass_before_min_play(monkeypatch):
    runtime, game_state = _make_selector_runtime_and_state()
    runtime.set_offense_intent_state(
        active=True,
        intent_index=2,
        intent_age=2,
        intent_commitment_remaining=2,
        game_state=game_state,
    )
    runtime._last_completed_pass_boundary = True

    def _fail_sample(_game_state):
        raise AssertionError("completed pass before min play should not sample selector")

    monkeypatch.setattr(runtime, "_sample_selector_intent", _fail_sample)
    monkeypatch.setattr(runtime, "_selector_preferences", lambda _game_state: None)

    body = runtime.step(
        ActionRequest(actions={}, player_deterministic=True, opponent_deterministic=True),
        game_state,
    )

    assert body["status"] == "success"
    assert body["selector_transition"] is None
    assert body["state"]["intent_index_current"] == 2
    debug = body["state"]["selector_debug"]
    assert debug["min_play_steps"] == 3
    assert debug["completed_pass_min_steps_met"] is False


def test_jax_dev_runtime_self_play_reselects_after_completed_pass_boundary(monkeypatch):
    runtime, game_state = _make_selector_runtime_and_state()
    runtime.set_offense_intent_state(
        active=True,
        intent_index=2,
        intent_age=3,
        intent_commitment_remaining=2,
        game_state=game_state,
    )
    runtime._last_completed_pass_boundary = True
    monkeypatch.setattr(
        runtime,
        "_sample_selector_intent",
        lambda _game_state: {
            "intent_index": 6,
            "used_selector": True,
            "alpha": 1.0,
            "eps": 0.0,
            "value": 0.5,
        },
    )
    monkeypatch.setattr(runtime, "_selector_preferences", lambda _game_state: None)

    body = runtime.step(
        ActionRequest(actions={}, player_deterministic=True, opponent_deterministic=True),
        game_state,
    )

    assert body["status"] == "success"
    assert body["selector_transition"]["reason"] == "completed_pass"
    assert body["selector_transition"]["intent_index"] == 6
    assert body["state"]["intent_index_current"] == 6
    assert body["state"]["selector_last_boundary_reason"] == "completed_pass"


def test_jax_dev_runtime_unseeded_resets_advance_rng_key():
    runtime = _make_runtime(rng_seed=0, reset_seed=None)

    key0 = np.asarray(jax.device_get(runtime._rng_key))
    runtime.reset()
    key1 = np.asarray(jax.device_get(runtime._rng_key))
    runtime.reset()
    key2 = np.asarray(jax.device_get(runtime._rng_key))

    assert not np.array_equal(key0, key1)
    assert not np.array_equal(key1, key2)


def test_jax_dev_runtime_apply_resolved_template_updates_kernel_state():
    runtime = _make_runtime()
    game_state = GameState()
    game_state.jax_runtime = runtime
    game_state.env = runtime.display_env
    game_state.unified_policy = runtime.unified_policy
    game_state.defense_policy = runtime.opponent_policy
    game_state.user_team = Team.OFFENSE

    cells = [
        tuple(int(v) for v in cell)
        for cell in sorted(runtime.display_env._move_mask_by_cell.keys())
        if tuple(cell) != tuple(runtime.display_env.basket_position)
    ][: runtime.n_players]
    resolved = {
        "initial_positions": cells,
        "ball_holder": 1,
        "shot_clock": 19,
    }

    runtime.apply_resolved_start_template(resolved, game_state)

    assert runtime.positions == cells
    assert runtime.ball_holder == 1
    assert runtime.shot_clock == 19
    assert game_state.turn_start_positions == cells
    assert game_state.turn_start_ball_holder == 1
    assert game_state.turn_start_shot_clock == 19
    assert isinstance(game_state.obs, dict)


def test_jax_dev_runtime_apply_display_env_edits_updates_kernel_state():
    runtime = _make_runtime()
    game_state = GameState()
    game_state.jax_runtime = runtime
    game_state.env = runtime.display_env
    game_state.unified_policy = runtime.unified_policy
    game_state.defense_policy = runtime.opponent_policy
    game_state.user_team = Team.OFFENSE

    cells = [
        tuple(int(v) for v in cell)
        for cell in sorted(runtime.display_env._move_mask_by_cell.keys())
        if tuple(cell) != tuple(runtime.display_env.basket_position)
    ][: runtime.n_players]
    runtime.display_env.positions = cells
    runtime.display_env.ball_holder = 1
    runtime.display_env.shot_clock = 17
    runtime.display_env.offense_layup_pct_by_player = [0.11, 0.22, 0.33]
    runtime.display_env.offense_three_pt_pct_by_player = [0.44, 0.55, 0.66]
    runtime.display_env.offense_dunk_pct_by_player = [0.77, 0.88, 0.99]

    runtime.apply_display_env_edits(game_state)

    assert runtime.positions == cells
    assert runtime.ball_holder == 1
    assert runtime.shot_clock == 17
    np.testing.assert_allclose(
        np.asarray(runtime.jax.device_get(runtime.state.layup_pct))[0, runtime.offense_ids],
        [0.11, 0.22, 0.33],
    )
    np.testing.assert_allclose(
        np.asarray(runtime.jax.device_get(runtime.state.three_pt_pct))[0, runtime.offense_ids],
        [0.44, 0.55, 0.66],
    )
    np.testing.assert_allclose(
        np.asarray(runtime.jax.device_get(runtime.state.dunk_pct))[0, runtime.offense_ids],
        [0.77, 0.88, 0.99],
    )
    assert isinstance(game_state.obs, dict)


def test_apply_start_template_route_updates_jax_runtime(monkeypatch):
    class DummyRuntime:
        def __init__(self):
            self.resolved = None

        def apply_resolved_start_template(self, resolved, game_state):
            self.resolved = dict(resolved)
            game_state.obs = {"action_mask": []}

    fresh = GameState()
    runtime = DummyRuntime()
    fresh.jax_runtime = runtime
    fresh.env = _ExplodingPythonEnv()
    fresh.obs = {"action_mask": []}
    monkeypatch.setattr(backend_state, "game_state", fresh)
    monkeypatch.setattr(admin_routes, "game_state", fresh)
    monkeypatch.setattr(
        admin_routes,
        "_resolve_start_template_request",
        lambda req: (
            {
                "template_id": "wing_entry",
                "initial_positions": [(0, 0), (1, 0)],
                "ball_holder": 0,
                "shot_clock": 24,
            },
            False,
        ),
    )
    monkeypatch.setattr(
        admin_routes,
        "get_ui_game_state",
        lambda: {"positions": [(0, 0), (1, 0)], "ball_holder": 0},
    )

    body = admin_routes.apply_start_template(
        ApplyStartTemplateRequest(template_id="wing_entry", apply_to_state=True)
    )

    assert body["status"] == "success"
    assert runtime.resolved["template_id"] == "wing_entry"
    assert body["state"]["ball_holder"] == 0


def test_update_player_position_route_syncs_jax_runtime(monkeypatch):
    class DummyEnv:
        n_players = 2
        positions = [(0, 0), (1, 0)]
        episode_ended = False

        def _is_valid_position(self, q, r):
            return True

    class DummyRuntime:
        def __init__(self):
            self.synced = False

        def apply_display_env_edits(self, game_state):
            self.synced = True
            game_state.obs = {"action_mask": []}

    fresh = GameState()
    runtime = DummyRuntime()
    env = DummyEnv()
    fresh.jax_runtime = runtime
    fresh.env = env
    fresh.obs = {"action_mask": []}
    monkeypatch.setattr(backend_state, "game_state", fresh)
    monkeypatch.setattr(admin_routes, "game_state", fresh)
    monkeypatch.setattr(
        admin_routes,
        "get_ui_game_state",
        lambda: {"positions": [tuple(pos) for pos in fresh.env.positions]},
    )

    body = admin_routes.update_player_position(
        UpdatePositionRequest(player_id=0, q=2, r=0)
    )

    assert body["status"] == "success"
    assert env.positions[0] == (2, 0)
    assert runtime.synced is True


def test_set_offense_skills_route_syncs_jax_runtime(monkeypatch):
    class DummyEnv:
        players_per_side = 3
        offense_layup_pct_by_player = [0.5, 0.5, 0.5]
        offense_three_pt_pct_by_player = [0.35, 0.35, 0.35]
        offense_dunk_pct_by_player = [0.8, 0.8, 0.8]

    class DummyRuntime:
        def __init__(self):
            self.synced = False

        def apply_display_env_edits(self, game_state):
            self.synced = True
            game_state.obs = {"action_mask": []}

    fresh = GameState()
    runtime = DummyRuntime()
    env = DummyEnv()
    fresh.jax_runtime = runtime
    fresh.env = env
    fresh.obs = {"action_mask": []}
    fresh.sampled_offense_skills = {
        "layup": [0.5, 0.5, 0.5],
        "three_pt": [0.35, 0.35, 0.35],
        "dunk": [0.8, 0.8, 0.8],
    }
    monkeypatch.setattr(backend_state, "game_state", fresh)
    monkeypatch.setattr(admin_routes, "game_state", fresh)
    monkeypatch.setattr(
        admin_routes,
        "get_ui_game_state",
        lambda: {
            "offense_shooting_pct_by_player": {
                "layup": list(fresh.env.offense_layup_pct_by_player),
                "three_pt": list(fresh.env.offense_three_pt_pct_by_player),
                "dunk": list(fresh.env.offense_dunk_pct_by_player),
            }
        },
    )

    body = admin_routes.set_offense_skills(
        SetOffenseSkillsRequest(
            skills={
                "layup": [0.61, 0.62, 0.63],
                "three_pt": [0.41, 0.42, 0.43],
                "dunk": [0.81, 0.82, 0.83],
            }
        )
    )

    assert body["status"] == "success"
    assert runtime.synced is True
    assert env.offense_layup_pct_by_player == [0.61, 0.62, 0.63]
    assert env.offense_three_pt_pct_by_player == [0.41, 0.42, 0.43]
    assert env.offense_dunk_pct_by_player == [0.81, 0.82, 0.83]


def test_jax_dev_runtime_set_offense_intent_state_updates_kernel_state():
    runtime = _make_runtime()
    game_state = GameState()
    game_state.jax_runtime = runtime
    game_state.env = runtime.display_env
    game_state.unified_policy = runtime.unified_policy
    game_state.defense_policy = runtime.opponent_policy
    game_state.user_team = Team.OFFENSE

    runtime.set_offense_intent_state(
        active=True,
        intent_index=5,
        intent_age=1,
        intent_commitment_remaining=3,
        game_state=game_state,
    )
    state = runtime.get_full_game_state(game_state, include_policy_probs=False)

    assert state["intent_active_current"] is True
    assert state["intent_index_current"] == 5
    assert state["intent_age"] == 1
    assert state["intent_commitment_remaining"] == 3
    assert runtime.display_env.intent_index == 5


def test_set_intent_state_route_updates_jax_runtime(monkeypatch):
    class DummyRuntime:
        def __init__(self):
            self.payload = None

        def set_offense_intent_state(self, **kwargs):
            self.payload = dict(kwargs)
            kwargs["game_state"].obs = {"action_mask": []}

    fresh = GameState()
    runtime = DummyRuntime()
    fresh.jax_runtime = runtime
    fresh.env = type(
        "IntentEnv",
        (),
        {
            "episode_ended": False,
            "enable_intent_learning": True,
            "num_intents": 8,
            "intent_commitment_steps": 4,
        },
    )()
    fresh.obs = {"action_mask": []}
    monkeypatch.setattr(backend_state, "game_state", fresh)
    monkeypatch.setattr(admin_routes, "game_state", fresh)
    monkeypatch.setattr(
        admin_routes,
        "get_ui_game_state",
        lambda: {
            "intent_active_current": True,
            "intent_index_current": 6,
        },
    )

    body = admin_routes.set_intent_state(
        SetIntentStateRequest(active=True, intent_index=6, intent_age=1)
    )

    assert body["status"] == "success"
    assert runtime.payload["active"] is True
    assert runtime.payload["intent_index"] == 6
    assert runtime.payload["intent_age"] == 1
    assert runtime.payload["intent_commitment_remaining"] == 3
    assert body["state"]["intent_index_current"] == 6
