from __future__ import annotations

from pathlib import Path

import app.backend.inference_adapters as adapters
from app.backend.inference_adapters import (
    InferencePolicyAdapter,
    JAXInferenceAdapter,
    SB3PPOInferenceAdapter,
    get_policy_backend_kind,
    get_policy_capabilities,
    get_policy_metadata,
    policy_observation_tokens,
    policy_attention_payload,
    policy_action_probabilities,
    prepare_policy_for_role,
    unwrap_inference_model,
    unwrap_policy_module,
)


class _FakePolicyModule:
    def __init__(self):
        self.pass_mode = None

    def set_pass_mode(self, mode_value: str) -> None:
        self.pass_mode = str(mode_value)


class _FakeRawModel:
    def __init__(self):
        self.policy = _FakePolicyModule()
        self.predict_calls = []

    def predict(self, obs, deterministic: bool = False):
        self.predict_calls.append((obs, bool(deterministic)))
        return [1, 2, 3], None


class _FakeJAXRawModel:
    def __init__(self, model_type: str, policy_spec: dict | None = None):
        self.metadata = {
            "policy_spec": {
                "model_type": model_type,
                **dict(policy_spec or {}),
            }
        }
        self.attention_payload_calls = []

    def predict(self, obs, deterministic: bool = False):
        return [0], None

    def action_probabilities(self, obs):
        return [[1.0]]

    def prepare_for_role(self, env, *, observer_is_offense: bool) -> None:
        return None

    def observation_vector(self, env, *, observer_is_offense: bool):
        return [0.0]

    def observation_tokens(self, env, *, observer_is_offense: bool):
        return {"globals": [0.5], "globals_labels": ["shot_clock_norm"]}

    def attention_payload(self, env, *, observer_is_offense: bool):
        self.attention_payload_calls.append((env, bool(observer_is_offense)))
        return {"weights_avg": [[1.0]], "weights_heads": [[[1.0]]], "labels": ["T0"], "heads": 1}

    def state_value(self, env, *, observer_is_offense: bool):
        return 0.75 if observer_is_offense else -0.25


class _FakeCustomAdapter(InferencePolicyAdapter):
    def __init__(self):
        super().__init__(raw_model=object(), backend_kind="jax")
        self.prepare_calls = []
        self.probability_calls = []

    def predict(self, obs, deterministic: bool = False):
        return [0], None

    def action_probabilities(self, obs):
        self.probability_calls.append(obs)
        return [[0.25, 0.75]]

    def prepare_for_role(self, env, *, observer_is_offense: bool) -> None:
        self.prepare_calls.append((env, bool(observer_is_offense)))


def test_sb3_adapter_delegates_predict_and_pass_mode():
    raw_model = _FakeRawModel()
    adapter = SB3PPOInferenceAdapter(raw_model)

    actions, _ = adapter.predict({"obs": [0]}, deterministic=True)
    adapter.set_pass_mode("directional")

    assert actions == [1, 2, 3]
    assert raw_model.predict_calls == [({"obs": [0]}, True)]
    assert raw_model.policy.pass_mode == "directional"


def test_adapter_unwrap_and_capabilities_helpers():
    raw_model = _FakeRawModel()
    adapter = SB3PPOInferenceAdapter(raw_model)

    assert unwrap_inference_model(adapter) is raw_model
    assert unwrap_policy_module(adapter) is raw_model.policy
    assert get_policy_backend_kind(adapter) == "sb3"

    capabilities = get_policy_capabilities(adapter)
    assert capabilities is not None
    assert capabilities["player_controls"] is True
    assert capabilities["attention"] is True


def test_jax_adapter_enables_attention_only_for_attention_checkpoints():
    env = object()
    attention_raw = _FakeJAXRawModel("attention")
    attention_adapter = JAXInferenceAdapter(attention_raw)
    mlp_adapter = JAXInferenceAdapter(_FakeJAXRawModel("mlp"))

    assert get_policy_capabilities(attention_adapter)["attention"] is True
    assert get_policy_capabilities(mlp_adapter)["attention"] is False
    assert get_policy_capabilities(attention_adapter)["state_values"] is True
    assert attention_adapter.state_value(env, observer_is_offense=True) == 0.75
    assert attention_adapter.state_value(env, observer_is_offense=False) == -0.25
    assert policy_attention_payload(
        attention_adapter,
        env,
        observer_is_offense=False,
    ) == {"weights_avg": [[1.0]], "weights_heads": [[[1.0]]], "labels": ["T0"], "heads": 1}
    assert policy_observation_tokens(
        attention_adapter,
        env,
        observer_is_offense=False,
    ) == {"globals": [0.5], "globals_labels": ["shot_clock_norm"]}
    assert attention_raw.attention_payload_calls == [(env, False)]
    assert policy_attention_payload(
        mlp_adapter,
        env,
        observer_is_offense=True,
    ) is None


def test_jax_adapter_exposes_play_capabilities_from_policy_spec():
    raw = _FakeJAXRawModel(
        "attention",
        {
            "num_intents": 4,
            "intent_embedding_enabled": True,
            "intent_selector_enabled": True,
        },
    )
    raw.metadata["env_config"] = {"enable_intent_learning": True, "num_intents": 4}
    raw.metadata["play_name_map"] = {"0": "cut", "1": "flare"}

    capabilities = get_policy_capabilities(JAXInferenceAdapter(raw))

    assert capabilities["play_metadata"] is True
    assert capabilities["selector_distribution"] is True
    assert capabilities["per_intent_eval"] is True
    assert capabilities["play_shot_charts"] is True
    assert capabilities["manual_intent_override"] is True
    assert capabilities["playbook"] is True


def test_helper_dispatch_uses_adapter_surface():
    adapter = _FakeCustomAdapter()
    env = object()
    obs = {"obs": [1]}

    prepare_policy_for_role(adapter, env, observer_is_offense=True)
    probs = policy_action_probabilities(adapter, obs)

    assert adapter.prepare_calls == [(env, True)]
    assert adapter.probability_calls == [obs]
    assert probs == [[0.25, 0.75]]


def test_generic_loader_detects_jax_checkpoint(tmp_path, monkeypatch):
    checkpoint_dir = Path(tmp_path) / "latest"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "metadata.json").write_text("{}", encoding="utf-8")
    (checkpoint_dir / "state").mkdir()

    sentinel = object()
    monkeypatch.setattr(adapters, "load_jax_adapter", lambda path: sentinel)

    loaded = adapters.load_inference_policy(str(checkpoint_dir))
    assert loaded is sentinel


def test_get_policy_metadata_returns_adapter_metadata():
    adapter = _FakeCustomAdapter()
    adapter.metadata = {"checkpoint_path": "/tmp/model", "backend": "jax"}

    metadata = get_policy_metadata(adapter)
    assert metadata == {"checkpoint_path": "/tmp/model", "backend": "jax"}
