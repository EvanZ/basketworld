from __future__ import annotations

from pathlib import Path

import app.backend.inference_adapters as adapters
from app.backend.inference_adapters import (
    InferencePolicyAdapter,
    SB3PPOInferenceAdapter,
    get_policy_backend_kind,
    get_policy_capabilities,
    get_policy_metadata,
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
