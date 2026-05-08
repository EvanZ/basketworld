from __future__ import annotations

from app.backend.observations import _compute_state_values_from_obs
from app.backend.state import game_state


class _DummyJaxValuePolicy:
    backend_kind = "jax"
    capabilities = {"state_values": True, "q_values": False}

    def __init__(self):
        self.calls = []

    def state_value(self, env, *, observer_is_offense: bool):
        self.calls.append((env, bool(observer_is_offense)))
        return 1.25 if observer_is_offense else -0.5


def test_compute_state_values_uses_jax_adapter_value_head(monkeypatch):
    env = object()
    policy = _DummyJaxValuePolicy()
    monkeypatch.setattr(game_state, "env", env)
    monkeypatch.setattr(game_state, "unified_policy", policy)

    values = _compute_state_values_from_obs({"obs": []})

    assert values == {"offensive_value": 1.25, "defensive_value": -0.5}
    assert policy.calls == [(env, True), (env, False)]
