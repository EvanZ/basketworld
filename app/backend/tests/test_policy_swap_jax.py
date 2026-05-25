from types import SimpleNamespace

import app.backend.state as backend_state
from app.backend.routes import admin_routes
from app.backend.schemas import SwapPoliciesRequest


class _DummyJaxPolicy:
    backend_kind = "jax"
    capabilities = {"player_controls": True}

    def __init__(self):
        self.pass_mode = None

    def set_pass_mode(self, mode_value: str) -> None:
        self.pass_mode = str(mode_value)


class _DummyJaxRuntime:
    def __init__(self):
        self.replace_calls = []

    def replace_policies(self, *, unified_policy, opponent_policy, game_state):
        self.replace_calls.append(
            {
                "unified_policy": unified_policy,
                "opponent_policy": opponent_policy,
                "game_state": game_state,
            }
        )
        game_state.obs = {"refreshed": True}


def test_swap_policies_uses_inference_loader_for_jax_checkpoint(monkeypatch):
    fresh = backend_state.GameState()
    fresh.env = SimpleNamespace(pass_mode="pointer_targeted")
    fresh.obs = {"action_mask": []}
    fresh.run_id = "run-1"
    fresh.unified_policy_key = "update_0000000"
    fresh.unified_policy = _DummyJaxPolicy()
    fresh.defense_policy = None
    fresh.jax_runtime = _DummyJaxRuntime()

    loaded_policy = _DummyJaxPolicy()
    load_calls = []

    monkeypatch.setattr(backend_state, "game_state", fresh)
    monkeypatch.setattr(admin_routes, "game_state", fresh)
    monkeypatch.setattr(admin_routes.mlflow.tracking, "MlflowClient", lambda: object())
    monkeypatch.setattr(
        admin_routes,
        "get_unified_policy_path",
        lambda client, run_id, policy_name=None: f"/tmp/{policy_name}",
    )

    def fake_load_inference_policy(path, **kwargs):
        load_calls.append((path, kwargs))
        return loaded_policy

    monkeypatch.setattr(admin_routes, "load_inference_policy", fake_load_inference_policy)
    monkeypatch.setattr(admin_routes, "_compute_param_counts_from_policy", lambda policy: None)
    monkeypatch.setattr(
        admin_routes,
        "get_ui_game_state",
        lambda: {
            "unified_policy_name": fresh.unified_policy_key,
            "model_backend": fresh.unified_policy_backend,
        },
    )

    body = admin_routes.swap_policies(
        SwapPoliciesRequest(user_policy_name="update_0000100")
    )

    assert body["status"] == "success"
    assert fresh.unified_policy is loaded_policy
    assert fresh.unified_policy_key == "update_0000100"
    assert fresh.unified_policy_path == "/tmp/update_0000100"
    assert fresh.unified_policy_backend == "jax"
    assert loaded_policy.pass_mode == "pointer_targeted"
    assert fresh.obs == {"refreshed": True}
    assert fresh.jax_runtime.replace_calls == [
        {
            "unified_policy": loaded_policy,
            "opponent_policy": None,
            "game_state": fresh,
        }
    ]
    assert len(load_calls) == 1
    assert load_calls[0][0] == "/tmp/update_0000100"
    assert load_calls[0][1]["device"] == "cpu"
    assert "custom_objects" in load_calls[0][1]


def test_swap_policies_no_change_still_syncs_jax_runtime(monkeypatch):
    fresh = backend_state.GameState()
    fresh.env = SimpleNamespace(pass_mode="pointer_targeted")
    fresh.obs = {"action_mask": []}
    fresh.run_id = "run-1"
    fresh.unified_policy_key = "update_0000100"
    fresh.unified_policy = _DummyJaxPolicy()
    fresh.defense_policy = None
    fresh.jax_runtime = _DummyJaxRuntime()

    monkeypatch.setattr(backend_state, "game_state", fresh)
    monkeypatch.setattr(admin_routes, "game_state", fresh)
    monkeypatch.setattr(admin_routes.mlflow.tracking, "MlflowClient", lambda: object())
    monkeypatch.setattr(
        admin_routes,
        "get_ui_game_state",
        lambda: {
            "unified_policy_name": fresh.unified_policy_key,
            "model_backend": fresh.unified_policy_backend,
        },
    )

    body = admin_routes.swap_policies(
        SwapPoliciesRequest(user_policy_name="update_0000100")
    )

    assert body["status"] == "no_change"
    assert fresh.obs == {"refreshed": True}
    assert fresh.jax_runtime.replace_calls == [
        {
            "unified_policy": fresh.unified_policy,
            "opponent_policy": None,
            "game_state": fresh,
        }
    ]
