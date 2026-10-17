import numpy as np
import pytest

from app.backend.routes import lifecycle_routes
from app.backend.schemas import ActionRequest
from app.backend.state import game_state


class DummyJaxRuntime:
    def __init__(self):
        self.requests = []

    def step(self, request, state):
        self.requests.append((request, state))
        first_action = request.actions.get("0")
        metadata = first_action if isinstance(first_action, dict) else {"action": first_action}
        return {
            "status": "success",
            "state": {"positions": np.asarray([[1, 2], [3, 4]])},
            "actions_taken": {"0": "PASS_0" if metadata.get("type") == "PASS" else "NOOP"},
            "actions_taken_meta": {"0": metadata},
            "step_rewards": {"offense": np.float32(0.1), "defense": np.float32(-0.1)},
            "episode_rewards": {"offense": np.float32(0.1), "defense": np.float32(-0.1)},
        }


@pytest.fixture(autouse=True)
def setup_jax_runtime():
    runtime = DummyJaxRuntime()
    game_state.jax_runtime = runtime
    yield runtime
    game_state.jax_runtime = None


def test_step_route_dispatches_action_payload_and_serializes_response(setup_jax_runtime):
    body = lifecycle_routes.step(
        ActionRequest(actions={"0": {"action": 0}, "1": 1}, team="OFFENSE")
    )

    assert body["status"] == "success"
    assert body["state"]["positions"] == [[1, 2], [3, 4]]
    assert isinstance(body["step_rewards"]["offense"], float)
    assert setup_jax_runtime.requests[0][0].actions["0"] == {"action": 0}


def test_step_route_preserves_structured_pointer_pass_payload(setup_jax_runtime):
    body = lifecycle_routes.step(
        ActionRequest(actions={"0": {"type": "PASS", "target": 1}, "1": 0})
    )

    assert body["actions_taken"]["0"] == "PASS_0"
    assert body["actions_taken_meta"]["0"] == {"type": "PASS", "target": 1}
    assert setup_jax_runtime.requests[0][0].actions["0"] == {
        "type": "PASS",
        "target": 1,
    }
