import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.backend.routes import lifecycle_routes
from app.backend.state import game_state


class DummyEnv:
    def __init__(self, n_players: int = 2):
        self.n_players = n_players
        self.offense_ids = list(range(n_players // 2))
        self.defense_ids = list(range(n_players // 2, n_players))
        self.training_team = None
        self.ball_holder = 0
        self.shot_clock = 24

    def step(self, actions):
        # Return numpy-heavy structures to exercise JSON encoding
        obs = {"obs": np.array([[1, 2], [3, 4]]), "action_mask": np.ones((self.n_players, 2))}
        reward = [0.1] * self.n_players
        done = False
        info = {"phi_ep_by_player": np.array([0.5] * self.n_players)}
        return obs, reward, done, False, info


@pytest.fixture(autouse=True)
def setup_dummy_env():
    # Reset game_state to a minimal dummy env before each test
    game_state.env = DummyEnv(n_players=2)
    game_state.obs = {"obs": np.zeros((2, 2)), "action_mask": np.ones((2, 2))}
    game_state.prev_obs = None
    game_state.reward_history = []
    game_state.episode_rewards = {"offense": 0.0, "defense": 0.0}
    game_state.frames = []
    game_state.actions_log = []
    yield
    # Teardown: clear env
    game_state.env = None
    game_state.obs = None


def test_step_route_handles_legacy_action_payloads_and_serializes_response():
    app = FastAPI()
    app.include_router(lifecycle_routes.router)
    client = TestClient(app)

    payload = {
        "actions": {
            "0": {"action": 0},
            "1": 1,
        },
        "team": "OFFENSE",
    }

    resp = client.post("/api/step", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["status"] == "success"
    assert "state" in body and isinstance(body["state"], dict)
    assert "episode_rewards" in body and isinstance(body["episode_rewards"], dict)
    # Step rewards/episode rewards should be JSON-friendly
    assert isinstance(body["step_rewards"]["offense"], float)
    assert isinstance(body["step_rewards"]["defense"], float)
