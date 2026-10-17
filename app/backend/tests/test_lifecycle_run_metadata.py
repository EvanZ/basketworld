from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.backend.routes import lifecycle_routes


class _RunClient:
    def __init__(self, params):
        self.params = params

    def get_run(self, _run_id):
        return SimpleNamespace(data=SimpleNamespace(params=self.params))


def test_run_metadata_recognizes_multi_possession_mode():
    assert lifecycle_routes._run_uses_multi_possession(
        _RunClient({"enable_multi_possession": "true"}), "run-1"
    ) is True


def test_run_metadata_exposes_clock_mode_and_legacy_defaults():
    assert lifecycle_routes._run_game_mode_config(
        _RunClient({
            "jax/env/enable_multi_possession": "true",
            "jax/env/episode_end_mode": "game_clock",
            "jax/env/game_clock_seconds": "300",
            "jax/env/overtime_seconds": "120",
        }),
        "clock-run",
    ) == {
        "enable_multi_possession": True,
        "episode_end_mode": "game_clock",
        "game_clock_seconds": 300,
        "overtime_seconds": 120,
    }
    assert lifecycle_routes._run_game_mode_config(
        _RunClient({"enable_multi_possession": "true"}),
        "legacy-run",
    )["episode_end_mode"] == "possessions"


def test_dev_game_rejects_checkpoint_mode_mismatch_before_override():
    lifecycle_routes._validate_requested_episode_end_mode(
        {"episode_end_mode": "game_clock"},
        "game_clock",
    )
    with pytest.raises(HTTPException, match="Checkpoint episode mode mismatch") as error:
        lifecycle_routes._validate_requested_episode_end_mode(
            {"episode_end_mode": "possessions"},
            "game_clock",
        )
    assert error.value.status_code == 400
    assert lifecycle_routes._run_uses_multi_possession(
        _RunClient({"enable_multi_possession": "false"}), "run-2"
    ) is False
    assert lifecycle_routes._run_uses_multi_possession(
        _RunClient({"jax/env/enable_multi_possession": "true"}), "run-3"
    ) is True
