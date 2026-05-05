from __future__ import annotations

from basketworld.envs.basketworld_env_v2 import Team

from app.backend.routes.lifecycle_routes import (
    _jax_local_env_config_from_metadata,
    _overlay_jax_mlflow_env_params,
)


def test_overlay_jax_mlflow_env_params_applies_skill_stds():
    optional = {
        "training_team": Team.OFFENSE,
        "layup_pct": 0.60,
        "three_pt_pct": 0.40,
        "dunk_pct": 0.85,
    }
    params = {
        "jax/env/layup_pct": "0.55",
        "jax/env/three_pt_pct": "0.37",
        "jax/env/dunk_pct": "0.6",
        "jax/env/layup_std": "0.05",
        "jax/env/three_pt_std": "0.05",
        "jax/env/dunk_std": "0.3",
        "jax/env/training_team": "offense",
    }

    merged = _overlay_jax_mlflow_env_params(optional, params)

    assert merged["training_team"] == Team.OFFENSE
    assert merged["layup_pct"] == 0.55
    assert merged["three_pt_pct"] == 0.37
    assert merged["dunk_pct"] == 0.6
    assert merged["layup_std"] == 0.05
    assert merged["three_pt_std"] == 0.05
    assert merged["dunk_std"] == 0.3


def test_jax_local_env_config_prefers_checkpoint_env_config():
    class Policy:
        metadata = {
            "frozen_config": {
                "training_team": "offense",
                "layup_std": 0.0,
            },
            "env_config": {
                "training_team": "offense",
                "layup_std": 0.05,
                "three_pt_std": 0.05,
                "dunk_std": 0.3,
                "offensive_three_seconds": False,
            },
            "trainer_config": {"kernel_batch_size": 128},
        }

    required, optional, trainer, phi = _jax_local_env_config_from_metadata(Policy())

    assert required == {}
    assert optional["training_team"] == Team.OFFENSE
    assert optional["layup_std"] == 0.05
    assert optional["three_pt_std"] == 0.05
    assert optional["dunk_std"] == 0.3
    assert optional["offensive_three_seconds_enabled"] is False
    assert "offensive_three_seconds" not in optional
    assert trainer == {"kernel_batch_size": 128}
    assert phi == {}
