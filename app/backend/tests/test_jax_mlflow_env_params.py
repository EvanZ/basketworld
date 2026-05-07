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
        "jax/env/allow_dunks": "true",
        "jax/env/training_team": "offense",
        "jax/env/enable_intent_learning": "true",
        "jax/env/enable_defense_intent_learning": "true",
        "jax/env/num_intents": "6",
        "jax/env/intent_commitment_steps": "5",
        "jax/env/intent_null_prob": "0.1",
        "jax/env/defense_intent_null_prob": "0.2",
        "jax/env/intent_visible_to_defense_prob": "0.3",
    }

    merged = _overlay_jax_mlflow_env_params(optional, params)

    assert merged["training_team"] == Team.OFFENSE
    assert merged["allow_dunks"] is True
    assert merged["layup_pct"] == 0.55
    assert merged["three_pt_pct"] == 0.37
    assert merged["dunk_pct"] == 0.6
    assert merged["layup_std"] == 0.05
    assert merged["three_pt_std"] == 0.05
    assert merged["dunk_std"] == 0.3
    assert merged["enable_intent_learning"] is True
    assert merged["enable_defense_intent_learning"] is True
    assert merged["num_intents"] == 6
    assert merged["intent_commitment_steps"] == 5
    assert merged["intent_null_prob"] == 0.1
    assert merged["defense_intent_null_prob"] == 0.2
    assert merged["intent_visible_to_defense_prob"] == 0.3


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
                "enable_intent_learning": True,
                "enable_defense_intent_learning": True,
                "num_intents": 6,
                "intent_commitment_steps": 5,
                "intent_null_prob": 0.1,
                "defense_intent_null_prob": 0.2,
                "intent_visible_to_defense_prob": 0.3,
            },
            "trainer_config": {"kernel_batch_size": 128},
        }

    required, optional, trainer, phi = _jax_local_env_config_from_metadata(Policy())

    assert required == {}
    assert optional["training_team"] == Team.OFFENSE
    assert optional["allow_dunks"] is True
    assert optional["layup_std"] == 0.05
    assert optional["three_pt_std"] == 0.05
    assert optional["dunk_std"] == 0.3
    assert optional["offensive_three_seconds_enabled"] is False
    assert optional["enable_intent_learning"] is True
    assert optional["enable_defense_intent_learning"] is True
    assert optional["num_intents"] == 6
    assert optional["intent_commitment_steps"] == 5
    assert optional["intent_null_prob"] == 0.1
    assert optional["defense_intent_null_prob"] == 0.2
    assert optional["intent_visible_to_defense_prob"] == 0.3
    assert "offensive_three_seconds" not in optional
    assert trainer == {"kernel_batch_size": 128}
    assert phi == {}
