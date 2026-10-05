from __future__ import annotations

from app.training_backend.mlflow_import import config_from_mlflow_values


def test_resolved_mlflow_values_populate_application_fields_and_overrides():
    schema = [
        {
            "name": "num_updates",
            "type": "int",
            "default": 500,
            "editable": False,
        },
        {
            "name": "gamma",
            "type": "float",
            "default": 0.99,
            "editable": True,
        },
        {
            "name": "policy_hidden_dims",
            "type": "list",
            "item_type": "int",
            "default": [128, 128],
            "editable": True,
        },
    ]
    config, matched, missing = config_from_mlflow_values(
        values={
            "num_updates": 5000,
            "gamma": 1.0,
            "policy_hidden_dims": [64, 32],
        },
        params={},
        schema=schema,
        tracking_uri="http://localhost:5000",
        experiment_name="halfcourt_multi_possessions",
        run_name="source",
    )

    assert config.name == "source copy"
    assert config.num_updates == 5000
    assert config.overrides == {
        "gamma": 1.0,
        "policy_hidden_dims": [64, 32],
    }
    assert matched == ["gamma", "num_updates", "policy_hidden_dims"]
    assert missing == []


def test_legacy_mlflow_params_are_typed_from_jax_namespaces():
    schema = [
        {
            "name": "gamma",
            "type": "float",
            "default": 0.99,
            "editable": True,
        },
        {
            "name": "enable_rebounds",
            "type": "bool",
            "default": False,
            "editable": True,
        },
    ]
    config, matched, _ = config_from_mlflow_values(
        values={},
        params={"jax/gamma": "1.0", "jax/env/enable_rebounds": "True"},
        schema=schema,
        tracking_uri="http://localhost:5000",
        experiment_name="legacy",
        run_name="legacy run",
    )

    assert config.overrides == {"gamma": 1.0, "enable_rebounds": True}
    assert matched == ["enable_rebounds", "gamma"]
