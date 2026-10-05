from __future__ import annotations

import pytest

from app.training_backend.config_schema import compile_overrides


def test_compile_overrides_rejects_internal_fields():
    with pytest.raises(ValueError, match="read-only in the training application"):
        compile_overrides(
            {"num_updates": 10},
            [
                {
                    "name": "num_updates",
                    "type": "int",
                    "option_strings": ["--num-updates"],
                    "editable": False,
                }
            ],
        )


def test_compile_overrides_handles_lists_and_choices():
    schema = [
        {
            "name": "policy_hidden_dims",
            "type": "list",
            "item_type": "int",
            "option_strings": ["--policy-hidden-dims"],
            "editable": True,
        },
        {
            "name": "policy_model",
            "type": "str",
            "option_strings": ["--policy-model"],
            "choices": ["mlp", "attention"],
            "editable": True,
        },
    ]
    assert compile_overrides(
        {"policy_hidden_dims": [64, 32], "policy_model": "attention"}, schema
    ) == [
        "--policy-hidden-dims",
        "64",
        "32",
        "--policy-model",
        "attention",
    ]


def test_compile_overrides_handles_value_style_boolean():
    assert compile_overrides(
        {"shot_pressure_enabled": False},
        [
            {
                "name": "shot_pressure_enabled",
                "type": "bool",
                "boolean_mode": "value",
                "option_strings": ["--shot-pressure-enabled"],
                "editable": True,
            }
        ],
    ) == ["--shot-pressure-enabled", "false"]
