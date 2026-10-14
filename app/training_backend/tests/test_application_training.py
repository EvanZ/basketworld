from __future__ import annotations

import pytest

from app.training_backend.application_training import (
    JAX_INAPPLICABLE_FIELDS,
    application_preset_values,
    load_application_config_schema,
    resolve_run_config_schema,
)
from app.training_backend.config_schema import compile_overrides


def test_application_schema_excludes_sb3_only_fields(tmp_path):
    schema = load_application_config_schema(str(tmp_path))
    names = {field["name"] for field in schema}

    assert not names.intersection(JAX_INAPPLICABLE_FIELDS)
    assert {
        "rollout_horizon",
        "policy_update_epochs",
        "ppo_minibatches",
        "kernel_batch_size",
        "policy_hidden_dims",
        "eval_deploy_every_updates",
        "shot_contest_skill_mean",
        "shot_contest_skill_std",
        "shot_contest_skill_effect_weight",
    }.issubset(names)

    fields = {field["name"]: field for field in schema}
    for name in (
        "shot_contest_skill_mean",
        "shot_contest_skill_std",
        "shot_contest_skill_effect_weight",
    ):
        assert fields[name]["editable"] is True
        assert fields[name]["category"] == "Shooting"


def test_application_preset_does_not_depend_on_inapplicable_fields(tmp_path):
    assert not set(application_preset_values(tmp_path)).intersection(
        JAX_INAPPLICABLE_FIELDS
    )


def test_application_preset_runs_ordinary_eval_every_250_updates(tmp_path):
    assert application_preset_values(tmp_path)["eval_every_updates"] == 250


def test_application_schema_rejects_excluded_sb3_override(tmp_path):
    schema = load_application_config_schema(str(tmp_path))
    with pytest.raises(ValueError, match="Unknown training configuration"):
        compile_overrides({"n_steps": 128}, schema)


def test_resolve_run_config_schema_reports_effective_values():
    fields = resolve_run_config_schema(
        (
            "/repo/.env/bin/python",
            "-m",
            "basketworld_jax.train.main",
            "--num-updates",
            "321",
            "--gamma",
            "0.97",
        )
    )
    values = {field["name"]: field["value"] for field in fields}

    assert values["num_updates"] == 321
    assert values["gamma"] == 0.97


def test_resolve_run_config_schema_tolerates_retired_launch_arguments():
    fields = resolve_run_config_schema(
        (
            "/repo/.env/bin/python",
            "-m",
            "basketworld_jax.train.main",
            "--check-clearance",
            "1",
            "--check-clearance-warmup-updates",
            "500",
            "--num-updates",
            "321",
        )
    )
    values = {field["name"]: field["value"] for field in fields}

    assert values["check_clearance"] == 1
    assert values["num_updates"] == 321
    assert "check_clearance_warmup_updates" not in values
