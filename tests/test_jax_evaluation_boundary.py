from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys

import pytest

from basketworld_jax.config import (
    JaxEnvironmentConfig,
    normalize_runtime_environment_config,
)
from basketworld_jax.eval.native import _resolve_native_environment_config
from basketworld_jax.inference import (
    JAXInferenceModel,
    UnsupportedCheckpointFormatError,
    load_inference_model,
    require_checkpoint_path,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_supported_evaluation_and_inference_sources_have_no_retired_imports():
    forbidden = (
        "basketworld.envs.basketworld_env_v2",
        "stable_baselines3",
        "torch",
    )
    paths = (
        PROJECT_ROOT / "basketworld_jax" / "eval" / "native.py",
        PROJECT_ROOT / "basketworld_jax" / "eval" / "fixed_pool_study.py",
        PROJECT_ROOT / "basketworld_jax" / "inference" / "policy.py",
    )

    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        for module in imported:
            assert not module.startswith(forbidden), (path, module)


def test_public_backend_selectors_do_not_dispatch_to_retired_runtimes():
    cases = (
        (
            PROJECT_ROOT / "app" / "backend" / "evaluation.py",
            "run_evaluation",
            {
                "_run_sequential_evaluation",
                "_run_parallel_evaluation",
                "HexagonBasketballEnv",
            },
        ),
        (
            PROJECT_ROOT / "app" / "backend" / "inference_adapters.py",
            "load_inference_policy",
            {"load_sb3_policy_adapter"},
        ),
    )

    for path, function_name, forbidden_names in cases:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == function_name
        )
        referenced = {
            node.id for node in ast.walk(function) if isinstance(node, ast.Name)
        }
        assert referenced.isdisjoint(forbidden_names), (path, referenced & forbidden_names)


def test_evaluation_and_skill_ratings_import_with_retired_runtime_unavailable():
    script = r'''
import importlib.abc
import sys

FORBIDDEN = (
    "basketworld.envs.basketworld_env_v2",
    "stable_baselines3",
    "torch",
)

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in FORBIDDEN):
            raise AssertionError(f"forbidden import: {fullname}")
        return None

sys.meta_path.insert(0, Blocker())
from basketworld_jax.eval.fixed_pool import generate_player_pool
from basketworld_jax.eval.lineup_schedule import build_balanced_lineup_schedule
from basketworld_jax.eval.native import run_native_jax_evaluation
from basketworld_jax.eval.fixed_pool_study import run_fixed_pool_study
from basketworld_jax.inference import JAXInferenceModel

pool = generate_player_pool(player_count=10, seed=19)
schedule = build_balanced_lineup_schedule(pool, game_count=2, seed=23)
assert len(schedule.games) == 2
assert callable(run_native_jax_evaluation)
assert callable(run_fixed_pool_study)
assert JAXInferenceModel.__module__ == "basketworld_jax.inference.policy"
'''
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_non_jax_checkpoint_has_clear_unsupported_format_error(tmp_path):
    sb3_artifact = tmp_path / "policy.zip"
    sb3_artifact.write_bytes(b"not-a-jax-checkpoint")

    with pytest.raises(UnsupportedCheckpointFormatError, match="SB3/PyTorch"):
        require_checkpoint_path(sb3_artifact)
    with pytest.raises(UnsupportedCheckpointFormatError, match="JAX checkpoint"):
        load_inference_model(sb3_artifact)


def test_historical_opponent_environment_contract_is_cross_validated():
    unified = {
        "env_config": {
            "players": 5,
            "court_rows": 16,
            "court_cols": 16,
            "enable_multi_possession": True,
            "multi_possession_schema_version": 4,
        }
    }
    opponent = {
        "env_config": {
            "players": 3,
            "court_rows": 16,
            "court_cols": 16,
            "enable_multi_possession": True,
            "multi_possession_schema_version": 4,
        }
    }

    with pytest.raises(ValueError, match="players=5/3"):
        _resolve_native_environment_config(
            required_params={},
            optional_params={},
            unified_payload=unified,
            opponent_payload=opponent,
        )

    overridden = _resolve_native_environment_config(
        required_params={"players": 5},
        optional_params={},
        unified_payload=unified,
        opponent_payload=opponent,
    )
    assert overridden.players == 5


@pytest.mark.parametrize("saved_schema", [2, 3, 4])
def test_supported_additive_checkpoint_schemas_use_current_runtime(saved_schema):
    config = JaxEnvironmentConfig(
        enable_multi_possession=True,
        multi_possession_schema_version=saved_schema,
    )

    normalized = normalize_runtime_environment_config(config)

    assert normalized.multi_possession_schema_version == 4


def test_inference_can_prepare_from_snapshot_without_python_environment():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    model = object.__new__(JAXInferenceModel)
    model.environment_config = JaxEnvironmentConfig(
        players=1,
        court_rows=9,
        court_cols=8,
    )
    model.jnp = jnp
    model._static_cache = {}
    model._prepared_env = None
    model._prepared_observer_is_offense = True
    snapshot = {"marker": 17}

    model.prepare_for_snapshot(snapshot, observer_is_offense=False)

    assert model._prepared_env == snapshot
    assert model._prepared_observer_is_offense is False
    assert int(model._static_cache[False].role_encoding.shape[0]) == 2
