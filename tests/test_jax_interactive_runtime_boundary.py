from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = (
    "basketworld.envs.basketworld_env_v2",
    "stable_baselines3",
    "torch",
)


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    return imported


def test_active_jax_interactive_sources_have_no_retired_imports():
    paths = (
        PROJECT_ROOT / "app" / "backend" / "jax_dev_runtime.py",
        PROJECT_ROOT / "basketworld_jax" / "config" / "environment.py",
        PROJECT_ROOT / "basketworld_jax" / "env" / "minimal.py",
    )
    for path in paths:
        for module in _imports(path):
            assert not module.startswith(FORBIDDEN), (path, module)

    runtime_source = paths[0].read_text(encoding="utf-8")
    assert "display_env" not in runtime_source
    assert "HexagonBasketballEnv" not in runtime_source


def test_jax_interactive_runtime_imports_with_retired_stack_unavailable():
    script = r'''
import importlib.abc
import numpy as np
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
from app.backend.jax_dev_runtime import JaxDevRuntime
from basketworld.domain import Team
from basketworld_jax.config.environment import JaxEnvironmentConfig, JaxStaticEnvironment

assert JaxDevRuntime.__module__ == "app.backend.jax_dev_runtime"
assert JaxStaticEnvironment(JaxEnvironmentConfig(players=1)).n_players == 2

class FakeRandom:
    @staticmethod
    def PRNGKey(seed):
        return np.asarray([0, int(seed)], dtype=np.uint32)

class FakeJax:
    random = FakeRandom()

class FakeSpec:
    multi_possession_features = False

class FakeModel:
    jax = FakeJax()
    jnp = np
    spec = FakeSpec()

runtime = JaxDevRuntime(
    required_params={"players": 1},
    env_params={"pass_mode": "pointer_targeted"},
    unified_policy=FakeModel(),
    opponent_policy=None,
    user_team=Team.OFFENSE,
    rng_seed=7,
)
assert isinstance(runtime.environment, JaxStaticEnvironment)
assert not hasattr(runtime, "display_env")
assert not hasattr(runtime.environment, "step")
'''
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
