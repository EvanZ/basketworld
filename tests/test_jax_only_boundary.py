from __future__ import annotations

import ast
import importlib
import tokenize
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_ROOTS = (
    REPO_ROOT / "basketworld",
    REPO_ROOT / "basketworld_jax",
    REPO_ROOT / "app" / "backend",
    REPO_ROOT / "app" / "training_backend",
    REPO_ROOT / "analytics",
    REPO_ROOT / "benchmarks",
    REPO_ROOT / "scripts",
)
RETIRED_IMPORT_PREFIXES = (
    "stable_baselines3",
    "torch",
    "gymnasium",
    "basketworld.algorithms",
    "basketworld.envs",
    "basketworld.policies",
    "train",
)
SUPPORTED_IMPORTS = (
    "basketworld_jax.config.environment",
    "basketworld_jax.env.minimal",
    "basketworld_jax.inference.policy",
    "basketworld_jax.train.cli",
    "app.backend.inference_adapters",
    "app.backend.policies",
    "app.backend.routes.lifecycle_routes",
    "app.backend.routes.playable_routes",
    "app.training_backend.application_training",
)


def _production_python_files() -> list[Path]:
    files: list[Path] = []
    for root in PRODUCTION_ROOTS:
        if not root.exists():
            continue
        files.extend(
            path
            for path in root.rglob("*.py")
            if "tests" not in path.parts
            and "__pycache__" not in path.parts
            and not any(part.startswith(".") for part in path.relative_to(root).parts)
        )
    return sorted(files)


def _is_retired_import(module: str) -> bool:
    return any(
        module == prefix or module.startswith(f"{prefix}.")
        for prefix in RETIRED_IMPORT_PREFIXES
    )


def test_supported_python_has_no_retired_runtime_imports() -> None:
    violations: list[str] = []
    for path in _production_python_files():
        with tokenize.open(path) as source_file:
            tree = ast.parse(source_file.read(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            else:
                continue
            for module in modules:
                if _is_retired_import(module):
                    relative = path.relative_to(REPO_ROOT)
                    violations.append(f"{relative}:{node.lineno}: {module}")

    assert violations == []


def test_retired_runtime_source_trees_are_absent() -> None:
    retired_paths = (
        REPO_ROOT / "basketworld" / "algorithms",
        REPO_ROOT / "basketworld" / "envs",
        REPO_ROOT / "basketworld" / "policies",
        REPO_ROOT / "train" / "train.py",
        REPO_ROOT / "app" / "backend" / "rollout_runtime.py",
        REPO_ROOT / "app" / "backend" / "selector_runtime.py",
    )
    assert [str(path.relative_to(REPO_ROOT)) for path in retired_paths if path.exists()] == []


def test_supported_dependency_manifests_exclude_retired_packages() -> None:
    manifests = (
        REPO_ROOT / "pyproject.toml",
        REPO_ROOT / "setup.py",
        REPO_ROOT / "requirements.txt",
        REPO_ROOT / "app" / "backend" / "requirements.txt",
    )
    retired_packages = ("stable-baselines3", "torch", "torchvision", "gymnasium")
    violations: list[str] = []
    for manifest in manifests:
        text = manifest.read_text(encoding="utf-8").lower()
        for package in retired_packages:
            if package in text:
                violations.append(f"{manifest.relative_to(REPO_ROOT)}: {package}")
    assert violations == []


@pytest.mark.parametrize("module_name", SUPPORTED_IMPORTS)
def test_supported_workflow_module_imports(module_name: str) -> None:
    importlib.import_module(module_name)
