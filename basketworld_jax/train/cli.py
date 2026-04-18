from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from basketworld.envs.basketworld_env_v2 import Team
from train.config import get_parser


try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is expected but optional
    tqdm = None


class NullProgress:
    def update(self, n: int = 1) -> None:
        return None

    def set_postfix_str(self, s: str, refresh: bool = True) -> None:
        return None

    def close(self) -> None:
        return None


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = get_parser()
    parser.description = description
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to write the run result as JSON.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress indicators.",
    )
    return parser


def resolve_training_team(name: str) -> Team:
    if str(name).lower() == "defense":
        return Team.DEFENSE
    return Team.OFFENSE


def to_builtin(value: Any) -> Any:
    if isinstance(value, argparse.Namespace):
        return {k: to_builtin(v) for k, v in vars(value).items()}
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Team):
        return value.name
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(to_builtin(payload), indent=2, sort_keys=True) + "\n")


def ensure_jax_available(script_name: str):
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover - exercised only without JAX installed
        raise SystemExit(
            f"{script_name} requires JAX, but it is not installed in the active environment. "
            "Install both 'jax' and 'jaxlib' before running this trainer."
        ) from exc
    return jax, jnp


def build_progress(total: int, desc: str, disable: bool, unit: str = "step"):
    if disable or tqdm is None:
        return NullProgress()
    return tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True)
