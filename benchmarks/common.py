from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Iterable

import numpy as np

from basketworld.envs.basketworld_env_v2 import Team
from train.config import get_parser


BENCHMARK_ONLY_KEYS = {
    "action_mode",
    "benchmark_iters",
    "episodes",
    "horizon",
    "kernel_batch_size",
    "kernel_horizon",
    "mode",
    "no_progress",
    "num_envs",
    "output_json",
    "profile_top_k",
    "runner",
    "sample_reset_seed",
    "seed",
    "training_team",
    "warmup_iters",
}

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is expected but optional
    tqdm = None


class Timer:
    """Small timing helper backed by perf_counter_ns."""

    def __enter__(self) -> "Timer":
        self.start_ns = perf_counter_ns()
        self.elapsed_ns = 0
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.elapsed_ns = perf_counter_ns() - self.start_ns


class NullProgress:
    def update(self, n: int = 1) -> None:
        return None

    def set_postfix_str(self, s: str, refresh: bool = True) -> None:
        return None

    def close(self) -> None:
        return None


def build_benchmark_parser(description: str) -> argparse.ArgumentParser:
    parser = get_parser()
    parser.description = description
    parser.set_defaults(num_envs=1, enable_env_profiling=False)
    parser.add_argument(
        "--mode",
        choices=["throughput", "hotspot"],
        default="throughput",
        help="Benchmark mode. Throughput disables profiling; hotspot enables env profiling.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Number of episodes to run per env instance.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=64,
        help="Maximum number of steps to execute per episode before forcing a reset.",
    )
    parser.add_argument(
        "--action-mode",
        choices=["random_legal", "pregenerated_legal"],
        default="pregenerated_legal",
        help="How legal actions are chosen for the benchmark.",
    )
    parser.add_argument(
        "--training-team",
        choices=["offense", "defense"],
        default="offense",
        help="Training-side observer/team to benchmark.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed used for env resets and deterministic action generation.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to write the benchmark result as JSON.",
    )
    parser.add_argument(
        "--profile-top-k",
        type=int,
        default=10,
        help="Number of profile sections to keep in ranked output.",
    )
    parser.add_argument(
        "--runner",
        choices=["sequential", "dummy_vec", "subproc_vec"],
        default="sequential",
        help=(
            "Execution path to benchmark. "
            "'sequential' steps wrapped envs in-process one at a time; "
            "'dummy_vec' uses SB3 DummyVecEnv; "
            "'subproc_vec' uses SB3 SubprocVecEnv like the current training stack."
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress indicators for benchmark runs.",
    )
    return parser


def benchmark_args_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: to_builtin(value)
        for key, value in vars(args).items()
        if key not in BENCHMARK_ONLY_KEYS
    }


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


def try_import_jax():
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        return None, None, exc
    return jax, jnp, None


def ensure_jax_available(script_name: str):
    jax, jnp, exc = try_import_jax()
    if exc is not None:
        raise SystemExit(
            f"{script_name} requires JAX, but it is not installed in the active environment. "
            "Install both 'jax' and 'jaxlib' before running this benchmark."
        ) from exc
    return jax, jnp


def build_progress(total: int, desc: str, disable: bool, unit: str = "ep"):
    if disable or tqdm is None:
        return NullProgress()
    return tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True)


def extract_action_mask(obs: Any) -> np.ndarray:
    if not isinstance(obs, dict) or "action_mask" not in obs:
        raise ValueError("Benchmark expects dict observations with an 'action_mask' field.")
    return np.asarray(obs["action_mask"], dtype=np.int8)


def generate_action_ranks(
    rng: np.random.Generator,
    num_envs: int,
    episodes: int,
    horizon: int,
    n_players: int,
) -> np.ndarray:
    return rng.integers(
        low=0,
        high=np.iinfo(np.int64).max,
        size=(num_envs, episodes, horizon, n_players),
        dtype=np.int64,
    )


def choose_legal_actions(
    action_mask: np.ndarray,
    rng: np.random.Generator,
    action_mode: str,
    action_ranks: np.ndarray | None = None,
) -> np.ndarray:
    mask = np.asarray(action_mask, dtype=np.int8)
    out = np.zeros(mask.shape[0], dtype=np.int64)

    for player_idx in range(mask.shape[0]):
        legal = np.flatnonzero(mask[player_idx])
        if legal.size == 0:
            out[player_idx] = 0
            continue

        if action_mode == "pregenerated_legal":
            if action_ranks is None:
                raise ValueError("pregenerated_legal mode requires action ranks.")
            rank = int(action_ranks[player_idx])
            out[player_idx] = int(legal[rank % legal.size])
            continue

        out[player_idx] = int(rng.choice(legal))

    return out


def aggregate_profile_stats(
    stats_by_env: Iterable[dict[str, dict[str, float]]],
) -> dict[str, dict[str, float]]:
    aggregated: dict[str, dict[str, float]] = {}

    for env_stats in stats_by_env:
        for section_name, metrics in env_stats.items():
            entry = aggregated.setdefault(
                section_name,
                {"total_ms": 0.0, "calls": 0.0, "avg_us": 0.0},
            )
            entry["total_ms"] += float(metrics.get("total_ms", 0.0))
            entry["calls"] += float(metrics.get("calls", 0.0))

    for section_name, metrics in aggregated.items():
        calls = max(1.0, float(metrics["calls"]))
        metrics["avg_us"] = (float(metrics["total_ms"]) * 1000.0) / calls
        aggregated[section_name] = {
            "total_ms": float(metrics["total_ms"]),
            "calls": float(metrics["calls"]),
            "avg_us": float(metrics["avg_us"]),
        }

    return aggregated


def rank_profile_sections(
    profile_stats: dict[str, dict[str, float]],
    sort_key: str,
    limit: int,
) -> list[dict[str, float | str]]:
    ranked = []
    for section_name, metrics in profile_stats.items():
        ranked.append(
            {
                "section": section_name,
                "total_ms": float(metrics.get("total_ms", 0.0)),
                "calls": float(metrics.get("calls", 0.0)),
                "avg_us": float(metrics.get("avg_us", 0.0)),
            }
        )

    ranked.sort(key=lambda item: float(item.get(sort_key, 0.0)), reverse=True)
    return ranked[: max(0, int(limit))]


def require_jax():
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover - used by future JAX scripts
        raise SystemExit(
            "JAX is not installed in this environment. Install 'jax' and 'jaxlib' "
            "before running JAX benchmark scripts."
        ) from exc
    return jax, jnp
