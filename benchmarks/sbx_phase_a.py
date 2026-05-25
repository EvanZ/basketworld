from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Sequence
import sys

import numpy as np

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from basketworld.envs.basketworld_env_v2 import Team
try:
    from benchmarks.common import (
        build_benchmark_parser,
        build_progress,
        choose_legal_actions,
        ensure_jax_available,
        resolve_training_team,
        to_builtin,
        write_json,
    )
except ImportError:  # pragma: no cover - direct script execution path
    from common import (  # type: ignore[no-redef]
        build_benchmark_parser,
        build_progress,
        choose_legal_actions,
        ensure_jax_available,
        resolve_training_team,
        to_builtin,
        write_json,
    )
from train.env_factory import setup_environment


PHASE_A_FLAT_OBS_KEYS = ("obs", "role_flag", "skills")
PHASE_A_IGNORED_OBS_KEYS = {"action_mask"}
NOOP_ACTION_INDEX = 0
MASKED_LOGIT_FLOOR = -1.0e9

PHASE_A_FROZEN_VALUES: dict[str, Any] = {
    "training_team": "offense",
    "players": 3,
    "court_rows": 9,
    "court_cols": 8,
    "shot_clock": 24,
    "min_shot_clock": 14,
    "layup_pct": 0.60,
    "three_pt_pct": 0.37,
    "dunk_pct": 0.60,
    "three_point_distance": 4.25,
    "three_point_short_distance": 3,
    "three_pt_extra_hex_decay": 0.05,
    "shot_pressure_enabled": True,
    "shot_pressure_max": 0.25,
    "shot_pressure_lambda": 1.0,
    "shot_pressure_arc_degrees": 300,
    "defender_pressure_distance": 3,
    "defender_pressure_turnover_chance": 0.02,
    "defender_pressure_decay_lambda": 1.0,
    "base_steal_rate": 0.3,
    "steal_perp_decay": 1.5,
    "steal_distance_factor": 0.2,
    "spawn_distance": 4,
    "max_spawn_distance": 7,
    "defender_spawn_distance": 2,
    "defender_guard_distance": 1,
    "assist_window": 3,
    "mask_occupied_moves": False,
    "enable_pass_gating": True,
    "pass_mode": "directional",
    "use_set_obs": False,
    "use_dual_policy": False,
    "use_dual_critic": False,
    "enable_intent_learning": False,
    "enable_defense_intent_learning": False,
    "intent_selector_enabled": False,
    "intent_diversity_enabled": False,
    "start_template_enabled": False,
    "enable_phi_shaping": False,
}


@dataclass(frozen=True)
class PhaseAPolicySpec:
    flat_obs_dim: int
    training_player_count: int
    action_dim_per_player: int
    total_action_dim: int
    hidden_dims: tuple[int, ...]


def configure_phase_a_parser(parser) -> None:
    parser.set_defaults(**PHASE_A_FROZEN_VALUES)


def parse_args(argv=None):
    parser = build_benchmark_parser(
        "Phase A SBX feasibility prototype: flat JAX MLP policy forward pass with legal-action masking."
    )
    configure_phase_a_parser(parser)
    parser.set_defaults(mode="throughput", runner="sequential")
    parser.add_argument(
        "--kernel-batch-size",
        type=int,
        default=256,
        help="Number of sampled env states to pack into one batched policy forward pass.",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=3,
        help="Number of warm iterations to run before timing steady-state execution.",
    )
    parser.add_argument(
        "--benchmark-iters",
        type=int,
        default=100,
        help="Number of timed iterations to run for the policy forward pass.",
    )
    parser.add_argument(
        "--sample-reset-seed",
        type=int,
        default=0,
        help="Base reset seed used when sampling representative env states.",
    )
    parser.add_argument(
        "--policy-hidden-dims",
        type=int,
        nargs="+",
        default=[128, 128],
        help="Hidden layer widths for the flat MLP prototype.",
    )
    parser.add_argument(
        "--policy-seed",
        type=int,
        default=0,
        help="Random seed used to initialize the JAX MLP weights.",
    )
    return parser.parse_args(argv)


def _values_match(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        return np.isclose(float(actual), float(expected), atol=1e-8, rtol=0.0)
    return actual == expected


def validate_phase_a_args(args) -> None:
    mismatches: list[str] = []
    for key, expected in PHASE_A_FROZEN_VALUES.items():
        actual = getattr(args, key)
        if not _values_match(actual, expected):
            mismatches.append(f"{key}={actual!r} expected {expected!r}")
    if mismatches:
        joined = ", ".join(mismatches)
        raise SystemExit(
            "Phase A uses a frozen reduced config. The following overrides are not supported: "
            f"{joined}"
        )


def build_phase_a_args(extra_argv: Sequence[str] | None = None):
    argv = list(extra_argv or [])
    args = parse_args(argv)
    validate_phase_a_args(args)
    return args


def flatten_phase_a_observation(obs_payload: Any) -> np.ndarray:
    if not isinstance(obs_payload, dict):
        return np.asarray(obs_payload, dtype=np.float32).reshape(-1)

    parts: list[np.ndarray] = []
    for key in PHASE_A_FLAT_OBS_KEYS:
        value = obs_payload.get(key)
        if value is None:
            continue
        parts.append(np.asarray(value, dtype=np.float32).reshape(-1))
    if parts:
        return np.concatenate(parts).astype(np.float32, copy=False)

    fallback_parts: list[np.ndarray] = []
    for key in sorted(obs_payload.keys()):
        if key in PHASE_A_IGNORED_OBS_KEYS:
            continue
        value = obs_payload.get(key)
        if value is None:
            continue
        fallback_parts.append(np.asarray(value, dtype=np.float32).reshape(-1))
    if fallback_parts:
        return np.concatenate(fallback_parts).astype(np.float32, copy=False)

    return np.zeros((1,), dtype=np.float32)


def flatten_phase_a_observation_batch(obs_payload: Any) -> np.ndarray:
    if not isinstance(obs_payload, dict):
        arr = np.asarray(obs_payload, dtype=np.float32)
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        return arr.reshape(arr.shape[0], -1)

    parts: list[np.ndarray] = []
    batch_size: int | None = None
    for key in PHASE_A_FLAT_OBS_KEYS:
        value = obs_payload.get(key)
        if value is None:
            continue
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)
        else:
            arr = arr.reshape(arr.shape[0], -1)
        if batch_size is None:
            batch_size = int(arr.shape[0])
        elif int(arr.shape[0]) != batch_size:
            raise ValueError(
                f"Observation batch key '{key}' has batch={arr.shape[0]} but expected {batch_size}."
            )
        parts.append(arr)

    if parts:
        return np.concatenate(parts, axis=1).astype(np.float32, copy=False)

    return np.zeros((1, 1), dtype=np.float32)


def extract_phase_a_training_action_mask(
    action_mask: np.ndarray,
    training_player_ids: Sequence[int],
) -> np.ndarray:
    mask = np.asarray(action_mask, dtype=np.int8)
    indices = np.asarray(training_player_ids, dtype=np.int64)
    if mask.ndim == 2:
        return mask[indices]
    if mask.ndim == 3:
        return mask[:, indices, :]
    raise ValueError(
        f"Expected action_mask with rank 2 or 3, received shape {tuple(mask.shape)}."
    )


def _resolve_training_player_ids(env, training_team: Team) -> np.ndarray:
    base_env = env.unwrapped
    if training_team == Team.DEFENSE:
        return np.asarray(base_env.defense_ids, dtype=np.int32)
    return np.asarray(base_env.offense_ids, dtype=np.int32)


def collect_phase_a_samples(args) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    training_team = resolve_training_team(args.training_team)
    env = setup_environment(args, training_team)
    base_env = env.unwrapped
    training_player_ids = _resolve_training_player_ids(env, training_team)
    sample_rng = np.random.default_rng(int(args.seed))
    progress = build_progress(
        total=int(args.kernel_batch_size),
        desc="phase_a:samples",
        disable=bool(args.no_progress),
        unit="state",
    )

    flat_obs_rows: list[np.ndarray] = []
    action_mask_rows: list[np.ndarray] = []
    obs, _ = env.reset(seed=int(args.sample_reset_seed))
    steps_in_episode = 0

    try:
        while len(flat_obs_rows) < int(args.kernel_batch_size):
            flat_obs_rows.append(flatten_phase_a_observation(obs))
            action_mask_rows.append(
                extract_phase_a_training_action_mask(obs["action_mask"], training_player_ids)
            )
            progress.update(1)
            progress.set_postfix_str(f"states={len(flat_obs_rows)}", refresh=False)

            if len(flat_obs_rows) >= int(args.kernel_batch_size):
                break

            action_ranks = None
            if args.action_mode == "pregenerated_legal":
                action_ranks = sample_rng.integers(
                    low=0,
                    high=np.iinfo(np.int64).max,
                    size=(base_env.n_players,),
                    dtype=np.int64,
                )
            actions = choose_legal_actions(
                obs["action_mask"],
                rng=sample_rng,
                action_mode=args.action_mode,
                action_ranks=action_ranks,
            )
            obs, _, terminated, truncated, _ = env.step(actions)
            steps_in_episode += 1
            if bool(terminated or truncated or steps_in_episode >= int(args.horizon)):
                obs, _ = env.reset(
                    seed=int(args.sample_reset_seed) + len(flat_obs_rows) + 1
                )
                steps_in_episode = 0
    finally:
        progress.close()
        env.close()

    flat_obs_batch = np.stack(flat_obs_rows, axis=0).astype(np.float32, copy=False)
    action_mask_batch = np.stack(action_mask_rows, axis=0).astype(np.int8, copy=False)
    return flat_obs_batch, action_mask_batch, training_player_ids


def build_phase_a_policy_spec(
    flat_obs_batch: np.ndarray,
    action_mask_batch: np.ndarray,
    hidden_dims: Sequence[int],
) -> PhaseAPolicySpec:
    if action_mask_batch.ndim != 3:
        raise ValueError(
            f"Expected action_mask batch shape (batch, players, actions), got {action_mask_batch.shape}."
        )
    training_player_count = int(action_mask_batch.shape[1])
    action_dim_per_player = int(action_mask_batch.shape[2])
    return PhaseAPolicySpec(
        flat_obs_dim=int(flat_obs_batch.shape[1]),
        training_player_count=training_player_count,
        action_dim_per_player=action_dim_per_player,
        total_action_dim=training_player_count * action_dim_per_player,
        hidden_dims=tuple(int(v) for v in hidden_dims),
    )


def init_phase_a_policy_params(jax, jnp, spec: PhaseAPolicySpec, *, seed: int):
    dims = [int(spec.flat_obs_dim), *[int(v) for v in spec.hidden_dims], int(spec.total_action_dim)]
    keys = jax.random.split(jax.random.PRNGKey(int(seed)), len(dims) - 1)
    params = []
    for key, in_dim, out_dim in zip(keys, dims[:-1], dims[1:]):
        scale = np.sqrt(2.0 / max(1, int(in_dim)))
        weights = (
            jax.random.normal(key, shape=(int(in_dim), int(out_dim)), dtype=jnp.float32)
            * jnp.asarray(scale, dtype=jnp.float32)
        )
        bias = jnp.zeros((int(out_dim),), dtype=jnp.float32)
        params.append((weights, bias))
    return tuple(params)


def _build_jitted_policy_forward(jax, jnp, spec: PhaseAPolicySpec):
    total_action_dim = int(spec.total_action_dim)
    training_player_count = int(spec.training_player_count)
    action_dim_per_player = int(spec.action_dim_per_player)

    def _forward_logits(params, flat_obs):
        x = flat_obs
        for layer_idx, (weights, bias) in enumerate(params):
            x = jnp.matmul(x, weights) + bias
            if layer_idx < len(params) - 1:
                x = jnp.tanh(x)
        return x

    def _apply_action_mask(flat_logits, action_mask):
        batch_size = flat_logits.shape[0]
        logits = flat_logits.reshape(batch_size, training_player_count, action_dim_per_player)
        legal = action_mask > 0
        has_legal = jnp.any(legal, axis=-1, keepdims=True)
        noop_mask = jnp.zeros_like(legal)
        noop_mask = noop_mask.at[..., NOOP_ACTION_INDEX].set(True)
        effective_legal = jnp.where(has_legal, legal, noop_mask)
        masked_logits = jnp.where(
            effective_legal,
            logits,
            jnp.full_like(logits, MASKED_LOGIT_FLOOR),
        )
        probs = jax.nn.softmax(masked_logits, axis=-1)
        deterministic_actions = jnp.argmax(masked_logits, axis=-1).astype(jnp.int32)
        return masked_logits, probs, deterministic_actions

    @jax.jit
    def _forward(params, flat_obs, action_mask, sample_key):
        flat_logits = _forward_logits(params, flat_obs)
        masked_logits, probs, deterministic_actions = _apply_action_mask(
            flat_logits, action_mask
        )
        sampled_actions = jax.random.categorical(
            sample_key, masked_logits, axis=-1
        ).astype(jnp.int32)
        return {
            "flat_logits": flat_logits,
            "masked_logits": masked_logits,
            "probs": probs,
            "deterministic_actions": deterministic_actions,
            "sampled_actions": sampled_actions,
        }

    return _forward


def run_phase_a_policy_benchmark(args) -> dict[str, Any]:
    jax, jnp = ensure_jax_available("benchmarks/sbx_phase_a.py")

    flat_obs_batch, action_mask_batch, training_player_ids = collect_phase_a_samples(args)
    spec = build_phase_a_policy_spec(
        flat_obs_batch,
        action_mask_batch,
        hidden_dims=args.policy_hidden_dims,
    )
    params = init_phase_a_policy_params(
        jax,
        jnp,
        spec,
        seed=int(args.policy_seed),
    )
    flat_obs_device = jnp.asarray(flat_obs_batch, dtype=jnp.float32)
    action_mask_device = jnp.asarray(action_mask_batch, dtype=jnp.int8)
    forward_fn = _build_jitted_policy_forward(jax, jnp, spec)

    total_iters = int(args.warmup_iters) + int(args.benchmark_iters)
    progress = build_progress(
        total=total_iters,
        desc="phase_a:policy",
        disable=bool(args.no_progress),
        unit="iter",
    )

    sample_key = jax.random.PRNGKey(int(args.policy_seed) + 1)
    for idx in range(int(args.warmup_iters)):
        sample_key = jax.random.fold_in(sample_key, idx)
        out = forward_fn(params, flat_obs_device, action_mask_device, sample_key)
        jax.block_until_ready(out)
        progress.update(1)
        progress.set_postfix_str("warmup", refresh=False)

    timed_ns = 0
    final_out = None
    for idx in range(int(args.benchmark_iters)):
        sample_key = jax.random.fold_in(sample_key, idx + 10_000)
        start_ns = perf_counter_ns()
        out = forward_fn(params, flat_obs_device, action_mask_device, sample_key)
        jax.block_until_ready(out)
        timed_ns += perf_counter_ns() - start_ns
        final_out = out
        progress.update(1)
        progress.set_postfix_str("benchmark", refresh=False)
    progress.close()

    total_states = int(args.kernel_batch_size) * int(args.benchmark_iters)
    total_seconds = max(timed_ns / 1e9, 1e-12)
    states_per_sec = total_states / total_seconds
    mean_batch_latency_ms = (timed_ns / 1e6) / max(1, int(args.benchmark_iters))

    preview_det = None
    preview_sampled = None
    if final_out is not None:
        preview_det = np.asarray(final_out["deterministic_actions"][:3], dtype=np.int32)
        preview_sampled = np.asarray(final_out["sampled_actions"][:3], dtype=np.int32)

    return {
        "phase": "A",
        "prototype": "dependency_light_jax_mlp",
        "frozen_config": {key: to_builtin(getattr(args, key)) for key in PHASE_A_FROZEN_VALUES},
        "policy_hidden_dims": [int(v) for v in spec.hidden_dims],
        "kernel_batch_size": int(args.kernel_batch_size),
        "warmup_iters": int(args.warmup_iters),
        "benchmark_iters": int(args.benchmark_iters),
        "flat_obs_dim": int(spec.flat_obs_dim),
        "training_player_ids": [int(v) for v in training_player_ids.tolist()],
        "training_player_count": int(spec.training_player_count),
        "action_dim_per_player": int(spec.action_dim_per_player),
        "total_action_dim": int(spec.total_action_dim),
        "policy_forward_states_per_sec": float(states_per_sec),
        "mean_batch_latency_ms": float(mean_batch_latency_ms),
        "deterministic_action_preview": preview_det,
        "sampled_action_preview": preview_sampled,
    }


def main(argv=None):
    args = parse_args(argv)
    validate_phase_a_args(args)
    result = run_phase_a_policy_benchmark(args)

    print("Phase A SBX feasibility prototype")
    print(f"flat_obs_dim: {result['flat_obs_dim']}")
    print(f"training_player_ids: {result['training_player_ids']}")
    print(f"action_dim_per_player: {result['action_dim_per_player']}")
    print(
        "policy_forward_masked:"
        f" states_per_sec={result['policy_forward_states_per_sec']:.2f}"
        f" mean_batch_latency_ms={result['mean_batch_latency_ms']:.4f}"
    )

    if args.output_json:
        write_json(args.output_json, result)
        print(f"wrote_json: {args.output_json}")


if __name__ == "__main__":
    main()
