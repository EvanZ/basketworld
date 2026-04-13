from __future__ import annotations

from pathlib import Path
from time import perf_counter_ns
from typing import Any, Sequence
import sys

import numpy as np

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

try:
    from benchmarks.common import build_benchmark_parser, build_progress, write_json
    from benchmarks.sbx_phase_a import (
        MASKED_LOGIT_FLOOR,
        NOOP_ACTION_INDEX,
        PhaseAPolicySpec,
        build_phase_a_policy_spec,
        collect_phase_a_samples,
        configure_phase_a_parser,
        validate_phase_a_args,
    )
except ImportError:  # pragma: no cover - direct script execution path
    from common import build_benchmark_parser, build_progress, write_json  # type: ignore[no-redef]
    from sbx_phase_a import (  # type: ignore[no-redef]
        MASKED_LOGIT_FLOOR,
        NOOP_ACTION_INDEX,
        PhaseAPolicySpec,
        build_phase_a_policy_spec,
        collect_phase_a_samples,
        configure_phase_a_parser,
        validate_phase_a_args,
    )


def parse_args(argv=None):
    parser = build_benchmark_parser(
        "Phase A Torch baseline: flat MLP policy forward pass with legal-action masking."
    )
    configure_phase_a_parser(parser)
    parser.set_defaults(mode="throughput", runner="sequential", device="cpu")
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
        help="Random seed used to initialize the Torch MLP weights.",
    )
    return parser.parse_args(argv)


def require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only when torch is absent
        raise SystemExit(
            "benchmarks/sbx_phase_a_torch.py requires PyTorch in the active environment."
        ) from exc
    return torch


class PhaseATorchPolicy:
    def __init__(
        self,
        torch_mod,
        spec: PhaseAPolicySpec,
        *,
        hidden_dims: Sequence[int],
        seed: int,
        device,
    ) -> None:
        self.torch = torch_mod
        self.spec = spec
        self.device = device
        torch_mod.manual_seed(int(seed))

        layers = []
        in_dim = int(spec.flat_obs_dim)
        for hidden_dim in hidden_dims:
            linear = torch_mod.nn.Linear(in_dim, int(hidden_dim), device=device)
            self._init_linear(linear, in_dim)
            layers.append(linear)
            layers.append(torch_mod.nn.Tanh())
            in_dim = int(hidden_dim)

        output = torch_mod.nn.Linear(in_dim, int(spec.total_action_dim), device=device)
        self._init_linear(output, in_dim)
        layers.append(output)

        self.model = torch_mod.nn.Sequential(*layers).to(device)
        self.model.eval()

    def _init_linear(self, layer, in_dim: int) -> None:
        std = float(np.sqrt(2.0 / max(1, int(in_dim))))
        self.torch.nn.init.normal_(layer.weight, mean=0.0, std=std)
        self.torch.nn.init.zeros_(layer.bias)

    def __call__(self, flat_obs):
        return self.model(flat_obs)


def _apply_action_mask_torch(torch_mod, flat_logits, action_mask, spec: PhaseAPolicySpec):
    batch_size = int(flat_logits.shape[0])
    logits = flat_logits.reshape(
        batch_size,
        int(spec.training_player_count),
        int(spec.action_dim_per_player),
    )
    legal = action_mask > 0
    has_legal = legal.any(dim=-1, keepdim=True)
    noop_mask = torch_mod.zeros_like(legal)
    noop_mask[..., NOOP_ACTION_INDEX] = True
    effective_legal = torch_mod.where(has_legal, legal, noop_mask)
    masked_logits = torch_mod.where(
        effective_legal,
        logits,
        torch_mod.full_like(logits, float(MASKED_LOGIT_FLOOR)),
    )
    probs = torch_mod.softmax(masked_logits, dim=-1)
    deterministic_actions = torch_mod.argmax(masked_logits, dim=-1)
    sampled_actions = torch_mod.distributions.Categorical(logits=masked_logits).sample()
    return {
        "flat_logits": flat_logits,
        "masked_logits": masked_logits,
        "probs": probs,
        "deterministic_actions": deterministic_actions,
        "sampled_actions": sampled_actions,
    }


def run_torch_policy_once(torch_mod, policy, flat_obs, action_mask, spec: PhaseAPolicySpec):
    flat_logits = policy(flat_obs)
    return _apply_action_mask_torch(torch_mod, flat_logits, action_mask, spec)


def _sync_torch(torch_mod, device) -> None:
    if str(device).startswith("cuda"):
        torch_mod.cuda.synchronize(device)


def run_phase_a_torch_benchmark(args) -> dict[str, Any]:
    torch_mod = require_torch()

    flat_obs_batch, action_mask_batch, training_player_ids = collect_phase_a_samples(args)
    spec = build_phase_a_policy_spec(
        flat_obs_batch,
        action_mask_batch,
        hidden_dims=args.policy_hidden_dims,
    )
    device_name = str(getattr(args, "device", "cpu") or "cpu")
    if device_name == "auto":
        device_name = "cuda" if torch_mod.cuda.is_available() else "cpu"
    device = torch_mod.device(device_name)

    flat_obs = torch_mod.as_tensor(flat_obs_batch, dtype=torch_mod.float32, device=device)
    action_mask = torch_mod.as_tensor(action_mask_batch, dtype=torch_mod.int8, device=device)
    policy = PhaseATorchPolicy(
        torch_mod,
        spec,
        hidden_dims=args.policy_hidden_dims,
        seed=int(args.policy_seed),
        device=device,
    )

    total_iters = int(args.warmup_iters) + int(args.benchmark_iters)
    progress = build_progress(
        total=total_iters,
        desc="phase_a_torch:policy",
        disable=bool(args.no_progress),
        unit="iter",
    )

    final_out = None
    with torch_mod.inference_mode():
        for _ in range(int(args.warmup_iters)):
            out = run_torch_policy_once(torch_mod, policy, flat_obs, action_mask, spec)
            _sync_torch(torch_mod, device)
            final_out = out
            progress.update(1)
            progress.set_postfix_str("warmup", refresh=False)

        timed_ns = 0
        for _ in range(int(args.benchmark_iters)):
            start_ns = perf_counter_ns()
            out = run_torch_policy_once(torch_mod, policy, flat_obs, action_mask, spec)
            _sync_torch(torch_mod, device)
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
        preview_det = (
            final_out["deterministic_actions"][:3].detach().cpu().numpy().astype(np.int32)
        )
        preview_sampled = (
            final_out["sampled_actions"][:3].detach().cpu().numpy().astype(np.int32)
        )

    return {
        "phase": "A",
        "prototype": "torch_mlp_baseline",
        "device": str(device),
        "policy_hidden_dims": [int(v) for v in spec.hidden_dims],
        "kernel_batch_size": int(args.kernel_batch_size),
        "warmup_iters": int(args.warmup_iters),
        "benchmark_iters": int(args.benchmark_iters),
        "flat_obs_dim": int(spec.flat_obs_dim),
        "training_player_ids": [int(v) for v in training_player_ids.tolist()],
        "training_player_count": int(spec.training_player_count),
        "action_dim_per_player": int(spec.action_dim_per_player),
        "total_action_dim": int(spec.total_action_dim),
        "policy_forward_masked_states_per_sec": float(states_per_sec),
        "mean_batch_latency_ms": float(mean_batch_latency_ms),
        "deterministic_action_preview": preview_det,
        "sampled_action_preview": preview_sampled,
    }


def main(argv=None):
    args = parse_args(argv)
    validate_phase_a_args(args)
    result = run_phase_a_torch_benchmark(args)

    print("Phase A Torch baseline")
    print(f"device: {result['device']}")
    print(f"flat_obs_dim: {result['flat_obs_dim']}")
    print(f"training_player_ids: {result['training_player_ids']}")
    print(f"action_dim_per_player: {result['action_dim_per_player']}")
    print(
        "policy_forward_masked:"
        f" states_per_sec={result['policy_forward_masked_states_per_sec']:.2f}"
        f" mean_batch_latency_ms={result['mean_batch_latency_ms']:.4f}"
    )

    if args.output_json:
        write_json(args.output_json, result)
        print(f"wrote_json: {args.output_json}")


if __name__ == "__main__":
    main()
