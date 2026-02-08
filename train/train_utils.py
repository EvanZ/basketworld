import math
import random
import torch
from typing import List, Optional


def get_device(device_arg):
    """Return torch.device from user argument."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("CUDA requested but not available, falling back to CPU")
        return torch.device("cpu")
    return torch.device(device_arg)


def linear_schedule(start, end):
    """Return a function for a linear schedule from start to end over training."""
    def f(progress_remaining: float):
        return end + (start - end) * progress_remaining

    return f


def get_steps_for_alternation(
    alternation_idx: int,
    total_alternations: int,
    start_steps: int,
    end_steps: int,
    schedule_type: str = "linear",
) -> int:
    """Compute steps for an alternation using linear or log schedule."""
    if total_alternations <= 1 or start_steps == end_steps:
        return start_steps
    if schedule_type == "constant":
        return start_steps

    progress = alternation_idx / (total_alternations - 1)

    if schedule_type == "log":
        k = 9.0
        log_progress = math.log1p(progress * k) / math.log1p(k)
        return int(round(start_steps + (end_steps - start_steps) * log_progress))

    return int(round(start_steps + (end_steps - start_steps) * progress))


def calculate_total_timesteps_with_schedule(
    total_alternations: int,
    start_steps: int,
    end_steps: int,
    schedule_type: str,
    num_envs: int,
    n_steps: int,
    offset: int = 0,
    schedule_total_alternations: Optional[int] = None,
) -> int:
    """Total timesteps across a segment given the step schedule."""
    if schedule_total_alternations is None:
        schedule_total_alternations = total_alternations
    schedule_total_alternations = max(
        schedule_total_alternations, total_alternations + max(0, offset)
    )
    total = 0
    for i in range(total_alternations):
        steps = get_steps_for_alternation(
            i + offset,
            schedule_total_alternations,
            start_steps,
            end_steps,
            schedule_type,
        )
        total += steps * num_envs * n_steps
    return total


def sample_geometric(indices: List[int], beta: float) -> int:
    """Sample an index with geometric decay weighting (newest highest)."""
    weights = [(1 - beta) * (beta ** (len(indices) - i)) for i in range(1, len(indices) + 1)]
    total = sum(weights)
    probs = [w / total for w in weights]
    return random.choices(indices, weights=probs, k=1)[0]


def resolve_phi_beta_schedule(args, previous_schedule_meta, new_training_timesteps, schedule_mode: str):
    """Compute phi_beta schedule parameters based on args and continuation mode."""
    if (
        args.continue_run_id
        and previous_schedule_meta
        and schedule_mode == "extend"
    ):
        phi_beta_start = previous_schedule_meta.get("phi_beta_start")
        phi_beta_end = previous_schedule_meta.get("phi_beta_end")
        phi_beta_schedule_type = previous_schedule_meta.get(
            "phi_beta_schedule", "exp"
        )
        phi_beta_bump_updates = previous_schedule_meta.get("phi_bump_updates", 0)
        phi_beta_bump_multiplier = previous_schedule_meta.get(
            "phi_bump_multiplier", 1.0
        )
        total_planned_ts = previous_schedule_meta.get(
            "total_planned_timesteps", new_training_timesteps
        )
    elif args.continue_run_id and schedule_mode == "constant":
        phi_beta_start = previous_schedule_meta.get("phi_beta_end")
        phi_beta_end = previous_schedule_meta.get("phi_beta_end")
        phi_beta_schedule_type = previous_schedule_meta.get("phi_beta_schedule", "exp")
        phi_beta_bump_updates = 0
        phi_beta_bump_multiplier = 1.0
        total_planned_ts = new_training_timesteps
    elif args.continue_run_id and schedule_mode == "restart":
        phi_beta_start = None
        phi_beta_end = None
        phi_beta_schedule_type = "exp"
        phi_beta_bump_updates = 0
        phi_beta_bump_multiplier = 1.0
        total_planned_ts = new_training_timesteps
    else:
        phi_beta_start = getattr(args, "phi_beta_start", None)
        phi_beta_end = getattr(args, "phi_beta_end", None)
        phi_beta_schedule_type = getattr(args, "phi_beta_schedule", "exp")
        phi_beta_bump_updates = getattr(args, "phi_bump_updates", 0)
        phi_beta_bump_multiplier = getattr(args, "phi_bump_multiplier", 1.0)
        total_planned_ts = new_training_timesteps

    return (
        phi_beta_start,
        phi_beta_end,
        phi_beta_schedule_type,
        phi_beta_bump_updates,
        phi_beta_bump_multiplier,
        total_planned_ts,
    )


def resolve_spa_schedule(args, schedule_mode: str, previous_schedule_meta, base_alt_idx: int):
    """Resolve steps-per-alternation start/end/schedule and continuation offset."""
    if args.continue_run_id and previous_schedule_meta:
        # Use prior schedule metadata if present
        spa_start = previous_schedule_meta.get("spa_start", args.steps_per_alternation)
        spa_end = previous_schedule_meta.get(
            "spa_end", args.steps_per_alternation_end or args.steps_per_alternation
        )
        spa_schedule = previous_schedule_meta.get("spa_schedule", args.steps_per_alternation_schedule)
    else:
        spa_start = args.steps_per_alternation
        spa_end = args.steps_per_alternation_end if args.steps_per_alternation_end is not None else args.steps_per_alternation
        spa_schedule = args.steps_per_alternation_schedule

    spa_offset = 0
    spa_total_alternations = args.alternations
    if args.continue_run_id and schedule_mode == "extend":
        spa_offset = max(0, base_alt_idx)
        spa_total_alternations = spa_offset + args.alternations

    # If continuing and schedule_mode=constant, lock to last known steps
    if args.continue_run_id and schedule_mode == "constant":
        spa_start = spa_end
    return spa_start, spa_end, spa_schedule, spa_offset, spa_total_alternations
