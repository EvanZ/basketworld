import os
import sys
import tempfile
from typing import Optional, List

import numpy as np
import mlflow
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger, HumanOutputFormat

from basketworld.utils.callbacks import (
    RolloutUpdateTimingCallback,
    MLflowCallback,
    AccumulativeMetricsCallback,
    EntropyScheduleCallback,
    EntropyExpScheduleCallback,
    PotentialBetaExpScheduleCallback,
    PassLogitBiasExpScheduleCallback,
    PassCurriculumExpScheduleCallback,
    EpisodeSampleLogger,
)
from basketworld.utils.mlflow_logger import MLflowWriter


def build_timing_callbacks():
    """Return offense/defense timing callbacks."""
    return RolloutUpdateTimingCallback(), RolloutUpdateTimingCallback()


def build_entropy_callback(args, total_planned_ts, timestep_offset):
    """Build entropy schedule callback if requested."""
    entropy_callback: Optional[BaseCallback] = None
    if args.ent_coef_start is not None or args.ent_coef_end is not None:
        start = args.ent_coef_start if args.ent_coef_start is not None else args.ent_coef
        end = args.ent_coef_end if args.ent_coef_end is not None else start
        if args.ent_schedule == "exp":
            entropy_callback = EntropyExpScheduleCallback(
                start,
                end,
                total_planned_timesteps=total_planned_ts,
                bump_updates=args.ent_bump_updates or args.ent_bump_rollouts,
                bump_multiplier=args.ent_bump_multiplier,
                timestep_offset=timestep_offset,
            )
        else:
            entropy_callback = EntropyScheduleCallback(
                ent_coef_start=start,
                ent_coef_end=end,
                ent_coef_bump_updates=args.ent_bump_updates or args.ent_bump_rollouts,
                ent_coef_bump_multiplier=args.ent_bump_multiplier,
                total_timesteps=total_planned_ts,
                timestep_offset=timestep_offset,
            )
    return entropy_callback


def build_beta_callback(args, total_planned_ts, timestep_offset):
    """Build phi beta schedule callback if requested."""
    if (args.phi_beta_start is None) and (args.phi_beta_end is None):
        return None
    start = args.phi_beta_start if args.phi_beta_start is not None else 0.0
    end = args.phi_beta_end if args.phi_beta_end is not None else start
    return PotentialBetaExpScheduleCallback(
        start,
        end,
        total_planned_timesteps=total_planned_ts,
        bump_updates=args.phi_bump_updates,
        bump_multiplier=args.phi_bump_multiplier,
        timestep_offset=timestep_offset,
    )


def build_pass_bias_callback(args, total_planned_ts, timestep_offset):
    """Build pass logit bias schedule if requested."""
    if not args.pass_logit_bias_enabled:
        return None
    return PassLogitBiasExpScheduleCallback(
        args.pass_logit_bias_start,
        args.pass_logit_bias_end,
        total_planned_timesteps=total_planned_ts,
        timestep_offset=timestep_offset,
    )


def build_pass_curriculum_callback(args, total_planned_ts, timestep_offset):
    """Build pass arc/OOB curriculum callback if requested."""
    if not (
        getattr(args, "pass_arc_end", None) is not None
        or getattr(args, "pass_arc_start", None) is not None
        or getattr(args, "pass_oob_turnover_prob_start", None) is not None
        or getattr(args, "pass_oob_turnover_prob_end", None) is not None
    ):
        return None
    arc_start = (
        args.pass_arc_start
        if getattr(args, "pass_arc_start", None) is not None
        else 60.0
    )
    arc_end = (
        args.pass_arc_end
        if getattr(args, "pass_arc_end", None) is not None
        else 60.0
    )
    oob_start = (
        args.pass_oob_turnover_prob_start
        if getattr(args, "pass_oob_turnover_prob_start", None) is not None
        else 1.0
    )
    oob_end = (
        args.pass_oob_turnover_prob_end
        if getattr(args, "pass_oob_turnover_prob_end", None) is not None
        else 1.0
    )
    arc_power = args.pass_arc_power if getattr(args, "pass_arc_power", None) is not None else 2.0
    oob_power = args.pass_oob_power if getattr(args, "pass_oob_power", None) is not None else 2.0
    return PassCurriculumExpScheduleCallback(
        arc_start,
        arc_end,
        oob_start,
        oob_end,
        total_planned_ts,
        arc_power=arc_power,
        oob_power=oob_power,
        timestep_offset=timestep_offset,
    )


def build_mixed_callbacks(
    args,
    global_alt: int,
    offense_timing_cb: BaseCallback,
    entropy_cb: Optional[BaseCallback],
    beta_cb: Optional[BaseCallback],
    pass_bias_cb: Optional[BaseCallback],
    pass_curriculum_cb: Optional[BaseCallback],
) -> List[BaseCallback]:
    """Assemble callbacks for mixed training."""
    callbacks: List[BaseCallback] = [
        AccumulativeMetricsCallback(),
        offense_timing_cb,
    ]
    if entropy_cb is not None:
        callbacks.append(entropy_cb)
    if beta_cb is not None:
        callbacks.append(beta_cb)
    if pass_bias_cb is not None:
        callbacks.append(pass_bias_cb)
    if pass_curriculum_cb is not None:
        callbacks.append(pass_curriculum_cb)
    if args.episode_sample_prob > 0.0 and args.log_episode_artifacts:
        callbacks.append(
            EpisodeSampleLogger(
                team_name="Offense",
                alternation_id=global_alt,
                sample_prob=args.episode_sample_prob,
            )
        )
        callbacks.append(
            EpisodeSampleLogger(
                team_name="Defense",
                alternation_id=global_alt,
                sample_prob=args.episode_sample_prob,
            )
        )
    return callbacks


def build_mixed_logger():
    """Logger for mixed training (stdout + MLflow)."""
    return Logger(
        folder=None,
        output_formats=[HumanOutputFormat(sys.stdout), MLflowWriter("Mixed")],
    )


def log_opponent_mapping(opponents, global_alt: int):
    """Log which opponent(s) were used this alternation."""
    try:
        with tempfile.TemporaryDirectory() as _tmp_note_dir:
            note_path = os.path.join(
                _tmp_note_dir, f"opponent_alt_{global_alt}.txt"
            )
            with open(note_path, "w") as f:
                if isinstance(opponents, list):
                    f.write("Per-environment opponent sampling (geometric distribution):\n")
                    f.write("\nEnvironment-to-Opponent Mapping:\n")
                    from collections import Counter
                    policy_counts = Counter(
                        os.path.basename(str(p)) for p in opponents
                    )
                    for env_idx, policy_path in enumerate(opponents):
                        f.write(f"  Env {env_idx:2d}: {os.path.basename(str(policy_path))}\n")
                    f.write(f"\nSummary:\n")
                    f.write(f"  Total environments: {len(opponents)}\n")
                    f.write(f"  Unique policies: {len(policy_counts)}\n")
                    f.write(f"\nPolicy Usage Counts:\n")
                    for policy_name, count in sorted(
                        policy_counts.items(), key=lambda x: x[1], reverse=True
                    ):
                        f.write(f"  {policy_name}: used by {count} env(s)\n")
                else:
                    f.write("Single policy for all environments:\n")
                    f.write(f"  {os.path.basename(str(opponents))}\n")
            mlflow.log_artifact(note_path, artifact_path="opponents")
    except Exception:
        pass
