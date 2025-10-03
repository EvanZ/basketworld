"""
Utilities for saving and restoring schedule state across training runs.

This module enables continuing training with schedules (entropy, phi-beta, etc.)
that pick up where they left off rather than restarting from scratch.
"""

from typing import Dict, Optional, Any
import mlflow


def save_schedule_metadata(
    ent_coef_start: Optional[float] = None,
    ent_coef_end: Optional[float] = None,
    ent_schedule: str = "linear",
    ent_bump_updates: int = 0,
    ent_bump_multiplier: float = 1.0,
    phi_beta_start: Optional[float] = None,
    phi_beta_end: Optional[float] = None,
    phi_beta_schedule: str = "exp",
    phi_bump_updates: int = 0,
    phi_bump_multiplier: float = 1.0,
    pass_logit_bias_start: Optional[float] = None,
    pass_logit_bias_end: Optional[float] = None,
    pass_arc_start: Optional[float] = None,
    pass_arc_end: Optional[float] = None,
    pass_oob_turnover_prob_start: Optional[float] = None,
    pass_oob_turnover_prob_end: Optional[float] = None,
    total_planned_timesteps: int = 0,
    current_timesteps: int = 0,
) -> None:
    """
    Save schedule configuration and state to MLflow for later continuation.

    This allows a resumed run to reconstruct schedules with the correct
    total_planned_timesteps and current progress.

    Args:
        ent_coef_start: Starting entropy coefficient (None if not scheduled)
        ent_coef_end: Ending entropy coefficient
        ent_schedule: Schedule type ("linear" or "exp")
        ent_bump_updates: Number of updates for entropy bump
        ent_bump_multiplier: Multiplier for entropy bump
        phi_beta_start: Starting phi beta value
        phi_beta_end: Ending phi beta value
        phi_beta_schedule: Schedule type for phi beta
        phi_bump_updates: Number of updates for phi beta bump
        phi_bump_multiplier: Multiplier for phi beta bump
        pass_logit_bias_start: Starting pass logit bias
        pass_logit_bias_end: Ending pass logit bias
        pass_arc_start: Starting pass arc degrees
        pass_arc_end: Ending pass arc degrees
        pass_oob_turnover_prob_start: Starting OOB turnover probability
        pass_oob_turnover_prob_end: Ending OOB turnover probability
        total_planned_timesteps: Total timesteps planned for this run
        current_timesteps: Current timesteps completed (from model.num_timesteps)
    """
    metadata: Dict[str, Any] = {
        "schedule_total_planned_timesteps": total_planned_timesteps,
        "schedule_current_timesteps": current_timesteps,
    }

    # Entropy schedule metadata
    if ent_coef_start is not None:
        metadata.update(
            {
                "schedule_ent_coef_start": ent_coef_start,
                "schedule_ent_coef_end": (
                    ent_coef_end if ent_coef_end is not None else 0.0
                ),
                "schedule_ent_schedule": ent_schedule,
                "schedule_ent_bump_updates": ent_bump_updates,
                "schedule_ent_bump_multiplier": ent_bump_multiplier,
            }
        )

    # Phi beta schedule metadata
    if phi_beta_start is not None:
        metadata.update(
            {
                "schedule_phi_beta_start": phi_beta_start,
                "schedule_phi_beta_end": (
                    phi_beta_end if phi_beta_end is not None else 0.0
                ),
                "schedule_phi_beta_schedule": phi_beta_schedule,
                "schedule_phi_bump_updates": phi_bump_updates,
                "schedule_phi_bump_multiplier": phi_bump_multiplier,
            }
        )

    # Pass logit bias schedule metadata
    if pass_logit_bias_start is not None:
        metadata.update(
            {
                "schedule_pass_logit_bias_start": pass_logit_bias_start,
                "schedule_pass_logit_bias_end": (
                    pass_logit_bias_end if pass_logit_bias_end is not None else 0.0
                ),
            }
        )

    # Pass curriculum schedule metadata
    if pass_arc_start is not None:
        metadata.update(
            {
                "schedule_pass_arc_start": pass_arc_start,
                "schedule_pass_arc_end": (
                    pass_arc_end if pass_arc_end is not None else 60.0
                ),
            }
        )

    if pass_oob_turnover_prob_start is not None:
        metadata.update(
            {
                "schedule_pass_oob_turnover_prob_start": pass_oob_turnover_prob_start,
                "schedule_pass_oob_turnover_prob_end": (
                    pass_oob_turnover_prob_end
                    if pass_oob_turnover_prob_end is not None
                    else 1.0
                ),
            }
        )

    # Log to MLflow
    mlflow.log_params(metadata)


def load_schedule_metadata(
    client: mlflow.tracking.MlflowClient, run_id: str
) -> Dict[str, Any]:
    """
    Load schedule configuration from a previous MLflow run.

    Args:
        client: MLflow client
        run_id: Run ID to load from

    Returns:
        Dictionary containing schedule metadata
    """
    run = client.get_run(run_id)
    params = run.data.params

    metadata = {}

    # Helper to safely extract parameters
    def get_param(key: str, cast_fn=float, default=None):
        if key in params:
            try:
                return cast_fn(params[key])
            except (ValueError, TypeError):
                return default
        return default

    # Load timestep information
    metadata["total_planned_timesteps"] = get_param(
        "schedule_total_planned_timesteps", int, 0
    )
    metadata["current_timesteps"] = get_param("schedule_current_timesteps", int, 0)

    # Load entropy schedule
    metadata["ent_coef_start"] = get_param("schedule_ent_coef_start")
    metadata["ent_coef_end"] = get_param("schedule_ent_coef_end")
    metadata["ent_schedule"] = get_param("schedule_ent_schedule", str, "linear")
    metadata["ent_bump_updates"] = get_param("schedule_ent_bump_updates", int, 0)
    metadata["ent_bump_multiplier"] = get_param(
        "schedule_ent_bump_multiplier", float, 1.0
    )

    # Load phi beta schedule
    metadata["phi_beta_start"] = get_param("schedule_phi_beta_start")
    metadata["phi_beta_end"] = get_param("schedule_phi_beta_end")
    metadata["phi_beta_schedule"] = get_param("schedule_phi_beta_schedule", str, "exp")
    metadata["phi_bump_updates"] = get_param("schedule_phi_bump_updates", int, 0)
    metadata["phi_bump_multiplier"] = get_param(
        "schedule_phi_bump_multiplier", float, 1.0
    )

    # Load pass logit bias schedule
    metadata["pass_logit_bias_start"] = get_param("schedule_pass_logit_bias_start")
    metadata["pass_logit_bias_end"] = get_param("schedule_pass_logit_bias_end")

    # Load pass curriculum schedule
    metadata["pass_arc_start"] = get_param("schedule_pass_arc_start")
    metadata["pass_arc_end"] = get_param("schedule_pass_arc_end")
    metadata["pass_oob_turnover_prob_start"] = get_param(
        "schedule_pass_oob_turnover_prob_start"
    )
    metadata["pass_oob_turnover_prob_end"] = get_param(
        "schedule_pass_oob_turnover_prob_end"
    )

    return metadata


def calculate_continued_total_timesteps(
    original_total: int,
    original_current: int,
    new_alternations: int,
    steps_per_alternation: int,
    num_envs: int,
    n_steps: int,
) -> int:
    """
    Calculate the new total_planned_timesteps when continuing training.

    This adds the new training duration to the original total, allowing
    schedules to continue smoothly.

    Args:
        original_total: Total timesteps planned in original run
        original_current: Timesteps completed in original run
        new_alternations: Number of alternations in continuation
        steps_per_alternation: Steps per alternation
        num_envs: Number of environments
        n_steps: Steps per rollout

    Returns:
        New total timesteps for schedule calculation
    """
    new_timesteps = int(
        2 * new_alternations * steps_per_alternation * num_envs * n_steps
    )
    return original_total + new_timesteps
