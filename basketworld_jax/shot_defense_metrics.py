"""Shot-release counters, with shooter-side denominators and defensive blocks."""

from __future__ import annotations

import numpy as np


SHOT_DEFENSE_COUNTERS = (
    "attempts",
    "makes",
    "layup_attempts",
    "layup_makes",
    "dunk_attempts",
    "dunk_makes",
    "open_finish_attempts",
    "open_finish_makes",
    "contested_finish_attempts",
    "contested_finish_makes",
    "open_jump_attempts",
    "open_jump_makes",
    "contested_jump_attempts",
    "contested_jump_makes",
    "rim_opportunities",
    "blocked",
    "blocked_offensive_recoveries",
    "blocked_defensive_recoveries",
    "open_probability_sum",
    "conditional_probability_sum",
    "block_probability_sum",
    "marginal_probability_sum",
    "jump_suppression_before_sum",
    "jump_suppression_after_sum",
    "jump_attempts",
    "open_layup_attempts",
    "open_layup_makes",
    "contested_layup_attempts",
    "contested_layup_makes",
    "open_dunk_attempts",
    "open_dunk_makes",
    "contested_dunk_attempts",
    "contested_dunk_makes",
    "completed_possessions",
    "rim_contest_probability_drop_sum",
)
SHOT_DEFENSE_SHAPE = (2, len(SHOT_DEFENSE_COUNTERS))


def build_shot_defense_diagnostics(static, out, active, xp):
    attempt = active & out.shot_attempt.astype(bool)
    made = attempt & out.shot_success.astype(bool)
    zero = xp.zeros_like(out.shot_attempt)
    finish = attempt & getattr(out, "shot_is_finish", zero).astype(bool)
    contested = attempt & getattr(out, "shot_contested", zero).astype(bool)
    blocked = attempt & getattr(out, "shot_blocked", zero).astype(bool)
    # Use the shared enum rather than interpreting all two-pointers as finishes.
    from basketworld_jax.env.minimal import SHOT_TYPE_DUNK

    dunk = attempt & (out.shot_type == SHOT_TYPE_DUNK)
    jump = attempt & ~finish
    masks = (
        attempt,
        made,
        finish & ~dunk,
        made & finish & ~dunk,
        dunk,
        made & dunk,
        finish & ~contested,
        made & finish & ~contested,
        finish & contested,
        made & finish & contested,
        jump & ~contested,
        made & jump & ~contested,
        jump & contested,
        made & jump & contested,
        finish & contested & static.rim_protection_enabled.astype(bool),
        blocked,
        blocked & out.offensive_rebound.astype(bool),
        blocked & out.defensive_rebound.astype(bool),
    )
    values = [value.astype(xp.float32) for value in masks]
    for field in (
        "shot_open_probability",
        "shot_conditional_probability",
        "shot_block_probability",
    ):
        values.append(xp.where(attempt, getattr(out, field, zero), 0.0))
    values.append(
        xp.where(
            attempt,
            getattr(out, "shot_conditional_probability", zero)
            * (1.0 - getattr(out, "shot_block_probability", zero)),
            0.0,
        )
    )
    for field in (
        "shot_suppression_before_resistance",
        "shot_suppression_after_resistance",
    ):
        values.append(xp.where(jump, getattr(out, field, zero), 0.0))
    values.append(jump.astype(xp.float32))
    for category in (
        finish & ~dunk & ~contested,
        finish & ~dunk & contested,
        dunk & ~contested,
        dunk & contested,
    ):
        values.extend(
            [category.astype(xp.float32), (made & category).astype(xp.float32)]
        )
    values.append(xp.zeros_like(attempt, dtype=xp.float32))
    rim_contested = finish & contested & static.rim_protection_enabled.astype(bool)
    # Conditional make probability excludes the independent block probability.
    # Include blocked attempts too, so block outcomes cannot bias this average.
    values.append(xp.where(
        rim_contested,
        xp.maximum(0.0, out.shot_open_probability - out.shot_conditional_probability),
        0.0,
    ))
    counters = xp.stack(values, axis=-1)
    shooter = xp.clip(out.shot_shooter, 0, static.training_player_mask.shape[0] - 1)
    learner_shot = static.training_player_mask[shooter] > 0.5
    sides = xp.stack([learner_shot, ~learner_shot], axis=-1).astype(xp.float32)
    result = sides[..., :, None] * counters[..., None, :]
    # A shared denominator counts each completed possession exactly once.
    return result.at[..., 0, SHOT_DEFENSE_COUNTERS.index("completed_possessions")].set(
        (active & out.possession_ended.astype(bool)).astype(xp.float32)
    )


def summarize_shot_defense_diagnostics(diagnostics):
    if diagnostics is None:
        return {}
    raw = np.asarray(diagnostics, dtype=np.float64)
    # Historical diagnostic arrays predate the appended contest counter.
    counter_names = SHOT_DEFENSE_COUNTERS
    if raw.shape[-1] == len(SHOT_DEFENSE_COUNTERS) - 1:
        counter_names = SHOT_DEFENSE_COUNTERS[:-1]
    array = raw.reshape(-1, 2, len(counter_names)).sum(axis=0)
    output = {}
    rows = {"learner": array[0], "opponent": array[1], "all": array.sum(axis=0)}
    for side, row in rows.items():
        counters = dict(zip(counter_names, map(float, row), strict=True))
        prefix = f"{side}_shot_defense"
        output.update({f"{prefix}_{key}": value for key, value in counters.items()})
        for label in (
            "layup",
            "dunk",
            "open_finish",
            "contested_finish",
            "open_jump",
            "contested_jump",
            "open_layup",
            "contested_layup",
            "open_dunk",
            "contested_dunk",
        ):
            denominator = counters[f"{label}_attempts"]
            if denominator:
                output[f"{prefix}_{label}_make_rate"] = (
                    counters[f"{label}_makes"] / denominator
                )
        for name, numerator, denominator in (
            ("block_rate_per_eligible_attempt", "blocked", "rim_opportunities"),
            ("blocked_rate_per_shot", "blocked", "attempts"),
            (
                "mean_block_probability_per_eligible_attempt",
                "block_probability_sum",
                "rim_opportunities",
            ),
            (
                "blocked_offensive_recovery_rate",
                "blocked_offensive_recoveries",
                "blocked",
            ),
            (
                "blocked_defensive_recovery_rate",
                "blocked_defensive_recoveries",
                "blocked",
            ),
            ("mean_open_probability", "open_probability_sum", "attempts"),
            ("mean_conditional_probability", "conditional_probability_sum", "attempts"),
            ("mean_marginal_probability", "marginal_probability_sum", "attempts"),
            (
                "mean_jump_suppression_before",
                "jump_suppression_before_sum",
                "jump_attempts",
            ),
            (
                "mean_jump_suppression_after",
                "jump_suppression_after_sum",
                "jump_attempts",
            ),
        ):
            if counters[denominator]:
                output[f"{prefix}_{name}"] = counters[numerator] / counters[denominator]
    blocked_index = SHOT_DEFENSE_COUNTERS.index("blocked")
    output["learner_blocks"] = float(array[1, blocked_index])
    output["opponent_blocks"] = float(array[0, blocked_index])
    output["blocks"] = output["learner_blocks"] + output["opponent_blocks"]
    possessions = float(array[:, SHOT_DEFENSE_COUNTERS.index("completed_possessions")].sum())
    if possessions:
        output["blocks_per_completed_possession"] = output["blocks"] / possessions
    return output
