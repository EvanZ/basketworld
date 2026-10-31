"""Role-attributed live-state shot opportunity/contest diagnostics.

Final axes are [learner attacking, learner defending] and additive counters.
Summaries share identical masks for uncontested/defended ESV and suppression.
"""

from __future__ import annotations

import numpy as np

from basketworld_jax.env.minimal import (
    GAME_PHASE_LIVE,
    _active_offense_ids_batch,
    _active_defense_ids_batch,
    _hex_distance,
    _lookup_cell_indices,
    build_shot_profile_batch,
)

PRESSURE_SHAPE = (2, 10)
PRESSURE_ROLES = ("learner_attacking", "learner_defending")


def build_pressure_diagnostics(static, state, active, jnp):
    offense = _active_offense_ids_batch(static, state, jnp)
    defense = _active_defense_ids_batch(static, state, jnp)
    rows = jnp.arange(state.positions.shape[0])[:, None]
    _, on_court = _lookup_cell_indices(static.cell_coords, state.positions, jnp)
    offensive_on_court = on_court[rows, offense]
    defensive_on_court = on_court[rows, defense]
    live = (
        active.astype(jnp.bool_)
        & ~state.episode_ended.astype(jnp.bool_)
        & (state.game_phase == GAME_PHASE_LIVE)
    )
    count = jnp.sum(offensive_on_court, axis=1)
    valid_live = live & (count > 0)
    holder = jnp.clip(state.ball_holder, 0, state.positions.shape[1] - 1)
    valid_holder = (
        valid_live
        & (state.ball_holder >= 0)
        & on_court[jnp.arange(len(holder)), holder]
        & jnp.any(offense == state.ball_holder[:, None], axis=1)
    )
    learner_attacking = static.training_player_mask[offense[:, 0]] > 0.5
    profile = build_shot_profile_batch(static, state, jnp)
    raw = profile["shot_value"] * profile["base_probability"]
    defended = profile["shot_value"] * profile["probability"]
    distances = _hex_distance(
        state.positions[rows, offense][:, :, None, :],
        state.positions[rows, defense][:, None, :, :],
        jnp,
    )
    nearest = jnp.min(
        jnp.where(
            defensive_on_court[:, None, :], distances.astype(jnp.float32), jnp.inf
        ),
        axis=-1,
    )
    # No on-court defender has no finite distance; omit distance/coverage sample.
    distance_live = valid_live & jnp.any(defensive_on_court, axis=1)
    nearest = jnp.where(jnp.isfinite(nearest), nearest, 0.0)
    holder_slot = jnp.argmax(offense == state.ball_holder[:, None], axis=1)
    holder_distance = nearest[jnp.arange(len(holder)), holder_slot]

    def offense_mean(values):
        return jnp.sum(
            jnp.where(offensive_on_court, values, 0.0), axis=1
        ) / jnp.maximum(count, 1)

    def masked(value, valid):
        return jnp.where(valid, value, 0.0)

    # Real live games always have on-court defenders. Keep the sample definition
    # explicit for malformed/imported states rather than emitting infinity.
    valid_live = distance_live
    valid_holder = valid_holder & distance_live
    values = jnp.stack(
        [
            valid_live.astype(jnp.float32),
            valid_holder.astype(jnp.float32),
            masked(offense_mean(nearest), valid_live),
            masked(holder_distance, valid_holder),
            masked(
                (holder_distance <= static.defender_pressure_distance).astype(
                    jnp.float32
                ),
                valid_holder,
            ),
            masked(offense_mean((nearest > 2).astype(jnp.float32)), valid_live),
            masked(raw[jnp.arange(len(holder)), holder], valid_holder),
            masked(defended[jnp.arange(len(holder)), holder], valid_holder),
            masked(offense_mean(raw[rows, offense]), valid_live),
            masked(offense_mean(defended[rows, offense]), valid_live),
        ],
        axis=-1,
    )
    role_mask = jnp.stack([learner_attacking, ~learner_attacking], axis=-1)
    return values[:, None, :] * role_mask[:, :, None]


def summarize_pressure_diagnostics(values):
    if values is None:
        return {}
    values = np.asarray(values, dtype=np.float64)
    if values.shape[-2:] != PRESSURE_SHAPE:
        return {}  # Legacy synthetic traces/totals have no diagnostics.
    totals = values.sum(axis=tuple(range(values.ndim - 2)))
    metrics = {}
    for role, row in zip(PRESSURE_ROLES, totals, strict=True):
        prefix = f"pressure/{role}/"
        live, holder = row[:2]
        metrics[prefix + "live_samples"] = float(live)
        metrics[prefix + "ball_handler_samples"] = float(holder)
        for name, numerator, denominator in (
            ("mean_offense_nearest_defender_distance", row[2], live),
            ("mean_ball_handler_nearest_defender_distance", row[3], holder),
            ("ball_handler_pressure_rate", row[4], holder),
            ("unguarded_offense_fraction", row[5], live),
        ):
            if denominator > 0:
                metrics[prefix + name] = float(numerator / denominator)
        for subject, raw, defended, denominator in (
            ("ball_handler", row[6], row[7], holder),
            ("offense", row[8], row[9], live),
        ):
            if denominator > 0:
                for name, value in (
                    ("uncontested_esv", raw),
                    ("defended_esv", defended),
                    ("esv_reduction", raw - defended),
                ):
                    metrics[prefix + f"mean_{subject}_{name}"] = float(
                        value / denominator
                    )
    return metrics
