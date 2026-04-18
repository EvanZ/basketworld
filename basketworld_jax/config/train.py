from __future__ import annotations

from typing import Any


# Historical reduced-scope reference values. This keeps the older
# directional-passing baseline visible, but it is not the canonical config for
# the current JAX trainer/runtime.
DIRECTIONAL_REFERENCE_VALUES: dict[str, Any] = {
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


# Canonical JAX trainer/runtime config. Despite the name "pointer_targeted",
# this does not imply an attention-based pointer network. The current reduced
# JAX stack uses a flat MLP head with fixed pass slots that map to teammate
# target IDs.
TRAIN_FROZEN_VALUES: dict[str, Any] = {
    **DIRECTIONAL_REFERENCE_VALUES,
    "pass_mode": "pointer_targeted",
    "use_set_obs": False,
    "training_team": "offense",
    "enable_phi_shaping": False,
    "illegal_defense_enabled": False,
    "offensive_three_seconds": False,
    "include_hoop_vector": True,
    "phi_aggregation_mode": "team_best",
    "phi_use_ball_handler_only": False,
}
