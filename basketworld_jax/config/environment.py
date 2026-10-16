"""JAX-owned environment configuration and dependency-neutral court setup."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import math
from os import PathLike
from typing import Any, Mapping

import numpy as np

from basketworld.domain import Team
from basketworld import geometry


HEX_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
)


def _team(value: Team | str | int) -> Team:
    if isinstance(value, Team):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"defense", "def", "team_b", "b", "1"}:
            return Team.DEFENSE
        if normalized in {"offense", "off", "team_a", "a", "0"}:
            return Team.OFFENSE
    try:
        return Team(int(value))
    except (TypeError, ValueError):
        raise ValueError(f"training_team has unsupported value {value!r}.") from None


@dataclass(frozen=True)
class JaxEnvironmentConfig:
    """All inputs needed to build JAX static state and reset an episode."""

    grid_size: int = 16
    court_rows: int | None = None
    court_cols: int | None = None
    players: int = 3
    shot_clock: int = 24
    min_shot_clock: int = 10
    training_team: Team | str | int = Team.OFFENSE
    seed: int = 0

    defender_pressure_distance: int = 1
    defender_pressure_turnover_chance: float = 0.05
    defender_pressure_decay_lambda: float = 1.0
    pressure_skill_mean: float = 0.0
    pressure_skill_std: float = 0.0
    shot_contest_skill_mean: float = 0.0
    shot_contest_skill_std: float = 0.0
    ball_handling_skill_mean: float = 0.0
    ball_handling_skill_std: float = 0.0
    pressure_skill_effect_weight: float = 0.0
    shot_contest_skill_effect_weight: float = 0.0
    base_steal_rate: float = 0.35
    steal_perp_decay: float = 1.5
    steal_distance_factor: float = 0.08
    steal_position_weight_min: float = 0.3
    pass_interception_model: str = "line"
    pass_passer_pressure_weight: float = 0.0
    pass_receiver_pressure_weight: float = 0.0
    pass_lob_lane_multiplier: float = 0.35
    pass_lob_receiver_distance: float = 1.0
    pass_speed: float = 3.5
    defender_reaction_time: float = 0.35
    defender_speed: float = 1.25
    defender_reach_radius: float = 0.65
    reaction_softness: float = 0.55
    base_passer_risk: float = 0.06
    passer_pressure_decay: float = 1.35
    base_receiver_risk: float = 0.35
    receiver_alignment_min: float = 0.35
    receiver_alignment_width: float = 2.0
    max_receiver_hazard: float = 0.85
    lane_weight: float = 0.0
    steal_skill_mean: float = 0.0
    steal_skill_std: float = 0.0
    passing_skill_mean: float = 0.0
    passing_skill_std: float = 0.0
    steal_skill_effect_weight: float = 0.0

    three_point_distance: float = 4.0
    three_point_short_distance: float | None = None
    three_pt_extra_hex_decay: float = 0.05
    layup_pct: float = 0.6
    three_pt_pct: float = 0.37
    layup_std: float = 0.0
    three_pt_std: float = 0.0
    allow_dunks: bool = False
    dunk_pct: float = 0.9
    dunk_std: float = 0.0
    three_second_lane_width: int = 1
    three_second_lane_height: int = 3
    three_second_max_steps: int = 3
    illegal_defense_enabled: bool = True
    offensive_three_seconds_enabled: bool = False
    shot_pressure_enabled: bool = True
    shot_pressure_max: float = 0.5
    shot_pressure_lambda: float = 1.0
    shot_pressure_arc_degrees: float = 60.0

    spawn_distance: int = 3
    max_spawn_distance: int | None = None
    defender_spawn_distance: int = 0
    defender_guard_distance: int = 1
    offense_spawn_boundary_margin: int = 0
    normalize_obs: bool = True
    mask_occupied_moves: bool = False
    enable_pass_gating: bool = True
    include_hoop_vector: bool = True

    # Backend/UI metadata that is part of the live JAX environment contract.
    # These values do not require a Gym environment, but keeping them in the
    # JAX-owned config lets interactive serialization preserve its payload.
    pass_arc_degrees: float = 60.0
    pass_target_strategy: str = "nearest"
    pass_mode: str = "directional"
    illegal_action_policy: str = "noop"
    intent_obs_mode: str = "private_offense"

    pass_reward: float = 0.0
    violation_reward: float = 1.0
    reward_shaping_gamma: float | None = None
    enable_phi_shaping: bool = False
    phi_beta: float = 0.0
    phi_use_ball_handler_only: bool = False
    phi_blend_weight: float = 0.0
    phi_aggregation_mode: str = "team_best"
    pass_oob_turnover_prob: float = 1.0
    assist_window: int = 2
    potential_assist_pct: float = 0.1
    full_assist_bonus_pct: float = 0.05
    task_reward_scale: float = 1.0

    enable_intent_learning: bool = False
    enable_defense_intent_learning: bool = False
    num_intents: int = 8
    intent_commitment_steps: int = 4
    intent_null_prob: float = 0.2
    defense_intent_null_prob: float = 1.0
    intent_visible_to_defense_prob: float = 0.0

    start_template_enabled: bool = False
    start_template_library: Mapping[str, Any] | None = None
    start_template_prob: float = 0.0
    start_template_jitter_scale: float = 1.0
    start_template_mirror_prob: float = 0.5
    start_template_strict: bool = False

    enable_rebounds: bool = False
    rebound_table_model_dir: str = ""
    rebound_target_temperature: float = 1.0
    rebound_target_uniform_mix: float = 0.0
    rebound_winner_distance_weight: float = 1.0
    rebound_basket_position_weight: float = 0.0
    rebound_winner_temperature: float = 1.0
    rebound_skill_std: float = 0.0
    rebound_skill_sampling_mode: str = "gaussian"
    rebound_skill_high: float = 1.0
    rebound_skill_low: float = -0.25
    rebound_skill_weight: float = 0.0
    rebound_contest_mode: str = "global_contest"
    rebound_contest_radius: int = 1
    rebound_counterfactual_positioning_enabled: bool = False
    rebound_obs_top_n_targets: int = 0
    offensive_rebound_shot_clock_reset: int = 14
    rebound_terminal_reward_mode: str = "actual_points"
    enable_rebound_reward_redistribution: bool = False
    offensive_rebound_reward_advance: float = 0.4
    rebound_reward_once_per_possession: bool = True

    enable_multi_possession: bool = False
    multi_possession_limit: int = 25
    multi_possession_overtime_round_cap: int = 0
    multi_possession_reward_mode: str = "win_loss"
    score_potential_scale: float = 1.0
    game_winner_reward: float = 0.0
    episode_end_mode: str = "possessions"
    game_clock_seconds: int = 300
    overtime_seconds: int = 120
    game_clock_max_total_steps: int = 10_000
    multi_possession_aux_rewards_enabled: bool = False
    multi_possession_schema_version: int | None = None
    multi_possession_use_inbounds: bool = True
    made_basket_restart_mode: str | None = None
    inbound_deadline_steps: int = 5
    check_deadline_steps: int = 5
    check_setup_steps: int = 0
    check_clearance: int = 0
    random_check_location: bool = False

    def __post_init__(self) -> None:
        positive = {
            "grid_size": self.grid_size,
            "court_rows": self.court_rows if self.court_rows is not None else self.grid_size,
            "court_cols": self.court_cols if self.court_cols is not None else self.grid_size,
            "players": self.players,
            "shot_clock": self.shot_clock,
            "min_shot_clock": self.min_shot_clock,
            "game_clock_seconds": self.game_clock_seconds,
            "overtime_seconds": self.overtime_seconds,
            "game_clock_max_total_steps": self.game_clock_max_total_steps,
        }
        for name, value in positive.items():
            if int(value) < 1:
                raise ValueError(f"{name} must be at least 1.")
        if int(self.min_shot_clock) > int(self.shot_clock):
            raise ValueError("min_shot_clock must not exceed shot_clock.")
        mode = str(self.episode_end_mode or "possessions").strip().lower()
        if mode not in {"possessions", "game_clock"}:
            raise ValueError("episode_end_mode must be 'possessions' or 'game_clock'.")
        if mode == "game_clock" and not bool(self.enable_multi_possession):
            raise ValueError("game_clock episode mode requires enable_multi_possession.")
        if int(self.check_clearance) < 0:
            raise ValueError("check_clearance must be >= 0.")
        if int(self.check_clearance) > 0 and int(self.check_setup_steps) < 1:
            raise ValueError("check_clearance requires at least one check_setup_steps tick.")
        _team(self.training_team)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "JaxEnvironmentConfig":
        raw = dict(values or {})
        aliases = {
            "court_height": "court_rows",
            "court_width": "court_cols",
            "players_per_side": "players",
            "shot_clock_steps": "shot_clock",
            "offensive_three_seconds": "offensive_three_seconds_enabled",
            "phi_beta_start": "phi_beta",
            "pass_oob_turnover_prob_start": "pass_oob_turnover_prob",
        }
        for old, new in aliases.items():
            if new not in raw and old in raw:
                raw[new] = raw[old]
        library = raw.get("start_template_library")
        if isinstance(library, (str, PathLike)) and str(library).strip():
            from basketworld.utils.start_templates import load_start_template_library

            raw["start_template_library"] = load_start_template_library(
                str(library),
                players_per_side=int(raw.get("players", 3)),
            )
        allowed = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in raw.items() if key in allowed})

    @classmethod
    def from_args(cls, args: Any) -> "JaxEnvironmentConfig":
        raw = dict(vars(args) if hasattr(args, "__dict__") else args)
        if raw.get("reward_shaping_gamma") is None and raw.get("gamma") is not None:
            raw["reward_shaping_gamma"] = raw["gamma"]
        return cls.from_mapping(raw)

    @classmethod
    def from_checkpoint_metadata(
        cls, metadata: Mapping[str, Any]
    ) -> "JaxEnvironmentConfig":
        raw = dict(metadata or {})
        merged: dict[str, Any] = {}
        nested = False
        for key in ("frozen_config", "trainer_config", "env_config"):
            value = raw.get(key)
            if isinstance(value, Mapping):
                merged.update(value)
                nested = True
        if not nested:
            merged.update(raw)
        return cls.from_mapping(merged)

    @classmethod
    def from_evaluation_request(
        cls,
        required_params: Mapping[str, Any],
        optional_params: Mapping[str, Any] | None = None,
        *,
        checkpoint_metadata: Mapping[str, Any] | None = None,
    ) -> "JaxEnvironmentConfig":
        merged: dict[str, Any] = {}
        if checkpoint_metadata:
            merged.update(asdict(cls.from_checkpoint_metadata(checkpoint_metadata)))
        merged.update(required_params or {})
        merged.update(optional_params or {})
        return cls.from_mapping(merged)

    @classmethod
    def from_dev_game_params(
        cls,
        required_params: Mapping[str, Any],
        env_params: Mapping[str, Any] | None = None,
    ) -> "JaxEnvironmentConfig":
        return cls.from_evaluation_request(required_params, env_params)

    @classmethod
    def from_legacy_env(cls, env: Any) -> "JaxEnvironmentConfig":
        raw = {
            field.name: getattr(env, field.name)
            for field in fields(cls)
            if hasattr(env, field.name)
        }
        raw.update(
            court_rows=getattr(env, "court_height"),
            court_cols=getattr(env, "court_width"),
            players=getattr(env, "players_per_side"),
            shot_clock=getattr(env, "shot_clock_steps"),
        )
        return cls.from_mapping(raw)

    def as_metadata(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["training_team"] = _team(self.training_team).name.lower()
        return payload


class JaxStaticEnvironment:
    """Pure-Python geometry/config source consumed by the JAX static builder."""

    def __init__(self, config: JaxEnvironmentConfig):
        self.config = config
        self.court_width = int(config.court_cols or config.grid_size)
        self.court_height = int(config.court_rows or config.grid_size)
        self.players_per_side = int(config.players)
        self.n_players = self.players_per_side * 2
        self.offense_ids = list(range(self.players_per_side))
        self.defense_ids = list(range(self.players_per_side, self.n_players))
        self.training_team = _team(config.training_team)
        self.shot_clock_steps = int(config.shot_clock)
        self.shot_pressure_arc_rad = math.radians(float(config.shot_pressure_arc_degrees))
        self.reward_shaping_gamma = (
            1.0 if config.reward_shaping_gamma is None else float(config.reward_shaping_gamma)
        )
        self.multi_possession_schema_version = (
            int(config.multi_possession_schema_version)
            if config.multi_possession_schema_version is not None
            else (5 if int(config.check_clearance) > 0 else 4)
            if bool(config.enable_multi_possession)
            else 1
        )
        self.made_basket_restart_mode = config.made_basket_restart_mode or (
            "baseline_inbound"
            if bool(config.multi_possession_use_inbounds)
            else "direct_handoff"
        )
        self.hex_directions = list(HEX_DIRECTIONS)
        self._rng = np.random.default_rng(int(config.seed))

        cells = [
            geometry.offset_to_axial_formula(col, row)
            for row in range(self.court_height)
            for col in range(self.court_width)
        ]
        self._valid_axial = set(cells)
        self._cell_index = {cell: index for index, cell in enumerate(cells)}
        self.basket_position = geometry.offset_to_axial_formula(0, self.court_height // 2)
        self._move_mask_by_cell: dict[tuple[int, int], np.ndarray] = {}
        for cell in cells:
            allowed = np.ones(len(self.hex_directions), dtype=np.int8)
            for index, direction in enumerate(self.hex_directions):
                neighbor = (cell[0] + direction[0], cell[1] + direction[1])
                if neighbor not in self._valid_axial or (
                    neighbor == self.basket_position and not bool(config.allow_dunks)
                ):
                    allowed[index] = 0
            self._move_mask_by_cell[cell] = allowed

        self._three_point_hexes, self._three_point_line_hexes, self._three_point_outline_points = (
            geometry.compute_three_point_geometry(self)
        )
        lane = self._calculate_lane_hexes()
        self.offensive_lane_hexes = lane if config.offensive_three_seconds_enabled else set()
        self.defensive_lane_hexes = lane if config.illegal_defense_enabled else set()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.config, name)

    @staticmethod
    def _offset_to_axial(col: int, row: int) -> tuple[int, int]:
        return geometry.offset_to_axial_formula(col, row)

    @staticmethod
    def _axial_to_offset(q: int, r: int) -> tuple[int, int]:
        return geometry.axial_to_offset_formula(q, r)

    @staticmethod
    def _axial_to_cartesian(q: int, r: int) -> tuple[float, float]:
        return geometry.axial_to_cartesian_formula(q, r)

    @staticmethod
    def _hex_distance(first: tuple[int, int], second: tuple[int, int]) -> int:
        return geometry.hex_distance_formula(first[0], first[1], second[0], second[1])

    def _calculate_lane_hexes(self) -> set[tuple[int, int]]:
        lane: set[tuple[int, int]] = set()
        _, basket_row = self._axial_to_offset(*self.basket_position)
        for col in range(self.court_width):
            for row in range(self.court_height):
                cell = self._offset_to_axial(col, row)
                if self._hex_distance(cell, self.basket_position) >= int(
                    self.config.three_second_lane_height
                ):
                    continue
                if abs(row - basket_row) <= int(self.config.three_second_lane_width):
                    lane.add(cell)
        return lane


def build_static_environment(config: JaxEnvironmentConfig) -> JaxStaticEnvironment:
    return JaxStaticEnvironment(config)


def normalize_runtime_environment_config(
    config: JaxEnvironmentConfig,
) -> JaxEnvironmentConfig:
    """Map supported additive checkpoint schemas to the current JAX runtime."""
    if not bool(config.enable_multi_possession):
        return config
    runtime_schema = 5 if int(config.check_clearance) > 0 else 4
    saved_schema = config.multi_possession_schema_version
    compatible = {5} if runtime_schema == 5 else {2, 3, 4}
    if saved_schema is not None and int(saved_schema) not in compatible:
        raise ValueError(
            "Unsupported JAX environment observation schema "
            f"{saved_schema}; runtime schema {runtime_schema} accepts "
            f"{sorted(compatible)}."
        )
    return replace(config, multi_possession_schema_version=runtime_schema)


__all__ = [
    "HEX_DIRECTIONS",
    "JaxEnvironmentConfig",
    "JaxStaticEnvironment",
    "build_static_environment",
    "normalize_runtime_environment_config",
]
