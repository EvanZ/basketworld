# basketworld_env_v2.py
"""
Hexagon-tessellated basketball environment for reinforcement learning.
Features alternating policy optimization (self-play) on a hexagonal grid.

Key Features:
- Hexagonal grid (16x16 default, parameterized)
- N vs N players (2v2, 3v3, 5x5 configurable)
- 24-second shot clock
- Simultaneous turn-based actions
- Collision resolution
- Pass/shot probability mechanics
- Self-play training support
"""

from __future__ import annotations

import random
import math
from typing import Callable, Dict, List, Tuple, Optional, Union, Set
from enum import Enum
from collections import defaultdict

# Use a non-interactive backend so rendering works in headless/threaded contexts
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from basketworld.utils.evaluation_helpers import profile_section
from basketworld.envs.core import geometry
from basketworld.envs.core import movement as movement_core
from basketworld.envs.core import passing as passing_core
from basketworld.envs.core import shooting as shooting_core
from basketworld.envs.core import rewards as rewards_core
from basketworld.envs.core import state as state_core
from basketworld.envs.core import rendering as rendering_core


class Team(Enum):
    OFFENSE = 0
    DEFENSE = 1


class ActionType(Enum):
    NOOP = 0
    MOVE_E = 1
    MOVE_NE = 2
    MOVE_NW = 3
    MOVE_W = 4
    MOVE_SW = 5
    MOVE_SE = 6
    SHOOT = 7
    PASS_E = 8
    PASS_NE = 9
    PASS_NW = 10
    PASS_W = 11
    PASS_SW = 12
    PASS_SE = 13


class IllegalActionPolicy(Enum):
    """Strategy for resolving illegal actions supplied to env.step()."""

    NOOP = "noop"  # Map illegal actions to NOOP
    BEST_PROB = "best_prob"  # Use highest-probability legal action
    SAMPLE_PROB = "sample_prob"  # Sample among legal actions using probabilities


class HexagonBasketballEnv(gym.Env):
    """Hexagon-tessellated basketball environment for self-play RL."""

    # Small tolerance to treat pass-arc boundary angles as in-arc (avoids floating error gaps)
    _PASS_ARC_COS_EPS = 1e-9

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self,
        grid_size: int = 16,
        court_rows: int | None = None,
        court_cols: int | None = None,
        players: int | None = None,
        players_per_side: int | None = None,
        shot_clock: int | None = None,
        shot_clock_steps: int = 24,
        min_shot_clock: int = 10,
        training_team: Team = Team.OFFENSE,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        defender_pressure_distance: int = 1,
        defender_pressure_turnover_chance: float = 0.05,
        defender_pressure_decay_lambda: float = 1.0,  # Exponential decay rate for pressure
        # Realistic passing steal parameters
        base_steal_rate: float = 0.35,
        steal_perp_decay: float = 1.5,
        steal_distance_factor: float = 0.08,
        steal_position_weight_min: float = 0.3,
        three_point_distance: float = 4.0,
        three_point_short_distance: Optional[float] = None,
        layup_pct: float = 0.60,
        three_pt_pct: float = 0.37,
        # Baseline shooting variability (per-player, sampled each episode)
        layup_std: float = 0.0,
        three_pt_std: float = 0.0,
        # Dunk controls
        allow_dunks: bool = False,
        dunk_pct: float = 0.90,
        dunk_std: float = 0.0,
        # 3-second violation controls (shared lane configuration)
        three_second_lane_width: int = 1,
        three_second_lane_height: int = 3,
        three_second_max_steps: int = 3,
        # Illegal defense (3-in-the-key) controls
        illegal_defense_enabled: bool = True,
        # Offensive 3-second violation controls
        offensive_three_seconds_enabled: bool = False,
        # Shot pressure parameters
        shot_pressure_enabled: bool = True,
        shot_pressure_max: float = 0.5,  # max reduction at distance=1 (multiplier = 1 - max)
        shot_pressure_lambda: float = 1.0,  # decay rate per hex away from shooter
        shot_pressure_arc_degrees: float = 60.0,  # arc width centered toward basket
        # Pass/OOB curriculum controls
        pass_arc_degrees: float = 60.0,
        pass_oob_turnover_prob: float = 1.0,
        pass_target_strategy: str = "nearest",
        enable_profiling: bool = False,
        profiling_sample_rate: float = 1.0,  # Fraction of episodes to profile (0.0-1.0)
        spawn_distance: int = 3,
        max_spawn_distance: Optional[int] = None,
        defender_spawn_distance: int = 0,
        defender_guard_distance: int = 1,
        # Reward shaping parameters
        pass_reward: float = 0.0,
        turnover_penalty: float = 0.0,
        made_shot_reward_inside: float = 2.0,
        made_shot_reward_three: float = 3.0,
        missed_shot_penalty: float = 0.0,
        potential_assist_reward: float = 0.1,
        full_assist_bonus: float = 0.2,
        violation_reward: float = 1.0,
        # Potential-based shaping (Phi) controls
        enable_phi_shaping: bool = False,
        reward_shaping_gamma: Optional[float] = None,
        phi_beta: float = 0.0,
        phi_use_ball_handler_only: bool = False,
        phi_blend_weight: float = 0.0,
        phi_aggregation_mode: str = "team_best",  # "team_best", "teammates_best", "teammates_avg", "team_avg"
        # Preferred: percentage-based assist shaping
        potential_assist_pct: float = 0.10,
        full_assist_bonus_pct: float = 0.05,
        # Observation controls
        use_egocentric_obs: bool = True,
        egocentric_rotate_to_hoop: bool = True,
        include_hoop_vector: bool = True,
        normalize_obs: bool = True,
        # Movement mask controls
        mask_occupied_moves: bool = False,
        # Pass gating: mask out passes without teammates in arc
        enable_pass_gating: bool = True,
        # Deterministic overrides (optional)
        initial_positions: Optional[List[Tuple[int, int]]] = None,
        initial_ball_holder: Optional[int] = None,
        fixed_shot_clock: Optional[int] = None,
        assist_window: int = 2,
        # Illegal action resolution policy (see IllegalActionPolicy)
        illegal_action_policy: str = "noop",
        # If True, raise a clear error when an illegal action is passed to step()
        raise_on_illegal_action: bool = False,
    ):
        super().__init__()

        # Initialize caches early (used by coordinate helpers before full precompute)
        self._offset_to_axial_cache: list[list[Tuple[int, int]]] = []
        self._axial_to_offset_cache: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self._valid_axial: set[Tuple[int, int]] = set()
        self._axial_to_cart_cache: Dict[Tuple[int, int], Tuple[float, float]] = {}
        self._hex_distance_lut = None
        self._cell_index: Dict[Tuple[int, int], int] = {}

        self.grid_size = grid_size
        self.court_width = int(court_cols) if court_cols is not None else int(grid_size * 1.0)
        self.court_height = int(court_rows) if court_rows is not None else grid_size
        # Prefer new canonical name `players`; fall back to legacy `players_per_side`
        if players is None and players_per_side is not None:
            players = players_per_side
        if players is None:
            players = 3
        self.players_per_side = int(players)
        # Prefer new canonical name `shot_clock`; fall back to legacy `shot_clock_steps`
        if shot_clock is None:
            shot_clock = shot_clock_steps
        self.shot_clock_steps = int(shot_clock)
        # Minimum starting value when randomly initializing the shot clock at reset
        self.min_shot_clock = int(min_shot_clock)
        self.training_team = training_team  # Which team is currently training
        self.render_mode = render_mode
        self.defender_pressure_distance = defender_pressure_distance
        self.defender_pressure_turnover_chance = defender_pressure_turnover_chance
        self.defender_pressure_decay_lambda = defender_pressure_decay_lambda
        # Realistic passing steal parameters
        self.base_steal_rate = base_steal_rate
        self.steal_perp_decay = steal_perp_decay
        self.steal_distance_factor = steal_distance_factor
        self.steal_position_weight_min = steal_position_weight_min
        self.spawn_distance = spawn_distance
        self.max_spawn_distance = max_spawn_distance
        self.defender_spawn_distance = defender_spawn_distance
        self.defender_guard_distance = max(0, int(defender_guard_distance))
        self.use_egocentric_obs = bool(use_egocentric_obs)
        self.egocentric_rotate_to_hoop = bool(egocentric_rotate_to_hoop)
        self.include_hoop_vector = bool(include_hoop_vector)
        self.normalize_obs = bool(normalize_obs)
        # Movement mask behavior
        self.mask_occupied_moves = bool(mask_occupied_moves)
        # Pass gating: mask out passes without teammates in arc
        self.enable_pass_gating = bool(enable_pass_gating)
        # Illegal action handling configuration/state
        try:
            self.illegal_action_policy = IllegalActionPolicy(illegal_action_policy)
        except Exception:
            self.illegal_action_policy = IllegalActionPolicy.NOOP
        # Optional per-step action probabilities provided by caller
        self._pending_action_probs: Optional[np.ndarray] = None
        self._legal_actions_offense: float = 0.0
        self._legal_actions_defense: float = 0.0
        # Honor constructor flag for strict illegal action handling
        self.raise_on_illegal_action = bool(raise_on_illegal_action)
        # Three-point configuration and shot model parameters
        self.three_point_distance = float(three_point_distance)
        self.three_point_short_distance = (
            float(three_point_short_distance)
            if three_point_short_distance is not None
            else None
        )
        self.layup_pct = float(layup_pct)
        self.three_pt_pct = float(three_pt_pct)
        # Std deviations for per-player variability (sampled each episode)
        self.layup_std = float(layup_std)
        self.three_pt_std = float(three_pt_std)
        # Dunk configuration
        self.allow_dunks = bool(allow_dunks)
        self.dunk_pct = float(dunk_pct)
        self.dunk_std = float(dunk_std)
        # Defender shot pressure
        self.shot_pressure_enabled = bool(shot_pressure_enabled)
        self.shot_pressure_max = float(shot_pressure_max)
        self.shot_pressure_lambda = float(shot_pressure_lambda)
        self.shot_pressure_arc_degrees = float(shot_pressure_arc_degrees)
        self.shot_pressure_arc_rad = math.radians(shot_pressure_arc_degrees)
        # Passing arc (degrees) centered on chosen pass direction; schedulable
        self.pass_arc_degrees = float(max(1.0, min(360.0, pass_arc_degrees)))
        # How to pick the receiver when multiple teammates are in the arc
        self.pass_target_strategy = (
            str(pass_target_strategy).lower()
            if str(pass_target_strategy).lower() in ("nearest", "best_ev")
            else "nearest"
        )
        # Probability an attempted pass with no receiver in arc is ruled OOB turnover
        # (1.0 retains prior behavior). Can be annealed by scheduler.
        self.pass_oob_turnover_prob = float(max(0.0, min(1.0, pass_oob_turnover_prob)))
        # Profiling
        self.enable_profiling = bool(enable_profiling)
        self.profiling_sample_rate = float(max(0.0, min(1.0, profiling_sample_rate)))  # Clamp to [0, 1]
        self._profiling_this_episode: bool = False  # Set per episode in reset()
        self._profile_ns: Dict[str, int] = {}
        self._profile_calls: Dict[str, int] = {}
        # Basket position, using offset coordinates for placement
        basket_col = 0
        basket_row = self.court_height // 2
        self.basket_position = self._offset_to_axial(basket_col, basket_row)
        self._three_point_hexes: Set[Tuple[int, int]] = set()
        self._three_point_line_hexes: Set[Tuple[int, int]] = set()
        self._three_point_outline_points: List[Tuple[float, float]] = []
        self._compute_three_point_geometry()
        # Shared 3-second violation configuration (used by both offense and defense)
        self.three_second_lane_width = int(three_second_lane_width)
        self.three_second_lane_height = int(three_second_lane_height)
        self.three_second_max_steps = int(three_second_max_steps)
        
        # Illegal defense (defensive 3-second) configuration
        self.illegal_defense_enabled = bool(illegal_defense_enabled)
        
        # Offensive 3-second violation configuration
        self.offensive_three_seconds_enabled = bool(offensive_three_seconds_enabled)
        
        # Total players
        self.n_players = self.players_per_side * 2
        self.offense_ids = list(range(self.players_per_side))
        self.defense_ids = list(range(self.players_per_side, self.n_players))

        # Assist window
        self.assist_window = int(assist_window)

        # Action space: each player can take one of 9 actions
        self.action_space = spaces.MultiDiscrete([len(ActionType)] * self.n_players)

        # Define the two parts of our observation space
        # Observation length depends on configuration flags
        # Role flag moved to separate observation key `role_flag` (Box shape=(1,))
        # +players_per_side for per-offense nearest-defender distances
        # +n_players for lane step counts (both offensive and defensive players)
        # +players_per_side for expected points (EP) for each offensive player
        # +players_per_side for turnover probability (one per offensive player, 0 if not ball holder)
        # +players_per_side for steal risks (one per offensive player, 0 for ball holder)
        # Player skills moved to separate observation key `skills` (shape=(players_per_side*3,))
        # Note: Using fixed-position encoding (one slot per player) instead of dynamic ordering
        # for better learning - position i always corresponds to offensive player i
        # Base components of observation vector
        base_len = (self.n_players * 2) + self.n_players + 1  # Player positions, ball holder one-hot, shot clock
        team_encoding_extra = self.n_players  # Per-player team encoding (+1 offense, -1 defense)
        ball_handler_pos_extra = 2  # Absolute position of ball handler (NEW - helps distinguish court regions)
        hoop_extra = 2 if self.include_hoop_vector else 0
        # All-pairs offense-defense distances and angles (replaces nearest defender distances)
        offense_defense_distances = self.players_per_side * self.players_per_side
        offense_defense_angles = self.players_per_side * self.players_per_side
        lane_steps_extra = self.n_players  # Lane violation counters for all players
        ep_extra = self.players_per_side  # EP for each offensive player
        turnover_risk_extra = self.players_per_side  # Turnover prob per offensive player (0 if not ball holder)
        steal_risk_extra = self.players_per_side  # Steal risk per offensive player (0 for ball holder)
        teammate_distance_pairs = (self.players_per_side * (self.players_per_side - 1)) // 2
        teammate_angle_pairs = self.players_per_side * (self.players_per_side - 1)
        teammate_distance_extra = 2 * teammate_distance_pairs
        teammate_angle_extra = 2 * teammate_angle_pairs
        state_vector_length = (
            base_len
            + team_encoding_extra
            + ball_handler_pos_extra
            + hoop_extra
            + offense_defense_distances
            + offense_defense_angles
            + lane_steps_extra
            + ep_extra
            + turnover_risk_extra
            + steal_risk_extra
            + teammate_distance_extra
            + teammate_angle_extra
        )
        state_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_vector_length,),
            dtype=np.float32,
        )
        action_mask_space = spaces.Box(
            low=0, high=1, shape=(self.n_players, len(ActionType)), dtype=np.int8
        )

        # The full observation space is a dictionary containing the state and the mask
        # role_flag: -1.0 = defense, +1.0 = offense (symmetric encoding)
        role_flag_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        skills_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.players_per_side * 3,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                "obs": state_space,
                "action_mask": action_mask_space,
                "role_flag": role_flag_space,
                "skills": skills_space,
            }
        )

        # --- Hexagonal Grid Directions ---
        # These are the 6 axial direction vectors for a pointy-topped hexagonal grid.
        # In our (q, r) axial system:
        # - Moving E/W changes only the q-axis.
        # - Moving NW/SE changes only the r-axis.
        # - Moving NE/SW changes both q and r axes.
        # The previous vectors were incorrect, causing bugs in movement and passing.
        # These vectors correctly map ActionType enums to their corresponding axial changes.
        self.hex_directions = [
            (+1, 0),  # E:  Move one hex to the right.
            (+1, -1),  # NE: Move diagonally up-right.
            (0, -1),  # NW: Move diagonally up-left.
            (-1, 0),  # W:  Move one hex to the left.
            (-1, +1),  # SW: Move diagonally down-left.
            (0, +1),  # SE: Move diagonally down-right.
        ]

        self._rng = np.random.default_rng(seed)
        self._offset_to_axial_cache: list[list[Tuple[int, int]]] = []
        self._axial_to_offset_cache: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self._valid_axial: set[Tuple[int, int]] = set()
        self._axial_to_cart_cache: Dict[Tuple[int, int], Tuple[float, float]] = {}

        # Game state
        self.positions: List[Tuple[int, int]] = []  # (q, r) axial coordinates
        self.ball_holder: int = 0
        self.shot_clock: int = 0
        self.step_count: int = 0
        self.episode_ended: bool = False
        self.last_action_results: Dict = {}
        # Track consecutive defender steps in the basket (key) cell
        self._defender_in_key_steps: Dict[int, int] = {}
        # Assist tracking: window after a successful pass where a shot counts as assisted
        # Structure: { 'passer_id': int, 'recipient_id': int, 'expires_at_step': int }
        self._assist_candidate: Optional[Dict[str, int]] = None

        # Precompute per-cell move validity mask (6 directions) to speed up action mask building
        # 1 = allowed, 0 = blocked (OOB or basket hex if dunks disabled)
        self._move_mask_by_cell: Dict[Tuple[int, int], np.ndarray] = {}
        all_cells: List[Tuple[int, int]] = []
        for row in range(self.court_height):
            for col in range(self.court_width):
                cell = self._offset_to_axial(col, row)
                all_cells.append(cell)
                allowed = np.ones(6, dtype=np.int8)
                for dir_idx in range(6):
                    nbr = (
                        cell[0] + self.hex_directions[dir_idx][0],
                        cell[1] + self.hex_directions[dir_idx][1],
                    )
                    if (not self._is_valid_position(*nbr)) or (
                        (nbr == self.basket_position) and (not self.allow_dunks)
                    ):
                        allowed[dir_idx] = 0
                self._move_mask_by_cell[cell] = allowed
        self._precompute_coord_caches(all_cells)
        self._precompute_hex_distance_lut(all_cells)

        # Precompute shoot/pass action indices
        self._shoot_pass_action_indices = [ActionType.SHOOT.value] + [
            a.value for a in ActionType if "PASS" in a.name
        ]
        
        # Calculate lane hexes for offensive and defensive 3-second rules
        if self.offensive_three_seconds_enabled:
            self.offensive_lane_hexes = self._calculate_offensive_lane_hexes()
        else:
            self.offensive_lane_hexes = set()
        
        if self.illegal_defense_enabled:
            self.defensive_lane_hexes = self._calculate_defensive_lane_hexes()
        else:
            self.defensive_lane_hexes = set()

        # --- Reward parameters (stored on env for evaluation compatibility) ---
        self.pass_reward: float = float(pass_reward)
        self.turnover_penalty: float = float(turnover_penalty)
        self.violation_reward: float = float(violation_reward)
        self.made_shot_reward_inside: float = float(made_shot_reward_inside)
        self.made_shot_reward_three: float = float(made_shot_reward_three)
        self.missed_shot_penalty: float = float(missed_shot_penalty)
        # Absolute assist rewards (legacy; prefer percentage fields below)
        self.potential_assist_reward: float = float(potential_assist_reward)
        self.full_assist_bonus: float = float(full_assist_bonus)
        # Percentage-based assist rewards (preferred)
        self.potential_assist_pct: float = float(potential_assist_pct)
        self.full_assist_bonus_pct: float = float(full_assist_bonus_pct)

        # --- Potential-based shaping (Phi) configuration ---
        self.enable_phi_shaping: bool = bool(enable_phi_shaping)
        # Must match the agent's discount to preserve policy invariance
        self.reward_shaping_gamma: float = (
            float(reward_shaping_gamma) if reward_shaping_gamma is not None else 1.0
        )
        # Beta can be scheduled during training via VecEnv.set_attr
        self.phi_beta: float = float(phi_beta)
        # If True, use only the ball handler's make prob; else best among offense
        self.phi_use_ball_handler_only: bool = bool(phi_use_ball_handler_only)
        # How to aggregate teammate EPs when not using ball-handler-only mode
        self.phi_aggregation_mode: str = str(phi_aggregation_mode)
        try:
            self.phi_blend_weight: float = float(max(0.0, min(1.0, phi_blend_weight)))
        except Exception:
            self.phi_blend_weight = 0.0

        # Cache for phi value to avoid redundant calculations (telescoping)
        self._cached_phi: Optional[float] = None

        # Optional deterministic start overrides
        self._initial_positions_override: Optional[List[Tuple[int, int]]] = None
        if initial_positions is not None:
            # Shallow copy to avoid accidental external mutation
            self._initial_positions_override = [
                (int(p[0]), int(p[1])) for p in initial_positions
            ]
        self._initial_ball_holder_override: Optional[int] = initial_ball_holder
        self._fixed_shot_clock: Optional[int] = fixed_shot_clock

        # Per-episode, per-offensive-player baseline shooting percentages
        # Initialized to means; re-sampled each reset.
        self.offense_layup_pct_by_player: List[float] = [
            float(self.layup_pct)
        ] * self.players_per_side
        self.offense_three_pt_pct_by_player: List[float] = [
            float(self.three_pt_pct)
        ] * self.players_per_side
        self.offense_dunk_pct_by_player: List[float] = [
            float(self.dunk_pct)
        ] * self.players_per_side

    @staticmethod
    def _offset_to_axial_formula(col: int, row: int) -> Tuple[int, int]:
        """Pure conversion odd-r offset -> axial."""
        return geometry.offset_to_axial_formula(col, row)

    @staticmethod
    def _axial_to_offset_formula(q: int, r: int) -> Tuple[int, int]:
        """Pure conversion axial -> odd-r offset."""
        return geometry.axial_to_offset_formula(q, r)

    @staticmethod
    def _axial_to_cartesian_formula(q: int, r: int) -> Tuple[float, float]:
        """Convert axial (q, r) to cartesian (x, y) matching rendering geometry."""
        return geometry.axial_to_cartesian_formula(q, r)

    @profile_section("_offset_to_axial")
    def _offset_to_axial(self, col: int, row: int) -> Tuple[int, int]:
        """Converts odd-r offset coordinates to axial coordinates."""
        cache = getattr(self, "_offset_to_axial_cache", None)
        if cache and 0 <= row < len(cache) and 0 <= col < len(cache[row]):
            return cache[row][col]
        return self._offset_to_axial_formula(col, row)

    @profile_section("_axial_to_offset")
    def _axial_to_offset(self, q: int, r: int) -> Tuple[int, int]:
        """Converts axial coordinates to odd-r offset coordinates."""
        cached = self._axial_to_offset_cache.get((q, r))
        if cached is not None:
            return cached
        return self._axial_to_offset_formula(q, r)

    @profile_section("_axial_to_cartesian")
    def _axial_to_cartesian(self, q: int, r: int) -> Tuple[float, float]:
        """Convert axial (q, r) to cartesian (x, y) matching rendering geometry."""
        cached = self._axial_to_cart_cache.get((q, r))
        if cached is not None:
            return cached
        return self._axial_to_cartesian_formula(q, r)

    def _precompute_coord_caches(self, cells: List[Tuple[int, int]]) -> None:
        """Precompute common coordinate conversions for all on-court cells."""
        (
            self._offset_to_axial_cache,
            self._axial_to_offset_cache,
            self._axial_to_cart_cache,
            self._valid_axial,
        ) = geometry.precompute_coord_caches(self.court_width, self.court_height, cells)

    @profile_section("_axial_to_cube")
    def _axial_to_cube(self, q: int, r: int) -> Tuple[int, int, int]:
        """Convert axial (q, r) to cube (x, y, z) coordinates."""
        return geometry.axial_to_cube(q, r)

    @profile_section("_cube_to_axial")
    def _cube_to_axial(self, x: int, y: int, z: int) -> Tuple[int, int]:
        """Convert cube (x, y, z) to axial (q, r) coordinates."""
        return geometry.cube_to_axial(x, y, z)

    @profile_section("_rotate60_cw_cube")
    def _rotate60_cw_cube(self, x: int, y: int, z: int) -> Tuple[int, int, int]:
        """Rotate cube (x, y, z) by 60 degrees clockwise."""
        return geometry.rotate60_cw_cube(x, y, z)

    @profile_section("_rotate_k60_axial")
    def _rotate_k60_axial(self, q: int, r: int, k: int) -> Tuple[int, int]:
        """Rotate axial (q, r) by k*60 degrees clockwise."""
        return geometry.rotate_k60_axial(q, r, k)

    @staticmethod
    def _hex_distance_formula(q1: int, r1: int, q2: int, r2: int) -> int:
        """Closed-form hex distance on axial coords."""
        return geometry.hex_distance_formula(q1, r1, q2, r2)

    @profile_section("_hex_distance")
    def _hex_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate distance between two hexagon positions."""
        if self._hex_distance_lut is not None:
            idx1 = self._cell_index.get(pos1)
            idx2 = self._cell_index.get(pos2)
            if idx1 is not None and idx2 is not None:
                return int(self._hex_distance_lut[idx1, idx2])
        q1, r1 = pos1
        q2, r2 = pos2
        return self._hex_distance_formula(q1, r1, q2, r2)

    def _precompute_hex_distance_lut(self, cells: List[Tuple[int, int]]) -> None:
        """Precompute hex distances between all on-court cells for fast lookup."""
        lut, cell_index = geometry.precompute_hex_distance_lut(cells)
        self._hex_distance_lut = lut
        self._cell_index = cell_index

    @profile_section("_defender_is_guarding_offense")
    def _defender_is_guarding_offense(self, defender_id: int) -> bool:
        """Return True if any offensive player is within guard distance of the defender."""
        return movement_core.defender_is_guarding_offense(self, defender_id)

    @profile_section("_point_to_line_distance")
    def _point_to_line_distance(
        self,
        point: Tuple[int, int],
        line_start: Tuple[int, int],
        line_end: Tuple[int, int]
    ) -> float:
        """
        Calculate perpendicular distance from a point to a line segment.
        
        Args:
            point: The point to measure distance from (axial coords)
            line_start: Start of line segment (axial coords)
            line_end: End of line segment (axial coords)
            
        Returns:
            Perpendicular distance in Cartesian space
        """
        return geometry.point_to_line_distance(point, line_start, line_end)

    @profile_section("_get_position_on_line")
    def _get_position_on_line(
        self,
        point: Tuple[int, int],
        line_start: Tuple[int, int],
        line_end: Tuple[int, int]
    ) -> float:
        """
        Get the position parameter t of a point's projection onto a line.
        
        Args:
            point: The point to project (axial coords)
            line_start: Start of line segment (axial coords)
            line_end: End of line segment (axial coords)
            
        Returns:
            Position parameter t where:
            - t = 0.0 means projection is at line_start
            - t = 1.0 means projection is at line_end
            - 0 < t < 1 means projection is between start and end
        """
        return geometry.get_position_on_line(point, line_start, line_end)

    @profile_section("_is_between_points")
    def _is_between_points(
        self,
        point: Tuple[int, int],
        line_start: Tuple[int, int],
        line_end: Tuple[int, int]
    ) -> bool:
        """
        Check if a point's projection onto the line falls between start and end.
        
        Args:
            point: The point to check (axial coords)
            line_start: Start of line segment (axial coords)
            line_end: End of line segment (axial coords)
            
        Returns:
            True if point projects onto the line segment (not just the infinite line)
        """
        return geometry.is_between_points(point, line_start, line_end)

    @profile_section("reset")
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment to initial state."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        
        # Decide whether to profile this episode (reduces overhead for large-scale training)
        if self.enable_profiling:
            self._profiling_this_episode = (self._rng.random() < self.profiling_sample_rate)
        else:
            self._profiling_this_episode = False
        
        # Resolve overrides from options (takes precedence over ctor overrides)
        opt_positions = None
        opt_ball_holder = None
        opt_shot_clock = None
        if options:
            opt_positions = options.get("initial_positions")
            opt_ball_holder = options.get("ball_holder")
            opt_shot_clock = options.get("shot_clock")

        # Shot clock setup: option > ctor fixed > random
        if opt_shot_clock is not None:
            self.shot_clock = int(opt_shot_clock)
        elif self._fixed_shot_clock is not None:
            self.shot_clock = int(self._fixed_shot_clock)
        else:
            # Sample uniformly between min and max (inclusive)
            min_sc = int(self.min_shot_clock)
            max_sc = int(self.shot_clock_steps)
            if max_sc < min_sc:
                min_sc, max_sc = max_sc, min_sc
            # numpy Generator.integers samples [low, high) → add 1 to include max
            self.shot_clock = int(self._rng.integers(min_sc, max_sc + 1))
        self.step_count = 0
        self.episode_ended = False
        self.last_action_results = {}
        # Track steps in lane for both offensive and defensive players
        self._defender_in_key_steps = {pid: 0 for pid in range(self.n_players)}
        self._offensive_lane_steps = {pid: 0 for pid in range(self.n_players)}
        # Initialize scores (for tracking defensive violations that award points)
        self.offense_score = 0
        self.defense_score = 0
        self._assist_candidate = None
        # Clear any pending external probabilities on reset
        self._pending_action_probs = None
        # Track first step after reset for phi shaping (Φ(s₀) = 0)
        self._first_step_after_reset = True
        # Initialize cached phi value (Φ(s₀) = 0 for PBRS)
        self._cached_phi = 0.0

        # Sample per-player baseline shooting percentages for OFFENSE for this episode
        # Use truncated normal around means with stds, clamped to [0.01, 0.99]
        # Allow override via options to maintain consistent skills across resets
        opt_skills = options.get("offense_skills") if options else None
        
        if opt_skills is not None:
            # Use provided skills (for replay/self-play consistency)
            for i in range(self.players_per_side):
                self.offense_layup_pct_by_player[i] = float(opt_skills["layup"][i])
                self.offense_three_pt_pct_by_player[i] = float(opt_skills["three_pt"][i])
                self.offense_dunk_pct_by_player[i] = float(opt_skills["dunk"][i])
        else:
            # Sample new skills
            def _sample_prob(mean: float, std: float) -> float:
                if std <= 0.0:
                    return float(mean)
                val = float(self._rng.normal(loc=float(mean), scale=float(std)))
                return float(max(0.01, min(0.99, val)))

            for i in range(self.players_per_side):
                self.offense_layup_pct_by_player[i] = _sample_prob(
                    self.layup_pct, self.layup_std
                )
                self.offense_three_pt_pct_by_player[i] = _sample_prob(
                    self.three_pt_pct, self.three_pt_std
                )
                self.offense_dunk_pct_by_player[i] = _sample_prob(
                    self.dunk_pct, self.dunk_std
                )

        # Initialize positions
        if opt_positions is not None:
            self._set_initial_positions_from_override(opt_positions)
        elif self._initial_positions_override is not None:
            self._set_initial_positions_from_override(self._initial_positions_override)
        else:
            # Offense on right side, defense on left (randomized)
            self.positions = self._generate_initial_positions()

        # Determine ball holder: option > ctor override > random offense
        if opt_ball_holder is not None:
            self._set_initial_ball_holder(int(opt_ball_holder))
        elif self._initial_ball_holder_override is not None:
            self._set_initial_ball_holder(int(self._initial_ball_holder_override))
        else:
            # Random offensive player starts with ball
            self.ball_holder = int(self._rng.choice(self.offense_ids))

        obs = {
            "obs": self._get_observation(),
            "action_mask": self._get_action_masks(),
            "role_flag": np.array(
                [1.0 if self.training_team == Team.OFFENSE else -1.0],
                dtype=np.float32,
            ),
            "skills": self._get_offense_skills_array(),
        }
        mask = obs["action_mask"]
        try:
            offense_mask = mask[self.offense_ids]
            defense_mask = mask[self.defense_ids]
            self._legal_actions_offense = float(np.mean(np.sum(offense_mask, axis=1)))
            self._legal_actions_defense = float(np.mean(np.sum(defense_mask, axis=1)))
        except Exception:
            self._legal_actions_offense = 0.0
            self._legal_actions_defense = 0.0

        info = self._attach_legal_action_stats(
            {"training_team": self.training_team.name}
        )

        return obs, info

    @profile_section("step")
    def step(self, actions: Union[np.ndarray, List[int]]):
        """Execute one step of the environment."""
        # Initialize rewards
        rewards = np.zeros(self.n_players)

        if self.episode_ended:
            raise ValueError("Episode has ended. Call reset() to start a new episode.")

        self.step_count += 1

        # Compute Phi(s) at start of step (for logs and optional shaping)
        # Use cached value from previous step to avoid redundant calculation (telescoping)
        phi_prev: Optional[float] = None
        if self.enable_phi_shaping:
            phi_prev = self._cached_phi  # Use cached value from end of previous step

        # Clear the flag regardless of whether phi shaping is enabled
        if self._first_step_after_reset:
            self._first_step_after_reset = False

        # --- Defender Pressure Mechanic ---
        # Use shared calculation for consistency between gameplay and observations
        defender_pressure_info = self._calculate_defender_pressure_info()
        
        # Variables needed for action results and turnover logging
        ball_handler_pos = self.positions[self.ball_holder] if self.ball_holder is not None else (0, 0)
        total_pressure_prob = self.calculate_defender_pressure_turnover_probability()
        
        if defender_pressure_info:
            # Check for turnovers
            for pressure in defender_pressure_info:
                if self._rng.random() < pressure["turnover_prob"]:
                    # Turnover occurs!
                    defender_id = pressure["defender_id"]
                    turnover_results = {
                        "turnovers": [
                            {
                                "player_id": self.ball_holder,
                                "reason": "defender_pressure",
                                "stolen_by": defender_id,
                                "turnover_pos": ball_handler_pos,
                            }
                        ]
                    }
                    self.last_action_results = turnover_results
                    self.ball_holder = defender_id  # Defender gets the ball

                    done = True
                    self.episode_ended = done

                    obs = {
                        "obs": self._get_observation(),
                        "action_mask": self._get_action_masks(),
                        "role_flag": np.array(
                            [1.0 if self.training_team == Team.OFFENSE else -1.0],
                            dtype=np.float32,
                        ),
                        "skills": self._get_offense_skills_array(),
                    }
                    info = self._attach_legal_action_stats(
                        {
                            "training_team": self.training_team.name,
                            "action_results": turnover_results,
                            "shot_clock": self.shot_clock,
                        }
                    )

                    # Phi diagnostics and optional shaping on early turnover path
                    if self.enable_phi_shaping:
                        # Terminal step → define Phi(s')=0 to preserve policy invariance
                        phi_next_term = 0.0

                        # Calculate phi shaping reward
                        r_shape = float(self.reward_shaping_gamma) * float(
                            phi_next_term
                        ) - float(phi_prev if phi_prev is not None else 0.0)
                        shaped = float(self.phi_beta) * float(r_shape)
                        per_team = shaped / self.players_per_side

                        # Apply to actual rewards
                        rewards[self.offense_ids] += per_team
                        rewards[self.defense_ids] -= per_team

                        # Cache phi_next for next step (terminal, so it's 0.0)
                        self._cached_phi = phi_next_term

                        # Report the calculated value (for UI/logging)
                        info["phi_r_shape"] = float(per_team)
                        info["phi_prev"] = float(
                            phi_prev if phi_prev is not None else 0.0
                        )
                        info["phi_next"] = float(phi_next_term)
                        info["phi_beta"] = float(self.phi_beta)
                    else:
                        # Phi shaping disabled - provide zero values for Monitor compatibility
                        info["phi_r_shape"] = 0.0
                        info["phi_prev"] = 0.0
                        info["phi_next"] = 0.0
                        info["phi_beta"] = 0.0
                        # Per-player EP breakdown for UI
                        try:
                            team_best, ball_ep = self._phi_ep_breakdown()
                            info["phi_team_best_ep"] = float(team_best)
                            info["phi_ball_handler_ep"] = float(ball_ep)
                            # Add per-player EPs for accurate UI recalculation
                            ep_by_player = []
                            for pid in range(self.n_players):
                                pos = self.positions[pid]
                                dist = self._hex_distance(pos, self.basket_position)
                                is_three = self._is_three_point_hex(tuple(pos))
                                shot_value = 2.0
                                if self.allow_dunks and dist == 0:
                                    shot_value = 2.0
                                elif is_three:
                                    shot_value = 3.0
                                p = float(
                                    self._calculate_shot_probability(pid, dist)
                                )
                                ep_by_player.append(float(shot_value * p))
                            info["phi_ep_by_player"] = ep_by_player
                        except Exception:
                            pass

                    return obs, rewards, done, False, info

                # break  # Only check the first defender applying pressure each step

        actions = np.array(actions)
        # Resolve illegal actions according to configured policy
        try:
            masks = self._get_action_masks()
            offense_mask = masks[self.offense_ids]
            defense_mask = masks[self.defense_ids]
            self._legal_actions_offense = float(
                np.mean(np.sum(offense_mask, axis=1))
            )
            self._legal_actions_defense = float(
                np.mean(np.sum(defense_mask, axis=1))
            )
            for i in range(self.n_players):
                a = int(actions[i])
                # Legal and in-bounds → keep
                if 0 <= a < masks.shape[1] and masks[i, a] == 1:
                    continue

                legal = np.where(masks[i] == 1)[0]
                if len(legal) == 0:
                    actions[i] = ActionType.NOOP.value
                    continue

                # Optional strict mode: surface illegal action immediately
                if self.raise_on_illegal_action:
                    raise ValueError(
                        f"Illegal action {a} by player {i}. Legal actions: {legal.tolist()}"
                    )

                if self.illegal_action_policy == IllegalActionPolicy.NOOP:
                    actions[i] = ActionType.NOOP.value
                else:
                    # Use external probabilities if provided
                    probs_vec = None
                    if (
                        self._pending_action_probs is not None
                        and i < self._pending_action_probs.shape[0]
                    ):
                        probs_vec = self._pending_action_probs[i]

                    chosen = None
                    if (
                        probs_vec is not None
                        and len(probs_vec) >= int(np.max(legal)) + 1
                    ):
                        masked = probs_vec[legal]
                        total = float(np.sum(masked))
                        if total > 0.0:
                            if (
                                self.illegal_action_policy
                                == IllegalActionPolicy.BEST_PROB
                            ):
                                chosen = int(legal[int(np.argmax(masked))])
                            elif (
                                self.illegal_action_policy
                                == IllegalActionPolicy.SAMPLE_PROB
                            ):
                                normed = masked / total
                                chosen = int(np.random.choice(legal, p=normed))

                    if chosen is None:
                        non_noop = [
                            int(idx) for idx in legal if idx != ActionType.NOOP.value
                        ]
                        chosen = (
                            int(non_noop[0]) if len(non_noop) > 0 else int(legal[0])
                        )
                    actions[i] = chosen
        except Exception:
            # If masking fails for any reason, proceed without remapping
            pass

        # Decrement shot clock
        self.shot_clock -= 1

        # Process all actions simultaneously
        action_results = self._process_simultaneous_actions(actions)
        self.last_action_results = action_results
        
        # Add defender pressure info to action results
        if defender_pressure_info and self.ball_holder in self.offense_ids:
            action_results["defender_pressure"][self.ball_holder] = {
                "defenders": defender_pressure_info,
                "total_pressure_prob": total_pressure_prob,
            }

        # Update illegal defense counters based on resulting positions
        # Defenders cannot camp in the full lane area (not just basket)
        # Check for defensive 3-second violations BEFORE calculating rewards
        if self.illegal_defense_enabled:
            for did in self.defense_ids:
                if self.positions and tuple(self.positions[did]) in self.defensive_lane_hexes:
                    if self._defender_is_guarding_offense(did):
                        self._defender_in_key_steps[did] = 0
                        continue

                    steps = self._defender_in_key_steps.get(did, 0) + 1
                    self._defender_in_key_steps[did] = steps
                    
                    # If defender exceeds max steps, it's a violation
                    if steps > self.three_second_max_steps:
                        action_results["defensive_lane_violations"].append({
                            "player_id": did,
                            "steps": steps,
                            "position": tuple(self.positions[did]),
                        })
                        # Award offense 1 point (like a technical free throw)
                        self.offense_score += 1
                        # Reset the counter to avoid repeated violations
                        self._defender_in_key_steps[did] = 0
                        break  # Only one violation per step
                else:
                    self._defender_in_key_steps[did] = 0

        # Check for episode termination and calculate rewards (after checking violations)
        done, episode_rewards = self._check_termination_and_rewards(action_results)
        rewards += episode_rewards

        # Check shot clock expiration
        if self.shot_clock <= 0:
            done = True

        self.episode_ended = done
        
        # Note: Offensive lane step counters are now updated in _process_simultaneous_actions
        # immediately after moves are resolved and before violation checks

        # Clear per-step external probabilities after use to avoid reuse
        self._pending_action_probs = None

        obs = {
            "obs": self._get_observation(),
            "action_mask": self._get_action_masks(),
            "role_flag": np.array(
                [1.0 if self.training_team == Team.OFFENSE else -1.0],
                dtype=np.float32,
            ),
            "skills": self._get_offense_skills_array(),
        }
        info = {
            "training_team": self.training_team.name,
            "action_results": action_results,
            "shot_clock": self.shot_clock,
        }

        # Phi diagnostics and shaping (only if enabled)
        if self.enable_phi_shaping:
            # If this is a terminal step, force Phi(s')=0 to preserve policy invariance.
            phi_next = 0.0 if done else 0.0
            if not done:
                try:
                    phi_next = float(self._phi_shot_quality())
                except Exception:
                    phi_next = 0.0

            # Calculate phi shaping reward
            r_shape = float(self.reward_shaping_gamma) * float(phi_next) - float(
                phi_prev if phi_prev is not None else 0.0
            )
            shaped = float(self.phi_beta) * float(r_shape)
            per_team = shaped / self.players_per_side

            # Apply to actual rewards
            rewards[self.offense_ids] += per_team
            rewards[self.defense_ids] -= per_team

            # Cache phi_next for next step (to avoid redundant calculation)
            self._cached_phi = phi_next

            # Report the calculated value (for UI/logging)
            info["phi_r_shape"] = float(per_team)
            info["phi_prev"] = float(phi_prev if phi_prev is not None else 0.0)
            info["phi_next"] = float(phi_next)
            info["phi_beta"] = float(self.phi_beta)
            # Per-player EP breakdown for UI
            try:
                team_best, ball_ep = self._phi_ep_breakdown()
                info["phi_team_best_ep"] = float(team_best)
                info["phi_ball_handler_ep"] = float(ball_ep)
                # Add per-player EPs for accurate UI recalculation
                ep_by_player = []
                for pid in range(self.n_players):
                    pos = self.positions[pid]
                    dist = self._hex_distance(pos, self.basket_position)
                    is_three = self._is_three_point_hex(pos)
                    if self.allow_dunks and dist == 0:
                        shot_value = 2.0
                    else:
                        shot_value = 3.0 if is_three else 2.0
                    p = float(self._calculate_shot_probability(pid, dist))
                    ep_by_player.append(float(shot_value * p))
                info["phi_ep_by_player"] = ep_by_player
            except Exception:
                pass
        else:
            # Phi shaping disabled - provide zero values for Monitor compatibility
            info["phi_r_shape"] = 0.0
            info["phi_prev"] = 0.0
            info["phi_next"] = 0.0
            info["phi_beta"] = 0.0

        info = self._attach_legal_action_stats(info)
        return obs, rewards, done, False, info

    @profile_section("set_illegal_action_probs")
    def set_illegal_action_probs(self, probs: Optional[np.ndarray]) -> None:
        """Provide per-player action probabilities for resolving illegal actions.

        Shape expected: (n_players, n_actions). Only used when
        illegal_action_policy is BEST_PROB or SAMPLE_PROB. Passing None clears
        previously set probabilities.
        """
        if probs is None:
            self._pending_action_probs = None
            return
        try:
            arr = np.asarray(probs)
            self._pending_action_probs = arr
        except Exception:
            self._pending_action_probs = None

    @profile_section("action_masks")
    def _get_action_masks(self) -> np.ndarray:
        """Generate a mask of legal actions for each player."""
        from basketworld.envs.core import actions as actions_core

        return actions_core.build_action_masks(
            n_players=self.n_players,
            positions=self.positions,
            ball_holder=self.ball_holder,
            move_mask_by_cell=self._move_mask_by_cell,
            hex_directions=self.hex_directions,
            mask_occupied_moves=self.mask_occupied_moves,
            enable_pass_gating=self.enable_pass_gating,
            has_teammate_in_pass_arc=self._has_teammate_in_pass_arc,
        )

    @profile_section("_generate_initial_positions")
    def _generate_initial_positions(self) -> List[Tuple[int, int]]:
        """
        Generate initial positions with distances defined RELATIVE to the basket:
        - Offense: any valid cell with distance >= spawn_distance (negative => 0)
                   and distance <= max_spawn_distance (if set)
        - Defense: closer to basket than the matched offense and distance >= spawn_distance (negative => 0)
                   and distance <= max_spawn_distance (if set)
          If no such cells, broaden progressively to avoid spawn failures.
        """
        return state_core.generate_initial_positions(self)

    @profile_section("_set_initial_positions_from_override")
    def _set_initial_positions_from_override(
        self, positions: List[Tuple[int, int]]
    ) -> None:
        """Validate and set fixed initial positions by player index.
        Expects length == n_players. Positions must be on-court and unique.
        Basket cell is allowed only when dunks are enabled.
        """
        if len(positions) != self.n_players:
            raise ValueError(
                f"initial_positions must have length {self.n_players}, got {len(positions)}"
            )
        normalized: List[Tuple[int, int]] = [tuple(p) for p in positions]
        # Validate legality
        seen: set[Tuple[int, int]] = set()
        for pos in normalized:
            if not isinstance(pos, tuple) or len(pos) != 2:
                raise ValueError("Each initial position must be a (q, r) tuple")
            q, r = int(pos[0]), int(pos[1])
            if (q, r) == tuple(self.basket_position) and not self.allow_dunks:
                raise ValueError(
                    "initial position cannot be the basket cell when dunks are disabled"
                )
            if not self._is_valid_position(q, r):
                raise ValueError(f"initial position {(q, r)} is out of bounds")
            if (q, r) in seen:
                raise ValueError(f"duplicate initial position {(q, r)}")
            seen.add((q, r))
        self.positions = normalized

    @profile_section("_set_initial_ball_holder")
    def _set_initial_ball_holder(self, player_id: int) -> None:
        if not (0 <= player_id < self.n_players):
            raise ValueError(
                f"ball_holder must be in [0, {self.n_players - 1}], got {player_id}"
            )
        self.ball_holder = int(player_id)

    @profile_section("_is_valid_position")
    def _is_valid_position(self, q: int, r: int) -> bool:
        """Check if a hexagon position is within the rectangular court bounds."""
        if self._valid_axial:
            return (q, r) in self._valid_axial
        col, row = self._axial_to_offset_formula(q, r)
        return 0 <= col < self.court_width and 0 <= row < self.court_height
    
    @profile_section("_calculate_offensive_lane_hexes")
    def _calculate_offensive_lane_hexes(self) -> set:
        """Calculate the hexes that make up the offensive lane (painted area).
        
        The lane extends from the basket along the +q axis (toward offensive side)
        up to (but not including) the 3-point line distance.
        The lane has symmetric width on both sides.
        
        The lane is defined by two criteria:
        1. Hex distance from basket < lane_height
        2. Row offset from basket row <= lane_width
        
        Returns:
            Set of (q, r) tuples representing lane hexes
        """
        return movement_core.calculate_offensive_lane_hexes(self)
    
    @profile_section("_calculate_defensive_lane_hexes")
    def _calculate_defensive_lane_hexes(self) -> set:
        """Calculate the defensive lane (full painted area, same as offensive lane).
        
        Defenders cannot camp in the lane for more than max_steps, enforcing 
        the defensive 3-second violation rule across the entire lane area.
        """
        return movement_core.calculate_defensive_lane_hexes(self)

    def _compute_three_point_geometry(self) -> None:
        """Precompute which hexes qualify for threes and the outline cells."""
        hexes, line_hexes, outline_points = geometry.compute_three_point_geometry(self)
        self._three_point_hexes = set(hexes)
        self._three_point_line_hexes = set(line_hexes)
        self._three_point_outline_points = list(outline_points)

    def _is_three_point_hex(self, coord: Tuple[int, int]) -> bool:
        if coord is None:
            return False
        return tuple(coord) in self._three_point_hexes

    def is_three_point_location(self, coord: Tuple[int, int]) -> bool:
        """Public helper for external modules (analytics, wrappers, etc.)."""
        return self._is_three_point_hex(coord)

    @profile_section("process_actions")
    def _process_simultaneous_actions(self, actions: np.ndarray) -> Dict:
        """Process all player actions simultaneously with collision resolution."""
        results = {
            "moves": {},
            "passes": {},
            "shots": {},
            "collisions": [],
            "turnovers": [],
            "defensive_lane_violations": [],
            "defender_pressure": {},  # Track defender pressure on ball handler
        }

        current_positions = self.positions
        final_positions = current_positions.copy()

        # 1. Handle non-movement actions first (shots, passes)
        for player_id, action_val in enumerate(actions):
            action = ActionType(action_val)
            if action == ActionType.SHOOT and player_id == self.ball_holder:
                results["shots"][player_id] = self._attempt_shot(player_id)
            elif "PASS" in action.name and player_id == self.ball_holder:
                direction_idx = action.value - ActionType.PASS_E.value
                self._attempt_pass(player_id, direction_idx, results)

        movement_core.resolve_movement(self, actions, results, current_positions)

        return results

    @profile_section("_get_adjacent_position")
    def _get_adjacent_position(
        self, pos: Tuple[int, int], direction_idx: int
    ) -> Tuple[int, int]:
        """Get adjacent hexagon position in given direction."""
        q, r = pos
        dq, dr = self.hex_directions[direction_idx]
        return (q + dq, r + dr)

    @profile_section("_has_teammate_in_pass_arc")
    def _has_teammate_in_pass_arc(self, passer_id: int, direction_idx: int) -> bool:
        """
        Check if there is at least one teammate within the pass arc for the given direction.

        Args:
            passer_id: ID of the player attempting to pass
            direction_idx: Direction index (0-5 for the 6 hex directions)

        Returns:
            True if at least one teammate is in the arc, False otherwise
        """
        return passing_core.has_teammate_in_pass_arc(self, passer_id, direction_idx)

    @profile_section("_attempt_pass")
    def _attempt_pass(self, passer_id: int, direction_idx: int, results: Dict) -> None:
        """
        Arc-based passing with line-of-sight steal mechanics:
        
        1. Determine arc centered on chosen direction (configurable, default 60 degrees)
        2. Find nearest teammate in arc as the target receiver
        3. If no eligible teammate, treat as pass out of bounds (configurable turnover probability)
        4. Line-of-sight steal evaluation:
           - Only defenders in 180° forward hemisphere (chosen direction ± adjacent directions) are considered
           - Of those, only defenders between passer and receiver (along the pass line) are evaluated
           - Each defender's steal contribution depends on:
             * Perpendicular distance from the pass line (closer = higher chance)
             * Total pass distance (longer passes = higher chance)
             * Position along pass line (defenders closer to receiver are more dangerous)
           - Formula: steal_i = base_rate * exp(-perp_decay * perp_dist) * (1 + dist_factor * pass_dist) * position_weight
           - Position weight: position_weight = min_weight + (1 - min_weight) * t, where t ∈ [0,1] (0=passer, 1=receiver)
           - Multiple defender contributions are compounded: total_steal = 1 - ∏(1 - steal_i)
           - Defender with highest contribution gets the ball if interception occurs
        5. If no defenders in forward hemisphere between passer and receiver, pass always succeeds
        """
        return passing_core.attempt_pass(self, passer_id, direction_idx, results)

    @profile_section("_compute_shot_pressure_multiplier")
    def _compute_shot_pressure_multiplier(
        self,
        shooter_id: Optional[int],
        shooter_pos: Tuple[int, int],
        distance_to_basket: int,
    ) -> float:
        return shooting_core.compute_shot_pressure_multiplier(
            self, shooter_id, shooter_pos, distance_to_basket
        )

    @profile_section("attempt_shot")
    def _attempt_shot(self, shooter_id: int) -> Dict:
        """Attempt a shot from the ball holder."""
        return shooting_core.attempt_shot(self, shooter_id)

    @profile_section("_turnover_to_defense")
    def _turnover_to_defense(self, from_player: int):
        """Handle turnover - ball goes to nearest defender."""
        if from_player in self.offense_ids:
            # Find nearest defender
            nearest_defender = min(
                self.defense_ids,
                key=lambda d: self._hex_distance(
                    self.positions[from_player], self.positions[d]
                ),
            )
        else:
            # Find nearest offensive player
            nearest_defender = min(
                self.offense_ids,
                key=lambda o: self._hex_distance(
                    self.positions[from_player], self.positions[o]
                ),
            )

        self.ball_holder = nearest_defender

    @profile_section("_calculate_shot_probability")
    def _calculate_shot_probability(self, shooter_id: int, distance: int) -> float:
        return shooting_core.calculate_shot_probability(self, shooter_id, distance)

    @profile_section("_check_termination_and_rewards")
    def _check_termination_and_rewards(
        self, action_results: Dict
    ) -> Tuple[bool, np.ndarray]:
        return rewards_core.check_termination_and_rewards(self, action_results)

    # -------------------- Expected Points Calculation --------------------
    
    @profile_section("_calculate_expected_points_for_player")
    def _calculate_expected_points_for_player(self, player_id: int) -> float:
        """Calculate expected points for a single player.
        
        Computes expected points using pressure-adjusted make probability times
        shot value (3 for beyond the arc, otherwise 2; dunk treated as 2).
        
        Args:
            player_id: ID of the player to calculate EP for
            
        Returns:
            Expected points value (shot_value × pressure_adjusted_probability)
        """
        return rewards_core.calculate_expected_points_for_player(self, player_id)

    # -------------------- Potential Function Phi(s) --------------------
    @profile_section("_phi_shot_quality")
    def _phi_shot_quality(self) -> float:
        """Potential function Phi(s): team's current best expected points.

        Computes expected points using pressure-adjusted make probability times
        shot value (3 for beyond the arc, otherwise 2; dunk treated as 2).
        If `phi_use_ball_handler_only` is True, returns only the current ball
        handler's expected points; otherwise returns the best expected points
        among players on the team currently in possession.
        """
        return rewards_core.phi_shot_quality(self)

    @profile_section("_phi_ep_breakdown")
    def _phi_ep_breakdown(self) -> Tuple[float, float]:
        """Return (team_best_ep, ball_handler_ep) for current possession team."""
        return rewards_core.phi_ep_breakdown(self)

    # Allow VecEnv.env_method to update phi_beta dynamically
    @profile_section("set_phi_beta")
    def set_phi_beta(self, value: float) -> None:
        try:
            self.phi_beta = float(value)
        except Exception:
            pass

    # --- Schedulable setters for passing curriculum ---
    @profile_section("set_pass_arc_degrees")
    def set_pass_arc_degrees(self, value: float) -> None:
        try:
            self.pass_arc_degrees = float(max(1.0, min(360.0, value)))
        except Exception:
            pass

    @profile_section("set_pass_oob_turnover_prob")
    def set_pass_oob_turnover_prob(self, value: float) -> None:
        try:
            self.pass_oob_turnover_prob = float(max(0.0, min(1.0, value)))
        except Exception:
            pass

    @profile_section("set_pass_target_strategy")
    def set_pass_target_strategy(self, strategy: str) -> None:
        try:
            normalized = str(strategy).lower()
            if normalized in ("nearest", "best_ev"):
                self.pass_target_strategy = normalized
        except Exception:
            pass

    @profile_section("_get_player_distances")
    def _get_player_distances(self, base_id: int, target_ids: List[int]) -> np.ndarray:
        """Return hex distances from `base_id` to each ID in `target_ids`."""
        from basketworld.envs.core import observations as obs_core

        return obs_core.get_player_distances(self, base_id, target_ids)

    @profile_section("_get_player_angles")
    def _get_player_angles(self, base_id: int, target_ids: List[int]) -> np.ndarray:
        """Return cosine angles between base→target and base→basket for each target."""
        from basketworld.envs.core import observations as obs_core

        return obs_core.get_player_angles(self, base_id, target_ids)

    
    @profile_section("_collect_pairwise_features")
    def _collect_pairwise_features(
        self,
        base_ids: List[int],
        target_ids: List[int],
        getter: Callable[[int, List[int]], np.ndarray],
    ) -> np.ndarray:
        """Collect features for each (base, target) pair using the provided getter."""
        values: List[float] = []
        if not target_ids:
            return np.array([], dtype=np.float32)

        for base_id in base_ids:
            feature_vec = getter(base_id, target_ids)
            values.extend(feature_vec.tolist())
        return np.array(values, dtype=np.float32)

    @profile_section("_calculate_offense_defense_distances")
    def _calculate_offense_defense_distances(self) -> np.ndarray:
        from basketworld.envs.core import observations as obs_core

        return obs_core.calculate_offense_defense_distances(self)

    @profile_section("_calculate_offense_defense_angles")
    def _calculate_offense_defense_angles(self) -> np.ndarray:
        from basketworld.envs.core import observations as obs_core

        return obs_core.calculate_offense_defense_angles(self)

    def _collect_teammate_features(
        self,
        team_ids: List[int],
        getter: Callable[[int, List[int]], np.ndarray],
    ) -> np.ndarray:
        """Collect features from the first teammate to their squad-mates."""
        if len(team_ids) <= 1:
            return np.array([], dtype=np.float32)
        return self._collect_pairwise_features([team_ids[0]], team_ids[1:], getter)

    @profile_section("_calculate_teammate_distances")
    def _calculate_teammate_distances(self) -> np.ndarray:
        from basketworld.envs.core import observations as obs_core

        return obs_core.calculate_teammate_distances(self)

    @profile_section("_calculate_teammate_angles")
    def _calculate_teammate_angles(self) -> np.ndarray:
        from basketworld.envs.core import observations as obs_core

        return obs_core.calculate_teammate_angles(self)

    @profile_section("_get_observation")
    def _get_observation(self) -> np.ndarray:
        """Get current observation of the game state.

        Absolute position vector (NOT egocentric):
        - For each player i: (q_i, r_i) in absolute court coordinates, normalized
        - One-hot ball holder
        - Shot clock (raw value)
        - Absolute ball handler position (q, r) - helps distinguish court regions (center vs. sides)
        - Hoop vector: (q, r) in absolute coordinates, normalized
        """
        from basketworld.envs.core import observations as obs_core

        return obs_core.build_observation(self)

    @profile_section("_get_offense_skills_array")
    def _get_offense_skills_array(self) -> np.ndarray:
        """Return per-offense-player skill deltas as a flat array of length players_per_side*3.

        Order per offense player: (layup_delta, three_pt_delta, dunk_delta)
        """
        skills: List[float] = []
        for i in range(self.players_per_side):
            skills.append(
                float(self.offense_layup_pct_by_player[i]) - float(self.layup_pct)
            )
            skills.append(
                float(self.offense_three_pt_pct_by_player[i]) - float(self.three_pt_pct)
            )
            skills.append(
                float(self.offense_dunk_pct_by_player[i]) - float(self.dunk_pct)
            )
        return np.array(skills, dtype=np.float32)

    def _attach_legal_action_stats(self, info: Optional[Dict]) -> Dict:
        if info is None:
            info = {}
        info.setdefault(
            "legal_actions_offense",
            float(getattr(self, "_legal_actions_offense", 0.0)),
        )
        info.setdefault(
            "legal_actions_defense",
            float(getattr(self, "_legal_actions_defense", 0.0)),
        )
        return info

    @profile_section("render")
    def render(self):
        """Render the current state of the environment."""
        if self.render_mode == "human":
            return rendering_core.render_ascii(self)
        elif self.render_mode == "rgb_array":
            return rendering_core.render_visual(self)

    @profile_section("_render_ascii")
    def _render_ascii(self):
        """Simple ASCII rendering for training."""
        return rendering_core.render_ascii(self)

    @profile_section("_render_visual")
    def _render_visual(self):
        """Visual rendering using matplotlib."""
        return rendering_core.render_visual(self)

    @profile_section("switch_training_team")
    def switch_training_team(self):
        """Switch which team is currently training (for alternating optimization)."""
        self.training_team = (
            Team.DEFENSE if self.training_team == Team.OFFENSE else Team.OFFENSE
        )

    @profile_section("_calculate_steal_probability_for_pass")
    def _calculate_steal_probability_for_pass(
        self, 
        passer_pos: Tuple[int, int], 
        recv_pos: Tuple[int, int],
        passer_id: int
    ) -> Tuple[float, List[Tuple[int, float, float, float]]]:
        """
        Calculate steal probability for a specific pass from passer to receiver.
        
        Returns:
            (total_steal_prob, defender_contributions)
            where defender_contributions is a list of (defender_id, steal_contrib, perp_dist, position_t)
        
        This is the core calculation used both for:
        1. Actual steal resolution during gameplay  
        2. Observation features for agents to see steal risk
        """
        return passing_core.calculate_steal_probability_for_pass(self, passer_pos, recv_pos, passer_id)

    @profile_section("calculate_pass_steal_probabilities")
    def calculate_pass_steal_probabilities(self, passer_id: int) -> Dict[int, float]:
        """
        Calculate hypothetical steal probabilities for passes to each teammate.
        Returns a dict mapping teammate_id -> steal_probability.
        """
        return passing_core.calculate_pass_steal_probabilities(self, passer_id)

    @profile_section("_calculate_defender_pressure_info")
    def _calculate_defender_pressure_info(self) -> List[Dict]:
        """
        Calculate defender pressure information for the current ball handler.
        Returns list of dicts with defender_id, distance, and turnover_prob.
        
        This is the core calculation used both for:
        1. Actual turnover resolution during gameplay
        2. Observation features for agents to see turnover risk
        
        Only considers defenders in front (180° arc toward basket, cos(angle) >= 0).
        
        Refactored to use the distance and angle calculation methods.
        """
        from basketworld.envs.core import pressure as pressure_core

        return pressure_core.calculate_defender_pressure_info(self)

    @profile_section("calculate_defender_pressure_turnover_probability")
    def calculate_defender_pressure_turnover_probability(self) -> float:
        """
        Calculate the total turnover probability from defender pressure on the ball handler.
        Returns the compound probability of turnover from all nearby defenders.
        """
        from basketworld.envs.core import pressure as pressure_core

        return pressure_core.calculate_defender_pressure_turnover_probability(self)

    @profile_section("calculate_expected_points_all_players")
    def calculate_expected_points_all_players(self) -> np.ndarray:
        """
        Calculate expected points for offensive players based on their current position
        and defender pressure. Uses pressure-adjusted shot probability.
        
        EP only makes sense for the team that can score (offense), so we only calculate
        it for offensive players.
        
        Returns array of shape (players_per_side,) with EP for each offensive player.
        """
        return rewards_core.calculate_expected_points_all_players(self)

    # --- Profiling helpers ---
    @profile_section("get_profile_stats")
    def get_profile_stats(self) -> Dict[str, Dict[str, float]]:
        stats: Dict[str, Dict[str, float]] = {}
        for k, total_ns in self._profile_ns.items():
            calls = max(1, self._profile_calls.get(k, 1))
            stats[k] = {
                "total_ms": total_ns / 1e6,
                "avg_us": (total_ns / calls) / 1e3,
                "calls": float(calls),
            }
        return stats

    @profile_section("reset_profile_stats")
    def reset_profile_stats(self) -> None:
        self._profile_ns.clear()
        self._profile_calls.clear()


# Test/Demo code
if __name__ == "__main__":
    # Test the environment
    env = HexagonBasketballEnv(
        grid_size=16,
        players_per_side=3,
        shot_clock_steps=24,
        training_team=Team.OFFENSE,
        seed=42,
    )

    print("=== Hexagon Basketball Environment Demo ===")
    obs, info = env.reset()
    print(f"Initial observation shape: {obs['obs'].shape}")
    print(f"Action space: {env.action_space}")

    env.render()

    # Run a few random steps
    for step in range(5):
        actions = env.action_space.sample()
        obs, rewards, done, truncated, info = env.step(actions)

        print(f"\nStep {step + 1}:")
        print(f"Actions: {actions}")
        print(f"Rewards: {rewards}")
        print(f"Done: {done}")

        env.render()

        if done:
            print("Episode ended!")
            break

    print("\n=== Switching to Defense Training ===")
    env.switch_training_team()
    obs, info = env.reset()
    env.render()
