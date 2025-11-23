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

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self,
        grid_size: int = 16,
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

        self.grid_size = grid_size
        self.court_width = int(grid_size * 1.0)
        self.court_height = grid_size
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
        teammate_distance_extra = 2 * max(0, self.players_per_side - 1)
        teammate_angle_extra = 2 * max(0, self.players_per_side - 1)
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
        for row in range(self.court_height):
            for col in range(self.court_width):
                cell = self._offset_to_axial(col, row)
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

    @profile_section("_offset_to_axial")
    def _offset_to_axial(self, col: int, row: int) -> Tuple[int, int]:
        """Converts odd-r offset coordinates to axial coordinates."""
        q = col - (row - (row & 1)) // 2
        r = row
        return q, r

    @profile_section("_axial_to_offset")
    def _axial_to_offset(self, q: int, r: int) -> Tuple[int, int]:
        """Converts axial coordinates to odd-r offset coordinates."""
        col = q + (r - (r & 1)) // 2
        row = r
        return col, row

    @profile_section("_axial_to_cartesian")
    def _axial_to_cartesian(self, q: int, r: int) -> Tuple[float, float]:
        """Convert axial (q, r) to cartesian (x, y) matching rendering geometry."""
        size = 1.0
        x = size * (math.sqrt(3) * q + math.sqrt(3) / 2 * r)
        y = size * (1.5 * r)
        return x, y

    @profile_section("_axial_to_cube")
    def _axial_to_cube(self, q: int, r: int) -> Tuple[int, int, int]:
        """Convert axial (q, r) to cube (x, y, z) coordinates."""
        x, z = q, r
        y = -x - z
        return x, y, z

    @profile_section("_cube_to_axial")
    def _cube_to_axial(self, x: int, y: int, z: int) -> Tuple[int, int]:
        """Convert cube (x, y, z) to axial (q, r) coordinates."""
        return x, z

    @profile_section("_rotate60_cw_cube")
    def _rotate60_cw_cube(self, x: int, y: int, z: int) -> Tuple[int, int, int]:
        """Rotate cube (x, y, z) by 60 degrees clockwise."""
        return -z, -x, -y

    @profile_section("_rotate_k60_axial")
    def _rotate_k60_axial(self, q: int, r: int, k: int) -> Tuple[int, int]:
        """Rotate axial (q, r) by k*60 degrees clockwise."""
        x, y, z = self._axial_to_cube(q, r)
        for _ in range(k % 6):
            x, y, z = self._rotate60_cw_cube(x, y, z)
        return self._cube_to_axial(x, y, z)

    @profile_section("_hex_distance")
    def _hex_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate distance between two hexagon positions."""
        q1, r1 = pos1
        q2, r2 = pos2
        return (abs(q1 - q2) + abs(q1 + r1 - q2 - r2) + abs(r1 - r2)) // 2

    @profile_section("_defender_is_guarding_offense")
    def _defender_is_guarding_offense(self, defender_id: int) -> bool:
        """Return True if any offensive player is within guard distance of the defender."""
        if self.defender_guard_distance <= 0:
            return False
        if not self.positions:
            return False

        def_pos = self.positions[defender_id]
        for off_id in self.offense_ids:
            if self._hex_distance(def_pos, self.positions[off_id]) <= self.defender_guard_distance:
                return True
        return False

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
        # Convert all positions to Cartesian coordinates
        px, py = self._axial_to_cartesian(point[0], point[1])
        sx, sy = self._axial_to_cartesian(line_start[0], line_start[1])
        ex, ey = self._axial_to_cartesian(line_end[0], line_end[1])
        
        # Vector from start to end
        dx = ex - sx
        dy = ey - sy
        line_length_sq = dx * dx + dy * dy
        
        if line_length_sq == 0:
            # Line start and end are the same point
            return math.hypot(px - sx, py - sy)
        
        # Project point onto line: t = [(P-S) · (E-S)] / |E-S|²
        t = ((px - sx) * dx + (py - sy) * dy) / line_length_sq
        
        # Find closest point on line segment (clamped to [0, 1])
        t = max(0.0, min(1.0, t))
        closest_x = sx + t * dx
        closest_y = sy + t * dy
        
        # Return distance from point to closest point on line
        return math.hypot(px - closest_x, py - closest_y)

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
        # Convert to Cartesian
        px, py = self._axial_to_cartesian(point[0], point[1])
        sx, sy = self._axial_to_cartesian(line_start[0], line_start[1])
        ex, ey = self._axial_to_cartesian(line_end[0], line_end[1])
        
        # Vector from start to end
        dx = ex - sx
        dy = ey - sy
        line_length_sq = dx * dx + dy * dy
        
        if line_length_sq == 0:
            # Line has no length
            return 0.0
        
        # Project point onto line: t = [(P-S) · (E-S)] / |E-S|²
        t = ((px - sx) * dx + (py - sy) * dy) / line_length_sq
        return t

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
        t = self._get_position_on_line(point, line_start, line_end)
        # Point is between start and end if 0 < t < 1
        return 0.0 < t < 1.0

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
        masks = np.ones((self.n_players, len(ActionType)), dtype=np.int8)

        # Only the ball holder can shoot or pass
        for i in range(self.n_players):
            if i != self.ball_holder:
                masks[i, self._shoot_pass_action_indices] = 0

            # Apply precomputed movement validity from the player's current cell
            cell = self.positions[i]
            move_mask = self._move_mask_by_cell.get(cell)
            if move_mask is not None:
                for dir_idx in range(6):
                    masks[i, ActionType.MOVE_E.value + dir_idx] = move_mask[dir_idx]

        # Optionally disallow moving into any currently occupied neighboring hex
        if self.mask_occupied_moves:
            occupied = set(self.positions)
            for i in range(self.n_players):
                curr_q, curr_r = self.positions[i]
                for dir_idx in range(6):
                    action_idx = ActionType.MOVE_E.value + dir_idx
                    if masks[i, action_idx] == 0:
                        continue
                    dq, dr = self.hex_directions[dir_idx]
                    nbr = (curr_q + dq, curr_r + dr)
                    if nbr in occupied:
                        masks[i, action_idx] = 0

        # Pass gating: mask out passes that don't have a teammate in the arc
        # This prevents the policy from learning to avoid passing due to OOB turnovers
        if self.enable_pass_gating and self.ball_holder is not None:
            for dir_idx in range(6):
                pass_action_idx = ActionType.PASS_E.value + dir_idx
                if not self._has_teammate_in_pass_arc(self.ball_holder, dir_idx):
                    masks[self.ball_holder, pass_action_idx] = 0

        return masks

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
        taken_positions: set[Tuple[int, int]] = set()

        # List all axial cells on court
        all_cells: List[Tuple[int, int]] = []
        for row in range(self.court_height):
            for col in range(self.court_width):
                all_cells.append(self._offset_to_axial(col, row))

        # Minimum distance from basket (negative means no minimum)
        min_spawn_dist_offense = max(0, self.spawn_distance)
        # Defenders can spawn 1 unit closer than offense minimum
        min_spawn_dist_defense = max(0, self.spawn_distance - 1)
        # Maximum distance from basket (None means no maximum)
        max_spawn_dist = self.max_spawn_distance

        # Offense candidates: any valid cell at least min_spawn_dist_offense from basket
        # and at most max_spawn_dist (if set)
        offense_candidates = []
        for cell in all_cells:
            if cell == self.basket_position:
                continue
            if not self._is_valid_position(*cell):
                continue
            dist = self._hex_distance(cell, self.basket_position)
            if dist >= min_spawn_dist_offense:
                # Apply max distance constraint if set
                if max_spawn_dist is None or dist <= max_spawn_dist:
                    offense_candidates.append(cell)

        if len(offense_candidates) < self.players_per_side:
            # Fallback to any valid non-basket cell
            offense_candidates = [
                cell
                for cell in all_cells
                if cell != self.basket_position and self._is_valid_position(*cell)
            ]
            if len(offense_candidates) < self.players_per_side:
                raise ValueError("Not enough valid cells to spawn offense.")

        # Sample unique offense positions
        offense_positions = []
        for cell in self._rng.choice(
            len(offense_candidates), size=self.players_per_side, replace=False
        ):
            pos = offense_candidates[cell]
            offense_positions.append(pos)
            taken_positions.add(pos)

        # Defense candidates: closer to basket than the offense counterpart, with a
        # minimum distance from the basket defined RELATIVE to the basket.
        # Defenders can spawn 1 unit closer than the offense minimum.
        # A negative spawn_distance means no minimum (allow anywhere on court).
        # Also enforce max_spawn_distance if set.
        # If defender_spawn_distance > 0, defenders spawn with randomized distance from their matched offense
        defense_positions: List[Tuple[int, int]] = []
        for off_pos in offense_positions:
            off_dist = self._hex_distance(off_pos, self.basket_position)
            
            # If defender_spawn_distance is set, use it to randomize spawn distance from offense
            if self.defender_spawn_distance > 0:
                # Random distance from matched offense player (1 to defender_spawn_distance hexes away)
                target_dist_from_offense = self._rng.integers(1, self.defender_spawn_distance + 1)
                # Find candidates at approximately that distance from the offense player
                # Allow +/- 1 hex tolerance for flexibility
                candidates = [
                    cell
                    for cell in all_cells
                    if cell != self.basket_position
                    and cell not in taken_positions
                    and self._is_valid_position(*cell)
                    and self._hex_distance(cell, self.basket_position) < off_dist
                    and self._hex_distance(cell, self.basket_position) >= min_spawn_dist_defense
                    and (max_spawn_dist is None or self._hex_distance(cell, self.basket_position) <= max_spawn_dist)
                    and abs(self._hex_distance(cell, off_pos) - target_dist_from_offense) <= 1
                ]
            else:
                # Original behavior: spawn as close as possible to offense (distance 1)
                candidates = [
                    cell
                    for cell in all_cells
                    if cell != self.basket_position
                    and cell not in taken_positions
                    and self._is_valid_position(*cell)
                    and self._hex_distance(cell, self.basket_position) < off_dist
                    and self._hex_distance(cell, self.basket_position) >= min_spawn_dist_defense
                    and (max_spawn_dist is None or self._hex_distance(cell, self.basket_position) <= max_spawn_dist)
                ]

            if not candidates:
                # Fallback: pick any valid empty cell meeting min/max spawn distance constraints
                candidates = [
                    cell
                    for cell in all_cells
                    if cell != self.basket_position
                    and cell not in taken_positions
                    and self._is_valid_position(*cell)
                    and self._hex_distance(cell, self.basket_position) >= min_spawn_dist_defense
                    and (max_spawn_dist is None or self._hex_distance(cell, self.basket_position) <= max_spawn_dist)
                ]

            if not candidates:
                # Final fallback: any valid empty cell (avoid crashing)
                candidates = [
                    cell
                    for cell in all_cells
                    if cell != self.basket_position
                    and cell not in taken_positions
                    and self._is_valid_position(*cell)
                ]

            # If using defender_spawn_distance, pick randomly from candidates; otherwise pick nearest
            if self.defender_spawn_distance > 0:
                def_pos = candidates[self._rng.integers(0, len(candidates))]
            else:
                # Original: choose the candidate nearest to the offensive player (to simulate marking)
                candidates.sort(key=lambda c: self._hex_distance(c, off_pos))
                def_pos = candidates[0]
            defense_positions.append(def_pos)
            taken_positions.add(def_pos)

        # offense first then defense
        return offense_positions + defense_positions

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
        col, row = self._axial_to_offset(q, r)
        return 0 <= col < self.court_width and 0 <= row < self.court_height
    
    @profile_section("_calculate_offensive_lane_hexes")
    def _calculate_offensive_lane_hexes(self) -> set:
        """Calculate the hexes that make up the offensive lane (painted area).
        
        The lane extends from the basket along the +q axis (toward offensive side)
        up to (but not including) the 3-point line distance.
        The lane has symmetric width on both sides.
        
        Returns:
            Set of (q, r) tuples representing lane hexes
        """
        lane_hexes = set()
        basket_q, basket_r = self.basket_position
        lane_width = self.three_second_lane_width
        lane_height = self.three_second_lane_height
        
        # Lane extends from distance 0 (basket) to just before 3pt line
        for dist in range(0, lane_height):
            # For each distance, add hexes within lane_width perpendicular distance
            # We'll explore in the +q direction from basket
            # At each step along q-axis, check r offsets within lane_width
            for q_offset in range(dist + 1):
                for r_offset in range(-dist, dist + 1):
                    q = basket_q + q_offset
                    r = basket_r + r_offset
                    
                    # Check if this hex is within the lane width and at the right distance
                    if self._hex_distance((q, r), self.basket_position) == dist:
                        # Calculate perpendicular distance from center line
                        # Center line is along +q axis from basket
                        # For simplicity, check if r_offset is within bounds
                        if abs(r - basket_r) <= lane_width and self._is_valid_position(q, r):
                            lane_hexes.add((q, r))
        
        return lane_hexes
    
    @profile_section("_calculate_defensive_lane_hexes")
    def _calculate_defensive_lane_hexes(self) -> set:
        """Calculate the defensive lane (full painted area, same as offensive lane).
        
        Defenders cannot camp in the lane for more than max_steps, enforcing 
        the defensive 3-second violation rule across the entire lane area.
        """
        # Defensive lane is the same area as offensive lane
        return self._calculate_offensive_lane_hexes()

    def _compute_three_point_geometry(self) -> None:
        """Precompute which hexes qualify for threes and the outline cells."""
        self._three_point_hexes.clear()
        self._three_point_line_hexes.clear()
        self._three_point_outline_points.clear()

        if self.three_point_distance <= 0:
            return

        basket_axial = self.basket_position
        hoop_x, hoop_y = self._axial_to_cartesian(*basket_axial)
        radius_cart = float(self.three_point_distance) * math.sqrt(3.0)
        short_band = (
            float(self.three_point_short_distance) * math.sqrt(3.0)
            if self.three_point_short_distance is not None
            else None
        )
        tolerance = 0.35  # tuned to keep outline contiguous on discrete grid

        outline_seen: Set[Tuple[int, int]] = set()

        for row in range(self.court_height):
            for col in range(self.court_width):
                q, r = self._offset_to_axial(col, row)
                cell = (q, r)
                cx, cy = self._axial_to_cartesian(q, r)
                dx = cx - hoop_x
                dy = cy - hoop_y
                abs_dy = abs(dy)
                dist_cart = math.hypot(dx, dy)

                qualifies = False
                is_outline = False

                # Sampling logic to determine qualification (at least 50% of hex area)
                # We use a 7-point sampling pattern: center + 6 vertices
                hex_size_px = 1.0
                px_per_hex = hex_size_px * math.sqrt(3)
                # Radius of circumcircle for hex with height 1.0 (pointy-top) is 1/sqrt(3) * height?
                # No, our hexes are pointy-topped.
                # Height of pointy-topped hex is 2 * size. Width is sqrt(3) * size.
                # If we assume axial coordinates map to Cartesian where dist((0,0), (1,0)) = sqrt(3) * size,
                # then spacing is sqrt(3) * size.
                # In _axial_to_cartesian: x = size * sqrt(3) * (q + r/2), y = size * 3/2 * r
                # Normalized hex size in _axial_to_cartesian seems to be 1.0 implicit.
                # Let's assume size=1 for sampling.
                
                # Vertices of a pointy-topped hex at (cx, cy) with size=1
                # angles: 30, 90, 150, 210, 270, 330 degrees
                # x = cx + size * cos(angle)
                # y = cy + size * sin(angle)
                
                samples = [(cx, cy)] # Center
                for i in range(6):
                    angle_deg = 30 + 60 * i
                    angle_rad = math.radians(angle_deg)
                    # Use slightly smaller radius to stay within the hex
                    sx = cx + 0.95 * math.cos(angle_rad)
                    sy = cy + 0.95 * math.sin(angle_rad)
                    samples.append((sx, sy))
                
                qualified_samples = 0
                for sx, sy in samples:
                    sdx = sx - hoop_x
                    sdy = sy - hoop_y
                    sabs_dy = abs(sdy)
                    sdist = math.hypot(sdx, sdy)
                    
                    is_3pt_pt = False
                    if short_band is not None and sabs_dy >= short_band - tolerance:
                         # Behind short line (extended infinitely? No, usually just horizontal band)
                         # The visualizer logic:
                         # if abs(dy) >= short_band: qualifies
                         # But wait, the short line is a vertical line segments at x? 
                         # No, in NBA, short lines are parallel to sidelines (horizontal in our rotated view?)
                         # Our visualizer had: "short distance refers to distance from basket to lines on SIDES"
                         # Basket is at (0, H/2). Court is WxH.
                         # Sidelines are top/bottom in our (x,y) or left/right?
                         # _axial_to_cartesian: x is roughly col, y is roughly row.
                         # Basket at (0, H/2) is left-center.
                         # So sidelines are top (y=0) and bottom (y=max).
                         # Short distance is distance from basket center Y to the straight lines.
                         # So straight lines are at hoop_y +/- short_band.
                         # Yes, so abs(dy) >= short_band means OUTSIDE the central lane, i.e. near the sidelines.
                         # Wait, NBA short corner: The line is straight near the sidelines, then curves.
                         # The straight part is CLOSER to the basket than the arc would be.
                         # The 3pt line is the arc, UNLESS the straight line is closer.
                         # So you are a 3pt shooter if you are OUTSIDE the shape formed by min(arc, straight).
                         # The straight lines are at y = +/- 22ft (NBA). Arc is r=23.75ft.
                         # If |y| > 22, the distance to hoop > 22. 
                         # But the line is AT 22. So if |y| > 22, you are "behind" the line?
                         # No, the line is x = ...?
                         # Let's re-read visualizer logic.
                         pass

                    # Re-implementing precise visualizer logic for point classification
                    # Visualizer: pointIsThree(px, py)
                    # if (shortPx defined)
                    #    horizontalReach = sqrt(R^2 - short^2)
                    #    if abs(dy) >= shortPx:
                    #        return dx >= horizontalReach (Wait, this implies a box corner?)
                    # No, NBA line:
                    # You are 3pt if:
                    # 1. |y| > 22ft ? No.
                    # The line is defined by x = 22ft (corner) -- wait, in NBA corners are at bottom/top of visual.
                    # In our view (hoop left), corners are top/bottom.
                    # Straight lines are y = hoop_y +/- short_band?
                    # If |y - hoop_y| > short_band... that means you are FAR from center.
                    # Actually, the straight lines in NBA are parallel to the sidelines.
                    # Distance from center stripe is fixed (22ft).
                    # So if you are in the "corner" (high |y|), the line is straight at x = constant? 
                    # No, the line is parallel to side, so y = const? No, line is parallel to side...
                    # Sideline is y=0 and y=H.
                    # Parallel to sideline means y = const.
                    # So the 3pt line is y = hoop_y + 22 and y = hoop_y - 22? 
                    # That would be a horizontal line running full court.
                    # No, the 3pt line *segment* is straight there.
                    # It connects to the baseline (x=0).
                    # So for x < some_intersection, the line is y = +/- 22.
                    # For x > intersection, it's the arc.
                    # Wait, if line is y = 22, then distance from hoop (0,0) is sqrt(x^2 + 22^2).
                    # If x=0, dist=22. This is < 23.75.
                    # So yes, near baseline, the line is closer (22ft).
                    # So a point (x,y) is a 3pt attempt if:
                    #   if x < intersection:
                    #       return |y| > short_band (You must be "outside" the line towards the sideline)
                    #   else:
                    #       return dist > radius
                    
                    # BUT our visualizer had specific logic.
                    # Let's assume the visualizer logic was:
                    # if short_band is set:
                    #    horizontal_reach = sqrt(radius^2 - short^2)
                    #    if dx < horizontal_reach:
                    #        is_3pt = abs(dy) >= short_band
                    #    else:
                    #        is_3pt = dist >= radius
                    
                    # Let's validate this.
                    # If dx is small (near baseline), we check if |dy| is large enough.
                    # If |dy| < short_band, we are inside the paint/2pt zone.
                    # If |dy| > short_band, we are outside -> 3pt.
                    # Correct.
                    
                    # Calculate horizontal_reach (x-coord where arc meets straight line)
                    # straight line at y = short_band. Circle x^2 + y^2 = R^2.
                    # x^2 + short^2 = R^2 => x = sqrt(R^2 - short^2)
                    
                    horizontal_reach = 0.0
                    if short_band is not None and short_band < radius_cart:
                        horizontal_reach = math.sqrt(radius_cart**2 - short_band**2)
                    
                    if short_band is not None:
                        if dx < horizontal_reach:
                            # In the straight-line region (corner)
                            if sabs_dy >= short_band:
                                is_3pt_pt = True
                        else:
                            # In the arc region
                            if sdist >= radius_cart:
                                is_3pt_pt = True
                    else:
                        # Just arc
                        if sdist >= radius_cart:
                            is_3pt_pt = True
                            
                    if is_3pt_pt:
                        qualified_samples += 1
                
                qualifies = (qualified_samples / len(samples)) >= 0.5
                
                # Outline check (simplified: if center is close to boundary)
                # Ideally we check if hex crosses the boundary, but distance check is decent proxy
                if short_band is not None:
                    if dx < horizontal_reach:
                        # Distance to straight line
                        dist_to_line = abs(sabs_dy - short_band)
                        if dist_to_line <= tolerance:
                            is_outline = True
                    else:
                        # Distance to arc
                        if abs(dist_cart - radius_cart) <= tolerance:
                            is_outline = True
                else:
                    if abs(dist_cart - radius_cart) <= tolerance:
                        is_outline = True

                if qualifies:
                    self._three_point_hexes.add(cell)
                if is_outline:
                    self._three_point_line_hexes.add(cell)
                    outline_seen.add(cell)

        outline_points = [self._axial_to_cartesian(q, r) for q, r in outline_seen]
        outline_points.sort(
            key=lambda pt: math.atan2(pt[1] - hoop_y, pt[0] - hoop_x)
        )
        self._three_point_outline_points = outline_points

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

        # If a shot was taken or a turnover occurred, the episode will end.
        # Skip movement processing to avoid spurious state changes in terminal observations.
        if results.get("shots") or results.get("turnovers"):
            return results

        # 2. Determine intended moves for all players
        intended_moves = {}
        for player_id, action_val in enumerate(actions):
            action = ActionType(action_val)
            if ActionType.MOVE_E.value <= action.value <= ActionType.MOVE_SE.value:
                direction_idx = action.value - ActionType.MOVE_E.value
                new_pos = self._get_adjacent_position(
                    current_positions[player_id], direction_idx
                )

                if self._is_valid_position(*new_pos):
                    if (new_pos == self.basket_position) and (not self.allow_dunks):
                        # Basket hex blocked when dunks disabled
                        if player_id == self.ball_holder:
                            results["turnovers"].append(
                                {
                                    "player_id": player_id,
                                    "reason": "move_out_of_bounds",
                                    "turnover_pos": new_pos,
                                }
                            )
                            self._turnover_to_defense(player_id)
                        results["moves"][player_id] = {
                            "success": False,
                            "reason": "basket_collision",
                        }
                    else:
                        intended_moves[player_id] = new_pos
                else:
                    # Out of bounds move
                    if player_id == self.ball_holder:
                        results["turnovers"].append(
                            {
                                "player_id": player_id,
                                "reason": "move_out_of_bounds",
                                "turnover_pos": new_pos,
                            }
                        )
                        self._turnover_to_defense(player_id)
                    results["moves"][player_id] = {
                        "success": False,
                        "reason": "out_of_bounds",
                    }

        # Always block moves into any cell that was occupied at the start of the step
        if intended_moves:
            occupied_start = set(current_positions)
            to_remove = []
            for pid, dest in intended_moves.items():
                if dest in occupied_start:
                    results["moves"][pid] = {
                        "success": False,
                        "reason": "occupied_neighbor",
                    }
                    to_remove.append(pid)
            for pid in to_remove:
                intended_moves.pop(pid, None)

        # 3. Resolve movement based on conflicts

        # Players who are not moving this turn
        static_players = set(range(self.n_players)) - set(intended_moves.keys())

        # Positions that will be occupied by players who are not moving
        occupied_by_static = {current_positions[pid] for pid in static_players}

        # Group moving players by their intended destination
        move_destinations = defaultdict(list)
        for player_id, dest in intended_moves.items():
            move_destinations[dest].append(player_id)

        # Iterate through destinations and resolve conflicts
        for dest, players_intending_to_move in move_destinations.items():
            # a. Check for collisions with players who aren't moving
            if dest in occupied_by_static:
                for player_id in players_intending_to_move:
                    results["moves"][player_id] = {
                        "success": False,
                        "reason": "collision_static",
                    }
                continue  # No one can move here

            # b. Check for collisions between multiple moving players
            if len(players_intending_to_move) > 1:
                # Collision occurs, pick one winner
                winner = self._rng.choice(players_intending_to_move)
                final_positions[winner] = dest
                results["moves"][winner] = {"success": True, "new_position": dest}

                # Others fail
                for player_id in players_intending_to_move:
                    if player_id != winner:
                        results["moves"][player_id] = {
                            "success": False,
                            "reason": "collision_dynamic",
                        }

                results["collisions"].append(
                    {
                        "position": dest,
                        "players": players_intending_to_move,
                        "winner": winner,
                    }
                )
            else:
                # No collision, single player moves
                player_id = players_intending_to_move[0]
                final_positions[player_id] = dest
                results["moves"][player_id] = {"success": True, "new_position": dest}

        self.positions = final_positions
        
        # Check for offensive 3-second violations
        # First update the lane counters based on new positions, then check for violations
        if self.offensive_three_seconds_enabled:
            # Update lane step counters based on new positions
            for oid in self.offense_ids:
                if tuple(self.positions[oid]) in self.offensive_lane_hexes:
                    self._offensive_lane_steps[oid] = (
                        self._offensive_lane_steps.get(oid, 0) + 1
                    )
                else:
                    self._offensive_lane_steps[oid] = 0
            
            # Now check for violations using the updated counters
            for oid in self.offense_ids:
                steps_in_lane = self._offensive_lane_steps.get(oid, 0)
                in_lane = tuple(self.positions[oid]) in self.offensive_lane_hexes
                has_ball = (oid == self.ball_holder)
                
                # Violation occurs if:
                # 1. Player has been in lane for max_steps
                # 2. Player is still in lane or just entered
                # 3. Player doesn't have the ball (exception for ball handler)
                # 4. If player has ball and at max_steps+1, they MUST shoot
                
                if in_lane:
                    if steps_in_lane >= self.three_second_max_steps:
                        if not has_ball:
                            # Violation: been in lane too long without ball
                            results["turnovers"].append({
                                "player_id": oid,
                                "reason": "offensive_three_seconds",
                                "turnover_pos": self.positions[oid],
                            })
                            # Transfer possession to defense
                            if self.ball_holder is not None:
                                self._turnover_to_defense(self.ball_holder)
                            break  # Only one violation per step
                        elif has_ball and steps_in_lane > self.three_second_max_steps:
                            # Ball handler at max_steps+1: if they didn't shoot, it's a violation
                            # Check if they took a shot action
                            action_taken = ActionType(actions[oid])
                            if action_taken != ActionType.SHOOT:
                                results["turnovers"].append({
                                    "player_id": oid,
                                    "reason": "offensive_three_seconds",
                                    "turnover_pos": self.positions[oid],
                                })
                                self._turnover_to_defense(oid)
                                break
        
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
        passer_pos = self.positions[passer_id]
        dir_dq, dir_dr = self.hex_directions[direction_idx]

        # Compute angles in cartesian space
        dir_x, dir_y = self._axial_to_cartesian(dir_dq, dir_dr)
        dir_norm = math.hypot(dir_x, dir_y) or 1.0
        # Arc total in degrees -> half-angle in radians
        half_angle_rad = math.radians(max(1.0, min(360.0, self.pass_arc_degrees))) / 2.0
        cos_threshold = math.cos(half_angle_rad)

        # Check if any teammate is in arc
        team_ids = (
            self.offense_ids if passer_id in self.offense_ids else self.defense_ids
        )
        for pid in team_ids:
            if pid == passer_id:
                continue
            tq, tr = self.positions[pid]
            # Check if this teammate is in the arc
            vx, vy = self._axial_to_cartesian(tq - passer_pos[0], tr - passer_pos[1])
            vnorm = math.hypot(vx, vy)
            if vnorm == 0:
                continue
            cosang = (vx * dir_x + vy * dir_y) / (vnorm * dir_norm)
            if cosang >= cos_threshold:
                return True

        return False

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
        passer_pos = self.positions[passer_id]
        dir_dq, dir_dr = self.hex_directions[direction_idx]

        # Compute angles in cartesian space
        dir_x, dir_y = self._axial_to_cartesian(dir_dq, dir_dr)
        dir_norm = math.hypot(dir_x, dir_y) or 1.0
        # Arc total in degrees -> half-angle in radians
        half_angle_rad = (
            math.radians(max(1.0, min(360.0, getattr(self, "pass_arc_degrees", 60.0))))
            / 2.0
        )
        cos_threshold = math.cos(half_angle_rad)

        def in_arc(to_q: int, to_r: int) -> bool:
            vx, vy = self._axial_to_cartesian(
                to_q - passer_pos[0], to_r - passer_pos[1]
            )
            vnorm = math.hypot(vx, vy)
            if vnorm == 0:
                return False
            cosang = (vx * dir_x + vy * dir_y) / (vnorm * dir_norm)
            return cosang >= cos_threshold

        def in_defender_arc(to_q: int, to_r: int) -> bool:
            """Check if position is within 180° arc (chosen direction ± 1 adjacent direction)."""
            vx, vy = self._axial_to_cartesian(
                to_q - passer_pos[0], to_r - passer_pos[1]
            )
            vnorm = math.hypot(vx, vy)
            if vnorm == 0:
                return False
            cosang = (vx * dir_x + vy * dir_y) / (vnorm * dir_norm)
            # 180° arc means cos >= 0 (angles from -90° to +90°)
            return cosang >= 0.0

        # Pick closest teammate in arc
        team_ids = (
            self.offense_ids if passer_id in self.offense_ids else self.defense_ids
        )
        recv_id = None
        recv_dist = None
        for pid in team_ids:
            if pid == passer_id:
                continue
            tq, tr = self.positions[pid]
            if not in_arc(tq, tr):
                continue
            d = self._hex_distance(passer_pos, (tq, tr))
            if recv_id is None or d < recv_dist:
                recv_id = pid
                recv_dist = d

        if recv_id is None:
            # No teammate in arc: roll for OOB turnover; on non-turnover, treat as NOOP
            if self._rng.random() < float(getattr(self, "pass_oob_turnover_prob", 1.0)):
                # Determine first out-of-bounds cell along direction for logging
                step = 1
                target = passer_pos
                while True:
                    target = (
                        passer_pos[0] + dir_dq * step,
                        passer_pos[1] + dir_dr * step,
                    )
                    if not self._is_valid_position(*target):
                        break
                    step += 1
                self.ball_holder = None
                results["turnovers"].append(
                    {
                        "player_id": passer_id,
                        "reason": "pass_out_of_bounds",
                        "turnover_pos": target,
                    }
                )
                results["passes"][passer_id] = {
                    "success": False,
                    "reason": "out_of_bounds",
                }
            else:
                # No turnover -> fallback to NOOP (ball handler keeps the ball)
                results["passes"][passer_id] = {
                    "success": False,
                    "reason": "no_receiver_noop",
                }
            return

        # Line-of-sight based interception: defenders between passer and receiver
        # Use shared calculation for consistency between gameplay and observations
        recv_pos = self.positions[recv_id]
        pass_distance = self._hex_distance(passer_pos, recv_pos)
        
        total_steal_prob, defender_contributions = self._calculate_steal_probability_for_pass(
            passer_pos, recv_pos, passer_id
        )
        
        # Roll for interception
        if total_steal_prob > 0 and self._rng.random() < total_steal_prob:
            # Interception occurs - defender with highest steal contribution gets ball
            defender_contributions.sort(key=lambda t: t[1], reverse=True)
            thief_id = defender_contributions[0][0]
            
            self.ball_holder = thief_id
            results["turnovers"].append(
                {
                    "player_id": passer_id,
                    "reason": "intercepted",
                    "stolen_by": thief_id,
                    "turnover_pos": self.positions[thief_id],
                }
            )
            results["passes"][passer_id] = {
                "success": False,
                "reason": "intercepted",
                "interceptor_id": thief_id,
                "pass_distance": pass_distance,
                "total_steal_prob": total_steal_prob,
                "defenders_evaluated": [
                    {
                        "id": did,
                        "steal_contribution": contrib,
                        "perp_distance": perp_dist,
                        "position_on_line": pos_t,
                    }
                    for did, contrib, perp_dist, pos_t in defender_contributions
                ],
            }
            return

        # Successful pass to receiver in arc
        self.ball_holder = recv_id
        results["passes"][passer_id] = {
            "success": True,
            "target": recv_id,
            "pass_distance": pass_distance,
            "total_steal_prob": total_steal_prob,
            "defenders_evaluated": [
                {
                    "id": did,
                    "steal_contribution": contrib,
                    "perp_distance": perp_dist,
                    "position_on_line": pos_t,
                }
                for did, contrib, perp_dist, pos_t in defender_contributions
            ],
        }
        # Start/refresh assist window (configurable steps including current step)
        self._assist_candidate = {
            "passer_id": int(passer_id),
            "recipient_id": int(recv_id),
            "expires_at_step": int(self.step_count + self.assist_window),
        }
        return

    @profile_section("_compute_shot_pressure_multiplier")
    def _compute_shot_pressure_multiplier(
        self,
        shooter_id: Optional[int],
        shooter_pos: Tuple[int, int],
        distance_to_basket: int,
    ) -> float:
        """Compute multiplicative reduction to shot probability due to nearest defender
        in a forward arc toward the basket. Returns 1.0 if no qualifying defender.
        """
        if not self.shot_pressure_enabled or shooter_id is None:
            return 1.0

        # Direction from shooter to basket in cartesian
        dir_q = self.basket_position[0] - shooter_pos[0]
        dir_r = self.basket_position[1] - shooter_pos[1]
        dir_x, dir_y = self._axial_to_cartesian(dir_q, dir_r)
        dir_norm = math.hypot(dir_x, dir_y) or 1.0
        cos_threshold = math.cos(self.shot_pressure_arc_rad / 2.0)

        # Opposing team
        opp_ids = (
            self.defense_ids if shooter_id in self.offense_ids else self.offense_ids
        )

        # Track the defender that applies the most pressure (considering both distance and alignment)
        best_pressure_reduction: Optional[float] = None

        for did in opp_ids:
            dq = self.positions[did][0] - shooter_pos[0]
            dr = self.positions[did][1] - shooter_pos[1]
            vx, vy = self._axial_to_cartesian(dq, dr)
            if vx == 0 and vy == 0:
                continue
            vnorm = math.hypot(vx, vy)
            if vnorm == 0:
                continue
            cosang = (vx * dir_x + vy * dir_y) / (vnorm * dir_norm)
            in_arc = cosang >= cos_threshold
            d_def = self._hex_distance(shooter_pos, self.positions[did])
            # Defender must be at or closer than the basket along this direction
            if in_arc and d_def <= distance_to_basket:
                # Calculate pressure for this defender using both distance and angle alignment
                # cosang ranges from cos_threshold to 1.0, where 1.0 = perfectly aligned
                # Normalize to [0, 1] where 1.0 = perfectly aligned, 0 = at arc edge
                angle_factor = (
                    (cosang - cos_threshold) / (1.0 - cos_threshold)
                    if cos_threshold < 1.0
                    else 1.0
                )

                # Base pressure from distance: stronger when closer
                exponent_arg = d_def - 1
                distance_reduction = self.shot_pressure_max * math.exp(
                    -self.shot_pressure_lambda * exponent_arg
                )

                # Scale by angle alignment - more aligned = more pressure
                # Use a power to make alignment matter more (e.g., angle_factor^2)
                pressure_reduction = distance_reduction * (angle_factor**2)

                if (
                    best_pressure_reduction is None
                    or pressure_reduction > best_pressure_reduction
                ):
                    best_pressure_reduction = pressure_reduction

        if best_pressure_reduction is None:
            return 1.0

        return max(0.0, 1.0 - best_pressure_reduction)

    @profile_section("attempt_shot")
    def _attempt_shot(self, shooter_id: int) -> Dict:
        """Attempt a shot from the ball holder."""
        shooter_pos = self.positions[shooter_id]
        basket_pos = self.basket_position

        distance = self._hex_distance(shooter_pos, basket_pos)

        # Use the distance-based probability calculation
        shot_success_prob = self._calculate_shot_probability(shooter_id, distance)

        # For diagnostics: compute base probability (no pressure) and pressure multiplier
        d0 = 1
        d1 = max(self.three_point_distance, d0 + 1)
        # Use per-offense-player baselines if shooter is offense; else global baselines
        if shooter_id in self.offense_ids:
            idx = int(shooter_id)
            dunk_p = self.offense_dunk_pct_by_player[idx]
            layup_p = self.offense_layup_pct_by_player[idx]
            three_p = self.offense_three_pt_pct_by_player[idx]
        else:
            dunk_p = self.dunk_pct
            layup_p = self.layup_pct
            three_p = self.three_pt_pct
        if self.allow_dunks and distance == 0:
            base_prob = dunk_p
        elif distance <= d0:
            base_prob = layup_p
        else:
            t = (distance - d0) / (d1 - d0)
            base_prob = layup_p + (three_p - layup_p) * t
        # Clamp similar to backend
        base_prob = max(0.01, min(0.99, base_prob))

        pressure_mult = self._compute_shot_pressure_multiplier(
            shooter_id, shooter_pos, distance
        )

        # Sample RNG and decide outcome
        rng_u = self._rng.random()
        shot_made = rng_u < shot_success_prob

        if not shot_made:
            # Missed shot - possession ends
            self.ball_holder = None

        is_three = self._is_three_point_hex(tuple(shooter_pos))

        return {
            "success": shot_made,
            "distance": distance,
            "probability": shot_success_prob,
            "rng": rng_u,
            "base_probability": base_prob,
            "pressure_multiplier": pressure_mult,
            "is_three": bool(is_three),
        }

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
        """Calculate probability of successful shot using a simple linear model
        anchored at layup (distance 1) and three-point (distance = three_point_distance).
        Beyond the arc, we linearly extrapolate and clamp to [0.01, 0.99].
        """
        # Anchors
        d0 = 1
        # Fix: When three_point_distance is float (e.g., 4.75), max() needs comparable types.
        # Use ceil() for integer hex distance logic, or just keep as float for interpolation.
        # Here we need the 'distance' value that corresponds to the 3pt line anchor.
        # Since 'distance' argument is integer hex distance, we should round up or use the raw float.
        # Let's use the raw float for the anchor to allow sub-hex precision in the probability curve.
        d1 = max(float(self.three_point_distance), float(d0 + 1))
        # Use per-offense-player anchors if applicable
        if shooter_id in self.offense_ids:
            idx = int(shooter_id)
            p0 = float(self.offense_layup_pct_by_player[idx])
            p1 = float(self.offense_three_pt_pct_by_player[idx])
            dunk_p = float(self.offense_dunk_pct_by_player[idx])
        else:
            p0 = float(self.layup_pct)
            p1 = float(self.three_pt_pct)
            dunk_p = float(self.dunk_pct)

        if self.allow_dunks and distance == 0:
            prob = dunk_p
        elif distance <= d0:
            prob = p0
        else:
            t = (distance - d0) / (d1 - d0)
            prob = (
                p0 + (p1 - p0) * t
            )  # linear interpolation (or extrapolation if distance>d1)

        # Apply defender shot pressure if any qualifying defender is between shooter and basket
        if self.shot_pressure_enabled and shooter_id is not None:
            shooter_pos = self.positions[shooter_id]
            pressure_mult = self._compute_shot_pressure_multiplier(
                shooter_id, shooter_pos, distance
            )
            prob *= pressure_mult

        # Clamp to sensible bounds
        prob = max(0.01, min(0.99, prob))
        return float(prob)

    @profile_section("_check_termination_and_rewards")
    def _check_termination_and_rewards(
        self, action_results: Dict
    ) -> Tuple[bool, np.ndarray]:
        """Check if episode should terminate and calculate rewards."""
        rewards = np.zeros(self.n_players)
        done = False
        pass_reward = self.pass_reward
        violation_reward = self.violation_reward
        turnover_penalty = self.turnover_penalty
        # Assist shaping
        potential_assist_reward = self.potential_assist_reward
        full_assist_bonus = self.full_assist_bonus
        # Define the reward magnitude for shots (2PT vs 3PT)
        made_shot_reward_inside = self.made_shot_reward_inside
        made_shot_reward_three = self.made_shot_reward_three
        missed_shot_penalty = self.missed_shot_penalty

        # --- Reward successful passes ---
        for _, pass_result in action_results.get("passes", {}).items():
            if pass_result.get("success"):
                rewards[self.offense_ids] += pass_reward / self.players_per_side
                rewards[self.defense_ids] -= pass_reward / self.players_per_side

        # --- Handle all turnovers from actions ---
        if action_results.get("turnovers"):
            done = True
            # Penalize offense, reward defense for the turnover
            # We assume only one turnover can happen per step
            rewards[self.offense_ids] -= turnover_penalty / self.players_per_side
            rewards[self.defense_ids] += turnover_penalty / self.players_per_side

        # --- Handle defensive lane violations (illegal defense) ---
        # Only apply violation reward if there's no shot on this step
        # (to avoid double-rewarding offense when episode ends anyway)
        if action_results.get("defensive_lane_violations") and not action_results.get("shots"):
            done = True
            # Defense committed a violation, offense gets a point (like a technical free throw)
            # Reward offense, penalize defense
            violation_reward = self.violation_reward
            rewards[self.offense_ids] += violation_reward / self.players_per_side
            rewards[self.defense_ids] -= violation_reward / self.players_per_side

        # Check for shots
        for player_id, shot_result in action_results.get("shots", {}).items():
            done = True  # Episode ends after any shot attempt

            # Compute distance to basket and value of the attempted shot
            shooter_pos = self.positions[player_id]
            dist_to_basket = self._hex_distance(shooter_pos, self.basket_position)
            is_three_point = self._is_three_point_hex(tuple(shooter_pos))

            if shot_result["success"]:
                # Basket was made
                made_shot_reward = (
                    made_shot_reward_three
                    if is_three_point
                    else made_shot_reward_inside
                )
                # Offense scored, good for them, bad for defense
                rewards[self.offense_ids] += made_shot_reward / self.players_per_side
                rewards[self.defense_ids] -= made_shot_reward / self.players_per_side
                # else: handle rare case of defense scoring on own basket
            else:
                # Offense missed, bad for them, good for defense
                rewards[self.offense_ids] -= missed_shot_penalty / self.players_per_side
                rewards[self.defense_ids] += missed_shot_penalty / self.players_per_side

            # --- Assist rewards and annotations ---
            assist_potential = False
            assist_full = False
            assist_passer: Optional[int] = None
            if self._assist_candidate is not None and self._assist_candidate.get(
                "recipient_id"
            ) == int(player_id):
                if self.step_count <= int(
                    self._assist_candidate.get("expires_at_step", -1)
                ):
                    assist_potential = True
                    assist_passer = int(self._assist_candidate["passer_id"])
                    # Reward potential assist
                    # Prefer % of shot reward if configured; otherwise fall back to absolute
                    if shot_result["success"]:
                        base_for_pct = made_shot_reward
                    else:
                        # If missed, use would-be shot reward magnitude as base.
                        # Recompute distance locally to avoid scope issues.
                        _dist_for_pct = self._hex_distance(
                            self.positions[player_id], self.basket_position
                        )
                        base_for_pct = (
                            made_shot_reward_three if is_three_point else made_shot_reward_inside
                        )
                    potential_assist_amt = (
                        max(0.0, float(self.potential_assist_pct) * float(base_for_pct))
                        if hasattr(self, "potential_assist_pct")
                        and self.potential_assist_pct is not None
                        else potential_assist_reward
                    )
                    # Assist shaping is symmetric: reward offense, penalize defense
                    rewards[self.offense_ids] += (
                        potential_assist_amt / self.players_per_side
                    )
                    rewards[self.defense_ids] -= (
                        potential_assist_amt / self.players_per_side
                    )
                    if shot_result["success"]:
                        assist_full = True
                        # Full assist bonus proportional to made shot reward if configured
                        full_bonus_amt = (
                            max(
                                0.0,
                                float(self.full_assist_bonus_pct)
                                * float(made_shot_reward),
                            )
                            if hasattr(self, "full_assist_bonus_pct")
                            and self.full_assist_bonus_pct is not None
                            else full_assist_bonus
                        )
                        # Full assist bonus is symmetric as well
                        rewards[self.offense_ids] += (
                            full_bonus_amt / self.players_per_side
                        )
                        rewards[self.defense_ids] -= (
                            full_bonus_amt / self.players_per_side
                        )
            # Annotate shot result for evaluation
            shot_result["assist_potential"] = bool(assist_potential)
            shot_result["assist_full"] = bool(assist_full)
            shot_result["is_three"] = bool(is_three_point)
            if assist_passer is not None:
                shot_result["assist_passer_id"] = assist_passer
            # Clear assist window after a shot attempt (episode ends anyway)
            self._assist_candidate = None

        return done, rewards

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
        player_pos = self.positions[player_id]
        dist = self._hex_distance(player_pos, self.basket_position)
        # Shot value: 3 if at/behind arc and not a dunk; else 2
        if self.allow_dunks and dist == 0:
            shot_value = 2.0
        else:
            shot_value = 3.0 if self._is_three_point_hex(tuple(player_pos)) else 2.0
        p_make = float(self._calculate_shot_probability(player_id, dist))
        return float(shot_value * p_make)

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
        # If no ball holder (e.g., post-shot terminal before reset), fall back to 0
        if self.ball_holder is None:
            return 0.0
        # Determine which team's opportunities to evaluate: the team in possession
        team_ids = (
            self.offense_ids
            if (self.ball_holder in self.offense_ids)
            else self.defense_ids
        )

        if self.phi_use_ball_handler_only:
            return self._calculate_expected_points_for_player(int(self.ball_holder))

        # Compute ball-handler EP and aggregate teammate EPs based on mode
        ball_ep = self._calculate_expected_points_for_player(int(self.ball_holder))
        ball_holder_id = int(self.ball_holder)

        # Collect teammate EPs (may or may not exclude ball handler depending on mode)
        mode = getattr(self, "phi_aggregation_mode", "team_best")

        if mode == "team_avg":
            # Simple average of all players (including ball handler)
            eps = [self._calculate_expected_points_for_player(int(pid)) for pid in team_ids]
            return float(sum(eps) / max(1, len(eps)))

        # For other modes, separate ball handler from teammates
        teammate_eps = [
            self._calculate_expected_points_for_player(int(pid)) for pid in team_ids if pid != ball_holder_id
        ]

        if not teammate_eps:  # No teammates (1v1 or edge case)
            return ball_ep

        # Aggregate teammate EPs based on mode
        if mode == "teammates_best":
            teammate_aggregate = max(teammate_eps)
        elif mode == "teammates_avg":
            teammate_aggregate = sum(teammate_eps) / len(teammate_eps)
        elif mode == "teammates_worst":
            teammate_aggregate = min(teammate_eps)
        elif mode == "team_worst":
            # Include ball handler in the "worst" calculation
            teammate_aggregate = min(min(teammate_eps), ball_ep)
        else:  # "team_best" (default/legacy behavior)
            # Include ball handler in the "best" calculation
            teammate_aggregate = max(max(teammate_eps), ball_ep)

        # Blend teammate aggregate with ball handler EP
        w = float(max(0.0, min(1.0, getattr(self, "phi_blend_weight", 0.0))))
        blended = (1.0 - w) * float(teammate_aggregate) + w * float(ball_ep)
        return float(blended)

    @profile_section("_phi_ep_breakdown")
    def _phi_ep_breakdown(self) -> Tuple[float, float]:
        """Return (team_best_ep, ball_handler_ep) for current possession team."""
        if self.ball_holder is None:
            return 0.0, 0.0
        team_ids = (
            self.offense_ids
            if (self.ball_holder in self.offense_ids)
            else self.defense_ids
        )
        team_best = 0.0
        ball_ep = 0.0
        for pid in team_ids:
            ep = self._calculate_expected_points_for_player(pid)
            if pid == self.ball_holder:
                ball_ep = ep
            if ep > team_best:
                team_best = ep
        return float(team_best), float(ball_ep)

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

    @profile_section("_get_player_distances")
    def _get_player_distances(self, base_id: int, target_ids: List[int]) -> np.ndarray:
        """Return hex distances from `base_id` to each ID in `target_ids`."""
        if not self.positions or base_id >= len(self.positions):
            return np.zeros(0, dtype=np.float32)

        base_pos = self.positions[base_id]
        distances = [
            float(self._hex_distance(base_pos, self.positions[target_id]))
            for target_id in target_ids
        ]
        return np.array(distances, dtype=np.float32)

    @profile_section("_get_player_angles")
    def _get_player_angles(self, base_id: int, target_ids: List[int]) -> np.ndarray:
        """Return cosine angles between base→target and base→basket for each target."""
        if not self.positions or base_id >= len(self.positions):
            return np.zeros(0, dtype=np.float32)

        base_pos = self.positions[base_id]
        to_basket_q = self.basket_position[0] - base_pos[0]
        to_basket_r = self.basket_position[1] - base_pos[1]
        basket_mag_sq = to_basket_q**2 + to_basket_r**2 + to_basket_q * to_basket_r
        basket_mag = math.sqrt(max(0.0, basket_mag_sq))

        angles: List[float] = []
        for target_id in target_ids:
            target_pos = self.positions[target_id]
            to_target_q = target_pos[0] - base_pos[0]
            to_target_r = target_pos[1] - base_pos[1]
            target_mag_sq = (
                to_target_q**2 + to_target_r**2 + to_target_q * to_target_r
            )
            target_mag = math.sqrt(max(0.0, target_mag_sq))

            if basket_mag < 1e-6 or target_mag < 1e-6:
                angles.append(0.0)
                continue

            dot = (
                to_basket_q * to_target_q
                + to_basket_r * to_target_r
                + 0.5 * (to_basket_q * to_target_r + to_basket_r * to_target_q)
            )
            cos_angle = dot / (basket_mag * target_mag)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angles.append(float(cos_angle))

        return np.array(angles, dtype=np.float32)

    
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
        return self._collect_pairwise_features(
            self.offense_ids,
            self.defense_ids,
            self._get_player_distances,
        )

    @profile_section("_calculate_offense_defense_angles")
    def _calculate_offense_defense_angles(self) -> np.ndarray:
        return self._collect_pairwise_features(
            self.offense_ids,
            self.defense_ids,
            self._get_player_angles,
        )

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
        offense_distances = self._collect_teammate_features(
            self.offense_ids, self._get_player_distances
        )
        defense_distances = self._collect_teammate_features(
            self.defense_ids, self._get_player_distances
        )
        return np.concatenate((offense_distances, defense_distances)) if offense_distances.size or defense_distances.size else np.array([], dtype=np.float32)

    @profile_section("_calculate_teammate_angles")
    def _calculate_teammate_angles(self) -> np.ndarray:
        offense_angles = self._collect_teammate_features(
            self.offense_ids, self._get_player_angles
        )
        defense_angles = self._collect_teammate_features(
            self.defense_ids, self._get_player_angles
        )
        return np.concatenate((offense_angles, defense_angles)) if offense_angles.size or defense_angles.size else np.array([], dtype=np.float32)

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
        obs: List[float] = []

        # Normalization factor to put coordinates roughly in [-1, 1]
        # Using the larger of width/height bounds axial deltas conservatively
        norm_den: float = float(max(self.court_width, self.court_height)) or 1.0
        if not self.normalize_obs:
            norm_den = 1.0

        # Player positions in absolute coordinates, normalized
        for q, r in self.positions:
            obs.extend([q / norm_den, r / norm_den])

        # One-hot encode the ball holder
        ball_holder_one_hot = np.zeros(self.n_players, dtype=np.float32)
        if self.ball_holder is not None:
            ball_holder_one_hot[self.ball_holder] = 1.0
        obs.extend(ball_holder_one_hot.tolist())

        # Shot clock (kept unnormalized)
        obs.append(float(self.shot_clock))

        # Team encoding: per-player team identification (+1 for offense, -1 for defense)
        # Format: [team_0, team_1, ..., team_n] where team_i ∈ {+1, -1}
        # Example (2v2): [+1, +1, -1, -1] means players 0,1 are offense, players 2,3 are defense
        for pid in range(self.n_players):
            if pid in self.offense_ids:
                obs.append(1.0)  # Offense
            else:
                obs.append(-1.0)  # Defense

        # Absolute position of the ball handler (allows network to distinguish court regions)
        # This helps the network learn different strategies for center court vs. sidelines
        if self.ball_holder is not None:
            ball_handler_q, ball_handler_r = self.positions[self.ball_holder]
            obs.extend([ball_handler_q / norm_den, ball_handler_r / norm_den])
        else:
            # If no ball holder (terminal state), use basket position as reference
            obs.extend([self.basket_position[0] / norm_den, self.basket_position[1] / norm_den])

        # Hoop vector in absolute coordinates (optional)
        if self.include_hoop_vector:
            hoop_q, hoop_r = self.basket_position
            obs.extend([hoop_q / norm_den, hoop_r / norm_den])

        # All-pairs offense-defense distances
        # For each offensive player, distance to each defender
        # Shape: (players_per_side * players_per_side,)
        # For 3v3: [O0→D0, O0→D1, O0→D2, O1→D0, O1→D1, O1→D2, O2→D0, O2→D1, O2→D2]
        distances = self._calculate_offense_defense_distances()
        if self.normalize_obs:
            distances = distances / norm_den
        obs.extend(distances.tolist())
        
        # All-pairs offense-defense angle cosines
        # For each offensive player, cos(angle) between defender and basket direction
        # Shape: (players_per_side * players_per_side,)
        # Values in [-1, 1]: +1 = defender in front, 0 = perpendicular, -1 = behind
        angles = self._calculate_offense_defense_angles()
        obs.extend(angles.tolist())
        
        # Teammate distances / angles
        teammate_distances = self._calculate_teammate_distances()
        if self.normalize_obs:
            teammate_distances = teammate_distances / norm_den
        obs.extend(teammate_distances.tolist())

        teammate_angles = self._calculate_teammate_angles()
        obs.extend(teammate_angles.tolist())
        
        # Lane step counts for all players (offensive and defensive)
        # This allows agents to learn to manage their time in the lane
        # Offensive players track time in offensive lane, defensive players track time in defensive lane (basket)
        for pid in range(self.n_players):
            if pid in self.offense_ids:
                lane_steps = self._offensive_lane_steps.get(pid, 0)
            else:
                lane_steps = self._defender_in_key_steps.get(pid, 0)
            obs.append(float(lane_steps))
        
        # Expected Points (EP) for each offensive player
        # Pressure-adjusted expected value of a shot from their current position
        # Only offensive players have EP since they're the ones who can score
        ep_values = self.calculate_expected_points_all_players()
        obs.extend(ep_values.tolist())
        
        # Turnover probability for each offensive player (fixed-position encoding)
        # Position i corresponds to offensive player i
        # Non-zero only for the current ball handler
        turnover_probs = np.zeros(self.players_per_side, dtype=np.float32)
        if self.ball_holder is not None and self.ball_holder in self.offense_ids:
            ball_holder_idx = self.offense_ids.index(self.ball_holder)
            turnover_prob = self.calculate_defender_pressure_turnover_probability()
            turnover_probs[ball_holder_idx] = float(turnover_prob)
        obs.extend(turnover_probs.tolist())
        
        # Steal risks for each offensive player (fixed-position encoding)
        # Position i corresponds to offensive player i
        # Non-zero for potential pass receivers, 0 for ball holder and if no ball
        steal_risks = np.zeros(self.players_per_side, dtype=np.float32)
        if self.ball_holder is not None and self.ball_holder in self.offense_ids:
            steal_probs_dict = self.calculate_pass_steal_probabilities(self.ball_holder)
            for offense_id in self.offense_ids:
                if offense_id != self.ball_holder:
                    offense_idx = self.offense_ids.index(offense_id)
                    steal_prob = steal_probs_dict.get(offense_id, 0.0)
                    steal_risks[offense_idx] = float(steal_prob)
        obs.extend(steal_risks.tolist())

        return np.array(obs, dtype=np.float32)

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
            return self._render_ascii()
        elif self.render_mode == "rgb_array":
            return self._render_visual()

    @profile_section("_render_ascii")
    def _render_ascii(self):
        """Simple ASCII rendering for training."""
        print(f"\nShot Clock: {self.shot_clock}")
        print(f"Ball Holder: Player {self.ball_holder}")

        grid = [
            [" · " for _ in range(self.court_width)] for _ in range(self.court_height)
        ]

        # Place players and basket
        for r in range(self.court_height):
            for c in range(self.court_width):
                q, r_ax = self._offset_to_axial(c, r)

                if (q, r_ax) == self.basket_position:
                    grid[r][c] = " B "

                for i, pos in enumerate(self.positions):
                    if pos == (q, r_ax):
                        symbol = f"O{i}" if i in self.offense_ids else f"D{i}"
                        if i == self.ball_holder:
                            symbol = f"*{symbol[1]}*"
                        grid[r][c] = f" {symbol} "

        # Print grid
        print("\nCourt Layout (O=Offense, D=Defense, *=Ball):")
        for r_idx, row in enumerate(grid):
            # Indent odd rows for hex layout
            indent = " " if r_idx % 2 != 0 else ""
            print(indent + "".join(row))

        if self.last_action_results:
            print(f"\nLast Action Results: {self.last_action_results}")
        print("-" * 40)

    @profile_section("_render_visual")
    def _render_visual(self):
        """Visual rendering using matplotlib."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import RegularPolygon
        import io
        from PIL import Image

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_aspect("equal")

        # Convert axial coordinates to cartesian for pointy-topped hexes
        def axial_to_cartesian(q, r):
            size = 1.0  # Defines the size (radius) of the hexagons
            x = size * (math.sqrt(3) * q + math.sqrt(3) / 2 * r)
            y = size * (3.0 / 2.0 * r)
            return x, y

        hex_radius = 1.0

        # Helper: map standardized delta (in units of std dev) to a color (blue→green→orange)
        def _lerp(c1, c2, t):
            return (
                c1[0] + (c2[0] - c1[0]) * t,
                c1[1] + (c2[1] - c1[1]) * t,
                c1[2] + (c2[2] - c1[2]) * t,
            )

        def color_from_zscore(delta: float, std: float, zmax: float = 2.0):
            # Convert delta to z = (sample - baseline) / std; handle std≈0
            if std is None or std <= 1e-9:
                z = 0.0
            else:
                z = float(delta) / float(std)
            # Clamp to [−zmax, +zmax]
            if z < -zmax:
                z = -zmax
            elif z > zmax:
                z = zmax
            # Anchors
            blue = (0.20, 0.40, 0.90)
            green = (0.20, 0.70, 0.20)
            orange = (1.00, 0.60, 0.00)
            if z < 0.0:
                # Map [−zmax, 0] → [blue, green]
                t = 1.0 - (abs(z) / zmax)  # z=-zmax→0, z=0→1
                return _lerp(blue, green, t)
            else:
                # Map [0, +zmax] → [green, orange]
                t = z / zmax  # z=0→0, z=+zmax→1
                return _lerp(green, orange, t)

        # Draw hexagonal grid for the entire court
        for c in range(self.court_width):
            for r in range(self.court_height):
                q, r_ax = self._offset_to_axial(c, r)
                x, y = axial_to_cartesian(q, r_ax)

                # Draw the base grid hexagon
                hexagon = RegularPolygon(
                    (x, y),
                    numVertices=6,
                    radius=hex_radius,
                    orientation=0,  # for pointy-topped
                    facecolor="lightgray",
                    edgecolor="white",
                    alpha=0.5,
                    linewidth=1,
                )
                ax.add_patch(hexagon)

                # Paint the lane area with light red if violations are enabled
                # Check if lane hexes exist (for backward compatibility with old environments)
                if hasattr(self, 'offensive_lane_hexes') and self.offensive_lane_hexes:
                    is_in_lane = (q, r_ax) in self.offensive_lane_hexes
                    offense_enabled = getattr(self, 'offensive_three_seconds_enabled', False)
                    defense_enabled = getattr(self, 'illegal_defense_enabled', False)
                    
                    if is_in_lane and (offense_enabled or defense_enabled):
                        lane_hexagon = RegularPolygon(
                            (x, y),
                            numVertices=6,
                            radius=hex_radius,
                            orientation=0,
                            facecolor=(1.0, 0.39, 0.39, 0.15),  # Light red with alpha
                            edgecolor=(1.0, 0.39, 0.39, 0.3),   # Light red edge
                            linewidth=1.5,
                            zorder=2,
                        )
                        ax.add_patch(lane_hexagon)

                # For the basket, add a thick red ring around it
                if (q, r_ax) == self.basket_position:
                    basket_ring = plt.Circle(
                        (x, y),
                        hex_radius * 1.05,
                        fill=False,
                        edgecolor="red",
                        linewidth=4,
                        zorder=6,
                    )
                    ax.add_patch(basket_ring)

                # Paint the three-point line
                if (q, r_ax) in self._three_point_line_hexes:
                    tp_outline = RegularPolygon(
                        (x, y),
                        numVertices=6,
                        radius=hex_radius,
                        orientation=0,
                        facecolor="none",
                        edgecolor="red",
                        linewidth=2.5,
                        zorder=7,
                    )
                    ax.add_patch(tp_outline)

        # Draw players by filling their hexagon
        for i, (q, r) in enumerate(self.positions):
            x, y = axial_to_cartesian(q, r)
            color = "blue" if i in self.offense_ids else "red"

            player_hexagon = RegularPolygon(
                (x, y),
                numVertices=6,
                radius=hex_radius,
                orientation=0,  # for pointy-topped
                facecolor=color,
                edgecolor="white",
                alpha=0.9,
                zorder=10,
            )
            ax.add_patch(player_hexagon)
            ax.text(
                x,
                y,
                str(i),
                ha="center",
                va="center",
                fontsize=24,
                fontweight="bold",
                color="white",
                zorder=11,
            )

            if i == self.ball_holder:
                ball_ring = plt.Circle(
                    (x, y),
                    hex_radius * 0.9,
                    fill=False,
                    color="orange",
                    linewidth=4,
                    zorder=12,
                )
                ax.add_patch(ball_ring)

            # Baseline skill labels at the three lower vertices (Offense only),
            # with background color coded by (sampled − baseline) delta.
            try:
                if i in self.offense_ids:
                    lay = float(self.offense_layup_pct_by_player[int(i)])
                    three = float(self.offense_three_pt_pct_by_player[int(i)])
                    dunk = float(self.offense_dunk_pct_by_player[int(i)])
                    # Baselines
                    base_lay = float(self.layup_pct)
                    base_three = float(self.three_pt_pct)
                    base_dunk = float(self.dunk_pct)
                    # Deltas
                    d_lay = lay - base_lay
                    d_three = three - base_three
                    d_dunk = dunk - base_dunk

                    rscale = hex_radius * 1.05
                    sqrt3 = math.sqrt(3.0)
                    verts = [
                        # bottom-left: Dunk value with dunk delta color
                        (
                            x - (sqrt3 / 2.0) * rscale,
                            y - 0.5 * rscale,
                            f"{int(round(dunk * 100))}%D",
                            color_from_zscore(d_dunk, float(self.dunk_std)),
                        ),
                        # bottom-center: Layup value with layup delta color
                        (
                            x + 0.0,
                            y - 1.0 * rscale,
                            f"{int(round(lay * 100))}%L",
                            color_from_zscore(d_lay, float(self.layup_std)),
                        ),
                        # bottom-right: Three value with three-point delta color
                        (
                            x + (sqrt3 / 2.0) * rscale,
                            y - 0.5 * rscale,
                            f"{int(round(three * 100))}%3",
                            color_from_zscore(d_three, float(self.three_pt_std)),
                        ),
                    ]

                    for vx, vy, text, fc_col in verts:
                        ax.text(
                            vx,
                            vy,
                            text,
                            ha="center",
                            va="center",
                            fontsize=9,
                            fontweight="bold",
                            color="white",
                            bbox=dict(
                                boxstyle="round,pad=0.15",
                                fc=fc_col,
                                ec=fc_col,
                                alpha=0.85,
                            ),
                            zorder=13,
                        )
            except Exception:
                pass

        # --- Draw pass arrow (successful pass in the last action) ---
        try:
            if self.last_action_results and self.last_action_results.get("passes"):
                for passer_id_str, pass_res in self.last_action_results[
                    "passes"
                ].items():
                    if pass_res.get("success") and "target" in pass_res:
                        passer_id = int(passer_id_str)
                        target_id = int(pass_res.get("target"))
                        pq, pr = self.positions[passer_id]
                        tq, tr = self.positions[target_id]
                        x1, y1 = axial_to_cartesian(pq, pr)
                        x2, y2 = axial_to_cartesian(tq, tr)
                        # Solid black arrow from passer to receiver
                        ax.annotate(
                            "",
                            xy=(x2, y2),
                            xytext=(x1, y1),
                            arrowprops=dict(
                                arrowstyle="->", color="black", linewidth=3
                            ),
                            zorder=19,
                        )
        except Exception:
            pass

        # --- Annotate per-offensive-player shot make percentages (with pressure) ---
        try:
            for oid in self.offense_ids:
                q, r = self.positions[oid]
                # Distance to basket in hex metric
                dist = self._hex_distance((q, r), self.basket_position)
                # Use the env's probability model (accounts for player skill and pressure)
                prob = float(self._calculate_shot_probability(int(oid), int(dist)))
                pct_text = f"{int(round(prob * 100))}%"
                x, y = axial_to_cartesian(q, r)
                # Offset label slightly above-right of the player
                tx = x + hex_radius * 0.0
                ty = y + hex_radius * 1.2
                ax.text(
                    tx,
                    ty,
                    pct_text,
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    color="white",
                    bbox=dict(
                        boxstyle="round,pad=0.2", fc="black", ec="black", alpha=0.7
                    ),
                    zorder=13,
                )
        except Exception:
            pass

        # Calculate court boundaries to set axis limits
        cartesian_coords = [
            axial_to_cartesian(*self._offset_to_axial(c, r))
            for c in range(self.court_width)
            for r in range(self.court_height)
        ]
        x_coords = [c[0] for c in cartesian_coords]
        y_coords = [c[1] for c in cartesian_coords]

        margin = 2.0
        ax.set_xlim(min(x_coords) - margin, max(x_coords) + margin)
        ax.set_ylim(min(y_coords) - margin, max(y_coords) + margin)

        ax.set_title(
            f"Hexagon Basketball {self.players_per_side}v{self.players_per_side}"
        )
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

        # --- Draw Final Action Result Indicators ---
        if self.episode_ended and self.last_action_results:
            # Shot results
            if self.last_action_results.get("shots"):
                shot_result = list(self.last_action_results["shots"].values())[0]
                basket_x, basket_y = axial_to_cartesian(*self.basket_position)
                # Determine Dunk vs 2PT/3PT
                shooter_id = list(self.last_action_results["shots"].keys())[0]
                shooter_pos = self.positions[int(shooter_id)]
                dist_to_basket = self._hex_distance(shooter_pos, self.basket_position)
                is_dunk = shot_result.get("distance") == 0
                if is_dunk:
                    label_text = "Dunk"
                else:
                    is_three = self._is_three_point_hex(tuple(shooter_pos))
                    label_text = "3" if is_three else "2"

                if shot_result["success"]:
                    ax.add_patch(
                        plt.Circle(
                            (basket_x, basket_y),
                            hex_radius,
                            color="green",
                            alpha=0.7,
                            zorder=20,
                        )
                    )
                    ax.text(
                        0.5,
                        0.9,
                        f"Made {label_text}!",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=50,
                        fontweight="bold",
                        color="green",
                        alpha=0.9,
                    )
                else:
                    ax.add_patch(
                        plt.Circle(
                            (basket_x, basket_y),
                            hex_radius,
                            color="red",
                            alpha=0.7,
                            zorder=20,
                        )
                    )
                    ax.text(
                        0.5,
                        0.9,
                        f"Missed {label_text}!",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=50,
                        fontweight="bold",
                        color="red",
                        alpha=0.9,
                    )

            # Turnover results
            turnovers = self.last_action_results.get("turnovers", [])
            if self.last_action_results.get("passes"):
                for pass_res in self.last_action_results["passes"].values():
                    if pass_res.get("turnover"):
                        turnovers.append(pass_res)

            if turnovers:
                first_turnover = turnovers[0]
                turnover_pos = first_turnover.get("turnover_pos")
                if turnover_pos:
                    tx, ty = axial_to_cartesian(*turnover_pos)
                    ax.text(
                        tx,
                        ty,
                        "X",
                        ha="center",
                        va="center",
                        fontsize=60,
                        fontweight="bold",
                        color="darkred",
                        zorder=21,
                    )

                # Map backend reason codes to short labels for the banner
                reason_code = first_turnover.get("reason", "")
                reason_map = {
                    "defender_pressure": "PRESSURE",
                    "pass_out_of_bounds": "OOB",
                    "move_out_of_bounds": "OOB",
                    "intercepted": "STEAL",
                }
                reason_label = reason_map.get(
                    reason_code, reason_code.replace("_", " ").upper()
                )
                ax.text(
                    0.5,
                    0.9,
                    f"TOV - {reason_label}!",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=50,
                    fontweight="bold",
                    color="darkred",
                    alpha=0.9,
                )

            # Shot clock violation
            elif self.shot_clock <= 0:
                ax.text(
                    0.5,
                    0.9,
                    "SHOT CLOCK VIOLATION",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=40,
                    fontweight="bold",
                    color="darkred",
                    alpha=0.9,
                )

        # Add shot clock text to the bottom right corner
        ax.text(
            0.95,
            0.05,
            f"{self.shot_clock}",
            fontsize=48,
            fontweight="bold",
            color="black",
            ha="right",
            va="bottom",
            alpha=0.5,
            transform=ax.transAxes,
        )

        fig.canvas.draw()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        buf.seek(0)

        img = Image.open(buf)
        rgb_array = np.array(img)
        plt.close(fig)
        buf.close()

        return rgb_array

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
        opp_ids = (
            self.defense_ids if passer_id in self.offense_ids else self.offense_ids
        )
        
        pass_distance = self._hex_distance(passer_pos, recv_pos)
        
        # Calculate vector from passer to receiver for arc checking
        recv_dx = recv_pos[0] - passer_pos[0]
        recv_dy = recv_pos[1] - passer_pos[1]
        recv_norm = math.hypot(*self._axial_to_cartesian(recv_dx, recv_dy))
        
        if recv_norm < 1e-6:
            return 0.0, []
        
        recv_cart = self._axial_to_cartesian(recv_dx, recv_dy)
        
        # Evaluate each defender's steal contribution
        defender_contributions: List[Tuple[int, float, float, float]] = []
        
        for did in opp_ids:
            defender_pos = self.positions[did]
            
            # Check if defender is in forward hemisphere (toward receiver)
            def_dx = defender_pos[0] - passer_pos[0]
            def_dy = defender_pos[1] - passer_pos[1]
            
            def_cart = self._axial_to_cartesian(def_dx, def_dy)
            def_norm = math.hypot(*def_cart)
            
            if def_norm < 1e-6:
                continue
            
            # Dot product: only consider defenders in forward hemisphere (cosine >= 0)
            cosang = (def_cart[0] * recv_cart[0] + def_cart[1] * recv_cart[1]) / (def_norm * recv_norm)
            
            if cosang < 0.0:
                continue
            
            # Only consider defenders between passer and receiver
            if not self._is_between_points(defender_pos, passer_pos, recv_pos):
                continue
            
            # Calculate perpendicular distance from pass line
            perp_distance = self._point_to_line_distance(defender_pos, passer_pos, recv_pos)
            
            # Calculate position along pass line (0 = at passer, 1 = at receiver)
            position_t = self._get_position_on_line(defender_pos, passer_pos, recv_pos)
            
            # Position weight: defenders closer to receiver are more dangerous
            position_weight = self.steal_position_weight_min + (1.0 - self.steal_position_weight_min) * position_t
            
            # Calculate steal contribution for this defender
            steal_contrib = (
                self.base_steal_rate *
                math.exp(-self.steal_perp_decay * perp_distance) *
                (1.0 + self.steal_distance_factor * pass_distance) *
                position_weight
            )
            
            # Clamp to [0, 1] for safety
            steal_contrib = max(0.0, min(1.0, steal_contrib))
            
            defender_contributions.append((did, steal_contrib, perp_distance, position_t))
        
        # Compound steal probabilities: total = 1 - ∏(1 - steal_i)
        total_steal_prob = 0.0
        if defender_contributions:
            complement_product = 1.0
            for _, steal_contrib, _, _ in defender_contributions:
                complement_product *= (1.0 - steal_contrib)
            total_steal_prob = 1.0 - complement_product
        
        return total_steal_prob, defender_contributions

    @profile_section("calculate_pass_steal_probabilities")
    def calculate_pass_steal_probabilities(self, passer_id: int) -> Dict[int, float]:
        """
        Calculate hypothetical steal probabilities for passes to each teammate.
        Returns a dict mapping teammate_id -> steal_probability.
        """
        if self.ball_holder != passer_id:
            return {}
        
        passer_pos = self.positions[passer_id]
        team_ids = (
            self.offense_ids if passer_id in self.offense_ids else self.defense_ids
        )
        
        steal_probs = {}
        
        for teammate_id in team_ids:
            if teammate_id == passer_id:
                continue
            
            recv_pos = self.positions[teammate_id]
            # Use shared calculation for consistency between gameplay and observations
            total_steal_prob, _ = self._calculate_steal_probability_for_pass(
                passer_pos, recv_pos, passer_id
            )
            steal_probs[teammate_id] = total_steal_prob
        
        return steal_probs

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
        if self.ball_holder is None:
            return []
        
        if self.ball_holder not in self.offense_ids:
            return []  # Only offensive ball handlers face defender pressure
        
        # Get distances and angles for the ball handler
        distances = self._get_player_distances(self.ball_holder, self.defense_ids)
        angles = self._get_player_angles(self.ball_holder, self.defense_ids)
        
        defender_pressure_info = []
        
        for i, def_id in enumerate(self.defense_ids):
            distance = distances[i]
            cos_angle = angles[i]
            
            # Only consider defenders within pressure range and in front (cos_angle >= 0)
            if distance <= self.defender_pressure_distance and cos_angle >= 0:
                # Calculate turnover probability with exponential decay
                # At distance=1 (adjacent), probability = baseline
                # As distance increases beyond 1, probability decays exponentially
                turnover_prob = self.defender_pressure_turnover_chance * math.exp(
                    -self.defender_pressure_decay_lambda * max(0, distance - 1)
                )
                
                defender_pressure_info.append({
                    "defender_id": int(def_id),
                    "distance": int(distance),
                    "turnover_prob": float(turnover_prob),
                    "cos_angle": float(cos_angle),  # Added for debugging/visualization
                })
        
        return defender_pressure_info

    @profile_section("calculate_defender_pressure_turnover_probability")
    def calculate_defender_pressure_turnover_probability(self) -> float:
        """
        Calculate the total turnover probability from defender pressure on the ball handler.
        Returns the compound probability of turnover from all nearby defenders.
        """
        defender_pressure_info = self._calculate_defender_pressure_info()
        
        # Calculate total turnover probability (compound probability)
        if not defender_pressure_info:
            return 0.0
        
        complement_product = 1.0
        for pressure in defender_pressure_info:
            complement_product *= (1.0 - pressure["turnover_prob"])
        total_pressure_prob = 1.0 - complement_product
        
        return total_pressure_prob

    @profile_section("calculate_expected_points_all_players")
    def calculate_expected_points_all_players(self) -> np.ndarray:
        """
        Calculate expected points for offensive players based on their current position
        and defender pressure. Uses pressure-adjusted shot probability.
        
        EP only makes sense for the team that can score (offense), so we only calculate
        it for offensive players.
        
        Returns array of shape (players_per_side,) with EP for each offensive player.
        """
        eps = np.zeros(self.players_per_side, dtype=np.float32)
        
        for idx, player_id in enumerate(self.offense_ids):
            eps[idx] = self._calculate_expected_points_for_player(player_id)
        
        return eps

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
