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
from typing import Dict, List, Tuple, Optional, Union
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
        steal_chance: float = 0.05,
        three_point_distance: int = 4,
        layup_pct: float = 0.60,
        three_pt_pct: float = 0.37,
        # Baseline shooting variability (per-player, sampled each episode)
        layup_std: float = 0.0,
        three_pt_std: float = 0.0,
        # Dunk controls
        allow_dunks: bool = False,
        dunk_pct: float = 0.90,
        dunk_std: float = 0.0,
        # Illegal defense (3-in-the-key) controls
        illegal_defense_enabled: bool = True,
        illegal_defense_max_steps: int = 3,
        # Shot pressure parameters
        shot_pressure_enabled: bool = True,
        shot_pressure_max: float = 0.5,  # max reduction at distance=1 (multiplier = 1 - max)
        shot_pressure_lambda: float = 1.0,  # decay rate per hex away from shooter
        shot_pressure_arc_degrees: float = 60.0,  # arc width centered toward basket
        # Pass/OOB curriculum controls
        pass_arc_degrees: float = 60.0,
        pass_oob_turnover_prob: float = 1.0,
        enable_profiling: bool = False,
        spawn_distance: int = 3,
        max_spawn_distance: Optional[int] = None,
        # Reward shaping parameters
        pass_reward: float = 0.0,
        turnover_penalty: float = 0.0,
        made_shot_reward_inside: float = 2.0,
        made_shot_reward_three: float = 3.0,
        missed_shot_penalty: float = 0.0,
        potential_assist_reward: float = 0.1,
        full_assist_bonus: float = 0.2,
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
        self.steal_chance = steal_chance
        self.spawn_distance = spawn_distance
        self.max_spawn_distance = max_spawn_distance
        self.use_egocentric_obs = bool(use_egocentric_obs)
        self.egocentric_rotate_to_hoop = bool(egocentric_rotate_to_hoop)
        self.include_hoop_vector = bool(include_hoop_vector)
        self.normalize_obs = bool(normalize_obs)
        # Movement mask behavior
        self.mask_occupied_moves = bool(mask_occupied_moves)
        # Illegal action handling configuration/state
        try:
            self.illegal_action_policy = IllegalActionPolicy(illegal_action_policy)
        except Exception:
            self.illegal_action_policy = IllegalActionPolicy.NOOP
        # Optional per-step action probabilities provided by caller
        self._pending_action_probs: Optional[np.ndarray] = None
        # Honor constructor flag for strict illegal action handling
        self.raise_on_illegal_action = bool(raise_on_illegal_action)
        # Three-point configuration and shot model parameters
        self.three_point_distance = three_point_distance
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
        self._profile_ns: Dict[str, int] = {}
        self._profile_calls: Dict[str, int] = {}
        # Basket position, using offset coordinates for placement
        basket_col = 0
        basket_row = self.court_height // 2
        self.basket_position = self._offset_to_axial(basket_col, basket_row)
        # Illegal defense configuration
        self.illegal_defense_enabled = bool(illegal_defense_enabled)
        self.illegal_defense_max_steps = int(illegal_defense_max_steps)

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
        # Player skills moved to separate observation key `skills` (shape=(players_per_side*3,))
        base_len = (self.n_players * 2) + self.n_players + 1 + self.players_per_side
        hoop_extra = 2 if self.include_hoop_vector else 0
        state_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(base_len + hoop_extra,),
            dtype=np.float32,
        )
        action_mask_space = spaces.Box(
            low=0, high=1, shape=(self.n_players, len(ActionType)), dtype=np.int8
        )

        # The full observation space is a dictionary containing the state and the mask
        role_flag_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
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

        # --- Reward parameters (stored on env for evaluation compatibility) ---
        self.pass_reward: float = float(pass_reward)
        self.turnover_penalty: float = float(turnover_penalty)
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

    def _offset_to_axial(self, col: int, row: int) -> Tuple[int, int]:
        """Converts odd-r offset coordinates to axial coordinates."""
        q = col - (row - (row & 1)) // 2
        r = row
        return q, r

    def _axial_to_offset(self, q: int, r: int) -> Tuple[int, int]:
        """Converts axial coordinates to odd-r offset coordinates."""
        col = q + (r - (r & 1)) // 2
        row = r
        return col, row

    def _axial_to_cartesian(self, q: int, r: int) -> Tuple[float, float]:
        """Convert axial (q, r) to cartesian (x, y) matching rendering geometry."""
        size = 1.0
        x = size * (math.sqrt(3) * q + math.sqrt(3) / 2 * r)
        y = size * (1.5 * r)
        return x, y

    def _axial_to_cube(self, q: int, r: int) -> Tuple[int, int, int]:
        """Convert axial (q, r) to cube (x, y, z) coordinates."""
        x, z = q, r
        y = -x - z
        return x, y, z

    def _cube_to_axial(self, x: int, y: int, z: int) -> Tuple[int, int]:
        """Convert cube (x, y, z) to axial (q, r) coordinates."""
        return x, z

    def _rotate60_cw_cube(self, x: int, y: int, z: int) -> Tuple[int, int, int]:
        """Rotate cube (x, y, z) by 60 degrees clockwise."""
        return -z, -x, -y

    def _rotate_k60_axial(self, q: int, r: int, k: int) -> Tuple[int, int]:
        """Rotate axial (q, r) by k*60 degrees clockwise."""
        x, y, z = self._axial_to_cube(q, r)
        for _ in range(k % 6):
            x, y, z = self._rotate60_cw_cube(x, y, z)
        return self._cube_to_axial(x, y, z)

    def _hex_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate distance between two hexagon positions."""
        q1, r1 = pos1
        q2, r2 = pos2
        return (abs(q1 - q2) + abs(q1 + r1 - q2 - r2) + abs(r1 - r2)) // 2

    @profile_section("reset")
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment to initial state."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
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
        self._defender_in_key_steps = {pid: 0 for pid in range(self.n_players)}
        self._assist_candidate = None
        # Clear any pending external probabilities on reset
        self._pending_action_probs = None
        # Track first step after reset for phi shaping (Φ(s₀) = 0)
        self._first_step_after_reset = True

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
                [1.0 if self.training_team == Team.OFFENSE else 0.0],
                dtype=np.float32,
            ),
            "skills": self._get_offense_skills_array(),
        }
        info = {"training_team": self.training_team.name}

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
        # For proper PBRS, Φ(s₀) = 0 to ensure shaping sums to zero over episode
        phi_prev: Optional[float] = None
        if self._first_step_after_reset:
            phi_prev = 0.0
            self._first_step_after_reset = False
        else:
            try:
                phi_prev = float(self._phi_shot_quality())
            except Exception:
                phi_prev = 0.0

        # --- Defender Pressure Mechanic ---
        if self.ball_holder in self.offense_ids:
            ball_handler_pos = self.positions[self.ball_holder]
            for defender_id in self.defense_ids:
                defender_pos = self.positions[defender_id]
                distance = self._hex_distance(ball_handler_pos, defender_pos)

                if distance <= self.defender_pressure_distance:
                    # This defender is applying pressure. Roll for a turnover.
                    if self._rng.random() < self.defender_pressure_turnover_chance:
                        # Turnover occurs!
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
                                [1.0 if self.training_team == Team.OFFENSE else 0.0],
                                dtype=np.float32,
                            ),
                            "skills": self._get_offense_skills_array(),
                        }
                        info = {
                            "training_team": self.training_team.name,
                            "action_results": turnover_results,
                            "shot_clock": self.shot_clock,
                        }

                        # Phi diagnostics and optional shaping on early turnover path
                        # Terminal step → define Phi(s')=0 to preserve policy invariance
                        phi_next_term = 0.0

                        # Always calculate phi shaping reward for display/logging purposes
                        r_shape = float(self.reward_shaping_gamma) * float(
                            phi_next_term
                        ) - float(phi_prev if phi_prev is not None else 0.0)
                        shaped = float(self.phi_beta) * float(r_shape)
                        per_team = shaped / self.players_per_side

                        # Only apply to actual rewards if enabled
                        if self.enable_phi_shaping:
                            rewards[self.offense_ids] += per_team
                            rewards[self.defense_ids] -= per_team

                        # Always report the calculated value (for UI/logging)
                        info["phi_r_shape"] = float(per_team)
                        info["phi_prev"] = float(
                            phi_prev if phi_prev is not None else 0.0
                        )
                        info["phi_next"] = float(phi_next_term)
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
                                shot_value = (
                                    2.0
                                    if (self.allow_dunks and dist == 0)
                                    else (
                                        3.0
                                        if dist >= self.three_point_distance
                                        else 2.0
                                    )
                                )
                                p = float(self._calculate_shot_probability(pid, dist))
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

        # Check for episode termination and calculate rewards
        done, episode_rewards = self._check_termination_and_rewards(action_results)
        rewards += episode_rewards

        # Check shot clock expiration
        if self.shot_clock <= 0:
            done = True

        self.episode_ended = done

        # Update illegal defense counters based on resulting positions
        if self.illegal_defense_enabled:
            for did in self.defense_ids:
                if self.positions and tuple(self.positions[did]) == tuple(
                    self.basket_position
                ):
                    self._defender_in_key_steps[did] = (
                        self._defender_in_key_steps.get(did, 0) + 1
                    )
                else:
                    self._defender_in_key_steps[did] = 0

        # Clear per-step external probabilities after use to avoid reuse
        self._pending_action_probs = None

        obs = {
            "obs": self._get_observation(),
            "action_mask": self._get_action_masks(),
            "role_flag": np.array(
                [1.0 if self.training_team == Team.OFFENSE else 0.0],
                dtype=np.float32,
            ),
            "skills": self._get_offense_skills_array(),
        }
        info = {
            "training_team": self.training_team.name,
            "action_results": action_results,
            "shot_clock": self.shot_clock,
        }

        # Phi diagnostics; always calculate for UI display, but only apply if enabled.
        # If this is a terminal step, force Phi(s')=0 to preserve policy invariance.
        phi_next = 0.0 if done else 0.0
        if not done:
            try:
                phi_next = float(self._phi_shot_quality())
            except Exception:
                phi_next = 0.0

        # Always calculate phi shaping reward for display/logging purposes
        r_shape = float(self.reward_shaping_gamma) * float(phi_next) - float(
            phi_prev if phi_prev is not None else 0.0
        )
        shaped = float(self.phi_beta) * float(r_shape)
        per_team = shaped / self.players_per_side

        # Only apply to actual rewards if enabled
        if self.enable_phi_shaping:
            rewards[self.offense_ids] += per_team
            rewards[self.defense_ids] -= per_team

        # Always report the calculated value (for UI/logging)
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
                shot_value = (
                    2.0
                    if (self.allow_dunks and dist == 0)
                    else (3.0 if dist >= self.three_point_distance else 2.0)
                )
                p = float(self._calculate_shot_probability(pid, dist))
                ep_by_player.append(float(shot_value * p))
            info["phi_ep_by_player"] = ep_by_player
        except Exception:
            pass

        return obs, rewards, done, False, info

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

        # Enforce illegal defense: after max steps in basket, mask NOOP so defender must move
        if self.illegal_defense_enabled and self.illegal_defense_max_steps > 0:
            for did in self.defense_ids:
                if (
                    self._defender_in_key_steps.get(did, 0)
                    >= self.illegal_defense_max_steps
                ):
                    if tuple(self.positions[did]) == tuple(self.basket_position):
                        masks[did, ActionType.NOOP.value] = 0

        return masks

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
        defense_positions: List[Tuple[int, int]] = []
        for off_pos in offense_positions:
            off_dist = self._hex_distance(off_pos, self.basket_position)
            candidates = [
                cell
                for cell in all_cells
                if cell != self.basket_position
                and cell not in taken_positions
                and self._is_valid_position(*cell)
                and self._hex_distance(cell, self.basket_position) < off_dist
                and self._hex_distance(cell, self.basket_position)
                >= min_spawn_dist_defense
                and (
                    max_spawn_dist is None
                    or self._hex_distance(cell, self.basket_position) <= max_spawn_dist
                )
            ]

            if not candidates:
                # Fallback: pick any valid empty cell meeting min/max spawn distance constraints
                candidates = [
                    cell
                    for cell in all_cells
                    if cell != self.basket_position
                    and cell not in taken_positions
                    and self._is_valid_position(*cell)
                    and self._hex_distance(cell, self.basket_position)
                    >= min_spawn_dist_defense
                    and (
                        max_spawn_dist is None
                        or self._hex_distance(cell, self.basket_position)
                        <= max_spawn_dist
                    )
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

            # Choose the candidate nearest to the offensive player (to simulate marking)
            candidates.sort(key=lambda c: self._hex_distance(c, off_pos))
            def_pos = candidates[0]
            defense_positions.append(def_pos)
            taken_positions.add(def_pos)

        # offense first then defense
        return offense_positions + defense_positions

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

    def _set_initial_ball_holder(self, player_id: int) -> None:
        if not (0 <= player_id < self.n_players):
            raise ValueError(
                f"ball_holder must be in [0, {self.n_players - 1}], got {player_id}"
            )
        self.ball_holder = int(player_id)

    def _is_valid_position(self, q: int, r: int) -> bool:
        """Check if a hexagon position is within the rectangular court bounds."""
        col, row = self._axial_to_offset(q, r)
        return 0 <= col < self.court_width and 0 <= row < self.court_height

    @profile_section("process_actions")
    def _process_simultaneous_actions(self, actions: np.ndarray) -> Dict:
        """Process all player actions simultaneously with collision resolution."""
        results = {
            "moves": {},
            "passes": {},
            "shots": {},
            "collisions": [],
            "turnovers": [],
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

        # If configured, block moves into any cell that was occupied at the start of the step
        if self.mask_occupied_moves and intended_moves:
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
        return results

    def _get_adjacent_position(
        self, pos: Tuple[int, int], direction_idx: int
    ) -> Tuple[int, int]:
        """Get adjacent hexagon position in given direction."""
        q, r = pos
        dq, dr = self.hex_directions[direction_idx]
        return (q + dq, r + dr)

    @profile_section("pass_logic")
    def _attempt_pass(self, passer_id: int, direction_idx: int, results: Dict) -> None:
        """
        Arc-based passing:
        - Determine a 60-degree arc centered on the chosen direction.
        - Eligible receivers are same-team players whose angle from passer lies within the arc;
          choose the nearest such teammate as target.
        - If no eligible teammate, treat as pass out of bounds in that direction.
        - If at least one defender lies in the same arc and is closer than the receiver,
          a fixed 25% chance of interception applies (closest defender steals).
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

        # Possible interception by defender in same arc who is closer than receiver
        opp_ids = (
            self.defense_ids if passer_id in self.offense_ids else self.offense_ids
        )
        intercept_candidates: List[Tuple[int, int]] = []  # (defender_id, distance)
        for did in opp_ids:
            dq, dr = self.positions[did]
            if not in_arc(dq, dr):
                continue
            dist_d = self._hex_distance(passer_pos, (dq, dr))
            if recv_dist is not None and dist_d < recv_dist:
                intercept_candidates.append((did, dist_d))

        if intercept_candidates:
            # Closest defender in arc
            intercept_candidates.sort(key=lambda t: t[1])
            thief_id = intercept_candidates[0][0]
            if self._rng.random() < self.steal_chance:
                # Interception occurs
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
                }
                return

        # Successful pass to receiver in arc
        self.ball_holder = recv_id
        results["passes"][passer_id] = {"success": True, "target": recv_id}
        # Start/refresh assist window (configurable steps including current step)
        self._assist_candidate = {
            "passer_id": int(passer_id),
            "recipient_id": int(recv_id),
            "expires_at_step": int(self.step_count + self.assist_window),
        }
        return

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

        return {
            "success": shot_made,
            "distance": distance,
            "probability": shot_success_prob,
            "rng": rng_u,
            "base_probability": base_prob,
            "pressure_multiplier": pressure_mult,
        }

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

    @profile_section("shot_prob")
    def _calculate_shot_probability(self, shooter_id: int, distance: int) -> float:
        """Calculate probability of successful shot using a simple linear model
        anchored at layup (distance 1) and three-point (distance = three_point_distance).
        Beyond the arc, we linearly extrapolate and clamp to [0.01, 0.99].
        """
        # Anchors
        d0 = 1
        d1 = max(self.three_point_distance, d0 + 1)
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

    @profile_section("rewards")
    def _check_termination_and_rewards(
        self, action_results: Dict
    ) -> Tuple[bool, np.ndarray]:
        """Check if episode should terminate and calculate rewards."""
        rewards = np.zeros(self.n_players)
        done = False
        pass_reward = self.pass_reward
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

        # Check for shots
        for player_id, shot_result in action_results.get("shots", {}).items():
            done = True  # Episode ends after any shot attempt

            # Compute distance to basket and value of the attempted shot
            shooter_pos = self.positions[player_id]
            dist_to_basket = self._hex_distance(shooter_pos, self.basket_position)

            if shot_result["success"]:
                # Basket was made
                made_shot_reward = (
                    made_shot_reward_three
                    if dist_to_basket >= self.three_point_distance
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
                            made_shot_reward_three
                            if _dist_for_pct >= self.three_point_distance
                            else made_shot_reward_inside
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
            if assist_passer is not None:
                shot_result["assist_passer_id"] = assist_passer
            # Clear assist window after a shot attempt (episode ends anyway)
            self._assist_candidate = None

        return done, rewards

    # -------------------- Potential Function Phi(s) --------------------
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

        # If using only the ball handler
        def expected_points_for(player_id: int) -> float:
            player_pos = self.positions[player_id]
            dist = self._hex_distance(player_pos, self.basket_position)
            # Shot value: 3 if at/behind arc and not a dunk; else 2
            if self.allow_dunks and dist == 0:
                shot_value = 2.0
            else:
                shot_value = 3.0 if dist >= self.three_point_distance else 2.0
            p_make = float(self._calculate_shot_probability(player_id, dist))
            return float(shot_value * p_make)

        if self.phi_use_ball_handler_only:
            return expected_points_for(int(self.ball_holder))

        # Compute ball-handler EP and aggregate teammate EPs based on mode
        ball_ep = expected_points_for(int(self.ball_holder))
        ball_holder_id = int(self.ball_holder)

        # Collect teammate EPs (may or may not exclude ball handler depending on mode)
        mode = getattr(self, "phi_aggregation_mode", "team_best")

        if mode == "team_avg":
            # Simple average of all players (including ball handler)
            eps = [expected_points_for(int(pid)) for pid in team_ids]
            return float(sum(eps) / max(1, len(eps)))

        # For other modes, separate ball handler from teammates
        teammate_eps = [
            expected_points_for(int(pid)) for pid in team_ids if pid != ball_holder_id
        ]

        if not teammate_eps:  # No teammates (1v1 or edge case)
            return ball_ep

        # Aggregate teammate EPs based on mode
        if mode == "teammates_best":
            teammate_aggregate = max(teammate_eps)
        elif mode == "teammates_avg":
            teammate_aggregate = sum(teammate_eps) / len(teammate_eps)
        else:  # "team_best" (default/legacy behavior)
            # Include ball handler in the "best" calculation
            teammate_aggregate = max(max(teammate_eps), ball_ep)

        # Blend teammate aggregate with ball handler EP
        w = float(max(0.0, min(1.0, getattr(self, "phi_blend_weight", 0.0))))
        blended = (1.0 - w) * float(teammate_aggregate) + w * float(ball_ep)
        return float(blended)

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
            pos = self.positions[pid]
            dist = self._hex_distance(pos, self.basket_position)
            shot_value = (
                2.0
                if (self.allow_dunks and dist == 0)
                else (3.0 if dist >= self.three_point_distance else 2.0)
            )
            p = float(self._calculate_shot_probability(pid, dist))
            ep = float(shot_value * p)
            if pid == self.ball_holder:
                ball_ep = ep
            if ep > team_best:
                team_best = ep
        return float(team_best), float(ball_ep)

    # Allow VecEnv.env_method to update phi_beta dynamically
    def set_phi_beta(self, value: float) -> None:
        try:
            self.phi_beta = float(value)
        except Exception:
            pass

    # --- Schedulable setters for passing curriculum ---
    def set_pass_arc_degrees(self, value: float) -> None:
        try:
            self.pass_arc_degrees = float(max(1.0, min(360.0, value)))
        except Exception:
            pass

    def set_pass_oob_turnover_prob(self, value: float) -> None:
        try:
            self.pass_oob_turnover_prob = float(max(0.0, min(1.0, value)))
        except Exception:
            pass

    def _get_observation(self) -> np.ndarray:
        """Get current observation of the game state.

        Ego-centric vector:
        - For each player i: (dq_i, dr_i) relative to current ball handler, normalized
        - One-hot ball holder (redundant but retained for compatibility/debugging)
        - Shot clock (raw value)
        - Hoop vector relative to ball handler, normalized
        - Role flag now provided separately as observation key `role_flag`
        - For each offensive player: distance to nearest defender (normalized if enabled)
        """
        obs: List[float] = []

        # Choose the ego-center. If there is no ball holder (e.g., after a missed shot in a terminal state),
        # fall back to the basket position to avoid undefined relative coordinates.
        if self.ball_holder is not None:
            center_q, center_r = self.positions[self.ball_holder]
        else:
            center_q, center_r = self.basket_position

        # Normalization factor to put deltas roughly in [-1, 1]
        # Using the larger of width/height bounds axial deltas conservatively
        norm_den: float = float(max(self.court_width, self.court_height)) or 1.0
        if not self.normalize_obs:
            norm_den = 1.0

        # Compute rotation k so that hoop vector is aligned toward +q (if enabled)
        # Work with unnormalized axial deltas, rotate, then normalize.
        hoop_dq_raw = self.basket_position[0] - center_q
        hoop_dr_raw = self.basket_position[1] - center_r

        best_k = 0
        if self.egocentric_rotate_to_hoop:
            best_score = None
            for k in range(6):
                rq, rr = self._rotate_k60_axial(hoop_dq_raw, hoop_dr_raw, k)
                # Prefer minimal |rr| (close to +q axis), then prefer rq>=0, then maximize rq
                score = (abs(rr), 0 if rq >= 0 else 1, -rq)
                if best_score is None or score < best_score:
                    best_score = score
                    best_k = k

        # Player positions relative to ego-center, rotated if configured, then normalized
        for q, r in self.positions:
            rdq = q - center_q
            rdr = r - center_r
            if best_k:
                rdq, rdr = self._rotate_k60_axial(rdq, rdr, best_k)
            obs.extend([rdq / norm_den, rdr / norm_den])

        # One-hot encode the ball holder
        ball_holder_one_hot = np.zeros(self.n_players, dtype=np.float32)
        if self.ball_holder is not None:
            ball_holder_one_hot[self.ball_holder] = 1.0
        obs.extend(ball_holder_one_hot.tolist())

        # Shot clock (kept unnormalized)
        obs.append(float(self.shot_clock))

        # Hoop vector relative to ego-center, rotated consistently (optional)
        if self.include_hoop_vector:
            hoop_dq, hoop_dr = hoop_dq_raw, hoop_dr_raw
            if best_k:
                hoop_dq, hoop_dr = self._rotate_k60_axial(hoop_dq, hoop_dr, best_k)
            hoop_dq /= norm_den
            hoop_dr /= norm_den
            obs.extend([hoop_dq, hoop_dr])

        # Distances from each offensive player to the nearest defender
        # Appended in ascending offensive player id order
        for offense_id in self.offense_ids:
            offense_pos = self.positions[offense_id]
            nearest_defender_distance = min(
                self._hex_distance(offense_pos, self.positions[defender_id])
                for defender_id in self.defense_ids
            )
            if self.normalize_obs:
                obs.append(float(nearest_defender_distance) / norm_den)
            else:
                obs.append(float(nearest_defender_distance))

        return np.array(obs, dtype=np.float32)

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

    def render(self):
        """Render the current state of the environment."""
        if self.render_mode == "human":
            return self._render_ascii()
        elif self.render_mode == "rgb_array":
            return self._render_visual()

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

                # Paint the three-point line: all hexes at exactly self.three_point_distance
                cell_distance = self._hex_distance((q, r_ax), self.basket_position)
                if cell_distance == self.three_point_distance:
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
                    is_three = dist_to_basket >= self.three_point_distance
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

    def switch_training_team(self):
        """Switch which team is currently training (for alternating optimization)."""
        self.training_team = (
            Team.DEFENSE if self.training_team == Team.OFFENSE else Team.OFFENSE
        )

    # --- Profiling helpers ---
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
