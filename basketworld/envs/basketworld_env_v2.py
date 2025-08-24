
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
from time import perf_counter_ns

# Use a non-interactive backend so rendering works in headless/threaded contexts
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
from gymnasium import spaces


def profile_section(section_name: str):
    """Decorator to measure method wall time in ns when env.enable_profiling is True.
    Placed before class definition so it's available for method decorators.
    """
    def _decorator(func):
        def _wrapped(self, *args, **kwargs):
            if not getattr(self, "enable_profiling", False):
                return func(self, *args, **kwargs)
            t0 = perf_counter_ns()
            try:
                return func(self, *args, **kwargs)
            finally:
                dt = perf_counter_ns() - t0
                # Lazy init if constructor did not run yet
                if not hasattr(self, "_profile_ns"):
                    self._profile_ns = {}
                    self._profile_calls = {}
                self._profile_ns[section_name] = self._profile_ns.get(section_name, 0) + dt
                self._profile_calls[section_name] = self._profile_calls.get(section_name, 0) + 1
        return _wrapped
    return _decorator


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


class HexagonBasketballEnv(gym.Env):
    """Hexagon-tessellated basketball environment for self-play RL."""
    
    metadata = {"render.modes": ["human", "rgb_array"]}
    
    def __init__(
        self,
        grid_size: int = 16,
        players_per_side: int = 3,
        shot_clock_steps: int = 24,
        training_team: Team = Team.OFFENSE,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        defender_pressure_distance: int = 1,
        defender_pressure_turnover_chance: float = 0.05,
        three_point_distance: int = 4,
        layup_pct: float = 0.60,
        three_pt_pct: float = 0.37,
        # Dunk controls
        allow_dunks: bool = False,
        dunk_pct: float = 0.90,
        # Shot pressure parameters
        shot_pressure_enabled: bool = True,
        shot_pressure_max: float = 0.5,   # max reduction at distance=1 (multiplier = 1 - max)
        shot_pressure_lambda: float = 1.0, # decay rate per hex away from shooter
        shot_pressure_arc_degrees: float = 60.0, # arc width centered toward basket
        enable_profiling: bool = False,
        spawn_distance: int = 3,
        # Observation controls
        use_egocentric_obs: bool = True,
        egocentric_rotate_to_hoop: bool = True,
        include_hoop_vector: bool = True,
        normalize_obs: bool = True,
        # Movement mask controls
        mask_occupied_moves: bool = False,
    ):
        super().__init__()
        
        self.grid_size = grid_size
        self.court_width = int(grid_size * 1.0)
        self.court_height = grid_size
        self.players_per_side = players_per_side
        self.shot_clock_steps = shot_clock_steps
        self.training_team = training_team  # Which team is currently training
        self.render_mode = render_mode
        self.defender_pressure_distance = defender_pressure_distance
        self.defender_pressure_turnover_chance = defender_pressure_turnover_chance
        self.spawn_distance = spawn_distance
        self.use_egocentric_obs = bool(use_egocentric_obs)
        self.egocentric_rotate_to_hoop = bool(egocentric_rotate_to_hoop)
        self.include_hoop_vector = bool(include_hoop_vector)
        self.normalize_obs = bool(normalize_obs)
        # Movement mask behavior
        self.mask_occupied_moves = bool(mask_occupied_moves)
        # Three-point configuration and shot model parameters
        self.three_point_distance = three_point_distance
        self.layup_pct = float(layup_pct)
        self.three_pt_pct = float(three_pt_pct)
        # Dunk configuration
        self.allow_dunks = bool(allow_dunks)
        self.dunk_pct = float(dunk_pct)
        # Back-compat field kept (UI may use it if shot_params absent). Not authoritative anymore.
        self.shot_probs = None
        # New descriptive params for UI
        self.shot_params = {
            "model": "linear",
            "layup_pct": self.layup_pct,
            "three_pt_pct": self.three_pt_pct,
            "dunk_pct": self.dunk_pct,
            "allow_dunks": self.allow_dunks,
        }
        # Defender shot pressure
        self.shot_pressure_enabled = bool(shot_pressure_enabled)
        self.shot_pressure_max = float(shot_pressure_max)
        self.shot_pressure_lambda = float(shot_pressure_lambda)
        self.shot_pressure_arc_degrees = float(shot_pressure_arc_degrees)
        self.shot_pressure_arc_rad = math.radians(shot_pressure_arc_degrees)
        # Profiling
        self.enable_profiling = bool(enable_profiling)
        self._profile_ns: Dict[str, int] = {}
        self._profile_calls: Dict[str, int] = {}
        # Basket position, using offset coordinates for placement
        basket_col = 0
        basket_row = self.court_height // 2
        self.basket_position = self._offset_to_axial(basket_col, basket_row)
        
        # Total players
        self.n_players = players_per_side * 2
        self.offense_ids = list(range(players_per_side))
        self.defense_ids = list(range(players_per_side, self.n_players))
        
        # Action space: each player can take one of 9 actions
        self.action_space = spaces.MultiDiscrete([len(ActionType)] * self.n_players)
        
        # Define the two parts of our observation space
        # Observation length depends on configuration flags
        base_len = (self.n_players * 2) + self.n_players + 1
        hoop_extra = 2 if self.include_hoop_vector else 0
        state_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(base_len + hoop_extra,),
            dtype=np.float32,
        )
        action_mask_space = spaces.Box(
            low=0, 
            high=1, 
            shape=(self.n_players, len(ActionType)), 
            dtype=np.int8
        )
        
        # The full observation space is a dictionary containing the state and the mask
        self.observation_space = spaces.Dict({
            "obs": state_space,
            "action_mask": action_mask_space
        })
        
        # --- Hexagonal Grid Directions ---
        # These are the 6 axial direction vectors for a pointy-topped hexagonal grid.
        # In our (q, r) axial system:
        # - Moving E/W changes only the q-axis.
        # - Moving NW/SE changes only the r-axis.
        # - Moving NE/SW changes both q and r axes.
        # The previous vectors were incorrect, causing bugs in movement and passing.
        # These vectors correctly map ActionType enums to their corresponding axial changes.
        self.hex_directions = [
            (+1,  0), # E:  Move one hex to the right.
            (+1, -1), # NE: Move diagonally up-right.
            ( 0, -1), # NW: Move diagonally up-left.
            (-1,  0), # W:  Move one hex to the left.
            (-1, +1), # SW: Move diagonally down-left.
            ( 0, +1), # SE: Move diagonally down-right.
        ]
        
        self._rng = np.random.default_rng(seed)
        
        # Game state
        self.positions: List[Tuple[int, int]] = []  # (q, r) axial coordinates
        self.ball_holder: int = 0
        self.shot_clock: int = 0
        self.step_count: int = 0
        self.episode_ended: bool = False
        self.last_action_results: Dict = {}

        # Precompute per-cell move validity mask (6 directions) to speed up action mask building
        # 1 = allowed, 0 = blocked (OOB or basket hex if dunks disabled)
        self._move_mask_by_cell: Dict[Tuple[int, int], np.ndarray] = {}
        for row in range(self.court_height):
            for col in range(self.court_width):
                cell = self._offset_to_axial(col, row)
                allowed = np.ones(6, dtype=np.int8)
                for dir_idx in range(6):
                    nbr = (cell[0] + self.hex_directions[dir_idx][0], cell[1] + self.hex_directions[dir_idx][1])
                    if (not self._is_valid_position(*nbr)) or ((nbr == self.basket_position) and (not self.allow_dunks)):
                        allowed[dir_idx] = 0
                self._move_mask_by_cell[cell] = allowed

        # Precompute shoot/pass action indices
        self._shoot_pass_action_indices = [ActionType.SHOOT.value] + [a.value for a in ActionType if "PASS" in a.name]
        
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

        max_shot_clock = self.shot_clock_steps    
        self.shot_clock = max(10, math.floor(max_shot_clock * self._rng.random()))
        self.step_count = 0
        self.episode_ended = False
        self.last_action_results = {}
        
        # Initialize positions (offense on right side, defense on left)
        self.positions = self._generate_initial_positions()
        
        # Random offensive player starts with ball
        self.ball_holder = self._rng.choice(self.offense_ids)
        
        obs = {
            "obs": self._get_observation(),
            "action_mask": self._get_action_masks()
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
                            "turnovers": [{
                                "player_id": self.ball_holder,
                                "reason": "defender_pressure",
                                "stolen_by": defender_id,
                                "turnover_pos": ball_handler_pos
                            }]
                        }
                        self.last_action_results = turnover_results
                        self.ball_holder = defender_id  # Defender gets the ball

                        done = True
                        self.episode_ended = done

                        obs = {"obs": self._get_observation(), "action_mask": self._get_action_masks()}
                        info = {"training_team": self.training_team.name, "action_results": turnover_results, "shot_clock": self.shot_clock}
                        
                        return obs, rewards, done, False, info
                    
                    break # Only check the first defender applying pressure each step

        actions = np.array(actions)
        
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
        
        obs = {
            "obs": self._get_observation(),
            "action_mask": self._get_action_masks()
        }
        info = {
            "training_team": self.training_team.name,
            "action_results": action_results,
            "shot_clock": self.shot_clock
        }
        
        return obs, rewards, done, False, info

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
                 
        return masks
         
    def _generate_initial_positions(self) -> List[Tuple[int, int]]:
        """
        Generate initial positions with:
        - Offense spawned "behind" the 3pt line (distance >= three_point_distance)
        - Defense spawned "inside" the 3pt line (distance < three_point_distance) and
          closer to the basket than the matched offensive player, preferring the nearest
          valid position to that offensive player.
        """
        taken_positions: set[Tuple[int, int]] = set()

        # List all axial cells on court
        all_cells: List[Tuple[int, int]] = []
        for row in range(self.court_height):
            for col in range(self.court_width):
                all_cells.append(self._offset_to_axial(col, row))

        # Offense candidates: behind the arc but within spawn_distance hexes of it
        offense_candidates = []
        for cell in all_cells:
            if cell == self.basket_position:
                continue
            if not self._is_valid_position(*cell):
                continue
            dist = self._hex_distance(cell, self.basket_position)
            if self.three_point_distance + self.spawn_distance <= dist <= self.court_width:
                offense_candidates.append(cell)

        if len(offense_candidates) < self.players_per_side:
            raise ValueError("Not enough cells behind the 3pt line to spawn offense.")

        # Sample unique offense positions
        offense_positions = []
        for cell in self._rng.choice(len(offense_candidates), size=self.players_per_side, replace=False):
            pos = offense_candidates[cell]
            offense_positions.append(pos)
            taken_positions.add(pos)

        # Defense candidates: inside arc and closer to basket than offense counterpart
        defense_positions: List[Tuple[int, int]] = []
        for off_pos in offense_positions:
            off_dist = self._hex_distance(off_pos, self.basket_position)
            candidates = [
                cell for cell in all_cells
                if cell != self.basket_position
                and cell not in taken_positions
                and self._is_valid_position(*cell)
                and self._hex_distance(cell, self.basket_position) < off_dist
                and self._hex_distance(cell, self.basket_position) >= self.three_point_distance + self.spawn_distance
            ]

            if not candidates:
                # Fallback: pick any valid empty cell that is closer to basket
                candidates = [
                    cell for cell in all_cells
                    if cell != self.basket_position
                    and cell not in taken_positions
                    and self._is_valid_position(*cell)
                    and self._hex_distance(cell, self.basket_position) < off_dist + self.spawn_distance
                ]

            if not candidates:
                raise RuntimeError("Could not find a valid inside-arc spawn for a defender.")

            # Choose the candidate nearest to the offensive player (to simulate marking)
            candidates.sort(key=lambda c: self._hex_distance(c, off_pos))
            def_pos = candidates[0]
            defense_positions.append(def_pos)
            taken_positions.add(def_pos)

        # offense first then defense
        return offense_positions + defense_positions
    
    def _is_valid_position(self, q: int, r: int) -> bool:
        """Check if a hexagon position is within the rectangular court bounds."""
        col, row = self._axial_to_offset(q, r)
        return 0 <= col < self.court_width and 0 <= row < self.court_height
    
    @profile_section("process_actions")
    def _process_simultaneous_actions(self, actions: np.ndarray) -> Dict:
        """Process all player actions simultaneously with collision resolution."""
        results = {
            "moves": {}, "passes": {}, "shots": {}, "collisions": [], "turnovers": []
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
                new_pos = self._get_adjacent_position(current_positions[player_id], direction_idx)
                
                if self._is_valid_position(*new_pos):
                    if (new_pos == self.basket_position) and (not self.allow_dunks):
                        # Basket hex blocked when dunks disabled
                        if player_id == self.ball_holder:
                            results["turnovers"].append({
                                "player_id": player_id,
                                "reason": "move_out_of_bounds",
                                "turnover_pos": new_pos
                            })
                            self._turnover_to_defense(player_id)
                        results["moves"][player_id] = {"success": False, "reason": "basket_collision"}
                    else:
                        intended_moves[player_id] = new_pos
                else:
                    # Out of bounds move
                    if player_id == self.ball_holder:
                        results["turnovers"].append({
                            "player_id": player_id,
                            "reason": "move_out_of_bounds",
                            "turnover_pos": new_pos
                        })
                        self._turnover_to_defense(player_id)
                    results["moves"][player_id] = {"success": False, "reason": "out_of_bounds"}

        # If configured, block moves into any cell that was occupied at the start of the step
        if self.mask_occupied_moves and intended_moves:
            occupied_start = set(current_positions)
            to_remove = []
            for pid, dest in intended_moves.items():
                if dest in occupied_start:
                    results["moves"][pid] = {"success": False, "reason": "occupied_neighbor"}
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
                    results["moves"][player_id] = {"success": False, "reason": "collision_static"}
                continue # No one can move here

            # b. Check for collisions between multiple moving players
            if len(players_intending_to_move) > 1:
                # Collision occurs, pick one winner
                winner = self._rng.choice(players_intending_to_move)
                final_positions[winner] = dest
                results["moves"][winner] = {"success": True, "new_position": dest}
                
                # Others fail
                for player_id in players_intending_to_move:
                    if player_id != winner:
                        results["moves"][player_id] = {"success": False, "reason": "collision_dynamic"}
                
                results["collisions"].append({
                    "position": dest, "players": players_intending_to_move, "winner": winner
                })
            else:
                # No collision, single player moves
                player_id = players_intending_to_move[0]
                final_positions[player_id] = dest
                results["moves"][player_id] = {"success": True, "new_position": dest}
        
        self.positions = final_positions
        return results
    
    def _get_adjacent_position(self, pos: Tuple[int, int], direction_idx: int) -> Tuple[int, int]:
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
        # 60-degree arc total -> 30-degree half-angle
        cos_threshold = math.cos(math.pi / 6)

        def in_arc(to_q: int, to_r: int) -> bool:
            vx, vy = self._axial_to_cartesian(to_q - passer_pos[0], to_r - passer_pos[1])
            vnorm = math.hypot(vx, vy)
            if vnorm == 0:
                return False
            cosang = (vx * dir_x + vy * dir_y) / (vnorm * dir_norm)
            return cosang >= cos_threshold

        # Pick closest teammate in arc
        team_ids = self.offense_ids if passer_id in self.offense_ids else self.defense_ids
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
            # No teammate in arc: pass sails out of bounds in that direction
            step = 1
            while True:
                target = (passer_pos[0] + dir_dq * step, passer_pos[1] + dir_dr * step)
                if not self._is_valid_position(*target):
                    self.ball_holder = None
                    results["turnovers"].append({
                        "player_id": passer_id,
                        "reason": "pass_out_of_bounds",
                        "turnover_pos": target,
                    })
                    results["passes"][passer_id] = {"success": False, "reason": "out_of_bounds"}
                    return
                step += 1

        # Possible interception by defender in same arc who is closer than receiver
        opp_ids = self.defense_ids if passer_id in self.offense_ids else self.offense_ids
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
            if self._rng.random() < 0.05:
                # Interception occurs
                self.ball_holder = thief_id
                results["turnovers"].append({
                    "player_id": passer_id,
                    "reason": "intercepted",
                    "stolen_by": thief_id,
                    "turnover_pos": self.positions[thief_id],
                })
                results["passes"][passer_id] = {"success": False, "reason": "intercepted", "interceptor_id": thief_id}
                return

        # Successful pass to receiver in arc
        self.ball_holder = recv_id
        results["passes"][passer_id] = {"success": True, "target": recv_id}
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
        opp_ids = self.defense_ids if shooter_id in self.offense_ids else self.offense_ids

        closest_d: Optional[int] = None
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
            # Defender must be closer than basket along this direction
            if in_arc and d_def < distance_to_basket:
                if closest_d is None or d_def < closest_d:
                    closest_d = d_def

        if closest_d is None:
            return 1.0

        # Pressure multiplier: 1 - A * exp(-lambda * (d-1))
        exponent_arg = max(0, closest_d - 1)
        reduction = self.shot_pressure_max * math.exp(-self.shot_pressure_lambda * exponent_arg)
        return max(0.0, 1.0 - reduction)

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
        if self.allow_dunks and distance == 0:
            base_prob = self.dunk_pct
        elif distance <= d0:
            base_prob = self.layup_pct
        else:
            t = (distance - d0) / (d1 - d0)
            base_prob = self.layup_pct + (self.three_pt_pct - self.layup_pct) * t
        # Clamp similar to backend
        base_prob = max(0.01, min(0.99, base_prob))

        pressure_mult = self._compute_shot_pressure_multiplier(shooter_id, shooter_pos, distance)

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
            nearest_defender = min(self.defense_ids, 
                                 key=lambda d: self._hex_distance(self.positions[from_player], 
                                                                self.positions[d]))
        else:
            # Find nearest offensive player
            nearest_defender = min(self.offense_ids,
                                 key=lambda o: self._hex_distance(self.positions[from_player],
                                                                self.positions[o]))
        
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
        p0 = self.layup_pct
        p1 = self.three_pt_pct

        if self.allow_dunks and distance == 0:
            prob = self.dunk_pct
        elif distance <= d0:
            prob = p0
        else:
            t = (distance - d0) / (d1 - d0)
            prob = p0 + (p1 - p0) * t  # linear interpolation (or extrapolation if distance>d1)

        # Apply defender shot pressure if any qualifying defender is between shooter and basket
        if self.shot_pressure_enabled and shooter_id is not None:
            shooter_pos = self.positions[shooter_id]
            pressure_mult = self._compute_shot_pressure_multiplier(shooter_id, shooter_pos, distance)
            prob *= pressure_mult

        # Clamp to sensible bounds
        prob = max(0.01, min(0.99, prob))
        return float(prob)

    @profile_section("rewards")
    def _check_termination_and_rewards(self, action_results: Dict) -> Tuple[bool, np.ndarray]:
        """Check if episode should terminate and calculate rewards."""
        rewards = np.zeros(self.n_players)
        done = False
        pass_reward = 0.0
        turnover_penalty = 0.0
        # Define the reward magnitude for shots (3PT outside the line)
        # Inside arc: 1.0, At/Outside arc (>= distance) : 1.5
        made_shot_reward_inside = 2.0
        made_shot_reward_three = 3.0
        missed_shot_penalty = 0.0 # No penalty for missed shots
        
        # --- Reward successful passes ---
        for _, pass_result in action_results.get("passes", {}).items():
            if pass_result.get("success"):
                rewards[self.offense_ids] += pass_reward/self.players_per_side
                rewards[self.defense_ids] -= pass_reward/self.players_per_side
        
        # --- Handle all turnovers from actions ---
        if action_results.get("turnovers"):
            done = True
            # Penalize offense, reward defense for the turnover
            # We assume only one turnover can happen per step
            rewards[self.offense_ids] -= turnover_penalty/self.players_per_side 
            rewards[self.defense_ids] += turnover_penalty/self.players_per_side
        

        # Check for shots
        for player_id, shot_result in action_results.get("shots", {}).items():
            done = True  # Episode ends after any shot attempt
                        
            if shot_result["success"]:
                # Basket was made
                # Distance of the shot to determine 2PT vs 3PT
                shooter_pos = self.positions[player_id]
                dist_to_basket = self._hex_distance(shooter_pos, self.basket_position)
                made_shot_reward = (
                    made_shot_reward_three
                    if dist_to_basket >= self.three_point_distance
                    else made_shot_reward_inside
                )
                # Offense scored, good for them, bad for defense
                rewards[self.offense_ids] += made_shot_reward/self.players_per_side
                rewards[self.defense_ids] -= made_shot_reward/self.players_per_side
                # else: handle rare case of defense scoring on own basket
            else:
                # Offense missed, bad for them, good for defense
                rewards[self.offense_ids] -= missed_shot_penalty/self.players_per_side
                rewards[self.defense_ids] += missed_shot_penalty/self.players_per_side
        
        return done, rewards
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation of the game state.

        Ego-centric vector:
        - For each player i: (dq_i, dr_i) relative to current ball handler, normalized
        - One-hot ball holder (redundant but retained for compatibility/debugging)
        - Shot clock (raw value)
        - Hoop vector relative to ball handler, normalized
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

        return np.array(obs, dtype=np.float32)
    
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
        
        grid = [[' · ' for _ in range(self.court_width)] for _ in range(self.court_height)]
        
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
        ax.set_aspect('equal')
        
        # Convert axial coordinates to cartesian for pointy-topped hexes
        def axial_to_cartesian(q, r):
            size = 1.0  # Defines the size (radius) of the hexagons
            x = size * (math.sqrt(3) * q + math.sqrt(3) / 2 * r)
            y = size * (3. / 2. * r)
            return x, y
        
        hex_radius = 1.0
        
        # Draw hexagonal grid for the entire court
        for c in range(self.court_width):
            for r in range(self.court_height):
                q, r_ax = self._offset_to_axial(c, r)
                x, y = axial_to_cartesian(q, r_ax)
                
                # Draw the base grid hexagon
                hexagon = RegularPolygon(
                    (x, y), numVertices=6, radius=hex_radius, 
                    orientation=0, # for pointy-topped
                    facecolor='lightgray', edgecolor='white', alpha=0.5,
                    linewidth=1
                )
                ax.add_patch(hexagon)

                # For the basket, add a thick red ring around it
                if (q, r_ax) == self.basket_position:
                    basket_ring = plt.Circle((x, y), hex_radius * 1.05, fill=False, edgecolor='red', linewidth=4, zorder=6)
                    ax.add_patch(basket_ring)

                # Paint the three-point line: all hexes at exactly self.three_point_distance
                cell_distance = self._hex_distance((q, r_ax), self.basket_position)
                if cell_distance == self.three_point_distance:
                    tp_outline = RegularPolygon(
                        (x, y), numVertices=6, radius=hex_radius,
                        orientation=0, facecolor='none', edgecolor='red', linewidth=2.5, zorder=7
                    )
                    ax.add_patch(tp_outline)

        # Draw players by filling their hexagon
        for i, (q, r) in enumerate(self.positions):
            x, y = axial_to_cartesian(q, r)
            color = 'blue' if i in self.offense_ids else 'red'
            
            player_hexagon = RegularPolygon(
                (x, y), numVertices=6, radius=hex_radius,
                orientation=0, # for pointy-topped
                facecolor=color,
                edgecolor='white',
                alpha=0.9,
                zorder=10
            )
            ax.add_patch(player_hexagon)
            ax.text(x, y, str(i), ha='center', va='center', fontsize=24, fontweight='bold', color='white', zorder=11)
            
            if i == self.ball_holder:
                ball_ring = plt.Circle((x, y), hex_radius * 0.9, fill=False, color='orange', linewidth=4, zorder=12)
                ax.add_patch(ball_ring)
        
        # Calculate court boundaries to set axis limits
        cartesian_coords = [axial_to_cartesian(*self._offset_to_axial(c, r))
                            for c in range(self.court_width) for r in range(self.court_height)]
        x_coords = [c[0] for c in cartesian_coords]
        y_coords = [c[1] for c in cartesian_coords]
        
        margin = 2.0
        ax.set_xlim(min(x_coords) - margin, max(x_coords) + margin)
        ax.set_ylim(min(y_coords) - margin, max(y_coords) + margin)

        ax.set_title(f'Hexagon Basketball {self.players_per_side}v{self.players_per_side}')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

        # --- Draw Final Action Result Indicators ---
        if self.episode_ended and self.last_action_results:
            # Shot results
            if self.last_action_results.get("shots"):
                shot_result = list(self.last_action_results["shots"].values())[0]
                basket_x, basket_y = axial_to_cartesian(*self.basket_position)
                # Determine 2PT or 3PT by shooter position at shot
                shooter_id = list(self.last_action_results["shots"].keys())[0]
                shooter_pos = self.positions[int(shooter_id)]
                dist_to_basket = self._hex_distance(shooter_pos, self.basket_position)
                is_three = dist_to_basket >= self.three_point_distance
                label_suffix = "3" if is_three else "2"
                if shot_result["success"]:
                    ax.add_patch(plt.Circle((basket_x, basket_y), hex_radius, color='green', alpha=0.7, zorder=20))
                    ax.text(0.5, 0.9, f"Made {label_suffix}!", transform=ax.transAxes, ha='center', va='center', fontsize=50, fontweight='bold', color='green', alpha=0.9)
                else:
                    ax.add_patch(plt.Circle((basket_x, basket_y), hex_radius, color='red', alpha=0.7, zorder=20))
                    ax.text(0.5, 0.9, f"Missed {label_suffix}!", transform=ax.transAxes, ha='center', va='center', fontsize=50, fontweight='bold', color='red', alpha=0.9)

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
                    ax.text(tx, ty, "X", ha='center', va='center', fontsize=60, fontweight='bold', color='darkred', zorder=21)

                # Map backend reason codes to short labels for the banner
                reason_code = first_turnover.get("reason", "")
                reason_map = {
                    "defender_pressure": "PRESSURE",
                    "pass_out_of_bounds": "OOB",
                    "move_out_of_bounds": "OOB",
                    "intercepted": "STEAL",
                }
                reason_label = reason_map.get(reason_code, reason_code.replace("_", " ").upper())
                ax.text(0.5, 0.9, f"TOV - {reason_label}!", transform=ax.transAxes, ha='center', va='center', fontsize=50, fontweight='bold', color='darkred', alpha=0.9)

            # Shot clock violation
            elif self.shot_clock <= 0:
                ax.text(0.5, 0.9, "SHOT CLOCK VIOLATION", transform=ax.transAxes, ha='center', va='center', fontsize=40, fontweight='bold', color='darkred', alpha=0.9)

 
        # Add shot clock text to the bottom right corner
        ax.text(0.95, 0.05, f"{self.shot_clock}",
                fontsize=48,
                fontweight='bold',
                color='black',
                ha='right',
                va='bottom',
                alpha=0.5,
                transform=ax.transAxes)
        
        fig.canvas.draw()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        
        img = Image.open(buf)
        rgb_array = np.array(img)
        plt.close(fig)
        buf.close()
        
        return rgb_array
    
    def switch_training_team(self):
        """Switch which team is currently training (for alternating optimization)."""
        self.training_team = Team.DEFENSE if self.training_team == Team.OFFENSE else Team.OFFENSE

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
        seed=42
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
