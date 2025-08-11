
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
        # Three-point configuration
        self.three_point_distance = three_point_distance
        # Shot probability table (exposed for UI). Keep in sync with _calculate_shot_probability.
        self.shot_probs = {
            "layup": 0.67,
            "hook": 0.55,
            "jumper": 0.45,
            "three": 0.35,
            "heave": 0.1,
        }
        
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
        state_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=((self.n_players * 2) + self.n_players + 1,), 
            dtype=np.float32
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
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment to initial state."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            
        self.shot_clock = self.shot_clock_steps
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
    
    def step(self, actions: Union[np.ndarray, List[int]]):
        """Execute one step of the environment."""
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
                        rewards = np.zeros(self.n_players)
                        rewards[self.offense_ids] -= 0.2
                        rewards[self.defense_ids] += 0.2
                        self.episode_ended = done

                        obs = {"obs": self._get_observation(), "action_mask": self._get_action_masks()}
                        info = {"training_team": self.training_team.name, "action_results": turnover_results, "shot_clock": self.shot_clock}
                        
                        return obs, rewards, done, False, info
                    
                    break # Only check the first defender applying pressure each step

        actions = np.array(actions)
        
        # Decrement shot clock
        self.shot_clock -= 1
        
        # Initialize rewards
        rewards = np.zeros(self.n_players)
        
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

    def _get_action_masks(self) -> np.ndarray:
        """Generate a mask of legal actions for each player."""
        masks = np.ones((self.n_players, len(ActionType)), dtype=np.int8)
        
        # Only the ball holder can shoot or pass
        shoot_pass_actions = [ActionType.SHOOT.value] + [a.value for a in ActionType if "PASS" in a.name]
        
        for i in range(self.n_players):
            if i != self.ball_holder:
                masks[i, shoot_pass_actions] = 0
                
        return masks
         
    def _generate_initial_positions(self) -> List[Tuple[int, int]]:
        """
        Generate initial positions with offense on the far edge and defense adjacent.
        """
        offense_positions = []
        defense_positions = []
        taken_positions = set()

        # 1. Spawn offense on the far edge of the court (right side)
        edge_col = self.court_width - 1
        
        # Ensure we don't try to sample more rows than available
        num_to_sample = min(self.players_per_side, self.court_height)
        offensive_rows = self._rng.choice(self.court_height, size=num_to_sample, replace=False)

        for row in offensive_rows:
            axial_pos = self._offset_to_axial(edge_col, row)
            if axial_pos != self.basket_position:
                offense_positions.append(axial_pos)
                taken_positions.add(axial_pos)
        
        if len(offense_positions) < self.players_per_side:
            raise ValueError("Could not spawn all offensive players on the edge. Check court dimensions and player count.")

        # 2. Spawn defense adjacent to each offensive player
        # Directions ordered from "in front of" (towards basket) to "behind"
        # W, SW, NW, SE, NE, E
        preferred_directions_indices = [3, 4, 2, 5, 1, 0]

        for off_pos in offense_positions:
            placed_defender = False
            for dir_idx in preferred_directions_indices:
                direction_vector = self.hex_directions[dir_idx]
                def_pos = (off_pos[0] + direction_vector[0], off_pos[1] + direction_vector[1])

                if self._is_valid_position(*def_pos) and def_pos not in taken_positions and def_pos != self.basket_position:
                    defense_positions.append(def_pos)
                    taken_positions.add(def_pos)
                    placed_defender = True
                    break  # Move to the next offensive player
            
            if not placed_defender:
                # This should be rare, but is possible on very crowded/small courts
                raise RuntimeError(f"Could not find a valid adjacent spawn for a defender near {off_pos}")

        # The final list must have offense positions first, then defense
        return offense_positions + defense_positions
    
    def _is_valid_position(self, q: int, r: int) -> bool:
        """Check if a hexagon position is within the rectangular court bounds."""
        col, row = self._axial_to_offset(q, r)
        return 0 <= col < self.court_width and 0 <= row < self.court_height
    
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
                
                if self._is_valid_position(*new_pos) and new_pos != self.basket_position:
                    intended_moves[player_id] = new_pos
                else:
                    reason = "out_of_bounds" if new_pos != self.basket_position else "basket_collision"
                    if player_id == self.ball_holder:
                        results["turnovers"].append({
                            "player_id": player_id,
                            "reason": "move_out_of_bounds",
                            "turnover_pos": new_pos
                        })
                        self._turnover_to_defense(player_id) # This must happen after storing results
                    results["moves"][player_id] = {"success": False, "reason": reason}

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
        dir_angle = math.atan2(dir_y, dir_x)
        arc_half = math.pi / 6  # 30 degrees

        def angle_in_arc(to_q: int, to_r: int) -> Tuple[float, float, bool]:
            vx, vy = self._axial_to_cartesian(to_q - passer_pos[0], to_r - passer_pos[1])
            ang = math.atan2(vy, vx)
            # Smallest signed angle difference
            diff = (ang - dir_angle + math.pi) % (2 * math.pi) - math.pi
            return ang, diff, abs(diff) <= arc_half

        # Pick closest teammate in arc
        team_ids = self.offense_ids if passer_id in self.offense_ids else self.defense_ids
        recv_id = None
        recv_dist = None
        for pid in team_ids:
            if pid == passer_id:
                continue
            tq, tr = self.positions[pid]
            _, _, in_arc = angle_in_arc(tq, tr)
            if not in_arc:
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
            _, _, in_arc = angle_in_arc(dq, dr)
            if not in_arc:
                continue
            dist_d = self._hex_distance(passer_pos, (dq, dr))
            if recv_dist is not None and dist_d < recv_dist:
                intercept_candidates.append((did, dist_d))

        if intercept_candidates:
            # Closest defender in arc
            intercept_candidates.sort(key=lambda t: t[1])
            thief_id = intercept_candidates[0][0]
            if self._rng.random() < 0.25:
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

    def _attempt_shot(self, shooter_id: int) -> Dict:
        """Attempt a shot from the ball holder."""
        shooter_pos = self.positions[shooter_id]
        basket_pos = self.basket_position
        
        distance = self._hex_distance(shooter_pos, basket_pos)
        
        # Use the distance-based probability calculation
        shot_success_prob = self._calculate_shot_probability(shooter_id, distance)
        shot_made = self._rng.random() < shot_success_prob
        
        if not shot_made:
            # Missed shot - possession ends
            self.ball_holder = None
        
        return {
            "success": shot_made,
            "distance": distance,
            "probability": shot_success_prob
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
    
    def _hex_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate distance between two hexagon positions."""
        q1, r1 = pos1
        q2, r2 = pos2
        return (abs(q1 - q2) + abs(q1 + r1 - q2 - r2) + abs(r1 - r2)) // 2

    def _calculate_shot_probability(self, shooter_id: int, distance: int) -> float:
        """Calculate probability of successful shot based on distance."""
        if distance <= 1:
            return self.shot_probs["layup"]    # Dunk/Layup
        elif distance <= 2:
            return self.shot_probs["hook"]  # Close shot
        elif distance <= self.three_point_distance - 1:
            return self.shot_probs["jumper"]  # Mid-range
        elif distance <= self.three_point_distance + 1:
            return self.shot_probs["three"]  # Three
        else:
            return self.shot_probs["heave"] # Long-range heave

    def _check_termination_and_rewards(self, action_results: Dict) -> Tuple[bool, np.ndarray]:
        """Check if episode should terminate and calculate rewards."""
        rewards = np.zeros(self.n_players)
        done = False
        
        # --- Reward successful passes ---
        for _, pass_result in action_results.get("passes", {}).items():
            if pass_result.get("success"):
                rewards[self.offense_ids] += 0.05
        
        # --- Handle all turnovers from actions ---
        if action_results.get("turnovers"):
            done = True
            # Penalize offense, reward defense for the turnover
            # We assume only one turnover can happen per step
            rewards[self.offense_ids] -= 0.2 
            rewards[self.defense_ids] += 0.2
        
        # Check for shots
        for player_id, shot_result in action_results.get("shots", {}).items():
            done = True  # Episode ends after any shot attempt
            
            # --- Time-based penalty for shooting too early ---
            if self.step_count <= 3:
                # Penalty is high at step 1 and decrements
                # Step 1: -0.5, Step 2: -0.3, Step 3: -0.1
                time_penalty = - (0.7 - self.step_count * 0.2)
                
                # We only apply this penalty to the team that is currently training
                if self.training_team == Team.OFFENSE and player_id in self.offense_ids:
                    rewards[self.offense_ids] += time_penalty
            
            # Define the reward magnitude for shots (3PT outside the line)
            # Inside arc: 1.0, At/Outside arc (>= distance) : 1.5
            made_shot_reward_inside = 1.0
            made_shot_reward_three = 1.5
            missed_shot_penalty = 0.1 # Less punishing than a turnover (-0.2)

            if shot_result["success"]:
                # Basket was made
                if player_id in self.offense_ids:
                    # Distance of the shot to determine 2PT vs 3PT
                    shooter_pos = self.positions[player_id]
                    dist_to_basket = self._hex_distance(shooter_pos, self.basket_position)
                    made_shot_reward = (
                        made_shot_reward_three
                        if dist_to_basket >= self.three_point_distance
                        else made_shot_reward_inside
                    )
                    # Offense scored, good for them, bad for defense
                    rewards[self.offense_ids] += made_shot_reward
                    rewards[self.defense_ids] -= made_shot_reward
                # else: handle rare case of defense scoring on own basket
            else:
                # Basket was missed
                if player_id in self.offense_ids:
                    # Offense missed, bad for them, good for defense
                    rewards[self.offense_ids] -= missed_shot_penalty
                    rewards[self.defense_ids] += missed_shot_penalty
        
        return done, rewards
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation of the game state."""
        obs = []
        
        # Player positions (q, r for each player)
        for q, r in self.positions:
            obs.extend([q, r])
        
        # One-hot encode the ball holder
        ball_holder_one_hot = np.zeros(self.n_players)
        if self.ball_holder is not None:
            ball_holder_one_hot[self.ball_holder] = 1.0
        obs.extend(ball_holder_one_hot)
        
        # Shot clock
        obs.append(self.shot_clock)
        
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
