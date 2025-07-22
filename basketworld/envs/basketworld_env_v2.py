
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
    MOVE_S = 6
    SHOOT = 7
    PASS_E = 8
    PASS_NE = 9
    PASS_NW = 10
    PASS_W = 11
    PASS_SW = 12
    PASS_S = 13


class HexagonBasketballEnv(gym.Env):
    """Hexagon-tessellated basketball environment for self-play RL."""
    
    metadata = {"render.modes": ["human", "rgb_array"]}
    
    def __init__(
        self,
        grid_size: int = 16,
        players_per_side: int = 3,
        shot_clock_steps: int = 24,
        training_team: Team = Team.OFFENSE,
        seed: Optional[int] = None
    ):
        super().__init__()
        
        self.grid_size = grid_size
        self.court_width = int(grid_size * 1.0)
        self.court_height = grid_size
        self.players_per_side = players_per_side
        self.shot_clock_steps = shot_clock_steps
        self.training_team = training_team  # Which team is currently training
        
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
        
        # Observation space:
        # - Player positions: (q, r) for each of n_players -> n_players * 2
        # - Ball holder: one-hot encoded vector of size n_players
        # - Shot clock: 1 value
        obs_space_size = (self.n_players * 2) + self.n_players + 1
        obs_low = np.full(obs_space_size, -np.inf)
        obs_high = np.full(obs_space_size, np.inf)
        
        # Set specific bounds for known values
        # Player positions (can be anything, so keep as inf)
        # Ball holder one-hot (0 to 1)
        obs_low[self.n_players * 2:-1] = 0
        obs_high[self.n_players * 2:-1] = 1
        # Shot clock (0 to max)
        obs_low[-1] = 0
        obs_high[-1] = self.shot_clock_steps
        
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        
        # Hexagon direction vectors for POINTY-TOPPED hexagons
        self.hex_directions = [
            (+1,  0), # E
            ( 0, -1), # NE
            (-1, -1), # NW
            (-1,  0), # W
            ( 0, +1), # SW
            (+1, +1), # S
        ]
        
        self._rng = np.random.default_rng(seed)
        
        # Game state
        self.positions: List[Tuple[int, int]] = []  # (q, r) axial coordinates
        self.ball_holder: int = 0
        self.shot_clock: int = 0
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
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment to initial state."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            
        self.shot_clock = self.shot_clock_steps
        self.episode_ended = False
        self.last_action_results = {}
        
        # Initialize positions (offense on right side, defense on left)
        self.positions = self._generate_initial_positions()
        
        # Random offensive player starts with ball
        self.ball_holder = self._rng.choice(self.offense_ids)
        
        obs = self._get_observation()
        info = {"training_team": self.training_team.name}
        
        return obs, info
    
    def step(self, actions: Union[np.ndarray, List[int]]):
        """Execute one step of the environment."""
        if self.episode_ended:
            raise ValueError("Episode has ended. Call reset() to start a new episode.")
            
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
        
        obs = self._get_observation()
        info = {
            "training_team": self.training_team.name,
            "action_results": action_results,
            "shot_clock": self.shot_clock
        }
        
        return obs, rewards, done, False, info
    
    def _generate_initial_positions(self) -> List[Tuple[int, int]]:
        """Generate initial non-overlapping positions for all players."""
        positions = []
        
        # Offense starts on right side (higher q values)
        offense_start_q = self.grid_size // 2
        defense_start_q = -self.grid_size // 2
        
        # Generate positions ensuring no overlaps
        taken_positions = set()
        
        # Place offense players
        for i in range(self.players_per_side):
            while True:
                col = self._rng.integers(self.court_width // 2, self.court_width)
                row = self._rng.integers(0, self.court_height)
                q, r = self._offset_to_axial(col, row)
                if (q, r) not in taken_positions:
                    positions.append((q, r))
                    taken_positions.add((q, r))
                    break
        
        # Place defense players  
        for i in range(self.players_per_side):
            while True:
                col = self._rng.integers(0, self.court_width // 2)
                row = self._rng.integers(0, self.court_height)
                q, r = self._offset_to_axial(col, row)
                if (q, r) not in taken_positions:
                    positions.append((q, r))
                    taken_positions.add((q, r))
                    break
                    
        return positions
    
    def _is_valid_position(self, q: int, r: int) -> bool:
        """Check if a hexagon position is within the rectangular court bounds."""
        col, row = self._axial_to_offset(q, r)
        return 0 <= col < self.court_width and 0 <= row < self.court_height
    
    def _process_simultaneous_actions(self, actions: np.ndarray) -> Dict:
        """Process all player actions simultaneously with collision resolution."""
        results = {
            "moves": {},
            "passes": {},
            "shots": {},
            "collisions": [],
            "out_of_bounds_turnover": False
        }
        
        # First, collect all intended moves and non-movement actions
        intended_moves = {}
        for player_id, action in enumerate(actions):
            action_type = ActionType(action)
            
            if action_type == ActionType.NOOP:
                continue
            elif 1 <= action_type.value <= 6: # Movement actions
                direction_idx = action_type.value - 1
                new_pos = self._get_adjacent_position(self.positions[player_id], direction_idx)
                if self._is_valid_position(*new_pos):
                    intended_moves[player_id] = new_pos
                else:
                    # Player tried to move out of bounds
                    results["moves"][player_id] = {"success": False, "reason": "out_of_bounds"}
                    if player_id == self.ball_holder:
                        self._turnover_to_defense(player_id)
                        results["out_of_bounds_turnover"] = True
            elif action_type == ActionType.SHOOT:
                if player_id == self.ball_holder:
                    results["shots"][player_id] = self._attempt_shot(player_id)
            elif 8 <= action_type.value <= 13: # Pass actions
                if player_id == self.ball_holder:
                    direction_idx = action_type.value - 8
                    results["passes"][player_id] = self._attempt_pass(player_id, direction_idx)

        # Resolve movement collisions
        final_positions = self.positions.copy()
        position_conflicts = {}
        
        # Group players by intended destination
        for player_id, new_pos in intended_moves.items():
            if new_pos not in position_conflicts:
                position_conflicts[new_pos] = []
            position_conflicts[new_pos].append(player_id)
        
        # Resolve conflicts
        for new_pos, conflicting_players in position_conflicts.items():
            if len(conflicting_players) == 1:
                # No conflict, move normally
                player_id = conflicting_players[0]
                final_positions[player_id] = new_pos
                results["moves"][player_id] = {"success": True, "new_position": new_pos}
            else:
                # Collision! Pick random winner
                winner = self._rng.choice(conflicting_players)
                final_positions[winner] = new_pos
                
                for player_id in conflicting_players:
                    if player_id == winner:
                        results["moves"][player_id] = {"success": True, "new_position": new_pos}
                    else:
                        results["moves"][player_id] = {"success": False, "reason": "collision"}
                
                results["collisions"].append({
                    "position": new_pos,
                    "players": conflicting_players,
                    "winner": winner
                })
        
        # Check for players trying to move into occupied spaces
        occupied_positions = set(final_positions)
        for player_id, action in enumerate(actions):
            if player_id in intended_moves:
                intended_pos = intended_moves[player_id]
                if intended_pos in occupied_positions and player_id not in results["moves"]:
                    results["moves"][player_id] = {"success": False, "reason": "occupied"}
        
        self.positions = final_positions
        return results
    
    def _get_adjacent_position(self, pos: Tuple[int, int], direction_idx: int) -> Tuple[int, int]:
        """Get adjacent hexagon position in given direction."""
        q, r = pos
        dq, dr = self.hex_directions[direction_idx]
        return (q + dq, r + dr)
    
    def _attempt_pass(self, passer_id: int, direction_idx: int) -> Dict:
        """Attempt a pass from the ball holder in a specific direction."""
        passer_pos = self.positions[passer_id]
        
        # Project a line of sight from the passer
        # We check up to a max pass distance of 5 hexes
        for i in range(1, 6): 
            vec = self.hex_directions[direction_idx]
            target_pos = (passer_pos[0] + vec[0] * i, passer_pos[1] + vec[1] * i)

            # If pass goes out of bounds, it's a turnover
            if not self._is_valid_position(*target_pos):
                self._turnover_to_defense(passer_id)
                return {"success": False, "reason": "out_of_bounds", "turnover": True}
            
            # Check if any player is at the target position
            for player_id, pos in enumerate(self.positions):
                if pos == target_pos:
                    is_teammate = (player_id in self.offense_ids and passer_id in self.offense_ids) or \
                                  (player_id in self.defense_ids and passer_id in self.defense_ids)
                    
                    if is_teammate:
                        # Successful pass
                        self.ball_holder = player_id
                        return {"success": True, "target": player_id, "turnover": False}
                    else:
                        # Intercepted by opponent
                        self._turnover_to_defense(passer_id)
                        return {"success": False, "reason": "intercepted", "turnover": True}

        # If the pass finds no one in range, it's a turnover
        self._turnover_to_defense(passer_id)
        return {"success": False, "reason": "no_target", "turnover": True}
    
    def _attempt_shot(self, shooter_id: int) -> Dict:
        """Attempt a shot from the ball holder."""
        shooter_pos = self.positions[shooter_id]
        basket_pos = self.basket_position
        
        distance = self._hex_distance(shooter_pos, basket_pos)
        
        # Simple fixed probability for now
        shot_success_prob = 0.4 
        shot_made = self._rng.random() < shot_success_prob
        
        if not shot_made:
            # Missed shot - rebound to defense
            self._turnover_to_defense(shooter_id)
        
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
        if distance <= 2:
            return 0.7  # Close shot
        elif distance <= 4:
            return 0.5  # Mid-range
        else:
            return 0.3  # Long shot

    def _check_termination_and_rewards(self, action_results: Dict) -> Tuple[bool, np.ndarray]:
        """Check if episode should terminate and calculate rewards."""
        rewards = np.zeros(self.n_players)
        done = False
        
        # Check for shots
        for player_id, shot_result in action_results["shots"].items():
            done = True  # Episode ends after any shot attempt
            
            if shot_result["success"]:
                # Basket made
                if player_id in self.offense_ids:
                    # Offense scored
                    if self.training_team == Team.OFFENSE:
                        rewards[self.offense_ids] = 1.0  # Reward offense
                    else:
                        rewards[self.defense_ids] = -1.0  # Penalty for defense
                else:
                    # Defense scored (rare but possible)
                    if self.training_team == Team.DEFENSE:
                        rewards[self.defense_ids] = 1.0
                    else:
                        rewards[self.offense_ids] = -1.0
        
        # Check for turnovers from passes or moving out of bounds
        if action_results.get("out_of_bounds_turnover", False):
            done = True
        
        for player_id, pass_result in action_results["passes"].items():
            if pass_result.get("turnover", False):
                done = True  # Episode ends on turnover
        
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
    
    def render(self, mode: str = "human"):
        """Render the current state of the environment."""
        if mode == "human":
            return self._render_ascii()
        elif mode == "rgb_array":
            return self._render_visual()
        else:
            raise ValueError(f"Unsupported render mode: {mode}")
    
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
        
        # Convert to RGB array
        fig.canvas.draw()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
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
    print(f"Initial observation shape: {obs.shape}")
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
