# basketworld_3x3_env.py
"""
Gym‑compatible grid‑world environment for half‑court 3‑on‑3 basketball with
simultaneous actions and a *single shared* policy.  The environment is designed
as a pedagogical starting point for experiments in multi‑agent/parameter‑sharing
reinforcement learning with PyTorch RL algorithms such as PPO or A2C.

Key design decisions
--------------------
* **Simultaneous turns**: Every timestep the environment receives one action per
  player (6 actions total) and resolves them together.
* **Shared policy**:  All six agents share one neural network; the agent's role
  (offense/defense), whether it currently has the ball, and its unique ID are
  embedded in the observation so the policy can condition behaviour.
* **Role‑based rewards**:  Zero‑sum rewards are reshaped into *individual* (+/‑)
  signals so that gradients do not cancel when using a shared policy.

The code is intentionally compact and heavily commented so you can customise it
quickly.  Mechanics such as collision resolution, steal/shot success formulas,
pass validation, and reward magnitudes are simple heuristics — feel free to
refine them for more realism.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class Basketball3v3Env(gym.Env):
    """Half‑court 3‑on‑3 basketball grid world (12×8)."""

    metadata = {"render.modes": ["human"]}

    WIDTH = 12
    HEIGHT = 8
    N_PLAYERS = 6  # 0‑2 offense, 3‑5 defense
    OFFENSE_IDS = (0, 1, 2)
    DEFENSE_IDS = (3, 4, 5)
    SHOOT_RANGE = 3  # Manhattan distance
    MAX_STEPS = 48   # ~24‑second shot clock at 0.5s per grid step

    ACTIONS = {
        0: "stay",
        1: "up",
        2: "down",
        3: "left",
        4: "right",
        5: "pass",
        6: "shoot",
        7: "steal",
    }
    N_ACTIONS = len(ACTIONS)

    def __init__(self, seed: int | None = None):
        super().__init__()

        # All agents share the same discrete action space
        self.single_action_space = spaces.Discrete(self.N_ACTIONS)
        # Environment accepts a *dict* of agent_id -> action
        # Gym's top‑level action_space must still be defined; we use MultiDiscrete
        self.action_space = spaces.MultiDiscrete([self.N_ACTIONS] * self.N_PLAYERS)

        # Observation encoding:
        #   channels = [offense_map, defense_map, ball_map, self_map]
        obs_shape = (4, self.HEIGHT, self.WIDTH)
        self.single_observation_space = spaces.Box(0, 1, shape=obs_shape, dtype=np.float32)
        self.observation_space = spaces.Tuple([self.single_observation_space] * self.N_PLAYERS)

        self._rng = np.random.default_rng(seed)

        # State
        self.positions: List[Tuple[int, int]] = []  # (x, y) for each player
        self.ball_handler: int = 0                 # player ID w/ the ball
        self.steps: int = 0

    # ---------------------------------------------------------------------
    # Gym API
    # ---------------------------------------------------------------------
    def reset(self):
        self.steps = 0
        # Random non‑overlapping spawn positions within half court
        self.positions = self._sample_start_positions()
        # Random offensive player starts with ball
        self.ball_handler = self._rng.choice(self.OFFENSE_IDS)
        return self._get_all_observations()

    def step(self, actions: Dict[int, int] | np.ndarray):
        """Accepts dict {player_id: action} or ndarray of shape (6,)."""
        self.steps += 1
        # Convert ndarray to dict if necessary
        if isinstance(actions, np.ndarray):
            actions = {pid: int(a) for pid, a in enumerate(actions.tolist())}

        rewards = {pid: 0.0 for pid in range(self.N_PLAYERS)}
        done = False
        info = {}

        # First resolve *movement* intents to avoid sequential bias
        target_positions = self.positions.copy()
        for pid, action in actions.items():
            if action in (1, 2, 3, 4):  # movement actions
                target_positions[pid] = self._move(self.positions[pid], action)

        # Handle collisions: if two players target same cell, both stay
        occupied: Dict[Tuple[int, int], int] = {}
        for pid, pos in enumerate(target_positions):
            if pos in occupied:
                # collision → both stay at original positions
                other_pid = occupied[pos]
                target_positions[pid] = self.positions[pid]
                target_positions[other_pid] = self.positions[other_pid]
            else:
                occupied[pos] = pid
        self.positions = target_positions

        # Second, resolve *ball actions* (pass, shoot, steal)
        for pid, action in actions.items():
            role_offense = pid in self.OFFENSE_IDS
            has_ball = pid == self.ball_handler

            # -------------------- PASS --------------------
            if action == 5 and has_ball:
                teammate_id = self._adjacent_teammate(pid)
                if teammate_id is not None:
                    self.ball_handler = teammate_id
                    # small positive for successful pass
                    rewards[pid] += 0.1
                else:
                    # failed pass turnover to defense
                    self.ball_handler = self._nearest_defender(pid)
                    rewards[pid] -= 1.0  # penalise risky pass

            # -------------------- SHOOT -------------------
            elif action == 6 and has_ball:
                dist = self._manhattan(self.positions[pid], (self.WIDTH - 1, self.HEIGHT // 2))
                if dist <= self.SHOOT_RANGE:
                    success_prob = max(0.1, 0.9 - 0.2 * dist)  # simple formula
                    if random.random() < success_prob:
                        # basket!
                        for o in self.OFFENSE_IDS:
                            rewards[o] += 1.0
                        for d in self.DEFENSE_IDS:
                            rewards[d] -= 1.0
                        done = True
                    else:
                        # missed shot, rebound to random defender
                        self.ball_handler = self._rng.choice(self.DEFENSE_IDS)
                        for o in self.OFFENSE_IDS:
                            rewards[o] -= 1.0
                        for d in self.DEFENSE_IDS:
                            rewards[d] += 1.0
                else:
                    # out‑of‑range attempt is turnover
                    self.ball_handler = self._rng.choice(self.DEFENSE_IDS)
                    rewards[pid] -= 1.0

            # -------------------- STEAL -------------------
            elif action == 7 and (not role_offense):
                bh_pos = self.positions[self.ball_handler]
                if self._manhattan(self.positions[pid], bh_pos) == 1:
                    if random.random() < 0.5:  # steal probability
                        # Successful steal!
                        self.ball_handler = pid
                        for d in self.DEFENSE_IDS:
                            rewards[d] += 1.0
                        for o in self.OFFENSE_IDS:
                            rewards[o] -= 1.0

        # Shot clock / max step termination
        if self.steps >= self.MAX_STEPS:
            done = True

        obs = self._get_all_observations()
        # Convert rewards dict to list aligned with obs tuple ordering
        rewards_tuple = tuple(rewards[pid] for pid in range(self.N_PLAYERS))
        return obs, rewards_tuple, done, info

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------
    def _sample_start_positions(self) -> List[Tuple[int, int]]:
        """Random, non‑overlapping positions within court."""
        coords = set()
        positions = []
        while len(positions) < self.N_PLAYERS:
            x = self._rng.integers(0, self.WIDTH // 2)  # start left half‑court
            y = self._rng.integers(0, self.HEIGHT)
            if (x, y) not in coords:
                coords.add((x, y))
                positions.append((x, y))
        return positions

    def _move(self, pos: Tuple[int, int], action: int) -> Tuple[int, int]:
        x, y = pos
        if action == 1:  # up
            y = max(0, y - 1)
        elif action == 2:  # down
            y = min(self.HEIGHT - 1, y + 1)
        elif action == 3:  # left
            x = max(0, x - 1)
        elif action == 4:  # right
            x = min(self.WIDTH - 1, x + 1)
        return x, y

    def _adjacent_teammate(self, pid: int) -> int | None:
        for tid in (self.OFFENSE_IDS if pid in self.OFFENSE_IDS else self.DEFENSE_IDS):
            if tid == pid:
                continue
            if self._manhattan(self.positions[pid], self.positions[tid]) == 1:
                return tid
        return None

    def _nearest_defender(self, pid: int) -> int:
        # Return closest defender to pid (break ties randomly)
        distances = [
            (self._manhattan(self.positions[pid], self.positions[d]), d) for d in self.DEFENSE_IDS
        ]
        min_dist = min(distances, key=lambda t: t[0])[0]
        candidates = [d for dist, d in distances if dist == min_dist]
        return self._rng.choice(candidates)

    @staticmethod
    def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # ------------------------------------------------------------------
    # Observation Encoding
    # ------------------------------------------------------------------
    def _encode_observation(self, agent_id: int) -> np.ndarray:
        grid = np.zeros((4, self.HEIGHT, self.WIDTH), dtype=np.float32)
        # Channel 0: offense positions
        for o in self.OFFENSE_IDS:
            x, y = self.positions[o]
            grid[0, y, x] = 1.0
        # Channel 1: defense positions
        for d in self.DEFENSE_IDS:
            x, y = self.positions[d]
            grid[1, y, x] = 1.0
        # Channel 2: ball position
        bx, by = self.positions[self.ball_handler]
        grid[2, by, bx] = 1.0
        # Channel 3: self position
        sx, sy = self.positions[agent_id]
        grid[3, sy, sx] = 1.0
        return grid

    def _get_all_observations(self):
        return tuple(self._encode_observation(pid) for pid in range(self.N_PLAYERS))

    # ------------------------------------------------------------------
    # Rendering (ASCII and Visual)
    # ------------------------------------------------------------------
    def render(self, mode="human"):
        """Render the environment. Returns visual grid if mode='rgb_array'."""
        grid = [["·" for _ in range(self.WIDTH)] for _ in range(self.HEIGHT)]
        # Draw hoop
        hx, hy = self.WIDTH - 1, self.HEIGHT // 2
        grid[hy][hx] = "⊙"
        # Draw players
        for pid, (x, y) in enumerate(self.positions):
            symbol = str(pid)
            grid[y][x] = symbol
        # Mark ball handler with '*'
        bx, by = self.positions[self.ball_handler]
        grid[by][bx] = f"*{self.ball_handler}*"[-1]  # keep single char
        
        if mode == "human":
            print("\n".join(" ".join(row) for row in grid))
        elif mode == "rgb_array":
            return self._grid_to_rgb()
        return grid

    def _grid_to_rgb(self):
        """Convert ASCII grid to RGB array for visualization."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlim(0, self.WIDTH)
        ax.set_ylim(0, self.HEIGHT)
        ax.set_aspect('equal')
        
        # Draw court
        court = patches.Rectangle((0, 0), self.WIDTH, self.HEIGHT, 
                                linewidth=2, edgecolor='black', facecolor='white')
        ax.add_patch(court)
        
        # Draw realistic basketball hoop
        hoop_x, hoop_y = self.WIDTH - 0.5, self.HEIGHT // 2
        
        # Backboard
        backboard = patches.Rectangle((self.WIDTH - 0.15, hoop_y - 0.5), 0.15, 1.0,
                                    linewidth=2, edgecolor='black', facecolor='white')
        ax.add_patch(backboard)
        
        # Rim (outer ring)
        rim_outer = patches.Circle((hoop_x - 0.2, hoop_y), 0.35, 
                                 fill=False, color='orange', linewidth=4)
        ax.add_patch(rim_outer)
        
        # Rim (inner circle for depth)
        rim_inner = patches.Circle((hoop_x - 0.2, hoop_y), 0.3, 
                                 fill=False, color='darkorange', linewidth=2)
        ax.add_patch(rim_inner)
        
        # Net lines (simple representation)
        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
            x_start = (hoop_x - 0.2) + 0.3 * np.cos(angle)
            y_start = hoop_y + 0.3 * np.sin(angle)
            x_end = (hoop_x - 0.2) + 0.2 * np.cos(angle)
            y_end = hoop_y - 0.3 + 0.2 * np.sin(angle)
            ax.plot([x_start, x_end], [y_start, y_end], 'white', linewidth=1, alpha=0.7)
        
        # Add NBA-style BW watermark with center circle
        center_x, center_y = self.WIDTH / 2, self.HEIGHT / 2
        
        # Center circle (like NBA courts)
        center_circle = patches.Circle((center_x, center_y), 2.0, 
                                     fill=False, color='lightgray', linewidth=2, alpha=0.3)
        ax.add_patch(center_circle)
        
        # Inner circle for logo background
        logo_bg = patches.Circle((center_x, center_y), 1.5, 
                               facecolor='white', edgecolor='lightgray', 
                               linewidth=1, alpha=0.2)
        ax.add_patch(logo_bg)
        
        # BW logo text with stylistic font
        ax.text(center_x, center_y, 'BW', fontsize=48, 
               ha='center', va='center', alpha=0.15, fontweight='bold', 
               color='darkblue', family='serif', style='italic')
        
        # Draw players
        colors = ['blue', 'blue', 'blue', 'red', 'red', 'red']  # offense=blue, defense=red
        for pid, (x, y) in enumerate(self.positions):
            player_circle = patches.Circle((x + 0.5, self.HEIGHT - y - 0.5), 0.3, 
                                         color=colors[pid], alpha=0.7)
            ax.add_patch(player_circle)
            
            # Add player number
            ax.text(x + 0.5, self.HEIGHT - y - 0.5, str(pid), 
                   ha='center', va='center', fontweight='bold', color='white')
            
            # Mark ball handler with ring
            if pid == self.ball_handler:
                ball_ring = patches.Circle((x + 0.5, self.HEIGHT - y - 0.5), 0.4, 
                                         fill=False, color='orange', linewidth=5)
                ax.add_patch(ball_ring)
        
        ax.set_title(f'BasketWorld 3v3 - Step {self.steps}')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(self.WIDTH + 1))
        ax.set_yticks(range(self.HEIGHT + 1))
        
        # Convert plot to RGB array
        import io
        fig.canvas.draw()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        
        from PIL import Image
        img = Image.open(buf)
        rgb_array = np.array(img)
        plt.close(fig)
        buf.close()
        
        return rgb_array


# ---------------------------------------------------------------------
# Quick sanity test / example usage
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter, FFMpegWriter, FuncAnimation
    
    # Configuration
    output_format = "gif"  # Options: "gif" or "mp4"
    fps = 2
    
    env = Basketball3v3Env(seed=42)
    obs = env.reset()
    
    # Collect frames for animation
    frames = []
    frames.append(env.render(mode="rgb_array"))  # Initial frame
    
    done = False
    step_count = 0
    max_steps = 48  # Limit for demo
    
    while not done and step_count < max_steps:
        # Random simultaneous actions
        actions = env.action_space.sample()
        obs, rewards, done, _ = env.step(actions)
        
        # Collect frame
        frame = env.render(mode="rgb_array")
        frames.append(frame)
        
        # Also print to console
        env.render(mode="human")
        print("Rewards:", rewards, "\n")
        step_count += 1
    
    # Create animation
    if frames:
        print(f"Creating animated {output_format}...")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('off')
        
        # Create animation
        im = ax.imshow(frames[0])
        
        def animate(frame_idx):
            im.set_array(frames[frame_idx])
            return [im]
        
        anim = FuncAnimation(fig, animate, frames=len(frames), 
                           interval=1000//fps, blit=True)
        
        # Save based on format
        if output_format == "gif":
            writer = PillowWriter(fps=fps)
            filename = 'basketball_game.gif'
            anim.save(filename, writer=writer)
        elif output_format == "mp4":
            writer = FFMpegWriter(fps=fps, bitrate=1800)
            filename = 'basketball_game.mp4'
            anim.save(filename, writer=writer)
        else:
            raise ValueError(f"Unsupported format: {output_format}")
        
        plt.close()
        print(f"Animation saved as '{filename}' with {len(frames)} frames!")
