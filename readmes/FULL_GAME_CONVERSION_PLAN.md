# Full-Game Basketball Conversion Plan

## Executive Summary

This document outlines a plan to convert the current single-possession `HexagonBasketballEnv` into a full-game basketball simulation featuring:
- Full-court basketball (mirrored court with baskets at each end)
- Multiple possessions per episode
- Game quarters with clock management
- Rebounds (offensive and defensive) with appropriate shot clock resets
- Inbounding mechanics after made baskets
- Dynamic role switching for the training team based on possession
- Continued use of frozen opponent pool for self-play training

---

## Table of Contents

1. [Current State Analysis](#current-state-analysis)
2. [High-Level Architecture Changes](#high-level-architecture-changes)
3. [Implementation Phases](#implementation-phases)
4. [Detailed Component Design](#detailed-component-design)
5. [Training Loop Modifications](#training-loop-modifications)
6. [Testing Strategy](#testing-strategy)
7. [Potential Challenges](#potential-challenges)

---

## 1. Current State Analysis

### Current Environment Structure
- **Grid**: 16x16 hexagonal grid (configurable)
- **Game Flow**: Single possession ending on made basket, miss, turnover, or shot clock violation
- **Training**: Alternating between offense and defense training phases
- **Teams**: Offense (Team.OFFENSE) vs Defense (Team.DEFENSE)
- **Termination**: Episode ends after single possession outcome

### Key Components to Preserve
- Hexagonal grid mechanics
- Action types (movement, passing, shooting)
- Defender pressure system
- Shot probability mechanics (layups, 3-pointers, dunks)
- 3-second violation rules
- Action masking
- Observation space structure
- Self-play training infrastructure

---

## 2. High-Level Architecture Changes

### 2.1 Court Representation

**Current**: Half-court with single basket
```
Offense attacks → [Basket]
```

**Proposed**: Full-court with mirrored baskets
```
[Basket A] ← Team A attacks / Team B defends
    ... court ...
[Basket B] ← Team B attacks / Team A defends
```

**Implementation Approach**:
- Mirror the current court geometry along the half-court line
- Each basket retains its own 3-point arc, layup zone, and 3-second lane
- Basket locations become `(basket_a_row, basket_a_col)` and `(basket_b_row, basket_b_col)`
- Current basket is determined by which team has possession

### 2.2 Team Role System

**Current**: 
- Fixed roles: Team.OFFENSE (training) vs Team.DEFENSE (opponent)
- Switch between training phases

**Proposed**:
- Two persistent teams: Team.A and Team.B
- **Role flag** (0 or 1) indicates possession:
  - Role 0: Has possession (attacks)
  - Role 1: Plays defense
- Training team always maintains same identity but switches roles based on possession
- Opponent pool contains frozen policies that can handle both offensive and defensive roles

### 2.3 Episode Structure

**Current**: Single possession → Episode ends

**Proposed**: 
```
Game Start
├── Quarter 1 (e.g., 12 minutes → ~288 game steps at 2.5s per step)
│   ├── Initial tip-off possession
│   ├── Multiple possessions with role switches
│   └── Quarter end
├── Quarter 2
├── Quarter 3
├── Quarter 4
└── Game End (score determines winner)
```

Alternative simpler structure (Phase 1):
```
Game Start
├── First Half (e.g., 24 minutes)
├── Second Half (e.g., 24 minutes)
└── Game End
```

Or simplest (Phase 0):
```
Game Start
├── Single period (e.g., 10 minutes or N possessions)
└── Game End
```

---

## 3. Implementation Phases

### Phase 0: Foundation (No Game Clock)
**Goal**: Get basic full-court multi-possession working without game clock

**Components**:
1. Full-court geometry with two baskets
2. Basic possession tracking
3. Simple inbounding after made basket
4. Shot clock for each possession (24 seconds)
5. Role switching on possession change
6. Fixed number of possessions per episode (e.g., 10 possessions)

**Training**: 
- Training team switches roles dynamically
- Episode ends after N possessions or large score differential
- Reward shaping for cumulative score difference

### Phase 1: Game Clock & Periods
**Goal**: Add game clock and structured periods

**Components**:
1. Game clock (counts down from period length)
2. Period structure (quarters or halves)
3. End-of-period logic
4. Timeout management (optional)
5. Possession at end of period handling

### Phase 2: Rebounds
**Goal**: Add offensive and defensive rebounding mechanics

**Components**:
1. Rebound probability based on shot distance and defender proximity
2. Rebound location distribution
3. Box-out mechanics (optional)
4. Shot clock reset rules:
   - Offensive rebound: Reset to 14 seconds (or 24 if current > 14)
   - Defensive rebound: Reset to 24 seconds
5. Loose ball mechanics

### Phase 3: Advanced Mechanics
**Goal**: Add remaining basketball rules and refinements

**Components**:
1. Fast break detection and rewards
2. Half-court transition rules
3. Backcourt violation (8-second rule, can't return to backcourt)
4. Jump balls (held ball, simultaneous possession)
5. Advanced inbounding (sideline, baseline)
6. Fouls (optional but impactful)

---

## 4. Detailed Component Design

### 4.1 Court Geometry

#### Grid Layout (16x16 example)
```python
# Current basket at row 0, middle column
BASKET_A_ROW = 0
BASKET_A_COL = grid_size // 2

# Mirror basket at row grid_size-1, middle column  
BASKET_B_ROW = grid_size - 1
BASKET_B_COL = grid_size // 2

# Half-court line
HALF_COURT_ROW = grid_size // 2
```

#### Coordinate Transformation
When possession changes, need to map:
- Shot distance calculation to appropriate basket
- 3-point arcs to appropriate basket
- Layup zones to appropriate basket
- 3-second lanes to appropriate basket

```python
def get_target_basket(self, possessing_team):
    """Return (row, col) of basket that possessing team attacks."""
    if possessing_team == Team.A:
        return (self.basket_b_row, self.basket_b_col)
    else:
        return (self.basket_a_row, self.basket_a_col)

def distance_to_target_basket(self, row, col, possessing_team):
    """Calculate distance to the basket being attacked."""
    target_row, target_col = self.get_target_basket(possessing_team)
    return self.hex_distance((row, col), (target_row, target_col))
```

### 4.2 Possession Tracking

```python
class PossessionState:
    """Track current possession information."""
    def __init__(self):
        self.possessing_team = Team.A  # Team.A or Team.B
        self.possession_number = 0
        self.shot_clock = 24
        self.last_possession_change_reason = None  # "made_basket", "miss", "turnover", "steal"
        
class GameState:
    """Track overall game state."""
    def __init__(self):
        self.game_clock = 720  # 12 minutes = 720 seconds
        self.period = 1
        self.score_a = 0
        self.score_b = 0
        self.possession = PossessionState()
```

### 4.3 Role Flag System

The role flag determines whether agents see themselves as offense or defense:

```python
def get_role_flag(self, player_idx):
    """
    Return role flag for a player.
    0 = Offense (has possession)
    1 = Defense (opposing possession)
    """
    player_team = self.get_player_team(player_idx)
    if player_team == self.possession.possessing_team:
        return 0  # Offense
    else:
        return 1  # Defense
```

**Key Insight**: The training team (e.g., Team.A) always trains, but sees different role flags depending on possession. This means:
- Same policy must handle both offensive and defensive play
- Observation space includes role flag to condition behavior
- Opponent pool policies must also handle both roles

### 4.4 Observation Space Modifications

Add to current observation space:

```python
obs_additions = {
    'role_flag': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
    'game_clock': spaces.Box(low=0, high=3600, shape=(1,), dtype=np.float32),  # Normalized
    'period': spaces.Box(low=1, high=4, shape=(1,), dtype=np.int32),
    'score_difference': spaces.Box(low=-100, high=100, shape=(1,), dtype=np.float32),
    'possession_team': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
    'basket_a_direction': spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),  # Direction to basket A
    'basket_b_direction': spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),  # Direction to basket B
    'target_basket_distance': spaces.Box(low=0, high=grid_size*2, shape=(1,), dtype=np.float32),
}
```

### 4.5 Inbounding Mechanics

After a made basket, the ball must be inbounded:

```python
class InboundState(Enum):
    NONE = 0  # Normal play
    BASELINE_INBOUND = 1  # After made basket
    SIDELINE_INBOUND = 2  # After out-of-bounds
    
def handle_made_basket(self, points):
    """Handle scoring and possession change."""
    # Update score
    if self.possession.possessing_team == Team.A:
        self.game_state.score_a += points
        self.possession.possessing_team = Team.B
    else:
        self.game_state.score_b += points
        self.possession.possessing_team = Team.A
    
    # Set up inbound
    self.inbound_state = InboundState.BASELINE_INBOUND
    self.place_inbounder()
    self.place_defenders()
    self.shot_clock = 24
    
def place_inbounder(self):
    """Place inbounder at baseline behind their own basket."""
    # Inbounder has special "inbound pass" actions available
    # Ball is at inbounder's location
    # Inbounder can pass to any teammate
    target_basket = self.get_target_basket(self.possession.possessing_team)
    baseline_row = 0 if target_basket[0] == self.grid_size - 1 else self.grid_size - 1
    inbound_position = (baseline_row, self.grid_size // 2)
    # Place ball and designate inbounder
    self.ball_holder = self.get_player_at_position(inbound_position)
```

**Simplified Inbounding (Phase 0)**:
- After made basket, ball spawns at designated inbound location
- Random player from new possession team gets the ball
- Opponents must be at least N hexes away
- Normal play resumes immediately

**Advanced Inbounding (Phase 3)**:
- Designated inbounder (can't move until pass)
- 5-second violation if pass not made
- Full-court press opportunities
- Special inbound actions

### 4.6 Rebound System (Phase 2)

```python
def resolve_missed_shot(self, shooter_pos, shot_distance):
    """Handle missed shot and rebound."""
    # Determine rebound probability
    rebound_probs = self.calculate_rebound_probabilities(shooter_pos, shot_distance)
    
    # Sample rebounder or no rebound (out of bounds)
    rebounder_idx = self.sample_rebounder(rebound_probs)
    
    if rebounder_idx is None:
        # Ball out of bounds - possession to defending team
        self.change_possession(reason="out_of_bounds")
        self.inbound_state = InboundState.BASELINE_INBOUND
        return
    
    rebounder_team = self.get_player_team(rebounder_idx)
    
    if rebounder_team == self.possession.possessing_team:
        # Offensive rebound
        self.handle_offensive_rebound(rebounder_idx)
    else:
        # Defensive rebound
        self.handle_defensive_rebound(rebounder_idx)

def calculate_rebound_probabilities(self, shooter_pos, shot_distance):
    """
    Calculate rebound probability for each player.
    Factors:
    - Distance to basket
    - Box-out positioning (defenders between offensive players and basket)
    - Height/strength (could add player attributes)
    """
    basket_pos = self.get_target_basket(self.possession.possessing_team)
    probs = np.zeros(self.n_players)
    
    for idx in range(self.n_players):
        player_pos = self.player_positions[idx]
        distance_to_basket = self.hex_distance(player_pos, basket_pos)
        
        # Closer players more likely to rebound
        # Longer shots tend to rebound further
        ideal_rebound_distance = min(3, shot_distance // 2)
        distance_factor = np.exp(-abs(distance_to_basket - ideal_rebound_distance))
        
        probs[idx] = distance_factor
    
    # Normalize
    total = probs.sum()
    if total > 0:
        probs /= total
    
    return probs

def handle_offensive_rebound(self, rebounder_idx):
    """Handle offensive rebound."""
    self.ball_holder = rebounder_idx
    # Reset shot clock to 14 seconds (or keep current if < 14)
    self.shot_clock = max(self.shot_clock, 14)
    # Optional: Small reward for offensive rebound
    self.add_reward(rebounder_idx, 0.1, reason="offensive_rebound")

def handle_defensive_rebound(self, rebounder_idx):
    """Handle defensive rebound - possession change."""
    self.possession.possessing_team = self.get_player_team(rebounder_idx)
    self.ball_holder = rebounder_idx
    self.shot_clock = 24
    # Optional: Small reward for defensive rebound
    self.add_reward(rebounder_idx, 0.1, reason="defensive_rebound")
```

### 4.7 Game Clock & Period Management (Phase 1)

```python
def step(self, actions):
    """Execute one step - now includes game clock management."""
    # ... existing step logic ...
    
    # Update game clock (each step = ~2.5 seconds of game time)
    self.game_state.game_clock -= 2.5
    
    # Check for end of period
    if self.game_state.game_clock <= 0:
        self.end_period()
    
    # ... rest of step logic ...

def end_period(self):
    """Handle end of period."""
    self.game_state.period += 1
    
    if self.game_state.period > 4:  # Or 2 for halves
        # Game over
        self.episode_ended = True
        self.calculate_final_rewards()
        return
    
    # Reset clock for new period
    self.game_state.game_clock = 720  # 12 minutes
    
    # Possession alternates each period (or use jump ball)
    # Could also give possession to team that started previous period on defense
    self.alternate_possession()
    
    # Set up for new period start
    self.reset_positions_for_period_start()

def calculate_final_rewards(self):
    """Calculate final game outcome rewards."""
    score_diff = self.game_state.score_a - self.game_state.score_b
    
    # Large win bonus/penalty
    if self.training_team == Team.A:
        win_reward = 10.0 if score_diff > 0 else -10.0
    else:
        win_reward = 10.0 if score_diff < 0 else -10.0
    
    # Add to all training team players
    for idx in self.get_training_team_indices():
        self.add_reward(idx, win_reward, reason="final_outcome")
```

### 4.8 Training Loop Integration

The training loop needs minimal changes because:

1. **Single Training Team**: Only one team (e.g., Team.A) is actively training
2. **Opponent Pool**: Contains frozen policies that can play both offense and defense
3. **Role Switching**: Handled automatically by environment based on possession

```python
def make_env(training_team=Team.A, opponent_policy=None):
    """Create environment for training."""
    env = HexagonBasketballFullGameEnv(
        grid_size=16,
        players_per_side=3,
        training_team=training_team,  # Always Team.A during training
        # New parameters
        game_mode='full_game',
        possessions_per_episode=20,  # Phase 0: fixed possessions
        # use_game_clock=False,  # Phase 0: no clock
    )
    
    env = SelfPlayEnvWrapper(
        env, 
        opponent_policy=opponent_policy,
        opponent_handles_both_roles=True,  # NEW FLAG
    )
    
    return env
```

**Key Change in SelfPlayEnvWrapper**:
```python
def step(self, actions):
    """
    Wrapper now needs to route actions based on current possession.
    """
    # Get current possession from environment
    current_possession = self.env.possession.possessing_team
    
    if current_possession == self.env.training_team:
        # Training team has ball (offense role)
        # Use provided actions for training team
        # Use opponent policy for opponents (defense role)
        full_actions = self.combine_actions(
            training_actions=actions,
            role_for_training=0,  # offense
            role_for_opponent=1,  # defense
        )
    else:
        # Opponent has ball (training team on defense)
        # Use provided actions for training team  
        # Use opponent policy for opponents (offense role)
        full_actions = self.combine_actions(
            training_actions=actions,
            role_for_training=1,  # defense
            role_for_opponent=0,  # offense
        )
    
    return self.env.step(full_actions)
```

---

## 5. Training Loop Modifications

### 5.1 Reward Structure Changes

**Current**: 
- Binary outcome (+1 for made basket, -1 for loss of possession)
- Optional potential-based shaping

**Proposed Multi-Level Rewards**:

```python
# Immediate rewards (per-step)
immediate_rewards = {
    'made_basket_2pt': +2.0,
    'made_basket_3pt': +3.0,
    'defensive_stop': +0.5,
    'offensive_rebound': +0.3,
    'defensive_rebound': +0.2,
    'assist': +0.5,
    'steal': +0.8,
    'turnover': -0.8,
    'shot_clock_violation': -0.5,
}

# Cumulative rewards (per-episode)
cumulative_rewards = {
    'final_score_difference': score_diff * 0.1,  # Continuous reward
    'win_bonus': +10.0 if won else -10.0,
    'efficiency_bonus': (points / possessions) * 0.5,
}

# Potential-based shaping (optional)
# Updated to consider score and clock
def phi(state):
    score_diff = state.score_a - state.score_b  # From training team perspective
    time_remaining = state.game_clock + (4 - state.period) * 720
    possession_value = 1.0 if state.possession == training_team else -0.5
    
    return score_diff * 0.5 + possession_value + (time_remaining / 2880) * score_diff
```

### 5.2 Episode Length Considerations

**Phase 0 (No Game Clock)**:
- Fixed N possessions per episode (e.g., 10-20 possessions)
- Or fixed time limit (e.g., 500 steps)
- Episodes shorter initially for faster iteration

**Phase 1 (With Game Clock)**:
- Full game = 4 quarters × 12 minutes = 48 minutes
- At 2.5 seconds per step: 48 min × 60 sec / 2.5 sec = ~1152 steps
- May want shorter games initially (e.g., 4 minute quarters = ~384 steps)

**Training Implications**:
- Longer episodes → fewer episodes per training iteration
- May need to adjust `n_steps` in PPO to accumulate enough data
- Potentially use episode truncation for early training phases

### 5.3 Opponent Pool Updates

Opponent pool policies must now handle both roles:

```python
def save_policy_to_pool(policy, timestep, metrics):
    """Save policy with both offensive and defensive capabilities."""
    policy_entry = {
        'policy': policy,
        'timestep': timestep,
        'metrics': metrics,
        'offensive_skill': metrics.get('points_per_possession', 0.0),
        'defensive_skill': metrics.get('stops_per_possession', 0.0),
    }
    opponent_pool.append(policy_entry)

def sample_opponent(strategy='recent'):
    """Sample opponent from pool."""
    if strategy == 'recent':
        # Weight toward recent policies
        weights = np.exp(np.linspace(-2, 0, len(opponent_pool)))
    elif strategy == 'skill_matched':
        # Match current policy skill level
        current_skill = estimate_current_skill()
        weights = [1.0 / (1.0 + abs(p['metrics']['overall_skill'] - current_skill)) 
                   for p in opponent_pool]
    
    weights = np.array(weights) / np.sum(weights)
    return np.random.choice(opponent_pool, p=weights)
```

### 5.4 Evaluation Metrics

New metrics to track:

```python
evaluation_metrics = {
    # Game-level
    'games_won': int,
    'average_score_difference': float,
    'average_final_score': float,
    
    # Possession-level  
    'points_per_possession': float,
    'offensive_efficiency': float,  # Points per 100 possessions
    'defensive_efficiency': float,  # Points allowed per 100 possessions
    
    # Advanced
    'offensive_rebound_rate': float,
    'defensive_rebound_rate': float,
    'turnover_rate': float,
    'steal_rate': float,
    'fast_break_points': int,  # Phase 3
    
    # Clock management (Phase 1)
    'average_possession_length': float,
    'late_game_performance': float,  # Performance in last 2 minutes
}
```

---

## 6. Testing Strategy

### 6.1 Unit Tests

```python
# Test court geometry
def test_full_court_geometry():
    env = HexagonBasketballFullGameEnv()
    assert env.basket_a_row == 0
    assert env.basket_b_row == env.grid_size - 1
    
def test_distance_to_baskets():
    env = HexagonBasketballFullGameEnv()
    # Test distance calculations for both baskets
    
def test_role_flag_switching():
    env = HexagonBasketballFullGameEnv()
    # Verify role flags switch correctly on possession change

# Test possession mechanics
def test_possession_change_on_made_basket():
    env = HexagonBasketballFullGameEnv()
    env.reset()
    initial_possession = env.possession.possessing_team
    # Simulate made basket
    env.handle_made_basket(points=2)
    assert env.possession.possessing_team != initial_possession
    assert env.shot_clock == 24
    
def test_inbounding():
    env = HexagonBasketballFullGameEnv()
    # Test inbound setup after made basket
    
# Test shot clock
def test_shot_clock_reset_offensive_rebound():
    env = HexagonBasketballFullGameEnv()
    env.shot_clock = 5
    env.handle_offensive_rebound(player_idx=0)
    assert env.shot_clock == 14
    
def test_shot_clock_reset_defensive_rebound():
    env = HexagonBasketballFullGameEnv()
    env.handle_defensive_rebound(player_idx=3)
    assert env.shot_clock == 24
```

### 6.2 Integration Tests

```python
def test_full_game_flow():
    """Test complete game from start to finish."""
    env = HexagonBasketballFullGameEnv(possessions_per_episode=10)
    obs = env.reset()
    
    possession_count = 0
    done = False
    
    while not done:
        actions = env.action_space.sample()  # Random actions
        obs, rewards, done, info = env.step(actions)
        
        if info.get('possession_changed'):
            possession_count += 1
    
    assert possession_count >= 10
    assert env.game_state.score_a + env.game_state.score_b > 0  # Someone scored

def test_role_switching_in_game():
    """Verify agents see correct role flags throughout game."""
    env = HexagonBasketballFullGameEnv()
    obs = env.reset()
    
    # Track role flags for training team player
    training_player_idx = 0
    role_history = []
    possession_history = []
    
    for _ in range(100):
        role_flag = obs['role_flag'][training_player_idx]
        role_history.append(role_flag)
        possession_history.append(env.possession.possessing_team)
        
        actions = env.action_space.sample()
        obs, _, done, _ = env.step(actions)
        
        if done:
            break
    
    # Verify role flag correlates with possession
    for role, poss in zip(role_history, possession_history):
        if poss == env.training_team:
            assert role == 0  # Offense
        else:
            assert role == 1  # Defense
```

### 6.3 Training Tests

```python
def test_training_with_role_switching():
    """Test that training works with dynamic role switching."""
    env = make_vec_env(n_envs=4)
    model = PPO('MultiInputPolicy', env, n_steps=128)
    
    # Train for a few iterations
    model.learn(total_timesteps=10000)
    
    # Verify model can handle both roles
    test_env = HexagonBasketballFullGameEnv()
    obs = test_env.reset()
    
    # Test with offense role
    test_env.possession.possessing_team = test_env.training_team
    obs['role_flag'] = np.array([0.0])
    action_offense, _ = model.predict(obs, deterministic=False)
    
    # Test with defense role
    test_env.possession.possessing_team = Team.B if test_env.training_team == Team.A else Team.A
    obs['role_flag'] = np.array([1.0])
    action_defense, _ = model.predict(obs, deterministic=False)
    
    # Actions should be different (stochastic policy)
    assert action_offense != action_defense  # (Might fail occasionally with random seed)
```

---

## 7. Potential Challenges

### 7.1 Learning Complexity

**Challenge**: Agents now need to learn both offensive and defensive strategies simultaneously.

**Mitigations**:
1. **Curriculum Learning**: 
   - Start with Phase 0 (no clock, fixed possessions)
   - Gradually increase episode length
   - Add game clock only after basic competency

2. **Separate Policy Heads**: Consider architecture with shared features but separate heads for offense/defense roles

3. **Role-Conditioned Training**: Ensure role flag is prominent in observations

4. **Reward Shaping**: Maintain clear reward signals for both roles

### 7.2 Episode Length & Sample Efficiency

**Challenge**: Full games are ~1000 steps vs current ~20-30 steps, reducing sample efficiency.

**Mitigations**:
1. Start with shorter games (e.g., 5 possessions per episode)
2. Increase PPO `n_steps` to collect more experience per update
3. Use episode truncation during early training
4. Consider hierarchical RL approaches for long-term planning

### 7.3 Credit Assignment

**Challenge**: Actions early in game have delayed impact on final outcome.

**Mitigations**:
1. Per-possession rewards (not just final game outcome)
2. Intermediate rewards for offensive/defensive stops
3. Value function improvements through GAE (Generalized Advantage Estimation)
4. Potential-based shaping that considers game state evolution

### 7.4 Opponent Pool Diversity

**Challenge**: Opponents must be competent at both offense and defense.

**Mitigations**:
1. Save policies more frequently to opponent pool
2. Evaluate policies on both offensive and defensive metrics
3. Sample opponents based on balanced skill levels
4. Include diversity metrics when selecting opponents

### 7.5 Observation Space Dimensionality

**Challenge**: Full court doubles the spatial information agents must process.

**Mitigations**:
1. Maintain relative positioning (focus on relevant court area)
2. Use attention mechanisms to handle variable player positions
3. Separate encoding for own half vs opponent half
4. Consider ego-centric observations centered on ball/player

### 7.6 Action Space & Strategy Complexity

**Challenge**: Fast breaks, transition defense, clock management add strategic depth.

**Mitigations**:
1. Introduce mechanics gradually (phases)
2. Use demonstrations or heuristic policies for bootstrapping
3. Reward intermediate strategic behaviors (fast breaks, clock running)
4. Monitor action distributions to detect degenerate strategies

### 7.7 Balance Between Teams

**Challenge**: Training team plays against frozen opponents, creating moving target problem.

**Mitigations**:
1. Frequent opponent pool updates
2. Diversity in opponent sampling
3. Monitor win rates and adjust opponent difficulty
4. Consider using ELO ratings for opponent selection

### 7.8 Computational Cost

**Challenge**: Longer episodes mean slower training iterations.

**Mitigations**:
1. Parallel environments (already implemented)
2. Shorter games during initial training phases
3. Efficient episode serialization for replay buffer
4. Profile and optimize environment step time

---

## 8. Implementation Roadmap

### Milestone 1: Core Full-Court Infrastructure (2-3 weeks)
- [ ] Implement full-court geometry
- [ ] Add possession tracking system
- [ ] Implement role flag system
- [ ] Update observation space
- [ ] Modify distance calculations for dual baskets
- [ ] Unit tests for basic mechanics

### Milestone 2: Phase 0 - Multi-Possession (2-3 weeks)
- [ ] Simple inbounding after made basket
- [ ] Possession change on miss (no rebounds yet)
- [ ] Fixed-possession episode termination
- [ ] Update reward structure
- [ ] Integration with training loop
- [ ] Test training convergence with short games

### Milestone 3: Evaluation & Refinement (1-2 weeks)
- [ ] Comprehensive testing
- [ ] Performance profiling
- [ ] Visualization updates for full court
- [ ] Metrics tracking for both roles
- [ ] Documentation updates

### Milestone 4: Phase 1 - Game Clock (2-3 weeks)
- [ ] Game clock implementation
- [ ] Period structure
- [ ] Clock management strategy learning
- [ ] End-of-game situations
- [ ] Training adjustments for longer episodes

### Milestone 5: Phase 2 - Rebounds (3-4 weeks)
- [ ] Rebound probability system
- [ ] Offensive rebound mechanics
- [ ] Defensive rebound mechanics
- [ ] Shot clock reset logic
- [ ] Reward adjustments for rebounding

### Milestone 6: Phase 3 - Advanced Mechanics (3-4 weeks)
- [ ] Fast breaks
- [ ] Backcourt violations
- [ ] Advanced inbounding
- [ ] Jump balls
- [ ] Additional rule refinements

**Total Estimated Time**: 4-6 months for full implementation

---

## 9. Alternative Approaches

### 9.1 Two-Model Approach
Instead of one model learning both roles, train separate offensive and defensive models:

**Pros**:
- Simpler learning problem for each model
- Can specialize each model's architecture
- Easier credit assignment

**Cons**:
- More complex training orchestration
- Doubles model storage requirements
- Loses potential for shared representations

### 9.2 Hierarchical RL
Use high-level controller to choose strategies, low-level controller for actions:

**Pros**:
- Better long-term planning
- Explicit strategy representation
- Easier curriculum learning

**Cons**:
- More complex implementation
- Requires defining strategy space
- Training instability possible

### 9.3 Simplified Game Mode
Instead of full quarters, use simpler game structure:

**Options**:
- First to N points wins
- Fixed number of possessions, highest score wins
- Sudden death after tied possessions

**Pros**:
- Faster episodes
- Simpler implementation
- Clearer learning signal

**Cons**:
- Loses clock management aspect
- Less realistic basketball

---

## 10. Success Metrics

Track these metrics to evaluate implementation success:

### Technical Metrics
- [ ] Environment runs without errors for 10,000 full games
- [ ] Average episode length matches expected value
- [ ] Role switching occurs correctly 100% of time
- [ ] Shot clock resets correctly in all scenarios

### Learning Metrics
- [ ] Training converges to non-trivial policies
- [ ] Win rate against random policy > 90%
- [ ] Points per game increases over training
- [ ] Both offensive and defensive competency improve

### Behavioral Metrics
- [ ] Agents learn to run fast breaks (Phase 3)
- [ ] Agents learn clock management (Phase 1)
- [ ] Agents learn to crash offensive boards (Phase 2)
- [ ] Agents show coordinated team behavior

### Performance Metrics
- [ ] Environment step time < 50ms
- [ ] Training iteration time comparable to current setup
- [ ] Memory usage remains manageable

---

## 11. Documentation Requirements

When implementing, ensure documentation for:

1. **Architecture Document**: Full court geometry, possession system, role flags
2. **Training Guide**: How to train with new environment, hyperparameter recommendations
3. **API Reference**: New observation space, new environment parameters
4. **Migration Guide**: Converting existing models/training runs
5. **Debugging Guide**: Common issues and solutions
6. **Visualization Updates**: How to render full court games

---

## 12. Conclusion

Converting to a full-game basketball environment is a significant but achievable undertaking. The phased approach allows for:

1. **Incremental validation** - Each phase can be tested independently
2. **Flexible implementation** - Can stop at Phase 0 or 1 if sufficient
3. **Backward compatibility** - Existing single-possession mode can coexist
4. **Research opportunities** - Each phase opens new research questions

The key insight is that the **role flag system** allows a single training team to learn both offense and defense without fundamentally changing the training loop architecture. This maintains compatibility with existing self-play infrastructure while enabling richer, more realistic basketball gameplay.

**Recommended Starting Point**: Implement Phase 0 (Foundation) as a proof-of-concept. This will reveal any unforeseen challenges and validate the core architecture before committing to the full implementation.

