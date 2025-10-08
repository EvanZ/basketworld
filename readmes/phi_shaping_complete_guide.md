# Potential-Based Reward Shaping (PBRS) - Complete Guide

## Table of Contents
1. [Theory and Foundations](#theory-and-foundations)
2. [Quick Start](#quick-start)
3. [Implementation Overview](#implementation-overview)
4. [Aggregation Modes](#aggregation-modes)
5. [Web App Usage](#web-app-usage)
6. [Training Configuration](#training-configuration)
7. [Technical Implementation](#technical-implementation)
8. [Examples and Use Cases](#examples-and-use-cases)
9. [Troubleshooting](#troubleshooting)

---

## Theory and Foundations

### What is Potential-Based Reward Shaping?

**Potential-Based Reward Shaping (PBRS)** is a technique to accelerate reinforcement learning by adding auxiliary rewards that guide agents without changing the optimal policy. Based on Ng et al. (1999), PBRS provides "hints" that help agents learn faster while maintaining policy invariance.

### The Core Concept

PBRS defines a **potential function** Φ(s) that represents the "quality" or "promise" of a state. The shaping reward is:

```
r_shape = β * (γ * Φ(s') - Φ(s))
```

Where:
- **Φ(s)**: Potential of current state
- **Φ(s')**: Potential of next state
- **β**: Shaping strength multiplier
- **γ**: Discount factor (must match agent's γ)

### Why It Works

1. **Policy Invariance**: If γ_shaping = γ_agent, the optimal policy remains unchanged
2. **Credit Assignment**: Moves credit to actions that improve state quality (positive shaping)
3. **Telescoping Property**: Over an episode, shaping rewards sum to zero when Φ(s₀)=0 and Φ(terminal)=0
4. **Variance Reduction**: Dense shaping signals reduce variance in returns

### Our Implementation: Expected Points as Φ

In BasketWorld, we define:

```
Φ(s) = Expected Points (EP) based on shot quality and positions
EP(player) = shot_value × shot_probability
  where shot_value = 3 (beyond arc) or 2 (inside arc or dunk)
```

This makes Φ:
- **State-dependent only** (policy-invariant)
- **Interpretable** (directly relates to scoring potential)
- **Dense** (every action affects EP via positioning)

### Telescoping to Zero

Proper PBRS requires:
1. **Φ(s₀) = 0**: Initial state has zero potential
2. **Φ(s_terminal) = 0**: Terminal states have zero potential

This ensures:
```
Sum_episode r_shape = β[γΦ(s₁) - 0] + β[γΦ(s₂) - Φ(s₁)] + ... + β[0 - Φ(s_T)]
                     = βγ·0 - β·0 = 0
```

Benefits:
- ✅ Guaranteed policy invariance
- ✅ Zero-sum shaping reduces variance
- ✅ No spurious baseline shifts

---

## Quick Start

### For Web App Users

1. **Load a model** trained with phi shaping
2. **Go to Rewards Tab**: See actual rewards with MLflow phi shaping
3. **Go to Phi Shaping Tab**: See detailed diagnostics and experiment with parameters

### For Training

```bash
python train/train.py \
  --enable-phi-shaping true \
  --phi-aggregation-mode teammates_best \
  --phi-blend-weight 0.3 \
  --phi-beta-start 0.15 \
  --phi-beta-end 0.0 \
  --reward-shaping-gamma 0.99 \  # Must match --gamma
  --gamma 0.99
```

### Recommended Starting Parameters

| Parameter | Recommended | Why |
|-----------|-------------|-----|
| `phi_aggregation_mode` | `teammates_best` | Clean individual accountability |
| `phi_blend_weight` | `0.3` | Balance team and individual |
| `phi_beta_start` | `0.10-0.20` | Strong enough signal |
| `phi_beta_end` | `0.0` | Anneal to zero |
| `reward_shaping_gamma` | Same as `gamma` | Policy invariance |

---

## Implementation Overview

### Architecture

```
Training Environment (basketworld_env_v2.py)
  ↓ phi_beta, enable_phi_shaping
  ├─ Calculates Φ(s) each step
  ├─ Applies r_shape to rewards
  └─ Stores phi diagnostics in info

MLflow (utils/mlflow_params.py)
  ↓ logs phi parameters
  └─ phi_beta_end, phi_aggregation_mode, etc.

Web App Backend (app/backend/main.py)
  ↓ loads MLflow params
  ├─ Calculates rewards with MLflow phi shaping
  ├─ Replaces environment phi with MLflow phi
  └─ Serves to frontend

Web App Frontend
  ├─ Rewards Tab: Shows actual training rewards
  └─ Phi Shaping Tab: Shows diagnostics & experiments
```

### Key Files

| File | Purpose |
|------|---------|
| `basketworld/envs/basketworld_env_v2.py` | Core phi calculation and shaping |
| `basketworld/utils/callbacks.py` | Beta scheduling during training |
| `basketworld/utils/mlflow_params.py` | MLflow parameter loading |
| `app/backend/main.py` | Web app reward calculation |
| `app/frontend/src/components/PlayerControls.vue` | Rewards tab UI |
| `app/frontend/src/components/PhiShaping.vue` | Phi diagnostics UI |

---

## Aggregation Modes

### The Problem

How should we aggregate expected points from multiple teammates into a single team potential Φ(s)?

### Available Modes

#### 1. **team_best** (legacy)
```
Φ = (1-w) × max(all_teammates_including_ball_handler) + w × ball_handler_EP
```

**Use case**: Backward compatibility

**Problem**: Ball handler appears in both terms when they have the best shot, dampening individual penalties

#### 2. **teammates_best** ⭐ RECOMMENDED
```
Φ = (1-w) × max(teammates_excluding_ball_handler) + w × ball_handler_EP
```

**Use case**: Clean separation of individual vs team accountability

**Advantages**:
- Ball handler's decisions only affect the `w` term
- Teammate spacing/positioning only affects the `(1-w)` term
- No double-counting
- Strong penalties for degrading your own shot quality

**Example**:
```
Step 8: Ball EP=1.194, Teammate best=1.151
  Φ = 0.7×1.151 + 0.3×1.194 = 1.164

Step 9: Ball EP=0.870, Teammate best=1.151  
  Φ = 0.7×1.151 + 0.3×0.870 = 1.067

Penalty = β × (1.067 - 1.164) = 0.15 × (-0.097) = -0.0145 per team
```

#### 3. **teammates_avg**
```
Φ = (1-w) × mean(teammates_excluding_ball_handler) + w × ball_handler_EP
```

**Use case**: Smoother signal, rewards overall team spacing

**Advantages**:
- Less sensitive to single teammate movement
- Encourages balanced team positioning
- Still separates ball handler accountability

**Trade-offs**:
- Weaker signal for "find the best shot"
- Might not encourage passing to the open player as strongly

#### 4. **team_avg**
```
Φ = mean(all_teammates_including_ball_handler)
```

**Use case**: Simplest option, no blend weight needed

**Advantages**:
- Conceptually simple
- Automatically balances individual (1/N) vs team ((N-1)/N)
- No hyperparameter w to tune

**Trade-offs**:
- With 3 players, ball handler is only 33% of Φ
- Might need higher β to get sufficient signal

#### 5. **teammates_worst**
```
Φ = (1-w) × min(teammates_excluding_ball_handler) + w × ball_handler_EP
```

**Use case**: Raise the floor - encourage behaviors that help the worst-positioned teammate

**Advantages**:
- Encourages team coordination to improve weakest player's position
- Rewards actions that prevent any teammate from being left in a bad spot

**Trade-offs**:
- May lead to overly cautious play
- Might not encourage taking advantage of best opportunities

#### 6. **team_worst**
```
Φ = (1-w) × min(all_teammates_including_ball_handler) + w × ball_handler_EP
```

**Use case**: Similar to teammates_worst but includes ball handler

**Trade-offs**:
- Ball handler appears in both terms when they have the worst EP
- May dampen individual accountability

### Mathematical Comparison

Given a 3v3 scenario:
- Ball handler: EP = 0.870
- Teammate 1: EP = 1.151  
- Teammate 2: EP = 0.850

| Mode | Φ(s) with w=0.3 | Φ(s) with w=0.5 |
|------|-----------------|-----------------|
| `team_best` | 0.7×1.151 + 0.3×0.870 = 1.067 | 0.5×1.151 + 0.5×0.870 = 1.011 |
| `teammates_best` | 0.7×1.151 + 0.3×0.870 = 1.067 | 0.5×1.151 + 0.5×0.870 = 1.011 |
| `teammates_avg` | 0.7×1.001 + 0.3×0.870 = 0.962 | 0.5×1.001 + 0.5×0.870 = 0.936 |
| `team_avg` | (0.870+1.151+0.850)/3 = 0.957 | N/A (no blend) |
| `teammates_worst` | 0.7×0.850 + 0.3×0.870 = 0.856 | 0.5×0.850 + 0.5×0.870 = 0.860 |

### Recommended Configurations

#### For Learning Shot Selection
```bash
--phi-aggregation-mode teammates_best \
--phi-blend-weight 0.5 \
--phi-beta-start 0.15
```

#### For Emphasizing Team Play
```bash
--phi-aggregation-mode teammates_avg \
--phi-blend-weight 0.3 \
--phi-beta-start 0.20
```

#### For Simplicity
```bash
--phi-aggregation-mode team_avg \
--phi-beta-start 0.20
```

---

## Web App Usage

### Overview

The web app displays phi-based reward shaping in two tabs:
- **Rewards Tab**: Shows actual game rewards using MLflow training parameters
- **Phi Shaping Tab**: Shows detailed phi diagnostics and allows experimentation

Both tabs automatically use the phi shaping parameters from your MLflow training run.

### Rewards Tab

**Purpose**: Show what rewards the model actually experienced during training

#### Display

| Column | Shows | Format |
|--------|-------|--------|
| Turn | Step number (0 = initial state) | Integer |
| Offense | Total offensive rewards with MLflow phi | X.XXX (3 decimals) |
| Off. Reason | Why reward was given | Text |
| Defense | Total defensive rewards with MLflow phi | X.XXX (3 decimals) |
| Def. Reason | Why reward was given | Text |
| Φ | Phi potential (state quality) | X.XXX (3 decimals) |

**What you see**:
```
Turn  Offense  Defense  Φ
0     0.000    0.000    0.602  <- Initial state
1     -0.014   0.014    0.545  <- After first action
2     -0.013   0.013    0.494  <- After second action
```

**Key Features**:
- Turn 0 shows initial state with starting Φ
- Rewards include MLflow phi shaping (not environment's)
- Φ column shows state quality for reference
- Fixed to training parameters (can't be changed)
- Auto-refreshes when switching to tab

### Phi Shaping Tab

**Purpose**: Detailed phi diagnostics and experimentation

#### Display

| Column | Shows | Format |
|--------|-------|--------|
| Step | Step number (0 = initial state) | Integer |
| Clock | Shot clock value | Integer |
| β | Beta parameter at this step | X.XXX (3 decimals) |
| Φprev | Phi before action | X.XXX (3 decimals) |
| Φnext | Phi after action | X.XXX (3 decimals) |
| TeamBestEP | Best/avg expected points among teammates | X.XXX (3 decimals) |
| BallEP | Ball handler's expected points | X.XXX (3 decimals) |
| BestP | Player ID with best shot | Integer |
| r_shape | Shaping reward (team total) | X.XXX (3 decimals) |

**What you see**:
```
Step  β     Φprev  Φnext  r_shape
0     0.250 0.000  0.602  0.000   <- Initial state
1     0.250 0.602  0.545  -0.014  <- Shaping reward for turn 1
2     0.250 0.545  0.494  -0.013  <- Shaping reward for turn 2
```

**Key Features**:
- Initializes with MLflow parameters on load
- Can be adjusted for experimentation
- Shows "Total" row summing all r_shape values
- Step 0 shows initial state (r_shape always 0)
- **Auto-updates** after each game step
- **Live recalculation** when parameters change

### Interactive Features

#### Auto-Refresh on Every Step
- Table automatically updates after each game step
- No manual refresh needed
- Uses Vue watchers to detect gameState changes

#### Live Parameter Recalculation
When you change parameters in the Phi Shaping tab:
- All rows recalculate instantly
- Move sliders or change dropdowns → see updated values immediately
- Explore "what would phi shaping have been?" scenarios

**Use Cases**:

1. **Finding the Right Beta**
   ```
   1. Play through an episode
   2. Adjust beta slider: 0.05 → 0.10 → 0.15 → 0.20
   3. Watch r_shape values change in real-time
   4. Pick beta that gives meaningful signal
   5. Click Apply to set it
   ```

2. **Comparing Aggregation Modes**
   ```
   1. Play through an episode
   2. Switch aggregation mode dropdown:
      - team_best → see current values
      - teammates_best → see how excluding ball handler changes Φ
      - teammates_avg → see smoother aggregation
      - team_avg → see simple average
   3. Compare Total row to see cumulative impact
   4. Pick mode that best penalizes bad decisions
   5. Click Apply to set it
   ```

### Verifying Consistency

**Important**: The Rewards tab Offense/Defense should equal the Phi Shaping tab r_shape (when only phi shaping is active):
- Rewards tab Turn 1 Offense: **-0.014** 
- Phi Shaping tab Step 1 r_shape: **-0.014**
- ✓ They match!

If your environment gives other rewards (passes, shots), they'll be added to the phi shaping.

### Independence of Systems

#### Rewards Tab
- **Fixed to MLflow parameters**
- Shows what model experienced during training
- Cannot be modified
- **Purpose**: Verify training rewards

#### Phi Shaping Tab  
- **Initialized with MLflow, but adjustable**
- Can experiment with different parameters
- Does NOT affect Rewards tab or gameplay
- **Purpose**: Explore different shaping configurations

#### Environment
- Can have its own phi shaping enabled/disabled
- Rewards tab removes environment's phi shaping and replaces with MLflow's
- **Purpose**: Test model with different runtime configurations

---

## Training Configuration

### CLI Arguments

```bash
python train/train.py \
  --enable-phi-shaping true \
  --phi-aggregation-mode teammates_best \
  --phi-blend-weight 0.3 \
  --phi-beta-start 0.15 \
  --phi-beta-end 0.0 \
  --phi-beta-schedule exp \
  --reward-shaping-gamma 0.99 \
  --gamma 0.99 \
  --phi-bump-updates 1 \
  --phi-bump-multiplier 1.25
```

### Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--enable-phi-shaping` | bool | `false` | Enable PBRS |
| `--phi-aggregation-mode` | str | `team_best` | How to aggregate teammate EPs |
| `--phi-blend-weight` | float | `0.0` | Weight w for blending (0=team only, 1=ball only) |
| `--phi-beta-start` | float | `0.0` | Initial beta value |
| `--phi-beta-end` | float | `0.0` | Final beta value |
| `--phi-beta-schedule` | str | `exp` | Beta decay schedule |
| `--reward-shaping-gamma` | float | `1.0` | Discount for shaping (should match `--gamma`) |
| `--phi-use-ball-handler-only` | bool | `false` | Force ball-handler-only mode |
| `--phi-bump-updates` | int | `0` | Bump beta for N updates at alternation start |
| `--phi-bump-multiplier` | float | `1.0` | Multiply beta by this during bump |

### Scheduling

The `PotentialBetaExpScheduleCallback` provides exponential decay:

```python
β(t) = β_end + (β_start - β_end) * exp(-decay_constant * t)
```

With optional "bumps" at alternation starts to reinforce learning.

### Example Configurations

#### Standard Setup (Recommended)
```bash
--enable-phi-shaping true \
--phi-aggregation-mode teammates_best \
--phi-blend-weight 0.3 \
--phi-beta-start 0.15 \
--phi-beta-end 0.0 \
--reward-shaping-gamma 1.0 \
--gamma 1.0
```

#### Strong Individual Accountability
```bash
--enable-phi-shaping true \
--phi-aggregation-mode teammates_best \
--phi-blend-weight 0.5 \
--phi-beta-start 0.20 \
--phi-beta-end 0.0
```

#### Emphasis on Team Coordination
```bash
--enable-phi-shaping true \
--phi-aggregation-mode teammates_avg \
--phi-blend-weight 0.2 \
--phi-beta-start 0.20 \
--phi-beta-end 0.0
```

#### Simple Average (No Tuning)
```bash
--enable-phi-shaping true \
--phi-aggregation-mode team_avg \
--phi-beta-start 0.20 \
--phi-beta-end 0.0
```

---

## Technical Implementation

### Environment (`basketworld_env_v2.py`)

#### Initialization
```python
self.enable_phi_shaping: bool = bool(enable_phi_shaping)
self.reward_shaping_gamma: float = float(reward_shaping_gamma)
self.phi_beta: float = float(phi_beta)
self.phi_use_ball_handler_only: bool = bool(phi_use_ball_handler_only)
self.phi_blend_weight: float = float(phi_blend_weight)
self.phi_aggregation_mode: str = str(phi_aggregation_mode)
self._first_step_after_reset: bool = False
```

#### Step Function
```python
# Calculate Φ(s) before action
phi_prev = 0.0 if self._first_step_after_reset else self._phi_shot_quality()

# Take action, observe next state

# Calculate Φ(s') after action
phi_next = 0.0 if done else self._phi_shot_quality()

# Apply shaping
if self.enable_phi_shaping:
    r_shape = self.phi_beta * (
        self.reward_shaping_gamma * phi_next - phi_prev
    )
    rewards[self.offense_ids] += r_shape / self.players_per_side
    rewards[self.defense_ids] -= r_shape / self.players_per_side
    
    # Store diagnostics
    info["phi_prev"] = phi_prev
    info["phi_next"] = phi_next
    info["phi_beta"] = self.phi_beta
    info["phi_r_shape"] = r_shape / self.players_per_side  # Per-player
```

#### Phi Calculation
```python
def _phi_shot_quality(self) -> float:
    """Calculate potential Φ(s) based on expected points."""
    
    # Get EPs for all offensive players
    eps = []
    for pid in self.offense_ids:
        pos = self.positions[pid]
        dist = self._hex_distance(pos, self.basket_position)
        shot_value = 3.0 if dist >= self.three_point_distance else 2.0
        if self.allow_dunks and dist == 0:
            shot_value = 2.0
        p = self._calculate_shot_probability(pid, dist)
        eps.append(shot_value * p)
    
    ball_ep = eps[self.ball_holder] if 0 <= self.ball_holder < len(eps) else 0.0
    
    # Ball-handler-only mode
    if self.phi_use_ball_handler_only:
        return ball_ep
    
    # Aggregation modes
    if self.phi_aggregation_mode == "team_avg":
        return sum(eps) / len(eps)
    
    # Get teammate EPs (excluding ball handler)
    teammate_eps = [ep for i, ep in enumerate(eps) if i != self.ball_holder]
    
    if not teammate_eps:
        return ball_ep
    
    if self.phi_aggregation_mode == "teammates_best":
        team_aggregate = max(teammate_eps)
    elif self.phi_aggregation_mode == "teammates_avg":
        team_aggregate = sum(teammate_eps) / len(teammate_eps)
    elif self.phi_aggregation_mode == "teammates_worst":
        team_aggregate = min(teammate_eps)
    elif self.phi_aggregation_mode == "team_worst":
        team_aggregate = min(eps)
    else:  # team_best (default)
        team_aggregate = max(eps)
    
    # Blend
    w = float(self.phi_blend_weight)
    w = max(0.0, min(1.0, w))
    return (1.0 - w) * team_aggregate + w * ball_ep
```

### Backend (`app/backend/main.py`)

#### Loading MLflow Parameters
```python
from basketworld.utils.mlflow_params import get_mlflow_phi_shaping_params

# In init_game()
mlflow_phi_params = get_mlflow_phi_shaping_params(client, request.run_id)
game_state.mlflow_phi_shaping_params = mlflow_phi_params
```

#### Reward Calculation
```python
# In get_rewards()
if mlflow_phi_params and mlflow_phi_params.get("enable_phi_shaping"):
    beta = mlflow_phi_params["phi_beta"]
    gamma = mlflow_phi_params["reward_shaping_gamma"]
    
    # Calculate initial Φ
    phi_prev = 0.0  # Initial state
    if game_state.phi_log and len(game_state.phi_log) > 0:
        initial_entry = game_state.phi_log[0]
        if initial_entry.get("step") == 0:
            initial_ep = initial_entry.get("ep_by_player", [])
            initial_ball = initial_entry.get("ball_handler", -1)
            initial_offense = initial_entry.get("offense_ids", [])
            if initial_ep and initial_ball >= 0 and initial_offense:
                phi_prev = calculate_phi_from_ep_data(
                    initial_ep, initial_ball, initial_offense, mlflow_phi_params
                )
    
    # Calculate r_shape for each step
    for i, reward in enumerate(game_state.reward_history):
        ep_by_player = reward.get("ep_by_player", [])
        ball_handler = reward.get("ball_handler", -1)
        offense_ids = reward.get("offense_ids", [])
        is_terminal = reward.get("is_terminal", False)
        
        phi_next = 0.0  # Terminal states
        if not is_terminal and ep_by_player:
            phi_next = calculate_phi_from_ep_data(
                ep_by_player, ball_handler, offense_ids, mlflow_phi_params
            )
        
        r_shape = beta * (gamma * phi_next - phi_prev)
        mlflow_phi_r_shape_values.append(r_shape)
        mlflow_phi_potential_values.append(phi_next)
        
        phi_prev = phi_next
    
    # Remove environment phi, add MLflow phi
    for i, reward in enumerate(game_state.reward_history):
        env_phi_r_shape_per_player = reward.get("phi_r_shape", 0.0)
        num_offensive_players = len(reward.get("offense_ids", []))
        env_phi_r_shape_total = env_phi_r_shape_per_player * num_offensive_players
        
        base_offense = float(reward["offense"]) - env_phi_r_shape_total
        base_defense = float(reward["defense"]) + env_phi_r_shape_total
        
        mlflow_phi_r_shape = mlflow_phi_r_shape_values[i]
        offense_with_mlflow = base_offense + mlflow_phi_r_shape
        defense_with_mlflow = base_defense - mlflow_phi_r_shape
```

### Frontend (`PhiShaping.vue`)

#### Initialization with MLflow Params
```javascript
async function loadParams() {
  try {
    // Try to load MLflow params first
    const rewardsData = await getRewards();
    if (rewardsData.mlflow_phi_params && rewardsData.mlflow_phi_params.enable_phi_shaping) {
      params.value = { ...params.value, ...rewardsData.mlflow_phi_params };
      console.log('[PhiShaping] Initialized with MLflow params:', rewardsData.mlflow_phi_params);
    } else {
      // Fall back to environment params
      const p = await getPhiParams();
      params.value = { ...params.value, ...p };
    }
  } catch (e) {
    console.warn('[PhiShaping] Failed to load params:', e);
  }
}
```

#### Live Recalculation
```javascript
const displayRows = computed(() => {
  if (!rawLogData.value || rawLogData.value.length === 0) {
    return [];
  }

  const beta = params.value.phi_beta;
  const gamma = params.value.reward_shaping_gamma;
  const mode = params.value.phi_aggregation_mode;
  const blendWeight = params.value.phi_blend_weight;
  const useBallHandlerOnly = params.value.phi_use_ball_handler_only;

  return rawLogData.value.map((row, idx) => {
    let phi_prev = 0.0;
    if (idx > 0) {
      const prevRow = rawLogData.value[idx - 1];
      phi_prev = prevRow.is_terminal ? 0.0 : calculatePhi(prevRow, mode, blendWeight, useBallHandlerOnly);
    }
    
    const phi_next = row.is_terminal ? 0.0 : calculatePhi(row, mode, blendWeight, useBallHandlerOnly);
    const r_shape = row.step === 0 ? 0 : beta * (gamma * phi_next - phi_prev);
    
    return {
      step: row.step,
      shot_clock: row.shot_clock,
      phi_beta: beta,
      phi_prev: phi_prev,
      phi_next: phi_next,
      team_best_ep: row.team_best_ep,
      ball_handler_ep: row.ball_handler_ep,
      best_ep_player: calculateBestEPPlayer(row),
      phi_r_shape: r_shape
    };
  });
});
```

---

## Examples and Use Cases

### Example 1: Understanding a Single Step

**Scenario**: 3v3 game, MLflow params: β=0.25, γ=1.0, mode=team_avg

**Phi Shaping Tab shows**:
```
Step 1: Φprev=0.602, Φnext=0.545, r_shape=-0.014
```

**Calculation**:
```
r_shape = 0.25 * (1.0 * 0.545 - 0.602)
        = 0.25 * (-0.057)
        = -0.014  (team total)
```

**Meaning**: 
- State got worse (Φ decreased from 0.602 to 0.545)
- Team penalized by -0.014 total
- Each of 3 players gets: -0.014 / 3 = -0.005

**Rewards Tab shows**:
```
Turn 1: Offense=-0.014, Defense=0.014, Φ=0.545
```

### Example 2: Full Episode Analysis

**Phi Shaping Tab**:
```
Step  Φprev  Φnext  r_shape
0     0.000  0.602  0.000   <- Initial state
1     0.602  0.545  -0.014  <- Bad action
2     0.545  0.494  -0.013  <- Bad action
3     0.494  0.539  0.011   <- Good action!
4     0.539  0.606  0.017   <- Good action!
Total:                0.001  <- Net positive
```

**Interpretation**:
- Started at Φ=0.602
- First two actions degraded position (negative r_shape)
- Actions 3-4 improved position (positive r_shape)
- Net phi shaping: slightly positive

### Example 3: How PPO Uses Shaping

At each step:
1. **Policy gets reward** = base task reward + r_shape/team
2. **PPO computes returns/advantages** (e.g., GAE) from these per-step rewards
3. **Actions that increased Φ** get higher advantages → positive reinforcement
4. **Actions that decreased Φ** get lower/negative advantages → negative reinforcement

**Concrete actions**:
- Drive that raises team-best EP → positive shaping → higher advantage → policy repeats that drive
- Off-ball cut/screen that raises team-best EP → same reinforcement → teammates learn to move
- Pass when teammate's EP > ball-handler's EP:
  - With ball-handler-only or hybrid (w>0): Φ jumps to recipient's EP → positive shaping at pass step
  - With pure team-best only: passing is directly credited only if team-best EP increases

**Terminal step**:
- Has negative shaping "payback" (Φ_next = 0)
- But outcome reward (2/3 pts) determines shot selection
- Shaping mainly redistributes credit to earlier creators

### Example 4: Comparing Aggregation Modes

**Same game state, different modes**:

| Mode | Team EPs | Ball EP | Φ | How it affects behavior |
|------|----------|---------|---|------------------------|
| team_avg | [0.4, 0.6, 0.8] | 0.6 | 0.600 | Balanced, 1/3 weight each |
| team_best | [0.4, 0.6, 0.8] | 0.6 | 0.800 | Rewards creating best shot |
| teammates_best | [0.4, 0.8] | 0.6 | 0.800 | Clean ball handler separation |
| teammates_avg | [0.4, 0.8] | 0.6 | 0.600 | Smoother team signal |

Different modes lead to different policies!

### Example 5: Zero-Sum Property

With proper Φ(s₀)=0 and Φ(terminal)=0:

```
Episode: 5 steps, β=0.15, γ=1.0

Step 0: Φ=0.000 → Φ=0.602, r_shape = 0.15*(0.602-0) = +0.090
Step 1: Φ=0.602 → Φ=0.545, r_shape = 0.15*(0.545-0.602) = -0.009
Step 2: Φ=0.545 → Φ=0.494, r_shape = 0.15*(0.494-0.545) = -0.008
Step 3: Φ=0.494 → Φ=0.539, r_shape = 0.15*(0.539-0.494) = +0.007
Step 4: Φ=0.539 → 0.000, r_shape = 0.15*(0-0.539) = -0.081

Total: +0.090 - 0.009 - 0.008 + 0.007 - 0.081 ≈ 0.000 ✓
```

The shaping telescopes to zero, redistributing credit without changing total return!

---

## Troubleshooting

### Training Issues

#### Q: Model isn't learning to pass

**Check**:
1. Is `phi_blend_weight` > 0? (Gives direct pass credit)
2. Try `teammates_best` mode for cleaner signals
3. Increase beta (0.15-0.25 range)
4. Consider pass logit bias (`--pass-logit-bias-start 1.0 --pass-logit-bias-end 0.0`)

**Recommended**:
```bash
--phi-aggregation-mode teammates_best \
--phi-blend-weight 0.5 \
--phi-beta-start 0.20
```

#### Q: Shaping rewards seem too weak/strong

**Adjust beta**:
- Too weak: Increase `--phi-beta-start` (try 0.20-0.30)
- Too strong: Decrease `--phi-beta-start` (try 0.05-0.10)
- Check r_shape values in web app Phi Shaping tab

**Rule of thumb**: r_shape should be ~10-30% of typical base rewards

#### Q: Should I anneal beta?

**Usually yes**:
```bash
--phi-beta-start 0.15 \
--phi-beta-end 0.0
```

**Why**: Early training needs dense signals; later training should rely on learned values

**Exception**: Keep constant if learning is very slow

#### Q: Which aggregation mode should I use?

**For shot selection**: `teammates_best` with w=0.5
**For team coordination**: `teammates_avg` with w=0.3
**For simplicity**: `team_avg` (no blend weight)

**Start with `teammates_best` and adjust based on observed behavior**

### Web App Issues

#### Q: Rewards tab shows all zeros for phi shaping

**Check**:
1. Was the model trained with phi shaping enabled?
   - Look for `--enable-phi-shaping true` in launch config
2. Does MLflow run have phi parameters?
   - Check if `mlflow_phi_params.enable_phi_shaping` is true
3. Is beta = 0?
   - If beta is 0, r_shape will always be 0

#### Q: Rewards tab and Phi Shaping tab don't match

**Common causes**:
1. **Different episodes**: Make sure you're viewing the same episode
2. **Different parameters**: Check that Phi Shaping tab parameters match MLflow
3. **Step numbering**: Phi Shaping step N = Rewards turn N
4. **Other rewards**: Base game rewards (shots, passes) are added to phi shaping

#### Q: Phi Shaping tab values are all zero

**Most common**: Beta = 0
- Set beta slider to non-zero value (e.g., 0.15)
- Click "Apply"

**Other causes**:
- Enable phi shaping is off (Apply button auto-enables it)
- No game steps taken yet
- Backend not storing phi data

#### Q: Initial state (step 0) is missing

**This was fixed**: Both tabs now show step 0 with initial Φ value. If you don't see it:
1. Restart backend
2. Refresh browser
3. Load a new game

#### Q: Phi potentials seem wrong

**Verify**:
1. Check aggregation mode (team_avg, team_best, etc.)
2. Verify blend weight setting
3. Look at TeamBestEP and BallEP diagnostics in table
4. Ensure shot probabilities are reasonable (not all 0 or 1)

**Debug**:
- Look at "BestP" column - which player has best shot?
- Compare TeamBestEP vs BallEP - does aggregation make sense?
- Try changing mode and see if values change as expected

#### Q: Auto-update not working in Phi Shaping tab

**Check**:
1. Is gameState prop being passed to component?
2. Look for JavaScript errors in browser console
3. Verify backend is storing phi_log data
4. Try manual refresh to see if data is available

### General Issues

#### Q: Does phi shaping change the optimal policy?

**No, if** `reward_shaping_gamma` = agent's `gamma`. This is **policy invariance**.

**Yes, if** they differ. Always match them!

#### Q: Why does total r_shape ≈ 0?

**This is correct!** With proper PBRS:
- Φ(s₀) = 0
- Φ(terminal) = 0
- Shaping telescopes to zero over episode

**This is a feature**, not a bug. It means shaping redistributes credit without changing total return.

#### Q: Should I change mode mid-training?

**No** - this changes the reward structure. The policy learned Q-values under the original mode.

Pick a mode at the start and keep it constant.

#### Q: How do I know which params were used during training?

**Check launch configuration**:
```bash
.vscode/launch.json
```

Look for:
```json
"--enable-phi-shaping", "true",
"--phi-beta-start", "0.25",
"--phi-beta-end", "0.25",
"--reward-shaping-gamma", "1.0",
"--phi-aggregation-mode", "team_avg"
```

**Or check MLflow**: The web app automatically loads these!

#### Q: Can I use phi shaping with other reward engineering?

**Yes**, phi shaping is additive:
```
total_reward = base_reward + r_shape
```

Where base_reward can include:
- Shot outcomes (2/3 pts)
- Pass rewards
- Turnover penalties
- Other custom rewards

Just ensure `reward_shaping_gamma` matches agent's discount.

---

## Key Takeaways

1. **Policy Invariance**: PBRS doesn't change optimal policy if γ_shaping = γ_agent
2. **Dense Credit**: Provides reward signal every step, not just at outcomes
3. **Telescoping**: Shaping sums to zero over episode with proper Φ(s₀) and Φ(terminal)
4. **Aggregation Matters**: Different modes lead to different behaviors
5. **MLflow Integration**: Web app automatically uses training parameters
6. **Experimentation**: Phi Shaping tab allows safe parameter exploration
7. **Annealing**: Usually beneficial to anneal beta → 0 over training
8. **Team Coordination**: Blend weight balances individual and team accountability

---

## Related Files

### Core Implementation
- `basketworld/envs/basketworld_env_v2.py`: Environment phi calculation
- `basketworld/utils/callbacks.py`: Beta scheduling
- `basketworld/utils/policies.py`: PassBiasMultiInputPolicy (complementary exploration aid)

### MLflow Integration
- `basketworld/utils/mlflow_params.py`: Parameter loading

### Web App
- `app/backend/main.py`: Backend reward calculation
- `app/frontend/src/components/PlayerControls.vue`: Rewards tab
- `app/frontend/src/components/PhiShaping.vue`: Phi Shaping tab diagnostics

### Configuration
- `train/train.py`: Training setup and CLI arguments
- `.vscode/launch.json`: Example training configurations

---

## References

- Ng, A. Y., Harada, D., & Russell, S. (1999). "Policy invariance under reward transformations: Theory and application to reward shaping." ICML.
- Potential-based reward shaping theory ensures optimal policy preservation
- Expected points (EP) as potential function grounds shaping in domain knowledge

---

**Last Updated**: Following complete consolidation of phi shaping documentation

**Version**: 2.0 (Consolidated from 5 separate documents)

