> Can you explain the logic behind how you defined the observation space?
---

## Overview

The observation space is one of the most critical design decisions in any reinforcement learning environment. It's the "eyes" of the AI agent, and what you choose to include (and how you format it) directly impacts how well the agent can learn.

Our environment uses a **structured dictionary observation space** with multiple components, each serving a specific purpose for learning basketball strategy.

---

## Observation Space Structure

The observation space is a `gymnasium.spaces.Dict` with four keys:

```python
observation_space = spaces.Dict({
    "obs": state_space,           # Main game state vector
    "action_mask": action_mask_space,  # Legal actions per player
    "role_flag": role_flag_space,      # Offense (+1) or Defense (-1)
    "skills": skills_space,            # Per-player shooting abilities
})
```

---

## 1. Main State Vector (`"obs"`)

This is the primary observation vector containing all dynamic game state information. It's built using **absolute court coordinates**, allowing the network to learn location-dependent strategies (e.g., center court vs. sidelines).

**Note:** This is ONLY the `"obs"` key in the dictionary. Skills, action masks, and role flags are separate dictionary keys (see sections below).

### Components (in order):

#### **A. Player Positions** (`n_players × 2` floats)
- **Format:** `(q, r)` pairs in absolute axial hex coordinates
- **Reference Frame:** Absolute court coordinates (NOT relative to ball handler)
- **Normalization:** Divided by `max(court_width, court_height)` to keep values roughly in `[-1, 1]`
- **Purpose:** Allows the network to learn strategies that are sensitive to court position (e.g., more space in center, constraints on sidelines)

**Why absolute?** This allows the network to distinguish between center court (more space) and sidelines (more constraints), which is crucial for realistic basketball play.

#### **B. Team Encoding** (`n_players` floats)
- **Format:** `[team_0, team_1, ..., team_N]` where each value is ±1
- **Values:** `+1.0` = offense, `-1.0` = defense
- **Example (3v3):** `[1, 1, 1, -1, -1, -1]` means players 0-2 on offense, 3-5 on defense
- **Purpose:** Explicit per-player team identification
- **Why:** Allows network to directly condition strategies on team membership rather than inferring it

#### **C. Ball Holder One-Hot** (`n_players` floats)
- **Format:** One-hot vector indicating which player has the ball
- **Example (3v3):** `[0, 0, 1, 0, 0, 0]` means player 2 has the ball
- **Purpose:** Explicit encoding of ball possession state

#### **D. Shot Clock** (`1` float)
- **Format:** Raw integer value (unnormalized)
- **Range:** `[0, shot_clock_steps]` (typically 24)
- **Purpose:** Encodes time pressure for decision-making

#### **E. Ball Handler Position** (`2` floats)
- **Format:** `(q, r)` - absolute position of the player with the ball
- **Normalization:** Same as player positions
- **Purpose:** Explicitly encodes court region information (center vs. sidelines) to help the network learn location-dependent strategies
- **Fallback:** If no ball holder exists (terminal states), uses basket position

#### **F. Hoop Vector** (`2` floats, optional)
- **Format:** `(hoop_q, hoop_r)` - absolute position of basket
- **Included if:** `include_hoop_vector=True`
- **Normalization:** Same as player positions
- **Purpose:** Explicit encoding of basket location in absolute coordinates

#### **G. All-Pairs Offense-Defense Distances** (`players_per_side²` floats)
- **Format:** For each offensive player, distance to each defender (row-major order)
- **Example (3v3):** 9 values `[O0→D0, O0→D1, O0→D2, O1→D0, O1→D1, O1→D2, O2→D0, O2→D1, O2→D2]`
- **Calculation:** Hex distance between each offense-defense pair
- **Normalization:** Divided by `max(court_width, court_height)` if `normalize_obs=True`
- **Purpose:** 
  - Provides complete defensive coverage information
  - Enables understanding of help defense positioning
  - Network can learn "nearest defender" (minimum per row) if needed
  - Reveals double-team situations and defensive rotations

#### **H. All-Pairs Offense-Defense Angle Cosines** (`players_per_side²` floats)
- **Format:** For each offensive player, cos(angle) to each defender (same ordering as distances)
- **Example (3v3):** 9 values matching the distance matrix ordering
- **Calculation:** For each (offense, defender) pair, cos(angle) where angle is at the offensive player between:
  - Vector from offensive player to basket (direction they want to go)
  - Vector from offensive player to defender (where the defender is)
- **Values:** Range `[-1.0, +1.0]`
  - `+1.0`: Defender directly between offensive player and basket (optimal defensive position)
  - `0.0`: Defender perpendicular/to the side (neutral/help defense)
  - `-1.0`: Defender directly behind offensive player (beaten or weakside help)
- **Purpose:**
  - Captures defensive positioning quality, not just distance
  - "Close but behind" vs "close and in front" have very different meanings
  - Enables learning about on-ball pressure vs help defense
  - Combined with distance, provides complete defensive context

#### **H. Lane Step Counts** (`n_players` floats)
- **Format:** One count per player (offensive and defensive)
- **Offensive Players:** Steps spent in offensive lane (3-second violation tracking)
- **Defensive Players:** Steps spent in defensive key (illegal defense tracking)
- **Range:** `[0, three_second_max_steps]` (typically 3)
- **Purpose:** Enables agents to learn rule compliance (avoid violations)

#### **I. Expected Points (EP)** (`players_per_side` floats)
- **Format:** One EP value per offensive player (in order of player ID)
- **Calculation:** Pressure-adjusted expected value of a shot from their current position
- **Factors:** 
  - Distance to basket (layup vs three-pointer percentages)
  - Defender pressure (shot pressure mechanics)
  - Player's individual shooting skill
- **Purpose:** Helps agents evaluate shot quality and make better shooting decisions

#### **J. Turnover Probabilities** (`players_per_side` floats)
- **Format:** One probability per offensive player (fixed-position encoding)
- **Values:** Non-zero only for the current ball handler, zero for all others
- **Calculation:** Based on defender proximity using exponential decay
- **Purpose:** Explicit risk signal for holding the ball under pressure

#### **K. Steal Risks** (`players_per_side` floats)
- **Format:** One probability per offensive player (fixed-position encoding)
- **Values:** Non-zero for potential pass receivers, zero for ball holder
- **Calculation:** Geometric steal probability based on:
  - Pass distance
  - Defenders along pass trajectory
  - Perpendicular distance of defenders from pass line
- **Purpose:** Helps agent evaluate passing risks and make safer pass decisions

### Size Calculation for `"obs"` Vector (3v3 example):

```python
n_players = 6
players_per_side = 3

# Size of observation["obs"] - the main state vector only
obs_size = (
    n_players * 2                      # Player positions (absolute): 12
    + n_players                        # Ball holder one-hot: 6
    + 1                                # Shot clock: 1
    + n_players                        # Team encoding (±1 per player): 6 [NEW]
    + 2                                # Ball handler position (absolute): 2
    + 2                                # Hoop vector (if included): 2
    + players_per_side * players_per_side  # All-pairs distances: 9
    + players_per_side * players_per_side  # All-pairs angles: 9
    + n_players                        # Lane step counts: 6
    + players_per_side                 # Expected Points: 3
    + players_per_side                 # Turnover probabilities: 3
    + players_per_side                 # Steal risks: 3
) = 62 floats (with hoop vector) [was 56, +6 for team encoding]

# Additional dictionary keys (separate from "obs"):
# observation["skills"] = 9 floats (3 per offensive player)
# observation["action_mask"] = (6, 14) int8 array
# observation["role_flag"] = 1 float
```

---

## 2. Action Mask (`"action_mask"`)

- **Shape:** `(n_players, 14)` - one row per player, one column per action type
- **Format:** Binary mask (0 = illegal, 1 = legal)
- **Purpose:** Prevents agents from attempting illegal actions (moving off-court, shooting from too far, passing to non-existent teammates, etc.)
- **Actions:** `NOOP, MOVE_E, MOVE_NE, MOVE_NW, MOVE_W, MOVE_SW, MOVE_SE, SHOOT, PASS_E, PASS_NE, PASS_NW, PASS_W, PASS_SW, PASS_SE`

**Why important?** Masking illegal actions dramatically accelerates learning by preventing the agent from wasting time exploring invalid moves.

---

## 3. Role Flag (`"role_flag"`)

- **Shape:** `(1,)` - single float
- **Values:** `+1.0` for offensive team, `-1.0` for defensive team
- **Purpose:** Indicates which team the agent is currently controlling
- **Symmetric Encoding:** Using +1/-1 instead of 0/1 provides better gradient flow and symmetry

**Why separate?** This allows a single shared policy to play both offense and defense by conditioning its behavior on this flag.

---

## 4. Skills (`"skills"`)

- **Shape:** `(players_per_side × 3,)` - three skills per offensive player
- **Format:** Skill deltas relative to baseline percentages
- **Order per player:** `(layup_delta, three_pt_delta, dunk_delta)`
- **Example (3v3):** 9 floats total `[P0_layup_Δ, P0_3pt_Δ, P0_dunk_Δ, P1_layup_Δ, P1_3pt_Δ, P1_dunk_Δ, P2_layup_Δ, P2_3pt_Δ, P2_dunk_Δ]`
- **Concrete Example:** If player 0 has `layup_pct = 0.65` and baseline is `0.60`, the first value is `+0.05`

**Purpose:** Enables agents to:
- Make better decisions about who should shoot
- Learn role specialization (e.g., three-point specialists vs. inside scorers)
- Adapt strategy to team composition

**Why deltas?** Expressing skills as differences from baseline makes it easier for the network to learn relative strengths.

**Why separate from main `"obs"`?** Skills are sampled once per episode and remain constant, while the main `"obs"` vector changes every step. This separation can help the network architecture process static vs dynamic information differently.

---

## Design Principles

### **1. Absolute Court Coordinates**
- All spatial information is in absolute court coordinates (not relative to ball handler)
- Allows network to learn location-dependent strategies
- Enables awareness of court regions: center court (more space) vs. sidelines (constrained)
- Maintains interpretability of positions

### **2. Explicit Ball Handler Position**
- Ball handler position is included as a separate feature
- Helps the network explicitly model court position effects
- Reduces reliance on implicit positional inference

### **3. Fixed-Position Encoding**
- EP, turnover risk, and steal risk use fixed player indices
- Position `i` always corresponds to offensive player `i`
- More stable for learning than dynamic ordering

### **4. Rich Defensive Context**
- All-pairs distance and angle matrices provide complete defensive information
- Replaces single "nearest defender" with full defensive coverage picture
- Angle cosines capture positioning quality: defender "in front" vs "behind" or "helping"
- Enables learning about help defense, switches, and double-teams
- Network can derive simpler features (e.g., nearest defender) if useful

### **5. Explicit Risk Signals**
- EP, turnover probability, and steal risks are calculated by the environment
- Provides clear learning signals about decision quality
- Accelerates learning of risk-aware behavior

### **6. Normalization**
- Spatial coordinates normalized to `[-1, 1]` range
- Angle cosines naturally in `[-1, 1]` range
- Improves neural network training stability
- Can be disabled with `normalize_obs=False` for debugging

---

## Configuration Flags

Key environment parameters that affect observation structure:

- `include_hoop_vector`: Add explicit hoop position vector (default: `True`)
- `normalize_obs`: Normalize spatial coordinates (default: `True`)

---

## Notes on Architecture Change

**As of the latest update, the environment uses absolute coordinates instead of egocentric coordinates.** This change allows the network to:
- Learn that center court offers more space than sidelines
- Develop position-aware strategies (e.g., different tactics near basket vs. mid-court)
- Explicitly condition decisions on court location
- Better model real basketball dynamics where position matters

The old flags `use_egocentric_obs` and `egocentric_rotate_to_hoop` are deprecated and no longer used.

---

By providing this rich, structured observation space with absolute coordinates, we give the RL agent all the information needed to learn sophisticated, location-aware basketball strategies while maintaining computational efficiency through careful normalization and representation choices.