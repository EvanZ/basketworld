# Per-Player Skill Heatmaps (Pure CNN Mode)

## Overview

Instead of encoding player skills as a single uniform channel, we now create **one spatial heatmap per offensive player**, showing their unpressured shooting probability from every position on the court.

## What Changed

### Before (Old Implementation)
```
Channel 7: Average skill (uniform grid with mean skill delta)
```
- ❌ Lost individual player information
- ❌ No spatial information
- ❌ CNN couldn't learn player-specific positioning

### After (New Implementation)
```
Channels 7-9 (for 3v3): One heatmap per offensive player
  - Player 0: Their shooting probability from each position
  - Player 1: Their shooting probability from each position  
  - Player 2: Their shooting probability from each position
```
- ✅ Individual player skills preserved
- ✅ Spatially meaningful (shows distance-based falloff)
- ✅ CNN learns optimal positioning per player
- ✅ Enables learning player specialization (3PT shooter vs layup specialist)

## Channel Configuration

### Pure CNN Mode (for 3v3)

| Channel | Content | Values |
|---------|---------|--------|
| 0 | Players | +1 (offense), -1 (defense) |
| 1 | Ball holder | 0 or 1 |
| 2 | Basket | 0 or 1 |
| 3 | Expected points (current ball holder) | 0.0-3.0 |
| 4 | Three-point arc | 0 (inside), 1 (outside) |
| 5 | Shot clock | uniform, 0-1 normalized |
| 6 | Role flag | uniform, 0 or 1 |
| **7** | **Player 0 skill heatmap** | **0.01-0.99 (probability)** |
| **8** | **Player 1 skill heatmap** | **0.01-0.99 (probability)** |
| **9** | **Player 2 skill heatmap** | **0.01-0.99 (probability)** |

**Total: 10 channels for 3v3** (7 + N where N = players_per_side)

## What Each Skill Heatmap Shows

Each heatmap encodes that player's **base shooting probability** (without defender pressure) from every position:

```python
# Example: Player 0's skill heatmap
Channel 7:
  Near basket:    0.548  ← High layup %
  Mid-range:      0.314  ← Medium
  3PT line:       0.331  ← This player's 3PT %
  Far corner:     0.264  ← Extrapolated
```

### Key Properties

1. **Distance-based**: Probability decreases with distance from basket
2. **Player-specific**: Each player has different skills (sampled per episode)
3. **Unpressured**: Shows base skill without defender pressure
4. **Computed once**: Calculated after skill sampling in `reset()`, cached for episode

## Example: Learning Player Specialization

Imagine two players with different skills:

**Player 0 (3PT Specialist):**
- Layup: 55%
- 3PT: 40%
- Heatmap shows relatively flat falloff (good from distance)

**Player 1 (Layup Specialist):**
- Layup: 68%
- 3PT: 30%
- Heatmap shows steep falloff (much better close)

**What CNN Learns:**
- Pass to Player 0 when at 3PT line
- Pass to Player 1 when driving to basket
- Optimal positioning for each player's skills

## Benefits for Learning

### 1. Strategic Passing
CNN learns **who** to pass to based on **where** they are:
- "Player 2 is good from the corner, pass to them!"
- "Player 0 is better driving, don't pass for 3PT shot"

### 2. Optimal Positioning
Model learns where each player should position themselves:
- 3PT specialists stay outside
- Layup specialists cut to basket
- Match player strengths to court positions

### 3. Role Specialization
Naturally emerges without explicit coding:
- Some players become shooters
- Some players become drivers
- Some become facilitators (passers)

### 4. Skill-Aware Decision Making
Actions depend on individual abilities:
- High-skill player shoots contested shots
- Low-skill player waits for open looks
- Team coordination based on complementary skills

## Implementation Details

### Computation (Once Per Episode)

```python
def _compute_skill_heatmaps(self) -> np.ndarray:
    """
    Compute per-player skill heatmaps after skill sampling in reset().
    
    Returns: (players_per_side, court_height, court_width)
    """
    heatmaps = np.zeros((self.players_per_side, H, W))
    
    for player_idx in range(self.players_per_side):
        for each position (q, r) on court:
            dist = distance_to_basket(q, r)
            prob = player_base_probability(player_idx, dist)
            heatmaps[player_idx, row, col] = prob
    
    return heatmaps
```

### Key Points:
- **Unpressured**: Uses `_calculate_base_shot_probability()` (no defender pressure)
- **Cached**: Stored in `self._cached_skill_heatmaps` for the episode
- **Per-episode**: Recomputed after each `reset()` when skills are resampled

## Comparison: Different Team Sizes

### 2v2 → 9 channels
- Base: 5
- Scalars: 2 (shot clock, role)
- Skills: **2** (one per player)

### 3v3 → 10 channels  
- Base: 5
- Scalars: 2
- Skills: **3** (one per player)

### 5v5 → 12 channels
- Base: 5
- Scalars: 2
- Skills: **5** (one per player)

**Scales linearly with team size!**

## Training with Per-Player Heatmaps

```bash
# 3v3 with per-player skill heatmaps
python train/train.py \
    --use-pure-cnn \
    --players 3 \
    --alternations 150 \
    --cnn-channels 32 64 128 128 \
    --mlflow-experiment-name "Pure CNN with Per-Player Skills"
```

**Expected benefits:**
- Better passing decisions
- Natural role specialization  
- More efficient team coordination
- Skill-aware positioning

## Visualization Example

For a 3v3 game with varied skills:

```
Player 0 Heatmap (Channel 7):          Player 1 Heatmap (Channel 8):
0.26 0.28 0.31 0.34 0.37              0.19 0.22 0.25 0.28 0.31
0.28 0.31 0.34 0.38 0.41              0.22 0.25 0.28 0.32 0.35
0.31 0.35 0.39 0.43 0.46              0.25 0.29 0.33 0.37 0.40
0.35 0.40 0.45 0.50 0.52     vs      0.29 0.34 0.39 0.44 0.47
0.40 0.46 0.52 0.55 0.55              0.34 0.40 0.46 0.52 0.56
         (3PT shooter)                        (Layup specialist)
```

**CNN learns:** Player 0 for distance shots, Player 1 for close shots

## Summary

| Aspect | Old (Uniform Skill) | New (Per-Player Heatmaps) |
|--------|---------------------|---------------------------|
| **Information** | Average only | Individual + spatial |
| **Player differentiation** | ❌ Lost | ✅ Preserved |
| **Spatial meaning** | ❌ None | ✅ Distance-based |
| **Enables specialization** | ❌ No | ✅ Yes |
| **Optimal positioning** | ❌ Hard to learn | ✅ Natural |
| **Strategic passing** | ❌ Random | ✅ Skill-aware |
| **Channels** | 8 (fixed) | 7 + N (scales) |

---

**This is a significant improvement for Pure CNN mode!** The CNN now has access to fine-grained, spatially-meaningful skill information that enables learning sophisticated team coordination strategies.

