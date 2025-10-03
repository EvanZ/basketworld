# Phi Aggregation Modes

## Overview

The potential function Φ(s) can now aggregate teammate expected points in different ways to better separate individual vs team accountability.

## The Problem

With the original `team_best` mode and blend weight w=0.3:
```
Φ(s) = 0.7 × max(all_team_EPs) + 0.3 × ball_handler_EP
```

**Issue:** When ball handler degrades shot quality but teammates maintain good positions, the max includes the ball handler's current EP, so Φ barely changes. The agent doesn't feel strong penalties for individual bad decisions.

## Available Modes

### 1. **team_best** (default/legacy)
```
Φ = (1-w) × max(all_teammates_including_ball_handler) + w × ball_handler_EP
```

**Use case:** Original behavior, maintained for backward compatibility.

**Problem:** Ball handler appears in both terms when they have the best shot, dampening penalties.

### 2. **teammates_best** (RECOMMENDED)
```
Φ = (1-w) × max(teammates_excluding_ball_handler) + w × ball_handler_EP
```

**Use case:** Clean separation of individual vs team accountability.

**Advantages:**
- Ball handler's decisions only affect the `w` term
- Teammate spacing/positioning only affects the `(1-w)` term
- No double-counting
- Strong penalties for degrading your own shot quality

**Example from user's episode:**
```
Step 8: Ball EP=1.194, Teammate best=1.151
  Φ = 0.7×1.151 + 0.3×1.194 = 1.164

Step 9: Ball EP=0.870, Teammate best=1.151  
  Φ = 0.7×1.151 + 0.3×0.870 = 1.067

Penalty = β × (1.067 - 1.164) = 0.15 × (-0.097) = -0.0145 per team
```

With w=0.5, penalty would be even stronger: -0.024 per team.

### 3. **teammates_avg**
```
Φ = (1-w) × mean(teammates_excluding_ball_handler) + w × ball_handler_EP
```

**Use case:** Smoother signal, rewards overall team spacing.

**Advantages:**
- Less sensitive to single teammate movement
- Encourages balanced team positioning
- Still separates ball handler accountability

**Trade-offs:**
- Weaker signal for "find the best shot" (uses average, not max)
- Might not encourage passing to the open player as strongly

### 4. **team_avg**
```
Φ = mean(all_teammates_including_ball_handler)
```

**Use case:** Simplest option, no blend weight needed.

**Advantages:**
- Conceptually simple
- Automatically balances individual (1/N) vs team ((N-1)/N)
- No hyperparameter w to tune

**Trade-offs:**
- With 3 players, ball handler is only 33% of Φ
- Might need higher β to get sufficient signal

## Recommended Configurations

### For Learning Shot Selection (Current Issue)
```bash
--phi-aggregation-mode teammates_best \
--phi-blend-weight 0.5 \  # Balance individual and team
--phi-beta-start 0.15 \
--phi-beta-end 0.15  # Keep constant
```

**Why:** Provides strong individual accountability while maintaining team coordination incentives.

### For Emphasizing Team Play
```bash
--phi-aggregation-mode teammates_avg \
--phi-blend-weight 0.3 \
--phi-beta-start 0.20  # Slightly higher β
```

**Why:** Rewards overall team spacing and balanced positioning.

### For Simplicity
```bash
--phi-aggregation-mode team_avg \
--phi-beta-start 0.20
# (phi-blend-weight ignored in this mode)
```

**Why:** No need to tune blend weight, natural 1/N weighting.

## Mathematical Comparison

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

**Note:** `team_best` and `teammates_best` give same result in this state because ball handler doesn't have the max. The difference appears when ball handler *does* have the max.

## When Each Mode Shines

### Use `teammates_best` when:
- Learning shot selection is critical
- You want strong individual accountability
- Agents need to learn "don't degrade your shot"
- Passing to the open player is important

### Use `teammates_avg` when:
- Emphasizing team coordination and spacing
- You want smoother, more stable shaping signals
- All teammates' positions matter, not just the best

### Use `team_avg` when:
- You prefer simplicity over fine-tuning
- You have many players (N>3) where 1/N is reasonable
- You want guaranteed equal treatment

### Use `team_best` when:
- You need backward compatibility with existing runs
- You've already tuned w for this mode

## Implementation Details

The aggregation mode is set via CLI:
```bash
--phi-aggregation-mode teammates_best
```

And stored in the environment:
```python
env = HexagonBasketballEnv(
    phi_aggregation_mode="teammates_best",
    phi_blend_weight=0.5,
    ...
)
```

The mode can be checked in UI or logs via the `phi_aggregation_mode` parameter.

## Troubleshooting

**Q: My penalties seem too weak even with teammates_best?**

A: Try increasing blend weight `w` or beta `β`:
```bash
--phi-blend-weight 0.6  # More weight on ball handler
--phi-beta-start 0.20   # Stronger signal overall
```

**Q: Should I change mode mid-training?**

A: No - this changes the reward structure. Pick a mode at the start and keep it constant.

**Q: Which mode for evaluation?**

A: Use the same mode the policy was trained with. The policy learned values under that specific Φ definition.


