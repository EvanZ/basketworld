# CNN Policy Implementation - Clarification & Comparison

## TL;DR - Answering Your Questions

### Q1: "Why not CnnPolicy?"

**Answer:** We **intentionally don't use** SB3's built-in `CnnPolicy` because:

1. `CnnPolicy` expects a simple **Box observation space** (e.g., image with shape `(C, H, W)`)
2. We have a **Dict observation space** with multiple components
3. We need to preserve `PassBiasMultiInputPolicy` for curriculum learning features

**What we do instead:**
- Use `PassBiasMultiInputPolicy` (the policy type)
- Plug in a **custom feature extractor** via `policy_kwargs["features_extractor_class"]`
- This is the **recommended approach** in SB3 docs for custom observation spaces

```python
# This is how it works:
PPO(
    PassBiasMultiInputPolicy,  # ← Policy type (handles action distribution)
    env,
    policy_kwargs={
        "features_extractor_class": HexCNNFeatureExtractor,  # ← Custom CNN
        "features_extractor_kwargs": {...}
    }
)
```

### Q2: "Why still need MLP if we have CNN?"

**Answer:** Because **not all features are in the CNN**! We have three observation components:

| Component | Content | What Processes It |
|-----------|---------|-------------------|
| `spatial` | 5-channel grid (players, ball, basket, etc.) | **CNN** |
| `obs` | Positions, shot clock, defender distances | **MLP** |
| `skills` | Per-player shooting percentages | **MLP** |
| `role_flag` | Training offense or defense | **MLP** |

The MLP processes precise numerical values that don't need spatial convolution.

---

## Three Approaches - Detailed Comparison

### 1. Pure MLP (Original)

**When to use:** Traditional RL approach, fast training

```bash
python train/train.py \
    --alternations 50 \
    --num-envs 16
```

**Architecture:**
```
obs (vector) ──┐
skills        ─┤
role_flag     ─┤→ MaskAgnosticCombinedExtractor → MLP → Policy/Value
action_mask   ─┘   (zeros out mask)
```

**Pros:**
- ✅ Fastest training
- ✅ Smallest model
- ✅ Works well for numerical features

**Cons:**
- ❌ No spatial reasoning
- ❌ Harder to learn formations/positioning

---

### 2. Hybrid CNN+MLP (Recommended)

**When to use:** Best of both worlds - spatial patterns + precise numerical features

```bash
python train/train.py \
    --use-spatial-obs \
    --alternations 100 \
    --num-envs 16
```

**Architecture:**
```
spatial (5, H, W) → CNN → 256-dim ─┐
                                    ├→ Concat (384-dim) → Policy/Value
obs, skills, role → MLP → 128-dim ─┘
```

**Observation Breakdown:**
- **CNN processes:** 5 spatial channels
  - Ch 0: Players (+1/-1 encoding)
  - Ch 1: Ball holder
  - Ch 2: Basket position
  - Ch 3: Expected points heatmap
  - Ch 4: Three-point arc
  
- **MLP processes:** Vector features
  - Ego-centric positions
  - Shot clock (as number: 17, 16, 15...)
  - Player skills (shooting percentages)
  - Role flag (offense/defense)

**Pros:**
- ✅ CNN learns spatial patterns (formations, spacing)
- ✅ MLP handles precise values (shot clock=17 is clearer as number)
- ✅ Best performance expected
- ✅ More parameter efficient than pure CNN

**Cons:**
- ❌ Slightly slower than pure MLP
- ❌ Two branches to tune

**Why MLP for shot clock?** 
- Shot clock as number: 17 → model easily learns "urgency increases as it decreases"
- Shot clock spatially: uniform grid filled with 0.85 → CNN must learn relationship indirectly

---

### 3. Pure CNN (New Alternative)

**When to use:** Research/experimentation, or when you want CNN to learn everything

```bash
python train/train.py \
    --use-pure-cnn \
    --alternations 150 \
    --num-envs 12
```

**Architecture:**
```
spatial (8, H, W) → CNN (deeper) → 512-dim → Policy/Value
```

**Observation:** 7 + N spatial channels (where N = players_per_side)
- Ch 0-4: Same as hybrid (players, ball, basket, EP, 3pt arc)
- Ch 5: Shot clock (uniform grid, normalized 0-1)
- Ch 6: Role flag (uniform grid, 1.0 or 0.0)
- Ch 7+: Per-player skill heatmaps (each player's unpressured shooting probability map)
  - For 3v3: Channels 7, 8, 9 (one per offensive player)
  - **Shows individual player skills spatially!**

**Pros:**
- ✅ Single end-to-end learning path
- ✅ CNN learns to process all features together
- ✅ Simpler architecture (one branch)

**Cons:**
- ❌ Slower training (more conv layers needed)
- ❌ Uniform channels (shot clock, skills) don't benefit from convolution
- ❌ Needs more samples to converge
- ❌ Less parameter efficient

**Example:** Shot clock encoded as uniform channel
```
All cells = 0.85  (17/20 remaining)
All cells = 0.80  (16/20 remaining)
...
```

---

## Detailed Comparison Table

| Aspect | Pure MLP | Hybrid CNN+MLP | Pure CNN |
|--------|----------|----------------|----------|
| **Spatial reasoning** | ❌ None | ✅ Excellent | ✅ Excellent |
| **Numerical precision** | ✅ Excellent | ✅ Excellent | ⚠️ Indirect |
| **Training speed** | ✅ Fastest | ⚠️ Medium | ❌ Slowest |
| **Sample efficiency** | ✅ Good | ⚠️ Medium | ❌ Needs more |
| **Parameters** | ✅ Smallest | ⚠️ Medium | ❌ Largest |
| **Architecture complexity** | ✅ Simple | ⚠️ Two branches | ✅ Simple |
| **Expected performance** | ⚠️ Good | ✅ Best | ⚠️ Good |

---

## Recommendation

**Start with Hybrid CNN+MLP** (`--use-spatial-obs`):
- Gets spatial reasoning from CNN
- Gets numerical precision from MLP
- Best bang for buck
- Most flexible

**Use Pure MLP** if:
- You want fastest experiments
- Your env is small/simple
- You need baseline comparison

**Use Pure CNN** if:
- You're doing CNN architecture research
- You want to study end-to-end spatial learning
- You have lots of compute/samples

---

## Visual Comparison

### Hybrid Mode (5 spatial + vectors)
```
Environment State
├─ 5 Spatial Channels        → CNN (learns formations)
│  ├─ Players (±1)
│  ├─ Ball (binary)
│  ├─ Basket (binary) 
│  ├─ Expected Points (0-3)
│  └─ 3PT Arc (binary)
│
└─ Vector Features            → MLP (learns precise timing)
   ├─ obs: positions, distances
   ├─ Shot clock: 17 ← precise number
   ├─ Skills: [0.05, -0.02, 0.01]  ← precise deltas
   └─ Role: 1.0 (offense)
```

### Pure CNN Mode (7+N spatial only, N=players_per_side)
```
Environment State (3v3 example: 10 channels)
└─ Spatial Channels            → CNN (learns everything spatially)
   ├─ Players (±1)
   ├─ Ball (binary)
   ├─ Basket (binary)
   ├─ Expected Points (0-3)
   ├─ 3PT Arc (binary)
   ├─ Shot Clock (uniform: 0.85)  ← spatial encoding
   ├─ Role (uniform: 1.0)          ← spatial encoding
   ├─ Player 0 skill heatmap       ← per-player, spatially varying!
   ├─ Player 1 skill heatmap       ← per-player, spatially varying!
   └─ Player 2 skill heatmap       ← per-player, spatially varying!
```

---

## Example Training Commands

### Hybrid CNN+MLP (Recommended)
```bash
python train/train.py \
    --use-spatial-obs \
    --cnn-channels 32 64 64 \
    --cnn-features-dim 256 \
    --mlp-features-dim 128 \
    --alternations 100 \
    --learning-rate 3e-4 \
    --mlflow-experiment-name "Hybrid CNN+MLP"
```

### Pure CNN (Experimental)
```bash
python train/train.py \
    --use-pure-cnn \
    --cnn-channels 32 64 128 128 \
    --cnn-features-dim 512 \
    --alternations 150 \
    --learning-rate 2e-4 \
    --mlflow-experiment-name "Pure CNN"
```

### Ablation Study (Compare All Three)
```bash
# 1. Pure MLP baseline
python train/train.py \
    --mlflow-experiment-name "Ablation Study" \
    --mlflow-run-name "Pure MLP" \
    --alternations 100

# 2. Hybrid CNN+MLP
python train/train.py \
    --use-spatial-obs \
    --mlflow-experiment-name "Ablation Study" \
    --mlflow-run-name "Hybrid" \
    --alternations 100

# 3. Pure CNN
python train/train.py \
    --use-pure-cnn \
    --mlflow-experiment-name "Ablation Study" \
    --mlflow-run-name "Pure CNN" \
    --alternations 150
```

---

## Key Takeaways

1. **We DO use CNNs** - just not SB3's `CnnPolicy` directly
2. **Custom feature extractors** are the right way to handle Dict observations
3. **MLP is still valuable** for precise numerical features (shot clock, skills)
4. **Hybrid approach** combines strengths of both CNN and MLP
5. **Pure CNN** is an option but typically less efficient

The implementation gives you **full flexibility** to choose the best approach for your needs!

