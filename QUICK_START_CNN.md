# Quick Start: CNN Training for BasketWorld

## Three Ways to Train

### 1. Pure MLP (Original - Fastest)
```bash
python train/train.py \
    --alternations 50 \
    --mlflow-experiment-name "Pure MLP Baseline"
```
- No spatial reasoning
- Processes vectors only
- Fastest training

### 2. Hybrid CNN+MLP (Recommended)
```bash
python train/train.py \
    --use-spatial-obs \
    --alternations 100 \
    --mlflow-experiment-name "Hybrid CNN+MLP"
```
- CNN learns spatial patterns (5 channels)
- MLP processes precise numbers (shot clock, skills)
- **Best performance expected**

### 3. Pure CNN (Experimental)
```bash
python train/train.py \
    --use-pure-cnn \
    --alternations 150 \
    --mlflow-experiment-name "Pure CNN"
```
- Everything encoded spatially (8 channels)
- Single learning path
- Needs more training

---

## Quick Answers to Common Questions

### "Why not use SB3's CnnPolicy?"

We **don't use** `CnnPolicy` because:
- It expects Box observations, we have Dict
- We need `PassBiasMultiInputPolicy` for curriculum learning
- Custom feature extractors are the right approach

**What happens under the hood:**
```python
PPO(
    PassBiasMultiInputPolicy,  # ← Policy type
    env,
    policy_kwargs={
        "features_extractor_class": HexCNNFeatureExtractor,  # ← Our custom CNN
    }
)
```

### "Why do we need MLP if we have CNN?"

**Short answer:** Not all features are in the CNN!

**Hybrid Mode (5 spatial + vectors):**
- CNN sees: Players, ball, basket, expected points, 3pt arc
- MLP sees: Shot clock (as number 17), skills (percentages), role

**Why?** Shot clock=17 is clearer as a number than a uniform spatial grid

**Pure CNN Mode (8 spatial only):**
- CNN sees everything, including shot clock encoded spatially
- No MLP branch
- Simpler but less efficient

---

## Which Should I Use?

| Use Case | Recommendation |
|----------|----------------|
| **Best performance** | Hybrid CNN+MLP (`--use-spatial-obs`) |
| **Fastest experiments** | Pure MLP (no flags) |
| **Research/ablation** | Try all three |
| **Learning formations** | Hybrid or Pure CNN |
| **Limited compute** | Pure MLP |

---

## Architecture Comparison

### Hybrid (Recommended)
```
spatial (5×12×12) → CNN → 256-dim ──┐
                                     ├─→ 384-dim → Policy
obs, skills, role → MLP → 128-dim ──┘
```

### Pure CNN
```
spatial (8×12×12) → CNN (deeper) → 512-dim → Policy
```

---

## Testing Your Setup

```bash
# Verify implementation
source .env/bin/activate
python test_cnn_implementation.py

# Visualize what CNN sees
python visualize_spatial_obs.py
```

---

## Advanced Configuration

```bash
# Hybrid with custom architecture
python train/train.py \
    --use-spatial-obs \
    --cnn-channels 16 32 64 \
    --cnn-features-dim 128 \
    --mlp-features-dim 64 \
    --net-arch-pi 128 128 \
    --net-arch-vf 128 128

# Pure CNN with deep network
python train/train.py \
    --use-pure-cnn \
    --cnn-channels 32 64 128 256 \
    --cnn-features-dim 512
```

---

## Channel Summary

### Hybrid Mode (5 channels)
0. Players: +1 (offense), -1 (defense)
1. Ball holder: binary
2. Basket: binary
3. Expected points: 0.0-3.0
4. Three-point arc: 0 (inside), 1 (outside)

### Pure CNN Mode (adds 2 + N more, where N = players_per_side)
5. Shot clock: uniform, 0-1 normalized
6. Role flag: uniform, 0 or 1
7-N. Per-player skill heatmaps: each player's unpressured shooting probability from each position

---

## Documentation

- `CNN_POLICY_CLARIFICATION.md` - Detailed explanation of architectures
- `CNN_IMPLEMENTATION_SUMMARY.md` - Technical implementation details
- `test_cnn_implementation.py` - Test suite
- `visualize_spatial_obs.py` - Visualization tool

---

**Ready to train? Start with:**
```bash
python train/train.py --use-spatial-obs --alternations 100
```

