# Spacing Features Plan

## Goal

Add explicit spacing features at two levels:

- Team/global spacing for offense and defense.
- Per-player self-spacing relative to the team baseline.

This should work for both observation paths:

- Flat vector: `obs` (from `basketworld/envs/core/observations.py`).
- Set tokens: `players` and `globals` (from `basketworld/utils/wrappers.py`).

## Candidate Metrics

### Option A: Centroid Radius (your proposed baseline)

For each team `T` with player coordinates `x_i`:

- `c_T = mean_i(x_i)` (team centroid)
- `r_i = ||x_i - c_T||`
- `G_centroid(T) = mean_i(r_i)` (global team spacing)
- `S_centroid(i) = r_i / (G_centroid(T) + eps)` (player normalized spacing)

Pros:

- Simple and interpretable.
- Cheap to compute.

Risks:

- Can miss formation shape (different geometries can share same mean radius).
- Sensitive to one outlier player.

### Option B: Pairwise Team Distances (recommended v1)

Use normalized teammate distances `d(i, j)` (hex distance divided by `norm_den`):

- `G_pair(T) = mean_{i<j} d(i, j)` (global team spacing)
- `G_nn(T) = mean_i min_{j!=i} d(i, j)` (optional cluster signal)
- `S_pair(i) = mean_{j!=i} d(i, j) / (G_pair(T) + eps)`
- `S_nn(i) = min_{j!=i} d(i, j) / (G_nn(T) + eps)` (optional)

Pros:

- More robust than centroid to outliers.
- Fits existing geometry usage (hex distances already used heavily).
- Directly captures teammate spread/clustering.

Risks:

- Slightly more compute than centroid (still small at current team sizes).

### Option C: Covariance + Mahalanobis (experimental)

Use Cartesian coordinates for each team:

- `Sigma_T = cov(X_T) + lambda*I`
- `G_cov(T) = logdet(Sigma_T)` (or `trace(Sigma_T)`)
- `S_cov(i) = sqrt((x_i-c_T)^T * inv(Sigma_T) * (x_i-c_T))`

Pros:

- Shape-aware (anisotropy/orientation).
- Better for width-vs-depth spacing interpretation.

Risks:

- More numerically sensitive for small `N`.
- More implementation and testing complexity.

## Recommended v1

Start with **Option B (pairwise)** as default, keep centroid as an ablation option.

Recommended feature set:

- Global (offense + defense): `G_pair(off)`, `G_pair(def)`.
- Per-player: `S_pair(i)` for every player.

Optional if needed after first ablation:

- Add `G_nn(off/def)` and/or `S_nn(i)`.

Why this first:

- More robust than centroid.
- Easy to reason about on hex grids.
- Minimizes numerical edge cases and implementation risk.

## Integration Plan

### 1. Core Geometry Helpers

Add spacing helper functions in `basketworld/envs/core/observations.py`:

- Team-level spacing computation for a given player-id set.
- Per-player normalized self-spacing vector.
- Edge-case handling:
  - Team size `<= 1`: return zeros.
  - Guard with `eps` to avoid divide-by-zero.

### 2. Flat Observation (`obs`) Path

In `basketworld/envs/core/observations.py` and `basketworld/envs/basketworld_env_v2.py`:

- Append new spacing features to `build_observation`.
- Update `state_vector_length` accounting in env init.
- Keep ordering stable and explicit (document final offsets).

### 3. Set-Observation Tokens/Globals

In `basketworld/utils/wrappers.py`:

- Add team spacing signals to `globals` (offense, defense).
- Add self-spacing ratio(s) to per-player token rows.
- Update `TOKEN_DIM`, `GLOBAL_DIM`, and index constants.
- Keep values normalized and finite.

Note: if spacing features are purely distance-based, mirror transforms should preserve them. Still verify mirror behavior in tests.

### 4. Frontend/Debug Display

Update `app/frontend/src/components/PlayerControls.vue`:

- `obsMeta` offsets for new flat `obs` fields.
- Token feature/global labels for new set-token fields.
- Keep existing debug tables aligned with updated dimensions.

### 5. Tests

Update/add tests:

- `tests/test_observations.py`
  - Shape/length checks include new spacing fields.
  - Finite-value assertions.
- `tests/test_set_observation_wrapper.py`
  - New token/global dimensions.
  - Finite-value checks.
- Mirror test
  - Confirm derived spacing fields remain consistent after mirroring.

## Evaluation Plan (before default-on)

Run a small ablation:

1. Baseline (no spacing additions).
2. Centroid-only.
3. Pairwise-only.
4. Pairwise + centroid.

Compare:

- Learning speed (early sample efficiency).
- Final performance (offense/defense metrics already tracked in training).
- Stability (variance across seeds).
- Permutation consistency diagnostics (using your existing workflow artifacts).

Decision rule:

- Promote whichever variant gives best stability/performance tradeoff with minimal complexity.

## Rollout Strategy

Add a feature flag in config (example):

- `--spacing-feature-mode none|centroid|pairwise|pairwise_plus_centroid|covariance`

Initial default:

- `none` for compatibility until ablation is complete.

After validation:

- switch default to `pairwise`.

## Open Questions

- Should spacing use hex distance everywhere, or Cartesian Euclidean for centroid/covariance modes?
- Do we want offense/defense spacing both in `obs` and `globals`, or one canonical location?
- Is a second per-player spacing channel (`S_nn`) worth the extra token dimension?
