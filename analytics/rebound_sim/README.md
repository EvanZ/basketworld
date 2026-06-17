# Rebound Simulation Prototype

Standalone Python prototype for tuning BW rebound landing and rebound-winner
logic before porting the settled model to JAX.

The simulator uses the current canonical JAX half-court geometry by default:

- `court_rows=9`
- `court_cols=8`
- left-baseline rim at the middle row
- existing odd-r offset, axial, and cartesian hex geometry
- existing 3PT geometry for shot classification/reporting

Run:

```bash
.env/bin/python analytics/rebound_sim/run_rebound_sim.py --samples 50000 --seed 0
```

Outputs:

- `analytics/rebound_sim/outputs/rebound_sim_summary.json`
- `analytics/rebound_sim/outputs/rebound_sim_summary.png`
- `analytics/rebound_sim/outputs/rebound_sim_region_flow.png`

Condition rebound heatmaps on specific shot hexes by passing flat axial `q,r` pairs. Use `--shot-cells=-4,8` syntax when the first coordinate is negative so argparse does not treat it as a new flag.

```bash
.env/bin/python analytics/rebound_sim/run_rebound_sim.py \
  --samples 50000 \
  --shot-cells=-4,8,-3,8,0,4
```

This keeps the normal aggregate summary and also writes one file per shot cell, for example:

- `analytics/rebound_sim/outputs/rebound_heatmap_shot_q-4_r8.png`
- `analytics/rebound_sim/outputs/rebound_heatmap_shot_q-3_r8.png`
- `analytics/rebound_sim/outputs/rebound_sim_conditioned_summary.json`

Set `--conditioned-samples` if you want a different sample count per fixed shot cell.


## Fitted MuJoCo Target Table

The prototype can now use the Milestone 4 fitted catch table for rebound target
sampling while keeping the existing prototype winner model. This is the intended
Milestone 6 bridge before porting the table lookup/sampling logic into JAX.

```bash
.env/bin/python analytics/rebound_sim/run_rebound_sim.py \
  --samples 50000 \
  --fitted-target-model-dir analytics/rebound_physics/outputs/dataset_9x8/fitted_catch_model
```

In this mode:

1. The missed-shot cell is mapped to the fitted model's canonical side.
2. The shot type (`dunk`, `finger_roll`, `layup`, `jumper`) selects the matching table.
3. A rebound target cell is sampled from the fitted MuJoCo distribution.
4. If the original shot was on the mirrored side, the sampled target is reflected back.
5. The existing prototype logic still samples player positions and rebound winner.

The fitted table is a smoothed empirical categorical distribution, not a
continuous analytic rebound formula. For shot type `k`, canonical shot cell `s`,
and rebound target cell `t`:

```text
P(target = t | shot_type = k, canonical_shot_cell = s)
  = (count[k, s, t] + pseudocount)
    / sum_j(count[k, s, j] + pseudocount)
```

The current fitted catch model uses `catch_cell_index` as the target cell and
excludes made shots, missing target cells, and behind-backboard target artifacts.
The default pseudocount is `0.05`.

At runtime:

```text
shot_cell
  -> canonical_shot_cell via L/R reflection
  -> shot_type table
  -> categorical target sample
  -> reflected target if the original shot was mirrored
```

This samples only the rebound target. The prototype winner model then samples
which already-placed player gets that target:

```text
score_i = -target_distance_weight * distance(player_i, target)

P(winner = i) = softmax(score_i / winner_temperature)
```

There is no explicit offense/defense side bonus, boxout term, inside-position bonus, player-skill bonus, or shooter penalty in this winner formula. At this prototype stage, only distance to the sampled rebound target affects the winner distribution.

## JAX Training Feature Bridge

The JAX halfcourt environment now uses the fitted target table for both rebound
sampling and rebound-aware policy observations. For the current offensive ball
handler's hypothetical shot, the observation exposes distributional features
before any miss/rebound sample is drawn:

Global features:

- `expected_rebound_target_q`
- `expected_rebound_target_r`
- `target_entropy`
- `orb_prob_if_current_shot_misses`

Per-player features:

- `dist_to_expected_rebound_target`
- `rebound_win_prob_if_current_shot_misses`

`drb_prob_if_current_shot_misses` is intentionally omitted because it is exactly
`1 - orb_prob_if_current_shot_misses`. These features are zeroed when rebounds
are disabled or when there is no valid offensive ball holder. They use the full
target distribution, not the sampled future rebound target, so they provide
learnable rebounding context without leaking the random outcome.

The conditioned heatmap flow also works with the fitted table:

```bash
.env/bin/python analytics/rebound_sim/run_rebound_sim.py \
  --samples 10000 \
  --conditioned-samples 5000 \
  --shot-cells=-4,8,3,4 \
  --fitted-target-model-dir analytics/rebound_physics/outputs/dataset_9x8/fitted_catch_model
```

All `ReboundParams` fields are exposed as kebab-case CLI flags. The most
important fitted-table runtime knobs are the winner parameters:

```bash
--target-distance-weight 1.10
--winner-temperature 0.75
```

Shot-location parameters are used when `--shot-cells` is omitted:

```bash
--shot-rim-weight 0.16
--shot-paint-weight 0.24
--shot-midrange-weight 0.20
--shot-three-weight 0.40
--shot-rim-center 1.0
--shot-paint-center 2.0
--shot-midrange-center 3.4
--shot-three-center 4.8
--shot-rim-sigma 0.65
--shot-paint-sigma 0.75
--shot-midrange-sigma 0.95
--shot-three-sigma 0.85
```

Synthetic target parameters are only used when `--fitted-target-model-dir` is
not supplied:

```bash
--short-rebound-base-hex 1.0
--normal-rebound-base-hex 2.0
--long-rebound-base-hex 3.8
--short-rebound-slope 0.10
--normal-rebound-slope 0.18
--long-rebound-slope 0.28
--short-rebound-sigma-hex 0.90
--normal-rebound-sigma-hex 1.10
--long-rebound-sigma-hex 1.35
--lateral-bias-hex 1.15
--centerline-deadband 0.12
--short-weak-prob 0.56
--short-same-prob 0.25
--long-weak-prob 0.22
--long-same-prob 0.47
```

Current model:

1. Sample a missed-shot location from real court hexes.
2. Sample a rebound target hex. In default mode this uses the old synthetic
   `short`/`normal`/`long` heuristic; with `--fitted-target-model-dir` it uses
   the MuJoCo-fitted catch table instead.
3. Place 5 offense and 5 defense players in plausible half-court positions.
4. Sample rebound winner from distance to the sampled target only. There is
   no explicit offense/defense team-side bonus, boxout term, inside-position
   bonus, player-skill bonus, or shooter penalty.

Parameter groups:

- Shot-location parameters (`shot_*`) only control synthetic selection of missed
  shot hexes when explicit `--shot-cells` are not provided. They are not the
  rebound physics model.
- Synthetic target parameters (`short_*`, `normal_*`, `long_*`,
  `lateral_bias_hex`, `centerline_deadband`, `*_weak_prob`, `*_same_prob`) are
  only used when the fitted MuJoCo table is not supplied.
- Winner parameters (`target_distance_weight`, `winner_temperature`) decide
  which already-placed player gets the sampled target. These still apply in
  fitted-table mode.

This is deliberately not integrated into training yet. The goal is to tune
aggregate distributions first, then port the stable vectorized math into the
JAX environment.
