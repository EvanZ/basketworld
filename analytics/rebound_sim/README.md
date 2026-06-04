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

Useful tuning flags:

```bash
--defense-rebound-bias 0.55
--boxout-bias 0.35
--target-distance-weight 1.10
--winner-temperature 0.75
```

Current model:

1. Sample a missed-shot location from real court hexes.
2. Sample rebound type: `short`, `normal`, or `long`.
3. Sample rebound region: `weak`, `same`, or `middle`.
4. Sample a rebound landing target hex.
5. Place 5 offense and 5 defense players in plausible half-court positions.
6. Sample rebound winner from distance, skill, inside position, boxout, and
   defensive rebounding bias.

This is deliberately not integrated into training yet. The goal is to tune
aggregate distributions first, then port the stable vectorized math into the
JAX environment.
