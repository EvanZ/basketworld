# Direct JAX policy cross-play

`analytics/jax_policy_crossplay.py` measures the head-to-head strength of two
distinct trained JAX policies. The first checkpoint must have been trained with
`team_action_mode=autoregressive`; the second must have been trained with
`team_action_mode=independent` (or be a legacy checkpoint where that field is
absent and therefore means independent).

The script never changes either policy's decoder mode or action order. It loads
the saved policy spec and weights from each checkpoint and rejects aliases of
the same checkpoint, reversed/mislabeled modes, mismatched team sizes, and
native-evaluation incompatibilities.

## Default 1,000-game study

```bash
.env/bin/python analytics/jax_policy_crossplay.py \
  --autoregressive-checkpoint /path/to/ar/checkpoint \
  --independent-checkpoint /path/to/independent/checkpoint \
  --output-dir artifacts/policy_crossplay/ar-vs-independent-u2000 \
  --episodes 1000 \
  --action-mode deterministic \
  --seed 1
```

Run sampled actions separately by changing `--action-mode` to `sampled`. Do not
combine deterministic and sampled games in one inference statement.

The 1,000 games comprise 250 matched four-game scenario blocks. Each base
environment reset is evaluated with both starting-possession assignments and
both checkpoint-to-stable-team assignments. Confidence intervals bootstrap
these 250 blocks, rather than treating the 1,000 correlated games as independent.

`--episodes` must be a multiple of four, and `--batch-size` must be even. Both
checkpoint environment contracts are used by default and must agree. Explicit
environment changes require `--environment-overrides-json`; overrides are
recorded in provenance and never alter decoder mode or ordering.

## Output

The output directory must not already exist. The evaluator writes:

- `manifest.json`: checkpoint paths, metadata hashes and saved specs, source
  revision/dirty state, seed, action mode, environment overrides, and status;
- `games.jsonl`: policy-relative outcomes plus the raw native episode record;
- `summary.json`: wins/ties/losses, win score, score margin, points per
  possession, assist-to-turnover ratio, violations, spatial diagnostics,
  assignment splits, and matched-block 95% confidence intervals;
- `progress.json`: the latest completed-game estimates and available paired
  interval while the job runs;
- `native_ar_team_a.json` and `native_ar_team_b.json`: native aggregate
  diagnostics for each checkpoint assignment.

The console reports phase/liveness updates and refreshes the current aggregate
after every completed evaluation batch.
