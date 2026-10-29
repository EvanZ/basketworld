# JAX defensive feature probe

The offline defensive feature probe tests whether explicit defensive geometry
makes MCTS-selected actions easier to represent than the current policy
observation alone. It consumes completed artifacts from the JAX MCTS regret
study and does not run search, restore SB3, or alter PPO training.

## Inputs

The required input is a completed `root_action_comparison.json` with defensive
records and its adjacent replay artifacts:

- `sampled_states.json` for state IDs and provenance;
- `sampled_states.npz` for lossless `KernelState` arrays.

The probe validates state IDs, depth results, defensive action labels, token
observation dimensions, and environment provenance before training.

## Feature variants

All variants use the same small token-attention classifier, optimization
settings, state split, and training seeds.

1. `current_observation` uses the existing policy player tokens and globals.
2. `directional_geometry` adds normalized axial direction vectors from each
   defender to the ball handler and nearest offensive player, plus validity
   flags.
3. `threat_and_responsibility` additionally adds the nearest offensive
   player's expected shot value, whether the defender is closest to the ball
   handler, and whether the defender is closest to the offensive player with
   the highest expected shot value.

The environment implementation retains some legacy internal names for expected
shot value. Probe feature labels and output artifacts use `expected_shot_value`.

## Run the study

```bash
.env/bin/python analytics/jax_defensive_feature_probe.py \
  --mcts-results <study-dir>/root_action_comparison.json \
  --output-dir <new-probe-output-dir>
```

Existing output directories are rejected rather than overwritten. By default,
states are split 70%/15%/15% into train/validation/test groups, and three
initialization seeds are ensembled for the final held-out metrics. Data-size
learning curves use the first configured seed consistently at every fraction,
so changing the sample size is not confounded with ensembling.

When the soft-action diagnostic is enabled, the evaluated checkpoint path
recorded in the MCTS artifact must remain available. The probe restores it to
obtain the original policy's full legal action distribution for the held-out
comparison. Ordinary hard-label probe runs do not require checkpoint restore.

If completed searches over the same states are available at other budgets or
seeds, include them to identify stable joint-action labels:

```bash
.env/bin/python analytics/jax_defensive_feature_probe.py \
  --mcts-results <budget-128>/root_action_comparison.json \
  --stability-result <budget-256>/root_action_comparison.json \
  --stability-result <budget-512>/root_action_comparison.json \
  --output-dir <new-probe-output-dir>
```

Without at least one comparison artifact, stability metrics are marked
unavailable. A single MCTS budget is never silently treated as stable.

### Visit-weighted soft-action diagnostic

Supplying two `--stability-result` search-seed reruns, for three total searches
over the same roots, also enables a soft-action learnability test. For each
search and defender, the probe sums root joint-action visit counts by that
defender's action and normalizes the resulting seven-way marginal. It then
averages the three run-level marginals equally. This preserves uncertainty when
different search seeds choose different hard argmax actions.

The diagnostic trains one additional current-observation attention probe with
soft cross-entropy. It deliberately does not retest the explicit feature
variants, alter PPO, run MCTS, or change deployment. Held-out results cover:

- move versus no-op;
- no-op plus three directional move buckets;
- exact seven-way defensive actions;
- pairwise search-seed Jensen-Shannon divergence;
- target cross-entropy, KL divergence, probability overlap, teacher mass at the
  selected action, and teacher-mode agreement;
- paired exact-action cross-entropy differences against the restored evaluated
  policy and a training-set soft-frequency baseline.

The outcome-rollout gate passes only when the student's paired held-out
cross-entropy is reliably lower than both baselines (the complete 95% bootstrap
interval is below zero). Passing that gate justifies a separate paired rollout
test of whether student-selected actions close any MCTS regret. This offline
diagnostic itself does not claim an outcome improvement.

### Same-state search-seed stability

Before interpreting probe accuracy as a representation limit, repeat the
primary depth/budget search over the exact persisted roots with at least two
new top-level seeds. Keep checkpoints, environment, action modes, depth,
simulation budget, and sampled-state metadata fixed. Including the original
artifact, this gives at least three independent MCTS decisions per state.
Use the existing root-only evaluator command with the original
`--sample-metadata`, changing only `--seed` and `--output-dir` for each rerun.

Analyze the completed searches with:

```bash
.env/bin/python analytics/jax_mcts_seed_stability.py \
  --result <seed-1>/root_action_comparison.json \
  --result <seed-2>/root_action_comparison.json \
  --result <seed-3>/root_action_comparison.json \
  --output-dir <new-seed-stability-output-dir>
```

The immutable output reports state-bootstrapped per-defender pairwise action
agreement, full joint-action agreement, all-seed consensus, modal agreement,
move/no-op agreement, and selected-versus-runner-up visit and value margins.
The analyzer rejects protocol, checkpoint/environment provenance, state-order,
sample-label, or search-seed mismatches. It does not impose an arbitrary
stability cutoff.

## Output and interpretation

`defensive_feature_probe.json` contains:

- exact split membership and training seeds;
- feature definitions and model settings;
- per-seed training histories and data-size learning curves;
- state-bootstrapped held-out confidence intervals;
- exact per-player and five-player joint action accuracy;
- move/no-op, exact six-way direction, three-bucket direction, and top-k
  diagnostics;
- multiclass cross-entropy (reported as negative log likelihood),
  legal-action-conditioned macro one-vs-rest ROC-AUC and PR-AUC, and binary
  move-vs-no-op ROC-AUC and PR-AUC;
- confusion matrices;
- separate high-regret and, when available, stable-label results;
- paired bootstrap differences between feature variants;
- stable-label subsets derived from completed same-state search artifacts when
  supplied;
- the visit-weighted soft-action diagnostic when three same-state search runs
  are supplied;
- per-state ensemble probabilities and predictions;
- input hashes and source-code provenance.

Cross-entropy is the primary probability-quality metric because it evaluates
the full seven-way distribution and penalizes confident errors. Exact accuracy
checks whether the probe actually selects the MCTS action. AUC is supplemental:
it can detect improved action ranking before the argmax changes, but does not
measure calibration. Macro action AUC excludes decisions where the candidate
action was illegal, avoiding artificially easy negatives. PR-AUC is included
because the one-vs-rest action labels are imbalanced.

If the current observation predicts stable MCTS actions well, PPO credit or
coordinated exploration is the leading limitation. A reliable held-out gain
from directional or responsibility features supports a subsequent production
observation experiment. Failure on all variants is not evidence that the
features are useless unless the MCTS labels are themselves stable.
