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

## Output and interpretation

`defensive_feature_probe.json` contains:

- exact split membership and training seeds;
- feature definitions and model settings;
- per-seed training histories and data-size learning curves;
- state-bootstrapped held-out confidence intervals;
- exact per-player and five-player joint action accuracy;
- move/no-op, exact six-way direction, three-bucket direction, top-k, negative
  log likelihood, and confusion-matrix diagnostics;
- separate high-regret and, when available, stable-label results;
- paired bootstrap differences between feature variants;
- per-state ensemble probabilities and predictions;
- input hashes and source-code provenance.

If the current observation predicts stable MCTS actions well, PPO credit or
coordinated exploration is the leading limitation. A reliable held-out gain
from directional or responsibility features supports a subsequent production
observation experiment. Failure on all variants is not evidence that the
features are useless unless the MCTS labels are themselves stable.
