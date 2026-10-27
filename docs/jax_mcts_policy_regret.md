# JAX MCTS policy-regret evaluator

`analytics/jax_mcts_policy_regret.py` samples policy-play states and asks whether
a JAX-native MCTS planner can recover more possession value than the evaluated
policy's action. Both checkpoint paths are explicit. The evaluated checkpoint
controls stable Team A and the opponent checkpoint controls stable Team B;
their offense and defense roles change normally between possessions.

The first implementation mode is **root-only**:

1. Run independently seeded policy episodes.
2. Uniformly reservoir-sample one eligible live decision per episode.
3. Run depth-1, depth-3, and depth-5 MCTS from the exact saved state.
4. Force either the policy action or planner action at the root, then return
   both teams to their fixed policies through possession end.
5. Estimate regret with matched rollout seeds so policy/planner differences do
   not include avoidable independent-rollout noise.

The learned value head is not used as the return target. Search transitions and
leaf continuations use actual Team A minus Team B point changes through the
possession boundary.

## Run a study

From the repository root:

```bash
.env/bin/python analytics/jax_mcts_policy_regret.py \
  --evaluated-checkpoint /path/to/evaluated/checkpoint \
  --opponent-checkpoint /path/to/opponent/checkpoint \
  --output-dir results/mcts-regret/study-name \
  --episodes 1000 \
  --seed 1
```

Defaults follow Issue #76: depths 1/3/5, depth 5 primary, and 512 MCTS
simulations per action selection. Use `--evaluated-action-mode sampled` and/or
`--opponent-action-mode sampled` for a separately labeled stochastic-policy
study. Do not mix deterministic and sampled results into one headline number.

An output directory must not already exist. Each study writes:

- `sampled_states.npz`: lossless `KernelState` arrays for exact replay;
- `sampled_states.json`: state context, all sampling seeds, exclusion counts,
  checkpoint/environment settings, and source provenance;
- `root_action_comparison.json`: per-depth search diagnostics, paired returns,
  regret, offense/defense splits, uncertainty, and aggregate runtime.

The sampler excludes terminal states, non-live setup/inbound phases, and joint
decisions where every controlled player is forced to one action. It alternates
the opening stable team across episodes and reports the observed offense and
defense sample counts separately.

## Calibrate the search budget first

Print the agreed commands for the same 50 episode seeds at 64, 128, 256, 512,
and 1,024 simulations:

```bash
.env/bin/python analytics/jax_mcts_policy_regret.py \
  --evaluated-checkpoint /path/to/evaluated/checkpoint \
  --opponent-checkpoint /path/to/opponent/checkpoint \
  --output-dir results/mcts-regret/calibration-name \
  --seed 1 \
  --print-calibration-command
```

Retain the 512-simulation main-study budget only after comparing its selected
actions and regret estimates with the 1,024-simulation artifacts.

## Current boundary

The command currently implements the root-only comparison. Full-possession
receding-horizon control, in which depth-5 MCTS replans at every evaluated-team
decision, is the next Issue #76 implementation slice. Checkpoints with an
enabled learned intent selector are rejected rather than simulated with
incorrect selector state; no-intent checkpoints are supported.
