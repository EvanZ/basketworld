# JAX MCTS policy-regret evaluator

`analytics/jax_mcts_policy_regret.py` samples policy-play states and asks whether
a JAX-native MCTS planner can recover more possession value than the evaluated
policy's action. Both checkpoint paths are explicit. The evaluated checkpoint
controls stable Team A and the opponent checkpoint controls stable Team B;
their offense and defense roles change normally between possessions.

The evaluator reports the **root-only** comparison and, by default, the
**full-possession** comparison:

1. Run independently seeded policy episodes.
2. Uniformly reservoir-sample one eligible live decision per episode.
3. Run depth-1, depth-3, and depth-5 MCTS from the exact saved state.
4. Force either the policy action or planner action at the root, then return
   both teams to their fixed policies through possession end.
5. For full-possession mode, keep the selected root action and run a fresh
   primary-depth search at each subsequent meaningful evaluated-team decision.
6. Estimate gains with matched rollout seeds so policy/planner differences do
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

Defaults follow Issue #76: depths 1/3/5, depth 5 primary, 512 MCTS simulations
per action selection, and root plus full-possession planning. Pass
`--planning-mode root-only` to omit receding-horizon evaluation. Use
`--evaluated-action-mode sampled` and/or `--opponent-action-mode sampled` for a
separately labeled stochastic-policy study. Do not mix deterministic and
sampled results into one headline number.

An output directory must not already exist. Each study writes:

- `sampled_states.npz`: lossless `KernelState` arrays for exact replay;
- `sampled_states.json`: state context, all sampling seeds, exclusion counts,
  checkpoint/environment settings, and source provenance;
- `policy_regret_study.json`: per-depth root search diagnostics, full-possession
  replanning traces, paired returns and gains, offense/defense splits, exact
  action agreement, coarse action-family agreement, three-way movement-direction
  agreement (northward/lateral/southward), uncertainty, and aggregate runtime.
  Root-only runs instead write
  `root_action_comparison.json`.
- `regret_trajectory.json`: an atomically updated running mean, bootstrap 95%
  interval, and offense/defense split after every analyzed state.

The sampler excludes terminal states, non-live setup/inbound phases, and joint
decisions where every controlled player is forced to one action. It compiles
the episode batch as one JAX scan, performs uniform reservoir replacement on
device, and retains only one full state per episode. It alternates the opening
stable team across episodes and reports the observed offense and defense sample
counts separately. Checkpoint restore and the first JAX compile can take time;
the command prints those phases explicitly.

## Calibrate the search budget first

Run the agreed budgets over one persisted set of 50 sampled states:

```bash
.env/bin/python analytics/jax_mcts_policy_regret.py \
  --evaluated-checkpoint /path/to/evaluated/checkpoint \
  --opponent-checkpoint /path/to/opponent/checkpoint \
  --output-dir results/mcts-regret/calibration-name \
  --seed 1 \
  --calibration
```

This writes `budget_calibration.json`, including per-budget records, adjacent
budget comparisons, and a focused 512-versus-1,024 summary of selected-action
stability and paired regret deltas. The evaluator records the evidence but does
not impose an arbitrary automatic stability threshold. Retain the
512-simulation main-study budget only after inspecting that comparison.

Calibration searches only the primary depth (depth 5 by default), because the
budget decision is based on the primary planner. It also reuses identical
paired policy/action returns across budgets for each state. The main study
still reports depths 1, 3, and 5 separately.

To rerun a budget over the exact same states, pass the earlier metadata file.
An optional role filter makes offense-only or defense-only sweeps possible:

```bash
.env/bin/python analytics/jax_mcts_policy_regret.py \
  --evaluated-checkpoint /path/to/evaluated/checkpoint \
  --opponent-checkpoint /path/to/opponent/checkpoint \
  --output-dir results/mcts-regret/defense-budget-256 \
  --sample-metadata results/mcts-regret/main/sampled_states.json \
  --evaluated-role defense \
  --planning-mode root-only \
  --search-depths 5 \
  --primary-search-depth 5 \
  --simulations-per-action 256
```

Replay validates checkpoint, policy, action-mode, and environment provenance
before analysis. The filtered states are copied into the new immutable output
directory with their source metadata recorded.

`--print-calibration-command` remains available when separate scheduler jobs
are preferable. Those printed commands explicitly use root-only mode. Both
calibration paths isolate search-budget stability without multiplying the sweep
by every subsequent possession decision.

## Current boundary

Checkpoints with an enabled learned intent selector are rejected rather than
simulated with incorrect selector state; no-intent checkpoints are supported.
