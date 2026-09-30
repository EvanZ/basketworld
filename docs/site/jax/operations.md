# Training operations

The training entry point combines a compiled numerical core with Python-side
experiment management. This page covers local research workflows only.

## Representative commands

### Trainer scaffold

Benchmark model forward, rollout, and PPO update paths:

```bash
python -m basketworld_jax.train.main \
  --kernel-batch-size 256 \
  --rollout-horizon 64 \
  --policy-model attention \
  --action-head-mode pointer_targeted
```

### Short training run

```bash
python -m basketworld_jax.train.main \
  --run-train-loop \
  --num-updates 100 \
  --kernel-batch-size 256 \
  --rollout-horizon 64 \
  --policy-model attention \
  --action-head-mode pointer_targeted \
  --ppo-minibatches 8 \
  --eval-every-updates 25 \
  --checkpoint-dir checkpoints/local_jax \
  --checkpoint-every-updates 25
```

For a large experiment with start templates, fitted rebounds, intent learning,
scheduled shaping, grouped opponents, and MLflow, use
`scripts/run_jax_5v5_rebound_continuation.sh` as a concrete configuration
reference. Read the flags before running it: it is a long research job, not a
quickstart.

### Continuous half-court training

`scripts/run_jax_5v5_halfcourt_multi_possession.sh` starts the agreed fresh
25-possession half-court experiment. It intentionally does not load a
pretrained policy, continuation checkpoint, or starting templates. The two
fixed-team cohorts feed one shared PPO update, and unfinished games continue
across rollout and optimizer-update boundaries.

Multi-possession attention checkpoints created before the stable-team token
canonicalization fix in issue #37 must be retrained. Those checkpoints kept
player tokens in stable Team A/Team B order while the action head selected a
token half from the current offense/defense role. Team B offensive possessions
therefore mapped logits from the wrong roster into Team B action slots. The
checkpoint file format remains loadable, but its learned behavior and symmetric
self-play results are not trustworthy for continued training or comparison.

The launcher enables the established offense-only latent-intent stack: eight
intents, attention embeddings, the learned selector, and the diversity
objective. The team currently on offense receives its private intent context;
defensive intent learning remains disabled. Its 100-update warmup and
400-update ramp fit the launcher's 500-update experiment rather than copying
the much longer continuation schedule.

In this mode, rollout cuts bootstrap the value function; only true game endings
cut the bootstrap. Historical opponents and their sampled action mode are
pinned per game row until that game ends. Do not combine multi-possession mode
with `--single-episode-rollouts` or `--ppo-completed-episodes-only`; validation
rejects both because they would discard or respawn unfinished games.

### Inspecting and evaluating continuous games

The JAX development board shows stable user/AI scores, the current offense,
completed and remaining possessions, shot clock, inbound countdown, and the
clearance requirement. An inbounder is rendered beyond the baseline; after a
successful inbound, the board marks its legal re-entry cells. Start-template
controls are disabled in this mode because the normal opening spawn occurs
only once. Both stable teams receive neutral, distinct on-court locations;
the 50/50 jump-ball result then assigns the opening ball-handler without
rearranging either team.

For top-of-key check restarts, `--check-setup-steps N` optionally inserts N
movement-only dead-ball ticks before the existing pickup countdown. Both teams
may reposition during setup, but neither may enter the protected ball cell;
the shot clock, lane clocks, and `--check-deadline-steps` countdown remain
paused. The default is `0`, which preserves the immediate-check behavior.
`CHECK_SETUP_STEPS` exposes the same setting in the multi-possession launcher.
Training and native evaluation report setup opportunities, executed setup
steps, and mean executed setup duration separately from live-possession pace
and spatial diagnostics.

Native evaluation pairs reset seeds and swaps the Team A/Team B starter within
each pair while preserving the same neutral opening positions. It reports game W/L/T, scores, margins,
possessions and points per possession only for completed games. Horizon cutoffs
are reported separately as incomplete games—not as ties. Clearance events/time,
turnovers before clearance, inbound outcomes, rebound totals, and
`mean_live_steps_per_completed_possession` remain separate diagnostics. The
pace metric counts the terminal live action but excludes dead-ball inbound
ticks, so mechanically correct transitions are not mistaken for a learned
clearing strategy.

## MLflow

`--log-mlflow` starts a run through the repository's MLflow configuration. The
trainer logs:

- CLI and resolved environment parameters;
- policy and trainer specifications;
- schedule values;
- PPO, value, reward, event, opponent, selector, and discriminator metrics;
- evaluation summaries;
- checkpoint and intent-sample artifacts.

`--mlflow-metric-profile core` keeps the lower-volume metric set.
`full` publishes every scalar retained by the trainer.

MLflow is optional for local training. Without it, metrics remain in console
and returned summaries, and checkpoints require a local checkpoint directory.

## Checkpoints

JAX checkpoints are directories with:

- Orbax-managed array state under a state subdirectory;
- JSON metadata describing reconstruction and experiment state.

The payload includes:

- update index;
- trainer, policy, frozen structural, and environment config;
- actor-critic parameters and optimizer state;
- optional selector optimizer state;
- current training and evaluation states;
- PRNG key;
- recent evaluation traces and metrics;
- opponent information;
- optional opponent candidate parameters, per-row assignments and action modes,
  and opponent-pool RNG state for continuous games;
- optional offense/defense discriminator state;
- optional play-name metadata.

The trainer writes numbered checkpoints and a latest checkpoint. The final
update is saved whenever checkpoint publication is enabled.

## Checkpoint cadence

`--checkpoint-schedule fixed` uses a constant modulo interval.

`--checkpoint-schedule log` starts with
`--checkpoint-log-initial-updates`, grows the interval over
`--checkpoint-log-ramp-updates`, and caps it at
`--checkpoint-every-updates`. This captures early learning changes densely
without maintaining that frequency for a long run.

## Resume and continuation

Resume a local checkpoint with:

```bash
python -m basketworld_jax.train.main \
  --run-train-loop \
  --resume-checkpoint checkpoints/local_jax/jax_checkpoint_latest \
  --num-updates 200 \
  ...same structural and environment arguments...
```

The target `num_updates` is the total update index, not the number of
additional updates. The trainer validates policy and environment compatibility
before restoring.

`--continue-run-id` resolves a checkpoint artifact from an MLflow run and
starts a continuation workflow. It can also seed the new opponent pool with
recent checkpoints from that run. Continuation resets transient batched
environment state and can reset auxiliary discriminator state while preserving
the primary model and schedule metadata needed for a coherent new run.

## Historical opponent pool

A frozen opponent can come from:

- `--frozen-opponent-checkpoint`;
- `--frozen-opponent-run-id` plus an optional artifact hint;
- compatible checkpoints saved during the current run;
- compatible checkpoints loaded from a continuation run.

The pool maintains a bounded recent history. Sampling uses recency-biased
geometric selection plus an exploration probability.

With `--grouped-opponent-sampling`, several checkpoint parameter trees are
stacked and assigned to contiguous groups of environment rows. Opponent
inference stays batched and compiled instead of issuing one Python model call
per environment.

The probability of deterministic argmax opponent actions can be constant or
linearly scheduled. The sampled deterministic/stochastic mode is held for the
life of each episode row.

## Evaluation

Training supports:

- periodic compiled role evaluation through `--eval-every-updates`;
- fixed-seed same-policy argmax-versus-argmax deploy evaluation through
  `--eval-deploy-every-updates`;
- a maximum number of serialized trajectory examples for summaries.

`basketworld_jax/eval/native.py` provides a separate high-throughput
checkpoint evaluation path with per-player/team aggregates, shot and pass
diagnostics, turnovers, rebounds, values, and intent metrics. It advances the
same JAX environment semantics used by training.

## Operational checks

Before a long run:

1. run the one-update command from [Getting started](../getting-started.md);
2. inspect `jax.devices()`;
3. ensure PPO sample count is divisible by minibatches;
4. validate rebound artifact geometry if rebounds are enabled;
5. confirm checkpoint output or MLflow artifact storage;
6. budget for a new compilation when shape-bearing settings change;
7. inspect early rollout, loss, entropy, reward-component, and opponent
   metrics before scaling the run.

### Multi-possession rollout performance check

Use the trainer scaffold to compare environment/compiler changes without
creating checkpoints, MLflow runs, or policy-training artifacts. Keep the
hardware idle, exclude compilation with warmup iterations, and compare the
same command before and after the change:

```bash
python -m basketworld_jax.train.main \
  --enable-multi-possession \
  --multi-possession-limit 25 \
  --multi-possession-reward-mode scoring_events \
  --multi-possession-use-inbounds \
  --made-basket-restart-mode check \
  --inbound-deadline-steps 5 \
  --check-deadline-steps 5 \
  --check-setup-steps 5 \
  --players 5 \
  --court-rows 9 \
  --court-cols 8 \
  --shot-clock 24 \
  --min-shot-clock 24 \
  --illegal-defense-enabled true \
  --offensive-three-seconds true \
  --enable-pass-gating true \
  --enable-rebounds \
  --rebound-table-model-dir analytics/rebound_physics/outputs/dataset_9x8/fitted_catch_model \
  --kernel-batch-size 512 \
  --rollout-horizon 64 \
  --policy-update-epochs 1 \
  --ppo-minibatches 16 \
  --policy-model attention \
  --action-head-mode pointer_targeted \
  --intent-embedding-enabled \
  --intent-embedding-dim 16 \
  --enable-intent-learning true \
  --num-intents 8 \
  --intent-commitment-steps 8 \
  --warmup-iters 2 \
  --benchmark-iters 10 \
  --no-progress
```

The issue #39 reference on the local CPU backend used the same 512-by-64
rollout shape. Before sharing movement resolution between check setup and the
ordinary check phase, the three-iteration baseline produced 5,321 rollout
states/sec with 6.16 seconds mean rollout latency. The first identical
three-iteration optimized run produced 6,339 states/sec with 5.17 seconds
latency; a longer ten-iteration confirmation produced 6,058 states/sec with
5.41 seconds latency. This is a repeatable rollout improvement of roughly
14-19%, depending on the comparison run.

Performance work for this path must pass the exact parity regression tests.
In particular, identical states, actions, and PRNG keys must preserve the full
environment output tree, not merely aggregate scores or approximate metrics.
