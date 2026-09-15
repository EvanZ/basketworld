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
