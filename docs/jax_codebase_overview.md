# JAX Codebase Overview

This document is a study guide for the JAX stack. It focuses on how the current SB3/Python/Torch training architecture was transformed into the JAX/Flax/Optax/Orbax path, and where to look when studying model architecture, training, inference, and evaluation.

## Mental Model

The old SB3 stack was organized around:

- Python Gym environment stepping
- SB3 `VecEnv` / `SubprocVecEnv`
- SB3 PPO rollout collection
- PyTorch policy/value/discriminator modules
- callback-driven logging, checkpointing, evaluation, and intent diagnostics

The JAX stack is organized around:

- a batched JAX environment kernel
- compiled rollout loops
- Flax model modules
- JAX-native PPO, selector, and discriminator updates
- explicit MLflow logging
- Orbax checkpoint state plus metadata artifacts
- JAX-native dev-runtime and evaluation paths

The most important architectural change is that the environment state is no longer advanced one Python env at a time. The JAX path represents the env as static constants plus batched state arrays, then advances that state inside compiled JAX functions.

## Package Map

- [basketworld_jax/env/minimal.py](/home/evanzamir/basketworld/basketworld_jax/env/minimal.py) defines the JAX-native environment kernel: reset, step, action masks, rewards, lane rules, shot/pass/turnover logic, start templates, and compact observations.

- [basketworld_jax/models/actor_critic.py](/home/evanzamir/basketworld/basketworld_jax/models/actor_critic.py) defines the Flax actor-critic model used for policy, value, attention, intent conditioning, pointer-targeted passing, and selector outputs.

- [basketworld_jax/train/types.py](/home/evanzamir/basketworld/basketworld_jax/train/types.py) defines the typed data structures passed through training: trajectory batches, PPO batches, selector batches, eval traces, rollout outputs, and summary containers.

- [basketworld_jax/train/runtime.py](/home/evanzamir/basketworld/basketworld_jax/train/runtime.py) contains the compiled runtime pieces: rollout runners, grouped frozen-opponent runners, PPO update runner, selector update runner, eval runners, and metric summarizers.

- [basketworld_jax/train/main.py](/home/evanzamir/basketworld/basketworld_jax/train/main.py) is the training orchestrator: CLI args, MLflow logging, checkpointing, opponent pool management, rollout/update loop, discriminator training, selector training, eval, and summaries.

- [basketworld_jax/intent/discriminator.py](/home/evanzamir/basketworld/basketworld_jax/intent/discriminator.py) defines the intent discriminator, intent bonus computation, discriminator update loop, holdout metrics, AUC/top-1 metrics, and sample dumping for t-SNE.

- [basketworld_jax/inference/policy.py](/home/evanzamir/basketworld/basketworld_jax/inference/policy.py) loads JAX checkpoints and exposes a backend-friendly inference wrapper for action probabilities, values, attention info, and selector/intent state.

- [basketworld_jax/eval/native.py](/home/evanzamir/basketworld/basketworld_jax/eval/native.py) is the fast JAX-native evaluation path. It avoids Python env stepping and computes UI stats directly from batched JAX rollouts.

- [basketworld_jax/checkpoints/checkpoint.py](/home/evanzamir/basketworld/basketworld_jax/checkpoints/checkpoint.py) handles Orbax checkpoint save/load plus metadata serialization for MLflow artifacts.

- [app/backend/jax_dev_runtime.py](/home/evanzamir/basketworld/app/backend/jax_dev_runtime.py) is the JAX-native dev app runtime. It keeps the canonical live game state as a JAX `KernelState`, steps it through the JAX env, and serializes it back into the existing frontend payload shape.

## SB3-To-JAX Mapping

- SB3 `VecEnv` maps to batched JAX `KernelState`.

- Python env `step()` maps to compiled JAX transition functions in [minimal.py](/home/evanzamir/basketworld/basketworld_jax/env/minimal.py).

- SB3 rollout buffers map to `TrajectoryBatch` and `PPOBatch` in [types.py](/home/evanzamir/basketworld/basketworld_jax/train/types.py).

- SB3 PPO policy/value modules map to `ActorCriticModule` in [actor_critic.py](/home/evanzamir/basketworld/basketworld_jax/models/actor_critic.py).

- SB3 callback-driven metrics map to explicit summary and logging code in [main.py](/home/evanzamir/basketworld/basketworld_jax/train/main.py) and [runtime.py](/home/evanzamir/basketworld/basketworld_jax/train/runtime.py).

- SB3 frozen-opponent wrappers map to grouped frozen-opponent rollout runners in [runtime.py](/home/evanzamir/basketworld/basketworld_jax/train/runtime.py).

- PyTorch checkpoints map to Orbax state plus JSON metadata in [checkpoint.py](/home/evanzamir/basketworld/basketworld_jax/checkpoints/checkpoint.py).

- SB3 eval mode maps to `run_native_jax_evaluation()` in [native.py](/home/evanzamir/basketworld/basketworld_jax/eval/native.py).

- SB3 play/intent tooling maps to JAX selector, discriminator, intent samples, and UI intent controls.

## Environment Kernel

The JAX env is deliberately compact. It does not try to wrap the existing Python env. Instead it ports the rollout-critical semantics into JAX arrays.

The main split is:

- static config: court constants, player counts, rule settings, skill distributions, action dimensions, template metadata, and other values that should not change during a compiled rollout

- dynamic state: positions, ball holder, shot clock, scores, done flags, lane counters, intent state, selector state, template state, and per-env random keys

This split matters because JAX can efficiently compile functions that operate on arrays with stable shapes. The old Python env had rich object state, which is useful for debugging but too expensive for high-throughput training.

## Model Architecture

The main model is `ActorCriticModule` in [actor_critic.py](/home/evanzamir/basketworld/basketworld_jax/models/actor_critic.py).

It supports two model families:

- `mlp`: a simple flat-observation baseline retained for speed and debugging

- `attention`: the current parity-oriented model using player tokens, global features, role flags, attention, intent embeddings, pointer-targeted passing, and selector outputs

The attention model has these major subcomponents:

- token encoder: converts per-player observation tokens into hidden embeddings

- global conditioning: appends global features and role information to token inputs

- attention block: applies self-attention over player tokens plus optional CLS tokens

- policy heads: separate offense and defense policy heads

- value heads: separate offense and defense value heads

- pointer-targeted action head: decomposes pass decisions into action-type logits plus teammate-target logits

- intent embedding: injects the active play/intent into the policy representation

- selector head: predicts play preferences and selector value from the current state

The current parity direction is the attention model with pointer-targeted passing, intent conditioning, and selector support.

## Training Runtime

The training runtime is split between orchestration and compiled computation.

The orchestrator is [main.py](/home/evanzamir/basketworld/basketworld_jax/train/main.py). It owns CLI config, MLflow logging, checkpointing, opponent pool state, schedule values, and the outer update loop.

The compiled work is built in [runtime.py](/home/evanzamir/basketworld/basketworld_jax/train/runtime.py). It owns rollout, PPO update, selector update, eval, and grouped frozen-opponent inference.

A normal training update is:

- run a compiled rollout for `kernel_batch_size * rollout_horizon` env states per role path

- select actions using the current actor-critic policy

- optionally infer frozen historical opponents through grouped opponent sampling

- collect rewards, dones, values, log-probs, masks, action stats, selector state, intent labels, lane stats, and debug metrics into a trajectory batch

- compute advantages and returns

- run one or more PPO update epochs over configured minibatches

- optionally update the selector

- optionally update the discriminator after warmup/ramp scheduling

- log metrics and periodically save checkpoints

Compared with SB3, this is less hidden. The JAX code makes rollout, PPO update, selector update, discriminator update, opponent sampling, checkpointing, and metric logging explicit.

## PPO Update

The PPO update is JIT-compiled and consumes a flattened `PPOBatch`.

Important concepts:

- rollout size is controlled by `kernel_batch_size * rollout_horizon`

- PPO batch size may include both offense and defense samples when both roles are trained

- `--ppo-minibatches` divides the PPO batch into shuffled minibatches

- `--policy-update-epochs` repeats PPO updates over the same rollout batch

- `approx_kl`, `clip_fraction`, entropy, value loss, policy loss, and end-to-end speed metrics are logged explicitly

This replaces SB3’s internal PPO optimizer loop.

## Opponent Sampling

The SB3 production path used historical or frozen opponent checkpoints. The JAX path implements the same idea without returning to Python-side env stepping.

The grouped opponent path:

- keeps an opponent pool from historical checkpoints

- groups envs by sampled opponent

- runs opponent inference inside compiled/batched rollout logic

- avoids one Python model call per env

This is necessary because the original JAX speedup only survives if self-play and action selection stay mostly inside the JAX rollout path.

## Intent System

The intent system has three related pieces.

- Policy conditioning: the actor-critic receives an active intent/play id and uses a learned embedding to condition action selection.

- Selector: the actor-critic has a selector head that chooses the next play at valid segment boundaries.

- Discriminator: a separate Flax model learns whether rollout samples are distinguishable by intent and provides metrics and optional intent reward signal.

The JAX discriminator intentionally follows the current production direction: offensive intent learning, not defensive intent learning. It uses the set-step architecture rather than porting the older GRU path.

Useful files:

- [intent/discriminator.py](/home/evanzamir/basketworld/basketworld_jax/intent/discriminator.py)

- [train/runtime.py](/home/evanzamir/basketworld/basketworld_jax/train/runtime.py)

- [analytics/jax_intent_sample_embed.py](/home/evanzamir/basketworld/analytics/jax_intent_sample_embed.py)

## Inference

Inference starts in [policy.py](/home/evanzamir/basketworld/basketworld_jax/inference/policy.py).

The inference wrapper:

- loads checkpoint metadata and Orbax state

- reconstructs the actor-critic spec

- runs JAX policy inference for the current observation/state

- returns action probabilities, value estimates, attention payloads, selector preferences, and intent metadata in backend-friendly form

The backend uses this through the existing adapter seam rather than calling model internals directly.

## Dev App Runtime

The dev app path for JAX checkpoints is [jax_dev_runtime.py](/home/evanzamir/basketworld/app/backend/jax_dev_runtime.py).

This runtime exists because using JAX inference with the old Python env transition path was not honest parity. A JAX checkpoint should be visualized in the same environment dynamics it was trained on.

The runtime:

- stores canonical JAX `KernelState`

- resets and steps through the JAX env kernel

- supports self-play, manual controls, templates, selector state, value outputs, and attention payloads

- serializes JAX state into the existing frontend payload shape

The Python env may still be constructed for compatibility helpers, but it is no longer the canonical stepping runtime for JAX checkpoints.

## Native Evaluation

The fast eval path is [native.py](/home/evanzamir/basketworld/basketworld_jax/eval/native.py).

It exists because stepping thousands of eval episodes through the UI/Python path would be unnecessarily slow.

Native eval:

- loads JAX checkpoint state

- runs many episodes through batched compiled JAX rollout

- computes per-player and team stats

- computes shot charts, pass/assist links, turnovers, rewards, lane violations, selector behavior, and intent stats

- feeds the Eval/Stats UI path with data shaped like the existing frontend expects

## Checkpoints

JAX checkpoints use Orbax for array state and JSON metadata for config/run fields.

The checkpoint layer lives in [checkpoint.py](/home/evanzamir/basketworld/basketworld_jax/checkpoints/checkpoint.py).

At a high level:

- model params and optimizer state are saved through Orbax

- metadata records model spec, env config, training config, selector settings, intent settings, template settings, and run identifiers

- MLflow stores the checkpoint artifact

- backend loading uses MLflow run id as the preferred entrypoint

This replaces SB3’s PyTorch policy serialization while keeping the MLflow-centered workflow.

## What Is Still Shared With The Old Stack

The JAX stack is separate, but it is not completely isolated.

Shared or reused pieces include:

- MLflow run id workflow

- frontend routes and payload shapes

- start template libraries

- some config naming conventions

- some Python env construction for static metadata or display compatibility

- analytics concepts such as t-SNE sample dumps

The intended direction is not to remove all shared concepts. The intended direction is to keep JAX training/eval/interactive stepping on the JAX-native environment path while preserving enough backend/frontend compatibility to avoid rewriting the whole app.

## Suggested Study Order

1. Read [env/minimal.py](/home/evanzamir/basketworld/basketworld_jax/env/minimal.py) to understand what the JAX environment state actually contains.

2. Read [models/actor_critic.py](/home/evanzamir/basketworld/basketworld_jax/models/actor_critic.py) to understand policy, value, attention, pointer passing, selector, and intent conditioning.

3. Read [train/types.py](/home/evanzamir/basketworld/basketworld_jax/train/types.py) so the data flowing through rollout and update is clear.

4. Read [train/runtime.py](/home/evanzamir/basketworld/basketworld_jax/train/runtime.py) for compiled rollout, grouped opponent sampling, PPO update, selector update, and eval mechanics.

5. Read [train/main.py](/home/evanzamir/basketworld/basketworld_jax/train/main.py) after the runtime files, because it is orchestration-heavy.

6. Read [intent/discriminator.py](/home/evanzamir/basketworld/basketworld_jax/intent/discriminator.py) once the rollout and model paths are clear.

7. Read [inference/policy.py](/home/evanzamir/basketworld/basketworld_jax/inference/policy.py), [jax_dev_runtime.py](/home/evanzamir/basketworld/app/backend/jax_dev_runtime.py), and [eval/native.py](/home/evanzamir/basketworld/basketworld_jax/eval/native.py) to understand deployment, UI behavior, and eval behavior.
