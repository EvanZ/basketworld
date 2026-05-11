# JAX / Flax / Optax Full Migration Plan

## Goal

Replace the current rollout-critical Torch/SB3 stack with a JAX-native training and inference stack that is fast enough, operationally usable, and incrementally extensible back toward current BasketWorld capabilities.

This plan assumes the reduced JAX path has already passed proof-of-value on throughput.

For a study-oriented overview of the current JAX package structure and how the SB3 architecture maps to the JAX implementation, see [jax_codebase_overview.md](/home/evanzamir/basketworld/docs/jax_codebase_overview.md).

## What Has Already Been Proven

The benchmark phase established:

- a JAX environment rewrite only helps if the rollout loop stays mostly on device
- small boundary-preserving swaps into the current SB3/Torch stack do not preserve the speedup
- the reduced JAX-native rollout + PPO path is fast enough to justify migration work

What is **not** proven yet:

- full feature parity with the current architecture
- long-run learning quality
- deployment readiness

So this is no longer a benchmark plan. It is a migration plan.

## Core Migration Decision

Use a **parallel-stack migration**, not an in-place rewrite.

That means:

- keep the current Torch/SB3 stack intact during migration
- build a new JAX-native stack alongside it
- cut over only when the new stack is operational and behaviorally credible

This matters because earlier experiments showed that mixing JAX env code back into the current Python/Torch/self-play path destroys most of the speed benefit.

## Target Stack

The intended target stack is:

- `JAX` for env state, rollout, GAE, sampling, metrics, and trainer loop
- `Flax Linen` for model definition
- `Optax` for optimization
- `MLflow` for experiment logging
- `Orbax` for checkpointing

The current JAX prototype already uses JAX and Optax. Flax and checkpointing should be the next formalization steps.

## Repository Strategy

Current exploratory JAX code lives in:

- [benchmarks/jax_kernel.py](/home/evanzamir/basketworld/benchmarks/jax_kernel.py)
- [benchmarks/jax_phase_a_policy.py](/home/evanzamir/basketworld/benchmarks/jax_phase_a_policy.py)
- [benchmarks/jax_phase_a_optim.py](/home/evanzamir/basketworld/benchmarks/jax_phase_a_optim.py)
- [basketworld_jax/train/main.py](/home/evanzamir/basketworld/basketworld_jax/train/main.py)

The benchmark files are still useful for older exploratory slices, but the reduced trainer itself now belongs under `basketworld_jax`, not under `benchmarks/`.

The long-term JAX package should live in a new parallel package, for example:

- `basketworld_jax/env/`
- `basketworld_jax/models/`
- `basketworld_jax/rollout/`
- `basketworld_jax/train/`
- `basketworld_jax/checkpoints/`
- `basketworld_jax/inference/`
- `basketworld_jax/config/`

Rule:

- prototype code starts in `benchmarks/`
- once a piece is proven necessary and stable, move it into `basketworld_jax/`

## Non-Goals

This migration should **not** start by:

- replacing the existing Torch/SB3 code path immediately
- chasing full current feature parity from day one
- porting every current model or callback before the reduced JAX path is operational
- threading JAX through the current self-play wrapper and expecting the speedup to survive

## Terminology

Do not use migration-stage labels like "Phase 1 / 2 / 3" in ongoing planning.

Use:

- `Foundation` for the package/trainer/checkpointing baseline
- `Learnability` for training-quality validation
- `Dev App Integration` for backend/frontend checkpoint loading and interactive testing
- `Representation Parity`, `Self-Play Parity`, `Advanced Features`, and `Cutover` for later capability milestones

Important distinction:

- migration milestone names should stay capability-based, not implementation-specific

## Migration Principle

The migration should expand capability in this order:

1. foundation
2. learnability
3. dev-app integration
4. representation parity
5. self-play parity
6. advanced feature parity
7. cutover

That order is important. It keeps the project moving through real milestones instead of disappearing into full-parity work too early.

## Foundation

Purpose:

- turn the current prototype into a usable trainer, not just a benchmark

Scope:

- keep the current reduced pointer-targeted JAX semantics
- keep the flat observation path
- keep the MLP actor-critic
- keep the legal-random opponent
- keep the reduced rule scope already used by the JAX trainer

Primary tasks:

- create `basketworld_jax/`
- move stable pieces out of `benchmarks/` into package modules
- replace the hand-rolled MLP with Flax Linen modules
- keep Optax as the optimizer layer
- add checkpoint save/load
- add resume support
- standardize config objects and run metadata
- standardize MLflow logging
- keep deterministic eval trajectory dumping

Exit criteria:

- train
- resume
- evaluate
- checkpoint
- reproduce a run from config

At the end of this milestone, the reduced JAX trainer should be a real training system.

Current status:

- `basketworld_jax/` package skeleton exists
- stable reduced actor-critic and optimizer code has moved into:
  - [actor_critic.py](/home/evanzamir/basketworld/basketworld_jax/models/actor_critic.py)
  - [adam.py](/home/evanzamir/basketworld/basketworld_jax/optim/adam.py)
- the reduced actor-critic now uses Flax Linen under the existing helper API
- stable trainer datatypes and PPO batch / GAE helpers have moved into:
  - [types.py](/home/evanzamir/basketworld/basketworld_jax/train/types.py)
- rollout, eval, PPO update, and benchmark/runtime helpers now exist in:
  - [runtime.py](/home/evanzamir/basketworld/basketworld_jax/train/runtime.py)
- the reduced env subset needed by the current JAX trainer now exists in:
  - [minimal.py](/home/evanzamir/basketworld/basketworld_jax/env/minimal.py)
- the canonical reduced trainer entrypoint now exists in:
  - [main.py](/home/evanzamir/basketworld/basketworld_jax/train/main.py)
- checkpointing is live through:
  - [checkpoint.py](/home/evanzamir/basketworld/basketworld_jax/checkpoints/checkpoint.py)
- checkpoint state now uses Orbax, with sidecar metadata for run/config/history fields
- the reduced train loop now supports:
  - periodic checkpoint save
  - final checkpoint save
  - resume from checkpoint
- checkpoint validation currently enforces rollout-shape and PPO-config compatibility while allowing `num_updates` to increase on resume
- legacy pickle checkpoints remain loadable for transition purposes

Foundation is now considered complete.

Non-blocking cleanup that can happen later:

- remove leftover benchmark-era compatibility shims once they are no longer needed
- standardize run/config objects beyond the current argparse path

## Learnability

Purpose:

- establish that the reduced JAX system is not only fast but useful

Primary tasks:

- run longer training jobs
- tune PPO update behavior
- improve eval metrics
- inspect replay traces
- add best-checkpoint selection

Metrics to care about:

- end-to-end env-steps/sec
- return
- score
- shot / pass / turnover behavior
- stability across seeds

Exit criteria:

- stable long-run throughput
- credible behavioral learning signal
- repeatable training outcomes

This is an active workstream. If learnability fails, do not move to parity work yet. Fix training quality first.

## Dev App Integration

Purpose:

- prove that a trained JAX model can actually be used outside the trainer

Primary tasks:

- create a JAX checkpoint loader
- build a backend inference adapter exposing `predict(...)`
- support masked action selection in inference
- keep the model server-side
- test it through the backend first, not the browser

Likely package area:

- `basketworld_jax/inference/`

Exit criteria:

- load trained JAX checkpoint
- run backend inference
- drive the web app backend against a simple opponent

This is the first real deployment milestone.

Initial dev-app scope for JAX models:

- keep player controls
- keep self-play
- keep the Observation panel
- keep Eval
- add backend capability flags so unsupported panels can be disabled cleanly
- disable Playbook, MCTS, and Attention panels for JAX models at first
- disable the current Environment and Training tabs for JAX models if needed
- add a dedicated JAX tab showing JAX-specific env/training/checkpoint metadata

Pragmatic rule:

- the frontend should branch on backend-reported model capabilities, not on checkpoint naming conventions or ad hoc heuristics

Current backend status:

- the backend now has a unified inference adapter seam for both SB3 and JAX models
- the current JAX trainer logs Orbax checkpoints to MLflow artifacts
- the current JAX path can now be loaded in the dev app from MLflow `run_id`
- `/api/init_game` can now resolve a JAX checkpoint artifact from MLflow and build the reduced env directly from checkpoint metadata
- backend state payloads now expose:
  - `model_backend`
  - `model_capabilities`
  - `model_metadata`
- the dev frontend can now initialize JAX models through the same MLflow `run_id` flow used by SB3
- the dev controls tabs now capability-gate unsupported panels and expose a JAX metadata tab
- self-play and interactive controls now run without app-level errors against the JAX model
- the remaining work here is feature hardening, not basic load/run viability

Dev App Integration is now considered complete for the current reduced JAX stack.

Important correction:

- the current dev app uses JAX checkpoint inference, but it still steps the Python `HexagonBasketballEnv`
- this means interactive self-play and visual testing are not yet running on the exact JAX-native transition system used during training
- that bridge was useful for early checkpoint loading and UI debugging, but it is now a parity risk
- achieving honest deployment parity requires a JAX-native UI/runtime stepping path

## JAX-Native Dev Runtime Parity

Purpose:

- make the dev app step the same JAX environment state/transition logic used by JAX training and native eval
- remove the current Python-env bridge for JAX checkpoints during interactive play, self-play, template resets, and visual debugging
- ensure visual testing is evaluating the model in the environment it was trained on

Priority:

- this is the next top-priority implementation slice before adding more advanced feature parity
- further UI parity work should assume this runtime path exists, or explicitly state when it is still using the Python bridge

Current risk:

- JAX inference currently snapshots a live Python `HexagonBasketballEnv` and runs the JAX actor-critic against that snapshot
- actions are then applied through `game_state.env.step(...)`, which is the Python production env transition path
- if the Python env and `basketworld_jax.env.minimal` diverge in reset logic, rule handling, reward calculation, action masks, intent/template handling, or terminal semantics, the UI can make a checkpoint look better or worse than it is under the actual training environment

Target architecture:

- introduce a JAX dev-runtime session object for JAX checkpoints
- store canonical JAX `KernelState` plus static env config in backend session state
- run reset/step through `basketworld_jax.env` functions, not through `HexagonBasketballEnv.step`
- run policy inference from the same JAX state/static pair, avoiding Python-env snapshot reconstruction
- serialize JAX state into the existing frontend game-state payload shape so `GameBoard`, Policy, Eval, Stats, Attention, and template controls can continue to work
- keep the Python env path only for SB3 checkpoints and for temporary fallback diagnostics

Implementation milestones:

1. Add a backend runtime abstraction.
   - define a small interface for `reset`, `step`, `render_state_payload`, `policy_outputs`, `state_values`, `attention_payload`, and `selector_preferences`
   - implement adapters for the existing Python env runtime and the new JAX runtime
   - route JAX checkpoints to the JAX runtime based on `model_backend == "jax"`

2. Build a JAX game-state serializer.
   - convert `KernelState` and static config into the existing frontend fields:
     `positions`, `ball_holder`, `shot_clock`, `offense_ids`, `defense_ids`, `scores`, `done`, `action_mask`, skills, lane counters, intent/play metadata, template metadata, and policy metadata
   - preserve raw display units such as shot-clock seconds while keeping normalized values only in observations/tokens
   - include `last_action_results` parity fields needed by the board, moves table, stats, and eval diagnostics

3. Add JAX-native interactive step.
   - accept the same UI action override payloads
   - combine manual/user actions with JAX policy actions in JAX-native action format
   - support deterministic/stochastic policy choices, pointer-targeted passing, selector state, and frozen/opponent policy inference
   - update replay buffers, rewards, episode states, and frontend response payloads without calling the Python env

4. Add JAX-native reset and template reset.
   - support normal reset, New Game reset, self-play reset, selected start template, jitter, and mirror
   - preserve selected start-template UI state across reset
   - ensure eval-time template behavior and interactive template behavior use the same JAX resolver

5. Port visual diagnostics that currently depend on Python env helpers.
   - shot probability and shot type
   - pass steal probabilities / pointer target metadata
   - lane violation counters and rewards
   - value-function overlay
   - attention/token payloads
   - selector/intent preference payloads

6. Add parity tests.
   - golden reset snapshots for Python bridge vs JAX serializer where semantics intentionally overlap
   - action-mask parity for supported JAX rules
   - one-step transition tests for shoot, pass, turnover, lane violations, and terminal states
   - template reset determinism tests
   - backend route tests proving JAX checkpoints do not call `HexagonBasketballEnv.step`

Current implementation status:

- backend session state now has an explicit `jax_runtime` slot
- JAX checkpoint initialization now creates a `JaxDevRuntime` session instead of treating the Python env as the canonical stepping runtime
- `/api/step`, `/api/start_self_play`, `/api/reset_turn_state`, and state serialization dispatch to the JAX runtime when a JAX checkpoint is loaded
- `JaxDevRuntime` stores canonical JAX `KernelState` plus static env config, runs reset/step through `basketworld_jax.env.minimal`, and serializes back into the existing frontend payload shape
- the Python env is still constructed for static config/display helper compatibility, but JAX interactive stepping no longer calls `HexagonBasketballEnv.step`
- JAX self-play now supports selector multiselect reselection at commitment-timeout and completed-pass boundaries when the checkpoint/run metadata enables it
- the first parity tests now prove state serialization dispatch, route dispatch, and a direct JAX runtime step that fails if the Python env `step()` is called

Remaining runtime hardening:

- replace any remaining Python-env helper dependence with explicit JAX serializers where practical
- harden selector parity with real-checkpoint UI smoke tests and any missing diagnostics around selector segment starts
- harden start-template reset parity and determinism under the JAX resolver
- add focused one-step parity tests for passes, shots, turnovers, lane violations, and terminal states
- exercise the dev app manually with real checkpoints, including templates, value overlay, attention, selector preferences, and replay

Exit criteria:

- loading a JAX checkpoint in the dev app creates a JAX runtime session, not a Python env session
- `New Game`, self-play, manual controls, templates, value overlay, attention, selector preferences, and replay state all work through JAX-native reset/step
- the frontend payload remains stable enough that the board and existing tabs do not need a full rewrite
- route tests prove that JAX interactive stepping does not use the SB3/Python-env transition path
- any remaining fallback to Python env is explicit, logged, and disabled by default for JAX checkpoints

## Representation Parity

Purpose:

- bring back stronger model structure without breaking the JAX-native rollout path

Primary tasks:

- keep the flat MLP policy/value network available as a stable baseline
- add set-observation support
- add a Flax set-attention encoder
- add pointer-targeted action head support under the JAX-native sampling path

Important rule:

- do not reintroduce Python-side action repair or host-side rollout assembly

Exit criteria:

- JAX-native set-observation training works
- pointer-targeted action semantics work on device

This is the point where the reduced stack starts to resemble the current model class more closely.

Current status:

- the trainer now supports `--policy-model attention` alongside the default `--policy-model mlp`
- attention mode uses packed JAX-native set observations: player tokens plus global features plus role flag
- the token layout mirrors the production set-observation wrapper: 15 per-player features and 4 global features
- the Flax actor-critic now has a shared token MLP, learned CLS tokens, self-attention, role-selected player action heads, and role-selected value heads
- attention mode now supports configurable post-attention PI/VF token-head MLPs through `--attention-pi-head-hidden-dims`, `--attention-vf-head-hidden-dims`, and `--attention-head-activation`
- the checkpointed dev launch is set to the production-style parity cost: 64-dim attention, 4 heads, 2 CLS tokens, 4x64 ReLU PI head, and 4x64 ReLU VF head
- rollout, PPO update, checkpointing, inference loading, and native eval route through the checkpoint `policy_spec["model_type"]`
- attention mode now supports a JAX-native pointer-targeted action head through `--action-head-mode pointer_targeted`
- the pointer-targeted head factorizes pass decisions into non-pass action type vs `PASS`, then learned teammate slot logits, while still emitting the existing final action ids for rollout, PPO, eval, and UI compatibility

## Environment Parity

Purpose:

- restore production-relevant rules while keeping the JAX rollout loop compiled

Current status:

- JAX reset/step now supports active offensive three-seconds counters and turnovers
- JAX reset/step now supports active illegal-defense lane counters, guard-distance reset behavior, defensive lane violations, technical point assignment, and violation rewards
- lane config is now logged with JAX run metadata:
  - `illegal_defense_enabled`
  - `offensive_three_seconds`
  - `three_second_lane_width`
  - `three_second_lane_height`
  - `three_second_max_steps`
  - `violation_reward`
- the checkpointed dev launch now enables both lane-rule families for more honest parity/performance measurement

Remaining environment parity gaps:

- start-template resets/curricula
- phi shaping
- advanced intent/template hooks
- richer Python-env `action_results` diagnostics

## GPU Viability Test

Purpose:

- determine whether attention training is viable when the PPO update runs on GPU instead of CPU

Current finding:

- attention rollout is not the main bottleneck
- attention PPO update/backprop over the full rollout batch is the dominant CPU cost
- with the current full-batch PPO update, each update can contain roughly `2 * kernel_batch_size * rollout_horizon` samples
- for `kernel_batch_size=4096` and `rollout_horizon=128`, that is `1,048,576` samples before multiplying by `policy_update_epochs`

Expected GPU behavior:

- no JAX code change should be required for a single-GPU test if CUDA-enabled JAX is installed
- JAX should choose the GPU backend automatically when the GPU is visible and the installed `jaxlib` supports CUDA
- the test should verify `jax.default_backend()` and `jax.devices()` before training
- the important comparison metrics are:
  - `rollout_latency_ms`
  - `update_latency_ms`
  - `end_to_end_steps_per_sec`
- a successful GPU test should show `update_latency_ms` dropping materially for attention runs

Setup notes:

- install a CUDA-enabled JAX wheel in the training environment, for example `jax[cuda13]` or `jax[cuda12]`
- use `nvidia-smi` to verify the EC2 GPU and driver
- use a single GPU first; the current trainer is not designed to automatically shard one run across multiple GPUs

Decision rule:

- if GPU reduces attention update latency enough to restore acceptable end-to-end throughput, continue with attention parity work
- if GPU does not materially improve the update bottleneck, revisit PPO minibatching, smaller attention dimensions, or delayed attention adoption

## PPO Minibatching

Purpose:

- reduce the cost and memory pressure of attention PPO updates without changing rollout semantics

Current implementation:

- the trainer now supports `--ppo-minibatches`
- `--ppo-minibatches 1` preserves the original full-batch PPO update
- values greater than 1 shuffle the PPO batch with a JAX PRNG key and scan over minibatches inside the compiled update path
- minibatch count must evenly divide the compiled PPO batch size
- the effective PPO batch size is:
  - train loop: `2 * kernel_batch_size * rollout_horizon`
  - scaffold: `kernel_batch_size * rollout_horizon`

Why this matters:

- full-batch PPO was acceptable for the MLP policy
- attention made the full-batch backward pass the dominant CPU cost
- standard PPO usually trains from shuffled minibatches rather than one giant full-batch gradient

Suggested starting points:

- CPU attention smoke runs: `--ppo-minibatches 16`
- larger batches per minibatch: `8`
- smaller batches per minibatch: `32` or `64`
- keep `--policy-update-epochs 1` initially while measuring speed and learning behavior

Decision rule:

- if minibatching preserves or improves throughput and learning signal, keep it as the default for attention runs
- if minibatching slows CPU but improves GPU utilization, keep it as a GPU-oriented option
- if learning quality requires multiple epochs, tune minibatch count before increasing full-batch update cost again

## Self-Play Parity

Purpose:

- port the training architecture features that matter for real use

Primary tasks:

- mixed offense/defense training with one shared policy
- frozen opponent checkpoint inference inside the compiled rollout
- opponent pool sampling from MLflow checkpoints
- per-env opponent sampling once single-opponent sampling is stable
- JAX-native self-play logic
- on-device opponent action selection
- on-device action assembly

This is where the old experiments matter most:

- the speedup only survives if the self-play and action-selection path stay mostly on device

Recommended order:

1. Train both roles first, still against legal-random opponents.
2. Add a single frozen opponent checkpoint to the compiled rollout.
3. Add checkpoint-pool sampling from MLflow artifacts.
4. Add per-env opponent sampling only after the single-opponent path is correct.

Rationale:

- the current production SB3 path uses one unified policy and trains mixed offense/defense envs simultaneously
- frozen opponent sampling is only meaningful once a JAX checkpoint can act as both offense and defense
- per-env opponent sampling is the closest production behavior, but it adds stacked/vmapped opponent params and should come after the simpler frozen-opponent path

Current status:

- mixed offense/defense training is implemented with one shared actor-critic
- frozen opponent action selection stays inside the compiled JAX rollout/eval path
- local checkpoints and MLflow `run_id` checkpoints can bootstrap the opponent pool
- newly saved checkpoints are added to the opponent candidate pool during training
- the trainer resamples frozen opponents from the checkpoint pool at checkpoint boundaries using the existing `opponent_pool_size`, `opponent_pool_beta`, and `opponent_pool_exploration` knobs
- grouped opponent sampling is implemented behind the JAX-specific `--grouped-opponent-sampling` flag
- in the JAX path, `--opponent-group-count` controls how many sampled checkpoint opponents split the rollout batch; this keeps the production-style opponent diversity while preserving batched JAX forwards
- fully vmapped per-row opponent params remains optional benchmark work, not the preferred default

Exit criteria:

- no dependency on current Torch/SB3 rollout plumbing
- self-play training works end to end in the JAX path

## Intent / Learnable Plays Parity

Purpose:

- restore the production learnable-play architecture without reintroducing ordinal intent features into the low-level policy input
- keep intent/play identity nominal, policy-side, and compatible with compiled JAX rollout/update paths

Design stance:

- do not add `intent_index_norm` as a normal global feature
- keep intent/runtime state in JAX environment state and run metadata
- condition the low-level attention policy through learned embeddings, matching the current production direction
- learnable plays require a discriminator/diversity objective; sampled intent IDs and embeddings alone are not sufficient
- implement the selector only after the low-level policy can demonstrably respond to different play IDs

Milestone 1: low-level JAX intent runtime state

Status: implemented.

- add offense intent fields to `KernelState`:
  - `intent_index`
  - `intent_active`
  - `intent_age`
  - `intent_commitment_remaining`
- add matching defense intent fields if we keep offense/defense parity from the start
- add static config fields for:
  - `num_intents`
  - `intent_commitment_steps`
  - `intent_null_prob`
  - `defense_intent_null_prob`
  - `intent_visible_to_defense_prob`
  - intent enable flags
- sample intent state during JAX reset
- advance/expire intent state during JAX step
- expose intent state in training/eval metrics and checkpoint metadata

Exit criteria:

- JAX reset/step carries active intent state deterministically under a fixed PRNG seed
- raw `shot_clock`, scoring, rollout, and action semantics remain unchanged
- native eval can report active intent usage and episode-level play IDs

Implementation notes:

- JAX reset now samples offense and defense intent state from static config with deterministic PRNG behavior.
- JAX step advances and expires active intent commitments without changing action or scoring semantics.
- Null/inactive intents carry `commitment_remaining=0`; active intents start at `intent_commitment_steps`.
- Train/eval metrics now include offense and defense intent active rates, ages, remaining commitment, and per-ID usage shares.
- Checkpoint/MLflow env metadata now persists the intent runtime config.
- Native JAX eval now emits episode-level offense and defense intent IDs and active/inactive summary counts.
- Intent state is runtime/diagnostic only in Milestone 1; Milestone 2 now consumes the role-selected intent in the policy path.

Milestone 2: low-level attention policy embedding conditioning

Status: implemented.

- add intent embedding config to `ActorCriticSpec`:
  - `intent_embedding_enabled`
  - `intent_embedding_dim`
  - `num_intents`
- add offense and defense intent embedding tables in the Flax attention model
- project the selected role-specific embedding into token embedding space
- add the gated intent delta to each player token before self-attention:
  - `token_i = token_mlp(player_i, globals) + gate * W_role * e_z`
- pass intent context into the actor-critic forward path without treating `z` as an ordered scalar feature

Exit criteria:

- same state with different active `intent_index` can produce different logits
- inactive intent gate produces the same logits as no intent conditioning
- checkpoint metadata fully records intent conditioning config
- checkpoint/MLflow metadata persists the same deterministic play-code-name mapping used by SB3
- the Attention tab can show current intent state for debugging

Implementation notes:

- `ActorCriticSpec` now records `intent_embedding_enabled`, `intent_embedding_dim`, and `num_intents`.
- The Flax attention policy has separate offense and defense intent embedding tables plus role-specific projections into token embedding space.
- Runtime policy context is passed separately from observations as `intent_index` and `intent_gate`; the PPO rollout batch stores both fields and feeds them back through the update loss.
- The JAX rollout, frozen-opponent, grouped-opponent, native eval, and web inference paths all pass role-selected intent context into actor-critic forward calls.
- Inactive/zero intent gate is tested to match no-context logits, while active different intent IDs are tested to change logits.
- MLflow/checkpoint policy metadata records the intent embedding config, and the Attention tab payload now reports the live JAX runtime intent gate/index.
- JAX training now logs `metadata/play_names.json`, tags `model_codename`, and stores `play_name_map` in checkpoint metadata so the UI and analytics can use the same code names as the SB3 path.

Milestone 3: discriminator / diversity objective

Status: current-policy offense discriminator path is implemented with both `set_step` and `mlp_mean`; `set_step` is the intended parity path.

- add a JAX-native discriminator that predicts active `intent_index` from behavior traces
- match the current SB3 production architecture:
  - collect only active offense intent behavior by default
  - keep a `current_policy_only`-style filter so frozen-opponent offense episodes do not train the discriminator unless explicitly enabled
  - compute the DIAYN-style bonus as `log q(z | trajectory) + log(num_intents)`
  - normalize the raw bonus with running mean/std
  - clip the normalized bonus
  - scale it with a scheduled beta
  - inject the bonus into rollout rewards before PPO advantage/return computation
- define the behavior window used by the discriminator:
  - player-token trajectory summaries
  - action/pass/shot features
  - role-aware offense and defense features if defense intents are enabled
- train the discriminator with Optax alongside PPO without feeding ordinal intent features into the low-level policy
- add a DIAYN-style intrinsic reward or auxiliary bonus based on discriminator confidence/log-prob
- keep reward scaling explicit and logged so this cannot silently dominate basketball rewards
- log discriminator trainbatch and holdout top-1 accuracy, macro OVR AUC, entropy, intrinsic reward mean, and per-intent classification stats
- save capped active-offense rollout samples for offline t-SNE/UMAP analysis:
  - `intent_index`
  - behavior feature vector used by the discriminator
  - discriminator embedding / penultimate layer output
  - action/pass/shot/outcome summary fields
  - update index and current-policy-vs-frozen provenance
  - default to opt-in or capped dumps so artifacts stay bounded
- support the production-relevant discriminator path in JAX:
  - set-step discriminator over `players`, `globals`, and `role_flag`
  - mean-pooled MLP fallback for simpler smoke tests
  - GRU/sequence discriminator is intentionally not planned for JAX
- preserve optional auxiliary heads as follow-up parity:
  - shot-end prior
  - shot-quality prior

Exit criteria:

- discriminator can predict intent above chance on held-out rollout windows
- intrinsic reward is nonzero and bounded by configured scale
- per-intent behavior begins to separate under uniform sampled intents
- discriminator/update cost is measurable and acceptable

Implementation notes:

- JAX train loop now supports `--intent-diversity-enabled` with offense-only, current-policy-only discriminator training.
- The JAX discriminator supports `--intent-disc-encoder-type set_step`, which consumes the attention policy's tokenized player observations plus global context and role flag.
- The mean-pooled `mlp_mean` discriminator remains available as a flat-feature smoke/debug path built from observations, selected actions, pass/shot/turnover events, and score deltas.
- `--intent-disc-include-shot-clock false` and `--intent-disc-include-pressure-exposure false` zero those globals for the set-step discriminator while leaving the policy observation unchanged.
- JAX supports update-count diversity scheduling with `--intent-diversity-warmup-updates` and `--intent-diversity-ramp-updates`, which is preferred over step-count scheduling for JAX runs.
- While scheduled beta is zero, JAX skips discriminator training, bonus computation, and discriminator sample dumps to match SB3 warmup behavior.
- JAX now supports SB3-style task reward scaling for DIAYN-first curricula. `--task-reward-scale-start/end` down-weight normal basketball reward before the discriminator bonus is added, and JAX-preferred `--task-reward-scale-warmup-updates` / `--task-reward-scale-ramp-updates` take precedence over the legacy step-count schedule.
- JAX low-level PPO now supports SB3-style entropy coefficient schedules through `--ent-coef-start`, `--ent-coef-end`, and `--ent-schedule linear|exp`. The scheduled value is logged as `jax/train/entropy_coef`.
- The low-level PPO reward path now injects normalized/clipped DIAYN-style bonus into active offense intent steps before GAE.
- Metrics include discriminator loss/entropy, trainbatch and holdout top-1 accuracy, trainbatch and holdout macro OVR AUC, trainbatch/holdout sizes, label/predicted intent distributions, active sample count, raw bonus stats, normalized bonus stats, beta, and bonus normalizer state.
- Checkpoints now persist discriminator params, optimizer state, config, and bonus normalizer stats.
- `--disc-eval-batch-output true` plus `--intent-sample-dump-size N` saves capped `.npz` active-offense samples for t-SNE/UMAP:
  - local checkpoints save under `intent_samples/`
  - MLflow runs log under `intent_samples/update_*`
- The set-step sample dump includes `features`, `players`, `globals`, `role_flag`, discriminator `embedding`, intent labels, actions, and pass/shot/outcome summary fields.
- `analytics/jax_intent_sample_embed.py` resolves the same deterministic play labels and writes them into plots, CSVs, and summaries.
- GRU/sequence discriminator support is not a JAX migration goal because the production-relevant path is the set-step discriminator.

Milestone 4: manual / sampled play conditioning smoke tests

- train with uniformly sampled active intents
- run native eval grouped by `intent_index`
- display per-intent eval rows and shot-chart filters with play code names instead of raw `z=N` labels
- log per-intent behavior summaries:
  - usage count
  - points per completed episode
  - pass attempts
  - assists
  - turnovers
  - shot profile
- verify that conditioning does not collapse to identical behavior across all play IDs

Exit criteria:

- model behavior changes measurably by active play ID
- speed remains acceptable with intent embeddings enabled
- UI can load and inspect an intent-conditioned JAX checkpoint

Milestone 5: selector head and segment runtime

Status: selector-runtime parity slice is implemented for current-policy offense rollouts.

- add a selector head on top of the attention/CLS representation
- keep selector observation neutralized with respect to current intent, matching SB3
- apply selected intent through runtime state/override, not by mutating normal observation features
- define JAX-native segment boundary logic for when a new play can be chosen
  - episode start
  - commitment timeout
  - optional completed-pass boundary after `intent_selector_min_play_steps`
- generate selector observations with current low-level intent neutralized
- sample or argmax a play ID and apply it to JAX runtime state
- add selector alpha schedule and epsilon-to-uniform exploration schedule
- log selector entropy, usage, and chosen-play distribution

Exit criteria:

- JAX runtime can choose and apply plays without Python-side rollout intervention
- selector decisions are reproducible under fixed seeds
- low-level policy receives selector-chosen play IDs through the same embedding conditioning path

Implementation notes:

- The Flax attention actor-critic now has production-style selector logits and selector-value heads on the offense CLS context.
- Selector final logits and value heads are zero-initialized, matching the SB3 convention of starting near uniform with zero value bias.
- Compiled JAX rollouts can now neutralize low-level intent context, run selector inference, sample a play with epsilon-to-uniform mixing, and apply the selected offense intent through `KernelState`.
- Selector application now happens at eligible offense segment starts where active offense intent age is zero and, when `--intent-selector-multiselect-enabled true`, at commitment-timeout boundaries where the active offense intent has exhausted `intent_commitment_steps`.
- Selector application also supports completed-pass reselection once the current play has reached `--intent-selector-min-play-steps`; the completed-pass event is carried to the next compiled scan step so reselection occurs before the next policy action is sampled.
- Commitment-timeout and completed-pass boundaries mirror the SB3 integrated selector behavior:
  - if the learned selector branch is active under alpha, record a selector sample and apply the selected play
  - if alpha does not select the learned branch, apply an unrecorded uniform fallback play so the expired segment does not continue indefinitely
- The train loop supports update-count selector alpha/epsilon schedules through:
  - `--intent-selector-alpha-warmup-updates`
  - `--intent-selector-alpha-ramp-updates`
  - `--intent-selector-eps-warmup-updates`
  - `--intent-selector-eps-ramp-updates`
- MLflow/train metrics now include selector alpha, epsilon, usage count/rate, applied count/rate, uniform fallback count/rate, commitment-timeout and completed-pass boundary counts, entropy, max probability, selected log-prob/value means, and per-intent usage shares.

Milestone 6: selector training objective

Status: selector PPO objective and rollout-cadence accounting are implemented for current-policy offense selector samples.

- implement selector PPO-style training over completed segments
- add selector value prediction
- add selector entropy regularization
- add usage regularization toward non-collapsed play usage
- train on segment returns, including optional bootstrap value at segment boundaries
- log selector return, advantage, KL, clip fraction, usage-by-intent, top-1-by-intent, and per-segment usage
- keep selector update inside the JAX/Optax training path

Exit criteria:

- selector learns non-trivial play usage
- selector metrics are logged to MLflow
- selector does not destabilize low-level PPO training

Implementation notes:

- The JAX train loop now builds selector batches from offense rollout selector-start records.
- Selector returns now use segment-aware discounted rewards: a selector sample bootstraps from the next selector value when another selector segment starts before episode termination, and unfinished rollout-window segments bootstrap from the final selector value.
- Selector advantages are normalized over actually-used selector samples and computed against the stored selector value from rollout time.
- The selector update runs after the low-level PPO update, uses the same Optax optimizer state as the actor-critic, and applies:
  - clipped PPO selector policy loss
  - selector value loss
  - selector entropy bonus
  - KL-to-uniform usage regularization
- The selector loss uses neutralized low-level intent context, preserving the SB3 architecture where the selector does not observe the currently active low-level play.
- The selector objective respects the active alpha/epsilon schedules:
  - if alpha is zero, no selector samples are generated and no selector gradient is applied
  - epsilon-to-uniform is mixed into selector probabilities for both rollout sampling and selector PPO log-prob computation
- MLflow/train metrics now include selector train loss, policy loss, value loss, entropy, usage KL, approximate KL, clip fraction, gradient norm, sample count, return/advantage/value means, and per-intent usage/probability.
- When `--intent-selector-train-every-rollouts > 1`, selector batches are accumulated across skipped rollout updates and trained together on the scheduled selector update rather than discarded.
- `--intent-selector-max-samples-per-update` is applied to the accumulated selector batch by masking excess active selector samples while preserving the compiled batch shape.
- Remaining optional selector cleanup is per-segment bucket diagnostics for deeper selector debugging.

## Play-Aware Dev UI Parity

Purpose:

- make the dev app useful for visually debugging learned JAX plays, not just loading and stepping the model
- preserve the useful SB3 play diagnostics while using JAX-native metadata, eval, and selector outputs
- keep unsupported SB3-only panels explicitly gated instead of showing stale controls

Current implemented hooks:

- the backend state payload exposes `play_name_map`, current offense/defense play IDs, play names, intent ages, commitment remaining, selector boundary diagnostics, and model metadata
- the Policy tab can show selector intent preferences with play names, raw selector probabilities, epsilon-mixed probabilities, deployed alpha/epsilon probabilities, entropy, KL-to-uniform, logits, and the current play row
- the Eval tab can choose selector intent selection mode: learned sample, argmax, or uniform random
- native JAX eval returns per-intent aggregate stats and intent start counts
- the Stats panel shows per-intent eval rows with play labels
- the board shot-chart dropdown can switch from team/player charts to per-play charts after eval
- the Attention tab shows the runtime intent index/gate/visibility context for the current tokenized observation

Milestone UI-1: explicit play capability contract

Status: first slice implemented for JAX play metadata and selector distribution gating.

- add or audit backend capability flags for:
  - play names / intent metadata
  - selector distribution in current state
  - per-intent eval stats
  - play-conditioned shot chart filtering
  - playbook / counterfactual play preview
  - manual intent override
- make JAX UI gates derive from capability flags plus metadata, not from assumptions about model names or training params
- show a clear disabled reason when a play-aware panel cannot run for the loaded checkpoint

Exit criteria:

- a JAX checkpoint without selector/discriminator metadata does not show misleading play controls
- a selector-enabled JAX checkpoint enables only the supported play diagnostics

Implementation notes:

- Inference capabilities now include play-aware flags for metadata, selector distribution, per-intent eval, play shot charts, and manual intent override.
- JAX inference derives those flags from checkpoint `policy_spec`, `env_config`, and play-name metadata.
- JAX selector checkpoints expose generic selector outputs so the existing backend selector preference payload can populate the Policy tab without SB3-specific `.policy` assumptions.

Milestone UI-2: Policy tab parity and cleanup

Status: live-play header implemented; selector distribution table now works through the generic selector interface.

- keep the existing selector intent distribution table as the primary play distribution view
- add a compact live play header:
  - current play code name
  - active/inactive state
  - age and commitment remaining
  - selector segment index
  - last selector boundary reason
  - offense visibility to defense
- clearly separate selector distribution `p(z | s)` from low-level action distributions `pi(a | s, z)`
- show a no-data state that distinguishes "selector not present" from "policy probabilities were not requested"

Exit criteria:

- while running self-play, the Policy tab tells us both which play is active and what play the selector would choose from the current state
- play labels match MLflow/checkpoint `play_name_map`

Implementation notes:

- The Policy tab now shows current offense play, active flag, age, commitment remaining, selector segment, last boundary, defense visibility, and defense play.
- If selector training is enabled but the current state cannot produce selector preferences, the Policy tab now shows an explicit reason instead of silently hiding the section.

Milestone UI-3: Eval tab play controls

- keep learned sample, argmax, and uniform random selector modes
- add explicit eval modes for play diagnostics:
  - force one selected play for all eval episodes
  - sweep all plays with equal episode counts
  - sweep a user-provided subset of plays
  - optionally include a no-intent/null bucket if the checkpoint supports null intents
- route these modes through the JAX-native eval fast path where possible
- surface the eval seed and actual sampled/forced play counts in the response

Exit criteria:

- we can run a 10K-episode eval that produces fair per-play comparisons without manually changing runtime state
- repeated evals are reproducible only when the same seed/mode is used

Milestone UI-4: Stats, shot charts, and assist-flow by play

- keep the existing per-intent aggregate table and board shot-chart dropdown
- extend per-play eval stats to include the same debugging views as team/player eval:
  - shot chart
  - shot type mix
  - pass attempts and completions
  - assists and potential assists
  - turnovers by reason
  - lane violations
  - PPP and reward/episode
- add play filtering for assist/Sankey diagnostics, not only shot charts
- ensure copied Markdown stats include per-play summaries with code names

Exit criteria:

- the Eval/Stats workflow can answer "what does play X do?" without leaving the UI
- per-play filters use the same labels as Policy and MLflow artifacts

Milestone UI-5: Playbook / counterfactual parity decision

- decide whether the Playbook tab should remain enabled for JAX checkpoints
- if enabled, ensure it uses JAX-compatible policy inference, play override, and selector boundary semantics
- if not enabled, replace it with a narrower JAX play preview panel that runs fixed-play rollouts from the current state or captured snapshot
- keep trajectory overlays grouped by play code name

Exit criteria:

- play-conditioned counterfactual rollouts no longer depend on SB3-only assumptions
- the UI clearly distinguishes statistical eval from current-state play previews

Milestone UI-6: Tests and fixtures

- add backend tests for the JAX state payload play fields and selector intent preference payload
- add eval-route tests for forced/sweep play modes and per-play stats
- add frontend fixture coverage for:
  - selector-enabled JAX checkpoint
  - intent-conditioned but selector-disabled JAX checkpoint
  - legacy/SB3 checkpoint
- run frontend build after UI changes

Exit criteria:

- play-aware UI behavior is stable across JAX, SB3, and unsupported checkpoints

## Starting Templates Parity

Purpose:

- restore the SB3-style reset/start-template system after the current selector run proves stable
- reduce spawn-distribution artifacts and make visual/eval comparisons more controlled

Primary tasks:

- inspect the current SB3 start-template and jitter implementation: done
- add JAX-native template metadata to env static/config state: done
- implement compiled reset sampling from template IDs: done
- expose template config in MLflow/checkpoint metadata: done for train metadata/artifacts
- support eval-time use of a session-loaded template library even when the checkpoint was not trained with templates: done
- support Policy-tab playable template controls and self-play starts from selected templates: done
- implement deterministic PRNG jitter/mirror sampling per reset rather than pre-resolved candidate variants: done
- expose start-template metadata in the dev app JAX metadata panel
- add eval/train launch options for fixed template, sampled templates, and zero-jitter diagnostic runs: sampled-template train launch enabled for the active JAX diagnostic config

Exit criteria:

- JAX train/eval can run from the same family of start templates as the SB3 path: done for train, eval, and Policy-tab self-play
- fixed-template eval is deterministic under a fixed seed and sampled-template eval varies under different seeds
- start-template choices are visible in logs/artifacts so shot charts and behavior can be compared honestly

Recommended first implementation slice:

- implement Milestone 1, Milestone 2, and Milestone 3 before selector work
- add tests proving intent state propagation, embedding-conditioned logits, and discriminator learning on synthetic/separable traces
- do not start selector work until low-level conditioning plus discriminator-driven diversity is working

## Advanced Feature Parity

Purpose:

- selectively restore the current advanced features after the JAX base is already operational

Candidates:

- dual critic / dual policy
- templates / curricula
- phi shaping
- intent learning and selector/discriminator path, tracked explicitly in the `Intent / Learnable Plays Parity` section
- evaluation and analytics hooks

Important constraint:

- these should be restored in order of product value, not in order of current code size

Exit criteria:

- only the features that still justify their complexity come back

This phase should be explicitly selective, not automatic.

## Cutover

Purpose:

- make the JAX stack the primary path

Primary tasks:

- compare trained JAX checkpoints against the current stack
- run backend shadow testing
- define a champion/challenger process
- cut over inference paths when stable
- retire or freeze old Torch/SB3 training paths only after confidence is high

Exit criteria:

- JAX path is faster
- JAX path learns well enough
- JAX path serves inference reliably
- operational tooling is good enough for normal use

## Recommended Immediate Sequence

The next concrete work should be:

1. run longer JAX training jobs and judge learning quality, not just throughput
2. tighten eval and checkpoint-selection criteria
3. decide which representation upgrade is actually needed next
4. expand only the features required for useful testing and migration

That is the shortest path from “package-native JAX stack that runs” to “new stack we can actually rely on.”

## Mapping From Current Code To Future JAX Ownership

Current rollout-critical ownership:

- env: [basketworld_env_v2.py](/home/evanzamir/basketworld/basketworld/envs/basketworld_env_v2.py)
- PPO: [integrated_mu_selector_ppo.py](/home/evanzamir/basketworld/basketworld/algorithms/integrated_mu_selector_ppo.py)
- policy: [set_attention_policy.py](/home/evanzamir/basketworld/basketworld/policies/set_attention_policy.py)
- self-play: [self_play_wrapper.py](/home/evanzamir/basketworld/basketworld/utils/self_play_wrapper.py)

Future JAX ownership should become:

- `basketworld_jax.env`
  owns reset, step, rollout state, reward, masks, observation assembly

- `basketworld_jax.models`
  owns actor-critic modules and later set-attention / pointer-targeted heads

- `basketworld_jax.rollout`
  owns compiled rollout loops, GAE, action sampling, opponent selection

- `basketworld_jax.train`
  owns PPO updates, config, run loop, logging

- `basketworld_jax.inference`
  owns checkpoint load, backend prediction, masking, deterministic inference

This ownership split is what keeps the migration coherent.

## Risks

Main risks now are no longer raw speed.

They are:

- learning quality on the reduced stack
- complexity growth while restoring parity
- deployment compatibility
- keeping the rollout path on device as complexity returns

That means migration decisions should now be driven by:

- operational readiness
- learning quality
- scope discipline

not by more isolated kernel benchmarks.

## Decision Rule

Continue the migration if all three remain true:

1. the reduced JAX stack stays materially faster than the current stack
2. the reduced JAX stack learns behavior worth keeping
3. the new package path stays cleaner than trying to hybridize with the old stack

If any of those stop being true, pause and reassess before expanding parity scope.
