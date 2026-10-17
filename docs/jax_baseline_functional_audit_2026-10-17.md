# JAX baseline functional audit — 2026-10-17

## Conclusion

The range from `b4626ba` to `bbc08aa` contains a confirmed possession-mode
mechanics change: a pending inbounder is restored to the court when a live
possession boundary interrupts re-entry. The baseline can clear the inbound
metadata while leaving that player outside the court. This change affects the
environment used by both training and deployment evaluation.

The final SB3-removal commit by itself did not change the active JAX trainer,
optimizer, model, or transition logic. The run comparison spans earlier
mechanics changes as well as that cleanup. The audit establishes a concrete
reason to expect different trajectories; it does not establish how much of the
observed pickup-rate or turnover difference that reason explains.

## Scope and provenance

- Baseline run: `07f4d9401aae4d0186fa694ee143b6c7`.
- Current run: `12c4f5dabbcf4a9b9dde76092f5da775`.
- Requested baseline commit: `b4626ba62a7c68c7b451961644e2eedd853a639c`.
- Current commit: `bbc08aa190ceb1ec2ed92682b60007ced48bb4e9`.
- Cleanup-only comparison: `a4b6971..bbc08aa`.
- Both saved trainer configurations and commands specify `policy_seed=0`.
- Baseline launch history and update-100/update-2500 checkpoint metadata show
  that the early baseline actually ran `76cb218`. Its later launch used
  `b4626ba`. The intervening changes affect the trainer application and
  continuation controls, not the JAX kernel or training algorithm, so using
  `b4626ba` for the kernel comparison is appropriate.

The latest saved baseline command is a resume command with an extended target.
It is not presented here as the original launch command. Static comparisons
use its saved environment arguments; the sampled early checkpoint metadata
also confirms schema 5, clearance 1, and actual-points rebound rewards.

## Confirmed change affecting these runs

### Interrupted inbounder restoration: `a13de76`, completed by `7700ad7`

Relevant code: `basketworld_jax/env/minimal.py`,
`_return_pending_inbounder_to_court_single` and
`_finalize_possession_single` (current boundary-position selection near line 5247).

Before: when a live turnover or defensive rebound ended a possession before
the previous inbounder re-entered, the finalizer could clear `inbound_player`
but retain that player's out-of-bounds position. The player then lost the
special pending-inbounder handling while remaining outside the court.

After: at that possession boundary, the player is restored to the nearest free
court cell before the metadata is cleared. During an ordinary live step with
no boundary, the player stays outside and waits for legal re-entry.

`a13de76` introduced restoration but initially also applied it without a
possession boundary. `7700ad7` added the `advance` guard. The latter repairs an
intermediate regression; relative to the requested baseline, the net change
is restoration at interrupted live possession boundaries.

Reproduced with identical inputs and random keys:

- Live turnover finalizer: player 0 changes from `[-3, 4]` outside the court to
  `[-2, 4]` on court. Both versions clear inbound metadata. The only differing
  shared state field is `positions`.
- Defensive rebound finalizer: the same position difference occurs.
- A complete forced pressure-turnover step reproduces the same difference;
  the other 122 shared state/output fields match exactly.
- No-boundary, made-basket/check, and dead-ball inbound finalizer probes match
  across all 60 shared state fields.

Restoring a participant changes future movement, spacing, passing targets,
pressure and rebound opportunities, and therefore future observations and
learning data. There was no direct change to the pickup-rate calculation or
the check-location/clearance mechanics in this range.

## Other functional changes, inactive for this comparison

### Timed games: `6a544d1`, trainer configuration exposed in `a74c195`

Added regulation and overtime clocks, period expiration, total-step truncation,
and clock behavior during check pickup and inbounds. Possession mode retains
its prior check shot-clock behavior; pickup countdown changes are gated by
`episode_end_mode=game_clock`.

These runs use `episode_end_mode=possessions`. The new clock fields therefore
do not activate timed-game endings. Reset adds an overtime key via `fold_in`,
without changing the existing random split sequence. The sampled reset,
observation, mask, and transition comparisons below confirm parity outside
the inbounder case.

Timed-game integration is still incomplete for compiled trainer/native-eval
callers: `step_batch_minimal` suppresses period handling when the mode is a
tracer, and those callers still use that entry point rather than
`step_batch_game_clock`. The source explicitly defers this wiring to issue
#64. Exposing the flag is not evidence that compiled timed training is ready.
This limitation does not affect the possession-mode comparison.

### Shaping-discount fallback: activated by `a74c195`

The old factory passed `reward_shaping_gamma=None` to a constructor that used
`1.0`. `JaxEnvironmentConfig.from_args` now replaces an omitted shaping gamma
with PPO's `gamma`.

Confirmed static probe: with PPO gamma `0.97` and no explicit shaping gamma,
the kernel value changes from `1.0` to approximately `0.97`. The default CLI
configuration likewise changes this field from `1.0` to `0.99`.

This can change rewards when potential shaping is active. Both compared runs
use gamma `1.0`, shaping disabled, and scoring-event rewards, so it cannot
explain their difference.

### Rebound terminal-reward flag now reaches the kernel: `a74c195`

The old training bootstrap neither passed `rebound_terminal_reward_mode` to
the legacy constructor nor copied it in its static override list. It therefore
used `actual_points` even if another mode was requested. The new typed config
honors the argument.

Confirmed static probe: requesting `last_shot_ep` yields mode ID `0`
(`actual_points`) before the migration and ID `2` (`last_shot_ep`) afterward.
That changes rewards in modes where the auxiliary/shot rewards are consumed.
Both compared runs explicitly use `actual_points`, so their effective values
match.

### Single-possession scaffold reset: `a74c195`

`sample_state_batch` now uses the JAX reset for every mode. Previously its
single-possession branch sampled the Python environment with sequential
integer seeds. That changes the reset stream for consumers of that branch.
The multi-possession branch already used JAX resets. In addition, the actual
training loop constructs its fresh states using `reset_batch_minimal` and
`policy_seed` after obtaining the static configuration, both before and after
the migration. The scaffold reset replacement does not explain these runs.

## Evaluation/inference changes and one reproduced regression

### Explicit alias overrides can be ignored: `921ae50` evaluation migration

`JaxEnvironmentConfig.from_evaluation_request` first expands checkpoint values
into canonical field names, then merges request fields. `from_mapping` applies
an alias only when the canonical field is absent. As a result, an explicit
request using an accepted alias can lose to an inherited checkpoint default.

Reproduction through `_resolve_native_environment_config`:

- Checkpoint: `shot_clock=24`, `min_shot_clock=10`.
- Explicit request: `shot_clock_steps=18`.
- Actual resolved clock: **24**, despite the explicit override.
- A canonical request `shot_clock=18` resolves correctly to **18**.

The old evaluator passed `shot_clock_steps` directly to the environment
constructor. This is a real configuration-precedence regression. The same
merge problem affects newly supported aliases such as `court_height` and
`offensive_three_seconds` when a canonical inherited value exists.

Recommended fix: canonicalize each input layer independently before merging
checkpoint, required-request, and optional-request values. Add a focused test
that overrides an inherited value through an alias. No fix was applied as
part of this audit.

This is in the separate native evaluation path. The trainer's
`jax/eval_deploy/*` metrics use its own static configurations and compiled
deployment runner, so this bug does not explain the reported training-run
comparison.

### Changed contracts, not demonstrated causes of the current metrics

`921ae50` makes native evaluation/inference derive environment settings from
checkpoint metadata plus explicit overrides, checks compatible checkpoint
contracts, and normalizes supported observation schemas. Inference static
state now belongs to the model configuration rather than being rebuilt from
the supplied Python environment. Skill Ratings also uses the typed checkpoint
configuration. These changes can reject previously accepted incompatible or
incomplete metadata and change omitted-setting behavior in external callers.

`a4b6971` removes the Python environment backing the interactive runtime and
supports model-free JAX sessions. `bbc08aa` removes SB3 checkpoint/runtime
dispatch, legacy analytics tools, and MCTS-related API fields. These are
intentional product/API changes; they are not called by the JAX training loop.

The parser's `intent_selector_mode` default changes from `callback` to
`integrated` in the cleanup. It is not used to select the JAX training
implementation. Native evaluation already normalizes the old `callback`
metadata to `integrated` for selector-enabled JAX policies.

## Verification performed

All execution probes were CPU-only and restricted to two CPU affinity slots.
No training job, GPU evaluation, full pytest suite, or service restart was
started. No runtime source code was modified.

1. **Actual-run static comparison:** all 155 common fields match exactly for
   offense and defense, including geometry, spawn/check/inbound tables,
   rebound probabilities, rewards, skills, and observation settings. Current
   code adds four inactive game-clock configuration fields.
2. **Cleanup-only static comparison:** all 159 fields match exactly between
   `a4b6971` and `bbc08aa` for both roles.
3. **Seeded resets:** 16 probe states using the training root/role key scheme;
   all 60 shared state fields match. Nine additive clock/truncation fields
   account for the new state shape.
4. **Initial attention observations and action masks:** exact equality across
   those 16 probe states.
5. **Possession finalizer:** five scenarios, with differences limited to the
   two interrupted-inbounder live-boundary cases described above.
6. **Complete scalar transitions:** seven scenarios. Live no-op, check setup,
   check wait, check timeout, inbound wait, and shot-clock expiry match across
   all 123 shared state/output fields. Forced pressure turnover with a pending
   inbounder differs only in that player's position.
7. **Source comparison:** `train/runtime.py`, PPO/GAE types, optimizers, model
   implementation, intent implementation, and frozen training defaults are
   unchanged. Training-loop function code is unchanged; `main.py` changes
   are clock argument validation and additive configuration/metadata support.
8. **Conditional configuration probes:** shaping gamma and rebound reward
   mode differences reproduced; evaluation alias precedence reproduced.

The legacy static probe executed the original environment-constructor argument
expression and the original static override loop. It omitted SB3 wrappers,
which do not modify the base static settings consumed by this builder. Both
versions used the same currently available fitted rebound assets and installed
libraries, isolating source/config changes rather than historical dependency
or asset versions.

Machine-readable evidence, raw arrays, and probe scripts are saved locally in
`artifacts/audits/jax-baseline-2026-10-17/` (gitignored). Temporary historical
source snapshots and saved launch records remain under
`/tmp/bw-functional-audit/`; the probes depend on those temporary inputs.
The durable audit conclusions are in this document.

## What remains unproven

These are targeted source and transition checks, not an exhaustive proof of
all possible states or a same-checkpoint long-horizon evaluation. They do not
quantify how much the inbounder repair contributes to the learned pickup-rate
gap. GPU/compiler numerical effects or other historical execution differences
also have not been ruled out. Both runs sharing a seed does not itself prove
bitwise reproducibility across code and execution changes.

The most informative next experiment is to evaluate one fixed checkpoint and
identical episode seeds with the two transition implementations, count the
first differing events, and measure how often interrupted inbounders occur.
That would isolate mechanics from learned-policy differences before deciding
whether another long training run is necessary.

## Approved follow-up fixes

Implemented after review of this audit:

- Normalize alias keys separately within checkpoint, required-request, and
  optional-request layers. Explicit evaluation overrides now win consistently;
  unspecified fields remain inherited.
- Preserve requested configuration and add `effective_env_config` to checkpoint
  payloads and metadata. Evaluation/inference prefer that record. Log effective
  environment parameters and `jax/policy_seed` for new MLflow runs; append
  continuation artifacts without overwriting historical parameters.
- Preserve the corrected rebound reward flag and the inbounder restoration.
  Require an explicit shaping discount on resume when older metadata is
  ambiguous, and reject disagreement with a recorded effective discount.
- Reject compiled game-clock training/native evaluation pending #64. The trainer
  settings schema offers only the supported possession mode. Scalar/reset clock
  support remains available for development.
- Replace the previous timed-training smoke assertion with rejection coverage;
  that assertion checked metadata/horizon, not whether compiled clocks expired.

Focused CPU regression verification: 37 tests passed, including override
precedence, kernel-versus-metadata reward values, checkpoint metadata round trips,
continuation logging, ambiguous/changed resume discounts, trainer schema, and
pre-compilation clock guards. No full test suite or training run was launched.

## Fixed-checkpoint paired replay

Replayed the baseline update-2500 checkpoint unchanged against `b4626ba` and
current kernels, with 16 episode seeds (`2000000..2000015`), deterministic
policy-only self-play, and identical per-step RNG keys. Selector execution was
disabled consistently with native multi-possession evaluation. Both runs used
the audited static arrays, the same installed dependencies, CPU only, and two
cores, sequentially. All 16 games completed on both versions within the
2,000-tick cap. No optimization or training was performed.

All 60 shared initial state fields matched exactly. Five games retained identical
shared states through completion. Eleven first diverged solely in `positions`
at an interrupted inbounder restoration; actions matched through each first
difference. Nine first divergences followed intercepted passes from the awaiting-
inbound phase; two followed live turnovers (interception and defender pressure).
This broadens the original hand-constructed live-boundary example: an inbound
pass that is intercepted can also strand the inbounder in the baseline.

Across these 16 games:

- Mean per-game pickup rate: baseline 76.03%, current 73.19%.
- Pooled pickups / (pickups + check violations): 277/361 (76.73%) versus
  271/365 (74.25%).
- Mean turnovers per game: 16.44 versus 16.81.
- Total possessions: 800 versus 802 (the current replay included overtime).
- Combined points per possession: 1.190 versus 1.207.

These paired observations confirm that the inbounder repair affects real policy
trajectories. The pickup-rate direction in this small fixed-policy sample is
opposite the larger difference between the separately trained runs. It does not
attribute that training-run gap or establish the repair's long-run effect on
learning. The mechanics fix remains warranted because leaving a player outside
without inbound metadata is invalid state.

The reusable replay command is
`scripts/audits/compare_jax_baseline_checkpoint.py`. Raw traces, run metadata,
comparison results, dependency versions, source/input hashes, and the summary
script are under the gitignored
`artifacts/audits/jax-baseline-2026-10-17/` directory, with filenames beginning
`checkpoint-`. The original audit arrays and historical run/checkpoint metadata
were preserved.
