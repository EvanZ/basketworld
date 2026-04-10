# JAX Batched Environment Benchmark Plan

## Goal

Estimate whether a batched JAX rewrite of the BasketWorld environment is worth pursuing, without changing the current SB3 training path.

This plan is for a standalone benchmark and prototype track. It should not affect:

- current training
- current evaluation
- current backend/UI
- current env implementation

## Why

Current rough wall-clock split:

- rollouts: ~40s
- PPO updates: ~15s
- total: ~55s

If rollouts can be reduced substantially with a batched JAX environment, training throughput may improve enough to justify a rewrite. The benchmark should answer:

1. How much of the current rollout cost is actually in env stepping?
2. Which env sections dominate runtime?
3. What kernel-level speedup is plausible from a batched JAX transition?
4. What end-to-end speedup would that imply for training?

## Non-Goals

This effort should not:

- replace the current env yet
- integrate JAX into SB3 yet
- rewrite PPO updates yet
- support full feature parity on day one
- optimize rendering, backend routes, or UI code

## Success Criteria

The benchmark is useful only if it gives a credible go/no-go signal.

Target thresholds:

1. Kernel-level speedup
- `>= 5x` at realistic batch sizes
- ideally `>= 8x`

2. Rollout-equivalent speedup
- `>= 3x` including observation and action-mask generation

3. Projected end-to-end training speedup
- `>= 4x` is promising
- `>= 6x` strongly justifies a rewrite

If the JAX prototype cannot clear these thresholds, the rewrite should be reconsidered.

## High-Level Approach

Build a separate benchmark/prototype path in parallel with the current env.

The work splits into three layers:

1. Baseline measurement on the current env
2. Standalone batched JAX transition prototype
3. Apples-to-apples comparison and end-to-end speed estimate

## Deliverables

Planned new files:

- `benchmarks/rollout_baseline.py`
- `benchmarks/jax_kernel.py`
- `benchmarks/compare_rollout.py`
- `tests/test_jax_transition_equivalence.py`

Optional later:

- `benchmarks/jax_kernel_obs.py`
- `benchmarks/compare_rollout_report.py`

## Benchmark Scope

The benchmark should measure rollout-side performance only.

It should include:

- reset throughput
- step throughput
- observation generation cost
- action mask generation cost
- end-to-end rollout steps/sec

It should not include:

- PPO backward/update time
- MLflow logging
- rendering
- backend route overhead
- frontend interactions

## Existing Instrumentation to Reuse

The current env already has profiling hooks and should be used as the baseline.

Relevant code:

- [train/config.py](/home/evanzamir/basketworld/train/config.py)
- [train/profiling.py](/home/evanzamir/basketworld/train/profiling.py)
- [basketworld_env_v2.py](/home/evanzamir/basketworld/basketworld/envs/basketworld_env_v2.py)

Important profiled sections already exist for:

- `step`
- `_process_simultaneous_actions`
- `_get_observation`
- `_get_action_masks`
- geometry helpers
- reward helpers

## Phase 1: Baseline Benchmark

### Script

`benchmarks/rollout_baseline.py`

### Purpose

Measure the current non-JAX env as it exists today.

### Inputs

- env config
- number of envs
- number of episodes
- horizon
- action mode
  - random legal
  - pre-generated legal
- profiling enabled/disabled

### Outputs

- env steps/sec
- resets/sec
- mean step latency
- mean reset latency
- top profile sections by total time
- top profile sections by per-call time

### Notes

Use pre-generated or deterministic legal actions for stable comparisons. Avoid policy inference in this script. The point is env cost, not policy cost.

## Phase 2: JAX Prototype

### Script

`benchmarks/jax_kernel.py`

### Purpose

Implement a minimal pure-state batched transition kernel in JAX.

### First-Class Constraints

The JAX prototype should be:

- pure functional
- batched from the start
- JIT-friendly
- independent from Gym
- independent from SB3

### Initial API

Suggested functions:

```python
reset_batch(key, config, batch_size) -> state
step_batch(state, actions, rng_key, config) -> next_state, reward, done, info
build_obs_batch(state, config) -> obs
build_action_mask_batch(state, config) -> action_mask
```

### Initial State Fields

The first prototype should carry only the state needed for rollout correctness:

- player positions
- ball holder
- shot clock
- step count
- offense/defense scores
- episode ended flag
- assist candidate state
- lane violation counters
- pressure-related state if required
- offense intent state if required by rollout semantics
- stochastic state or explicit RNG inputs

Avoid UI-only or logging-only fields initially.

### Config Fields

Use a small immutable config object for:

- court size
- players per side
- shot clock settings
- passing mode
- pressure parameters
- reward parameters
- phi shaping flags if included
- intent-learning toggles if included

Start with one fixed training-relevant configuration. Do not attempt every mode at once.

## Phase 3: Comparison Harness

### Script

`benchmarks/compare_rollout.py`

### Purpose

Run the current env and the JAX prototype under matched conditions and compare throughput.

### Benchmark Rules

1. Warm up JAX before timing
2. Use the same batch size and horizon
3. Use the same action-generation policy
4. Report both:
- core transition speed
- transition + obs + mask speed
5. Include projected end-to-end training speedup using the current rollout/update split

### Projection Formula

Given:

- current rollout time `R`
- current update time `U`
- measured rollout speedup `S`

Projected total time:

`T_new = (R / S) + U`

If update speedup is later estimated too:

`T_new = (R / S_rollout) + (U / S_update)`

## Correctness Strategy

Performance numbers are meaningless without parity checks.

### Test File

`tests/test_jax_transition_equivalence.py`

### Required Test Categories

1. Reset parity
- seeded reset produces equivalent initial states

2. Movement/collision parity
- positions and occupancy resolve identically

3. Passing parity
- pass success/failure and recipient state match

4. Shot parity
- expected points / outcomes / episode termination match

5. Turnover and violation parity
- reasons and rewards match

6. Reward parity
- offense and defense rewards match numerically

7. Done parity
- terminal flags match

### Stochastic Control

Do not let NumPy and JAX sample independently.

Instead:

- generate explicit random uniforms or random keys externally
- feed the same stochastic draws into both implementations

This avoids false mismatches caused by RNG differences.

## Feature Scope for the First Prototype

The first prototype should target only the features needed to answer the speed question.

Recommended scope:

1. one env config
2. one player count
3. one pass mode
4. no rendering
5. no wrappers
6. no backend-specific state

Likely good initial target:

- current main training configuration
- without optional UI-only paths

## What to Exclude Initially

Exclude these until the core benchmark is working:

- mirror observation logic
- Playbook paths
- backend routes
- evaluation routes
- MCTS
- artifact logging
- full replay compatibility

These do not help answer the rollout-speed question.

## Risks

### 1. Python boundary still dominates

If the prototype is called through a Python loop one env at a time, results will understate JAX potential.

Mitigation:

- batch from the beginning
- measure large enough batch sizes

### 2. Observation generation dominates

If transition speed improves but obs/mask generation stays expensive, end-to-end rollout gains may be much smaller.

Mitigation:

- benchmark both transition-only and transition+obs+mask

### 3. JAX prototype cheats by omitting important logic

This would produce inflated speedups that are not actionable.

Mitigation:

- document included/excluded features explicitly
- add parity tests before trusting benchmark numbers

### 4. GPU assumptions are optimistic

Even if env stepping gets much faster, PPO updates may not get a `10x` gain without enough batch size and reduced Python overhead.

Mitigation:

- treat rollout speedup and update speedup as separate estimates

## Milestones

### Milestone 1: Baseline

Deliver:

- `benchmarks/rollout_baseline.py`
- profile report for current env

Decision:

- identify top 3 rollout hotspots

### Milestone 2: Transition Kernel

Deliver:

- `benchmarks/jax_kernel.py`
- parity tests for reset, step, reward, done

Decision:

- is batched transition speedup materially large?

### Milestone 3: Full Rollout-Equivalent Benchmark

Deliver:

- obs + mask generation in JAX
- `benchmarks/compare_rollout.py`

Decision:

- estimate rollout speedup and total training speedup ceiling

### Milestone 4: Rewrite Decision

Go if:

- parity is credible
- rollout-equivalent speedup is large enough
- projected total speedup justifies rewrite complexity

No-go if:

- gains are too small after obs/mask generation is included
- parity complexity is too high relative to the benefit

## Recommended Implementation Order

1. Build `rollout_baseline.py`
2. Capture current hotspot profile
3. Define JAX `State` and `Config`
4. Implement batched `step_batch`
5. Add transition parity tests
6. Implement obs + mask generation
7. Add full comparison harness
8. Estimate end-to-end speedup
9. Decide whether to proceed to a real env rewrite

## Decision Standard

This work is justified only if the benchmark proves that a batched JAX env can move the bottleneck enough to matter.

The standard should be:

- measurable
- reproducible
- isolated from training noise
- strong enough to justify a rewrite, not merely interesting

