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

Before applying thresholds, re-measure the rollout/update split on the same machine and config used for the benchmark.

With the current rough split:

- rollouts `R ~= 40s`
- updates `U ~= 15s`
- total `T ~= 55s`

the theoretical maximum speedup from rollout-only acceleration is:

- `T / U ~= 55 / 15 ~= 3.67x`

That means rollout-only work cannot justify a `4x+` end-to-end claim under the current SB3 update path. Success criteria should therefore separate:

- rollout-only value under the current training path
- longer-term upside if update-side work is revisited later

Target thresholds:

1. Kernel-level speedup
- `>= 5x` at realistic batch sizes
- ideally `>= 8x`

2. Rollout-equivalent speedup
- `>= 3x` including observation and action-mask generation

3. Projected end-to-end training speedup under the current SB3 update path
- `>= 2.0x` is promising
- `>= 2.5x` strongly justifies a rewrite investigation
- `>= 3.0x` is exceptional and already near the rollout-only ceiling for the current split

4. Longer-term ceiling estimate if update-side work is also revisited later
- report separately
- do not use this as the main go/no-go criterion for the env-only rewrite decision

If the JAX prototype cannot clear these thresholds, the rewrite should be reconsidered.

## High-Level Approach

Build a separate benchmark/prototype path in parallel with the current env.

The work splits into three layers:

1. Baseline measurement on the current env
2. Standalone batched JAX transition prototype
3. Apples-to-apples comparison and end-to-end speed estimate

## Deliverables

Planned new files:

- `benchmarks/__init__.py`
- `benchmarks/common.py`
- `benchmarks/rollout_baseline.py`
- `benchmarks/jax_kernel.py`
- `benchmarks/compare_rollout.py`
- `tests/test_jax_transition_equivalence.py`

Optional later:

- `benchmarks/jax_kernel_obs.py`
- `benchmarks/compare_rollout_report.py`

## Concrete Implementation Plan

This section turns the benchmark into an execution sequence I can implement directly.

### Phase 0: Scaffolding And Dependency Gate

Purpose:

- create the benchmark package and shared helpers
- unblock phase 1 without requiring JAX yet
- make the later JAX dependency explicit instead of implicit

Files:

- `benchmarks/__init__.py`
- `benchmarks/common.py`
- [pyproject.toml](/home/evanzamir/basketworld/pyproject.toml) or [setup.py](/home/evanzamir/basketworld/setup.py) only if/when we decide to declare JAX explicitly

Tasks:

- create a small `benchmarks` package so scripts can share helpers cleanly
- add timing utilities and simple stdout/JSON reporting helpers
- add a benchmark config loader that reuses current env arguments rather than inventing a parallel config format
- add deterministic legal-action generation helpers for benchmark runs
- add an import guard so JAX-dependent scripts fail with a clear message if JAX is not installed

Important constraint:

- the repo does not currently declare `jax` or `jaxlib` in [pyproject.toml](/home/evanzamir/basketworld/pyproject.toml), so phase 1 should not depend on them

Done when:

- `benchmarks/` exists and imports cleanly
- the baseline benchmark can be implemented on top of shared helpers without touching training code
- JAX absence is handled explicitly rather than by a cryptic import error

### Phase 1: Baseline Benchmark

Purpose:

- measure the current env first
- produce the throughput and hotspot data that determines what the JAX prototype must beat

Primary file:

- `benchmarks/rollout_baseline.py`

CLI shape:

```bash
python benchmarks/rollout_baseline.py \
  --episodes 200 \
  --horizon 64 \
  --num-envs 1 \
  --mode throughput \
  --action-mode pregenerated_legal
```

Recommended arguments:

- `--episodes`
- `--horizon`
- `--num-envs`
- `--mode {throughput,hotspot}`
- `--action-mode {random_legal,pregenerated_legal}`
- `--seed`
- `--output-json <path>` optional
- benchmark-specific overrides for the chosen fixed training config

Tasks:

- build one env repeatedly under the chosen benchmark config
- implement throughput mode with profiling disabled
- implement hotspot mode with env profiling enabled
- time resets and steps separately
- report aggregate steps/sec, reset/sec, mean latencies, and top profile sections
- keep policy inference out of the measurement path

Done when:

- the script can run in both throughput and hotspot modes
- it prints stable rollout metrics for a fixed seed/config
- hotspot output identifies the current top rollout sections from the existing env profiler

Artifacts to keep:

- one saved baseline JSON for throughput mode
- one saved baseline JSON for hotspot mode

### Current Baseline Findings (2026-04-10)

The first baseline runs used the current `Train BasketWorld` env semantics from
[launch.json](/home/evanzamir/basketworld/.vscode/launch.json):

- `pass_mode=pointer_targeted`
- `players=3`
- `court_rows=9`, `court_cols=8`
- `use_set_obs=true`
- `enable_intent_learning=true`
- `intent_obs_mode=private_offense`
- `start_template_enabled=true`
- `start_template_library=configs/start_templates_v2.json`
- `start_template_prob=1.0`

The most important throughput result is the current production-style vectorized baseline:

- `runner=subproc_vec`
- `num_envs=16`
- `episodes_per_env=200`
- `horizon=64`
- throughput run:
  - `rollout_steps_per_sec ~= 8335.67`
  - `steps_per_sec ~= 8441.91`
  - `mean_env_step_latency_ms ~= 0.1185`
  - `mean_env_reset_latency_ms ~= 0.0966`
  - `mean_step_call_latency_ms ~= 1.8953`

The clean single-env baseline is still useful for hotspot attribution:

- `runner=sequential`
- `num_envs=1`
- `episodes_per_env=200`
- `horizon=64`
- throughput run:
  - `rollout_steps_per_sec ~= 1995.99`
  - `steps_per_sec ~= 2447.29`
  - `mean_env_step_latency_ms ~= 0.4086`
  - `mean_env_reset_latency_ms ~= 0.6033`

The hotspot runs show a consistent ranking in both the single-env and vectorized paths.

Top recurring hotspots:

- `step`
- `_get_observation`
- `_calculate_shot_probability`
- `reset`
- `_hex_distance`
- `calculate_expected_points_all_players`
- `_calculate_expected_points_for_player`
- `calculate_pass_steal_probabilities`
- `action_masks`

Important interpretation notes:

- The vectorized hotspot run (`subproc_vec`, `16` envs) reached `rollout_steps_per_sec ~= 6973.61`, versus `8335.67` for the unprofiled throughput run.
- This implies roughly `16%` profiling overhead in the production-style vectorized path.
- Hotspot runs should therefore be used for attribution, not as the final speed target.
- In the vectorized path, there were many auto-resets inside `vec_env.step()` (`auto_resets_from_done ~= 32913`), so some reset work is already folded into step cost.
- The sequential and vectorized runners do not have identical semantics:
  - sequential mode ends the episode on `done`
  - vectorized mode measures fixed rollout windows with auto-reset on `done`
- Because of that, the `1`-env sequential throughput number should not be compared directly to the `16`-env vectorized throughput number as a scaling ratio.

Current benchmark implication:

- The main practical bar for a JAX rewrite is the current unprofiled production-style baseline:
  - beat `~8.3k` rollout env-steps/sec under the same env semantics
- A transition-only JAX kernel is unlikely to be sufficient evidence on its own.
- The hotspot rankings suggest that a credible JAX prototype must absorb not only transition logic but also:
  - observation generation
  - action-mask generation
  - repeated geometry / distance calculations
  - shot-probability and expected-points calculations
  - pass steal-probability calculations

### Initial JAX Kernel Findings (2026-04-11)

The first JAX benchmark slice is now running on CPU with `jax` / `jaxlib` installed in the
project virtualenv.

Implemented and validated so far:

- import-safe JAX benchmark scaffold in
  [benchmarks/jax_kernel.py](/home/evanzamir/basketworld/benchmarks/jax_kernel.py)
- fixed-state parity tests in
  [tests/test_jax_transition_equivalence.py](/home/evanzamir/basketworld/tests/test_jax_transition_equivalence.py)
- reduced-scope legal-action transition kernel in
  [benchmarks/jax_kernel.py](/home/evanzamir/basketworld/benchmarks/jax_kernel.py)
- current parity scope:
  - pointer-targeted action masks
  - shot probability / pressure / expected-points profiles
  - pass steal-probability computation
  - raw observation vector assembly
  - reduced-scope `step_batch` transition parity for noop/move, pointer pass, and shot

Initial validation result:

- `4` parity tests passed with JAX installed
- `7` parity tests now pass after adding reduced-scope transition parity

Initial kernel benchmark result:

- fixed benchmark config based on current `Train BasketWorld` env semantics
- device reported by JAX: `TFRT_CPU_0`
- kernel batch size: `256`
- timed kernels:
  - `action_masks ~= 4.44M states/sec`
  - `shot_profiles ~= 2.34M states/sec`
  - `offense_expected_points ~= 2.86M states/sec`
  - `turnover_probabilities ~= 20.57M states/sec`
  - `pass_steal_probabilities ~= 9.90M states/sec`
  - `raw_observation_vector ~= 2.07M states/sec`

Initial reduced-scope transition benchmark result:

- same fixed `3v3` pointer-targeted benchmark config, but with these features disabled to
  match the currently implemented `step_batch` scope:
  - `enable_phi_shaping=false`
  - `illegal_defense_enabled=false`
  - `offensive_three_seconds_enabled=false`
- batch size: `256`
- timed reduced-scope transition kernels:
  - `step_batch_minimal ~= 298k states/sec`
  - `rollout_like_minimal ~= 311k states/sec`

Initial adapter benchmark result:

- same fixed `3v3` pointer-targeted benchmark config and same reduced transition scope
- first adapter-oriented run used `kernel_batch_size=256`, `warmup_iters=2`, `benchmark_iters=10`
- additional metrics:
  - `set_observation_payload ~= 1.58M states/sec`
  - `sb3_payload_minimal_device ~= 365k states/sec`
  - `sb3_payload_minimal_host ~= 166k states/sec`

Interpretation of the first adapter result:

- host transfer plus Python payload assembly is material
- the host-side SB3-style payload path is about `2.2x` slower than the same reduced-scope
  payload staying on device (`365k -> 166k states/sec`)
- but even this first host-side number still sits about `20x` above the current
  `~8.3k rollout env-steps/sec` wrapped Python baseline
- this does **not** yet include:
  - self-play wrapper logic
  - frozen-opponent policy inference
  - SB3 policy inference
  - PPO rollout-buffer bookkeeping

Initial policy/self-play bridge benchmark result:

- first smoke run used the same reduced transition scope and a smaller
  `kernel_batch_size=32` to validate the bridge path end-to-end
- the benchmark now instantiates the real set-attention policy shape and measures:
  - training policy `predict(...)`
  - opponent policy `predict(...)`
  - opponent action-probability extraction
  - illegal-action resolution and full-action assembly
  - end-to-end bridge cost starting from the JAX adapter output
- initial smoke numbers:
  - `training_policy_predict_host ~= 29.6k states/sec`
  - `opponent_policy_predict_host ~= 31.5k states/sec`
  - `opponent_action_probabilities_host ~= 26.7k states/sec`
  - `self_play_bridge_host ~= 8.17k states/sec`
  - `self_play_bridge_with_adapter ~= 6.87k states/sec`

Interpretation of the first bridge result:

- this is the first benchmark slice where the throughput drops into the same range as the
  current wrapped Python baseline (`~8.3k env-steps/sec`)
- the biggest remaining cost center is no longer the JAX env kernel itself
- it is the policy/self-play bridge:
  - batched policy inference on host observations
  - opponent action-probability extraction
  - Python illegal-action resolution and full-action assembly
- this makes the next decision boundary much clearer:
  - widening env rule coverage is not the highest-value next step
  - the highest-value next step is reducing or bypassing the current Python
    self-play / action-resolution bridge

Important caveat:

- these are kernel-only fixed-state throughput numbers
- they are not directly comparable to the current `~8.3k rollout env-steps/sec`
  production-style baseline
- the current JAX slice does **not** yet include:
  - reset logic
  - full-scope transition / `step_batch`
  - SB3/Gym adapter overhead
  - intent observation fields
  - set-observation wrapper assembly

Current implication:

- the hotspot-heavy derived computations still appear highly promising in batched JAX
- raw observation assembly remains in the same throughput range as shot-profile computation,
  which is encouraging because observation work was one of the major Python hotspots
- a compiled legal-action `step_batch` now exists and already benchmarks in the
  `~300k states/sec` range for the reduced transition scope, which materially lowers the
  implementation risk of the next rollout-equivalent phase
- the first host-side adapter benchmark suggests that the JAX-to-NumPy boundary is a real
  cost center, but not obviously a rewrite-killer on its own
- the first policy/self-play bridge benchmark suggests that the current Python bridge logic
  can erase most of the raw JAX env win unless that layer is redesigned too
- the next credibility step is to move from fixed-state derived kernels toward
  current-stack-equivalent measurement:
  - measure the current Python self-play bridge more precisely and identify the dominant
    sub-costs inside that layer
  - test whether a lower-Python bridge design can recover the JAX env advantage
  - only then decide how much value there is in widening env rule coverage immediately

Current reduced transition scope caveat:

- the new `step_batch` currently assumes:
  - legal actions only
  - pointer-targeted pass mode
  - phi shaping disabled
  - illegal defense disabled
  - offensive three-seconds disabled
- this is useful for phase-2 transition validation, but it is not yet sufficient for a
  full apples-to-apples comparison against the current production benchmark config

### Final Findings And Decision (2026-04-11)

The experiment now has enough data to make a practical decision about the current
training architecture.

Most important measured results:

- current wrapped Python env baseline, production-style env-only path:
  - `subproc_vec`, `16` envs
  - `rollout_steps_per_sec ~= 8301.62`
- reduced-scope JAX env plus host payload path:
  - `sb3_payload_minimal_host ~= 281,599 states/sec`
- reduced-scope JAX env plus current host-side policy/self-play bridge:
  - `training_policy_predict_host ~= 102,040 states/sec`
  - `opponent_policy_predict_host ~= 107,661 states/sec`
  - `opponent_action_probabilities_host ~= 99,270 states/sec`
  - `self_play_bridge_host ~= 17,981 states/sec`
  - `self_play_bridge_with_adapter ~= 13,807 states/sec`

What these results mean:

- the JAX env kernels themselves are not the problem
- the large kernel-level win survives through the env-to-host payload boundary
- the win then compresses sharply once the current host-side rollout stack is included:
  - policy inference
  - opponent action-probability extraction
  - Python illegal-action resolution
  - self-play action assembly

This is the key conclusion of the experiment:

- a batched JAX env alone is not enough to make the current SB3/PyTorch rollout stack
  compellingly faster
- the current architecture keeps too much work in the Python / PyTorch / wrapper path
  after env stepping

Important comparison caveat:

- `self_play_bridge_with_adapter ~= 13.8k states/sec` is **not** a strict apples-to-apples
  comparison against the `~8.3k` env-only baseline, because the JAX bridge number includes
  policy and self-play work that the env-only baseline excludes
- this means `~1.66x` should not be interpreted as a hard end-to-end ceiling
- but it **does** show that simply replacing the env while keeping the current host-side
  rollout stack mostly intact is unlikely to justify the rewrite effort

Additional current-stack benchmark result:

- a dedicated Python policy/self-play benchmark was added in
  [benchmarks/policy_bridge_baseline.py](/home/evanzamir/basketworld/benchmarks/policy_bridge_baseline.py)
- the `dummy_vec` `16`-env run produced:
  - `rollout_steps_per_sec ~= 303.71`
  - `policy_predict_steps_per_sec ~= 11,580.42`
  - `env_step_steps_per_sec ~= 312.84`
- this is **not** a production comparison number, because `DummyVecEnv` serializes env stepping
- it is still useful because it shows the policy forward pass is not the dominant cost in that path;
  the env plus self-play wrapper step dominates once everything is forced through the current Python loop

Subprocess benchmark caveat:

- the new `subproc_vec` policy/self-play benchmark path behaved pathologically in this harness
  and did not produce a trustworthy throughput number within a reasonable time budget
- that path should be debugged before using it as a precise production-style apples-to-apples number
- however, the existing evidence is already strong enough for the present decision

Decision for the current stack:

- stop the env-only JAX rewrite investigation here
- keep the benchmark code and results as documentation
- do **not** invest more work in widening env rule coverage or deeper env parity under the
  current SB3/PyTorch rollout architecture

When to revisit:

- only revisit this direction as a broader rollout-stack redesign, not an env-only rewrite
- a future attempt would likely need to move most rollout-critical inference-time work into JAX too:
  - policy forward pass
  - opponent policy forward pass
  - action-probability extraction
  - self-play action assembly / illegal-action resolution
- in practice that implies either:
  - a much more JAX-native training stack
  - or a substantial port of the current custom PPO/policy/self-play logic to a JAX-based framework
    such as SBX

Bottom line:

- `JAX env only`: technically promising
- `JAX env inside the current training stack`: not promising enough to justify further rewrite work

### SBX Feasibility Prototype (2026-04-12)

The local SBX PPO code suggests a narrower follow-up question is still worth testing:

- not "can we speed up the exact current SB3/PyTorch stack?"
- but "can a nearby JAX-native approximation of the older simpler BasketWorld training setup
  deliver materially higher throughput while still learning acceptably?"

This is a different project from the env-only rewrite. It should be treated as a reduced-scope
prototype, not a continuation of the current-stack parity effort.

#### Why This Is Still Plausible

From the local SBX clone:

- `~/sbx/sbx/ppo/ppo.py` is small and delegates most common PPO behavior to shared JAX
  infrastructure
- `~/sbx/sbx/ppo/policies.py` is also small and already supports `MultiDiscrete`
  action spaces

That said, the small surface is partly because SBX assumes a simpler policy/training stack than
 the current BasketWorld codebase:

- no built-in `MultiInputPolicy` alias in PPO
- no set-attention policy
- no integrated intent-selector PPO
- no custom dual-critic / dual-policy logic
- no BasketWorld-specific pointer-pass distribution or pass-bias behavior

So SBX is best viewed as a promising base for a reduced prototype, not a drop-in replacement for
the current architecture.

#### Prototype Goal

Answer one concrete question:

- if we move both env stepping and policy inference to a JAX-native stack with reduced feature
  scope, do we recover the large theoretical speedup that disappeared in the current
  SB3/PyTorch bridge?

The prototype only needs to be good enough to answer:

1. training throughput
2. rough sample-efficiency / learning viability

It does **not** need current-feature parity.

#### Recommended Scope

Use a deliberately reduced target that is still behaviorally relevant and historically close to the
earlier simpler training setup.

Keep:

- batched JAX env core
- PPO in SBX
- `pointer_targeted` passing if it is considered essential to current gameplay
- one fixed training team, preferably offense-only for the first pass
- one fixed training config
- legal-action masking

Drop for phase 1:

- integrated intent-selector PPO
- runtime intent-conditioning extras
- dual critic / dual policy
- mixed offense/defense training
- current self-play wrapper behavior
- pass-bias schedules unless they are clearly required for learning to start
- current callback / logging complexity

Prefer one of these two policy choices:

1. simpler flat or lightly structured observation encoder
- fastest path to a first answer
- best if the earlier project version learned reasonably well without the current set-attention stack

2. reduced set-style policy in Flax
- use only if flattening the observation would make the comparison too unrealistic
- still omit selector, dual-policy, and other current high-complexity features

#### What Must Be Ported

Minimum required for a meaningful SBX test:

1. JAX-friendly env / vec-env interface
2. BasketWorld observation pipeline for the reduced scope
3. legal-action masking path compatible with SBX policy inference
4. BasketWorld-compatible action distribution for the chosen action space
5. one training script that can run PPO rollouts and updates end to end

Likely first custom BasketWorld pieces:

1. custom SBX policy class
- either flat MLP-based or reduced set-attention

2. custom action distribution if `pointer_targeted` remains in scope
- SBX already supports generic `MultiDiscrete`, but BasketWorld may still need custom routing
  semantics depending on how pointer-targeted pass selection is encoded

3. minimal environment adapter
- enough to expose batched observations, masks, rewards, dones, and resets to SBX

#### Explicit Non-Goals

Do not attempt these in the first prototype:

- current-feature parity
- checkpoint compatibility with current SB3 models
- integrated selector parity
- mixed-team self-play parity
- production-quality logging / MLflow parity
- exact reproduction of current training callbacks

If the reduced prototype is not fast and viable, there is no reason to broaden scope.

#### Concrete Execution Plan

Phase A: Design Freeze

- choose the reduced historical-style config to approximate
- choose one policy family:
  - flat MLP
  - reduced set-style encoder
- decide whether `pointer_targeted` is mandatory in phase 1

Done when:

- there is one exact prototype target config
- the intentionally dropped current features are written down explicitly

Phase B: Inference-Only Policy Port

- implement the reduced SBX policy forward pass in Flax
- support the chosen observation schema
- support legal-action masking
- if needed, implement a reduced custom action distribution for `pointer_targeted`

Done when:

- the policy can run batched inference on sampled env observations
- masked action sampling works
- per-batch inference throughput can be measured on CPU

Phase C: Minimal End-to-End PPO Training Loop

- wire the reduced JAX env path into SBX PPO
- run end-to-end rollouts and updates with one fixed config
- keep logging minimal

Done when:

- one training command runs for at least several PPO updates without crashing
- rollout and update throughput are both measurable

Phase D: Viability Benchmark

Measure:

- rollout env-steps/sec
- total training updates/sec
- wall-clock time per rollout/update cycle
- short-run reward / learning sanity curve

Compare against:

- current env-only Python baseline
- current JAX env-only benchmark
- a historically simpler BasketWorld training baseline if available

Done when:

- we know whether the reduced SBX path is materially faster
- we know whether it learns at all on the reduced target

#### Success Criteria

The reduced SBX prototype is worth pursuing further only if it clears both bars:

1. Throughput bar
- at least `3x` faster than the practical current-stack rollout path we can achieve with the
  reduced feature set

2. Learning bar
- reaches a nontrivial learning signal in a short sanity run
- does not obviously collapse relative to the older simpler training setup it is meant to approximate

If it misses either bar, stop.

#### Stop Conditions

Stop the SBX prototype early if any of these become true:

- the policy port requires near-full reproduction of the current custom PPO/policy stack
- the reduced action-space semantics still demand too much custom distribution machinery
- throughput is only modestly better after moving both env and policy inference into JAX
- the reduced prototype does not learn even at a coarse sanity-check level

#### Recommended First Slice

The best first slice is not a full trainer. It is:

1. choose the reduced target config
2. implement one reduced SBX policy forward pass
3. feed it real BasketWorld batched observations
4. measure inference throughput with masking
5. only then decide whether to wire it into PPO

This is the cheapest way to test whether the SBX route is genuinely lower-friction than it looks.

#### Phase A Design Freeze

The following decisions are now frozen for the first SBX feasibility pass.

These choices are intentionally biased toward:

- minimizing porting work
- approximating the older simpler project setup
- preserving enough BasketWorld structure to keep the result meaningful

They are **not** intended to match the current production stack.

##### Frozen Prototype Target

Environment semantics:

- keep current core `3v3` court geometry:
  - `players=3`
  - `court_rows=9`
  - `court_cols=8`
- keep these main env values from the current benchmark config:
  - `shot_clock=24`
  - `min_shot_clock=14`
  - `layup_pct=0.60`
  - `three_pt_pct=0.37`
  - `dunk_pct=0.60`
  - `three_point_distance=4.25`
  - `three_point_short_distance=3`
  - `three_pt_extra_hex_decay=0.05`
  - `shot_pressure_enabled=true`
  - `shot_pressure_max=0.25`
  - `shot_pressure_lambda=1.0`
  - `shot_pressure_arc_degrees=300`
  - `defender_pressure_distance=3`
  - `defender_pressure_turnover_chance=0.02`
  - `defender_pressure_decay_lambda=1`
  - `base_steal_rate=0.3`
  - `steal_perp_decay=1.5`
  - `steal_distance_factor=0.2`
  - `spawn_distance=4`
  - `max_spawn_distance=7`
  - `defender_spawn_distance=2`
  - `defender_guard_distance=1`
  - `assist_window=3`
  - `mask_occupied_moves=false`
  - `enable_pass_gating=true`
- drop current template / selector / set-observation complexity

Architecture and training semantics:

- `use_set_obs=false`
- `use_dual_policy=false`
- `use_dual_critic=false`
- `enable_intent_learning=false`
- `enable_defense_intent_learning=false`
- `intent_selector_enabled=false`
- `intent_diversity_enabled=false`
- `start_template_enabled=false`
- `enable_phi_shaping=false`

Training scope:

- offense-only training for phase 1
- no current self-play wrapper parity requirement
- non-training team handled by a fixed legal-random opponent in phase 1

##### Policy Family Choice

Phase-1 policy family:

- flat MLP over the standard non-set observation path

Reason:

- SBX PPO currently has a much simpler stock policy surface than the current BasketWorld
  set-attention stack
- a flat MLP gives the cheapest path to answer the real question:
  "does moving both env stepping and policy inference into a JAX-native stack recover large speedups?"
- if this path is not clearly fast and viable, there is no reason to spend time porting
  set-attention next

Explicitly deferred:

- reduced Flax set-attention policy

That becomes phase-2 only if the flat policy prototype is promising.

##### Pass-Mode Decision

Phase-1 pass mode:

- `directional`

Reason:

- it matches the older simpler project style and the parser defaults
- it avoids spending the first prototype slice on custom pointer-targeted action-distribution work
- it gives a cleaner test of whether SBX plus a reduced BasketWorld stack is fundamentally promising

Explicitly not required in phase 1:

- `pointer_targeted`

Revisit `pointer_targeted` only if the directional prototype is both:

- materially faster
- still learns acceptably

##### Opponent / Control Simplification

Phase-1 opponent handling:

- do **not** port the current self-play wrapper behavior
- use a simpler opponent/control scheme that keeps the prototype easy to wire into SBX

Frozen phase-1 choice:

1. fixed legal-random opponent for non-training players

Deferred:

1. scripted opponent
2. frozen-policy opponent path

Reason:

- the current self-play bridge was one of the main bottlenecks in the JAX env investigation
- reproducing it early would recreate exactly the complexity we are trying to bypass

##### Immediate Deliverable For Phase A

The first concrete deliverable after this design freeze should be:

1. a short SBX prototype spec with the exact reduced config
2. one flat-observation preprocessing path for that config
3. one flat MLP SBX policy forward pass with legal-action masking

Done when:

- the reduced scope is written down unambiguously
- there is no open question about set obs vs flat obs
- there is no open question about directional vs pointer-targeted for phase 1
- there is no open question about whether current self-play parity is in scope for phase 1

##### Phase A Initial Implementation (2026-04-12)

The first executable Phase-A prototype is now implemented in:

- [benchmarks/sbx_phase_a.py](/home/evanzamir/basketworld/benchmarks/sbx_phase_a.py)
- [tests/test_sbx_phase_a.py](/home/evanzamir/basketworld/tests/test_sbx_phase_a.py)

Important implementation choice:

- the project venv currently has `jax` but does not have `flax`
- so the first slice is a dependency-light JAX MLP benchmark, not a full SBX/Flax policy class yet

What this prototype covers:

- frozen reduced config from Phase A
- flat observation preprocessing using:
  - `obs`
  - `role_flag`
  - `skills`
- explicit exclusion of `action_mask` from policy inputs
- legal-action masking applied at action selection time
- policy head sized only for the training team players (`offense` in phase 1), not all players

Current smoke validation result:

- `flat_obs_dim = 91`
- training-player IDs: `[0, 1, 2]`
- action dim per controlled player: `14`
- small smoke benchmark at batch size `32` reached roughly:
  - `policy_forward_masked ~= 161k states/sec`

First comparable batch-size result:

- JAX flat policy benchmark at batch size `256`:
  - `policy_forward_masked ~= 816k states/sec`
- matching Torch flat policy benchmark at batch size `256`:
  - `policy_forward_masked ~= 582k states/sec`

Initial inference-side speedup estimate for the reduced Phase-A policy path:

- `816k / 582k ~= 1.40x`

Interpretation:

- this is only a first inference-side sanity check
- it is useful because it validates the reduced observation schema and action-head assumptions
- it does **not** yet answer the full SBX feasibility question
- the first apples-to-apples policy-only comparison suggests a modest CPU inference win,
  not a dramatic one

Next useful step after this point:

1. compare this reduced JAX flat-policy path against a matching Torch non-set policy forward-pass baseline
2. if still promising, either:
   - install `flax`/`optax` and port this exact schema into a real SBX-style policy class
   - or continue using this harness to estimate rollout-time inference cost before wiring PPO

#### What A Larger JAX-Native Redesign Would Require

The current evidence suggests that a larger redesign would be needed to recover large rollout
speedups.

Why:

- env-only JAX kernels were very fast
- reduced policy-only JAX inference is only modestly faster than Torch (`~1.4x`)
- earlier bridge benchmarks showed that much of the env-side win disappeared once rollout logic
  returned to host-side Python/Torch code

That means the missing speedup is not likely to come from a simple "swap in JAX for policy
predict" change.

The broader issue is that the current rollout path still contains several host-side stages that
sit outside the compiled env kernel:

- action-probability extraction to host arrays/lists
- illegal-action repair after policy prediction
- self-play action assembly in Python
- per-step rollout plumbing in Python

In principle, all of these can be addressed by a more fused JAX-native design.

The key architectural shift would be:

- do **not** treat the env, policy, masking, and action assembly as separate host-side stages
- instead keep the whole action-selection path on device, with fixed-shape arrays, inside one
  compiled rollout function

##### Why This Redesign Is Needed

The smaller intermediate designs tested so far are not enough:

1. JAX env + current Python/Torch rollout stack
- not enough, because the host-side bridge eats most of the env-side win

2. JAX env + JAX policy, but still with Python-side self-play / masking / action assembly
- likely still not enough, because the isolated policy-forward win is only modest

3. JAX env + fused JAX rollout loop
- this is the first design that has a realistic chance of preserving a large fraction of the
  theoretical env-side gain

So if this direction is ever revisited, it should be framed as:

- a mostly JAX-native rollout redesign

not:

- an env rewrite
- or a light SBX port with the rest of the rollout path unchanged

##### What That Might Look Like

Below are concrete examples of how the remaining host-side bottlenecks would need to change.

1. Action-probability extraction

Current pattern:

- policy logits or distributions are materialized
- probabilities are extracted to host arrays or Python lists
- downstream logic consumes those host-side values

JAX-native transformation:

- keep probabilities on device as fixed-shape arrays, for example `probs[B, P, A]`
- consume them directly inside the masked sampling / action-selection path

Example:

```python
masked_probs = jnp.where(action_mask, probs, 0.0)
sampled_actions = sample_masked(masked_probs, rng_key)
```

This avoids:

- `.cpu().numpy()` / host copies
- Python list construction per player

2. Illegal-action resolution

Current pattern:

- sample or predict actions first
- repair illegal actions afterward in Python / NumPy using masks

JAX-native transformation:

- make legality part of the distribution itself
- never emit illegal actions in the normal path

Example:

```python
masked_logits = jnp.where(action_mask, logits, -1e9)
actions = jax.random.categorical(rng_key, masked_logits, axis=-1)
```

Fallback semantics such as "noop when nothing is legal" can also stay on device:

```python
has_legal = jnp.any(action_mask, axis=-1)
actions = jnp.where(has_legal, actions, 0)
```

This deletes the host-side repair pass entirely.

3. Self-play action assembly

Current pattern:

- build opponent observation on host
- call training and opponent policies separately
- resolve each side's actions separately
- build the full action vector in Python

JAX-native transformation:

- compute both sides inside one compiled function
- produce the full per-env action tensor on device

Example:

```python
training_logits = training_policy(obs_training)
opponent_logits = opponent_policy(obs_opponent)
training_actions = masked_sample(training_logits, training_mask, key_a)
opponent_actions = masked_sample(opponent_logits, opponent_mask, key_b)
full_actions = scatter_team_actions(
    training_actions,
    opponent_actions,
    training_ids,
    opponent_ids,
)
```

This removes:

- Python per-env loops
- repeated host-side slicing / copying
- separate post-processing stages for each team

4. Python rollout plumbing

Current pattern:

- step-by-step host orchestration around the env and policy
- repeated Python function calls per rollout step

JAX-native transformation:

- compile the whole rollout step
- ideally compile a full horizon using `lax.scan`

Example:

```python
def rollout_step(carry, _):
    state, policy_state, rng = carry
    obs = build_obs(state)
    mask = build_action_mask(state)
    actions = select_actions(obs, mask, policy_state, rng)
    next_state, reward, done = step_batch(state, actions, rng)
    return (next_state, policy_state, next_rng(rng)), (obs, actions, reward, done)

carry, traj = jax.lax.scan(rollout_step, init_carry, xs=None, length=horizon)
```

This is the real route to removing:

- per-step Python overhead
- host/device synchronization between sub-stages

##### Practical Conclusion

This is **not** the current recommendation.

Given the measured numbers so far, a larger redesign should only be revisited if there is
explicit interest in:

- replacing most of the rollout-time host-side logic
- accepting a substantially larger rewrite
- treating the result as a new JAX-native training stack, not as an incremental optimization of
  the current one

If that is not the goal, the current stop decision still stands.

##### First Incremental Redesign Check: Masked Sampling (2026-04-12)

The first concrete redesign-style checkpoint has now been added to
[benchmarks/jax_kernel.py](/home/evanzamir/basketworld/benchmarks/jax_kernel.py):

- keep the same host-side bridge benchmark structure
- replace "predict, then repair illegal actions" with:
  - get batched policy probabilities
  - apply the legal action mask before sampling
  - sample masked actions directly
  - skip host-side illegal-action repair for both training and opponent actions

This does **not** move the whole bridge into JAX. It is still a host-side benchmark. The point is
to measure whether removing the illegal-action repair path is materially helpful.

Initial smoke result at batch size `64`:

- old repair-based bridge:
  - `self_play_bridge_host ~= 14.5k states/sec`
  - `self_play_bridge_with_adapter ~= 10.6k states/sec`
- masked-sampling bridge:
  - `self_play_bridge_masked_sampling_host ~= 22.7k states/sec`
  - `self_play_bridge_masked_sampling_with_adapter ~= 17.4k states/sec`

Approximate improvement in this smoke run:

- host-side bridge only:
  - `22.7k / 14.5k ~= 1.57x`
- host bridge including adapter:
  - `17.4k / 10.6k ~= 1.64x`

Interpretation:

- this is more encouraging than the policy-only `~1.4x` comparison
- it suggests the repair-style action path was a real source of overhead
- it still does **not** justify restarting the larger project on its own
- but it does support the claim that a more fused action-selection path could matter

Methodological note:

- this was a small smoke benchmark, not a final production-style result
- if this redesign path is explored further, this exact metric should be rerun with a larger timed
  window and compared alongside any future rollout-loop fusion changes

##### Second Incremental Redesign Check: Compiled Rollout Loop (2026-04-12)

The next redesign-style checkpoint has now been added to
[benchmarks/jax_kernel.py](/home/evanzamir/basketworld/benchmarks/jax_kernel.py):

- a pure JAX rollout loop using `lax.scan`
- on-device action selection for every rollout step
- on-device legal-random opponent actions
- a reduced flat JAX policy inside the loop for the training team
- no host-side per-step rollout orchestration

Important limitations of this first version:

- still uses the reduced minimal transition scope
- uses a reduced flat JAX policy, not the current Torch policy stack
- uses a legal-random opponent, not self-play parity

Initial smoke result at batch size `64` and horizon `64`:

- compiled rollout with legal-random actions for both sides:
  - `compiled_rollout_legal_random_minimal ~= 180.9k states/sec`
- compiled rollout with reduced flat JAX policy for the training side and legal-random opponent:
  - `compiled_rollout_phase_a_policy_minimal ~= 161.1k states/sec`

Why this result matters:

- it is the first measured version of redesign item `#2`
- it keeps the rollout loop itself on device instead of paying host-side per-step orchestration
- it is therefore much closer to the kind of architecture a larger JAX-native redesign would
  actually need

Interpretation:

- this is much more promising than the isolated policy-only `~1.4x` result
- it suggests that rollout-loop fusion is a more meaningful lever than policy inference alone
- it is still only a reduced-scope smoke benchmark, not evidence that the full redesign is worth
  doing

Follow-up result at batch size `256` and horizon `64`:

- compiled rollout with legal-random actions for both sides:
  - `compiled_rollout_legal_random_minimal ~= 306.6k states/sec`
- compiled rollout with reduced flat JAX policy for the training side and legal-random opponent:
  - `compiled_rollout_phase_a_policy_minimal ~= 270.7k states/sec`

Relevant bridge baselines from the same batch-`256` run:

- old repair-based bridge with adapter:
  - `self_play_bridge_with_adapter ~= 21.6k states/sec`
- masked-sampling bridge with adapter:
  - `self_play_bridge_masked_sampling_with_adapter ~= 40.9k states/sec`

This makes the current strongest reduced-scope comparison:

- `270.7k / 21.6k ~= 12.5x` versus the old repair-based host bridge
- `270.7k / 40.9k ~= 6.6x` versus the masked-sampling host bridge with adapter

This is the first result in the investigation that clearly qualifies as a major positive signal.

Practical takeaway:

- if this redesign path is explored methodically, the current ordering still makes sense:
  1. masked sampling
  2. compiled rollout loop
  3. only then consider on-device self-play action assembly or broader parity work

Reset realism follow-up (same reduced scope, batch size `256`, horizon `64`):

- the compiled rollout loop now uses a real reduced `reset_batch` sampled on device when
  episodes terminate
- direct reset benchmark:
  - `reset_batch_minimal ~= 213.3k states/sec`
- compiled rollout with legal-random actions for both sides:
  - `compiled_rollout_legal_random_minimal ~= 126.6k states/sec`
- compiled rollout with reduced flat JAX policy for the training side and legal-random opponent:
  - `compiled_rollout_phase_a_policy_minimal ~= 119.9k states/sec`

Relevant bridge baselines from the same run:

- old repair-based bridge with adapter:
  - `self_play_bridge_with_adapter ~= 20.2k states/sec`
- masked-sampling bridge with adapter:
  - `self_play_bridge_masked_sampling_with_adapter ~= 40.9k states/sec`

What changed:

- the earlier fixed-reset compiled-rollout number (`~270.7k`) was materially optimistic
- real reset cost is large enough to cut the compiled-rollout throughput by more than half

What still holds:

- even after paying real reset cost, the compiled rollout remains clearly faster than the
  host-side bridge variants:
  - `119.9k / 20.2k ~= 5.9x` versus the old repair-based bridge with adapter
  - `119.9k / 40.9k ~= 2.9x` versus the masked-sampling bridge with adapter

Interpretation:

- reset handling is a real cost center and has to be included in any honest redesign estimate
- but the main architectural conclusion still survives:
  - keeping the rollout loop on device is materially more important than the smaller isolated
    wins from policy-only JAX or masked sampling alone

Matching reduced Python rollout baseline:

- a new comparison harness now exists in
  [benchmarks/sbx_phase_a_rollout_torch.py](/home/evanzamir/basketworld/benchmarks/sbx_phase_a_rollout_torch.py)
- it uses:
  - batched host-side Torch flat-policy inference
  - legal-random opponent actions
  - per-env Python stepping and reset
  - the same fixed rollout horizon structure as the compiled JAX rollout benchmark

Longer comparison run (same reduced scope, batch size `256`, horizon `64`, matched to the
reset-aware JAX benchmark timing window with `warmup_iters=3`, `benchmark_iters=20`):

- Python rollout with legal-random actions for both sides:
  - `compiled_rollout_python_legal_random_minimal ~= 2.71k states/sec`
- Python rollout with reduced flat Torch policy for the training side and legal-random opponent:
  - `compiled_rollout_python_phase_a_policy_minimal ~= 2.87k states/sec`

Compared against the reset-aware compiled JAX rollout:

- `119.9k / 2.71k ~= 44.3x` versus the reduced Python legal-random rollout
- `119.9k / 2.87k ~= 41.8x` versus the reduced Python flat-policy rollout

Interpretation:

- the compiled JAX rollout is not merely beating the old host bridge abstractions
- it is also beating the matching reduced Python rollout path by a very large multiple
- this materially strengthens the case that the important architectural lever is the compiled
  on-device rollout loop itself, not just any one intermediate micro-optimization
- the longer timing window landed very close to the earlier short-window estimate, which makes the
  `40x+` order-of-magnitude gap look stable rather than a short-run artifact

#### Deployment Note If The Prototype Succeeds

If the reduced SBX prototype shows acceptable training speed and acceptable learning behavior,
it should be possible to test the resulting model in the web app.

Important constraint:

- the blocker is not the frontend
- the blocker is the backend inference contract, which is currently SB3/PPO-specific

Current backend assumptions include:

- SB3/Torch checkpoint loading
- `policy.predict(obs, deterministic=...)`
- action-probability extraction for legal-action resolution
- some tooling that directly accesses policy internals for value/Q inspection

So a successful SBX prototype would need a small backend adapter layer before app testing:

1. `load_sbx_for_inference(...)` equivalent to the current SB3 loader
2. a policy adapter exposing:
   - `predict(obs, deterministic=...)`
   - action probabilities for masked action resolution
3. route-level support for selecting SB3 vs SBX model backends
4. optional fallback or temporary disablement of app features that depend on SB3-specific
   policy internals

Expected feasibility:

- basic manual playtesting in the web app should be feasible if the prototype trains well
- full current app-feature parity is a separate follow-up task
- inference should remain server-side; browser-side deployment is not a phase-1 goal

Recommended follow-up order if the SBX speed/learning test is successful:

1. train one reduced SBX model end to end
2. add the backend inference adapter
3. support live app play against that model
4. only then consider broader app/runtime parity

### Phase 2: Minimal JAX Transition Kernel

Purpose:

- implement only enough JAX state and logic to answer the transition-speed question

Primary file:

- `benchmarks/jax_kernel.py`

Scope:

- one fixed benchmark config
- one player count
- one pass mode
- no rendering
- no backend/UI paths

Tasks:

- define a small immutable config object for the benchmark case
- define a batched pure state container
- implement `reset_batch`
- implement `step_batch`
- keep transition logic explicit and feature-scoped rather than trying to mirror every env mode immediately
- add clear `NotImplementedError` or equivalent for excluded features instead of silently diverging

Done when:

- the kernel can run batched reset and batched step for the chosen config
- a warm JIT run produces a valid transition-only steps/sec number
- excluded features are documented and fail loudly if accidentally used

### Phase 3: Deterministic Parity Harness

Purpose:

- prove that the prototype is close enough to trust benchmark numbers

Primary file:

- `tests/test_jax_transition_equivalence.py`

Tasks:

- construct a small fixed benchmark fixture shared by env and JAX paths
- feed explicit stochastic draws to both implementations where randomness matters
- compare reset parity
- compare transition parity on representative movement, pass, shot, turnover, and terminal cases
- compare raw observation fields for the benchmark config
- compare action masks exactly for the benchmark config

Done when:

- deterministic parity tests pass for the fixed benchmark config
- stochastic branches are controlled externally rather than relying on NumPy vs JAX RNG coincidence
- failures localize the mismatch category clearly

### Phase 4: Obs/Mask And Comparison Harness

Purpose:

- convert the JAX prototype from a transition microbenchmark into a rollout-equivalent benchmark

Primary files:

- `benchmarks/jax_kernel.py`
- `benchmarks/compare_rollout.py`

CLI shape:

```bash
python benchmarks/compare_rollout.py \
  --batch-size 256 \
  --horizon 128 \
  --seed 0 \
  --output-json benchmarks/latest_compare.json
```

Tasks:

- implement `build_obs_batch`
- implement `build_action_mask_batch`
- measure kernel-only performance
- measure transition + obs + mask performance
- measure current-stack-equivalent adapter cost needed to return data to the existing Python/SB3 path
- compute projected rollout and total-training speedups from the measured numbers

Done when:

- one script reports kernel-only, rollout-equivalent, and current-stack-equivalent numbers
- adapter overhead is measured separately instead of hidden inside a single blended number
- results can be saved and compared across runs

### Phase 5: Decision Output

Purpose:

- turn benchmark results into a concrete go/no-go outcome

Primary output:

- benchmark JSON artifacts
- optional short report in `docs/` or `benchmarks/compare_rollout_report.py`

Tasks:

- compare measured speedups against the thresholds in this document
- record the included feature scope for the JAX prototype
- state whether the current-stack-equivalent result justifies more rewrite work

Done when:

- the benchmark produces a decision that is reproducible on the same machine/config
- the report distinguishes near-term current-stack value from longer-term architectural upside

## First Implementation Slice

The first coding tranche should be:

1. create `benchmarks/__init__.py` and `benchmarks/common.py`
2. implement `benchmarks/rollout_baseline.py`
3. capture one throughput run and one hotspot run
4. use those results to finalize the exact fixed benchmark config for the JAX kernel

This is the right first slice because it:

- requires no JAX dependency yet
- gives us real hotspot data before committing to the kernel shape
- reduces the chance of optimizing the wrong part of the env

## Benchmark Scope

The benchmark should measure rollout-side performance only.

It should include:

- reset throughput
- step throughput
- observation generation cost
- action mask generation cost
- end-to-end rollout steps/sec
- current-stack bridge overhead required to feed the existing training stack

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

Run the baseline in two modes:

1. throughput mode
- profiling disabled
- used for steps/sec and reset/sec numbers

2. hotspot mode
- profiling enabled with a fixed sample rate
- used only for section attribution and hotspot ranking

Do not compare absolute throughput numbers from profiled and unprofiled runs as if they were the same measurement.

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

This independence is for the prototype kernel only.

The comparison harness must still account for the cost of adapting JAX state/output back into the current Python + SB3 training path, because that is the near-term decision being made.

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

Report three tiers of results:

1. kernel-only
- transition only

2. rollout-equivalent
- transition + obs + mask inside the prototype path

3. current-stack-equivalent
- any Python-side packing/unpacking, wrapper conversion, and adapter overhead needed to preserve the current training path

### Benchmark Rules

1. Warm up JAX before timing
2. Use the same batch size and horizon
3. Use the same action-generation policy
4. Report both:
- core transition speed
- transition + obs + mask speed
5. Also report current-stack-equivalent speed using the adapter path you would actually need first
6. Include projected end-to-end training speedup using the current rollout/update split

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

8. Observation parity
- raw observation fields match numerically within tolerance for the chosen benchmark config
- include role-conditioned fields that affect rollout semantics

9. Action-mask parity
- legal action masks match exactly for the chosen benchmark config
- include directional vs pointer-targeted pass behavior if relevant to that config

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
5. no wrappers inside the kernel prototype itself
6. no backend-specific state

Likely good initial target:

- current main training configuration
- without optional UI-only paths
- with the exact observation and mask semantics needed by the current training stack for that configuration

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

### 2b. Adapter overhead dominates

If the JAX kernel is fast but converting results back into the current Python + SB3 path is expensive, near-term gains may be much smaller than the kernel benchmark suggests.

Mitigation:

- report a current-stack-equivalent benchmark tier
- keep adapter logic explicit and measured, not hand-waved

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
- separate throughput and hotspot measurements

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
- current-stack-equivalent adapter measurement

Decision:

- estimate rollout speedup and total training speedup ceiling
- estimate realistic near-term speedup under the current SB3 path

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
