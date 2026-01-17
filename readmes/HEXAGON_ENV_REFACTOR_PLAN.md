# HexagonBasketballEnv Refactor Plan

Goal: split `basketworld/envs/basketworld_env_v2.py` (~2500 LOC) into cohesive modules/utilities while preserving exact behavior, API, and serialized artifacts (policies/checkpoints). Add regression tests to guard core mechanics.

## Current Shape (to preserve)
- Env class `HexagonBasketballEnv`: gym-compatible API (`reset/step/render`), attributes (`positions`, `ball_holder`, `offense_ids/defense_ids`, `shot_clock`, etc.), config params (spawn distances, dunk flags, pressure settings, pass/shot reward knobs, illegal action policy), and side effects (`last_action_results`, `shot_probs`, `_three_point_hexes`, etc.).
- Key subsystems: initialization/reset logic, action masking, movement/shot/pass resolution, reward computation (including phi shaping hooks), episode termination, rendering helpers, pass/steal probabilities, hex geometry helpers, skill sampling/assignment, illegal move handling.

## Target Module Structure
- `basketworld/envs/core/geometry.py` — hex grid math: axial coords, distance, neighbors, bounds checks, three-point line/lane calculations, pressure arcs, pathfinding helpers if present.
- `basketworld/envs/core/state.py` — lightweight data structs for positions/skills/shot clock, utility to clone/restore state, random spawn helpers, offense/defense assignment, sampled skills persistence.
- `basketworld/envs/core/actions.py` — action mask building, illegal action policy handling, movement validation, pass targeting strategies, turnover checks.
- `basketworld/envs/core/shooting.py` — shot probability computation, pressure adjustment, dunk rules, shot logging payloads, EP helpers, shot clock updates.
- `basketworld/envs/core/passing.py` — pass resolution, steal probability, assist/potential assist tracking.
- `basketworld/envs/core/rewards.py` — reward calculation (base + shaping), phi-related per-step values, aggregation by team, turnover/violation rewards.
- `basketworld/envs/core/rendering.py` — render_frame helpers, PNG/frame generation, any GIF/export helpers (if still in env).
- `basketworld/envs/hex_env.py` — the public `HexagonBasketballEnv` class that wires all helpers together; maintains existing attributes and method signatures, delegates to core modules.

## Refactor Phases
1) **Survey & map**: add a temporary checklist mapping functions/blocks in `basketworld_env_v2.py` to target modules; note shared state/attributes.
2) **Extract pure utilities**: move geometry helpers (distance, neighbors, bounds, three-point sets) to `core/geometry.py`. Update imports in-place, no behavior change.
3) **Action mask & validation**: extract mask computation and illegal action handling to `core/actions.py`. Keep mask shapes and NOOP handling identical.
4) **Movement & positioning**: extract movement resolution, spawn/reset logic, lane/three-second tracking to `core/state.py` and `core/actions.py` as appropriate.
5) **Shooting module**: move shot probability calc, pressure modifiers, dunk handling, shot logging payload construction to `core/shooting.py`. Preserve RNG usage/order.
6) **Passing/steals**: extract pass resolution, steal probability, assist/potential assist bookkeeping to `core/passing.py`.
7) **Rewards/shaping**: move reward aggregation (offense/defense), violation/turnover rewards, phi shaping hooks to `core/rewards.py`. Ensure env exposes `last_action_results` unchanged.
8) **Rendering utilities**: move rendering helpers to `core/rendering.py`; keep `render()` surface compatible.
9) **Wrap in `hex_env.py`**: reassemble `HexagonBasketballEnv` by delegating to extracted modules; keep public API, attributes, and defaults identical. Leave a thin shim in `basketworld_env_v2.py` importing the new class for backward compatibility.
10) **Tests & verification**: add regression tests before each major move; maintain green as code migrates.

## Test Plan (add under `tests/env/`)
- **API smoke**: instantiate env with default params; ensure attributes exist (`positions`, `offense_ids`, `shot_clock`, `last_action_results`); run `reset/step` no-op actions for a few steps without error.
- **Action masks**: verify mask shape and illegal action masking behavior matches current env for a fixed seed; check NOOP allowed.
- **Movement**: move a player to boundary/out-of-bounds; assert positions and turnover/violation behavior unchanged.
- **Shooting**: seed RNG, force a shot action; assert `last_action_results["shots"]` fields (probability, distance, success flag shape) unchanged.
- **Passing/steals**: compute `calculate_pass_steal_probabilities` for a fixed layout; compare against fixture values.
- **Rewards**: run a scripted mini-episode to compare offense/defense reward sums and `phi_r_shape` logs to current implementation.
- **Rendering**: basic `render()` returns an array with expected shape (if available in headless test).
- **Parallel eval compatibility**: ensure `_init_evaluation_worker` can import/recreate env with new module paths (integration smoke).

## Safety/compat
- Keep class name/module path (`basketworld.envs.basketworld_env_v2.HexagonBasketballEnv`) available via import shim.
- Preserve RNG call order (no reordering of random draws).
- Avoid renaming public attributes or altering `info`/`last_action_results` payload shapes.
- Keep illegal action strategy and action space enum values identical.

## Next Steps
- Lock in this plan, then start with geometry extraction + tests. Move module-by-module with CI/tests after each phase. Preserve git history to aid bisecting if regressions appear.
