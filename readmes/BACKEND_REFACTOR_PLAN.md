# Backend Refactor Plan (FastAPI)

## Goals
- Shrink `app/backend/main.py` (~4500 LOC) into focused modules to improve readability and change velocity.
- Make game lifecycle, policy/MLflow integration, evaluation/MCTS, and admin/debug controls testable in isolation.
- Preserve API compatibility and current behavior while creating seams for future multi-user/session work.

## Target Structure (implemented)
- `app/backend/state.py`: `GameState`, turn snapshots, reward/history tracking, serialization helpers (`get_full_game_state`, phi logs).
- `app/backend/schemas.py`: all Pydantic models (Init/Action/MCTS/phi params/skills/position updates/etc.).
- `app/backend/observations.py`: role-flag/obs cloning, EP/shot/pass probability helpers, phi recompute helpers, Q/state-value calculators.
- `app/backend/policies.py`: MLflow interaction (list/download), policy loading/swap, param counts, action prediction/masking utilities.
- `app/backend/evaluation.py`: sequential/parallel evaluation (1000+ episodes spawn up to 16 workers), worker init/state, episode save/replay helpers.
- `app/backend/mcts.py`: MCTSAdvisor and related helpers (hashing, priors, rollout).
- `app/backend/media.py`: GIF/PNG episode saving utilities.
- `app/backend/routes/`:
  - `lifecycle_routes.py`: init game, step/action (restored legacy behavior), reset turn, self-play, mcts_advise alias, init phi log seed.
  - `policy_routes.py`: phi params/log, rewards, action/state values, policy probabilities.
  - `admin_routes.py`: skills overrides, shot clock/pass target adjustments, position updates, policy swap.
  - `evaluation_routes.py`: run evaluation, pass_steal_preview; wraps `evaluation.py`.
  - `media_routes.py`: save_episode, save_episode_from_pngs, replay helpers.
  - `analytics_routes.py`: shot stats, rewards, replay_last_episode.
- `app/backend/main.py`: FastAPI app setup + router registration only.

## Refactor Phases
1) **Baseline map**: annotate current functions/endpoints with target modules; note shared globals (`game_state`, `_worker_state`). Write a short mapping comment or checklist to track moves.
2) **Extract pure helpers**: obs/role/phi/value calculators moved to `observations.py`.
3) **Policies/MLflow seam**: list/download/swap helpers and param counting in `policies.py`.
4) **State + serialization**: `GameState`, turn snapshots, reward/history serialization, `get_full_game_state` in `state.py` (single shared instance).
5) **Routerization**: endpoints grouped in `routes/` (lifecycle, policy, admin, evaluation, media, analytics); legacy shapes restored where needed.
6) **Evaluation/MCTS**: worker init/run eval/MCTSAdvisor in `evaluation.py`/`mcts.py`; parallel path reinstated for large runs; mcts_advise endpoint restored to legacy signature.
7) **Main cleanup**: `main.py` is app setup + router include; no feature logic.
8) **Testing pass**: step route serialization test added; manual smoke via init_game/step/rewards/eval; pending broader API test suite.

## Migration Notes / Risks
- Shared mutable `game_state` stays in `state.py`; all routers import that singleton.
- Deep copies in MCTS/Q calc remain; monitor perf.
- MLflow URIs/policy paths centralized in `policies.py`; init_game keeps role_flag compatibility.
- Parallel eval workers rely on `evaluation.py` init; large runs now spawn up to 16 workers.
- Backward-compat JSON shapes restored for step/action and evaluation results (final_state, last_action_results).

## Verification Checklist
- Uvicorn import/app start OK (`uvicorn main:app`).
- Endpoints: `/api/init_game` (MLflow params load), `/api/step` (legacy overrides + reward logging), `/api/rewards` (phi shaping with MLflow params), `/api/mcts_advise`, `/api/run_evaluation` (sequential + parallel >1000 episodes), media save/replay.
- Unit: step serialization test passing; add more API coverage if time.
- Confirm no circular imports; routers only pull from shared state/helpers.
