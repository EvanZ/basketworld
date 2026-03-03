# Web Playable Mode

This project now supports a public/playable deployment mode intended for web users.

## Runtime Modes

- Frontend default mode: full dev UI (existing controls/tabs)
- Frontend playable mode: public human-vs-AI UI
- Backend default mode: all API routes
- Backend public mode: playable + media-only routes

## Frontend Env

Set in `app/frontend/.env` (or build-time env):

```bash
VITE_APP_MODE=playable
VITE_API_BASE_URL=https://basketworld.toplines.app
```

- `VITE_APP_MODE=playable` mounts `PlayableApp.vue`
- Any other value (or unset) mounts the existing `App.vue`

## Backend Env

Set in `app/backend/.env.app` (loaded automatically by `app/backend/main.py`).
Fallbacks are `app/backend/.env.local` and `app/backend/.env` if those paths are files.

```bash
# Route gating
BW_PUBLIC_MODE=true

# CORS
BACKEND_CORS_ALLOW_ORIGINS=https://basketworld.toplines.app

# Optional MLflow URI (example)
MLFLOW_TRACKING_URI=http://127.0.0.1:5001
```

When `BW_PUBLIC_MODE=true`:

- mounted: `/api/playable/*`, media routes
- not mounted: eval/admin/lifecycle/policy/analytics HTTP routes

## Playable Policy Matrix Config

You can configure playable options through one of two methods.

### Method 1: JSON Matrix

```bash
BW_PLAYABLE_POLICY_MATRIX_JSON='{
  "1": {"easy": {"run_id": "<run-id>", "checkpoint_index": 101}},
  "3": {"medium": {"run_id": "<run-id>", "checkpoint_index": 220}}
}'
```

### Method 2: Explicit Per-Option Vars

```bash
BW_PLAYABLE_1_EASY_RUN_ID=<run-id>
BW_PLAYABLE_1_EASY_CHECKPOINT=101

BW_PLAYABLE_3_MEDIUM_RUN_ID=<run-id>
BW_PLAYABLE_3_MEDIUM_CHECKPOINT=220
```

Rules:

- valid players per side: `1..5`
- valid difficulties: `easy`, `medium`, `hard`
- checkpoint maps to `unified_iter_<checkpoint>.zip`
- missing `RUN_ID` or `CHECKPOINT` makes that option unavailable in UI
- explicit vars override JSON entries

## Playable Behavior Summary

- user controls canonical IDs `0..(n-1)`
- AI controls canonical IDs `n..(2n-1)`
- coin toss decides first possession
- possessions alternate between user and AI offense
- every possession resets with random spawn and 24-second shot clock
- score tracks continuously across possessions
- no terminal winner in v1; user can start a new game
- episode replay/save routes are disabled in backend public mode
- single-board PNG/GIF capture remains available in UI
