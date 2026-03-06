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

# Playable concurrency controls
BW_PLAYABLE_MAX_ACTIVE_SESSIONS=8
BW_PLAYABLE_SESSION_TTL_MINUTES=120

# Optional playable analytics export (S3 JSONL batches)
BW_ANALYTICS_S3_ENABLED=true
BW_ANALYTICS_S3_BUCKET=basketworld
BW_ANALYTICS_S3_PREFIX=basketworld/playable-analytics
BW_ANALYTICS_S3_FLUSH_EVENTS=50
BW_ANALYTICS_S3_FLUSH_SECONDS=15
BW_ANALYTICS_S3_MAX_QUEUE_EVENTS=5000
BW_ANALYTICS_ENVIRONMENT=prod
BW_ANALYTICS_DEBUG_ENABLED=false
BW_ANALYTICS_DEBUG_BUFFER_EVENTS=200

# Optional MLflow URI (example)
MLFLOW_TRACKING_URI=http://127.0.0.1:5001
```

When `BW_PUBLIC_MODE=true`:

- mounted: `/api/playable/*`, media routes
- not mounted: eval/admin/lifecycle/policy/analytics HTTP routes

## Playable Session Isolation

- each browser client is assigned a playable session ID
- frontend sends this via `X-Playable-Session-Id` on playable requests
- backend isolates game state per session ID (no cross-user overwrites)
- once active sessions hit `BW_PLAYABLE_MAX_ACTIVE_SESSIONS`, `/api/playable/start` returns `429`
- idle sessions are evicted after `BW_PLAYABLE_SESSION_TTL_MINUTES`

## Playable Analytics Export

- when `BW_ANALYTICS_S3_ENABLED=true`, backend emits analytics events to S3 in `.jsonl.gz` batches
- credentials follow the same AWS chain already used by MLflow (`MLFLOW_AWS_*` then `AWS_*`, then IAM role)
- uploads are asynchronous and non-blocking for gameplay
- object key pattern:
  - `<prefix>/date=YYYY-MM-DD/hour=HH/events_<epoch_ms>_<host>_<id>.jsonl.gz`

Event envelope (`bw.analytics.v1`):

```json
{
  "schema_version": "bw.analytics.v1",
  "event_type": "turn_resolved",
  "event_id": "uuid",
  "event_ts": "2026-03-06T20:40:12.345Z",
  "session_id": "playable-session-id",
  "game_id": "uuid",
  "seq": 42,
  "app_mode": "playable",
  "environment": "prod",
  "game_config": {
    "players_per_side": 3,
    "difficulty": "hard",
    "period_mode": "quarters",
    "period_length_minutes": 5,
    "policy_run_id": "<run_id>",
    "policy_checkpoint_index": 200,
    "pass_mode": "pointer_targeted"
  },
  "payload": {}
}
```

Current `event_type` values:

- `game_started`
- `turn_submitted`
- `turn_resolved`
- `period_ended`
- `game_ended`

Optional local debug endpoint:

- enable with `BW_ANALYTICS_DEBUG_ENABLED=true`
- inspect recent queued/uploaded events:
  - `GET /api/playable/analytics_debug?limit=50`

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
