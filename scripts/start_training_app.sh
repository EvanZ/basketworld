#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.env/bin/python}"
BACKEND_PORT="${TRAINING_BACKEND_PORT:-8090}"
FRONTEND_PORT="${TRAINING_FRONTEND_PORT:-5174}"

cd "$ROOT"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found at $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -d "$ROOT/app/training_frontend/node_modules" ]]; then
  echo "Training frontend dependencies are missing. Run: cd app/training_frontend && npm install" >&2
  exit 1
fi

"$PYTHON_BIN" -m uvicorn app.training_backend.main:app \
  --host 0.0.0.0 \
  --port "$BACKEND_PORT" \
  --reload &
BACKEND_PID=$!

npm --prefix "$ROOT/app/training_frontend" run dev -- \
  --host 0.0.0.0 \
  --port "$FRONTEND_PORT" &
FRONTEND_PID=$!

cleanup() {
  kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
  wait "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Training app frontend: http://localhost:$FRONTEND_PORT"
echo "Training app API:      http://localhost:$BACKEND_PORT/docs"
echo "Training workers are detached and are not stopped when this launcher exits."

wait -n "$BACKEND_PID" "$FRONTEND_PID"
