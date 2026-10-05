#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.env/bin/python}"
PORT="${TRAINING_APP_PORT:-8090}"

cd "$ROOT"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found at $PYTHON_BIN" >&2
  exit 1
fi

npm --prefix "$ROOT/app/training_frontend" run build

export BW_TRAINING_SERVE_FRONTEND=true
exec "$PYTHON_BIN" -m uvicorn app.training_backend.main:app \
  --host 0.0.0.0 \
  --port "$PORT"
