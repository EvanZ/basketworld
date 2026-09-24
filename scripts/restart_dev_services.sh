#!/usr/bin/env bash
# Restart BasketWorld's local MLflow, FastAPI, and Vite development services.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="${BASKETWORLD_DEV_RUNTIME_DIR:-/tmp/basketworld-dev-services}"
MLFLOW_PORT=5000
BACKEND_PORT=8080
FRONTEND_PORT=5173
HOST=127.0.0.1
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.env/bin/python}"

mkdir -p "$RUNTIME_DIR"

port_listener_pids() {
  local port="$1"
  lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true
}

wait_for_port() {
  local label="$1"
  local port="$2"
  local attempts=30

  for ((attempt = 1; attempt <= attempts; attempt += 1)); do
    if (exec 3<>"/dev/tcp/$HOST/$port") 2>/dev/null; then
      exec 3>&-
      exec 3<&-
      return 0
    fi
    sleep 1
  done

  echo "Timed out waiting for $label on $HOST:$port." >&2
  return 1
}

stop_port_listener() {
  local label="$1"
  local port="$2"
  local pid
  local pids

  pids="$(port_listener_pids "$port")"
  if [ -z "$pids" ]; then
    return 0
  fi

  echo "Stopping existing $label listener(s) on port $port: $pids"
  while read -r pid; do
    [ -n "$pid" ] || continue
    kill -TERM "$pid" 2>/dev/null || true
  done <<< "$pids"

  for ((attempt = 1; attempt <= 10; attempt += 1)); do
    [ -z "$(port_listener_pids "$port")" ] && return 0
    sleep 1
  done

  pids="$(port_listener_pids "$port")"
  if [ -n "$pids" ]; then
    echo "Force-stopping remaining $label listener(s) on port $port: $pids" >&2
    while read -r pid; do
      [ -n "$pid" ] || continue
      kill -KILL "$pid" 2>/dev/null || true
    done <<< "$pids"
  fi
}

stop_managed_process_group() {
  local label="$1"
  local pid_file="$2"
  local port="$3"
  local pid
  local listener
  local group_id
  local listener_group_id

  [ -f "$pid_file" ] || return 0
  pid="$(<"$pid_file")"
  rm -f "$pid_file"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 0
  kill -0 "$pid" 2>/dev/null || return 0
  group_id="$(ps -o pgid= -p "$pid" | tr -d ' ')"

  # Only target the full process group when it still owns this service's
  # listener. This prevents a stale PID file from killing an unrelated process.
  while read -r listener; do
    [ -n "$listener" ] || continue
    listener_group_id="$(ps -o pgid= -p "$listener" | tr -d ' ')"
    if [ "$group_id" = "$listener_group_id" ]; then
      echo "Stopping managed $label process group $group_id"
      kill -TERM -- "-$group_id" 2>/dev/null || true
      return 0
    fi
  done <<< "$(port_listener_pids "$port")"
}

start_service() {
  local label="$1"
  local working_dir="$2"
  local pid_file="$3"
  local log_file="$4"
  shift 4

  echo "Starting $label (log: $log_file)"
  (
    cd "$working_dir"
    setsid "$@" >"$log_file" 2>&1 &
    echo "$!" >"$pid_file"
  )
}

show_start_failure() {
  local label="$1"
  local log_file="$2"
  echo "$label failed to start. Recent log output:" >&2
  tail -n 80 "$log_file" >&2 || true
}

MLFLOW_PID_FILE="$RUNTIME_DIR/mlflow.pid"
BACKEND_PID_FILE="$RUNTIME_DIR/backend.pid"
FRONTEND_PID_FILE="$RUNTIME_DIR/frontend.pid"
MLFLOW_LOG_FILE="$RUNTIME_DIR/mlflow.log"
BACKEND_LOG_FILE="$RUNTIME_DIR/backend.log"
FRONTEND_LOG_FILE="$RUNTIME_DIR/frontend.log"

stop_managed_process_group "MLflow" "$MLFLOW_PID_FILE" "$MLFLOW_PORT"
stop_managed_process_group "backend" "$BACKEND_PID_FILE" "$BACKEND_PORT"
stop_managed_process_group "frontend" "$FRONTEND_PID_FILE" "$FRONTEND_PORT"
stop_port_listener "MLflow" "$MLFLOW_PORT"
stop_port_listener "backend" "$BACKEND_PORT"
stop_port_listener "frontend" "$FRONTEND_PORT"

# This deliberately invokes the existing profile-aware launcher so MLflow uses
# the identical AWS profile, SQLite store, and S3 artifact destination.
start_service "MLflow" "$ROOT" "$MLFLOW_PID_FILE" "$MLFLOW_LOG_FILE" \
  "$ROOT/start_mlflow_with_profile.sh"
if ! wait_for_port "MLflow" "$MLFLOW_PORT"; then
  show_start_failure "MLflow" "$MLFLOW_LOG_FILE"
  exit 1
fi

start_service "backend" "$ROOT" "$BACKEND_PID_FILE" "$BACKEND_LOG_FILE" \
  "$PYTHON_BIN" -m uvicorn app.backend.main:app --reload --host "$HOST" --port "$BACKEND_PORT"
if ! wait_for_port "backend" "$BACKEND_PORT"; then
  show_start_failure "backend" "$BACKEND_LOG_FILE"
  exit 1
fi

start_service "frontend" "$ROOT/app/frontend" "$FRONTEND_PID_FILE" "$FRONTEND_LOG_FILE" \
  npm run dev -- --host "$HOST" --port "$FRONTEND_PORT" --strictPort
if ! wait_for_port "frontend" "$FRONTEND_PORT"; then
  show_start_failure "frontend" "$FRONTEND_LOG_FILE"
  exit 1
fi

echo
echo "BasketWorld development services are ready:"
echo "  MLflow:   http://$HOST:$MLFLOW_PORT"
echo "  Backend:  http://$HOST:$BACKEND_PORT"
echo "  Frontend: http://$HOST:$FRONTEND_PORT"
echo "Logs and PID files: $RUNTIME_DIR"
