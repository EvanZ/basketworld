#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND_DIR="$ROOT_DIR/app/frontend"
WEB_ROOT="/var/www/basketworld"
BACKEND_SERVICE="basketworld-backend"
MLFLOW_SERVICE="mlflow-basketworld"

DEPLOY_FRONTEND=1
DEPLOY_BACKEND=1
INSTALL_DEPS=0
RESTART_MLFLOW=0
SKIP_HEALTH=0

HEALTH_TIMEOUT_SECONDS="${HEALTH_TIMEOUT_SECONDS:-60}"
HEALTH_RETRY_SECONDS="${HEALTH_RETRY_SECONDS:-2}"

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Deploy Basketworld public app (frontend + backend).
Run from anywhere; script resolves project root automatically.

Options:
  --frontend-only      Build/sync frontend only.
  --backend-only       Restart backend service only.
  --with-deps          Install Python deps with uv before deploy.
  --restart-mlflow     Restart MLflow service too.
  --skip-health-check  Skip post-deploy health checks.
  -h, --help           Show this help.

Examples:
  $(basename "$0")
  $(basename "$0") --backend-only
  $(basename "$0") --frontend-only
  $(basename "$0") --with-deps --restart-mlflow
USAGE
}

log() {
  printf '[deploy] %s\n' "$*"
}

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command: $cmd" >&2
    exit 1
  fi
}

wait_for_service_active() {
  local service="$1"
  local timeout_s="$2"
  local retry_s="$3"
  local deadline=$((SECONDS + timeout_s))

  while (( SECONDS < deadline )); do
    if sudo systemctl is-active "$service" >/dev/null 2>&1; then
      return 0
    fi
    sleep "$retry_s"
  done

  echo "Timed out waiting for service to become active: $service" >&2
  sudo systemctl --no-pager --full status "$service" || true
  return 1
}

wait_for_http_ok() {
  local url="$1"
  local timeout_s="$2"
  local retry_s="$3"
  local deadline=$((SECONDS + timeout_s))

  while (( SECONDS < deadline )); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep "$retry_s"
  done

  echo "Timed out waiting for healthy HTTP endpoint: $url" >&2
  return 1
}

for arg in "$@"; do
  case "$arg" in
    --frontend-only)
      DEPLOY_FRONTEND=1
      DEPLOY_BACKEND=0
      ;;
    --backend-only)
      DEPLOY_FRONTEND=0
      DEPLOY_BACKEND=1
      ;;
    --with-deps)
      INSTALL_DEPS=1
      ;;
    --restart-mlflow)
      RESTART_MLFLOW=1
      ;;
    --skip-health-check)
      SKIP_HEALTH=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $arg" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "$DEPLOY_FRONTEND" -eq 0 && "$DEPLOY_BACKEND" -eq 0 ]]; then
  echo "Nothing to do: both frontend and backend deploy are disabled." >&2
  exit 1
fi

if [[ "$INSTALL_DEPS" -eq 1 ]]; then
  require_cmd uv
  log "Installing backend dependencies via uv"
  cd "$ROOT_DIR"
  uv venv .venv
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.venv/bin/activate"
  uv pip install -r "$ROOT_DIR/requirements.txt"
  (cd "$ROOT_DIR/app/backend" && uv pip install -r requirements.txt)
fi

if [[ "$DEPLOY_FRONTEND" -eq 1 ]]; then
  require_cmd npm
  require_cmd rsync
  log "Building frontend"
  cd "$FRONTEND_DIR"
  npm ci
  npm run build

  log "Syncing frontend to $WEB_ROOT"
  sudo mkdir -p "$WEB_ROOT"
  sudo rsync -a --delete "$FRONTEND_DIR/dist/" "$WEB_ROOT/"
  sudo chown -R www-data:www-data "$WEB_ROOT"

  log "Reloading nginx"
  sudo nginx -t
  sudo systemctl reload nginx
fi

if [[ "$DEPLOY_BACKEND" -eq 1 ]]; then
  log "Restarting backend service: $BACKEND_SERVICE"
  sudo systemctl restart "$BACKEND_SERVICE"
fi

if [[ "$RESTART_MLFLOW" -eq 1 ]]; then
  log "Restarting MLflow service: $MLFLOW_SERVICE"
  sudo systemctl restart "$MLFLOW_SERVICE"
fi

if [[ "$SKIP_HEALTH" -eq 0 ]]; then
  require_cmd curl
  log "Running health checks"

  if [[ "$RESTART_MLFLOW" -eq 1 ]]; then
    wait_for_service_active "$MLFLOW_SERVICE" "$HEALTH_TIMEOUT_SECONDS" "$HEALTH_RETRY_SECONDS"
    wait_for_http_ok "http://127.0.0.1:5000/health" "$HEALTH_TIMEOUT_SECONDS" "$HEALTH_RETRY_SECONDS"
  fi

  if [[ "$DEPLOY_BACKEND" -eq 1 ]]; then
    wait_for_service_active "$BACKEND_SERVICE" "$HEALTH_TIMEOUT_SECONDS" "$HEALTH_RETRY_SECONDS"
    wait_for_http_ok "http://127.0.0.1:8080/api/playable/config" "$HEALTH_TIMEOUT_SECONDS" "$HEALTH_RETRY_SECONDS"
  fi

  if [[ "$DEPLOY_FRONTEND" -eq 1 ]]; then
    wait_for_http_ok "https://basketworld.toplines.app" "$HEALTH_TIMEOUT_SECONDS" "$HEALTH_RETRY_SECONDS"
  fi

  log "Health checks passed"
fi

log "Deploy complete"
