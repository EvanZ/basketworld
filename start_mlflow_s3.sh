#!/bin/bash
# Start MLflow server with S3 support.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
DB_PATH="$PROJECT_ROOT/mlflow.db"
BACKEND_STORE_URI="sqlite:///$DB_PATH"

cd "$PROJECT_ROOT"

# Load credentials from .env.aws
set -a
source .env.aws
set +a

# Activate virtual environment
source .env/bin/activate
export MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false

# Start MLflow server with correct flag: --artifacts-destination (not --default-artifact-root!)
# This is critical for proper S3 artifact access
AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
AWS_DEFAULT_REGION="$AWS_DEFAULT_REGION" \
mlflow server \
    --backend-store-uri "$BACKEND_STORE_URI" \
    --artifacts-destination s3://basketworld/mlflow-artifacts \
    --host 127.0.0.1 \
    --port 5000
