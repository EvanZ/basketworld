#!/bin/bash
# Start MLflow server using AWS profile.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
DB_PATH="$PROJECT_ROOT/mlflow.db"
BACKEND_STORE_URI="sqlite:///$DB_PATH"

cd "$PROJECT_ROOT"
source .env/bin/activate
export MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false

# Use basketworld AWS profile
export AWS_PROFILE=default

echo "Using AWS profile: $AWS_PROFILE"
echo "Starting MLflow server..."
echo "Project root: $PROJECT_ROOT"
echo "Backend store: $BACKEND_STORE_URI"

mlflow server \
    --backend-store-uri "$BACKEND_STORE_URI" \
    --artifacts-destination s3://basketworld/mlflow-artifacts \
    --host 127.0.0.1 \
    --port 5000
