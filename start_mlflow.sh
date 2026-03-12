#!/bin/bash
# Start MLflow server with optional S3 support.
# Usage:
#   ./start_mlflow.sh              # Use S3 (default)
#   ./start_mlflow.sh --disable-s3 # Use local storage

USE_S3=true
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
DB_PATH="$PROJECT_ROOT/mlflow.db"
BACKEND_STORE_URI="sqlite:///$DB_PATH"
LOCAL_ARTIFACT_ROOT="file://$PROJECT_ROOT/mlartifacts"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --disable-s3)
            USE_S3=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--disable-s3]"
            exit 1
            ;;
    esac
done

cd "$PROJECT_ROOT"
source .env/bin/activate

echo "MLflow Server Startup"
echo "===================="
echo "Project root: $PROJECT_ROOT"
echo "Backend store: $BACKEND_STORE_URI"
export MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false

if [ "$USE_S3" = true ]; then
    echo "Mode: S3 Storage"
    echo "Artifacts: s3://basketworld/mlflow-artifacts"
    echo "AWS Profile: basketworld"
    echo ""
    
    # Use basketworld AWS profile for S3 access
    export AWS_PROFILE=basketworld
    
    mlflow server \
        --backend-store-uri "$BACKEND_STORE_URI" \
        --artifacts-destination s3://basketworld/mlflow-artifacts \
        --host 127.0.0.1 \
        --port 5000
else
    echo "Mode: Local Storage"
    echo "Artifacts: $LOCAL_ARTIFACT_ROOT"
    echo ""
    
    # Create local directories if they don't exist
    mkdir -p mlruns
    mkdir -p mlartifacts
    
    mlflow server \
        --backend-store-uri "$BACKEND_STORE_URI" \
        --default-artifact-root "$LOCAL_ARTIFACT_ROOT" \
        --host 127.0.0.1 \
        --port 5000
fi
