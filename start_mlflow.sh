#!/bin/bash
# Start MLflow server with optional S3 support
# Usage:
#   ./start_mlflow.sh              # Use S3 (default)
#   ./start_mlflow.sh --disable-s3 # Use local storage

USE_S3=true

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

cd /Users/evanzamir/projects/basketworld
source .env/bin/activate

echo "MLflow Server Startup"
echo "===================="

if [ "$USE_S3" = true ]; then
    echo "Mode: S3 Storage"
    echo "Artifacts: s3://basketworld/mlflow-artifacts"
    echo "AWS Profile: basketworld"
    echo ""
    
    # Use basketworld AWS profile for S3 access
    export AWS_PROFILE=basketworld
    
    mlflow server \
        --backend-store-uri sqlite:///mlflow.db \
        --artifacts-destination s3://basketworld/mlflow-artifacts \
        --host 0.0.0.0 \
        --port 5000
else
    echo "Mode: Local Storage"
    echo "Artifacts: ./mlartifacts"
    echo ""
    
    # Create local directories if they don't exist
    mkdir -p mlruns
    mkdir -p mlartifacts
    
    mlflow server \
        --backend-store-uri sqlite:///mlflow.db \
        --default-artifact-root file://$(pwd)/mlartifacts \
        --host 0.0.0.0 \
        --port 5000
fi

