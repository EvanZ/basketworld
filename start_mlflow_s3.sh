#!/bin/bash
# Start MLflow server with S3 support

cd /Users/evanzamir/projects/basketworld

# Load credentials from .env.aws
set -a
source .env.aws
set +a

# Activate virtual environment
source .env/bin/activate

# Start MLflow server with correct flag: --artifacts-destination (not --default-artifact-root!)
# This is critical for proper S3 artifact access
AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
AWS_DEFAULT_REGION="$AWS_DEFAULT_REGION" \
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --artifacts-destination s3://basketworld/mlflow-artifacts \
    --host 0.0.0.0 \
    --port 5000
