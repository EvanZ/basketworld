#!/bin/bash
# Start MLflow server using AWS profile

cd /Users/evanzamir/projects/basketworld
source .env/bin/activate

# Use basketworld AWS profile
export AWS_PROFILE=basketworld

echo "Using AWS profile: $AWS_PROFILE"
echo "Starting MLflow server..."

mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --artifacts-destination s3://basketworld/mlflow-artifacts \
    --host 0.0.0.0 \
    --port 5000

