#!/bin/bash
# Start MLflow server using AWS profile

cd /Users/evanzamir/projects/basketworld
source .env/bin/activate

# Use basketworld AWS profile
export AWS_PROFILE=default

echo "Using AWS profile: $AWS_PROFILE"
echo "Starting MLflow server..."

mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --artifacts-destination s3://basketworld/mlflow-artifacts \
    --host 127.0.0.1 \
    --port 5000

