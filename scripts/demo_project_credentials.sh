#!/bin/bash
# Demo: Project-Specific AWS Credentials for MLflow
# This demonstrates how MLFLOW_AWS_* credentials don't conflict with global AWS credentials

set -e

echo "================================================================================"
echo "  Demo: Project-Specific AWS Credentials for MLflow"
echo "================================================================================"
echo

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}Step 1: Show current global AWS credentials${NC}"
echo "----------------------------------------------------------------------------"
if [ -n "$AWS_ACCESS_KEY_ID" ]; then
    echo "  AWS_ACCESS_KEY_ID: ${AWS_ACCESS_KEY_ID:0:10}..."
    echo "  ✓ Global credentials are set"
else
    echo "  AWS_ACCESS_KEY_ID: (not set)"
    echo "  ℹ️  No global credentials (that's okay for this demo)"
fi
echo

echo -e "${BLUE}Step 2: Create a test .env file with project-specific credentials${NC}"
echo "----------------------------------------------------------------------------"
cat > .env.demo << 'EOF'
# Project-specific MLflow credentials
MLFLOW_ARTIFACT_ROOT=s3://demo-mlflow-bucket/artifacts
MLFLOW_AWS_ACCESS_KEY_ID=AKIADEMO_MLFLOW_KEY
MLFLOW_AWS_SECRET_ACCESS_KEY=demo_mlflow_secret_key_xyz123
MLFLOW_AWS_DEFAULT_REGION=us-west-2
EOF

echo "  Created .env.demo with MLFLOW_AWS_* credentials"
echo "  Contents:"
cat .env.demo | sed 's/^/    /'
echo

echo -e "${BLUE}Step 3: Test MLflow configuration (simulated)${NC}"
echo "----------------------------------------------------------------------------"
echo "  Loading .env.demo file..."
set -a
source .env.demo
set +a

echo "  ✓ Environment variables loaded"
echo

echo -e "${BLUE}Step 4: Check which credentials MLflow will use${NC}"
echo "----------------------------------------------------------------------------"
python3 << 'PYEOF'
import os

print("  Checking credential sources:")
print()

# Project-specific credentials
mlflow_key = os.environ.get("MLFLOW_AWS_ACCESS_KEY_ID")
aws_key = os.environ.get("AWS_ACCESS_KEY_ID")

if mlflow_key:
    print("  🎯 MLflow will use PROJECT-SPECIFIC credentials:")
    print(f"     MLFLOW_AWS_ACCESS_KEY_ID: {mlflow_key[:15]}...")
    print(f"     MLFLOW_AWS_SECRET_ACCESS_KEY: {os.environ.get('MLFLOW_AWS_SECRET_ACCESS_KEY', '')[:15]}...")
    print(f"     MLFLOW_AWS_DEFAULT_REGION: {os.environ.get('MLFLOW_AWS_DEFAULT_REGION')}")
else:
    print("  ⚠️  MLflow will use global credentials (or none)")

print()
if aws_key:
    print("  🌍 Global AWS credentials REMAIN UNCHANGED:")
    print(f"     AWS_ACCESS_KEY_ID: {aws_key[:15]}...")
    print("     (Other AWS tools will continue using these)")
else:
    print("  ℹ️  No global AWS credentials set")
    print("     (Other AWS tools won't be affected)")

print()
print("  ✅ No conflict! Project-specific credentials are isolated.")
PYEOF
echo

echo -e "${BLUE}Step 5: Show how MLflow configuration detects this${NC}"
echo "----------------------------------------------------------------------------"
python3 << 'PYEOF'
import sys
sys.path.insert(0, '/home/ubuntu/basketworld')

from basketworld.utils.mlflow_config import get_mlflow_config

config = get_mlflow_config(load_env=False)  # Already loaded
print(config)
PYEOF
echo

echo -e "${BLUE}Step 6: Cleanup${NC}"
echo "----------------------------------------------------------------------------"
rm -f .env.demo
unset MLFLOW_ARTIFACT_ROOT
unset MLFLOW_AWS_ACCESS_KEY_ID
unset MLFLOW_AWS_SECRET_ACCESS_KEY
unset MLFLOW_AWS_DEFAULT_REGION
echo "  ✓ Cleaned up demo environment"
echo

echo "================================================================================"
echo -e "${GREEN}Demo Complete!${NC}"
echo "================================================================================"
echo
echo "Summary:"
echo "  • Created .env file with MLFLOW_AWS_* credentials"
echo "  • MLflow uses project-specific credentials automatically"
echo "  • Global AWS credentials remain unchanged"
echo "  • No conflicts with other AWS projects!"
echo
echo "To use this in your project:"
echo "  1. Copy .env.mlflow.example to .env"
echo "  2. Fill in your MLflow-specific AWS credentials"
echo "  3. Run your scripts normally - credentials are auto-loaded!"
echo
echo "See docs/mlflow_project_credentials.md for more details."
echo

