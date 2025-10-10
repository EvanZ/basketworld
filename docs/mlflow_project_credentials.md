# MLflow Project-Specific AWS Credentials

This guide explains how to use project-specific AWS credentials for MLflow that won't conflict with your other AWS projects.

## The Problem

When working with multiple AWS projects, you typically have global AWS credentials set:

```bash
# Global credentials for your main AWS account
export AWS_ACCESS_KEY_ID="main-project-key"
export AWS_SECRET_ACCESS_KEY="main-project-secret"
```

If you use these same environment variables for MLflow S3 storage, they will affect ALL your AWS operations, potentially causing conflicts with other projects.

## The Solution

The MLflow configuration module supports **project-specific credentials** using two methods:

### Method 1: Using `.env` File (Recommended)

Create a `.env` file in the project root (already in `.gitignore`):

```bash
# .env
MLFLOW_ARTIFACT_ROOT=s3://my-mlflow-bucket/artifacts
MLFLOW_AWS_ACCESS_KEY_ID=mlflow-specific-key
MLFLOW_AWS_SECRET_ACCESS_KEY=mlflow-specific-secret
MLFLOW_AWS_DEFAULT_REGION=us-east-1
```

The `.env` file is automatically loaded when any script runs. No need to source it manually!

**Key Features:**
- ✅ Automatically loaded by all scripts
- ✅ Project-specific credentials
- ✅ Won't affect other AWS projects
- ✅ Already in `.gitignore` (safe from commits)
- ✅ No need to export or source manually

### Method 2: Using `MLFLOW_AWS_*` Environment Variables

Set project-specific environment variables with the `MLFLOW_AWS_` prefix:

```bash
# Project-specific credentials (for MLflow only)
export MLFLOW_AWS_ACCESS_KEY_ID="mlflow-specific-key"
export MLFLOW_AWS_SECRET_ACCESS_KEY="mlflow-specific-secret"
export MLFLOW_AWS_DEFAULT_REGION="us-east-1"
export MLFLOW_ARTIFACT_ROOT="s3://my-mlflow-bucket/artifacts"

# Your global credentials remain unchanged and work for other projects
export AWS_ACCESS_KEY_ID="main-project-key"
export AWS_SECRET_ACCESS_KEY="main-project-secret"
```

## Credential Precedence

The system checks for credentials in this order:

1. **`MLFLOW_AWS_ACCESS_KEY_ID`** (project-specific) - **HIGHEST PRIORITY**
2. **`AWS_ACCESS_KEY_ID`** (global) - Fallback if above not set
3. **.env file** - Loaded automatically before checking environment variables

Same for `SECRET_ACCESS_KEY` and `DEFAULT_REGION`.

## Complete Example

### Step 1: Create `.env` File

```bash
cd /home/ubuntu/basketworld
cp .env.mlflow.example .env
```

Edit `.env`:
```bash
# .env
MLFLOW_ARTIFACT_ROOT=s3://my-mlflow-bucket/mlflow-artifacts
MLFLOW_AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
MLFLOW_AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
MLFLOW_AWS_DEFAULT_REGION=us-east-1
```

### Step 2: Start MLflow Server

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root s3://my-mlflow-bucket/mlflow-artifacts \
  --port 5000
```

### Step 3: Run Training

```bash
python train/train.py --mlflow-experiment-name my-experiment
```

Output will show:
```
MLflow Configuration:
  Tracking URI: http://localhost:5000
  Storage Type: S3 (Remote)
  Artifact Root: s3://my-mlflow-bucket/mlflow-artifacts
  AWS Credentials: project-specific (MLFLOW_AWS_*)
  AWS Region: us-east-1
```

## Verification

### Test 1: Verify Configuration

```bash
python -c "from basketworld.utils.mlflow_config import get_mlflow_config; print(get_mlflow_config())"
```

This should show "project-specific (MLFLOW_AWS_*)" if using `.env` or `MLFLOW_AWS_*` variables.

### Test 2: Run Full Test

```bash
python scripts/test_mlflow_s3.py
```

This tests all aspects of your S3 configuration.

### Test 3: Verify Credentials Don't Conflict

```bash
# Check global AWS credentials (should be your main project)
aws sts get-caller-identity

# Run MLflow training (uses project-specific credentials)
python train/train.py --mlflow-experiment-name test

# Check global AWS credentials again (should still be your main project)
aws sts get-caller-identity
```

The global AWS CLI should still use your main credentials, while MLflow uses the project-specific ones.

## Multiple Projects Example

Here's how this helps with multiple projects:

```bash
# Terminal 1: Main project (uses global AWS credentials)
cd /home/ubuntu/main-project
export AWS_ACCESS_KEY_ID="main-project-key"
export AWS_SECRET_ACCESS_KEY="main-project-secret"
python main_script.py  # Uses global credentials

# Terminal 2: MLflow project (uses project-specific credentials from .env)
cd /home/ubuntu/basketworld
# .env file has MLFLOW_AWS_* credentials
python train/train.py  # Uses project-specific credentials from .env
```

Both projects work independently without credential conflicts!

## Security Best Practices

1. **Never commit `.env`**: Already in `.gitignore`, but double-check
2. **Use separate IAM users**: Create a dedicated IAM user for MLflow
3. **Least privilege**: Only grant S3 permissions needed for MLflow bucket
4. **Rotate credentials**: Regularly rotate your AWS access keys
5. **Use different buckets**: Don't mix MLflow artifacts with other project data

## Troubleshooting

### Problem: Credentials still conflict

**Solution**: Make sure you're using `MLFLOW_AWS_*` prefix, not just `AWS_*`:

```bash
# ✗ Wrong - will affect all AWS operations
AWS_ACCESS_KEY_ID=key

# ✓ Correct - only affects MLflow
MLFLOW_AWS_ACCESS_KEY_ID=key
```

### Problem: `.env` file not loaded

**Check**:
1. File is named exactly `.env` (not `.env.txt` or `.env.example`)
2. File is in the project root (`/home/ubuntu/basketworld/.env`)
3. File is not a directory
4. File has proper KEY=VALUE format

**Test**:
```bash
cd /home/ubuntu/basketworld
python -c "
from basketworld.utils.mlflow_config import _load_env_file
_load_env_file()
import os
print('MLFLOW_AWS_ACCESS_KEY_ID:', os.environ.get('MLFLOW_AWS_ACCESS_KEY_ID', 'NOT SET'))
"
```

### Problem: Still using global credentials

**Debug**:
```bash
python -c "
import os
print('Checking credential sources...')
print('MLFLOW_AWS_ACCESS_KEY_ID:', 'SET' if os.environ.get('MLFLOW_AWS_ACCESS_KEY_ID') else 'NOT SET')
print('AWS_ACCESS_KEY_ID:', 'SET' if os.environ.get('AWS_ACCESS_KEY_ID') else 'NOT SET')

from basketworld.utils.mlflow_config import get_mlflow_config
config = get_mlflow_config()
print('\\n' + str(config))
"
```

### Problem: Permission denied with project-specific credentials

**Solution**: Verify the IAM user has correct permissions:

```bash
# Test with AWS CLI using project-specific credentials
export AWS_ACCESS_KEY_ID="mlflow-specific-key"
export AWS_SECRET_ACCESS_KEY="mlflow-specific-secret"
aws s3 ls s3://my-mlflow-bucket/
```

If this fails, the IAM user needs S3 permissions (see main S3 setup guide).

## Migration from Global Credentials

If you're currently using global `AWS_*` credentials, here's how to migrate:

### Step 1: Create Project-Specific IAM User

1. Go to AWS IAM Console
2. Create new user: `mlflow-basketworld`
3. Attach S3 permissions policy
4. Generate access keys

### Step 2: Create `.env` File

```bash
cd /home/ubuntu/basketworld
cat > .env << 'EOF'
MLFLOW_ARTIFACT_ROOT=s3://my-mlflow-bucket/mlflow-artifacts
MLFLOW_AWS_ACCESS_KEY_ID=<new-mlflow-key>
MLFLOW_AWS_SECRET_ACCESS_KEY=<new-mlflow-secret>
MLFLOW_AWS_DEFAULT_REGION=us-east-1
EOF
```

### Step 3: Test

```bash
python scripts/test_mlflow_s3.py
```

Should show "project-specific (MLFLOW_AWS_*)" in the output.

### Step 4: Restore Global Credentials

```bash
# Restore your main project credentials
export AWS_ACCESS_KEY_ID="main-project-key"
export AWS_SECRET_ACCESS_KEY="main-project-secret"
```

### Step 5: Verify Both Work

```bash
# Test main project
aws s3 ls  # Should use global credentials

# Test MLflow
python train/train.py --mlflow-experiment-name test  # Should use project-specific credentials
```

## Summary

- ✅ Use **`.env` file** for automatic, project-specific credentials
- ✅ Use **`MLFLOW_AWS_*`** prefix to avoid conflicts with global AWS credentials
- ✅ Credentials are loaded **automatically** - no manual sourcing needed
- ✅ Safe and secure - `.env` is already in `.gitignore`
- ✅ Works seamlessly with all scripts (training, analytics, backend)
- ✅ Doesn't affect other AWS projects

**Quick Start**: Copy `.env.mlflow.example` to `.env`, fill in your MLflow-specific credentials, and you're done!

