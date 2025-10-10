# MLflow S3 Storage Implementation Summary

This document summarizes the implementation of S3 storage support for MLflow in the BasketWorld project.

## Overview

MLflow can now use Amazon S3 (or any S3-compatible service) for storing experiment artifacts, while maintaining backward compatibility with local storage. This allows users to:

- Store artifacts in the cloud for persistence and sharing
- Access experiments from multiple machines
- Avoid local disk space limitations
- Keep existing local storage workflows intact

## What Was Implemented

### 1. Core Configuration Module

**File**: `basketworld/utils/mlflow_config.py`

This new module provides:
- `MLflowConfig` dataclass for storing configuration
- `get_mlflow_config()` function to read configuration from environment variables
- `setup_mlflow()` function to initialize MLflow with the correct backend
- Automatic detection of S3 vs. local storage
- Validation of required dependencies and credentials

### 2. Updated Dependencies

**File**: `requirements.txt`

Added `boto3==1.37.3` for AWS S3 support.

### 3. Updated Training Script

**File**: `train/train.py`

Modified to:
- Import and use `setup_mlflow()` instead of hardcoded tracking URI
- Display storage type (Local vs S3) when training starts
- Show helpful error messages with S3 server start command if connection fails

### 4. Updated Backend API

**File**: `app/backend/main.py`

Modified to:
- Use `setup_mlflow()` for configuration
- Handle S3 storage configuration errors gracefully

### 5. Updated Analytics Scripts

**Files**:
- `analytics/elo_evolution.py`
- `analytics/evaluate.py`
- `analytics/heatmap.py`
- `analytics/shotchart.py`
- `analytics/assist_skill_delta.py`
- `analytics/trajectory_accumulation.py`
- `scripts/cleanup_mlflow_deleted_runs.py`

All scripts now use `setup_mlflow()` to automatically detect and use the configured storage backend.

### 6. Documentation

Created comprehensive documentation:

**`docs/mlflow_s3_setup.md`** (Full Guide)
- Detailed setup instructions
- IAM policy examples
- Troubleshooting guide
- Cost considerations
- Security best practices
- Migration guide

**`docs/mlflow_s3_quickstart.md`** (Quick Reference)
- 5-minute setup guide
- Essential commands
- Common troubleshooting
- Quick reference table

**`docs/MLFLOW_S3_IMPLEMENTATION.md`** (This Document)
- Implementation details
- Technical overview

### 7. Test Script

**File**: `scripts/test_mlflow_s3.py`

A comprehensive test script that:
- Checks environment variables
- Verifies boto3 installation
- Tests AWS credentials
- Tests S3 connectivity and permissions
- Tests MLflow configuration
- Tests MLflow server connectivity

Usage: `python scripts/test_mlflow_s3.py`

### 8. Updated README

**File**: `README.md`

Added a new section "☁️ Remote Storage (S3)" with:
- Quick setup instructions
- Links to detailed documentation
- Example commands

## How It Works

### Configuration Detection

The `get_mlflow_config()` function reads environment variables:

1. **MLFLOW_TRACKING_URI**: The MLflow tracking server URI (default: `http://localhost:5000`)
2. **MLFLOW_ARTIFACT_ROOT**: The artifact storage location (default: local)
3. **AWS_ACCESS_KEY_ID**: AWS access key (required for S3)
4. **AWS_SECRET_ACCESS_KEY**: AWS secret key (required for S3)
5. **AWS_DEFAULT_REGION**: AWS region (default: `us-east-1`)
6. **MLFLOW_S3_ENDPOINT_URL**: Custom S3 endpoint (optional, for MinIO/LocalStack)

If `MLFLOW_ARTIFACT_ROOT` starts with `s3://`, S3 storage is enabled.

### Storage Determination

```python
use_s3 = artifact_root is not None and artifact_root.startswith("s3://")
```

### Validation

When S3 is enabled, `setup_mlflow()`:
1. Checks if boto3 is installed
2. Verifies AWS credentials are set
3. Sets the tracking URI in MLflow

### Backward Compatibility

If no S3 configuration is present, the system defaults to:
- Tracking URI: `http://localhost:5000`
- Artifact storage: Local filesystem (`mlruns/` and `mlartifacts/`)

## Usage Examples

### Local Storage (Default)

```bash
# No configuration needed
mlflow ui
python train/train.py --mlflow-experiment-name my-experiment
```

### S3 Storage

```bash
# Set environment variables
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
export MLFLOW_ARTIFACT_ROOT="s3://my-bucket/mlflow-artifacts"

# Start MLflow server with S3 backend
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root s3://my-bucket/mlflow-artifacts

# Run training (automatically uses S3)
python train/train.py --mlflow-experiment-name my-experiment
```

Output will show:
```
MLflow Configuration:
  Tracking URI: http://localhost:5000
  Storage Type: S3 (Remote)
  Artifact Root: s3://my-bucket/mlflow-artifacts
```

## Architecture

### Storage Architecture

```
┌─────────────────────────────────────────────────────┐
│                 MLflow Server                        │
│  ┌────────────────────┐  ┌────────────────────────┐│
│  │  Backend Store     │  │   Artifact Store       ││
│  │  (SQLite/Postgres) │  │   (Local or S3)        ││
│  │                    │  │                        ││
│  │  - Run metadata    │  │   - Model files        ││
│  │  - Parameters      │  │   - Plots/Charts       ││
│  │  - Metrics         │  │   - Custom artifacts   ││
│  │  - Tags            │  │                        ││
│  └────────────────────┘  └────────────────────────┘│
└─────────────────────────────────────────────────────┘
           │                          │
           │                          │
           ▼                          ▼
    Local SQLite DB              S3 Bucket or
    (mlflow.db)                  Local mlartifacts/
```

### Code Flow

```
Application (train.py, backend/main.py, etc.)
    │
    ├─► Import setup_mlflow()
    │
    ├─► Call setup_mlflow(verbose=True)
    │      │
    │      ├─► Read environment variables
    │      ├─► Detect S3 vs Local
    │      ├─► Validate dependencies & credentials
    │      └─► Set mlflow.set_tracking_uri()
    │
    └─► Use MLflow normally (mlflow.log_*, mlflow.start_run(), etc.)
           │
           └─► MLflow automatically uses configured storage
```

## Testing

### Test Local Configuration

```bash
python scripts/test_mlflow_s3.py
```

Expected output:
```
MLflow S3 Configuration Test
============================================================

1. Checking environment variables...
------------------------------------------------------------
  MLFLOW_TRACKING_URI: http://localhost:5000
  MLFLOW_ARTIFACT_ROOT: (not set - using local storage)
  ...

ℹ️  S3 storage is NOT configured (using local storage)
```

### Test S3 Configuration

```bash
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
export MLFLOW_ARTIFACT_ROOT="s3://my-bucket/mlflow-artifacts"

python scripts/test_mlflow_s3.py
```

Expected output:
```
MLflow S3 Configuration Test
============================================================

1. Checking environment variables...
------------------------------------------------------------
  MLFLOW_TRACKING_URI: http://localhost:5000
  MLFLOW_ARTIFACT_ROOT: s3://my-bucket/mlflow-artifacts
  AWS_ACCESS_KEY_ID: ✓ set
  AWS_SECRET_ACCESS_KEY: ✓ set
  ...

✅ All checks passed! MLflow S3 storage is properly configured.
```

## Security Considerations

### Credentials Management

- Never commit AWS credentials to version control
- Use environment variables or AWS credentials file
- Consider using IAM roles on EC2 instances
- Use least-privilege IAM policies

### Recommended IAM Policy

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::YOUR-BUCKET-NAME/*",
        "arn:aws:s3:::YOUR-BUCKET-NAME"
      ]
    }
  ]
}
```

### S3 Bucket Security

- Enable server-side encryption
- Enable versioning for artifact protection
- Use bucket policies to restrict access
- Enable access logging for audit trails
- Consider using VPC endpoints for private access

## Future Enhancements

Potential improvements:

1. **Azure Blob Storage Support**: Add support for Azure as an alternative to S3
2. **Google Cloud Storage Support**: Add support for GCS
3. **Credential Rotation**: Automatic credential refresh for long-running training jobs
4. **Multi-Region Support**: Store artifacts in multiple regions for redundancy
5. **Artifact Caching**: Local caching of frequently accessed artifacts
6. **Migration Tool**: Automated tool to migrate existing local artifacts to S3

## Troubleshooting

### Common Issues

1. **boto3 not found**: Install with `pip install boto3`
2. **Credentials not set**: Export AWS credentials as environment variables
3. **Bucket access denied**: Check IAM permissions
4. **Server not started with S3**: Ensure `--default-artifact-root` matches `MLFLOW_ARTIFACT_ROOT`

See `docs/mlflow_s3_setup.md` for detailed troubleshooting.

## References

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow S3 Artifact Store](https://mlflow.org/docs/latest/tracking.html#amazon-s3-and-s3-compatible-storage)
- [AWS S3 Documentation](https://docs.aws.amazon.com/s3/)
- [boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)

## Contact

For questions or issues:
- Open an issue on GitHub
- Check the documentation in `docs/mlflow_s3_setup.md`
- Run the test script: `python scripts/test_mlflow_s3.py`

