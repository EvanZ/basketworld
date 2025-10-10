# MLflow S3 Setup - Final Working Configuration

## Overview
This project uses MLflow with S3 for artifact storage. This document describes the final working setup.

## Prerequisites
- AWS S3 bucket: `s3://basketworld/mlflow-artifacts`
- AWS credentials with S3 access

## Setup Steps

### 1. AWS Credentials Setup

Create AWS credentials in `~/.aws/credentials` using a named profile:

```ini
[basketworld]
aws_access_key_id = YOUR_ACCESS_KEY
aws_secret_access_key = YOUR_SECRET_KEY
```

And configure the region in `~/.aws/config`:
```ini
[profile basketworld]
region = us-west-1
```

This is the **only place** you need to store credentials. Both the MLflow server and training script use this profile.

### 2. Starting the MLflow Server

Use the unified startup script:

```bash
# S3 storage (default)
./start_mlflow.sh

# Local storage (for development without AWS)
./start_mlflow.sh --disable-s3
```

**S3 Mode** (default):
- Sets `AWS_PROFILE=basketworld`
- Uses `--artifacts-destination s3://basketworld/mlflow-artifacts`
- New runs get `mlflow-artifacts:/` URIs (server-proxied access)

**Local Mode** (`--disable-s3`):
- Uses `--default-artifact-root file://./mlartifacts`
- No AWS credentials needed
- Artifacts stored locally in `./mlartifacts/`

### 3. Running Training

The training script (`train/train.py`) automatically:
- Checks if `~/.aws/credentials` exists
- Sets `AWS_PROFILE=basketworld` if available
- Works with both S3 and local storage modes

Just run training normally from Cursor or terminal - it will work with whichever server mode you're using.

## Important Notes

### Server Flag: --artifacts-destination vs --default-artifact-root

**✅ USE (S3 Mode):** `--artifacts-destination s3://...`
- New runs get `mlflow-artifacts:/` URIs
- Client accesses artifacts through server (proxy mode)
- Only server needs AWS credentials

**✅ USE (Local Mode):** `--default-artifact-root file://...`
- New runs get `file://` URIs
- Artifacts stored locally
- No AWS credentials needed

**❌ DON'T USE:** `--default-artifact-root s3://...`
- Runs get `s3://` URIs
- Client accesses S3 directly (requires client credentials)
- Deprecated approach

### Experiments Created with Old Server Configuration

If you have experiments created when the server used `--default-artifact-root s3://...`, those experiments will ALWAYS use `s3://` URIs. Create a NEW experiment to use the correct `mlflow-artifacts:/` URIs with server-proxied access.

### Storage Separation

MLflow separates storage into two parts:

1. **Backend Store** (Metadata): `mlflow.db`
   - Stores metrics, parameters, tags, run info
   - Local SQLite database
   - Can be copied to S3 for sharing/backup

2. **Artifact Store** (Files): `s3://basketworld/mlflow-artifacts` or `./mlartifacts/`
   - Stores models, plots, large files
   - S3 (production) or local (development)

### Web App Backend

The web app backend doesn't need credentials. It:
- Connects to `http://localhost:5000` (MLflow server)
- Server proxies all S3 access
- Backend never touches S3 directly

## Files

### Scripts
- `start_mlflow.sh` - Unified startup (S3 or local mode)
- `start_mlflow_with_profile.sh` - S3-only startup (legacy, still works)
- `stop_mlflow.sh` - Stop MLflow server

### Configuration
- `~/.aws/credentials` - AWS credentials with [basketworld] profile
- `~/.aws/config` - AWS region configuration
- `.vscode/launch.json` - VS Code debug configurations

### Cleanup
- `scripts/cleanup_mlflow_deleted_runs_s3.py` - Clean deleted runs from S3
- `scripts/cleanup_mlflow_deleted_runs.py` - Clean deleted runs from local storage (legacy)

## Troubleshooting

### Server returns 500 errors
Check that:
- Server was started with `AWS_PROFILE=basketworld` (for S3 mode)
- Credentials exist in `~/.aws/credentials` under `[basketworld]` profile
- Restart the server: `./stop_mlflow.sh && ./start_mlflow.sh`

### Training can't access artifacts
If using S3 mode:
- Verify `~/.aws/credentials` exists with `[basketworld]` profile
- This only affects old runs with `s3://` URIs
- New runs with `mlflow-artifacts:/` URIs don't need client credentials

### First run in new experiment hangs
- First run has no opponent policies (expected behavior)
- Training continues after a moment with current policy

## Testing

Test that the server can access S3:

```python
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("test")

with mlflow.start_run():
    mlflow.log_param("test", "value")
    # Check run.info.artifact_uri - should start with "mlflow-artifacts:/"
```

## Development Workflow

For **local development** (no AWS needed):
```bash
./start_mlflow.sh --disable-s3
# Train normally - artifacts go to ./mlartifacts/
```

For **production** (with S3):
```bash
./start_mlflow.sh
# Train normally - artifacts go to S3
```

Both modes use the same `mlflow.db` for metadata, so you can switch between them.
