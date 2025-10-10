# MLflow S3 Storage - Quick Start

This is a quick reference for setting up MLflow with S3 storage. For detailed explanations, see [mlflow_s3_setup.md](./mlflow_s3_setup.md).

## Quick Setup (5 minutes)

### 1. Install boto3

```bash
pip install boto3
```

### 2. Configure Credentials (Choose One Method)

**Method A: Using `.env` file (Recommended - No conflicts with other projects)**

```bash
# Create .env file in project root
cat > .env << 'EOF'
MLFLOW_ARTIFACT_ROOT=s3://your-bucket-name/mlflow-artifacts
MLFLOW_AWS_ACCESS_KEY_ID=your-access-key-id
MLFLOW_AWS_SECRET_ACCESS_KEY=your-secret-access-key
MLFLOW_AWS_DEFAULT_REGION=us-east-1
EOF

# No need to source it - automatically loaded!
```

**Method B: Using environment variables**

```bash
# Project-specific (won't conflict with other projects)
export MLFLOW_AWS_ACCESS_KEY_ID="your-access-key-id"
export MLFLOW_AWS_SECRET_ACCESS_KEY="your-secret-access-key"
export MLFLOW_AWS_DEFAULT_REGION="us-east-1"
export MLFLOW_ARTIFACT_ROOT="s3://your-bucket-name/mlflow-artifacts"

# OR use global credentials (may conflict with other projects)
export AWS_ACCESS_KEY_ID="your-access-key-id"
export AWS_SECRET_ACCESS_KEY="your-secret-access-key"
export AWS_DEFAULT_REGION="us-east-1"
export MLFLOW_ARTIFACT_ROOT="s3://your-bucket-name/mlflow-artifacts"
```

### 3. Start MLflow Server

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root s3://your-bucket-name/mlflow-artifacts \
  --host 0.0.0.0 \
  --port 5000
```

### 4. Run Training

```bash
python train/train.py --mlflow-experiment-name my-experiment
```

You should see:
```
MLflow Configuration:
  Tracking URI: http://localhost:5000
  Storage Type: S3 (Remote)
  Artifact Root: s3://your-bucket-name/mlflow-artifacts
```

## Persistent Configuration

### Option 1: Shell Configuration File

Add to `~/.bashrc` or `~/.zshrc`:

```bash
export AWS_ACCESS_KEY_ID="your-access-key-id"
export AWS_SECRET_ACCESS_KEY="your-secret-access-key"
export AWS_DEFAULT_REGION="us-east-1"
export MLFLOW_ARTIFACT_ROOT="s3://your-bucket-name/mlflow-artifacts"
```

Then `source ~/.bashrc` or `source ~/.zshrc`.

### Option 2: .env File

Create a `.env` file in the project root (already in `.gitignore`):

```bash
# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_ARTIFACT_ROOT=s3://your-bucket-name/mlflow-artifacts

# AWS Credentials
AWS_ACCESS_KEY_ID=your-access-key-id
AWS_SECRET_ACCESS_KEY=your-secret-access-key
AWS_DEFAULT_REGION=us-east-1
```

Load before running scripts:
```bash
set -a; source .env; set +a
python train/train.py ...
```

## IAM Policy (Minimum Required)

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
        "arn:aws:s3:::your-bucket-name/*",
        "arn:aws:s3:::your-bucket-name"
      ]
    }
  ]
}
```

## Switch Back to Local Storage

```bash
unset MLFLOW_ARTIFACT_ROOT
mlflow ui  # Defaults to local storage
```

## Troubleshooting

| Error | Solution |
|-------|----------|
| "boto3 is required" | `pip install boto3` |
| "AWS_ACCESS_KEY_ID must be set" | Set AWS credentials in environment |
| "NoSuchBucket" | Check bucket name, region, and permissions |
| "Access Denied" | Verify IAM permissions |

## What Gets Stored in S3?

- Model checkpoints (`.zip` files)
- Plots and visualizations
- Custom artifacts logged during training
- **Not stored**: Experiment metadata (run IDs, parameters, metrics) - these stay in the backend store (SQLite/PostgreSQL)

## Cost Estimate

Approximate costs for 100 GB of artifacts:
- **S3 Standard**: ~$2.30/month
- **S3 Infrequent Access**: ~$1.25/month (after 30 days)
- **Data Transfer**: Out to internet ~$9/GB (free within AWS region)

Use lifecycle policies to automatically move old artifacts to cheaper storage tiers.

## Project-Specific Credentials (No Conflicts!)

To avoid conflicts with other AWS projects, use `MLFLOW_AWS_*` prefixed credentials:

```bash
# In .env file or as environment variables
MLFLOW_AWS_ACCESS_KEY_ID=mlflow-key       # For this project only
MLFLOW_AWS_SECRET_ACCESS_KEY=mlflow-secret # For this project only

# Your global credentials remain unchanged
AWS_ACCESS_KEY_ID=main-project-key         # For other projects
AWS_SECRET_ACCESS_KEY=main-project-secret  # For other projects
```

The `MLFLOW_AWS_*` credentials take precedence and won't affect other projects!

See [docs/mlflow_project_credentials.md](./mlflow_project_credentials.md) for details.

## Important Notes

1. **Backend store** (experiment metadata) is separate from **artifact store** (files)
2. MLflow server must be started with `--default-artifact-root s3://...` 
3. All scripts automatically detect and use S3 when configured
4. No code changes needed - just environment variables or `.env` file
5. Can switch between local and S3 anytime
6. `.env` file is automatically loaded - no manual sourcing needed
7. Use `MLFLOW_AWS_*` prefix to avoid conflicts with other AWS projects

## Need Help?

- **Full setup guide**: [docs/mlflow_s3_setup.md](./mlflow_s3_setup.md)
- **Project credentials**: [docs/mlflow_project_credentials.md](./mlflow_project_credentials.md)
- **Test your setup**: `python scripts/test_mlflow_s3.py`

