# MLflow S3 Storage Setup Guide

This guide explains how to configure MLflow to use your personal S3 bucket for storing experiments and artifacts, instead of local storage.

## Overview

By default, MLflow stores experiment data and artifacts locally in the `mlruns/` and `mlartifacts/` directories. With S3 storage, you can:

- **Persist data in the cloud**: Keep your experiments and artifacts safe even if your local machine fails
- **Share across teams**: Multiple users can access the same experiments
- **Scale storage**: No local disk space limitations
- **Access from anywhere**: Train on one machine, analyze on another

## Prerequisites

1. **AWS Account**: You need an AWS account with S3 access
2. **S3 Bucket**: Create a dedicated S3 bucket for MLflow artifacts
3. **AWS Credentials**: IAM user with appropriate S3 permissions

## Step 1: Install Dependencies

The required boto3 package has been added to `requirements.txt`. Install it with:

```bash
pip install -r requirements.txt
```

Or install boto3 directly:

```bash
pip install boto3
```

## Step 2: Create an S3 Bucket

1. Log into the AWS Console
2. Navigate to S3
3. Create a new bucket (e.g., `my-basketworld-mlflow`)
4. Choose a region close to your training infrastructure
5. Keep default settings (or adjust based on your security requirements)

## Step 3: Set Up IAM User and Permissions

Create an IAM user with the following S3 permissions:

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

Replace `YOUR-BUCKET-NAME` with your actual bucket name.

After creating the IAM user, generate and save the **Access Key ID** and **Secret Access Key**.

## Step 4: Configure Environment Variables

Set the following environment variables in your shell:

```bash
# AWS Credentials
export AWS_ACCESS_KEY_ID="your-access-key-id"
export AWS_SECRET_ACCESS_KEY="your-secret-access-key"
export AWS_DEFAULT_REGION="us-east-1"  # Or your preferred region

# MLflow Configuration
export MLFLOW_TRACKING_URI="http://localhost:5000"
export MLFLOW_ARTIFACT_ROOT="s3://YOUR-BUCKET-NAME/mlflow-artifacts"
```

### Making Configuration Persistent

To avoid setting these every time, add them to your shell configuration file:

**For bash** (`~/.bashrc` or `~/.bash_profile`):
```bash
echo 'export AWS_ACCESS_KEY_ID="your-access-key-id"' >> ~/.bashrc
echo 'export AWS_SECRET_ACCESS_KEY="your-secret-access-key"' >> ~/.bashrc
echo 'export AWS_DEFAULT_REGION="us-east-1"' >> ~/.bashrc
echo 'export MLFLOW_ARTIFACT_ROOT="s3://YOUR-BUCKET-NAME/mlflow-artifacts"' >> ~/.bashrc
source ~/.bashrc
```

**For zsh** (`~/.zshrc`):
```bash
echo 'export AWS_ACCESS_KEY_ID="your-access-key-id"' >> ~/.zshrc
echo 'export AWS_SECRET_ACCESS_KEY="your-secret-access-key"' >> ~/.zshrc
echo 'export AWS_DEFAULT_REGION="us-east-1"' >> ~/.zshrc
echo 'export MLFLOW_ARTIFACT_ROOT="s3://YOUR-BUCKET-NAME/mlflow-artifacts"' >> ~/.zshrc
source ~/.zshrc
```

### Using a .env File (Alternative)

You can also create a `.env` file in the project root:

```bash
# .env
AWS_ACCESS_KEY_ID=your-access-key-id
AWS_SECRET_ACCESS_KEY=your-secret-access-key
AWS_DEFAULT_REGION=us-east-1
MLFLOW_ARTIFACT_ROOT=s3://YOUR-BUCKET-NAME/mlflow-artifacts
```

Then load it before running scripts:
```bash
set -a; source .env; set +a
```

**Note**: Add `.env` to `.gitignore` to avoid committing credentials!

## Step 5: Start MLflow Server with S3 Backend

When using S3 storage, you need to start the MLflow server with the S3 artifact root:

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root s3://YOUR-BUCKET-NAME/mlflow-artifacts \
  --host 0.0.0.0 \
  --port 5000
```

**Important**: The `--default-artifact-root` must match the `MLFLOW_ARTIFACT_ROOT` environment variable.

### Option: Using PostgreSQL for Metadata

For production use, consider using PostgreSQL instead of SQLite:

```bash
mlflow server \
  --backend-store-uri postgresql://user:password@localhost/mlflow \
  --default-artifact-root s3://YOUR-BUCKET-NAME/mlflow-artifacts \
  --host 0.0.0.0 \
  --port 5000
```

## Step 6: Run Training with S3 Storage

Once configured, simply run your training as usual:

```bash
python train/train.py --mlflow-experiment-name my-experiment
```

The script will automatically detect the S3 configuration and print:

```
MLflow Configuration:
  Tracking URI: http://localhost:5000
  Storage Type: S3 (Remote)
  Artifact Root: s3://YOUR-BUCKET-NAME/mlflow-artifacts
```

## Verification

To verify S3 storage is working:

1. **Check the training logs**: You should see "Storage Type: S3 (Remote)" when training starts
2. **Check your S3 bucket**: After training starts, check the bucket for artifacts
3. **MLflow UI**: Open http://localhost:5000 and verify experiments are being logged

## Switching Between Local and S3 Storage

### Using Local Storage

Simply unset the S3 environment variable:

```bash
unset MLFLOW_ARTIFACT_ROOT
```

Start MLflow with local storage:

```bash
mlflow ui
```

### Using S3 Storage

Set the environment variable:

```bash
export MLFLOW_ARTIFACT_ROOT="s3://YOUR-BUCKET-NAME/mlflow-artifacts"
```

Start MLflow server with S3:

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root s3://YOUR-BUCKET-NAME/mlflow-artifacts
```

## Advanced Configuration

### Using Custom S3 Endpoints (MinIO, LocalStack)

If you're using MinIO or another S3-compatible service:

```bash
export MLFLOW_S3_ENDPOINT_URL="http://localhost:9000"
export AWS_ACCESS_KEY_ID="minioadmin"
export AWS_SECRET_ACCESS_KEY="minioadmin"
export MLFLOW_ARTIFACT_ROOT="s3://mlflow"
```

### S3 Transfer Configuration

For large artifacts, you can configure S3 transfer settings:

```bash
export AWS_S3_MAX_CONCURRENT_REQUESTS=50
export AWS_S3_MULTIPART_THRESHOLD=8388608  # 8MB
export AWS_S3_MULTIPART_CHUNKSIZE=8388608  # 8MB
```

## Troubleshooting

### "boto3 is required for S3 storage"

**Solution**: Install boto3:
```bash
pip install boto3
```

### "AWS_ACCESS_KEY_ID environment variable must be set"

**Solution**: Set your AWS credentials:
```bash
export AWS_ACCESS_KEY_ID="your-access-key-id"
export AWS_SECRET_ACCESS_KEY="your-secret-access-key"
```

### "NoSuchBucket" Error

**Solution**: Ensure:
1. The bucket exists
2. The region is correct
3. Your IAM user has access to the bucket

### Artifacts Not Uploading to S3

**Solution**: 
1. Check that MLflow server was started with `--default-artifact-root s3://...`
2. Verify AWS credentials are set
3. Check S3 bucket permissions
4. Check the MLflow server logs for errors

### "Access Denied" Error

**Solution**: Verify your IAM user has the correct S3 permissions (PutObject, GetObject, ListBucket, DeleteObject)

## Cost Considerations

MLflow artifacts can be large (model checkpoints, plots, etc.). Consider:

1. **S3 Storage Class**: Use S3 Standard for frequently accessed data, S3 Infrequent Access or Glacier for older experiments
2. **Lifecycle Policies**: Automatically transition old artifacts to cheaper storage tiers
3. **Delete Old Experiments**: Use the cleanup script to remove experiments you no longer need

Example S3 lifecycle policy to transition old artifacts:

```json
{
  "Rules": [
    {
      "Id": "Archive old MLflow artifacts",
      "Status": "Enabled",
      "Transitions": [
        {
          "Days": 30,
          "StorageClass": "STANDARD_IA"
        },
        {
          "Days": 90,
          "StorageClass": "GLACIER"
        }
      ],
      "Prefix": "mlflow-artifacts/"
    }
  ]
}
```

## Security Best Practices

1. **Never commit credentials**: Add `.env` to `.gitignore`
2. **Use IAM roles**: If running on EC2, use IAM instance roles instead of access keys
3. **Restrict bucket access**: Use bucket policies to limit access
4. **Enable encryption**: Enable S3 server-side encryption
5. **Enable versioning**: Protect against accidental deletions
6. **Audit access**: Enable S3 access logging and CloudTrail

## Migration from Local to S3

To migrate existing local experiments to S3:

1. Start MLflow server with S3 backend
2. Use the MLflow API to copy artifacts:

```python
import mlflow
from mlflow.tracking import MlflowClient

# Set up source (local) and destination (S3)
mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

# This is a manual process - artifacts are not automatically migrated
# You'll need to re-run experiments or manually copy artifacts to S3
```

Note: There's no automated migration tool. Consider starting fresh with S3 or manually copying important artifacts.

## Summary

With S3 storage configured:
- ✅ Experiments and artifacts are stored in S3
- ✅ Data persists beyond local machine lifecycle
- ✅ Multiple users can access the same experiments
- ✅ No local disk space limitations
- ✅ All existing code works without changes

The codebase automatically detects and uses S3 when `MLFLOW_ARTIFACT_ROOT` is set to an S3 URI.

