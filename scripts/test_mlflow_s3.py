#!/usr/bin/env python3
"""
Test MLflow S3 Configuration Script

This script verifies that MLflow is properly configured to use S3 storage.
It checks environment variables, AWS credentials, and connectivity.

Usage:
    python scripts/test_mlflow_s3.py
"""

import os
import sys


def test_s3_configuration():
    """Test MLflow S3 configuration and connectivity."""
    print("=" * 60)
    print("MLflow S3 Configuration Test")
    print("=" * 60)
    print()

    # Check environment variables
    print("1. Checking environment variables...")
    print("-" * 60)

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    artifact_root = os.environ.get("MLFLOW_ARTIFACT_ROOT")
    aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    aws_region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    s3_endpoint = os.environ.get("MLFLOW_S3_ENDPOINT_URL")

    print(f"  MLFLOW_TRACKING_URI: {tracking_uri}")
    print(
        f"  MLFLOW_ARTIFACT_ROOT: {artifact_root or '(not set - using local storage)'}"
    )
    print(f"  AWS_ACCESS_KEY_ID: {'✓ set' if aws_access_key else '✗ not set'}")
    print(f"  AWS_SECRET_ACCESS_KEY: {'✓ set' if aws_secret_key else '✗ not set'}")
    print(f"  AWS_DEFAULT_REGION: {aws_region}")
    if s3_endpoint:
        print(f"  MLFLOW_S3_ENDPOINT_URL: {s3_endpoint}")
    print()

    use_s3 = artifact_root and artifact_root.startswith("s3://")

    if not use_s3:
        print("ℹ️  S3 storage is NOT configured (using local storage)")
        print()
        print("To enable S3 storage, set:")
        print("  export MLFLOW_ARTIFACT_ROOT='s3://your-bucket/mlflow-artifacts'")
        print()
        return True

    print("✓ S3 storage is configured")
    print()

    # Check boto3
    print("2. Checking boto3 installation...")
    print("-" * 60)
    try:
        import boto3

        print(f"  ✓ boto3 version: {boto3.__version__}")
        print()
    except ImportError:
        print("  ✗ boto3 is not installed")
        print()
        print("Please install boto3:")
        print("  pip install boto3")
        print()
        return False

    # Check AWS credentials
    print("3. Checking AWS credentials...")
    print("-" * 60)
    if not aws_access_key or not aws_secret_key:
        print("  ✗ AWS credentials are not set")
        print()
        print("Please set your AWS credentials:")
        print("  export AWS_ACCESS_KEY_ID='your-access-key'")
        print("  export AWS_SECRET_ACCESS_KEY='your-secret-key'")
        print()
        return False

    print("  ✓ AWS credentials are set")
    print()

    # Test S3 connectivity
    print("4. Testing S3 connectivity...")
    print("-" * 60)
    try:
        # Parse bucket name from artifact root
        bucket_name = artifact_root.replace("s3://", "").split("/")[0]
        print(f"  Bucket: {bucket_name}")

        # Create S3 client
        s3_kwargs = {"region_name": aws_region}
        if s3_endpoint:
            s3_kwargs["endpoint_url"] = s3_endpoint

        s3 = boto3.client("s3", **s3_kwargs)

        # Try to head the bucket
        s3.head_bucket(Bucket=bucket_name)
        print(f"  ✓ Successfully connected to bucket '{bucket_name}'")
        print()

        # Try to list objects (just to verify we have permissions)
        prefix = artifact_root.replace(f"s3://{bucket_name}/", "")
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, MaxKeys=1)
        print(
            f"  ✓ Successfully listed objects in '{prefix}' (found {response.get('KeyCount', 0)} objects)"
        )
        print()

    except Exception as e:
        print(f"  ✗ Failed to connect to S3: {e}")
        print()
        print("Please verify:")
        print("  1. The bucket exists")
        print("  2. The bucket name is correct")
        print("  3. Your AWS credentials have access to the bucket")
        print("  4. The AWS region is correct")
        print()
        return False

    # Test MLflow configuration
    print("5. Testing MLflow configuration...")
    print("-" * 60)
    try:
        from basketworld.utils.mlflow_config import get_mlflow_config

        config = get_mlflow_config()
        print(config)
        print()
        print("  ✓ MLflow configuration loaded successfully")
        print()

        if config.use_s3:
            print("  ✓ S3 storage will be used for artifacts")
        else:
            print("  ℹ️  Local storage will be used for artifacts")
        print()

    except Exception as e:
        print(f"  ✗ Failed to load MLflow configuration: {e}")
        print()
        return False

    # Test MLflow server connectivity
    print("6. Testing MLflow server connectivity...")
    print("-" * 60)
    try:
        import mlflow
        from basketworld.utils.mlflow_config import setup_mlflow

        setup_mlflow(verbose=False)

        # Try to list experiments
        experiments = mlflow.search_experiments()
        print(f"  ✓ Successfully connected to MLflow server")
        print(f"  Found {len(experiments)} experiment(s)")
        print()

    except Exception as e:
        print(f"  ✗ Failed to connect to MLflow server: {e}")
        print()
        print("Please ensure the MLflow server is running:")
        if use_s3:
            print(f"  mlflow server \\")
            print(f"    --backend-store-uri sqlite:///mlflow.db \\")
            print(f"    --default-artifact-root {artifact_root} \\")
            print(f"    --port 5000")
        else:
            print("  mlflow ui")
        print()
        return False

    print("=" * 60)
    print("✅ All checks passed! MLflow S3 storage is properly configured.")
    print("=" * 60)
    print()
    return True


if __name__ == "__main__":
    success = test_s3_configuration()
    sys.exit(0 if success else 1)
