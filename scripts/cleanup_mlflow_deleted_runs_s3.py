#!/usr/bin/env python3
"""
MLflow Deleted Runs Cleanup Script (S3 Version)

This script cleans up deleted MLflow runs by removing their associated
artifacts from S3 and run data from the local database/filesystem.

Usage:
    python cleanup_mlflow_deleted_runs_s3.py [--dry-run] [--tracking-uri URI]

Options:
    --dry-run           Show what would be deleted without actually deleting
    --tracking-uri URI  MLflow tracking URI (default: http://localhost:5000)
    --s3-bucket BUCKET  S3 bucket name (default: basketworld)
    --s3-prefix PREFIX  S3 prefix (default: mlflow-artifacts)
    --aws-profile NAME  AWS profile name (optional; uses default chain if omitted)
    --skip-s3           Skip S3 deletion and only remove deleted-run metadata from mlflow.db
"""

import argparse
import os
import sqlite3
from pathlib import Path
from typing import List, Tuple

import mlflow
from mlflow.tracking import MlflowClient
import boto3
from botocore.exceptions import ClientError, ProfileNotFound


def apply_mlflow_aws_env_fallbacks() -> list[str]:
    """
    Map MLFLOW_AWS_* vars into AWS_* vars for boto3 if AWS_* are unset.
    Returns a list of keys that were populated via fallback.
    """
    mapped = []
    fallback_pairs = [
        ("MLFLOW_AWS_ACCESS_KEY_ID", "AWS_ACCESS_KEY_ID"),
        ("MLFLOW_AWS_SECRET_ACCESS_KEY", "AWS_SECRET_ACCESS_KEY"),
        ("MLFLOW_AWS_SESSION_TOKEN", "AWS_SESSION_TOKEN"),
        ("MLFLOW_AWS_DEFAULT_REGION", "AWS_DEFAULT_REGION"),
    ]
    for src, dst in fallback_pairs:
        if not os.environ.get(dst) and os.environ.get(src):
            os.environ[dst] = os.environ[src]
            mapped.append(f"{src}->{dst}")
    return mapped


def get_deleted_runs(client: MlflowClient) -> List[Tuple[str, str, str]]:
    """
    Get all deleted runs from all experiments.

    Returns:
        List of tuples: (experiment_id, run_id, run_name)
    """
    deleted_runs = []

    # Get all experiments (including deleted ones)
    experiments = client.search_experiments(view_type=mlflow.entities.ViewType.ALL)

    for experiment in experiments:
        # Search for deleted runs in this experiment
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="",
            run_view_type=mlflow.entities.ViewType.DELETED_ONLY,
        )

        for run in runs:
            run_name = run.data.tags.get("mlflow.runName", "unnamed")
            deleted_runs.append((experiment.experiment_id, run.info.run_id, run_name))

    return deleted_runs


def clean_s3_artifacts(
    s3_client,
    bucket: str,
    prefix: str,
    experiment_id: str,
    run_id: str,
    dry_run: bool = False,
) -> int:
    """
    Remove artifacts from S3 for a deleted run.

    Returns:
        Number of objects deleted.
    """
    # S3 path: s3://bucket/prefix/experiment_id/run_id/artifacts/
    s3_prefix = f"{prefix}/{experiment_id}/{run_id}/"

    try:
        # List all objects with this prefix
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=s3_prefix)

        if "Contents" not in response:
            return 0

        objects = response["Contents"]
        num_objects = len(objects)

        if dry_run:
            print(
                f"  [DRY RUN] Would delete {num_objects} object(s) from s3://{bucket}/{s3_prefix}"
            )
            for obj in objects[:5]:  # Show first 5
                print(f"    - {obj['Key']}")
            if num_objects > 5:
                print(f"    ... and {num_objects - 5} more")
        else:
            # Delete objects (in batches of 1000 if needed)
            objects_to_delete = [{"Key": obj["Key"]} for obj in objects]

            # Delete in batches of 1000 (S3 limit)
            for i in range(0, len(objects_to_delete), 1000):
                batch = objects_to_delete[i : i + 1000]
                s3_client.delete_objects(
                    Bucket=bucket, Delete={"Objects": batch, "Quiet": True}
                )

            print(
                f"  ✓ Deleted {num_objects} object(s) from s3://{bucket}/{s3_prefix}"
            )

        return num_objects

    except ClientError as e:
        print(f"  ✗ Failed to delete S3 artifacts: {e}")
        return 0


def permanently_delete_run(
    db_path: str, run_id: str, dry_run: bool = False
) -> bool:
    """
    Permanently delete a run from the MLflow database using direct SQL.

    This removes all traces of the run from mlflow.db, including:
    - metrics
    - params
    - tags
    - run metadata
    """
    if dry_run:
        print(f"  [DRY RUN] Would permanently delete run from database")
        return True
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Delete from all related tables
        tables_to_clean = [
            "metrics",
            "params",
            "tags",
            "latest_metrics",
        ]
        
        rows_deleted = 0
        for table in tables_to_clean:
            cursor.execute(f"DELETE FROM {table} WHERE run_uuid = ?", (run_id,))
            rows_deleted += cursor.rowcount
        
        # Finally, delete the run itself
        cursor.execute("DELETE FROM runs WHERE run_uuid = ?", (run_id,))
        rows_deleted += cursor.rowcount
        
        conn.commit()
        conn.close()
        
        print(f"  ✓ Permanently deleted run from database ({rows_deleted} rows)")
        return True
        
    except Exception as e:
        print(f"  ✗ Failed to delete run from database: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Clean up deleted MLflow runs (S3 version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default="http://localhost:5000",
        help="MLflow tracking URI (default: http://localhost:5000)",
    )
    parser.add_argument(
        "--s3-bucket",
        type=str,
        default="basketworld",
        help="S3 bucket name (default: basketworld)",
    )
    parser.add_argument(
        "--s3-prefix",
        type=str,
        default="mlflow-artifacts",
        help="S3 prefix for artifacts (default: mlflow-artifacts)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="./mlflow.db",
        help="Path to mlflow.db (default: ./mlflow.db)",
    )
    parser.add_argument(
        "--aws-profile",
        type=str,
        default=os.environ.get("AWS_PROFILE"),
        help="AWS profile name to use for S3 (default: AWS_PROFILE env or boto3 default chain)",
    )
    parser.add_argument(
        "--aws-region",
        type=str,
        default=(
            os.environ.get("AWS_DEFAULT_REGION")
            or os.environ.get("MLFLOW_AWS_DEFAULT_REGION")
            or "us-east-1"
        ),
        help="AWS region for S3/STS checks (default: AWS_DEFAULT_REGION or MLFLOW_AWS_DEFAULT_REGION or us-east-1)",
    )
    parser.add_argument(
        "--skip-s3",
        action="store_true",
        help="Skip S3 artifact deletion and only clean deleted-run metadata from mlflow.db",
    )

    args = parser.parse_args()

    print(f"MLflow S3 Cleanup Script")
    print(f"{'=' * 80}")
    print(f"Tracking URI: {args.tracking_uri}")
    print(f"Database: {args.db_path}")
    print(f"S3 Location: s3://{args.s3_bucket}/{args.s3_prefix}/")
    print(f"AWS Region: {args.aws_region}")
    if args.skip_s3:
        print("S3 Mode: skipped (metadata-only cleanup)")
    elif args.aws_profile:
        print(f"AWS Profile: {args.aws_profile}")

    mapped_env = apply_mlflow_aws_env_fallbacks()
    if mapped_env:
        print("AWS env fallback applied:", ", ".join(mapped_env))

    if args.dry_run:
        print("\n⚠️  DRY RUN MODE - No files will be deleted\n")
    else:
        print("\n⚠️  WARNING: This will permanently delete:")
        if not args.skip_s3:
            print("   - Artifacts from S3")
        print("   - Run metadata from mlflow.db")
        response = input("Continue? (yes/no): ")
        if response.lower() not in ["yes", "y"]:
            print("Aborted.")
            return
        print()

    # Connect to MLflow
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()

    s3_client = None
    if not args.skip_s3:
        try:
            session = None
            used_profile_label = None
            if args.aws_profile:
                try:
                    session = boto3.Session(profile_name=args.aws_profile)
                    used_profile_label = args.aws_profile
                except ProfileNotFound:
                    print(
                        f"Warning: AWS profile '{args.aws_profile}' was not found. "
                        "Falling back to default credential chain."
                    )
                    # If AWS_PROFILE points to a missing profile, boto3.Session() can
                    # fail as well, so temporarily clear it for fallback resolution.
                    saved_aws_profile = os.environ.pop("AWS_PROFILE", None)
                    try:
                        session = boto3.Session()
                        used_profile_label = "default credential chain"
                    finally:
                        if saved_aws_profile is not None:
                            os.environ["AWS_PROFILE"] = saved_aws_profile
            else:
                session = boto3.Session()
                used_profile_label = session.profile_name or "default credential chain"

            print(f"Using AWS auth source: {used_profile_label}")
            s3_client = session.client("s3", region_name=args.aws_region)
            # Test S3 connection
            s3_client.head_bucket(Bucket=args.s3_bucket)
        except ProfileNotFound as e:
            print(f"Error: AWS profile '{args.aws_profile}' was not found.")
            print(f"Please configure it or rerun with --aws-profile <existing-profile> or --skip-s3")
            print(f"Details: {e}")
            return
        except Exception as e:
            print(f"Error: Could not connect to S3 bucket {args.s3_bucket}")
            print(f"Please ensure AWS credentials are set up or use --skip-s3: {e}")
            return

    try:
        # Test MLflow connection
        client.search_experiments()
    except Exception as e:
        print(f"Error: Could not connect to MLflow at {args.tracking_uri}")
        print(f"Please ensure MLflow server is running: {e}")
        return

    # Get all deleted runs
    print("Searching for deleted runs...")
    deleted_runs = get_deleted_runs(client)

    if not deleted_runs:
        print("No deleted runs found. Nothing to clean up!")
        return

    print(f"Found {len(deleted_runs)} deleted run(s)\n")

    total_objects = 0

    for i, (exp_id, run_id, run_name) in enumerate(deleted_runs, 1):
        print(f"{i}. Run: {run_name} (ID: {run_id}, Experiment: {exp_id})")

        if not args.skip_s3:
            # Clean S3 artifacts
            num_objects = clean_s3_artifacts(
                s3_client, args.s3_bucket, args.s3_prefix, exp_id, run_id, args.dry_run
            )
            total_objects += num_objects
        else:
            print("  - Skipped S3 artifact deletion (--skip-s3)")

        # Permanently delete from database
        permanently_delete_run(args.db_path, run_id, args.dry_run)

        print()

    print(f"{'=' * 80}")
    if args.dry_run:
        print(f"Would delete:")
        if not args.skip_s3:
            print(f"  - {total_objects} object(s) from S3")
        print(f"  - {len(deleted_runs)} run(s) from database")
        print("\nRun without --dry-run to actually delete.")
    else:
        print(f"✅ Successfully cleaned up:")
        if not args.skip_s3:
            print(f"  - {total_objects} object(s) from S3")
        print(f"  - {len(deleted_runs)} run(s) from database (permanently)")
        print("\nAll traces of deleted runs have been removed.")


if __name__ == "__main__":
    main()
