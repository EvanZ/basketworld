#!/usr/bin/env python3
"""
MLflow Deleted Runs Cleanup Script

This script cleans up deleted MLflow runs by removing their associated
artifacts and run data from the filesystem. When you delete a run in MLflow UI,
it only marks the run as deleted in the metadata but doesn't remove the actual
files from disk, which can take up significant space.

Usage:
    python cleanup_mlflow_deleted_runs.py [--dry-run] [--tracking-uri URI]

Options:
    --dry-run           Show what would be deleted without actually deleting
    --tracking-uri URI  MLflow tracking URI (default: http://localhost:5000)
    --mlruns-dir DIR    Path to mlruns directory (default: ./mlruns)
    --mlartifacts-dir DIR  Path to mlartifacts directory (default: ./mlartifacts)
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import List, Tuple

import mlflow
from mlflow.tracking import MlflowClient


def format_size(bytes_size: int) -> str:
    """Format bytes into human-readable size."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def get_dir_size(path: Path) -> int:
    """Calculate total size of a directory recursively."""
    total = 0
    try:
        if path.is_file():
            return path.stat().st_size
        elif path.is_dir():
            for entry in path.rglob("*"):
                if entry.is_file():
                    try:
                        total += entry.stat().st_size
                    except (OSError, PermissionError):
                        pass
    except (OSError, PermissionError):
        pass
    return total


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


def clean_mlruns(
    mlruns_dir: Path, experiment_id: str, run_id: str, dry_run: bool = False
) -> int:
    """
    Remove run directory from mlruns.

    Returns:
        Size in bytes of what was (or would be) deleted.
    """
    run_path = mlruns_dir / experiment_id / run_id

    if not run_path.exists():
        return 0

    size = get_dir_size(run_path)

    if not dry_run:
        try:
            shutil.rmtree(run_path)
            print(f"  ✓ Removed mlruns data: {run_path} ({format_size(size)})")
        except Exception as e:
            print(f"  ✗ Failed to remove {run_path}: {e}")
            return 0
    else:
        print(f"  [DRY RUN] Would remove mlruns data: {run_path} ({format_size(size)})")

    return size


def clean_mlartifacts(
    mlartifacts_dir: Path, experiment_id: str, run_id: str, dry_run: bool = False
) -> int:
    """
    Remove artifacts directory for a run.

    Returns:
        Size in bytes of what was (or would be) deleted.
    """
    # MLflow stores artifacts in mlartifacts/{experiment_id}/{run_id}/artifacts/
    artifact_path = mlartifacts_dir / experiment_id / run_id

    if not artifact_path.exists():
        return 0

    size = get_dir_size(artifact_path)

    if not dry_run:
        try:
            shutil.rmtree(artifact_path)
            print(f"  ✓ Removed artifacts: {artifact_path} ({format_size(size)})")
        except Exception as e:
            print(f"  ✗ Failed to remove {artifact_path}: {e}")
            return 0
    else:
        print(
            f"  [DRY RUN] Would remove artifacts: {artifact_path} ({format_size(size)})"
        )

    return size


def main():
    parser = argparse.ArgumentParser(
        description="Clean up deleted MLflow runs from filesystem",
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
        "--mlruns-dir",
        type=str,
        default="./mlruns",
        help="Path to mlruns directory (default: ./mlruns)",
    )
    parser.add_argument(
        "--mlartifacts-dir",
        type=str,
        default="./mlartifacts",
        help="Path to mlartifacts directory (default: ./mlartifacts)",
    )

    args = parser.parse_args()

    # Set up paths
    mlruns_dir = Path(args.mlruns_dir).resolve()
    mlartifacts_dir = Path(args.mlartifacts_dir).resolve()

    print(f"MLflow Cleanup Script")
    print(f"{'=' * 80}")
    print(f"Tracking URI: {args.tracking_uri}")
    print(f"MLruns directory: {mlruns_dir}")
    print(f"MLartifacts directory: {mlartifacts_dir}")

    if args.dry_run:
        print("\n⚠️  DRY RUN MODE - No files will be deleted\n")
    else:
        print("\n⚠️  WARNING: This will permanently delete files from disk!")
        response = input("Continue? (yes/no): ")
        if response.lower() not in ["yes", "y"]:
            print("Aborted.")
            return
        print()

    # Connect to MLflow
    if args.tracking_uri == "http://localhost:5000":
        # Use the default config if not explicitly overridden
        from basketworld.utils.mlflow_config import setup_mlflow

        setup_mlflow(verbose=False)
    else:
        mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()

    try:
        # Test connection
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

    total_size = 0

    for i, (exp_id, run_id, run_name) in enumerate(deleted_runs, 1):
        print(f"{i}. Run: {run_name} (ID: {run_id}, Experiment: {exp_id})")

        # Clean mlruns
        size_mlruns = clean_mlruns(mlruns_dir, exp_id, run_id, args.dry_run)
        total_size += size_mlruns

        # Clean mlartifacts
        size_mlartifacts = clean_mlartifacts(
            mlartifacts_dir, exp_id, run_id, args.dry_run
        )
        total_size += size_mlartifacts

        print()

    print(f"{'=' * 80}")
    if args.dry_run:
        print(f"Would reclaim approximately: {format_size(total_size)}")
        print("\nRun without --dry-run to actually delete the files.")
    else:
        print(f"Successfully reclaimed: {format_size(total_size)}")
        print(f"Cleaned up {len(deleted_runs)} deleted run(s)")


if __name__ == "__main__":
    main()
