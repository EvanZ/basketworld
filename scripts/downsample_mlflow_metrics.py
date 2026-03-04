#!/usr/bin/env python3
"""
Downsample MLflow metric history in a SQLite backend store.

Keeps every Nth metric point per (run_uuid, key) and always keeps the latest step.
This is useful for shrinking mlflow.db while preserving trend curves.
"""

import argparse
import sqlite3
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Downsample MLflow metrics in mlflow.db"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="./mlflow.db",
        help="Path to SQLite MLflow backend DB (default: ./mlflow.db)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=10,
        help="Keep every Nth point per metric series (default: 10)",
    )
    parser.add_argument(
        "--run-id",
        action="append",
        default=[],
        help="Optional run_uuid to include (repeatable). If omitted, uses active runs.",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Target all runs in metrics table instead of active runs only.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show counts only; do not modify database.",
    )
    parser.add_argument(
        "--vacuum",
        action="store_true",
        help="Run VACUUM after downsampling (requires free disk space).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive confirmation prompt.",
    )
    return parser.parse_args()


def resolve_target_runs(conn: sqlite3.Connection, run_ids: list[str], all_runs: bool) -> list[str]:
    cur = conn.cursor()
    if run_ids:
        return sorted(set(run_ids))
    if all_runs:
        cur.execute("SELECT DISTINCT run_uuid FROM metrics")
        return sorted(r[0] for r in cur.fetchall())
    cur.execute("SELECT run_uuid FROM runs WHERE lifecycle_stage = 'active'")
    return sorted(r[0] for r in cur.fetchall())


def create_temp_target_runs(conn: sqlite3.Connection, run_ids: list[str]) -> None:
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS _target_runs")
    cur.execute("CREATE TEMP TABLE _target_runs (run_uuid TEXT PRIMARY KEY)")
    cur.executemany(
        "INSERT INTO _target_runs(run_uuid) VALUES (?)",
        [(rid,) for rid in run_ids],
    )
    conn.commit()


def count_target_metrics(conn: sqlite3.Connection) -> int:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT COUNT(*)
        FROM metrics m
        JOIN _target_runs t ON t.run_uuid = m.run_uuid
        """
    )
    return int(cur.fetchone()[0])


def estimate_delete_count(conn: sqlite3.Connection, stride: int) -> int:
    cur = conn.cursor()
    cur.execute(
        """
        WITH ranked AS (
            SELECT
                m.rowid AS rowid,
                m.step AS step,
                ROW_NUMBER() OVER (
                    PARTITION BY m.run_uuid, m.key
                    ORDER BY m.step, m.timestamp, m.rowid
                ) AS rn,
                MAX(m.step) OVER (
                    PARTITION BY m.run_uuid, m.key
                ) AS max_step
            FROM metrics m
            JOIN _target_runs t ON t.run_uuid = m.run_uuid
        )
        SELECT COUNT(*)
        FROM ranked
        WHERE ((rn - 1) % ?) != 0
          AND step != max_step
        """,
        (int(stride),),
    )
    return int(cur.fetchone()[0])


def refresh_latest_metrics(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        DELETE FROM latest_metrics
        WHERE run_uuid IN (SELECT run_uuid FROM _target_runs)
        """
    )
    cur.execute(
        """
        INSERT INTO latest_metrics (key, value, timestamp, step, is_nan, run_uuid)
        SELECT key, value, timestamp, step, is_nan, run_uuid
        FROM (
            SELECT
                m.key,
                m.value,
                m.timestamp,
                m.step,
                m.is_nan,
                m.run_uuid,
                ROW_NUMBER() OVER (
                    PARTITION BY m.run_uuid, m.key
                    ORDER BY m.step DESC, m.timestamp DESC, m.rowid DESC
                ) AS rn
            FROM metrics m
            JOIN _target_runs t ON t.run_uuid = m.run_uuid
        )
        WHERE rn = 1
        """
    )


def downsample_metrics(conn: sqlite3.Connection, stride: int) -> int:
    before = conn.total_changes
    cur = conn.cursor()
    cur.execute("BEGIN IMMEDIATE")
    cur.execute(
        """
        WITH ranked AS (
            SELECT
                m.rowid AS rowid,
                m.step AS step,
                ROW_NUMBER() OVER (
                    PARTITION BY m.run_uuid, m.key
                    ORDER BY m.step, m.timestamp, m.rowid
                ) AS rn,
                MAX(m.step) OVER (
                    PARTITION BY m.run_uuid, m.key
                ) AS max_step
            FROM metrics m
            JOIN _target_runs t ON t.run_uuid = m.run_uuid
        )
        DELETE FROM metrics
        WHERE rowid IN (
            SELECT rowid
            FROM ranked
            WHERE ((rn - 1) % ?) != 0
              AND step != max_step
        )
        """,
        (int(stride),),
    )
    refresh_latest_metrics(conn)
    conn.commit()
    return int(conn.total_changes - before)


def main() -> int:
    args = parse_args()
    if args.stride < 2:
        print("Error: --stride must be >= 2")
        return 1

    db_path = Path(args.db_path).resolve()
    if not db_path.exists():
        print(f"Error: database not found: {db_path}")
        return 1

    conn = sqlite3.connect(str(db_path))
    try:
        # Avoid filesystem temp-file issues during large window queries.
        conn.execute("PRAGMA temp_store = MEMORY;")
        run_ids = resolve_target_runs(conn, args.run_id, args.all_runs)
        if not run_ids:
            print("No target runs found.")
            return 0

        create_temp_target_runs(conn, run_ids)
        total_target_rows = count_target_metrics(conn)
        to_delete = estimate_delete_count(conn, args.stride)

        print("MLflow Metric Downsampling")
        print("=" * 80)
        print(f"Database: {db_path}")
        print(f"Target runs: {len(run_ids)}")
        print(f"Stride: keep every {args.stride}th point + latest")
        print(f"Target metric rows: {total_target_rows:,}")
        print(f"Rows to delete (estimated): {to_delete:,}")
        if total_target_rows > 0:
            kept = total_target_rows - to_delete
            print(f"Rows kept (estimated): {kept:,} ({(kept / total_target_rows) * 100:.1f}%)")

        if args.dry_run:
            print("\nDry run complete. No changes made.")
            return 0

        if not args.yes:
            response = input("\nProceed with downsampling? (yes/no): ").strip().lower()
            if response not in {"yes", "y"}:
                print("Aborted.")
                return 0

        deleted = downsample_metrics(conn, args.stride)
        print(f"\nDeleted metric rows: {deleted:,}")

        if args.vacuum:
            print("Running VACUUM...")
            conn.execute("PRAGMA temp_store = MEMORY;")
            conn.execute("VACUUM;")
            print("VACUUM complete.")

        print("Done.")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
