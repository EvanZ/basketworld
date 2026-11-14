"""Compact local trainer evaluation JSON, retaining verified gzip originals.

Dry-run by default. Only explicit --apply replaces files, and never MLflow data.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from basketworld_jax.eval.artifacts import compact_existing_historical_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Trainer runs directory")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--indices-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    paths = sorted(root.glob("*/checkpoints/historical_opponent_evaluation.json"))
    if not args.indices_only:
        paths += sorted(root.glob("*/checkpoints/historical_opponent_evaluations/update_*.json"))
    totals = {"files": 0, "before_bytes": 0, "after_bytes": 0, "archive_bytes": 0}
    for path in paths:
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Path outside requested directory: {path}")
        if not args.apply:
            print(json.dumps({"path": str(path), "bytes": path.stat().st_size}), flush=True)
            continue
        result = compact_existing_historical_artifact(path)
        print(json.dumps(result), flush=True)
        if result["changed"]:
            totals["files"] += 1
            for name in ("before_bytes", "after_bytes", "archive_bytes"):
                totals[name] += result[name]
        gc.collect()
    print(json.dumps({"totals": totals}), flush=True)


if __name__ == "__main__":
    main()
