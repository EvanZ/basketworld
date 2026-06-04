#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analytics.rebound_sim.model import CourtSpec, ReboundParams, build_court, simulate_rebounds
from analytics.rebound_sim.plotting import (
    plot_conditioned_rebound_heatmap,
    plot_rebound_region_flow,
    plot_rebound_summary,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prototype BW rebound landing/winner model.")
    parser.add_argument("--samples", type=int, default=50000, help="Number of missed shots to simulate.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--court-rows", type=int, default=9)
    parser.add_argument("--court-cols", type=int, default=8)
    parser.add_argument("--three-point-distance", type=float, default=4.25)
    parser.add_argument("--three-point-short-distance", type=float, default=3.0)
    parser.add_argument("--defense-rebound-bias", type=float, default=ReboundParams.defense_rebound_bias)
    parser.add_argument("--boxout-bias", type=float, default=ReboundParams.boxout_bias)
    parser.add_argument("--target-distance-weight", type=float, default=ReboundParams.target_distance_weight)
    parser.add_argument("--winner-temperature", type=float, default=ReboundParams.winner_temperature)
    parser.add_argument(
        "--shot-cells",
        type=str,
        default="",
        help="Optional flat comma-separated axial q,r shot cells, e.g. --shot-cells=-4,8,-3,8.",
    )
    parser.add_argument(
        "--conditioned-samples",
        type=int,
        default=None,
        help="Samples per conditioned shot cell. Defaults to --samples.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("analytics/rebound_sim/outputs"))
    parser.add_argument("--no-plot", action="store_true", help="Skip PNG generation.")
    argv_list = _normalize_shot_cells_arg(sys.argv[1:] if argv is None else list(argv))
    return parser.parse_args(argv_list)


def _normalize_shot_cells_arg(argv: list[str]) -> list[str]:
    # argparse treats a separate value like "-4,8" as an option. Normalize
    # VSCode-style ["--shot-cells", "-4,8"] into "--shot-cells=-4,8".
    normalized: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--shot-cells" and i + 1 < len(argv):
            normalized.append(f"--shot-cells={argv[i + 1]}")
            i += 2
            continue
        normalized.append(arg)
        i += 1
    return normalized


def main() -> None:
    args = parse_args()
    court = build_court(
        CourtSpec(
            rows=args.court_rows,
            cols=args.court_cols,
            three_point_distance=args.three_point_distance,
            three_point_short_distance=args.three_point_short_distance,
        )
    )
    params = ReboundParams(
        defense_rebound_bias=args.defense_rebound_bias,
        boxout_bias=args.boxout_bias,
        target_distance_weight=args.target_distance_weight,
        winner_temperature=args.winner_temperature,
    )
    result = simulate_rebounds(args.samples, seed=args.seed, court=court, params=params)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "rebound_sim_summary.json"
    result.save_summary_json(summary_path)
    written_paths: list[Path] = [summary_path]
    if not args.no_plot:
        summary_png = args.out_dir / "rebound_sim_summary.png"
        flow_png = args.out_dir / "rebound_sim_region_flow.png"
        plot_rebound_summary(result, summary_png)
        plot_rebound_region_flow(result, flow_png)
        written_paths.extend([summary_png, flow_png])

    shot_cells = _parse_shot_cells(args.shot_cells)
    conditioned_summaries: dict[str, object] = {}
    conditioned_samples = int(args.conditioned_samples or args.samples)
    for i, cell in enumerate(shot_cells):
        if cell not in court.cell_index:
            valid = ", ".join(f"{q},{r}" for q, r in court.cells[:8])
            raise SystemExit(f"Shot cell {cell} is not on the court. Example valid cells: {valid}, ...")
        shot_idx = int(court.cell_index[cell])
        shot_indices = np.full(conditioned_samples, shot_idx, dtype=np.int32)
        conditioned = simulate_rebounds(
            conditioned_samples,
            seed=int(args.seed) + 1000 + i,
            court=court,
            params=params,
            shot_indices=shot_indices,
        )
        key = _shot_cell_key(cell)
        conditioned_summary = conditioned.summary()
        conditioned_summary["shot_cell"] = {"q": int(cell[0]), "r": int(cell[1])}
        conditioned_summary["shot_index"] = shot_idx
        conditioned_summaries[key] = conditioned_summary
        if not args.no_plot:
            conditioned_png = args.out_dir / f"rebound_heatmap_shot_{key}.png"
            plot_conditioned_rebound_heatmap(conditioned, conditioned_png, shot_cell=cell)
            written_paths.append(conditioned_png)
    if conditioned_summaries:
        conditioned_path = args.out_dir / "rebound_sim_conditioned_summary.json"
        conditioned_path.write_text(json.dumps(conditioned_summaries, indent=2, sort_keys=True) + "\n")
        written_paths.append(conditioned_path)

    print(json.dumps(result.summary(), indent=2, sort_keys=True))
    for path in written_paths:
        print(f"Wrote {path}")


def _parse_shot_cells(value: str) -> list[tuple[int, int]]:
    cleaned = value.strip()
    if not cleaned:
        return []
    for char in "();":
        cleaned = cleaned.replace(char, ",")
    cleaned = cleaned.replace(" ", "")
    parts = [part for part in cleaned.split(",") if part]
    if len(parts) % 2 != 0:
        raise SystemExit("--shot-cells expects an even number of comma-separated integers: q,r[,q,r...]")
    try:
        values = [int(part) for part in parts]
    except ValueError as exc:
        raise SystemExit("--shot-cells only accepts integer axial coordinates") from exc
    return [(values[i], values[i + 1]) for i in range(0, len(values), 2)]


def _shot_cell_key(cell: tuple[int, int]) -> str:
    q, r = cell
    return f"q{q}_r{r}"


if __name__ == "__main__":
    main()
