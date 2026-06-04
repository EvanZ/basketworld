from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from analytics.rebound_physics.model import PhysicsConfig, ShotOrigin
from analytics.rebound_physics.scene import build_scene_xml
from analytics.rebound_physics.simulate import require_mujoco

FT_TO_M = 0.3048


@dataclass(frozen=True)
class BounceResult:
    contact_solref_timeconst: float
    contact_solref_dampratio: float
    drop_height_ft: float
    target_bounce_height_ft: float
    first_contact_time: float | None
    rebound_peak_center_ft: float
    rebound_peak_bottom_ft: float
    rebound_height_error_ft: float
    rebound_ratio_bottom: float
    sim_time: float


def _parse_float_grid(text: str) -> list[float]:
    values: list[float] = []
    for piece in text.split(","):
        piece = piece.strip()
        if piece:
            values.append(float(piece))
    return values


def _floor_ball_contact(mujoco, model, data) -> bool:
    for contact_idx in range(data.ncon):
        contact = data.contact[contact_idx]
        names = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)) or "",
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)) or "",
        )
        if "ball" in names and "floor" in names:
            return True
    return False


def simulate_bounce(
    *,
    config: PhysicsConfig,
    drop_height_ft: float,
    target_bounce_height_ft: float,
) -> BounceResult:
    """Drop a stationary ball and measure the first post-impact rebound peak.

    Heights are reported relative to the bottom of the ball, because basketball
    bounce specs are usually easier to reason about as floor-to-ball clearance.
    """
    mujoco = require_mujoco()
    drop_bottom_m = drop_height_ft * FT_TO_M
    shot_origin = ShotOrigin(x=0.0, y=-2.0, z=drop_bottom_m + config.ball_radius)
    model = mujoco.MjModel.from_xml_string(
        build_scene_xml(config=config, shot_origin=shot_origin, model_name="basketworld_bounce_calibration")
    )
    data = mujoco.MjData(model)

    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    qpos_adr = int(model.jnt_qposadr[joint_id])
    qvel_adr = int(model.jnt_dofadr[joint_id])

    data.qpos[qpos_adr : qpos_adr + 3] = np.array([shot_origin.x, shot_origin.y, shot_origin.z])
    data.qpos[qpos_adr + 3 : qpos_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    data.qvel[qvel_adr : qvel_adr + 6] = 0.0
    mujoco.mj_forward(model, data)

    first_contact_time: float | None = None
    rebound_peak_center_m = float(data.xpos[body_id][2])
    bounced_up = False
    prev_vel_z = 0.0

    steps = int(config.duration / config.timestep)
    for _ in range(steps):
        mujoco.mj_step(model, data)
        z = float(data.xpos[body_id][2])
        vel_z = float(data.qvel[qvel_adr + 2])

        if first_contact_time is None:
            if _floor_ball_contact(mujoco, model, data):
                first_contact_time = float(data.time)
                rebound_peak_center_m = z
        else:
            rebound_peak_center_m = max(rebound_peak_center_m, z)
            if vel_z > 0.0:
                bounced_up = True
            if bounced_up and prev_vel_z > 0.0 and vel_z <= 0.0:
                break
        prev_vel_z = vel_z

    rebound_peak_bottom_ft = max(0.0, (rebound_peak_center_m - config.ball_radius) / FT_TO_M)
    rebound_peak_center_ft = rebound_peak_center_m / FT_TO_M
    error_ft = rebound_peak_bottom_ft - target_bounce_height_ft
    return BounceResult(
        contact_solref_timeconst=config.contact_solref_timeconst,
        contact_solref_dampratio=config.contact_solref_dampratio,
        drop_height_ft=drop_height_ft,
        target_bounce_height_ft=target_bounce_height_ft,
        first_contact_time=first_contact_time,
        rebound_peak_center_ft=float(rebound_peak_center_ft),
        rebound_peak_bottom_ft=float(rebound_peak_bottom_ft),
        rebound_height_error_ft=float(error_ft),
        rebound_ratio_bottom=float(rebound_peak_bottom_ft / max(1e-9, drop_height_ft)),
        sim_time=float(data.time),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate MuJoCo ball/court bounce against a drop-test target.")
    parser.add_argument("--drop-height-ft", type=float, default=6.0)
    parser.add_argument("--target-bounce-height-ft", type=float, default=4.0)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--timestep", type=float, default=0.002)
    parser.add_argument(
        "--contact-timeconst",
        type=float,
        default=None,
        help="Run one contact time constant. If omitted, sweep --timeconst-grid.",
    )
    parser.add_argument(
        "--timeconst-grid",
        type=str,
        default="0.008,0.010,0.012,0.015,0.020",
    )
    parser.add_argument(
        "--contact-dampratio",
        type=float,
        default=None,
        help="Run one damping ratio. If omitted, sweep --dampratio-grid.",
    )
    parser.add_argument(
        "--dampratio-grid",
        type=str,
        default="0.05,0.08,0.10,0.12,0.15,0.20,0.25,0.35,0.50,0.75,1.00",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("analytics/rebound_physics/outputs"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    timeconsts = [args.contact_timeconst] if args.contact_timeconst is not None else _parse_float_grid(args.timeconst_grid)
    dampratios = [args.contact_dampratio] if args.contact_dampratio is not None else _parse_float_grid(args.dampratio_grid)
    if not timeconsts:
        raise SystemExit("No contact time constants provided.")
    if not dampratios:
        raise SystemExit("No damping ratios provided.")

    results: list[BounceResult] = []
    for timeconst in timeconsts:
        for dampratio in dampratios:
            config = PhysicsConfig(
                duration=args.duration,
                timestep=args.timestep,
                contact_solref_timeconst=float(timeconst),
                contact_solref_dampratio=float(dampratio),
            )
            results.append(
                simulate_bounce(
                    config=config,
                    drop_height_ft=args.drop_height_ft,
                    target_bounce_height_ft=args.target_bounce_height_ft,
                )
            )

    ranked = sorted(results, key=lambda result: abs(result.rebound_height_error_ft))
    summary = {
        "best": asdict(ranked[0]),
        "results": [asdict(result) for result in ranked],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "bounce_calibration_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("rank  timeconst  dampratio  rebound_bottom_ft  error_ft  ratio")
    for idx, result in enumerate(ranked[: min(12, len(ranked))], start=1):
        print(
            f"{idx:>4}  {result.contact_solref_timeconst:>9.3f}  "
            f"{result.contact_solref_dampratio:>9.3f}  "
            f"{result.rebound_peak_bottom_ft:>17.3f}  "
            f"{result.rebound_height_error_ft:>8.3f}  "
            f"{result.rebound_ratio_bottom:>5.3f}"
        )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
