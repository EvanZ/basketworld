from __future__ import annotations

import numpy as np

from analytics.rebound_physics.model import (
    ContactPoint,
    PhysicsConfig,
    ShotSamplerConfig,
    ShotOrigin,
    ShotParams,
    TrajectorySample,
    sample_shot_params,
)
from analytics.rebound_physics.scene import build_scene_xml


def require_mujoco():
    try:
        import mujoco  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "MuJoCo is required for rebound physics simulation. "
            "Install it with `.env/bin/pip install mujoco`."
        ) from exc
    return mujoco


def _geom_name(mujoco, model, geom_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id)) or ""


def _contact_label(mujoco, model, contact) -> str | None:
    names = (_geom_name(mujoco, model, contact.geom1), _geom_name(mujoco, model, contact.geom2))
    if "ball" not in names:
        return None
    if any(name.startswith("rim_") for name in names):
        return "rim"
    if "backboard" in names:
        return "backboard"
    if "floor" in names:
        return "floor"
    return "+".join(name for name in names if name and name != "ball") or "other"


def run_shot(
    shot: ShotParams,
    *,
    config: PhysicsConfig | None = None,
    seed: int = 0,
    shot_index: int = 0,
) -> TrajectorySample:
    mujoco = require_mujoco()
    config = config or PhysicsConfig()
    model = mujoco.MjModel.from_xml_string(build_scene_xml(config=config, shot_origin=shot.origin))
    data = mujoco.MjData(model)

    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    qpos_adr = int(model.jnt_qposadr[joint_id])
    qvel_adr = int(model.jnt_dofadr[joint_id])

    data.qpos[qpos_adr : qpos_adr + 3] = np.array([shot.origin.x, shot.origin.y, shot.origin.z])
    data.qpos[qpos_adr + 3 : qpos_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    data.qvel[qvel_adr : qvel_adr + 3] = np.array(shot.velocity)
    data.qvel[qvel_adr + 3 : qvel_adr + 6] = np.array(shot.angular_velocity)
    mujoco.mj_forward(model, data)

    made = False
    net_caught = False
    first_contact: str | None = None
    contact_sequence: list[str] = []
    contact_points: list[ContactPoint] = []
    contact_count = 0
    landing_xy: tuple[float, float] | None = None
    landing_time: float | None = None
    max_height = float(shot.origin.z)
    prev_pos = np.asarray(data.xpos[body_id], dtype=np.float64).copy()
    prev_z = float(prev_pos[2])
    prev_time = float(data.time)
    rim_crossing_xy: tuple[float, float] | None = None
    rim_crossing_distance: float | None = None
    rim_crossing_time: float | None = None
    trajectory_stride = max(1, int(config.trajectory_stride))
    trajectory: list[tuple[float, float, float]] = []
    trajectory_quat_wxyz: list[tuple[float, float, float, float]] = []

    def _current_ball_quat() -> tuple[float, float, float, float]:
        quat = np.asarray(data.qpos[qpos_adr + 3 : qpos_adr + 7], dtype=np.float64)
        return (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))

    def _append_trajectory_point(pos_xyz) -> None:
        point = (float(pos_xyz[0]), float(pos_xyz[1]), float(pos_xyz[2]))
        if not trajectory or trajectory[-1] != point:
            trajectory.append(point)
            trajectory_quat_wxyz.append(_current_ball_quat())

    _append_trajectory_point((shot.origin.x, shot.origin.y, shot.origin.z))

    steps = int(config.duration / config.timestep)
    for step_idx in range(steps):
        mujoco.mj_step(model, data)
        pos = np.asarray(data.xpos[body_id], dtype=np.float64)
        vel_z = float(data.qvel[qvel_adr + 2])
        max_height = max(max_height, float(pos[2]))

        if step_idx % trajectory_stride == 0:
            _append_trajectory_point(pos)

        if rim_crossing_xy is None and prev_z > config.rim_height >= float(pos[2]) and vel_z < 0.0:
            denom = prev_z - float(pos[2])
            alpha = 0.0 if abs(denom) < 1e-9 else (prev_z - config.rim_height) / denom
            alpha = float(np.clip(alpha, 0.0, 1.0))
            cross_xy = prev_pos[:2] + alpha * (pos[:2] - prev_pos[:2])
            rim_crossing_xy = (float(cross_xy[0]), float(cross_xy[1]))
            rim_crossing_distance = float(np.linalg.norm(cross_xy))
            rim_crossing_time = float(prev_time + alpha * (float(data.time) - prev_time))
            if rim_crossing_distance <= config.make_radius:
                made = True
                if config.net_catch_made and not net_caught:
                    catch_z = config.rim_height - 2.0 * config.ball_radius
                    data.qpos[qpos_adr : qpos_adr + 3] = np.array([cross_xy[0], cross_xy[1], catch_z])
                    data.qpos[qpos_adr + 3 : qpos_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
                    data.qvel[qvel_adr : qvel_adr + 3] = np.array([0.0, 0.0, -abs(config.net_downward_speed)])
                    data.qvel[qvel_adr + 3 : qvel_adr + 6] = np.zeros(3, dtype=np.float64)
                    mujoco.mj_forward(model, data)
                    pos = np.asarray(data.xpos[body_id], dtype=np.float64)
                    vel_z = float(data.qvel[qvel_adr + 2])
                    net_caught = True
                    _append_trajectory_point(pos)
        prev_pos = pos.copy()
        prev_z = float(pos[2])
        prev_time = float(data.time)

        for contact_idx in range(data.ncon):
            contact = data.contact[contact_idx]
            label = _contact_label(mujoco, model, contact)
            if label is None:
                continue
            contact_count += 1
            if first_contact is None:
                first_contact = label
            if not contact_sequence or contact_sequence[-1] != label:
                contact_sequence.append(label)
            if label in {"rim", "backboard"} and len(contact_points) < config.max_contact_points:
                cpos = np.asarray(contact.pos, dtype=np.float64)
                contact_points.append(
                    ContactPoint(
                        kind=label,
                        x=float(cpos[0]),
                        y=float(cpos[1]),
                        z=float(cpos[2]),
                        time=float(data.time),
                    )
                )
            if label == "floor" and landing_xy is None:
                landing_xy = (float(pos[0]), float(pos[1]))
                landing_time = float(data.time)
                if len(contact_points) < config.max_contact_points:
                    cpos = np.asarray(contact.pos, dtype=np.float64)
                    contact_points.append(
                        ContactPoint(
                            kind="floor",
                            x=float(cpos[0]),
                            y=float(cpos[1]),
                            z=float(cpos[2]),
                            time=float(data.time),
                        )
                    )

        if landing_time is not None and float(data.time) - landing_time >= 0.35:
            break

    settled_pos = np.asarray(data.xpos[body_id], dtype=np.float64)
    _append_trajectory_point(settled_pos)
    return TrajectorySample(
        seed=seed,
        shot_index=shot_index,
        shot=shot,
        made=made,
        first_contact=first_contact,
        contact_sequence=tuple(contact_sequence),
        contact_count=contact_count,
        landing_xy=landing_xy,
        settled_xy=(float(settled_pos[0]), float(settled_pos[1])),
        max_height=float(max_height),
        sim_time=float(data.time),
        rim_crossing_xy=rim_crossing_xy,
        rim_crossing_distance=rim_crossing_distance,
        rim_crossing_time=rim_crossing_time,
        contact_points=tuple(contact_points),
        trajectory_xyz=tuple(trajectory),
        trajectory_quat_wxyz=tuple(trajectory_quat_wxyz),
    )


def run_batch(
    *,
    samples: int,
    seed: int,
    origin: ShotOrigin | None = None,
    config: PhysicsConfig | None = None,
    sampler_config: ShotSamplerConfig | None = None,
) -> list[TrajectorySample]:
    config = config or PhysicsConfig()
    origin = origin or ShotOrigin()
    sampler_config = sampler_config or ShotSamplerConfig()
    seed_rng = np.random.default_rng(seed)
    output: list[TrajectorySample] = []
    for shot_index in range(samples):
        shot_seed = int(seed_rng.integers(0, 2**31 - 1))
        shot_rng = np.random.default_rng(shot_seed)
        shot = sample_shot_params(
            shot_rng,
            origin=origin,
            config=config,
            sampler_config=sampler_config,
        )
        output.append(run_shot(shot, config=config, seed=shot_seed, shot_index=shot_index))
    return output
