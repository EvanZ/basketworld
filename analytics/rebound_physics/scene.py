from __future__ import annotations

import math
from xml.sax.saxutils import escape

from analytics.rebound_physics.model import PhysicsConfig, ShotOrigin


def _fmt(value: float) -> str:
    return f"{value:.6f}"


def _geom_contact_attrs(config: PhysicsConfig, prefix: str) -> str:
    timeconst = getattr(config, f"{prefix}_contact_solref_timeconst")
    dampratio = getattr(config, f"{prefix}_contact_solref_dampratio")
    solimp_width = getattr(config, f"{prefix}_contact_solimp_width")
    solimp_midpoint = getattr(config, f"{prefix}_contact_solimp_midpoint")
    solimp_power = getattr(config, f"{prefix}_contact_solimp_power")
    attrs: list[str] = []
    if timeconst is not None or dampratio is not None:
        attrs.append(
            f'solref="{_fmt(config.contact_solref_timeconst if timeconst is None else timeconst)} '
            f'{_fmt(config.contact_solref_dampratio if dampratio is None else dampratio)}"'
        )
    if solimp_width is not None or solimp_midpoint is not None or solimp_power is not None:
        attrs.append(
            f'solimp="{_fmt(config.contact_solimp_width if solimp_width is None else solimp_width)} '
            f'{_fmt(config.contact_solimp_midpoint if solimp_midpoint is None else solimp_midpoint)} '
            f'{_fmt(config.contact_solimp_power if solimp_power is None else solimp_power)}"'
        )
    return "" if not attrs else " " + " ".join(attrs)


def _rim_segment_xml(config: PhysicsConfig) -> str:
    parts: list[str] = []
    rim_contact_attrs = _geom_contact_attrs(config, "rim")
    n = max(8, int(config.rim_segments))
    for idx in range(n):
        a0 = 2.0 * math.pi * idx / n
        a1 = 2.0 * math.pi * (idx + 1) / n
        x0 = config.rim_radius * math.cos(a0)
        y0 = config.rim_radius * math.sin(a0)
        x1 = config.rim_radius * math.cos(a1)
        y1 = config.rim_radius * math.sin(a1)
        parts.append(
            "    "
            f'<geom name="rim_{idx:02d}" type="capsule" size="{_fmt(config.rim_tube_radius)}" '
            f'fromto="{_fmt(x0)} {_fmt(y0)} {_fmt(config.rim_height)} '
            f'{_fmt(x1)} {_fmt(y1)} {_fmt(config.rim_height)}" '
            f'rgba="1.0 0.35 0.05 1" friction="0.65 0.02 0.001"{rim_contact_attrs}/>\n'
        )
    return "".join(parts)


def build_scene_xml(
    *,
    config: PhysicsConfig | None = None,
    shot_origin: ShotOrigin | None = None,
    model_name: str = "basketworld_rebound_physics",
) -> str:
    """Build a small MJCF scene with floor, backboard, rim approximation, and ball."""
    config = config or PhysicsConfig()
    shot_origin = shot_origin or ShotOrigin()
    board_z = config.rim_height + config.backboard_center_z_offset
    board_y = config.backboard_y + config.backboard_thickness * 0.5
    backboard_contact_attrs = _geom_contact_attrs(config, "backboard")

    return f"""<mujoco model="{escape(model_name)}">
  <compiler angle="radian" inertiafromgeom="true"/>
  <option timestep="{_fmt(config.timestep)}" gravity="0 0 -{_fmt(config.gravity)}"/>
  <default>
    <geom contype="1" conaffinity="1" condim="3"
          solref="{_fmt(config.contact_solref_timeconst)} {_fmt(config.contact_solref_dampratio)}"
          solimp="{_fmt(config.contact_solimp_width)} {_fmt(config.contact_solimp_midpoint)} {_fmt(config.contact_solimp_power)}"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="{_fmt(config.floor_size)} {_fmt(config.floor_size)} 0.1"
          rgba="0.08 0.10 0.12 1" friction="0.8 0.02 0.001"/>
    <body name="backboard" pos="0 {_fmt(board_y)} {_fmt(board_z)}">
      <geom name="backboard" type="box"
            size="{_fmt(config.backboard_width * 0.5)} {_fmt(config.backboard_thickness * 0.5)} {_fmt(config.backboard_height * 0.5)}"
            rgba="0.85 0.9 1.0 0.45" friction="0.55 0.02 0.001"{backboard_contact_attrs}/>
    </body>
{_rim_segment_xml(config)}    <body name="ball" pos="{_fmt(shot_origin.x)} {_fmt(shot_origin.y)} {_fmt(shot_origin.z)}">
      <freejoint name="ball_free"/>
      <geom name="ball" type="sphere" size="{_fmt(config.ball_radius)}" mass="{_fmt(config.ball_mass)}"
            rgba="0.95 0.43 0.12 1" friction="0.75 0.02 0.001"/>
    </body>
  </worldbody>
</mujoco>
"""

