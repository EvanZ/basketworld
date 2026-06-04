from __future__ import annotations

import math

DEFAULT_BW_THREE_POINT_DISTANCE = 4.25
NBA_MAX_THREE_POINT_DISTANCE_METERS = 7.24
NBA_CORNER_THREE_POINT_DISTANCE_METERS = 6.70


def meters_per_bw_unit_for_three_point_radius(
    *,
    three_point_distance: float = DEFAULT_BW_THREE_POINT_DISTANCE,
    three_point_radius_meters: float = NBA_MAX_THREE_POINT_DISTANCE_METERS,
) -> float:
    """Scale BW cartesian units so the arc apex matches a target metric radius."""

    distance = float(three_point_distance)
    radius = float(three_point_radius_meters)
    if distance <= 0.0:
        raise ValueError("three_point_distance must be positive")
    if radius <= 0.0:
        raise ValueError("three_point_radius_meters must be positive")
    return radius / (distance * math.sqrt(3.0))


DEFAULT_METERS_PER_BW_UNIT = meters_per_bw_unit_for_three_point_radius()
