from __future__ import annotations

import math
from typing import Dict, List, Tuple, Set

import numpy as np


def offset_to_axial_formula(col: int, row: int) -> Tuple[int, int]:
    """Pure conversion odd-r offset -> axial."""
    q = col - (row - (row & 1)) // 2
    r = row
    return q, r


def axial_to_offset_formula(q: int, r: int) -> Tuple[int, int]:
    """Pure conversion axial -> odd-r offset."""
    col = q + (r - (r & 1)) // 2
    row = r
    return col, row


def axial_to_cartesian_formula(q: int, r: int) -> Tuple[float, float]:
    """Convert axial (q, r) to cartesian (x, y) matching rendering geometry."""
    size = 1.0
    x = size * (math.sqrt(3) * q + math.sqrt(3) / 2 * r)
    y = size * (1.5 * r)
    return x, y


def axial_to_cube(q: int, r: int) -> Tuple[int, int, int]:
    x, z = q, r
    y = -x - z
    return x, y, z


def cube_to_axial(x: int, y: int, z: int) -> Tuple[int, int]:
    return x, z


def rotate60_cw_cube(x: int, y: int, z: int) -> Tuple[int, int, int]:
    """Rotate cube (x, y, z) by 60 degrees clockwise."""
    return -z, -x, -y


def rotate_k60_axial(q: int, r: int, k: int) -> Tuple[int, int]:
    """Rotate axial (q, r) by k*60 degrees clockwise."""
    x, y, z = axial_to_cube(q, r)
    for _ in range(k % 6):
        x, y, z = rotate60_cw_cube(x, y, z)
    return cube_to_axial(x, y, z)


def hex_distance_formula(q1: int, r1: int, q2: int, r2: int) -> int:
    """Closed-form hex distance on axial coords."""
    return (abs(q1 - q2) + abs(q1 + r1 - q2 - r2) + abs(r1 - r2)) // 2


def precompute_coord_caches(
    court_width: int, court_height: int, cells: List[Tuple[int, int]]
) -> tuple[list[list[Tuple[int, int]]], Dict[Tuple[int, int], Tuple[int, int]], Dict[Tuple[int, int], Tuple[float, float]], set[Tuple[int, int]]]:
    """
    Precompute coordinate conversion caches for all on-court cells.
    Returns (offset_to_axial_cache, axial_to_offset_cache, axial_to_cart_cache, valid_axial_set).
    """
    offset_to_axial_cache = [
        [offset_to_axial_formula(c, r) for c in range(court_width)]
        for r in range(court_height)
    ]
    axial_to_offset_cache: Dict[Tuple[int, int], Tuple[int, int]] = {}
    axial_to_cart_cache: Dict[Tuple[int, int], Tuple[float, float]] = {}
    for r in range(court_height):
        for c in range(court_width):
            axial = offset_to_axial_cache[r][c]
            axial_to_offset_cache[axial] = (c, r)
            axial_to_cart_cache[axial] = axial_to_cartesian_formula(*axial)
    valid_axial = set(cells) if cells else set(axial_to_offset_cache.keys())
    return offset_to_axial_cache, axial_to_offset_cache, axial_to_cart_cache, valid_axial


def precompute_hex_distance_lut(
    cells: List[Tuple[int, int]],
) -> tuple[np.ndarray, Dict[Tuple[int, int], int]]:
    """Precompute hex distances between all on-court cells for fast lookup."""
    cell_index = {cell: idx for idx, cell in enumerate(cells)}
    n = len(cells)
    lut = np.zeros((n, n), dtype=np.int16)
    for i, (q1, r1) in enumerate(cells):
        for j, (q2, r2) in enumerate(cells):
            lut[i, j] = hex_distance_formula(q1, r1, q2, r2)
    return lut, cell_index


def point_to_line_distance(
    point: Tuple[int, int],
    line_start: Tuple[int, int],
    line_end: Tuple[int, int],
) -> float:
    """Perpendicular distance from point to line segment (axial coords; uses cartesian transform)."""
    px, py = axial_to_cartesian_formula(point[0], point[1])
    sx, sy = axial_to_cartesian_formula(line_start[0], line_start[1])
    ex, ey = axial_to_cartesian_formula(line_end[0], line_end[1])

    dx = ex - sx
    dy = ey - sy
    line_length_sq = dx * dx + dy * dy

    if line_length_sq == 0:
        return math.hypot(px - sx, py - sy)

    t = ((px - sx) * dx + (py - sy) * dy) / line_length_sq
    t = max(0.0, min(1.0, t))
    closest_x = sx + t * dx
    closest_y = sy + t * dy
    return math.hypot(px - closest_x, py - closest_y)


def get_position_on_line(
    point: Tuple[int, int],
    line_start: Tuple[int, int],
    line_end: Tuple[int, int],
) -> float:
    """Return projection parameter t for a point onto a line segment (axial coords)."""
    px, py = axial_to_cartesian_formula(point[0], point[1])
    sx, sy = axial_to_cartesian_formula(line_start[0], line_start[1])
    ex, ey = axial_to_cartesian_formula(line_end[0], line_end[1])

    dx = ex - sx
    dy = ey - sy
    line_length_sq = dx * dx + dy * dy
    if line_length_sq == 0:
        return 0.0
    return ((px - sx) * dx + (py - sy) * dy) / line_length_sq


def is_between_points(
    point: Tuple[int, int],
    line_start: Tuple[int, int],
    line_end: Tuple[int, int],
) -> bool:
    """Return True if projection of point onto line lies strictly between endpoints."""
    t = get_position_on_line(point, line_start, line_end)
    return 0.0 < t < 1.0


def compute_three_point_geometry(env, tolerance: float = 0.35) -> tuple[Set[Tuple[int, int]], Set[Tuple[int, int]], List[Tuple[float, float]]]:
    """
    Compute three-point regions and outline points given an environment that exposes
    basket position, court dimensions, and axial/cartesian helpers.
    Returns (three_point_hexes, three_point_line_hexes, outline_points_cartesian).
    """
    three_point_hexes: Set[Tuple[int, int]] = set()
    three_point_line_hexes: Set[Tuple[int, int]] = set()
    outline_points: List[Tuple[float, float]] = []

    if getattr(env, "three_point_distance", 0) <= 0:
        return three_point_hexes, three_point_line_hexes, outline_points

    basket_axial = env.basket_position
    hoop_x, hoop_y = env._axial_to_cartesian(*basket_axial)
    radius_cart = float(env.three_point_distance) * math.sqrt(3.0)
    short_band = (
        float(env.three_point_short_distance) * math.sqrt(3.0)
        if getattr(env, "three_point_short_distance", None) is not None
        else None
    )

    horizontal_reach = 0.0
    if short_band is not None and short_band < radius_cart:
        horizontal_reach = math.sqrt(radius_cart**2 - short_band**2)

    outline_seen: Set[Tuple[int, int]] = set()

    for row in range(env.court_height):
        for col in range(env.court_width):
            q, r = env._offset_to_axial(col, row)
            cell = (q, r)
            cx, cy = env._axial_to_cartesian(q, r)
            dx = cx - hoop_x
            dy = cy - hoop_y
            abs_dy = abs(dy)
            dist_cart = math.hypot(dx, dy)

            qualifies = False
            is_outline = False

            # Sampling logic to determine qualification (at least 50% of hex area)
            samples = [(cx, cy)]  # Center
            for i in range(6):
                angle_deg = 30 + 60 * i
                angle_rad = math.radians(angle_deg)
                sx = cx + 0.95 * math.cos(angle_rad)
                sy = cy + 0.95 * math.sin(angle_rad)
                samples.append((sx, sy))

            qualified_samples = 0
            for sx, sy in samples:
                sdx = sx - hoop_x
                sdy = sy - hoop_y
                sabs_dy = abs(sdy)
                sdist = math.hypot(sdx, sdy)

                is_3pt_pt = False

                if short_band is not None:
                    if dx < horizontal_reach:
                        if sabs_dy >= short_band:
                            is_3pt_pt = True
                    else:
                        if sdist >= radius_cart:
                            is_3pt_pt = True
                else:
                    if sdist >= radius_cart:
                        is_3pt_pt = True

                if is_3pt_pt:
                    qualified_samples += 1

            qualifies = (qualified_samples / len(samples)) >= 0.5

            # Outline check (distance-to-boundary heuristic)
            if short_band is not None:
                if dx < horizontal_reach:
                    dist_to_line = abs(abs_dy - short_band)
                    if dist_to_line <= tolerance:
                        is_outline = True
                else:
                    if abs(dist_cart - radius_cart) <= tolerance:
                        is_outline = True
            else:
                if abs(dist_cart - radius_cart) <= tolerance:
                    is_outline = True

            if qualifies:
                three_point_hexes.add(cell)
            if is_outline:
                three_point_line_hexes.add(cell)
                outline_seen.add(cell)

    outline_points = [env._axial_to_cartesian(q, r) for q, r in outline_seen]
    outline_points.sort(key=lambda pt: math.atan2(pt[1] - hoop_y, pt[0] - hoop_x))

    return three_point_hexes, three_point_line_hexes, outline_points
