from __future__ import annotations

from typing import List, Tuple


def generate_initial_positions(env) -> List[Tuple[int, int]]:
    """
    Generate initial positions with distances defined RELATIVE to the basket:
    - Offense: any valid cell with distance >= spawn_distance (negative => 0)
               and distance <= max_spawn_distance (if set)
    - Defense: closer to basket than the matched offense and distance >= spawn_distance (negative => 0)
               and distance <= max_spawn_distance (if set)
      If no such cells, broaden progressively to avoid spawn failures.
    """
    taken_positions: set[Tuple[int, int]] = set()

    all_cells: List[Tuple[int, int]] = []
    for row in range(env.court_height):
        for col in range(env.court_width):
            all_cells.append(env._offset_to_axial(col, row))

    boundary_margin = max(0, int(getattr(env, "offense_spawn_boundary_margin", 0)))
    boundary_distance: dict[Tuple[int, int], int] = {}
    if boundary_margin > 0:
        boundary_cells: List[Tuple[int, int]] = []
        for cell in all_cells:
            if not env._is_valid_position(*cell):
                continue
            for dq, dr in env.hex_directions:
                nbr = (cell[0] + dq, cell[1] + dr)
                if not env._is_valid_position(*nbr):
                    boundary_cells.append(cell)
                    break
        if boundary_cells:
            for cell in all_cells:
                if not env._is_valid_position(*cell):
                    continue
                boundary_distance[cell] = min(
                    env._hex_distance(cell, boundary) for boundary in boundary_cells
                )

    min_spawn_dist_offense = max(0, env.spawn_distance)
    min_spawn_dist_defense = max(0, env.spawn_distance - 1)
    max_spawn_dist = env.max_spawn_distance

    offense_candidates = []
    for cell in all_cells:
        if cell == env.basket_position:
            continue
        if not env._is_valid_position(*cell):
            continue
        if boundary_margin > 0:
            if boundary_distance.get(cell, 0) < boundary_margin:
                continue
        dist = env._hex_distance(cell, env.basket_position)
        if dist >= min_spawn_dist_offense:
            if max_spawn_dist is None or dist <= max_spawn_dist:
                offense_candidates.append(cell)

    if len(offense_candidates) < env.players_per_side:
        offense_candidates = [
            cell
            for cell in all_cells
            if cell != env.basket_position and env._is_valid_position(*cell)
        ]
        if len(offense_candidates) < env.players_per_side:
            raise ValueError("Not enough valid cells to place offense")

    rng = getattr(env, "_rng")
    offense_positions = [
        tuple(p)
        for p in rng.choice(offense_candidates, size=env.players_per_side, replace=False).tolist()
    ]
    for pos in offense_positions:
        taken_positions.add(pos)

    # Randomize offense order to avoid fixed defense-to-offense matchups.
    offense_for_match = list(offense_positions)
    rng.shuffle(offense_for_match)

    defense_positions = []
    for off_pos in offense_for_match:
        candidates = []
        target_dist_from_offense = max(0, env.defender_spawn_distance)
        for cell in all_cells:
            if cell == env.basket_position:
                continue
            if not env._is_valid_position(*cell):
                continue
            if cell in taken_positions:
                continue
            off_dist = env._hex_distance(off_pos, env.basket_position)
            cell_dist = env._hex_distance(cell, env.basket_position)
            if (
                cell_dist < off_dist
                and cell_dist >= min_spawn_dist_defense
                and (max_spawn_dist is None or cell_dist <= max_spawn_dist)
                and abs(env._hex_distance(cell, off_pos) - target_dist_from_offense) <= 1
            ):
                candidates.append(cell)

        if len(candidates) < 2:
            candidates = [
                cell
                for cell in all_cells
                if cell != env.basket_position
                and env._is_valid_position(*cell)
                and cell not in taken_positions
                and env._hex_distance(cell, env.basket_position) < env._hex_distance(off_pos, env.basket_position)
                and env._hex_distance(cell, env.basket_position) >= min_spawn_dist_defense
                and (max_spawn_dist is None or env._hex_distance(cell, env.basket_position) <= max_spawn_dist)
            ]
        if len(candidates) == 0:
            candidates = [
                cell
                for cell in all_cells
                if cell != env.basket_position
                and env._is_valid_position(*cell)
                and cell not in taken_positions
                and env._hex_distance(cell, env.basket_position) >= min_spawn_dist_defense
                and (max_spawn_dist is None or env._hex_distance(cell, env.basket_position) <= max_spawn_dist)
            ]
        if len(candidates) == 0:
            candidates = [
                cell
                for cell in all_cells
                if cell != env.basket_position
                and env._is_valid_position(*cell)
                and cell not in taken_positions
            ]
        if len(candidates) == 0:
            raise ValueError("Not enough valid cells to place defense")

        dist_pairs = [(cell, env._hex_distance(cell, off_pos)) for cell in candidates]
        dist_pairs.sort(key=lambda item: item[1])
        min_dist = dist_pairs[0][1]
        closest = [cell for cell, dist in dist_pairs if dist == min_dist]
        # Randomize tie-breaks among closest cells to avoid directional bias.
        chosen = closest[int(rng.integers(0, len(closest)))]
        defense_positions.append(tuple(chosen))
        taken_positions.add(tuple(chosen))

    return offense_positions + defense_positions
