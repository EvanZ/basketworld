from __future__ import annotations

def defender_is_guarding_offense(env, defender_id: int) -> bool:
    """Return True if any offensive player is within guard distance of the defender."""
    if env.defender_guard_distance <= 0:
        return False
    if not env.positions:
        return False

    def_pos = env.positions[defender_id]
    for off_id in env.offense_ids:
        if env._hex_distance(def_pos, env.positions[off_id]) <= env.defender_guard_distance:
            return True
    return False


def calculate_offensive_lane_hexes(env) -> set:
    """Calculate the hexes that make up the offensive lane (painted area)."""
    lane_hexes = set()
    lane_width = env.three_second_lane_width
    lane_height = env.three_second_lane_height

    basket_col, basket_row = env._axial_to_offset(*env.basket_position)
    for col in range(env.court_width):
        for row in range(env.court_height):
            q, r = env._offset_to_axial(col, row)
            dist = env._hex_distance((q, r), env.basket_position)
            if dist >= lane_height:
                continue
            row_offset = abs(row - basket_row)
            if row_offset <= lane_width:
                lane_hexes.add((q, r))
    return lane_hexes


def calculate_defensive_lane_hexes(env) -> set:
    """Defensive lane uses same geometry as offensive lane."""
    return calculate_offensive_lane_hexes(env)


def update_offensive_lane_counters(env, actions, results) -> None:
    """
    Update offensive lane counters and append violations to results (mutates results).
    Mirrors original logic in HexagonBasketballEnv._resolve_movement.
    """
    if not env.offensive_three_seconds_enabled:
        return

    for oid in env.offense_ids:
        if tuple(env.positions[oid]) in env.offensive_lane_hexes:
            env._offensive_lane_steps[oid] = env._offensive_lane_steps.get(oid, 0) + 1
        else:
            env._offensive_lane_steps[oid] = 0

    for oid in env.offense_ids:
        steps_in_lane = env._offensive_lane_steps.get(oid, 0)
        in_lane = tuple(env.positions[oid]) in env.offensive_lane_hexes
        has_ball = oid == env.ball_holder

        if in_lane:
            if steps_in_lane >= env.three_second_max_steps:
                if not has_ball:
                    results["turnovers"].append(
                        {
                            "player_id": oid,
                            "reason": "offensive_three_seconds",
                            "turnover_pos": env.positions[oid],
                        }
                    )
                    if env.ball_holder is not None:
                        env._turnover_to_defense(env.ball_holder)
                    break
                elif has_ball and steps_in_lane > env.three_second_max_steps:
                    from basketworld.envs.basketworld_env_v2 import ActionType

                    action_taken = ActionType(actions[oid])
                    if action_taken != ActionType.SHOOT:
                        results["turnovers"].append(
                            {
                                "player_id": oid,
                                "reason": "offensive_three_seconds",
                                "turnover_pos": env.positions[oid],
                            }
                        )
                        env._turnover_to_defense(oid)
                        break


def resolve_movement(env, actions, results, current_positions):
    """
    Resolve movement and collisions; updates env.positions and results.
    Mirrors original _process_simultaneous_actions movement block.
    """
    # Skip movement processing if already terminal (shots/turnovers)
    if results.get("shots") or results.get("turnovers"):
        return

    intended_moves = {}
    for player_id, action_val in enumerate(actions):
        from basketworld.envs.basketworld_env_v2 import ActionType

        action = ActionType(action_val)
        if ActionType.MOVE_E.value <= action.value <= ActionType.MOVE_SE.value:
            direction_idx = action.value - ActionType.MOVE_E.value
            new_pos = env._get_adjacent_position(current_positions[player_id], direction_idx)

            if env._is_valid_position(*new_pos):
                if (new_pos == env.basket_position) and (not env.allow_dunks):
                    if player_id == env.ball_holder:
                        results["turnovers"].append(
                            {"player_id": player_id, "reason": "move_out_of_bounds", "turnover_pos": new_pos}
                        )
                        env._turnover_to_defense(player_id)
                    results["moves"][player_id] = {"success": False, "reason": "basket_collision"}
                else:
                    intended_moves[player_id] = new_pos
            else:
                if player_id == env.ball_holder:
                    results["turnovers"].append(
                        {"player_id": player_id, "reason": "move_out_of_bounds", "turnover_pos": new_pos}
                    )
                    env._turnover_to_defense(player_id)
                results["moves"][player_id] = {"success": False, "reason": "out_of_bounds"}

    if intended_moves:
        occupied_start = set(current_positions)
        to_remove = []
        for pid, dest in intended_moves.items():
            if dest in occupied_start:
                results["moves"][pid] = {"success": False, "reason": "occupied_neighbor"}
                to_remove.append(pid)
        for pid in to_remove:
            intended_moves.pop(pid, None)

    static_players = set(range(env.n_players)) - set(intended_moves.keys())
    occupied_by_static = {current_positions[pid] for pid in static_players}

    move_destinations = {}
    from collections import defaultdict

    move_destinations = defaultdict(list)
    for player_id, dest in intended_moves.items():
        move_destinations[dest].append(player_id)

    final_positions = list(current_positions)

    for dest, players_intending_to_move in move_destinations.items():
        if dest in occupied_by_static:
            for player_id in players_intending_to_move:
                results["moves"][player_id] = {"success": False, "reason": "collision_static"}
            continue

        if len(players_intending_to_move) > 1:
            winner = env._rng.choice(players_intending_to_move)
            final_positions[winner] = dest
            results["moves"][winner] = {"success": True, "new_position": dest}
            for player_id in players_intending_to_move:
                if player_id != winner:
                    results["moves"][player_id] = {"success": False, "reason": "collision_dynamic"}
            results["collisions"].append(
                {"position": dest, "players": players_intending_to_move, "winner": winner}
            )
        else:
            player_id = players_intending_to_move[0]
            final_positions[player_id] = dest
            results["moves"][player_id] = {"success": True, "new_position": dest}

    env.positions = final_positions
    update_offensive_lane_counters(env, actions, results)
