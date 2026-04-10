from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_template_file(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"start template library not found: {file_path}")
    suffix = file_path.suffix.lower()
    text = file_path.read_text(encoding="utf-8")
    if suffix == ".json":
        data = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError(
                "YAML template libraries require PyYAML to be installed"
            ) from exc
        data = yaml.safe_load(text)
    else:
        raise ValueError(
            f"unsupported start template library format: {file_path.suffix or '<none>'}"
        )
    if not isinstance(data, dict):
        raise ValueError("start template library root must be a mapping")
    return data


def _normalize_anchor_entry(
    entry: dict[str, Any],
    *,
    players_per_side: int,
    team_name: str,
    entry_index: int,
) -> dict[str, Any]:
    slot_value = entry.get("slot")
    slot: int | None = None
    if slot_value is not None:
        slot = int(slot_value)
        if not (0 <= slot < players_per_side):
            raise ValueError(
                f"{team_name} slot must be in [0, {players_per_side - 1}], got {slot}"
            )

    anchor = entry.get("anchor")
    label = f"slot {slot}" if slot is not None else f"entry {entry_index}"
    if not isinstance(anchor, (list, tuple)) or len(anchor) != 2:
        raise ValueError(f"{team_name} {label} anchor must be a [q, r] pair")
    try:
        q, r = int(anchor[0]), int(anchor[1])
    except Exception as exc:
        raise ValueError(
            f"{team_name} {label} anchor must contain integer coordinates"
        ) from exc

    jitter_radius = int(entry.get("jitter_radius", 0) or 0)
    if jitter_radius < 0:
        raise ValueError(f"{team_name} {label} jitter_radius must be >= 0")

    normalized = {
        "anchor": [q, r],
        "jitter_radius": jitter_radius,
    }
    if slot is not None:
        normalized["slot"] = slot

    role = entry.get("role")
    if role is not None:
        normalized["role"] = str(role)

    if bool(entry.get("has_ball", False)):
        normalized["has_ball"] = True

    return normalized


def _normalize_team_entries(
    template_id: str,
    team_name: str,
    entries: Any,
    *,
    players_per_side: int,
) -> list[dict[str, Any]]:
    if not isinstance(entries, list) or len(entries) != players_per_side:
        raise ValueError(
            f"template '{template_id}' must define exactly {players_per_side} {team_name} entries"
        )
    normalized_entries = [
        _normalize_anchor_entry(
            entry,
            players_per_side=players_per_side,
            team_name=team_name,
            entry_index=entry_idx,
        )
        for entry_idx, entry in enumerate(entries)
    ]

    slots_present = [("slot" in entry) for entry in normalized_entries]
    if any(slots_present) and not all(slots_present):
        raise ValueError(
            f"template '{template_id}' {team_name} entries must either all include slot or all omit it"
        )

    if all(slots_present):
        slots = sorted(int(entry["slot"]) for entry in normalized_entries)
        expected_slots = list(range(players_per_side))
        if slots != expected_slots:
            raise ValueError(
                f"template '{template_id}' {team_name} slots must be exactly {expected_slots}"
            )
        normalized_entries = sorted(
            normalized_entries, key=lambda entry: int(entry["slot"])
        )
        for entry in normalized_entries:
            entry.pop("slot", None)

    return normalized_entries


def validate_start_template_library(
    library: dict[str, Any], players_per_side: int
) -> dict[str, Any]:
    if not isinstance(library, dict):
        raise ValueError("start template library must be a mapping")
    version = int(library.get("version", 1) or 1)
    if version != 1:
        raise ValueError(f"unsupported start template library version: {version}")
    library_pps = int(
        library.get("players_per_side", players_per_side) or players_per_side
    )
    if library_pps != int(players_per_side):
        raise ValueError(
            f"template library players_per_side={library_pps} does not match env players_per_side={players_per_side}"
        )
    templates_raw = library.get("templates")
    if not isinstance(templates_raw, list) or not templates_raw:
        raise ValueError(
            "start template library must contain a non-empty 'templates' list"
        )

    normalized_templates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for idx, template in enumerate(templates_raw):
        if not isinstance(template, dict):
            raise ValueError(f"template index {idx} must be a mapping")
        template_id = str(template.get("id", "")).strip()
        if not template_id:
            raise ValueError(f"template index {idx} is missing a non-empty id")
        if template_id in seen_ids:
            raise ValueError(f"duplicate start template id: {template_id}")
        seen_ids.add(template_id)

        weight = float(template.get("weight", 1.0) or 0.0)
        if weight <= 0.0:
            raise ValueError(f"template '{template_id}' weight must be > 0")
        mirrorable = bool(template.get("mirrorable", False))

        normalized_template: dict[str, Any] = {
            "id": template_id,
            "weight": weight,
            "mirrorable": mirrorable,
        }

        if template.get("shot_clock") is not None:
            normalized_template["shot_clock"] = int(template["shot_clock"])

        normalized_template["offense"] = _normalize_team_entries(
            template_id,
            "offense",
            template.get("offense"),
            players_per_side=players_per_side,
        )
        normalized_template["defense"] = _normalize_team_entries(
            template_id,
            "defense",
            template.get("defense"),
            players_per_side=players_per_side,
        )

        total_ball_entries = sum(
            1
            for team_name in ("offense", "defense")
            for entry in normalized_template[team_name]
            if bool(entry.get("has_ball", False))
        )
        legacy_ball_holder = template.get("ball_holder")
        if legacy_ball_holder is not None:
            if total_ball_entries > 0:
                raise ValueError(
                    f"template '{template_id}' cannot define both ball_holder and entry-level has_ball"
                )
            if not isinstance(legacy_ball_holder, dict):
                raise ValueError(
                    f"template '{template_id}' ball_holder must be a mapping"
                )
            team = str(legacy_ball_holder.get("team", "")).strip().lower()
            if team not in {"offense", "defense"}:
                raise ValueError(
                    f"template '{template_id}' ball_holder.team must be offense or defense"
                )
            slot = int(legacy_ball_holder.get("slot", -1))
            if not (0 <= slot < players_per_side):
                raise ValueError(
                    f"template '{template_id}' ball_holder.slot must be in [0, {players_per_side - 1}]"
                )
            normalized_template[team][slot]["has_ball"] = True
            total_ball_entries = 1

        if total_ball_entries > 1:
            raise ValueError(
                f"template '{template_id}' may mark at most one entry with has_ball"
            )

        normalized_templates.append(normalized_template)

    return {
        "version": version,
        "players_per_side": int(players_per_side),
        "templates": normalized_templates,
    }


def load_start_template_library(
    path: str | Path, *, players_per_side: int
) -> dict[str, Any]:
    return validate_start_template_library(
        _load_template_file(path), players_per_side=players_per_side
    )


def sample_start_template(
    library: dict[str, Any], rng: np.random.Generator
) -> dict[str, Any]:
    templates = list(library.get("templates") or [])
    if not templates:
        raise ValueError("start template library contains no templates")
    weights = np.asarray(
        [float(template.get("weight", 1.0) or 0.0) for template in templates],
        dtype=np.float64,
    )
    if np.any(weights < 0.0) or not np.any(weights > 0.0):
        raise ValueError(
            "start template weights must contain at least one positive value"
        )
    probs = weights / float(np.sum(weights))
    idx = int(rng.choice(len(templates), p=probs))
    return dict(templates[idx])


def _mirror_anchor(env: Any, anchor: tuple[int, int]) -> tuple[int, int]:
    col, row = env._axial_to_offset(int(anchor[0]), int(anchor[1]))
    # Strategic left/right symmetry on the rendered court is the sideline axis
    # (R at the top, L at the bottom), so mirroring must flip the offset-row.
    mirrored_row = int(env.court_height - 1 - row)
    return tuple(env._offset_to_axial(int(col), int(mirrored_row)))


def _resolve_entry_position(
    env: Any,
    *,
    anchor: tuple[int, int],
    effective_radius: int,
    taken_positions: set[tuple[int, int]],
    valid_cells: list[tuple[int, int]] | None = None,
) -> tuple[int, int]:
    valid_cells = list(
        valid_cells
        or getattr(env, "_valid_axial", ())
        or getattr(env, "_cell_index", {}).keys()
    )
    if not valid_cells:
        raise ValueError("env does not expose any valid cells for template resolution")
    candidates: list[tuple[tuple[int, int], int]] = []
    for cell in valid_cells:
        if tuple(cell) in taken_positions:
            continue
        if (tuple(cell) == tuple(env.basket_position)) and (
            not bool(getattr(env, "allow_dunks", False))
        ):
            continue
        dist = int(env._hex_distance(anchor, tuple(cell)))
        candidates.append((tuple(cell), dist))
    if not candidates:
        raise ValueError("no available cells remain while resolving start template")

    in_radius = [cell for cell, dist in candidates if dist <= effective_radius]
    if in_radius:
        choice_idx = int(getattr(env, "_rng").integers(0, len(in_radius)))
        return tuple(in_radius[choice_idx])

    min_dist = min(dist for _, dist in candidates)
    nearest = [cell for cell, dist in candidates if dist == min_dist]
    choice_idx = int(getattr(env, "_rng").integers(0, len(nearest)))
    return tuple(nearest[choice_idx])


def _project_anchor_to_valid_cell(
    env: Any,
    anchor: tuple[int, int],
    *,
    valid_cells: list[tuple[int, int]],
) -> tuple[int, int]:
    anchor = (int(anchor[0]), int(anchor[1]))
    if anchor in valid_cells:
        return anchor
    return _resolve_entry_position(
        env,
        anchor=anchor,
        effective_radius=0,
        taken_positions=set(),
        valid_cells=valid_cells,
    )


def _ensure_positions_in_bounds(
    env: Any,
    positions: list[tuple[int, int]],
    *,
    valid_cells: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    fixed_positions: list[tuple[int, int]] = []
    taken_positions: set[tuple[int, int]] = set()
    for pos in positions:
        candidate = (int(pos[0]), int(pos[1]))
        if candidate not in valid_cells or candidate in taken_positions:
            candidate = _resolve_entry_position(
                env,
                anchor=candidate,
                effective_radius=0,
                taken_positions=taken_positions,
                valid_cells=valid_cells,
            )
        fixed_positions.append(candidate)
        taken_positions.add(candidate)
    return fixed_positions


def _team_player_ids(env: Any, team_name: str, players_per_side: int) -> list[int]:
    if team_name == "offense":
        team_ids = getattr(env, "offense_ids", None)
        if team_ids is None:
            team_ids = list(range(players_per_side))
    else:
        team_ids = getattr(env, "defense_ids", None)
        if team_ids is None:
            team_ids = list(range(players_per_side, players_per_side * 2))
    return [int(pid) for pid in team_ids]


def resolve_start_template(
    env: Any,
    template: dict[str, Any],
    *,
    jitter_scale: float = 1.0,
    mirror: bool = False,
) -> dict[str, Any]:
    players_per_side = int(getattr(env, "players_per_side"))
    n_players = int(getattr(env, "n_players"))
    if n_players != players_per_side * 2:
        raise ValueError(
            "start template resolver currently expects offense and defense sides only"
        )

    placements: list[dict[str, Any]] = []
    ball_holder: int | None = None
    rng = getattr(env, "_rng")
    valid_cells = list(
        getattr(env, "_valid_axial", ())
        or getattr(env, "_cell_index", {}).keys()
    )
    if not valid_cells:
        raise ValueError("env does not expose any valid cells for template resolution")
    for team_name in ("offense", "defense"):
        entries = list(template.get(team_name, []) or [])
        team_ids = _team_player_ids(env, team_name, players_per_side)
        if len(team_ids) != len(entries):
            raise ValueError(
                f"template '{template.get('id', '')}' {team_name} entry count does not match team ids"
            )
        assignment_order = [team_ids[int(idx)] for idx in rng.permutation(len(team_ids))]
        for entry, assigned_player_id in zip(entries, assignment_order):
            anchor = _project_anchor_to_valid_cell(
                env,
                (int(entry["anchor"][0]), int(entry["anchor"][1])),
                valid_cells=valid_cells,
            )
            placements.append(
                {
                    "player_id": int(assigned_player_id),
                    "anchor": anchor,
                    "jitter_radius": max(
                        0,
                        int(
                            round(
                                float(entry.get("jitter_radius", 0))
                                * max(0.0, float(jitter_scale))
                            )
                        ),
                    ),
                }
            )
            if bool(entry.get("has_ball", False)):
                ball_holder = int(assigned_player_id)

    # Place the most constrained anchors first.
    placements.sort(key=lambda item: (int(item["jitter_radius"]), int(item["player_id"])))
    taken_positions: set[tuple[int, int]] = set()
    positions: list[tuple[int, int] | None] = [None] * n_players
    for item in placements:
        chosen = _resolve_entry_position(
            env,
            anchor=tuple(item["anchor"]),
            effective_radius=int(item["jitter_radius"]),
            taken_positions=taken_positions,
            valid_cells=valid_cells,
        )
        positions[int(item["player_id"])] = chosen
        taken_positions.add(chosen)

    resolved_positions = [tuple(pos) for pos in positions if pos is not None]
    if len(resolved_positions) != n_players:
        raise ValueError("failed to resolve all player positions for start template")

    effective_mirror = bool(mirror and bool(template.get("mirrorable", False)))
    if effective_mirror:
        resolved_positions = [
            _mirror_anchor(env, tuple(pos)) for pos in resolved_positions
        ]
    resolved_positions = _ensure_positions_in_bounds(
        env,
        resolved_positions,
        valid_cells=valid_cells,
    )

    result: dict[str, Any] = {
        "template_id": str(template.get("id", "")),
        "mirrored": effective_mirror,
        "initial_positions": resolved_positions,
    }
    if ball_holder is not None:
        result["ball_holder"] = int(ball_holder)
    if template.get("shot_clock") is not None:
        result["shot_clock"] = int(template["shot_clock"])
    return result
