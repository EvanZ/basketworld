"""Immutable raw outcomes and possession ledgers for fixed-player studies.

The fixed-pool evaluation runner deliberately keeps fitted models out of its
output.  This module owns the durable raw layer instead: two source manifests,
an episode table, and an append-only possession-event ledger.  The tables are
CSV so they work with spreadsheet and data-frame tooling without a project
specific reader; every cell is JSON encoded so the round trip preserves types
and nested roster/skill metadata exactly.
"""

from __future__ import annotations

import copy
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from basketworld_jax.eval.fixed_pool import PlayerPool
from basketworld_jax.eval.lineup_schedule import LineupSchedule


FIXED_POOL_STUDY_SCHEMA_VERSION = 1
FIXED_POOL_EPISODE_TABLE_SCHEMA_VERSION = 1
FIXED_POOL_EVENT_LEDGER_SCHEMA_VERSION = 1
FIXED_POOL_ARTIFACT_NAMESPACE = "fixed_pool_studies"


class FixedPoolStudyArtifactError(ValueError):
    """A raw fixed-pool study artifact is incomplete or incompatible."""


# These ordered field lists are the schemas.  Fields may only be added in a
# later version; do not infer fields from individual rows and silently drop
# data.  JSON-encoded cells allow CSV to preserve nullable numeric values,
# booleans, lists, and named skill descriptors without a lossy conversion.
EPISODE_TABLE_FIELDS = (
    "study_id",
    "episode_table_schema_version",
    "episode",
    "pool_id",
    "pool_schema_version",
    "skill_schema_version",
    "skill_schema_hash",
    "schedule_id",
    "schedule_schema_version",
    "schedule_seed",
    "schedule_game_id",
    "pair_id",
    "starting_team",
    "reset_rng_root",
    "environment_rng_root",
    "policy_rng_root",
    "action_mode",
    "unified_checkpoint_path",
    "unified_checkpoint_update_index",
    "opponent_checkpoint_path",
    "opponent_checkpoint_update_index",
    "checkpoint_observation_schema_versions",
    "environment_observation_schema_version",
    "team_a_player_ids",
    "team_b_player_ids",
    "team_a_player_skills",
    "team_b_player_skills",
    "completed",
    "truncated",
    "termination_status",
    "analysis_default_included",
    "analysis_exclusion_reason",
    "team_a_score",
    "team_b_score",
    "team_a_minus_team_b_margin",
    "completed_possessions",
    "team_a_completed_possessions",
    "team_b_completed_possessions",
    "team_a_points_per_offensive_possession",
    "team_b_points_per_offensive_possession",
    "team_a_minus_team_b_net_rating",
    "overtime_round",
    "overtime_possessions_completed",
)


EVENT_LEDGER_FIELDS = (
    "study_id",
    "event_ledger_schema_version",
    "episode",
    "schedule_game_id",
    "pair_id",
    "event_id",
    "step_index",
    "event_type",
    "event_subtype",
    "offense_team",
    "actor_team",
    "actor_player_slot",
    "actor_player_id",
    "target_player_slot",
    "target_player_id",
    "completed_possessions_after",
    "team_a_completed_possessions_after",
    "team_b_completed_possessions_after",
    "possession_number",
    "possession_ended",
    "possession_end_reason_code",
    "possession_end_reason",
    "prior_game_phase",
    "game_phase",
    "team_a_score_after",
    "team_b_score_after",
    "shot_success",
    "shot_value",
    "shot_expected_points",
    "shot_distance",
    "shot_type",
    "pass_completed",
    "assist",
    "potential_assist",
    "turnover_reason_code",
    "turnover_reason",
    "rebound_offensive",
    "rebound_defensive",
    "rebound_target_cell",
    "rebound_global_contest",
    "inbound_team",
    "inbound_player_slot",
    "inbound_player_id",
    "inbound_steps_remaining",
    "clearance_elapsed_steps",
    "turnovers_before_clearance",
    "check_pickup_steps",
    "payload",
)


def _json_value(value: Any) -> Any:
    """Return a defensive JSON-compatible copy with NumPy scalars normalized."""
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return copy.deepcopy(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _study_id_payload(
    pool: PlayerPool,
    schedule: LineupSchedule,
    episode_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not episode_rows:
        raise FixedPoolStudyArtifactError(
            "A fixed-pool study needs at least one episode."
        )
    first = episode_rows[0]
    shared_fields = (
        "pool_id",
        "schedule_id",
        "action_mode",
        "unified_checkpoint_path",
        "unified_checkpoint_update_index",
        "opponent_checkpoint_path",
        "opponent_checkpoint_update_index",
        "checkpoint_observation_schema_versions",
        "environment_observation_schema_version",
    )
    metadata = {field: _json_value(first.get(field)) for field in shared_fields}
    for row in episode_rows[1:]:
        for field, expected in metadata.items():
            if _json_value(row.get(field)) != expected:
                raise FixedPoolStudyArtifactError(
                    f"Fixed-pool episode rows disagree on study field {field!r}."
                )
    return {
        "study_schema_version": FIXED_POOL_STUDY_SCHEMA_VERSION,
        "pool_id": pool.pool_id,
        "schedule_id": schedule.schedule_id,
        "metadata": metadata,
    }


def fixed_pool_study_id(
    pool: PlayerPool,
    schedule: LineupSchedule,
    episode_rows: Sequence[Mapping[str, Any]],
) -> str:
    """Return the deterministic identity for one raw fixed-pool study."""
    digest = hashlib.sha256(
        _canonical_json(_study_id_payload(pool, schedule, episode_rows)).encode("utf-8")
    ).hexdigest()
    return f"fixed-pool-study-v{FIXED_POOL_STUDY_SCHEMA_VERSION}-{digest[:20]}"


def _team_label(code: int | None) -> str | None:
    if code == 0:
        return "team_a"
    if code == 1:
        return "team_b"
    return None


def _phase_label(code: int | None) -> str | None:
    return {
        0: "live",
        1: "awaiting_inbound",
        2: "awaiting_check",
        3: "check_setup",
    }.get(code, "unknown" if code is not None else None)


def _possession_end_label(code: int | None) -> str | None:
    return {
        0: "none",
        1: "made_basket",
        2: "defensive_rebound",
        3: "turnover",
        4: "defensive_violation",
        5: "inbound_violation",
        6: "check_violation",
    }.get(code, "unknown" if code is not None else None)


def _turnover_reason_label(code: int | None) -> str | None:
    return {
        0: "none",
        1: "pass_out_of_bounds",
        2: "intercepted",
        3: "defender_pressure",
        4: "move_out_of_bounds",
        5: "shot_clock_violation",
        6: "offensive_three_seconds",
        7: "inbound_timeout",
        8: "inbound_invalid_pass",
        9: "clearance_violation",
        10: "check_timeout",
    }.get(code, "unknown" if code is not None else None)


def _shot_type_label(code: int | None) -> str | None:
    # The kernel uses zero for a two, one for a three, and two for a dunk.
    return {0: "two", 1: "three", 2: "dunk"}.get(
        code, "unknown" if code is not None else None
    )


def _slot_player_map(game_metadata: Mapping[str, Any]) -> dict[int, tuple[str, str]]:
    team_a = tuple(str(value) for value in game_metadata.get("team_a_player_ids", []))
    team_b = tuple(str(value) for value in game_metadata.get("team_b_player_ids", []))
    if len(team_a) != 5 or len(team_b) != 5:
        raise FixedPoolStudyArtifactError(
            "A fixed-pool trace needs exactly five named players on each team."
        )
    return {
        **{slot: ("team_a", player_id) for slot, player_id in enumerate(team_a)},
        **{5 + slot: ("team_b", player_id) for slot, player_id in enumerate(team_b)},
    }


def _trace_value(
    trace: Mapping[str, Any],
    key: str,
    step_index: int,
    env_index: int,
    default: Any = 0,
) -> Any:
    values = trace.get(key)
    if values is None:
        return default
    array = np.asarray(values)
    try:
        value = array[step_index, env_index]
    except (IndexError, TypeError):
        return default
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    value = int(value)
    return value if value >= 0 else None


def _player_fields(
    slot_player_map: Mapping[int, tuple[str, str]], slot: Any
) -> tuple[str | None, int | None, str | None]:
    normalized_slot = _optional_int(slot)
    if normalized_slot is None:
        return None, None, None
    entry = slot_player_map.get(normalized_slot)
    if entry is None:
        return None, normalized_slot, None
    team, player_id = entry
    return team, normalized_slot, player_id


def _new_event_row(
    *,
    episode: int,
    game_metadata: Mapping[str, Any],
    step_index: int,
    event_type: str,
    event_subtype: str | None,
    base: Mapping[str, Any],
    actor: tuple[str | None, int | None, str | None] = (None, None, None),
    target: tuple[str | None, int | None, str | None] = (None, None, None),
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        field: None
        for field in EVENT_LEDGER_FIELDS
        if field
        not in {
            "event_ledger_schema_version",
            "episode",
            "schedule_game_id",
            "pair_id",
            "event_id",
            "step_index",
            "event_type",
            "event_subtype",
            "actor_team",
            "actor_player_slot",
            "actor_player_id",
            "target_player_slot",
            "target_player_id",
            "payload",
        }
    }
    row.update(base)
    row.update(
        {
            "event_ledger_schema_version": FIXED_POOL_EVENT_LEDGER_SCHEMA_VERSION,
            "episode": int(episode),
            "schedule_game_id": str(game_metadata["schedule_game_id"]),
            "pair_id": str(game_metadata["pair_id"]),
            "event_id": f"{game_metadata['schedule_game_id']}:{int(step_index)}:{event_type}",
            "step_index": int(step_index),
            "event_type": str(event_type),
            "event_subtype": event_subtype,
            "actor_team": actor[0],
            "actor_player_slot": actor[1],
            "actor_player_id": actor[2],
            "target_player_slot": target[1],
            "target_player_id": target[2],
            "payload": _json_value(payload or {}),
        }
    )
    return row


def possession_event_rows_from_trace(
    trace: Mapping[str, Any],
    *,
    env_index: int,
    active_steps: int,
    episode: int,
    game_metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Extract durable possession events from one native-evaluation trace row.

    A timestep can yield more than one row (for example, a completed pass can
    also be an assist, and a made shot can end a possession).  The fixed field
    schema is intentionally broad and event-specific details stay in
    ``payload`` so a later schema can append fields without rewriting history.
    """
    slot_player_map = _slot_player_map(game_metadata)
    rows: list[dict[str, Any]] = []
    for step_index in range(max(0, int(active_steps))):
        offense_code = int(
            _trace_value(trace, "offense_team", step_index, env_index, -1)
        )
        offense_team = _team_label(offense_code)
        possession_ended = bool(
            _trace_value(trace, "possession_ended", step_index, env_index, 0)
        )
        completed_after = int(
            _trace_value(trace, "completed_possessions", step_index, env_index, 0)
        )
        team_a_possessions = int(
            _trace_value(
                trace, "team_a_completed_possessions", step_index, env_index, 0
            )
        )
        team_b_possessions = int(
            _trace_value(
                trace, "team_b_completed_possessions", step_index, env_index, 0
            )
        )
        end_reason_code = int(
            _trace_value(trace, "possession_end_reason", step_index, env_index, 0)
        )
        prior_phase_code = int(
            _trace_value(trace, "prior_game_phase", step_index, env_index, -1)
        )
        game_phase_code = int(
            _trace_value(trace, "game_phase", step_index, env_index, -1)
        )
        base = {
            "study_id": None,
            "offense_team": offense_team,
            "completed_possessions_after": completed_after,
            "team_a_completed_possessions_after": team_a_possessions,
            "team_b_completed_possessions_after": team_b_possessions,
            "possession_number": max(
                1, completed_after + (0 if possession_ended else 1)
            ),
            "possession_ended": possession_ended,
            "possession_end_reason_code": end_reason_code,
            "possession_end_reason": _possession_end_label(end_reason_code),
            "prior_game_phase": _phase_label(prior_phase_code),
            "game_phase": _phase_label(game_phase_code),
            "team_a_score_after": float(
                _trace_value(trace, "team_a_score", step_index, env_index, 0.0)
            ),
            "team_b_score_after": float(
                _trace_value(trace, "team_b_score", step_index, env_index, 0.0)
            ),
            "shot_success": None,
            "shot_value": None,
            "shot_expected_points": None,
            "shot_distance": None,
            "shot_type": None,
            "pass_completed": None,
            "assist": None,
            "potential_assist": None,
            "turnover_reason_code": None,
            "turnover_reason": None,
            "rebound_offensive": None,
            "rebound_defensive": None,
            "rebound_target_cell": None,
            "rebound_global_contest": None,
            "inbound_team": None,
            "inbound_player_slot": None,
            "inbound_player_id": None,
            "inbound_steps_remaining": None,
            "clearance_elapsed_steps": None,
            "turnovers_before_clearance": None,
            "check_pickup_steps": None,
        }

        shot_attempt = bool(
            _trace_value(trace, "shot_attempt", step_index, env_index, 0)
        )
        if shot_attempt:
            actor = _player_fields(
                slot_player_map,
                _trace_value(trace, "shot_shooter", step_index, env_index, -1),
            )
            shot_type_code = int(
                _trace_value(trace, "shot_type", step_index, env_index, -1)
            )
            shot_data = {
                "shot_success": bool(
                    _trace_value(trace, "shot_success", step_index, env_index, 0)
                ),
                "shot_value": float(
                    _trace_value(trace, "shot_value", step_index, env_index, 0.0)
                ),
                "shot_expected_points": float(
                    _trace_value(
                        trace, "shot_expected_points", step_index, env_index, 0.0
                    )
                ),
                "shot_distance": float(
                    _trace_value(trace, "shot_distance", step_index, env_index, 0.0)
                ),
                "shot_type": _shot_type_label(shot_type_code),
            }
            rows.append(
                _new_event_row(
                    episode=episode,
                    game_metadata=game_metadata,
                    step_index=step_index,
                    event_type="shot",
                    event_subtype=shot_data["shot_type"],
                    base={**base, **shot_data},
                    actor=actor,
                    payload={"shot_type_code": shot_type_code},
                )
            )

        pass_attempt = bool(
            _trace_value(trace, "pass_attempts", step_index, env_index, 0)
        )
        if pass_attempt:
            actor = _player_fields(
                slot_player_map,
                _trace_value(trace, "pass_passer", step_index, env_index, -1),
            )
            target = _player_fields(
                slot_player_map,
                _trace_value(trace, "pass_receiver", step_index, env_index, -1),
            )
            pass_data = {
                "pass_completed": bool(
                    _trace_value(trace, "completed_passes", step_index, env_index, 0)
                ),
                "assist": bool(
                    _trace_value(trace, "assists", step_index, env_index, 0)
                ),
                "potential_assist": bool(
                    _trace_value(trace, "potential_assist", step_index, env_index, 0)
                ),
            }
            rows.append(
                _new_event_row(
                    episode=episode,
                    game_metadata=game_metadata,
                    step_index=step_index,
                    event_type="pass",
                    event_subtype=(
                        "completed" if pass_data["pass_completed"] else "attempted"
                    ),
                    base={**base, **pass_data},
                    actor=actor,
                    target=target,
                )
            )

        turnover = bool(_trace_value(trace, "turnovers", step_index, env_index, 0))
        if turnover:
            reason_code = int(
                _trace_value(trace, "turnover_reason", step_index, env_index, 0)
            )
            actor = _player_fields(
                slot_player_map,
                _trace_value(trace, "turnover_player", step_index, env_index, -1),
            )
            target = _player_fields(
                slot_player_map,
                _trace_value(trace, "steal_player", step_index, env_index, -1),
            )
            turnover_data = {
                "turnover_reason_code": reason_code,
                "turnover_reason": _turnover_reason_label(reason_code),
            }
            rows.append(
                _new_event_row(
                    episode=episode,
                    game_metadata=game_metadata,
                    step_index=step_index,
                    event_type="turnover",
                    event_subtype=turnover_data["turnover_reason"],
                    base={**base, **turnover_data},
                    actor=actor,
                    target=target,
                )
            )

        rebound_attempt = bool(
            _trace_value(trace, "rebound_attempt", step_index, env_index, 0)
        )
        if rebound_attempt:
            actor = _player_fields(
                slot_player_map,
                _trace_value(trace, "rebound_winner", step_index, env_index, -1),
            )
            rebound_data = {
                "rebound_offensive": bool(
                    _trace_value(trace, "offensive_rebound", step_index, env_index, 0)
                ),
                "rebound_defensive": bool(
                    _trace_value(trace, "defensive_rebound", step_index, env_index, 0)
                ),
                "rebound_target_cell": _optional_int(
                    _trace_value(
                        trace, "rebound_target_cell", step_index, env_index, -1
                    )
                ),
                "rebound_global_contest": bool(
                    _trace_value(
                        trace, "rebound_global_contest", step_index, env_index, 0
                    )
                ),
            }
            rows.append(
                _new_event_row(
                    episode=episode,
                    game_metadata=game_metadata,
                    step_index=step_index,
                    event_type="rebound",
                    event_subtype=(
                        "offensive"
                        if rebound_data["rebound_offensive"]
                        else (
                            "defensive"
                            if rebound_data["rebound_defensive"]
                            else "unresolved"
                        )
                    ),
                    base={**base, **rebound_data},
                    actor=actor,
                )
            )

        clearance_event = bool(
            _trace_value(trace, "clearance_events", step_index, env_index, 0)
        )
        if clearance_event:
            clearance_data = {
                "clearance_elapsed_steps": int(
                    _trace_value(
                        trace, "clearance_elapsed_steps", step_index, env_index, 0
                    )
                ),
                "turnovers_before_clearance": bool(
                    _trace_value(
                        trace, "turnovers_before_clearance", step_index, env_index, 0
                    )
                ),
            }
            rows.append(
                _new_event_row(
                    episode=episode,
                    game_metadata=game_metadata,
                    step_index=step_index,
                    event_type="clearance",
                    event_subtype="achieved",
                    base={**base, **clearance_data},
                )
            )

        inbound_player_slot = _optional_int(
            _trace_value(trace, "inbound_player", step_index, env_index, -1)
        )
        inbound_actor = _player_fields(slot_player_map, inbound_player_slot)
        inbound_team_code = int(
            _trace_value(trace, "inbound_team", step_index, env_index, -1)
        )
        inbound_data = {
            "inbound_team": _team_label(inbound_team_code),
            "inbound_player_slot": inbound_actor[1],
            "inbound_player_id": inbound_actor[2],
            "inbound_steps_remaining": int(
                _trace_value(trace, "inbound_steps_remaining", step_index, env_index, 0)
            ),
        }
        inbound_started = possession_ended and game_phase_code == 1
        inbound_completed = prior_phase_code == 1 and bool(
            _trace_value(trace, "completed_passes", step_index, env_index, 0)
        )
        if inbound_started or inbound_completed:
            rows.append(
                _new_event_row(
                    episode=episode,
                    game_metadata=game_metadata,
                    step_index=step_index,
                    event_type="inbound",
                    event_subtype="started" if inbound_started else "completed",
                    base={**base, **inbound_data},
                    actor=inbound_actor,
                )
            )

        check_values = {
            "check_opportunity": bool(
                _trace_value(trace, "check_opportunities", step_index, env_index, 0)
            ),
            "check_setup_opportunity": bool(
                _trace_value(
                    trace, "check_setup_opportunities", step_index, env_index, 0
                )
            ),
            "check_setup_step": bool(
                _trace_value(trace, "check_setup_steps", step_index, env_index, 0)
            ),
            "check_pickup": bool(
                _trace_value(trace, "check_pickups", step_index, env_index, 0)
            ),
            "check_violation": bool(
                _trace_value(trace, "check_violations", step_index, env_index, 0)
            ),
            "check_pickup_steps": int(
                _trace_value(trace, "check_pickup_steps", step_index, env_index, 0)
            ),
        }
        if any(check_values.values()):
            if check_values["check_violation"]:
                subtype = "violation"
            elif check_values["check_pickup"]:
                subtype = "pickup"
            elif check_values["check_setup_step"]:
                subtype = "setup_step"
            elif check_values["check_setup_opportunity"]:
                subtype = "setup_opportunity"
            else:
                subtype = "opportunity"
            rows.append(
                _new_event_row(
                    episode=episode,
                    game_metadata=game_metadata,
                    step_index=step_index,
                    event_type="check",
                    event_subtype=subtype,
                    base={
                        **base,
                        "check_pickup_steps": check_values["check_pickup_steps"],
                    },
                    payload=check_values,
                )
            )

        defensive_lane = bool(
            _trace_value(trace, "defensive_lane_violation", step_index, env_index, 0)
        )
        offensive_three = bool(
            _trace_value(trace, "offensive_three_seconds", step_index, env_index, 0)
        )
        if defensive_lane or offensive_three:
            actor = _player_fields(
                slot_player_map,
                (
                    _trace_value(
                        trace,
                        "defensive_lane_violation_player",
                        step_index,
                        env_index,
                        -1,
                    )
                    if defensive_lane
                    else -1
                ),
            )
            subtype = "defensive_lane" if defensive_lane else "offensive_three_seconds"
            rows.append(
                _new_event_row(
                    episode=episode,
                    game_metadata=game_metadata,
                    step_index=step_index,
                    event_type="violation",
                    event_subtype=subtype,
                    base=base,
                    actor=actor,
                )
            )

        if possession_ended:
            rows.append(
                _new_event_row(
                    episode=episode,
                    game_metadata=game_metadata,
                    step_index=step_index,
                    event_type="possession_end",
                    event_subtype=_possession_end_label(end_reason_code),
                    base=base,
                )
            )
    return rows


def episode_rows_from_results(
    pool: PlayerPool,
    schedule: LineupSchedule,
    results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten fixed-pool evaluation outcomes into one typed row per game."""
    players = {player.player_id: player for player in pool.players}
    schedule_games = {game.game_id: game for game in schedule.games}
    rows: list[dict[str, Any]] = []
    seen_game_ids: set[str] = set()
    for result in results:
        fixed = result.get("fixed_pool_study")
        game = result.get("game")
        if not isinstance(fixed, Mapping) or not isinstance(game, Mapping):
            raise FixedPoolStudyArtifactError(
                "Fixed-pool raw artifacts require native fixed-pool episode results."
            )
        game_id = str(fixed.get("schedule_game_id", ""))
        scheduled_game = schedule_games.get(game_id)
        if scheduled_game is None or game_id in seen_game_ids:
            raise FixedPoolStudyArtifactError(
                "Episode results must contain each scheduled game exactly once."
            )
        seen_game_ids.add(game_id)
        if (
            fixed.get("pool_id") != pool.pool_id
            or fixed.get("schedule_id") != schedule.schedule_id
        ):
            raise FixedPoolStudyArtifactError(
                "Episode provenance does not match the supplied pool and schedule."
            )
        team_a_ids = [str(value) for value in fixed.get("team_a_player_ids", [])]
        team_b_ids = [str(value) for value in fixed.get("team_b_player_ids", [])]
        if (
            tuple(team_a_ids) != scheduled_game.team_a_player_ids
            or tuple(team_b_ids) != scheduled_game.team_b_player_ids
        ):
            raise FixedPoolStudyArtifactError(
                f"Episode roster does not match scheduled game {game_id!r}."
            )
        team_a_possessions = int(game.get("team_a_completed_possessions", 0))
        team_b_possessions = int(game.get("team_b_completed_possessions", 0))
        team_a_score = float(game.get("team_a_score", 0.0))
        team_b_score = float(game.get("team_b_score", 0.0))
        team_a_ppp = (
            team_a_score / team_a_possessions if team_a_possessions > 0 else None
        )
        team_b_ppp = (
            team_b_score / team_b_possessions if team_b_possessions > 0 else None
        )
        net_rating = (
            100.0 * (team_a_ppp - team_b_ppp)
            if team_a_ppp is not None and team_b_ppp is not None
            else None
        )
        completed = bool(game.get("completed", result.get("completed", False)))
        if team_a_possessions <= 0 or team_b_possessions <= 0:
            termination_status = "zero_possession"
            inclusion_reason = "zero_offensive_possessions"
        elif not completed:
            termination_status = "incomplete"
            inclusion_reason = "incomplete_game"
        else:
            termination_status = "completed"
            inclusion_reason = None
        row = {
            "study_id": None,
            "episode_table_schema_version": FIXED_POOL_EPISODE_TABLE_SCHEMA_VERSION,
            "episode": int(result.get("episode", 0)),
            "pool_id": pool.pool_id,
            "pool_schema_version": int(pool.schema_version),
            "skill_schema_version": int(pool.skill_schema_version),
            "skill_schema_hash": pool.skill_schema_hash,
            "schedule_id": schedule.schedule_id,
            "schedule_schema_version": int(schedule.schema_version),
            "schedule_seed": int(schedule.schedule_seed),
            "schedule_game_id": game_id,
            "pair_id": str(fixed.get("pair_id", "")),
            "starting_team": str(fixed.get("starting_team", "")),
            "reset_rng_root": int(fixed.get("reset_rng_root", 0)),
            "environment_rng_root": int(fixed.get("environment_rng_root", 0)),
            "policy_rng_root": int(fixed.get("policy_rng_root", 0)),
            "action_mode": fixed.get("action_mode"),
            "unified_checkpoint_path": fixed.get("unified_checkpoint_path"),
            "unified_checkpoint_update_index": fixed.get(
                "unified_checkpoint_update_index"
            ),
            "opponent_checkpoint_path": fixed.get("opponent_checkpoint_path"),
            "opponent_checkpoint_update_index": fixed.get(
                "opponent_checkpoint_update_index"
            ),
            "checkpoint_observation_schema_versions": fixed.get(
                "checkpoint_observation_schema_versions"
            ),
            "environment_observation_schema_version": fixed.get(
                "environment_observation_schema_version"
            ),
            "team_a_player_ids": team_a_ids,
            "team_b_player_ids": team_b_ids,
            "team_a_player_skills": [
                {
                    "player_id": player_id,
                    "skills": _json_value(players[player_id].skills),
                }
                for player_id in team_a_ids
            ],
            "team_b_player_skills": [
                {
                    "player_id": player_id,
                    "skills": _json_value(players[player_id].skills),
                }
                for player_id in team_b_ids
            ],
            "completed": completed,
            "truncated": not completed,
            "termination_status": termination_status,
            "analysis_default_included": termination_status == "completed",
            "analysis_exclusion_reason": inclusion_reason,
            "team_a_score": team_a_score,
            "team_b_score": team_b_score,
            "team_a_minus_team_b_margin": team_a_score - team_b_score,
            "completed_possessions": int(game.get("completed_possessions", 0)),
            "team_a_completed_possessions": team_a_possessions,
            "team_b_completed_possessions": team_b_possessions,
            "team_a_points_per_offensive_possession": team_a_ppp,
            "team_b_points_per_offensive_possession": team_b_ppp,
            "team_a_minus_team_b_net_rating": net_rating,
            "overtime_round": int(game.get("overtime_round", 0)),
            "overtime_possessions_completed": int(
                game.get("overtime_possessions_completed", 0)
            ),
        }
        rows.append(row)
    expected_game_ids = set(schedule_games)
    if seen_game_ids != expected_game_ids:
        missing = sorted(expected_game_ids - seen_game_ids)
        extra = sorted(seen_game_ids - expected_game_ids)
        raise FixedPoolStudyArtifactError(
            "Episode results do not cover the immutable schedule "
            f"(missing={missing}, extra={extra})."
        )
    return rows


def _validate_rows(
    rows: Sequence[Mapping[str, Any]], *, fields: Sequence[str], table_name: str
) -> None:
    allowed = set(fields)
    for row in rows:
        unexpected = set(row) - allowed
        if unexpected:
            raise FixedPoolStudyArtifactError(
                f"{table_name} row has fields outside its append-only schema: "
                f"{sorted(unexpected)}."
            )


def _csv_contents(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _canonical_json(row.get(field)) for field in fields})
    return stream.getvalue()


def _read_csv_table(path: Path, fields: Sequence[str]) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != tuple(fields):
            raise FixedPoolStudyArtifactError(
                f"Unexpected schema for {path.name}; expected versioned field order."
            )
        return [{field: json.loads(raw[field]) for field in fields} for raw in reader]


@dataclass(frozen=True)
class FixedPoolStudyArtifactPaths:
    """Locations of a persisted raw study and its optional MLflow mirror."""

    study_id: str
    root: Path
    manifest_path: Path
    pool_manifest_path: Path
    schedule_manifest_path: Path
    episode_table_path: Path
    event_ledger_path: Path
    mlflow_artifact_path: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "study_id": self.study_id,
            "root": str(self.root),
            "manifest": str(self.manifest_path),
            "pool_manifest": str(self.pool_manifest_path),
            "schedule_manifest": str(self.schedule_manifest_path),
            "episodes": str(self.episode_table_path),
            "possession_events": str(self.event_ledger_path),
            "mlflow_artifact_path": self.mlflow_artifact_path,
        }


def persist_fixed_pool_study_artifacts(
    *,
    pool: PlayerPool,
    schedule: LineupSchedule,
    episode_results: Sequence[Mapping[str, Any]],
    possession_events: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    mlflow_client: Any | None = None,
) -> FixedPoolStudyArtifactPaths:
    """Write immutable raw artifacts and, optionally, mirror them to MLflow.

    ``output_dir`` is a parent directory; study files are always placed below
    ``fixed_pool_studies/<study-id>/`` so they cannot mingle with Elo or
    standard-evaluation artifacts.  Repeating the exact same write is allowed;
    attempting to reuse a study identity with changed bytes raises an error.
    """
    episode_rows = episode_rows_from_results(pool, schedule, episode_results)
    _validate_rows(episode_rows, fields=EPISODE_TABLE_FIELDS, table_name="episode")
    study_id = fixed_pool_study_id(pool, schedule, episode_rows)
    normalized_events: list[dict[str, Any]] = []
    expected_game_ids = {game.game_id for game in schedule.games}
    for raw_event in possession_events:
        event = _json_value(raw_event)
        if str(event.get("schedule_game_id", "")) not in expected_game_ids:
            raise FixedPoolStudyArtifactError(
                "Possession ledger contains a game outside the study schedule."
            )
        event["study_id"] = study_id
        normalized_events.append(event)
    _validate_rows(
        normalized_events, fields=EVENT_LEDGER_FIELDS, table_name="possession ledger"
    )
    for row in episode_rows:
        row["study_id"] = study_id
    study_payload = _study_id_payload(pool, schedule, episode_rows)
    manifest = {
        "schema_version": FIXED_POOL_STUDY_SCHEMA_VERSION,
        "study_id": study_id,
        "artifact_namespace": FIXED_POOL_ARTIFACT_NAMESPACE,
        "pool_id": pool.pool_id,
        "pool_schema_version": int(pool.schema_version),
        "skill_schema_version": int(pool.skill_schema_version),
        "skill_schema_hash": pool.skill_schema_hash,
        "schedule_id": schedule.schedule_id,
        "schedule_schema_version": int(schedule.schema_version),
        "study_identity": study_payload,
        "tables": {
            "episodes": {
                "path": "episodes.csv",
                "format": "csv-json-cells",
                "schema_version": FIXED_POOL_EPISODE_TABLE_SCHEMA_VERSION,
                "fields": list(EPISODE_TABLE_FIELDS),
                "row_count": len(episode_rows),
            },
            "possession_events": {
                "path": "possession_events.csv",
                "format": "csv-json-cells",
                "schema_version": FIXED_POOL_EVENT_LEDGER_SCHEMA_VERSION,
                "fields": list(EVENT_LEDGER_FIELDS),
                "row_count": len(normalized_events),
                "append_only": True,
            },
        },
        "analysis_rules": {
            "raw_rows_dropped_on_write": False,
            "default_episode_inclusion": (
                "completed games with nonzero offensive possessions for both teams"
            ),
            "zero_or_incomplete_games": (
                "persisted explicitly; downstream fitting must report exclusions"
            ),
            "net_rating": (
                "100 * (team_a_points / team_a_offensive_possessions - "
                "team_b_points / team_b_offensive_possessions)"
            ),
        },
    }
    files = {
        "manifest.json": _canonical_json(manifest) + "\n",
        "pool_manifest.json": _canonical_json(pool.to_payload()) + "\n",
        "schedule_manifest.json": _canonical_json(schedule.to_payload()) + "\n",
        "episodes.csv": _csv_contents(episode_rows, EPISODE_TABLE_FIELDS),
        "possession_events.csv": _csv_contents(normalized_events, EVENT_LEDGER_FIELDS),
    }
    namespace_root = Path(output_dir) / FIXED_POOL_ARTIFACT_NAMESPACE
    target = namespace_root / study_id
    if target.exists():
        if not target.is_dir():
            raise FixedPoolStudyArtifactError(
                f"Fixed-pool study target is not a directory: {target}"
            )
        for filename, expected in files.items():
            existing = target / filename
            if (
                not existing.is_file()
                or existing.read_text(encoding="utf-8") != expected
            ):
                raise FixedPoolStudyArtifactError(
                    "Refusing to overwrite immutable fixed-pool artifact " f"{target}."
                )
    else:
        namespace_root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{study_id}-", dir=namespace_root))
        try:
            for filename, content in files.items():
                (staging / filename).write_text(content, encoding="utf-8")
            staging.replace(target)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    mlflow_artifact_path = f"{FIXED_POOL_ARTIFACT_NAMESPACE}/{study_id}"
    if mlflow_client is not None:
        log_artifacts = getattr(mlflow_client, "log_artifacts", None)
        if not callable(log_artifacts):
            raise FixedPoolStudyArtifactError(
                "MLflow client must provide log_artifacts(local_dir, artifact_path=...)."
            )
        log_artifacts(str(target), artifact_path=mlflow_artifact_path)
    return FixedPoolStudyArtifactPaths(
        study_id=study_id,
        root=target,
        manifest_path=target / "manifest.json",
        pool_manifest_path=target / "pool_manifest.json",
        schedule_manifest_path=target / "schedule_manifest.json",
        episode_table_path=target / "episodes.csv",
        event_ledger_path=target / "possession_events.csv",
        mlflow_artifact_path=(
            mlflow_artifact_path if mlflow_client is not None else None
        ),
    )


def _persist_fixed_pool_study_rows(
    *,
    pool: PlayerPool,
    schedule: LineupSchedule,
    episode_rows: Sequence[Mapping[str, Any]],
    possession_events: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    source_studies: Sequence[Mapping[str, Any]] | None = None,
) -> FixedPoolStudyArtifactPaths:
    """Persist already-materialized rows for a derived immutable study."""
    schedule_games = {game.game_id: game for game in schedule.games}
    normalized_rows: list[dict[str, Any]] = []
    seen_game_ids: set[str] = set()
    for raw_row in episode_rows:
        row = _json_value(raw_row)
        game_id = str(row.get("schedule_game_id", ""))
        game = schedule_games.get(game_id)
        if game is None or game_id in seen_game_ids:
            raise FixedPoolStudyArtifactError(
                "Derived study rows must cover every combined schedule game once."
            )
        if (
            row.get("pool_id") != pool.pool_id
            or tuple(row.get("team_a_player_ids") or ()) != game.team_a_player_ids
            or tuple(row.get("team_b_player_ids") or ()) != game.team_b_player_ids
            or row.get("pair_id") != game.pair_id
            or row.get("starting_team") != game.starting_team
        ):
            raise FixedPoolStudyArtifactError(
                f"Derived row provenance does not match scheduled game {game_id!r}."
            )
        seen_game_ids.add(game_id)
        row["episode"] = len(normalized_rows)
        row["schedule_id"] = schedule.schedule_id
        row["schedule_schema_version"] = int(schedule.schema_version)
        row["schedule_seed"] = int(schedule.schedule_seed)
        row["study_id"] = None
        normalized_rows.append(row)
    if seen_game_ids != set(schedule_games):
        raise FixedPoolStudyArtifactError(
            "Derived study rows do not cover the complete combined schedule."
        )
    _validate_rows(
        normalized_rows, fields=EPISODE_TABLE_FIELDS, table_name="derived episode"
    )

    episode_index = {
        str(row["schedule_game_id"]): int(row["episode"])
        for row in normalized_rows
    }
    normalized_events: list[dict[str, Any]] = []
    for raw_event in possession_events:
        event = _json_value(raw_event)
        game_id = str(event.get("schedule_game_id", ""))
        if game_id not in episode_index:
            raise FixedPoolStudyArtifactError(
                "Derived possession event is outside the combined schedule."
            )
        event["episode"] = episode_index[game_id]
        event["study_id"] = None
        normalized_events.append(event)
    _validate_rows(
        normalized_events,
        fields=EVENT_LEDGER_FIELDS,
        table_name="derived possession ledger",
    )

    study_id = fixed_pool_study_id(pool, schedule, normalized_rows)
    for row in normalized_rows:
        row["study_id"] = study_id
    for event in normalized_events:
        event["study_id"] = study_id
    study_payload = _study_id_payload(pool, schedule, normalized_rows)
    manifest: dict[str, Any] = {
        "schema_version": FIXED_POOL_STUDY_SCHEMA_VERSION,
        "study_id": study_id,
        "artifact_namespace": FIXED_POOL_ARTIFACT_NAMESPACE,
        "pool_id": pool.pool_id,
        "pool_schema_version": int(pool.schema_version),
        "skill_schema_version": int(pool.skill_schema_version),
        "skill_schema_hash": pool.skill_schema_hash,
        "schedule_id": schedule.schedule_id,
        "schedule_schema_version": int(schedule.schema_version),
        "study_identity": study_payload,
        "tables": {
            "episodes": {
                "path": "episodes.csv",
                "format": "csv-json-cells",
                "schema_version": FIXED_POOL_EPISODE_TABLE_SCHEMA_VERSION,
                "fields": list(EPISODE_TABLE_FIELDS),
                "row_count": len(normalized_rows),
            },
            "possession_events": {
                "path": "possession_events.csv",
                "format": "csv-json-cells",
                "schema_version": FIXED_POOL_EVENT_LEDGER_SCHEMA_VERSION,
                "fields": list(EVENT_LEDGER_FIELDS),
                "row_count": len(normalized_events),
                "append_only": True,
            },
        },
        "analysis_rules": {
            "raw_rows_dropped_on_write": False,
            "default_episode_inclusion": (
                "completed games with nonzero offensive possessions for both teams"
            ),
            "zero_or_incomplete_games": (
                "persisted explicitly; downstream fitting must report exclusions"
            ),
            "net_rating": (
                "100 * (team_a_points / team_a_offensive_possessions - "
                "team_b_points / team_b_offensive_possessions)"
            ),
        },
    }
    if source_studies:
        manifest["source_studies"] = [_json_value(value) for value in source_studies]
    files = {
        "manifest.json": _canonical_json(manifest) + "\n",
        "pool_manifest.json": _canonical_json(pool.to_payload()) + "\n",
        "schedule_manifest.json": _canonical_json(schedule.to_payload()) + "\n",
        "episodes.csv": _csv_contents(normalized_rows, EPISODE_TABLE_FIELDS),
        "possession_events.csv": _csv_contents(normalized_events, EVENT_LEDGER_FIELDS),
    }
    namespace_root = Path(output_dir) / FIXED_POOL_ARTIFACT_NAMESPACE
    target = namespace_root / study_id
    if target.exists():
        if not target.is_dir():
            raise FixedPoolStudyArtifactError(
                f"Fixed-pool study target is not a directory: {target}"
            )
        for filename, expected in files.items():
            existing = target / filename
            if not existing.is_file() or existing.read_text(encoding="utf-8") != expected:
                raise FixedPoolStudyArtifactError(
                    "Refusing to overwrite immutable fixed-pool artifact " f"{target}."
                )
    else:
        namespace_root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{study_id}-", dir=namespace_root))
        try:
            for filename, content in files.items():
                (staging / filename).write_text(content, encoding="utf-8")
            staging.replace(target)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return FixedPoolStudyArtifactPaths(
        study_id=study_id,
        root=target,
        manifest_path=target / "manifest.json",
        pool_manifest_path=target / "pool_manifest.json",
        schedule_manifest_path=target / "schedule_manifest.json",
        episode_table_path=target / "episodes.csv",
        event_ledger_path=target / "possession_events.csv",
    )


def combine_fixed_pool_study_artifacts(
    *,
    pool: PlayerPool,
    combined_schedule: LineupSchedule,
    base_study_root: str | Path,
    additional_study_root: str | Path,
    output_dir: str | Path,
) -> FixedPoolStudyArtifactPaths:
    """Join a prior study and a newly evaluated extension without replaying it."""
    base = load_fixed_pool_study_artifacts(base_study_root)
    additional = load_fixed_pool_study_artifacts(additional_study_root)
    base_games = {str(row["schedule_game_id"]) for row in base["episodes"]}
    additional_games = {
        str(row["schedule_game_id"]) for row in additional["episodes"]
    }
    combined_games = {game.game_id for game in combined_schedule.games}
    if base_games & additional_games or base_games | additional_games != combined_games:
        raise FixedPoolStudyArtifactError(
            "Extension source games do not exactly partition the combined schedule."
        )
    metadata_keys = (
        "pool_id",
        "action_mode",
        "unified_checkpoint_path",
        "unified_checkpoint_update_index",
        "opponent_checkpoint_path",
        "opponent_checkpoint_update_index",
        "checkpoint_observation_schema_versions",
        "environment_observation_schema_version",
    )
    base_metadata = (base["manifest"].get("study_identity") or {}).get("metadata") or {}
    additional_metadata = (
        (additional["manifest"].get("study_identity") or {}).get("metadata") or {}
    )
    if any(
        _json_value(base_metadata.get(key)) != _json_value(additional_metadata.get(key))
        for key in metadata_keys
    ):
        raise FixedPoolStudyArtifactError(
            "Extension evidence must use the same pool, policy, and action mode."
        )
    if str(base_metadata.get("pool_id")) != pool.pool_id:
        raise FixedPoolStudyArtifactError(
            "Extension base study does not match the requested player pool."
        )
    source_rows = [*base["episodes"], *additional["episodes"]]
    source_events = [*base["possession_events"], *additional["possession_events"]]
    return _persist_fixed_pool_study_rows(
        pool=pool,
        schedule=combined_schedule,
        episode_rows=source_rows,
        possession_events=source_events,
        output_dir=output_dir,
        source_studies=(
            {"role": "base", "study_id": base["manifest"].get("study_id"), "root": str(base_study_root)},
            {"role": "additional", "study_id": additional["manifest"].get("study_id"), "root": str(additional_study_root)},
        ),
    )


def load_fixed_pool_study_artifacts(
    root: str | Path,
) -> dict[str, Any]:
    """Load and validate a raw fixed-pool study written by this module."""
    root = Path(root)
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        pool_manifest = json.loads(
            (root / "pool_manifest.json").read_text(encoding="utf-8")
        )
        schedule_manifest = json.loads(
            (root / "schedule_manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise FixedPoolStudyArtifactError(
            f"Cannot read fixed-pool study artifact at {root}."
        ) from exc
    if int(manifest.get("schema_version", -1)) != FIXED_POOL_STUDY_SCHEMA_VERSION:
        raise FixedPoolStudyArtifactError(
            "Unsupported fixed-pool study schema version."
        )
    episodes = _read_csv_table(root / "episodes.csv", EPISODE_TABLE_FIELDS)
    events = _read_csv_table(root / "possession_events.csv", EVENT_LEDGER_FIELDS)
    study_id = manifest.get("study_id")
    if any(row.get("study_id") != study_id for row in episodes + events):
        raise FixedPoolStudyArtifactError(
            "Artifact tables do not agree with manifest study ID."
        )
    return {
        "manifest": manifest,
        "pool_manifest": pool_manifest,
        "schedule_manifest": schedule_manifest,
        "episodes": episodes,
        "possession_events": events,
    }
