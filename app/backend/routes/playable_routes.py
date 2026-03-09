import copy
import json
import os
import random
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response

from basketworld.envs.basketworld_env_v2 import Team
from app.backend.playable_analytics import playable_analytics_emitter
from app.backend.playable_session_store import (
    PLAYABLE_SESSION_HEADER,
    PlayableCapacityError,
    bind_game_state,
    playable_session_store,
)
from app.backend.routes.lifecycle_routes import init_game as init_game_route
from app.backend.routes.lifecycle_routes import step as lifecycle_step
from app.backend.schemas import (
    ActionRequest,
    InitGameRequest,
    PlayableStartRequest,
    PlayableStepRequest,
)
from app.backend.state import _capture_turn_start_snapshot, game_state, get_ui_game_state


router = APIRouter()

PLAYABLE_DIFFICULTIES = ("easy", "medium", "hard")
PLAYABLE_PLAYERS = tuple(range(1, 6))
PLAYABLE_PERIOD_MODES = ("period", "halves", "quarters")
PLAYABLE_PERIOD_MODE_TO_COUNT = {
    "period": 1,
    "halves": 2,
    "quarters": 4,
}
PLAYABLE_MIN_PERIOD_MINUTES = 1
PLAYABLE_MAX_PERIOD_MINUTES = 60
PLAYABLE_CLOCK_TICK_SECONDS = 1
PLAYABLE_DEMO_ENABLED_ENV = "BW_PLAYABLE_DEMO_ENABLED"
PLAYABLE_DEMO_RUN_ID_ENV = "BW_PLAYABLE_DEMO_RUN_ID"
PLAYABLE_DEMO_ALTERNATION_ENV = "BW_PLAYABLE_DEMO_ALTERNATION_NUMBER"
PLAYABLE_DEMO_CHECKPOINT_ENV = "BW_PLAYABLE_DEMO_CHECKPOINT"
PLAYABLE_DEMO_PERIOD_MODE_ENV = "BW_PLAYABLE_DEMO_PERIOD_MODE"
PLAYABLE_DEMO_PERIOD_LENGTH_ENV = "BW_PLAYABLE_DEMO_PERIOD_LENGTH_MINUTES"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _attach_session_id(
    payload: dict[str, Any],
    session_id: str,
    response: Response | None = None,
) -> dict[str, Any]:
    payload["session_id"] = session_id
    if response is not None:
        response.headers[PLAYABLE_SESSION_HEADER] = session_id
    return payload


def _resolve_session_state(
    request: Request,
    *,
    require_active: bool,
) -> tuple[str, Any]:
    raw_session_id = request.headers.get(PLAYABLE_SESSION_HEADER)
    if not raw_session_id:
        raise HTTPException(
            status_code=400,
            detail=f"Missing {PLAYABLE_SESSION_HEADER} header. Start a new game first.",
        )

    session_id, target_state = playable_session_store.get(raw_session_id)
    if not session_id:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {PLAYABLE_SESSION_HEADER} header. Start a new game.",
        )
    if target_state is None:
        raise HTTPException(status_code=404, detail="Playable session not found. Start a new game.")

    if require_active:
        session = getattr(target_state, "playable_session", None)
        if not (isinstance(session, dict) and session.get("active")):
            raise HTTPException(status_code=400, detail="Playable session is not active.")

    return session_id, target_state


def _ensure_playable_analytics_fields(session: dict[str, Any]) -> None:
    analytics = session.get("analytics")
    if not isinstance(analytics, dict):
        analytics = {}
    game_id = str(analytics.get("game_id") or "").strip() or uuid.uuid4().hex
    next_seq = _to_int(analytics.get("next_seq"))
    if next_seq is None or int(next_seq) < 1:
        next_seq = 1
    analytics["game_id"] = game_id
    analytics["next_seq"] = int(next_seq)
    session["analytics"] = analytics

    turn_index = _to_int(session.get("turn_index"))
    if turn_index is None or int(turn_index) < 1:
        session["turn_index"] = 1


def _next_playable_analytics_seq(session: dict[str, Any]) -> int:
    _ensure_playable_analytics_fields(session)
    analytics = session["analytics"]
    seq = int(analytics.get("next_seq", 1) or 1)
    analytics["next_seq"] = seq + 1
    return seq


def _emit_playable_event(
    *,
    session_id: str,
    session: dict[str, Any],
    event_type: str,
    payload: dict[str, Any],
) -> None:
    try:
        seq = _next_playable_analytics_seq(session)
        env = getattr(game_state, "env", None)
        pass_mode = str(getattr(env, "pass_mode", "directional")) if env is not None else "unknown"
        configured_environment = str(os.getenv("BW_ANALYTICS_ENVIRONMENT") or "").strip().lower()
        environment = configured_environment or ("public" if _is_truthy(os.getenv("BW_PUBLIC_MODE")) else "dev")
        game_config = {
            "players_per_side": int(session.get("players_per_side", 0) or 0),
            "difficulty": str(session.get("difficulty", "")).lower(),
            "period_mode": str(session.get("period_mode", "period")).lower(),
            "period_length_minutes": int(session.get("period_length_minutes", 5) or 5),
            "policy_run_id": str(session.get("run_id", "")),
            "policy_checkpoint_index": int(session.get("checkpoint_index", 0) or 0),
            "session_kind": "demo" if bool(session.get("demo_mode", False)) else "human",
            "pass_mode": pass_mode,
        }
        event = {
            "schema_version": "bw.analytics.v1",
            "event_type": str(event_type),
            "event_id": uuid.uuid4().hex,
            "event_ts": _utc_now_iso(),
            "session_id": str(session_id),
            "game_id": str(session["analytics"]["game_id"]),
            "seq": int(seq),
            "app_mode": "playable",
            "environment": environment,
            "game_config": game_config,
            "payload": payload if isinstance(payload, dict) else {},
        }
        playable_analytics_emitter.emit(event)
    except Exception:
        # Analytics should never interrupt gameplay.
        return


def _normalize_period_mode(raw: Any) -> str:
    mode = str(raw or "period").strip().lower()
    if mode not in PLAYABLE_PERIOD_MODE_TO_COUNT:
        raise HTTPException(
            status_code=400,
            detail="period_mode must be one of: period, halves, quarters.",
        )
    return mode


def _normalize_period_length_minutes(raw: Any) -> int:
    minutes = _to_int(raw)
    if minutes is None:
        raise HTTPException(status_code=400, detail="period_length_minutes must be an integer.")
    minutes = int(minutes)
    if minutes < PLAYABLE_MIN_PERIOD_MINUTES or minutes > PLAYABLE_MAX_PERIOD_MINUTES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"period_length_minutes must be between "
                f"{PLAYABLE_MIN_PERIOD_MINUTES} and {PLAYABLE_MAX_PERIOD_MINUTES}."
            ),
        )
    return minutes


def _build_playable_config_info(session: dict[str, Any]) -> dict[str, Any]:
    mode = _normalize_period_mode(session.get("period_mode", "period"))
    total_periods = int(
        session.get("total_periods", PLAYABLE_PERIOD_MODE_TO_COUNT[mode])
        or PLAYABLE_PERIOD_MODE_TO_COUNT[mode]
    )
    return {
        "players_per_side": int(session["players_per_side"]),
        "difficulty": str(session.get("difficulty", "")).lower(),
        "period_mode": mode,
        "total_periods": total_periods,
        "period_length_minutes": _normalize_period_length_minutes(session.get("period_length_minutes", 5)),
        "demo_mode": bool(session.get("demo_mode", False)),
        "session_kind": "demo" if bool(session.get("demo_mode", False)) else "human",
    }


def _build_playable_demo_config() -> dict[str, Any]:
    run_id = str(os.getenv(PLAYABLE_DEMO_RUN_ID_ENV) or "").strip()
    checkpoint = _to_int(os.getenv(PLAYABLE_DEMO_ALTERNATION_ENV))
    if checkpoint is None:
        checkpoint = _to_int(os.getenv(PLAYABLE_DEMO_CHECKPOINT_ENV))

    raw_enabled = str(os.getenv(PLAYABLE_DEMO_ENABLED_ENV) or "").strip()
    if not raw_enabled:
        enabled = bool(run_id) and checkpoint is not None
    else:
        enabled = _is_truthy(raw_enabled)

    raw_period_mode = str(os.getenv(PLAYABLE_DEMO_PERIOD_MODE_ENV) or "period").strip().lower()
    period_mode = raw_period_mode if raw_period_mode in PLAYABLE_PERIOD_MODE_TO_COUNT else "period"

    raw_period_length = _to_int(os.getenv(PLAYABLE_DEMO_PERIOD_LENGTH_ENV))
    if raw_period_length is None:
        period_length_minutes = 5
    else:
        period_length_minutes = max(
            PLAYABLE_MIN_PERIOD_MINUTES,
            min(PLAYABLE_MAX_PERIOD_MINUTES, int(raw_period_length)),
        )

    reason = ""
    if enabled and not run_id:
        reason = f"Missing {PLAYABLE_DEMO_RUN_ID_ENV}"
    elif enabled and checkpoint is None:
        reason = (
            f"Missing {PLAYABLE_DEMO_ALTERNATION_ENV}"
            f" (or {PLAYABLE_DEMO_CHECKPOINT_ENV})"
        )

    return {
        "enabled": bool(enabled),
        "available": bool(enabled and run_id and checkpoint is not None),
        "run_id": run_id,
        "checkpoint_index": int(checkpoint) if checkpoint is not None else None,
        "policy_name": f"unified_iter_{int(checkpoint)}.zip" if checkpoint is not None else "",
        "period_mode": period_mode,
        "total_periods": int(PLAYABLE_PERIOD_MODE_TO_COUNT[period_mode]),
        "period_length_minutes": int(period_length_minutes),
        "reason": reason,
    }


def _sanitize_demo_config_for_response(config: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "enabled": bool(config.get("enabled", False)),
        "available": bool(config.get("available", False)),
        "period_mode": str(config.get("period_mode", "period")),
        "total_periods": int(config.get("total_periods", 1) or 1),
        "period_length_minutes": int(config.get("period_length_minutes", 5) or 5),
        "session_kind": "demo",
    }
    reason = str(config.get("reason") or "").strip()
    if reason:
        payload["reason"] = reason
    return payload


def _build_new_playable_session(
    *,
    players_per_side: int,
    difficulty: str,
    run_id: str,
    checkpoint_index: int,
    policy_name: str,
    period_mode: str,
    total_periods: int,
    period_length_minutes: int,
    demo_mode: bool,
) -> dict[str, Any]:
    return {
        "active": True,
        "players_per_side": int(players_per_side),
        "difficulty": str(difficulty).lower(),
        "run_id": str(run_id),
        "checkpoint_index": int(checkpoint_index),
        "policy_name": str(policy_name),
        "score": {"user": 0, "ai": 0},
        "possession_number": 1,
        "user_on_offense": bool(random.getrandbits(1)),
        "side_skills": _build_playable_side_skills(),
        "period_mode": period_mode,
        "total_periods": int(total_periods),
        "current_period": 1,
        "period_length_minutes": int(period_length_minutes),
        "period_seconds_remaining": int(period_length_minutes) * 60,
        "game_over": False,
        "turn_index": 1,
        "demo_mode": bool(demo_mode),
        "analytics": {
            "game_id": uuid.uuid4().hex,
            "next_seq": 1,
        },
    }


def _segment_label_for_mode(mode: str, period_index: int) -> str:
    idx = max(1, int(period_index))
    if mode == "halves":
        return f"Half {idx}"
    if mode == "quarters":
        return f"Quarter {idx}"
    return f"Period {idx}"


def _format_clock_display(seconds_remaining: int) -> str:
    seconds = max(0, int(seconds_remaining))
    minutes = seconds // 60
    secs = seconds % 60
    return f"{minutes:02d}:{secs:02d}"


def _ensure_playable_game_fields(session: dict[str, Any]) -> None:
    _ensure_playable_analytics_fields(session)

    mode = _normalize_period_mode(session.get("period_mode", "period"))
    session["period_mode"] = mode

    total_default = int(PLAYABLE_PERIOD_MODE_TO_COUNT.get(mode, 1))
    total_periods = _to_int(session.get("total_periods"))
    if total_periods not in (1, 2, 4):
        total_periods = total_default
    session["total_periods"] = int(total_periods)

    current_period = _to_int(session.get("current_period"))
    if current_period is None:
        current_period = 1
    current_period = max(1, min(int(total_periods), int(current_period)))
    session["current_period"] = int(current_period)

    period_length_minutes = _normalize_period_length_minutes(session.get("period_length_minutes", 5))
    session["period_length_minutes"] = int(period_length_minutes)

    max_seconds = int(period_length_minutes) * 60
    seconds_remaining = _to_int(session.get("period_seconds_remaining"))
    if seconds_remaining is None:
        seconds_remaining = max_seconds
    seconds_remaining = max(0, min(max_seconds, int(seconds_remaining)))
    session["period_seconds_remaining"] = int(seconds_remaining)
    session["game_over"] = bool(session.get("game_over", False))


def _build_game_clock_info(session: dict[str, Any]) -> dict[str, Any]:
    _ensure_playable_game_fields(session)
    mode = str(session.get("period_mode"))
    total_periods = int(session.get("total_periods", 1) or 1)
    current_period = int(session.get("current_period", 1) or 1)
    period_length_minutes = int(session.get("period_length_minutes", 5) or 5)
    seconds_remaining = int(session.get("period_seconds_remaining", period_length_minutes * 60) or 0)
    return {
        "period_mode": mode,
        "period_mode_label": {
            "period": "1 period",
            "halves": "2 halves",
            "quarters": "4 quarters",
        }[mode],
        "total_periods": max(1, total_periods),
        "current_period": max(1, current_period),
        "period_length_minutes": max(1, period_length_minutes),
        "seconds_remaining": max(0, seconds_remaining),
        "display": _format_clock_display(seconds_remaining),
        "segment_label": _segment_label_for_mode(mode, current_period),
    }


def _resolve_game_result(score: dict[str, Any]) -> dict[str, Any]:
    user_score = int((score or {}).get("user", 0) or 0)
    ai_score = int((score or {}).get("ai", 0) or 0)
    if user_score > ai_score:
        winner = "user"
        message = f"Game over. You win {user_score}-{ai_score}."
    elif ai_score > user_score:
        winner = "ai"
        message = f"Game over. AI wins {ai_score}-{user_score}."
    else:
        winner = "tie"
        message = f"Game over. Tie game {user_score}-{ai_score}."
    return {
        "game_over": True,
        "winner": winner,
        "message": message,
        "score": {
            "user": user_score,
            "ai": ai_score,
        },
    }


def _build_game_result(session: dict[str, Any]) -> dict[str, Any]:
    if not bool(session.get("game_over", False)):
        return {
            "game_over": False,
            "winner": None,
            "message": "",
            "score": _build_score_info(session),
        }
    score = session.get("score") if isinstance(session.get("score"), dict) else {"user": 0, "ai": 0}
    return _resolve_game_result(score)


def _tick_game_clock(session: dict[str, Any]) -> int:
    remaining = int(session.get("period_seconds_remaining", 0) or 0)
    remaining = max(0, remaining - PLAYABLE_CLOCK_TICK_SECONDS)
    session["period_seconds_remaining"] = remaining
    return remaining


def _clone_skill_set(raw: Any) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {
        "layup": [],
        "three_pt": [],
        "dunk": [],
    }
    if not isinstance(raw, dict):
        return out
    for key in ("layup", "three_pt", "dunk"):
        values = raw.get(key)
        if not isinstance(values, list):
            continue
        parsed: list[float] = []
        for value in values:
            try:
                parsed.append(float(value))
            except Exception:
                continue
        out[key] = parsed
    return out


def _read_env_offense_skills() -> dict[str, list[float]]:
    env = getattr(game_state, "env", None)
    if env is None:
        return {"layup": [], "three_pt": [], "dunk": []}
    return {
        "layup": [float(x) for x in getattr(env, "offense_layup_pct_by_player", [])],
        "three_pt": [float(x) for x in getattr(env, "offense_three_pt_pct_by_player", [])],
        "dunk": [float(x) for x in getattr(env, "offense_dunk_pct_by_player", [])],
    }


def _sample_playable_skill_set_once() -> dict[str, list[float]]:
    env = getattr(game_state, "env", None)
    if env is None:
        return {"layup": [], "three_pt": [], "dunk": []}
    env.reset(options={"shot_clock": 24})
    return _read_env_offense_skills()


def _build_playable_side_skills() -> dict[str, dict[str, list[float]]]:
    return {
        "user": _sample_playable_skill_set_once(),
        "ai": _sample_playable_skill_set_once(),
    }


def _is_truthy(raw: str | None) -> bool:
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on"}


def _to_int(raw: Any) -> int | None:
    try:
        return int(raw)
    except Exception:
        return None


def _get_playable_matrix_from_json() -> dict[int, dict[str, dict[str, Any]]]:
    raw = os.getenv("BW_PLAYABLE_POLICY_MATRIX_JSON")
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}

    matrix: dict[int, dict[str, dict[str, Any]]] = {}
    if not isinstance(parsed, dict):
        return matrix

    for players_key, diff_map in parsed.items():
        players = _to_int(players_key)
        if players is None or players not in PLAYABLE_PLAYERS:
            continue
        if not isinstance(diff_map, dict):
            continue
        matrix[players] = {}
        for diff, cfg in diff_map.items():
            diff_key = str(diff or "").strip().lower()
            if diff_key not in PLAYABLE_DIFFICULTIES:
                continue
            if not isinstance(cfg, dict):
                continue
            run_id = str(cfg.get("run_id") or "").strip()
            checkpoint = _to_int(cfg.get("checkpoint_index"))
            if not run_id or checkpoint is None:
                continue
            matrix[players][diff_key] = {
                "run_id": run_id,
                "checkpoint_index": int(checkpoint),
                "policy_name": f"unified_iter_{int(checkpoint)}.zip",
                "available": True,
            }
    return matrix


def _load_playable_matrix() -> dict[int, dict[str, dict[str, Any]]]:
    matrix: dict[int, dict[str, dict[str, Any]]] = {
        players: {diff: {"available": False} for diff in PLAYABLE_DIFFICULTIES}
        for players in PLAYABLE_PLAYERS
    }

    json_matrix = _get_playable_matrix_from_json()
    for players, diff_map in json_matrix.items():
        for diff, cfg in diff_map.items():
            matrix[players][diff] = cfg

    # Explicit env vars override JSON entries.
    for players in PLAYABLE_PLAYERS:
        for diff in PLAYABLE_DIFFICULTIES:
            prefix = f"BW_PLAYABLE_{players}_{diff.upper()}"
            run_id = str(os.getenv(f"{prefix}_RUN_ID") or "").strip()
            checkpoint = _to_int(os.getenv(f"{prefix}_CHECKPOINT"))
            if run_id and checkpoint is not None:
                matrix[players][diff] = {
                    "run_id": run_id,
                    "checkpoint_index": int(checkpoint),
                    "policy_name": f"unified_iter_{int(checkpoint)}.zip",
                    "available": True,
                }
            elif run_id or checkpoint is not None:
                matrix[players][diff] = {
                    "available": False,
                    "reason": "Missing run_id/checkpoint",
                }

    return matrix


def _get_playable_session(required: bool = False) -> dict[str, Any] | None:
    session = getattr(game_state, "playable_session", None)
    if isinstance(session, dict) and session.get("active"):
        return session
    if required:
        raise HTTPException(status_code=400, detail="Playable session is not active.")
    return None


def _build_id_maps(players_per_side: int, user_on_offense: bool) -> tuple[dict[int, int], dict[int, int]]:
    total = int(players_per_side) * 2
    if user_on_offense:
        env_to_canonical = {pid: pid for pid in range(total)}
        canonical_to_env = dict(env_to_canonical)
        return env_to_canonical, canonical_to_env

    env_to_canonical: dict[int, int] = {}
    canonical_to_env: dict[int, int] = {}
    for idx in range(players_per_side):
        env_off = idx
        env_def = players_per_side + idx
        can_user = idx
        can_ai = players_per_side + idx

        env_to_canonical[env_off] = can_ai
        env_to_canonical[env_def] = can_user
        canonical_to_env[can_user] = env_def
        canonical_to_env[can_ai] = env_off

    return env_to_canonical, canonical_to_env


def _map_player_id(value: Any, env_to_canonical: dict[int, int]) -> Any:
    if value is None:
        return None
    as_int = _to_int(value)
    if as_int is None:
        return value
    return env_to_canonical.get(as_int, as_int)


def _map_id_list(values: Any, env_to_canonical: dict[int, int]) -> Any:
    if not isinstance(values, list):
        return values
    mapped = []
    for value in values:
        mapped.append(_map_player_id(value, env_to_canonical))
    return mapped


def _remap_player_references(payload: Any, env_to_canonical: dict[int, int], parent_key: str | None = None) -> Any:
    id_fields = {
        "player_id",
        "stolen_by",
        "pass_target",
        "intended_target",
        "target",
        "ball_holder",
        "winner",
        "id",
        "passer_id",
        "recipient_id",
        "assist_passer_id",
    }
    id_list_fields = {"players", "offense_ids", "defense_ids"}

    if isinstance(payload, dict):
        out: dict[Any, Any] = {}
        for key, val in payload.items():
            mapped_key = key
            if _to_int(key) is not None:
                mapped_id = _map_player_id(key, env_to_canonical)
                mapped_key = str(mapped_id) if isinstance(key, str) else mapped_id

            if str(key) in id_fields:
                out[mapped_key] = _map_player_id(val, env_to_canonical)
            elif str(key) in id_list_fields:
                out[mapped_key] = _map_id_list(val, env_to_canonical)
            else:
                out[mapped_key] = _remap_player_references(val, env_to_canonical, str(key))
        return out

    if isinstance(payload, list):
        return [
            _remap_player_references(item, env_to_canonical, parent_key)
            for item in payload
        ]

    return payload


def _map_state_for_playable(state: dict[str, Any], session: dict[str, Any]) -> dict[str, Any]:
    players_per_side = int(session["players_per_side"])
    user_on_offense = bool(session["user_on_offense"])
    total = players_per_side * 2
    env_to_canonical, canonical_to_env = _build_id_maps(players_per_side, user_on_offense)

    mapped = copy.deepcopy(state or {})

    positions = mapped.get("positions") or []
    if isinstance(positions, list) and len(positions) == total:
        mapped["positions"] = [positions[canonical_to_env[idx]] for idx in range(total)]

    action_mask = mapped.get("action_mask") or []
    if isinstance(action_mask, list) and len(action_mask) == total:
        mapped["action_mask"] = [action_mask[canonical_to_env[idx]] for idx in range(total)]

    ep_by_player = mapped.get("ep_by_player") or []
    if isinstance(ep_by_player, list) and len(ep_by_player) == total:
        mapped["ep_by_player"] = [ep_by_player[canonical_to_env[idx]] for idx in range(total)]

    if mapped.get("offense_ids") is not None:
        mapped["offense_ids"] = _map_id_list(mapped.get("offense_ids"), env_to_canonical)
    if mapped.get("defense_ids") is not None:
        mapped["defense_ids"] = _map_id_list(mapped.get("defense_ids"), env_to_canonical)

    mapped["ball_holder"] = _map_player_id(mapped.get("ball_holder"), env_to_canonical)

    for dict_field in (
        "pass_steal_probabilities",
        "policy_probabilities",
        "action_values",
        "offensive_lane_steps",
        "defensive_lane_steps",
    ):
        value = mapped.get(dict_field)
        if isinstance(value, dict):
            remapped_dict: dict[Any, Any] = {}
            for key, val in value.items():
                mapped_id = _map_player_id(key, env_to_canonical)
                mapped_key = str(mapped_id) if isinstance(key, str) else mapped_id
                remapped_dict[mapped_key] = val
            mapped[dict_field] = remapped_dict

    mapped["last_action_results"] = _remap_player_references(
        mapped.get("last_action_results") or {}, env_to_canonical
    )

    mapped["playable_user_ids"] = list(range(players_per_side))
    mapped["playable_ai_ids"] = list(range(players_per_side, total))
    mapped["playable_user_on_offense"] = bool(user_on_offense)
    side_skills = session.get("side_skills") if isinstance(session.get("side_skills"), dict) else {}
    mapped["playable_side_skills"] = {
        "user": _clone_skill_set(side_skills.get("user")),
        "ai": _clone_skill_set(side_skills.get("ai")),
    }

    return mapped


def _build_possession_info(session: dict[str, Any]) -> dict[str, Any]:
    user_on_offense = bool(session["user_on_offense"])
    return {
        "number": int(session["possession_number"]),
        "offense_team": "user" if user_on_offense else "ai",
        "defense_team": "ai" if user_on_offense else "user",
        "user_on_offense": user_on_offense,
    }


def _build_score_info(session: dict[str, Any]) -> dict[str, int]:
    score = session.get("score") or {}
    return {
        "user": int(score.get("user", 0)),
        "ai": int(score.get("ai", 0)),
    }


def _current_possession_owner_is_user(session: dict[str, Any]) -> bool:
    return bool(session.get("user_on_offense", True))


def _resolve_shot_points(shot_result: dict[str, Any]) -> int:
    if not isinstance(shot_result, dict) or not shot_result.get("success"):
        return 0
    if bool(shot_result.get("is_three")):
        return 3
    return 2


def _build_possession_result(
    action_results: dict[str, Any],
    shot_clock: int,
    offense_is_user: bool,
    env_to_canonical: dict[int, int],
) -> dict[str, Any]:
    action_results = action_results or {}

    if action_results.get("shots"):
        first_shot_key, first_shot = next(iter(action_results["shots"].items()))
        points = _resolve_shot_points(first_shot)
        scorer_id = _map_player_id(first_shot_key, env_to_canonical)
        if points > 0:
            return {
                "type": "made_shot",
                "points": int(points),
                "scoring_team": "user" if offense_is_user else "ai",
                "scorer_id": scorer_id,
                "message": f"{('User' if offense_is_user else 'AI')} made a {points}-point shot.",
            }
        return {
            "type": "missed_shot",
            "points": 0,
            "scoring_team": None,
            "scorer_id": scorer_id,
            "message": "Shot attempt missed.",
        }

    if action_results.get("defensive_lane_violations"):
        first_violation = action_results["defensive_lane_violations"][0]
        violator_id = _map_player_id(first_violation.get("player_id"), env_to_canonical)
        return {
            "type": "defensive_violation",
            "points": 1,
            "scoring_team": "user" if offense_is_user else "ai",
            "violator_id": violator_id,
            "message": "Defensive lane violation (+1 point).",
        }

    offensive_violations = action_results.get("offensive_lane_violations") or []
    if offensive_violations:
        first_violation = offensive_violations[0]
        violator_id = _map_player_id(first_violation.get("player_id"), env_to_canonical)
        return {
            "type": "offensive_violation",
            "points": 0,
            "scoring_team": None,
            "violator_id": violator_id,
            "message": "Offensive lane violation.",
        }

    turnovers = action_results.get("turnovers") or []
    if turnovers:
        turnover = turnovers[0]
        player_id = _map_player_id(turnover.get("player_id"), env_to_canonical)
        reason = str(turnover.get("reason") or "turnover")
        return {
            "type": "turnover",
            "points": 0,
            "scoring_team": None,
            "player_id": player_id,
            "reason": reason,
            "message": f"Possession ended with turnover ({reason}).",
        }

    if int(shot_clock) <= 0:
        return {
            "type": "shot_clock_violation",
            "points": 0,
            "scoring_team": None,
            "message": "Shot clock violation.",
        }

    return {
        "type": "possession_end",
        "points": 0,
        "scoring_team": None,
        "message": "Possession ended.",
    }


def _reset_playable_possession(session: dict[str, Any]) -> dict[str, Any]:
    if not game_state.env:
        raise HTTPException(status_code=400, detail="Game not initialized.")

    env = game_state.env
    players_per_side = int(session["players_per_side"])
    user_on_offense = bool(session["user_on_offense"])
    current_team = Team.OFFENSE if user_on_offense else Team.DEFENSE

    game_state.user_team = current_team
    env.training_team = current_team
    env.shot_clock_steps = 24
    env.min_shot_clock = 24

    game_state.frames = []
    game_state.reward_history = []
    game_state.episode_rewards = {"offense": 0.0, "defense": 0.0}
    game_state.actions_log = []
    game_state.episode_states = []
    game_state.phi_log = []

    offense_team_key = "user" if user_on_offense else "ai"
    side_skills = session.get("side_skills") if isinstance(session.get("side_skills"), dict) else {}
    offense_skill_set = _clone_skill_set(side_skills.get(offense_team_key))
    has_expected_skill_len = all(
        len(offense_skill_set.get(key, [])) == players_per_side
        for key in ("layup", "three_pt", "dunk")
    )

    reset_options: dict[str, Any] = {"shot_clock": 24}
    if has_expected_skill_len:
        reset_options["offense_skills"] = offense_skill_set

    game_state.obs, _ = env.reset(options=reset_options)
    game_state.prev_obs = None
    _capture_turn_start_snapshot()

    game_state.sampled_offense_skills = (
        copy.deepcopy(offense_skill_set)
        if has_expected_skill_len
        else _read_env_offense_skills()
    )

    state = get_ui_game_state()
    game_state.episode_states.append(copy.deepcopy(state))
    return state


def _sanitize_options_for_response(matrix: dict[int, dict[str, dict[str, Any]]]) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for players in PLAYABLE_PLAYERS:
        out[str(players)] = {}
        for diff in PLAYABLE_DIFFICULTIES:
            cfg = matrix[players][diff]
            out[str(players)][diff] = {
                "available": bool(cfg.get("available", False)),
            }
            if cfg.get("reason"):
                out[str(players)][diff]["reason"] = str(cfg.get("reason"))
    return out


@router.get("/api/playable/options")
def get_playable_options():
    matrix = _load_playable_matrix()
    runtime_metrics = playable_session_store.metrics()
    demo_config = _build_playable_demo_config()
    return {
        "status": "success",
        "players_per_side": list(PLAYABLE_PLAYERS),
        "difficulties": list(PLAYABLE_DIFFICULTIES),
        "period_modes": list(PLAYABLE_PERIOD_MODES),
        "period_length_minutes": {
            "min": PLAYABLE_MIN_PERIOD_MINUTES,
            "max": PLAYABLE_MAX_PERIOD_MINUTES,
            "default": 5,
        },
        "options": _sanitize_options_for_response(matrix),
        "demo": _sanitize_demo_config_for_response(demo_config),
        "runtime_limits": {
            "max_active_sessions": int(runtime_metrics["max_active_sessions"]),
            "session_ttl_minutes": int(runtime_metrics["session_ttl_minutes"]),
        },
    }


@router.post("/api/playable/start")
async def start_playable_game(
    request: PlayableStartRequest,
    http_request: Request,
    http_response: Response,
):
    players_per_side = int(request.players_per_side)
    difficulty = str(request.difficulty).strip().lower()
    period_mode = _normalize_period_mode(request.period_mode)
    period_length_minutes = _normalize_period_length_minutes(request.period_length_minutes)
    total_periods = int(PLAYABLE_PERIOD_MODE_TO_COUNT[period_mode])

    if players_per_side not in PLAYABLE_PLAYERS:
        raise HTTPException(status_code=400, detail="players_per_side must be between 1 and 5.")
    if difficulty not in PLAYABLE_DIFFICULTIES:
        raise HTTPException(status_code=400, detail="difficulty must be easy, medium, or hard.")

    matrix = _load_playable_matrix()
    cfg = matrix.get(players_per_side, {}).get(difficulty, {})
    if not cfg.get("available"):
        raise HTTPException(status_code=400, detail=f"{players_per_side}v{players_per_side} {difficulty} is unavailable.")

    try:
        session_id, target_state, _ = playable_session_store.get_or_create_for_start(
            http_request.headers.get(PLAYABLE_SESSION_HEADER)
        )
    except PlayableCapacityError as exc:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Playable server is full ({exc.max_active} active games). "
                "Please try again shortly."
            ),
        ) from exc

    with bind_game_state(target_state):
        init_request = InitGameRequest(
            run_id=str(cfg["run_id"]),
            user_team_name="OFFENSE",
            unified_policy_name=str(cfg["policy_name"]),
            opponent_unified_policy_name=None,
        )

        await init_game_route(init_request)

        env_players = int(getattr(game_state.env, "players_per_side", 0) or 0)
        if env_players != players_per_side:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Configured playable option expects {players_per_side} players per side, "
                    f"but loaded run provides {env_players}."
                ),
            )

        game_state.playable_session = _build_new_playable_session(
            players_per_side=players_per_side,
            difficulty=difficulty,
            run_id=str(cfg["run_id"]),
            checkpoint_index=int(cfg["checkpoint_index"]),
            policy_name=str(cfg["policy_name"]),
            period_mode=period_mode,
            total_periods=total_periods,
            period_length_minutes=period_length_minutes,
            demo_mode=False,
        )

        session = game_state.playable_session
        state = _reset_playable_possession(session)
        game_clock = _build_game_clock_info(session)
        score = _build_score_info(session)
        possession = _build_possession_info(session)
        game_result = _build_game_result(session)

        payload = {
            "status": "success",
            "state": _map_state_for_playable(state, session),
            "score": score,
            "possession": possession,
            "game_clock": game_clock,
            "game_result": game_result,
            "config": _build_playable_config_info(session),
        }
        _emit_playable_event(
            session_id=session_id,
            session=session,
            event_type="game_started",
            payload={
                "source": "start",
                "coin_toss_winner": "user" if bool(session.get("user_on_offense")) else "ai",
                "user_on_offense": bool(session.get("user_on_offense")),
                "score": score,
                "possession": possession,
                "game_clock": game_clock,
                "game_result": game_result,
                "sampled_skills": copy.deepcopy(session.get("side_skills") or {}),
            },
        )
    return _attach_session_id(payload, session_id, http_response)


@router.post("/api/playable/demo/start")
async def start_playable_demo(
    http_request: Request,
    http_response: Response,
):
    demo_config = _build_playable_demo_config()
    if not bool(demo_config.get("enabled", False)):
        raise HTTPException(status_code=404, detail="Playable demo mode is disabled.")
    if not bool(demo_config.get("available", False)):
        raise HTTPException(
            status_code=400,
            detail=str(demo_config.get("reason") or "Playable demo mode is not configured."),
        )

    try:
        session_id, target_state, _ = playable_session_store.get_or_create_for_start(
            http_request.headers.get(PLAYABLE_SESSION_HEADER)
        )
    except PlayableCapacityError as exc:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Playable server is full ({exc.max_active} active games). "
                "Please try again shortly."
            ),
        ) from exc

    with bind_game_state(target_state):
        init_request = InitGameRequest(
            run_id=str(demo_config["run_id"]),
            user_team_name="OFFENSE",
            unified_policy_name=str(demo_config["policy_name"]),
            opponent_unified_policy_name=None,
        )

        await init_game_route(init_request)

        players_per_side = int(getattr(game_state.env, "players_per_side", 0) or 0)
        if players_per_side not in PLAYABLE_PLAYERS:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Playable demo run must load a supported players_per_side value "
                    f"(got {players_per_side})."
                ),
            )

        game_state.playable_session = _build_new_playable_session(
            players_per_side=players_per_side,
            difficulty="demo",
            run_id=str(demo_config["run_id"]),
            checkpoint_index=int(demo_config["checkpoint_index"]),
            policy_name=str(demo_config["policy_name"]),
            period_mode=str(demo_config["period_mode"]),
            total_periods=int(demo_config["total_periods"]),
            period_length_minutes=int(demo_config["period_length_minutes"]),
            demo_mode=True,
        )

        session = game_state.playable_session
        state = _reset_playable_possession(session)
        game_clock = _build_game_clock_info(session)
        score = _build_score_info(session)
        possession = _build_possession_info(session)
        game_result = _build_game_result(session)

        payload = {
            "status": "success",
            "state": _map_state_for_playable(state, session),
            "score": score,
            "possession": possession,
            "game_clock": game_clock,
            "game_result": game_result,
            "config": _build_playable_config_info(session),
        }
        _emit_playable_event(
            session_id=session_id,
            session=session,
            event_type="game_started",
            payload={
                "source": "demo_start",
                "coin_toss_winner": "user" if bool(session.get("user_on_offense")) else "ai",
                "user_on_offense": bool(session.get("user_on_offense")),
                "score": score,
                "possession": possession,
                "game_clock": game_clock,
                "game_result": game_result,
                "sampled_skills": copy.deepcopy(session.get("side_skills") or {}),
            },
        )
    return _attach_session_id(payload, session_id, http_response)


@router.post("/api/playable/new_game")
def playable_new_game(http_request: Request, http_response: Response):
    session_id, target_state = _resolve_session_state(http_request, require_active=True)
    with bind_game_state(target_state):
        session = _get_playable_session(required=True)
        period_mode = _normalize_period_mode(session.get("period_mode", "period"))
        period_length_minutes = _normalize_period_length_minutes(session.get("period_length_minutes", 5))
        total_periods = int(PLAYABLE_PERIOD_MODE_TO_COUNT.get(period_mode, 1))

        session["score"] = {"user": 0, "ai": 0}
        session["possession_number"] = 1
        session["user_on_offense"] = bool(random.getrandbits(1))
        session["side_skills"] = _build_playable_side_skills()
        session["period_mode"] = period_mode
        session["total_periods"] = total_periods
        session["current_period"] = 1
        session["period_length_minutes"] = period_length_minutes
        session["period_seconds_remaining"] = period_length_minutes * 60
        session["game_over"] = False
        session["turn_index"] = 1
        session["analytics"] = {
            "game_id": uuid.uuid4().hex,
            "next_seq": 1,
        }

        state = _reset_playable_possession(session)
        game_clock = _build_game_clock_info(session)
        score = _build_score_info(session)
        possession = _build_possession_info(session)
        game_result = _build_game_result(session)
        payload = {
            "status": "success",
            "state": _map_state_for_playable(state, session),
            "score": score,
            "possession": possession,
            "game_clock": game_clock,
            "game_result": game_result,
            "config": _build_playable_config_info(session),
        }
        _emit_playable_event(
            session_id=session_id,
            session=session,
            event_type="game_started",
            payload={
                "source": "demo_new_game" if bool(session.get("demo_mode", False)) else "new_game",
                "coin_toss_winner": "user" if bool(session.get("user_on_offense")) else "ai",
                "user_on_offense": bool(session.get("user_on_offense")),
                "score": score,
                "possession": possession,
                "game_clock": game_clock,
                "game_result": game_result,
                "sampled_skills": copy.deepcopy(session.get("side_skills") or {}),
            },
        )
    return _attach_session_id(payload, session_id, http_response)


@router.post("/api/playable/step")
def playable_step(
    request: PlayableStepRequest,
    http_request: Request,
    http_response: Response,
):
    session_id, target_state = _resolve_session_state(http_request, require_active=True)
    with bind_game_state(target_state):
        session = _get_playable_session(required=True)
        _ensure_playable_game_fields(session)
        if bool(session.get("game_over", False)):
            raise HTTPException(status_code=400, detail="Game is over. Start a new game.")

        players_per_side = int(session["players_per_side"])
        auto_user_actions = bool(request.auto_user_actions or session.get("demo_mode", False))

        env_to_canonical, canonical_to_env = _build_id_maps(
            players_per_side,
            bool(session["user_on_offense"]),
        )

        user_actions: dict[str, Any] = {}
        submitted_actions: dict[str, Any] = {}
        provided = request.actions or {}
        turn_index = int(session.get("turn_index", 1) or 1)
        score_before = _build_score_info(session)
        game_clock_before = _build_game_clock_info(session)
        shot_clock_before = int(getattr(game_state.env, "shot_clock", 0) or 0)

        for canonical_pid in range(players_per_side):
            raw_action = provided.get(str(canonical_pid), provided.get(canonical_pid))
            if raw_action is None:
                if auto_user_actions:
                    continue
                raw_action = "NOOP"
            env_pid = canonical_to_env[canonical_pid]
            user_actions[str(env_pid)] = raw_action
            submitted_actions[str(canonical_pid)] = raw_action

        _emit_playable_event(
            session_id=session_id,
            session=session,
            event_type="turn_submitted",
            payload={
                "turn_index": int(turn_index),
                "score_before": score_before,
                "game_clock_before": game_clock_before,
                "shot_clock_before": int(shot_clock_before),
                "auto_user_actions": bool(auto_user_actions),
                "submitted_actions": copy.deepcopy(submitted_actions),
            },
        )

        step_payload = lifecycle_step(
            ActionRequest(
                actions=user_actions,
                player_deterministic=not auto_user_actions,
                opponent_deterministic=False,
                use_mcts=False,
            )
        )

        if not isinstance(step_payload, dict) or step_payload.get("status") != "success":
            raise HTTPException(status_code=500, detail="Playable step failed.")

        actions_taken = _remap_player_references(step_payload.get("actions_taken") or {}, env_to_canonical)
        actions_taken_meta = _remap_player_references(step_payload.get("actions_taken_meta") or {}, env_to_canonical)

        step_state = step_payload.get("state") or {}
        possession_done = bool(step_state.get("done"))
        offense_is_user = _current_possession_owner_is_user(session)
        possession_result = None
        ended_state = None

        if possession_done:
            ended_state = _map_state_for_playable(step_state, session)
            possession_result = _build_possession_result(
                getattr(game_state.env, "last_action_results", {}) or {},
                int(getattr(game_state.env, "shot_clock", 0)),
                offense_is_user,
                env_to_canonical,
            )

            points = int(possession_result.get("points", 0) or 0)
            scoring_team = possession_result.get("scoring_team")
            if points > 0 and scoring_team in {"user", "ai"}:
                session["score"][str(scoring_team)] = int(session["score"].get(str(scoring_team), 0)) + points

            # Alternate possession at end of each possession.
            session["user_on_offense"] = not offense_is_user
            session["possession_number"] = int(session.get("possession_number", 1)) + 1

        # Every playable tick decrements the game clock by exactly one second.
        _tick_game_clock(session)
        period_ended = int(session.get("period_seconds_remaining", 0) or 0) <= 0
        period_result = None

        next_state_raw = None
        if period_ended:
            current_period = int(session.get("current_period", 1) or 1)
            total_periods = int(session.get("total_periods", 1) or 1)
            period_mode = str(session.get("period_mode", "period"))
            ended_segment_label = _segment_label_for_mode(period_mode, current_period)

            if current_period >= total_periods:
                session["game_over"] = True
                period_result = {
                    "type": "final_buzzer",
                    "message": f"End of {ended_segment_label}.",
                }
            else:
                session["current_period"] = current_period + 1
                period_minutes = int(session.get("period_length_minutes", 5) or 5)
                session["period_seconds_remaining"] = period_minutes * 60
                if not possession_done:
                    # Force a fresh possession at the start of a new segment.
                    session["possession_number"] = int(session.get("possession_number", 1)) + 1
                period_result = {
                    "type": "period_end",
                    "message": f"End of {ended_segment_label}.",
                }
                next_state_raw = _reset_playable_possession(session)

        score_after = _build_score_info(session)
        game_clock_after = _build_game_clock_info(session)
        shot_clock_after = int(getattr(game_state.env, "shot_clock", 0) or 0)
        game_result = _build_game_result(session)

        _emit_playable_event(
            session_id=session_id,
            session=session,
            event_type="turn_resolved",
            payload={
                "turn_index": int(turn_index),
                "score_before": score_before,
                "score_after": score_after,
                "game_clock_before": game_clock_before,
                "game_clock_after": game_clock_after,
                "shot_clock_before": int(shot_clock_before),
                "shot_clock_after": int(shot_clock_after),
                "auto_user_actions": bool(auto_user_actions),
                "submitted_actions": copy.deepcopy(submitted_actions),
                "actions_taken": copy.deepcopy(actions_taken),
                "actions_taken_meta": copy.deepcopy(actions_taken_meta),
                "possession_ended": bool(possession_done),
                "possession_result": copy.deepcopy(possession_result) if isinstance(possession_result, dict) else possession_result,
                "period_ended": bool(period_ended),
                "period_result": copy.deepcopy(period_result) if isinstance(period_result, dict) else period_result,
                "game_over": bool(session.get("game_over", False)),
            },
        )

        if period_ended and not bool(session.get("game_over", False)):
            _emit_playable_event(
                session_id=session_id,
                session=session,
                event_type="period_ended",
                payload={
                    "turn_index": int(turn_index),
                    "score": score_after,
                    "game_clock_after": game_clock_after,
                    "period_result": copy.deepcopy(period_result) if isinstance(period_result, dict) else period_result,
                },
            )

        if bool(session.get("game_over", False)):
            _emit_playable_event(
                session_id=session_id,
                session=session,
                event_type="game_ended",
                payload={
                    "turn_index": int(turn_index),
                    "winner": str(game_result.get("winner")) if isinstance(game_result, dict) else None,
                    "final_score": score_after,
                    "game_result": game_result,
                },
            )

        session["turn_index"] = int(turn_index) + 1

        if bool(session.get("game_over", False)):
            # Preserve the final board state at the buzzer.
            final_state = _map_state_for_playable(step_state, session)
            payload = {
                "status": "success",
                "state": final_state,
                "ended_state": ended_state,
                "actions_taken": actions_taken,
                "actions_taken_meta": actions_taken_meta,
                "score": score_after,
                "possession": _build_possession_info(session),
                "game_clock": game_clock_after,
                "game_result": game_result,
                "possession_ended": bool(possession_done),
                "possession_result": possession_result,
                "period_ended": bool(period_ended),
                "period_result": period_result,
            }
            return _attach_session_id(payload, session_id, http_response)

        if next_state_raw is None:
            if possession_done:
                next_state_raw = _reset_playable_possession(session)
            else:
                next_state_raw = step_state

        payload = {
            "status": "success",
            "state": _map_state_for_playable(next_state_raw, session),
            "ended_state": ended_state if possession_done else None,
            "actions_taken": actions_taken,
            "actions_taken_meta": actions_taken_meta,
            "score": score_after,
            "possession": _build_possession_info(session),
            "game_clock": game_clock_after,
            "game_result": game_result,
            "possession_ended": bool(possession_done),
            "possession_result": possession_result,
            "period_ended": bool(period_ended),
            "period_result": period_result,
        }
        return _attach_session_id(payload, session_id, http_response)


@router.get("/api/playable/state")
def get_playable_state(http_request: Request, http_response: Response):
    session_id, target_state = _resolve_session_state(http_request, require_active=True)
    with bind_game_state(target_state):
        session = _get_playable_session(required=True)
        _ensure_playable_game_fields(session)
        state = get_ui_game_state()
        payload = {
            "status": "success",
            "state": _map_state_for_playable(state, session),
            "score": _build_score_info(session),
            "possession": _build_possession_info(session),
            "game_clock": _build_game_clock_info(session),
            "game_result": _build_game_result(session),
            "config": _build_playable_config_info(session),
        }
    return _attach_session_id(payload, session_id, http_response)


@router.get("/api/playable/config")
def get_playable_runtime_config():
    runtime_metrics = playable_session_store.metrics()
    return {
        "status": "success",
        "public_mode": _is_truthy(os.getenv("BW_PUBLIC_MODE")),
        "matrix_source_json": bool(os.getenv("BW_PLAYABLE_POLICY_MATRIX_JSON")),
        "demo": _sanitize_demo_config_for_response(_build_playable_demo_config()),
        "session_header": PLAYABLE_SESSION_HEADER,
        "active_sessions": int(runtime_metrics["active_sessions"]),
        "max_active_sessions": int(runtime_metrics["max_active_sessions"]),
        "session_ttl_minutes": int(runtime_metrics["session_ttl_minutes"]),
        "analytics": playable_analytics_emitter.runtime_status(),
    }


@router.get("/api/playable/analytics_debug")
def get_playable_analytics_debug(limit: int = 50):
    if not _is_truthy(os.getenv("BW_ANALYTICS_DEBUG_ENABLED")):
        raise HTTPException(status_code=404, detail="Analytics debug endpoint is disabled.")
    return {
        "status": "success",
        "analytics": playable_analytics_emitter.runtime_status(),
        "debug": playable_analytics_emitter.debug_snapshot(limit=limit),
    }
