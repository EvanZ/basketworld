import copy
import json
import os
import random
from typing import Any

from fastapi import APIRouter, HTTPException

from basketworld.envs.basketworld_env_v2 import Team
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
    return {
        "status": "success",
        "players_per_side": list(PLAYABLE_PLAYERS),
        "difficulties": list(PLAYABLE_DIFFICULTIES),
        "options": _sanitize_options_for_response(matrix),
    }


@router.post("/api/playable/start")
async def start_playable_game(request: PlayableStartRequest):
    players_per_side = int(request.players_per_side)
    difficulty = str(request.difficulty).strip().lower()

    if players_per_side not in PLAYABLE_PLAYERS:
        raise HTTPException(status_code=400, detail="players_per_side must be between 1 and 5.")
    if difficulty not in PLAYABLE_DIFFICULTIES:
        raise HTTPException(status_code=400, detail="difficulty must be easy, medium, or hard.")

    matrix = _load_playable_matrix()
    cfg = matrix.get(players_per_side, {}).get(difficulty, {})
    if not cfg.get("available"):
        raise HTTPException(status_code=400, detail=f"{players_per_side}v{players_per_side} {difficulty} is unavailable.")

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

    user_starts = bool(random.getrandbits(1))
    side_skills = _build_playable_side_skills()
    game_state.playable_session = {
        "active": True,
        "players_per_side": players_per_side,
        "difficulty": difficulty,
        "run_id": str(cfg["run_id"]),
        "checkpoint_index": int(cfg["checkpoint_index"]),
        "policy_name": str(cfg["policy_name"]),
        "score": {"user": 0, "ai": 0},
        "possession_number": 1,
        "user_on_offense": user_starts,
        "side_skills": side_skills,
    }

    session = game_state.playable_session
    state = _reset_playable_possession(session)

    return {
        "status": "success",
        "state": _map_state_for_playable(state, session),
        "score": _build_score_info(session),
        "possession": _build_possession_info(session),
        "config": {
            "players_per_side": players_per_side,
            "difficulty": difficulty,
        },
    }


@router.post("/api/playable/new_game")
def playable_new_game():
    session = _get_playable_session(required=True)
    session["score"] = {"user": 0, "ai": 0}
    session["possession_number"] = 1
    session["user_on_offense"] = bool(random.getrandbits(1))
    session["side_skills"] = _build_playable_side_skills()

    state = _reset_playable_possession(session)
    return {
        "status": "success",
        "state": _map_state_for_playable(state, session),
        "score": _build_score_info(session),
        "possession": _build_possession_info(session),
    }


@router.post("/api/playable/step")
def playable_step(request: PlayableStepRequest):
    session = _get_playable_session(required=True)
    players_per_side = int(session["players_per_side"])

    env_to_canonical, canonical_to_env = _build_id_maps(
        players_per_side,
        bool(session["user_on_offense"]),
    )

    user_actions: dict[str, Any] = {}
    provided = request.actions or {}

    for canonical_pid in range(players_per_side):
        raw_action = provided.get(str(canonical_pid), provided.get(canonical_pid))
        if raw_action is None:
            raw_action = "NOOP"
        env_pid = canonical_to_env[canonical_pid]
        user_actions[str(env_pid)] = raw_action

    step_payload = lifecycle_step(
        ActionRequest(
            actions=user_actions,
            player_deterministic=True,
            opponent_deterministic=True,
            use_mcts=False,
        )
    )

    if not isinstance(step_payload, dict) or step_payload.get("status") != "success":
        raise HTTPException(status_code=500, detail="Playable step failed.")

    actions_taken = _remap_player_references(step_payload.get("actions_taken") or {}, env_to_canonical)
    actions_taken_meta = _remap_player_references(step_payload.get("actions_taken_meta") or {}, env_to_canonical)

    step_state = step_payload.get("state") or {}
    possession_done = bool(step_state.get("done"))

    if not possession_done:
        return {
            "status": "success",
            "state": _map_state_for_playable(step_state, session),
            "actions_taken": actions_taken,
            "actions_taken_meta": actions_taken_meta,
            "score": _build_score_info(session),
            "possession": _build_possession_info(session),
            "possession_ended": False,
            "possession_result": None,
        }

    offense_is_user = _current_possession_owner_is_user(session)
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

    # Alternate possession and spawn a new one immediately.
    session["user_on_offense"] = not offense_is_user
    session["possession_number"] = int(session.get("possession_number", 1)) + 1

    next_state = _reset_playable_possession(session)

    return {
        "status": "success",
        "state": _map_state_for_playable(next_state, session),
        "ended_state": ended_state,
        "actions_taken": actions_taken,
        "actions_taken_meta": actions_taken_meta,
        "score": _build_score_info(session),
        "possession": _build_possession_info(session),
        "possession_ended": True,
        "possession_result": possession_result,
    }


@router.get("/api/playable/state")
def get_playable_state():
    session = _get_playable_session(required=True)
    state = get_ui_game_state()
    return {
        "status": "success",
        "state": _map_state_for_playable(state, session),
        "score": _build_score_info(session),
        "possession": _build_possession_info(session),
    }


@router.get("/api/playable/config")
def get_playable_runtime_config():
    return {
        "status": "success",
        "public_mode": _is_truthy(os.getenv("BW_PUBLIC_MODE")),
        "matrix_source_json": bool(os.getenv("BW_PLAYABLE_POLICY_MATRIX_JSON")),
    }
