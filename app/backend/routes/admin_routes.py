import copy
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import logging
import math
import multiprocessing as mp
import os
import queue
from typing import List

from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
import mlflow
import numpy as np
from stable_baselines3 import PPO
from basketworld.utils.policies import PassBiasDualCriticPolicy, PassBiasMultiInputPolicy
from basketworld.policies import SetAttentionDualCriticPolicy, SetAttentionExtractor

import app.backend.evaluation as backend_evaluation
from app.backend.observations import rebuild_observation_from_env
from app.backend.rollout_runtime import (
    apply_post_step_rollout_updates,
    combine_team_actions,
    predict_joint_policy_actions,
)
from app.backend.env_access import env_view
from app.backend.schemas import (
    ActionRequest,
    BatchUpdatePositionRequest,
    PlaybookAnalysisRequest,
    ReplayCounterfactualRequest,
    SetBallHolderRequest,
    SetIntentStateRequest,
    SetOffenseSkillsRequest,
    SetPassLogitBiasRequest,
    SetPressureParamsRequest,
    SetPassTargetStrategyRequest,
    SwapPoliciesRequest,
    UpdatePositionRequest,
    UpdateShotClockRequest,
)
from app.backend.state import (
    _capture_restorable_backend_state,
    _rebuild_cached_obs,
    _restore_restorable_backend_state,
    capture_counterfactual_snapshot,
    fail_playbook_progress,
    get_playbook_progress,
    get_ui_game_state,
    game_state,
    reset_playbook_progress,
    restore_counterfactual_snapshot,
    update_playbook_progress,
)
from app.backend.policies import _compute_param_counts_from_policy, get_unified_policy_path


router = APIRouter()
logger = logging.getLogger(__name__)
_NUMPY_SAFE_ENCODER = {
    np.integer: int,
    np.floating: float,
    np.bool_: bool,
    np.ndarray: lambda arr: arr.tolist(),
}


def _base_env():
    env = game_state.env
    return getattr(env, "unwrapped", env)


def _heatmap_increment(heatmap: dict[str, list[int]], pos) -> None:
    if not isinstance(pos, (list, tuple)) or len(pos) < 2:
        return
    try:
        q = int(pos[0])
        r = int(pos[1])
    except Exception:
        return
    key = f"{q},{r}"
    slot = heatmap.setdefault(key, [0, 0])
    slot[0] += 1
    slot[1] += 1


def _increment_count(counts: dict[str, int], key: str, amount: int = 1) -> None:
    counts[str(key)] = int(counts.get(str(key), 0)) + int(amount)


def _classify_playbook_terminal_outcome(
    state: dict,
    done: bool,
    shot_clock: int | None,
) -> tuple[str, str | None]:
    if not done:
        return "horizon_cutoff", None

    action_results = state.get("last_action_results") or {}
    raw_shots = action_results.get("shots") or {}
    if raw_shots:
        any_make = False
        for shot_result in raw_shots.values():
            if isinstance(shot_result, dict) and bool(shot_result.get("success")):
                any_make = True
                break
        return ("shot_make" if any_make else "shot_miss"), None

    if action_results.get("defensive_lane_violations"):
        return "defensive_violation", None

    raw_turnovers = action_results.get("turnovers") or []
    if raw_turnovers:
        turnover_reason = "unknown"
        first = raw_turnovers[0]
        if isinstance(first, dict):
            turnover_reason = str(first.get("reason") or "unknown").lower()
        return "turnover", turnover_reason

    try:
        if shot_clock is not None and int(shot_clock) <= 0:
            return "shot_clock_expiration", None
    except Exception:
        pass

    return "other_terminal", None


def _count_rollout_state(
    state: dict,
    offense_ids: list[int],
    player_heatmaps: dict[str, dict[str, list[int]]],
    ball_heatmap: dict[str, list[int]],
    shot_heatmap: dict[str, list[int]],
    player_shot_heatmaps: dict[str, dict[str, list[int]]],
    player_shot_stats: dict[str, dict[str, int]],
    pass_links: dict[str, int],
    pass_path_segments: dict[str, int] | None = None,
) -> tuple[int, int]:
    positions = state.get("positions") or []
    passes_this_state = 0
    shots_this_state = 0

    for pid in offense_ids:
        if pid < 0 or pid >= len(positions):
            continue
        player_map = player_heatmaps.setdefault(str(pid), {})
        _heatmap_increment(player_map, positions[pid])

    ball_holder = state.get("ball_holder")
    if ball_holder is not None:
        try:
            bh = int(ball_holder)
        except Exception:
            bh = None
        if bh is not None and 0 <= bh < len(positions):
            _heatmap_increment(ball_heatmap, positions[bh])

    action_results = state.get("last_action_results") or {}
    raw_passes = action_results.get("passes") or {}
    for raw_passer, pass_result in raw_passes.items():
        if not isinstance(pass_result, dict) or not pass_result.get("success"):
            continue
        try:
            passer = int(raw_passer)
        except Exception:
            continue
        raw_target = pass_result.get("target")
        if raw_target is None:
            raw_target = pass_result.get("intended_target")
        try:
            target = int(raw_target)
        except Exception:
            continue
        pass_links[f"{passer}->{target}"] = int(pass_links.get(f"{passer}->{target}", 0)) + 1
        if pass_path_segments is not None and 0 <= passer < len(positions) and 0 <= target < len(positions):
            _pass_segment_increment(pass_path_segments, passer, target, positions[passer], positions[target])
        passes_this_state += 1

    raw_shots = action_results.get("shots") or {}
    for raw_shooter, shot_result in raw_shots.items():
        try:
            shooter = int(raw_shooter)
        except Exception:
            continue
        if 0 <= shooter < len(positions):
            _heatmap_increment(shot_heatmap, positions[shooter])
            _heatmap_increment(
                player_shot_heatmaps.setdefault(str(shooter), {}),
                positions[shooter],
            )
            shooter_stats = player_shot_stats.setdefault(
                str(shooter),
                {"attempts": 0, "makes": 0},
            )
            shooter_stats["attempts"] = int(shooter_stats.get("attempts", 0)) + 1
            if isinstance(shot_result, dict) and bool(shot_result.get("success")):
                shooter_stats["makes"] = int(shooter_stats.get("makes", 0)) + 1
            shots_this_state += 1

    return passes_this_state, shots_this_state


def _init_playbook_panel_accumulator(offense_ids: list[int]) -> dict:
    return {
        "player_heatmaps": {str(pid): {} for pid in offense_ids},
        "ball_heatmap": {},
        "shot_heatmap": {},
        "player_shot_heatmaps": {str(pid): {} for pid in offense_ids},
        "player_shot_stats": {
            str(pid): {"attempts": 0, "makes": 0} for pid in offense_ids
        },
        "terminal_outcomes": {},
        "turnover_reasons": {},
        "pass_links": {},
        "player_path_segments": {str(pid): {} for pid in offense_ids},
        "ball_path_segments": {},
        "pass_path_segments": {},
        "rollout_lengths_sum": 0,
        "rollout_passes_sum": 0,
        "rollout_shots_sum": 0,
        "terminated_rollouts": 0,
        "num_rollouts": 0,
        "base_state": None,
    }


def _merge_heatmap_counts(dest: dict[str, list[int]], src: dict[str, list[int]]) -> None:
    for key, vals in (src or {}).items():
        try:
            src_0 = int(vals[0]) if isinstance(vals, (list, tuple)) and len(vals) > 0 else 0
            src_1 = int(vals[1]) if isinstance(vals, (list, tuple)) and len(vals) > 1 else 0
        except Exception:
            src_0, src_1 = 0, 0
        slot = dest.setdefault(key, [0, 0])
        slot[0] += src_0
        slot[1] += src_1


def _merge_player_heatmaps(
    dest: dict[str, dict[str, list[int]]],
    src: dict[str, dict[str, list[int]]],
) -> None:
    for pid, player_map in (src or {}).items():
        _merge_heatmap_counts(dest.setdefault(str(pid), {}), player_map or {})


def _segment_key(start_pos, end_pos) -> str | None:
    if not isinstance(start_pos, (list, tuple)) or not isinstance(end_pos, (list, tuple)):
        return None
    if len(start_pos) < 2 or len(end_pos) < 2:
        return None
    try:
        start_q = int(start_pos[0])
        start_r = int(start_pos[1])
        end_q = int(end_pos[0])
        end_r = int(end_pos[1])
    except Exception:
        return None
    if start_q == end_q and start_r == end_r:
        return None
    return f"{start_q},{start_r}|{end_q},{end_r}"


def _segment_increment(segment_counts: dict[str, int], start_pos, end_pos) -> None:
    key = _segment_key(start_pos, end_pos)
    if not key:
        return
    segment_counts[key] = int(segment_counts.get(key, 0)) + 1


def _pass_segment_increment(
    segment_counts: dict[str, int],
    passer_id: int,
    receiver_id: int,
    start_pos,
    end_pos,
) -> None:
    key = _segment_key(start_pos, end_pos)
    if not key:
        return
    scoped_key = f"{int(passer_id)}->{int(receiver_id)}::{key}"
    segment_counts[scoped_key] = int(segment_counts.get(scoped_key, 0)) + 1


def _count_rollout_transition(
    prev_state: dict,
    state: dict,
    offense_ids: list[int],
    player_path_segments: dict[str, dict[str, int]],
    ball_path_segments: dict[str, int],
) -> None:
    prev_positions = prev_state.get("positions") or []
    curr_positions = state.get("positions") or []

    for pid in offense_ids:
        if pid < 0 or pid >= len(prev_positions) or pid >= len(curr_positions):
            continue
        _segment_increment(
            player_path_segments.setdefault(str(pid), {}),
            prev_positions[pid],
            curr_positions[pid],
        )

    try:
        prev_holder = int(prev_state.get("ball_holder"))
        curr_holder = int(state.get("ball_holder"))
    except Exception:
        return
    if (
        prev_holder < 0
        or curr_holder < 0
        or prev_holder >= len(prev_positions)
        or curr_holder >= len(curr_positions)
    ):
        return
    _segment_increment(
        ball_path_segments,
        prev_positions[prev_holder],
        curr_positions[curr_holder],
    )


def _merge_segment_counts(dest: dict[str, int], src: dict[str, int]) -> None:
    for key, count in (src or {}).items():
        dest[key] = int(dest.get(key, 0)) + int(count or 0)


def _merge_player_segment_counts(
    dest: dict[str, dict[str, int]],
    src: dict[str, dict[str, int]],
) -> None:
    for pid, player_segments in (src or {}).items():
        _merge_segment_counts(dest.setdefault(str(pid), {}), player_segments or {})


def _serialize_segment_counts(segment_counts: dict[str, int]) -> list[dict]:
    serialized: list[dict] = []
    for key, raw_count in sorted(
        (segment_counts or {}).items(),
        key=lambda item: (-int(item[1] or 0), str(item[0])),
    ):
        try:
            start_key, end_key = str(key).split("|", 1)
            start_q, start_r = (int(part) for part in start_key.split(",", 1))
            end_q, end_r = (int(part) for part in end_key.split(",", 1))
            count = int(raw_count)
        except Exception:
            continue
        if count <= 0:
            continue
        serialized.append(
            {
                "from": [start_q, start_r],
                "to": [end_q, end_r],
                "count": count,
            }
        )
    return serialized


def _serialize_pass_segment_counts(segment_counts: dict[str, int]) -> list[dict]:
    serialized: list[dict] = []
    for key, raw_count in sorted(
        (segment_counts or {}).items(),
        key=lambda item: (-int(item[1] or 0), str(item[0])),
    ):
        try:
            participants_key, segment_key = str(key).split("::", 1)
            passer_key, receiver_key = participants_key.split("->", 1)
            passer_id = int(passer_key)
            receiver_id = int(receiver_key)
            start_key, end_key = str(segment_key).split("|", 1)
            start_q, start_r = (int(part) for part in start_key.split(",", 1))
            end_q, end_r = (int(part) for part in end_key.split(",", 1))
            count = int(raw_count)
        except Exception:
            continue
        if count <= 0:
            continue
        serialized.append(
            {
                "passer_id": passer_id,
                "receiver_id": receiver_id,
                "from": [start_q, start_r],
                "to": [end_q, end_r],
                "count": count,
            }
        )
    return serialized


def _merge_playbook_panel_accumulator(dest: dict, src: dict) -> None:
    _merge_player_heatmaps(dest["player_heatmaps"], src.get("player_heatmaps") or {})
    _merge_heatmap_counts(dest["ball_heatmap"], src.get("ball_heatmap") or {})
    _merge_heatmap_counts(dest["shot_heatmap"], src.get("shot_heatmap") or {})
    _merge_player_heatmaps(
        dest["player_shot_heatmaps"], src.get("player_shot_heatmaps") or {}
    )
    for pid, raw_stats in (src.get("player_shot_stats") or {}).items():
        slot = dest["player_shot_stats"].setdefault(
            str(pid),
            {"attempts": 0, "makes": 0},
        )
        slot["attempts"] = int(slot.get("attempts", 0)) + int(
            (raw_stats or {}).get("attempts", 0) or 0
        )
        slot["makes"] = int(slot.get("makes", 0)) + int(
            (raw_stats or {}).get("makes", 0) or 0
        )
    for key, count in (src.get("terminal_outcomes") or {}).items():
        _increment_count(dest["terminal_outcomes"], str(key), int(count or 0))
    for key, count in (src.get("turnover_reasons") or {}).items():
        _increment_count(dest["turnover_reasons"], str(key), int(count or 0))
    _merge_player_segment_counts(dest["player_path_segments"], src.get("player_path_segments") or {})
    _merge_segment_counts(dest["ball_path_segments"], src.get("ball_path_segments") or {})
    _merge_segment_counts(dest["pass_path_segments"], src.get("pass_path_segments") or {})
    for key, count in (src.get("pass_links") or {}).items():
        dest["pass_links"][key] = int(dest["pass_links"].get(key, 0)) + int(count or 0)
    for key in (
        "rollout_lengths_sum",
        "rollout_passes_sum",
        "rollout_shots_sum",
        "terminated_rollouts",
        "num_rollouts",
    ):
        dest[key] = int(dest.get(key, 0)) + int(src.get(key, 0) or 0)
    if dest.get("base_state") is None and src.get("base_state") is not None:
        dest["base_state"] = copy.deepcopy(src.get("base_state"))


def _build_playbook_rollout_state(env, action_results: dict | None = None) -> dict:
    return {
        "positions": copy.deepcopy(getattr(env, "positions", [])),
        "ball_holder": getattr(env, "ball_holder", None),
        "last_action_results": action_results or {},
    }


def _build_playbook_ui_state(template_state: dict, env, action_results: dict | None = None) -> dict:
    env_read = env_view(env)
    state = copy.deepcopy(template_state or {})
    state["positions"] = [
        [int(pos[0]), int(pos[1])] for pos in getattr(env, "positions", [])
    ]
    ball_holder = getattr(env, "ball_holder", None)
    state["ball_holder"] = int(ball_holder) if ball_holder is not None else None
    state["shot_clock"] = int(getattr(env, "shot_clock", 0))
    state["done"] = bool(env_read.episode_ended)
    state["last_action_results"] = copy.deepcopy(action_results or {})
    state["intent_active_current"] = bool(getattr(env, "intent_active", False))
    state["intent_index_current"] = int(getattr(env, "intent_index", 0))
    state["intent_age_current"] = int(getattr(env, "intent_age", 0))
    state["pass_steal_probabilities"] = {}
    state["ball_handler_shot_probability"] = None
    return state


def _build_playbook_worker_obs(env, user_team) -> dict:
    worker_state = backend_evaluation._worker_state
    _Team = worker_state["Team"]
    role_value = (
        worker_state.get("role_flag_offense", 1.0)
        if user_team == _Team.OFFENSE
        else worker_state.get("role_flag_defense", -1.0)
    )
    return rebuild_observation_from_env(env, role_flag_value=float(role_value))


def _run_playbook_batch_worker(args: tuple) -> dict:
    (
        rollout_specs,
        base_env,
        base_ui_state,
        user_team_name,
        offense_ids,
        commitment_steps,
        max_steps,
        run_to_end,
        player_deterministic,
        opponent_deterministic,
    ) = args

    worker_state = backend_evaluation._worker_state
    _np = worker_state["np"]
    _Team = worker_state["Team"]
    training_params = worker_state.get("training_params", {}) or {}
    unified_policy = worker_state["unified_policy"]
    opponent_policy = worker_state["opponent_policy"]
    progress_queue = worker_state.get("progress_queue")
    user_team = _Team.OFFENSE if user_team_name == "OFFENSE" else _Team.DEFENSE

    try:
        import torch as _torch
    except Exception:
        _torch = None

    payload: dict[int, dict] = {}

    for _, intent_index, rollout_seed in rollout_specs:
        if _torch is not None:
            try:
                _torch.manual_seed(int(rollout_seed))
            except Exception:
                pass
        try:
            _np.random.seed(int(rollout_seed))
        except Exception:
            pass

        env_copy = copy.deepcopy(base_env)
        env = getattr(env_copy, "unwrapped", env_copy)
        env.training_team = user_team
        try:
            env._rng = _np.random.default_rng(int(rollout_seed))
        except Exception:
            pass

        env.intent_active = True
        env.intent_index = int(intent_index)
        env.intent_age = 0
        env.intent_commitment_remaining = int(commitment_steps)

        obs = _build_playbook_worker_obs(env, user_team)
        selector_segment_index = 0
        panel = payload.setdefault(int(intent_index), _init_playbook_panel_accumulator(offense_ids))
        initial_ui_state = _build_playbook_ui_state(base_ui_state, env)
        if panel.get("base_state") is None:
            panel["base_state"] = copy.deepcopy(initial_ui_state)

        rollout_state = _build_playbook_rollout_state(env)
        passes_count, shots_count = _count_rollout_state(
            rollout_state,
            offense_ids,
            panel["player_heatmaps"],
            panel["ball_heatmap"],
            panel["shot_heatmap"],
            panel["player_shot_heatmaps"],
            panel["player_shot_stats"],
            panel["pass_links"],
            panel["pass_path_segments"],
        )
        rollout_steps = 0
        done = bool(env_view(env).episode_ended)

        while not done and (run_to_end or rollout_steps < max_steps):
            rollout_actions = predict_joint_policy_actions(
                unified_policy=unified_policy,
                opponent_policy=opponent_policy if opponent_policy is not None else unified_policy,
                obs=obs,
                env=env,
                player_deterministic=player_deterministic,
                opponent_deterministic=opponent_deterministic,
                role_flag_offense=float(worker_state.get("role_flag_offense", 1.0)),
                role_flag_defense=float(worker_state.get("role_flag_defense", -1.0)),
            )
            full_action = combine_team_actions(
                env=env,
                user_team=user_team,
                resolved_unified=rollout_actions["resolved_unified"],
                resolved_opponent=rollout_actions["resolved_opponent"],
            )

            obs, _, terminated, truncated, info = env.step(full_action)
            action_results = info.get("action_results", {}) if info else {}
            if not action_results:
                action_results = info.get("last_action_results", {}) if info else {}
            if not action_results:
                action_results = getattr(env, "last_action_results", {}) or {}
            done = bool(terminated or truncated)
            post_step = apply_post_step_rollout_updates(
                env=env,
                next_obs=obs,
                info=info if isinstance(info, dict) else None,
                done=done,
                training_params=training_params,
                unified_policy=unified_policy,
                opponent_policy=opponent_policy,
                user_team=user_team,
                role_flag_offense=float(worker_state.get("role_flag_offense", 1.0)),
                role_flag_defense=float(worker_state.get("role_flag_defense", -1.0)),
                selector_segment_index=selector_segment_index,
            )
            obs = post_step["obs"]
            selector_segment_index = int(post_step["selector_segment_index"])

            next_rollout_state = _build_playbook_rollout_state(env, action_results)
            _count_rollout_transition(
                rollout_state,
                next_rollout_state,
                offense_ids,
                panel["player_path_segments"],
                panel["ball_path_segments"],
            )
            more_passes, more_shots = _count_rollout_state(
                next_rollout_state,
                offense_ids,
                panel["player_heatmaps"],
                panel["ball_heatmap"],
                panel["shot_heatmap"],
                panel["player_shot_heatmaps"],
                panel["player_shot_stats"],
                panel["pass_links"],
                panel["pass_path_segments"],
            )
            passes_count += more_passes
            shots_count += more_shots
            rollout_steps += 1
            rollout_state = next_rollout_state

        panel["rollout_lengths_sum"] += int(rollout_steps)
        panel["rollout_passes_sum"] += int(passes_count)
        panel["rollout_shots_sum"] += int(shots_count)
        panel["terminated_rollouts"] += int(bool(done))
        panel["num_rollouts"] += 1
        terminal_key, turnover_reason = _classify_playbook_terminal_outcome(
            rollout_state,
            done=bool(done),
            shot_clock=getattr(env, "shot_clock", None),
        )
        _increment_count(panel["terminal_outcomes"], terminal_key, 1)
        if turnover_reason:
            _increment_count(panel["turnover_reasons"], turnover_reason, 1)

        if progress_queue is not None:
            try:
                progress_queue.put_nowait(1)
            except queue.Full:
                pass
            except Exception:
                pass

    return payload


@router.post("/api/batch_update_player_positions")
def batch_update_player_positions(req: BatchUpdatePositionRequest):
    """Updates positions for multiple players at once."""
    if not game_state.env or game_state.obs is None:
        raise HTTPException(status_code=400, detail="Game not initialized.")
    env = _base_env()
    if env_view(env).episode_ended:
        raise HTTPException(status_code=400, detail="Cannot move players after episode has ended.")

    updates = req.updates or []
    if not updates:
        raise HTTPException(status_code=400, detail="No position updates provided.")

    for upd in updates:
        pid = upd.player_id
        new_pos = (upd.q, upd.r)
        if pid < 0 or pid >= env.n_players:
            raise HTTPException(status_code=400, detail=f"Invalid player ID: {pid}")
        if not env._is_valid_position(*new_pos):
            raise HTTPException(status_code=400, detail=f"Position {new_pos} is out of bounds.")
        for i, pos in enumerate(env.positions):
            if i != pid and pos == new_pos:
                raise HTTPException(status_code=400, detail=f"Position {new_pos} is occupied by Player {i}.")
        env.positions[pid] = new_pos

    _rebuild_cached_obs()

    updated_state = get_ui_game_state()
    if game_state.episode_states:
        game_state.episode_states[-1] = updated_state
    return {
        "status": "success",
        "state": updated_state,
    }


@router.post("/api/update_player_position")
def update_player_position(req: UpdatePositionRequest):
    """Updates a single player's position during an ongoing episode."""
    if not game_state.env or game_state.obs is None:
        raise HTTPException(status_code=400, detail="Game not initialized.")
    env = _base_env()
    if env_view(env).episode_ended:
        raise HTTPException(status_code=400, detail="Cannot move players after episode has ended.")

    pid = req.player_id
    new_pos = (req.q, req.r)

    if pid < 0 or pid >= env.n_players:
        raise HTTPException(status_code=400, detail=f"Invalid player ID: {pid}")
    if not env._is_valid_position(*new_pos):
        raise HTTPException(status_code=400, detail=f"Position {new_pos} is out of bounds.")
    for i, pos in enumerate(env.positions):
        if i != pid and pos == new_pos:
            raise HTTPException(status_code=400, detail=f"Position {new_pos} is occupied by Player {i}.")

    env.positions[pid] = new_pos

    _rebuild_cached_obs()

    updated_state = get_ui_game_state()
    if game_state.episode_states:
        game_state.episode_states[-1] = updated_state
    return {
        "status": "success",
        "state": updated_state,
    }


@router.post("/api/update_shot_clock")
def update_shot_clock(req: UpdateShotClockRequest):
    """Adjust the shot clock by a delta (see UpdateShotClockRequest)."""
    if not game_state.env:
        raise HTTPException(status_code=400, detail="Game not initialized.")
    env = _base_env()
    try:
        delta = int(req.delta)
        current = int(getattr(env, "shot_clock", 0))
        max_val = int(getattr(env, "shot_clock_steps", current))
        new_val = current + delta
        if max_val > 0:
            new_val = max(0, min(max_val, new_val))
        else:
            new_val = max(0, new_val)
        env.shot_clock = int(new_val)
        _rebuild_cached_obs()
        return {
            "status": "success",
            "shot_clock": int(env.shot_clock),
            "state": get_ui_game_state(),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/api/set_shot_clock")
def set_shot_clock(req: UpdateShotClockRequest):
    """Backwards-compatible alias for update_shot_clock (expects delta)."""
    return update_shot_clock(req)


@router.post("/api/set_ball_holder")
def set_ball_holder(req: SetBallHolderRequest):
    """Manually set the ball holder during a live game (offense only)."""
    if not game_state.env:
        raise HTTPException(status_code=400, detail="Game not initialized.")

    env = _base_env()
    if req.player_id not in env.offense_ids:
        raise HTTPException(status_code=400, detail="Ball holder must be an offensive player.")

    env.ball_holder = int(req.player_id)
    try:
        _rebuild_cached_obs()
        updated_state = get_ui_game_state()
        if game_state.episode_states:
            game_state.episode_states[-1] = updated_state
        return {"status": "success", "state": updated_state}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to set ball holder: {e}")


@router.post("/api/set_intent_state")
def set_intent_state(req: SetIntentStateRequest):
    """Override the live offense intent state for the current possession."""
    if not game_state.env or game_state.obs is None:
        raise HTTPException(status_code=400, detail="Game not initialized.")
    env = _base_env()
    if env_view(env).episode_ended:
        raise HTTPException(status_code=400, detail="Cannot edit intent after episode has ended.")

    if not bool(getattr(env, "enable_intent_learning", False)):
        raise HTTPException(
            status_code=400,
            detail="Offense intent learning is not enabled for this environment.",
        )

    try:
        num_intents = max(1, int(getattr(env, "num_intents", 1)))
        max_age = max(0, int(getattr(env, "intent_commitment_steps", 0)))

        active = bool(req.active)
        intent_index = max(0, min(num_intents - 1, int(req.intent_index)))
        intent_age = max(0, min(max_age, int(req.intent_age)))

        if not active:
            intent_index = 0
            intent_age = 0
            intent_commitment_remaining = 0
        else:
            intent_commitment_remaining = max(0, max_age - intent_age)

        env.intent_active = active
        env.intent_index = int(intent_index)
        env.intent_age = int(intent_age)
        env.intent_commitment_remaining = int(intent_commitment_remaining)

        _rebuild_cached_obs()

        updated_state = get_ui_game_state()
        if game_state.episode_states:
            game_state.episode_states[-1] = updated_state
        return {"status": "success", "state": updated_state}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to set intent state: {e}")


@router.post("/api/capture_counterfactual_snapshot")
def capture_counterfactual_snapshot_route():
    """Capture the current env/session state as a branch point for counterfactual testing."""
    if not game_state.env or game_state.obs is None:
        raise HTTPException(status_code=400, detail="Game not initialized.")
    try:
        snapshot = capture_counterfactual_snapshot()
        updated_state = get_ui_game_state()
        if game_state.episode_states:
            game_state.episode_states[-1] = updated_state
        return {
            "status": "success",
            "state": updated_state,
            "snapshot": snapshot,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to capture snapshot: {e}")


@router.post("/api/restore_counterfactual_snapshot")
def restore_counterfactual_snapshot_route():
    """Restore the current env/session to the most recently captured counterfactual snapshot."""
    if not game_state.counterfactual_snapshot:
        raise HTTPException(status_code=400, detail="No counterfactual snapshot available.")
    try:
        snapshot = restore_counterfactual_snapshot()
        updated_state = get_ui_game_state()
        if game_state.episode_states:
            game_state.episode_states[-1] = updated_state
        return {
            "status": "success",
            "state": updated_state,
            "snapshot": snapshot,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to restore snapshot: {e}")


@router.post("/api/replay_counterfactual_snapshot")
def replay_counterfactual_snapshot_route(req: ReplayCounterfactualRequest):
    """Autoplay deterministically from the current branch state after snapshot-based edits."""
    if not game_state.counterfactual_snapshot:
        raise HTTPException(status_code=400, detail="No counterfactual snapshot available.")
    if not game_state.env or game_state.obs is None:
        raise HTTPException(status_code=400, detail="Game not initialized.")

    from app.backend.routes.lifecycle_routes import step as step_route

    try:
        max_steps = max(1, int(req.max_steps))
        states = [get_ui_game_state()]
        done = bool(states[-1].get("done")) or bool(env_view(game_state.env).episode_ended)
        steps_taken = 0

        while not done and steps_taken < max_steps:
            step_body = step_route(
                ActionRequest(
                    actions={},
                    player_deterministic=bool(req.player_deterministic),
                    opponent_deterministic=bool(req.opponent_deterministic),
                )
            )
            next_state = step_body.get("state")
            if not next_state:
                raise RuntimeError("Counterfactual replay step did not return a state.")
            states.append(next_state)
            steps_taken += 1
            done = bool(next_state.get("done")) or bool(env_view(game_state.env).episode_ended)

        return {
            "status": "success",
            "states": states,
            "state": states[-1],
            "steps_taken": int(steps_taken),
            "terminated": bool(done),
            "max_steps": int(max_steps),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to replay counterfactual snapshot: {e}")


@router.post("/api/playbook_analysis")
def playbook_analysis_route(req: PlaybookAnalysisRequest):
    """Run repeated snapshot/current-state rollouts and aggregate intent-conditioned spatial patterns."""
    if not game_state.env or game_state.obs is None:
        raise HTTPException(status_code=400, detail="Game not initialized.")
    if req.use_snapshot and not game_state.counterfactual_snapshot:
        raise HTTPException(status_code=400, detail="No counterfactual snapshot available.")

    from app.backend.routes.lifecycle_routes import step as step_route

    env = _base_env()
    num_intents = max(1, int(getattr(env, "num_intents", 1)))
    if not bool(getattr(env, "enable_intent_learning", False)):
        raise HTTPException(status_code=400, detail="Intent learning is not enabled for this environment.")

    raw_indices = list(req.intent_indices or [])
    if not raw_indices:
        raise HTTPException(status_code=400, detail="Provide at least one intent index.")

    intent_indices: list[int] = []
    for raw_idx in raw_indices:
        idx = int(raw_idx)
        if idx < 0 or idx >= num_intents:
            raise HTTPException(
                status_code=400,
                detail=f"Intent index {idx} is out of range for {num_intents} intents.",
            )
        if idx not in intent_indices:
            intent_indices.append(idx)

    num_rollouts = max(1, min(512, int(req.num_rollouts)))
    max_steps = max(1, min(256, int(req.max_steps)))
    run_to_end = bool(req.run_to_end)
    total_rollouts = int(num_rollouts * len(intent_indices))
    commitment_steps = max(0, int(getattr(env, "intent_commitment_steps", 0)))
    offense_ids = [int(pid) for pid in getattr(env, "offense_ids", [])]
    source_label = "snapshot" if bool(req.use_snapshot) else "current"
    training_params = getattr(game_state, "mlflow_training_params", None) or {}
    can_parallelize = bool(
        game_state.env_required_params is not None
        and game_state.unified_policy_path is not None
        and game_state.user_team is not None
        and total_rollouts > 1
    )
    num_workers = None
    if can_parallelize and total_rollouts >= 8:
        num_workers = max(2, min(mp.cpu_count(), 16, total_rollouts))

    live_restore_state = _capture_restorable_backend_state()
    original_counterfactual_snapshot = copy.deepcopy(game_state.counterfactual_snapshot)

    try:
        reset_playbook_progress(total_rollouts)
        if req.use_snapshot:
            base_state = copy.deepcopy(game_state.counterfactual_snapshot)
        else:
            base_state = _capture_restorable_backend_state()

        if req.use_snapshot:
            _restore_restorable_backend_state(base_state)
            base_ui_state = get_ui_game_state()
            _restore_restorable_backend_state(live_restore_state)
            game_state.counterfactual_snapshot = copy.deepcopy(original_counterfactual_snapshot)
        else:
            base_ui_state = get_ui_game_state()

        panel_accumulators = {
            int(intent_index): _init_playbook_panel_accumulator(offense_ids)
            for intent_index in intent_indices
        }

        if num_workers is not None:
            parallel_base_env = copy.deepcopy(getattr(base_state["env"], "unwrapped", base_state["env"]))
            rollout_specs = [
                (int(order), int(intent_index), int(np.random.randint(0, 2**31 - 1)))
                for order, intent_index in enumerate(
                    [
                        int(intent_index)
                        for intent_index in intent_indices
                        for _ in range(num_rollouts)
                    ]
                )
            ]
            target_batches = max(1, num_workers * 4)
            batch_size = max(1, (len(rollout_specs) + target_batches - 1) // target_batches)
            batches = []
            for idx in range(0, len(rollout_specs), batch_size):
                batch_specs = rollout_specs[idx : idx + batch_size]
                if not batch_specs:
                    continue
                batches.append(
                    (
                        batch_specs,
                        parallel_base_env,
                        base_ui_state,
                        str(game_state.user_team.name),
                        offense_ids,
                        int(commitment_steps),
                        int(max_steps),
                        bool(run_to_end),
                        bool(req.player_deterministic),
                        bool(req.opponent_deterministic),
                    )
                )

            ctx = mp.get_context("spawn")
            progress_queue = ctx.Queue()
            completed = 0
            payloads: list[dict] = []

            with ProcessPoolExecutor(
                max_workers=num_workers,
                mp_context=ctx,
                initializer=backend_evaluation._init_evaluation_worker,
                initargs=(
                    game_state.env_required_params,
                    game_state.env_optional_params,
                    game_state.mlflow_training_params,
                    game_state.unified_policy_path,
                    game_state.opponent_policy_path,
                    game_state.user_team.name,
                    game_state.role_flag_offense,
                    game_state.role_flag_defense,
                    "learned_sample",
                    progress_queue,
                ),
            ) as executor:
                pending = {
                    executor.submit(_run_playbook_batch_worker, batch)
                    for batch in batches
                }
                while pending:
                    done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                    while True:
                        try:
                            completed += int(progress_queue.get_nowait())
                        except queue.Empty:
                            break
                        except Exception:
                            break
                    update_playbook_progress(min(completed, total_rollouts), total_rollouts)
                    if not done:
                        continue
                    for future in done:
                        payloads.append(future.result())

                while True:
                    try:
                        completed += int(progress_queue.get_nowait())
                    except queue.Empty:
                        break
                    except Exception:
                        break

            update_playbook_progress(total_rollouts, total_rollouts)
            for payload in payloads:
                for intent_index, partial in (payload or {}).items():
                    _merge_playbook_panel_accumulator(
                        panel_accumulators[int(intent_index)],
                        partial or {},
                    )
        else:
            completed = 0
            for intent_index in intent_indices:
                panel = panel_accumulators[int(intent_index)]
                for _ in range(num_rollouts):
                    _restore_restorable_backend_state(base_state)
                    game_state.counterfactual_snapshot = copy.deepcopy(original_counterfactual_snapshot)

                    base_env = _base_env()
                    base_env.intent_active = True
                    base_env.intent_index = int(intent_index)
                    base_env.intent_age = 0
                    base_env.intent_commitment_remaining = int(commitment_steps)
                    _rebuild_cached_obs()
                    initial_state = copy.deepcopy(get_ui_game_state())
                    if panel.get("base_state") is None:
                        panel["base_state"] = copy.deepcopy(initial_state)

                    rollout_state = _build_playbook_rollout_state(game_state.env)
                    passes_count, shots_count = _count_rollout_state(
                        rollout_state,
                        offense_ids,
                        panel["player_heatmaps"],
                        panel["ball_heatmap"],
                        panel["shot_heatmap"],
                        panel["player_shot_heatmaps"],
                        panel["player_shot_stats"],
                        panel["pass_links"],
                        panel["pass_path_segments"],
                    )
                    rollout_steps = 0
                    done = bool(env_view(game_state.env).episode_ended)

                    while not done and (run_to_end or rollout_steps < max_steps):
                        step_body = step_route(
                            ActionRequest(
                                actions={},
                                player_deterministic=bool(req.player_deterministic),
                                opponent_deterministic=bool(req.opponent_deterministic),
                            )
                        )
                        next_state = step_body.get("state")
                        if not next_state:
                            raise RuntimeError("Playbook analysis step did not return a state.")
                        next_rollout_state = _build_playbook_rollout_state(
                            game_state.env,
                            next_state.get("last_action_results") or {},
                        )
                        _count_rollout_transition(
                            rollout_state,
                            next_rollout_state,
                            offense_ids,
                            panel["player_path_segments"],
                            panel["ball_path_segments"],
                        )
                        more_passes, more_shots = _count_rollout_state(
                            next_rollout_state,
                            offense_ids,
                            panel["player_heatmaps"],
                            panel["ball_heatmap"],
                            panel["shot_heatmap"],
                            panel["player_shot_heatmaps"],
                            panel["player_shot_stats"],
                            panel["pass_links"],
                            panel["pass_path_segments"],
                        )
                        passes_count += more_passes
                        shots_count += more_shots
                        rollout_steps += 1
                        done = bool(next_state.get("done")) or bool(env_view(game_state.env).episode_ended)
                        rollout_state = next_rollout_state

                    panel["rollout_lengths_sum"] += int(rollout_steps)
                    panel["rollout_passes_sum"] += int(passes_count)
                    panel["rollout_shots_sum"] += int(shots_count)
                    panel["terminated_rollouts"] += int(bool(done))
                    panel["num_rollouts"] += 1
                    terminal_key, turnover_reason = _classify_playbook_terminal_outcome(
                        rollout_state,
                        done=bool(done),
                        shot_clock=getattr(game_state.env, "shot_clock", None),
                    )
                    _increment_count(panel["terminal_outcomes"], terminal_key, 1)
                    if turnover_reason:
                        _increment_count(panel["turnover_reasons"], turnover_reason, 1)
                    completed += 1
                    update_playbook_progress(completed, total_rollouts)

        panels = []
        for intent_index in intent_indices:
            panel = panel_accumulators[int(intent_index)]
            rollouts_done = max(1, int(panel["num_rollouts"]))
            shot_stats_by_player = {
                str(pid): {
                    "attempts": int((stats or {}).get("attempts", 0) or 0),
                    "makes": int((stats or {}).get("makes", 0) or 0),
                }
                for pid, stats in (panel["player_shot_stats"] or {}).items()
            }
            total_shot_attempts = int(
                sum(stats["attempts"] for stats in shot_stats_by_player.values())
            )
            total_shot_makes = int(
                sum(stats["makes"] for stats in shot_stats_by_player.values())
            )
            terminal_outcomes = {
                str(key): int(value or 0)
                for key, value in (panel["terminal_outcomes"] or {}).items()
            }
            turnover_reasons = {
                str(key): int(value or 0)
                for key, value in (panel["turnover_reasons"] or {}).items()
            }
            panels.append(
                {
                    "intent_index": int(intent_index),
                    "num_rollouts": int(panel["num_rollouts"]),
                    "avg_steps": float(panel["rollout_lengths_sum"] / rollouts_done),
                    "avg_passes": float(panel["rollout_passes_sum"] / rollouts_done),
                    "avg_shots": float(panel["rollout_shots_sum"] / rollouts_done),
                    "terminated_rate": float(panel["terminated_rollouts"] / rollouts_done),
                    "player_heatmaps": panel["player_heatmaps"],
                    "ball_heatmap": panel["ball_heatmap"],
                    "shot_heatmap": panel["shot_heatmap"],
                    "player_shot_heatmaps": panel["player_shot_heatmaps"],
                    "shot_stats": {
                        "by_player": shot_stats_by_player,
                        "total": {
                            "attempts": total_shot_attempts,
                            "makes": total_shot_makes,
                        },
                    },
                    "terminal_outcomes": terminal_outcomes,
                    "turnover_reasons": turnover_reasons,
                    "pass_links": dict(sorted(panel["pass_links"].items())),
                    "base_state": panel["base_state"],
                    "player_path_segments": {
                        str(pid): _serialize_segment_counts(segments)
                        for pid, segments in (panel["player_path_segments"] or {}).items()
                    },
                    "ball_path_segments": _serialize_segment_counts(panel["ball_path_segments"]),
                    "pass_path_segments": _serialize_pass_segment_counts(panel["pass_path_segments"]),
                }
            )

        return jsonable_encoder({
            "status": "success",
            "source": source_label,
            "num_rollouts": int(num_rollouts),
            "total_rollouts": int(total_rollouts),
            "max_steps": int(max_steps),
            "run_to_end": bool(run_to_end),
            "offense_ids": offense_ids,
            "used_parallel": bool(num_workers is not None),
            "num_workers": int(num_workers or 1),
            "panels": panels,
        }, custom_encoder=_NUMPY_SAFE_ENCODER)
    except HTTPException:
        raise
    except Exception as e:
        fail_playbook_progress(str(e))
        raise HTTPException(status_code=500, detail=f"Failed to run playbook analysis: {e}")
    finally:
        if get_playbook_progress().get("status") == "running":
            update_playbook_progress(total_rollouts, total_rollouts)
        _restore_restorable_backend_state(live_restore_state)
        game_state.counterfactual_snapshot = copy.deepcopy(original_counterfactual_snapshot)


@router.get("/api/playbook_progress")
def playbook_progress():
    return get_playbook_progress()


@router.post("/api/offense_skills")
def set_offense_skills(req: SetOffenseSkillsRequest):
    """Override or reset the per-offensive-player shooting percentages for the current episode."""
    if not game_state.env:
        raise HTTPException(status_code=400, detail="Game not initialized.")

    env = _base_env()
    count = env.players_per_side

    def _normalize(values: List[float] | None, name: str) -> List[float]:
        if values is None:
            raise HTTPException(status_code=400, detail=f"Missing {name} values.")
        if len(values) != count:
            raise HTTPException(
                status_code=400,
                detail=f"{name} must include {count} values (one per offensive player).",
            )
        normalized: List[float] = []
        for v in values:
            try:
                val = float(v)
            except Exception:
                raise HTTPException(status_code=400, detail=f"Invalid {name} value: {v}")
            normalized.append(val)
        return normalized

    try:
        if req.reset_to_sampled:
            env.offense_layup_pct_by_player = copy.deepcopy(game_state.sampled_offense_skills.get("layup"))
            env.offense_three_pt_pct_by_player = copy.deepcopy(game_state.sampled_offense_skills.get("three_pt"))
            env.offense_dunk_pct_by_player = copy.deepcopy(game_state.sampled_offense_skills.get("dunk"))
        else:
            if not req.skills:
                raise HTTPException(status_code=400, detail="Missing skills payload.")
            layup = _normalize(req.skills.layup, "layup")
            three_pt = _normalize(req.skills.three_pt, "three_pt")
            dunk = _normalize(req.skills.dunk, "dunk")

            env.offense_layup_pct_by_player = layup
            env.offense_three_pt_pct_by_player = three_pt
            env.offense_dunk_pct_by_player = dunk

        return {
            "status": "success",
            "state": get_ui_game_state(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to set offense skills: {e}")


@router.post("/api/update_pass_target_strategy")
def set_pass_target_strategy(req: SetPassTargetStrategyRequest):
    """Update pass target strategy (admin)."""
    if not game_state.env:
        raise HTTPException(status_code=400, detail="Game not initialized.")
    try:
        strategy = req.strategy
        env = _base_env()
        env.pass_target_strategy = strategy
        return {"status": "success", "pass_target_strategy": strategy}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update pass target strategy: {e}")


@router.post("/api/set_pass_logit_bias")
def set_pass_logit_bias(req: SetPassLogitBiasRequest):
    """Update pass logit bias for the active policies."""
    if game_state.unified_policy is None:
        raise HTTPException(status_code=400, detail="Game not initialized.")
    try:
        bias = float(req.bias) if req.bias is not None else 0.0
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid pass logit bias: {req.bias}")

    def _apply(policy):
        policy_obj = getattr(policy, "policy", None)
        if policy_obj is None:
            return
        if hasattr(policy_obj, "set_pass_logit_bias"):
            policy_obj.set_pass_logit_bias(bias)
        else:
            try:
                setattr(policy_obj, "pass_logit_bias", float(bias))
            except Exception:
                pass

    _apply(game_state.unified_policy)
    if game_state.defense_policy is not None:
        _apply(game_state.defense_policy)

    return {
        "status": "success",
        "state": get_ui_game_state(),
    }


def _set_pressure_params_impl(
    req: SetPressureParamsRequest, forced_scope: str | None = None
):
    """Internal implementation for pressure/interception parameter updates."""
    if not game_state.env:
        raise HTTPException(status_code=400, detail="Game not initialized.")
    env = _base_env()

    mlflow_defaults = (
        game_state.mlflow_env_optional_defaults
        or game_state.env_optional_params
        or {}
    )
    all_param_keys = {
        "three_pt_extra_hex_decay",
        "shot_pressure_enabled",
        "shot_pressure_max",
        "shot_pressure_lambda",
        "shot_pressure_arc_degrees",
        "base_steal_rate",
        "steal_perp_decay",
        "steal_distance_factor",
        "steal_position_weight_min",
        "defender_pressure_distance",
        "defender_pressure_turnover_chance",
        "defender_pressure_decay_lambda",
    }
    scoped_param_keys = {
        "all": set(all_param_keys),
        "shot_pressure": {
            "three_pt_extra_hex_decay",
            "shot_pressure_enabled",
            "shot_pressure_max",
            "shot_pressure_lambda",
            "shot_pressure_arc_degrees",
        },
        "pass_interception": {
            "base_steal_rate",
            "steal_perp_decay",
            "steal_distance_factor",
            "steal_position_weight_min",
        },
        "defender_pressure": {
            "defender_pressure_distance",
            "defender_pressure_turnover_chance",
            "defender_pressure_decay_lambda",
        },
    }
    requested_scope = (
        str(forced_scope).strip().lower()
        if forced_scope is not None
        else str(req.scope or req.reset_group or "").strip().lower()
    )
    if requested_scope and requested_scope not in scoped_param_keys:
        raise HTTPException(status_code=400, detail=f"Unknown scope: {requested_scope}")
    active_scope = requested_scope or ""

    if req.reset_to_mlflow_defaults:
        def _default_value(key: str, fallback):
            val = mlflow_defaults.get(key, fallback)
            return fallback if val is None else val

        default_payload = {
            "three_pt_extra_hex_decay": _default_value(
                "three_pt_extra_hex_decay", getattr(env, "three_pt_extra_hex_decay", 0.05)
            ),
            "shot_pressure_enabled": _default_value(
                "shot_pressure_enabled", getattr(env, "shot_pressure_enabled", True)
            ),
            "shot_pressure_max": _default_value(
                "shot_pressure_max", getattr(env, "shot_pressure_max", 0.5)
            ),
            "shot_pressure_lambda": _default_value(
                "shot_pressure_lambda", getattr(env, "shot_pressure_lambda", 1.0)
            ),
            "shot_pressure_arc_degrees": _default_value(
                "shot_pressure_arc_degrees", getattr(env, "shot_pressure_arc_degrees", 60.0)
            ),
            "base_steal_rate": _default_value(
                "base_steal_rate", getattr(env, "base_steal_rate", 0.35)
            ),
            "steal_perp_decay": _default_value(
                "steal_perp_decay", getattr(env, "steal_perp_decay", 1.5)
            ),
            "steal_distance_factor": _default_value(
                "steal_distance_factor", getattr(env, "steal_distance_factor", 0.08)
            ),
            "steal_position_weight_min": _default_value(
                "steal_position_weight_min", getattr(env, "steal_position_weight_min", 0.3)
            ),
            "defender_pressure_distance": _default_value(
                "defender_pressure_distance", getattr(env, "defender_pressure_distance", 1)
            ),
            "defender_pressure_turnover_chance": _default_value(
                "defender_pressure_turnover_chance",
                getattr(env, "defender_pressure_turnover_chance", 0.05),
            ),
            "defender_pressure_decay_lambda": _default_value(
                "defender_pressure_decay_lambda",
                getattr(env, "defender_pressure_decay_lambda", 1.0),
            ),
        }
        if active_scope:
            selected_keys = set(scoped_param_keys[active_scope])
        elif req.reset_keys is not None:
            selected_keys = {str(k) for k in req.reset_keys}
        else:
            raise HTTPException(
                status_code=400,
                detail="reset_to_mlflow_defaults requires scope/reset_group or reset_keys",
            )

        unknown_keys = sorted(selected_keys - set(all_param_keys))
        if unknown_keys:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown reset_keys: {', '.join(unknown_keys)}",
            )
        payload = {k: v for k, v in default_payload.items() if k in selected_keys}
    else:
        payload = {
            "three_pt_extra_hex_decay": req.three_pt_extra_hex_decay,
            "shot_pressure_enabled": req.shot_pressure_enabled,
            "shot_pressure_max": req.shot_pressure_max,
            "shot_pressure_lambda": req.shot_pressure_lambda,
            "shot_pressure_arc_degrees": req.shot_pressure_arc_degrees,
            "base_steal_rate": req.base_steal_rate,
            "steal_perp_decay": req.steal_perp_decay,
            "steal_distance_factor": req.steal_distance_factor,
            "steal_position_weight_min": req.steal_position_weight_min,
            "defender_pressure_distance": req.defender_pressure_distance,
            "defender_pressure_turnover_chance": req.defender_pressure_turnover_chance,
            "defender_pressure_decay_lambda": req.defender_pressure_decay_lambda,
        }
        payload = {k: v for k, v in payload.items() if v is not None}
        if active_scope and active_scope != "all":
            allowed = scoped_param_keys[active_scope]
            payload = {k: v for k, v in payload.items() if k in allowed}

    if not payload:
        return {
            "status": "no_change",
            "updated_keys": [],
            "applied_scope": active_scope or "all",
            "state": get_ui_game_state(),
        }

    def _as_bool(v, key: str) -> bool:
        if isinstance(v, bool):
            return v
        if v in (0, 1):
            return bool(v)
        raise HTTPException(status_code=400, detail=f"{key} must be a boolean.")

    def _as_float(v, key: str) -> float:
        try:
            return float(v)
        except Exception:
            raise HTTPException(status_code=400, detail=f"{key} must be a number.")

    def _as_int(v, key: str) -> int:
        try:
            return int(v)
        except Exception:
            raise HTTPException(status_code=400, detail=f"{key} must be an integer.")

    def _validate_range(v: float, key: str, lo: float, hi: float) -> float:
        if v < lo or v > hi:
            raise HTTPException(
                status_code=400,
                detail=f"{key} must be between {lo} and {hi}.",
            )
        return v

    def _validate_min(v: float, key: str, lo: float) -> float:
        if v < lo:
            raise HTTPException(
                status_code=400,
                detail=f"{key} must be >= {lo}.",
            )
        return v

    normalized = {}
    if "three_pt_extra_hex_decay" in payload:
        normalized["three_pt_extra_hex_decay"] = _validate_range(
            _as_float(payload["three_pt_extra_hex_decay"], "three_pt_extra_hex_decay"),
            "three_pt_extra_hex_decay",
            0.0,
            1.0,
        )

    if "shot_pressure_enabled" in payload:
        normalized["shot_pressure_enabled"] = _as_bool(
            payload["shot_pressure_enabled"], "shot_pressure_enabled"
        )
    if "shot_pressure_max" in payload:
        normalized["shot_pressure_max"] = _validate_range(
            _as_float(payload["shot_pressure_max"], "shot_pressure_max"),
            "shot_pressure_max",
            0.0,
            1.0,
        )
    if "shot_pressure_lambda" in payload:
        normalized["shot_pressure_lambda"] = _validate_min(
            _as_float(payload["shot_pressure_lambda"], "shot_pressure_lambda"),
            "shot_pressure_lambda",
            0.0,
        )
    if "shot_pressure_arc_degrees" in payload:
        normalized["shot_pressure_arc_degrees"] = _validate_range(
            _as_float(payload["shot_pressure_arc_degrees"], "shot_pressure_arc_degrees"),
            "shot_pressure_arc_degrees",
            0.0,
            360.0,
        )

    if "base_steal_rate" in payload:
        normalized["base_steal_rate"] = _validate_range(
            _as_float(payload["base_steal_rate"], "base_steal_rate"),
            "base_steal_rate",
            0.0,
            1.0,
        )
    if "steal_perp_decay" in payload:
        normalized["steal_perp_decay"] = _validate_min(
            _as_float(payload["steal_perp_decay"], "steal_perp_decay"),
            "steal_perp_decay",
            0.0,
        )
    if "steal_distance_factor" in payload:
        normalized["steal_distance_factor"] = _validate_min(
            _as_float(payload["steal_distance_factor"], "steal_distance_factor"),
            "steal_distance_factor",
            0.0,
        )
    if "steal_position_weight_min" in payload:
        normalized["steal_position_weight_min"] = _validate_range(
            _as_float(payload["steal_position_weight_min"], "steal_position_weight_min"),
            "steal_position_weight_min",
            0.0,
            1.0,
        )

    if "defender_pressure_distance" in payload:
        normalized["defender_pressure_distance"] = _as_int(
            payload["defender_pressure_distance"], "defender_pressure_distance"
        )
        if normalized["defender_pressure_distance"] < 0:
            raise HTTPException(
                status_code=400,
                detail="defender_pressure_distance must be >= 0.",
            )
    if "defender_pressure_turnover_chance" in payload:
        normalized["defender_pressure_turnover_chance"] = _validate_range(
            _as_float(
                payload["defender_pressure_turnover_chance"],
                "defender_pressure_turnover_chance",
            ),
            "defender_pressure_turnover_chance",
            0.0,
            1.0,
        )
    if "defender_pressure_decay_lambda" in payload:
        normalized["defender_pressure_decay_lambda"] = _validate_min(
            _as_float(
                payload["defender_pressure_decay_lambda"],
                "defender_pressure_decay_lambda",
            ),
            "defender_pressure_decay_lambda",
            0.0,
        )

    try:
        for key, val in normalized.items():
            setattr(env, key, val)
        if "shot_pressure_arc_degrees" in normalized:
            env.shot_pressure_arc_rad = math.radians(
                float(normalized["shot_pressure_arc_degrees"])
            )

        if game_state.env_optional_params is None:
            game_state.env_optional_params = {}
        game_state.env_optional_params.update(normalized)

        _rebuild_cached_obs()

        updated_state = get_ui_game_state()
        if game_state.episode_states:
            game_state.episode_states[-1] = updated_state
        return {
            "status": "success",
            "updated_keys": sorted(normalized.keys()),
            "applied_scope": active_scope or "all",
            "state": updated_state,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update pressure/interception parameters: {e}",
        )


@router.post("/api/set_pressure_params")
def set_pressure_params(req: SetPressureParamsRequest):
    """Backward-compatible catchall endpoint for pressure/interception settings."""
    return _set_pressure_params_impl(req)


@router.post("/api/set_shot_pressure_params")
def set_shot_pressure_params(req: SetPressureParamsRequest):
    """Update only shot-pressure and shot-distance-decay parameters."""
    return _set_pressure_params_impl(req, forced_scope="shot_pressure")


@router.post("/api/set_pass_interception_params")
def set_pass_interception_params(req: SetPressureParamsRequest):
    """Update only pass-interception parameters."""
    return _set_pressure_params_impl(req, forced_scope="pass_interception")


@router.post("/api/set_defender_pressure_params")
def set_defender_pressure_params(req: SetPressureParamsRequest):
    """Update only defender turnover-pressure parameters."""
    return _set_pressure_params_impl(req, forced_scope="defender_pressure")


@router.post("/api/swap_policies")
def swap_policies(req: SwapPoliciesRequest):
    """Swap the active PPO policies without resetting the environment."""
    if not game_state.env or game_state.obs is None:
        raise HTTPException(status_code=400, detail="Game not initialized.")
    if not game_state.run_id:
        raise HTTPException(status_code=400, detail="No MLflow run associated with current game.")

    requested_user_policy = req.user_policy_name
    requested_opponent_policy = req.opponent_policy_name

    if requested_user_policy is None and requested_opponent_policy is None:
        raise HTTPException(status_code=400, detail="No policy requested for swap.")

    client = mlflow.tracking.MlflowClient()
    custom_objects = {
        "PassBiasDualCriticPolicy": PassBiasDualCriticPolicy,
        "PassBiasMultiInputPolicy": PassBiasMultiInputPolicy,
        "SetAttentionDualCriticPolicy": SetAttentionDualCriticPolicy,
        "SetAttentionExtractor": SetAttentionExtractor,
    }

    def _apply_pass_mode(policy_obj) -> None:
        policy = getattr(policy_obj, "policy", None)
        if policy is None:
            return
        mode_value = str(getattr(game_state.env, "pass_mode", "directional"))
        if hasattr(policy, "set_pass_mode"):
            try:
                policy.set_pass_mode(mode_value)
            except Exception:
                pass

    policies_changed = False

    if requested_user_policy is not None and requested_user_policy != game_state.unified_policy_key:
        try:
            user_path = get_unified_policy_path(client, game_state.run_id, requested_user_policy)
            game_state.unified_policy = PPO.load(user_path, custom_objects=custom_objects)
            _apply_pass_mode(game_state.unified_policy)
            game_state.unified_policy_key = os.path.basename(user_path)
            game_state.unified_policy_path = user_path
            try:
                counts = _compute_param_counts_from_policy(game_state.unified_policy)
                if counts:
                    if game_state.mlflow_training_params is None:
                        game_state.mlflow_training_params = {}
                    game_state.mlflow_training_params["param_counts"] = counts
            except Exception:
                pass
            policies_changed = True
        except Exception as e:
            logger.exception("swap_policies: failed loading user policy %s", requested_user_policy)
            raise HTTPException(status_code=500, detail=f"Failed to load user policy '{requested_user_policy}': {e}")

    if requested_opponent_policy is not None:
        if requested_opponent_policy == "":
            if game_state.defense_policy is not None or game_state.opponent_unified_policy_key is not None:
                game_state.defense_policy = None
                game_state.opponent_unified_policy_key = None
                game_state.opponent_policy_path = None
                policies_changed = True
        elif requested_opponent_policy != game_state.opponent_unified_policy_key:
            try:
                opp_path = get_unified_policy_path(client, game_state.run_id, requested_opponent_policy)
                game_state.defense_policy = PPO.load(opp_path, custom_objects=custom_objects)
                _apply_pass_mode(game_state.defense_policy)
                game_state.opponent_unified_policy_key = os.path.basename(opp_path)
                game_state.opponent_policy_path = opp_path
                try:
                    counts = _compute_param_counts_from_policy(game_state.unified_policy)
                    if counts:
                        if game_state.mlflow_training_params is None:
                            game_state.mlflow_training_params = {}
                        game_state.mlflow_training_params["param_counts"] = counts
                except Exception:
                    pass
                policies_changed = True
            except Exception as e:
                logger.exception("swap_policies: failed loading opponent policy %s", requested_opponent_policy)
                raise HTTPException(status_code=500, detail=f"Failed to load opponent policy '{requested_opponent_policy}': {e}")

    if not policies_changed:
        return {
            "status": "no_change",
            "state": get_ui_game_state(),
        }

    updated_state = get_ui_game_state()
    if game_state.episode_states:
        game_state.episode_states[-1] = updated_state

    logger.info(
        "swap_policies success: user=%s opponent=%s",
        game_state.unified_policy_key,
        game_state.opponent_unified_policy_key,
    )
    return {"status": "success", "state": updated_state}
