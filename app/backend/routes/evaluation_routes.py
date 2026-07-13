import copy
import time
import multiprocessing as mp

from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
import numpy as np
from basketworld.envs.basketworld_env_v2 import Team

from app.backend.evaluation import (
    pass_steal_preview as eval_pass_steal_preview,
    run_evaluation as eval_run_evaluation,
    validate_custom_eval_setup as eval_validate_custom_eval_setup,
)
from app.backend.schemas import EvaluationRequest, PassStealPreviewRequest
from app.backend.state import (
    game_state,
    get_ui_game_state,
    reset_evaluation_progress,
    update_evaluation_progress,
    fail_evaluation_progress,
    get_evaluation_progress,
)


router = APIRouter()


_NUMPY_SAFE_ENCODER = {
    np.integer: int,
    np.floating: float,
    np.bool_: bool,
    np.ndarray: lambda arr: arr.tolist(),
}


def _normalize_jsonable(value):
    """Convert common scientific-Python containers before FastAPI encoding."""
    if isinstance(value, np.ndarray):
        return [_normalize_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    value_module = getattr(type(value), "__module__", "")
    if value_module.startswith(("jax", "jaxlib")) and hasattr(value, "tolist"):
        return _normalize_jsonable(value.tolist())
    if isinstance(value, dict):
        normalized = {}
        for key, item in value.items():
            if isinstance(key, np.generic):
                key = key.item()
            if not isinstance(key, (str, int, float, bool, type(None))):
                key = str(key)
            normalized[key] = _normalize_jsonable(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize_jsonable(item) for item in value]
    return value


def _to_jsonable(value):
    """Force NumPy-safe, JSON-serializable payloads for FastAPI responses."""
    return jsonable_encoder(
        _normalize_jsonable(value), custom_encoder=_NUMPY_SAFE_ENCODER
    )


def _clip_probability(value, *, default: float) -> float:
    try:
        numeric = float(default if value is None else value)
    except (TypeError, ValueError):
        numeric = float(default)
    return float(max(0.0, min(1.0, numeric)))


_EVAL_ENV_OVERRIDE_KEYS = {
    "rebound_winner_distance_weight",
    "rebound_basket_position_weight",
    "rebound_winner_temperature",
    "rebound_skill_std",
    "rebound_skill_sampling_mode",
    "rebound_skill_high",
    "rebound_skill_low",
    "rebound_skill_weight",
    "rebound_contest_mode",
    "rebound_contest_radius",
}


def _coerce_eval_env_override(key: str, value):
    if key == "rebound_contest_mode":
        raw_value = value
        if isinstance(raw_value, dict):
            raw_value = raw_value.get("value", raw_value.get("mode", raw_value.get("label")))
        if isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool):
            return "local_contest" if int(raw_value) == 1 else "global_contest"
        mode = str(raw_value or "global_contest").strip().lower().replace("-", "_").replace(" ", "_")
        if mode in {"0", "global", "global_contest", "global_softmax"}:
            return "global_contest"
        if mode in {"1", "local", "local_contest"}:
            return "local_contest"
        raise HTTPException(
            status_code=400,
            detail="rebound_contest_mode must be 'global_contest' or 'local_contest'.",
        )
    if key == "rebound_skill_sampling_mode":
        raw_value = value
        if isinstance(raw_value, dict):
            raw_value = raw_value.get("value", raw_value.get("mode", raw_value.get("label")))
        mode = str(raw_value or "gaussian").strip().lower().replace("-", "_").replace(" ", "_")
        if mode in {"gaussian", "normal"}:
            return "gaussian"
        if mode in {"one_high", "one_high_per_team", "specialist", "specialist_per_team"}:
            return "one_high_per_team"
        raise HTTPException(
            status_code=400,
            detail="rebound_skill_sampling_mode must be 'gaussian' or 'one_high_per_team'.",
        )
    if key == "rebound_contest_radius":
        try:
            return max(0, int(value))
        except Exception:
            raise HTTPException(status_code=400, detail=f"{key} must be an integer.")
    if key in {"rebound_skill_high", "rebound_skill_low"}:
        try:
            return float(value)
        except Exception:
            raise HTTPException(status_code=400, detail=f"{key} must be numeric.")
    minimum = 1.0e-6 if key == "rebound_winner_temperature" else 0.0
    try:
        return max(minimum, float(value))
    except Exception:
        raise HTTPException(status_code=400, detail=f"{key} must be numeric.")


def _request_eval_env_overrides(request: EvaluationRequest) -> dict:
    raw = getattr(request, "env_overrides", None) or {}
    if not isinstance(raw, dict):
        return {}
    normalized = {}
    for key, value in raw.items():
        if key not in _EVAL_ENV_OVERRIDE_KEYS or value is None:
            continue
        normalized[key] = _coerce_eval_env_override(key, value)
    return normalized


def _build_evaluation_optional_params(request: EvaluationRequest) -> tuple[dict, dict]:
    """Apply eval-only overrides without mutating session metadata."""
    optional_params = copy.deepcopy(game_state.env_optional_params or {})
    eval_env_overrides = _request_eval_env_overrides(request)
    if eval_env_overrides:
        optional_params.update(eval_env_overrides)
    template_library = copy.deepcopy(
        getattr(game_state, "mlflow_start_template_library", None)
    )
    mode = str(getattr(request, "start_template_mode", "checkpoint") or "checkpoint")
    if mode not in {"checkpoint", "enabled", "disabled"}:
        mode = "checkpoint"

    diagnostics = {
        "start_template_mode": mode,
        "start_template_library_available": bool(template_library),
        "start_template_source": getattr(game_state, "start_template_library_source", None),
        "env_overrides": copy.deepcopy(eval_env_overrides),
    }

    if mode == "disabled":
        optional_params["start_template_enabled"] = False
        optional_params.pop("start_template_library", None)
        diagnostics["start_template_enabled"] = False
        return optional_params, diagnostics

    if mode == "enabled":
        if not template_library:
            raise HTTPException(
                status_code=400,
                detail="No start-template library is loaded for this UI session.",
            )
        optional_params["start_template_enabled"] = True
        optional_params["start_template_library"] = template_library
        optional_params["start_template_prob"] = _clip_probability(
            getattr(request, "start_template_prob", None),
            default=float(optional_params.get("start_template_prob", 1.0) or 1.0),
        )
        optional_params["start_template_jitter_scale"] = max(
            0.0,
            float(
                getattr(request, "start_template_jitter_scale", None)
                if getattr(request, "start_template_jitter_scale", None) is not None
                else optional_params.get("start_template_jitter_scale", 1.0)
            ),
        )
        optional_params["start_template_mirror_prob"] = _clip_probability(
            getattr(request, "start_template_mirror_prob", None),
            default=float(optional_params.get("start_template_mirror_prob", 0.0) or 0.0),
        )
        diagnostics["start_template_enabled"] = True
        diagnostics["start_template_prob"] = optional_params["start_template_prob"]
        diagnostics["start_template_jitter_scale"] = optional_params["start_template_jitter_scale"]
        diagnostics["start_template_mirror_prob"] = optional_params["start_template_mirror_prob"]
        return optional_params, diagnostics

    if bool(optional_params.get("start_template_enabled", False)) and template_library:
        optional_params["start_template_library"] = template_library
        diagnostics["start_template_enabled"] = True
        diagnostics["start_template_prob"] = optional_params.get("start_template_prob")
        diagnostics["start_template_jitter_scale"] = optional_params.get(
            "start_template_jitter_scale"
        )
        diagnostics["start_template_mirror_prob"] = optional_params.get(
            "start_template_mirror_prob"
        )
    else:
        optional_params.pop("start_template_library", None)
        diagnostics["start_template_enabled"] = bool(
            optional_params.get("start_template_enabled", False)
        )
    return optional_params, diagnostics


@router.post("/api/pass_steal_preview")
def pass_steal_preview(req: PassStealPreviewRequest):
    """Return pass steal probabilities for a hypothetical placement (positions + ball holder)."""
    if not game_state.env:
        raise HTTPException(status_code=400, detail="Game not initialized.")
    try:
        result = eval_pass_steal_preview(game_state.env, req.positions, req.ball_holder)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compute pass steal preview: {e}")


@router.post("/api/run_evaluation")
def run_evaluation(request: EvaluationRequest):
    """Run N episodes of self-play for evaluation purposes."""
    if not game_state.env:
        raise HTTPException(
            status_code=400, detail="Game not initialized. Call /api/init_game first."
        )

    if not game_state.unified_policy:
        raise HTTPException(
            status_code=400, detail="Unified policy required for evaluation."
        )

    if game_state.env_required_params is None or game_state.unified_policy_path is None:
        raise HTTPException(
            status_code=400,
            detail="Missing environment parameters. Please re-initialize the game with /api/init_game.",
        )

    num_episodes = max(1, min(request.num_episodes, 1000000))
    player_deterministic = request.player_deterministic
    opponent_deterministic = request.opponent_deterministic
    custom_setup = eval_validate_custom_eval_setup(request.custom_setup, game_state.env)
    if custom_setup.get("rebound_skills") is not None:
        rebound_skill_values = [float(v) for v in custom_setup.get("rebound_skills") or []]
        print(
            "[Evaluation] Custom rebound skills received: "
            f"count={len(rebound_skill_values)} "
            f"positive={sum(1 for value in rebound_skill_values if value > 0.0)} "
            f"sum={sum(rebound_skill_values):.3f} "
            f"values={rebound_skill_values}"
        )
    else:
        sampling = custom_setup.get("rebound_skill_sampling") if isinstance(custom_setup, dict) else None
        if sampling:
            print(f"[Evaluation] Custom rebound skill sampling received: {sampling}")
        else:
            print("[Evaluation] Custom rebound skills received: none")
    randomize_offense_perm = bool(getattr(request, "randomize_offense_permutation", False))
    intent_selection_mode = str(getattr(request, "intent_selection_mode", "learned_sample") or "learned_sample")
    eval_optional_params, eval_template_diagnostics = _build_evaluation_optional_params(request)

    # Log shot clock configuration before evaluation
    print(f"[Evaluation] Starting {num_episodes} episodes (parallel)")
    print("[Evaluation] Configuration:")
    print(f"  - Player deterministic: {player_deterministic}")
    print(f"  - Opponent deterministic: {opponent_deterministic}")
    print(f"  - Intent selection mode: {intent_selection_mode}")
    print(f"  - Start-template eval mode: {eval_template_diagnostics.get('start_template_mode')}")
    print(f"  - Start-template enabled: {eval_template_diagnostics.get('start_template_enabled')}")
    print(f"  - Using opponent policy: {game_state.defense_policy is not None}")
    print(f"  - User team: {game_state.user_team.name}")
    print(f"  - Unified policy (user): {game_state.unified_policy_key}")
    print(
        f"  - Opponent policy: {game_state.opponent_unified_policy_key or 'same as unified'}"
    )
    print(f"  - shot_clock (max): {game_state.env.shot_clock_steps}")
    print(f"  - min_shot_clock: {game_state.env.min_shot_clock}")
    print(
        f"  - Each episode starts with random shot clock in range: [{game_state.env.min_shot_clock}, {game_state.env.shot_clock_steps}] steps"
    )

    # Log policy assignment to teams
    if game_state.user_team == Team.OFFENSE:
        print("\n[Policy Assignment]")
        print(f"  - OFFENSE: {game_state.unified_policy_key} (user policy)")
        print(
            f"  - DEFENSE: {game_state.opponent_unified_policy_key or game_state.unified_policy_key} (opponent policy)"
        )
    else:
        print("\n[Policy Assignment]")
        print(
            f"  - OFFENSE: {game_state.opponent_unified_policy_key or game_state.unified_policy_key} (opponent policy)"
        )
        print(f"  - DEFENSE: {game_state.unified_policy_key} (user policy)")

    start_time = time.time()
    shot_accumulator: dict[str, list[int]] = {}
    rebound_accumulator: dict[str, dict] = {}

    PARALLEL_THRESHOLD = 1000
    num_workers = None
    if num_episodes >= PARALLEL_THRESHOLD:
        # Use up to 16 cores (or available CPU count) but not more than num_episodes
        num_workers = max(2, min(mp.cpu_count(), 16, num_episodes))

    try:
        reset_evaluation_progress(num_episodes)
        raw_results = eval_run_evaluation(
            num_episodes=num_episodes,
            player_deterministic=player_deterministic,
            opponent_deterministic=opponent_deterministic,
            required_params=game_state.env_required_params,
            optional_params=eval_optional_params,
            training_params=game_state.mlflow_training_params,
            unified_policy_path=game_state.unified_policy_path,
            opponent_policy_path=game_state.opponent_policy_path,
            user_team_name=game_state.user_team.name,
            role_flag_offense=game_state.role_flag_offense,
            role_flag_defense=game_state.role_flag_defense,
            shot_accumulator=shot_accumulator,
            custom_setup=custom_setup,
            randomize_offense_permutation=randomize_offense_perm,
            intent_selection_mode=intent_selection_mode,
            num_workers=num_workers,
            progress_callback=update_evaluation_progress,
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        fail_evaluation_progress(str(e))
        raise HTTPException(status_code=500, detail=f"Failed to run evaluation: {e}")

    if isinstance(raw_results, dict):
        per_player_stats = raw_results.get("per_player_stats", {}) or {}
        per_intent_stats = raw_results.get("per_intent_stats", {}) or {}
        eval_diagnostics = raw_results.get("eval_diagnostics", {}) or {}
        eval_diagnostics["start_template_eval"] = eval_template_diagnostics
        try:
            native_summary = eval_diagnostics.get("jax_native_summary") or {}
            value_diag = native_summary.get("value_diagnostics") or eval_diagnostics.get("value_diagnostics") or {}
            print(
                "[Evaluation] Value diagnostics response: "
                f"native={bool(native_summary)} "
                f"samples={int(value_diag.get('sample_count', 0) or 0)} "
                f"Vo={float(value_diag.get('offense_value_mean', 0.0) or 0.0):.3f} "
                f"Vd={float(value_diag.get('defense_value_mean', 0.0) or 0.0):.3f} "
                f"Vo+Vd={float(value_diag.get('value_sum_mean', 0.0) or 0.0):.3f}"
            )
        except Exception:
            pass
        raw_shots = raw_results.get("shot_accumulator")
        if isinstance(raw_shots, dict):
            shot_accumulator = raw_shots
        raw_rebounds = raw_results.get("rebound_accumulator")
        if isinstance(raw_rebounds, dict):
            rebound_accumulator = raw_rebounds
        episode_payload = raw_results.get("results", [])
    else:
        per_player_stats = {}
        per_intent_stats = {}
        eval_diagnostics = {}
        eval_diagnostics["start_template_eval"] = eval_template_diagnostics
        episode_payload = raw_results

    elapsed_time = time.time() - start_time
    if elapsed_time > 0:
        eps_per_sec = len(episode_payload) / elapsed_time if episode_payload else 0
        print(f"[Evaluation] Completed {len(episode_payload)} episodes in {elapsed_time:.1f}s ({eps_per_sec:.1f} episodes/sec)")
    else:
        print(f"[Evaluation] Completed {len(episode_payload)} episodes.")

    # Normalize episode results to legacy shape expected by UI (final_state with last_action_results)
    episode_results = []
    for r in episode_payload or []:
        outcome_info = r.get("outcome_info", {}) if isinstance(r, dict) else {}
        final_state = {
            "last_action_results": {
                "shots": _to_jsonable(outcome_info.get("shots", {})),
                "turnovers": _to_jsonable(outcome_info.get("turnovers", [])),
                "defensive_lane_violations": _to_jsonable(
                    outcome_info.get("defensive_lane_violations", [])
                ),
                "rebounds": _to_jsonable(outcome_info.get("rebounds", [])),
                "rebound": _to_jsonable(outcome_info.get("rebound")),
            },
            "shot_clock": outcome_info.get("shot_clock", 0),
            "three_point_distance": outcome_info.get("three_point_distance", 4.0),
            "user_team_name": game_state.user_team.name if game_state.user_team else None,
            "done": True,
        }
        episode_results.append(
            {
                "episode": r.get("episode") if isinstance(r, dict) else None,
                "intent_index": r.get("intent_index") if isinstance(r, dict) else None,
                "final_state": final_state,
                "steps": r.get("steps") if isinstance(r, dict) else None,
                "episode_rewards": r.get("episode_rewards") if isinstance(r, dict) else None,
            }
        )

    current_game_state = get_ui_game_state()
    game_state.episode_states = []
    game_state.frames = []

    try:
        sorted_items = sorted(shot_accumulator.items(), key=lambda kv: kv[0])
        print("[Evaluation] Shot location totals (q,r -> (FGA, FGM)):")
        if not sorted_items:
            print("  (no shots recorded during evaluation)")
        else:
            for loc, vals in sorted_items:
                att, mk = vals
                print(f"  {loc}: ({att}, {mk})")
    except Exception:
        pass

    update_evaluation_progress(len(episode_results), len(episode_results))

    return _to_jsonable({
        "status": "success",
        "num_episodes": len(episode_results),
        "results": episode_results,
        "current_state": current_game_state,
        "shot_accumulator": shot_accumulator,
        "rebound_accumulator": rebound_accumulator,
        "per_player_stats": per_player_stats,
        "per_intent_stats": per_intent_stats,
        "eval_diagnostics": eval_diagnostics,
    })


@router.get("/api/evaluation_progress")
def evaluation_progress():
    return _to_jsonable(get_evaluation_progress())
