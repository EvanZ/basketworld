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
}


def _to_jsonable(value):
    """Force NumPy-safe, JSON-serializable payloads for FastAPI responses."""
    return jsonable_encoder(value, custom_encoder=_NUMPY_SAFE_ENCODER)


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
    randomize_offense_perm = bool(getattr(request, "randomize_offense_permutation", False))
    intent_selection_mode = str(getattr(request, "intent_selection_mode", "learned_sample") or "learned_sample")

    # Log shot clock configuration before evaluation
    print(f"[Evaluation] Starting {num_episodes} episodes (parallel)")
    print("[Evaluation] Configuration:")
    print(f"  - Player deterministic: {player_deterministic}")
    print(f"  - Opponent deterministic: {opponent_deterministic}")
    print(f"  - Intent selection mode: {intent_selection_mode}")
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
            optional_params=game_state.env_optional_params,
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
        raw_shots = raw_results.get("shot_accumulator")
        if isinstance(raw_shots, dict):
            shot_accumulator = raw_shots
        episode_payload = raw_results.get("results", [])
    else:
        per_player_stats = {}
        per_intent_stats = {}
        eval_diagnostics = {}
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
        "per_player_stats": per_player_stats,
        "per_intent_stats": per_intent_stats,
        "eval_diagnostics": eval_diagnostics,
    })


@router.get("/api/evaluation_progress")
def evaluation_progress():
    return _to_jsonable(get_evaluation_progress())
