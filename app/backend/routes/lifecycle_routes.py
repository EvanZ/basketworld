import copy
import inspect
import os
import tempfile
from pathlib import Path

import basketworld
import mlflow
import numpy as np
import torch
from fastapi import APIRouter, HTTPException

from basketworld.utils.mlflow_params import (
    get_mlflow_params,
    get_mlflow_phi_shaping_params,
    get_mlflow_start_template_library,
    get_mlflow_training_params,
)
from basketworld.utils.mlflow_config import setup_mlflow
from basketworld.utils.policy_loading import load_ppo_for_inference
from basketworld.envs.basketworld_env_v2 import ActionType, Team
from basketworld.utils.wrappers import SetObservationWrapper

from app.backend.mcts import _run_mcts_advisor
from app.backend.observations import (
    _compute_q_values_for_player,
    validate_policy_observation_schema,
)
from app.backend.rollout_runtime import combine_team_actions, predict_joint_policy_actions
from app.backend.selector_runtime import (
    apply_rollout_segment_start,
    maybe_apply_rollout_multisegment_boundary,
    selector_runtime_active_for_rollout,
)
from app.backend.policies import (
    _compute_param_counts_from_policy,
    get_latest_policies_from_run,
    get_unified_policy_path,
    list_policies_from_run,
)
from app.backend.env_access import env_view
from app.backend.schemas import (
    ActionRequest,
    InitGameRequest,
    ListPoliciesRequest,
    MCTSAdviseRequest,
    TemplateBootstrapRequest,
)
from app.backend.state import (
    _capture_turn_start_snapshot,
    _role_flag_value_for_team,
    game_state,
    get_full_game_state,
    get_ui_game_state,
)
from fastapi.encoders import jsonable_encoder


router = APIRouter()


def _split_env_and_wrapper_params(optional_params: dict) -> tuple[dict, dict]:
    """Separate real env kwargs from wrapper/training-only metadata."""
    env_signature = inspect.signature(basketworld.HexagonBasketballEnv.__init__)
    valid_env_keys = {
        name for name in env_signature.parameters.keys() if name not in {"self", "render_mode"}
    }
    env_kwargs = {k: v for k, v in optional_params.items() if k in valid_env_keys}
    wrapper_kwargs = {k: v for k, v in optional_params.items() if k not in valid_env_keys}
    return env_kwargs, wrapper_kwargs


def _load_start_template_library_for_run(
    mlflow_client: mlflow.tracking.MlflowClient, run_id: str
) -> dict | None:
    """Best-effort load of a persisted start-template library artifact."""
    try:
        return get_mlflow_start_template_library(mlflow_client, run_id)
    except Exception as exc:
        print(f"[start_templates] Failed to load template library for run {run_id}: {exc}")
        return None


def _selector_runtime_active_for_app() -> bool:
    return selector_runtime_active_for_rollout(
        getattr(game_state, "mlflow_training_params", None),
        unified_policy=game_state.unified_policy,
        opponent_policy=game_state.defense_policy,
        user_team=game_state.user_team or Team.OFFENSE,
    )


def _apply_app_segment_start(*, allow_uniform_fallback: bool) -> bool:
    result = apply_rollout_segment_start(
        game_state.env,
        game_state.obs,
        training_params=getattr(game_state, "mlflow_training_params", None),
        unified_policy=game_state.unified_policy,
        opponent_policy=game_state.defense_policy,
        user_team=game_state.user_team or Team.OFFENSE,
        role_flag_offense=_role_flag_value_for_team(Team.OFFENSE),
        allow_uniform_fallback=allow_uniform_fallback,
    )
    if result.get("obs") is not None:
        game_state.obs = result["obs"]
    return bool(result.get("used_selector", False))


def _initialize_app_selector_runtime_for_episode() -> None:
    game_state.selector_segment_index = 0
    game_state.selector_last_boundary_reason = None
    _apply_app_segment_start(allow_uniform_fallback=False)


def _maybe_apply_app_multisegment_boundary(info: dict | None, done: bool) -> str | None:
    boundary = maybe_apply_rollout_multisegment_boundary(
        game_state.env,
        game_state.obs,
        info=info,
        done=bool(done),
        training_params=getattr(game_state, "mlflow_training_params", None),
        unified_policy=game_state.unified_policy,
        opponent_policy=game_state.defense_policy,
        user_team=game_state.user_team or Team.OFFENSE,
        role_flag_offense=_role_flag_value_for_team(Team.OFFENSE),
        selector_segment_index=int(getattr(game_state, "selector_segment_index", 0)),
    )
    game_state.obs = boundary.get("obs", game_state.obs)
    game_state.selector_segment_index = int(
        boundary.get("selector_segment_index", getattr(game_state, "selector_segment_index", 0))
    )
    game_state.selector_last_boundary_reason = boundary.get("reason")
    return boundary.get("reason")


@router.post("/api/list_policies")
def list_policies(request: ListPoliciesRequest):
    """Return available unified policy filenames for a run."""
    try:
        try:
            setup_mlflow(verbose=False)
        except Exception as setup_err:
            print(f"[list_policies] MLflow setup warning: {setup_err}")

        client = mlflow.tracking.MlflowClient()
        try:
            unified_paths = list_policies_from_run(client, request.run_id)
        except Exception as e:
            print(f"[list_policies] Failed to list artifacts for run {request.run_id}: {e}")
            return {"unified": []}
        if not unified_paths:
            return {"unified": []}
        return {"unified": [os.path.basename(p) for p in unified_paths]}
    except Exception as e:
        import traceback

        print(f"Error listing policies: {e}")
        traceback.print_exc()
        return {"unified": []}


@router.post("/api/init_game")
async def init_game(request: InitGameRequest):
    """Initializes a new game from an MLflow run."""
    global game_state

    from basketworld.utils.mlflow_config import setup_mlflow

    try:
        setup_mlflow()
    except Exception as e:
        print(f"Warning: MLflow setup failed: {e}")

    mlflow_client = mlflow.tracking.MlflowClient()

    run_name = request.run_name
    run_id = request.run_id
    unified_policy_name = request.unified_policy_name
    opponent_unified_policy_name = request.opponent_unified_policy_name

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            required_params, optional_params = get_mlflow_params(mlflow_client, run_id)
            mlflow_phi_params = get_mlflow_phi_shaping_params(mlflow_client, run_id)
            mlflow_training_params = get_mlflow_training_params(mlflow_client, run_id)
        except Exception as e:
            msg = str(e)
            if "RESOURCE_DOES_NOT_EXIST" in msg:
                raise HTTPException(status_code=404, detail=f"MLflow run not found: {run_id}")
            raise HTTPException(status_code=500, detail=f"Failed to fetch MLflow params: {e}")

        start_template_library = _load_start_template_library_for_run(
            mlflow_client, run_id
        )

        # Extract role_flag encoding for backward compatibility (not passed to env)
        game_state.role_flag_offense = optional_params.pop("role_flag_offense_value", 1.0)
        game_state.role_flag_defense = optional_params.pop("role_flag_defense_value", -1.0)
        optional_params.pop("role_flag_encoding_version", None)

        # Apply request overrides for optional parameters
        if request.spawn_distance is not None:
            optional_params["spawn_distance"] = request.spawn_distance
        if request.defender_spawn_distance is not None:
            optional_params["defender_spawn_distance"] = request.defender_spawn_distance
        if request.allow_dunks is not None:
            optional_params["allow_dunks"] = request.allow_dunks
        if request.dunk_pct is not None:
            optional_params["dunk_pct"] = request.dunk_pct

        try:
            unified_path = get_unified_policy_path(mlflow_client, run_id, unified_policy_name)
        except Exception as e:
            msg = str(e)
            if "RESOURCE_DOES_NOT_EXIST" in msg:
                raise HTTPException(status_code=404, detail=f"Unified policy not found for run {run_id}")
            raise HTTPException(status_code=500, detail=f"Failed to download unified policy: {e}")
        opponent_unified_path = None
        if opponent_unified_policy_name:
            try:
                opponent_unified_path = get_unified_policy_path(mlflow_client, run_id, opponent_unified_policy_name)
            except Exception as e:
                msg = str(e)
                if "RESOURCE_DOES_NOT_EXIST" in msg:
                    raise HTTPException(status_code=404, detail=f"Opponent policy not found for run {run_id}")
                raise HTTPException(status_code=500, detail=f"Failed to download opponent policy: {e}")

        game_state.unified_policy = load_ppo_for_inference(unified_path, device="cpu")
        game_state.offense_policy = None
        game_state.defense_policy = (
            load_ppo_for_inference(opponent_unified_path, device="cpu")
            if opponent_unified_path
            else None
        )
        game_state.unified_policy_key = os.path.basename(unified_path)
        game_state.opponent_unified_policy_key = os.path.basename(opponent_unified_path) if opponent_unified_path else None

        env_optional_params, wrapper_only_params = _split_env_and_wrapper_params(optional_params)
        use_set_obs = bool(wrapper_only_params.get("use_set_obs", False))
        game_state.env = basketworld.HexagonBasketballEnv(
            **required_params,
            **env_optional_params,
            render_mode="rgb_array",
        )
        if use_set_obs:
            game_state.env = SetObservationWrapper(game_state.env)

        def _apply_policy_pass_mode(policy_obj, mode_value: str) -> None:
            policy = getattr(policy_obj, "policy", None)
            if policy is None:
                return
            if hasattr(policy, "set_pass_mode"):
                try:
                    policy.set_pass_mode(mode_value)
                except Exception:
                    pass

        env_read = env_view(game_state.env)
        current_pass_mode = str(env_read.pass_mode or "directional")
        _apply_policy_pass_mode(game_state.unified_policy, current_pass_mode)
        _apply_policy_pass_mode(game_state.defense_policy, current_pass_mode)

        game_state.obs, _ = game_state.env.reset()
        try:
            game_state.obs = validate_policy_observation_schema(
                game_state.unified_policy,
                game_state.env,
                game_state.obs,
                policy_label="unified_policy",
            )
            _ = validate_policy_observation_schema(
                game_state.defense_policy,
                game_state.env,
                game_state.obs,
                policy_label="opponent_policy",
            )
        except ValueError as schema_err:
            raise HTTPException(
                status_code=400,
                detail=f"Observation schema mismatch during init_game: {schema_err}",
            )
        game_state.prev_obs = None

        game_state.env_required_params = copy.deepcopy(required_params)
        game_state.env_optional_params = copy.deepcopy(env_optional_params)
        game_state.mlflow_env_optional_defaults = copy.deepcopy(env_optional_params)
        game_state.unified_policy_path = unified_path
        game_state.opponent_policy_path = opponent_unified_path

        sampled_skills = {
            "layup": list(env_read.offense_layup_pct_by_player or []),
            "three_pt": list(env_read.offense_three_pt_pct_by_player or []),
            "dunk": list(env_read.offense_dunk_pct_by_player or []),
        }
        game_state.replay_offense_skills = copy.deepcopy(sampled_skills)
        game_state.sampled_offense_skills = copy.deepcopy(sampled_skills)

        _capture_turn_start_snapshot()

        game_state.user_team = Team[request.user_team_name.upper()]
        game_state.run_id = request.run_id
        game_state.run_name = run_name or request.run_id
        game_state.mlflow_phi_shaping_params = mlflow_phi_params
        game_state.mlflow_training_params = mlflow_training_params
        game_state.counterfactual_snapshot = None
        game_state.frames = []
        game_state.reward_history = []
        game_state.episode_rewards = {"offense": 0.0, "defense": 0.0}
        game_state.actions_log = []
        game_state.episode_states = []
        game_state.phi_log = []
        game_state.playable_session = None
        _initialize_app_selector_runtime_for_episode()

        try:
            frame = game_state.env.render()
            if frame is not None:
                game_state.frames.append(frame)
        except Exception:
            pass

        initial_state = get_ui_game_state()
        game_state.episode_states.append(initial_state)

        # Record initial phi entry (step 0) for Rewards tab calculations
        try:
            env = env_view(game_state.env)
            offense_ids = list(env.offense_ids or [])
            ball_holder_id = (
                int(env.ball_holder)
                if env.ball_holder is not None
                else (offense_ids[0] if offense_ids else -1)
            )

            ep_by_player: list[float] = []
            for pid in range(int(env.n_players or 0)):
                pos = env.positions[pid]
                dist = env._hex_distance(pos, env.basket_position)
                is_three = env.is_three_point_location(pos)
                if getattr(env, "allow_dunks", False) and dist == 0:
                    shot_value = 2.0
                else:
                    shot_value = 3.0 if is_three else 2.0
                p = float(env._calculate_shot_probability(pid, dist))
                ep_by_player.append(float(shot_value * p))

            phi_next = 0.0
            team_best_ep = 0.0
            ball_handler_ep = 0.0
            if offense_ids and ep_by_player and ball_holder_id >= 0:
                ball_ep = ep_by_player[ball_holder_id] if ball_holder_id < len(ep_by_player) else 0.0
                teammate_eps = [
                    ep_by_player[int(pid)] for pid in offense_ids if int(pid) != ball_holder_id and int(pid) < len(ep_by_player)
                ]
                if teammate_eps:
                    team_best_ep = max(max(teammate_eps), ball_ep)
                    ball_handler_ep = ball_ep
                    blend_weight = float(getattr(env, "phi_blend_weight", 0.0))
                    phi_next = (1.0 - blend_weight) * float(team_best_ep) + blend_weight * float(ball_ep)
                else:
                    phi_next = float(ball_ep)
                    team_best_ep = float(ball_ep)
                    ball_handler_ep = float(ball_ep)

            initial_phi_entry = {
                "step": 0,
                "phi_prev": 0.0,
                "phi_next": float(phi_next),
                "phi_beta": float(getattr(env, "phi_beta", 0.0)),
                "phi_r_shape": 0.0,
                "ball_handler": ball_holder_id,
                "offense_ids": offense_ids,
                "defense_ids": list(getattr(env, "defense_ids", [])),
                "shot_clock": int(getattr(env, "shot_clock", -1)),
                "ep_by_player": ep_by_player,
                "team_best_ep": float(team_best_ep),
                "ball_handler_ep": float(ball_handler_ep),
                "is_terminal": False,
            }
            game_state.phi_log.append(initial_phi_entry)
        except Exception as e:
            print(f"[init_game] Failed to record initial phi entry: {e}")

    counts = _compute_param_counts_from_policy(game_state.unified_policy)
    if counts:
        mlflow_training_params["param_counts"] = counts
    game_state.mlflow_training_params = mlflow_training_params
    game_state.mlflow_start_template_library = copy.deepcopy(start_template_library)

    return {
        "status": "success",
        "state": get_ui_game_state(),
        "seed": int(np.random.randint(0, 2**31 - 1)),
    }


@router.post("/api/template_bootstrap")
async def template_bootstrap(request: TemplateBootstrapRequest | None = None):
    """Initialize a model-free sandbox state for template authoring."""
    global game_state

    request = request or TemplateBootstrapRequest()
    players_per_side = int(max(1, request.players_per_side or 3))
    user_team_name = str(request.user_team_name or "OFFENSE").strip().upper()
    if user_team_name not in {"OFFENSE", "DEFENSE"}:
        raise HTTPException(status_code=400, detail="user_team_name must be OFFENSE or DEFENSE")

    required_params = {"players": players_per_side}
    env_optional_params = {}
    mlflow_training_params: dict = {}
    run_name = "Template Sandbox"
    source_run_id = str(request.run_id or "").strip() or None

    if source_run_id:
        try:
            setup_mlflow(verbose=False)
        except Exception as setup_err:
            print(f"[template_bootstrap] MLflow setup warning: {setup_err}")

        mlflow_client = mlflow.tracking.MlflowClient()
        try:
            required_params, optional_params = get_mlflow_params(mlflow_client, source_run_id)
            mlflow_training_params = get_mlflow_training_params(
                mlflow_client, source_run_id
            )
        except Exception as e:
            msg = str(e)
            if "RESOURCE_DOES_NOT_EXIST" in msg:
                raise HTTPException(status_code=404, detail=f"MLflow run not found: {source_run_id}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch MLflow params for template sandbox: {e}",
            )

        env_optional_params, _wrapper_only_params = _split_env_and_wrapper_params(optional_params)
        game_state.mlflow_start_template_library = _load_start_template_library_for_run(
            mlflow_client, source_run_id
        )
        if request.allow_dunks is not None:
            env_optional_params["allow_dunks"] = bool(request.allow_dunks)
        run_name = f"Template Sandbox ({source_run_id})"
    elif request.allow_dunks is not None:
        env_optional_params["allow_dunks"] = bool(request.allow_dunks)
        game_state.mlflow_start_template_library = None
    else:
        game_state.mlflow_start_template_library = None

    game_state.unified_policy = None
    game_state.offense_policy = None
    game_state.defense_policy = None
    game_state.unified_policy_key = None
    game_state.opponent_unified_policy_key = None
    game_state.unified_policy_path = None
    game_state.opponent_policy_path = None

    env_kwargs = {
        **required_params,
        **env_optional_params,
        "render_mode": "rgb_array",
    }

    game_state.env = basketworld.HexagonBasketballEnv(**env_kwargs)
    game_state.obs, _ = game_state.env.reset()
    game_state.prev_obs = None

    env_read = env_view(game_state.env)
    sampled_skills = {
        "layup": list(env_read.offense_layup_pct_by_player or []),
        "three_pt": list(env_read.offense_three_pt_pct_by_player or []),
        "dunk": list(env_read.offense_dunk_pct_by_player or []),
    }
    game_state.replay_offense_skills = copy.deepcopy(sampled_skills)
    game_state.sampled_offense_skills = copy.deepcopy(sampled_skills)

    game_state.env_required_params = copy.deepcopy(required_params)
    game_state.env_optional_params = copy.deepcopy(env_optional_params)
    if "allow_dunks" not in game_state.env_optional_params:
        game_state.env_optional_params["allow_dunks"] = bool(getattr(env_read, "allow_dunks", False))
    game_state.mlflow_env_optional_defaults = copy.deepcopy(game_state.env_optional_params)

    _capture_turn_start_snapshot()

    game_state.user_team = Team[user_team_name]
    game_state.run_id = source_run_id
    game_state.run_name = run_name
    game_state.mlflow_phi_shaping_params = None
    game_state.mlflow_training_params = mlflow_training_params
    game_state.counterfactual_snapshot = None
    game_state.frames = []
    game_state.reward_history = []
    game_state.episode_rewards = {"offense": 0.0, "defense": 0.0}
    game_state.actions_log = []
    game_state.episode_states = []
    game_state.phi_log = []
    game_state.playable_session = None
    game_state.selector_segment_index = 0
    game_state.selector_last_boundary_reason = None

    try:
        frame = game_state.env.render()
        if frame is not None:
            game_state.frames.append(frame)
    except Exception:
        pass

    initial_state = get_full_game_state(
        include_policy_probs=False,
        include_action_values=False,
        include_state_values=False,
    )
    game_state.episode_states.append(initial_state)

    return {
        "status": "success",
        "state": initial_state,
        "mode": "template_sandbox",
        "seed": int(np.random.randint(0, 2**31 - 1)),
    }


def _coerce_action_index(raw_action) -> int | None:
    """Convert incoming action representation to an action index when possible."""
    if isinstance(raw_action, str):
        token = raw_action.strip().upper()
        if token in ActionType.__members__:
            return int(ActionType[token].value)
    try:
        return int(raw_action)
    except Exception:
        return None


def _normalize_action_overrides(raw_actions, n_players: int) -> tuple[dict[int, int], dict[int, dict]]:
    """Normalize payloads into numeric action overrides and optional per-player metadata."""
    overrides: dict[int, int] = {}
    meta: dict[int, dict] = {}
    if isinstance(raw_actions, dict):
        for k, v in raw_actions.items():
            try:
                idx = int(k)
            except Exception:
                continue
            if idx < 0 or idx >= n_players:
                continue
            if isinstance(v, dict):
                action_type = str(v.get("type", "")).upper()
                if action_type == "PASS" and "target" in v:
                    action_idx = int(ActionType.PASS_E.value)
                    try:
                        target_id = int(v.get("target"))
                    except Exception:
                        target_id = v.get("target")
                    overrides[idx] = action_idx
                    meta[idx] = {"type": "PASS", "target": target_id}
                    continue
                if "action" in v:
                    v = v["action"]
                elif "selected_action" in v:
                    v = v["selected_action"]
            action_idx = _coerce_action_index(v)
            if action_idx is None:
                continue
            overrides[idx] = action_idx
    elif isinstance(raw_actions, (list, tuple, np.ndarray)):
        for idx, v in enumerate(raw_actions):
            if idx >= n_players:
                break
            action_idx = _coerce_action_index(v)
            if action_idx is None:
                continue
            overrides[idx] = action_idx
    return overrides, meta


@router.post("/api/step")
def step(request: ActionRequest):
    """Takes a single step in the environment (restored full behavior from pre-refactor)."""
    if not game_state.env or game_state.obs is None:
        raise HTTPException(status_code=400, detail="Game not initialized.")

    user_team = game_state.user_team or Team.OFFENSE
    ai_obs = game_state.obs
    mcts_results: dict[str, dict] = {}
    env_read = env_view(game_state.env)

    player_deterministic = True if request.player_deterministic is None else bool(request.player_deterministic)
    opponent_deterministic = True if request.opponent_deterministic is None else bool(request.opponent_deterministic)

    rollout_actions = predict_joint_policy_actions(
        unified_policy=game_state.unified_policy,
        opponent_policy=game_state.defense_policy,
        obs=ai_obs,
        env=game_state.env,
        player_deterministic=player_deterministic,
        opponent_deterministic=opponent_deterministic,
        role_flag_offense=game_state.role_flag_offense,
        role_flag_defense=game_state.role_flag_defense,
    )
    resolved_unified = rollout_actions["resolved_unified"]
    unified_probs = rollout_actions["unified_probs"]
    resolved_opponent = rollout_actions["resolved_opponent"]
    opponent_probs = rollout_actions["opponent_probs"]

    if request.use_mcts:
        try:
            target_pids: list[int] = []
            if request.mcts_player_ids:
                target_pids = [int(pid) for pid in request.mcts_player_ids if pid is not None]
            elif request.mcts_player_id is not None:
                target_pids = [int(request.mcts_player_id)]
            else:
                default_pid = env_read.ball_holder
                target_pids = [int(default_pid) if default_pid is not None else 0]

            if not target_pids:
                target_pids = [0]

            unique_targets: list[int] = []
            for pid in target_pids:
                if pid not in unique_targets:
                    unique_targets.append(pid)

            for target_pid in unique_targets:
                result = _run_mcts_advisor(
                    player_id=target_pid,
                    obs=ai_obs,
                    env=game_state.env,
                    max_depth=request.mcts_max_depth,
                    time_budget_ms=request.mcts_time_budget_ms,
                    exploration_c=request.mcts_exploration_c,
                    use_priors=request.mcts_use_priors,
                )
                mcts_results[str(target_pid)] = result
                best_action = result.get("action") if isinstance(result, dict) else None
                if best_action is not None:
                    if resolved_unified is None:
                        resolved_unified = np.zeros(int(env_read.n_players or 0), dtype=int)
                    if target_pid in (env_read.offense_ids or []):
                        resolved_unified[target_pid] = int(best_action)
                    elif resolved_opponent is not None:
                        resolved_opponent[target_pid] = int(best_action)
                    else:
                        resolved_unified[target_pid] = int(best_action)
        except Exception as err:
            print(f"[MCTS] Failed to run advisor inside step: {err}")

    action_mask = ai_obs.get("action_mask") if isinstance(ai_obs, dict) else None
    if action_mask is None:
        action_mask = np.ones((int(env_read.n_players or 0), len(ActionType)), dtype=int)

    # Combine user and AI actions, then apply any explicit per-player overrides.
    full_action = combine_team_actions(
        env=game_state.env,
        user_team=user_team,
        resolved_unified=resolved_unified,
        resolved_opponent=resolved_opponent,
    )
    overrides, action_meta = _normalize_action_overrides(
        request.actions,
        int(env_read.n_players or 0),
    )

    pointer_targets: dict[int, int] = {}
    for pid, meta in action_meta.items():
        if str(meta.get("type", "")).upper() != "PASS":
            continue
        try:
            pointer_targets[int(pid)] = int(meta.get("target"))
        except Exception:
            continue
    set_pointer_pass_targets = env_read.set_pointer_pass_targets
    if set_pointer_pass_targets is not None:
        set_pointer_pass_targets(pointer_targets)

    for i in range(int(env_read.n_players or 0)):
        if i in overrides:
            proposed = overrides[i]
            try:
                if proposed is not None and proposed >= 0 and proposed < len(action_mask[i]) and action_mask[i][proposed] == 1:
                    full_action[i] = proposed
                else:
                    full_action[i] = 0
            except Exception:
                full_action[i] = 0
            continue

        full_action[i] = int(full_action[i])

    # Pre-step state values (guard when unified policy missing)
    pre_step_offensive_value = None
    pre_step_defensive_value = None
    if game_state.unified_policy is not None:
        try:
            offense_ids = list(env_read.offense_ids or [])
            defense_ids = list(env_read.defense_ids or [])

            if offense_ids and unified_probs is not None:
                offense_rep = (
                    env_read.ball_holder
                    if env_read.ball_holder in offense_ids
                    else offense_ids[0]
                )
                offense_q_values = _compute_q_values_for_player(offense_rep, game_state)
                offense_probs = unified_probs
                pre_step_offensive_value = sum(
                    offense_probs[offense_rep][i] * offense_q_values.get(ActionType(i).name, 0.0)
                    for i in range(len(offense_probs[offense_rep]))
                )

            if defense_ids and unified_probs is not None:
                defense_rep = defense_ids[0]
                defense_q_values = _compute_q_values_for_player(defense_rep, game_state)
                defense_probs = unified_probs
                pre_step_defensive_value = sum(
                    defense_probs[defense_rep][i] * defense_q_values.get(ActionType(i).name, 0.0)
                    for i in range(len(defense_probs[defense_rep]))
                )
        except Exception as e:
            print(f"[WARNING] Failed to calculate pre-step state values: {e}")

    game_state.prev_obs = game_state.obs
    game_state.obs, rewards, done, _, info = game_state.env.step(full_action)
    _maybe_apply_app_multisegment_boundary(
        info if isinstance(info, dict) else None,
        bool(done),
    )

    try:
        game_state.actions_log.append([int(a) for a in full_action.tolist()])
    except Exception:
        game_state.actions_log.append([int(a) for a in list(full_action)])

    if isinstance(rewards, np.ndarray):
        rewards_list = rewards.tolist()
    elif isinstance(rewards, (list, tuple)):
        rewards_list = list(rewards)
    else:
        rewards_list = [rewards]

    if len(rewards_list) > 1:
        step_rewards = {"offense": 0.0, "defense": 0.0}
        for i, reward in enumerate(rewards_list):
            if i in (env_read.offense_ids or []):
                team_key = "offense"
            elif i in (env_read.defense_ids or []):
                team_key = "defense"
            else:
                continue
            step_rewards[team_key] += float(reward)
            game_state.episode_rewards[team_key] += float(reward)
    else:
        step_rewards = {"offense": float(rewards_list[0]), "defense": 0.0}
        game_state.episode_rewards["offense"] += float(rewards_list[0])

    offense_reasons: list[str] = []
    defense_reasons: list[str] = []
    action_results = info.get("action_results", {}) if info else {}

    if action_results.get("shots"):
        for pid, shot_res in action_results["shots"].items():
            try:
                pid_int = int(pid)
            except Exception:
                pid_int = pid
            dist_at_shot = int(shot_res.get("distance", 0))
            is_three = bool(
                shot_res.get("is_three")
                if "is_three" in shot_res
                else dist_at_shot >= getattr(game_state.env, "three_point_distance", dist_at_shot)
            )
            game_state.shot_log.append(
                {
                    "step": int(len(game_state.reward_history) + 1),
                    "player_id": int(pid_int),
                    "distance": dist_at_shot,
                    "probability": float(shot_res.get("probability", 0.0)),
                    "success": bool(shot_res.get("success", False)),
                    "is_three": is_three,
                    "rng": float(shot_res.get("rng", -1.0)),
                    "base_probability": float(shot_res.get("base_probability", -1.0)),
                    "pressure_multiplier": float(shot_res.get("pressure_multiplier", -1.0)),
                    "expected_points": float(shot_res.get("expected_points", 0.0)),
                    "shooter_fg_pct": float(shot_res.get("probability", 0.0)),
                }
            )

    if action_results.get("shots"):
        for _, shot_result in action_results["shots"].items():
            ep_val = float(shot_result.get("expected_points", 0.0))
            ep_str = f"{ep_val:.2f}"
            if shot_result.get("success"):
                offense_reasons.append(f"Shot Make (EP={ep_str})")
                defense_reasons.append(f"Opp Shot (EP={ep_str})")
            else:
                offense_reasons.append(f"Shot Miss (EP={ep_str})")
                defense_reasons.append(f"Opp Shot (EP={ep_str})")

    if action_results.get("passes"):
        successful_passes = 0
        for _, pass_result in action_results["passes"].items():
            if pass_result.get("success"):
                successful_passes += 1
        if successful_passes > 0:
            if successful_passes == 1:
                offense_reasons.append("Pass")
                defense_reasons.append("Opp Pass")
            else:
                offense_reasons.append(f"{successful_passes} Passes")
                defense_reasons.append(f"Opp {successful_passes} Passes")

    if action_results.get("turnovers"):
        for turnover_info in action_results["turnovers"]:
            reason = turnover_info.get("reason", "Unknown")
            if reason == "out_of_bounds":
                offense_reasons.append("TO - OOB")
                defense_reasons.append("Forced TO - OOB")
            elif reason == "pressure":
                offense_reasons.append("TO - Pressure")
                defense_reasons.append("Forced TO - Pressure")
            elif reason == "steal":
                offense_reasons.append("TO - Steal")
                defense_reasons.append("Forced TO - Steal")
            else:
                offense_reasons.append(f"TO - {reason}")
                defense_reasons.append(f"Forced TO - {reason}")

    if action_results.get("moves"):
        for player_id, move_result in action_results["moves"].items():
            if not move_result.get("success", True) and move_result.get("reason") == "out_of_bounds":
                if player_id in (env_read.offense_ids or []):
                    offense_reasons.append("OOB Move")
                    defense_reasons.append("Opp OOB")
                else:
                    defense_reasons.append("OOB Move")
                    offense_reasons.append("Opp OOB")

    off_reward = step_rewards["offense"]
    def_reward = step_rewards["defense"]
    if not offense_reasons and not defense_reasons:
        if abs(off_reward) < 0.001 and abs(def_reward) < 0.001:
            offense_reasons.append("None")
            defense_reasons.append("None")
        else:
            offense_reasons.append("Positive" if off_reward > 0 else "Negative" if off_reward < 0 else "None")
            defense_reasons.append("Positive" if def_reward > 0 else "Negative" if def_reward < 0 else "None")

    phi_r_shape_per_team = float(info.get("phi_r_shape", 0.0)) if info else 0.0
    ep_by_player = []
    if info and "phi_ep_by_player" in info:
        try:
            ep_by_player = [float(x) for x in info.get("phi_ep_by_player", [])]
        except Exception:
            ep_by_player = []

    if not ep_by_player and game_state.env:
        try:
            env = getattr(game_state.env, "unwrapped", game_state.env)
            for pid in range(getattr(env, "n_players", 0)):
                pos = getattr(env, "positions", [])[pid] if pid < len(getattr(env, "positions", [])) else (0, 0)
                dist = env._hex_distance(pos, getattr(env, "basket_position", (0, 0))) if hasattr(env, "_hex_distance") else 0
                is_three = env.is_three_point_location(pos) if hasattr(env, "is_three_point_location") else False
                if getattr(env, "allow_dunks", False) and dist == 0:
                    shot_value = 2.0
                else:
                    shot_value = 3.0 if is_three else 2.0
                p = float(env._calculate_shot_probability(pid, dist)) if hasattr(env, "_calculate_shot_probability") else 0.0
                ep_by_player.append(float(shot_value * p))
        except Exception as e:
            print(f"[WARNING] Failed to calculate EP data: {e}")
            ep_by_player = []

    env_read = env_view(game_state.env)
    game_state.reward_history.append(
        {
            "step": len(game_state.reward_history) + 1,
            "offense": float(step_rewards["offense"]),
            "defense": float(step_rewards["defense"]),
            "offense_reason": ", ".join(offense_reasons) if offense_reasons else "None",
            "defense_reason": ", ".join(defense_reasons) if defense_reasons else "None",
            "phi_r_shape": phi_r_shape_per_team,
            "ep_by_player": ep_by_player,
            "ball_handler": (
                int(env_read.ball_holder)
                if env_read.ball_holder is not None
                else -1
            ),
            "offense_ids": list(env_read.offense_ids or []),
            "is_terminal": bool(done),
            "shot_clock": int(env_read.shot_clock or -1),
        }
    )

    try:
        entry = {
            "step": int(len(game_state.reward_history)),
            "phi_prev": float(info.get("phi_prev", -1.0)) if info else -1.0,
            "phi_next": float(info.get("phi_next", -1.0)) if info else -1.0,
            "phi_beta": float(info.get("phi_beta", -1.0)) if info else -1.0,
            "phi_r_shape": float(info.get("phi_r_shape", 0.0)) if info else 0.0,
            "ball_handler": (
                int(env_read.ball_holder)
                if env_read.ball_holder is not None
                else -1
            ),
            "offense_ids": list(env_read.offense_ids or []),
            "defense_ids": list(env_read.defense_ids or []),
            "shot_clock": int(env_read.shot_clock or -1),
            "is_terminal": bool(done),
        }
        if info and "phi_ep_by_player" in info:
            try:
                ep_list = list(info.get("phi_ep_by_player", []))
                entry["ep_by_player"] = [float(x) for x in ep_list]
            except Exception:
                entry["ep_by_player"] = []
        if info and "phi_team_best_ep" in info:
            entry["team_best_ep"] = float(info.get("phi_team_best_ep", -1.0))
        if info and "phi_ball_handler_ep" in info:
            entry["ball_handler_ep"] = float(info.get("phi_ball_handler_ep", -1.0))
        game_state.phi_log.append(entry)
    except Exception:
        pass

    try:
        frame = game_state.env.render()
        if frame is not None:
            game_state.frames.append(frame)
    except Exception as e:
        print(f"Warning: Failed to capture frame at step {len(game_state.frames)}: {e}")

    action_names = [a.name for a in ActionType]
    actions_taken = {}
    actions_taken_meta = {}
    for pid, act_idx in enumerate(full_action):
        action_name = action_names[act_idx] if act_idx < len(action_names) else "UNKNOWN"
        actions_taken[str(pid)] = action_name
        actions_taken_meta[str(pid)] = {"type": action_name}

    env_read = env_view(game_state.env)
    is_pointer_mode = str(env_read.pass_mode or "directional").lower() == "pointer_targeted"
    if is_pointer_mode:
        for pid, act_idx in enumerate(full_action):
            action_name = action_names[act_idx] if act_idx < len(action_names) else "UNKNOWN"
            if not str(action_name).startswith("PASS"):
                continue

            intended_target = None
            meta = action_meta.get(pid)
            if isinstance(meta, dict) and str(meta.get("type", "")).upper() == "PASS":
                try:
                    intended_target = int(meta.get("target"))
                except Exception:
                    intended_target = None

            if intended_target is None:
                pass_entry = action_results.get("passes", {}).get(pid)
                if pass_entry is None:
                    pass_entry = action_results.get("passes", {}).get(str(pid))
                if isinstance(pass_entry, dict):
                    raw_target = pass_entry.get("intended_target")
                    if raw_target is None:
                        raw_target = pass_entry.get("target")
                    try:
                        intended_target = int(raw_target)
                    except Exception:
                        intended_target = None

            if intended_target is None:
                for turnover in action_results.get("turnovers", []):
                    try:
                        turnover_pid = int(turnover.get("player_id"))
                    except Exception:
                        continue
                    if turnover_pid != int(pid):
                        continue
                    raw_target = turnover.get("intended_target")
                    if raw_target is None:
                        raw_target = turnover.get("pass_target")
                    try:
                        intended_target = int(raw_target)
                    except Exception:
                        intended_target = None
                    break

            if intended_target is not None:
                actions_taken_meta[str(pid)] = {
                    "type": "PASS",
                    "target": intended_target,
                }
            else:
                actions_taken_meta[str(pid)] = {"type": "PASS"}

    state_with_policy = {}
    try:
        state_with_policy = get_ui_game_state()
        state_with_policy["actions_taken"] = actions_taken
        state_with_policy["actions_taken_meta"] = actions_taken_meta
        game_state.episode_states.append(dict(state_with_policy))
    except Exception as e:
        print(f"[get_full_game_state] Failed to capture episode state: {e}")

    _capture_turn_start_snapshot()
    if game_state.self_play_active and done:
        game_state.self_play_active = False

    return {
        "status": "success",
        "state": state_with_policy,
        "actions_taken": actions_taken,
        "actions_taken_meta": actions_taken_meta,
        "step_rewards": {
            "offense": float(step_rewards["offense"]),
            "defense": float(step_rewards["defense"]),
        },
        "episode_rewards": {
            "offense": float(game_state.episode_rewards["offense"]),
            "defense": float(game_state.episode_rewards["defense"]),
        },
        "pre_step_state_values": {
            "offensive_value": float(pre_step_offensive_value) if pre_step_offensive_value is not None else None,
            "defensive_value": float(pre_step_defensive_value) if pre_step_defensive_value is not None else None,
        },
        "mcts": jsonable_encoder(mcts_results) if mcts_results else None,
    }


@router.post("/api/action")
def action(request: ActionRequest):
    # Backward-compatible alias
    return step(request)


@router.post("/api/mcts_advise")
def mcts_advise(request: MCTSAdviseRequest):
    """Return an MCTS-recommended action without advancing the environment."""
    if not game_state.env or game_state.obs is None:
        raise HTTPException(status_code=400, detail="Game not initialized.")

    try:
        advice = _run_mcts_advisor(
            player_id=request.player_id,
            obs=game_state.obs,
            env=game_state.env,
            max_depth=request.max_depth,
            time_budget_ms=request.time_budget_ms,
            exploration_c=request.exploration_c,
            use_priors=request.use_priors,
        )
        return {"status": "success", "advice": jsonable_encoder(advice)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MCTS failed: {e}")


@router.post("/api/start_self_play")
def start_self_play():
    """Prepare deterministic self-play by resetting with current initial conditions and a fixed seed."""
    if not game_state.env:
        raise HTTPException(status_code=400, detail="Game not initialized.")

    env_read = env_view(game_state.env)
    init_positions = [(int(q), int(r)) for (q, r) in env_read.positions]
    init_ball_holder = (
        int(env_read.ball_holder) if env_read.ball_holder is not None else None
    )
    init_shot_clock = int(env_read.shot_clock or 24)

    import numpy as _np

    episode_seed = int(_np.random.SeedSequence().entropy % (2**32 - 1))

    game_state.replay_seed = episode_seed
    game_state.replay_initial_positions = init_positions
    game_state.replay_ball_holder = init_ball_holder
    game_state.replay_shot_clock = init_shot_clock
    game_state.actions_log = []
    game_state.self_play_active = True
    game_state.frames = []
    game_state.reward_history = []
    game_state.episode_rewards = {"offense": 0.0, "defense": 0.0}
    game_state.episode_states = []
    if game_state.replay_offense_skills is None:
        game_state.replay_offense_skills = {
            "layup": list(game_state.env.offense_layup_pct_by_player),
            "three_pt": list(game_state.env.offense_three_pt_pct_by_player),
            "dunk": list(game_state.env.offense_dunk_pct_by_player),
        }
    if game_state.sampled_offense_skills is None and game_state.replay_offense_skills is not None:
        game_state.sampled_offense_skills = copy.deepcopy(game_state.replay_offense_skills)

    options = {
        "initial_positions": init_positions,
        "ball_holder": init_ball_holder,
        "shot_clock": init_shot_clock,
        "offense_skills": game_state.replay_offense_skills,
    }
    game_state.obs, _ = game_state.env.reset(seed=episode_seed, options=options)
    game_state.prev_obs = None
    _initialize_app_selector_runtime_for_episode()
    _capture_turn_start_snapshot()

    try:
        frame = game_state.env.render()
        if frame is not None:
            game_state.frames.append(frame)
    except Exception:
        pass

    initial_state = get_ui_game_state()
    game_state.episode_states.append(initial_state)

    return {
        "status": "success",
        "state": get_ui_game_state(),
        "seed": episode_seed,
    }


@router.post("/api/reset_turn_state")
def reset_turn_state():
    """Restore positions/ball holder/shot clock to the start-of-turn snapshot."""
    if not game_state.env or game_state.obs is None:
        raise HTTPException(status_code=400, detail="Game not initialized.")
    if not game_state.turn_start_positions:
        raise HTTPException(status_code=400, detail="No turn snapshot available.")

    env = game_state.env
    try:
        for idx, pos in enumerate(game_state.turn_start_positions):
            if idx < len(env.positions):
                env.positions[idx] = (int(pos[0]), int(pos[1]))
        env.ball_holder = (
            int(game_state.turn_start_ball_holder)
            if game_state.turn_start_ball_holder is not None
            else None
        )
        if game_state.turn_start_shot_clock is not None:
            env.shot_clock = int(game_state.turn_start_shot_clock)

        role_value = (
            float(np.asarray(game_state.obs.get("role_flag"), dtype=np.float32).reshape(-1)[0])
            if game_state.obs is not None and game_state.obs.get("role_flag") is not None
            else (1.0 if env.training_team == Team.OFFENSE else -1.0)
        )
        observer_is_offense = bool(role_value > 0.0)
        if hasattr(env, "_build_observation_dict"):
            new_obs_dict = env._build_observation_dict(observer_is_offense)
            new_obs_dict["role_flag"] = np.array([role_value], dtype=np.float32)
        else:
            new_obs_dict = {
                "obs": env._get_observation(),
                "action_mask": env._get_action_masks(),
                "role_flag": np.array([role_value], dtype=np.float32),
                "skills": env._get_offense_skills_array(),
            }
        game_state.obs = new_obs_dict
        game_state.prev_obs = None

        return {
            "status": "success",
            "state": get_ui_game_state(),
        }
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to reset turn: {e}")
