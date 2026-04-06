import copy
import numpy as np
import torch
import time
from threading import Lock
from fastapi.encoders import jsonable_encoder
from basketworld.envs.basketworld_env_v2 import ActionType, Team
from basketworld.utils.play_names import (
    build_model_codename,
    build_play_name_mapping,
    lookup_play_name,
    play_name_seed_key,
)
from basketworld.utils.wrappers import SetObservationWrapper
from app.backend.env_access import env_view, get_env_attr


class GameState:
    """Lightweight container for backend session state (single-user demo)."""

    def __init__(self):
        self.env = None
        self.offense_policy = None
        self.defense_policy = None
        self.unified_policy = None
        self.user_team: Team = None
        self.obs = None
        self.frames = []  # List of RGB frames for the current episode
        self.reward_history = []  # Track rewards for each step
        self.episode_rewards = {"offense": 0.0, "defense": 0.0}  # Running totals
        self.shot_log = []  # Per-step shot attempts with probability and result
        self.phi_log = []  # Per-step Phi diagnostics and EPs
        # Track which policies are currently loaded so we can persist logs across episodes
        self.offense_policy_key: str | None = None
        self.defense_policy_key: str | None = None
        self.unified_policy_key: str | None = None
        # Opponent unified policy (if different from unified)
        self.opponent_unified_policy_key: str | None = None
        # Self-play / replay tracking
        self.self_play_active: bool = False
        self.replay_seed: int | None = None
        self.replay_initial_positions: list[tuple[int, int]] | None = None
        self.replay_ball_holder: int | None = None
        self.replay_shot_clock: int | None = None
        self.replay_offense_skills: dict | None = None  # Store sampled skills for consistency
        self.sampled_offense_skills: dict | None = None  # Baseline skills from initial game creation
        self.actions_log: list[list[int]] = []  # full action arrays per step
        # General replay buffers (manual or AI). We store full game states for instant replay
        self.episode_states: list[dict] = []
        # MLflow run metadata
        self.run_id: str | None = None
        self.run_name: str | None = None
        # MLflow phi shaping parameters (used for Rewards tab calculations)
        # This is separate from env.phi_beta etc which can be modified in Phi Shaping tab
        self.mlflow_phi_shaping_params: dict | None = None
        # MLflow training parameters (PPO hyperparameters)
        self.mlflow_training_params: dict | None = None
        # Persisted start-template library artifact loaded from the MLflow run.
        self.mlflow_start_template_library: dict | None = None
        # Role flag encoding (for backward compatibility with old models)
        self.role_flag_offense: float = 1.0  # Default to new encoding
        self.role_flag_defense: float = -1.0  # Default to new encoding
        # Cache previous observation to handle race condition between move-recorded and step
        self.prev_obs: dict | None = None
        # Turn-start snapshot for frontend resets
        self.turn_start_positions: list[tuple[int, int]] | None = None
        self.turn_start_ball_holder: int | None = None
        self.turn_start_shot_clock: int | None = None
        # Parallel evaluation support - store params/paths for worker recreation
        self.env_required_params: dict | None = None
        self.env_optional_params: dict | None = None
        # Immutable copy of MLflow-loaded environment options for UI reset actions.
        self.mlflow_env_optional_defaults: dict | None = None
        self.unified_policy_path: str | None = None
        self.opponent_policy_path: str | None = None
        # Public/playable mode session metadata.
        self.playable_session: dict | None = None
        # User-defined snapshot for counterfactual restore/replay.
        self.counterfactual_snapshot: dict | None = None
        # App-side selector runtime state for integrated multi-segment inference.
        self.selector_segment_index: int = 0
        self.selector_last_boundary_reason: str | None = None
        # Evaluation progress for frontend polling.
        self._evaluation_progress_lock = Lock()
        self.evaluation_progress: dict = {
            "running": False,
            "completed": 0,
            "total": 0,
            "started_at": None,
            "finished_at": None,
            "status": "idle",
            "error": None,
        }
        self._playbook_progress_lock = Lock()
        self.playbook_progress: dict = {
            "running": False,
            "completed": 0,
            "total": 0,
            "started_at": None,
            "finished_at": None,
            "status": "idle",
            "error": None,
        }


game_state = GameState()


def reset_evaluation_progress(total: int = 0) -> None:
    with game_state._evaluation_progress_lock:
        game_state.evaluation_progress = {
            "running": bool(total > 0),
            "completed": 0,
            "total": int(max(0, total)),
            "started_at": time.time() if total > 0 else None,
            "finished_at": None,
            "status": "running" if total > 0 else "idle",
            "error": None,
        }


def update_evaluation_progress(completed: int, total: int | None = None) -> None:
    with game_state._evaluation_progress_lock:
        current_total = int(game_state.evaluation_progress.get("total") or 0)
        next_total = current_total if total is None else int(max(0, total))
        next_completed = int(max(0, completed))
        running = next_completed < next_total if next_total > 0 else False
        game_state.evaluation_progress.update(
            {
                "completed": next_completed,
                "total": next_total,
                "running": running,
                "status": "running" if running else ("completed" if next_total > 0 else "idle"),
            }
        )
        if not running and next_total > 0:
            game_state.evaluation_progress["finished_at"] = time.time()


def fail_evaluation_progress(error: str) -> None:
    with game_state._evaluation_progress_lock:
        game_state.evaluation_progress.update(
            {
                "running": False,
                "status": "failed",
                "error": str(error),
                "finished_at": time.time(),
            }
        )


def get_evaluation_progress() -> dict:
    with game_state._evaluation_progress_lock:
        payload = dict(game_state.evaluation_progress)
    total = int(payload.get("total") or 0)
    completed = int(payload.get("completed") or 0)
    payload["fraction"] = float(completed / total) if total > 0 else 0.0
    return payload


def reset_playbook_progress(total: int = 0) -> None:
    with game_state._playbook_progress_lock:
        game_state.playbook_progress = {
            "running": bool(total > 0),
            "completed": 0,
            "total": int(max(0, total)),
            "started_at": time.time() if total > 0 else None,
            "finished_at": None,
            "status": "running" if total > 0 else "idle",
            "error": None,
        }


def update_playbook_progress(completed: int, total: int | None = None) -> None:
    with game_state._playbook_progress_lock:
        current_total = int(game_state.playbook_progress.get("total") or 0)
        next_total = current_total if total is None else int(max(0, total))
        next_completed = int(max(0, completed))
        running = next_completed < next_total if next_total > 0 else False
        game_state.playbook_progress.update(
            {
                "completed": next_completed,
                "total": next_total,
                "running": running,
                "status": "running" if running else ("completed" if next_total > 0 else "idle"),
            }
        )
        if not running and next_total > 0:
            game_state.playbook_progress["finished_at"] = time.time()


def fail_playbook_progress(error: str) -> None:
    with game_state._playbook_progress_lock:
        game_state.playbook_progress.update(
            {
                "running": False,
                "status": "failed",
                "error": str(error),
                "finished_at": time.time(),
            }
        )


def get_playbook_progress() -> dict:
    with game_state._playbook_progress_lock:
        payload = dict(game_state.playbook_progress)
    total = int(payload.get("total") or 0)
    completed = int(payload.get("completed") or 0)
    payload["fraction"] = float(completed / total) if total > 0 else 0.0
    return payload


def _role_flag_value_for_team(team: Team) -> float:
    """Resolve the role_flag value for offense vs defense with backward compatibility."""
    if team == Team.OFFENSE:
        value = getattr(game_state, "role_flag_offense", None)
        return float(value if value is not None else 1.0)
    value = getattr(game_state, "role_flag_defense", None)
    return float(value if value is not None else -1.0)


def _capture_turn_start_snapshot():
    """Store current positions/ball holder/shot clock as the baseline for the turn."""
    if not game_state.env:
        return
    env = env_view(game_state.env)
    try:
        game_state.turn_start_positions = [
            (int(pos[0]), int(pos[1])) for pos in (env.positions or [])
        ]
    except Exception:
        game_state.turn_start_positions = None
    game_state.turn_start_ball_holder = (
        int(env.ball_holder) if env.ball_holder is not None else None
    )
    game_state.turn_start_shot_clock = int(env.shot_clock or 0)


def _rebuild_cached_obs() -> None:
    """Rebuild game_state.obs from env while preserving the current viewer role."""
    if not game_state.env:
        return
    from app.backend.observations import rebuild_observation_from_env

    game_state.obs = rebuild_observation_from_env(
        game_state.env,
        current_obs=game_state.obs,
    )


def _build_counterfactual_snapshot_metadata() -> dict:
    env = env_view(game_state.env)
    return {
        "captured_step": int(len(getattr(game_state, "actions_log", []) or [])),
        "shot_clock": int(env.shot_clock or 0),
        "ball_holder": (
            int(env.ball_holder or 0)
            if env.ball_holder is not None
            else None
        ),
        "intent_active": bool(env.intent_active),
        "intent_index": int(env.intent_index or 0),
        "intent_age": int(env.intent_age or 0),
        "captured_at": float(time.time()),
    }


def get_current_play_name_map(num_intents: int | None = None) -> dict[str, str]:
    env_obj = env_view(game_state.env) if game_state.env else None
    count = int(
        max(
            0,
            num_intents
            if num_intents is not None
            else getattr(env_obj, "num_intents", 0) or 0,
        )
    )
    if count <= 0:
        return {}
    seed_key = play_name_seed_key(
        run_id=getattr(game_state, "run_id", None),
        unified_policy_key=getattr(game_state, "unified_policy_key", None),
        unified_policy_path=getattr(game_state, "unified_policy_path", None),
        run_name=getattr(game_state, "run_name", None),
        fallback="session",
    )
    mapping = build_play_name_mapping(seed_key, count)
    return {str(int(idx)): str(name) for idx, name in mapping.items()}


def get_current_model_codename() -> str | None:
    seed_key = play_name_seed_key(
        run_id=getattr(game_state, "run_id", None),
        unified_policy_key=getattr(game_state, "unified_policy_key", None),
        unified_policy_path=getattr(game_state, "unified_policy_path", None),
        run_name=getattr(game_state, "run_name", None),
        fallback="session",
    )
    try:
        codename = build_model_codename(seed_key)
    except Exception:
        return None
    return str(codename).strip() or None


def _capture_restorable_backend_state() -> dict:
    """Capture enough backend session state to restore the current live branch exactly."""
    if not game_state.env or game_state.obs is None:
        raise RuntimeError("Game not initialized.")

    return {
        "env": copy.deepcopy(game_state.env),
        "user_team": copy.deepcopy(game_state.user_team),
        "obs": copy.deepcopy(game_state.obs),
        "prev_obs": copy.deepcopy(game_state.prev_obs),
        "sampled_offense_skills": copy.deepcopy(game_state.sampled_offense_skills),
        "replay_seed": copy.deepcopy(game_state.replay_seed),
        "replay_initial_positions": copy.deepcopy(game_state.replay_initial_positions),
        "replay_ball_holder": copy.deepcopy(game_state.replay_ball_holder),
        "replay_shot_clock": copy.deepcopy(game_state.replay_shot_clock),
        "replay_offense_skills": copy.deepcopy(game_state.replay_offense_skills),
        "self_play_active": bool(getattr(game_state, "self_play_active", False)),
        "reward_history": copy.deepcopy(game_state.reward_history),
        "episode_rewards": copy.deepcopy(game_state.episode_rewards),
        "shot_log": copy.deepcopy(game_state.shot_log),
        "phi_log": copy.deepcopy(game_state.phi_log),
        "actions_log": copy.deepcopy(game_state.actions_log),
        "episode_states": copy.deepcopy(game_state.episode_states),
        "playable_session": copy.deepcopy(game_state.playable_session),
        "selector_segment_index": int(getattr(game_state, "selector_segment_index", 0)),
        "selector_last_boundary_reason": copy.deepcopy(
            getattr(game_state, "selector_last_boundary_reason", None)
        ),
    }


def _restore_restorable_backend_state(snapshot: dict) -> None:
    """Restore backend session state from `_capture_restorable_backend_state` output."""
    if not snapshot:
        raise RuntimeError("Missing backend restore snapshot.")

    game_state.env = copy.deepcopy(snapshot["env"])
    game_state.user_team = copy.deepcopy(snapshot["user_team"])
    game_state.obs = copy.deepcopy(snapshot["obs"])
    game_state.prev_obs = copy.deepcopy(snapshot["prev_obs"])
    game_state.sampled_offense_skills = copy.deepcopy(snapshot["sampled_offense_skills"])
    game_state.replay_seed = copy.deepcopy(snapshot["replay_seed"])
    game_state.replay_initial_positions = copy.deepcopy(snapshot["replay_initial_positions"])
    game_state.replay_ball_holder = copy.deepcopy(snapshot["replay_ball_holder"])
    game_state.replay_shot_clock = copy.deepcopy(snapshot["replay_shot_clock"])
    game_state.replay_offense_skills = copy.deepcopy(snapshot["replay_offense_skills"])
    game_state.self_play_active = bool(snapshot["self_play_active"])
    game_state.reward_history = copy.deepcopy(snapshot["reward_history"])
    game_state.episode_rewards = copy.deepcopy(snapshot["episode_rewards"])
    game_state.shot_log = copy.deepcopy(snapshot["shot_log"])
    game_state.phi_log = copy.deepcopy(snapshot["phi_log"])
    game_state.actions_log = copy.deepcopy(snapshot["actions_log"])
    game_state.episode_states = copy.deepcopy(snapshot["episode_states"])
    game_state.playable_session = copy.deepcopy(snapshot["playable_session"])
    game_state.selector_segment_index = int(snapshot.get("selector_segment_index", 0))
    game_state.selector_last_boundary_reason = copy.deepcopy(
        snapshot.get("selector_last_boundary_reason", None)
    )

    _rebuild_cached_obs()
    _capture_turn_start_snapshot()


def get_counterfactual_snapshot_summary() -> dict:
    snapshot = getattr(game_state, "counterfactual_snapshot", None)
    metadata = snapshot.get("metadata", {}) if isinstance(snapshot, dict) else {}
    return {
        "available": bool(snapshot),
        "captured_step": metadata.get("captured_step"),
        "shot_clock": metadata.get("shot_clock"),
        "ball_holder": metadata.get("ball_holder"),
        "intent_active": metadata.get("intent_active"),
        "intent_index": metadata.get("intent_index"),
        "intent_age": metadata.get("intent_age"),
        "captured_at": metadata.get("captured_at"),
    }


def capture_counterfactual_snapshot() -> dict:
    """Capture a restorable snapshot of the current env/session state."""
    snapshot = _capture_restorable_backend_state()
    snapshot["metadata"] = _build_counterfactual_snapshot_metadata()
    game_state.counterfactual_snapshot = snapshot
    return get_counterfactual_snapshot_summary()


def restore_counterfactual_snapshot() -> dict:
    """Restore the most recently captured counterfactual snapshot."""
    snapshot = getattr(game_state, "counterfactual_snapshot", None)
    if not snapshot:
        raise RuntimeError("No counterfactual snapshot available.")

    _restore_restorable_backend_state(snapshot)
    return get_counterfactual_snapshot_summary()


def get_full_game_state(
    include_policy_probs: bool = False,
    include_action_values: bool = False,
    include_state_values: bool = False,
):
    """Construct a JSON-friendly snapshot of the current game state."""
    if not game_state.env:
        return {}

    # Local imports to avoid circular dependencies at module load time
    from app.backend.observations import (
        compute_policy_probabilities,
        _compute_q_values_for_player,
        _compute_state_values_from_obs,
    )
    from app.backend.selector_runtime import selector_ranked_intent_preferences

    # Use FastAPI's jsonable_encoder for numpy-safe encoding
    custom_encoder = {
        np.integer: int,
        np.floating: float,
        np.bool_: bool,
    }

    env = env_view(game_state.env)
    last_action_results_py = jsonable_encoder(
        env.last_action_results, custom_encoder=custom_encoder
    )

    def _set_global_labels(obs_tokens_dict: dict, globals_like) -> None:
        if globals_like is None or obs_tokens_dict is None:
            return
        try:
            globals_arr = np.asarray(globals_like, dtype=np.float32).reshape(-1)
            n = int(globals_arr.shape[0])
        except Exception:
            return
        if n <= 0:
            return
        labels = []
        base_labels = ["shot_clock", "pressure_exposure", "hoop_q_norm", "hoop_r_norm"]
        base_take = min(len(base_labels), n)
        labels.extend(base_labels[:base_take])
        extras = max(0, n - base_take)
        if extras > 0:
            intent_labels = [
                "intent_index_norm",
                "intent_active",
                "intent_visible",
                "intent_age_norm",
            ]
            intent_take = min(len(intent_labels), extras)
            labels.extend(intent_labels[:intent_take])
            for idx in range(intent_take, extras):
                labels.append(f"global_{base_take + idx}")
        obs_tokens_dict["globals_labels"] = labels

    # Convert numpy types to standard Python types for JSON serialization
    positions_py = [(int(q), int(r)) for q, r in env.positions]
    ball_holder_py = (
        int(env.ball_holder)
        if env.ball_holder is not None
        else None
    )
    basket_pos_py = (
        int(env.basket_position[0]),
        int(env.basket_position[1]),
    )
    action_mask_py = game_state.obs["action_mask"].tolist()
    obs_tokens = None
    if game_state.obs:
        players_tokens = game_state.obs.get("players")
        globals_tokens = game_state.obs.get("globals")
        if players_tokens is not None or globals_tokens is not None:
            obs_tokens = {}
            if players_tokens is not None:
                obs_tokens["players"] = (
                    players_tokens.tolist()
                    if hasattr(players_tokens, "tolist")
                    else players_tokens
                )
            if globals_tokens is not None:
                obs_tokens["globals"] = (
                    globals_tokens.tolist()
                    if hasattr(globals_tokens, "tolist")
                    else globals_tokens
                )
                _set_global_labels(obs_tokens, globals_tokens)
    if obs_tokens is None and game_state.env and game_state.obs:
        try:
            wrapper = SetObservationWrapper(game_state.env)
            derived = wrapper.observation(game_state.obs)
            players_tokens = derived.get("players")
            globals_tokens = derived.get("globals")
            if players_tokens is not None or globals_tokens is not None:
                obs_tokens = {}
                if players_tokens is not None:
                    obs_tokens["players"] = (
                        players_tokens.tolist()
                        if hasattr(players_tokens, "tolist")
                        else players_tokens
                    )
                if globals_tokens is not None:
                    obs_tokens["globals"] = (
                        globals_tokens.tolist()
                        if hasattr(globals_tokens, "tolist")
                        else globals_tokens
                    )
                    _set_global_labels(obs_tokens, globals_tokens)
        except Exception:
            obs_tokens = obs_tokens

    attention_payload = None
    if obs_tokens is not None and game_state.unified_policy is not None:
        try:
            policy_obj = getattr(game_state.unified_policy, "policy", None)
            extractor = getattr(policy_obj, "features_extractor", None)
            if (
                extractor is not None
                and hasattr(extractor, "token_mlp")
                and hasattr(extractor, "attn")
                and obs_tokens.get("players") is not None
                and obs_tokens.get("globals") is not None
            ):
                players_np = np.asarray(obs_tokens["players"], dtype=np.float32)
                globals_np = np.asarray(obs_tokens["globals"], dtype=np.float32)
                device = next(extractor.parameters()).device
                with torch.no_grad():
                    players_t = torch.as_tensor(players_np, device=device).unsqueeze(0)
                    globals_t = torch.as_tensor(globals_np, device=device).unsqueeze(0)
                    g = globals_t.unsqueeze(1).expand(-1, players_t.size(1), -1)
                    tokens = torch.cat([players_t, g], dim=-1)
                    emb = extractor.token_mlp(tokens)
                    cls_tokens = getattr(extractor, "cls_tokens", None)
                    cls_tokens_list = None
                    if cls_tokens is not None:
                        cls_tokens_list = cls_tokens.detach().cpu().tolist()
                        cls = cls_tokens.unsqueeze(0).expand(emb.size(0), -1, -1)
                        emb = torch.cat([emb, cls], dim=1)
                    _, attn_weights = extractor.attn(
                        emb, emb, emb, need_weights=True, average_attn_weights=False
                    )
                    per_head = attn_weights[0].detach().cpu().numpy()
                    avg_weights = per_head.mean(axis=0).tolist()
                    per_head_weights = per_head.tolist()
                labels = []
                for pid in range(players_np.shape[0]):
                    if pid in (env.offense_ids or []):
                        labels.append(f"O{pid}")
                    elif pid in (env.defense_ids or []):
                        labels.append(f"D{pid}")
                    else:
                        labels.append(f"P{pid}")
                num_cls = int(getattr(extractor, "num_cls_tokens", 0))
                cls_labels = []
                if num_cls >= 1:
                    labels.append("CLS_OFF")
                    cls_labels.append("CLS_OFF")
                if num_cls >= 2:
                    labels.append("CLS_DEF")
                    cls_labels.append("CLS_DEF")
                if cls_tokens_list is not None and len(cls_labels) != len(cls_tokens_list):
                    cls_labels = [f"CLS_{idx + 1}" for idx in range(len(cls_tokens_list))]
                attention_payload = {
                    "weights_avg": avg_weights,
                    "weights_heads": per_head_weights,
                    "labels": labels,
                    "heads": int(getattr(extractor.attn, "num_heads", 0)),
                }
                if cls_tokens_list is not None:
                    attention_payload["cls_tokens"] = cls_tokens_list
                    attention_payload["cls_labels"] = cls_labels
        except Exception:
            attention_payload = None

    # Calculate ball handler's pressure-adjusted shot probability for replay
    ball_handler_shot_prob = None
    if ball_holder_py is not None:
        try:
            base_env = getattr(game_state.env, "unwrapped", game_state.env)
            player_pos = base_env.positions[ball_holder_py]
            basket_pos = base_env.basket_position
            distance = base_env._hex_distance(player_pos, basket_pos)
            ball_handler_shot_prob = float(
                base_env._calculate_shot_probability(ball_holder_py, distance)
            )
        except Exception:
            ball_handler_shot_prob = None

    # Calculate pass steal probabilities for replay
    pass_steal_probs = {}
    if ball_holder_py is not None:
        try:
            base_env = getattr(game_state.env, "unwrapped", game_state.env)
            steal_probs = base_env.calculate_pass_steal_probabilities(ball_holder_py)
            pass_steal_probs = {int(k): float(v) for k, v in steal_probs.items()}
        except Exception as e:
            print(f"[get_full_game_state] Failed to calculate pass steal probabilities: {e}")
            pass_steal_probs = {}

    # Calculate EP (expected points) for all players
    ep_by_player = []
    try:
        ep_env = getattr(game_state.env, "unwrapped", game_state.env)
        for pid in range(ep_env.n_players):
            pos = ep_env.positions[pid]
            dist = ep_env._hex_distance(pos, ep_env.basket_position)
            is_three = ep_env.is_three_point_location(pos)
            if getattr(ep_env, "allow_dunks", True) and dist == 0:
                shot_value = 2.0
            else:
                shot_value = 3.0 if is_three else 2.0
            p = float(ep_env._calculate_shot_probability(pid, dist))
            ep = float(shot_value * p)
            ep_by_player.append(ep)
    except Exception:
        ep_by_player = []

    sampled_offense_skills = getattr(game_state, "sampled_offense_skills", None) or {}

    counterfactual_snapshot = get_counterfactual_snapshot_summary()
    base_env = getattr(game_state.env, "unwrapped", game_state.env)
    play_name_map = get_current_play_name_map(int(getattr(env, "num_intents", 0) or 0))

    state = {
        "players_per_side": int(env.players_per_side or 3),
        "players": int(env.players_per_side or 3),
        "positions": positions_py,
        "ball_holder": ball_holder_py,
        "ball_handler_shot_probability": ball_handler_shot_prob,
        "pass_steal_probabilities": pass_steal_probs,
        "shot_clock": int(env.shot_clock or 0),
        "min_shot_clock": int(env.min_shot_clock or 10),
        "shot_clock_steps": int(
            env.shot_clock_steps if env.shot_clock_steps is not None else (env.shot_clock or 24)
        ),
        "user_team_name": game_state.user_team.name,
        "done": bool(env.episode_ended or False),
        "training_team": (
            env.training_team.name
            if env.training_team
            else None
        ),
        "counterfactual_snapshot_available": bool(counterfactual_snapshot["available"]),
        "counterfactual_snapshot_step": counterfactual_snapshot["captured_step"],
        "counterfactual_snapshot_shot_clock": counterfactual_snapshot["shot_clock"],
        "counterfactual_snapshot_ball_holder": counterfactual_snapshot["ball_holder"],
        "counterfactual_snapshot_intent_active": counterfactual_snapshot["intent_active"],
        "counterfactual_snapshot_intent_index": counterfactual_snapshot["intent_index"],
        "counterfactual_snapshot_intent_age": counterfactual_snapshot["intent_age"],
        "counterfactual_snapshot_captured_at": counterfactual_snapshot["captured_at"],
        "action_space": {action.name: action.value for action in ActionType},
        "action_mask": action_mask_py,
        "obs": game_state.obs["obs"].tolist() if game_state.obs and "obs" in game_state.obs else [],
        "obs_tokens": (
            {**obs_tokens, "attention": attention_payload} if obs_tokens is not None else None
        ),
        "obs_tokens_version": 1 if obs_tokens is not None else 0,
        "last_action_results": last_action_results_py,
        "offense_ids": env.offense_ids,
        "defense_ids": env.defense_ids,
        "basket_position": basket_pos_py,
        "court_width": env.court_width,
        "court_height": env.court_height,
        "three_point_distance": float(env.three_point_distance or 4.0),
        "three_point_short_distance": (
            float(env.three_point_short_distance)
            if env.three_point_short_distance is not None
            else None
        ),
        "three_point_hexes": [
            (int(q), int(r))
            for q, r in getattr(base_env, "_three_point_hexes", set())
        ],
        "three_point_line_hexes": [
            (int(q), int(r))
            for q, r in getattr(base_env, "_three_point_line_hexes", set())
        ],
        "three_point_outline": [
            (float(x), float(y))
            for x, y in getattr(base_env, "_three_point_outline_points", [])
        ],
        "shot_probs": env.shot_probs,
        "shot_params": {
            "layup_pct": float(env.layup_pct or 0.0),
            "three_pt_pct": float(env.three_pt_pct or 0.0),
            "three_pt_extra_hex_decay": float(env.three_pt_extra_hex_decay or 0.05),
            "dunk_pct": float(env.dunk_pct or 0.0),
            "layup_std": float(env.layup_std or 0.0),
            "three_pt_std": float(env.three_pt_std or 0.0),
            "dunk_std": float(env.dunk_std or 0.0),
            "allow_dunks": bool(env.allow_dunks),
        },
        "defender_pressure_distance": int(env.defender_pressure_distance or 1),
        "defender_pressure_turnover_chance": float(env.defender_pressure_turnover_chance or 0.05),
        "defender_pressure_decay_lambda": float(env.defender_pressure_decay_lambda or 1.0),
        "base_steal_rate": float(env.base_steal_rate or 0.35),
        "steal_perp_decay": float(env.steal_perp_decay or 1.5),
        "steal_distance_factor": float(env.steal_distance_factor or 0.08),
        "steal_position_weight_min": float(env.steal_position_weight_min or 0.3),
        "spawn_distance": int(env.spawn_distance or 3),
        "max_spawn_distance": (
            int(env.max_spawn_distance)
            if env.max_spawn_distance is not None
            else None
        ),
        "defender_spawn_distance": int(env.defender_spawn_distance or 0),
        "defender_guard_distance": int(env.defender_guard_distance or 1),
        "offense_spawn_boundary_margin": int(env.offense_spawn_boundary_margin or 0),
        "shot_pressure_enabled": bool(env.shot_pressure_enabled),
        "shot_pressure_max": float(env.shot_pressure_max or 0.5),
        "shot_pressure_lambda": float(env.shot_pressure_lambda or 1.0),
        "shot_pressure_arc_degrees": float(env.shot_pressure_arc_degrees or 60.0),
        "three_pt_extra_hex_decay": float(env.three_pt_extra_hex_decay or 0.05),
        "mask_occupied_moves": bool(env.mask_occupied_moves),
        "three_second_lane_width": int(env.three_second_lane_width or 1),
        "three_second_lane_height": int(env.three_second_lane_height or 3),
        "three_second_max_steps": int(env.three_second_max_steps or 3),
        "illegal_defense_enabled": bool(env.illegal_defense_enabled),
        "offensive_three_seconds_enabled": bool(env.offensive_three_seconds_enabled),
        "enable_intent_learning": bool(env.enable_intent_learning),
        "num_intents": int(env.num_intents or 0),
        "intent_commitment_steps": int(env.intent_commitment_steps or 0),
        "intent_null_prob": float(env.intent_null_prob or 0.0),
        "intent_visible_to_defense_prob": float(env.intent_visible_to_defense_prob or 0.0),
        "enable_defense_intent_learning": bool(env.enable_defense_intent_learning),
        "defense_intent_null_prob": float(env.defense_intent_null_prob or 1.0),
        "play_name_map": play_name_map,
        "intent_diversity_enabled": (
            bool(
                getattr(game_state, "mlflow_training_params", {}).get(
                    "intent_diversity_enabled", False
                )
            )
            if getattr(game_state, "mlflow_training_params", None) is not None
            else None
        ),
        "intent_obs_mode": str(
            env.intent_obs_mode or "private_offense"
        ),
        "intent_active_current": bool(env.intent_active),
        "intent_index_current": int(env.intent_index or 0),
        "current_play_name": lookup_play_name(play_name_map, int(env.intent_index or 0)),
        "intent_age": int(env.intent_age or 0),
        "intent_commitment_remaining": int(env.intent_commitment_remaining or 0),
        "selector_segment_index_current": int(
            getattr(game_state, "selector_segment_index", 0)
        ),
        "selector_last_boundary_reason": getattr(
            game_state, "selector_last_boundary_reason", None
        ),
        "intent_visible_to_defense_current": bool(
            get_env_attr(game_state.env, "_intent_visible_to_defense", False)
        ),
        "defense_intent_active_current": bool(env.defense_intent_active),
        "defense_intent_index_current": int(env.defense_intent_index or 0),
        "current_defense_play_name": lookup_play_name(
            play_name_map, int(env.defense_intent_index or 0)
        ),
        "defense_intent_age": int(env.defense_intent_age or 0),
        "defense_intent_commitment_remaining": int(env.defense_intent_commitment_remaining or 0),
        "include_hoop_vector": bool(env.include_hoop_vector),
        "offensive_lane_hexes": [
            (int(q), int(r)) for q, r in (env.offensive_lane_hexes or set())
        ],
        "defensive_lane_hexes": [
            (int(q), int(r)) for q, r in (env.defensive_lane_hexes or set())
        ],
        "offensive_lane_steps": {
            int(pid): int(steps)
            for pid, steps in get_env_attr(game_state.env, "_offensive_lane_steps", {}).items()
        },
        "defensive_lane_steps": {
            int(pid): int(steps)
            for pid, steps in get_env_attr(game_state.env, "_defender_in_key_steps", {}).items()
        },
        "pass_arc_degrees": float(env.pass_arc_degrees or 60.0),
        "pass_oob_turnover_prob": float(env.pass_oob_turnover_prob or 1.0),
        "pass_target_strategy": env.pass_target_strategy or "nearest",
        "pass_mode": env.pass_mode or "directional",
        "illegal_action_policy": (
            env.illegal_action_policy.value
            if env.illegal_action_policy
            else "noop"
        ),
        "pass_logit_bias": float(
            getattr(game_state.unified_policy.policy, "pass_logit_bias", 0.0)
            if game_state.unified_policy
            and hasattr(game_state.unified_policy, "policy")
            else 0.0
        ),
        "run_id": getattr(game_state, "run_id", None),
        "run_name": getattr(game_state, "run_name", None),
        "model_codename": get_current_model_codename(),
        "training_params": getattr(game_state, "mlflow_training_params", None),
        "start_template_library": (
            jsonable_encoder(
                copy.deepcopy(getattr(game_state, "mlflow_start_template_library", None))
            )
            if getattr(game_state, "mlflow_start_template_library", None) is not None
            else None
        ),
        "mlflow_env_defaults": (
            dict(getattr(game_state, "mlflow_env_optional_defaults", {}) or {})
        ),
        "unified_policy_name": getattr(game_state, "unified_policy_key", None),
        "opponent_unified_policy_name": getattr(
            game_state, "opponent_unified_policy_key", None
        ),
        "offense_shooting_pct_by_player": {
            "layup": [
                float(x)
                for x in (env.offense_layup_pct_by_player or [])
            ],
            "three_pt": [
                float(x)
                for x in (env.offense_three_pt_pct_by_player or [])
            ],
            "dunk": [
                float(x)
                for x in (env.offense_dunk_pct_by_player or [])
            ],
        },
        "offense_shooting_pct_sampled": {
            "layup": [
                float(x)
                for x in sampled_offense_skills.get("layup", [])
            ],
            "three_pt": [
                float(x)
                for x in sampled_offense_skills.get("three_pt", [])
            ],
            "dunk": [
                float(x)
                for x in sampled_offense_skills.get("dunk", [])
            ],
        },
        "ep_by_player": ep_by_player,
    }

    if include_policy_probs:
        policy_probs = compute_policy_probabilities()
        if policy_probs is not None:
            state["policy_probabilities"] = policy_probs
        selector_prefs = selector_ranked_intent_preferences(
            training_params=getattr(game_state, "mlflow_training_params", None),
            env=game_state.env,
            base_obs=game_state.obs,
            unified_policy=game_state.unified_policy,
            opponent_policy=getattr(game_state, "defense_policy", None),
            user_team=game_state.user_team,
            role_flag_offense=1.0,
        )
        if selector_prefs is not None:
            try:
                selector_prefs = dict(selector_prefs)
                selector_prefs["play_name_map"] = play_name_map
                selector_prefs["current_play_name"] = lookup_play_name(
                    play_name_map, selector_prefs.get("current_intent_index")
                )
                ranked_items = []
                for item in selector_prefs.get("intent_probs", []) or []:
                    if not isinstance(item, dict):
                        continue
                    row = dict(item)
                    row["play_name"] = lookup_play_name(
                        play_name_map, row.get("intent_index")
                    )
                    ranked_items.append(row)
                selector_prefs["intent_probs"] = ranked_items
            except Exception:
                pass
            state["selector_intent_preferences"] = selector_prefs

    if include_action_values:
        try:
            action_values_by_player = {}
            for pid in range(env.n_players):
                action_values_by_player[str(pid)] = _compute_q_values_for_player(pid, game_state)
            state["action_values"] = action_values_by_player
        except Exception as e:
            print(f"[get_full_game_state] Failed to compute action values: {e}")

    if include_state_values:
        state_values = _compute_state_values_from_obs(game_state.obs)
        if state_values:
            state["state_values"] = state_values

    return state


def get_ui_game_state():
    """Build the canonical backend snapshot expected by frontend game/replay views."""
    return get_full_game_state(
        include_policy_probs=True,
        include_action_values=True,
        include_state_values=True,
    )
