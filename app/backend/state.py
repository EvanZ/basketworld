import copy
import numpy as np
import time
from threading import Lock
from fastapi.encoders import jsonable_encoder
from basketworld.domain import ActionType, Team
from basketworld.utils.play_names import (
    build_model_codename,
    build_play_name_mapping,
    lookup_play_name,
    play_name_seed_key,
)
from app.backend.env_access import env_view, get_env_attr
from app.backend.inference_adapters import (
    get_policy_backend_kind,
    get_policy_capabilities,
    get_policy_metadata,
    policy_attention_payload,
    policy_observation_tokens,
    policy_observation_vector,
    prepare_policy_for_role,
    unwrap_policy_module,
)


class GameState:
    """Lightweight container for backend session state (single-user demo)."""

    def __init__(self):
        self.env = None
        # JAX-native dev runtime. When set, JAX checkpoints should use this for
        # interactive reset/step instead of stepping the Python env bridge.
        self.jax_runtime = None
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
        self.unified_policy_backend: str | None = None
        self.defense_policy_backend: str | None = None
        self.unified_policy_capabilities: dict | None = None
        self.defense_policy_capabilities: dict | None = None
        # Self-play / replay tracking
        self.self_play_active: bool = False
        self.replay_seed: int | None = None
        self.replay_initial_positions: list[tuple[int, int]] | None = None
        self.replay_ball_holder: int | None = None
        self.replay_shot_clock: int | None = None
        # Identifies the one recording that a self-play run is allowed to
        # append to. The browser sends it back with each self-play step so a
        # late request from an older run cannot contaminate a new replay.
        self.replay_session_id: str | None = None
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
        self.start_template_library_source: str | None = None
        self.start_template_library_path: str | None = None
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
            "cancel_requested": False,
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
            "cancel_requested": False,
        }


def update_playbook_progress(completed: int, total: int | None = None) -> None:
    with game_state._playbook_progress_lock:
        current_total = int(game_state.playbook_progress.get("total") or 0)
        next_total = current_total if total is None else int(max(0, total))
        next_completed = int(max(0, completed))
        running = next_completed < next_total if next_total > 0 else False
        cancel_requested = bool(game_state.playbook_progress.get("cancel_requested"))
        game_state.playbook_progress.update(
            {
                "completed": next_completed,
                "total": next_total,
                "running": running,
                "status": (
                    "cancel_requested"
                    if cancel_requested
                    else ("running" if running else ("completed" if next_total > 0 else "idle"))
                ),
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
                "cancel_requested": False,
            }
        )


def request_playbook_cancel() -> dict:
    with game_state._playbook_progress_lock:
        running = bool(game_state.playbook_progress.get("running"))
        game_state.playbook_progress.update(
            {
                "cancel_requested": True,
                "status": "cancel_requested" if running else "cancelled",
                "running": running,
                "finished_at": None if running else time.time(),
            }
        )
        return dict(game_state.playbook_progress)


def playbook_cancel_requested() -> bool:
    with game_state._playbook_progress_lock:
        return bool(game_state.playbook_progress.get("cancel_requested"))


def cancel_playbook_progress() -> None:
    with game_state._playbook_progress_lock:
        game_state.playbook_progress.update(
            {
                "running": False,
                "status": "cancelled",
                "finished_at": time.time(),
                "cancel_requested": False,
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
    """Rebuild the cached observation from the active JAX runtime."""
    runtime = getattr(game_state, "jax_runtime", None)
    if runtime is None:
        raise RuntimeError("A JAX runtime is required to rebuild observations.")
    game_state.env = runtime.environment
    game_state.obs = runtime.observation_dict(
        observer_is_offense=runtime.user_team_is_a
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


def _coerce_play_name_map(raw_map, *, num_intents: int | None = None) -> dict[str, str]:
    if not isinstance(raw_map, dict):
        return {}
    out: dict[str, str] = {}
    limit = None if num_intents is None else max(0, int(num_intents))
    for raw_idx, raw_name in raw_map.items():
        try:
            idx = int(raw_idx)
        except Exception:
            continue
        if idx < 0 or (limit is not None and idx >= limit):
            continue
        name = str(raw_name or "").strip()
        if name:
            out[str(idx)] = name
    return out


def _play_name_map_from_metadata(metadata: dict | None, *, num_intents: int | None = None) -> dict[str, str]:
    if not isinstance(metadata, dict):
        return {}
    direct = _coerce_play_name_map(metadata.get("play_name_map"), num_intents=num_intents)
    if direct:
        return direct
    play_meta = metadata.get("play_name_metadata")
    if not isinstance(play_meta, dict):
        return {}
    mapped: dict[str, str] = {}
    limit = None if num_intents is None else max(0, int(num_intents))
    for item in play_meta.get("play_names", []) or []:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("intent_index"))
        except Exception:
            continue
        if idx < 0 or (limit is not None and idx >= limit):
            continue
        name = str(item.get("play_name", "")).strip()
        if name:
            mapped[str(idx)] = name
    return mapped


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
    for policy_obj in (
        getattr(game_state, "unified_policy", None),
        getattr(game_state, "offense_policy", None),
        getattr(game_state, "defense_policy", None),
    ):
        mapping = _play_name_map_from_metadata(
            get_policy_metadata(policy_obj),
            num_intents=count,
        )
        if mapping:
            return mapping
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

    jax_runtime = getattr(game_state, "jax_runtime", None)
    if jax_runtime is None or not hasattr(jax_runtime, "capture_snapshot"):
        raise RuntimeError("A JAX runtime is required for restorable snapshots.")
    return {
        "env": None,
        "jax_runtime": jax_runtime.capture_snapshot(),
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

    jax_snapshot = snapshot.get("jax_runtime")
    jax_runtime = getattr(game_state, "jax_runtime", None)
    if jax_runtime is None or jax_snapshot is None:
        raise RuntimeError("A JAX runtime snapshot is required for restore.")
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

    jax_runtime.restore_snapshot(jax_snapshot, game_state=game_state)


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
    """Construct the frontend payload from the active JAX runtime."""
    runtime = getattr(game_state, "jax_runtime", None)
    if runtime is None:
        return {}
    return runtime.get_full_game_state(
        game_state,
        include_policy_probs=include_policy_probs,
        include_action_values=include_action_values,
        include_state_values=include_state_values,
    )

def get_ui_game_state():
    """Build the canonical backend snapshot expected by frontend game/replay views."""
    return get_full_game_state(
        include_policy_probs=True,
        include_action_values=True,
        include_state_values=True,
    )
