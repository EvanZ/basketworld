from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

from basketworld.envs.basketworld_env_v2 import ActionType, HexagonBasketballEnv, Team
from basketworld.utils.play_names import lookup_play_name
from basketworld.utils.start_templates import resolve_start_template
from basketworld_jax.env.minimal import (
    ACTION_COUNT,
    PASS_ACTION_START,
    REBOUND_CONTEST_MODE_LOCAL,
    SHOT_TYPE_DUNK,
    SHOT_TYPE_THREE,
    TURNOVER_REASON_DEFENDER_PRESSURE,
    TURNOVER_REASON_INTERCEPTED,
    TURNOVER_REASON_MOVE_OUT_OF_BOUNDS,
    TURNOVER_REASON_OFFENSIVE_THREE_SECONDS,
    TURNOVER_REASON_PASS_OUT_OF_BOUNDS,
    TURNOVER_REASON_SHOT_CLOCK,
    build_action_masks_batch,
    build_kernel_static_from_env,
    build_pass_steal_probabilities_batch,
    build_policy_intent_context_batch_with_role_flag,
    build_policy_observation_batch_with_role_flag,
    build_shot_profile_batch,
    build_token_observation_components_batch,
    reset_batch_minimal,
    stack_state_snapshots,
    step_batch_minimal,
)
from app.backend.inference_adapters import (
    get_policy_backend_kind,
    get_policy_capabilities,
    get_policy_metadata,
    unwrap_inference_model,
)

_JAX_STATIC_ONLY_ENV_KEYS = {
    "enable_rebounds",
    "rebound_table_model_dir",
    "rebound_target_temperature",
    "rebound_target_uniform_mix",
    "rebound_winner_distance_weight",
    "rebound_winner_temperature",
    "rebound_skill_std",
    "rebound_skill_weight",
    "rebound_contest_mode",
    "rebound_contest_radius",
    "rebound_obs_top_n_targets",
    "offensive_rebound_shot_clock_reset",
    "rebound_terminal_reward_mode",
}


_JAX_STATIC_ONLY_ENV_DEFAULTS = {
    "enable_rebounds": False,
    "rebound_table_model_dir": "",
    "rebound_target_temperature": 1.0,
    "rebound_target_uniform_mix": 0.0,
    "rebound_winner_distance_weight": 1.0,
    "rebound_winner_temperature": 1.0,
    "rebound_skill_std": 0.0,
    "rebound_skill_weight": 0.0,
    "rebound_contest_mode": "global_contest",
    "rebound_contest_radius": 1,
    "rebound_obs_top_n_targets": 0,
    "offensive_rebound_shot_clock_reset": 14,
    "rebound_terminal_reward_mode": "actual_points",
}

_JAX_STATIC_ONLY_ENV_CASTS = {
    "enable_rebounds": "bool",
    "rebound_table_model_dir": "str",
    "rebound_target_temperature": "float",
    "rebound_target_uniform_mix": "float",
    "rebound_winner_distance_weight": "float",
    "rebound_winner_temperature": "float",
    "rebound_skill_std": "float",
    "rebound_skill_weight": "float",
    "rebound_contest_mode": "str",
    "rebound_contest_radius": "int",
    "rebound_obs_top_n_targets": "int",
    "offensive_rebound_shot_clock_reset": "int",
    "rebound_terminal_reward_mode": "str",
}


def _canonical_rebound_contest_mode(value: Any) -> str:
    raw = str(value or "global_contest").strip().lower().replace("-", "_")
    if raw in {"local", "local_contest"}:
        return "local_contest"
    return "global_contest"


def _rebound_contest_mode_from_static(static: Any) -> str:
    try:
        mode_id = int(np.asarray(static.rebound_contest_mode).reshape(-1)[0])
    except Exception:
        return "global_contest"
    return "local_contest" if mode_id == int(REBOUND_CONTEST_MODE_LOCAL) else "global_contest"


def _int_from_static_field(static: Any, field_name: str, fallback: int) -> int:
    try:
        return int(np.asarray(getattr(static, field_name)).reshape(-1)[0])
    except Exception:
        return int(fallback)


def _infer_attention_observation_dims(
    current_dim: int,
    *,
    expected_dim: int,
    token_count: int,
    token_dim: int,
    global_dim: int,
) -> tuple[int, int] | None:
    if token_count <= 0:
        return None
    max_token_dim = max(1, (int(current_dim) - 1) // int(token_count))
    candidates: list[tuple[tuple[int, int, int], int, int]] = []
    for current_token_dim in range(1, max_token_dim + 1):
        current_global_dim = int(current_dim) - 1 - (int(token_count) * int(current_token_dim))
        if current_global_dim < 0:
            continue
        if current_dim >= expected_dim:
            if current_token_dim < token_dim or current_global_dim < global_dim:
                continue
            # Prefer interpreting extras as a small number of added globals, then
            # added per-token fields. This handles 10*18+8+1 -> 10*17+7+1.
            score = (current_global_dim - global_dim, current_token_dim - token_dim, -current_token_dim)
        else:
            if current_token_dim > token_dim or current_global_dim > global_dim:
                continue
            # When padding for older checkpoints, preserve as much current
            # structure as possible.
            score = (token_dim - current_token_dim, global_dim - current_global_dim, -current_token_dim)
        candidates.append((score, current_token_dim, current_global_dim))
    if not candidates:
        return None
    _score, current_token_dim, current_global_dim = min(candidates, key=lambda item: item[0])
    return int(current_token_dim), int(current_global_dim)


def _adapt_attention_observation_to_spec(flat_obs, spec, jnp):
    expected_dim = int(spec.flat_obs_dim)
    current_dim = int(flat_obs.shape[-1])
    token_count = int(spec.token_player_count)
    token_dim = int(spec.token_dim)
    global_dim = int(spec.global_dim)
    dims = _infer_attention_observation_dims(
        current_dim,
        expected_dim=expected_dim,
        token_count=token_count,
        token_dim=token_dim,
        global_dim=global_dim,
    )
    if dims is None:
        raise ValueError(f"Cannot adapt attention observation dim {current_dim} to checkpoint dim {expected_dim}.")
    current_token_dim, current_global_dim = dims
    if current_token_dim <= 0 or current_global_dim < 0:
        raise ValueError(f"Cannot adapt attention observation dim {current_dim} to checkpoint dim {expected_dim}.")

    players = flat_obs[:, : token_count * current_token_dim].reshape(
        flat_obs.shape[0], token_count, current_token_dim
    )
    if current_token_dim >= token_dim:
        players = players[:, :, :token_dim]
    else:
        pad = jnp.zeros((flat_obs.shape[0], token_count, token_dim - current_token_dim), dtype=flat_obs.dtype)
        players = jnp.concatenate([players, pad], axis=-1)

    global_start = token_count * current_token_dim
    globals_vec = flat_obs[:, global_start : global_start + current_global_dim]
    if current_global_dim >= global_dim:
        globals_vec = globals_vec[:, :global_dim]
    else:
        pad = jnp.zeros((flat_obs.shape[0], global_dim - current_global_dim), dtype=flat_obs.dtype)
        globals_vec = jnp.concatenate([globals_vec, pad], axis=-1)

    role_start = global_start + current_global_dim
    role_flag = flat_obs[:, role_start : role_start + 1]
    if role_flag.shape[-1] < 1:
        role_flag = jnp.zeros((flat_obs.shape[0], 1), dtype=flat_obs.dtype)

    return jnp.concatenate(
        [players.reshape(flat_obs.shape[0], token_count * token_dim), globals_vec, role_flag],
        axis=1,
    ).astype(jnp.float32)


def _adapt_policy_observation_to_spec(flat_obs, static, spec, jnp):
    if not hasattr(spec, "flat_obs_dim"):
        return flat_obs
    expected_dim = int(spec.flat_obs_dim)
    current_dim = int(flat_obs.shape[-1])
    if current_dim == expected_dim:
        return flat_obs
    if str(spec.model_type) == "attention":
        return _adapt_attention_observation_to_spec(flat_obs, spec, jnp)
    if current_dim < expected_dim:
        raise ValueError(f"Policy observation dim {current_dim} is smaller than checkpoint dim {expected_dim}.")
    n_players = int(static.role_encoding.shape[0])
    offense_count = int(static.offense_ids.shape[0])
    tail_dim = 4 + 1 + (3 * offense_count)
    if current_dim - expected_dim == n_players and current_dim > (n_players + tail_dim):
        remove_start = current_dim - tail_dim - n_players
        return jnp.concatenate([flat_obs[:, :remove_start], flat_obs[:, remove_start + n_players :]], axis=1).astype(jnp.float32)
    return flat_obs[:, :expected_dim].astype(jnp.float32)


_SELECTOR_METADATA_PRIORITY_KEYS = {
    "intent_selector_enabled",
    "intent_selector_mode",
    "intent_selector_multiselect_enabled",
    "intent_selector_min_play_steps",
    "intent_selector_hidden_dim",
    "intent_selector_value_coef",
}


def _as_int(value) -> int:
    return int(np.asarray(value).reshape(-1)[0])


def _as_float(value) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def _as_bool(value) -> bool:
    return bool(_as_int(value))


def _field0(state, name: str):
    return np.asarray(getattr(state, name))[0]


def _param_lookup(params: dict[str, Any] | None, key: str, default: Any = None) -> Any:
    if not isinstance(params, dict):
        return default
    if key in params:
        return params[key]
    prefixed = f"jax/{key}"
    if prefixed in params:
        return params[prefixed]
    return default


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "t", "on"}
    return bool(value)


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)



def _coerce_runtime_static_value(key: str, value: Any) -> Any:
    kind = _JAX_STATIC_ONLY_ENV_CASTS.get(key)
    if kind == "bool":
        return _coerce_bool(value)
    if kind == "float":
        return _coerce_float(value)
    if kind == "int":
        return _coerce_int(value)
    if kind == "str":
        return str(value)
    return value


def _jax_static_env_params_from_policy(policy_obj: Any) -> dict[str, Any]:
    """Extract JAX kernel-only env attrs from checkpoint metadata.

    These fields are intentionally not passed to the legacy Python env
    constructor, but the JAX kernel reads them from display-env attributes.
    """
    metadata = get_policy_metadata(policy_obj) or {}
    sources = (
        metadata.get("frozen_config"),
        metadata.get("env_config"),
        metadata.get("trainer_config"),
    )
    out: dict[str, Any] = {}
    if any(isinstance(source, dict) for source in sources[:2]):
        out.update(_JAX_STATIC_ONLY_ENV_DEFAULTS)
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in _JAX_STATIC_ONLY_ENV_KEYS:
            if key in source and source[key] not in (None, ""):
                out[key] = _coerce_runtime_static_value(key, source[key])
    return out

def _coerce_play_name_map(metadata: dict[str, Any] | None, num_intents: int) -> dict[str, str]:
    if not isinstance(metadata, dict):
        return {}
    raw = metadata.get("play_name_map")
    if isinstance(raw, dict):
        out = {}
        for key, value in raw.items():
            try:
                idx = int(key)
            except Exception:
                continue
            if 0 <= idx < int(num_intents):
                label = str(value or "").strip()
                if label:
                    out[str(idx)] = label
        if out:
            return out
    play_meta = metadata.get("play_name_metadata")
    if isinstance(play_meta, dict):
        out = {}
        for item in play_meta.get("play_names", []) or []:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("intent_index"))
            except Exception:
                continue
            if 0 <= idx < int(num_intents):
                label = str(item.get("play_name") or "").strip()
                if label:
                    out[str(idx)] = label
        return out
    return {}


@dataclass
class _TeamPolicyOutput:
    actions: np.ndarray
    probs_by_player: dict[int, list[float]]
    values: float
    attention_weights: np.ndarray | None
    selector_logits: np.ndarray | None = None
    selector_values: np.ndarray | None = None


class JaxDevRuntime:
    """Interactive backend runtime that steps the JAX environment state directly.

    A Python HexagonBasketballEnv is still used as a static/config/display template,
    but action transitions go through basketworld_jax.env.minimal.
    """

    def __init__(
        self,
        *,
        required_params: dict[str, Any],
        env_params: dict[str, Any],
        unified_policy: Any,
        opponent_policy: Any | None,
        user_team: Team,
        role_flag_offense: float = 1.0,
        role_flag_defense: float = -1.0,
        rng_seed: int | None = None,
    ) -> None:
        self.unified_policy = unified_policy
        self.opponent_policy = opponent_policy
        self.user_team = user_team
        self.role_flag_offense = float(role_flag_offense)
        self.role_flag_defense = float(role_flag_defense)
        self.required_params = copy.deepcopy(required_params)
        self.env_params = copy.deepcopy(env_params)
        display_env_params = {
            key: value
            for key, value in self.env_params.items()
            if key not in _JAX_STATIC_ONLY_ENV_KEYS
        }
        self.display_env = HexagonBasketballEnv(
            **self.required_params,
            **display_env_params,
            render_mode="rgb_array",
        )
        self._apply_jax_static_env_attrs()

        raw_model = unwrap_inference_model(unified_policy)
        if raw_model is None or not hasattr(raw_model, "jax"):
            raise TypeError("JAX dev runtime requires a loaded JAX inference model.")
        self.raw_model = raw_model
        self.jax = raw_model.jax
        self.jnp = raw_model.jnp
        self.static = build_kernel_static_from_env(self.display_env, self.jnp)
        self.state = None
        self.last_step_output = None
        self.last_action_results = self._empty_action_results()
        self.episode_rebounds: list[dict[str, Any]] = []
        if rng_seed is None:
            rng_seed = int(np.random.default_rng().integers(0, 2**31 - 1))
        self._rng_seed = int(rng_seed)
        self._rng_key = self.jax.random.PRNGKey(self._rng_seed)
        self._last_policy_probs: dict[int, list[float]] | None = None
        self._last_attention_payload: dict[str, Any] | None = None
        self._last_attention_payloads: dict[str, dict[str, Any] | None] = {
            "offense": None,
            "defense": None,
        }
        self._last_completed_pass_boundary = False
        self._last_selector_transition: dict[str, Any] | None = None

    def _apply_jax_static_env_attrs(self) -> None:
        for key in _JAX_STATIC_ONLY_ENV_KEYS:
            if key in self.env_params:
                setattr(self.display_env, key, self.env_params[key])


    @property
    def n_players(self) -> int:
        return int(np.asarray(self.static.role_encoding).shape[0])

    @property
    def offense_ids(self) -> list[int]:
        return [int(v) for v in np.asarray(self.static.offense_ids).reshape(-1).tolist()]

    @property
    def defense_ids(self) -> list[int]:
        return [int(v) for v in np.asarray(self.static.defense_ids).reshape(-1).tolist()]

    def _next_key(self):
        self._rng_key, key = self.jax.random.split(self._rng_key)
        return key

    def _clear_attention_payload_cache(self) -> None:
        self._last_attention_payload = None
        self._last_attention_payloads = {"offense": None, "defense": None}

    def _clone_jax_tree(self, value):
        if value is None:
            return None

        def _clone_leaf(leaf):
            if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
                return self.jnp.asarray(np.array(self.jax.device_get(leaf)))
            return copy.deepcopy(leaf)

        return self.jax.tree_util.tree_map(_clone_leaf, value)

    def capture_snapshot(self) -> dict[str, Any]:
        """Capture the canonical JAX runtime branch for counterfactual restore/replay."""
        opponent_raw = unwrap_inference_model(self.opponent_policy)
        return {
            "state": self._clone_jax_tree(self.state),
            "rng_seed": int(self._rng_seed),
            "rng_key": self._clone_jax_tree(self._rng_key),
            "raw_model_sample_key": self._clone_jax_tree(
                getattr(self.raw_model, "_sample_key", None)
            ),
            "opponent_model_sample_key": (
                self._clone_jax_tree(getattr(opponent_raw, "_sample_key", None))
                if opponent_raw is not None and opponent_raw is not self.raw_model
                else None
            ),
            "display_env": copy.deepcopy(self.display_env),
            "last_step_output": self._clone_jax_tree(self.last_step_output),
            "last_action_results": copy.deepcopy(self.last_action_results),
            "episode_rebounds": copy.deepcopy(self.episode_rebounds),
            "last_policy_probs": copy.deepcopy(self._last_policy_probs),
            "last_attention_payload": copy.deepcopy(self._last_attention_payload),
            "last_attention_payloads": copy.deepcopy(self._last_attention_payloads),
            "last_completed_pass_boundary": bool(self._last_completed_pass_boundary),
            "last_selector_transition": copy.deepcopy(self._last_selector_transition),
        }

    def restore_snapshot(self, snapshot: dict[str, Any], *, game_state: Any | None = None) -> None:
        """Restore a snapshot produced by `capture_snapshot`."""
        if not isinstance(snapshot, dict) or snapshot.get("state") is None:
            raise ValueError("Invalid JAX runtime snapshot.")
        self.state = self._clone_jax_tree(snapshot["state"])
        self._rng_seed = int(snapshot.get("rng_seed", self._rng_seed))
        if snapshot.get("rng_key") is not None:
            self._rng_key = self._clone_jax_tree(snapshot["rng_key"])
        if snapshot.get("raw_model_sample_key") is not None and hasattr(self.raw_model, "_sample_key"):
            self.raw_model._sample_key = self._clone_jax_tree(snapshot["raw_model_sample_key"])
        opponent_raw = unwrap_inference_model(self.opponent_policy)
        if (
            opponent_raw is not None
            and opponent_raw is not self.raw_model
            and snapshot.get("opponent_model_sample_key") is not None
            and hasattr(opponent_raw, "_sample_key")
        ):
            opponent_raw._sample_key = self._clone_jax_tree(snapshot["opponent_model_sample_key"])
        if snapshot.get("display_env") is not None:
            self.display_env = copy.deepcopy(snapshot["display_env"])
        self.last_step_output = self._clone_jax_tree(snapshot.get("last_step_output"))
        self.last_action_results = copy.deepcopy(
            snapshot.get("last_action_results") or self._empty_action_results()
        )
        self.episode_rebounds = copy.deepcopy(snapshot.get("episode_rebounds") or [])
        self._last_policy_probs = copy.deepcopy(snapshot.get("last_policy_probs"))
        self._last_attention_payload = copy.deepcopy(snapshot.get("last_attention_payload"))
        payloads = snapshot.get("last_attention_payloads")
        if isinstance(payloads, dict):
            self._last_attention_payloads = {
                "offense": copy.deepcopy(payloads.get("offense")),
                "defense": copy.deepcopy(payloads.get("defense")),
            }
        else:
            self._last_attention_payloads = {
                "offense": copy.deepcopy(self._last_attention_payload),
                "defense": None,
            }
        self._last_completed_pass_boundary = bool(snapshot.get("last_completed_pass_boundary", False))
        self._last_selector_transition = copy.deepcopy(snapshot.get("last_selector_transition"))
        self._sync_display_env()
        if game_state is not None:
            game_state.env = self.display_env
            game_state.obs = self.observation_dict(observer_is_offense=game_state.user_team != Team.DEFENSE)
            game_state.prev_obs = None
            self._capture_turn_start(game_state)

    def refresh_static_from_display_env(self) -> None:
        """Refresh immutable JAX kernel config after live display-env edits."""
        self._apply_jax_static_env_attrs()
        self.static = build_kernel_static_from_env(self.display_env, self.jnp)
        self._last_policy_probs = None
        self._clear_attention_payload_cache()
        if hasattr(self, "_playbook_batch_runner_cache"):
            self._playbook_batch_runner_cache = None

    def replace_policies(
        self,
        *,
        unified_policy: Any,
        opponent_policy: Any | None,
        game_state: Any | None = None,
    ) -> None:
        """Replace live inference policies after a UI checkpoint swap."""
        raw_model = unwrap_inference_model(unified_policy)
        if raw_model is None or not hasattr(raw_model, "jax"):
            raise TypeError("JAX dev runtime requires a loaded JAX inference model.")

        self.unified_policy = unified_policy
        self.opponent_policy = opponent_policy
        self.raw_model = raw_model
        self.jax = raw_model.jax
        self.jnp = raw_model.jnp

        checkpoint_static_params = _jax_static_env_params_from_policy(unified_policy)
        if checkpoint_static_params:
            self.env_params.update(checkpoint_static_params)
            self.refresh_static_from_display_env()

        self._last_policy_probs = None
        self._clear_attention_payload_cache()
        self._last_selector_transition = None
        if hasattr(self, "_playbook_batch_runner_cache"):
            self._playbook_batch_runner_cache = None

        if game_state is not None:
            game_state.env = self.display_env
            if self.state is not None:
                game_state.obs = self.observation_dict(
                    observer_is_offense=game_state.user_team != Team.DEFENSE
                )
            game_state.prev_obs = None

    def reset(
        self,
        *,
        seed: int | None = None,
        template_id: str | None = None,
        template_mirrored: bool | None = None,
    ) -> dict[str, Any]:
        key = self.jax.random.PRNGKey(int(seed or 0)) if seed is not None else self._next_key()
        self.state = reset_batch_minimal(self.static, self.jnp.asarray([key]), self.jax, self.jnp)
        template_metadata = self._apply_forced_template(
            template_id=template_id,
            template_mirrored=template_mirrored,
            seed=seed,
        )
        self.last_step_output = None
        self.last_action_results = self._empty_action_results()
        self.episode_rebounds = []
        self._last_policy_probs = None
        self._clear_attention_payload_cache()
        self._last_completed_pass_boundary = False
        self._last_selector_transition = None
        self._sync_display_env()
        return {"start_template": template_metadata} if template_metadata else {}

    def _playable_skill_updates(self, offense_skills: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(offense_skills, dict):
            return {}
        offense_ids = np.asarray(self.offense_ids, dtype=np.int32)
        expected = int(offense_ids.size)
        field_map = {
            "layup": "layup_pct",
            "three_pt": "three_pt_pct",
            "dunk": "dunk_pct",
        }
        updates: dict[str, Any] = {}
        for skill_key, field_name in field_map.items():
            raw_values = offense_skills.get(skill_key)
            if raw_values is None:
                continue
            values = np.asarray(raw_values, dtype=np.float32).reshape(-1)
            if values.size != expected:
                continue
            current = np.asarray(_field0(self.state, field_name), dtype=np.float32).copy()
            current[offense_ids] = values
            updates[field_name] = self.jnp.asarray(current[None, ...], dtype=self.jnp.float32)
        return updates

    def _apply_selector_episode_start(self, game_state: Any) -> dict[str, Any] | None:
        game_state.selector_segment_index = 0
        game_state.selector_last_boundary_reason = None
        if not self._selector_runtime_enabled(game_state):
            return None
        selection = self._sample_selector_intent(game_state)
        if selection is None:
            return None
        commitment_steps = max(1, _as_int(self.static.intent_commitment_steps))
        self.set_offense_intent_state(
            active=True,
            intent_index=int(selection["intent_index"]),
            intent_age=0,
            intent_commitment_remaining=commitment_steps,
            game_state=game_state,
        )
        transition = {
            "reason": "episode_start",
            "previous_intent_index": None,
            "intent_index": int(selection["intent_index"]),
            "changed_intent": None,
            "used_selector": bool(selection.get("used_selector", False)),
            "source": "learned_selector" if bool(selection.get("used_selector", False)) else "uniform_fallback",
            "alpha": float(selection.get("alpha", 0.0)),
            "eps": float(selection.get("eps", 0.0)),
            "value": selection.get("value"),
        }
        self._last_selector_transition = transition
        game_state.selector_last_boundary_reason = "episode_start"
        return dict(transition)

    def reset_playable_possession(
        self,
        *,
        game_state: Any,
        user_team: Team,
        offense_skills: dict[str, Any] | None = None,
        shot_clock: int = 24,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Reset the authoritative JAX state for playable-mode possession changes."""
        self.user_team = user_team
        self.reset(seed=seed)
        updates: dict[str, Any] = {
            "shot_clock": self.jnp.asarray([int(shot_clock)], dtype=self.jnp.int32),
        }
        updates.update(self._playable_skill_updates(offense_skills))
        self.state = self.state._replace(**updates)
        self.last_step_output = None
        self.last_action_results = self._empty_action_results()
        self.episode_rebounds = []
        self._last_policy_probs = None
        self._clear_attention_payload_cache()
        self._last_completed_pass_boundary = False
        self._last_selector_transition = None
        self._sync_display_env()
        self.display_env.training_team = user_team
        self.display_env.shot_clock_steps = int(shot_clock)
        self.display_env.min_shot_clock = int(shot_clock)

        game_state.user_team = user_team
        game_state.env = self.display_env
        game_state.obs = self.observation_dict(observer_is_offense=user_team != Team.DEFENSE)
        game_state.prev_obs = None
        self._apply_selector_episode_start(game_state)
        self._sync_display_env()
        game_state.obs = self.observation_dict(observer_is_offense=user_team != Team.DEFENSE)
        self._capture_turn_start(game_state)
        return self.get_full_game_state(
            game_state,
            include_policy_probs=True,
            include_action_values=True,
            include_state_values=True,
        )

    def _apply_forced_template(
        self,
        *,
        template_id: str | None,
        template_mirrored: bool | None,
        seed: int | None,
    ) -> dict[str, Any] | None:
        if not template_id:
            return None
        library = self.env_params.get("start_template_library")
        if not isinstance(library, dict):
            raise ValueError("No start-template library is loaded for this JAX runtime.")
        template = next(
            (
                dict(item)
                for item in list(library.get("templates") or [])
                if str(item.get("id", "")).strip() == str(template_id).strip()
            ),
            None,
        )
        if template is None:
            raise ValueError(f"Unknown start template: {template_id}")

        original_rng = getattr(self.display_env, "_rng", None)
        if seed is not None:
            self.display_env._rng = np.random.default_rng(int(seed))
        try:
            resolved = resolve_start_template(
                self.display_env,
                template,
                jitter_scale=float(self.env_params.get("start_template_jitter_scale", 1.0)),
                mirror=bool(template_mirrored) and bool(template.get("mirrorable", False)),
            )
        finally:
            if seed is not None:
                self.display_env._rng = original_rng

        positions = np.asarray(resolved["initial_positions"], dtype=np.int32).reshape(self.n_players, 2)
        ball_holder = int(resolved["ball_holder"]) if resolved.get("ball_holder") is not None else -1
        shot_clock = int(resolved["shot_clock"]) if resolved.get("shot_clock") is not None else None
        updates = {
            "positions": self.jnp.asarray(positions[None, ...], dtype=self.jnp.int32),
            "ball_holder": self.jnp.asarray([ball_holder], dtype=self.jnp.int32),
        }
        if shot_clock is not None and shot_clock >= 0:
            updates["shot_clock"] = self.jnp.asarray([shot_clock], dtype=self.jnp.int32)
        self.state = self.state._replace(**updates)
        return {
            "template_id": str(resolved.get("template_id") or template_id),
            "mirrored": bool(resolved.get("mirrored", bool(template_mirrored))),
        }

    def observation_dict(self, *, observer_is_offense: bool = True) -> dict[str, Any]:
        if self.state is None:
            raise RuntimeError("JAX runtime is not initialized.")
        role_flag = self.role_flag_offense if observer_is_offense else self.role_flag_defense
        obs = build_policy_observation_batch_with_role_flag(
            self.static,
            self.state,
            role_flag,
            self.jnp,
            model_type=str(self.raw_model.spec.model_type),
        )
        obs = _adapt_policy_observation_to_spec(obs, self.static, self.raw_model.spec, self.jnp)
        players, globals_vec, _ = build_token_observation_components_batch(
            self.static,
            self.state,
            role_flag,
            self.jnp,
        )
        skills = self.jnp.stack(
            [
                self.state.layup_pct[:, self.static.offense_ids] - self.static.base_layup_pct,
                self.state.three_pt_pct[:, self.static.offense_ids] - self.static.base_three_pt_pct,
                self.state.dunk_pct[:, self.static.offense_ids] - self.static.base_dunk_pct,
            ],
            axis=-1,
        ).reshape((1, -1))
        return {
            "obs": np.asarray(self.jax.device_get(obs[0]), dtype=np.float32),
            "action_mask": self.action_mask(),
            "role_flag": np.asarray([role_flag], dtype=np.float32),
            "skills": np.asarray(self.jax.device_get(skills[0]), dtype=np.float32),
            "players": np.asarray(self.jax.device_get(players[0]), dtype=np.float32),
            "globals": np.asarray(self.jax.device_get(globals_vec[0]), dtype=np.float32),
        }

    def action_mask(self) -> np.ndarray:
        mask = build_action_masks_batch(self.static, self.state, self.jnp)
        return np.asarray(self.jax.device_get(mask[0]), dtype=np.int8)

    def _team_policy_output(self, policy: Any, *, observer_is_offense: bool, deterministic: bool) -> _TeamPolicyOutput:
        raw = unwrap_inference_model(policy)
        if raw is None or not hasattr(raw, "_masked_runner"):
            return _TeamPolicyOutput(
                actions=np.zeros(self.n_players, dtype=np.int32),
                probs_by_player={},
                values=0.0,
                attention_weights=None,
            )
        role_flag = self.role_flag_offense if observer_is_offense else self.role_flag_defense
        flat_obs = build_policy_observation_batch_with_role_flag(
            self.static,
            self.state,
            role_flag,
            self.jnp,
            model_type=str(raw.spec.model_type),
        )
        flat_obs = _adapt_policy_observation_to_spec(flat_obs, self.static, raw.spec, self.jnp)
        full_action_mask = build_action_masks_batch(self.static, self.state, self.jnp)
        team_ids_device = self.static.offense_ids if observer_is_offense else self.static.defense_ids
        team_ids = [int(v) for v in np.asarray(team_ids_device).reshape(-1).tolist()]
        team_action_mask = self.jnp.take(full_action_mask, team_ids_device, axis=1)
        intent_context = build_policy_intent_context_batch_with_role_flag(
            self.static,
            self.state,
            role_flag,
            self.jnp,
        )
        masked_out = raw._masked_runner(
            raw.params,
            flat_obs,
            team_action_mask,
            intent_context,
        )
        if deterministic:
            team_actions_device = masked_out["deterministic_actions"]
        else:
            raw._sample_key, sample_key = raw.jax.random.split(raw._sample_key)
            team_actions_device = raw.jax.random.categorical(
                sample_key,
                masked_out["masked_logits"],
                axis=-1,
            ).astype(raw.jnp.int32)
        team_actions = np.asarray(raw.jax.device_get(team_actions_device[0]), dtype=np.int32)
        probs = np.asarray(raw.jax.device_get(masked_out["probs"][0]), dtype=np.float32)
        full_actions = np.zeros(self.n_players, dtype=np.int32)
        probs_by_player: dict[int, list[float]] = {}
        for idx, pid in enumerate(team_ids):
            full_actions[int(pid)] = int(team_actions[idx])
            probs_by_player[int(pid)] = probs[idx].astype(float).tolist()
        values = np.asarray(raw.jax.device_get(masked_out.get("values", [0.0])), dtype=np.float32).reshape(-1)
        attention = masked_out.get("attention_weights")
        attention_np = (
            np.asarray(raw.jax.device_get(attention[0]), dtype=np.float32)
            if attention is not None
            else None
        )
        return _TeamPolicyOutput(
            actions=full_actions,
            probs_by_player=probs_by_player,
            values=float(values[0]) if values.size else 0.0,
            attention_weights=attention_np,
        )

    def _choose_joint_policy_actions(self, *, player_deterministic: bool, opponent_deterministic: bool):
        if self.user_team == Team.OFFENSE:
            offense_policy = self.unified_policy
            offense_det = bool(player_deterministic)
            defense_policy = self.opponent_policy or self.unified_policy
            defense_det = bool(opponent_deterministic) if self.opponent_policy is not None else bool(player_deterministic)
        else:
            offense_policy = self.opponent_policy or self.unified_policy
            offense_det = bool(opponent_deterministic) if self.opponent_policy is not None else bool(player_deterministic)
            defense_policy = self.unified_policy
            defense_det = bool(player_deterministic)

        offense_out = self._team_policy_output(
            offense_policy,
            observer_is_offense=True,
            deterministic=offense_det,
        )
        defense_out = self._team_policy_output(
            defense_policy,
            observer_is_offense=False,
            deterministic=defense_det,
        )
        full_actions = np.zeros(self.n_players, dtype=np.int32)
        for pid in self.offense_ids:
            full_actions[pid] = offense_out.actions[pid]
        for pid in self.defense_ids:
            full_actions[pid] = defense_out.actions[pid]
        probs = {**offense_out.probs_by_player, **defense_out.probs_by_player}
        self._last_policy_probs = probs
        offense_attention = self._attention_payload_from_weights(offense_out.attention_weights, True)
        defense_attention = self._attention_payload_from_weights(defense_out.attention_weights, False)
        self._last_attention_payloads = {
            "offense": offense_attention,
            "defense": defense_attention,
        }
        self._last_attention_payload = offense_attention
        return full_actions, probs

    def _normalize_overrides(self, raw_actions: Any) -> tuple[dict[int, int], dict[int, dict[str, Any]]]:
        overrides: dict[int, int] = {}
        meta: dict[int, dict[str, Any]] = {}
        if not isinstance(raw_actions, dict):
            return overrides, meta
        for key, value in raw_actions.items():
            try:
                pid = int(key)
            except Exception:
                continue
            if pid < 0 or pid >= self.n_players:
                continue
            if isinstance(value, dict):
                action_type = str(value.get("type", "")).upper()
                if action_type == "PASS" and "target" in value:
                    try:
                        target = int(value.get("target"))
                    except Exception:
                        continue
                    action_idx = self._pass_action_for_target(pid, target)
                    overrides[pid] = action_idx
                    meta[pid] = {"type": "PASS", "target": target}
                    continue
                value = value.get("action", value.get("selected_action", value))
            try:
                overrides[pid] = int(ActionType[str(value).strip().upper()].value) if isinstance(value, str) else int(value)
            except Exception:
                continue
        return overrides, meta

    def _pass_action_for_target(self, passer_id: int, target_id: int) -> int:
        target_ids = np.asarray(self.static.pointer_pass_target_ids, dtype=np.int32)
        if passer_id < 0 or passer_id >= target_ids.shape[0]:
            return int(ActionType.NOOP.value)
        matches = np.where(target_ids[passer_id] == int(target_id))[0]
        if matches.size == 0:
            return int(ActionType.NOOP.value)
        return int(PASS_ACTION_START + int(matches[0]))

    def step(self, request: Any, game_state: Any) -> dict[str, Any]:
        if self.state is None:
            raise RuntimeError("JAX runtime is not initialized.")
        if bool(getattr(request, "use_mcts", False)):
            raise ValueError("MCTS is not supported by the JAX-native dev runtime.")

        selector_transition = self._maybe_apply_selector_boundary(game_state)
        player_det = True if request.player_deterministic is None else bool(request.player_deterministic)
        opponent_det = True if request.opponent_deterministic is None else bool(request.opponent_deterministic)
        full_actions, _ = self._choose_joint_policy_actions(
            player_deterministic=player_det,
            opponent_deterministic=opponent_det,
        )
        overrides, action_meta = self._normalize_overrides(getattr(request, "actions", {}))
        action_mask = self.action_mask()
        for pid, action_idx in overrides.items():
            if 0 <= action_idx < action_mask.shape[1] and int(action_mask[pid, action_idx]) == 1:
                full_actions[pid] = int(action_idx)
            else:
                full_actions[pid] = int(ActionType.NOOP.value)

        pre_values = self.state_values()
        prev_state = self.state
        step_key = self._next_key()
        out = step_batch_minimal(
            self.static,
            self.state,
            self.jnp.asarray(full_actions[None, :], dtype=self.jnp.int32),
            self.jnp.asarray([step_key]),
            self.jax,
            self.jnp,
        )
        self.state = out.state
        self.last_step_output = out
        self.last_action_results = self._action_results_from_step(prev_state, out)
        if self.last_action_results.get("rebounds"):
            self.episode_rebounds.extend(copy.deepcopy(self.last_action_results["rebounds"]))
        self._last_completed_pass_boundary = bool(_as_bool(out.completed_pass[0]) and not _as_bool(out.done[0]))
        self._last_policy_probs = None
        self._clear_attention_payload_cache()
        self._sync_display_env()
        game_state.obs = self.observation_dict(observer_is_offense=game_state.user_team != Team.DEFENSE)
        game_state.prev_obs = None
        game_state.actions_log.append([int(v) for v in full_actions.tolist()])

        rewards = np.asarray(self.jax.device_get(out.rewards[0]), dtype=np.float32)
        step_rewards = {
            "offense": float(np.sum(rewards[self.offense_ids])),
            "defense": float(np.sum(rewards[self.defense_ids])),
        }
        step_idx = len(game_state.reward_history) + 1
        ep_by_player = self.expected_points()
        ball_handler = self.ball_holder
        is_terminal = bool(_as_bool(out.done[0]))
        phi_r_shape = _as_float(out.phi_r_shape[0])
        phi_prev = _as_float(out.phi_prev[0])
        phi_next = _as_float(out.phi_next[0])
        phi_beta = _as_float(out.phi_beta[0])
        team_best_ep, ball_handler_ep = self._phi_ep_summary(ep_by_player, ball_handler)
        game_state.episode_rewards["offense"] += step_rewards["offense"]
        game_state.episode_rewards["defense"] += step_rewards["defense"]
        game_state.reward_history.append(
            {
                "step": step_idx,
                "offense": step_rewards["offense"],
                "defense": step_rewards["defense"],
                "offense_reason": self._reward_reason(),
                "defense_reason": self._reward_reason(defense=True),
                "phi_r_shape": phi_r_shape,
                "phi_prev": phi_prev,
                "phi_next": phi_next,
                "phi_beta": phi_beta,
                "ep_by_player": ep_by_player,
                "ball_handler": int(ball_handler) if ball_handler is not None else -1,
                "offense_ids": self.offense_ids,
                "is_terminal": is_terminal,
                "shot_clock": self.shot_clock,
            }
        )
        game_state.phi_log.append(
            {
                "step": step_idx,
                "phi_prev": phi_prev,
                "phi_next": phi_next,
                "phi_beta": phi_beta,
                "phi_r_shape": phi_r_shape,
                "ball_handler": int(ball_handler) if ball_handler is not None else -1,
                "offense_ids": self.offense_ids,
                "defense_ids": self.defense_ids,
                "shot_clock": self.shot_clock,
                "is_terminal": is_terminal,
                "ep_by_player": ep_by_player,
                "team_best_ep": team_best_ep,
                "ball_handler_ep": ball_handler_ep,
            }
        )
        self._append_shot_log_if_needed(game_state)
        self._capture_turn_start(game_state)

        actions_taken, actions_taken_meta = self._actions_taken_payload(full_actions, action_meta)
        state_payload = self.get_full_game_state(
            game_state,
            include_policy_probs=True,
            include_action_values=True,
            include_state_values=True,
        )
        state_payload["actions_taken"] = actions_taken
        state_payload["actions_taken_meta"] = actions_taken_meta
        game_state.episode_states.append(dict(state_payload))
        if bool(state_payload.get("done")):
            game_state.self_play_active = False

        return {
            "status": "success",
            "state": state_payload,
            "actions_taken": actions_taken,
            "actions_taken_meta": actions_taken_meta,
            "step_rewards": step_rewards,
            "episode_rewards": {
                "offense": float(game_state.episode_rewards["offense"]),
                "defense": float(game_state.episode_rewards["defense"]),
            },
            "pre_step_state_values": pre_values,
            "mcts": None,
            "selector_transition": selector_transition,
        }

    def start_self_play(self, request: Any, game_state: Any) -> dict[str, Any]:
        requested_seed = getattr(request, "template_seed", None)
        seed = (
            int(requested_seed)
            if requested_seed is not None
            else int(np.random.SeedSequence().entropy % (2**32 - 1))
        )
        template_id = str(getattr(request, "template_id", "") or "").strip()
        if template_id:
            meta = self.reset(
                seed=seed,
                template_id=template_id,
                template_mirrored=getattr(request, "template_mirrored", None),
            ).get("start_template")
        else:
            meta = None
            self._sync_display_env()
        if meta is not None and "source" not in meta:
            meta["source"] = getattr(game_state, "start_template_library_source", None)
        self._last_completed_pass_boundary = False
        self._last_selector_transition = None
        game_state.replay_seed = seed
        game_state.replay_initial_positions = [tuple(pos) for pos in self.positions]
        game_state.replay_ball_holder = self.ball_holder
        game_state.replay_shot_clock = self.shot_clock
        game_state.actions_log = []
        game_state.self_play_active = True
        game_state.selector_segment_index = 0
        game_state.selector_last_boundary_reason = None
        game_state.frames = []
        game_state.reward_history = []
        game_state.episode_rewards = {"offense": 0.0, "defense": 0.0}
        game_state.episode_states = []
        game_state.phi_log = []
        game_state.obs = self.observation_dict(observer_is_offense=game_state.user_team != Team.DEFENSE)
        self._append_initial_phi_log(game_state)
        self._capture_turn_start(game_state)
        state_payload = self.get_full_game_state(
            game_state,
            include_policy_probs=True,
            include_action_values=True,
            include_state_values=True,
        )
        game_state.episode_states.append(dict(state_payload))
        return {
            "status": "success",
            "state": state_payload,
            "seed": seed,
            "start_template": meta,
        }

    def reset_turn_state(self, game_state: Any) -> dict[str, Any]:
        if not game_state.turn_start_positions:
            raise ValueError("No turn snapshot available.")
        state_updates = {
            "positions": self.jnp.asarray(
                [game_state.turn_start_positions],
                dtype=self.jnp.int32,
            ),
            "ball_holder": self.jnp.asarray(
                [int(game_state.turn_start_ball_holder) if game_state.turn_start_ball_holder is not None else -1],
                dtype=self.jnp.int32,
            ),
        }
        if game_state.turn_start_shot_clock is not None:
            state_updates["shot_clock"] = self.jnp.asarray(
                [int(game_state.turn_start_shot_clock)],
                dtype=self.jnp.int32,
            )
        self.state = self.state._replace(**state_updates)
        self._last_completed_pass_boundary = False
        self._last_selector_transition = None
        self._sync_display_env()
        game_state.obs = self.observation_dict(observer_is_offense=game_state.user_team != Team.DEFENSE)
        game_state.prev_obs = None
        return {"status": "success", "state": self.get_full_game_state(game_state, include_policy_probs=True, include_state_values=True)}

    def apply_resolved_start_template(self, resolved: dict[str, Any], game_state: Any) -> None:
        if self.state is None:
            raise RuntimeError("JAX runtime is not initialized.")
        positions = np.asarray(
            resolved.get("initial_positions") or [],
            dtype=np.int32,
        ).reshape(self.n_players, 2)
        updates = {
            "positions": self.jnp.asarray(positions[None, ...], dtype=self.jnp.int32),
        }
        if resolved.get("ball_holder") is not None:
            updates["ball_holder"] = self.jnp.asarray(
                [int(resolved["ball_holder"])],
                dtype=self.jnp.int32,
            )
        if resolved.get("shot_clock") is not None:
            updates["shot_clock"] = self.jnp.asarray(
                [int(resolved["shot_clock"])],
                dtype=self.jnp.int32,
            )
        self.state = self.state._replace(**updates)
        self.last_step_output = None
        self.last_action_results = self._empty_action_results()
        self.episode_rebounds = []
        self._last_policy_probs = None
        self._clear_attention_payload_cache()
        self._last_completed_pass_boundary = False
        self._last_selector_transition = None
        self._sync_display_env()
        game_state.obs = self.observation_dict(observer_is_offense=game_state.user_team != Team.DEFENSE)
        game_state.prev_obs = None
        self._capture_turn_start(game_state)

    def apply_display_env_edits(self, game_state: Any) -> None:
        """Import live board edits from the display env into the JAX kernel state."""
        if self.state is None:
            raise RuntimeError("JAX runtime is not initialized.")
        env = self.display_env
        positions = np.asarray(getattr(env, "positions", []), dtype=np.int32).reshape(
            self.n_players,
            2,
        )
        ball_holder = getattr(env, "ball_holder", None)
        offense_ids = np.asarray(self.offense_ids, dtype=np.int32)

        def _skill_state(field_name: str, env_attr: str):
            current = np.asarray(_field0(self.state, field_name), dtype=np.float32).copy()
            raw_values = getattr(env, env_attr, None)
            if raw_values is None:
                return self.jnp.asarray(current[None, ...], dtype=self.jnp.float32)
            values = np.asarray(raw_values, dtype=np.float32).reshape(-1)
            if values.size != offense_ids.size:
                raise RuntimeError(
                    f"{env_attr} must contain {offense_ids.size} values, got {values.size}."
                )
            current[offense_ids] = values
            return self.jnp.asarray(current[None, ...], dtype=self.jnp.float32)

        updates = {
            "positions": self.jnp.asarray(positions[None, ...], dtype=self.jnp.int32),
            "ball_holder": self.jnp.asarray(
                [-1 if ball_holder is None else int(ball_holder)],
                dtype=self.jnp.int32,
            ),
            "shot_clock": self.jnp.asarray(
                [int(getattr(env, "shot_clock", self.shot_clock))],
                dtype=self.jnp.int32,
            ),
            "layup_pct": _skill_state("layup_pct", "offense_layup_pct_by_player"),
            "three_pt_pct": _skill_state("three_pt_pct", "offense_three_pt_pct_by_player"),
            "dunk_pct": _skill_state("dunk_pct", "offense_dunk_pct_by_player"),
        }
        self.state = self.state._replace(**updates)
        self._last_policy_probs = None
        self._clear_attention_payload_cache()
        self._sync_display_env()
        game_state.obs = self.observation_dict(observer_is_offense=game_state.user_team != Team.DEFENSE)
        game_state.prev_obs = None

    def set_offense_intent_state(
        self,
        *,
        active: bool,
        intent_index: int,
        intent_age: int,
        intent_commitment_remaining: int,
        game_state: Any,
    ) -> None:
        if self.state is None:
            raise RuntimeError("JAX runtime is not initialized.")
        self.state = self.state._replace(
            intent_active=self.jnp.asarray([1 if active else 0], dtype=self.jnp.int8),
            intent_index=self.jnp.asarray([int(intent_index)], dtype=self.jnp.int32),
            intent_age=self.jnp.asarray([int(intent_age)], dtype=self.jnp.int32),
            intent_commitment_remaining=self.jnp.asarray(
                [int(intent_commitment_remaining)],
                dtype=self.jnp.int32,
            ),
        )
        self._last_policy_probs = None
        self._clear_attention_payload_cache()
        self._last_completed_pass_boundary = False
        self._last_selector_transition = None
        self._sync_display_env()
        game_state.obs = self.observation_dict(observer_is_offense=game_state.user_team != Team.DEFENSE)
        game_state.prev_obs = None

    @property
    def positions(self) -> list[tuple[int, int]]:
        arr = np.asarray(self.jax.device_get(_field0(self.state, "positions")), dtype=np.int32)
        return [(int(q), int(r)) for q, r in arr.tolist()]

    @property
    def ball_holder(self) -> int | None:
        raw = _as_int(_field0(self.state, "ball_holder"))
        return raw if raw >= 0 else None

    @property
    def shot_clock(self) -> int:
        return _as_int(_field0(self.state, "shot_clock"))

    def _sync_display_env(self) -> None:
        env = self.display_env
        env.positions = self.positions
        env.ball_holder = self.ball_holder
        env.shot_clock = self.shot_clock
        env.step_count = _as_int(_field0(self.state, "step_count"))
        env.episode_ended = _as_bool(_field0(self.state, "episode_ended"))
        env.offense_score = _as_float(_field0(self.state, "offense_score"))
        env.defense_score = _as_float(_field0(self.state, "defense_score"))
        env.offense_layup_pct_by_player = [
            float(v) for v in np.asarray(_field0(self.state, "layup_pct"))[self.offense_ids].tolist()
        ]
        env.offense_three_pt_pct_by_player = [
            float(v) for v in np.asarray(_field0(self.state, "three_pt_pct"))[self.offense_ids].tolist()
        ]
        env.offense_dunk_pct_by_player = [
            float(v) for v in np.asarray(_field0(self.state, "dunk_pct"))[self.offense_ids].tolist()
        ]
        env._offensive_lane_steps = {
            int(pid): int(v)
            for pid, v in enumerate(np.asarray(_field0(self.state, "offense_lane_steps")).tolist())
            if int(v) > 0
        }
        env._defender_in_key_steps = {
            int(pid): int(v)
            for pid, v in enumerate(np.asarray(_field0(self.state, "defense_lane_steps")).tolist())
            if int(v) > 0
        }
        env.intent_index = _as_int(_field0(self.state, "intent_index"))
        env.intent_active = _as_bool(_field0(self.state, "intent_active"))
        env.intent_age = _as_int(_field0(self.state, "intent_age"))
        env.intent_commitment_remaining = _as_int(_field0(self.state, "intent_commitment_remaining"))
        env._intent_visible_to_defense = _as_bool(_field0(self.state, "intent_visible_to_defense"))
        env.defense_intent_index = _as_int(_field0(self.state, "defense_intent_index"))
        env.defense_intent_active = _as_bool(_field0(self.state, "defense_intent_active"))
        env.defense_intent_age = _as_int(_field0(self.state, "defense_intent_age"))
        env.defense_intent_commitment_remaining = _as_int(_field0(self.state, "defense_intent_commitment_remaining"))
        env.last_action_results = copy.deepcopy(self.last_action_results)

    def expected_points(self) -> list[float]:
        profile = build_shot_profile_batch(self.static, self.state, self.jnp)
        return [
            float(v)
            for v in np.asarray(self.jax.device_get(profile["expected_points"][0]), dtype=np.float32).tolist()
        ]

    def _phi_ep_summary(
        self,
        ep_by_player: list[float],
        ball_handler: int | None,
    ) -> tuple[float, float]:
        offense_ids = self.offense_ids
        team_eps = [
            float(ep_by_player[int(pid)])
            for pid in offense_ids
            if 0 <= int(pid) < len(ep_by_player)
        ]
        team_best_ep = max(team_eps) if team_eps else 0.0
        if ball_handler is not None and 0 <= int(ball_handler) < len(ep_by_player):
            ball_handler_ep = float(ep_by_player[int(ball_handler)])
        else:
            ball_handler_ep = 0.0
        return float(team_best_ep), float(ball_handler_ep)

    def _append_initial_phi_log(self, game_state: Any) -> None:
        ep_by_player = self.expected_points()
        ball_handler = self.ball_holder
        team_best_ep, ball_handler_ep = self._phi_ep_summary(ep_by_player, ball_handler)
        game_state.phi_log.append(
            {
                "step": 0,
                "phi_prev": 0.0,
                "phi_next": float(_as_float(_field0(self.state, "cached_phi"))),
                "phi_beta": float(np.asarray(self.static.phi_beta).reshape(-1)[0]),
                "phi_r_shape": 0.0,
                "ball_handler": int(ball_handler) if ball_handler is not None else -1,
                "offense_ids": self.offense_ids,
                "defense_ids": self.defense_ids,
                "shot_clock": self.shot_clock,
                "is_terminal": False,
                "ep_by_player": ep_by_player,
                "team_best_ep": team_best_ep,
                "ball_handler_ep": ball_handler_ep,
            }
        )

    def pass_steal_probabilities(self) -> dict[int, float]:
        if self.ball_holder is None:
            return {}
        probs = np.asarray(
            self.jax.device_get(build_pass_steal_probabilities_batch(self.static, self.state, self.jnp)[0]),
            dtype=np.float32,
        )
        out: dict[int, float] = {}
        for idx, pid in enumerate(self.offense_ids):
            if int(pid) != int(self.ball_holder):
                out[int(pid)] = float(probs[idx])
        return out

    def state_values(self) -> dict[str, float]:
        offense = self._team_policy_output(self.unified_policy, observer_is_offense=True, deterministic=True)
        defense = self._team_policy_output(self.unified_policy, observer_is_offense=False, deterministic=True)
        return {"offensive_value": float(offense.values), "defensive_value": float(defense.values)}

    def _softmax_np(self, logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float64)
        if logits.size == 0:
            return logits
        shifted = logits - np.nanmax(logits)
        exp = np.exp(shifted)
        total = float(np.sum(exp))
        if total <= 0.0 or not np.isfinite(total):
            return np.ones_like(logits, dtype=np.float64) / float(logits.size)
        return exp / total

    def _cell_index_for_position(self, position: np.ndarray) -> int | None:
        coords = np.asarray(self.jax.device_get(self.static.cell_coords), dtype=np.int32)
        pos = np.asarray(position, dtype=np.int32).reshape(-1)[:2]
        matches = np.nonzero(np.all(coords == pos, axis=1))[0]
        if matches.size == 0:
            return None
        return int(matches[0])

    def _rebound_target_distribution_payload(
        self,
        *,
        shot_type: int,
        shooter: int,
        sampled_target_cell: int,
        prev_positions: np.ndarray,
    ) -> tuple[list[dict[str, Any]], float | None]:
        coords = np.asarray(self.jax.device_get(self.static.cell_coords), dtype=np.int32)
        if coords.ndim != 2 or coords.shape[0] == 0:
            return [], None
        if shooter < 0 or shooter >= int(prev_positions.shape[0]):
            return [], None
        shot_cell_idx = self._cell_index_for_position(prev_positions[shooter])
        if shot_cell_idx is None:
            return [], None

        table = np.asarray(self.jax.device_get(self.static.rebound_target_probs), dtype=np.float64)
        if table.ndim != 3 or table.shape[-1] != coords.shape[0]:
            return [], None
        safe_shot_type = int(np.clip(int(shot_type), 0, table.shape[0] - 1))
        safe_shot_cell = int(np.clip(int(shot_cell_idx), 0, table.shape[1] - 1))
        raw_probs = np.asarray(table[safe_shot_type, safe_shot_cell], dtype=np.float64)
        if raw_probs.size == 0:
            return [], None

        raw_total = float(np.sum(raw_probs))
        if raw_total <= 0.0 or not np.isfinite(raw_total):
            probs = np.ones_like(raw_probs, dtype=np.float64) / float(raw_probs.size)
        else:
            raw_probs = raw_probs / raw_total
            uniform = np.ones_like(raw_probs, dtype=np.float64) / float(raw_probs.size)
            mix = float(np.asarray(self.jax.device_get(self.static.rebound_target_uniform_mix)))
            mix = float(np.clip(mix, 0.0, 1.0))
            mixed = (1.0 - mix) * raw_probs + mix * uniform
            temp = float(np.asarray(self.jax.device_get(self.static.rebound_target_temperature)))
            temp = max(1.0e-6, temp if np.isfinite(temp) else 1.0)
            logits = np.log(np.maximum(mixed, 1.0e-8)) / temp
            probs = self._softmax_np(logits)

        target_cells = [
            {
                "index": int(idx),
                "q": int(coords[idx, 0]),
                "r": int(coords[idx, 1]),
                "prob": float(prob),
            }
            for idx, prob in enumerate(probs.tolist())
            if float(prob) > 0.0
        ]
        sampled_prob = None
        if 0 <= sampled_target_cell < probs.shape[0]:
            sampled_prob = float(probs[int(sampled_target_cell)])
        return target_cells, sampled_prob

    def _rebound_winner_probabilities_payload(
        self,
        *,
        sampled_target_cell: int,
        winner: int,
        next_state: Any,
    ) -> tuple[list[dict[str, Any]], float | None, dict[str, Any]]:
        coords = np.asarray(self.jax.device_get(self.static.cell_coords), dtype=np.int32)
        kernel_contest_mode = _rebound_contest_mode_from_static(self.static)
        empty_info = {
            "contest_mode": kernel_contest_mode,
            "contest_radius_used": None,
            "contest_fallback_global": False,
        }
        if sampled_target_cell < 0 or sampled_target_cell >= int(coords.shape[0]):
            return [], None, empty_info
        positions = np.asarray(self.jax.device_get(_field0(next_state, "positions")), dtype=np.int32)
        player_cell_indices: list[int] = []
        for pos in positions:
            idx = self._cell_index_for_position(pos)
            player_cell_indices.append(0 if idx is None else int(idx))
        safe_indices = np.clip(np.asarray(player_cell_indices, dtype=np.int32), 0, coords.shape[0] - 1)
        distance_matrix = np.asarray(self.jax.device_get(self.static.cell_distance_matrix), dtype=np.float64)
        distances = distance_matrix[
            safe_indices,
            int(np.clip(sampled_target_cell, 0, distance_matrix.shape[1] - 1)),
        ]
        rebound_skill = np.asarray(self.jax.device_get(_field0(next_state, "rebound_skill")), dtype=np.float64)
        skill_weight = float(np.asarray(self.jax.device_get(self.static.rebound_skill_weight)))
        effective_distances = distances - (max(0.0, skill_weight) * rebound_skill)
        weight = float(np.asarray(self.jax.device_get(self.static.rebound_winner_distance_weight)))
        temp = float(np.asarray(self.jax.device_get(self.static.rebound_winner_temperature)))
        temp = max(1.0e-6, temp if np.isfinite(temp) else 1.0)
        global_logits = (-max(0.0, weight) * effective_distances) / temp

        contest_mode = kernel_contest_mode
        radius_used: int | None = None
        fallback_global = False
        eligible = np.ones_like(distances, dtype=bool)
        logits = np.asarray(global_logits, dtype=np.float64)

        if contest_mode == "local_contest":
            initial_radius = max(0, _int_from_static_field(self.static, "rebound_contest_radius", 1))
            radius_eligible = distances <= float(initial_radius)
            if bool(np.any(radius_eligible)):
                eligible = radius_eligible.astype(bool)
                radius_used = int(initial_radius)
                logits = np.where(eligible, global_logits, -1.0e9)
            else:
                fallback_global = True

        probs = self._softmax_np(logits)
        rows = []
        offense_ids = set(int(pid) for pid in self.offense_ids)
        defense_ids = set(int(pid) for pid in self.defense_ids)
        if contest_mode == "local_contest" and not fallback_global:
            row_indices = np.nonzero(eligible)[0].tolist()
        else:
            row_indices = list(range(int(probs.shape[0])))
        for pid in row_indices:
            prob = float(probs[int(pid)])
            team = "offense" if pid in offense_ids else ("defense" if pid in defense_ids else "unknown")
            rows.append({
                "player_id": int(pid),
                "team": team,
                "conditional_prob": prob,
                "distance_to_sampled_target": int(round(float(distances[pid]))),
                "effective_distance_to_sampled_target": float(effective_distances[pid]),
                "rebound_skill": float(rebound_skill[pid]),
                "eligible": bool(eligible[pid]),
                "contest_mode": contest_mode,
                "contest_radius_used": radius_used,
                "contest_fallback_global": bool(fallback_global),
            })
        rows.sort(key=lambda row: (-float(row["conditional_prob"]), int(row["player_id"])))
        winner_prob = None
        if 0 <= winner < probs.shape[0]:
            winner_prob = float(probs[int(winner)])
        contest_info = {
            "contest_mode": contest_mode,
            "contest_radius_used": radius_used,
            "contest_fallback_global": bool(fallback_global),
        }
        return rows, winner_prob, contest_info

    def _empty_action_results(self) -> dict[str, Any]:
        return {
            "moves": {},
            "passes": {},
            "shots": {},
            "rebounds": [],
            "collisions": [],
            "turnovers": [],
            "defensive_lane_violations": [],
            "offensive_lane_violations": [],
            "defender_pressure": {},
        }

    def _action_results_from_step(self, prev_state, out) -> dict[str, Any]:
        results = self._empty_action_results()
        prev_positions = np.asarray(self.jax.device_get(_field0(prev_state, "positions")), dtype=np.int32)
        next_state = out.state
        if _as_bool(out.shot_attempt[0]):
            shooter = _as_int(out.shot_shooter[0])
            shot_value = _as_float(out.shot_value[0])
            expected = _as_float(out.shot_expected_points[0])
            prob = expected / shot_value if shot_value > 0 else 0.0
            assist_potential = _as_bool(getattr(out, "potential_assist", np.asarray([0]))[0])
            assist_full = _as_bool(getattr(out, "assist", np.asarray([0]))[0])
            assist_passer = _as_int(getattr(out, "assist_passer", np.asarray([-1]))[0])
            results["shots"][str(shooter)] = {
                "success": _as_bool(out.shot_success[0]),
                "distance": int(round(_as_float(out.shot_distance[0]))),
                "probability": float(prob),
                "rng": -1.0,
                "base_probability": float(prob),
                "pressure_multiplier": 1.0,
                "is_three": _as_int(out.shot_type[0]) == SHOT_TYPE_THREE,
                "expected_points": expected,
                "assist_potential": assist_potential,
                "assist_full": assist_full,
                "assist_passer_id": assist_passer if assist_passer >= 0 else None,
            }
        if _as_bool(getattr(out, "rebound_attempt", np.asarray([0]))[0]):
            winner = _as_int(out.rebound_winner[0])
            target_cell_idx = _as_int(out.rebound_target_cell[0])
            target = None
            if 0 <= target_cell_idx < int(np.asarray(self.static.cell_coords).shape[0]):
                target_arr = np.asarray(
                    self.jax.device_get(self.static.cell_coords[target_cell_idx]),
                    dtype=np.int32,
                )
                target = [int(target_arr[0]), int(target_arr[1])]
            winner_team = None
            if 0 <= winner < self.n_players:
                winner_team = "OFFENSE" if winner in self.offense_ids else "DEFENSE"
            shot_type = _as_int(out.shot_type[0])
            shot_shooter = _as_int(out.shot_shooter[0])
            target_cells, target_prob = self._rebound_target_distribution_payload(
                shot_type=shot_type,
                shooter=shot_shooter,
                sampled_target_cell=target_cell_idx,
                prev_positions=prev_positions,
            )
            winner_probs, winner_prob, contest_info = self._rebound_winner_probabilities_payload(
                sampled_target_cell=target_cell_idx,
                winner=winner,
                next_state=next_state,
            )
            rebound = {
                "attempt": True,
                "offensive": _as_bool(out.offensive_rebound[0]),
                "defensive": _as_bool(out.defensive_rebound[0]),
                "winner": winner if winner >= 0 else None,
                "winner_team": winner_team,
                "winner_conditional_prob": winner_prob,
                "winner_probs": winner_probs,
                "contest_mode": contest_info.get("contest_mode"),
                "contest_radius_used": contest_info.get("contest_radius_used"),
                "contest_fallback_global": contest_info.get("contest_fallback_global"),
                "target_cell_index": target_cell_idx if target_cell_idx >= 0 else None,
                "target": target,
                "target_prob": target_prob,
                "target_cells": target_cells,
                "shot_clock_reset_14": _as_bool(out.shot_clock_reset_14[0]),
                "shot_shooter": shot_shooter,
                "shot_type": shot_type,
            }
            results["rebounds"].append(rebound)
            results["rebound"] = rebound
        if _as_bool(out.pass_attempt[0]):
            passer = _as_int(out.pass_passer[0])
            receiver = _as_int(out.pass_receiver[0])
            if _as_bool(out.completed_pass[0]):
                results["passes"][str(passer)] = {
                    "success": True,
                    "target": receiver,
                    "intended_target": receiver,
                    "pass_distance": 0,
                    "target_strategy": "explicit_target",
                    "target_value": None,
                    "total_steal_prob": 0.0,
                    "defenders_evaluated": [],
                }
            else:
                reason = _as_int(out.turnover_reason[0])
                stolen_by = _as_int(_field0(next_state, "ball_holder"))
                results["turnovers"].append(
                    {
                        "player_id": passer,
                        "reason": "pass_out_of_bounds" if reason == TURNOVER_REASON_PASS_OUT_OF_BOUNDS else "steal",
                        "stolen_by": stolen_by if stolen_by >= 0 else None,
                        "turnover_pos": tuple(int(v) for v in prev_positions[passer].tolist()),
                        "pass_target": receiver if receiver >= 0 else None,
                        "intended_target": receiver if receiver >= 0 else None,
                    }
                )
        if _as_bool(out.turnover[0]) and not results["turnovers"]:
            player = _as_int(out.turnover_player[0])
            reason = _as_int(out.turnover_reason[0])
            reason_map = {
                TURNOVER_REASON_DEFENDER_PRESSURE: "defender_pressure",
                TURNOVER_REASON_MOVE_OUT_OF_BOUNDS: "move_out_of_bounds",
                TURNOVER_REASON_OFFENSIVE_THREE_SECONDS: "offensive_three_seconds",
                TURNOVER_REASON_SHOT_CLOCK: "shot_clock_violation",
                TURNOVER_REASON_INTERCEPTED: "steal",
            }
            results["turnovers"].append(
                {
                    "player_id": player if player >= 0 else None,
                    "reason": reason_map.get(reason, "turnover"),
                    "stolen_by": self.ball_holder,
                    "turnover_pos": tuple(int(v) for v in prev_positions[max(0, player)].tolist()) if player >= 0 else None,
                }
            )
        if _as_bool(out.offensive_three_seconds[0]):
            player = _as_int(out.turnover_player[0])
            results["offensive_lane_violations"].append(
                {
                    "player_id": player,
                    "position": tuple(int(v) for v in prev_positions[max(0, player)].tolist()) if player >= 0 else None,
                }
            )
        if _as_bool(out.defensive_lane_violation[0]):
            player = _as_int(out.defensive_lane_violation_player[0])
            results["defensive_lane_violations"].append(
                {
                    "player_id": player,
                    "position": tuple(int(v) for v in prev_positions[max(0, player)].tolist()) if player >= 0 else None,
                }
            )
        return results

    def _reward_reason(self, *, defense: bool = False) -> str:
        results = self.last_action_results
        if results.get("rebounds"):
            rebound = results["rebounds"][0]
            if rebound.get("offensive"):
                return "Off Reb" if not defense else "Opp Off Reb"
            if rebound.get("defensive"):
                return "Def Reb" if defense else "Opp Def Reb"
        if results.get("shots"):
            shot = next(iter(results["shots"].values()))
            return "Opp Shot" if defense else ("Shot Make" if shot.get("success") else "Shot Miss")
        if results.get("passes"):
            return "Opp Pass" if defense else "Pass"
        if results.get("turnovers"):
            return "Forced TO" if defense else "TO"
        if results.get("defensive_lane_violations"):
            return "Lane Violation"
        return "None"

    def _append_shot_log_if_needed(self, game_state: Any) -> None:
        for raw_pid, shot in (self.last_action_results.get("shots") or {}).items():
            pid = int(raw_pid)
            game_state.shot_log.append(
                {
                    "step": int(len(game_state.reward_history)),
                    "player_id": pid,
                    "distance": int(shot.get("distance", 0)),
                    "probability": float(shot.get("probability", 0.0)),
                    "success": bool(shot.get("success", False)),
                    "is_three": bool(shot.get("is_three", False)),
                    "rng": float(shot.get("rng", -1.0)),
                    "base_probability": float(shot.get("base_probability", -1.0)),
                    "pressure_multiplier": float(shot.get("pressure_multiplier", -1.0)),
                    "expected_points": float(shot.get("expected_points", 0.0)),
                    "shooter_fg_pct": float(shot.get("probability", 0.0)),
                    "assist_potential": bool(shot.get("assist_potential", False)),
                    "assist_full": bool(shot.get("assist_full", False)),
                    "assist_passer_id": shot.get("assist_passer_id"),
                }
            )

    def _actions_taken_payload(self, full_actions: np.ndarray, action_meta: dict[int, dict[str, Any]]):
        names = [action.name for action in ActionType]
        actions_taken = {}
        actions_taken_meta = {}
        for pid, action_idx in enumerate(full_actions.tolist()):
            name = names[int(action_idx)] if 0 <= int(action_idx) < len(names) else "UNKNOWN"
            actions_taken[str(pid)] = name
            if str(name).startswith("PASS"):
                pass_entry = self.last_action_results.get("passes", {}).get(str(pid), {})
                target = action_meta.get(pid, {}).get("target")
                if target is None:
                    target = pass_entry.get("intended_target", pass_entry.get("target"))
                if target is None:
                    for turnover in self.last_action_results.get("turnovers", []):
                        if turnover.get("player_id") == pid:
                            target = turnover.get("intended_target", turnover.get("pass_target"))
                            break
                actions_taken_meta[str(pid)] = {"type": "PASS", "target": target} if target is not None else {"type": "PASS"}
            else:
                actions_taken_meta[str(pid)] = {"type": name}
        return actions_taken, actions_taken_meta

    def _attention_payload_from_weights(self, weights: np.ndarray | None, observer_is_offense: bool):
        if weights is None:
            return None
        labels = []
        for pid in range(self.n_players):
            if pid in self.offense_ids:
                labels.append(f"O{pid}")
            elif pid in self.defense_ids:
                labels.append(f"D{pid}")
            else:
                labels.append(f"P{pid}")
        cls_count = int(getattr(self.raw_model.spec, "attention_num_cls_tokens", 0) or 0)
        cls_labels = []
        if cls_count >= 1:
            cls_labels.append("CLS_OFF")
        if cls_count >= 2:
            cls_labels.append("CLS_DEF")
        for idx in range(2, cls_count):
            cls_labels.append(f"CLS_{idx + 1}")
        labels.extend(cls_labels)
        if len(labels) != int(weights.shape[-1]):
            labels = [f"T{idx}" for idx in range(int(weights.shape[-1]))]
            cls_labels = []
        if observer_is_offense:
            runtime_intent_index = int(_as_int(_field0(self.state, "intent_index")))
            runtime_intent_active = bool(_as_bool(_field0(self.state, "intent_active")))
            runtime_intent_visible = True
            runtime_intent_gate = runtime_intent_active
        else:
            runtime_intent_active = bool(_as_bool(_field0(self.state, "defense_intent_active")))
            runtime_intent_index = (
                int(_as_int(_field0(self.state, "defense_intent_index")))
                if runtime_intent_active
                else None
            )
            runtime_intent_visible = bool(_as_bool(_field0(self.state, "intent_visible_to_defense")))
            runtime_intent_gate = runtime_intent_active
        return {
            "weights_avg": weights.mean(axis=0).tolist(),
            "weights_heads": weights.tolist(),
            "labels": labels,
            "heads": int(weights.shape[0]),
            "runtime_intent_index": runtime_intent_index,
            "runtime_intent_active": runtime_intent_active,
            "runtime_intent_visible": runtime_intent_visible,
            "runtime_intent_gate": runtime_intent_gate,
            "observer_role": "offense" if observer_is_offense else "defense",
            "cls_labels": cls_labels,
        }

    def _attention_payloads_for_state(self, observer_is_offense: bool) -> dict[str, Any]:
        payloads = {
            "offense": self._last_attention_payloads.get("offense"),
            "defense": self._last_attention_payloads.get("defense"),
        }
        if str(getattr(self.raw_model.spec, "model_type", "")) == "attention":
            if payloads["offense"] is None:
                offense_weights = self._team_policy_output(
                    self.unified_policy,
                    observer_is_offense=True,
                    deterministic=True,
                ).attention_weights
                payloads["offense"] = self._attention_payload_from_weights(offense_weights, True)
            if payloads["defense"] is None:
                defense_policy = self.opponent_policy or self.unified_policy
                defense_weights = self._team_policy_output(
                    defense_policy,
                    observer_is_offense=False,
                    deterministic=True,
                ).attention_weights
                payloads["defense"] = self._attention_payload_from_weights(defense_weights, False)
        self._last_attention_payloads = payloads
        self._last_attention_payload = payloads["offense"] if observer_is_offense else payloads["defense"]
        player_key = "offense" if self.user_team != Team.DEFENSE else "defense"
        opponent_key = "defense" if player_key == "offense" else "offense"
        return {
            "default": self._last_attention_payload,
            "offense": payloads["offense"],
            "defense": payloads["defense"],
            "player": payloads[player_key],
            "opponent": payloads[opponent_key],
        }

    def _selector_training_params(self, game_state: Any) -> dict[str, Any]:
        params: dict[str, Any] = {}
        game_params = getattr(game_state, "mlflow_training_params", None)
        if isinstance(game_params, dict):
            params.update(game_params)

        metadata = get_policy_metadata(self.unified_policy) or get_policy_metadata(self.raw_model) or {}
        trainer_config = metadata.get("trainer_config")
        if isinstance(trainer_config, dict):
            for key, value in trainer_config.items():
                params.setdefault(str(key), value)
            # Static selector architecture/runtime flags should come from the
            # loaded checkpoint. MLflow training-param extraction may populate
            # default False values for older/incomplete runs.
            for key in _SELECTOR_METADATA_PRIORITY_KEYS:
                if key in trainer_config:
                    params[key] = trainer_config[key]

        policy_spec = metadata.get("policy_spec")
        if isinstance(policy_spec, dict) and "intent_selector_enabled" in policy_spec:
            params.setdefault(
                "intent_selector_enabled",
                bool(policy_spec.get("intent_selector_enabled")),
            )
        return params

    def _selector_runtime_enabled(self, game_state: Any) -> bool:
        if not bool(getattr(self.raw_model.spec, "intent_selector_enabled", False)):
            return False
        if not _as_bool(self.static.enable_intent_learning):
            return False
        params = self._selector_training_params(game_state)
        return _coerce_bool(
            _param_lookup(params, "intent_selector_enabled", True),
            default=True,
        )

    def _selector_multiselect_enabled(self, game_state: Any) -> bool:
        if not self._selector_runtime_enabled(game_state):
            return False
        params = self._selector_training_params(game_state)
        return _coerce_bool(
            _param_lookup(params, "intent_selector_multiselect_enabled", False),
            default=False,
        )

    def _selector_config_alpha_eps(self, game_state: Any) -> tuple[float, float]:
        params = self._selector_training_params(game_state)
        alpha = _coerce_float(
            _param_lookup(
                params,
                "intent_selector_alpha_current",
                _param_lookup(
                    params,
                    "intent_selector_alpha_end",
                    _param_lookup(params, "selector_alpha", 1.0),
                ),
            ),
            default=1.0,
        )
        eps = _coerce_float(
            _param_lookup(
                params,
                "intent_selector_eps_current",
                _param_lookup(
                    params,
                    "intent_selector_eps_end",
                    _param_lookup(params, "selector_eps", 0.0),
                ),
            ),
            default=0.0,
        )
        return float(np.clip(alpha, 0.0, 1.0)), float(np.clip(eps, 0.0, 1.0))

    def _selector_force_learned_runtime(self, game_state: Any) -> bool:
        params = self._selector_training_params(game_state)
        return _coerce_bool(
            _param_lookup(params, "intent_selector_force_learned_runtime", True),
            default=True,
        )

    def _selector_alpha_eps(self, game_state: Any) -> tuple[float, float]:
        alpha, eps = self._selector_config_alpha_eps(game_state)
        if self._selector_runtime_enabled(game_state) and self._selector_force_learned_runtime(game_state):
            alpha = 1.0
        return float(alpha), float(eps)

    def _selector_forward(self):
        if not bool(getattr(self.raw_model.spec, "intent_selector_enabled", False)):
            return None
        flat_obs = build_policy_observation_batch_with_role_flag(
            self.static,
            self.state,
            self.role_flag_offense,
            self.jnp,
            model_type=str(self.raw_model.spec.model_type),
        )
        flat_obs = _adapt_policy_observation_to_spec(flat_obs, self.static, self.raw_model.spec, self.jnp)
        batch_size = flat_obs.shape[0]
        neutral_context = {
            "intent_index": self.jnp.zeros((batch_size,), dtype=self.jnp.int32),
            "intent_gate": self.jnp.zeros((batch_size,), dtype=self.jnp.float32),
        }
        from basketworld_jax.models import actor_critic_forward

        forward_out = actor_critic_forward(
            self.raw_model.params,
            flat_obs,
            self.raw_model.spec,
            self.jnp,
            intent_context=neutral_context,
        )
        return forward_out

    def _selector_distribution(self, game_state: Any) -> dict[str, Any] | None:
        forward_out = self._selector_forward()
        if forward_out is None:
            return None
        logits_device = forward_out["selector_logits"][0]
        values_device = forward_out["selector_values"][0:1]
        raw_probs_device = self.jax.nn.softmax(logits_device, axis=-1)
        alpha, eps = self._selector_alpha_eps(game_state)
        num_intents = int(max(1, getattr(self.raw_model.spec, "num_intents", logits_device.shape[-1])))
        uniform_device = self.jnp.full_like(raw_probs_device, 1.0 / float(num_intents))
        mixed_probs_device = ((1.0 - eps) * raw_probs_device) + (eps * uniform_device)
        deployed_probs_device = (alpha * mixed_probs_device) + ((1.0 - alpha) * uniform_device)
        return {
            "logits_device": logits_device,
            "mixed_probs_device": mixed_probs_device,
            "logits": np.asarray(self.jax.device_get(logits_device), dtype=np.float32),
            "raw_probs": np.asarray(self.jax.device_get(raw_probs_device), dtype=np.float32),
            "mixed_probs": np.asarray(self.jax.device_get(mixed_probs_device), dtype=np.float32),
            "deployed_probs": np.asarray(self.jax.device_get(deployed_probs_device), dtype=np.float32),
            "alpha": float(alpha),
            "eps": float(eps),
            "value": float(np.asarray(self.jax.device_get(values_device), dtype=np.float32).reshape(-1)[0]),
            "num_intents": int(num_intents),
        }

    def _sample_selector_intent(self, game_state: Any) -> dict[str, Any] | None:
        dist = self._selector_distribution(game_state)
        if dist is None:
            return None
        sample_key, alpha_key, fallback_key = self.jax.random.split(self._next_key(), 3)
        alpha_used = bool(
            float(self.jax.device_get(self.jax.random.uniform(alpha_key, shape=()))) < float(dist["alpha"])
        )
        if alpha_used:
            log_probs = self.jnp.log(self.jnp.maximum(dist["mixed_probs_device"], 1.0e-8))
            intent_index = int(self.jax.device_get(self.jax.random.categorical(sample_key, log_probs, axis=-1)))
        else:
            intent_index = int(
                self.jax.device_get(
                    self.jax.random.randint(
                        fallback_key,
                        shape=(),
                        minval=0,
                        maxval=int(dist["num_intents"]),
                        dtype=self.jnp.int32,
                    )
                )
            )
        return {
            "intent_index": int(intent_index),
            "used_selector": bool(alpha_used),
            "alpha": float(dist["alpha"]),
            "eps": float(dist["eps"]),
            "value": float(dist["value"]),
        }

    def _selector_boundary_reason(self, game_state: Any) -> str | None:
        if not self._selector_runtime_enabled(game_state):
            return None
        alpha, _ = self._selector_alpha_eps(game_state)
        multiselect_enabled = self._selector_multiselect_enabled(game_state)
        if alpha <= 0.0 and not multiselect_enabled:
            return None
        age = _as_int(_field0(self.state, "intent_age"))
        if not _as_bool(_field0(self.state, "intent_active")):
            return None
        if not multiselect_enabled:
            return None
        remaining = _as_int(_field0(self.state, "intent_commitment_remaining"))
        if age > 0 and remaining <= 0:
            return "commitment_timeout"
        params = self._selector_training_params(game_state)
        min_play_steps = max(
            1,
            _coerce_int(_param_lookup(params, "intent_selector_min_play_steps", 3), default=3),
        )
        if bool(self._last_completed_pass_boundary) and age >= min_play_steps:
            return "completed_pass"
        return None

    def _maybe_apply_selector_boundary(self, game_state: Any) -> dict[str, Any] | None:
        reason = self._selector_boundary_reason(game_state)
        if reason is None:
            return None
        previous_intent = _as_int(_field0(self.state, "intent_index"))
        selection = self._sample_selector_intent(game_state)
        if selection is None:
            return None
        commitment_steps = max(1, _as_int(self.static.intent_commitment_steps))
        self.set_offense_intent_state(
            active=True,
            intent_index=int(selection["intent_index"]),
            intent_age=0,
            intent_commitment_remaining=commitment_steps,
            game_state=game_state,
        )
        game_state.selector_segment_index = int(getattr(game_state, "selector_segment_index", 0) or 0) + 1
        game_state.selector_last_boundary_reason = str(reason)
        self._last_completed_pass_boundary = False
        self._last_selector_transition = {
            "reason": str(reason),
            "previous_intent_index": int(previous_intent),
            "intent_index": int(selection["intent_index"]),
            "changed_intent": int(selection["intent_index"]) != int(previous_intent),
            "used_selector": bool(selection.get("used_selector", False)),
            "source": "learned_selector" if bool(selection.get("used_selector", False)) else "uniform_fallback",
            "alpha": float(selection.get("alpha", 0.0)),
            "eps": float(selection.get("eps", 0.0)),
            "value": selection.get("value"),
        }
        return dict(self._last_selector_transition)

    def _selector_debug_payload(
        self,
        game_state: Any,
        selector_prefs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        params = self._selector_training_params(game_state)
        alpha, eps = self._selector_alpha_eps(game_state)
        training_alpha, training_eps = self._selector_config_alpha_eps(game_state)
        min_play_steps = max(
            1,
            _coerce_int(_param_lookup(params, "intent_selector_min_play_steps", 3), default=3),
        )
        intent_age = _as_int(_field0(self.state, "intent_age"))
        last_completed_pass_boundary = bool(self._last_completed_pass_boundary)
        completed_pass_min_steps_remaining = max(0, int(min_play_steps) - int(intent_age))
        completed_pass_min_steps_met = (
            last_completed_pass_boundary and completed_pass_min_steps_remaining == 0
        )
        current_intent = _as_int(_field0(self.state, "intent_index"))
        current_probs = None
        current_rank = None
        rows = list((selector_prefs or {}).get("intent_probs") or [])
        ranked = sorted(rows, key=lambda row: float(row.get("deployed_prob", row.get("prob", 0.0))), reverse=True)
        for rank_idx, row in enumerate(ranked, start=1):
            if int(row.get("intent_index", -1)) == int(current_intent):
                current_rank = rank_idx
                current_probs = {
                    "raw_prob": float(row.get("raw_prob", 0.0)),
                    "mixed_prob": float(row.get("mixed_prob", 0.0)),
                    "deployed_prob": float(row.get("deployed_prob", row.get("prob", 0.0))),
                    "rank_deployed": int(rank_idx),
                    "play_name": row.get("play_name"),
                }
                break
        runtime_enabled = self._selector_runtime_enabled(game_state)
        last_transition = copy.deepcopy(self._last_selector_transition)
        return {
            "runtime_enabled": bool(runtime_enabled),
            "model_selector_enabled": bool(getattr(self.raw_model.spec, "intent_selector_enabled", False)),
            "env_intent_learning_enabled": bool(_as_bool(self.static.enable_intent_learning)),
            "training_selector_enabled": bool(
                _coerce_bool(_param_lookup(params, "intent_selector_enabled", True), default=True)
            ),
            "multiselect_enabled": bool(self._selector_multiselect_enabled(game_state)),
            "alpha_current": float(alpha),
            "eps_current": float(eps),
            "training_alpha_current": float(training_alpha),
            "training_eps_current": float(training_eps),
            "force_learned_runtime": bool(self._selector_force_learned_runtime(game_state)),
            "min_play_steps": int(min_play_steps),
            "commitment_steps": int(max(1, _as_int(self.static.intent_commitment_steps))),
            "eligible_boundary_reason": self._selector_boundary_reason(game_state),
            "last_completed_pass_boundary": bool(last_completed_pass_boundary),
            "completed_pass_min_steps_met": bool(completed_pass_min_steps_met),
            "completed_pass_min_steps_remaining": int(completed_pass_min_steps_remaining),
            "last_transition": last_transition,
            "last_transition_source": (
                None
                if not isinstance(last_transition, dict)
                else str(last_transition.get("source") or "")
            ),
            "current_play_probabilities": current_probs,
            "current_play_rank_deployed": current_rank,
        }

    def _selector_preferences(self, game_state: Any) -> dict[str, Any] | None:
        dist = self._selector_distribution(game_state)
        if dist is None:
            return None
        logits = dist["logits"]
        play_map = self._play_name_map(game_state)
        rows = []
        for idx, prob in enumerate(dist["deployed_probs"].tolist()):
            rows.append(
                {
                    "intent_index": int(idx),
                    "prob": float(prob),
                    "raw_prob": float(dist["raw_probs"][idx]),
                    "mixed_prob": float(dist["mixed_probs"][idx]),
                    "deployed_prob": float(prob),
                    "logit": float(logits[idx]),
                    "play_name": lookup_play_name(play_map, idx),
                }
            )
        rows.sort(key=lambda item: item["prob"], reverse=True)
        return {
            "current_intent_index": int(_as_int(_field0(self.state, "intent_index"))),
            "current_play_name": lookup_play_name(play_map, _as_int(_field0(self.state, "intent_index"))),
            "intent_probs": rows,
            "play_name_map": play_map,
            "alpha_current": float(dist["alpha"]),
            "eps_current": float(dist["eps"]),
            "selection_mode": "learned_sample_runtime" if self._selector_force_learned_runtime(game_state) else "learned_sample",
            "segment_index": int(_as_int(_field0(self.state, "intent_age"))),
            "value_estimate": float(dist["value"]),
        }

    def _play_name_map(self, game_state: Any) -> dict[str, str]:
        count = int(getattr(self.raw_model.spec, "num_intents", 0) or 0)
        metadata = get_policy_metadata(getattr(game_state, "unified_policy", None))
        return _coerce_play_name_map(metadata, count)

    def _counterfactual_snapshot_summary(self, game_state: Any) -> dict[str, Any]:
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

    def _capture_turn_start(self, game_state: Any) -> None:
        game_state.turn_start_positions = [tuple(pos) for pos in self.positions]
        game_state.turn_start_ball_holder = self.ball_holder
        game_state.turn_start_shot_clock = self.shot_clock

    def get_full_game_state(
        self,
        game_state: Any,
        *,
        include_policy_probs: bool = False,
        include_action_values: bool = False,
        include_state_values: bool = False,
    ) -> dict[str, Any]:
        self._sync_display_env()
        observer_is_offense = game_state.user_team != Team.DEFENSE
        obs_dict = self.observation_dict(observer_is_offense=observer_is_offense)
        game_state.obs = obs_dict
        positions = self.positions
        ball_holder = self.ball_holder
        profile = build_shot_profile_batch(self.static, self.state, self.jnp)
        shot_prob = np.asarray(self.jax.device_get(profile["probability"][0]), dtype=np.float32)
        shot_ep = np.asarray(self.jax.device_get(profile["expected_points"][0]), dtype=np.float32)
        shot_distance = np.asarray(self.jax.device_get(profile["distance"][0]), dtype=np.int32)
        shot_value = np.asarray(self.jax.device_get(profile["shot_value"][0]), dtype=np.float32)
        play_map = self._play_name_map(game_state)
        rebound_skill = np.asarray(self.jax.device_get(_field0(self.state, "rebound_skill")), dtype=np.float32)
        player_rebound_skills = {str(pid): float(rebound_skill[int(pid)]) for pid in range(int(rebound_skill.shape[0]))}
        metadata = get_policy_metadata(getattr(game_state, "unified_policy", None)) or {}
        counterfactual_snapshot = self._counterfactual_snapshot_summary(game_state)
        globals_labels = [
            "shot_clock_norm",
            "pressure_exposure",
            "hoop_q_norm",
            "hoop_r_norm",
            "expected_rebound_target_q",
            "expected_rebound_target_r",
            "target_entropy",
        ]
        attention_payloads = self._attention_payloads_for_state(observer_is_offense)
        attention_payload = attention_payloads["default"]

        state = {
            "players_per_side": int(self.display_env.players_per_side or 3),
            "players": int(self.display_env.players_per_side or 3),
            "positions": positions,
            "ball_holder": ball_holder,
            "ball_handler_shot_probability": float(shot_prob[ball_holder]) if ball_holder is not None else None,
            "pass_steal_probabilities": self.pass_steal_probabilities(),
            "shot_clock": self.shot_clock,
            "min_shot_clock": int(self.display_env.min_shot_clock or 10),
            "shot_clock_steps": int(self.display_env.shot_clock_steps or 24),
            "user_team_name": game_state.user_team.name,
            "done": _as_bool(_field0(self.state, "episode_ended")),
            "training_team": getattr(self.display_env.training_team, "name", None),
            "counterfactual_snapshot_available": bool(counterfactual_snapshot["available"]),
            "counterfactual_snapshot_step": counterfactual_snapshot["captured_step"],
            "counterfactual_snapshot_shot_clock": counterfactual_snapshot["shot_clock"],
            "counterfactual_snapshot_ball_holder": counterfactual_snapshot["ball_holder"],
            "counterfactual_snapshot_intent_active": counterfactual_snapshot["intent_active"],
            "counterfactual_snapshot_intent_index": counterfactual_snapshot["intent_index"],
            "counterfactual_snapshot_intent_age": counterfactual_snapshot["intent_age"],
            "counterfactual_snapshot_captured_at": counterfactual_snapshot["captured_at"],
            "action_space": {action.name: action.value for action in ActionType},
            "action_mask": obs_dict["action_mask"].tolist(),
            "obs": np.asarray(obs_dict["obs"], dtype=np.float32).reshape(-1).tolist(),
            "obs_tokens": {
                "players": np.asarray(obs_dict["players"], dtype=np.float32).tolist(),
                "globals": np.asarray(obs_dict["globals"], dtype=np.float32).tolist(),
                "globals_labels": globals_labels,
                "attention": attention_payload,
                "attention_by_observer": attention_payloads,
            },
            "obs_tokens_version": 1,
            "last_action_results": copy.deepcopy(self.last_action_results),
            "episode_rebounds": copy.deepcopy(self.episode_rebounds),
            "player_rebound_skills": player_rebound_skills,
            "offense_ids": self.offense_ids,
            "defense_ids": self.defense_ids,
            "basket_position": tuple(int(v) for v in np.asarray(self.static.basket_position).tolist()),
            "court_width": int(self.display_env.court_width),
            "court_height": int(self.display_env.court_height),
            "three_point_distance": float(self.display_env.three_point_distance or 4.0),
            "three_point_short_distance": (
                float(self.display_env.three_point_short_distance)
                if self.display_env.three_point_short_distance is not None
                else None
            ),
            "three_point_hexes": [
                (int(q), int(r)) for q, r in getattr(self.display_env, "_three_point_hexes", set())
            ],
            "three_point_line_hexes": [
                (int(q), int(r)) for q, r in getattr(self.display_env, "_three_point_line_hexes", set())
            ],
            "three_point_outline": [
                (float(x), float(y)) for x, y in getattr(self.display_env, "_three_point_outline_points", [])
            ],
            "shot_probs": getattr(self.display_env, "shot_probs", {}),
            "shot_params": {
                "layup_pct": float(self.display_env.layup_pct or 0.0),
                "three_pt_pct": float(self.display_env.three_pt_pct or 0.0),
                "three_pt_extra_hex_decay": float(self.display_env.three_pt_extra_hex_decay or 0.05),
                "dunk_pct": float(self.display_env.dunk_pct or 0.0),
                "layup_std": float(self.display_env.layup_std or 0.0),
                "three_pt_std": float(self.display_env.three_pt_std or 0.0),
                "dunk_std": float(self.display_env.dunk_std or 0.0),
                "allow_dunks": bool(self.display_env.allow_dunks),
            },
            "rebound_runtime": {
                "enabled": bool(getattr(self.display_env, "enable_rebounds", False)),
                "kernel_enabled": _as_bool(self.static.enable_rebounds),
                "table_model_dir": str(getattr(self.display_env, "rebound_table_model_dir", "") or ""),
                "target_temperature": float(getattr(self.display_env, "rebound_target_temperature", 1.0)),
                "target_uniform_mix": float(getattr(self.display_env, "rebound_target_uniform_mix", 0.0)),
                "winner_distance_weight": float(getattr(self.display_env, "rebound_winner_distance_weight", 1.0)),
                "winner_temperature": float(getattr(self.display_env, "rebound_winner_temperature", 1.0)),
                "skill_std": float(getattr(self.display_env, "rebound_skill_std", 0.0)),
                "skill_weight": float(getattr(self.display_env, "rebound_skill_weight", 0.0)),
                "contest_mode": _rebound_contest_mode_from_static(self.static),
                "contest_radius": _int_from_static_field(self.static, "rebound_contest_radius", 1),
                "obs_top_n_targets": _int_from_static_field(self.static, "rebound_obs_top_n_targets", 0),
                "offensive_rebound_shot_clock_reset": int(getattr(self.display_env, "offensive_rebound_shot_clock_reset", 14)),
                "terminal_reward_mode": str(getattr(self.display_env, "rebound_terminal_reward_mode", "actual_points") or "actual_points"),
                "target_table_shape": [int(v) for v in self.static.rebound_target_probs.shape],
            },
            "defender_pressure_distance": int(self.display_env.defender_pressure_distance or 1),
            "defender_pressure_turnover_chance": float(self.display_env.defender_pressure_turnover_chance or 0.05),
            "defender_pressure_decay_lambda": float(self.display_env.defender_pressure_decay_lambda or 1.0),
            "base_steal_rate": float(self.display_env.base_steal_rate or 0.35),
            "steal_perp_decay": float(self.display_env.steal_perp_decay or 1.5),
            "steal_distance_factor": float(self.display_env.steal_distance_factor or 0.08),
            "steal_position_weight_min": float(self.display_env.steal_position_weight_min or 0.3),
            "pass_interception_model": str(getattr(self.display_env, "pass_interception_model", "line") or "line"),
            "pass_passer_pressure_weight": float(getattr(self.display_env, "pass_passer_pressure_weight", 0.0)),
            "pass_receiver_pressure_weight": float(getattr(self.display_env, "pass_receiver_pressure_weight", 0.0)),
            "pass_lob_lane_multiplier": float(getattr(self.display_env, "pass_lob_lane_multiplier", 0.35)),
            "pass_lob_receiver_distance": float(getattr(self.display_env, "pass_lob_receiver_distance", 1.0)),
            "pass_speed": float(getattr(self.display_env, "pass_speed", 3.5)),
            "defender_reaction_time": float(getattr(self.display_env, "defender_reaction_time", 0.35)),
            "defender_speed": float(getattr(self.display_env, "defender_speed", 1.25)),
            "defender_reach_radius": float(getattr(self.display_env, "defender_reach_radius", 0.65)),
            "reaction_softness": float(getattr(self.display_env, "reaction_softness", 0.55)),
            "base_passer_risk": float(getattr(self.display_env, "base_passer_risk", 0.06)),
            "passer_pressure_decay": float(getattr(self.display_env, "passer_pressure_decay", 1.35)),
            "base_receiver_risk": float(getattr(self.display_env, "base_receiver_risk", 0.35)),
            "receiver_alignment_min": float(getattr(self.display_env, "receiver_alignment_min", 0.35)),
            "receiver_alignment_width": float(getattr(self.display_env, "receiver_alignment_width", 2.0)),
            "max_receiver_hazard": float(getattr(self.display_env, "max_receiver_hazard", 0.85)),
            "lane_weight": float(getattr(self.display_env, "lane_weight", 0.0)),
            "spawn_distance": int(self.display_env.spawn_distance or 3),
            "max_spawn_distance": (
                int(self.display_env.max_spawn_distance)
                if self.display_env.max_spawn_distance is not None
                else None
            ),
            "defender_spawn_distance": int(self.display_env.defender_spawn_distance or 0),
            "defender_guard_distance": int(self.display_env.defender_guard_distance or 1),
            "offense_spawn_boundary_margin": int(self.display_env.offense_spawn_boundary_margin or 0),
            "shot_pressure_enabled": bool(self.display_env.shot_pressure_enabled),
            "shot_pressure_max": float(self.display_env.shot_pressure_max or 0.5),
            "shot_pressure_lambda": float(self.display_env.shot_pressure_lambda or 1.0),
            "shot_pressure_arc_degrees": float(self.display_env.shot_pressure_arc_degrees or 60.0),
            "three_pt_extra_hex_decay": float(self.display_env.three_pt_extra_hex_decay or 0.05),
            "mask_occupied_moves": bool(self.display_env.mask_occupied_moves),
            "three_second_lane_width": int(self.display_env.three_second_lane_width or 1),
            "three_second_lane_height": int(self.display_env.three_second_lane_height or 3),
            "three_second_max_steps": int(self.display_env.three_second_max_steps or 3),
            "illegal_defense_enabled": bool(self.display_env.illegal_defense_enabled),
            "offensive_three_seconds_enabled": bool(self.display_env.offensive_three_seconds_enabled),
            "enable_intent_learning": bool(self.display_env.enable_intent_learning),
            "num_intents": int(self.display_env.num_intents or 0),
            "intent_commitment_steps": int(self.display_env.intent_commitment_steps or 0),
            "intent_null_prob": float(self.display_env.intent_null_prob or 0.0),
            "intent_visible_to_defense_prob": float(self.display_env.intent_visible_to_defense_prob or 0.0),
            "enable_defense_intent_learning": bool(self.display_env.enable_defense_intent_learning),
            "defense_intent_null_prob": float(self.display_env.defense_intent_null_prob or 1.0),
            "play_name_map": play_map,
            "intent_diversity_enabled": bool((getattr(game_state, "mlflow_training_params", None) or {}).get("intent_diversity_enabled", False)),
            "intent_obs_mode": str(getattr(self.display_env, "intent_obs_mode", "private_offense") or "private_offense"),
            "intent_active_current": _as_bool(_field0(self.state, "intent_active")),
            "intent_index_current": _as_int(_field0(self.state, "intent_index")),
            "current_play_name": lookup_play_name(play_map, _as_int(_field0(self.state, "intent_index"))),
            "intent_age": _as_int(_field0(self.state, "intent_age")),
            "intent_commitment_remaining": _as_int(_field0(self.state, "intent_commitment_remaining")),
            "selector_segment_index_current": int(getattr(game_state, "selector_segment_index", 0)),
            "selector_last_boundary_reason": getattr(game_state, "selector_last_boundary_reason", None),
            "intent_visible_to_defense_current": _as_bool(_field0(self.state, "intent_visible_to_defense")),
            "defense_intent_active_current": _as_bool(_field0(self.state, "defense_intent_active")),
            "defense_intent_index_current": _as_int(_field0(self.state, "defense_intent_index")),
            "current_defense_play_name": lookup_play_name(play_map, _as_int(_field0(self.state, "defense_intent_index"))),
            "defense_intent_age": _as_int(_field0(self.state, "defense_intent_age")),
            "defense_intent_commitment_remaining": _as_int(_field0(self.state, "defense_intent_commitment_remaining")),
            "include_hoop_vector": bool(self.display_env.include_hoop_vector),
            "offensive_lane_hexes": [
                (int(q), int(r)) for q, r in (self.display_env.offensive_lane_hexes or set())
            ],
            "defensive_lane_hexes": [
                (int(q), int(r)) for q, r in (self.display_env.defensive_lane_hexes or set())
            ],
            "offensive_lane_steps": {
                int(pid): int(v)
                for pid, v in enumerate(np.asarray(_field0(self.state, "offense_lane_steps")).tolist())
            },
            "defensive_lane_steps": {
                int(pid): int(v)
                for pid, v in enumerate(np.asarray(_field0(self.state, "defense_lane_steps")).tolist())
            },
            "pass_arc_degrees": float(self.display_env.pass_arc_degrees or 60.0),
            "pass_oob_turnover_prob": float(self.display_env.pass_oob_turnover_prob or 1.0),
            "pass_target_strategy": self.display_env.pass_target_strategy or "nearest",
            "pass_mode": self.display_env.pass_mode or "directional",
            "illegal_action_policy": (
                self.display_env.illegal_action_policy.value
                if self.display_env.illegal_action_policy
                else "noop"
            ),
            "pass_logit_bias": 0.0,
            "model_backend": get_policy_backend_kind(getattr(game_state, "unified_policy", None)),
            "model_capabilities": copy.deepcopy(get_policy_capabilities(getattr(game_state, "unified_policy", None))),
            "model_metadata": copy.deepcopy(metadata),
            "opponent_model_backend": get_policy_backend_kind(getattr(game_state, "defense_policy", None)),
            "opponent_model_capabilities": copy.deepcopy(get_policy_capabilities(getattr(game_state, "defense_policy", None))),
            "opponent_model_metadata": copy.deepcopy(get_policy_metadata(getattr(game_state, "defense_policy", None))),
            "run_id": getattr(game_state, "run_id", None),
            "run_name": getattr(game_state, "run_name", None),
            "model_codename": metadata.get("model_codename"),
            "training_params": getattr(game_state, "mlflow_training_params", None),
            "start_template_library": copy.deepcopy(getattr(game_state, "mlflow_start_template_library", None)),
            "start_template_library_source": getattr(game_state, "start_template_library_source", None),
            "start_template_library_path": getattr(game_state, "start_template_library_path", None),
            "mlflow_env_defaults": dict(getattr(game_state, "mlflow_env_optional_defaults", {}) or {}),
            "unified_policy_name": getattr(game_state, "unified_policy_key", None),
            "opponent_unified_policy_name": getattr(game_state, "opponent_unified_policy_key", None),
            "offense_shooting_pct_by_player": {
                "layup": [float(v) for v in np.asarray(_field0(self.state, "layup_pct"))[self.offense_ids].tolist()],
                "three_pt": [float(v) for v in np.asarray(_field0(self.state, "three_pt_pct"))[self.offense_ids].tolist()],
                "dunk": [float(v) for v in np.asarray(_field0(self.state, "dunk_pct"))[self.offense_ids].tolist()],
            },
            "offense_shooting_pct_sampled": copy.deepcopy(getattr(game_state, "sampled_offense_skills", None) or {}),
            "ep_by_player": [float(v) for v in shot_ep.tolist()],
        }
        if include_policy_probs:
            if self._last_policy_probs is None:
                _, probs = self._choose_joint_policy_actions(player_deterministic=True, opponent_deterministic=True)
            else:
                probs = self._last_policy_probs
            state["policy_probabilities"] = {str(pid): values for pid, values in probs.items()}
            selector_prefs = self._selector_preferences(game_state)
            if selector_prefs is not None:
                state["selector_intent_preferences"] = selector_prefs
            state["selector_debug"] = self._selector_debug_payload(game_state, selector_prefs)
        else:
            state["selector_debug"] = self._selector_debug_payload(game_state)
        if include_action_values:
            state["action_values"] = {}
        if include_state_values:
            state["state_values"] = self.state_values()
        return state
