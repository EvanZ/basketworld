from __future__ import annotations

from typing import Optional, Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from .action_resolution import (
    IllegalActionStrategy,
    get_policy_action_probabilities,
    resolve_illegal_actions,
)
from basketworld.envs.basketworld_env_v2 import Team
from basketworld.utils.intent_policy_sensitivity import (
    clone_observation_dict,
    patch_intent_in_observation,
    sync_policy_runtime_intent_override_from_env,
)
from basketworld.utils.policy_loading import load_ppo_for_inference


def _selector_runtime_enabled_for_policy(policy_model: Any) -> bool:
    if policy_model is None:
        return False
    policy_obj = getattr(policy_model, "policy", None)
    if policy_obj is None or not hasattr(policy_obj, "has_intent_selector"):
        return False
    try:
        return bool(policy_obj.has_intent_selector())
    except Exception:
        return False


def _selector_alpha_current(training_params: Optional[dict[str, Any]], policy_model: Any) -> float:
    if not isinstance(training_params, dict):
        return 0.0
    t = int(getattr(policy_model, "num_timesteps", 0) or 0)
    start = float(training_params.get("intent_selector_alpha_start", 0.0) or 0.0)
    end = float(training_params.get("intent_selector_alpha_end", 1.0) or 1.0)
    warmup = max(0, int(training_params.get("intent_selector_alpha_warmup_steps", 0) or 0))
    ramp = max(0, int(training_params.get("intent_selector_alpha_ramp_steps", 1) or 0))
    if t < warmup:
        return float(start)
    if ramp <= 0:
        return float(end)
    progress = min(1.0, max(0.0, (t - warmup) / float(ramp)))
    return float(start + progress * (end - start))


def _normalize_selector_intent_selection_mode(mode: Optional[str]) -> str:
    value = str(mode or "learned_sample").strip().lower()
    if value in {"best_intent", "best", "argmax", "greedy"}:
        return "best_intent"
    if value in {"uniform_random", "uniform", "random"}:
        return "uniform_random"
    return "learned_sample"


def _recondition_intent_fields_for_role(
    env: Any, obs_dict: dict, observer_is_offense: bool
) -> dict:
    if not isinstance(obs_dict, dict):
        return obs_dict
    if "players" in obs_dict and "globals" in obs_dict:
        return obs_dict
    base_env = getattr(env, "unwrapped", env)
    try:
        fields = base_env.get_intent_observation_fields(bool(observer_is_offense))
    except Exception:
        fields = {}
    if fields:
        for key, value in fields.items():
            obs_dict[key] = np.array(value, dtype=np.float32, copy=True)
    if "globals" in obs_dict:
        try:
            obs_dict["globals"] = base_env.patch_globals_with_intent_features(
                obs_dict["globals"], bool(observer_is_offense)
            )
        except Exception:
            pass
    return obs_dict


def _selector_neutralize_observation(single_obs: dict[str, Any], num_intents: int) -> dict[str, Any]:
    selector_obs = clone_observation_dict(single_obs)
    patch_intent_in_observation(
        selector_obs,
        0,
        max(1, int(num_intents)),
        active=0.0,
        visible=0.0,
        age_norm=0.0,
    )
    return selector_obs


def _build_selector_observation(
    env: Any,
    base_obs: dict[str, Any],
    *,
    role_flag_offense: float,
    num_intents: int,
) -> dict[str, Any]:
    selector_obs = clone_observation_dict(base_obs)
    selector_obs["role_flag"] = np.array([float(role_flag_offense)], dtype=np.float32)
    _recondition_intent_fields_for_role(env, selector_obs, True)
    return _selector_neutralize_observation(selector_obs, num_intents)


def _sample_selector_intent(
    *,
    training_params: Optional[dict[str, Any]],
    policy_model: Any,
    selector_obs: dict[str, Any],
    num_intents: int,
    allow_uniform_fallback: bool,
    selection_mode: Optional[str],
    rng: np.random.Generator | None,
) -> dict[str, Any]:
    rng = rng or np.random.default_rng()
    mode = _normalize_selector_intent_selection_mode(selection_mode)
    alpha = _selector_alpha_current(training_params, policy_model)
    policy_obj = getattr(policy_model, "policy", None)
    if policy_obj is None or not hasattr(policy_obj, "get_intent_selector_outputs"):
        if allow_uniform_fallback:
            return {
                "intent_index": int(rng.integers(0, max(1, int(num_intents)))),
                "used_selector": False,
                "alpha": float(alpha),
                "value": None,
            }
        return {"intent_index": None, "used_selector": False, "alpha": float(alpha), "value": None}

    if mode == "uniform_random":
        return {
            "intent_index": int(rng.integers(0, max(1, int(num_intents)))),
            "used_selector": False,
            "alpha": float(alpha),
            "value": None,
        }

    try:
        with torch.no_grad():
            logits, values = policy_obj.get_intent_selector_outputs(selector_obs)
            logits_t = torch.as_tensor(logits, dtype=torch.float32)
            values_t = torch.as_tensor(values, dtype=torch.float32).reshape(-1)
    except Exception:
        if allow_uniform_fallback:
            return {
                "intent_index": int(rng.integers(0, max(1, int(num_intents)))),
                "used_selector": False,
                "alpha": float(alpha),
                "value": None,
            }
        return {"intent_index": None, "used_selector": False, "alpha": float(alpha), "value": None}

    if mode == "best_intent":
        chosen = torch.argmax(logits_t.reshape(1, -1), dim=-1).reshape(-1)
        return {
            "intent_index": int(chosen[0].item()),
            "used_selector": True,
            "alpha": float(alpha),
            "value": float(values_t[0].item()) if values_t.numel() > 0 else None,
        }

    if alpha > 0.0 and float(rng.random()) < alpha:
        dist = torch.distributions.Categorical(logits=logits_t)
        chosen = dist.sample().reshape(-1)
        return {
            "intent_index": int(chosen[0].item()),
            "used_selector": True,
            "alpha": float(alpha),
            "value": float(values_t[0].item()) if values_t.numel() > 0 else None,
        }

    if allow_uniform_fallback:
        return {
            "intent_index": int(rng.integers(0, max(1, int(num_intents)))),
            "used_selector": False,
            "alpha": float(alpha),
            "value": None,
        }
    return {"intent_index": None, "used_selector": False, "alpha": float(alpha), "value": None}


def apply_rollout_segment_start(
    env: Any,
    base_obs: dict[str, Any] | None,
    *,
    training_params: dict[str, Any] | None,
    unified_policy: Any,
    opponent_policy: Any,
    user_team: Team | None,
    role_flag_offense: float,
    allow_uniform_fallback: bool,
    selection_mode: str | None = None,
) -> dict[str, Any]:
    del unified_policy, user_team
    if not isinstance(base_obs, dict):
        return {"applied": False, "obs": base_obs, "used_selector": False, "intent_index": None}
    base_env = getattr(env, "unwrapped", env)
    if not bool(getattr(base_env, "enable_intent_learning", False)):
        return {"applied": False, "obs": base_obs, "used_selector": False, "intent_index": None}
    if not bool(getattr(base_env, "intent_active", False)):
        return {"applied": False, "obs": base_obs, "used_selector": False, "intent_index": None}
    if not _selector_runtime_enabled_for_policy(opponent_policy):
        return {"applied": False, "obs": base_obs, "used_selector": False, "intent_index": None}
    num_intents = max(1, int(getattr(base_env, "num_intents", 1)))
    selector_obs = _build_selector_observation(
        env,
        base_obs,
        role_flag_offense=float(role_flag_offense),
        num_intents=num_intents,
    )
    result = _sample_selector_intent(
        training_params=training_params,
        policy_model=opponent_policy,
        selector_obs=selector_obs,
        num_intents=num_intents,
        allow_uniform_fallback=allow_uniform_fallback,
        selection_mode=selection_mode,
        rng=getattr(base_env, "_rng", None),
    )
    intent_index = result.get("intent_index")
    if intent_index is None:
        return {
            "applied": False,
            "obs": base_obs,
            "used_selector": False,
            "intent_index": None,
            "alpha": float(result.get("alpha", 0.0) or 0.0),
            "value": result.get("value"),
        }
    try:
        remaining = int(getattr(base_env, "intent_commitment_steps", 0) or 0)
        setter = getattr(base_env, "set_offense_intent_state", None)
        if callable(setter):
            setter(
                int(intent_index),
                intent_active=True,
                intent_age=0,
                intent_commitment_remaining=remaining,
            )
    except Exception:
        return {
            "applied": False,
            "obs": base_obs,
            "used_selector": False,
            "intent_index": None,
            "alpha": float(result.get("alpha", 0.0) or 0.0),
            "value": result.get("value"),
        }
    rebuilt_obs = clone_observation_dict(base_obs)
    role_flag_arr = np.asarray(base_obs.get("role_flag", np.asarray([-1.0], dtype=np.float32)))
    role_flag_value = float(role_flag_arr.reshape(-1)[0]) if role_flag_arr.size > 0 else -1.0
    rebuilt_obs["role_flag"] = np.array([role_flag_value], dtype=np.float32)
    _recondition_intent_fields_for_role(env, rebuilt_obs, bool(role_flag_value > 0.0))
    return {
        "applied": True,
        "obs": rebuilt_obs,
        "used_selector": bool(result.get("used_selector", False)),
        "intent_index": int(intent_index),
        "alpha": float(result.get("alpha", 0.0) or 0.0),
        "value": result.get("value"),
    }


def _selector_completed_pass_boundary(info: Any) -> bool:
    if not isinstance(info, dict):
        return False
    action_results = info.get("action_results", {})
    if not isinstance(action_results, dict):
        return False
    passes = action_results.get("passes", {})
    if not isinstance(passes, dict):
        return False
    for pass_result in passes.values():
        if isinstance(pass_result, dict) and bool(pass_result.get("success")):
            return True
    return False


def maybe_apply_rollout_multisegment_boundary(
    env: Any,
    base_obs: dict[str, Any] | None,
    *,
    info: Any,
    done: bool,
    training_params: dict[str, Any] | None,
    unified_policy: Any,
    opponent_policy: Any,
    user_team: Team | None,
    role_flag_offense: float,
    selector_segment_index: int,
    selection_mode: str | None = None,
) -> dict[str, Any]:
    del unified_policy, user_team
    if bool(done):
        return {
            "reason": None,
            "selector_segment_index": int(selector_segment_index),
            "obs": base_obs,
        }
    if not isinstance(training_params, dict) or not bool(
        training_params.get("intent_selector_multiselect_enabled", False)
    ):
        return {
            "reason": None,
            "selector_segment_index": int(selector_segment_index),
            "obs": base_obs,
        }
    base_env = getattr(env, "unwrapped", env)
    segment_length = int(getattr(base_env, "intent_age", 0) or 0)
    commitment_steps = int(getattr(base_env, "intent_commitment_steps", 0) or 0)
    min_play_steps = max(
        1, int(training_params.get("intent_selector_min_play_steps", 3) or 3)
    )
    reason = None
    if segment_length >= max(1, commitment_steps):
        reason = "commitment_timeout"
    elif segment_length >= min_play_steps and _selector_completed_pass_boundary(info):
        reason = "completed_pass"
    if reason is None:
        return {
            "reason": None,
            "selector_segment_index": int(selector_segment_index),
            "obs": base_obs,
        }
    result = apply_rollout_segment_start(
        env,
        base_obs,
        training_params=training_params,
        unified_policy=None,
        opponent_policy=opponent_policy,
        user_team=None,
        role_flag_offense=float(role_flag_offense),
        allow_uniform_fallback=True,
        selection_mode=selection_mode,
    )
    if isinstance(info, dict):
        info["intent_segment_boundary"] = 1.0
        info["intent_segment_boundary_reason"] = str(reason)
    return {
        "reason": str(reason),
        "selector_segment_index": int(selector_segment_index) + 1,
        "obs": result.get("obs", base_obs),
        "used_selector": bool(result.get("used_selector", False)),
        "start_source": (
            "selector" if bool(result.get("used_selector", False)) else "uniform_fallback"
        ),
    }


class SelfPlayEnvWrapper(gym.Wrapper):
    """Wrap an env to manage opponent actions and resolve illegal actions.

    - training_strategy: strategy for the current training team
    - opponent_strategy: strategy for the frozen opponent
    - deterministic_opponent: when True, use deterministic choice in sampling
    - action_space: exposes only the training team's actions to the learning policy
    """

    def __init__(
        self,
        env: gym.Env,
        opponent_policy: Any,  # PPO or str path
        training_strategy: IllegalActionStrategy = IllegalActionStrategy.SAMPLE_PROB,
        opponent_strategy: IllegalActionStrategy = IllegalActionStrategy.BEST_PROB,
        deterministic_opponent: bool = False,
    ) -> None:
        super().__init__(env)
        self.opponent_policy = opponent_policy
        self.training_strategy = training_strategy
        self.opponent_strategy = opponent_strategy
        self.deterministic_opponent = bool(deterministic_opponent)
        self._set_team_ids()
        self._configure_action_space()
        self._apply_pass_mode_to_policy(self.opponent_policy)
        self._opponent_selector_segment_index = 0

    def _current_pass_mode(self) -> str:
        try:
            return str(getattr(self.env.unwrapped, "pass_mode", "directional")).lower()
        except Exception:
            return "directional"

    def _apply_pass_mode_to_policy(self, policy_obj: Any) -> None:
        if policy_obj is None:
            return
        try:
            mode = self._current_pass_mode()
            policy = getattr(policy_obj, "policy", None)
            if policy is None:
                policy = policy_obj
            if hasattr(policy, "set_pass_mode"):
                policy.set_pass_mode(mode)
        except Exception:
            pass

    def _ensure_opponent_loaded(self):
        # Allow passing a policy path string to avoid pickling full PPO
        if isinstance(self.opponent_policy, str):
            self.opponent_policy = load_ppo_for_inference(
                self.opponent_policy,
                device="cpu",
            )
        self._apply_pass_mode_to_policy(self.opponent_policy)

    def _set_team_ids(self) -> None:
        if self.env.unwrapped.training_team == Team.OFFENSE:
            self.training_player_ids = list(self.env.unwrapped.offense_ids)
            self.opponent_player_ids = list(self.env.unwrapped.defense_ids)
        else:
            self.training_player_ids = list(self.env.unwrapped.defense_ids)
            self.opponent_player_ids = list(self.env.unwrapped.offense_ids)
        # Precompute numpy indices for fast masking
        self._training_player_indices = np.array(self.training_player_ids, dtype=int)
        self._opponent_player_indices = np.array(self.opponent_player_ids, dtype=int)

    def _configure_action_space(self) -> None:
        """Expose a reduced MultiDiscrete action space for the training team."""
        base_space = getattr(self.env, "action_space", None)
        if not isinstance(base_space, spaces.MultiDiscrete):
            raise TypeError(
                "SelfPlayEnvWrapper requires underlying env to use MultiDiscrete action space"
            )
        if len(self.training_player_ids) == 0:
            raise ValueError("Training player list cannot be empty")
        training_nvec = [int(base_space.nvec[i]) for i in self.training_player_ids]
        self.action_space = spaces.MultiDiscrete(training_nvec)

    def _recondition_intent_fields_for_role(
        self, obs_dict: dict, observer_is_offense: bool
    ) -> None:
        """Update role-dependent intent fields after role-flag flips."""
        if not isinstance(obs_dict, dict):
            return
        # Set-observation pipelines no longer carry intent fields in the low-level
        # observation dict. Runtime play conditioning is handled policy-side, so do
        # not re-inject legacy intent keys or overwrite the compact globals vector.
        if "players" in obs_dict and "globals" in obs_dict:
            return
        try:
            env = self.env.unwrapped
        except Exception:
            return
        try:
            fields = env.get_intent_observation_fields(bool(observer_is_offense))
        except Exception:
            fields = {}
        if fields:
            for key, value in fields.items():
                obs_dict[key] = np.array(value, dtype=np.float32, copy=True)
        if "globals" in obs_dict:
            try:
                obs_dict["globals"] = env.patch_globals_with_intent_features(
                    obs_dict["globals"], bool(observer_is_offense)
                )
            except Exception:
                pass

    def reset(self, **kwargs):
        try:
            obs, info = self.env.reset(**kwargs)
            self._last_obs = obs
            self._opponent_selector_segment_index = 0
            self._set_team_ids()
            # Ensure action space matches the (potentially updated) training team
            self._configure_action_space()
            self._apply_pass_mode_to_policy(self.opponent_policy)
            obs = self._maybe_apply_frozen_offense_segment_start(obs)
            self._last_obs = obs
            return obs, info
        except Exception as e:
            # Provide detailed context so SubprocVecEnv surfaces useful diagnostics
            raise RuntimeError(
                f"SelfPlayEnvWrapper.reset failed: {type(e).__name__}: {e}"
            ) from e

    def step(self, action):
        # Build opponent observation with flipped role flag for unified policy
        opponent_obs = dict(self._last_obs)
        try:
            # Opponent is offense when training team is defense, and vice versa
            opponent_is_offense = self.env.unwrapped.training_team != Team.OFFENSE
            opponent_obs["role_flag"] = (
                np.array([1.0], dtype=np.float32)
                if opponent_is_offense
                else np.array([-1.0], dtype=np.float32)
            )
            self._recondition_intent_fields_for_role(opponent_obs, opponent_is_offense)
        except Exception:
            pass

        # Opponent raw actions and probs
        try:
            self._ensure_opponent_loaded()
            sync_policy_runtime_intent_override_from_env(
                self.opponent_policy,
                self.env,
                observer_is_offense=bool(opponent_is_offense),
            )
            opp_actions_raw, _ = self.opponent_policy.predict(
                opponent_obs, deterministic=self.deterministic_opponent
            )
            action_mask = self._last_obs["action_mask"]
        except Exception as e:
            raise RuntimeError(
                f"SelfPlayEnvWrapper.step opponent predict failed: {type(e).__name__}: {e}"
            ) from e

        opp_actions_raw = np.array(opp_actions_raw, dtype=int).reshape(-1)
        action_len = int(opp_actions_raw.shape[0])
        action_mask = np.asarray(action_mask)

        # Legacy checkpoints may emit full n_players actions while newer set-attention
        # checkpoints emit players_per_side actions. Normalize both to opponent team size.
        if action_len == len(self.opponent_player_ids):
            opp_pred_actions = opp_actions_raw
        elif action_len == int(getattr(self.env.unwrapped, "n_players", action_len)):
            opp_pred_actions = opp_actions_raw[self._opponent_player_indices]
        else:
            opp_pred_actions = opp_actions_raw[: len(self.opponent_player_ids)]

        # Resolve opponent illegal actions using only their players
        opponent_mask = action_mask[self._opponent_player_indices]
        try:
            opp_probs_full = get_policy_action_probabilities(
                self.opponent_policy, opponent_obs
            )
        except Exception:
            opp_probs_full = None

        opponent_probs = None
        if opp_probs_full is not None:
            try:
                total_players = self.env.unwrapped.n_players
            except Exception:
                total_players = len(opp_probs_full)

            if len(opp_probs_full) == total_players:
                opponent_probs = [opp_probs_full[int(pid)] for pid in self.opponent_player_ids]
            else:
                opponent_probs = opp_probs_full[: len(self.opponent_player_ids)]

        opp_actions = resolve_illegal_actions(
            np.array(opp_pred_actions),
            opponent_mask,
            self.opponent_strategy,
            self.deterministic_opponent,
            opponent_probs,
        )

        action = np.array(action, dtype=int)
        expected_dims = len(self.training_player_ids)
        if action.shape[0] != expected_dims:
            raise ValueError(
                f"Expected {expected_dims} training actions, received shape {action.shape}"
            )
        training_mask = action_mask[self._training_player_indices]
        n_actions = action_mask.shape[1]
        uniform_probs = [
            np.ones(n_actions, dtype=np.float32) for _ in range(expected_dims)
        ]
        training_actions = resolve_illegal_actions(
            action,
            training_mask,
            self.training_strategy,
            False,
            uniform_probs,
        )

        # Build full action vector for the underlying environment
        full_action = np.zeros(self.env.unwrapped.n_players, dtype=int)
        for idx, pid in enumerate(self.training_player_ids):
            full_action[pid] = int(training_actions[idx])
        for idx, pid in enumerate(self.opponent_player_ids):
            full_action[pid] = int(opp_actions[idx])

        try:
            obs, reward, done, truncated, info = self.env.step(full_action)
        except Exception as e:
            raise RuntimeError(
                f"SelfPlayEnvWrapper.step env.step failed: {type(e).__name__}: {e}"
            ) from e
        obs = self._maybe_apply_frozen_offense_boundary(obs, info, done)
        # (Reverted) do not mutate info here
        self._last_obs = obs
        return obs, reward, done, truncated, info

    def _frozen_offense_selector_training_params(self) -> Optional[dict[str, Any]]:
        if getattr(self.env.unwrapped, "training_team", None) != Team.DEFENSE:
            return None
        try:
            self._ensure_opponent_loaded()
        except Exception:
            return None
        model = self.opponent_policy
        policy_obj = getattr(model, "policy", None)
        try:
            has_selector = bool(
                policy_obj is not None
                and hasattr(policy_obj, "has_intent_selector")
                and policy_obj.has_intent_selector()
            )
        except Exception:
            has_selector = False
        if not has_selector:
            return None
        has_runtime_attrs = all(
            hasattr(model, attr_name)
            for attr_name in (
                "intent_selector_alpha_start",
                "intent_selector_alpha_end",
                "intent_selector_alpha_warmup_steps",
                "intent_selector_alpha_ramp_steps",
                "intent_selector_multiselect_enabled",
                "intent_selector_min_play_steps",
            )
        )
        if not bool(getattr(model, "intent_selector_enabled", False)) and not has_runtime_attrs:
            return None
        return {
            "intent_selector_enabled": True,
            "intent_selector_mode": "integrated",
            "intent_selector_alpha_start": float(
                getattr(model, "intent_selector_alpha_start", 0.0)
            ),
            "intent_selector_alpha_end": float(
                getattr(model, "intent_selector_alpha_end", 0.0)
            ),
            "intent_selector_alpha_warmup_steps": int(
                getattr(model, "intent_selector_alpha_warmup_steps", 0)
            ),
            "intent_selector_alpha_ramp_steps": int(
                getattr(model, "intent_selector_alpha_ramp_steps", 1)
            ),
            "intent_selector_multiselect_enabled": bool(
                getattr(model, "intent_selector_multiselect_enabled", False)
            ),
            "intent_selector_min_play_steps": int(
                getattr(model, "intent_selector_min_play_steps", 3)
            ),
        }

    def _maybe_apply_frozen_offense_segment_start(self, obs):
        if not isinstance(obs, dict):
            return obs
        training_params = self._frozen_offense_selector_training_params()
        if not isinstance(training_params, dict):
            return obs
        try:
            result = apply_rollout_segment_start(
                self.env,
                obs,
                training_params=training_params,
                unified_policy=None,
                opponent_policy=self.opponent_policy,
                user_team=Team.DEFENSE,
                role_flag_offense=1.0,
                allow_uniform_fallback=False,
                selection_mode="learned_sample",
            )
            self._opponent_selector_segment_index = 0
            return result.get("obs", obs)
        except Exception:
            return obs

    def _maybe_apply_frozen_offense_boundary(self, obs, info, done):
        if not isinstance(obs, dict):
            if done:
                self._opponent_selector_segment_index = 0
            return obs
        training_params = self._frozen_offense_selector_training_params()
        if not isinstance(training_params, dict):
            if done:
                self._opponent_selector_segment_index = 0
            return obs
        try:
            result = maybe_apply_rollout_multisegment_boundary(
                self.env,
                obs,
                info=info,
                done=bool(done),
                training_params=training_params,
                unified_policy=None,
                opponent_policy=self.opponent_policy,
                user_team=Team.DEFENSE,
                role_flag_offense=1.0,
                selector_segment_index=int(self._opponent_selector_segment_index),
                selection_mode="learned_sample",
            )
            self._opponent_selector_segment_index = int(
                result.get(
                    "selector_segment_index", self._opponent_selector_segment_index
                )
            )
            if done:
                self._opponent_selector_segment_index = 0
            return result.get("obs", obs)
        except Exception:
            if done:
                self._opponent_selector_segment_index = 0
            return obs

    # Provide a top-level hook so VecEnv.env_method('set_phi_beta', ...) does not
    # trigger Gymnasium's deprecated attribute forwarding warnings. This forwards
    # directly to the unwrapped environment implementation.
    def set_phi_beta(self, value: float) -> None:  # pragma: no cover - thin shim
        try:
            self.env.unwrapped.set_phi_beta(float(value))
        except Exception:
            pass

    def set_pass_arc_degrees(
        self, value: float
    ) -> None:  # pragma: no cover - thin shim
        try:
            self.env.unwrapped.set_pass_arc_degrees(float(value))
        except Exception:
            pass

    def set_pass_oob_turnover_prob(
        self, value: float
    ) -> None:  # pragma: no cover - thin shim
        try:
            self.env.unwrapped.set_pass_oob_turnover_prob(float(value))
        except Exception:
            pass

    def set_intent_null_prob(self, value: float) -> None:  # pragma: no cover - thin shim
        try:
            self.env.unwrapped.set_intent_null_prob(float(value))
        except Exception:
            pass

    def set_intent_visible_to_defense_prob(
        self, value: float
    ) -> None:  # pragma: no cover - thin shim
        try:
            self.env.unwrapped.set_intent_visible_to_defense_prob(float(value))
        except Exception:
            pass

    def set_defense_intent_null_prob(
        self, value: float
    ) -> None:  # pragma: no cover - thin shim
        try:
            self.env.unwrapped.set_defense_intent_null_prob(float(value))
        except Exception:
            pass

    def set_offense_intent_state(
        self,
        intent_index: int,
        *,
        intent_active: bool = True,
        intent_age: int = 0,
        intent_commitment_remaining: Optional[int] = None,
    ) -> None:  # pragma: no cover - thin shim
        try:
            self.env.unwrapped.set_offense_intent_state(
                int(intent_index),
                intent_active=bool(intent_active),
                intent_age=int(intent_age),
                intent_commitment_remaining=intent_commitment_remaining,
            )
            if (
                getattr(self.env.unwrapped, "training_team", None) == Team.OFFENSE
                and isinstance(getattr(self, "_last_obs", None), dict)
            ):
                self._recondition_intent_fields_for_role(self._last_obs, True)
        except Exception:
            pass

    def get_offense_intent_override(self):  # pragma: no cover - thin shim
        try:
            getter = getattr(self.env.unwrapped, "get_offense_intent_override", None)
            if callable(getter):
                return getter()
        except Exception:
            pass
        return None

    def get_intent_observation_fields(
        self, observer_is_offense: bool
    ):  # pragma: no cover - thin shim
        try:
            getter = getattr(self.env.unwrapped, "get_intent_observation_fields", None)
            if callable(getter):
                return getter(bool(observer_is_offense))
        except Exception:
            pass
        return {}

    def get_profile_stats(self):  # pragma: no cover - thin shim
        try:
            return self.env.unwrapped.get_profile_stats()
        except Exception:
            return {}

    def reset_profile_stats(self) -> None:  # pragma: no cover - thin shim
        try:
            self.env.unwrapped.reset_profile_stats()
        except Exception:
            pass
