from __future__ import annotations

from typing import Any

import numpy as np
import torch
from basketworld.envs.basketworld_env_v2 import Team

from basketworld.utils.intent_policy_sensitivity import (
    clone_observation_dict,
    patch_intent_in_observation,
    sync_policy_runtime_intent_override_from_env,
)

from .observations import (
    _apply_intent_fields_for_role,
    _clone_obs_with_role_flag,
    _ensure_set_obs,
    rebuild_observation_from_env,
)


def selector_runtime_enabled(training_params: dict | None, policy_model: Any) -> bool:
    if not isinstance(training_params, dict):
        return False
    if not bool(training_params.get("intent_selector_enabled", False)):
        return False
    if str(training_params.get("intent_selector_mode", "callback")).lower() != "integrated":
        return False
    policy_obj = getattr(policy_model, "policy", None)
    if policy_obj is None or not hasattr(policy_obj, "has_intent_selector"):
        return False
    try:
        return bool(policy_obj.has_intent_selector())
    except Exception:
        return False


def selector_multiselect_enabled(training_params: dict | None) -> bool:
    if not isinstance(training_params, dict):
        return False
    return bool(training_params.get("intent_selector_multiselect_enabled", False))


def current_offense_controller_policy(
    *,
    unified_policy: Any,
    opponent_policy: Any,
    user_team: Team | None,
):
    if (user_team or Team.OFFENSE) == Team.OFFENSE:
        return unified_policy
    if opponent_policy is not None:
        return opponent_policy
    return unified_policy


def selector_runtime_active_for_rollout(
    training_params: dict | None,
    *,
    unified_policy: Any,
    opponent_policy: Any,
    user_team: Team | None,
) -> bool:
    return selector_runtime_enabled(
        training_params,
        current_offense_controller_policy(
            unified_policy=unified_policy,
            opponent_policy=opponent_policy,
            user_team=user_team,
        ),
    )


def selector_alpha_current(training_params: dict | None, policy_model: Any) -> float:
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


def selector_eps_current(training_params: dict | None, policy_model: Any) -> float:
    if not isinstance(training_params, dict):
        return 0.0
    t = int(getattr(policy_model, "num_timesteps", 0) or 0)
    start = float(training_params.get("intent_selector_eps_start", 0.0) or 0.0)
    end = float(training_params.get("intent_selector_eps_end", 0.0) or 0.0)
    warmup = max(0, int(training_params.get("intent_selector_eps_warmup_steps", 0) or 0))
    ramp = max(0, int(training_params.get("intent_selector_eps_ramp_steps", 1) or 0))
    if t < warmup:
        return float(start)
    if ramp <= 0:
        return float(end)
    progress = min(1.0, max(0.0, (t - warmup) / float(ramp)))
    return float(start + progress * (end - start))


def normalize_selector_intent_selection_mode(mode: str | None) -> str:
    value = str(mode or "learned_sample").strip().lower()
    if value in {"learned_sample", "sample", "runtime_sample", "runtime"}:
        return "learned_sample"
    if value in {"best_intent", "best", "argmax", "greedy"}:
        return "best_intent"
    if value in {"uniform_random", "uniform", "random"}:
        return "uniform_random"
    return "learned_sample"


def selector_neutralize_observation(single_obs: dict[str, Any], num_intents: int) -> dict[str, Any]:
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


def selector_apply_eps_floor(
    selector_probs: torch.Tensor, selector_eps: torch.Tensor | float
) -> torch.Tensor:
    probs = selector_probs.clamp_min(1e-8)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    eps = torch.as_tensor(selector_eps, dtype=probs.dtype, device=probs.device)
    while eps.ndim < probs.ndim:
        eps = eps.unsqueeze(-1)
    uniform = torch.full_like(probs, 1.0 / float(max(1, probs.shape[-1])))
    mixed = (1.0 - eps) * probs + eps * uniform
    return mixed / mixed.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def build_offense_selector_observation(
    policy_model: Any,
    env: Any,
    base_obs: dict[str, Any] | None,
    *,
    role_flag_offense: float,
) -> dict[str, Any] | None:
    if policy_model is None or env is None or base_obs is None:
        return None
    prepared_obs = _ensure_set_obs(policy_model, env, base_obs)
    if prepared_obs is None:
        return None
    conditioned_obs = _clone_obs_with_role_flag(prepared_obs, float(role_flag_offense))
    conditioned_obs = _apply_intent_fields_for_role(
        conditioned_obs,
        env,
        float(role_flag_offense),
    )
    num_intents = max(
        1,
        int(getattr(getattr(env, "unwrapped", env), "num_intents", 1)),
    )
    return selector_neutralize_observation(conditioned_obs, num_intents)


def selector_sample_intent(
    training_params: dict | None,
    policy_model: Any,
    selector_obs: dict[str, Any],
    *,
    num_intents: int,
    allow_uniform_fallback: bool,
    selection_mode: str | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    rng = rng or np.random.default_rng()
    mode = normalize_selector_intent_selection_mode(selection_mode)
    alpha = selector_alpha_current(training_params, policy_model)
    selector_eps = selector_eps_current(training_params, policy_model)

    if mode == "uniform_random":
        return {
            "intent_index": int(rng.integers(0, max(1, int(num_intents)))),
            "used_selector": False,
            "alpha": float(alpha),
            "selector_eps": float(selector_eps),
            "value": None,
            "selection_mode": mode,
        }

    try:
        with torch.no_grad():
            logits, values = policy_model.policy.get_intent_selector_outputs(selector_obs)
            logits_t = torch.as_tensor(logits, dtype=torch.float32).reshape(1, -1)
            values_t = torch.as_tensor(values, dtype=torch.float32).reshape(-1)
            raw_probs_t = torch.softmax(logits_t, dim=-1)
            mixed_probs_t = selector_apply_eps_floor(raw_probs_t, selector_eps)
    except Exception:
        if allow_uniform_fallback:
            return {
                "intent_index": int(rng.integers(0, max(1, int(num_intents)))),
                "used_selector": False,
                "alpha": float(alpha),
                "selector_eps": float(selector_eps),
                "value": None,
                "selection_mode": mode,
            }
        return {
            "intent_index": None,
            "used_selector": False,
            "alpha": float(alpha),
            "selector_eps": float(selector_eps),
            "value": None,
            "selection_mode": mode,
        }

    if mode == "best_intent":
        chosen = torch.argmax(logits_t, dim=-1).reshape(-1)
        return {
            "intent_index": int(chosen[0].item()),
            "used_selector": True,
            "alpha": float(alpha),
            "selector_eps": float(selector_eps),
            "value": float(values_t[0].item()) if values_t.numel() > 0 else None,
            "selection_mode": mode,
        }

    if alpha > 0.0 and float(rng.random()) < alpha:
        dist = torch.distributions.Categorical(probs=mixed_probs_t)
        chosen = dist.sample().reshape(-1)
        return {
            "intent_index": int(chosen[0].item()),
            "used_selector": True,
            "alpha": float(alpha),
            "selector_eps": float(selector_eps),
            "value": float(values_t[0].item()) if values_t.numel() > 0 else None,
            "selection_mode": mode,
        }
    if allow_uniform_fallback:
        return {
            "intent_index": int(rng.integers(0, max(1, int(num_intents)))),
            "used_selector": False,
            "alpha": float(alpha),
            "selector_eps": float(selector_eps),
            "value": None,
            "selection_mode": mode,
        }
    return {
        "intent_index": None,
        "used_selector": False,
        "alpha": float(alpha),
        "selector_eps": float(selector_eps),
        "value": None,
        "selection_mode": mode,
    }


def selector_ranked_intent_preferences(
    *,
    training_params: dict | None,
    env: Any,
    base_obs: dict[str, Any] | None,
    unified_policy: Any,
    opponent_policy: Any,
    user_team: Team | None,
    role_flag_offense: float,
) -> dict[str, Any] | None:
    if not selector_runtime_active_for_rollout(
        training_params,
        unified_policy=unified_policy,
        opponent_policy=opponent_policy,
        user_team=user_team,
    ):
        return None
    base_env = getattr(env, "unwrapped", env)
    if not bool(getattr(base_env, "enable_intent_learning", False)):
        return None
    if not bool(getattr(base_env, "intent_active", False)):
        return None
    offense_policy = current_offense_controller_policy(
        unified_policy=unified_policy,
        opponent_policy=opponent_policy,
        user_team=user_team,
    )
    selector_obs = build_offense_selector_observation(
        offense_policy,
        env,
        base_obs,
        role_flag_offense=float(role_flag_offense),
    )
    if selector_obs is None:
        return None
    policy_obj = getattr(offense_policy, "policy", None)
    if policy_obj is None or not hasattr(policy_obj, "get_intent_selector_outputs"):
        return None
    try:
        with torch.no_grad():
            logits, values = policy_obj.get_intent_selector_outputs(selector_obs)
            logits_t = torch.as_tensor(logits, dtype=torch.float32).reshape(1, -1)
            raw_probs_t = torch.softmax(logits_t, dim=-1)
            alpha_current = float(selector_alpha_current(training_params, offense_policy))
            eps_current = float(selector_eps_current(training_params, offense_policy))
            mixed_probs_t = selector_apply_eps_floor(raw_probs_t, eps_current)
        logits_np = logits_t.detach().cpu().numpy().reshape(-1)
        raw_probs_np = raw_probs_t.detach().cpu().numpy().reshape(-1)
        mixed_probs_np = mixed_probs_t.detach().cpu().numpy().reshape(-1)
        num_classes = max(1, raw_probs_np.shape[0])
        uniform_probs_np = np.full(num_classes, 1.0 / float(num_classes), dtype=np.float32)
        mode = normalize_selector_intent_selection_mode(None)
        if mode == "uniform_random":
            deployed_probs_np = uniform_probs_np
        elif mode == "best_intent":
            deployed_probs_np = np.zeros_like(raw_probs_np)
            deployed_probs_np[int(np.argmax(raw_probs_np))] = 1.0
        else:
            deployed_probs_np = (
                alpha_current * mixed_probs_np
                + (1.0 - alpha_current) * uniform_probs_np
            )
        order = np.argsort(-deployed_probs_np, kind="stable")
        ranked = []
        for rank_idx, class_idx in enumerate(order.tolist(), start=1):
            ranked.append(
                {
                    "rank": int(rank_idx),
                    "intent_index": int(class_idx),
                    "prob": float(deployed_probs_np[class_idx]),
                    "raw_prob": float(raw_probs_np[class_idx]),
                    "mixed_prob": float(mixed_probs_np[class_idx]),
                    "deployed_prob": float(deployed_probs_np[class_idx]),
                    "logit": float(logits_np[class_idx]),
                }
            )
        value_estimate = None
        if values is not None:
            values_t = torch.as_tensor(values, dtype=torch.float32).reshape(-1)
            if values_t.numel() > 0:
                value_estimate = float(values_t[0].item())
        return {
            "alpha_current": float(alpha_current),
            "eps_current": float(eps_current),
            "selection_mode": mode,
            "value_estimate": value_estimate,
            "current_intent_index": int(getattr(base_env, "intent_index", 0)),
            "segment_index": int(getattr(base_env, "intent_age", 0)),
            "intent_probs": ranked,
        }
    except Exception:
        return None


def apply_rollout_segment_start(
    env: Any,
    base_obs: dict[str, Any] | None,
    *,
    training_params: dict | None,
    unified_policy: Any,
    opponent_policy: Any,
    user_team: Team | None,
    role_flag_offense: float,
    allow_uniform_fallback: bool,
    selection_mode: str | None = None,
) -> dict[str, Any]:
    if not selector_runtime_active_for_rollout(
        training_params,
        unified_policy=unified_policy,
        opponent_policy=opponent_policy,
        user_team=user_team,
    ):
        return {"applied": False, "obs": base_obs, "used_selector": False, "intent_index": None}
    base_env = getattr(env, "unwrapped", env)
    if not bool(getattr(base_env, "enable_intent_learning", False)):
        return {"applied": False, "obs": base_obs, "used_selector": False, "intent_index": None}
    if not bool(getattr(base_env, "intent_active", False)):
        return {"applied": False, "obs": base_obs, "used_selector": False, "intent_index": None}
    offense_policy = current_offense_controller_policy(
        unified_policy=unified_policy,
        opponent_policy=opponent_policy,
        user_team=user_team,
    )
    selector_obs = build_offense_selector_observation(
        offense_policy,
        env,
        base_obs,
        role_flag_offense=float(role_flag_offense),
    )
    if selector_obs is None:
        return {"applied": False, "obs": base_obs, "used_selector": False, "intent_index": None}
    result = selector_sample_intent(
        training_params,
        offense_policy,
        selector_obs,
        num_intents=max(1, int(getattr(base_env, "num_intents", 1))),
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
    apply_selected_offense_intent(
        env,
        int(intent_index),
        intent_commitment_steps=int(getattr(base_env, "intent_commitment_steps", 0)),
    )
    rebuilt_obs = rebuild_observation_from_env(env, current_obs=base_obs)
    sync_policy_runtime_intent_override_from_env(
        offense_policy, env, observer_is_offense=True
    )
    return {
        "applied": True,
        "obs": rebuilt_obs,
        "used_selector": bool(result.get("used_selector", False)),
        "intent_index": int(intent_index),
        "alpha": float(result.get("alpha", 0.0) or 0.0),
        "value": result.get("value"),
    }


def selector_completed_pass_boundary(info: Any) -> bool:
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


def selector_segment_boundary_reason(
    training_params: dict | None,
    *,
    segment_length: int,
    info: Any,
    intent_commitment_steps: int,
) -> str | None:
    if not selector_multiselect_enabled(training_params):
        return None
    if int(segment_length) >= max(1, int(intent_commitment_steps)):
        return "commitment_timeout"
    min_play_steps = max(
        1,
        int((training_params or {}).get("intent_selector_min_play_steps", 3) or 3),
    )
    if int(segment_length) >= min_play_steps and selector_completed_pass_boundary(info):
        return "completed_pass"
    return None


def maybe_apply_rollout_multisegment_boundary(
    env: Any,
    base_obs: dict[str, Any] | None,
    *,
    info: Any,
    done: bool,
    training_params: dict | None,
    unified_policy: Any,
    opponent_policy: Any,
    user_team: Team | None,
    role_flag_offense: float,
    selector_segment_index: int,
    selection_mode: str | None = None,
) -> dict[str, Any]:
    if bool(done):
        return {
            "reason": None,
            "selector_segment_index": int(selector_segment_index),
            "obs": base_obs,
        }
    if not selector_runtime_active_for_rollout(
        training_params,
        unified_policy=unified_policy,
        opponent_policy=opponent_policy,
        user_team=user_team,
    ):
        return {
            "reason": None,
            "selector_segment_index": int(selector_segment_index),
            "obs": base_obs,
        }
    base_env = getattr(env, "unwrapped", env)
    reason = selector_segment_boundary_reason(
        training_params,
        segment_length=int(getattr(base_env, "intent_age", 0)),
        info=info,
        intent_commitment_steps=int(getattr(base_env, "intent_commitment_steps", 0)),
    )
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
        unified_policy=unified_policy,
        opponent_policy=opponent_policy,
        user_team=user_team,
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
            "selector"
            if bool(result.get("used_selector", False))
            else "uniform_fallback"
        ),
    }


def apply_selected_offense_intent(
    env: Any,
    intent_index: int,
    *,
    intent_commitment_steps: int | None = None,
) -> None:
    base_env = getattr(env, "unwrapped", env)
    remaining = (
        int(intent_commitment_steps)
        if intent_commitment_steps is not None
        else int(getattr(base_env, "intent_commitment_steps", 0) or 0)
    )
    setter = getattr(base_env, "set_offense_intent_state", None)
    if callable(setter):
        setter(
            int(intent_index),
            intent_active=True,
            intent_age=0,
            intent_commitment_remaining=remaining,
        )
        return
    base_env.intent_active = True
    base_env.intent_index = int(intent_index)
    base_env.intent_age = 0
    base_env.intent_commitment_remaining = max(0, int(remaining))
