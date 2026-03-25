from __future__ import annotations

from typing import Any

import numpy as np
import torch

from basketworld.utils.intent_policy_sensitivity import (
    clone_observation_dict,
    patch_intent_in_observation,
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


def selector_sample_intent(
    training_params: dict | None,
    policy_model: Any,
    selector_obs: dict[str, Any],
    *,
    num_intents: int,
    allow_uniform_fallback: bool,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    rng = rng or np.random.default_rng()
    alpha = selector_alpha_current(training_params, policy_model)
    if alpha > 0.0 and float(rng.random()) < alpha:
        try:
            with torch.no_grad():
                logits, values = policy_model.policy.get_intent_selector_outputs(selector_obs)
                dist = torch.distributions.Categorical(logits=logits)
                chosen = dist.sample().reshape(-1)
                return {
                    "intent_index": int(chosen[0].item()),
                    "used_selector": True,
                    "alpha": float(alpha),
                    "value": float(values.reshape(-1)[0].item()),
                }
        except Exception:
            pass
    if allow_uniform_fallback:
        return {
            "intent_index": int(rng.integers(0, max(1, int(num_intents)))),
            "used_selector": False,
            "alpha": float(alpha),
            "value": None,
        }
    return {
        "intent_index": None,
        "used_selector": False,
        "alpha": float(alpha),
        "value": None,
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
