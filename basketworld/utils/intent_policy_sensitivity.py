from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import torch


_BASE_SET_GLOBAL_DIM = 4
_INTENT_GLOBAL_DIM = 3
_EPS = 1e-12


def clone_observation_dict(obs: dict) -> dict:
    """Return a copy of a dict observation with array-like values duplicated."""
    cloned: dict = {}
    for key, value in obs.items():
        try:
            cloned[key] = np.array(value, copy=True)
        except Exception:
            cloned[key] = value
    return cloned


def extract_single_env_observation(
    obs_payload: dict,
    env_idx: int,
    expected_batch_size: Optional[int] = None,
) -> dict:
    """Slice one environment observation out of a batched VecEnv dict payload."""
    if not isinstance(obs_payload, dict):
        raise TypeError("Expected dict observation payload.")

    single_obs: dict = {}
    env_idx = int(env_idx)
    expected = None if expected_batch_size is None else int(expected_batch_size)
    for key, value in obs_payload.items():
        arr = np.asarray(value)
        if expected is not None and arr.ndim > 0 and arr.shape[0] == expected:
            single_obs[key] = np.array(arr[env_idx], copy=True)
        else:
            single_obs[key] = np.array(arr, copy=True)
    return single_obs


def infer_num_intents(policy_or_model, default: int = 8) -> int:
    """Infer num_intents from a loaded policy/model when possible."""
    default = max(1, int(default))
    policy_obj = getattr(policy_or_model, "policy", policy_or_model)
    try:
        features_extractor = getattr(policy_obj, "features_extractor", None)
        if features_extractor is not None and hasattr(features_extractor, "num_intents"):
            return max(1, int(getattr(features_extractor, "num_intents")))
    except Exception:
        pass
    try:
        if hasattr(policy_obj, "num_intents"):
            return max(1, int(getattr(policy_obj, "num_intents")))
    except Exception:
        pass
    return default


def _set_scalar_field(obs: dict, key: str, value: float, batch_index: Optional[int]) -> None:
    if key not in obs:
        return
    arr = np.asarray(obs[key], dtype=np.float32)
    if batch_index is None:
        if arr.ndim == 0:
            obs[key] = np.asarray(value, dtype=np.float32)
        else:
            arr = np.array(arr, copy=True)
            arr[...] = float(value)
            obs[key] = arr
        return

    if arr.ndim == 0:
        obs[key] = np.full((batch_index + 1,), float(value), dtype=np.float32)
        return
    obs[key][int(batch_index)] = float(value)


def patch_intent_in_observation(
    obs: dict,
    intent_index: int,
    num_intents: int,
    *,
    active: float = 1.0,
    visible: float = 1.0,
    age_norm: Optional[float] = None,
    commitment_remaining_norm: Optional[float] = None,
    batch_index: Optional[int] = None,
) -> dict:
    """Patch intent-conditioned observation fields in-place and return `obs`."""
    z = int(max(0, min(max(1, int(num_intents)) - 1, int(intent_index))))
    active_f = float(active)
    visible_f = float(visible)
    if age_norm is None:
        age_norm = 0.0 if active_f > 0.5 else 0.0
    if commitment_remaining_norm is None:
        commitment_remaining_norm = 1.0 if active_f > 0.5 else 0.0
    if num_intents <= 1:
        index_norm = 0.0
    else:
        index_norm = float(z) / float(max(1, int(num_intents) - 1))

    globals_arr = obs.get("globals")
    if globals_arr is not None:
        globals_np = np.asarray(globals_arr, dtype=np.float32)
        # Set-observation wrapper appends intent globals to the 4 base globals.
        if globals_np.shape[-1] >= (_BASE_SET_GLOBAL_DIM + _INTENT_GLOBAL_DIM):
            if batch_index is None:
                globals_np = np.array(globals_np, copy=True)
                globals_np[-_INTENT_GLOBAL_DIM :] = np.asarray(
                    [index_norm, active_f, visible_f], dtype=np.float32
                )
                obs["globals"] = globals_np
            else:
                obs["globals"][int(batch_index), -_INTENT_GLOBAL_DIM :] = np.asarray(
                    [index_norm, active_f, visible_f], dtype=np.float32
                )

    _set_scalar_field(obs, "intent_index", float(z), batch_index)
    _set_scalar_field(obs, "intent_active", active_f, batch_index)
    _set_scalar_field(obs, "intent_visible", visible_f, batch_index)
    _set_scalar_field(obs, "intent_age_norm", float(age_norm), batch_index)
    _set_scalar_field(
        obs,
        "intent_commitment_remaining_norm",
        float(commitment_remaining_norm),
        batch_index,
    )
    return obs


def build_intent_variant_batch(
    single_obs: dict,
    num_intents: int,
    candidate_intents: Optional[Sequence[int]] = None,
    *,
    active: float = 1.0,
    visible: float = 1.0,
    age_norm: float = 0.0,
    commitment_remaining_norm: float = 1.0,
) -> tuple[dict, list[int]]:
    """Repeat one observation into a batch and patch each row with a different intent."""
    intents = (
        [int(z) for z in candidate_intents]
        if candidate_intents is not None
        else list(range(max(1, int(num_intents))))
    )
    batch: dict = {}
    for key, value in single_obs.items():
        arr = np.asarray(value)
        batch[key] = np.repeat(arr[None, ...], len(intents), axis=0)
    for batch_idx, z in enumerate(intents):
        patch_intent_in_observation(
            batch,
            z,
            num_intents=num_intents,
            active=active,
            visible=visible,
            age_norm=age_norm,
            commitment_remaining_norm=commitment_remaining_norm,
            batch_index=batch_idx,
        )
    return batch, intents


def get_policy_action_probabilities_tensor(policy_or_model, obs) -> Optional[np.ndarray]:
    """Return action probabilities as a tensor with shape (B, P, A)."""
    policy_obj = None
    try:
        policy_obj = getattr(policy_or_model, "policy", None)
        if policy_obj is None:
            policy_obj = policy_or_model
        if hasattr(policy_obj, "_extract_role_flag") and hasattr(policy_obj, "_current_role_flags"):
            try:
                policy_obj._current_role_flags = policy_obj._extract_role_flag(obs)
            except Exception:
                policy_obj._current_role_flags = None
        obs_tensor = policy_obj.obs_to_tensor(obs)[0]
        distributions = policy_obj.get_distribution(obs_tensor)
        if hasattr(distributions, "action_probabilities"):
            probs = distributions.action_probabilities()
            if probs.ndim == 2:
                probs = probs.unsqueeze(0)
            return probs.detach().cpu().numpy().astype(np.float64, copy=False)

        raw_distribution = getattr(distributions, "distribution", None)
        if raw_distribution is None:
            return None
        if not isinstance(raw_distribution, (list, tuple)):
            raw_distribution = [raw_distribution]

        per_player = []
        for dist in raw_distribution:
            probs = dist.probs
            if probs.ndim == 1:
                probs = probs.unsqueeze(0)
            per_player.append(probs)
        if not per_player:
            return None
        stacked = torch.stack(per_player, dim=1)
        return stacked.detach().cpu().numpy().astype(np.float64, copy=False)
    except Exception:
        return None
    finally:
        if policy_obj is not None and hasattr(policy_obj, "_current_role_flags"):
            policy_obj._current_role_flags = None


def apply_action_mask_to_probabilities(
    probabilities: np.ndarray,
    action_mask: np.ndarray,
) -> np.ndarray:
    """Zero out illegal actions and renormalize probabilities."""
    probs = np.asarray(probabilities, dtype=np.float64)
    mask = np.asarray(action_mask)
    if probs.ndim != 3:
        raise ValueError("Expected probability tensor with shape (B, P, A).")
    if mask.ndim == 2:
        mask = np.repeat(mask[None, ...], probs.shape[0], axis=0)
    if mask.shape != probs.shape:
        return _normalize_probabilities(probs)

    out = np.zeros_like(probs, dtype=np.float64)
    for batch_idx in range(probs.shape[0]):
        for player_idx in range(probs.shape[1]):
            legal = np.asarray(mask[batch_idx, player_idx] > 0, dtype=bool)
            if not np.any(legal):
                out[batch_idx, player_idx, 0] = 1.0
                continue
            vec = np.clip(probs[batch_idx, player_idx], _EPS, None)
            vec = np.where(legal, vec, 0.0)
            total = float(np.sum(vec))
            if total <= _EPS:
                out[batch_idx, player_idx, legal] = 1.0 / float(np.sum(legal))
            else:
                out[batch_idx, player_idx] = vec / total
    return out


def _normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    probs = np.clip(np.asarray(probabilities, dtype=np.float64), _EPS, None)
    denom = np.clip(np.sum(probs, axis=-1, keepdims=True), _EPS, None)
    return probs / denom


def _symmetric_kl_by_player(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p_norm = _normalize_probabilities(p)
    q_norm = _normalize_probabilities(q)
    return 0.5 * (
        np.sum(p_norm * np.log(p_norm / q_norm), axis=-1)
        + np.sum(q_norm * np.log(q_norm / p_norm), axis=-1)
    )


def _tv_by_player(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p_norm = _normalize_probabilities(p)
    q_norm = _normalize_probabilities(q)
    return 0.5 * np.sum(np.abs(p_norm - q_norm), axis=-1)


def compute_policy_sensitivity_metrics(
    policy_or_model,
    observations: Iterable[dict],
    *,
    num_intents: int,
    candidate_intents: Optional[Sequence[int]] = None,
    active: float = 1.0,
    visible: float = 1.0,
    age_norm: float = 0.0,
    commitment_remaining_norm: float = 1.0,
) -> dict[str, float]:
    """Measure how much action distributions move when only intent changes."""
    intents = (
        [int(z) for z in candidate_intents]
        if candidate_intents is not None
        else list(range(max(1, int(num_intents))))
    )
    if len(intents) < 2:
        return {
            "policy_kl_mean": 0.0,
            "policy_kl_max": 0.0,
            "policy_tv_mean": 0.0,
            "action_flip_rate": 0.0,
            "num_states": 0.0,
            "num_pairs": 0.0,
        }

    pair_kl_values: list[float] = []
    pair_tv_values: list[float] = []
    pair_flip_values: list[float] = []
    valid_states = 0

    for obs in observations:
        batch_obs, used_intents = build_intent_variant_batch(
            clone_observation_dict(obs),
            num_intents=num_intents,
            candidate_intents=intents,
            active=active,
            visible=visible,
            age_norm=age_norm,
            commitment_remaining_norm=commitment_remaining_norm,
        )
        probs = get_policy_action_probabilities_tensor(policy_or_model, batch_obs)
        if probs is None or probs.ndim != 3 or probs.shape[0] != len(used_intents):
            continue
        if "action_mask" in batch_obs:
            probs = apply_action_mask_to_probabilities(probs, batch_obs["action_mask"])
        else:
            probs = _normalize_probabilities(probs)

        state_pairs = 0
        for left in range(len(used_intents)):
            for right in range(left + 1, len(used_intents)):
                p = probs[left]
                q = probs[right]
                pair_kl_values.append(float(np.mean(_symmetric_kl_by_player(p, q))))
                pair_tv_values.append(float(np.mean(_tv_by_player(p, q))))
                pair_flip_values.append(
                    float(np.mean(np.argmax(p, axis=-1) != np.argmax(q, axis=-1)))
                )
                state_pairs += 1
        if state_pairs > 0:
            valid_states += 1

    if not pair_kl_values:
        return {
            "policy_kl_mean": 0.0,
            "policy_kl_max": 0.0,
            "policy_tv_mean": 0.0,
            "action_flip_rate": 0.0,
            "num_states": float(valid_states),
            "num_pairs": 0.0,
        }

    return {
        "policy_kl_mean": float(np.mean(pair_kl_values)),
        "policy_kl_max": float(np.max(pair_kl_values)),
        "policy_tv_mean": float(np.mean(pair_tv_values)),
        "action_flip_rate": float(np.mean(pair_flip_values)),
        "num_states": float(valid_states),
        "num_pairs": float(len(pair_kl_values)),
    }
