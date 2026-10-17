import os
import re
from typing import Any

from fastapi import HTTPException

from app.backend.inference_adapters import unwrap_inference_model


def _compute_param_counts_from_policy(policy_obj):
    """Return the total parameter count for a JAX inference policy."""
    try:
        model = unwrap_inference_model(policy_obj)
    except Exception:
        return None
    if model is None:
        return None
    params = getattr(model, "params", None)
    if params is None:
        return None
    try:
        leaves = model.jax.tree_util.tree_leaves(params)
        total = sum(int(getattr(leaf, "size", 0)) for leaf in leaves)
    except Exception:
        return None
    return {
        "total": int(total),
        "shared_trunk": None,
        "policy_heads": None,
        "value_heads": None,
        "log_std": 0,
    }


def list_policies_from_run(client, run_id):
    """Return sorted list of loadable model artifact paths for a run."""
    artifacts = client.list_artifacts(run_id, "models")
    jax_checkpoints = [
        f.path
        for f in artifacts
        if f.path.endswith("latest")
        or f.path.endswith("phase_a_latest")
        or re.search(r"(?:phase_a_)?update_\d+$", f.path)
    ]

    combined = list(dict.fromkeys(jax_checkpoints))
    combined.sort(key=_model_artifact_sort_key)
    return combined


def _jax_update_index(path: str) -> int | None:
    match = re.search(r"(?:phase_a_)?update_(\d+)$", path)
    return int(match.group(1)) if match else None


def _model_artifact_sort_key(path: str) -> tuple[int, int, str]:
    jax_index = _jax_update_index(path)
    if jax_index is not None:
        return (0, jax_index, path)
    if path.endswith("latest") or path.endswith("phase_a_latest"):
        return (0, 10**12, path)
    return (1, 0, path)


def _get_run_tags(client, run_id) -> dict[str, str]:
    try:
        run = client.get_run(run_id)
    except Exception:
        return {}
    return dict(getattr(getattr(run, "data", None), "tags", {}) or {})


def get_unified_policy_path(client, run_id, policy_name: str | None):
    """Return artifact path for unified policy (downloaded locally). If name None, use latest."""
    # Use a persistent cache directory for MLflow checkpoint downloads.
    cache_dir = os.path.join("episodes", "_policy_cache")
    os.makedirs(cache_dir, exist_ok=True)

    choices = list_policies_from_run(client, run_id)
    if not choices:
        raise HTTPException(status_code=404, detail="No unified policy artifacts found.")

    run_tags = _get_run_tags(client, run_id)
    latest_jax_artifact = str(run_tags.get("jax_latest_checkpoint_artifact", "")).strip()

    if policy_name and any(p.endswith(policy_name) for p in choices):
        chosen_artifact = next(p for p in choices if p.endswith(policy_name))
    elif policy_name:
        target_update_index = _jax_update_index(policy_name)
        if target_update_index is not None:
            chosen_artifact = next(
                (p for p in choices if _jax_update_index(p) == target_update_index),
                None,
            )
        else:
            chosen_artifact = None
        if chosen_artifact is None:
            raise HTTPException(
                status_code=404,
                detail=f"JAX checkpoint artifact not found: {policy_name}",
            )
    elif latest_jax_artifact and latest_jax_artifact in choices:
        chosen_artifact = latest_jax_artifact
    else:
        chosen_artifact = sorted(choices, key=_model_artifact_sort_key)[-1]

    return client.download_artifacts(run_id, chosen_artifact, cache_dir)
