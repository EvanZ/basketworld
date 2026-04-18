import os
import re
from typing import Any

import torch
from fastapi import HTTPException

from app.backend.inference_adapters import unwrap_policy_module


def _compute_param_counts_from_policy(policy_obj):
    """Return trainable parameter counts for shared trunk, policy heads, and value heads."""
    try:
        model = unwrap_policy_module(policy_obj)
    except Exception:
        return None
    if model is None:
        return None

    def count_params(module):
        try:
            return sum(p.numel() for p in module.parameters() if getattr(p, "requires_grad", False))
        except Exception:
            return 0

    shared_trunk = 0
    for attr in ("features_extractor", "mlp_extractor"):
        if hasattr(model, attr):
            shared_trunk += count_params(getattr(model, attr))

    policy_heads = 0
    for attr in ("action_net", "action_net_offense", "action_net_defense"):
        if hasattr(model, attr):
            policy_heads += count_params(getattr(model, attr))

    log_std_count = 0
    if hasattr(model, "log_std") and isinstance(model.log_std, torch.nn.Parameter):
        if model.log_std.requires_grad:
            try:
                log_std_count = int(model.log_std.numel())
            except Exception:
                log_std_count = 0

    value_heads = 0
    for attr in ("value_net", "value_net_offense", "value_net_defense"):
        if hasattr(model, attr):
            value_heads += count_params(getattr(model, attr))

    total = shared_trunk + policy_heads + value_heads + log_std_count
    return {
        "total": int(total),
        "shared_trunk": int(shared_trunk),
        "policy_heads": int(policy_heads + log_std_count),
        "value_heads": int(value_heads),
        "log_std": int(log_std_count),
    }


def list_policies_from_run(client, run_id):
    """Return sorted list of loadable model artifact paths for a run."""
    artifacts = client.list_artifacts(run_id, "models")
    unified = [f.path for f in artifacts if f.path.endswith(".zip") and "unified" in f.path]
    jax_checkpoints = [
        f.path
        for f in artifacts
        if f.path.endswith("latest")
        or f.path.endswith("phase_a_latest")
        or re.search(r"(?:phase_a_)?update_\d+$", f.path)
    ]

    combined = list(dict.fromkeys(unified + jax_checkpoints))
    combined.sort(key=_model_artifact_sort_key)
    return combined


def _model_artifact_sort_key(path: str) -> tuple[int, int, str]:
    zip_match = re.search(r"_(\d+)\.zip$", path)
    if zip_match:
        return (0, int(zip_match.group(1)), path)
    jax_match = re.search(r"(?:phase_a_)?update_(\d+)$", path)
    if jax_match:
        return (1, int(jax_match.group(1)), path)
    if path.endswith("latest") or path.endswith("phase_a_latest"):
        return (1, 10**12, path)
    return (2, 0, path)


def _get_run_tags(client, run_id) -> dict[str, str]:
    try:
        run = client.get_run(run_id)
    except Exception:
        return {}
    return dict(getattr(getattr(run, "data", None), "tags", {}) or {})


def get_unified_policy_path(client, run_id, policy_name: str | None):
    """Return artifact path for unified policy (downloaded locally). If name None, use latest."""
    # Use a persistent cache directory to avoid deletion before PPO.load
    cache_dir = os.path.join("episodes", "_policy_cache")
    os.makedirs(cache_dir, exist_ok=True)

    choices = list_policies_from_run(client, run_id)
    if not choices:
        raise HTTPException(status_code=404, detail="No unified policy artifacts found.")

    run_tags = _get_run_tags(client, run_id)
    latest_jax_artifact = str(run_tags.get("jax_phase_a_latest_checkpoint_artifact", "")).strip()

    if policy_name and any(p.endswith(policy_name) for p in choices):
        chosen_artifact = next(p for p in choices if p.endswith(policy_name))
    elif latest_jax_artifact and latest_jax_artifact in choices and not any(
        p.endswith(".zip") for p in choices
    ):
        chosen_artifact = latest_jax_artifact
    else:
        chosen_artifact = sorted(choices, key=_model_artifact_sort_key)[-1]

    return client.download_artifacts(run_id, chosen_artifact, cache_dir)


def get_latest_policies_from_run(client, run_id, tmpdir):
    """Downloads the latest policies from a given MLflow run."""
    artifacts = client.list_artifacts(run_id, "models")
    if not artifacts:
        raise HTTPException(status_code=404, detail="No model artifacts found in the specified run.")

    latest_offense_path = max(
        [f.path for f in artifacts if "offense" in f.path],
        key=lambda p: int(re.search(r"_(\d+)\.zip", p).group(1)),
    )
    latest_defense_path = max(
        [f.path for f in artifacts if "defense" in f.path],
        key=lambda p: int(re.search(r"_(\d+)\.zip", p).group(1)),
    )

    offense_local_path = client.download_artifacts(run_id, latest_offense_path, tmpdir)
    defense_local_path = client.download_artifacts(run_id, latest_defense_path, tmpdir)

    return offense_local_path, defense_local_path
