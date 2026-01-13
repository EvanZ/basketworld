import os
import re

import torch
from fastapi import HTTPException


def _compute_param_counts_from_policy(policy_obj):
    """Return trainable parameter counts for shared trunk, policy heads, and value heads."""
    try:
        model = policy_obj.policy
    except Exception:
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
    """Return sorted list of unified policy artifact paths for a run."""
    artifacts = client.list_artifacts(run_id, "models")
    unified = [f.path for f in artifacts if f.path.endswith(".zip") and "unified" in f.path]

    def sort_key(p):
        match = re.search(r"_(\d+)\.zip$", p)
        return int(match.group(1)) if match else 0

    unified.sort(key=sort_key)
    return unified


def get_unified_policy_path(client, run_id, policy_name: str | None):
    """Return artifact path for unified policy (downloaded locally). If name None, use latest."""
    # Use a persistent cache directory to avoid deletion before PPO.load
    cache_dir = os.path.join("episodes", "_policy_cache")
    os.makedirs(cache_dir, exist_ok=True)

    unified_paths = list_policies_from_run(client, run_id)
    choices = unified_paths
    if not choices:
        raise HTTPException(status_code=404, detail="No unified policy artifacts found.")

    if policy_name and any(p.endswith(policy_name) for p in choices):
        chosen_artifact = next(p for p in choices if p.endswith(policy_name))
    else:
        chosen_artifact = choices[-1]  # latest

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
