from __future__ import annotations

from typing import Any, Optional

from stable_baselines3 import PPO
from stable_baselines3.common.base_class import _convert_space
from stable_baselines3.common.save_util import load_from_zip_file


def get_sb3_custom_objects() -> dict[str, Any]:
    from basketworld.policies import SetAttentionDualCriticPolicy, SetAttentionExtractor
    from basketworld.utils.policies import (
        PassBiasDualCriticPolicy,
        PassBiasMultiInputPolicy,
    )

    return {
        "PassBiasDualCriticPolicy": PassBiasDualCriticPolicy,
        "PassBiasMultiInputPolicy": PassBiasMultiInputPolicy,
        "SetAttentionDualCriticPolicy": SetAttentionDualCriticPolicy,
        "SetAttentionExtractor": SetAttentionExtractor,
    }


def load_ppo_for_inference(
    path: str,
    *,
    device: str = "cpu",
    custom_objects: Optional[dict[str, Any]] = None,
) -> PPO:
    """Load a PPO checkpoint for prediction only, skipping optimizer restore.

    This is used for frozen opponents and offline analysis. Older checkpoints may
    have optimizer state that no longer matches the current policy parameter
    groups after architecture changes. For inference we only need the policy
    weights, not the optimizer state.
    """

    merged_custom_objects = get_sb3_custom_objects()
    if custom_objects:
        merged_custom_objects.update(custom_objects)

    data, params, _ = load_from_zip_file(
        path,
        device=device,
        custom_objects=merged_custom_objects,
    )
    assert data is not None, "No data found in the saved file"
    assert params is not None, "No params found in the saved file"

    if "policy_kwargs" in data:
        if "device" in data["policy_kwargs"]:
            del data["policy_kwargs"]["device"]
        if (
            "net_arch" in data["policy_kwargs"]
            and len(data["policy_kwargs"]["net_arch"]) > 0
        ):
            saved_net_arch = data["policy_kwargs"]["net_arch"]
            if isinstance(saved_net_arch, list) and isinstance(saved_net_arch[0], dict):
                data["policy_kwargs"]["net_arch"] = saved_net_arch[0]

    for key in {"observation_space", "action_space"}:
        if key not in data:
            raise KeyError(f"Missing saved {key}, can't restore policy for inference")
        data[key] = _convert_space(data[key])

    data.pop("env", None)

    model = PPO(
        policy=data["policy_class"],
        env=None,
        device=device,
        _init_setup_model=False,
    )
    model.__dict__.update(data)
    model._setup_model()

    policy_state = params.get("policy")
    if policy_state is None:
        raise KeyError("Saved checkpoint is missing policy parameters")
    model.policy.load_state_dict(policy_state, strict=True)

    if model.use_sde:
        model.policy.reset_noise()
    return model
