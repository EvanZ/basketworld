from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from basketworld.utils.action_resolution import get_policy_action_probabilities
from basketworld.utils.intent_policy_sensitivity import (
    sync_policy_runtime_intent_override_from_env,
)
from basketworld.utils.policy_loading import load_ppo_for_inference
from basketworld_jax.inference import (
    is_checkpoint_path,
    load_inference_model,
)


@dataclass(frozen=True)
class InferenceCapabilities:
    player_controls: bool = True
    self_play: bool = True
    observation_panel: bool = True
    eval: bool = True
    playbook: bool = True
    mcts: bool = True
    attention: bool = True
    env_training_tabs: bool = True
    state_values: bool = True
    q_values: bool = True


class InferencePolicyAdapter:
    def __init__(
        self,
        raw_model: Any,
        *,
        backend_kind: str,
        capabilities: InferenceCapabilities | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.raw_model = raw_model
        self.backend_kind = str(backend_kind)
        self.capabilities = capabilities or InferenceCapabilities()
        self.metadata = dict(metadata or {})

    def capability_dict(self) -> dict[str, bool]:
        return asdict(self.capabilities)

    def predict(self, obs, deterministic: bool = False):
        raise NotImplementedError

    def action_probabilities(self, obs):
        return None

    def prepare_for_role(self, env, *, observer_is_offense: bool) -> None:
        return None

    def set_pass_mode(self, mode_value: str) -> None:
        policy = getattr(self.raw_model, "policy", None)
        if policy is None or not hasattr(policy, "set_pass_mode"):
            return
        try:
            policy.set_pass_mode(mode_value)
        except Exception:
            return

    def __getattr__(self, name: str):
        return getattr(self.raw_model, name)


class SB3PPOInferenceAdapter(InferencePolicyAdapter):
    def __init__(self, raw_model: Any) -> None:
        super().__init__(
            raw_model,
            backend_kind="sb3",
            capabilities=InferenceCapabilities(),
        )

    def predict(self, obs, deterministic: bool = False):
        return self.raw_model.predict(obs, deterministic=deterministic)

    def action_probabilities(self, obs):
        return get_policy_action_probabilities(self.raw_model, obs)

    def prepare_for_role(self, env, *, observer_is_offense: bool) -> None:
        sync_policy_runtime_intent_override_from_env(
            self.raw_model,
            env,
            observer_is_offense=bool(observer_is_offense),
        )


class JAXInferenceAdapter(InferencePolicyAdapter):
    def __init__(self, raw_model: Any) -> None:
        super().__init__(
            raw_model,
            backend_kind="jax_phase_a",
            capabilities=InferenceCapabilities(
                player_controls=True,
                self_play=True,
                observation_panel=True,
                eval=True,
                playbook=False,
                mcts=False,
                attention=False,
                env_training_tabs=False,
                state_values=False,
                q_values=False,
            ),
            metadata=dict(getattr(raw_model, "metadata", {}) or {}),
        )

    def predict(self, obs, deterministic: bool = False):
        return self.raw_model.predict(obs, deterministic=deterministic)

    def action_probabilities(self, obs):
        return self.raw_model.action_probabilities(obs)

    def prepare_for_role(self, env, *, observer_is_offense: bool) -> None:
        self.raw_model.prepare_for_role(
            env,
            observer_is_offense=bool(observer_is_offense),
        )


def load_sb3_policy_adapter(
    path: str,
    *,
    device: str = "cpu",
    custom_objects: dict[str, Any] | None = None,
) -> SB3PPOInferenceAdapter:
    model = load_ppo_for_inference(
        path,
        device=device,
        custom_objects=custom_objects,
    )
    return SB3PPOInferenceAdapter(model)


def load_jax_adapter(path: str) -> JAXInferenceAdapter:
    model = load_inference_model(path)
    return JAXInferenceAdapter(model)


def load_inference_policy(
    path: str,
    *,
    device: str = "cpu",
    custom_objects: dict[str, Any] | None = None,
) -> InferencePolicyAdapter:
    candidate_path = Path(path)
    if is_checkpoint_path(candidate_path):
        return load_jax_adapter(str(candidate_path))
    return load_sb3_policy_adapter(
        str(candidate_path),
        device=device,
        custom_objects=custom_objects,
    )


def unwrap_inference_model(policy_obj: Any) -> Any:
    return getattr(policy_obj, "raw_model", policy_obj)


def unwrap_policy_module(policy_obj: Any) -> Any:
    model = unwrap_inference_model(policy_obj)
    return getattr(model, "policy", None)


def prepare_policy_for_role(policy_obj: Any, env, *, observer_is_offense: bool) -> None:
    if policy_obj is None:
        return
    if hasattr(policy_obj, "prepare_for_role"):
        policy_obj.prepare_for_role(env, observer_is_offense=bool(observer_is_offense))
        return
    raw_model = unwrap_inference_model(policy_obj)
    sync_policy_runtime_intent_override_from_env(
        raw_model,
        env,
        observer_is_offense=bool(observer_is_offense),
    )


def policy_action_probabilities(policy_obj: Any, obs):
    if policy_obj is None:
        return None
    if hasattr(policy_obj, "action_probabilities"):
        return policy_obj.action_probabilities(obs)
    raw_model = unwrap_inference_model(policy_obj)
    return get_policy_action_probabilities(raw_model, obs)


def get_policy_backend_kind(policy_obj: Any) -> str | None:
    if policy_obj is None:
        return None
    return str(getattr(policy_obj, "backend_kind", "sb3"))


def get_policy_capabilities(policy_obj: Any) -> dict[str, bool] | None:
    if policy_obj is None:
        return None
    capabilities = getattr(policy_obj, "capabilities", None)
    if capabilities is None:
        return asdict(InferenceCapabilities())
    if isinstance(capabilities, InferenceCapabilities):
        return asdict(capabilities)
    if isinstance(capabilities, dict):
        return {str(k): bool(v) for k, v in capabilities.items()}
    return None


def get_policy_metadata(policy_obj: Any) -> dict[str, Any] | None:
    if policy_obj is None:
        return None
    metadata = getattr(policy_obj, "metadata", None)
    if metadata is None:
        return None
    if isinstance(metadata, dict):
        return dict(metadata)
    return None
