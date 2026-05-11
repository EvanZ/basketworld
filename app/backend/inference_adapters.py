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
    play_metadata: bool = True
    selector_distribution: bool = True
    per_intent_eval: bool = True
    play_shot_charts: bool = True
    manual_intent_override: bool = True


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

    def observation_vector(self, env, *, observer_is_offense: bool):
        return None

    def observation_tokens(self, env, *, observer_is_offense: bool):
        return None

    def attention_payload(self, env, *, observer_is_offense: bool):
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
        supports_attention = _jax_model_uses_attention(raw_model)
        metadata = dict(getattr(raw_model, "metadata", {}) or {})
        supports_play_metadata = _jax_model_uses_intents(raw_model, metadata)
        supports_selector_distribution = _jax_model_uses_selector(raw_model, metadata)
        super().__init__(
            raw_model,
            backend_kind="jax",
            capabilities=InferenceCapabilities(
                player_controls=True,
                self_play=True,
                observation_panel=True,
                eval=True,
                playbook=supports_play_metadata,
                mcts=False,
                attention=supports_attention,
                env_training_tabs=False,
                state_values=True,
                q_values=False,
                play_metadata=supports_play_metadata,
                selector_distribution=supports_selector_distribution,
                per_intent_eval=supports_play_metadata,
                play_shot_charts=supports_play_metadata,
                manual_intent_override=supports_play_metadata,
            ),
            metadata=metadata,
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

    def observation_vector(self, env, *, observer_is_offense: bool):
        return self.raw_model.observation_vector(
            env,
            observer_is_offense=bool(observer_is_offense),
        )

    def observation_tokens(self, env, *, observer_is_offense: bool):
        token_fn = getattr(self.raw_model, "observation_tokens", None)
        if not callable(token_fn):
            return None
        return token_fn(env, observer_is_offense=bool(observer_is_offense))

    def attention_payload(self, env, *, observer_is_offense: bool):
        if not _jax_model_uses_attention(self.raw_model):
            return None
        payload_fn = getattr(self.raw_model, "attention_payload", None)
        if not callable(payload_fn):
            return None
        return payload_fn(env, observer_is_offense=bool(observer_is_offense))

    def state_value(self, env, *, observer_is_offense: bool) -> float | None:
        value_fn = getattr(self.raw_model, "state_value", None)
        if not callable(value_fn):
            return None
        return value_fn(env, observer_is_offense=bool(observer_is_offense))

    def has_intent_selector(self) -> bool:
        selector_fn = getattr(self.raw_model, "has_intent_selector", None)
        if callable(selector_fn):
            try:
                return bool(selector_fn())
            except Exception:
                return False
        return _jax_model_uses_selector(self.raw_model, self.metadata)

    def get_intent_selector_outputs(self, obs=None):
        output_fn = getattr(self.raw_model, "get_intent_selector_outputs", None)
        if not callable(output_fn):
            return None, None
        return output_fn(obs)


def _jax_model_uses_attention(raw_model: Any) -> bool:
    spec_obj = getattr(raw_model, "spec", None)
    spec_model_type = getattr(spec_obj, "model_type", None)
    if spec_model_type is not None:
        return str(spec_model_type).lower() == "attention"
    metadata = getattr(raw_model, "metadata", None)
    if not isinstance(metadata, dict):
        return False
    policy_spec = metadata.get("policy_spec")
    if not isinstance(policy_spec, dict):
        return False
    return str(policy_spec.get("model_type", "")).lower() == "attention"


def _jax_policy_spec_dict(raw_model: Any, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    spec_obj = getattr(raw_model, "spec", None)
    if spec_obj is not None:
        keys = (
            "model_type",
            "num_intents",
            "intent_embedding_enabled",
            "intent_selector_enabled",
        )
        return {
            key: getattr(spec_obj, key)
            for key in keys
            if hasattr(spec_obj, key)
        }
    if isinstance(metadata, dict):
        policy_spec = metadata.get("policy_spec")
        if isinstance(policy_spec, dict):
            return dict(policy_spec)
    raw_metadata = getattr(raw_model, "metadata", None)
    if isinstance(raw_metadata, dict):
        policy_spec = raw_metadata.get("policy_spec")
        if isinstance(policy_spec, dict):
            return dict(policy_spec)
    return {}


def _jax_model_uses_intents(raw_model: Any, metadata: dict[str, Any] | None = None) -> bool:
    metadata = metadata if isinstance(metadata, dict) else getattr(raw_model, "metadata", None)
    policy_spec = _jax_policy_spec_dict(raw_model, metadata)
    env_config = (
        dict(metadata.get("env_config", {}) or {})
        if isinstance(metadata, dict)
        else {}
    )
    play_name_map = metadata.get("play_name_map") if isinstance(metadata, dict) else None
    try:
        num_intents = int(
            policy_spec.get("num_intents")
            or env_config.get("num_intents")
            or 0
        )
    except Exception:
        num_intents = 0
    return bool(
        num_intents > 0
        and (
            bool(policy_spec.get("intent_embedding_enabled", False))
            or bool(policy_spec.get("intent_selector_enabled", False))
            or bool(env_config.get("enable_intent_learning", False))
            or bool(play_name_map)
        )
    )


def _jax_model_uses_selector(raw_model: Any, metadata: dict[str, Any] | None = None) -> bool:
    policy_spec = _jax_policy_spec_dict(raw_model, metadata)
    return bool(policy_spec.get("intent_selector_enabled", False))


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


def policy_observation_vector(policy_obj: Any, env, *, observer_is_offense: bool):
    if policy_obj is None:
        return None
    if hasattr(policy_obj, "observation_vector"):
        return policy_obj.observation_vector(
            env,
            observer_is_offense=bool(observer_is_offense),
        )
    return None


def policy_observation_tokens(policy_obj: Any, env, *, observer_is_offense: bool):
    if policy_obj is None:
        return None
    token_fn = getattr(policy_obj, "observation_tokens", None)
    if callable(token_fn):
        return token_fn(env, observer_is_offense=bool(observer_is_offense))
    raw_model = unwrap_inference_model(policy_obj)
    token_fn = getattr(raw_model, "observation_tokens", None)
    if callable(token_fn):
        return token_fn(env, observer_is_offense=bool(observer_is_offense))
    return None


def policy_attention_payload(policy_obj: Any, env, *, observer_is_offense: bool):
    if policy_obj is None:
        return None
    payload_fn = getattr(policy_obj, "attention_payload", None)
    if callable(payload_fn):
        return payload_fn(env, observer_is_offense=bool(observer_is_offense))
    raw_model = unwrap_inference_model(policy_obj)
    payload_fn = getattr(raw_model, "attention_payload", None)
    if callable(payload_fn):
        return payload_fn(env, observer_is_offense=bool(observer_is_offense))
    return None


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
