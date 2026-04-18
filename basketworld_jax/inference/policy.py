from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from basketworld.envs.basketworld_env_v2 import Team

from basketworld_jax.checkpoints import load_checkpoint
from basketworld_jax.env.minimal import (
    build_action_masks_batch,
    build_kernel_static_from_env,
    build_flat_observation_batch,
    snapshot_state_from_env,
    stack_state_snapshots,
)
from basketworld_jax.models import (
    ActorCriticSpec,
    actor_critic_forward,
    apply_action_mask,
)
from basketworld_jax.train.cli import ensure_jax_available


METADATA_FILENAME = "metadata.json"
STATE_SUBDIR = "state"


def is_checkpoint_path(path: str | Path) -> bool:
    checkpoint_path = Path(path)
    return checkpoint_path.is_dir() and (checkpoint_path / METADATA_FILENAME).is_file() and (
        checkpoint_path / STATE_SUBDIR
    ).exists()


class JAXInferenceModel:
    def __init__(self, checkpoint_path: str | Path) -> None:
        self.checkpoint_path = str(checkpoint_path)
        self.jax, self.jnp = ensure_jax_available("basketworld_jax/inference/policy.py")
        payload = load_checkpoint(checkpoint_path)
        self.params = self.jax.device_put(payload["params"])
        self.spec = ActorCriticSpec(**dict(payload["policy_spec"]))
        self.metadata = {
            "checkpoint_path": str(checkpoint_path),
            "update_index": int(payload.get("update_index", 0)),
            "saved_at": payload.get("saved_at"),
            "policy_spec": dict(payload.get("policy_spec", {})),
            "trainer_config": dict(payload.get("trainer_config", {})),
            "frozen_config": dict(payload.get("frozen_config", {})),
            "checkpoint_version": int(payload.get("checkpoint_version", 0)),
        }
        self._prepared_env = None
        self._prepared_observer_is_offense = True
        self._sample_key = self.jax.random.PRNGKey(0)
        self._static_cache: dict[tuple[int, bool], Any] = {}

    def _resolve_base_env(self, env):
        return getattr(env, "unwrapped", env)

    def _build_static_for_role(self, env, observer_is_offense: bool):
        base_env = self._resolve_base_env(env)
        cache_key = (id(base_env), bool(observer_is_offense))
        cached = self._static_cache.get(cache_key)
        if cached is not None:
            return cached

        desired_team = Team.OFFENSE if bool(observer_is_offense) else Team.DEFENSE
        original_team = getattr(base_env, "training_team", Team.OFFENSE)
        try:
            base_env.training_team = desired_team
            static = build_kernel_static_from_env(base_env, self.jnp)
        finally:
            base_env.training_team = original_team

        self._static_cache[cache_key] = static
        return static

    def _state_from_env(self, env):
        base_env = self._resolve_base_env(env)
        snapshot = snapshot_state_from_env(base_env)
        return stack_state_snapshots([snapshot], self.jnp)

    def prepare_for_role(self, env, *, observer_is_offense: bool) -> None:
        self._prepared_env = env
        self._prepared_observer_is_offense = bool(observer_is_offense)
        self._build_static_for_role(env, observer_is_offense)

    def _team_outputs(self, env, observer_is_offense: bool):
        static = self._build_static_for_role(env, observer_is_offense)
        state = self._state_from_env(env)
        flat_obs = build_flat_observation_batch(static, state, self.jnp)
        full_action_mask = build_action_masks_batch(static, state, self.jnp)
        team_ids = static.offense_ids if bool(observer_is_offense) else static.defense_ids
        team_action_mask = self.jnp.take(full_action_mask, team_ids, axis=1)
        forward_out = actor_critic_forward(self.params, flat_obs, self.spec, self.jnp)
        masked_out = apply_action_mask(
            forward_out["flat_policy_logits"],
            team_action_mask,
            self.spec,
            self.jax,
            self.jnp,
        )
        return masked_out

    def predict(self, obs=None, deterministic: bool = False):
        if self._prepared_env is None:
            raise RuntimeError("JAXInferenceModel.predict called before prepare_for_role.")

        masked_out = self._team_outputs(
            self._prepared_env,
            self._prepared_observer_is_offense,
        )
        if deterministic:
            actions = masked_out["deterministic_actions"]
        else:
            self._sample_key, sample_key = self.jax.random.split(self._sample_key)
            actions = self.jax.random.categorical(
                sample_key,
                masked_out["masked_logits"],
                axis=-1,
            ).astype(self.jnp.int32)
        actions_np = np.asarray(self.jax.device_get(actions[0]), dtype=np.int32)
        return actions_np, None

    def action_probabilities(self, obs=None):
        if self._prepared_env is None:
            return None
        masked_out = self._team_outputs(
            self._prepared_env,
            self._prepared_observer_is_offense,
        )
        probs = np.asarray(self.jax.device_get(masked_out["probs"][0]), dtype=np.float32)
        return [probs[idx] for idx in range(probs.shape[0])]


def load_inference_model(path: str | Path) -> JAXInferenceModel:
    if not is_checkpoint_path(path):
        raise FileNotFoundError(f"Not a JAX checkpoint directory: {path}")
    return JAXInferenceModel(path)
