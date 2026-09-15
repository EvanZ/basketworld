from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from basketworld.envs.basketworld_env_v2 import Team

from basketworld_jax.checkpoints import load_checkpoint
from basketworld_jax.env.minimal import (
    build_action_masks_batch,
    build_kernel_static_from_env,
    build_policy_intent_context_batch,
    build_policy_observation_batch,
    build_token_observation_components_batch,
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
            "env_config": dict(payload.get("env_config", {}) or {}),
            "last_metrics": dict(payload.get("last_metrics", {}) or {}),
            "checkpoint_version": int(payload.get("checkpoint_version", 0)),
        }
        for key in ("play_name_metadata", "play_name_map", "model_codename"):
            if key in payload:
                self.metadata[key] = payload[key]
        self._prepared_env = None
        self._prepared_observer_is_offense = True
        self._sample_key = self.jax.random.PRNGKey(0)
        self._static_cache: dict[tuple[int, bool], Any] = {}
        self._last_team_outputs: dict[str, Any] | None = None
        self._masked_runner = self._build_masked_runner()
        self._selector_runner = self._build_selector_runner()

    def _build_masked_runner(self):
        @self.jax.jit
        def _runner(params, flat_obs, team_action_mask, intent_context):
            forward_out = actor_critic_forward(
                params,
                flat_obs,
                self.spec,
                self.jnp,
                intent_context=intent_context,
            )
            masked_out = apply_action_mask(
                forward_out["flat_policy_logits"],
                team_action_mask,
                self.spec,
                self.jax,
                self.jnp,
            )
            return {
                **masked_out,
                "attention_weights": forward_out["attention_weights"],
                "values": forward_out["values"],
            }

        return _runner

    def _build_selector_runner(self):
        @self.jax.jit
        def _runner(params, flat_obs):
            batch_size = flat_obs.shape[0]
            neutral_context = {
                "intent_index": self.jnp.zeros((batch_size,), dtype=self.jnp.int32),
                "intent_gate": self.jnp.zeros((batch_size,), dtype=self.jnp.float32),
            }
            forward_out = actor_critic_forward(
                params,
                flat_obs,
                self.spec,
                self.jnp,
                intent_context=neutral_context,
            )
            return {
                "selector_logits": forward_out["selector_logits"],
                "selector_values": forward_out["selector_values"],
            }

        return _runner

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

    def _state_from_snapshot(self, snapshot):
        return stack_state_snapshots([snapshot], self.jnp)

    def _snapshot_cache_key(self, env, observer_is_offense: bool, snapshot: dict[str, Any]):
        base_env = self._resolve_base_env(env)
        array_keys = (
            "positions",
            "offense_lane_steps",
            "defense_lane_steps",
            "layup_pct",
            "three_pt_pct",
            "dunk_pct",
        )
        scalar_keys = (
            "ball_holder",
            "shot_clock",
            "step_count",
            "episode_ended",
            "pressure_exposure",
            "cached_phi",
            "offense_score",
            "defense_score",
            "assist_active",
            "assist_passer",
            "assist_recipient",
            "assist_expires_at",
            "intent_index",
            "intent_active",
            "intent_age",
            "intent_commitment_remaining",
            "intent_visible_to_defense",
            "defense_intent_index",
            "defense_intent_active",
            "defense_intent_age",
            "defense_intent_commitment_remaining",
        )
        arrays = tuple(np.asarray(snapshot[key]).tobytes() for key in array_keys)
        scalars = tuple(snapshot[key] for key in scalar_keys)
        return (id(base_env), bool(observer_is_offense), arrays, scalars)

    def prepare_for_role(self, env, *, observer_is_offense: bool) -> None:
        self._prepared_env = env
        self._prepared_observer_is_offense = bool(observer_is_offense)
        self._build_static_for_role(env, observer_is_offense)

    def _team_outputs(self, env, observer_is_offense: bool):
        base_env = self._resolve_base_env(env)
        snapshot = snapshot_state_from_env(base_env)
        cache_key = self._snapshot_cache_key(env, observer_is_offense, snapshot)
        if self._last_team_outputs is not None and self._last_team_outputs.get("key") == cache_key:
            return self._last_team_outputs["masked_out"]

        static = self._build_static_for_role(env, observer_is_offense)
        state = self._state_from_snapshot(snapshot)
        flat_obs = build_policy_observation_batch(
            static,
            state,
            self.jnp,
            model_type=self.spec.model_type,
            rebound_win_prob_features=bool(getattr(self.spec, "rebound_win_prob_features", False)),
            rebound_target_observation_features=bool(getattr(self.spec, "rebound_target_observation_features", True)),
        )
        full_action_mask = build_action_masks_batch(static, state, self.jnp)
        team_ids = static.offense_ids if bool(observer_is_offense) else static.defense_ids
        team_action_mask = self.jnp.take(full_action_mask, team_ids, axis=1)
        intent_context = build_policy_intent_context_batch(static, state, self.jnp)
        masked_out = self._masked_runner(
            self.params,
            flat_obs,
            team_action_mask,
            intent_context,
        )
        self._last_team_outputs = {
            "key": cache_key,
            "masked_out": masked_out,
        }
        return masked_out

    def observation_vector(self, env, *, observer_is_offense: bool):
        base_env = self._resolve_base_env(env)
        static = self._build_static_for_role(env, observer_is_offense)
        state = self._state_from_snapshot(snapshot_state_from_env(base_env))
        flat_obs = build_policy_observation_batch(
            static,
            state,
            self.jnp,
            model_type=self.spec.model_type,
            rebound_win_prob_features=bool(getattr(self.spec, "rebound_win_prob_features", False)),
            rebound_target_observation_features=bool(getattr(self.spec, "rebound_target_observation_features", True)),
        )
        return np.asarray(self.jax.device_get(flat_obs[0]), dtype=np.float32)

    def observation_tokens(self, env, *, observer_is_offense: bool):
        if str(self.spec.model_type) != "attention":
            return None
        base_env = self._resolve_base_env(env)
        static = self._build_static_for_role(env, observer_is_offense)
        state = self._state_from_snapshot(snapshot_state_from_env(base_env))
        players, globals_vec, _ = build_token_observation_components_batch(
            static,
            state,
            static.training_role_flag,
            self.jnp,
            rebound_win_prob_features=bool(
                getattr(self.spec, "rebound_win_prob_features", False)
            ),
            rebound_target_observation_features=bool(getattr(self.spec, "rebound_target_observation_features", True)),
        )
        rebound_target_observation_features = bool(
            getattr(self.spec, "rebound_target_observation_features", True)
        )
        player_labels = [
            "q_norm", "r_norm", "role", "has_ball", "layup_pct", "three_pt_pct",
            "dunk_pct", "lane_steps_norm", "expected_points", "turnover_probability",
            "pass_steal_probability", "distance_to_ball", "distance_to_best_ep_player",
            "distance_to_nearest_opponent", "distance_to_nearest_teammate",
        ]
        if rebound_target_observation_features:
            player_labels.append("distance_to_expected_rebound_target")
        player_labels.append("rebound_skill")
        if rebound_target_observation_features:
            player_labels.append("rebound_skill_specialist")
        if bool(getattr(self.spec, "rebound_win_prob_features", False)):
            player_labels.append("rebound_win_probability")
        global_labels = ["shot_clock_norm", "pressure_exposure", "hoop_q_norm", "hoop_r_norm"]
        if rebound_target_observation_features:
            global_labels.extend(
                ["expected_rebound_target_q", "expected_rebound_target_r", "target_entropy"]
            )
        if bool(getattr(self.spec, "rebound_win_prob_features", False)):
            global_labels.append("offensive_rebound_probability")
        return {
            "players": np.asarray(self.jax.device_get(players[0]), dtype=np.float32),
            "players_labels": player_labels,
            "globals": np.asarray(self.jax.device_get(globals_vec[0]), dtype=np.float32),
            "globals_labels": global_labels,
        }

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

    def state_value(self, env, *, observer_is_offense: bool) -> float:
        masked_out = self._team_outputs(env, bool(observer_is_offense))
        values = np.asarray(self.jax.device_get(masked_out["values"]), dtype=np.float32).reshape(-1)
        if values.size == 0:
            return 0.0
        return float(values[0])

    def has_intent_selector(self) -> bool:
        return bool(getattr(self.spec, "intent_selector_enabled", False))

    def get_intent_selector_outputs(self, obs=None):
        if not self.has_intent_selector() or self._prepared_env is None:
            return None, None
        base_env = self._resolve_base_env(self._prepared_env)
        static = self._build_static_for_role(base_env, observer_is_offense=True)
        state = self._state_from_snapshot(snapshot_state_from_env(base_env))
        flat_obs = build_policy_observation_batch(
            static,
            state,
            self.jnp,
            model_type=self.spec.model_type,
            rebound_win_prob_features=bool(getattr(self.spec, "rebound_win_prob_features", False)),
            rebound_target_observation_features=bool(getattr(self.spec, "rebound_target_observation_features", True)),
        )
        selector_out = self._selector_runner(self.params, flat_obs)
        logits = np.asarray(
            self.jax.device_get(selector_out["selector_logits"][0]),
            dtype=np.float32,
        )
        values = np.asarray(
            self.jax.device_get(selector_out["selector_values"][0:1]),
            dtype=np.float32,
        )
        return logits, values

    def attention_payload(self, env, *, observer_is_offense: bool):
        if str(self.spec.model_type) != "attention":
            return None
        masked_out = self._team_outputs(env, bool(observer_is_offense))
        weights_device = masked_out.get("attention_weights")
        if weights_device is None:
            return None
        weights = np.asarray(self.jax.device_get(weights_device[0]), dtype=np.float32)
        if weights.ndim != 3 or weights.shape[0] <= 0 or weights.shape[1] <= 0:
            return None

        base_env = self._resolve_base_env(env)
        static = self._build_static_for_role(env, observer_is_offense)
        state = self._state_from_snapshot(snapshot_state_from_env(base_env))
        intent_context = build_policy_intent_context_batch(static, state, self.jnp)
        intent_index = np.asarray(self.jax.device_get(intent_context["intent_index"]), dtype=np.int32)
        intent_gate = np.asarray(self.jax.device_get(intent_context["intent_gate"]), dtype=np.float32)
        offense_ids = set(int(pid) for pid in getattr(base_env, "offense_ids", []) or [])
        defense_ids = set(int(pid) for pid in getattr(base_env, "defense_ids", []) or [])
        labels: list[str] = []
        for pid in range(int(self.spec.token_player_count)):
            if pid in offense_ids:
                labels.append(f"O{pid}")
            elif pid in defense_ids:
                labels.append(f"D{pid}")
            else:
                labels.append(f"P{pid}")

        cls_labels: list[str] = []
        cls_count = int(self.spec.attention_num_cls_tokens)
        if cls_count >= 1:
            cls_labels.append("CLS_OFF")
        if cls_count >= 2:
            cls_labels.append("CLS_DEF")
        for idx in range(2, cls_count):
            cls_labels.append(f"CLS_{idx + 1}")
        labels.extend(cls_labels)

        token_count = int(weights.shape[-1])
        if len(labels) != token_count:
            labels = [f"T{idx}" for idx in range(token_count)]
            cls_labels = []

        return {
            "weights_avg": weights.mean(axis=0).tolist(),
            "weights_heads": weights.tolist(),
            "labels": labels,
            "heads": int(weights.shape[0]),
            "runtime_intent_index": int(intent_index[0]) if intent_index.size else 0,
            "runtime_intent_active": bool(intent_gate[0] > 0.5) if intent_gate.size else False,
            "runtime_intent_visible": bool(intent_gate[0] > 0.5) if intent_gate.size else False,
            "runtime_intent_gate": bool(intent_gate[0] > 0.5) if intent_gate.size else False,
            "observer_role": "offense" if bool(observer_is_offense) else "defense",
            "cls_labels": cls_labels,
        }


def load_inference_model(path: str | Path) -> JAXInferenceModel:
    if not is_checkpoint_path(path):
        raise FileNotFoundError(f"Not a JAX checkpoint directory: {path}")
    return JAXInferenceModel(path)
