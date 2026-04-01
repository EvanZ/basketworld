from __future__ import annotations

from typing import Any, Optional

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
import warnings
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.save_util import load_from_zip_file, recursive_getattr
from stable_baselines3.common.utils import explained_variance, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

from basketworld.utils.intent_discovery import RunningMeanStd
from basketworld.utils.intent_rollout_buffer import (
    IntentConditionedDictRolloutBuffer,
)
from basketworld.utils.intent_policy_sensitivity import (
    clone_observation_dict,
    extract_single_env_observation,
    patch_intent_in_observation,
)


class IntegratedIntentSelectorPPO(PPO):
    """PPO with an integrated possession-start intent selector mu(z|s)."""

    def __init__(
        self,
        *args,
        intent_selector_enabled: bool = False,
        num_intents: int = 8,
        intent_selector_alpha_start: float = 0.0,
        intent_selector_alpha_end: float = 1.0,
        intent_selector_alpha_warmup_steps: int = 0,
        intent_selector_alpha_ramp_steps: int = 1,
        intent_selector_entropy_coef: float = 0.01,
        intent_selector_usage_reg_coef: float = 0.01,
        intent_selector_value_coef: float = 0.5,
        intent_selector_multiselect_enabled: bool = False,
        intent_selector_min_play_steps: int = 3,
        intent_commitment_steps: int = 4,
        **kwargs,
    ) -> None:
        self.intent_selector_enabled = bool(intent_selector_enabled)
        self.intent_selector_num_intents = max(1, int(num_intents))
        self.intent_selector_alpha_start = float(intent_selector_alpha_start)
        self.intent_selector_alpha_end = float(intent_selector_alpha_end)
        self.intent_selector_alpha_warmup_steps = max(
            0, int(intent_selector_alpha_warmup_steps)
        )
        self.intent_selector_alpha_ramp_steps = max(
            0, int(intent_selector_alpha_ramp_steps)
        )
        self.intent_selector_entropy_coef = float(max(0.0, intent_selector_entropy_coef))
        self.intent_selector_usage_reg_coef = float(
            max(0.0, intent_selector_usage_reg_coef)
        )
        self.intent_selector_value_coef = float(max(0.0, intent_selector_value_coef))
        self.intent_selector_multiselect_enabled = bool(
            intent_selector_multiselect_enabled
        )
        self.intent_selector_min_play_steps = max(1, int(intent_selector_min_play_steps))
        self.intent_commitment_steps = max(1, int(intent_commitment_steps))

        self._selector_return_stats = RunningMeanStd()
        self._selector_episode_start_records_by_env: dict[int, dict[str, Any]] = {}
        self._selector_completed_samples: list[dict[str, Any]] = []
        self._selector_segment_returns: Optional[np.ndarray] = None
        self._selector_segment_discounts: Optional[np.ndarray] = None
        self._selector_segment_lengths: Optional[np.ndarray] = None
        self._selector_episode_segment_counts: Optional[np.ndarray] = None
        self._selector_episode_trackable_mask: Optional[np.ndarray] = None
        self._selector_current_obs_selected_mask: Optional[np.ndarray] = None
        self._policy_runtime_intent_indices: Optional[np.ndarray] = None
        self._policy_runtime_intent_gate: Optional[np.ndarray] = None
        self._selector_rollout_completed_episode_segment_counts: list[int] = []
        self._selector_rollout_boundary_reason_counts: dict[str, int] = {}
        self._selector_last_metrics: dict[str, float] = {}
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self) -> list[str]:
        excluded = super()._excluded_save_params()
        excluded.extend(
            [
                "_selector_episode_start_records_by_env",
                "_selector_completed_samples",
                "_selector_segment_returns",
                "_selector_segment_discounts",
                "_selector_segment_lengths",
                "_selector_episode_segment_counts",
                "_selector_episode_trackable_mask",
                "_selector_current_obs_selected_mask",
                "_policy_runtime_intent_indices",
                "_policy_runtime_intent_gate",
                "_selector_rollout_completed_episode_segment_counts",
                "_selector_rollout_boundary_reason_counts",
                "_selector_last_metrics",
            ]
        )
        return excluded

    def _setup_model(self) -> None:
        super()._setup_model()
        if isinstance(self.rollout_buffer, DictRolloutBuffer) and not isinstance(
            self.rollout_buffer, IntentConditionedDictRolloutBuffer
        ):
            self.rollout_buffer = IntentConditionedDictRolloutBuffer(
                self.n_steps,
                self.observation_space,
                self.action_space,
                device=self.device,
                gae_lambda=self.gae_lambda,
                gamma=self.gamma,
                n_envs=self.n_envs,
            )
        self._selector_reset_runtime_state(self.n_envs)
        if self.intent_selector_enabled and not self._policy_has_intent_selector():
            raise RuntimeError(
                "Integrated intent selector requires a policy with intent_selector_enabled=True."
            )

    def _policy_has_intent_selector(self) -> bool:
        policy = getattr(self, "policy", None)
        return bool(
            policy is not None
            and hasattr(policy, "has_intent_selector")
            and policy.has_intent_selector()
        )

    def _policy_has_intent_selector_value_head(self) -> bool:
        policy = getattr(self, "policy", None)
        return bool(
            policy is not None
            and hasattr(policy, "has_intent_selector_value_head")
            and policy.has_intent_selector_value_head()
        )

    @staticmethod
    def _selector_value_head_state_mismatch(exc: RuntimeError) -> bool:
        message = str(exc)
        return (
            "intent_selector_value_head" in message
            and (
                "Missing key(s) in state_dict" in message
                or "Missing key(s):" in message
                or "Unexpected key(s) in state_dict" in message
                or "Unexpected key(s):" in message
            )
        )

    def set_parameters(
        self,
        load_path_or_dict: Any,
        exact_match: bool = True,
        device: torch.device | str = "auto",
    ) -> None:
        params = {}
        if isinstance(load_path_or_dict, dict):
            params = load_path_or_dict
        else:
            _, params, _ = load_from_zip_file(load_path_or_dict, device=device)

        objects_needing_update = set(self._get_torch_save_params()[0])
        updated_objects = set()
        used_selector_backcompat = False

        for name in params:
            try:
                attr = recursive_getattr(self, name)
            except Exception as exc:
                raise ValueError(f"Key {name} is an invalid object name.") from exc

            if isinstance(attr, torch.optim.Optimizer):
                try:
                    attr.load_state_dict(params[name])  # type: ignore[arg-type]
                except ValueError as exc:
                    if (
                        name == "policy.optimizer"
                        and self._policy_has_intent_selector_value_head()
                    ):
                        used_selector_backcompat = True
                        warnings.warn(
                            "Skipping legacy optimizer state while loading integrated selector checkpoint "
                            "without selector critic weights.",
                            stacklevel=2,
                        )
                    else:
                        raise
            else:
                try:
                    attr.load_state_dict(params[name], strict=exact_match)
                except RuntimeError as exc:
                    if (
                        name == "policy"
                        and exact_match
                        and self._selector_value_head_state_mismatch(exc)
                    ):
                        attr.load_state_dict(params[name], strict=False)
                        used_selector_backcompat = True
                        warnings.warn(
                            "Loaded legacy integrated selector checkpoint without selector critic weights. "
                            "The selector value head was freshly initialized.",
                            stacklevel=2,
                        )
                    else:
                        raise
            updated_objects.add(name)

        if exact_match and updated_objects != objects_needing_update:
            if not used_selector_backcompat:
                raise ValueError(
                    "Names of parameters do not match agents' parameters: "
                    f"expected {objects_needing_update}, got {updated_objects}"
                )

    def _selector_reset_runtime_state(self, n_envs: Optional[int] = None) -> None:
        env_count = int(self.n_envs if n_envs is None else n_envs)
        self._selector_episode_start_records_by_env = {}
        self._selector_completed_samples = []
        self._selector_segment_returns = np.zeros(env_count, dtype=np.float64)
        self._selector_segment_discounts = np.ones(env_count, dtype=np.float64)
        self._selector_segment_lengths = np.zeros(env_count, dtype=np.int64)
        self._selector_episode_segment_counts = np.zeros(env_count, dtype=np.int64)
        self._selector_episode_trackable_mask = np.zeros(env_count, dtype=bool)
        self._selector_current_obs_selected_mask = np.zeros(env_count, dtype=bool)
        self._policy_runtime_intent_indices = np.zeros(env_count, dtype=np.int64)
        self._policy_runtime_intent_gate = np.zeros(env_count, dtype=np.float32)
        self._selector_rollout_completed_episode_segment_counts = []
        self._selector_rollout_boundary_reason_counts = {}
        self._selector_last_metrics = {}
        self._selector_push_policy_runtime_intent_override()

    def _selector_ensure_runtime_state(self, n_envs: int) -> None:
        if (
            self._selector_segment_returns is None
            or self._selector_segment_discounts is None
            or self._selector_segment_lengths is None
            or self._selector_episode_segment_counts is None
            or self._selector_episode_trackable_mask is None
            or self._selector_current_obs_selected_mask is None
            or len(self._selector_segment_returns) != int(n_envs)
            or len(self._selector_segment_discounts) != int(n_envs)
            or len(self._selector_segment_lengths) != int(n_envs)
            or len(self._selector_episode_segment_counts) != int(n_envs)
            or len(self._selector_episode_trackable_mask) != int(n_envs)
            or len(self._selector_current_obs_selected_mask) != int(n_envs)
        ):
            self._selector_reset_runtime_state(n_envs)

    def _selector_reset_rollout_observability(self) -> None:
        self._selector_rollout_completed_episode_segment_counts = []
        self._selector_rollout_boundary_reason_counts = {}

    def _selector_push_policy_runtime_intent_override(self) -> None:
        policy = getattr(self, "policy", None)
        if policy is None:
            return
        if not hasattr(policy, "set_runtime_intent_override"):
            return
        if (
            self._policy_runtime_intent_indices is None
            or self._policy_runtime_intent_gate is None
        ):
            clearer = getattr(policy, "clear_runtime_intent_override", None)
            if callable(clearer):
                clearer()
            return
        try:
            policy.set_runtime_intent_override(
                self._policy_runtime_intent_indices,
                self._policy_runtime_intent_gate,
            )
        except Exception:
            pass

    def _selector_sync_policy_runtime_from_obs(self, obs_payload: Any) -> None:
        batch_size = self._selector_obs_batch_size(obs_payload)
        if batch_size <= 0:
            self._policy_runtime_intent_indices = None
            self._policy_runtime_intent_gate = None
            self._selector_push_policy_runtime_intent_override()
            return
        self._selector_ensure_runtime_state(batch_size)
        assert self._policy_runtime_intent_indices is not None
        assert self._policy_runtime_intent_gate is not None
        for env_idx in range(int(batch_size)):
            role_flag = self._selector_extract_scalar_from_obs(
                obs_payload, "role_flag", env_idx, default=0.0
            )
            fields = self._selector_get_role_intent_fields(
                env_idx, observer_is_offense=bool(role_flag > 0.0)
            )
            intent_index = int(max(0, float(fields.get("intent_index", 0.0))))
            intent_active = float(fields.get("intent_active", 0.0))
            intent_visible = float(fields.get("intent_visible", 1.0))
            self._policy_runtime_intent_indices[env_idx] = int(intent_index)
            self._policy_runtime_intent_gate[env_idx] = float(
                1.0 if (intent_active > 0.5 and intent_visible > 0.5) else 0.0
            )
        self._selector_push_policy_runtime_intent_override()

    def _selector_get_role_intent_fields(
        self, env_idx: int, *, observer_is_offense: bool
    ) -> dict[str, float]:
        vec_env = self.get_env()
        if vec_env is None:
            return {}
        try:
            values = vec_env.env_method(
                "get_intent_observation_fields",
                bool(observer_is_offense),
                indices=[int(env_idx)],
            )
        except Exception:
            return {}
        if not values:
            return {}
        fields = values[0]
        if not isinstance(fields, dict):
            return {}
        out: dict[str, float] = {}
        for key in ("intent_index", "intent_active", "intent_visible", "intent_age_norm"):
            try:
                out[key] = float(np.asarray(fields.get(key, 0.0)).reshape(-1)[0])
            except Exception:
                out[key] = 0.0
        return out

    @staticmethod
    def _selector_extract_scalar_from_obs(
        obs_payload: Any, key: str, env_idx: int, default: float = 0.0
    ) -> float:
        try:
            if not isinstance(obs_payload, dict) or key not in obs_payload:
                return float(default)
            arr = np.asarray(obs_payload[key], dtype=np.float32)
            if arr.ndim == 0:
                return float(arr)
            if arr.shape[0] <= int(env_idx):
                return float(default)
            return float(np.asarray(arr[int(env_idx)]).reshape(-1)[0])
        except Exception:
            return float(default)

    @staticmethod
    def _selector_obs_batch_size(obs_payload: Any) -> int:
        try:
            if not isinstance(obs_payload, dict):
                return 0
            for value in obs_payload.values():
                arr = np.asarray(value)
                if arr.ndim >= 1:
                    return int(arr.shape[0])
        except Exception:
            pass
        return 0

    @staticmethod
    def _selector_stack_single_observations(single_obs_list: list[dict[str, Any]]) -> dict[str, np.ndarray]:
        if not single_obs_list:
            raise ValueError("single_obs_list cannot be empty")
        batched: dict[str, np.ndarray] = {}
        keys = list(single_obs_list[0].keys())
        for key in keys:
            batched[key] = np.stack(
                [np.asarray(single_obs[key]) for single_obs in single_obs_list], axis=0
            )
        return batched

    def _selector_alpha_current(self) -> float:
        t = int(getattr(self, "num_timesteps", 0))
        if t < self.intent_selector_alpha_warmup_steps:
            return float(self.intent_selector_alpha_start)
        if self.intent_selector_alpha_ramp_steps <= 0:
            return float(self.intent_selector_alpha_end)
        progress = min(
            1.0,
            max(
                0.0,
                (t - self.intent_selector_alpha_warmup_steps)
                / float(self.intent_selector_alpha_ramp_steps),
            ),
        )
        return float(
            self.intent_selector_alpha_start
            + progress
            * (self.intent_selector_alpha_end - self.intent_selector_alpha_start)
        )

    def _selector_neutralize_observation(self, single_obs: dict[str, Any]) -> dict[str, Any]:
        selector_obs = clone_observation_dict(single_obs)
        patch_intent_in_observation(
            selector_obs,
            0,
            self.intent_selector_num_intents,
            active=0.0,
            visible=0.0,
            age_norm=0.0,
        )
        return selector_obs

    def _selector_has_forced_override(self, env_idx: int) -> bool:
        vec_env = self.get_env()
        if vec_env is None:
            return False
        try:
            values = vec_env.env_method(
                "get_offense_intent_override", indices=[int(env_idx)]
            )
            if not values:
                return False
            return values[0] is not None
        except Exception:
            return False

    def _selector_apply_selected_intent(
        self, obs_payload: dict[str, Any], env_idx: int, intent_index: int
    ) -> None:
        vec_env = self.get_env()
        if vec_env is not None:
            try:
                vec_env.env_method(
                    "set_offense_intent_state",
                    int(intent_index),
                    indices=[int(env_idx)],
                    intent_active=True,
                    intent_age=0,
                )
            except Exception:
                pass
        visible = float(
            self._selector_get_role_intent_fields(
                env_idx, observer_is_offense=True
            ).get("intent_visible", 1.0)
        )
        if (
            self._policy_runtime_intent_indices is not None
            and self._policy_runtime_intent_gate is not None
            and 0 <= int(env_idx) < len(self._policy_runtime_intent_indices)
        ):
            self._policy_runtime_intent_indices[int(env_idx)] = int(intent_index)
            self._policy_runtime_intent_gate[int(env_idx)] = float(1.0 if visible > 0.5 else 0.0)
            self._selector_push_policy_runtime_intent_override()
        # Keep observation metadata aligned for rollout storage and label extraction.
        patch_intent_in_observation(
            obs_payload,
            int(intent_index),
            self.intent_selector_num_intents,
            active=1.0,
            visible=visible,
            age_norm=0.0,
            batch_index=env_idx,
        )

    def _selector_reset_segment_tracking_for_env(self, env_idx: int) -> None:
        assert self._selector_segment_returns is not None
        assert self._selector_segment_discounts is not None
        assert self._selector_segment_lengths is not None
        self._selector_segment_returns[int(env_idx)] = 0.0
        self._selector_segment_discounts[int(env_idx)] = 1.0
        self._selector_segment_lengths[int(env_idx)] = 0

    @staticmethod
    def _selector_segment_bucket_label(segment_index_within_episode: int) -> str:
        idx = max(0, int(segment_index_within_episode))
        if idx >= 3:
            return "3_plus"
        return str(idx)

    def _selector_episode_is_trackable(self, obs_payload: Any, env_idx: int) -> bool:
        if not isinstance(obs_payload, dict):
            return False
        role_flag = self._selector_extract_scalar_from_obs(
            obs_payload, "role_flag", env_idx, default=0.0
        )
        if role_flag <= 0.0:
            return False
        fields = self._selector_get_role_intent_fields(
            env_idx, observer_is_offense=True
        )
        return (
            float(fields.get("intent_active", 0.0)) > 0.5
        )

    def _selector_mark_episode_start(self, obs_payload: Any, env_idx: int) -> None:
        assert self._selector_episode_segment_counts is not None
        assert self._selector_episode_trackable_mask is not None
        trackable = self._selector_episode_is_trackable(obs_payload, env_idx)
        self._selector_episode_trackable_mask[int(env_idx)] = bool(trackable)
        self._selector_episode_segment_counts[int(env_idx)] = 1 if trackable else 0

    def _selector_record_segment_boundary(self, env_idx: int, reason: str) -> None:
        assert self._selector_episode_segment_counts is not None
        assert self._selector_episode_trackable_mask is not None
        if not bool(self._selector_episode_trackable_mask[int(env_idx)]):
            return
        self._selector_episode_segment_counts[int(env_idx)] += 1
        self._selector_rollout_boundary_reason_counts[reason] = (
            self._selector_rollout_boundary_reason_counts.get(reason, 0) + 1
        )

    def _selector_finalize_episode_tracking(self, env_idx: int) -> None:
        assert self._selector_episode_segment_counts is not None
        assert self._selector_episode_trackable_mask is not None
        if bool(self._selector_episode_trackable_mask[int(env_idx)]):
            self._selector_rollout_completed_episode_segment_counts.append(
                int(self._selector_episode_segment_counts[int(env_idx)])
            )
        self._selector_episode_segment_counts[int(env_idx)] = 0
        self._selector_episode_trackable_mask[int(env_idx)] = False

    def _selector_accumulate_step_reward(
        self, rewards: np.ndarray, dones: np.ndarray
    ) -> None:
        assert self._selector_segment_returns is not None
        assert self._selector_segment_discounts is not None
        assert self._selector_segment_lengths is not None
        reward_arr = np.asarray(rewards, dtype=np.float64).reshape(-1)
        for env_idx in range(min(len(reward_arr), len(self._selector_segment_returns))):
            self._selector_segment_returns[env_idx] += float(reward_arr[env_idx])
            self._selector_segment_lengths[env_idx] += 1

    @staticmethod
    def _selector_completed_pass_boundary(info: Any) -> bool:
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

    def _selector_segment_boundary_reason(self, info: Any, env_idx: int) -> Optional[str]:
        assert self._selector_segment_lengths is not None
        if self._selector_segment_lengths[int(env_idx)] >= int(self.intent_commitment_steps):
            return "commitment_timeout"
        if (
            self._selector_segment_lengths[int(env_idx)] >= self.intent_selector_min_play_steps
            and self._selector_completed_pass_boundary(info)
        ):
            return "completed_pass"
        return None

    def _selector_prepare_segment_start(
        self,
        obs_payload: Any,
        env_idx: int,
        *,
        allow_uniform_fallback: bool,
    ) -> dict[str, Any]:
        prepared: dict[str, Any] = {
            "apply_intent": None,
            "used_selector": False,
            "record": None,
        }
        if (
            not self.intent_selector_enabled
            or not self._policy_has_intent_selector()
            or not isinstance(obs_payload, dict)
        ):
            return prepared
        if self._selector_has_forced_override(env_idx):
            return prepared
        if (
            self._selector_extract_scalar_from_obs(
                obs_payload, "role_flag", env_idx, default=0.0
            )
            <= 0.0
        ):
            return prepared
        fields = self._selector_get_role_intent_fields(
            env_idx, observer_is_offense=True
        )
        if float(fields.get("intent_active", 0.0)) <= 0.5:
            return prepared

        n_envs = self._selector_obs_batch_size(obs_payload)
        if n_envs <= 0:
            return prepared
        try:
            start_obs = extract_single_env_observation(
                obs_payload, env_idx=env_idx, expected_batch_size=n_envs
            )
        except Exception:
            return prepared

        alpha = self._selector_alpha_current()
        selector_obs = self._selector_neutralize_observation(start_obs)
        if alpha > 0.0 and np.random.random() < alpha:
            try:
                with torch.no_grad():
                    logits, values = self.policy.get_intent_selector_outputs(selector_obs)
                    dist = torch.distributions.Categorical(logits=logits)
                    chosen = dist.sample().reshape(-1)
                    chosen_z = int(chosen[0].item())
                    old_log_prob = float(dist.log_prob(chosen).reshape(-1)[0].item())
                    old_value = float(values.reshape(-1)[0].item())
            except Exception:
                return prepared
            prepared["apply_intent"] = int(chosen_z)
            prepared["used_selector"] = True
            prepared["record"] = {
                "selector_obs": selector_obs,
                "chosen_z": int(chosen_z),
                "segment_index_within_episode": int(
                    self._selector_episode_segment_counts[int(env_idx)] - 1
                )
                if self._selector_episode_segment_counts is not None
                else 0,
                "alpha": float(alpha),
                "old_log_prob": float(old_log_prob),
                "old_value": float(old_value),
            }
            return prepared

        if allow_uniform_fallback:
            prepared["apply_intent"] = int(
                np.random.randint(0, self.intent_selector_num_intents)
            )
        return prepared

    def _selector_commit_prepared_segment_start(
        self, obs_payload: Any, env_idx: int, prepared: dict[str, Any]
    ) -> bool:
        self._selector_episode_start_records_by_env.pop(int(env_idx), None)
        self._selector_reset_segment_tracking_for_env(env_idx)
        chosen_z = prepared.get("apply_intent")
        if chosen_z is None:
            return False
        if isinstance(obs_payload, dict):
            self._selector_apply_selected_intent(obs_payload, env_idx, int(chosen_z))
        record = prepared.get("record")
        if isinstance(record, dict) and bool(prepared.get("used_selector", False)):
            self._selector_episode_start_records_by_env[int(env_idx)] = record
            return True
        return False

    def _selector_finalize_current_segment(
        self, env_idx: int, *, bootstrap_value: Optional[float] = None
    ) -> None:
        assert self._selector_segment_returns is not None
        assert self._selector_segment_discounts is not None
        assert self._selector_segment_lengths is not None
        record = self._selector_episode_start_records_by_env.pop(int(env_idx), None)
        segment_return = float(self._selector_segment_returns[int(env_idx)])
        segment_discount = float(self._selector_segment_discounts[int(env_idx)])
        segment_length = int(self._selector_segment_lengths[int(env_idx)])
        if record is not None:
            target_return = segment_return
            if bootstrap_value is not None:
                target_return += segment_discount * float(bootstrap_value)
            self._selector_completed_samples.append(
                {
                    **record,
                    "segment_return": float(segment_return),
                    "target_return": float(target_return),
                    "segment_length": float(segment_length),
                    "bootstrap_value": (
                        float(bootstrap_value) if bootstrap_value is not None else 0.0
                    ),
                    "used_bootstrap": bool(bootstrap_value is not None),
                }
            )
        self._selector_reset_segment_tracking_for_env(env_idx)

    def _selector_log_rollout_segment_metrics(self) -> None:
        completed_counts = np.asarray(
            self._selector_rollout_completed_episode_segment_counts, dtype=np.float64
        )
        episode_count = int(completed_counts.size)
        boundary_total = int(sum(self._selector_rollout_boundary_reason_counts.values()))
        metrics: dict[str, float] = {
            "intent/segment_episode_count": float(episode_count),
            "intent/multisegment_episode_rate": float(
                np.mean(completed_counts >= 2.0) if episode_count > 0 else 0.0
            ),
            "intent/segments_per_episode_mean": float(
                np.mean(completed_counts) if episode_count > 0 else 0.0
            ),
            "intent/segment_boundary_count": float(boundary_total),
            "intent/segment_boundary_reason_completed_pass_rate": float(
                self._selector_rollout_boundary_reason_counts.get("completed_pass", 0)
                / boundary_total
            )
            if boundary_total > 0
            else 0.0,
            "intent/segment_boundary_reason_commitment_timeout_rate": float(
                self._selector_rollout_boundary_reason_counts.get(
                    "commitment_timeout", 0
                )
                / boundary_total
            )
            if boundary_total > 0
            else 0.0,
        }
        self._selector_last_metrics = {
            **self._selector_last_metrics,
            **metrics,
        }
        global_step = int(getattr(self, "num_timesteps", 0))
        for key, value in metrics.items():
            try:
                mlflow.log_metric(key, float(value), step=global_step)
            except Exception:
                pass

    def _selector_maybe_select_for_episode_start(
        self, obs_payload: Any, env_idx: int
    ) -> bool:
        prepared = self._selector_prepare_segment_start(
            obs_payload,
            env_idx,
            allow_uniform_fallback=False,
        )
        return bool(
            self._selector_commit_prepared_segment_start(obs_payload, env_idx, prepared)
        )

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        assert self._last_obs is not None, "No previous observation was provided"

        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        self._selector_ensure_runtime_state(env.num_envs)
        self._selector_reset_rollout_observability()
        assert self._selector_segment_returns is not None
        assert self._selector_segment_discounts is not None
        assert self._selector_segment_lengths is not None
        assert self._selector_current_obs_selected_mask is not None

        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        episode_starts = np.asarray(self._last_episode_starts, dtype=bool).reshape(-1)
        for env_idx, is_start in enumerate(episode_starts):
            if not bool(is_start):
                continue
            self._selector_mark_episode_start(self._last_obs, env_idx)
            if not bool(self._selector_current_obs_selected_mask[env_idx]):
                selected = self._selector_maybe_select_for_episode_start(
                    self._last_obs, env_idx
                )
                self._selector_current_obs_selected_mask[env_idx] = bool(selected)
        if isinstance(self._last_obs, dict):
            self._selector_sync_policy_runtime_from_obs(self._last_obs)

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with torch.no_grad():
                if isinstance(self._last_obs, dict):
                    self._selector_sync_policy_runtime_from_obs(self._last_obs)
                else:
                    self._selector_sync_policy_runtime_from_obs(None)
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            clipped_actions = actions
            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    clipped_actions = np.clip(
                        actions, self.action_space.low, self.action_space.high
                    )

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            self.num_timesteps += env.num_envs
            self._selector_accumulate_step_reward(rewards, dones)

            next_selected_mask = np.zeros(env.num_envs, dtype=bool)
            pending_segment_starts: dict[int, dict[str, Any]] = {}
            if isinstance(new_obs, dict):
                done_mask = np.asarray(dones, dtype=bool).reshape(-1)
                for env_idx, done in enumerate(done_mask):
                    if not bool(done):
                        boundary_reason = None
                        if (
                            self.intent_selector_enabled
                            and self.intent_selector_multiselect_enabled
                        ):
                            boundary_reason = self._selector_segment_boundary_reason(
                                infos[env_idx] if env_idx < len(infos) else None,
                                env_idx,
                            )
                        if boundary_reason is not None:
                            self._selector_record_segment_boundary(
                                env_idx, boundary_reason
                            )
                            prepared = self._selector_prepare_segment_start(
                                new_obs,
                                env_idx,
                                allow_uniform_fallback=True,
                            )
                            bootstrap_value = None
                            if bool(prepared.get("used_selector", False)):
                                record = prepared.get("record")
                                if isinstance(record, dict):
                                    bootstrap_value = float(
                                        record.get("old_value", 0.0)
                                    )
                            self._selector_finalize_current_segment(
                                env_idx, bootstrap_value=bootstrap_value
                            )
                            if env_idx < len(infos) and isinstance(infos[env_idx], dict):
                                infos[env_idx]["intent_segment_boundary"] = 1.0
                                infos[env_idx]["intent_segment_boundary_reason"] = boundary_reason
                            pending_segment_starts[int(env_idx)] = prepared
                        continue
                    self._selector_finalize_current_segment(
                        env_idx, bootstrap_value=None
                    )
                    self._selector_finalize_episode_tracking(env_idx)
                    self._selector_mark_episode_start(new_obs, env_idx)
                    selected = self._selector_maybe_select_for_episode_start(
                        new_obs, env_idx
                    )
                    next_selected_mask[env_idx] = bool(selected)

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            if isinstance(new_obs, dict) and pending_segment_starts:
                for env_idx, prepared in pending_segment_starts.items():
                    used_selector = self._selector_commit_prepared_segment_start(
                        new_obs, env_idx, prepared
                    )
                    next_selected_mask[int(env_idx)] = bool(used_selector)
                self._selector_sync_policy_runtime_from_obs(new_obs)

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(
                        infos[idx]["terminal_observation"]
                    )[0]
                    with torch.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            if isinstance(rollout_buffer, IntentConditionedDictRolloutBuffer):
                rollout_buffer.add(
                    self._last_obs,
                    actions,
                    rewards,
                    self._last_episode_starts,
                    values,
                    log_probs,
                    intent_indices=self._policy_runtime_intent_indices,
                    intent_gate=self._policy_runtime_intent_gate,
                )
            else:
                rollout_buffer.add(
                    self._last_obs,
                    actions,
                    rewards,
                    self._last_episode_starts,
                    values,
                    log_probs,
                )
            self._last_obs = new_obs
            self._last_episode_starts = dones
            self._selector_current_obs_selected_mask = next_selected_mask

        with torch.no_grad():
            if isinstance(new_obs, dict):
                self._selector_sync_policy_runtime_from_obs(new_obs)
            else:
                self._selector_sync_policy_runtime_from_obs(None)
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    def train(self) -> None:
        self.policy.set_training_mode(True)
        clear_override = getattr(self.policy, "clear_runtime_intent_override", None)
        set_override = getattr(self.policy, "set_runtime_intent_override", None)
        if callable(clear_override):
            clear_override()

        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses: list[float] = []
        pg_losses: list[float] = []
        value_losses: list[float] = []
        clip_fractions: list[float] = []

        continue_training = True
        approx_kl_divs: list[float] = []
        loss = None
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                if callable(set_override):
                    intent_indices = getattr(rollout_data, "intent_indices", None)
                    intent_gate = getattr(rollout_data, "intent_gate", None)
                    if intent_indices is not None:
                        set_override(intent_indices, intent_gate)
                    elif callable(clear_override):
                        clear_override()

                try:
                    values, log_prob, entropy = self.policy.evaluate_actions(
                        rollout_data.observations, actions
                    )
                finally:
                    if callable(clear_override):
                        clear_override()

                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(
                    ratio, 1 - clip_range, 1 + clip_range
                )
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean(
                    (torch.abs(ratio - 1) > clip_range).float()
                ).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean(
                        (torch.exp(log_ratio) - 1) - log_ratio
                    ).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}"
                        )
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", 0.0 if loss is None else loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

        self._selector_log_rollout_segment_metrics()
        self._train_intent_selector()

    def _train_intent_selector(self) -> None:
        if (
            not self.intent_selector_enabled
            or not self._policy_has_intent_selector()
            or not self._selector_completed_samples
        ):
            return

        samples = list(self._selector_completed_samples)
        self._selector_completed_samples = []

        selector_obs_batch = self._selector_stack_single_observations(
            [sample["selector_obs"] for sample in samples]
        )
        returns_np = np.asarray(
            [sample["target_return"] for sample in samples], dtype=np.float32
        )
        segment_return_np = np.asarray(
            [sample.get("segment_return", sample["target_return"]) for sample in samples],
            dtype=np.float32,
        )
        segment_length_np = np.asarray(
            [sample.get("segment_length", 0.0) for sample in samples], dtype=np.float32
        )
        bootstrap_value_np = np.asarray(
            [sample.get("bootstrap_value", 0.0) for sample in samples], dtype=np.float32
        )
        segment_index_np = np.asarray(
            [sample.get("segment_index_within_episode", 0) for sample in samples],
            dtype=np.int64,
        )
        chosen_z_np = np.asarray(
            [sample["chosen_z"] for sample in samples], dtype=np.int64
        )
        old_log_prob_np = np.asarray(
            [sample.get("old_log_prob", 0.0) for sample in samples], dtype=np.float32
        )
        old_value_np = np.asarray(
            [sample.get("old_value", 0.0) for sample in samples], dtype=np.float32
        )
        raw_advantages_np = returns_np - old_value_np
        adv_std = float(np.std(raw_advantages_np))
        if adv_std <= 1e-8:
            advantages_np = raw_advantages_np - float(np.mean(raw_advantages_np))
        else:
            advantages_np = (
                raw_advantages_np - float(np.mean(raw_advantages_np))
            ) / adv_std
        self._selector_return_stats.update(returns_np)

        self.policy.set_training_mode(True)
        obs_tensor, _ = self.policy.obs_to_tensor(selector_obs_batch)
        logits, value_pred = self.policy.get_intent_selector_outputs(obs_tensor)
        dist = torch.distributions.Categorical(logits=logits)
        chosen_z = torch.as_tensor(chosen_z_np, dtype=torch.long, device=logits.device)
        advantage = torch.as_tensor(
            advantages_np, dtype=torch.float32, device=logits.device
        )
        returns = torch.as_tensor(
            returns_np, dtype=torch.float32, device=logits.device
        )
        old_log_prob = torch.as_tensor(
            old_log_prob_np, dtype=torch.float32, device=logits.device
        )
        old_value = torch.as_tensor(
            old_value_np, dtype=torch.float32, device=logits.device
        )
        log_prob = dist.log_prob(chosen_z)
        entropy = dist.entropy().mean()
        prob_tensor = torch.softmax(logits, dim=-1)
        mean_probs = prob_tensor.mean(dim=0)
        usage_kl = torch.sum(
            mean_probs
            * torch.log(
                mean_probs.clamp_min(1e-8) * float(self.intent_selector_num_intents)
            )
        )
        clip_range = self.clip_range(self._current_progress_remaining)
        ratio = torch.exp(log_prob - old_log_prob)
        policy_loss_1 = advantage * ratio
        policy_loss_2 = advantage * torch.clamp(
            ratio, 1.0 - clip_range, 1.0 + clip_range
        )
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
        if self.clip_range_vf is None:
            value_loss = F.mse_loss(value_pred, returns)
        else:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)
            value_pred_clipped = old_value + torch.clamp(
                value_pred - old_value, -clip_range_vf, clip_range_vf
            )
            value_loss = 0.5 * torch.max(
                (value_pred - returns) ** 2,
                (value_pred_clipped - returns) ** 2,
            ).mean()
        total_loss = (
            policy_loss
            + float(self.intent_selector_value_coef) * value_loss
            - float(self.intent_selector_entropy_coef) * entropy
            + float(self.intent_selector_usage_reg_coef) * usage_kl
        )

        self.policy.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()

        usage_counts = np.bincount(
            chosen_z_np, minlength=self.intent_selector_num_intents
        ).astype(np.float64)
        usage_probs = usage_counts / max(1.0, float(np.sum(usage_counts)))
        nonzero = usage_probs > 0.0
        usage_entropy = float(
            -np.sum(usage_probs[nonzero] * np.log(usage_probs[nonzero]))
        )
        mean_probs_np = mean_probs.detach().cpu().numpy().astype(np.float64, copy=False)
        top1_np = (
            torch.argmax(logits, dim=-1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.int64, copy=False)
        )
        top1_counts = np.bincount(
            top1_np, minlength=self.intent_selector_num_intents
        ).astype(np.float64)
        top1_probs = top1_counts / max(1.0, float(np.sum(top1_counts)))
        max_prob_np = (
            torch.max(prob_tensor, dim=-1)
            .values.detach()
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )
        top2_vals = torch.topk(
            prob_tensor, k=min(2, self.intent_selector_num_intents), dim=-1
        ).values
        if top2_vals.shape[-1] >= 2:
            margin_np = (
                top2_vals[:, 0] - top2_vals[:, 1]
            ).detach().cpu().numpy().astype(np.float64, copy=False)
        else:
            margin_np = max_prob_np.copy()

        metrics: dict[str, float] = {
            **self._selector_last_metrics,
            "intent/selector_alpha_current": float(self._selector_alpha_current()),
            "intent/selector_loss": float(total_loss.detach().cpu().item()),
            "intent/selector_policy_loss": float(policy_loss.detach().cpu().item()),
            "intent/selector_value_loss": float(value_loss.detach().cpu().item()),
            "intent/selector_entropy": float(entropy.detach().cpu().item()),
            "intent/selector_usage_entropy": float(usage_entropy),
            "intent/selector_usage_kl_uniform": float(
                usage_kl.detach().cpu().item()
            ),
            "intent/selector_return_mean": float(np.mean(returns_np)),
            "intent/selector_segment_return_mean": float(np.mean(segment_return_np)),
            "intent/selector_segment_steps_mean": float(np.mean(segment_length_np)),
            "intent/selector_bootstrap_value_mean": float(
                np.mean(bootstrap_value_np)
            ),
            "intent/selector_samples": float(len(samples)),
            "intent/selector_advantage_raw_mean": float(np.mean(raw_advantages_np)),
            "intent/selector_value_mean": float(
                value_pred.detach().cpu().mean().item()
            ),
            "intent/selector_old_value_mean": float(np.mean(old_value_np)),
            "intent/selector_clip_fraction": float(
                torch.mean((torch.abs(ratio - 1.0) > clip_range).float())
                .detach()
                .cpu()
                .item()
            ),
            "intent/selector_approx_kl": float(
                torch.mean(old_log_prob - log_prob).detach().cpu().item()
            ),
            "intent/selector_confidence_mean": float(np.mean(max_prob_np))
            if max_prob_np.size > 0
            else 0.0,
            "intent/selector_margin_mean": float(np.mean(margin_np))
            if margin_np.size > 0
            else 0.0,
        }
        for z in range(self.intent_selector_num_intents):
            metrics[f"intent/selector_usage_by_intent/{z}"] = float(usage_probs[z])
            metrics[f"intent/selector_prob_mean_by_intent/{z}"] = float(
                mean_probs_np[z]
            )
            metrics[f"intent/selector_top1_by_intent/{z}"] = float(top1_probs[z])
            selected_mask = chosen_z_np == z
            if np.any(selected_mask):
                metrics[f"intent/selector_return_by_intent/{z}"] = float(
                    np.mean(returns_np[selected_mask])
                )

        segment_bucket_order = ["0", "1", "2", "3_plus"]
        for bucket in segment_bucket_order:
            if bucket == "3_plus":
                bucket_mask = segment_index_np >= 3
            else:
                bucket_mask = segment_index_np == int(bucket)
            bucket_count = int(np.sum(bucket_mask))
            metrics[f"intent/selector_segment_index_count/{bucket}"] = float(
                bucket_count
            )
            if bucket_count <= 0:
                metrics[f"intent/selector_usage_entropy_by_segment/{bucket}"] = 0.0
                for z in range(self.intent_selector_num_intents):
                    metrics[f"intent/selector_usage_by_segment/{bucket}/{z}"] = 0.0
                continue
            bucket_counts = np.bincount(
                chosen_z_np[bucket_mask], minlength=self.intent_selector_num_intents
            ).astype(np.float64)
            bucket_probs = bucket_counts / max(1.0, float(np.sum(bucket_counts)))
            nonzero_bucket = bucket_probs > 0.0
            metrics[f"intent/selector_usage_entropy_by_segment/{bucket}"] = float(
                -np.sum(bucket_probs[nonzero_bucket] * np.log(bucket_probs[nonzero_bucket]))
            )
            for z in range(self.intent_selector_num_intents):
                metrics[f"intent/selector_usage_by_segment/{bucket}/{z}"] = float(
                    bucket_probs[z]
                )

        self._selector_last_metrics = metrics
        global_step = int(getattr(self, "num_timesteps", 0))
        for key, value in metrics.items():
            try:
                mlflow.log_metric(key, float(value), step=global_step)
            except Exception:
                pass
