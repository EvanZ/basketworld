from __future__ import annotations

from typing import Any, Optional

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
import warnings
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.save_util import load_from_zip_file, recursive_getattr
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

from basketworld.utils.intent_discovery import RunningMeanStd
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

        self._selector_return_stats = RunningMeanStd()
        self._selector_episode_start_records_by_env: dict[int, dict[str, Any]] = {}
        self._selector_completed_samples: list[dict[str, Any]] = []
        self._selector_episode_returns: Optional[np.ndarray] = None
        self._selector_current_obs_selected_mask: Optional[np.ndarray] = None
        self._selector_last_metrics: dict[str, float] = {}
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self) -> list[str]:
        excluded = super()._excluded_save_params()
        excluded.extend(
            [
                "_selector_episode_start_records_by_env",
                "_selector_completed_samples",
                "_selector_episode_returns",
                "_selector_current_obs_selected_mask",
                "_selector_last_metrics",
            ]
        )
        return excluded

    def _setup_model(self) -> None:
        super()._setup_model()
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
        self._selector_episode_returns = np.zeros(env_count, dtype=np.float64)
        self._selector_current_obs_selected_mask = np.zeros(env_count, dtype=bool)
        self._selector_last_metrics = {}

    def _selector_ensure_runtime_state(self, n_envs: int) -> None:
        if (
            self._selector_episode_returns is None
            or self._selector_current_obs_selected_mask is None
            or len(self._selector_episode_returns) != int(n_envs)
            or len(self._selector_current_obs_selected_mask) != int(n_envs)
        ):
            self._selector_reset_runtime_state(n_envs)

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
        visible = self._selector_extract_scalar_from_obs(
            obs_payload, "intent_visible", env_idx, default=1.0
        )
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
        patch_intent_in_observation(
            obs_payload,
            int(intent_index),
            self.intent_selector_num_intents,
            active=1.0,
            visible=visible,
            age_norm=0.0,
            batch_index=env_idx,
        )

    def _selector_maybe_select_for_episode_start(
        self, obs_payload: Any, env_idx: int
    ) -> bool:
        if (
            not self.intent_selector_enabled
            or not self._policy_has_intent_selector()
            or not isinstance(obs_payload, dict)
        ):
            self._selector_episode_start_records_by_env.pop(int(env_idx), None)
            return False
        if self._selector_has_forced_override(env_idx):
            self._selector_episode_start_records_by_env.pop(int(env_idx), None)
            return False
        if (
            self._selector_extract_scalar_from_obs(
                obs_payload, "role_flag", env_idx, default=0.0
            )
            <= 0.0
        ):
            self._selector_episode_start_records_by_env.pop(int(env_idx), None)
            return False
        if (
            self._selector_extract_scalar_from_obs(
                obs_payload, "intent_active", env_idx, default=0.0
            )
            <= 0.5
        ):
            self._selector_episode_start_records_by_env.pop(int(env_idx), None)
            return False

        alpha = self._selector_alpha_current()
        if alpha <= 0.0 or np.random.random() >= alpha:
            self._selector_episode_start_records_by_env.pop(int(env_idx), None)
            return False

        n_envs = self._selector_obs_batch_size(obs_payload)
        if n_envs <= 0:
            self._selector_episode_start_records_by_env.pop(int(env_idx), None)
            return False
        try:
            start_obs = extract_single_env_observation(
                obs_payload, env_idx=env_idx, expected_batch_size=n_envs
            )
        except Exception:
            self._selector_episode_start_records_by_env.pop(int(env_idx), None)
            return False

        selector_obs = self._selector_neutralize_observation(start_obs)
        try:
            with torch.no_grad():
                logits, values = self.policy.get_intent_selector_outputs(selector_obs)
                dist = torch.distributions.Categorical(logits=logits)
                chosen = dist.sample().reshape(-1)
                chosen_z = int(chosen[0].item())
                old_log_prob = float(dist.log_prob(chosen).reshape(-1)[0].item())
                old_value = float(values.reshape(-1)[0].item())
        except Exception:
            self._selector_episode_start_records_by_env.pop(int(env_idx), None)
            return False

        self._selector_apply_selected_intent(obs_payload, env_idx, chosen_z)
        self._selector_episode_start_records_by_env[int(env_idx)] = {
            "selector_obs": selector_obs,
            "chosen_z": int(chosen_z),
            "alpha": float(alpha),
            "old_log_prob": float(old_log_prob),
            "old_value": float(old_value),
        }
        return True

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
        assert self._selector_episode_returns is not None
        assert self._selector_current_obs_selected_mask is not None

        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        episode_starts = np.asarray(self._last_episode_starts, dtype=bool).reshape(-1)
        for env_idx, is_start in enumerate(episode_starts):
            if not bool(is_start):
                continue
            self._selector_episode_returns[env_idx] = 0.0
            if not bool(self._selector_current_obs_selected_mask[env_idx]):
                selected = self._selector_maybe_select_for_episode_start(
                    self._last_obs, env_idx
                )
                self._selector_current_obs_selected_mask[env_idx] = bool(selected)

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with torch.no_grad():
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
            self._selector_episode_returns += np.asarray(rewards, dtype=np.float64)

            next_selected_mask = np.zeros(env.num_envs, dtype=bool)
            if isinstance(new_obs, dict):
                for env_idx, done in enumerate(np.asarray(dones, dtype=bool).reshape(-1)):
                    if not bool(done):
                        continue
                    start_record = self._selector_episode_start_records_by_env.pop(
                        int(env_idx), None
                    )
                    if start_record is not None:
                        self._selector_completed_samples.append(
                            {
                                **start_record,
                                "episode_return": float(
                                    self._selector_episode_returns[env_idx]
                                ),
                            }
                        )
                    self._selector_episode_returns[env_idx] = 0.0
                    selected = self._selector_maybe_select_for_episode_start(
                        new_obs, env_idx
                    )
                    next_selected_mask[env_idx] = bool(selected)

            callback.update_locals(locals())
            if not callback.on_step():
                return False

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
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    def train(self) -> None:
        super().train()
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
            [sample["episode_return"] for sample in samples], dtype=np.float32
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

        self._selector_last_metrics = metrics
        global_step = int(getattr(self, "num_timesteps", 0))
        for key, value in metrics.items():
            try:
                mlflow.log_metric(key, float(value), step=global_step)
            except Exception:
                pass
