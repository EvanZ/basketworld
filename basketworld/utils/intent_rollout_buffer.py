from __future__ import annotations

from typing import Generator, NamedTuple, Optional

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.vec_env import VecNormalize


class IntentConditionedDictRolloutBufferSamples(NamedTuple):
    observations: dict[str, th.Tensor]
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    intent_indices: th.Tensor
    intent_gate: th.Tensor


class IntentConditionedDictRolloutBuffer(DictRolloutBuffer):
    """Dict rollout buffer with an explicit per-transition play-conditioning channel."""

    def reset(self) -> None:
        super().reset()
        self.intent_indices = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.int64
        )
        self.intent_gate = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )

    def add(  # type: ignore[override]
        self,
        obs: dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        *,
        intent_indices: Optional[np.ndarray] = None,
        intent_gate: Optional[np.ndarray] = None,
    ) -> None:
        write_pos = int(self.pos)
        super().add(obs, action, reward, episode_start, value, log_prob)

        if intent_indices is None:
            indices_arr = np.zeros((self.n_envs,), dtype=np.int64)
        else:
            indices_arr = np.asarray(intent_indices, dtype=np.int64).reshape(-1)
            if indices_arr.shape[0] == 1 and self.n_envs > 1:
                indices_arr = np.repeat(indices_arr, self.n_envs)
        if intent_gate is None:
            gate_arr = np.zeros((self.n_envs,), dtype=np.float32)
        else:
            gate_arr = np.asarray(intent_gate, dtype=np.float32).reshape(-1)
            if gate_arr.shape[0] == 1 and self.n_envs > 1:
                gate_arr = np.repeat(gate_arr, self.n_envs)

        if indices_arr.shape[0] != self.n_envs:
            raise ValueError(
                f"intent_indices must have length {self.n_envs}, got {indices_arr.shape[0]}"
            )
        if gate_arr.shape[0] != self.n_envs:
            raise ValueError(
                f"intent_gate must have length {self.n_envs}, got {gate_arr.shape[0]}"
            )

        self.intent_indices[write_pos] = np.array(indices_arr, copy=True)
        self.intent_gate[write_pos] = np.array(gate_arr, copy=True)

    def get(  # type: ignore[override]
        self,
        batch_size: Optional[int] = None,
    ) -> Generator[IntentConditionedDictRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        if not self.generator_ready:
            for key, obs in self.observations.items():
                self.observations[key] = self.swap_and_flatten(obs)

            tensor_names = [
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "intent_indices",
                "intent_gate",
            ]
            for tensor_name in tensor_names:
                self.__dict__[tensor_name] = self.swap_and_flatten(
                    self.__dict__[tensor_name]
                )
            self.generator_ready = True

        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(  # type: ignore[override]
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> IntentConditionedDictRolloutBufferSamples:
        return IntentConditionedDictRolloutBufferSamples(
            observations={
                key: self.to_torch(obs[batch_inds])
                for (key, obs) in self.observations.items()
            },
            actions=self.to_torch(self.actions[batch_inds]),
            old_values=self.to_torch(self.values[batch_inds].flatten()),
            old_log_prob=self.to_torch(self.log_probs[batch_inds].flatten()),
            advantages=self.to_torch(self.advantages[batch_inds].flatten()),
            returns=self.to_torch(self.returns[batch_inds].flatten()),
            intent_indices=self.to_torch(self.intent_indices[batch_inds]).long(),
            intent_gate=self.to_torch(self.intent_gate[batch_inds]).float(),
        )
