from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn


@dataclass
class IntentTransition:
    """One transition tracked for intent diversity shaping."""

    feature: np.ndarray
    buffer_step_idx: int
    env_idx: int
    role_flag: float
    intent_active: bool
    intent_index: int


@dataclass
class CompletedIntentEpisode:
    """Completed episode assembled from transitions."""

    intent_index: int
    transitions: List[IntentTransition]

    @property
    def length(self) -> int:
        return len(self.transitions)

    @property
    def buffer_indices(self) -> List[Tuple[int, int]]:
        return [(tr.buffer_step_idx, tr.env_idx) for tr in self.transitions]

    @property
    def role_is_offense(self) -> bool:
        if not self.transitions:
            return False
        return bool(self.transitions[0].role_flag > 0.0)

    @property
    def intent_active(self) -> bool:
        if not self.transitions:
            return False
        return bool(self.transitions[0].intent_active)


class RunningMeanStd:
    """Numerically stable running mean/std estimator."""

    def __init__(self, epsilon: float = 1e-6) -> None:
        self.count = float(epsilon)
        self.mean = 0.0
        self.m2 = 1.0

    def update(self, values: np.ndarray) -> None:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            return
        batch_mean = float(np.mean(arr))
        batch_var = float(np.var(arr))
        batch_count = float(arr.size)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / max(1e-12, total_count))

        m2_a = self.m2 * self.count
        m2_b = batch_var * batch_count
        m2_total = m2_a + m2_b + (delta * delta) * (self.count * batch_count / max(1e-12, total_count))

        self.mean = new_mean
        self.count = total_count
        self.m2 = m2_total / max(1e-12, total_count)

    @property
    def std(self) -> float:
        return float(np.sqrt(max(1e-12, self.m2)))


class IntentEpisodeBuffer:
    """Tracks partial episodes across rollout boundaries per environment."""

    def __init__(self) -> None:
        self._ongoing: Dict[int, List[IntentTransition]] = {}
        self._completed: List[CompletedIntentEpisode] = []

    def append(self, env_idx: int, transition: IntentTransition) -> None:
        self._ongoing.setdefault(int(env_idx), []).append(transition)

    def close_episode(self, env_idx: int) -> None:
        key = int(env_idx)
        transitions = self._ongoing.pop(key, [])
        if not transitions:
            return
        intent_index = int(transitions[0].intent_index)
        self._completed.append(
            CompletedIntentEpisode(intent_index=intent_index, transitions=transitions)
        )

    def pop_completed(
        self, filter_fn: Optional[Callable[[CompletedIntentEpisode], bool]] = None
    ) -> List[CompletedIntentEpisode]:
        episodes = self._completed
        self._completed = []
        if filter_fn is None:
            return episodes
        return [ep for ep in episodes if bool(filter_fn(ep))]


class IntentDiscriminator(nn.Module):
    """Simple MLP discriminator over episode embeddings."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_intents: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(max(0.0, dropout))),
            nn.Linear(int(hidden_dim), int(num_intents)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def flatten_observation_for_env(obs_payload, env_idx: int) -> np.ndarray:
    """Extract one flattened observation vector for a specific vec-env index."""
    if obs_payload is None:
        return np.zeros(1, dtype=np.float32)

    if isinstance(obs_payload, dict):
        parts: List[np.ndarray] = []
        if "obs" in obs_payload:
            try:
                arr = np.asarray(obs_payload["obs"], dtype=np.float32)
                parts.append(arr[int(env_idx)].reshape(-1))
            except Exception:
                pass
        else:
            for key, value in sorted(obs_payload.items()):
                if key in {"action_mask"}:
                    continue
                try:
                    arr = np.asarray(value, dtype=np.float32)
                    if arr.ndim >= 1 and arr.shape[0] > int(env_idx):
                        parts.append(arr[int(env_idx)].reshape(-1))
                except Exception:
                    continue
        if parts:
            return np.concatenate(parts).astype(np.float32, copy=False)
        return np.zeros(1, dtype=np.float32)

    try:
        arr = np.asarray(obs_payload, dtype=np.float32)
        if arr.ndim >= 1 and arr.shape[0] > int(env_idx):
            return arr[int(env_idx)].reshape(-1)
    except Exception:
        pass
    return np.zeros(1, dtype=np.float32)


def extract_action_features_for_env(actions_payload, env_idx: int) -> np.ndarray:
    """Extract flattened action features for one vec-env index."""
    if actions_payload is None:
        return np.zeros(1, dtype=np.float32)
    try:
        arr = np.asarray(actions_payload, dtype=np.float32)
        if arr.ndim == 0:
            return np.array([float(arr)], dtype=np.float32)
        if arr.ndim >= 1 and arr.shape[0] > int(env_idx):
            return arr[int(env_idx)].reshape(-1).astype(np.float32, copy=False)
    except Exception:
        pass
    return np.zeros(1, dtype=np.float32)


def compute_episode_embeddings(
    episodes: List[CompletedIntentEpisode],
    max_obs_dim: int = 256,
    max_action_dim: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert completed episodes to fixed-size embedding and labels."""
    if not episodes:
        return np.zeros((0, 1), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    emb_rows: List[np.ndarray] = []
    labels: List[int] = []
    for ep in episodes:
        step_vecs: List[np.ndarray] = []
        for tr in ep.transitions:
            feat = np.asarray(tr.feature, dtype=np.float32).reshape(-1)
            step_vecs.append(feat)
        if not step_vecs:
            continue
        stacked = np.vstack(step_vecs)
        mean_vec = np.mean(stacked, axis=0)
        if mean_vec.shape[0] < (max_obs_dim + max_action_dim):
            padded = np.zeros((max_obs_dim + max_action_dim,), dtype=np.float32)
            padded[: mean_vec.shape[0]] = mean_vec
            mean_vec = padded
        else:
            mean_vec = mean_vec[: (max_obs_dim + max_action_dim)]
        emb_rows.append(mean_vec.astype(np.float32, copy=False))
        labels.append(int(ep.intent_index))

    if not emb_rows:
        return np.zeros((0, max_obs_dim + max_action_dim), dtype=np.float32), np.zeros(
            (0,), dtype=np.int64
        )
    return np.vstack(emb_rows), np.asarray(labels, dtype=np.int64)
