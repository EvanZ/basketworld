from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


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

    @property
    def active_prefix_transitions(self) -> List[IntentTransition]:
        prefix: List[IntentTransition] = []
        for tr in self.transitions:
            if not bool(tr.intent_active):
                break
            prefix.append(tr)
        return prefix

    @property
    def active_prefix_length(self) -> int:
        return len(self.active_prefix_transitions)

    @property
    def active_buffer_indices(self) -> List[Tuple[int, int]]:
        return [(tr.buffer_step_idx, tr.env_idx) for tr in self.active_prefix_transitions]


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


class StepEncoder(nn.Module):
    """Per-step feature projection before temporal aggregation."""

    def __init__(self, input_dim: int, step_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(step_dim)),
            nn.ReLU(),
            nn.Dropout(float(max(0.0, dropout))),
        )

    def forward(self, x_steps: torch.Tensor) -> torch.Tensor:
        if x_steps.ndim != 3:
            raise ValueError(f"StepEncoder expects [B, T, D], got {tuple(x_steps.shape)}")
        b, t, d = x_steps.shape
        flat = x_steps.reshape(b * t, d)
        step_emb = self.net(flat)
        return step_emb.reshape(b, t, -1)


class TrajectoryEncoderGRU(nn.Module):
    """GRU over variable-length step sequences."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        gru_dropout = float(max(0.0, dropout)) if int(num_layers) > 1 else 0.0
        self.gru = nn.GRU(
            input_size=int(input_dim),
            hidden_size=int(hidden_dim),
            num_layers=int(max(1, num_layers)),
            batch_first=True,
            dropout=gru_dropout,
        )

    def forward(self, x_steps: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if x_steps.ndim != 3:
            raise ValueError(
                f"TrajectoryEncoderGRU expects [B, T, D], got {tuple(x_steps.shape)}"
            )
        if lengths.ndim != 1:
            raise ValueError(f"TrajectoryEncoderGRU expects [B] lengths, got {tuple(lengths.shape)}")
        packed = pack_padded_sequence(
            x_steps,
            lengths.detach().to(device="cpu", dtype=torch.long),
            batch_first=True,
            enforce_sorted=False,
        )
        _, h_n = self.gru(packed)
        return h_n[-1]


class IntentDiscriminator(nn.Module):
    """Intent discriminator over either pooled or sequential episode features."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_intents: int,
        dropout: float = 0.1,
        encoder_type: str = "mlp_mean",
        step_dim: int = 64,
    ) -> None:
        super().__init__()
        self.encoder_type = str(encoder_type).strip().lower()
        if self.encoder_type not in {"mlp_mean", "gru"}:
            raise ValueError(f"Unsupported intent discriminator encoder_type={encoder_type!r}")

        if self.encoder_type == "mlp_mean":
            self.net = nn.Sequential(
                nn.Linear(int(input_dim), int(hidden_dim)),
                nn.ReLU(),
                nn.Dropout(float(max(0.0, dropout))),
                nn.Linear(int(hidden_dim), int(num_intents)),
            )
            self.step_encoder = None
            self.traj_encoder = None
            self.head = None
        else:
            self.net = None
            self.step_encoder = StepEncoder(
                input_dim=int(input_dim),
                step_dim=int(step_dim),
                dropout=float(max(0.0, dropout)),
            )
            self.traj_encoder = TrajectoryEncoderGRU(
                input_dim=int(step_dim),
                hidden_dim=int(hidden_dim),
                num_layers=1,
                dropout=0.0,
            )
            self.head = nn.Sequential(
                nn.Dropout(float(max(0.0, dropout))),
                nn.Linear(int(hidden_dim), int(num_intents)),
            )

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        episode_emb = self.encode(x, lengths)
        if self.encoder_type == "mlp_mean":
            assert self.net is not None
            return self.net[-1](episode_emb)
        assert self.head is not None
        return self.head(episode_emb)

    def encode(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return the episode embedding prior to the final classifier layer."""
        if self.encoder_type == "mlp_mean":
            if x.ndim == 1:
                x = x.unsqueeze(0)
            elif x.ndim == 3:
                x = torch.mean(x, dim=1)
            if x.ndim != 2:
                raise ValueError(f"MLP intent discriminator expects [B, D], got {tuple(x.shape)}")
            assert self.net is not None
            hidden = x
            for layer in list(self.net.children())[:-1]:
                hidden = layer(hidden)
            return hidden

        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(f"GRU intent discriminator expects [B, T, D], got {tuple(x.shape)}")
        if lengths is None:
            lengths = torch.full(
                (x.shape[0],),
                int(x.shape[1]),
                dtype=torch.long,
                device=x.device,
            )
        else:
            lengths = lengths.to(device=x.device, dtype=torch.long).reshape(-1)
        assert self.step_encoder is not None
        assert self.traj_encoder is not None
        step_emb = self.step_encoder(x)
        return self.traj_encoder(step_emb, lengths)


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


def _fixed_dim_step_feature(
    feat: np.ndarray,
    max_obs_dim: int,
    max_action_dim: int,
) -> np.ndarray:
    total_dim = int(max_obs_dim + max_action_dim)
    fixed = np.zeros((total_dim,), dtype=np.float32)
    flat = np.asarray(feat, dtype=np.float32).reshape(-1)
    take = min(total_dim, flat.shape[0])
    if take > 0:
        fixed[:take] = flat[:take]
    return fixed


def _episode_feature_matrix(
    episode: CompletedIntentEpisode,
    max_obs_dim: int = 256,
    max_action_dim: int = 16,
) -> Optional[np.ndarray]:
    step_vecs: List[np.ndarray] = []
    for tr in episode.active_prefix_transitions:
        step_vecs.append(
            _fixed_dim_step_feature(
                tr.feature,
                max_obs_dim=max_obs_dim,
                max_action_dim=max_action_dim,
            )
        )
    if not step_vecs:
        return None
    return np.vstack(step_vecs).astype(np.float32, copy=False)


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
        stacked = _episode_feature_matrix(
            ep,
            max_obs_dim=max_obs_dim,
            max_action_dim=max_action_dim,
        )
        if stacked is None or stacked.shape[0] == 0:
            continue
        mean_vec = np.mean(stacked, axis=0)
        emb_rows.append(mean_vec.astype(np.float32, copy=False))
        labels.append(int(ep.intent_index))

    if not emb_rows:
        return np.zeros((0, max_obs_dim + max_action_dim), dtype=np.float32), np.zeros(
            (0,), dtype=np.int64
        )
    return np.vstack(emb_rows), np.asarray(labels, dtype=np.int64)


def build_padded_episode_batch(
    episodes: List[CompletedIntentEpisode],
    max_obs_dim: int = 256,
    max_action_dim: int = 16,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert completed episodes to padded [B, T, D] batch with lengths and labels."""
    total_dim = int(max_obs_dim + max_action_dim)
    if not episodes:
        return (
            np.zeros((0, 1, total_dim), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )

    sequences: List[np.ndarray] = []
    lengths: List[int] = []
    labels: List[int] = []
    max_len = 0
    for ep in episodes:
        seq = _episode_feature_matrix(
            ep,
            max_obs_dim=max_obs_dim,
            max_action_dim=max_action_dim,
        )
        if seq is None or seq.shape[0] == 0:
            continue
        sequences.append(seq)
        lengths.append(int(seq.shape[0]))
        labels.append(int(ep.intent_index))
        max_len = max(max_len, int(seq.shape[0]))

    if not sequences:
        return (
            np.zeros((0, 1, total_dim), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )

    padded = np.zeros((len(sequences), max_len, total_dim), dtype=np.float32)
    for idx, seq in enumerate(sequences):
        padded[idx, : seq.shape[0], :] = seq
    return (
        padded,
        np.asarray(lengths, dtype=np.int64),
        np.asarray(labels, dtype=np.int64),
    )
