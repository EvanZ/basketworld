from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

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
class IntentStepExample:
    """One single-step discriminator example."""

    obs: Dict[str, np.ndarray]
    buffer_step_idx: int
    env_idx: int
    role_flag: float
    intent_active: bool
    intent_index: int
    training_team: str = ""
    boundary_reason: str = ""
    shot_end: float = 0.0
    shot_quality: float = 0.0
    shot_quality_mask: float = 0.0


@dataclass
class CompletedIntentEpisode:
    """Completed episode assembled from transitions."""

    intent_index: int
    transitions: List[IntentTransition]
    terminal_info: Optional[Dict[str, Any]] = None

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

    @property
    def terminal_action_results(self) -> Dict[str, Any]:
        info = self.terminal_info or {}
        action_results = info.get("action_results", {})
        if isinstance(action_results, dict):
            return action_results
        return {}

    @property
    def shot_end_label(self) -> float:
        action_results = self.terminal_action_results
        shots = action_results.get("shots", {})
        if isinstance(shots, dict) and len(shots) > 0:
            return 1.0
        info = self.terminal_info or {}
        if (
            float(info.get("shot_dunk", 0.0)) > 0.0
            or float(info.get("shot_2pt", 0.0)) > 0.0
            or float(info.get("shot_3pt", 0.0)) > 0.0
        ):
            return 1.0
        return 0.0

    @property
    def shot_quality_target(self) -> float:
        action_results = self.terminal_action_results
        shots = action_results.get("shots", {})
        if isinstance(shots, dict):
            for shot_result in shots.values():
                if isinstance(shot_result, dict) and "expected_points" in shot_result:
                    try:
                        return float(shot_result.get("expected_points", 0.0))
                    except Exception:
                        break
        info = self.terminal_info or {}
        return float(info.get("expected_points", 0.0) or 0.0)


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

    def close_episode(
        self,
        env_idx: int,
        *,
        terminal_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        key = int(env_idx)
        transitions = self._ongoing.pop(key, [])
        if not transitions:
            return
        intent_index = int(transitions[0].intent_index)
        self._completed.append(
            CompletedIntentEpisode(
                intent_index=intent_index,
                transitions=transitions,
                terminal_info=dict(terminal_info or {}),
            )
        )

    def current_intent_index(self, env_idx: int) -> Optional[int]:
        transitions = self._ongoing.get(int(env_idx), [])
        if not transitions:
            return None
        return int(transitions[0].intent_index)

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


class SetStepEncoder(nn.Module):
    """Single-state set-attention encoder over player tokens and globals."""

    def __init__(
        self,
        token_dim: int,
        global_dim: int,
        hidden_dim: int,
        *,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_cls_tokens: int = 1,
        include_role_flag: bool = True,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.global_dim = int(global_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(max(1, num_heads))
        self.num_cls_tokens = int(max(0, num_cls_tokens))
        self.include_role_flag = bool(include_role_flag)
        extra_global_dim = 1 if self.include_role_flag else 0
        self.token_mlp = nn.Sequential(
            nn.Linear(self.token_dim + self.global_dim + extra_global_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(max(0.0, dropout))),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=float(max(0.0, dropout)),
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(self.hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(max(0.0, dropout))),
        )
        self.ff_norm = nn.LayerNorm(self.hidden_dim)
        self.cls_tokens = (
            nn.Parameter(torch.zeros(self.num_cls_tokens, self.hidden_dim))
            if self.num_cls_tokens > 0
            else None
        )

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        if not isinstance(obs, dict):
            raise ValueError("SetStepEncoder expects a dict observation batch.")
        if "players" not in obs or "globals" not in obs:
            raise ValueError("SetStepEncoder requires 'players' and 'globals'.")
        players = obs["players"]
        globals_vec = obs["globals"]
        if players.ndim != 3:
            raise ValueError(f"Expected players [B, P, T], got {tuple(players.shape)}")
        if globals_vec.ndim != 2:
            raise ValueError(f"Expected globals [B, G], got {tuple(globals_vec.shape)}")
        pieces = [
            players,
            globals_vec.unsqueeze(1).expand(-1, players.shape[1], -1),
        ]
        if self.include_role_flag and "role_flag" in obs:
            role_flag = obs["role_flag"]
            if role_flag.ndim == 1:
                role_flag = role_flag.unsqueeze(-1)
            elif role_flag.ndim > 2:
                role_flag = role_flag.reshape(role_flag.shape[0], -1)
            pieces.append(role_flag.unsqueeze(1).expand(-1, players.shape[1], -1))
        tokens = torch.cat(pieces, dim=-1)
        emb = self.token_mlp(tokens)
        if self.cls_tokens is not None:
            cls = self.cls_tokens.unsqueeze(0).expand(emb.shape[0], -1, -1)
            emb = torch.cat([emb, cls], dim=1)
        attn_out, _ = self.attn(emb, emb, emb, need_weights=False)
        emb = self.attn_norm(emb + attn_out)
        ff_out = self.ff(emb)
        emb = self.ff_norm(emb + ff_out)
        if self.num_cls_tokens > 0:
            cls_slice = emb[:, -self.num_cls_tokens :, :]
            return cls_slice.mean(dim=1)
        return emb.mean(dim=1)


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
        set_token_dim: int = 0,
        set_global_dim: int = 0,
        set_heads: int = 4,
        set_cls_tokens: int = 1,
        enable_shot_end_head: bool = False,
        enable_shot_quality_head: bool = False,
    ) -> None:
        super().__init__()
        self.encoder_type = str(encoder_type).strip().lower()
        if self.encoder_type not in {"mlp_mean", "gru", "set_step"}:
            raise ValueError(f"Unsupported intent discriminator encoder_type={encoder_type!r}")
        self.enable_shot_end_head = bool(enable_shot_end_head)
        self.enable_shot_quality_head = bool(enable_shot_quality_head)

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
        elif self.encoder_type == "gru":
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
        else:
            self.net = None
            self.step_encoder = SetStepEncoder(
                token_dim=int(set_token_dim),
                global_dim=int(set_global_dim),
                hidden_dim=int(hidden_dim),
                num_heads=int(max(1, set_heads)),
                dropout=float(max(0.0, dropout)),
                num_cls_tokens=int(max(0, set_cls_tokens)),
                include_role_flag=True,
            )
            self.traj_encoder = None
            self.head = nn.Sequential(
                nn.Dropout(float(max(0.0, dropout))),
                nn.Linear(int(hidden_dim), int(num_intents)),
            )
        self.shot_end_head = (
            nn.Linear(int(hidden_dim), 1) if self.enable_shot_end_head else None
        )
        self.shot_quality_head = (
            nn.Linear(int(hidden_dim), 1) if self.enable_shot_quality_head else None
        )

    def forward(self, x: Any, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        episode_emb = self.encode(x, lengths)
        if self.encoder_type == "mlp_mean":
            assert self.net is not None
            return self.net[-1](episode_emb)
        assert self.head is not None
        return self.head(episode_emb)

    def forward_with_aux(
        self,
        x: Any,
        lengths: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        episode_emb = self.encode(x, lengths)
        if self.encoder_type == "mlp_mean":
            assert self.net is not None
            logits = self.net[-1](episode_emb)
        else:
            assert self.head is not None
            logits = self.head(episode_emb)
        shot_end = (
            self.shot_end_head(episode_emb).reshape(-1)
            if self.shot_end_head is not None
            else None
        )
        shot_quality = (
            self.shot_quality_head(episode_emb).reshape(-1)
            if self.shot_quality_head is not None
            else None
        )
        return logits, shot_end, shot_quality

    def encode(self, x: Any, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
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

        if self.encoder_type == "set_step":
            if not isinstance(x, dict):
                raise ValueError("set_step intent discriminator expects a dict input.")
            assert self.step_encoder is not None
            return self.step_encoder(x)

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


def _checkpoint_head_enabled(
    config: Dict[str, Any],
    state_dict: Dict[str, Any],
    *,
    enabled_key: str,
    lambda_key: str,
    head_prefix: str,
) -> bool:
    if enabled_key in config:
        return bool(config.get(enabled_key))
    if float(config.get(lambda_key, 0.0) or 0.0) > 0.0:
        return True
    prefix = f"{head_prefix}."
    return any(str(key).startswith(prefix) for key in state_dict.keys())


def load_intent_discriminator_from_checkpoint(
    payload: Dict[str, Any],
    *,
    device: str | torch.device,
) -> IntentDiscriminator:
    """Instantiate a discriminator from checkpoint payload with schema compatibility."""
    if not isinstance(payload, dict) or "state_dict" not in payload or "config" not in payload:
        raise RuntimeError("Unsupported discriminator checkpoint payload")
    config = dict(payload.get("config", {}) or {})
    state_dict = dict(payload.get("state_dict", {}) or {})
    disc = IntentDiscriminator(
        input_dim=int(config["input_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        num_intents=int(config["num_intents"]),
        dropout=float(config.get("dropout", 0.1)),
        encoder_type=str(config.get("encoder_type", "mlp_mean")),
        step_dim=int(config.get("step_dim", 64)),
        enable_shot_end_head=_checkpoint_head_enabled(
            config,
            state_dict,
            enabled_key="enable_shot_end_head",
            lambda_key="disc_lambda_shot",
            head_prefix="shot_end_head",
        ),
        enable_shot_quality_head=_checkpoint_head_enabled(
            config,
            state_dict,
            enabled_key="enable_shot_quality_head",
            lambda_key="disc_lambda_q",
            head_prefix="shot_quality_head",
        ),
        set_token_dim=int(config.get("set_token_dim", 0)),
        set_global_dim=int(config.get("set_global_dim", 0)),
        set_heads=int(config.get("set_heads", 4)),
        set_cls_tokens=int(config.get("set_cls_tokens", 1)),
    ).to(device)
    disc.load_state_dict(state_dict)
    disc.eval()
    return disc


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


def extract_set_observation_for_env(
    obs_payload: Any,
    env_idx: int,
    *,
    include_keys: Tuple[str, ...] = ("players", "globals", "role_flag"),
) -> Optional[Dict[str, np.ndarray]]:
    """Extract one single-env set observation dict for discriminator use."""
    if not isinstance(obs_payload, dict):
        return None
    out: Dict[str, np.ndarray] = {}
    for key in include_keys:
        if key not in obs_payload:
            continue
        try:
            arr = np.asarray(obs_payload[key], dtype=np.float32)
        except Exception:
            continue
        if arr.ndim == 0 or arr.shape[0] <= int(env_idx):
            continue
        out[key] = np.array(arr[int(env_idx)], copy=True, dtype=np.float32)
    if "players" not in out or "globals" not in out:
        return None
    if "role_flag" not in out:
        out["role_flag"] = np.array([1.0], dtype=np.float32)
    return out


def build_step_observation_batch(
    examples: List[IntentStepExample],
) -> tuple[Dict[str, np.ndarray], np.ndarray]:
    """Stack single-step set observations and labels into batched arrays."""
    if not examples:
        return {}, np.zeros((0,), dtype=np.int64)
    keys = ["players", "globals", "role_flag"]
    batch: Dict[str, np.ndarray] = {}
    for key in keys:
        values = [ex.obs[key] for ex in examples if key in ex.obs]
        if len(values) != len(examples):
            continue
        batch[key] = np.stack(values, axis=0).astype(np.float32, copy=False)
    labels = np.asarray([int(ex.intent_index) for ex in examples], dtype=np.int64)
    return batch, labels


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
