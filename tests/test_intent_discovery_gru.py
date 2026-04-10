import numpy as np
import torch

from basketworld.utils.intent_discovery import (
    CompletedIntentEpisode,
    IntentDiscriminator,
    IntentTransition,
    TrajectoryEncoderGRU,
    compute_episode_embeddings,
    build_padded_episode_batch,
)


def _make_episode(
    intent_index: int,
    step_features: list[list[float]],
    active_flags: list[bool] | None = None,
) -> CompletedIntentEpisode:
    transitions = []
    for step_idx, feat in enumerate(step_features):
        active = True if active_flags is None else bool(active_flags[step_idx])
        transitions.append(
            IntentTransition(
                feature=np.asarray(feat, dtype=np.float32),
                buffer_step_idx=step_idx,
                env_idx=0,
                role_flag=1.0,
                intent_active=active,
                intent_index=int(intent_index),
            )
        )
    return CompletedIntentEpisode(intent_index=int(intent_index), transitions=transitions)


def test_build_padded_episode_batch_shapes_and_lengths():
    ep_a = _make_episode(1, [[1, 2, 3], [4, 5, 6]])
    ep_b = _make_episode(2, [[7, 8, 9]])

    x_steps, lengths, labels = build_padded_episode_batch(
        [ep_a, ep_b],
        max_obs_dim=2,
        max_action_dim=1,
    )

    assert x_steps.shape == (2, 2, 3)
    assert lengths.tolist() == [2, 1]
    assert labels.tolist() == [1, 2]
    assert np.allclose(x_steps[0, 0], [1.0, 2.0, 3.0])
    assert np.allclose(x_steps[0, 1], [4.0, 5.0, 6.0])
    assert np.allclose(x_steps[1, 0], [7.0, 8.0, 9.0])
    assert np.allclose(x_steps[1, 1], [0.0, 0.0, 0.0])


def test_episode_batch_truncates_to_active_prefix():
    ep = _make_episode(
        1,
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        active_flags=[True, True, False],
    )

    x_steps, lengths, labels = build_padded_episode_batch(
        [ep],
        max_obs_dim=2,
        max_action_dim=1,
    )
    x_mean, y_mean = compute_episode_embeddings(
        [ep],
        max_obs_dim=2,
        max_action_dim=1,
    )

    assert lengths.tolist() == [2]
    assert labels.tolist() == [1]
    assert np.allclose(x_steps[0, 0], [1.0, 2.0, 3.0])
    assert np.allclose(x_steps[0, 1], [4.0, 5.0, 6.0])
    assert np.allclose(x_mean[0], [2.5, 3.5, 4.5])
    assert y_mean.tolist() == [1]


def test_gru_encoder_ignores_right_padding_when_lengths_match():
    torch.manual_seed(0)
    encoder = TrajectoryEncoderGRU(input_dim=4, hidden_dim=8)

    x_base = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    x_padded = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [99.0, 99.0, 99.0, 99.0]]],
        dtype=torch.float32,
    )
    lengths = torch.tensor([2], dtype=torch.long)

    h_base = encoder(x_base, lengths)
    h_padded = encoder(x_padded, lengths)

    assert h_base.shape == (1, 8)
    assert torch.allclose(h_base, h_padded, atol=1e-6)


def test_gru_discriminator_outputs_logits_and_uses_order():
    torch.manual_seed(0)
    disc = IntentDiscriminator(
        input_dim=4,
        hidden_dim=8,
        num_intents=3,
        encoder_type="gru",
        step_dim=6,
        dropout=0.0,
    )

    seq_a = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0]]],
        dtype=torch.float32,
    )
    seq_b = torch.tensor(
        [[[0.0, 0.0, 3.0, 0.0], [0.0, 2.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    lengths = torch.tensor([3], dtype=torch.long)

    logits_a = disc(seq_a, lengths)
    logits_b = disc(seq_b, lengths)

    assert logits_a.shape == (1, 3)
    assert logits_b.shape == (1, 3)
    assert not torch.allclose(logits_a, logits_b, atol=1e-6)


def test_intent_discriminator_encode_returns_episode_embedding():
    torch.manual_seed(0)
    disc = IntentDiscriminator(
        input_dim=4,
        hidden_dim=8,
        num_intents=3,
        encoder_type="gru",
        step_dim=6,
        dropout=0.0,
    )
    seq = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0]]],
        dtype=torch.float32,
    )
    lengths = torch.tensor([3], dtype=torch.long)

    emb = disc.encode(seq, lengths)
    logits = disc(seq, lengths)

    assert emb.shape == (1, 8)
    assert logits.shape == (1, 3)
