from __future__ import annotations

import numpy as np
import pytest

from benchmarks.sbx_phase_a import PhaseAPolicySpec
from benchmarks.sbx_phase_a_torch import _apply_action_mask_torch


def test_torch_masked_forward_never_selects_illegal_action():
    torch = pytest.importorskip("torch")

    spec = PhaseAPolicySpec(
        flat_obs_dim=2,
        training_player_count=2,
        action_dim_per_player=4,
        total_action_dim=8,
        hidden_dims=(),
    )
    flat_logits = torch.tensor(
        [[0.0, 10.0, -1.0, -2.0, 7.0, 0.0, -3.0, -4.0]],
        dtype=torch.float32,
    )
    action_mask = torch.tensor(
        [[[1, 0, 1, 0], [0, 1, 0, 0]]],
        dtype=torch.int8,
    )

    out = _apply_action_mask_torch(torch, flat_logits, action_mask, spec)

    deterministic = out["deterministic_actions"].detach().cpu().numpy().astype(np.int32)
    probs = out["probs"].detach().cpu().numpy().astype(np.float32)

    np.testing.assert_array_equal(deterministic, np.array([[0, 1]], dtype=np.int32))
    assert probs[0, 0, 1] == pytest.approx(0.0)
    assert probs[0, 0, 3] == pytest.approx(0.0)
    assert probs[0, 1, 0] == pytest.approx(0.0)
    assert probs[0, 1, 2] == pytest.approx(0.0)
    assert probs[0, 1, 3] == pytest.approx(0.0)


def test_torch_masked_forward_falls_back_to_noop_when_mask_is_empty():
    torch = pytest.importorskip("torch")

    spec = PhaseAPolicySpec(
        flat_obs_dim=1,
        training_player_count=1,
        action_dim_per_player=3,
        total_action_dim=3,
        hidden_dims=(),
    )
    flat_logits = torch.tensor([[3.0, 2.0, 1.0]], dtype=torch.float32)
    action_mask = torch.zeros((1, 1, 3), dtype=torch.int8)

    out = _apply_action_mask_torch(torch, flat_logits, action_mask, spec)

    deterministic = out["deterministic_actions"].detach().cpu().numpy().astype(np.int32)
    probs = out["probs"].detach().cpu().numpy().astype(np.float32)

    np.testing.assert_array_equal(deterministic, np.array([[0]], dtype=np.int32))
    assert probs[0, 0, 0] == pytest.approx(1.0)
    assert probs[0, 0, 1] == pytest.approx(0.0)
    assert probs[0, 0, 2] == pytest.approx(0.0)
