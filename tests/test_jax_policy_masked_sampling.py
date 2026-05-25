from __future__ import annotations

import numpy as np
import pytest

from benchmarks.jax_kernel import _masked_sample_actions_from_prob_tensor


def test_masked_sample_actions_zeroes_illegal_mass_and_samples_legal():
    torch = pytest.importorskip("torch")

    probs = torch.tensor(
        [[[0.1, 0.7, 0.2], [0.9, 0.05, 0.05]]],
        dtype=torch.float32,
    )
    action_mask = torch.tensor(
        [[[1, 0, 1], [0, 1, 0]]],
        dtype=torch.float32,
    )

    actions, normalized = _masked_sample_actions_from_prob_tensor(
        torch,
        probs,
        action_mask,
        deterministic=True,
    )

    np.testing.assert_array_equal(
        actions.detach().cpu().numpy().astype(np.int64),
        np.array([[2, 1]], dtype=np.int64),
    )
    norm_np = normalized.detach().cpu().numpy()
    assert norm_np[0, 0, 1] == pytest.approx(0.0)
    assert norm_np[0, 1, 0] == pytest.approx(0.0)
    assert norm_np[0, 1, 2] == pytest.approx(0.0)
    assert norm_np[0, 0].sum() == pytest.approx(1.0)
    assert norm_np[0, 1].sum() == pytest.approx(1.0)


def test_masked_sample_actions_falls_back_to_noop_when_no_action_is_legal():
    torch = pytest.importorskip("torch")

    probs = torch.tensor([[[0.2, 0.3, 0.5]]], dtype=torch.float32)
    action_mask = torch.zeros((1, 1, 3), dtype=torch.float32)

    actions, normalized = _masked_sample_actions_from_prob_tensor(
        torch,
        probs,
        action_mask,
        deterministic=False,
    )

    np.testing.assert_array_equal(
        actions.detach().cpu().numpy().astype(np.int64),
        np.array([[0]], dtype=np.int64),
    )
    norm_np = normalized.detach().cpu().numpy()
    np.testing.assert_allclose(norm_np, np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32))
