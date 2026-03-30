from __future__ import annotations

import numpy as np

from analytics.intent_disc_batch_embed import (
    _effective_tsne_perplexity,
    _raw_mean_episode_matrix,
)


def test_raw_mean_episode_matrix_respects_lengths() -> None:
    x = np.asarray(
        [
            [[1.0, 3.0], [5.0, 7.0], [100.0, 100.0]],
            [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]],
        ],
        dtype=np.float32,
    )
    lengths = np.asarray([2, 3], dtype=np.int64)

    result = _raw_mean_episode_matrix(x, lengths)

    assert result.shape == (2, 2)
    np.testing.assert_allclose(result[0], np.asarray([3.0, 5.0], dtype=np.float32))
    np.testing.assert_allclose(result[1], np.asarray([6.0, 8.0], dtype=np.float32))


def test_effective_tsne_perplexity_is_clipped_below_sample_count() -> None:
    assert _effective_tsne_perplexity(10, 30.0) < 10.0
    assert _effective_tsne_perplexity(10, 5.0) == 5.0
