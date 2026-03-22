from pathlib import Path

import numpy as np
import torch

from analytics.intent_disc_eval_batch import load_eval_batch, score_eval_batch
from basketworld.utils.callbacks import IntentDiversityCallback


class _DummyPolicy:
    def __init__(self):
        self.device = torch.device("cpu")


class _DummyModel:
    def __init__(self):
        self.num_timesteps = 321
        self.policy = _DummyPolicy()


def test_score_eval_batch_recomputes_metrics_from_saved_batch(tmp_path: Path):
    cb = IntentDiversityCallback(
        enabled=True,
        num_intents=4,
        disc_encoder_type="mlp_mean",
    )
    model = _DummyModel()
    cb.init_callback(model)
    cb._on_training_start()
    cb._maybe_build_discriminator(input_dim=272)
    assert cb._disc is not None

    with torch.no_grad():
        for param in cb._disc.parameters():
            param.zero_()

    batch_x = np.zeros((3, 272), dtype=np.float32)
    batch_y = np.array([0, 1, 0], dtype=np.int64)
    cb._last_disc_eval_acc = 2.0 / 3.0
    cb._last_disc_auc = 0.5
    cb._latest_disc_eval_batch = {
        "x": batch_x,
        "y": batch_y,
        "lengths": None,
    }

    disc_path = tmp_path / "intent_disc_iter_9.pt"
    batch_path = tmp_path / "intent_disc_eval_batch_iter_9.npz"
    saved_disc = cb.export_discriminator_checkpoint(
        str(disc_path), global_step=321, alternation_idx=9
    )
    saved_batch = cb.export_latest_discriminator_eval_batch(
        str(batch_path), global_step=321, alternation_idx=9
    )

    assert saved_disc is True
    assert saved_batch is True

    loaded = load_eval_batch(str(batch_path))
    assert loaded["x"].shape == (3, 272)
    assert loaded["y"].tolist() == [0, 1, 0]
    assert loaded["lengths"] is None
    assert int(loaded["meta"]["alternation_idx"]) == 9

    result = score_eval_batch(str(batch_path), str(disc_path), device="cpu")
    assert result["num_samples"] == 3
    assert np.isclose(result["recomputed_top1_acc"], 2.0 / 3.0, atol=1e-6)
    assert result["disc_checkpoint_config"]["num_intents"] == 4
    assert result["predicted_class_histogram"] == [3, 0, 0, 0]
    assert result["confusion_matrix_counts"][0][0] == 2
    assert result["confusion_matrix_counts"][1][0] == 1
