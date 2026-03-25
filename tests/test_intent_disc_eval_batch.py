from pathlib import Path

import numpy as np
import torch

from analytics.intent_disc_eval_batch import (
    load_eval_batch,
    _resolve_artifact_uri,
    _resolve_disc_path_input,
    _resolve_inputs,
    score_eval_batch,
    validate_eval_result,
)
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
        disc_lambda_shot=0.1,
        disc_lambda_q=0.05,
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
    assert bool(result["disc_checkpoint_config"]["enable_shot_end_head"]) is True
    assert bool(result["disc_checkpoint_config"]["enable_shot_quality_head"]) is True
    assert result["predicted_class_histogram"] == [3, 0, 0, 0]
    assert result["confusion_matrix_counts"][0][0] == 2
    assert result["confusion_matrix_counts"][1][0] == 1
    assert np.allclose(
        np.asarray(result["confusion_matrix_row_normalized"], dtype=np.float64)[0],
        [1.0, 0.0, 0.0, 0.0],
    )
    assert np.allclose(
        np.asarray(result["confusion_matrix_row_normalized"], dtype=np.float64)[1],
        [1.0, 0.0, 0.0, 0.0],
    )


def test_validate_eval_result_enforces_thresholds():
    result = {
        "recomputed_top1_acc": 0.62,
        "recomputed_auc_ovr_macro": 0.91,
    }

    passed = validate_eval_result(result, min_top1=0.6, min_auc=0.9)
    failed = validate_eval_result(result, min_top1=0.7, min_auc=0.95)

    assert passed["passed"] is True
    assert passed["failed_checks"] == []
    assert failed["passed"] is False
    assert len(failed["failed_checks"]) == 2


def test_resolve_artifact_uri_parses_mlflow_artifact_uri():
    run_id, artifact_path = _resolve_artifact_uri(
        "mlflow-artifacts:/8/925b131ae92a4b128b34d301897a2b65/artifacts/analysis/intent_pca/iter_66/intent_disc_eval_batch_from_pca.npz"
    )

    assert run_id == "925b131ae92a4b128b34d301897a2b65"
    assert artifact_path == "analysis/intent_pca/iter_66/intent_disc_eval_batch_from_pca.npz"


def test_resolve_inputs_falls_back_to_pca_batch_when_training_batch_missing(monkeypatch):
    def fake_download_matching(run_id, *, pattern, checkpoint_idx, tmp_prefix):
        if "eval_batch" in tmp_prefix:
            raise RuntimeError("missing training eval batch")
        return "/tmp/intent_disc_iter_66.pt"

    monkeypatch.setattr(
        "analytics.intent_disc_eval_batch._download_matching_artifact",
        fake_download_matching,
    )
    monkeypatch.setattr(
        "analytics.intent_disc_eval_batch._try_download_pca_batch_artifact",
        lambda run_id, checkpoint_idx: "/tmp/intent_disc_eval_batch_from_pca.npz",
    )

    batch_path, disc_path = _resolve_inputs(
        "925b131ae92a4b128b34d301897a2b65",
        disc_path=None,
        checkpoint_idx=66,
    )

    assert batch_path == "/tmp/intent_disc_eval_batch_from_pca.npz"
    assert disc_path == "/tmp/intent_disc_iter_66.pt"


def test_resolve_disc_path_input_downloads_mlflow_artifact_uri(monkeypatch):
    monkeypatch.setattr(
        "analytics.intent_disc_eval_batch._download_exact_artifact",
        lambda run_id, artifact_path, tmp_prefix: f"/tmp/{artifact_path.split('/')[-1]}",
    )

    resolved = _resolve_disc_path_input(
        "mlflow-artifacts:/8/925b131ae92a4b128b34d301897a2b65/artifacts/models/intent_disc_iter_66.pt"
    )

    assert resolved == "/tmp/intent_disc_iter_66.pt"
