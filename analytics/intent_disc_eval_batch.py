#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

import mlflow
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basketworld.utils.callbacks import IntentDiversityCallback
from basketworld.utils.intent_discovery import (
    IntentDiscriminator,
    load_intent_discriminator_from_checkpoint,
)
from basketworld.utils.mlflow_config import setup_mlflow


def _extract_checkpoint_index(path_or_name: str) -> int | None:
    name = os.path.basename(str(path_or_name))
    match = re.search(r"_(?:alternation|iter)_(\d+)\.(?:pt|npz)$", name)
    if not match:
        return None
    return int(match.group(1))


def _download_matching_artifact(
    run_id: str,
    *,
    pattern: re.Pattern[str],
    checkpoint_idx: Optional[int],
    tmp_prefix: str,
) -> str:
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run_id, "models")
    candidates: list[tuple[int, str]] = []
    for item in artifacts:
        match = pattern.search(item.path)
        if match:
            candidates.append((int(match.group(1)), item.path))
    if not candidates:
        raise RuntimeError(f"No matching artifacts found in run {run_id} for pattern {pattern.pattern}")

    artifact_path: Optional[str] = None
    if checkpoint_idx is not None:
        for idx, path in candidates:
            if idx == int(checkpoint_idx):
                artifact_path = path
                break
        if artifact_path is None:
            raise RuntimeError(
                f"No matching artifact with checkpoint/index={int(checkpoint_idx)} found in run {run_id}"
            )
    else:
        candidates.sort(key=lambda item: item[0])
        artifact_path = candidates[-1][1]

    tmpdir = tempfile.mkdtemp(prefix=tmp_prefix)
    return client.download_artifacts(run_id, artifact_path, tmpdir)


def _download_exact_artifact(
    run_id: str,
    *,
    artifact_path: str,
    tmp_prefix: str,
) -> str:
    client = mlflow.tracking.MlflowClient()
    tmpdir = tempfile.mkdtemp(prefix=tmp_prefix)
    return client.download_artifacts(run_id, artifact_path, tmpdir)


def _try_download_pca_batch_artifact(
    run_id: str,
    *,
    checkpoint_idx: Optional[int],
) -> str | None:
    if checkpoint_idx is None:
        return None
    idx = int(checkpoint_idx)
    candidate_paths = [
        f"analysis/intent_pca/iter_{idx}/intent_disc_eval_batch_from_pca.npz",
        f"analysis/intent_pca/alternation_{idx}/intent_disc_eval_batch_from_pca.npz",
    ]
    for artifact_path in candidate_paths:
        try:
            return _download_exact_artifact(
                run_id,
                artifact_path=artifact_path,
                tmp_prefix="intent_disc_eval_batch_pca_",
            )
        except Exception:
            continue
    return None


def _resolve_artifact_uri(uri: str) -> tuple[str, str]:
    match = re.match(
        r"^mlflow-artifacts:/\d+/([0-9a-f]+)/artifacts/(.+)$",
        str(uri).strip(),
    )
    if not match:
        raise RuntimeError(f"Unsupported MLflow artifact URI: {uri}")
    return match.group(1), match.group(2)


def _resolve_local_discriminator_path(batch_path: str) -> str | None:
    batch_dir = os.path.dirname(os.path.abspath(batch_path))
    checkpoint_idx = _extract_checkpoint_index(batch_path)
    if checkpoint_idx is not None:
        candidate = os.path.join(batch_dir, f"intent_disc_iter_{checkpoint_idx}.pt")
        if os.path.isfile(candidate):
            return candidate
    candidates = sorted(
        Path(batch_dir).glob("intent_disc_iter_*.pt"),
        key=lambda path: _extract_checkpoint_index(path.name) or -1,
    )
    return str(candidates[-1]) if candidates else None


def _resolve_disc_path_input(disc_path: Optional[str]) -> Optional[str]:
    if not disc_path:
        return None
    if os.path.isfile(disc_path):
        return disc_path
    if str(disc_path).startswith("mlflow-artifacts:/"):
        run_id, artifact_path = _resolve_artifact_uri(str(disc_path))
        return _download_exact_artifact(
            run_id,
            artifact_path=artifact_path,
            tmp_prefix="intent_disc_ckpt_uri_",
        )
    return disc_path


def _load_discriminator_checkpoint(path: str, device: str) -> tuple[IntentDiscriminator, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload or "config" not in payload:
        raise RuntimeError(f"Unsupported discriminator checkpoint format: {path}")
    disc = load_intent_discriminator_from_checkpoint(payload, device=device)
    return disc, payload


def load_eval_batch(path: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as payload:
        x_np = np.asarray(payload["x"], dtype=np.float32)
        y_np = np.asarray(payload["y"], dtype=np.int64).reshape(-1)
        has_lengths = bool(
            int(payload["has_lengths"][0]) if "has_lengths" in payload else 0
        )
        lengths_np = None
        if has_lengths and "lengths" in payload:
            lengths_np = np.asarray(payload["lengths"], dtype=np.int64).reshape(-1)

        meta: dict[str, Any] = {}
        for key in (
            "global_step",
            "alternation_idx",
            "rollout_counter",
            "num_intents",
            "max_obs_dim",
            "max_action_dim",
            "disc_top1_acc_eval",
            "disc_auc_ovr_macro",
        ):
            if key not in payload:
                continue
            arr = payload[key]
            if np.asarray(arr).size == 0:
                continue
            value = np.asarray(arr).reshape(-1)[0]
            if isinstance(value, np.generic):
                value = value.item()
            meta[key] = value
        if "encoder_type" in payload:
            raw_encoder = np.asarray(payload["encoder_type"]).reshape(-1)[0]
            if isinstance(raw_encoder, bytes):
                raw_encoder = raw_encoder.decode("utf-8")
            if isinstance(raw_encoder, np.generic):
                raw_encoder = raw_encoder.item()
            meta["encoder_type"] = str(raw_encoder)

    return {
        "x": x_np,
        "y": y_np,
        "lengths": lengths_np,
        "meta": meta,
    }


def score_eval_batch(
    batch_path: str,
    disc_path: str,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    batch = load_eval_batch(batch_path)
    disc, payload = _load_discriminator_checkpoint(disc_path, device=device)
    disc_config = dict(payload.get("config", {}) or {})
    x_np = np.asarray(batch["x"], dtype=np.float32)
    y_np = np.asarray(batch["y"], dtype=np.int64).reshape(-1)
    lengths_np = batch.get("lengths")

    x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
    lengths = None
    if lengths_np is not None:
        lengths = torch.as_tensor(lengths_np, dtype=torch.long, device=device)
    with torch.no_grad():
        logits = disc(x, lengths)
        pred = torch.argmax(logits, dim=-1).detach().cpu().numpy()
    logits_np = logits.detach().cpu().numpy()
    top1 = float(np.mean(pred == y_np)) if y_np.size else 0.0
    auc = IntentDiversityCallback._multiclass_auc_ovr_macro(
        logits_np, y_np, num_classes=int(disc_config.get("num_intents", logits_np.shape[1]))
    )
    confusion = np.zeros((0, 0), dtype=np.int64)
    confusion_row_normalized = np.zeros((0, 0), dtype=np.float64)
    if y_np.size and logits_np.ndim == 2:
        num_classes = int(disc_config.get("num_intents", logits_np.shape[1]))
        confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
        for truth, pred_idx in zip(y_np.tolist(), pred.tolist()):
            if 0 <= int(truth) < num_classes and 0 <= int(pred_idx) < num_classes:
                confusion[int(truth), int(pred_idx)] += 1
        row_totals = confusion.sum(axis=1, keepdims=True).astype(np.float64, copy=False)
        confusion_row_normalized = np.divide(
            confusion.astype(np.float64, copy=False),
            np.maximum(row_totals, 1.0),
        )

    return {
        "batch_path": os.path.abspath(batch_path),
        "discriminator_checkpoint_path": os.path.abspath(disc_path),
        "num_samples": int(y_np.size),
        "stored_batch_meta": dict(batch.get("meta", {})),
        "disc_checkpoint_config": disc_config,
        "disc_checkpoint_meta": dict(payload.get("meta", {}) or {}),
        "recomputed_top1_acc": top1,
        "recomputed_auc_ovr_macro": None if auc is None else float(auc),
        "predicted_class_histogram": np.bincount(
            pred.astype(np.int64), minlength=int(disc_config.get("num_intents", 0))
        ).tolist()
        if pred.size
        else [],
        "confusion_matrix_counts": confusion.tolist(),
        "confusion_matrix_row_normalized": [
            [float(v) for v in row] for row in confusion_row_normalized.tolist()
        ],
    }


def validate_eval_result(
    result: dict[str, Any],
    *,
    min_top1: float | None = None,
    min_auc: float | None = None,
) -> dict[str, Any]:
    failed_checks: list[str] = []
    if min_top1 is not None:
        actual_top1 = float(result.get("recomputed_top1_acc", 0.0))
        if actual_top1 < float(min_top1):
            failed_checks.append(
                f"top1 {actual_top1:.6f} < required {float(min_top1):.6f}"
            )
    if min_auc is not None:
        actual_auc = result.get("recomputed_auc_ovr_macro", None)
        if actual_auc is None:
            failed_checks.append("auc unavailable")
        elif float(actual_auc) < float(min_auc):
            failed_checks.append(
                f"auc {float(actual_auc):.6f} < required {float(min_auc):.6f}"
            )
    return {
        "passed": len(failed_checks) == 0,
        "failed_checks": failed_checks,
    }


def _resolve_inputs(
    batch_or_run_id: str,
    *,
    disc_path: Optional[str],
    checkpoint_idx: Optional[int],
) -> tuple[str, str]:
    resolved_disc_path_arg = _resolve_disc_path_input(disc_path)
    if os.path.isfile(batch_or_run_id):
        batch_path = batch_or_run_id
        resolved_disc_path = resolved_disc_path_arg or _resolve_local_discriminator_path(batch_path)
        if not resolved_disc_path or not os.path.isfile(resolved_disc_path):
            raise RuntimeError(
                "Could not resolve a local discriminator checkpoint. Pass --disc-path explicitly."
            )
        return batch_path, resolved_disc_path

    if str(batch_or_run_id).startswith("mlflow-artifacts:/"):
        run_id_from_uri, artifact_path = _resolve_artifact_uri(batch_or_run_id)
        batch_path = _download_exact_artifact(
            run_id_from_uri,
            artifact_path=artifact_path,
            tmp_prefix="intent_disc_eval_batch_uri_",
        )
        resolved_disc_path = resolved_disc_path_arg or _download_matching_artifact(
            run_id_from_uri,
            pattern=re.compile(r"intent_disc_(?:alternation|iter)_(\d+)\.pt$"),
            checkpoint_idx=checkpoint_idx,
            tmp_prefix="intent_disc_ckpt_",
        )
        return batch_path, resolved_disc_path

    run_id = batch_or_run_id
    batch_artifact_pattern = re.compile(r"intent_disc_eval_batch_(?:alternation|iter)_(\d+)\.npz$")
    disc_artifact_pattern = re.compile(r"intent_disc_(?:alternation|iter)_(\d+)\.pt$")
    try:
        batch_path = _download_matching_artifact(
            run_id,
            pattern=batch_artifact_pattern,
            checkpoint_idx=checkpoint_idx,
            tmp_prefix="intent_disc_eval_batch_",
        )
    except RuntimeError:
        batch_path = _try_download_pca_batch_artifact(
            run_id,
            checkpoint_idx=checkpoint_idx,
        )
        if batch_path is None:
            raise
    resolved_disc_path = resolved_disc_path_arg or _download_matching_artifact(
        run_id,
        pattern=disc_artifact_pattern,
        checkpoint_idx=checkpoint_idx,
        tmp_prefix="intent_disc_ckpt_",
    )
    return batch_path, resolved_disc_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute discriminator top1/AUC on an exported training eval batch."
    )
    parser.add_argument(
        "batch_path_or_run_id",
        help="Local intent_disc_eval_batch_iter_<N>.npz path, or an MLflow run id.",
    )
    parser.add_argument(
        "--disc-path",
        type=str,
        default=None,
        help="Optional local discriminator checkpoint path. Required for local batch paths if it cannot be inferred.",
    )
    parser.add_argument(
        "--alternation-index",
        type=int,
        default=None,
        help="Checkpoint/alternation index when downloading artifacts from an MLflow run.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for offline scoring.",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional path to write the result JSON.",
    )
    parser.add_argument(
        "--min-top1",
        type=float,
        default=None,
        help="Optional minimum required recomputed top1 accuracy. Exits nonzero if unmet.",
    )
    parser.add_argument(
        "--min-auc",
        type=float,
        default=None,
        help="Optional minimum required recomputed one-vs-rest macro AUC. Exits nonzero if unmet.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not os.path.isfile(args.batch_path_or_run_id):
        setup_mlflow(verbose=False)
    batch_path, disc_path = _resolve_inputs(
        args.batch_path_or_run_id,
        disc_path=args.disc_path,
        checkpoint_idx=args.alternation_index,
    )
    result = score_eval_batch(batch_path, disc_path, device=args.device)
    result["validation"] = validate_eval_result(
        result,
        min_top1=args.min_top1,
        min_auc=args.min_auc,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        out_path = os.path.abspath(args.json_out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(rendered)
            fh.write("\n")
    return 0 if bool(result["validation"]["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
