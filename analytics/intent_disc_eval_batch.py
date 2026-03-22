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
from basketworld.utils.intent_discovery import IntentDiscriminator
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


def _load_discriminator_checkpoint(path: str, device: str) -> tuple[IntentDiscriminator, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload or "config" not in payload:
        raise RuntimeError(f"Unsupported discriminator checkpoint format: {path}")
    config = dict(payload.get("config", {}) or {})
    disc = IntentDiscriminator(
        input_dim=int(config["input_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        num_intents=int(config["num_intents"]),
        dropout=float(config.get("dropout", 0.1)),
        encoder_type=str(config.get("encoder_type", "mlp_mean")),
        step_dim=int(config.get("step_dim", 64)),
    ).to(device)
    disc.load_state_dict(payload["state_dict"])
    disc.eval()
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
    if y_np.size and logits_np.ndim == 2:
        num_classes = int(disc_config.get("num_intents", logits_np.shape[1]))
        confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
        for truth, pred_idx in zip(y_np.tolist(), pred.tolist()):
            if 0 <= int(truth) < num_classes and 0 <= int(pred_idx) < num_classes:
                confusion[int(truth), int(pred_idx)] += 1

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
    }


def _resolve_inputs(
    batch_or_run_id: str,
    *,
    disc_path: Optional[str],
    checkpoint_idx: Optional[int],
) -> tuple[str, str]:
    if os.path.isfile(batch_or_run_id):
        batch_path = batch_or_run_id
        resolved_disc_path = disc_path or _resolve_local_discriminator_path(batch_path)
        if not resolved_disc_path or not os.path.isfile(resolved_disc_path):
            raise RuntimeError(
                "Could not resolve a local discriminator checkpoint. Pass --disc-path explicitly."
            )
        return batch_path, resolved_disc_path

    run_id = batch_or_run_id
    batch_artifact_pattern = re.compile(r"intent_disc_eval_batch_(?:alternation|iter)_(\d+)\.npz$")
    disc_artifact_pattern = re.compile(r"intent_disc_(?:alternation|iter)_(\d+)\.pt$")
    batch_path = _download_matching_artifact(
        run_id,
        pattern=batch_artifact_pattern,
        checkpoint_idx=checkpoint_idx,
        tmp_prefix="intent_disc_eval_batch_",
    )
    resolved_disc_path = disc_path or _download_matching_artifact(
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
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        out_path = os.path.abspath(args.json_out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(rendered)
            fh.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
