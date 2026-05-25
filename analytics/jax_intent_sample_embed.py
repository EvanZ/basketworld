#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basketworld.utils.mlflow_config import setup_mlflow
from basketworld.utils.play_names import (
    build_play_name_mapping,
    lookup_play_name,
    play_name_seed_key,
)


def _effective_tsne_perplexity(n_samples: int, requested: float) -> float:
    if int(n_samples) < 3:
        raise RuntimeError("Need at least three samples to run t-SNE.")
    max_valid = max(1.0, float(int(n_samples) - 1))
    return float(min(float(requested), max_valid - 1.0e-6))


def _resolve_artifact_uri(uri: str) -> tuple[str, str]:
    match = re.match(
        r"^mlflow-artifacts:/\d+/([0-9a-f]+)/artifacts/(.+)$",
        str(uri).strip(),
    )
    if not match:
        raise RuntimeError(f"Unsupported MLflow artifact URI: {uri}")
    return match.group(1), match.group(2)


def _extract_update_index(path_or_name: str) -> int | None:
    name = os.path.basename(str(path_or_name))
    match = re.search(r"update_(\d+)", name)
    if match:
        return int(match.group(1))
    match = re.search(r"intent_samples_update_(\d+)\.npz$", name)
    if match:
        return int(match.group(1))
    return None


def _list_artifacts_recursive(client, run_id: str, path: str = "") -> list[Any]:
    items = client.list_artifacts(run_id, path)
    out: list[Any] = []
    for item in items:
        if bool(getattr(item, "is_dir", False)):
            out.extend(_list_artifacts_recursive(client, run_id, item.path))
        else:
            out.append(item)
    return out


def _download_exact_artifact(run_id: str, artifact_path: str) -> str:
    tmpdir = tempfile.mkdtemp(prefix="jax_intent_sample_")
    return mlflow.tracking.MlflowClient().download_artifacts(run_id, artifact_path, tmpdir)


def _download_sample_from_run(run_id: str, *, update_index: int | None) -> tuple[str, str]:
    client = mlflow.tracking.MlflowClient()
    candidates: list[tuple[int, str]] = []
    for item in _list_artifacts_recursive(client, run_id, "intent_samples"):
        path = str(item.path)
        if not path.endswith(".npz"):
            continue
        idx = _extract_update_index(path)
        if idx is None:
            continue
        candidates.append((int(idx), path))
    if not candidates:
        raise RuntimeError(f"No JAX intent sample .npz artifacts found under run {run_id}")

    artifact_path: str | None = None
    if update_index is not None:
        for idx, path in candidates:
            if int(idx) == int(update_index):
                artifact_path = path
                break
        if artifact_path is None:
            raise RuntimeError(
                f"No JAX intent sample artifact with update_index={int(update_index)} "
                f"found under run {run_id}"
            )
    else:
        candidates.sort(key=lambda item: item[0])
        artifact_path = candidates[-1][1]
    return _download_exact_artifact(run_id, artifact_path), artifact_path


def _resolve_sample_input(path_or_run_id: str, *, update_index: int | None) -> tuple[str, str | None, str | None]:
    value = str(path_or_run_id).strip()
    if os.path.isfile(value):
        return value, None, None
    if value.startswith("mlflow-artifacts:/"):
        run_id, artifact_path = _resolve_artifact_uri(value)
        return _download_exact_artifact(run_id, artifact_path), run_id, artifact_path
    run_id = value
    local_path, artifact_path = _download_sample_from_run(run_id, update_index=update_index)
    return local_path, run_id, artifact_path


def _load_sample(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _matrix_from_sample(sample: dict[str, np.ndarray], repr_mode: str) -> np.ndarray:
    mode = str(repr_mode).strip().lower()
    if mode == "embedding":
        if "embedding" not in sample:
            raise RuntimeError("Sample file does not contain an 'embedding' array.")
        return np.asarray(sample["embedding"], dtype=np.float32)
    if mode == "features":
        if "features" not in sample:
            raise RuntimeError("Sample file does not contain a 'features' array.")
        return np.asarray(sample["features"], dtype=np.float32)
    if mode in {"players_mean", "players_flat"}:
        if "players" not in sample:
            raise RuntimeError("Sample file does not contain a 'players' array.")
        players = np.asarray(sample["players"], dtype=np.float32)
        if players.ndim != 3:
            raise RuntimeError(f"Expected players [N,P,D], got {tuple(players.shape)}")
        globals_vec = np.asarray(
            sample.get("globals", np.zeros((players.shape[0], 0), dtype=np.float32)),
            dtype=np.float32,
        ).reshape(players.shape[0], -1)
        role_flag = np.asarray(
            sample.get("role_flag", np.zeros((players.shape[0], 0), dtype=np.float32)),
            dtype=np.float32,
        ).reshape(players.shape[0], -1)
        if mode == "players_mean":
            player_part = np.mean(players, axis=1, dtype=np.float32)
        else:
            player_part = players.reshape(players.shape[0], -1)
        return np.concatenate([player_part, globals_vec, role_flag], axis=1).astype(np.float32)
    raise ValueError(f"Unsupported repr_mode={repr_mode!r}")


def _labels_from_sample(sample: dict[str, np.ndarray]) -> np.ndarray:
    if "intent_index" not in sample:
        raise RuntimeError("Sample file does not contain an 'intent_index' array.")
    return np.asarray(sample["intent_index"], dtype=np.int64).reshape(-1)


def _infer_num_intents(labels: np.ndarray, sample: dict[str, np.ndarray]) -> int:
    candidates: list[int] = []
    if labels.size:
        candidates.append(int(np.max(labels)) + 1)
    for key in ("intent_logits", "intent_probs"):
        if key not in sample:
            continue
        arr = np.asarray(sample[key])
        if arr.ndim >= 2:
            candidates.append(int(arr.shape[-1]))
    return max(candidates) if candidates else 0


def _build_plot_play_name_map(
    *,
    run_id: str | None,
    sample_path: str,
    artifact_path: str | None,
    labels: np.ndarray,
    sample: dict[str, np.ndarray],
) -> dict[int, str]:
    num_intents = _infer_num_intents(labels, sample)
    if num_intents <= 0:
        return {}
    seed_key = play_name_seed_key(
        run_id=run_id,
        unified_policy_path=artifact_path,
        fallback=Path(sample_path).stem,
    )
    return {
        int(idx): str(name)
        for idx, name in build_play_name_mapping(seed_key, num_intents).items()
    }


def _format_intent_label(intent_index: int, play_name_map: dict[int, str] | None) -> str:
    play_name = lookup_play_name(play_name_map, intent_index)
    if play_name:
        return f"{play_name} (z={int(intent_index)})"
    return f"z={int(intent_index)}"


def _plot_embedding_scatter(
    coords: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    title: str,
    *,
    xlabel: str,
    ylabel: str,
    play_name_map: dict[int, str] | None = None,
) -> None:
    unique_labels = sorted({int(x) for x in labels.tolist()})
    cmap = plt.get_cmap("tab10" if len(unique_labels) <= 10 else "tab20")
    fig, ax = plt.subplots(figsize=(11, 9))
    for idx, intent_index in enumerate(unique_labels):
        mask = labels == int(intent_index)
        color = cmap(idx % cmap.N)
        display_label = _format_intent_label(intent_index, play_name_map)
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=12,
            alpha=0.55,
            label=display_label,
            color=color,
            edgecolors="none",
        )
        centroid = np.mean(coords[mask], axis=0)
        ax.scatter(
            [centroid[0]],
            [centroid[1]],
            s=80,
            color=color,
            edgecolors="black",
            linewidths=0.8,
        )
        ax.text(centroid[0], centroid[1], f" {display_label}", fontsize=9, va="center")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", frameon=True, fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_points_csv(
    output_path: Path,
    coords: np.ndarray,
    labels: np.ndarray,
    sample: dict[str, np.ndarray],
    *,
    x_key: str,
    y_key: str,
    play_name_map: dict[int, str] | None = None,
) -> None:
    optional_fields = [
        "bonus",
        "pass_attempt",
        "completed_pass",
        "assist",
        "turnover",
        "shot_attempt",
        "shot_make",
        "shot_dunk",
        "shot_two",
        "shot_three",
        "offense_score_delta",
        "defense_score_delta",
    ]
    optional_arrays = {
        key: np.asarray(sample[key]).reshape(-1)
        for key in optional_fields
        if key in sample and np.asarray(sample[key]).reshape(-1).shape[0] == labels.shape[0]
    }
    with output_path.open("w", newline="") as f:
        fieldnames = ["sample_idx", "intent_index", "intent_label", x_key, y_key, *optional_arrays.keys()]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, coord in enumerate(coords):
            row: dict[str, Any] = {
                "sample_idx": int(idx),
                "intent_index": int(labels[idx]),
                "intent_label": _format_intent_label(int(labels[idx]), play_name_map),
                x_key: float(coord[0]),
                y_key: float(coord[1]),
            }
            for key, arr in optional_arrays.items():
                value = arr[idx]
                row[key] = value.item() if isinstance(value, np.generic) else value
            writer.writerow(row)


def _summarize_by_intent(
    labels: np.ndarray,
    sample: dict[str, np.ndarray],
    *,
    play_name_map: dict[int, str] | None = None,
) -> dict[str, Any]:
    fields = [
        "bonus",
        "pass_attempt",
        "completed_pass",
        "assist",
        "turnover",
        "shot_attempt",
        "shot_make",
        "shot_dunk",
        "shot_two",
        "shot_three",
        "offense_score_delta",
        "defense_score_delta",
    ]
    summary: dict[str, Any] = {}
    for intent_index in sorted({int(x) for x in labels.tolist()}):
        mask = labels == int(intent_index)
        item: dict[str, Any] = {
            "count": int(np.sum(mask)),
            "label": _format_intent_label(intent_index, play_name_map),
        }
        play_name = lookup_play_name(play_name_map, intent_index)
        if play_name:
            item["play_name"] = play_name
        for field in fields:
            if field not in sample:
                continue
            arr = np.asarray(sample[field]).reshape(-1)
            if arr.shape[0] != labels.shape[0]:
                continue
            item[f"{field}_mean"] = float(np.mean(arr[mask])) if np.any(mask) else 0.0
        summary[str(intent_index)] = item
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PCA/t-SNE on JAX intent sample dumps saved by the JAX trainer."
    )
    parser.add_argument(
        "sample_path_or_run_id",
        help="Local JAX intent sample .npz path, MLflow run id, or MLflow artifact URI.",
    )
    parser.add_argument(
        "--update-index",
        type=int,
        default=None,
        help="Specific JAX update index to download from MLflow. Defaults to latest sample.",
    )
    parser.add_argument(
        "--repr-mode",
        choices=["embedding", "features", "players_mean", "players_flat"],
        default="embedding",
        help="Representation to visualize.",
    )
    parser.add_argument(
        "--embedding-methods",
        nargs="+",
        choices=["pca", "tsne"],
        default=["pca", "tsne"],
        help="Embedding methods to run.",
    )
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--json-out", type=str, default=None)
    parser.add_argument(
        "--log-to-mlflow",
        action="store_true",
        help="Log generated plots/CSV/summary back to the source MLflow run.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not os.path.isfile(str(args.sample_path_or_run_id)):
        setup_mlflow(verbose=False)
    sample_path, run_id, artifact_path = _resolve_sample_input(
        str(args.sample_path_or_run_id),
        update_index=args.update_index,
    )
    sample = _load_sample(sample_path)
    labels = _labels_from_sample(sample)
    matrix = _matrix_from_sample(sample, str(args.repr_mode))
    if matrix.ndim != 2:
        raise RuntimeError(f"Expected representation matrix [N,D], got {tuple(matrix.shape)}")
    if matrix.shape[0] != labels.shape[0]:
        raise RuntimeError(
            f"Representation sample count {matrix.shape[0]} does not match labels {labels.shape[0]}"
        )
    if matrix.shape[0] < 2:
        raise RuntimeError("Need at least two samples to visualize intent samples.")
    matrix_scaled = StandardScaler().fit_transform(matrix)
    play_name_map = _build_plot_play_name_map(
        run_id=run_id,
        sample_path=sample_path,
        artifact_path=artifact_path,
        labels=labels,
        sample=sample,
    )

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        key = run_id or Path(sample_path).stem
        update_index = (
            int(args.update_index)
            if args.update_index is not None
            else _extract_update_index(artifact_path or sample_path)
        )
        suffix = f"update_{int(update_index):07d}" if update_index is not None else Path(sample_path).stem
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = REPO_ROOT / "analytics" / "jax_intent_sample_embed" / f"{key}_{suffix}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    label_hist = np.bincount(labels, minlength=max(1, int(np.max(labels)) + 1)).astype(int).tolist()
    summary: dict[str, Any] = {
        "run_id": run_id,
        "artifact_path": artifact_path,
        "sample_path": os.path.abspath(sample_path),
        "update_index": (
            int(args.update_index)
            if args.update_index is not None
            else _extract_update_index(artifact_path or sample_path)
        ),
        "repr_mode": str(args.repr_mode),
        "embedding_methods": [str(x) for x in args.embedding_methods],
        "num_samples": int(labels.shape[0]),
        "feature_dim": int(matrix.shape[1]),
        "play_name_map": {str(int(idx)): str(name) for idx, name in sorted(play_name_map.items())},
        "label_histogram": label_hist,
        "sample_arrays": {key: list(np.asarray(value).shape) for key, value in sorted(sample.items())},
        "by_intent": _summarize_by_intent(labels, sample, play_name_map=play_name_map),
    }

    title_prefix = (
        f"JAX Intent Samples ({args.repr_mode})\n"
        f"samples={labels.shape[0]}  update={summary['update_index']}"
    )

    if "pca" in args.embedding_methods:
        pca = PCA(n_components=2, random_state=int(args.seed))
        pca_coords = pca.fit_transform(matrix_scaled)
        pca_plot_path = output_dir / "jax_intent_samples_pca.png"
        pca_csv_path = output_dir / "jax_intent_samples_pca_points.csv"
        _plot_embedding_scatter(
            pca_coords,
            labels,
            pca_plot_path,
            title_prefix,
            xlabel=f"PC1 ({100.0 * float(pca.explained_variance_ratio_[0]):.2f}% var)",
            ylabel=f"PC2 ({100.0 * float(pca.explained_variance_ratio_[1]):.2f}% var)",
            play_name_map=play_name_map,
        )
        _write_points_csv(
            pca_csv_path,
            pca_coords,
            labels,
            sample,
            x_key="pc1",
            y_key="pc2",
            play_name_map=play_name_map,
        )
        summary["pca"] = {
            "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_.tolist()],
            "plot_path": str(pca_plot_path),
            "csv_path": str(pca_csv_path),
        }

    if "tsne" in args.embedding_methods:
        perplexity = _effective_tsne_perplexity(labels.shape[0], float(args.tsne_perplexity))
        tsne = TSNE(
            n_components=2,
            init="pca",
            random_state=int(args.seed),
            perplexity=float(perplexity),
            learning_rate="auto",
        )
        tsne_coords = tsne.fit_transform(matrix_scaled)
        tsne_plot_path = output_dir / "jax_intent_samples_tsne.png"
        tsne_csv_path = output_dir / "jax_intent_samples_tsne_points.csv"
        _plot_embedding_scatter(
            tsne_coords,
            labels,
            tsne_plot_path,
            title_prefix,
            xlabel="t-SNE 1",
            ylabel="t-SNE 2",
            play_name_map=play_name_map,
        )
        _write_points_csv(
            tsne_csv_path,
            tsne_coords,
            labels,
            sample,
            x_key="tsne1",
            y_key="tsne2",
            play_name_map=play_name_map,
        )
        summary["tsne"] = {
            "effective_perplexity": float(perplexity),
            "plot_path": str(tsne_plot_path),
            "csv_path": str(tsne_csv_path),
        }

    summary_path = output_dir / "jax_intent_sample_embed_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if args.json_out:
        json_out_path = Path(args.json_out).expanduser().resolve()
        json_out_path.parent.mkdir(parents=True, exist_ok=True)
        json_out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    if args.log_to_mlflow and run_id:
        client = mlflow.tracking.MlflowClient()
        update_index = summary.get("update_index")
        artifact_subpath = (
            f"analysis/jax_intent_sample_embed/update_{int(update_index):07d}"
            if update_index is not None
            else "analysis/jax_intent_sample_embed"
        )
        for path in sorted(output_dir.iterdir()):
            if path.is_file():
                client.log_artifact(run_id, str(path), artifact_path=artifact_subpath)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
