from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
from typing import Any

import numpy as np


CHECKPOINT_VERSION = 2
LEGACY_PICKLE_CHECKPOINT_VERSION = 1
STATE_SUBDIR = "state"
METADATA_FILENAME = "metadata.json"


def _tree_to_numpy(tree):
    if isinstance(tree, dict):
        return {key: _tree_to_numpy(value) for key, value in tree.items()}
    if isinstance(tree, tuple) and hasattr(tree, "_fields"):
        return type(tree)(*(_tree_to_numpy(value) for value in tree))
    if isinstance(tree, tuple):
        return tuple(_tree_to_numpy(value) for value in tree)
    if isinstance(tree, list):
        return [_tree_to_numpy(value) for value in tree]
    if isinstance(tree, np.ndarray):
        return np.asarray(tree)
    shape = getattr(tree, "shape", None)
    dtype = getattr(tree, "dtype", None)
    if shape is not None and dtype is not None:
        return np.asarray(tree)
    return tree


def _to_jsonable(value):
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return {
            "__kind__": "tuple",
            "items": [_to_jsonable(item) for item in value],
        }
    if isinstance(value, np.ndarray):
        return {
            "__kind__": "ndarray",
            "dtype": str(value.dtype),
            "data": value.tolist(),
        }
    if isinstance(value, np.generic):
        return value.item()
    return value


def _from_jsonable(value):
    if isinstance(value, dict):
        kind = value.get("__kind__")
        if kind == "ndarray":
            return np.asarray(value["data"], dtype=np.dtype(value["dtype"]))
        if kind == "tuple":
            return tuple(_from_jsonable(item) for item in value["items"])
        return {key: _from_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_from_jsonable(item) for item in value]
    return value


def build_checkpoint_payload(
    *,
    update_index: int,
    trainer_config: dict[str, Any],
    policy_spec: dict[str, Any],
    frozen_config: dict[str, Any],
    params,
    opt_state,
    current_state,
    eval_initial_state,
    base_key,
    eval_trajectories: list[dict[str, Any]],
    last_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "update_index": int(update_index),
        "trainer_config": dict(trainer_config),
        "policy_spec": dict(policy_spec),
        "frozen_config": dict(frozen_config),
        "state": {
            "params": _tree_to_numpy(params),
            "opt_state": _tree_to_numpy(opt_state),
            "current_state": _tree_to_numpy(current_state),
            "eval_initial_state": _tree_to_numpy(eval_initial_state),
            "base_key": _tree_to_numpy(base_key),
        },
        "eval_trajectories": list(eval_trajectories),
        "last_metrics": None if last_metrics is None else dict(last_metrics),
    }


def _metadata_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "checkpoint_version": int(payload["checkpoint_version"]),
        "saved_at": str(payload["saved_at"]),
        "update_index": int(payload["update_index"]),
        "trainer_config": _to_jsonable(dict(payload["trainer_config"])),
        "policy_spec": _to_jsonable(dict(payload["policy_spec"])),
        "frozen_config": _to_jsonable(dict(payload["frozen_config"])),
        "eval_trajectories": _to_jsonable(payload["eval_trajectories"]),
        "last_metrics": _to_jsonable(payload["last_metrics"]),
    }


def _payload_from_metadata(metadata: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    return {
        "checkpoint_version": int(metadata["checkpoint_version"]),
        "saved_at": metadata["saved_at"],
        "update_index": int(metadata["update_index"]),
        "trainer_config": _from_jsonable(metadata["trainer_config"]),
        "policy_spec": _from_jsonable(metadata["policy_spec"]),
        "frozen_config": _from_jsonable(metadata["frozen_config"]),
        "params": state["params"],
        "opt_state": state["opt_state"],
        "current_state": state["current_state"],
        "eval_initial_state": state["eval_initial_state"],
        "base_key": state["base_key"],
        "eval_trajectories": _from_jsonable(metadata["eval_trajectories"]),
        "last_metrics": _from_jsonable(metadata["last_metrics"]),
    }


def _state_dir(path: str | Path) -> Path:
    return Path(path) / STATE_SUBDIR


def _metadata_path(path: str | Path) -> Path:
    return Path(path) / METADATA_FILENAME


def _save_orbax_state(path: str | Path, state: dict[str, Any]) -> None:
    import orbax.checkpoint as ocp

    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(_state_dir(path), state, force=True)
    checkpointer.wait_until_finished()
    checkpointer.close()


def _load_orbax_state(path: str | Path) -> dict[str, Any]:
    import orbax.checkpoint as ocp

    checkpointer = ocp.StandardCheckpointer()
    try:
        restored = checkpointer.restore(_state_dir(path))
        return dict(restored)
    finally:
        checkpointer.close()


def _load_legacy_pickle_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path)
    with checkpoint_path.open("rb") as handle:
        payload = pickle.load(handle)
    version = int(payload.get("checkpoint_version", 0))
    if version != LEGACY_PICKLE_CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported legacy JAX checkpoint version {version}; expected {LEGACY_PICKLE_CHECKPOINT_VERSION}."
        )
    return payload


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    _save_orbax_state(checkpoint_path, dict(payload["state"]))
    with _metadata_path(checkpoint_path).open("w", encoding="utf-8") as handle:
        json.dump(_metadata_from_payload(payload), handle, indent=2, sort_keys=True)
    return checkpoint_path


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if checkpoint_path.is_file():
        return _load_legacy_pickle_checkpoint(checkpoint_path)

    metadata_file = _metadata_path(checkpoint_path)
    if not metadata_file.is_file():
        raise FileNotFoundError(f"JAX checkpoint metadata not found at {metadata_file}.")

    with metadata_file.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    version = int(metadata.get("checkpoint_version", 0))
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported JAX checkpoint version {version}; expected {CHECKPOINT_VERSION}."
        )
    state = _load_orbax_state(checkpoint_path)
    return _payload_from_metadata(metadata, state)


def build_checkpoint_paths(
    checkpoint_dir: str | Path,
    *,
    update_index: int,
) -> tuple[Path, Path]:
    root = Path(checkpoint_dir)
    numbered = root / f"update_{int(update_index):07d}"
    latest = root / "latest"
    return numbered, latest
