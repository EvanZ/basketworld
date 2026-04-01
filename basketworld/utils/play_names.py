from __future__ import annotations

import hashlib
import random
from typing import Any


PLAY_NAME_POOL_VERSION = 2

_PLAY_NAME_PREFIXES = [
    "Atlas",
    "Cinder",
    "Comet",
    "Copper",
    "Crimson",
    "Delta",
    "Echo",
    "Ember",
    "Ghost",
    "Glass",
    "Harbor",
    "Ivory",
    "Jade",
    "Jet",
    "Lunar",
    "Maple",
    "Nova",
    "Onyx",
    "Raven",
    "Silver",
    "Solar",
    "Static",
    "Turbo",
    "Velvet",
]

_PLAY_NAME_SUFFIXES = [
    "Arc",
    "Blade",
    "Burst",
    "Cut",
    "Dive",
    "Elbow",
    "Flare",
    "Fork",
    "Ghost",
    "Hammer",
    "Horn",
    "Lift",
    "Loop",
    "Needle",
    "Pin",
    "Ripple",
    "Screen",
    "Snap",
    "Spiral",
    "Twist",
]


def _normalize_seed_key(seed_key: Any) -> str:
    text = str(seed_key or "").strip()
    return text or "default"


def play_name_seed_key(
    *,
    run_id: str | None = None,
    unified_policy_key: str | None = None,
    unified_policy_path: str | None = None,
    run_name: str | None = None,
    fallback: str = "default",
) -> str:
    for value in (run_id, unified_policy_key, unified_policy_path, run_name):
        text = str(value or "").strip()
        if text:
            return text
    return str(fallback or "default")


def play_name_pool() -> list[str]:
    return [f"{prefix} {suffix}" for prefix in _PLAY_NAME_PREFIXES for suffix in _PLAY_NAME_SUFFIXES]


def _rng_for_seed_key(seed_key: Any) -> random.Random:
    seed_text = _normalize_seed_key(seed_key)
    seed = int.from_bytes(hashlib.sha256(seed_text.encode("utf-8")).digest()[:8], "big")
    return random.Random(seed)


def _build_unique_component_play_names(rng: random.Random, count: int) -> list[str]:
    max_unique = min(len(_PLAY_NAME_PREFIXES), len(_PLAY_NAME_SUFFIXES))
    if count <= max_unique:
        prefixes = list(_PLAY_NAME_PREFIXES)
        suffixes = list(_PLAY_NAME_SUFFIXES)
        rng.shuffle(prefixes)
        rng.shuffle(suffixes)
        return [f"{prefixes[idx]} {suffixes[idx]}" for idx in range(count)]

    # Best-effort fallback if a future config asks for more intents than we can
    # satisfy with globally unique prefixes and suffixes.
    names = _build_unique_component_play_names(rng, max_unique)
    pool = play_name_pool()
    used = set(names)
    remaining = [name for name in pool if name not in used]
    rng.shuffle(remaining)
    for idx in range(count - max_unique):
        base = remaining[idx % len(remaining)]
        cycle = idx // len(remaining)
        if cycle <= 0:
            names.append(base)
        else:
            names.append(f"{base} {cycle + 2}")
    return names


def build_play_name_mapping(seed_key: Any, num_intents: int) -> dict[int, str]:
    count = max(0, int(num_intents or 0))
    if count <= 0:
        return {}

    rng = _rng_for_seed_key(seed_key)
    names = _build_unique_component_play_names(rng, count)
    return {idx: str(name) for idx, name in enumerate(names)}


def build_model_codename(seed_key: Any) -> str:
    rng = _rng_for_seed_key(f"model::{_normalize_seed_key(seed_key)}")
    return _build_unique_component_play_names(rng, 1)[0]


def build_play_name_artifact_payload(seed_key: Any, num_intents: int) -> dict[str, Any]:
    mapping = build_play_name_mapping(seed_key, num_intents)
    return {
        "pool_version": int(PLAY_NAME_POOL_VERSION),
        "seed_key": _normalize_seed_key(seed_key),
        "num_intents": int(max(0, int(num_intents or 0))),
        "play_names": [
            {"intent_index": int(idx), "play_name": str(name)}
            for idx, name in mapping.items()
        ],
    }


def lookup_play_name(mapping: dict[Any, Any] | None, intent_index: Any) -> str | None:
    if not isinstance(mapping, dict):
        return None
    try:
        idx = int(intent_index)
    except Exception:
        return None
    for key in (idx, str(idx)):
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None
