from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


OUTCOME_CATEGORIES = (
    "made_2pt",
    "made_3pt",
    "made_dunk",
    "missed_2pt",
    "missed_3pt",
    "missed_dunk",
    "turnover",
    "unknown",
)

SUMMARY_FEATURE_NAMES = (
    "episode_length",
    "active_prefix_length",
    "pass_attempts",
    "pass_completions",
    "pass_intercepts",
    "pass_oob",
    "assist_potential",
    "assist_full",
    "points",
    "team_reward_offense",
    "outcome_made_2pt",
    "outcome_made_3pt",
    "outcome_made_dunk",
    "outcome_missed_2pt",
    "outcome_missed_3pt",
    "outcome_missed_dunk",
    "outcome_turnover",
    "outcome_unknown",
)


def infer_outcome_from_episode_info(ep_info: Dict) -> Tuple[str, str, float]:
    """Infer human-readable outcome, coarse shot type, and points from episode info."""
    made_dunk = float(ep_info.get("made_dunk", 0.0))
    shot_dunk = float(ep_info.get("shot_dunk", 0.0))
    made_2pt = float(ep_info.get("made_2pt", 0.0))
    shot_2pt = float(ep_info.get("shot_2pt", 0.0))
    made_3pt = float(ep_info.get("made_3pt", 0.0))
    shot_3pt = float(ep_info.get("shot_3pt", 0.0))

    if shot_dunk > 0.0:
        return ("Made Dunk" if made_dunk > 0.0 else "Missed Dunk", "dunk", 2.0 if made_dunk > 0.0 else 0.0)
    if shot_3pt > 0.0:
        return ("Made 3pt" if made_3pt > 0.0 else "Missed 3pt", "3pt", 3.0 if made_3pt > 0.0 else 0.0)
    if shot_2pt > 0.0:
        return ("Made 2pt" if made_2pt > 0.0 else "Missed 2pt", "2pt", 2.0 if made_2pt > 0.0 else 0.0)

    turnover = float(ep_info.get("turnover", 0.0))
    if turnover > 0.0:
        if float(ep_info.get("turnover_intercepted", 0.0)) > 0.0:
            return ("Turnover (Intercepted)", "turnover", 0.0)
        if float(ep_info.get("turnover_pass_oob", 0.0)) > 0.0:
            return ("Turnover (OOB - Pass)", "turnover", 0.0)
        if float(ep_info.get("turnover_move_oob", 0.0)) > 0.0:
            return ("Turnover (OOB - Move)", "turnover", 0.0)
        if float(ep_info.get("turnover_pressure", 0.0)) > 0.0:
            return ("Turnover (Pressure)", "turnover", 0.0)
        return ("Turnover (Other)", "turnover", 0.0)

    return ("Unknown", "unknown", 0.0)


def outcome_category_key(outcome: str) -> str:
    outcome_norm = str(outcome).strip().lower()
    if outcome_norm == "made 2pt":
        return "made_2pt"
    if outcome_norm == "made 3pt":
        return "made_3pt"
    if outcome_norm == "made dunk":
        return "made_dunk"
    if outcome_norm == "missed 2pt":
        return "missed_2pt"
    if outcome_norm == "missed 3pt":
        return "missed_3pt"
    if outcome_norm == "missed dunk":
        return "missed_dunk"
    if outcome_norm.startswith("turnover"):
        return "turnover"
    return "unknown"


def build_summary_feature(metadata: Dict) -> np.ndarray:
    """Build a compact interpretable episode-level feature vector for PCA."""
    outcome_key = outcome_category_key(str(metadata.get("outcome", "Unknown")))
    out_one_hot = np.zeros(len(OUTCOME_CATEGORIES), dtype=np.float32)
    try:
        out_idx = OUTCOME_CATEGORIES.index(outcome_key)
    except ValueError:
        out_idx = OUTCOME_CATEGORIES.index("unknown")
    out_one_hot[out_idx] = 1.0

    dense = np.array(
        [
            float(metadata.get("episode_length", 0.0)),
            float(metadata.get("active_prefix_length", 0.0)),
            float(metadata.get("pass_attempts", 0.0)),
            float(metadata.get("pass_completions", 0.0)),
            float(metadata.get("pass_intercepts", 0.0)),
            float(metadata.get("pass_oob", 0.0)),
            float(metadata.get("assist_potential", 0.0)),
            float(metadata.get("assist_full", 0.0)),
            float(metadata.get("points", 0.0)),
            float(metadata.get("team_reward_offense", 0.0)),
        ],
        dtype=np.float32,
    )
    return np.concatenate([dense, out_one_hot]).astype(np.float32, copy=False)

