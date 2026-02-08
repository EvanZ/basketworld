from __future__ import annotations

from enum import Enum
from typing import List, Optional

import numpy as np


class IllegalActionStrategy(Enum):
    NOOP = "noop"
    BEST_PROB = "best_prob"
    SAMPLE_PROB = "sample_prob"


def get_policy_action_probabilities(policy, obs) -> Optional[List[np.ndarray]]:
    """Return per-player action probabilities for a Stable-Baselines3 policy.

    Expects a PPO/ActorCritic policy with .policy.get_distribution. Returns a list
    of numpy arrays (one per player) or None if retrieval fails.
    """
    policy_obj = None
    try:
        policy_obj = getattr(policy, "policy", None)
        if policy_obj is None:
            return None
        if hasattr(policy_obj, "_extract_role_flag") and hasattr(policy_obj, "_current_role_flags"):
            try:
                policy_obj._current_role_flags = policy_obj._extract_role_flag(obs)
            except Exception:
                policy_obj._current_role_flags = None
        obs_tensor = policy_obj.obs_to_tensor(obs)[0]
        distributions = policy_obj.get_distribution(obs_tensor)
        return [
            dist.probs.detach().cpu().numpy().squeeze()
            for dist in distributions.distribution
        ]
    except Exception:
        return None
    finally:
        if policy_obj is not None and hasattr(policy_obj, "_current_role_flags"):
            policy_obj._current_role_flags = None


def _choose_legal(
    legal_indices: np.ndarray,
    probs_vec: Optional[np.ndarray],
    strategy: IllegalActionStrategy,
    deterministic: bool,
) -> int:
    if len(legal_indices) == 0:
        return 0

    if strategy == IllegalActionStrategy.NOOP:
        # Strict NOOP semantics: map to NOOP (0) when illegal
        return 0

    if probs_vec is None or int(np.max(legal_indices)) >= probs_vec.shape[0]:
        # No probabilities available; fall back to a sensible deterministic choice
        non_noop = [int(i) for i in legal_indices if i != 0]
        return int(non_noop[0]) if len(non_noop) > 0 else int(legal_indices[0])

    masked = probs_vec[legal_indices]
    total = float(np.sum(masked))
    if total <= 0.0:
        # Degenerate distribution; same fallback
        non_noop = [int(i) for i in legal_indices if i != 0]
        return int(non_noop[0]) if len(non_noop) > 0 else int(legal_indices[0])

    if strategy == IllegalActionStrategy.BEST_PROB:
        return int(legal_indices[int(np.argmax(masked))])

    # SAMPLE_PROB
    normed = masked / total
    if deterministic:
        return int(legal_indices[int(np.argmax(normed))])
    return int(np.random.choice(legal_indices, p=normed))


def resolve_illegal_actions(
    predicted_actions: np.ndarray,
    action_mask: np.ndarray,
    strategy: IllegalActionStrategy,
    deterministic: bool,
    probs_per_player: Optional[List[np.ndarray]] = None,
) -> np.ndarray:
    """Return an action array where illegal actions have been replaced per strategy.

    - predicted_actions: shape (n_players,)
    - action_mask: shape (n_players, n_actions) with 1 for legal
    - probs_per_player: list of arrays (n_actions,) for each player (optional)
    """
    n_players = action_mask.shape[0]
    out = np.array(predicted_actions, dtype=int)

    for pid in range(n_players):
        a = int(out[pid])
        if a < 0 or a >= action_mask.shape[1] or action_mask[pid][a] == 0:
            legal = np.where(action_mask[pid] == 1)[0]
            pvec = (
                probs_per_player[pid]
                if probs_per_player is not None and pid < len(probs_per_player)
                else None
            )
            out[pid] = _choose_legal(legal, pvec, strategy, deterministic)

    return out
