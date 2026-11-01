"""Host-side, checkpointable prioritized fictitious self-play estimates.

Evidence has a 500-update half-life, a 256-game effective-count cap and a
Beta(1, 1) neutral prior. Lifetime counts/freshness are reported separately.
The constants are estimator implementation details, not extra tuning knobs.
"""

from __future__ import annotations

import copy
import hashlib
import json

import numpy as np

MODES = ("sampled", "deterministic")
HALF_LIFE_UPDATES = 500.0
MAX_EFFECTIVE_GAMES = 256.0


def probabilities(win_scores, alpha: float, epsilon: float) -> np.ndarray:
    if not all(np.isfinite(x) and 0 <= x <= 1 for x in (alpha, epsilon)):
        raise ValueError("PFSP alpha and epsilon must be finite and in [0, 1]")
    w = np.asarray(win_scores, dtype=np.float64)
    if w.ndim != 1 or not np.all(np.isfinite(w) & (w >= 0) & (w <= 1)):
        raise ValueError("PFSP win scores must be a finite vector in [0, 1]")
    if not w.size:
        return np.zeros(0, dtype=np.float32)
    uniform = np.full(w.size, 1 / w.size)

    def normalize(weights):
        return weights / weights.sum() if weights.sum() > 0 else uniform

    blend = (1 - alpha) * normalize(w * (1 - w)) + alpha * normalize((1 - w) ** 2)
    return ((1 - epsilon) * blend + epsilon * uniform).astype(np.float32)


def opponent_id(info) -> str:
    if info.get("pfsp_id"):
        return str(info["pfsp_id"])
    update = int(info.get("update_index", 0))
    if (
        info.get("candidate_kind") == "self_checkpoint"
        or info.get("source") == "self_checkpoint"
    ):
        return f"self_u{update}"
    identity = {
        k: info.get(k)
        for k in (
            "source",
            "run_id",
            "artifact_path",
            "checkpoint_path",
            "update_index",
        )
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[
        :12
    ]
    return f"external_{digest}_u{update}"


class PFSPState:
    def __init__(self, payload=None):
        payload = payload or {}
        if payload and payload.get("version") != 1:
            raise ValueError("Unsupported PFSP estimator state version")
        self.entries = copy.deepcopy(payload.get("entries", {}))

    def payload(self):
        return {"version": 1, "entries": copy.deepcopy(self.entries)}

    def _entry(self, identity, mode):
        if mode not in MODES:
            raise ValueError(f"Unknown opponent action mode: {mode}")
        return self.entries.setdefault(identity, {}).setdefault(
            mode,
            {
                "score_sum": 0.0,
                "effective_games": 0.0,
                "evidence_update": 0,
                "completed_games": 0,
                "truncated_games": 0,
                "last_observed_update": None,
                "offense_start_games": 0,
                "defense_start_games": 0,
                "assigned_games": 0,
                "active_steps": 0,
            },
        )

    @staticmethod
    def _aged(entry, update):
        age = max(0, int(update) - int(entry["evidence_update"]))
        factor = 2.0 ** (-age / HALF_LIFE_UPDATES)
        return entry["score_sum"] * factor, entry["effective_games"] * factor

    def observe(
        self, identity, mode, scores, *, update, role, truncated=0, role_counts=None
    ):
        scores = np.asarray(scores, dtype=np.float64)
        if role not in ("offense", "defense") or not np.all(
            np.isin(scores, [0.0, 0.5, 1.0])
        ):
            raise ValueError("Invalid PFSP game outcomes or starting side")
        self._observe_totals(
            identity,
            mode,
            float(scores.sum()),
            int(scores.size),
            update=update,
            truncated=truncated,
            role_counts=role_counts or {role: int(scores.size)},
        )

    def _observe_totals(
        self, identity, mode, score_sum, games, *, update, truncated, role_counts
    ):
        entry = self._entry(identity, mode)
        entry["truncated_games"] += int(truncated)
        if not games:
            return
        total, count = self._aged(entry, update)
        total += score_sum
        count += games
        scale = min(1.0, MAX_EFFECTIVE_GAMES / count)
        entry.update(
            score_sum=total * scale,
            effective_games=count * scale,
            evidence_update=int(update),
            last_observed_update=int(update),
        )
        entry["completed_games"] += int(games)
        for side, role_games in role_counts.items():
            entry[f"{side}_start_games"] += int(role_games)

    def sampling_snapshot(self, candidates, *, update, alpha, epsilon):
        """Snapshot BEFORE the rollout: logged w equals the sampling input."""
        metrics, distributions = {}, []
        for mode in MODES:
            rows = []
            for candidate in candidates:
                identity = opponent_id(candidate["info"])
                entry = self._entry(identity, mode)
                score, count = self._aged(entry, update)
                rows.append(
                    (identity, entry, (score + 1.0) / (count + 2.0), count, candidate)
                )
            eligible = np.asarray(
                [not row[4]["info"].get("cache_retiring", False) for row in rows],
                dtype=bool,
            )
            probs = np.zeros(len(rows), dtype=np.float32)
            probs[eligible] = probabilities(
                [row[2] for row, keep in zip(rows, eligible) if keep], alpha, epsilon
            )
            distributions.append(probs)
            for (identity, entry, w, count, candidate), probability in zip(
                rows, probs, strict=True
            ):
                values = {
                    "snapshot_update": int(update),
                    "retiring": float(candidate["info"].get("cache_retiring", False)),
                    "w": w,
                    "probability": float(probability),
                    "effective_games": count,
                    "completed_games": entry["completed_games"],
                    "truncated_games": entry["truncated_games"],
                    "prior_only": float(entry["last_observed_update"] is None),
                    "last_observed_update": (
                        entry["last_observed_update"]
                        if entry["last_observed_update"] is not None
                        else -1
                    ),
                    "age_updates": (
                        update - entry["last_observed_update"]
                        if entry["last_observed_update"] is not None
                        else -1
                    ),
                    "checkpoint_update": int(candidate["info"].get("update_index", 0)),
                    "offense_start_games": entry["offense_start_games"],
                    "defense_start_games": entry["defense_start_games"],
                    "assigned_games": entry.get("assigned_games", 0),
                    "active_steps": entry.get("active_steps", 0),
                }
                metrics.update(
                    {
                        f"pfsp/{name}/{identity}/{mode}": float(value)
                        for name, value in values.items()
                    }
                )
        return np.asarray(distributions, dtype=np.float32), metrics

    def observe_rollout(self, trajectory, candidates, *, update, role):
        self.observe_rollouts({role: trajectory}, candidates, update=update)

    def observe_rollouts(self, trajectories, candidates, *, update):
        """Aggregate once per rollout, not one full-array scan per opponent."""
        count = len(candidates)
        if not count:
            return
        batches = {}
        for role, trajectory in trajectories.items():
            active = np.asarray(trajectory.active_mask) > 0
            assignments = np.asarray(trajectory.opponent_assignment, dtype=np.int32)
            modes = (np.asarray(trajectory.opponent_deterministic_episode) > 0).astype(
                np.int32
            )
            valid = active & (assignments >= 0) & (assignments < count)
            index = assignments * 2 + modes
            ended = valid & (np.asarray(trajectory.dones) > 0)
            truncated = np.asarray(trajectory.truncated) > 0
            completed = ended & ~truncated
            scores = np.asarray(trajectory.terminal_learner_win_score)
            if not np.all(np.isin(scores[completed], [0.0, 0.5, 1.0])):
                raise ValueError("Invalid PFSP game outcomes")
            starts = (
                np.asarray(getattr(trajectory, "episode_start", np.zeros_like(active)))
                > 0
            )

            def aggregate(mask, weights=None):
                return np.bincount(
                    index[mask], weights=weights, minlength=count * 2
                ).reshape(count, 2)

            batches[role] = {
                "games": aggregate(completed),
                "scores": aggregate(completed, scores[completed]),
                "truncated": aggregate(ended & truncated),
                "assigned": aggregate(valid & starts),
                "active": aggregate(valid),
            }
        for index, candidate in enumerate(candidates):
            for deterministic, mode in enumerate(MODES):
                entry = self._entry(opponent_id(candidate["info"]), mode)

                def total(name):
                    return sum(
                        batch[name][index, deterministic] for batch in batches.values()
                    )

                entry["assigned_games"] = entry.get("assigned_games", 0) + int(
                    total("assigned")
                )
                entry["active_steps"] = entry.get("active_steps", 0) + int(
                    total("active")
                )
                # Cap evidence once per update, not once per starting side:
                # batched simultaneous games must not favor the second side.
                self._observe_totals(
                    opponent_id(candidate["info"]),
                    mode,
                    total("scores"),
                    int(total("games")),
                    update=update,
                    truncated=int(total("truncated")),
                    role_counts={
                        role: int(batch["games"][index, deterministic])
                        for role, batch in batches.items()
                    },
                )
