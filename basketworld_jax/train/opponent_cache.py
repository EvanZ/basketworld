"""Bounded, game-safe opponent snapshots, independent of checkpoint IO."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from basketworld_jax.train.pfsp import opponent_id


USAGE_HALF_LIFE_UPDATES = 500.0


def validate_cache_config(interval, capacity, *, multi_possession=True, disabled=False):
    if int(interval) < 0:
        raise ValueError(
            "Opponent snapshot interval must be >= 0 (0 is legacy pooling)."
        )
    if int(interval) and (int(capacity) < 2 or not multi_possession or disabled):
        raise ValueError(
            "Opponent snapshot cache requires capacity >= 2, multi-possession games, "
            "and an enabled opponent pool."
        )


class OpponentSnapshotCache:
    """Stable slots; retirement drains games before a slot can be overwritten.

    One pending snapshot may wait outside the resident bank. New scheduled
    snapshots replace that pending snapshot if all usable slots remain pinned
    or protected. No unbounded queue or intermediate checkpoint files exist.
    """

    def __init__(
        self,
        candidates,
        *,
        capacity,
        interval,
        template,
        jax,
        jnp,
        state=None,
        pending_params=None,
    ):
        validate_cache_config(interval, capacity)
        self.capacity, self.interval = int(capacity), int(interval)
        self.candidates = candidates
        self.jax, self.jnp = jax, jnp
        if len(candidates) > self.capacity:
            raise ValueError("Restored/seeded opponent pool exceeds cache capacity.")
        state = state or {}
        if state and (
            state.get("version") != 1
            or state.get("capacity") != self.capacity
            or state.get("interval") != self.interval
        ):
            raise ValueError(
                "Opponent cache config differs from the resume checkpoint."
            )
        self.usage = np.asarray(state.get("usage", [0.0] * self.capacity), dtype=float)
        if self.usage.shape != (self.capacity,) or not np.all(np.isfinite(self.usage)):
            raise ValueError("Invalid restored opponent cache usage.")
        self.usage_update = int(state.get("usage_update", 0))
        self.last_assignments = np.zeros(self.capacity, dtype=np.int64)
        self.retiring = state.get("retiring")
        self.pending_info = deepcopy(state.get("pending_info"))
        self.pending_params = pending_params
        if (self.pending_info is None) != (pending_params is None):
            raise ValueError("Opponent cache pending snapshot is incomplete.")
        if self.retiring is not None and not (0 <= self.retiring < len(candidates)):
            raise ValueError("Invalid restored retiring opponent slot.")
        self.admissions = int(state.get("admissions", 0))
        self.evictions = int(state.get("evictions", 0))
        self.superseded_pending = int(state.get("superseded_pending", 0))
        self.eviction_history = deepcopy(state.get("eviction_history", []))
        self.bank_writes = int(state.get("bank_writes", 0))
        self.bank = jax.tree_util.tree_map(
            lambda leaf: jnp.zeros((self.capacity, *leaf.shape), dtype=leaf.dtype),
            template,
        )
        # Slot is dynamic: additions/replacements keep the same compiled shape.
        self._write = jax.jit(
            lambda bank, params, slot: jax.tree_util.tree_map(
                lambda array, value: array.at[slot].set(value), bank, params
            ),
            # The trainer calls advance only after all rollout consumers have
            # completed. Donation lets XLA reuse the bank storage for slot writes.
            donate_argnums=(0,),
        )
        for index, candidate in enumerate(candidates):
            self.bank = self._write(self.bank, candidate["params"], jnp.asarray(index))

    def observe_rollouts(self, trajectories, *, update):
        age = max(0, int(update) - self.usage_update)
        self.usage *= 2.0 ** (-age / USAGE_HALF_LIFE_UPDATES)
        self.last_assignments[:] = 0
        for trace in trajectories.values():
            assignment = np.asarray(trace.opponent_assignment, dtype=np.int32)
            selected = (
                (np.asarray(trace.active_mask) > 0)
                & (np.asarray(trace.episode_start) > 0)
                & (assignment >= 0)
                & (assignment < len(self.candidates))
            )
            counts = np.bincount(assignment[selected], minlength=self.capacity)
            self.usage += counts
            self.last_assignments += counts
        self.usage_update = int(update)

    def usage_metrics(self):
        total = int(self.last_assignments.sum())
        metrics = {
            "opponent_cache/usage_update": self.usage_update,
            "opponent_cache/assigned_games_latest": total,
        }
        for slot, candidate in enumerate(self.candidates):
            info = candidate["info"]
            identity = opponent_id(info)
            for field, value in {
                "usage_percent": (
                    100.0 * self.last_assignments[slot] / total if total else 0.0
                ),
                "assigned_games_latest": int(self.last_assignments[slot]),
                "checkpoint_update": int(info.get("update_index", 0)),
                "snapshot_update": self.usage_update,
            }.items():
                metrics[f"opponent_cache/{field}/{identity}"] = value
        return metrics

    def _victim(self, update):
        ordered = sorted(
            range(len(self.candidates)),
            key=lambda i: (
                int(self.candidates[i]["info"].get("cache_added_update", 0)),
                i,
            ),
        )
        recent_count = min(2, max(1, self.capacity // 4))
        protected = set(ordered[-recent_count:])
        historical = ordered[:-recent_count]
        anchors = min(3, max(0, (self.capacity - 2) // 8), len(historical))
        if anchors:
            protected.update(
                historical[int(i)] for i in np.linspace(0, len(historical) - 1, anchors)
            )
        eligible = [
            i
            for i in ordered
            if i not in protected
            and update
            - int(
                self.candidates[i]["info"].get(
                    "cache_admitted_update",
                    self.candidates[i]["info"].get("cache_added_update", 0),
                )
            )
            >= 2 * self.interval
        ]
        return (
            min(eligible, key=lambda i: (self.usage[i], ordered.index(i)))
            if eligible
            else None
        )

    def advance(self, *, update, assignments, snapshot_params=None):
        """Call after the update; return an admission event, if any."""
        if snapshot_params is not None:
            if int(update) != 1 and int(update) % self.interval:
                raise ValueError(
                    "Opponent snapshots must follow the configured cadence."
                )
            if self.pending_info is not None:
                self.superseded_pending += 1
            self.pending_params = snapshot_params
            self.pending_info = {
                "source": "in_memory_snapshot",
                "candidate_kind": "self_snapshot",
                "pfsp_id": f"self_u{int(update)}",
                "update_index": int(update),
                "opponent_pool_eligible": True,
                "cache_added_update": int(update),
            }
        if self.pending_info is None:
            return None
        evicted = None
        if len(self.candidates) < self.capacity:
            slot = len(self.candidates)
        else:
            if self.retiring is None:
                self.retiring = self._victim(update)
                if self.retiring is None:
                    return None
                self.candidates[self.retiring]["info"]["cache_retiring"] = True
            slot = self.retiring
            if any(
                np.any(np.asarray(values) == slot) for values in assignments.values()
            ):
                return None
            evicted = deepcopy(self.candidates[slot]["info"])
        info = {**self.pending_info, "cache_admitted_update": int(update)}
        candidate = {"params": self.pending_params, "info": info}
        self.bank = self._write(self.bank, self.pending_params, self.jnp.asarray(slot))
        self.bank_writes += 1
        self.usage[slot] = 0.0
        if evicted is None:
            self.candidates.append(candidate)
        else:
            self.candidates[slot] = candidate
            self.evictions += 1
            self.eviction_history.append(
                {
                    "update": int(update),
                    "slot": slot,
                    "removed": evicted,
                    "added": deepcopy(info),
                }
            )
        self.admissions += 1
        self.pending_info, self.pending_params, self.retiring = None, None, None
        return {
            "update": int(update),
            "slot": slot,
            "snapshot": info["update_index"],
            "evicted": None if evicted is None else evicted.get("pfsp_id"),
        }

    def probabilities(self, compact):
        """Two mode rows with exact zeros for unused and draining slots."""
        compact = np.asarray(compact, dtype=np.float32)
        if compact.ndim == 1:
            compact = np.stack((compact, compact))
        result = np.zeros((2, self.capacity), dtype=np.float32)
        result[:, : len(self.candidates)] = compact
        if self.retiring is not None:
            result[:, self.retiring] = 0.0
        total = result.sum(axis=1, keepdims=True)
        return np.divide(result, total, out=result, where=total > 0)

    def compact_params(self):
        return self.jax.tree_util.tree_map(
            lambda leaf: leaf[: len(self.candidates)], self.bank
        )

    def payload(self):
        return {
            "version": 1,
            "capacity": self.capacity,
            "interval": self.interval,
            "usage": self.usage.tolist(),
            "usage_update": self.usage_update,
            "retiring": self.retiring,
            "pending_info": deepcopy(self.pending_info),
            "admissions": self.admissions,
            "evictions": self.evictions,
            "bank_writes": self.bank_writes,
            "superseded_pending": self.superseded_pending,
            "eviction_history": deepcopy(self.eviction_history),
        }

    def metrics(self, update):
        newest = max(
            (int(c["info"].get("update_index", 0)) for c in self.candidates), default=-1
        )
        return {
            "opponent_cache/" + name: value
            for name, value in {
                "snapshot_update": int(update),
                "capacity": self.capacity,
                "size": len(self.candidates),
                "interval_updates": self.interval,
                "newest_snapshot_update": newest,
                "pending_snapshot_update": (
                    -1
                    if self.pending_info is None
                    else self.pending_info["update_index"]
                ),
                "retiring_count": int(self.retiring is not None),
                "admissions": self.admissions,
                "evictions": self.evictions,
                "bank_writes": self.bank_writes,
                "superseded_pending": self.superseded_pending,
                "bank_bytes": sum(
                    leaf.nbytes for leaf in self.jax.tree_util.tree_leaves(self.bank)
                ),
            }.items()
        }
