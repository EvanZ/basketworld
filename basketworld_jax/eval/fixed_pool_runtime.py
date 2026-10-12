"""Host-side preparation of fixed-player-pool games for the JAX evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from basketworld_jax.eval.fixed_pool import PlayerPool
from basketworld_jax.eval.lineup_schedule import (
    LineupSchedule,
    ScheduleCompatibilityError,
    ScheduledGame,
    validate_lineup_schedule,
)
from basketworld_jax.eval.skills import ACTIVE_PLAYER_SKILL_REGISTRY


class FixedPoolRuntimeError(ValueError):
    """A fixed-pool schedule cannot be mapped onto an evaluation runtime."""


@dataclass(frozen=True)
class FixedPoolGameBatch:
    """Per-row JAX inputs and trace metadata for scheduled fixed-pool games."""

    games: tuple[ScheduledGame, ...]
    player_skill_rows: Mapping[str, np.ndarray]
    starting_team_codes: np.ndarray
    reset_rng_roots: np.ndarray
    environment_rng_roots: np.ndarray
    policy_rng_roots: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "player_skill_rows",
            MappingProxyType(
                {
                    str(key): np.asarray(values, dtype=np.float32).copy()
                    for key, values in self.player_skill_rows.items()
                }
            ),
        )
        for field in (
            "starting_team_codes",
            "reset_rng_roots",
            "environment_rng_roots",
            "policy_rng_roots",
        ):
            values = np.asarray(getattr(self, field)).copy()
            values.setflags(write=False)
            object.__setattr__(self, field, values)
        for values in self.player_skill_rows.values():
            values.setflags(write=False)

    def trace_metadata(self, row: int) -> dict[str, object]:
        """Return the immutable provenance needed to join an episode to a study."""
        game = self.games[int(row)]
        return {
            "schedule_game_id": game.game_id,
            "pair_id": game.pair_id,
            "starting_team": game.starting_team,
            "team_a_player_ids": list(game.team_a_player_ids),
            "team_b_player_ids": list(game.team_b_player_ids),
            "reset_rng_root": int(game.reset_rng_root),
            "environment_rng_root": int(game.environment_rng_root),
            "policy_rng_root": int(game.policy_rng_root),
        }


def build_fixed_pool_game_batch(
    pool: PlayerPool,
    schedule: LineupSchedule,
    games: Sequence[ScheduledGame],
    *,
    team_a_slots: Sequence[int],
    team_b_slots: Sequence[int],
    total_slots: int,
) -> FixedPoolGameBatch:
    """Map scheduled global player IDs to the existing stable JAX slot layout."""
    try:
        validate_lineup_schedule(schedule, pool=pool)
    except ScheduleCompatibilityError as exc:
        raise FixedPoolRuntimeError(str(exc)) from exc
    selected_games = tuple(games)
    if not selected_games:
        raise FixedPoolRuntimeError("A fixed-pool evaluation batch cannot be empty.")
    expected_game_ids = {game.game_id for game in schedule.games}
    if any(game.game_id not in expected_game_ids for game in selected_games):
        raise FixedPoolRuntimeError(
            "Fixed-pool batch contains a game outside its schedule."
        )
    team_a_slots = tuple(int(slot) for slot in team_a_slots)
    team_b_slots = tuple(int(slot) for slot in team_b_slots)
    expected_team_size = int(schedule.team_size)
    if (
        len(team_a_slots) != expected_team_size
        or len(team_b_slots) != expected_team_size
    ):
        raise FixedPoolRuntimeError(
            "Fixed-pool schedules require exactly five stable JAX slots per team."
        )
    all_slots = (*team_a_slots, *team_b_slots)
    if (
        len(set(all_slots)) != len(all_slots)
        or min(all_slots, default=0) < 0
        or max(all_slots, default=-1) >= int(total_slots)
    ):
        raise FixedPoolRuntimeError(
            "Fixed-pool JAX team slots must be unique and in range."
        )
    if int(total_slots) != expected_team_size * 2:
        raise FixedPoolRuntimeError(
            "Fixed-pool evaluation currently supports exactly ten JAX player slots."
        )

    players = {player.player_id: player for player in pool.players}
    skill_rows = {
        descriptor.key: np.empty(
            (len(selected_games), int(total_slots)), dtype=np.float32
        )
        for descriptor in ACTIVE_PLAYER_SKILL_REGISTRY.descriptors
    }
    starting_team_codes = np.empty((len(selected_games),), dtype=np.int8)
    reset_rng_roots = np.empty((len(selected_games),), dtype=np.uint32)
    environment_rng_roots = np.empty((len(selected_games),), dtype=np.uint32)
    policy_rng_roots = np.empty((len(selected_games),), dtype=np.uint32)

    for row, game in enumerate(selected_games):
        for slots, player_ids in (
            (team_a_slots, game.team_a_player_ids),
            (team_b_slots, game.team_b_player_ids),
        ):
            if len(player_ids) != expected_team_size:
                raise FixedPoolRuntimeError(
                    f"Scheduled game {game.game_id!r} has an invalid team size."
                )
            for slot, player_id in zip(slots, player_ids, strict=True):
                player = players.get(player_id)
                if player is None:
                    raise FixedPoolRuntimeError(
                        f"Scheduled game {game.game_id!r} references unknown player "
                        f"{player_id!r}."
                    )
                for descriptor in ACTIVE_PLAYER_SKILL_REGISTRY.descriptors:
                    skill_rows[descriptor.key][row, slot] = np.float32(
                        player.skills[descriptor.key]
                    )
        if game.starting_team == "team_a":
            starting_team_codes[row] = 0
        elif game.starting_team == "team_b":
            starting_team_codes[row] = 1
        else:
            raise FixedPoolRuntimeError(
                f"Scheduled game {game.game_id!r} has an invalid starting team."
            )
        reset_rng_roots[row] = np.uint32(game.reset_rng_root)
        environment_rng_roots[row] = np.uint32(game.environment_rng_root)
        policy_rng_roots[row] = np.uint32(game.policy_rng_root)

    return FixedPoolGameBatch(
        games=selected_games,
        player_skill_rows=skill_rows,
        starting_team_codes=starting_team_codes,
        reset_rng_roots=reset_rng_roots,
        environment_rng_roots=environment_rng_roots,
        policy_rng_roots=policy_rng_roots,
    )
