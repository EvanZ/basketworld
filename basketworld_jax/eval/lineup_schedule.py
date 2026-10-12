"""Deterministic, immutable 5v5 lineup schedules for fixed-player studies."""

from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from basketworld_jax.eval.fixed_pool import (
    PlayerPool,
    PlayerPoolError,
    assert_pool_compatible,
)


LINEUP_SCHEDULE_SCHEMA_VERSION = 3
_LEGACY_LINEUP_SCHEDULE_SCHEMA_VERSION = 2
_SUPPORTED_LINEUP_SCHEDULE_SCHEMA_VERSIONS = frozenset(
    {_LEGACY_LINEUP_SCHEDULE_SCHEMA_VERSION, LINEUP_SCHEDULE_SCHEMA_VERSION}
)
_TEAM_A = "team_a"
_TEAM_B = "team_b"
_STARTING_TEAMS = (_TEAM_A, _TEAM_B)
_LEGACY_OPTIMIZER_VERSION = "balanced-greedy-v1"
_OPTIMIZER_VERSION = "balanced-greedy-v2-independent-games"


class LineupScheduleError(ValueError):
    """Base error for fixed-pool lineup schedules."""


class ScheduleCompatibilityError(LineupScheduleError):
    """A lineup schedule cannot safely be replayed against a requested pool."""


def _json_value(value: Any) -> Any:
    """Return a defensive JSON-compatible copy without dropping future fields."""
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return copy.deepcopy(value)


def _freeze_json(value: Any) -> Any:
    """Recursively freeze schedule data while retaining JSON serialization."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return copy.deepcopy(value)


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        _json_value(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def schedule_id_for_payload(payload: Mapping[str, Any]) -> str:
    """Return the content-addressed identity for a schedule payload."""
    content = {key: value for key, value in payload.items() if key != "schedule_id"}
    try:
        schema_version = int(content["schema_version"])
    except (KeyError, TypeError, ValueError) as exc:
        raise LineupScheduleError("Schedule identity needs a schema_version.") from exc
    digest = hashlib.sha256(_canonical_json(content).encode("utf-8")).hexdigest()
    return f"lineup-schedule-v{schema_version}-{digest[:20]}"


def _pair_key(first: str, second: str) -> tuple[str, str]:
    return tuple(sorted((first, second)))


@dataclass(frozen=True)
class ScheduledGame:
    """One replayable game and its immutable analysis-resampling group.

    ``pair_id`` is retained as the serialized field name so schema-v2 study
    artifacts remain readable.  In schema v3 it is a one-game sampling-group
    identifier, not evidence of a paired start.
    """

    game_id: str
    pair_id: str
    starting_team: str
    team_a_player_ids: tuple[str, ...]
    team_b_player_ids: tuple[str, ...]
    reset_rng_root: int
    environment_rng_root: int
    policy_rng_root: int
    extra_fields: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "team_a_player_ids",
            tuple(str(player_id) for player_id in self.team_a_player_ids),
        )
        object.__setattr__(
            self,
            "team_b_player_ids",
            tuple(str(player_id) for player_id in self.team_b_player_ids),
        )
        object.__setattr__(self, "extra_fields", _freeze_json(self.extra_fields or {}))

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "game_id": self.game_id,
            "pair_id": self.pair_id,
            "starting_team": self.starting_team,
            "team_a_player_ids": list(self.team_a_player_ids),
            "team_b_player_ids": list(self.team_b_player_ids),
            "reset_rng_root": int(self.reset_rng_root),
            "environment_rng_root": int(self.environment_rng_root),
            "policy_rng_root": int(self.policy_rng_root),
        }
        if self.extra_fields:
            payload.update(_json_value(self.extra_fields))
        return payload

    @property
    def sampling_group_id(self) -> str:
        """Return the generic CV/bootstrap group identifier."""
        return self.pair_id


@dataclass(frozen=True)
class LineupSchedule:
    """An immutable, self-validating schedule bound to one player-pool artifact."""

    schema_version: int
    schedule_id: str
    pool_id: str
    pool_skill_schema_hash: str
    pool_player_ids: tuple[str, ...]
    schedule_seed: int
    team_size: int
    optimizer_version: str
    candidate_count: int
    games: tuple[ScheduledGame, ...]
    diagnostics: Mapping[str, Any]
    extra_fields: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "pool_player_ids",
            tuple(str(player_id) for player_id in self.pool_player_ids),
        )
        object.__setattr__(self, "diagnostics", _freeze_json(self.diagnostics))
        object.__setattr__(self, "extra_fields", _freeze_json(self.extra_fields or {}))

    @property
    def matchup_count(self) -> int:
        """Legacy name for the number of immutable sampling groups."""
        return len({game.pair_id for game in self.games})

    @property
    def sampling_group_count(self) -> int:
        return len({game.sampling_group_id for game in self.games})

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": int(self.schema_version),
            "schedule_id": self.schedule_id,
            "pool_id": self.pool_id,
            "pool_skill_schema_hash": self.pool_skill_schema_hash,
            "pool_player_ids": list(self.pool_player_ids),
            "schedule_seed": int(self.schedule_seed),
            "team_size": int(self.team_size),
            "optimizer_version": self.optimizer_version,
            "candidate_count": int(self.candidate_count),
            "games": [game.to_payload() for game in self.games],
            "diagnostics": _json_value(self.diagnostics),
        }
        if self.extra_fields:
            payload.update(_json_value(self.extra_fields))
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "LineupSchedule":
        if not isinstance(payload, Mapping):
            raise LineupScheduleError("Lineup-schedule artifact must be a JSON object.")
        try:
            schema_version = int(payload["schema_version"])
            schedule_id = str(payload["schedule_id"])
            pool_id = str(payload["pool_id"])
            pool_skill_schema_hash = str(payload["pool_skill_schema_hash"])
            schedule_seed = int(payload["schedule_seed"])
            team_size = int(payload["team_size"])
            optimizer_version = str(payload["optimizer_version"])
            candidate_count = int(payload["candidate_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise LineupScheduleError(
                "Lineup-schedule artifact is missing required metadata."
            ) from exc
        if schema_version not in _SUPPORTED_LINEUP_SCHEDULE_SCHEMA_VERSIONS:
            raise LineupScheduleError(
                f"Unsupported lineup-schedule schema version {schema_version}; "
                f"expected one of {sorted(_SUPPORTED_LINEUP_SCHEDULE_SCHEMA_VERSIONS)}."
            )
        raw_player_ids = payload.get("pool_player_ids")
        if not isinstance(raw_player_ids, list) or not raw_player_ids:
            raise LineupScheduleError(
                "Lineup-schedule artifact must contain pool_player_ids."
            )
        raw_games = payload.get("games")
        if not isinstance(raw_games, list) or not raw_games:
            raise LineupScheduleError("Lineup-schedule artifact must contain games.")
        diagnostics = payload.get("diagnostics")
        if not isinstance(diagnostics, Mapping):
            raise LineupScheduleError(
                "Lineup-schedule artifact must contain diagnostics."
            )
        known_fields = {
            "schema_version",
            "schedule_id",
            "pool_id",
            "pool_skill_schema_hash",
            "pool_player_ids",
            "schedule_seed",
            "team_size",
            "optimizer_version",
            "candidate_count",
            "games",
            "diagnostics",
        }
        schedule = cls(
            schema_version=schema_version,
            schedule_id=schedule_id,
            pool_id=pool_id,
            pool_skill_schema_hash=pool_skill_schema_hash,
            pool_player_ids=tuple(str(player_id) for player_id in raw_player_ids),
            schedule_seed=schedule_seed,
            team_size=team_size,
            optimizer_version=optimizer_version,
            candidate_count=candidate_count,
            games=tuple(_game_from_payload(game) for game in raw_games),
            diagnostics=_json_value(diagnostics),
            extra_fields={
                str(key): _json_value(value)
                for key, value in payload.items()
                if key not in known_fields
            },
        )
        expected_id = schedule_id_for_payload(schedule.to_payload())
        if schedule.schedule_id != expected_id:
            raise LineupScheduleError(
                f"Schedule ID {schedule.schedule_id!r} does not match artifact content; "
                f"expected {expected_id!r}."
            )
        validate_lineup_schedule(schedule)
        return schedule


@dataclass(frozen=True)
class LineupScheduleExtension:
    """Immutable schedules for evaluating only an appended game segment.

    ``additional_schedule`` is dispatched to the evaluator, while
    ``combined_schedule`` records the old and new games as one new immutable
    study.  This avoids replaying evidence that is already on disk.
    """

    additional_schedule: LineupSchedule
    combined_schedule: LineupSchedule


def _game_from_payload(payload: Any) -> ScheduledGame:
    if not isinstance(payload, Mapping):
        raise LineupScheduleError("Every scheduled game must be a JSON object.")
    try:
        game_id = str(payload["game_id"])
        pair_id = str(payload["pair_id"])
        starting_team = str(payload["starting_team"])
        reset_rng_root = int(payload["reset_rng_root"])
        environment_rng_root = int(payload["environment_rng_root"])
        policy_rng_root = int(payload["policy_rng_root"])
    except (KeyError, TypeError, ValueError) as exc:
        raise LineupScheduleError(
            "Every scheduled game needs IDs, a starting team, and RNG roots."
        ) from exc
    team_a_player_ids = payload.get("team_a_player_ids")
    team_b_player_ids = payload.get("team_b_player_ids")
    if not isinstance(team_a_player_ids, list) or not isinstance(
        team_b_player_ids, list
    ):
        raise LineupScheduleError(
            f"Scheduled game {game_id!r} must contain both team rosters."
        )
    known_fields = {
        "game_id",
        "pair_id",
        "starting_team",
        "team_a_player_ids",
        "team_b_player_ids",
        "reset_rng_root",
        "environment_rng_root",
        "policy_rng_root",
    }
    return ScheduledGame(
        game_id=game_id,
        pair_id=pair_id,
        starting_team=starting_team,
        team_a_player_ids=tuple(str(player_id) for player_id in team_a_player_ids),
        team_b_player_ids=tuple(str(player_id) for player_id in team_b_player_ids),
        reset_rng_root=reset_rng_root,
        environment_rng_root=environment_rng_root,
        policy_rng_root=policy_rng_root,
        extra_fields={
            str(key): _json_value(value)
            for key, value in payload.items()
            if key not in known_fields
        },
    )


def _make_schedule(
    *,
    pool: PlayerPool,
    schedule_seed: int,
    team_size: int,
    candidate_count: int,
    games: Sequence[ScheduledGame],
    extra_fields: Mapping[str, Any] | None = None,
) -> LineupSchedule:
    provisional = {
        "schema_version": LINEUP_SCHEDULE_SCHEMA_VERSION,
        "pool_id": pool.pool_id,
        "pool_skill_schema_hash": pool.skill_schema_hash,
        "pool_player_ids": [player.player_id for player in pool.players],
        "schedule_seed": int(schedule_seed),
        "team_size": int(team_size),
        "optimizer_version": _OPTIMIZER_VERSION,
        "candidate_count": int(candidate_count),
        "games": [game.to_payload() for game in games],
        **(_json_value(extra_fields) if extra_fields else {}),
    }
    provisional_schedule = LineupSchedule(
        schema_version=LINEUP_SCHEDULE_SCHEMA_VERSION,
        schedule_id="",
        pool_id=pool.pool_id,
        pool_skill_schema_hash=pool.skill_schema_hash,
        pool_player_ids=tuple(player.player_id for player in pool.players),
        schedule_seed=int(schedule_seed),
        team_size=int(team_size),
        optimizer_version=_OPTIMIZER_VERSION,
        candidate_count=int(candidate_count),
        games=tuple(games),
        diagnostics={},
        extra_fields=_json_value(extra_fields or {}),
    )
    diagnostics = schedule_balance_diagnostics(provisional_schedule)
    provisional["diagnostics"] = diagnostics
    schedule_id = schedule_id_for_payload(provisional)
    schedule = LineupSchedule(
        schema_version=LINEUP_SCHEDULE_SCHEMA_VERSION,
        schedule_id=schedule_id,
        pool_id=pool.pool_id,
        pool_skill_schema_hash=pool.skill_schema_hash,
        pool_player_ids=tuple(player.player_id for player in pool.players),
        schedule_seed=int(schedule_seed),
        team_size=int(team_size),
        optimizer_version=_OPTIMIZER_VERSION,
        candidate_count=int(candidate_count),
        games=tuple(games),
        diagnostics=diagnostics,
        extra_fields=_json_value(extra_fields or {}),
    )
    validate_lineup_schedule(schedule, pool=pool)
    return schedule


def _candidate_roster(
    player_ids: tuple[str, ...],
    appearances: Mapping[str, int],
    *,
    roster_size: int,
    rng: np.random.Generator,
) -> tuple[str, ...]:
    """Choose only from the least-used players, preserving optimal appearances."""
    sorted_counts = sorted(appearances[player_id] for player_id in player_ids)
    cutoff = sorted_counts[roster_size - 1]
    mandatory = [
        player_id for player_id in player_ids if appearances[player_id] < cutoff
    ]
    tied = [player_id for player_id in player_ids if appearances[player_id] == cutoff]
    remaining = roster_size - len(mandatory)
    if remaining == len(tied):
        chosen_tied = tied
    else:
        chosen_tied = rng.choice(tied, size=remaining, replace=False).tolist()
    return tuple(sorted((*mandatory, *chosen_tied)))


def _pairing_score(
    team_a: Sequence[str],
    team_b: Sequence[str],
    teammate_counts: Mapping[tuple[str, str], int],
    opponent_counts: Mapping[tuple[str, str], int],
) -> tuple[int, int, int, int]:
    teammate_values = [
        teammate_counts[_pair_key(first, second)]
        for team in (team_a, team_b)
        for index, first in enumerate(team)
        for second in team[index + 1 :]
    ]
    opponent_values = [
        opponent_counts[_pair_key(first, second)]
        for first in team_a
        for second in team_b
    ]
    return (
        max(teammate_values, default=0),
        max(opponent_values, default=0),
        sum(teammate_values),
        sum(opponent_values),
    )


def _choose_balanced_lineup(
    player_ids: tuple[str, ...],
    appearances: Mapping[str, int],
    teammate_counts: Mapping[tuple[str, str], int],
    opponent_counts: Mapping[tuple[str, str], int],
    *,
    team_size: int,
    candidate_count: int,
    rng: np.random.Generator,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Minimize repeated teammate/opponent pairings among optimal rosters."""
    best_score: tuple[int, int, int, int] | None = None
    best_teams: tuple[tuple[str, ...], tuple[str, ...]] | None = None
    roster_size = team_size * 2
    for _ in range(candidate_count):
        roster = _candidate_roster(
            player_ids,
            appearances,
            roster_size=roster_size,
            rng=rng,
        )
        shuffled = tuple(rng.permutation(roster).tolist())
        team_a = tuple(sorted(shuffled[:team_size]))
        team_b = tuple(sorted(shuffled[team_size:]))
        score = _pairing_score(team_a, team_b, teammate_counts, opponent_counts)
        # Preserve the first candidate on a tie so the schedule remains
        # seeded and randomized rather than silently ordered by player IDs.
        if best_score is None or score < best_score:
            best_score = score
            best_teams = (team_a, team_b)
    assert best_teams is not None
    return best_teams


def _record_matchup(
    team_a: Sequence[str],
    team_b: Sequence[str],
    appearances: dict[str, int],
    teammate_counts: dict[tuple[str, str], int],
    opponent_counts: dict[tuple[str, str], int],
) -> None:
    for player_id in (*team_a, *team_b):
        appearances[player_id] += 1
    for team in (team_a, team_b):
        for index, first in enumerate(team):
            for second in team[index + 1 :]:
                teammate_counts[_pair_key(first, second)] += 1
    for first in team_a:
        for second in team_b:
            opponent_counts[_pair_key(first, second)] += 1


def _relationship_diagnostics(
    counts: Mapping[tuple[str, str], int],
) -> dict[str, int]:
    values = list(counts.values())
    return {
        "distinct_pair_count": len(values),
        "min_count": min(values, default=0),
        "max_count": max(values, default=0),
        "spread": max(values, default=0) - min(values, default=0),
        "total_assignments": sum(values),
    }


def schedule_balance_diagnostics(schedule: LineupSchedule) -> dict[str, Any]:
    """Recompute the balance report from a schedule without needing a checkpoint."""
    appearances = {player_id: 0 for player_id in schedule.pool_player_ids}
    team_a_appearances = {player_id: 0 for player_id in schedule.pool_player_ids}
    team_b_appearances = {player_id: 0 for player_id in schedule.pool_player_ids}
    starter_appearances = {player_id: 0 for player_id in schedule.pool_player_ids}
    teammate_counts: dict[tuple[str, str], int] = defaultdict(int)
    opponent_counts: dict[tuple[str, str], int] = defaultdict(int)
    if schedule.schema_version == _LEGACY_LINEUP_SCHEDULE_SCHEMA_VERSION:
        representatives = _paired_games(schedule.games, error_type=LineupScheduleError)
    else:
        representatives = _independent_games(
            schedule.games, error_type=LineupScheduleError
        )
    for game in representatives:
        _record_matchup(
            game.team_a_player_ids,
            game.team_b_player_ids,
            appearances,
            teammate_counts,
            opponent_counts,
        )
        for player_id in game.team_a_player_ids:
            team_a_appearances[player_id] += 1
        for player_id in game.team_b_player_ids:
            team_b_appearances[player_id] += 1
        starters = (
            game.team_a_player_ids
            if game.starting_team == _TEAM_A
            else game.team_b_player_ids
        )
        for player_id in starters:
            starter_appearances[player_id] += 1
    values = list(appearances.values())
    # Preserve the exact schema-v2 diagnostic payload for existing immutable
    # schedules.  Their appearances are matchup (not game) counts.
    if schedule.schema_version == _LEGACY_LINEUP_SCHEDULE_SCHEMA_VERSION:
        return {
            "matchup_count": len(representatives),
            "game_count": len(schedule.games),
            "player_appearances": dict(sorted(appearances.items())),
            "appearance_min": min(values, default=0),
            "appearance_max": max(values, default=0),
            "appearance_spread": max(values, default=0) - min(values, default=0),
            "teammate_pairs": _relationship_diagnostics(teammate_counts),
            "opponent_pairs": _relationship_diagnostics(opponent_counts),
        }

    team_a_starts = sum(game.starting_team == _TEAM_A for game in representatives)
    team_b_starts = sum(game.starting_team == _TEAM_B for game in representatives)
    starter_values = list(starter_appearances.values())
    return {
        "evaluation_design": "independent_balanced_games",
        "sampling_group_count": len(representatives),
        "game_count": len(schedule.games),
        "player_game_appearances": dict(sorted(appearances.items())),
        "game_appearance_min": min(values, default=0),
        "game_appearance_max": max(values, default=0),
        "game_appearance_spread": max(values, default=0) - min(values, default=0),
        "team_a_start_game_count": int(team_a_starts),
        "team_b_start_game_count": int(team_b_starts),
        "player_team_a_appearances": dict(sorted(team_a_appearances.items())),
        "player_team_b_appearances": dict(sorted(team_b_appearances.items())),
        "max_team_side_imbalance": max(
            (
                abs(team_a_appearances[player_id] - team_b_appearances[player_id])
                for player_id in schedule.pool_player_ids
            ),
            default=0,
        ),
        "player_start_appearances": dict(sorted(starter_appearances.items())),
        "start_appearance_min": min(starter_values, default=0),
        "start_appearance_max": max(starter_values, default=0),
        "start_appearance_spread": max(starter_values, default=0)
        - min(starter_values, default=0),
        "teammate_pairs": _relationship_diagnostics(teammate_counts),
        "opponent_pairs": _relationship_diagnostics(opponent_counts),
    }


def _paired_games(
    games: Sequence[ScheduledGame],
    *,
    error_type: type[LineupScheduleError],
) -> tuple[ScheduledGame, ...]:
    by_pair: dict[str, list[ScheduledGame]] = defaultdict(list)
    for game in games:
        by_pair[game.pair_id].append(game)
    representatives: list[ScheduledGame] = []
    seen_game_ids: set[str] = set()
    for pair_id, paired_games in by_pair.items():
        if not pair_id:
            raise error_type("Scheduled game pair IDs must be non-empty.")
        if len(paired_games) != 2:
            raise error_type(
                f"Pair {pair_id!r} must contain exactly two scheduled games."
            )
        starters = {game.starting_team for game in paired_games}
        if starters != set(_STARTING_TEAMS):
            raise error_type(
                f"Pair {pair_id!r} must have one {_TEAM_A} and one {_TEAM_B} start."
            )
        first, second = paired_games
        shared_fields = (
            "team_a_player_ids",
            "team_b_player_ids",
            "reset_rng_root",
            "environment_rng_root",
            "policy_rng_root",
        )
        for field in shared_fields:
            if getattr(first, field) != getattr(second, field):
                raise error_type(
                    f"Pair {pair_id!r} changes {field}; paired starts may only switch "
                    "the starting team."
                )
        for game in paired_games:
            if not game.game_id or game.game_id in seen_game_ids:
                raise error_type("Scheduled game IDs must be non-empty and unique.")
            seen_game_ids.add(game.game_id)
        representatives.append(first)
    return tuple(representatives)


def _independent_games(
    games: Sequence[ScheduledGame],
    *,
    error_type: type[LineupScheduleError],
) -> tuple[ScheduledGame, ...]:
    """Validate schema-v3 games: every sampling group contains one game."""
    seen_game_ids: set[str] = set()
    seen_group_ids: set[str] = set()
    for game in games:
        if not game.game_id or game.game_id in seen_game_ids:
            raise error_type("Scheduled game IDs must be non-empty and unique.")
        if not game.sampling_group_id or game.sampling_group_id in seen_group_ids:
            raise error_type(
                "Independent schedules need one non-empty sampling group per game."
            )
        seen_game_ids.add(game.game_id)
        seen_group_ids.add(game.sampling_group_id)
    return tuple(games)


def validate_lineup_schedule(
    schedule: LineupSchedule,
    *,
    pool: PlayerPool | None = None,
) -> None:
    """Validate pair parity, roster integrity, diagnostics, and optional pool binding."""
    if schedule.schema_version not in _SUPPORTED_LINEUP_SCHEDULE_SCHEMA_VERSIONS:
        raise LineupScheduleError(
            f"Unsupported lineup-schedule schema version {schedule.schema_version}."
        )
    if schedule.team_size != 5:
        raise LineupScheduleError(
            "Fixed-pool evaluation schedules must use 5-player teams."
        )
    expected_optimizer = (
        _LEGACY_OPTIMIZER_VERSION
        if schedule.schema_version == _LEGACY_LINEUP_SCHEDULE_SCHEMA_VERSION
        else _OPTIMIZER_VERSION
    )
    if schedule.optimizer_version != expected_optimizer:
        raise LineupScheduleError(
            f"Unsupported lineup optimizer {schedule.optimizer_version!r}."
        )
    if schedule.candidate_count < 1:
        raise LineupScheduleError(
            "Lineup-schedule candidate_count must be at least one."
        )
    if len(schedule.pool_player_ids) < schedule.team_size * 2:
        raise LineupScheduleError("Schedule pool must contain at least ten players.")
    if len(set(schedule.pool_player_ids)) != len(schedule.pool_player_ids):
        raise LineupScheduleError("Schedule pool_player_ids must be unique.")
    pool_ids = set(schedule.pool_player_ids)
    if schedule.schema_version == _LEGACY_LINEUP_SCHEDULE_SCHEMA_VERSION:
        representatives = _paired_games(schedule.games, error_type=LineupScheduleError)
    else:
        representatives = _independent_games(
            schedule.games, error_type=LineupScheduleError
        )
    for game in representatives:
        roster = (*game.team_a_player_ids, *game.team_b_player_ids)
        if game.starting_team not in _STARTING_TEAMS:
            raise LineupScheduleError(
                f"Scheduled game {game.game_id!r} has an invalid starting_team."
            )
        if (
            len(game.team_a_player_ids) != schedule.team_size
            or len(game.team_b_player_ids) != schedule.team_size
        ):
            raise LineupScheduleError(
                f"Scheduled game {game.game_id!r} must contain two {schedule.team_size}-player teams."
            )
        if len(set(roster)) != len(roster):
            raise LineupScheduleError(
                f"Scheduled game {game.game_id!r} repeats a player in one lineup."
            )
        unknown_ids = sorted(set(roster) - pool_ids)
        if unknown_ids:
            raise LineupScheduleError(
                f"Scheduled game {game.game_id!r} references players outside the pool: "
                f"{', '.join(unknown_ids)}."
            )
    expected_diagnostics = schedule_balance_diagnostics(schedule)
    if _canonical_json(schedule.diagnostics) != _canonical_json(expected_diagnostics):
        raise LineupScheduleError(
            "Schedule balance diagnostics do not match its game assignments."
        )
    if pool is not None:
        try:
            assert_pool_compatible(pool)
        except PlayerPoolError as exc:
            raise ScheduleCompatibilityError(
                f"Pool {pool.pool_id} is not compatible with fixed-pool evaluation: {exc}"
            ) from exc
        if schedule.pool_id != pool.pool_id:
            raise ScheduleCompatibilityError(
                f"Schedule {schedule.schedule_id} was created for pool {schedule.pool_id}, "
                f"not {pool.pool_id}."
            )
        if schedule.pool_skill_schema_hash != pool.skill_schema_hash:
            raise ScheduleCompatibilityError(
                "Schedule skill-schema hash does not match its requested player pool."
            )
        actual_player_ids = tuple(player.player_id for player in pool.players)
        if schedule.pool_player_ids != actual_player_ids:
            raise ScheduleCompatibilityError(
                "Schedule player IDs do not match the requested player-pool artifact."
            )


def build_balanced_lineup_schedule(
    pool: PlayerPool,
    *,
    game_count: int,
    seed: int,
    candidate_count: int = 128,
) -> LineupSchedule:
    """Build a deterministic schedule of distinct, balanced 5v5 games."""
    assert_pool_compatible(pool)
    if int(game_count) < 1:
        raise LineupScheduleError("game_count must be at least one.")
    if int(candidate_count) < 1:
        raise LineupScheduleError("candidate_count must be at least one.")
    team_size = 5
    player_ids = tuple(player.player_id for player in pool.players)
    if len(player_ids) < team_size * 2:
        raise LineupScheduleError(
            "A 5v5 lineup schedule requires at least ten pool players."
        )
    rng = np.random.default_rng(int(seed))
    appearances = {player_id: 0 for player_id in player_ids}
    teammate_counts: dict[tuple[str, str], int] = defaultdict(int)
    opponent_counts: dict[tuple[str, str], int] = defaultdict(int)
    team_a_appearances = {player_id: 0 for player_id in player_ids}
    team_b_appearances = {player_id: 0 for player_id in player_ids}
    starter_appearances = {player_id: 0 for player_id in player_ids}
    team_a_starts = 0
    team_b_starts = 0
    games: list[ScheduledGame] = []
    for game_index in range(int(game_count)):
        team_a, team_b = _choose_balanced_lineup(
            player_ids,
            appearances,
            teammate_counts,
            opponent_counts,
            team_size=team_size,
            candidate_count=int(candidate_count),
            rng=rng,
        )
        # Team labels determine the signed outcome column.  Assign them so
        # every player's Team-A / Team-B exposure remains nearly balanced.
        normal_side_cost = sum(
            (team_a_appearances[player_id] + 1 - team_b_appearances[player_id]) ** 2
            for player_id in team_a
        ) + sum(
            (team_b_appearances[player_id] + 1 - team_a_appearances[player_id]) ** 2
            for player_id in team_b
        )
        swapped_side_cost = sum(
            (team_a_appearances[player_id] + 1 - team_b_appearances[player_id]) ** 2
            for player_id in team_b
        ) + sum(
            (team_b_appearances[player_id] + 1 - team_a_appearances[player_id]) ** 2
            for player_id in team_a
        )
        if swapped_side_cost < normal_side_cost or (
            swapped_side_cost == normal_side_cost and bool(rng.integers(0, 2))
        ):
            team_a, team_b = team_b, team_a
        _record_matchup(
            team_a,
            team_b,
            appearances,
            teammate_counts,
            opponent_counts,
        )
        for player_id in team_a:
            team_a_appearances[player_id] += 1
        for player_id in team_b:
            team_b_appearances[player_id] += 1
        max_team_a_starts = (int(game_count) + 1) // 2
        max_team_b_starts = int(game_count) // 2
        if team_a_starts >= max_team_a_starts:
            starting_team = _TEAM_B
        elif team_b_starts >= max_team_b_starts:
            starting_team = _TEAM_A
        else:
            team_a_start_cost = sum(starter_appearances[player_id] for player_id in team_a)
            team_b_start_cost = sum(starter_appearances[player_id] for player_id in team_b)
            if team_a_start_cost < team_b_start_cost:
                starting_team = _TEAM_A
            elif team_b_start_cost < team_a_start_cost:
                starting_team = _TEAM_B
            else:
                starting_team = _TEAM_A if team_a_starts <= team_b_starts else _TEAM_B
        starters = team_a if starting_team == _TEAM_A else team_b
        for player_id in starters:
            starter_appearances[player_id] += 1
        if starting_team == _TEAM_A:
            team_a_starts += 1
        else:
            team_b_starts += 1
        game_id = f"game-{game_index:05d}"
        reset_rng_root = int(rng.integers(0, 2**32, dtype=np.uint32))
        environment_rng_root = int(rng.integers(0, 2**32, dtype=np.uint32))
        policy_rng_root = int(rng.integers(0, 2**32, dtype=np.uint32))
        games.append(
            ScheduledGame(
                game_id=game_id,
                # The serialized legacy field now names a one-game sampling
                # group.  CV and bootstrap therefore keep each independent
                # game intact without pretending it has a paired counterpart.
                pair_id=game_id,
                starting_team=starting_team,
                team_a_player_ids=team_a,
                team_b_player_ids=team_b,
                reset_rng_root=reset_rng_root,
                environment_rng_root=environment_rng_root,
                policy_rng_root=policy_rng_root,
            )
        )
    return _make_schedule(
        pool=pool,
        schedule_seed=int(seed),
        team_size=team_size,
        candidate_count=int(candidate_count),
        games=games,
    )


def build_lineup_schedule_extension(
    pool: PlayerPool,
    base_schedule: LineupSchedule,
    *,
    additional_game_count: int,
    seed: int,
) -> LineupScheduleExtension:
    """Append newly balanced games without changing an immutable base schedule.

    The returned additional schedule has only the new games, so callers can
    evaluate exactly that segment.  The combined schedule preserves every base
    game byte-for-byte and supplies the provenance for one larger raw study.
    """
    assert_pool_compatible(pool)
    validate_lineup_schedule(base_schedule, pool=pool)
    if base_schedule.schema_version != LINEUP_SCHEDULE_SCHEMA_VERSION:
        raise LineupScheduleError(
            "Only independent-game schema-v3 schedules can be extended."
        )
    if int(additional_game_count) < 1:
        raise LineupScheduleError("additional_game_count must be at least one.")

    player_ids = tuple(player.player_id for player in pool.players)
    appearances = {player_id: 0 for player_id in player_ids}
    teammate_counts: dict[tuple[str, str], int] = defaultdict(int)
    opponent_counts: dict[tuple[str, str], int] = defaultdict(int)
    team_a_appearances = {player_id: 0 for player_id in player_ids}
    team_b_appearances = {player_id: 0 for player_id in player_ids}
    starter_appearances = {player_id: 0 for player_id in player_ids}
    team_a_starts = 0
    team_b_starts = 0
    for game in base_schedule.games:
        _record_matchup(
            game.team_a_player_ids,
            game.team_b_player_ids,
            appearances,
            teammate_counts,
            opponent_counts,
        )
        for player_id in game.team_a_player_ids:
            team_a_appearances[player_id] += 1
        for player_id in game.team_b_player_ids:
            team_b_appearances[player_id] += 1
        starters = (
            game.team_a_player_ids
            if game.starting_team == _TEAM_A
            else game.team_b_player_ids
        )
        for player_id in starters:
            starter_appearances[player_id] += 1
        if game.starting_team == _TEAM_A:
            team_a_starts += 1
        else:
            team_b_starts += 1

    rng = np.random.default_rng(int(seed))
    extra_games: list[ScheduledGame] = []
    base_game_count = len(base_schedule.games)
    total_game_count = base_game_count + int(additional_game_count)
    for offset in range(int(additional_game_count)):
        team_a, team_b = _choose_balanced_lineup(
            player_ids,
            appearances,
            teammate_counts,
            opponent_counts,
            team_size=base_schedule.team_size,
            candidate_count=base_schedule.candidate_count,
            rng=rng,
        )
        normal_side_cost = sum(
            (team_a_appearances[player_id] + 1 - team_b_appearances[player_id]) ** 2
            for player_id in team_a
        ) + sum(
            (team_b_appearances[player_id] + 1 - team_a_appearances[player_id]) ** 2
            for player_id in team_b
        )
        swapped_side_cost = sum(
            (team_a_appearances[player_id] + 1 - team_b_appearances[player_id]) ** 2
            for player_id in team_b
        ) + sum(
            (team_b_appearances[player_id] + 1 - team_a_appearances[player_id]) ** 2
            for player_id in team_a
        )
        if swapped_side_cost < normal_side_cost or (
            swapped_side_cost == normal_side_cost and bool(rng.integers(0, 2))
        ):
            team_a, team_b = team_b, team_a
        _record_matchup(
            team_a,
            team_b,
            appearances,
            teammate_counts,
            opponent_counts,
        )
        for player_id in team_a:
            team_a_appearances[player_id] += 1
        for player_id in team_b:
            team_b_appearances[player_id] += 1
        max_team_a_starts = (total_game_count + 1) // 2
        max_team_b_starts = total_game_count // 2
        if team_a_starts >= max_team_a_starts:
            starting_team = _TEAM_B
        elif team_b_starts >= max_team_b_starts:
            starting_team = _TEAM_A
        else:
            team_a_start_cost = sum(starter_appearances[player_id] for player_id in team_a)
            team_b_start_cost = sum(starter_appearances[player_id] for player_id in team_b)
            if team_a_start_cost < team_b_start_cost:
                starting_team = _TEAM_A
            elif team_b_start_cost < team_a_start_cost:
                starting_team = _TEAM_B
            else:
                starting_team = _TEAM_A if team_a_starts <= team_b_starts else _TEAM_B
        starters = team_a if starting_team == _TEAM_A else team_b
        for player_id in starters:
            starter_appearances[player_id] += 1
        if starting_team == _TEAM_A:
            team_a_starts += 1
        else:
            team_b_starts += 1
        game_id = f"game-{base_game_count + offset:05d}"
        extra_games.append(
            ScheduledGame(
                game_id=game_id,
                pair_id=game_id,
                starting_team=starting_team,
                team_a_player_ids=team_a,
                team_b_player_ids=team_b,
                reset_rng_root=int(rng.integers(0, 2**32, dtype=np.uint32)),
                environment_rng_root=int(rng.integers(0, 2**32, dtype=np.uint32)),
                policy_rng_root=int(rng.integers(0, 2**32, dtype=np.uint32)),
            )
        )

    extension_metadata = {
        "base_schedule_id": base_schedule.schedule_id,
        "base_schedule_seed": int(base_schedule.schedule_seed),
        "base_game_count": base_game_count,
        "additional_game_count": int(additional_game_count),
        "extension_seed": int(seed),
    }
    additional_schedule = _make_schedule(
        pool=pool,
        schedule_seed=int(seed),
        team_size=base_schedule.team_size,
        candidate_count=base_schedule.candidate_count,
        games=extra_games,
        extra_fields={"extension_segment": extension_metadata},
    )
    combined_schedule = _make_schedule(
        pool=pool,
        schedule_seed=int(seed),
        team_size=base_schedule.team_size,
        candidate_count=base_schedule.candidate_count,
        games=(*base_schedule.games, *extra_games),
        extra_fields={"extension": extension_metadata},
    )
    return LineupScheduleExtension(
        additional_schedule=additional_schedule,
        combined_schedule=combined_schedule,
    )


def write_lineup_schedule(schedule: LineupSchedule, path: str | Path) -> Path:
    """Write an immutable schedule artifact, rejecting a conflicting overwrite."""
    target = Path(path)
    payload = schedule.to_payload()
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        existing = target.read_text(encoding="utf-8")
        if existing != text:
            raise FileExistsError(
                f"Refusing to overwrite immutable lineup-schedule artifact: {target}"
            )
        return target
    target.write_text(text, encoding="utf-8")
    return target


def load_lineup_schedule(
    path: str | Path,
    *,
    pool: PlayerPool | None = None,
) -> LineupSchedule:
    """Load a schedule and, when provided, validate it against its exact pool."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LineupScheduleError(f"Invalid lineup-schedule JSON: {source}") from exc
    schedule = LineupSchedule.from_payload(payload)
    validate_lineup_schedule(schedule, pool=pool)
    return schedule
