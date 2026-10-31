"""Direct matched cross-play between distinct trained JAX policies."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from basketworld_jax.checkpoints import load_checkpoint_metadata
from basketworld_jax.eval.native import run_native_jax_evaluation


DEFAULT_EPISODES = 1000
DEFAULT_BOOTSTRAP_SAMPLES = 2000
POLICY_AR = "autoregressive"
POLICY_INDEPENDENT = "independent"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")
    temporary.replace(path)


def _checkpoint_digest(path: Path) -> str:
    source = path / "metadata.json" if path.is_dir() else path
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _policy_mode(metadata: Mapping[str, Any]) -> str:
    return str(
        (metadata.get("policy_spec") or {}).get(
            "team_action_mode", "independent"
        )
    )


def validate_crossplay_pair(
    autoregressive_checkpoint: str | Path,
    independent_checkpoint: str | Path,
) -> dict[str, dict[str, Any]]:
    """Validate that two paths identify distinct, correctly trained policies."""
    ar_path = Path(autoregressive_checkpoint).expanduser().resolve()
    independent_path = Path(independent_checkpoint).expanduser().resolve()
    if ar_path == independent_path:
        raise ValueError("Cross-play requires two distinct checkpoint paths.")

    ar_metadata = load_checkpoint_metadata(ar_path)
    independent_metadata = load_checkpoint_metadata(independent_path)
    ar_mode = _policy_mode(ar_metadata)
    independent_mode = _policy_mode(independent_metadata)
    if ar_mode != POLICY_AR:
        raise ValueError(
            "The autoregressive checkpoint was not trained with "
            f"team_action_mode='autoregressive' (found {ar_mode!r})."
        )
    if independent_mode != POLICY_INDEPENDENT:
        raise ValueError(
            "The independent checkpoint was not trained with "
            f"team_action_mode='independent' (found {independent_mode!r})."
        )

    ar_digest = _checkpoint_digest(ar_path)
    independent_digest = _checkpoint_digest(independent_path)
    if ar_digest == independent_digest:
        raise ValueError(
            "The two paths have identical checkpoint metadata; cross-play must "
            "use separately trained policies, not aliases of one checkpoint."
        )

    ar_players = int(
        (ar_metadata.get("policy_spec") or {}).get("training_player_count", 0)
    )
    independent_players = int(
        (independent_metadata.get("policy_spec") or {}).get("training_player_count", 0)
    )
    if ar_players < 1 or ar_players != independent_players:
        raise ValueError(
            "Cross-play checkpoints must have the same positive training player count "
            f"(found {ar_players} and {independent_players})."
        )

    return {
        POLICY_AR: {
            "path": str(ar_path),
            "metadata_sha256": ar_digest,
            "metadata": ar_metadata,
        },
        POLICY_INDEPENDENT: {
            "path": str(independent_path),
            "metadata_sha256": independent_digest,
            "metadata": independent_metadata,
        },
    }


def validate_protocol(
    *, episodes: int, batch_size: int, bootstrap_samples: int
) -> None:
    if int(episodes) < 4 or int(episodes) % 4 != 0:
        raise ValueError(
            "episodes must be a positive multiple of four so every base scenario "
            "has both starting teams and both checkpoint assignments."
        )
    if int(batch_size) < 2 or int(batch_size) % 2 != 0:
        raise ValueError("batch_size must be a positive even number.")
    if int(bootstrap_samples) < 1:
        raise ValueError("bootstrap_samples must be positive.")


def _normalize_episode(
    episode: Mapping[str, Any],
    *,
    assignment: str,
    assignment_episode_index: int,
) -> dict[str, Any]:
    ar_team = "team_a" if assignment == "ar_team_a" else "team_b"
    independent_team = "team_b" if ar_team == "team_a" else "team_a"
    game = dict(episode["game"])
    starting_team = str(game["starting_offense_team"])
    ar_score = float(game["user_score"])
    independent_score = float(game["opponent_score"])
    margin = ar_score - independent_score
    completed = bool(episode["completed"])
    return {
        "assignment": assignment,
        "assignment_episode_index": int(assignment_episode_index),
        "scenario_id": int(assignment_episode_index // 2),
        "starting_offense_stable_team": starting_team,
        "starting_offense_policy": (
            POLICY_AR if starting_team == ar_team else POLICY_INDEPENDENT
        ),
        "autoregressive_stable_team": ar_team,
        "independent_stable_team": independent_team,
        "completed": completed,
        "truncated": bool(episode["truncated"]),
        "termination_kind": str(episode["termination_kind"]),
        "autoregressive_score": ar_score,
        "independent_score": independent_score,
        "autoregressive_margin": float(margin),
        "autoregressive_win_score": (
            (1.0 if margin > 0.0 else 0.5 if margin == 0.0 else 0.0)
            if completed
            else None
        ),
        "autoregressive_completed_possessions": int(
            game["user_completed_possessions"]
        ),
        "independent_completed_possessions": int(
            game["opponent_completed_possessions"]
        ),
        "native_episode": dict(episode),
    }


def _bootstrap_mean_interval(
    values: Sequence[float], *, seed: int, samples: int
) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"mean": None, "ci95_low": None, "ci95_high": None}
    mean = float(array.mean())
    if array.size == 1:
        return {"mean": mean, "ci95_low": mean, "ci95_high": mean}
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, array.size, size=(int(samples), array.size))
    means = array[indices].mean(axis=1)
    return {
        "mean": mean,
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def _outcome_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if bool(row["completed"])]
    margins = [float(row["autoregressive_margin"]) for row in completed]
    win_scores = [float(row["autoregressive_win_score"]) for row in completed]
    ar_possessions = sum(
        int(row["autoregressive_completed_possessions"]) for row in completed
    )
    independent_possessions = sum(
        int(row["independent_completed_possessions"]) for row in completed
    )
    ar_points = sum(float(row["autoregressive_score"]) for row in completed)
    independent_points = sum(float(row["independent_score"]) for row in completed)
    return {
        "games": len(rows),
        "completed_games": len(completed),
        "truncated_games": len(rows) - len(completed),
        "autoregressive_wins": sum(margin > 0.0 for margin in margins),
        "ties": sum(margin == 0.0 for margin in margins),
        "independent_wins": sum(margin < 0.0 for margin in margins),
        "autoregressive_win_score_mean": (
            float(np.mean(win_scores)) if win_scores else None
        ),
        "autoregressive_margin_mean": float(np.mean(margins)) if margins else None,
        "autoregressive_points_per_possession": (
            float(ar_points / ar_possessions) if ar_possessions else None
        ),
        "independent_points_per_possession": (
            float(independent_points / independent_possessions)
            if independent_possessions
            else None
        ),
        "autoregressive_points": float(ar_points),
        "independent_points": float(independent_points),
        "autoregressive_completed_possessions": int(ar_possessions),
        "independent_completed_possessions": int(independent_possessions),
    }


def summarize_crossplay(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    by_assignment = {
        assignment: _outcome_summary(
            [row for row in rows if row["assignment"] == assignment]
        )
        for assignment in ("ar_team_a", "ar_team_b")
    }
    by_start = {
        policy: _outcome_summary(
            [row for row in rows if row["starting_offense_policy"] == policy]
        )
        for policy in (POLICY_AR, POLICY_INDEPENDENT)
    }

    scenario_rows: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        scenario_rows[int(row["scenario_id"])].append(row)
    complete_blocks = [
        block
        for block in scenario_rows.values()
        if len(block) == 4
        and all(bool(row["completed"]) for row in block)
        and {
            (
                str(row["assignment"]),
                str(row["starting_offense_stable_team"]),
            )
            for row in block
        }
        == {
            ("ar_team_a", "team_a"),
            ("ar_team_a", "team_b"),
            ("ar_team_b", "team_a"),
            ("ar_team_b", "team_b"),
        }
    ]
    block_win_scores = [
        float(np.mean([float(row["autoregressive_win_score"]) for row in block]))
        for block in complete_blocks
    ]
    block_margins = [
        float(np.mean([float(row["autoregressive_margin"]) for row in block]))
        for block in complete_blocks
    ]
    return {
        "overall": _outcome_summary(rows),
        "by_assignment": by_assignment,
        "by_starting_offense_policy": by_start,
        "matched_scenarios": {
            "scenario_count_seen": len(scenario_rows),
            "complete_four_game_block_count": len(complete_blocks),
            "bootstrap_unit": "four_game_matched_scenario",
            "autoregressive_win_score": _bootstrap_mean_interval(
                block_win_scores,
                seed=int(bootstrap_seed),
                samples=int(bootstrap_samples),
            ),
            "autoregressive_margin": _bootstrap_mean_interval(
                block_margins,
                seed=int(bootstrap_seed) + 1,
                samples=int(bootstrap_samples),
            ),
        },
    }


def _aggregate_player_stats(
    assignment_results: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    player_count: int,
) -> dict[str, dict[str, Any]]:
    fields = (
        "shots",
        "makes",
        "assists",
        "potential_assists",
        "turnovers",
        "steals",
        "points",
        "offensive_rebounds",
        "defensive_rebounds",
    )
    totals = {
        POLICY_AR: {field: 0.0 for field in fields},
        POLICY_INDEPENDENT: {field: 0.0 for field in fields},
    }
    for assignment, result in assignment_results:
        ar_ids = (
            range(0, player_count)
            if assignment == "ar_team_a"
            else range(player_count, player_count * 2)
        )
        ar_ids = set(ar_ids)
        for raw_player_id, stats in (result.get("per_player_stats") or {}).items():
            policy = POLICY_AR if int(raw_player_id) in ar_ids else POLICY_INDEPENDENT
            for field in fields:
                totals[policy][field] += float(stats.get(field, 0.0) or 0.0)
    for policy, stats in totals.items():
        stats["assist_to_turnover_ratio"] = (
            float(stats["assists"] / stats["turnovers"])
            if stats["turnovers"] > 0.0
            else None
        )
        stats["field_goal_percentage"] = (
            float(stats["makes"] / stats["shots"]) if stats["shots"] > 0.0 else None
        )
    return totals


def _violation_summary(
    rows: Sequence[Mapping[str, Any]], *, player_count: int
) -> dict[str, Any]:
    turnovers = {
        POLICY_AR: Counter(),
        POLICY_INDEPENDENT: Counter(),
        "unattributed": Counter(),
    }
    lane_violations = {POLICY_AR: 0, POLICY_INDEPENDENT: 0}
    for row in rows:
        ar_team_a = row["assignment"] == "ar_team_a"

        def policy_for_player(player_id: int) -> str:
            team_a = int(player_id) < int(player_count)
            return POLICY_AR if team_a == ar_team_a else POLICY_INDEPENDENT

        outcome = (row["native_episode"].get("outcome_info") or {})
        for turnover in outcome.get("turnovers") or []:
            player_id = turnover.get("player_id")
            policy = (
                "unattributed" if player_id is None else policy_for_player(player_id)
            )
            turnovers[policy][str(turnover.get("reason") or "unknown")] += 1
        for violation in outcome.get("defensive_lane_violations") or []:
            player_id = violation.get("player_id")
            if player_id is not None:
                lane_violations[policy_for_player(player_id)] += 1
    return {
        "turnovers_by_reason": {
            policy: dict(sorted(counts.items())) for policy, counts in turnovers.items()
        },
        "defensive_lane_violations": lane_violations,
    }


def _source_provenance(repo_root: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() if completed.returncode == 0 else ""

    status = git("status", "--porcelain")
    return {
        "commit": git("rev-parse", "HEAD") or None,
        "branch": git("branch", "--show-current") or None,
        "dirty": bool(status),
        "changed_paths": [line[3:] for line in status.splitlines() if len(line) > 3],
    }


def _write_games(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=_json_default))
            handle.write("\n")


def run_policy_crossplay(
    *,
    autoregressive_checkpoint: str | Path,
    independent_checkpoint: str | Path,
    output_dir: str | Path,
    episodes: int = DEFAULT_EPISODES,
    seed: int = 1,
    action_mode: str = "deterministic",
    batch_size: int = 64,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    environment_overrides: Mapping[str, Any] | None = None,
    repo_root: str | Path | None = None,
    printer: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run a balanced AR-vs-independent study and persist reproducible artifacts."""
    validate_protocol(
        episodes=int(episodes),
        batch_size=int(batch_size),
        bootstrap_samples=int(bootstrap_samples),
    )
    if action_mode not in {"deterministic", "sampled"}:
        raise ValueError("action_mode must be 'deterministic' or 'sampled'.")
    checkpoints = validate_crossplay_pair(
        autoregressive_checkpoint, independent_checkpoint
    )
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Output directory already exists: {destination}")
    destination.mkdir(parents=True)

    root = (
        Path(repo_root).expanduser().resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    player_count = int(
        checkpoints[POLICY_AR]["metadata"]["policy_spec"]["training_player_count"]
    )
    manifest: dict[str, Any] = {
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "episodes_total": int(episodes),
            "episodes_per_assignment": int(episodes) // 2,
            "matched_four_game_scenarios": int(episodes) // 4,
            "seed": int(seed),
            "action_mode": action_mode,
            "batch_size": int(batch_size),
            "bootstrap_samples": int(bootstrap_samples),
            "environment_overrides": dict(environment_overrides or {}),
            "decoder_override_allowed": False,
        },
        "checkpoints": checkpoints,
        "source": _source_provenance(root),
    }
    manifest_path = destination / "manifest.json"
    progress_path = destination / "progress.json"
    _write_json(manifest_path, manifest)

    rows: list[dict[str, Any]] = []
    assignment_results: list[tuple[str, Mapping[str, Any]]] = []
    episodes_per_assignment = int(episodes) // 2
    deterministic = action_mode == "deterministic"

    try:
        for assignment, user_team_name in (
            ("ar_team_a", "OFFENSE"),
            ("ar_team_b", "DEFENSE"),
        ):
            assignment_rows: list[dict[str, Any]] = []

            def on_batch(
                batch: Sequence[Mapping[str, Any]], completed: int, total: int
            ) -> None:
                offset = len(assignment_rows)
                normalized = [
                    _normalize_episode(
                        episode,
                        assignment=assignment,
                        assignment_episode_index=offset + index,
                    )
                    for index, episode in enumerate(batch)
                ]
                assignment_rows.extend(normalized)
                current = [*rows, *assignment_rows]
                snapshot = summarize_crossplay(
                    current,
                    bootstrap_seed=int(seed),
                    bootstrap_samples=int(bootstrap_samples),
                )
                snapshot["progress"] = {
                    "assignment": assignment,
                    "assignment_completed": int(completed),
                    "assignment_total": int(total),
                    "total_completed": len(current),
                    "total_games": int(episodes),
                }
                _write_json(progress_path, snapshot)
                overall = snapshot["overall"]
                matched = snapshot["matched_scenarios"]
                message = (
                    f"{assignment}: {completed}/{total}; "
                    f"total {len(current)}/{episodes}; "
                    f"AR win score {overall['autoregressive_win_score_mean']!s}, "
                    f"margin {overall['autoregressive_margin_mean']!s}"
                )
                interval = matched["autoregressive_win_score"]
                if interval["mean"] is not None:
                    message += (
                        "; "
                        f"{matched['complete_four_game_block_count']} matched blocks, "
                        f"paired win-score 95% CI "
                        f"[{interval['ci95_low']:.3f}, {interval['ci95_high']:.3f}]"
                    )
                printer(message)

            def on_event(event: Mapping[str, Any]) -> None:
                phase = str(event.get("phase") or "")
                if phase == "batch_running":
                    printer(
                        f"{assignment}: batch still running after "
                        f"{float(event.get('batch_elapsed_sec', 0.0)):.0f}s "
                        f"({int(event.get('completed_episodes', 0))}/"
                        f"{int(event.get('total_episodes', episodes_per_assignment))})"
                    )
                elif phase in {
                    "checkpoint_loading",
                    "checkpoint_loaded",
                    "jax_initializing",
                    "jax_initialized",
                    "environment_building",
                    "runner_ready",
                    "batch_started",
                }:
                    printer(f"{assignment}: {phase.replace('_', ' ')}")

            result = run_native_jax_evaluation(
                num_episodes=episodes_per_assignment,
                player_deterministic=deterministic,
                opponent_deterministic=deterministic,
                required_params={},
                optional_params=dict(environment_overrides or {}),
                unified_policy_path=checkpoints[POLICY_AR]["path"],
                opponent_policy_path=checkpoints[POLICY_INDEPENDENT]["path"],
                user_team_name=user_team_name,
                role_flag_offense=1.0,
                role_flag_defense=-1.0,
                eval_seed=int(seed),
                evaluation_batch_size=int(batch_size),
                progress_event_callback=on_event,
                batch_result_callback=on_batch,
            )
            if len(assignment_rows) != episodes_per_assignment:
                raise RuntimeError(
                    f"Native evaluator returned {len(assignment_rows)} rows for "
                    f"{assignment}; expected {episodes_per_assignment}."
                )
            rows.extend(assignment_rows)
            assignment_results.append((assignment, result))
            native_payload = {
                key: value for key, value in result.items() if key != "results"
            }
            _write_json(destination / f"native_{assignment}.json", native_payload)

        summary = summarize_crossplay(
            rows,
            bootstrap_seed=int(seed),
            bootstrap_samples=int(bootstrap_samples),
        )
        summary["policy_box_stats"] = _aggregate_player_stats(
            assignment_results,
            player_count=player_count,
        )
        summary["violations"] = _violation_summary(rows, player_count=player_count)
        summary["protocol"] = manifest["protocol"]
        summary["assignment_spatial_diagnostics"] = {
            assignment: {
                key: result["eval_diagnostics"]["jax_native_summary"].get(key)
                for key in (
                    "mean_live_ball_handler_nearest_defender_distance",
                    "live_ball_handler_pressure_rate",
                    "mean_live_offense_nearest_defender_distance",
                    "mean_live_unguarded_offense_fraction",
                    "mean_live_team_centroid_distance",
                )
            }
            for assignment, result in assignment_results
        }
        games_path = destination / "games.jsonl"
        summary_path = destination / "summary.json"
        _write_games(games_path, rows)
        _write_json(summary_path, summary)
        manifest["status"] = "completed"
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest["artifacts"] = {
            "games": games_path.name,
            "summary": summary_path.name,
            "progress": progress_path.name,
            "native_assignments": [
                "native_ar_team_a.json",
                "native_ar_team_b.json",
            ],
        }
        _write_json(manifest_path, manifest)
        return {
            "output_dir": str(destination),
            "manifest": str(manifest_path),
            "games": str(games_path),
            "summary": str(summary_path),
            "summary_payload": summary,
        }
    except BaseException as exc:
        manifest["status"] = (
            "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        )
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        _write_json(manifest_path, manifest)
        raise
