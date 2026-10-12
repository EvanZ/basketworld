"""Opt-in operational runner for fixed-player-pool evaluation studies.

The training CLI dispatches here only for ``--run-fixed-pool-study``.  Keeping
this workflow separate makes it impossible for normal evaluation or Elo output
to acquire fixed-pool artifacts unless the study was explicitly requested.
"""

from __future__ import annotations

from contextlib import nullcontext
import inspect
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import basketworld

from basketworld.utils.mlflow_config import setup_mlflow
from basketworld_jax.checkpoints import load_checkpoint, load_checkpoint_metadata
from basketworld_jax.eval.fixed_pool import (
    PlayerPool,
    generate_player_pool,
    load_player_pool,
    write_player_pool,
)
from basketworld_jax.eval.fixed_pool_models import (
    fit_fixed_pool_study_models,
    fixed_pool_study_budget,
)
from basketworld_jax.eval.fixed_pool_results import (
    combine_fixed_pool_study_artifacts,
    load_fixed_pool_study_artifacts,
)
from basketworld_jax.eval.lineup_schedule import (
    LineupSchedule,
    build_balanced_lineup_schedule,
    build_lineup_schedule_extension,
    load_lineup_schedule,
    write_lineup_schedule,
)
from basketworld_jax.eval.native import (
    _JAX_STATIC_ONLY_ENV_KEYS,
    run_native_jax_evaluation,
)
from basketworld_jax.eval.skills import ACTIVE_PLAYER_SKILL_REGISTRY


FIXED_POOL_REPORT_NAMESPACE = "fixed_pool_reports"
FIXED_POOL_OPERATION_SCHEMA_VERSION = 1
FIXED_POOL_SAMPLING_SCHEME = "independent_skill_coverage_v1"
# Bump whenever the derived model semantics change so immutable historical
# reports remain interpretable alongside later analyses of the same raw games.
FIXED_POOL_REANALYSIS_SCHEMA_VERSION = 8


class FixedPoolStudyRunError(ValueError):
    """The requested operational study is incomplete or unsafe to run."""


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    if isinstance(value, Path):
        return str(value)
    return value


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> Path:
    content = json.dumps(
        _json_ready(payload), indent=2, sort_keys=True, allow_nan=False
    )
    content += "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FixedPoolStudyRunError(
                f"Refusing to overwrite immutable fixed-pool report: {path}"
            )
        return path
    path.write_text(content, encoding="utf-8")
    return path


def _parse_checkpoint_paths(args) -> list[Path]:
    explicit = [
        Path(path).expanduser()
        for path in list(getattr(args, "fixed_pool_checkpoint", []) or [])
        if str(path).strip()
    ]
    checkpoint_dir = str(getattr(args, "fixed_pool_checkpoint_dir", "") or "").strip()
    update_indices = list(getattr(args, "fixed_pool_checkpoint_update", []) or [])
    if checkpoint_dir:
        root = Path(checkpoint_dir).expanduser()
        if update_indices:
            explicit.extend(
                root / f"update_{int(index):07d}" for index in update_indices
            )
        else:
            numbered = sorted(root.glob("update_*"))
            if not numbered:
                raise FixedPoolStudyRunError(
                    f"No numbered checkpoints found in {root}."
                )
            explicit.append(numbered[-1])
    deduplicated: list[Path] = []
    seen: set[Path] = set()
    for path in explicit:
        resolved = path.resolve()
        if resolved not in seen:
            deduplicated.append(resolved)
            seen.add(resolved)
    if not deduplicated:
        raise FixedPoolStudyRunError(
            "Specify --fixed-pool-checkpoint or --fixed-pool-checkpoint-dir."
        )
    missing = [str(path) for path in deduplicated if not path.exists()]
    if missing:
        raise FixedPoolStudyRunError(
            "Checkpoint paths do not exist: " + ", ".join(missing)
        )
    return deduplicated


def _study_skill_sampling_overrides() -> dict[str, dict[str, float]]:
    """Return deliberate skill variation for an estimable study pool.

    Production defaults intentionally leave every synthetic skill at its
    baseline.  A fixed-pool study needs nonzero support for every registered
    coefficient, otherwise skill and player-value models are rank deficient
    before a game is played.
    """
    overrides: dict[str, dict[str, float]] = {}
    for descriptor in ACTIVE_PLAYER_SKILL_REGISTRY.descriptors:
        defaults = dict(descriptor.sampling_defaults)
        mean = float(defaults.get("default_mean", 0.0))
        if descriptor.minimum is not None and descriptor.maximum is not None:
            # Percentage skills stay plausible while still spanning useful
            # variation after their declared bounds are applied.
            std = 0.15 * (float(descriptor.maximum) - float(descriptor.minimum))
        else:
            # These runtime fields are centered effect scales, so one standard
            # deviation supplies balanced positive and negative support.
            std = 1.0
        overrides[descriptor.key] = {"mean": mean, "std": float(std)}
    return overrides


def _load_or_create_pool(args) -> tuple[PlayerPool, Path, bool]:
    path_text = str(getattr(args, "fixed_pool_path", "") or "").strip()
    if not path_text:
        raise FixedPoolStudyRunError("--fixed-pool-path is required for a study run.")
    path = Path(path_text).expanduser()
    create = bool(getattr(args, "fixed_pool_create", False))
    if path.exists():
        if create:
            raise FixedPoolStudyRunError(
                "--fixed-pool-create requires a new path; immutable pool already exists."
            )
        return load_player_pool(path), path, False
    if not create:
        raise FixedPoolStudyRunError(
            "Fixed pool does not exist. Pass --fixed-pool-create to create it."
        )
    configured_size = int(getattr(args, "fixed_pool_size", 0) or 0)
    default_size = fixed_pool_study_budget(
        len(ACTIVE_PLAYER_SKILL_REGISTRY.descriptors)
    )["seeded_players"]
    pool = generate_player_pool(
        player_count=configured_size if configured_size > 0 else default_size,
        seed=int(getattr(args, "fixed_pool_seed", 0)),
        sampling_overrides=_study_skill_sampling_overrides(),
        extra_fields={
            "fixed_pool_sampling": {
                "scheme": FIXED_POOL_SAMPLING_SCHEME,
                "overrides": _study_skill_sampling_overrides(),
            }
        },
    )
    return pool, write_player_pool(pool, path), True


def _load_or_create_schedule(
    args, pool: PlayerPool
) -> tuple[LineupSchedule, Path, bool]:
    path_text = str(getattr(args, "fixed_pool_schedule_path", "") or "").strip()
    if not path_text:
        raise FixedPoolStudyRunError(
            "--fixed-pool-schedule-path is required for a study run."
        )
    path = Path(path_text).expanduser()
    create = bool(getattr(args, "fixed_pool_schedule_create", False))
    if path.exists():
        if create:
            raise FixedPoolStudyRunError(
                "--fixed-pool-schedule-create requires a new path; immutable schedule already exists."
            )
        return load_lineup_schedule(path, pool=pool), path, False
    if not create:
        raise FixedPoolStudyRunError(
            "Fixed schedule does not exist. Pass --fixed-pool-schedule-create to create it."
        )
    configured_games = int(getattr(args, "fixed_pool_games", 0) or 0)
    default_games = fixed_pool_study_budget(
        len(ACTIVE_PLAYER_SKILL_REGISTRY.descriptors)
    )["games"]
    game_count = configured_games if configured_games > 0 else default_games
    schedule = build_balanced_lineup_schedule(
        pool,
        game_count=game_count,
        seed=int(getattr(args, "fixed_pool_schedule_seed", 1)),
    )
    return schedule, write_lineup_schedule(schedule, path), True


def _checkpoint_env_params(
    checkpoint_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recover environment options recorded with a checkpoint for native eval."""
    payload = load_checkpoint_metadata(checkpoint_path)
    signature = inspect.signature(basketworld.HexagonBasketballEnv.__init__)
    accepted = (set(signature.parameters) - {"self", "render_mode"}) | set(
        _JAX_STATIC_ONLY_ENV_KEYS
    )
    merged: dict[str, Any] = {}
    for source_name in ("frozen_config", "env_config", "trainer_config"):
        source = payload.get(source_name)
        if isinstance(source, Mapping):
            merged.update({str(key): value for key, value in source.items()})
    optional_params = {
        key: value
        for key, value in merged.items()
        if key in accepted and value is not None
    }
    # These are hard study constraints, not mutable evaluation settings.
    if int(optional_params.get("players", 5) or 5) != 5:
        raise FixedPoolStudyRunError(
            "Fixed-pool studies require a 5v5 checkpoint/environment."
        )
    if not bool(optional_params.get("enable_multi_possession", False)):
        raise FixedPoolStudyRunError(
            "Fixed-pool studies require a multi-possession checkpoint/environment."
        )
    training_params = dict(payload.get("trainer_config", {}) or {})
    return optional_params, training_params


def _specialist_vs_balanced(
    raw_study_root: str | Path, model_report: Mapping[str, Any]
) -> dict[str, Any]:
    """Summarize model value for dynamically defined specialist/balanced players."""
    from basketworld_jax.eval.fixed_pool_results import load_fixed_pool_study_artifacts

    artifact = load_fixed_pool_study_artifacts(raw_study_root)
    players = artifact["pool_manifest"].get("players", [])
    descriptors = artifact["pool_manifest"].get("skill_descriptors", [])
    skill_keys = [
        str(descriptor.get("key"))
        for descriptor in descriptors
        if isinstance(descriptor, Mapping) and descriptor.get("key") is not None
    ]
    if not players or not skill_keys:
        return {"status": "unavailable", "reason": "missing_pool_skill_data"}
    values = {
        skill: [float(player["skills"][skill]) for player in players]
        for skill in skill_keys
    }
    means = {skill: sum(items) / len(items) for skill, items in values.items()}
    stds = {
        skill: (sum((value - means[skill]) ** 2 for value in items) / len(items)) ** 0.5
        for skill, items in values.items()
    }
    groups: dict[str, list[float]] = {"specialist": [], "balanced": []}
    player_groups: dict[str, dict[str, Any]] = {}
    rapm = model_report.get("adjusted_plus_minus", {}).get("players", {})
    for player in players:
        player_id = str(player["player_id"])
        z_scores = {
            skill: (
                (float(player["skills"][skill]) - means[skill]) / stds[skill]
                if stds[skill] > 1.0e-12
                else 0.0
            )
            for skill in skill_keys
        }
        ranked = sorted(z_scores.items(), key=lambda item: abs(item[1]), reverse=True)
        peak_skill, peak_score = ranked[0]
        second_peak = abs(ranked[1][1]) if len(ranked) > 1 else 0.0
        is_specialist = abs(peak_score) >= 1.0 and abs(peak_score) - second_peak >= 0.25
        group = "specialist" if is_specialist else "balanced"
        value = rapm.get(player_id, {}).get("net_rating_points_vs_replacement")
        if value is not None:
            groups[group].append(float(value))
        player_groups[player_id] = {
            "group": group,
            "dominant_skill": peak_skill,
            "dominant_skill_z_score": float(peak_score),
        }
    return {
        "definition": (
            "specialist: largest absolute study-pool skill z-score is at least 1.0 "
            "and exceeds the second-largest absolute z-score by at least 0.25"
        ),
        "groups": {
            name: {
                "player_count": len(values),
                "mean_rapm": sum(values) / len(values) if values else None,
            }
            for name, values in groups.items()
        },
        "players": player_groups,
    }


def _report_payload(
    *,
    raw_artifacts: Mapping[str, Any],
    model_report: Mapping[str, Any],
    checkpoint_path: Path,
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": FIXED_POOL_OPERATION_SCHEMA_VERSION,
        "kind": "fixed_pool_checkpoint_report",
        "checkpoint": {"path": str(checkpoint_path), "update_index": None},
        "raw_artifact_dependencies": dict(raw_artifacts),
        "schema_compatibility": model_report.get("schema_compatibility"),
        "action_mode": model_report.get("action_mode"),
        "direct_skill_ratings": model_report.get("direct_skill_ratings"),
        "player_rapm": model_report.get("adjusted_plus_minus"),
        "skill_coefficients": model_report.get("skill_regression"),
        "factor_impact": model_report.get("factor_models"),
        "conventional_three_factor_impact": model_report.get(
            "conventional_three_factor_models"
        ),
        "execution": dict(execution),
        "specialist_vs_balanced": _specialist_vs_balanced(
            raw_artifacts["root"], model_report
        ),
        "sizing_and_rank_stability": {
            "sizing": model_report.get("sizing"),
            "warnings": model_report.get("warnings"),
        },
    }


def _fixed_pool_execution_observer(
    *,
    checkpoint_path: Path,
    batch_size: int,
) -> tuple[Callable[[Mapping[str, Any]], None], list[dict[str, Any]]]:
    """Return a console heartbeat and a reportable per-batch timing ledger."""
    started_at = perf_counter()
    last_event_at = started_at
    events: list[dict[str, Any]] = []
    event_lock = Lock()

    def observe(raw_event: Mapping[str, Any]) -> None:
        nonlocal last_event_at
        with event_lock:
            now = perf_counter()
            event = {
                **dict(raw_event),
                "elapsed_sec": float(now - started_at),
                "phase_elapsed_sec": float(now - last_event_at),
            }
            last_event_at = now
            phase = str(event.get("phase", "unknown"))
            completed = event.get("completed_episodes")
            total = event.get("total_episodes")
            batch_count = event.get("batch_episode_count")
            progress = (
                f" completed={completed}/{total} batch={batch_count}"
                if completed is not None and total is not None and batch_count is not None
                else ""
            )
            if phase == "study_dispatch":
                progress += f" batch_size={batch_size}"
            if phase == "runner_ready":
                progress += (
                    f" batch_size={event.get('batch_size')} horizon={event.get('horizon')}"
                )
            if phase == "batch_running":
                progress += f" batch_elapsed={event.get('batch_elapsed_sec', 0.0):.1f}s"
            print(
                f"[fixed-pool] {phase} checkpoint={checkpoint_path.name}{progress} "
                f"elapsed={event['elapsed_sec']:.1f}s "
                f"phase_elapsed={event['phase_elapsed_sec']:.1f}s",
                flush=True,
            )
            events.append(event)

    return observe, events


def _fixed_pool_execution_summary(
    result: Mapping[str, Any],
    *,
    requested_batch_size: int,
    progress_events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Keep the execution facts needed to interpret a study's throughput."""
    diagnostics = result.get("eval_diagnostics")
    native_summary = (
        diagnostics.get("jax_native_summary", {})
        if isinstance(diagnostics, Mapping)
        else {}
    )
    if not isinstance(native_summary, Mapping):
        native_summary = {}
    # Only the configuration belongs in the immutable report.  Wall-clock
    # timing changes between otherwise identical replays and is therefore
    # returned to the caller as an observation rather than persisted as if it
    # were study evidence.
    return {
        "configuration": {
            "requested_batch_size": int(requested_batch_size),
            "actual_batch_size": native_summary.get("batch_size"),
            "horizon": native_summary.get("horizon"),
            # Preserve the seed the native evaluator actually used alongside
            # the launch configuration.  This remains useful if the report is
            # later reanalysed without the original CLI invocation.
            "eval_seed": native_summary.get("eval_seed"),
        },
        "runtime_observation": {
            "elapsed_sec": native_summary.get("elapsed_sec"),
            "episodes_per_sec": native_summary.get("episodes_per_sec"),
            "states_per_sec": native_summary.get("states_per_sec"),
            "progress_events": [dict(event) for event in progress_events],
        },
    }


def _study_reproducibility(
    args,
    *,
    pool: PlayerPool,
    schedule: LineupSchedule,
    action_mode: str,
    pool_path: Path,
    schedule_path: Path,
    extension_source_root: Path | None = None,
) -> dict[str, Any]:
    """Persist the inputs needed to reproduce or extend a completed study."""
    pool_payload = pool.to_payload()
    return {
        "pool_path": str(pool_path),
        "create_pool": False,
        "pool_size": len(pool.players),
        "pool_seed": pool_payload.get("pool_seed"),
        "schedule_path": str(schedule_path),
        "create_schedule": False,
        "game_count": len(schedule.games),
        "schedule_seed": int(schedule.schedule_seed),
        "action_mode": action_mode,
        "batch_size": int(getattr(args, "fixed_pool_batch_size", 64) or 64),
        "eval_seed": int(getattr(args, "fixed_pool_eval_seed", 0)),
        "ridge_alpha": getattr(args, "fixed_pool_ridge_alpha", None),
        "validation_fraction": float(
            getattr(args, "fixed_pool_validation_fraction", 0.2)
        ),
        "split_seed": int(getattr(args, "fixed_pool_split_seed", 0)),
        "bootstrap_samples": int(
            getattr(args, "fixed_pool_bootstrap_samples", 200)
        ),
        "bootstrap_seed": int(getattr(args, "fixed_pool_bootstrap_seed", 1)),
        "extension_source_root": (
            str(extension_source_root) if extension_source_root is not None else None
        ),
    }


def write_fixed_pool_model_reanalysis(
    raw_study_root: str | Path,
    *,
    report_root: str | Path | None = None,
    ridge_alpha: float | None = None,
    validation_fraction: float = 0.2,
    split_seed: int = 0,
    bootstrap_samples: int = 200,
    bootstrap_seed: int = 1,
) -> Path:
    """Write a versioned derived analysis without mutating raw study evidence.

    Model and diagnostic logic can improve after a study has completed.  This
    preserves both the original immutable report and a new, content-addressed
    analysis revision derived only from the same durable raw manifests/tables.
    """
    from basketworld_jax.eval.fixed_pool_results import load_fixed_pool_study_artifacts

    raw_root = Path(raw_study_root).expanduser().resolve()
    artifact = load_fixed_pool_study_artifacts(raw_root)
    manifest = artifact["manifest"]
    study_id = str(manifest["study_id"])
    metadata = dict((manifest.get("study_identity") or {}).get("metadata") or {})
    checkpoint_path = Path(
        str(metadata.get("unified_checkpoint_path", "unknown-checkpoint"))
    )
    raw_artifacts = {
        "study_id": study_id,
        "root": str(raw_root),
        "manifest": str(raw_root / "manifest.json"),
        "pool_manifest": str(raw_root / "pool_manifest.json"),
        "schedule_manifest": str(raw_root / "schedule_manifest.json"),
        "episodes": str(raw_root / "episodes.csv"),
        "possession_events": str(raw_root / "possession_events.csv"),
        "mlflow_artifact_path": None,
    }
    model_report = fit_fixed_pool_study_models(
        raw_root,
        ridge_alpha=(float(ridge_alpha) if ridge_alpha is not None else None),
        validation_fraction=float(validation_fraction),
        split_seed=int(split_seed),
        bootstrap_samples=int(bootstrap_samples),
        bootstrap_seed=int(bootstrap_seed),
    )
    report = _report_payload(
        raw_artifacts=raw_artifacts,
        model_report=model_report,
        checkpoint_path=checkpoint_path,
        execution={
            "analysis_source": "offline_raw_reanalysis",
            "ridge_alpha": (
                float(ridge_alpha) if ridge_alpha is not None else "selected_by_five_fold_cv"
            ),
            "validation_fraction": float(validation_fraction),
            "split_seed": int(split_seed),
            "bootstrap_samples": int(bootstrap_samples),
            "bootstrap_seed": int(bootstrap_seed),
        },
    )
    report["kind"] = "fixed_pool_checkpoint_reanalysis"
    report["reanalysis"] = {
        "schema_version": FIXED_POOL_REANALYSIS_SCHEMA_VERSION,
        "raw_study_id": study_id,
        "derived_from": "immutable_raw_study_artifacts",
    }
    report["checkpoint"]["update_index"] = metadata.get(
        "unified_checkpoint_update_index"
    )
    root = (
        Path(report_root).expanduser()
        if report_root is not None
        else raw_root.parent.parent
    )
    path = (
        root
        / FIXED_POOL_REPORT_NAMESPACE
        / study_id
        / "analysis_revisions"
        / f"v{FIXED_POOL_REANALYSIS_SCHEMA_VERSION}.json"
    )
    _write_immutable_json(path, report)
    return path


def _longitudinal_payload(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    compatibility_keys = {
        (
            str(report.get("schema_compatibility", {}).get("skill_schema_hash")),
            tuple(
                report.get("schema_compatibility", {}).get(
                    "checkpoint_observation_schema_versions", []
                )
                or []
            ),
            report.get("schema_compatibility", {}).get(
                "environment_observation_schema_version"
            ),
        )
        for report in reports
    }
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for report in reports:
        by_mode.setdefault(str(report.get("action_mode")), []).append(
            {
                "study_id": report.get("raw_artifact_dependencies", {}).get("study_id"),
                "report_path": report.get("report_path"),
                "checkpoint": report.get("checkpoint"),
                "schema_compatibility": report.get("schema_compatibility"),
            }
        )
    compatible = len(compatibility_keys) <= 1
    return {
        "schema_version": FIXED_POOL_OPERATION_SCHEMA_VERSION,
        "kind": "fixed_pool_longitudinal_view",
        "compatible": compatible,
        "schema_hashes": sorted({key[0] for key in compatibility_keys}),
        "observation_schemas": [list(key[1]) for key in sorted(compatibility_keys)],
        "warnings": [] if compatible else ["cross_schema_comparison_not_reported"],
        "reports_by_action_mode": by_mode,
        "action_modes_are_separate": True,
    }


def _start_mlflow_if_requested(args):
    if not bool(getattr(args, "log_mlflow", False)):
        return None, nullcontext()
    import mlflow

    setup_mlflow(verbose=False)
    mlflow.set_experiment(str(getattr(args, "mlflow_experiment_name", "Default")))
    return mlflow, mlflow.start_run(
        run_name=str(getattr(args, "mlflow_run_name", "") or "fixed-pool-study")
    )


def run_fixed_pool_study(
    args,
    *,
    native_runner: Callable[..., Mapping[str, Any]] = run_native_jax_evaluation,
) -> dict[str, Any]:
    """Create/load inputs, evaluate requested checkpoints, and write reports."""
    checkpoint_paths = _parse_checkpoint_paths(args)
    pool, pool_path, pool_created = _load_or_create_pool(args)
    extension_source_text = str(
        getattr(args, "fixed_pool_extension_source", "") or ""
    ).strip()
    extension_source_root: Path | None = None
    combined_schedule: LineupSchedule | None = None
    if extension_source_text:
        extension_source_root = Path(extension_source_text).expanduser().resolve()
        source = load_fixed_pool_study_artifacts(extension_source_root)
        base_schedule = load_lineup_schedule(
            extension_source_root / "schedule_manifest.json", pool=pool
        )
        source_pool_id = str((source.get("manifest") or {}).get("pool_id", ""))
        if source_pool_id != pool.pool_id:
            raise FixedPoolStudyRunError(
                "Extension source study does not use the requested player pool."
            )
        additional_games = int(getattr(args, "fixed_pool_games", 0) or 0)
        if additional_games < 1:
            raise FixedPoolStudyRunError(
                "An extension requires --fixed-pool-games to be the number of new games."
            )
        extension = build_lineup_schedule_extension(
            pool,
            base_schedule,
            additional_game_count=additional_games,
            seed=int(getattr(args, "fixed_pool_schedule_seed", 1)),
        )
        schedule_path_text = str(
            getattr(args, "fixed_pool_schedule_path", "") or ""
        ).strip()
        if not schedule_path_text:
            raise FixedPoolStudyRunError(
                "--fixed-pool-schedule-path is required for an extension."
            )
        schedule_path = Path(schedule_path_text).expanduser()
        if schedule_path.exists():
            raise FixedPoolStudyRunError(
                "An extension schedule path must be new and immutable."
            )
        if not bool(getattr(args, "fixed_pool_schedule_create", False)):
            raise FixedPoolStudyRunError(
                "An extension must create its new schedule artifact."
            )
        schedule = extension.additional_schedule
        schedule_path = write_lineup_schedule(schedule, schedule_path)
        schedule_created = True
        combined_schedule = extension.combined_schedule
    else:
        schedule, schedule_path, schedule_created = _load_or_create_schedule(args, pool)
    artifact_root_text = str(getattr(args, "fixed_pool_artifact_dir", "") or "").strip()
    if not artifact_root_text:
        raise FixedPoolStudyRunError("--fixed-pool-artifact-dir is required.")
    artifact_root = Path(artifact_root_text).expanduser()
    report_root = Path(
        str(getattr(args, "fixed_pool_report_dir", "") or artifact_root)
    ).expanduser()
    action_mode = str(getattr(args, "fixed_pool_action_mode", "direct"))
    deterministic = action_mode == "direct"
    if action_mode not in {"direct", "sampled"}:
        raise FixedPoolStudyRunError(
            "fixed-pool action mode must be direct or sampled."
        )
    mlflow, context = _start_mlflow_if_requested(args)
    written_reports: list[dict[str, Any]] = []
    with context:
        for checkpoint_path in checkpoint_paths:
            requested_batch_size = int(
                getattr(args, "fixed_pool_batch_size", 64) or 64
            )
            progress_callback, progress_events = _fixed_pool_execution_observer(
                checkpoint_path=checkpoint_path,
                batch_size=requested_batch_size,
            )
            progress_callback(
                {
                    "phase": "study_dispatch",
                    "total_episodes": len(schedule.games),
                }
            )
            progress_callback({"phase": "checkpoint_environment_loading"})
            optional_params, training_params = _checkpoint_env_params(checkpoint_path)
            progress_callback({"phase": "checkpoint_environment_loaded"})
            # Restore policy tensors exactly once, at a phase the CLI and a
            # future background-job UI can report independently.  The native
            # evaluator receives this in-memory payload rather than making a
            # second, unobservable Orbax restore of the same checkpoint.
            progress_callback({"phase": "checkpoint_restore_loading"})
            unified_checkpoint_payload = load_checkpoint(checkpoint_path)
            progress_callback({"phase": "checkpoint_restore_loaded"})
            result = native_runner(
                num_episodes=len(schedule.games),
                player_deterministic=deterministic,
                opponent_deterministic=deterministic,
                required_params={},
                optional_params=optional_params,
                training_params=training_params,
                unified_policy_path=str(checkpoint_path),
                opponent_policy_path=None,
                user_team_name="OFFENSE",
                role_flag_offense=1.0,
                role_flag_defense=-1.0,
                eval_seed=int(getattr(args, "fixed_pool_eval_seed", 0)),
                fixed_player_pool=pool,
                fixed_lineup_schedule=schedule,
                fixed_pool_artifact_dir=artifact_root,
                fixed_pool_mlflow_client=mlflow,
                evaluation_batch_size=requested_batch_size,
                progress_event_callback=progress_callback,
                unified_checkpoint_payload=unified_checkpoint_payload,
            )
            progress_callback({"phase": "native_evaluation_completed"})
            raw_artifacts = result.get("fixed_pool_artifacts")
            if not isinstance(raw_artifacts, Mapping):
                raise FixedPoolStudyRunError(
                    "Native fixed-pool evaluation did not persist raw artifacts."
                )
            if combined_schedule is not None and extension_source_root is not None:
                combined_artifacts = combine_fixed_pool_study_artifacts(
                    pool=pool,
                    combined_schedule=combined_schedule,
                    base_study_root=extension_source_root,
                    additional_study_root=raw_artifacts["root"],
                    output_dir=artifact_root,
                )
                raw_artifacts = combined_artifacts.to_payload()
            progress_callback({"phase": "model_fitting_started"})
            model_report = fit_fixed_pool_study_models(
                raw_artifacts["root"],
                ridge_alpha=(
                    float(args.fixed_pool_ridge_alpha)
                    if getattr(args, "fixed_pool_ridge_alpha", None) is not None
                    else None
                ),
                validation_fraction=float(
                    getattr(args, "fixed_pool_validation_fraction", 0.2)
                ),
                split_seed=int(getattr(args, "fixed_pool_split_seed", 0)),
                bootstrap_samples=int(
                    getattr(args, "fixed_pool_bootstrap_samples", 200)
                ),
                bootstrap_seed=int(getattr(args, "fixed_pool_bootstrap_seed", 1)),
            )
            progress_callback({"phase": "model_fitting_completed"})
            execution = _fixed_pool_execution_summary(
                result,
                requested_batch_size=requested_batch_size,
                progress_events=progress_events,
            )
            execution["configuration"]["reproducibility"] = _study_reproducibility(
                args,
                pool=pool,
                schedule=(combined_schedule or schedule),
                action_mode=action_mode,
                pool_path=Path(str(raw_artifacts["pool_manifest"])),
                schedule_path=Path(str(raw_artifacts["schedule_manifest"])),
                extension_source_root=extension_source_root,
            )
            report = _report_payload(
                raw_artifacts=raw_artifacts,
                model_report=model_report,
                checkpoint_path=checkpoint_path,
                execution=execution["configuration"],
            )
            # The raw manifest contains the authoritative update ID.  Surface it
            # directly rather than parsing a checkpoint filename.
            episode_rows = result.get("results", [])
            if episode_rows:
                report["checkpoint"]["update_index"] = (
                    episode_rows[0]
                    .get("fixed_pool_study", {})
                    .get("unified_checkpoint_update_index")
                )
            report_path = (
                report_root
                / FIXED_POOL_REPORT_NAMESPACE
                / str(raw_artifacts["study_id"])
                / "checkpoint_report.json"
            )
            progress_callback({"phase": "report_writing_started"})
            _write_immutable_json(report_path, report)
            progress_callback({"phase": "report_writing_completed"})
            report["report_path"] = str(report_path)
            report["runtime_observation"] = execution["runtime_observation"]
            if mlflow is not None:
                mlflow.log_artifact(
                    str(report_path),
                    artifact_path=f"{FIXED_POOL_REPORT_NAMESPACE}/{raw_artifacts['study_id']}",
                )
            written_reports.append(report)
        longitudinal = _longitudinal_payload(written_reports)
        longitudinal_path = (
            report_root / FIXED_POOL_REPORT_NAMESPACE / "longitudinal.json"
        )
        _write_immutable_json(longitudinal_path, longitudinal)
        if mlflow is not None:
            mlflow.log_artifact(
                str(longitudinal_path), artifact_path=FIXED_POOL_REPORT_NAMESPACE
            )
    return {
        "enabled": True,
        "pool": {"path": str(pool_path), "id": pool.pool_id, "created": pool_created},
        "schedule": {
            "path": str(schedule_path),
            "id": schedule.schedule_id,
            "created": schedule_created,
            "game_count": len(schedule.games),
        },
        "action_mode": action_mode,
        "checkpoint_reports": written_reports,
        "longitudinal_report_path": str(longitudinal_path),
        "mlflow_enabled": mlflow is not None,
    }
