from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import HTTPException

from app.backend.state import game_state
from basketworld_jax.eval import can_run_native_jax_evaluation, run_native_jax_evaluation
from basketworld_jax.eval.skills import (
    ACTIVE_PLAYER_SKILL_REGISTRY,
    PlayerSkillValidationError,
)
from basketworld_jax.inference import require_checkpoint_path


def validate_custom_eval_setup(custom_setup, env) -> dict:
    """Validate and normalize a custom eval setup against the active environment."""
    if not custom_setup:
        return {}
    # Convert to plain dict (accept pydantic model or dict)
    setup = custom_setup.dict() if hasattr(custom_setup, "dict") else dict(custom_setup)
    if env is None:
        raise HTTPException(
            status_code=400, detail="Environment not initialized for custom setup validation."
        )
    base_env = getattr(env, "unwrapped", env)

    normalized: dict = {}
    shooting_mode = str(setup.get("shooting_mode") or "random").lower()
    if shooting_mode not in ("random", "fixed"):
        raise HTTPException(status_code=400, detail=f"Invalid shooting_mode: {shooting_mode}")
    normalized["shooting_mode"] = shooting_mode

    if setup.get("player_skills") is not None:
        raw_profiles = setup.get("player_skills")
        if not isinstance(raw_profiles, dict):
            raise HTTPException(status_code=400, detail="player_skills must be keyed by player id.")
        expected_ids = set(range(int(base_env.n_players)))
        parsed_ids: set[int] = set()
        normalized_profiles: dict[str, dict[str, Any]] = {}
        for raw_player_id, raw_profile in raw_profiles.items():
            try:
                player_id = int(raw_player_id)
            except Exception:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid player_skills player id: {raw_player_id}",
                )
            if player_id not in expected_ids or player_id in parsed_ids:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid or duplicate player_skills player id: {raw_player_id}",
                )
            if not isinstance(raw_profile, dict):
                raise HTTPException(
                    status_code=400,
                    detail=f"player_skills.{player_id} must be an object.",
                )
            try:
                profile = ACTIVE_PLAYER_SKILL_REGISTRY.normalize_profile(raw_profile)
            except PlayerSkillValidationError as exc:
                if exc.missing_keys:
                    detail = (
                        f"player_skills.{player_id} is missing: "
                        f"{', '.join(exc.missing_keys)}"
                    )
                elif exc.skill_key is not None:
                    detail = f"player_skills.{player_id}.{exc.skill_key} {exc}"
                else:
                    detail = f"player_skills.{player_id} {exc}"
                raise HTTPException(status_code=400, detail=detail) from exc
            parsed_ids.add(player_id)
            normalized_profiles[str(player_id)] = profile
        if parsed_ids != expected_ids:
            missing_ids = sorted(expected_ids - parsed_ids)
            raise HTTPException(
                status_code=400,
                detail=f"player_skills must define every player; missing ids: {missing_ids}",
            )
        normalized["player_skills"] = normalized_profiles

    if setup.get("initial_positions") is not None:
        positions_raw = setup["initial_positions"]
        if not isinstance(positions_raw, (list, tuple)):
            raise HTTPException(status_code=400, detail="initial_positions must be a list.")
        if len(positions_raw) != base_env.n_players:
            raise HTTPException(
                status_code=400,
                detail=f"initial_positions must have {base_env.n_players} entries (one per player).",
            )
        positions: list[tuple[int, int]] = []
        seen = set()
        for pos in positions_raw:
            if not isinstance(pos, (list, tuple)) or len(pos) != 2:
                raise HTTPException(status_code=400, detail=f"Invalid position entry: {pos}")
            q, r = int(pos[0]), int(pos[1])
            if not base_env._is_valid_position(q, r):
                raise HTTPException(status_code=400, detail=f"Position {(q, r)} is out of bounds.")
            if (q, r) in seen:
                raise HTTPException(status_code=400, detail=f"Duplicate position {(q, r)} is not allowed.")
            seen.add((q, r))
            positions.append((q, r))
        normalized["initial_positions"] = positions

    if setup.get("ball_holder") is not None:
        bh = int(setup["ball_holder"])
        if bh < 0 or bh >= base_env.n_players:
            raise HTTPException(status_code=400, detail=f"Invalid ball_holder id: {bh}")
        normalized["ball_holder"] = bh

    # Validate fixed offense skills if requested
    if setup.get("rebound_skills") is not None:
        rebound_raw = setup.get("rebound_skills")
        if not isinstance(rebound_raw, (list, tuple)) or len(rebound_raw) != base_env.n_players:
            raise HTTPException(
                status_code=400,
                detail=f"rebound_skills must have {base_env.n_players} values.",
            )
        rebound_values: list[float] = []
        for v in rebound_raw:
            try:
                rebound_values.append(float(v))
            except Exception:
                raise HTTPException(status_code=400, detail=f"Invalid rebound skill value: {v}")
        normalized["rebound_skills"] = rebound_values

    if setup.get("rebound_skill_sampling") is not None and normalized.get("rebound_skills") is None:
        raw_sampling = setup.get("rebound_skill_sampling")
        sampling = raw_sampling.dict() if hasattr(raw_sampling, "dict") else dict(raw_sampling)
        mode = str(sampling.get("mode") or "constrained_gaussian")
        if mode != "constrained_gaussian":
            raise HTTPException(status_code=400, detail=f"Invalid rebound_skill_sampling.mode: {mode}")
        try:
            std = float(sampling.get("std", 1.0))
            target_edge = float(sampling.get("target_edge", 0.0))
            tolerance = float(sampling.get("tolerance", 0.25))
            max_attempts = int(sampling.get("max_attempts", 5000))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid rebound_skill_sampling value: {exc}")
        if std <= 0.0:
            raise HTTPException(status_code=400, detail="rebound_skill_sampling.std must be > 0.")
        if tolerance < 0.0:
            raise HTTPException(status_code=400, detail="rebound_skill_sampling.tolerance must be >= 0.")
        if max_attempts < 1:
            raise HTTPException(status_code=400, detail="rebound_skill_sampling.max_attempts must be >= 1.")
        normalized["rebound_skill_sampling"] = {
            "mode": mode,
            "std": std,
            "target_edge": target_edge,
            "tolerance": tolerance,
            "max_attempts": max_attempts,
        }

    if shooting_mode == "fixed" and normalized.get("player_skills") is None:
        offense_ids = getattr(base_env, "offense_ids", [])
        offense_count = len(offense_ids)
        skills = setup.get("offense_skills")
        if not skills or not isinstance(skills, dict):
            raise HTTPException(
                status_code=400, detail="offense_skills are required when shooting_mode='fixed'."
            )
        normalized_skills: dict[str, list[float]] = {}
        for key in ("layup", "three_pt", "dunk"):
            arr = skills.get(key)
            if arr is None or len(arr) != offense_count:
                raise HTTPException(
                    status_code=400,
                    detail=f"offense_skills.{key} must have {offense_count} values.",
                )
            vals: list[float] = []
            for v in arr:
                try:
                    fv = float(v)
                except Exception:
                    raise HTTPException(status_code=400, detail=f"Invalid offense skill value: {v}")
                if fv < 0 or fv > 1:
                    raise HTTPException(
                        status_code=400, detail="Offense skill probabilities must be between 0 and 1."
                    )
                vals.append(fv)
            normalized_skills[key] = vals
        normalized["offense_skills"] = normalized_skills

    return normalized


def run_evaluation(
    num_episodes: int,
    player_deterministic: bool,
    opponent_deterministic: bool,
    required_params: dict,
    optional_params: dict,
    training_params: dict | None,
    unified_policy_path: str,
    opponent_policy_path: str | None,
    user_team_name: str,
    role_flag_offense: float,
    role_flag_defense: float,
    shot_accumulator: dict[str, list[int]] | None = None,
    custom_setup: dict | None = None,
    randomize_offense_permutation: bool = False,
    intent_selection_mode: str = "learned_sample",
    num_workers: int | None = None,
    progress_callback=None,
    eval_seed: int | None = None,
):
    native_ready = can_run_native_jax_evaluation(
        unified_policy_path=unified_policy_path,
        opponent_policy_path=opponent_policy_path,
        custom_setup=custom_setup,
        randomize_offense_permutation=randomize_offense_permutation,
    )
    if not native_ready:
        require_checkpoint_path(unified_policy_path)
        if opponent_policy_path:
            require_checkpoint_path(opponent_policy_path)
        raise ValueError(
            "The requested evaluation options are not supported by the JAX-native evaluator."
        )
    print("[Evaluation] Using JAX-native compiled evaluation")
    return run_native_jax_evaluation(
        num_episodes=num_episodes,
        player_deterministic=player_deterministic,
        opponent_deterministic=opponent_deterministic,
        required_params=required_params,
        optional_params=optional_params,
        training_params=training_params,
        unified_policy_path=unified_policy_path,
        opponent_policy_path=opponent_policy_path,
        user_team_name=user_team_name,
        role_flag_offense=role_flag_offense,
        role_flag_defense=role_flag_defense,
        intent_selection_mode=intent_selection_mode,
        custom_setup=custom_setup,
        progress_callback=progress_callback,
        eval_seed=eval_seed,
    )


def pass_steal_preview(_env, positions: list[tuple[int, int]], ball_holder: int):
    """Return JAX pass-steal and policy probabilities for a hypothetical placement."""
    runtime = getattr(game_state, "jax_runtime", None)
    if runtime is None or runtime.state is None:
        raise ValueError("JAX environment not initialized.")
    if len(positions) != int(runtime.n_players):
        raise ValueError(
            f"positions must have {runtime.n_players} entries (got {len(positions)})"
        )
    validated = []
    seen = set()
    for raw in positions:
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise ValueError(f"Invalid position entry: {raw}")
        pos = (int(raw[0]), int(raw[1]))
        if not runtime.environment._is_valid_position(*pos):
            raise ValueError(f"Position {pos} is out of bounds.")
        if pos in seen:
            raise ValueError(f"Duplicate position {pos} not allowed.")
        seen.add(pos)
        validated.append(pos)
    holder = int(ball_holder)
    if holder < 0 or holder >= int(runtime.n_players):
        raise ValueError(f"Invalid ball_holder id: {holder}")

    snapshot = runtime.capture_snapshot()
    try:
        runtime.state = runtime.state._replace(
            positions=runtime.jnp.asarray([validated], dtype=runtime.jnp.int32),
            ball_holder=runtime.jnp.asarray([holder], dtype=runtime.jnp.int32),
        )
        runtime._sync_environment()
        _, probs = runtime._choose_joint_policy_actions(
            player_deterministic=False,
            opponent_deterministic=False,
            include_diagnostics=True,
        )
        return {
            "steal_probabilities": runtime.pass_steal_probabilities(),
            "policy_probabilities": (
                None
                if probs is None
                else [
                    [float(x) for x in np.asarray(probs.get(pid, [])).reshape(-1)]
                    for pid in range(int(runtime.n_players))
                ]
            ),
        }
    finally:
        runtime.restore_snapshot(snapshot, game_state=game_state)
