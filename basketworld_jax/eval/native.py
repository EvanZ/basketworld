from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

import basketworld
from basketworld.envs.basketworld_env_v2 import ActionType, Team
from basketworld_jax.checkpoints import load_checkpoint
from basketworld_jax.env.minimal import (
    MOVE_ACTION_END,
    MOVE_ACTION_START,
    PASS_ACTION_END,
    PASS_ACTION_START,
    SHOT_TYPE_DUNK,
    SHOT_TYPE_THREE,
    TURNOVER_REASON_DEFENDER_PRESSURE,
    TURNOVER_REASON_INTERCEPTED,
    TURNOVER_REASON_MOVE_OUT_OF_BOUNDS,
    TURNOVER_REASON_OFFENSIVE_THREE_SECONDS,
    TURNOVER_REASON_PASS_OUT_OF_BOUNDS,
    TURNOVER_REASON_SHOT_CLOCK,
    assemble_full_actions_jax,
    build_action_masks_batch,
    build_kernel_static_from_env,
    build_policy_observation_batch_with_role_flag,
    reset_batch_minimal,
    step_batch_minimal,
)
from basketworld_jax.inference import is_checkpoint_path
from basketworld_jax.models import ActorCriticSpec, actor_critic_forward, apply_action_mask
from basketworld_jax.train.cli import ensure_jax_available


def can_run_native_jax_evaluation(
    *,
    unified_policy_path: str,
    opponent_policy_path: str | None,
    custom_setup: dict | None,
    randomize_offense_permutation: bool,
) -> bool:
    if custom_setup:
        return False
    if bool(randomize_offense_permutation):
        return False
    if not is_checkpoint_path(unified_policy_path):
        return False
    return opponent_policy_path is None or is_checkpoint_path(opponent_policy_path)


def _load_checkpoint_params(path: str | Path, jax) -> tuple[dict[str, Any], Any, ActorCriticSpec]:
    payload = load_checkpoint(path)
    params = jax.device_put(payload["params"])
    spec = ActorCriticSpec(**dict(payload["policy_spec"]))
    return payload, params, spec


def _build_native_eval_runner(jax, jnp, spec: ActorCriticSpec):
    def _runner(
        static,
        initial_state,
        offense_params,
        defense_params,
        eval_key,
        role_flag_offense,
        role_flag_defense,
        horizon: int,
        offense_deterministic: bool,
        defense_deterministic: bool,
    ):
        n_players = int(static.role_encoding.shape[0])
        offense_ids = static.offense_ids.astype(jnp.int32)
        defense_ids = static.defense_ids.astype(jnp.int32)

        def _team_actions(params, flat_obs, action_mask, key, deterministic: bool):
            forward_out = actor_critic_forward(params, flat_obs, spec, jnp)
            masked_out = apply_action_mask(
                forward_out["flat_policy_logits"],
                action_mask,
                spec,
                jax,
                jnp,
            )
            if deterministic:
                return masked_out["deterministic_actions"]
            return jax.random.categorical(
                key,
                masked_out["masked_logits"],
                axis=-1,
            ).astype(jnp.int32)

        def _scan_step(carry, _):
            state, key = carry
            key, offense_key, defense_key, env_key = jax.random.split(key, 4)
            full_action_mask = build_action_masks_batch(static, state, jnp)
            offense_mask = full_action_mask[:, offense_ids, :]
            defense_mask = full_action_mask[:, defense_ids, :]
            offense_obs = build_policy_observation_batch_with_role_flag(
                static,
                state,
                role_flag_offense,
                jnp,
                model_type=spec.model_type,
            )
            defense_obs = build_policy_observation_batch_with_role_flag(
                static,
                state,
                role_flag_defense,
                jnp,
                model_type=spec.model_type,
            )
            offense_actions = _team_actions(
                offense_params,
                offense_obs,
                offense_mask,
                offense_key,
                offense_deterministic,
            )
            defense_actions = _team_actions(
                defense_params,
                defense_obs,
                defense_mask,
                defense_key,
                defense_deterministic,
            )
            full_actions = assemble_full_actions_jax(
                offense_actions,
                defense_actions,
                offense_ids,
                defense_ids,
                n_players,
                jnp,
            )
            env_keys = jax.random.split(env_key, initial_state.positions.shape[0])
            env_out = step_batch_minimal(static, state, full_actions, env_keys, jax, jnp)
            trace = {
                "full_actions": full_actions.astype(jnp.int32),
                "done": env_out.done.astype(jnp.int8),
                "terminal_episode_steps": env_out.terminal_episode_steps.astype(jnp.int32),
                "offense_rewards": jnp.sum(env_out.rewards[:, offense_ids], axis=1),
                "defense_rewards": jnp.sum(env_out.rewards[:, defense_ids], axis=1),
                "offense_score_delta": (
                    env_out.state.offense_score - state.offense_score
                ).astype(jnp.float32),
                "defense_score_delta": (
                    env_out.state.defense_score - state.defense_score
                ).astype(jnp.float32),
                "pass_attempts": env_out.pass_attempt.astype(jnp.int8),
                "pass_passer": env_out.pass_passer.astype(jnp.int32),
                "pass_receiver": env_out.pass_receiver.astype(jnp.int32),
                "completed_passes": env_out.completed_pass.astype(jnp.int8),
                "assists": env_out.assist.astype(jnp.int8),
                "turnovers": env_out.turnover.astype(jnp.int8),
                "shot_attempt": env_out.shot_attempt.astype(jnp.int8),
                "shot_success": env_out.shot_success.astype(jnp.int8),
                "shot_shooter": env_out.shot_shooter.astype(jnp.int32),
                "shot_value": env_out.shot_value.astype(jnp.float32),
                "shot_expected_points": env_out.shot_expected_points.astype(jnp.float32),
                "shot_distance": env_out.shot_distance.astype(jnp.float32),
                "shot_type": env_out.shot_type.astype(jnp.int32),
                "shot_q": env_out.shot_q.astype(jnp.int32),
                "shot_r": env_out.shot_r.astype(jnp.int32),
                "potential_assist": env_out.potential_assist.astype(jnp.int8),
                "assist_passer": env_out.assist_passer.astype(jnp.int32),
                "turnover_player": env_out.turnover_player.astype(jnp.int32),
                "turnover_reason": env_out.turnover_reason.astype(jnp.int32),
                "offensive_three_seconds": env_out.offensive_three_seconds.astype(jnp.int8),
                "defensive_lane_violation": env_out.defensive_lane_violation.astype(jnp.int8),
                "defensive_lane_violation_player": env_out.defensive_lane_violation_player.astype(jnp.int32),
            }
            return (env_out.state, key), trace

        (_, _), trace = jax.lax.scan(
            _scan_step,
            (initial_state, eval_key),
            xs=None,
            length=int(horizon),
        )
        return trace

    return jax.jit(_runner, static_argnums=(7, 8, 9))


def _episode_stats_from_trace(trace: dict[str, np.ndarray], *, take: int, horizon: int) -> dict[str, np.ndarray]:
    terminal_steps = np.max(np.asarray(trace["terminal_episode_steps"])[:, :take], axis=0)
    completed = terminal_steps > 0
    steps = np.where(completed, terminal_steps, int(horizon)).astype(np.int32)
    return {
        "steps": steps,
        "completed": completed.astype(np.int8),
        "offense_rewards": np.asarray(trace["offense_rewards"])[:, :take].sum(axis=0),
        "defense_rewards": np.asarray(trace["defense_rewards"])[:, :take].sum(axis=0),
        "offense_points": np.asarray(trace["offense_score_delta"])[:, :take].sum(axis=0),
        "defense_points": np.asarray(trace["defense_score_delta"])[:, :take].sum(axis=0),
        "pass_attempts": np.asarray(trace["pass_attempts"])[:, :take].sum(axis=0),
        "completed_passes": np.asarray(trace["completed_passes"])[:, :take].sum(axis=0),
        "assists": np.asarray(trace["assists"])[:, :take].sum(axis=0),
        "turnovers": np.asarray(trace["turnovers"])[:, :take].sum(axis=0),
    }


def _mean(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float32)
    return float(arr.mean()) if arr.size else 0.0


def _sum(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float32)
    return float(arr.sum()) if arr.size else 0.0


def _init_player_stats(n_players: int) -> dict[int, dict[str, Any]]:
    return {
        pid: {
            "shots": 0,
            "makes": 0,
            "shot_types": {"dunk": [0, 0], "two": [0, 0], "three": [0, 0]},
            "assist_full_by_type": {"dunk": 0, "two": 0, "three": 0},
            "assists": 0,
            "potential_assists": 0,
            "turnovers": 0,
            "points": 0.0,
            "episodes": 0,
            "steps": 0,
            "shot_chart": {},
            "unassisted": {"dunk": 0, "two": 0, "three": 0},
        }
        for pid in range(int(n_players))
    }


def _init_aggregate_stats() -> dict[str, Any]:
    return {
        "shots": 0,
        "makes": 0,
        "shot_types": {"dunk": [0, 0], "two": [0, 0], "three": [0, 0]},
        "assist_full_by_type": {"dunk": 0, "two": 0, "three": 0},
        "assists": 0,
        "potential_assists": 0,
        "turnovers": 0,
        "points": 0.0,
        "episodes": 0,
        "steps": 0,
        "shot_chart": {},
        "unassisted": {"dunk": 0, "two": 0, "three": 0},
    }


def _init_eval_diagnostics() -> dict[str, Any]:
    return {
        "intent_selection_counts": {},
        "intent_inactive_count": 0,
        "turnover_reasons": {},
        "assist_links": {},
        "assist_links_by_type": {"dunk": {}, "two": {}, "three": {}},
        "potential_assist_links": {},
        "potential_assist_links_by_type": {"dunk": {}, "two": {}, "three": {}},
        "pass_links": {},
        "completed_pass_links": {},
        "shot_attempts_by_player": {},
        "made_shots_by_player": {},
        "action_mix": {
            "noop": 0,
            "move": 0,
            "shoot": 0,
            "pass": 0,
            "other": 0,
            "total": 0,
        },
        "reward_breakdown": {
            "total_reward": 0.0,
            "expected_points": 0.0,
            "pass_reward": 0.0,
            "violation_reward": 0.0,
            "assist_potential": 0.0,
            "assist_full_bonus": 0.0,
            "phi_shaping": 0.0,
            "unexplained": 0.0,
        },
    }


def _merge_aggregate_stats(dest: dict[str, Any] | None, src: dict[str, Any] | None) -> dict[str, Any]:
    if dest is None:
        dest = _init_aggregate_stats()
    if not src:
        return dest

    for key in ("shots", "makes", "assists", "potential_assists", "turnovers", "episodes", "steps"):
        dest[key] = int(dest.get(key, 0) or 0) + int(src.get(key, 0) or 0)
    dest["points"] = float(dest.get("points", 0.0) or 0.0) + float(src.get("points", 0.0) or 0.0)

    for shot_type in ("dunk", "two", "three"):
        src_pair = (src.get("shot_types") or {}).get(shot_type, [0, 0])
        dst_pair = dest["shot_types"].setdefault(shot_type, [0, 0])
        dst_pair[0] += int(src_pair[0] if isinstance(src_pair, (list, tuple)) else 0)
        dst_pair[1] += int(src_pair[1] if isinstance(src_pair, (list, tuple)) else 0)
        dest["assist_full_by_type"][shot_type] = int(dest["assist_full_by_type"].get(shot_type, 0)) + int(
            (src.get("assist_full_by_type") or {}).get(shot_type, 0) or 0
        )
        dest["unassisted"][shot_type] = int(dest["unassisted"].get(shot_type, 0)) + int(
            (src.get("unassisted") or {}).get(shot_type, 0) or 0
        )

    for loc, vals in (src.get("shot_chart") or {}).items():
        dst_pair = dest["shot_chart"].setdefault(str(loc), [0, 0])
        dst_pair[0] += int(vals[0] if isinstance(vals, (list, tuple)) and len(vals) > 0 else 0)
        dst_pair[1] += int(vals[1] if isinstance(vals, (list, tuple)) and len(vals) > 1 else 0)
    return dest


def _accumulate_team_stats_from_players(
    dest: dict[str, Any] | None,
    player_stats: dict[int, dict[str, Any]],
    team_ids: list[int],
    *,
    episodes: int,
    steps: int,
) -> dict[str, Any]:
    merged = _init_aggregate_stats()
    merged["episodes"] = int(episodes)
    merged["steps"] = int(steps)
    for pid in team_ids:
        entry = player_stats.get(int(pid))
        if entry:
            merged = _merge_aggregate_stats(merged, entry)
    return _merge_aggregate_stats(dest, merged)


def _shot_type_label(code: int) -> str:
    if int(code) == int(SHOT_TYPE_DUNK):
        return "dunk"
    if int(code) == int(SHOT_TYPE_THREE):
        return "three"
    return "two"


def _turnover_reason_label(code: int) -> str:
    labels = {
        int(TURNOVER_REASON_PASS_OUT_OF_BOUNDS): "pass_out_of_bounds",
        int(TURNOVER_REASON_INTERCEPTED): "intercepted",
        int(TURNOVER_REASON_DEFENDER_PRESSURE): "defender_pressure",
        int(TURNOVER_REASON_MOVE_OUT_OF_BOUNDS): "move_out_of_bounds",
        int(TURNOVER_REASON_SHOT_CLOCK): "shot_clock_violation",
        int(TURNOVER_REASON_OFFENSIVE_THREE_SECONDS): "offensive_three_seconds",
    }
    return labels.get(int(code), "unknown")


def _action_bucket(action_id: int) -> str:
    aid = int(action_id)
    if aid == int(ActionType.NOOP.value):
        return "noop"
    if int(MOVE_ACTION_START) <= aid < int(MOVE_ACTION_END):
        return "move"
    if aid == int(ActionType.SHOOT.value):
        return "shoot"
    if int(PASS_ACTION_START) <= aid < int(PASS_ACTION_END):
        return "pass"
    return "other"


def _record_action_mix(eval_diagnostics: dict[str, Any], actions: np.ndarray, user_team_ids: list[int]) -> None:
    action_mix = eval_diagnostics["action_mix"]
    for pid in user_team_ids:
        bucket = _action_bucket(int(actions[int(pid)]))
        action_mix[bucket] = int(action_mix.get(bucket, 0)) + 1
        action_mix["total"] = int(action_mix.get("total", 0)) + 1


def _record_shot_event(
    *,
    stats: dict[int, dict[str, Any]],
    shot_accumulator: dict[str, list[int]] | None,
    shooter_id: int,
    success: bool,
    shot_value: float,
    shot_type: str,
    q: int,
    r: int,
    assist_full: bool,
) -> None:
    entry = stats.get(int(shooter_id))
    if entry is None:
        return
    entry["shots"] += 1
    entry["makes"] += int(bool(success))
    entry["points"] += float(shot_value) if success else 0.0
    pair = entry["shot_types"].setdefault(shot_type, [0, 0])
    pair[0] += 1
    pair[1] += int(bool(success))
    loc = f"{int(q)},{int(r)}"
    chart_pair = entry["shot_chart"].setdefault(loc, [0, 0])
    chart_pair[0] += 1
    chart_pair[1] += int(bool(success))
    if shot_accumulator is not None:
        shot_pair = shot_accumulator.setdefault(loc, [0, 0])
        shot_pair[0] += 1
        shot_pair[1] += int(bool(success))
    if success:
        if assist_full:
            entry["assist_full_by_type"][shot_type] = int(entry["assist_full_by_type"].get(shot_type, 0)) + 1
        else:
            entry["unassisted"][shot_type] = int(entry["unassisted"].get(shot_type, 0)) + 1


def _record_shot_diagnostics(
    eval_diagnostics: dict[str, Any],
    *,
    shooter_id: int,
    success: bool,
    user_team_ids_set: set[int],
) -> None:
    if int(shooter_id) not in user_team_ids_set:
        return
    key = str(int(shooter_id))
    attempts = eval_diagnostics.setdefault("shot_attempts_by_player", {})
    attempts[key] = int(attempts.get(key, 0)) + 1
    if success:
        makes = eval_diagnostics.setdefault("made_shots_by_player", {})
        makes[key] = int(makes.get(key, 0)) + 1


def _record_pass_link_diagnostics(
    eval_diagnostics: dict[str, Any],
    *,
    passer_id: int,
    receiver_id: int,
    completed: bool,
    user_team_ids_set: set[int],
) -> None:
    if int(passer_id) < 0 or int(receiver_id) < 0:
        return
    if int(passer_id) not in user_team_ids_set or int(receiver_id) not in user_team_ids_set:
        return
    link_key = f"{int(passer_id)}->{int(receiver_id)}"
    pass_links = eval_diagnostics.setdefault("pass_links", {})
    pass_links[link_key] = int(pass_links.get(link_key, 0)) + 1
    if completed:
        completed_links = eval_diagnostics.setdefault("completed_pass_links", {})
        completed_links[link_key] = int(completed_links.get(link_key, 0)) + 1


def _record_assist_event(
    *,
    stats: dict[int, dict[str, Any]],
    eval_diagnostics: dict[str, Any],
    passer_id: int,
    shooter_id: int,
    shot_type: str,
    potential: bool,
    full: bool,
    user_team_ids_set: set[int],
) -> None:
    if int(passer_id) < 0:
        return
    entry = stats.get(int(passer_id))
    if entry is not None:
        entry["potential_assists"] += int(bool(potential))
        entry["assists"] += int(bool(full))
    if int(passer_id) == int(shooter_id):
        return
    if int(passer_id) not in user_team_ids_set or int(shooter_id) not in user_team_ids_set:
        return
    link_key = f"{int(passer_id)}->{int(shooter_id)}"
    if full:
        links = eval_diagnostics["assist_links"]
        links[link_key] = int(links.get(link_key, 0)) + 1
        by_type = eval_diagnostics["assist_links_by_type"].setdefault(shot_type, {})
        by_type[link_key] = int(by_type.get(link_key, 0)) + 1
    if potential and not full:
        links = eval_diagnostics["potential_assist_links"]
        links[link_key] = int(links.get(link_key, 0)) + 1
        by_type = eval_diagnostics["potential_assist_links_by_type"].setdefault(shot_type, {})
        by_type[link_key] = int(by_type.get(link_key, 0)) + 1


def _record_turnover_event(
    *,
    stats: dict[int, dict[str, Any]],
    eval_diagnostics: dict[str, Any],
    player_id: int,
    reason_code: int,
    user_team_ids_set: set[int],
) -> str:
    reason = _turnover_reason_label(int(reason_code))
    entry = stats.get(int(player_id))
    if entry is not None:
        entry["turnovers"] += 1
    if int(player_id) in user_team_ids_set:
        reasons = eval_diagnostics["turnover_reasons"]
        reasons[reason] = int(reasons.get(reason, 0)) + 1
    return reason


def run_native_jax_evaluation(
    *,
    num_episodes: int,
    player_deterministic: bool,
    opponent_deterministic: bool,
    required_params: dict,
    optional_params: dict,
    unified_policy_path: str,
    opponent_policy_path: str | None,
    user_team_name: str,
    role_flag_offense: float,
    role_flag_defense: float,
    eval_seed: int | None = None,
    progress_callback=None,
) -> dict[str, Any]:
    jax, jnp = ensure_jax_available("basketworld_jax/eval/native.py")
    unified_payload, unified_params, spec = _load_checkpoint_params(unified_policy_path, jax)
    if opponent_policy_path:
        _, opponent_params, opponent_spec = _load_checkpoint_params(opponent_policy_path, jax)
        if opponent_spec != spec:
            raise ValueError("JAX opponent checkpoint policy_spec does not match unified checkpoint.")
    else:
        opponent_params = unified_params

    user_team = Team.OFFENSE if str(user_team_name) == "OFFENSE" else Team.DEFENSE
    if user_team == Team.OFFENSE:
        offense_params = unified_params
        defense_params = opponent_params
        offense_deterministic = bool(player_deterministic)
        defense_deterministic = bool(opponent_deterministic)
    else:
        offense_params = opponent_params
        defense_params = unified_params
        offense_deterministic = bool(opponent_deterministic)
        defense_deterministic = bool(player_deterministic)

    env = basketworld.HexagonBasketballEnv(
        **required_params,
        **optional_params,
        render_mode=None,
    )
    static = build_kernel_static_from_env(env, jnp)
    horizon = int(getattr(env, "shot_clock_steps", 24)) + 2
    configured_batch_size = int(
        dict(unified_payload.get("trainer_config", {})).get("kernel_batch_size", 4096)
    )
    batch_size = max(1, min(int(num_episodes), configured_batch_size))
    runner = _build_native_eval_runner(jax, jnp, spec)

    n_players = int(getattr(env, "n_players", 0))
    offense_ids = [int(pid) for pid in getattr(env, "offense_ids", [])]
    defense_ids = [int(pid) for pid in getattr(env, "defense_ids", [])]
    user_team_ids = offense_ids if user_team == Team.OFFENSE else defense_ids
    user_team_ids_set = set(user_team_ids)
    offense_sign = 1.0 if user_team == Team.OFFENSE else -1.0
    pass_reward = float(getattr(env, "pass_reward", 0.0) or 0.0)
    violation_reward = float(getattr(env, "violation_reward", 0.0) or 0.0)
    potential_assist_pct = float(getattr(env, "potential_assist_pct", 0.0) or 0.0)
    full_assist_bonus_pct = float(getattr(env, "full_assist_bonus_pct", 0.0) or 0.0)
    shot_clock_steps = int(getattr(env, "shot_clock_steps", horizon))
    three_point_distance = float(getattr(env, "three_point_distance", 4.0))

    results: list[dict[str, Any]] = []
    shot_accumulator: dict[str, list[int]] = {}
    per_player_stats = _init_player_stats(n_players)
    per_intent_stats: dict[str, dict[str, Any]] = {}
    eval_diagnostics = _init_eval_diagnostics()
    all_steps: list[int] = []
    all_completed: list[int] = []
    all_offense_rewards: list[float] = []
    all_defense_rewards: list[float] = []
    all_offense_points: list[float] = []
    all_defense_points: list[float] = []
    all_pass_attempts: list[float] = []
    all_completed_passes: list[float] = []
    all_assists: list[float] = []
    all_turnovers: list[float] = []

    start = perf_counter()
    completed_episodes = 0
    if eval_seed is None:
        eval_seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
    key = jax.random.PRNGKey(int(eval_seed))
    while completed_episodes < int(num_episodes):
        take = min(batch_size, int(num_episodes) - completed_episodes)
        key, reset_key, eval_key = jax.random.split(key, 3)
        reset_keys = jax.random.split(reset_key, batch_size)
        initial_state = reset_batch_minimal(static, reset_keys, jax, jnp)
        trace_device = runner(
            static,
            initial_state,
            offense_params,
            defense_params,
            eval_key,
            jnp.asarray(float(role_flag_offense), dtype=jnp.float32),
            jnp.asarray(float(role_flag_defense), dtype=jnp.float32),
            int(horizon),
            bool(offense_deterministic),
            bool(defense_deterministic),
        )
        trace = jax.device_get(trace_device)
        stats = _episode_stats_from_trace(trace, take=take, horizon=horizon)
        for idx in range(take):
            episode_num = completed_episodes + idx + 1
            step_count = int(stats["steps"][idx])
            active_steps = max(0, min(step_count, int(horizon)))
            offense_reward = float(stats["offense_rewards"][idx])
            defense_reward = float(stats["defense_rewards"][idx])
            user_reward = offense_reward if user_team == Team.OFFENSE else defense_reward
            episode_player_stats = _init_player_stats(n_players)
            shots_payload: dict[str, dict[str, Any]] = {}
            turnovers_payload: list[dict[str, Any]] = []
            defensive_lane_violations_payload: list[dict[str, Any]] = []
            episode_shots: dict[str, list[int]] = {}
            eval_diagnostics["intent_inactive_count"] = int(eval_diagnostics.get("intent_inactive_count", 0)) + 1

            for pid in per_player_stats:
                per_player_stats[pid]["episodes"] += 1
                per_player_stats[pid]["steps"] += step_count

            for t in range(active_steps):
                _record_action_mix(eval_diagnostics, trace["full_actions"][t, idx], user_team_ids)
                if int(trace["pass_attempts"][t, idx]):
                    _record_pass_link_diagnostics(
                        eval_diagnostics,
                        passer_id=int(trace["pass_passer"][t, idx]),
                        receiver_id=int(trace["pass_receiver"][t, idx]),
                        completed=bool(int(trace["completed_passes"][t, idx])),
                        user_team_ids_set=user_team_ids_set,
                    )

                if int(trace["shot_attempt"][t, idx]):
                    shooter_id = int(trace["shot_shooter"][t, idx])
                    shot_success = bool(int(trace["shot_success"][t, idx]))
                    shot_value = float(trace["shot_value"][t, idx])
                    shot_type = _shot_type_label(int(trace["shot_type"][t, idx]))
                    q = int(trace["shot_q"][t, idx])
                    r = int(trace["shot_r"][t, idx])
                    assist_full = bool(int(trace["assists"][t, idx]))
                    potential_assist = bool(int(trace["potential_assist"][t, idx]))
                    passer_id = int(trace["assist_passer"][t, idx])
                    _record_shot_event(
                        stats=per_player_stats,
                        shot_accumulator=shot_accumulator,
                        shooter_id=shooter_id,
                        success=shot_success,
                        shot_value=shot_value,
                        shot_type=shot_type,
                        q=q,
                        r=r,
                        assist_full=assist_full,
                    )
                    _record_shot_diagnostics(
                        eval_diagnostics,
                        shooter_id=shooter_id,
                        success=shot_success,
                        user_team_ids_set=user_team_ids_set,
                    )
                    _record_shot_event(
                        stats=episode_player_stats,
                        shot_accumulator=None,
                        shooter_id=shooter_id,
                        success=shot_success,
                        shot_value=shot_value,
                        shot_type=shot_type,
                        q=q,
                        r=r,
                        assist_full=assist_full,
                    )
                    if potential_assist or assist_full:
                        _record_assist_event(
                            stats=per_player_stats,
                            eval_diagnostics=eval_diagnostics,
                            passer_id=passer_id,
                            shooter_id=shooter_id,
                            shot_type=shot_type,
                            potential=potential_assist,
                            full=assist_full,
                            user_team_ids_set=user_team_ids_set,
                        )
                        if passer_id in episode_player_stats:
                            episode_player_stats[passer_id]["potential_assists"] += int(potential_assist)
                            episode_player_stats[passer_id]["assists"] += int(assist_full)

                    loc = f"{q},{r}"
                    episode_shot_pair = episode_shots.setdefault(loc, [0, 0])
                    episode_shot_pair[0] += 1
                    episode_shot_pair[1] += int(shot_success)
                    shots_payload[str(shooter_id)] = {
                        "success": shot_success,
                        "distance": float(trace["shot_distance"][t, idx]),
                        "is_three": shot_type == "three",
                        "expected_points": float(trace["shot_expected_points"][t, idx]),
                        "shot_value": shot_value,
                        "assist_full": assist_full,
                        "assist_potential": potential_assist,
                        "assist_passer_id": passer_id if passer_id >= 0 else None,
                    }

                if int(trace["turnovers"][t, idx]):
                    turnover_player = int(trace["turnover_player"][t, idx])
                    if turnover_player >= 0:
                        reason = _record_turnover_event(
                            stats=per_player_stats,
                            eval_diagnostics=eval_diagnostics,
                            player_id=turnover_player,
                            reason_code=int(trace["turnover_reason"][t, idx]),
                            user_team_ids_set=user_team_ids_set,
                        )
                        if turnover_player in episode_player_stats:
                            episode_player_stats[turnover_player]["turnovers"] += 1
                        turnovers_payload.append(
                            {
                                "player_id": turnover_player,
                                "reason": reason,
                            }
                        )
                if int(trace["defensive_lane_violation"][t, idx]):
                    defender_id = int(trace["defensive_lane_violation_player"][t, idx])
                    defensive_lane_violations_payload.append(
                        {
                            "player_id": defender_id,
                            "reason": "illegal_defense",
                        }
                    )

            completed_passes = float(np.asarray(trace["completed_passes"])[:active_steps, idx].sum())
            defensive_lane_violations = float(
                np.asarray(trace["defensive_lane_violation"])[:active_steps, idx].sum()
            )
            shot_expected_points = np.asarray(trace["shot_expected_points"])[:active_steps, idx]
            potential_flags = np.asarray(trace["potential_assist"])[:active_steps, idx].astype(bool)
            assist_flags = np.asarray(trace["assists"])[:active_steps, idx].astype(bool)
            expected_amt = offense_sign * float(shot_expected_points.sum())
            pass_amt = offense_sign * pass_reward * completed_passes
            violation_amt = offense_sign * violation_reward * defensive_lane_violations
            potential_amt = offense_sign * potential_assist_pct * float(shot_expected_points[potential_flags].sum())
            full_amt = offense_sign * full_assist_bonus_pct * float(shot_expected_points[assist_flags].sum())
            known_reward = expected_amt + pass_amt + violation_amt + potential_amt + full_amt
            reward_breakdown = eval_diagnostics["reward_breakdown"]
            reward_breakdown["total_reward"] += user_reward
            reward_breakdown["expected_points"] += expected_amt
            reward_breakdown["pass_reward"] += pass_amt
            reward_breakdown["violation_reward"] += violation_amt
            reward_breakdown["assist_potential"] += potential_amt
            reward_breakdown["assist_full_bonus"] += full_amt
            reward_breakdown["unexplained"] += user_reward - known_reward

            per_intent_stats["none"] = _accumulate_team_stats_from_players(
                per_intent_stats.get("none"),
                episode_player_stats,
                user_team_ids,
                episodes=1,
                steps=step_count,
            )

            results.append(
                {
                    "episode": int(episode_num),
                    "intent_index": None,
                    "steps": step_count,
                    "episode_rewards": {
                        "offense": offense_reward,
                        "defense": defense_reward,
                    },
                    "outcome_info": {
                        "shots": shots_payload,
                        "turnovers": turnovers_payload,
                        "defensive_lane_violations": defensive_lane_violations_payload,
                        "shot_clock": shot_clock_steps,
                        "three_point_distance": three_point_distance,
                    },
                    "shot_counts": episode_shots,
                }
            )
        all_steps.extend([int(v) for v in stats["steps"].tolist()])
        all_completed.extend([int(v) for v in stats["completed"].tolist()])
        all_offense_rewards.extend([float(v) for v in stats["offense_rewards"].tolist()])
        all_defense_rewards.extend([float(v) for v in stats["defense_rewards"].tolist()])
        all_offense_points.extend([float(v) for v in stats["offense_points"].tolist()])
        all_defense_points.extend([float(v) for v in stats["defense_points"].tolist()])
        all_pass_attempts.extend([float(v) for v in stats["pass_attempts"].tolist()])
        all_completed_passes.extend([float(v) for v in stats["completed_passes"].tolist()])
        all_assists.extend([float(v) for v in stats["assists"].tolist()])
        all_turnovers.extend([float(v) for v in stats["turnovers"].tolist()])
        completed_episodes += take
        if progress_callback is not None:
            progress_callback(completed_episodes, int(num_episodes))

    elapsed = max(perf_counter() - start, 1.0e-12)
    shot_type_attempts = {
        shot_type: int(
            sum(
                int((stats.get("shot_types") or {}).get(shot_type, [0, 0])[0])
                for stats in per_player_stats.values()
            )
        )
        for shot_type in ("dunk", "two", "three")
    }
    total_shot_attempts = int(sum(shot_type_attempts.values()))
    shot_type_shares = {
        shot_type: (
            float(count / total_shot_attempts)
            if total_shot_attempts > 0
            else 0.0
        )
        for shot_type, count in shot_type_attempts.items()
    }
    summary = {
        "backend": "jax",
        "mode": "native_compiled",
        "num_episodes": int(num_episodes),
        "eval_seed": int(eval_seed),
        "allow_dunks": bool(getattr(env, "allow_dunks", False)),
        "batch_size": int(batch_size),
        "horizon": int(horizon),
        "elapsed_sec": float(elapsed),
        "episodes_per_sec": float(int(num_episodes) / elapsed),
        "states_per_sec": float((int(num_episodes) * int(horizon)) / elapsed),
        "completed_episodes": int(np.sum(np.asarray(all_completed, dtype=np.int32))),
        "completion_rate": _mean(all_completed),
        "mean_steps": _mean(all_steps),
        "offense_reward_per_episode": _mean(all_offense_rewards),
        "defense_reward_per_episode": _mean(all_defense_rewards),
        "offense_points_per_episode": _mean(all_offense_points),
        "defense_points_per_episode": _mean(all_defense_points),
        "score_margin_per_episode": _mean(np.asarray(all_offense_points) - np.asarray(all_defense_points)),
        "pass_attempts_per_episode": _mean(all_pass_attempts),
        "completed_passes_per_episode": _mean(all_completed_passes),
        "assists_per_episode": _mean(all_assists),
        "turnovers_per_episode": _mean(all_turnovers),
        "total_offense_points": _sum(all_offense_points),
        "total_defense_points": _sum(all_defense_points),
        "total_shot_attempts": int(total_shot_attempts),
        "total_shot_dunk_attempts": int(shot_type_attempts["dunk"]),
        "total_shot_two_attempts": int(shot_type_attempts["two"]),
        "total_shot_three_attempts": int(shot_type_attempts["three"]),
        "shot_dunk_share": float(shot_type_shares["dunk"]),
        "shot_two_share": float(shot_type_shares["two"]),
        "shot_three_share": float(shot_type_shares["three"]),
    }
    return {
        "results": results,
        "shot_accumulator": shot_accumulator,
        "per_player_stats": per_player_stats,
        "per_intent_stats": per_intent_stats,
        "eval_diagnostics": {
            **eval_diagnostics,
            "jax_native_summary": summary,
        },
    }
