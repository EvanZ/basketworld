import copy
import multiprocessing as mp
import queue
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from typing import Optional

import numpy as np
import basketworld
from basketworld.envs.basketworld_env_v2 import Team
from fastapi import HTTPException
from basketworld.utils.action_resolution import (
    IllegalActionStrategy,
    get_policy_action_probabilities,
    resolve_illegal_actions,
)
from basketworld.utils.policies import PassBiasDualCriticPolicy, PassBiasMultiInputPolicy
from basketworld.policies import SetAttentionDualCriticPolicy, SetAttentionExtractor
from stable_baselines3 import PPO
from basketworld.utils.policy_loading import load_ppo_for_inference

from app.backend.mcts import _run_mcts_advisor
from app.backend.observations import (
    _ensure_set_obs,
    _predict_policy_actions,
    validate_policy_observation_schema,
)
from app.backend.state import game_state


# Worker-local storage (each process has its own copy)
_worker_state = {}


def _print_eval_progress(prefix: str, current: int, total: int, start_time: float | None = None) -> None:
    if total <= 0:
        return
    current = min(current, total)
    pct = (current / total) * 100.0
    eta_str = ""
    elapsed_str = ""
    if start_time is not None and current > 0:
        elapsed = time.time() - start_time
        if elapsed > 0:
            rate = current / elapsed
            if rate > 0:
                remaining = max(0.0, (total - current) / rate)
                eta_str = f" ETA {remaining:.1f}s"
    if start_time is not None:
        elapsed_str = f" elapsed {time.time() - start_time:.1f}s"
    line = f"\r[{prefix}] {current}/{total} ({pct:.1f}%)" + eta_str + elapsed_str
    sys.stdout.write(line.ljust(80))
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")
        sys.stdout.flush()


def _init_evaluation_worker(
    required_params: dict,
    optional_params: dict,
    unified_policy_path: str,
    opponent_policy_path: str | None,
    user_team_name: str,
    role_flag_offense: float,
    role_flag_defense: float,
    progress_queue=None,
):
    """Initialize a worker process with its own environment and policies."""
    global _worker_state

    # Import all required modules for worker functions
    import numpy as _np
    import torch as _torch
    from basketworld.envs.basketworld_env_v2 import Team as _Team
    from basketworld.utils.action_resolution import (
        IllegalActionStrategy as _IllegalActionStrategy,
        get_policy_action_probabilities as _get_policy_action_probabilities,
        resolve_illegal_actions as _resolve_illegal_actions,
    )

    # Avoid thread oversubscription when running multiple workers.
    try:
        _torch.set_num_threads(1)
        _torch.set_num_interop_threads(1)
    except Exception:
        pass

    env = basketworld.HexagonBasketballEnv(
        **required_params,
        **optional_params,
        render_mode=None,
    )

    custom_objects = {
        "PassBiasDualCriticPolicy": PassBiasDualCriticPolicy,
        "PassBiasMultiInputPolicy": PassBiasMultiInputPolicy,
        "SetAttentionDualCriticPolicy": SetAttentionDualCriticPolicy,
        "SetAttentionExtractor": SetAttentionExtractor,
    }
    unified_policy = load_ppo_for_inference(
        unified_policy_path,
        device="cpu",
        custom_objects=custom_objects,
    )
    opponent_policy = (
        load_ppo_for_inference(
            opponent_policy_path,
            device="cpu",
            custom_objects=custom_objects,
        )
        if opponent_policy_path
        else None
    )

    pass_mode = str(optional_params.get("pass_mode", "directional"))
    for policy_obj in (unified_policy, opponent_policy):
        policy = getattr(policy_obj, "policy", None) if policy_obj is not None else None
        if policy is None:
            continue
        if hasattr(policy, "set_pass_mode"):
            try:
                policy.set_pass_mode(pass_mode)
            except Exception:
                pass

    user_team = _Team.OFFENSE if user_team_name == "OFFENSE" else _Team.DEFENSE

    _worker_state = {
        "env": env,
        "unified_policy": unified_policy,
        "opponent_policy": opponent_policy,
        "user_team": user_team,
        "role_flag_offense": role_flag_offense,
        "role_flag_defense": role_flag_defense,
        "np": _np,
        "Team": _Team,
        "IllegalActionStrategy": _IllegalActionStrategy,
        "get_policy_action_probabilities": _get_policy_action_probabilities,
        "resolve_illegal_actions": _resolve_illegal_actions,
        "progress_queue": progress_queue,
    }
    # Fail fast on model/env observation schema mismatches.
    try:
        obs0, _ = env.reset(seed=0)
        _ = validate_policy_observation_schema(
            unified_policy, env, obs0, policy_label="eval_unified_policy"
        )
        _ = validate_policy_observation_schema(
            opponent_policy, env, obs0, policy_label="eval_opponent_policy"
        )
    except Exception as e:
        raise RuntimeError(f"Evaluation worker schema validation failed: {e}") from e


def _worker_role_flag_value(team) -> float:
    _Team = _worker_state["Team"]
    if team == _Team.OFFENSE:
        return _worker_state.get("role_flag_offense", 1.0)
    return _worker_state.get("role_flag_defense", -1.0)


def _worker_clone_obs_with_role_flag(obs: dict, role_flag_value: float) -> dict:
    _np = _worker_state["np"]
    cloned = {
        "obs": _np.copy(obs["obs"]),
        "action_mask": obs["action_mask"],
        "role_flag": _np.array([role_flag_value], dtype=_np.float32),
    }
    skills = obs.get("skills")
    if skills is not None:
        cloned["skills"] = _np.copy(skills)
    else:
        cloned["skills"] = None
    players = obs.get("players") if isinstance(obs, dict) else None
    globals_vec = obs.get("globals") if isinstance(obs, dict) else None
    if players is not None:
        cloned["players"] = _np.copy(players)
    if globals_vec is not None:
        cloned["globals"] = _np.copy(globals_vec)
    if isinstance(obs, dict):
        for key, value in obs.items():
            if key in cloned:
                continue
            if isinstance(value, _np.ndarray):
                cloned[key] = _np.copy(value)
    observer_is_offense = bool(float(role_flag_value) > 0.0)
    try:
        env = _worker_state.get("env")
        fields = env.get_intent_observation_fields(observer_is_offense)
    except Exception:
        fields = {}
    if fields:
        for key, value in fields.items():
            cloned[key] = _np.array(value, dtype=_np.float32, copy=True)
    if "globals" in cloned:
        try:
            cloned["globals"] = _worker_state["env"].patch_globals_with_intent_features(
                cloned["globals"], observer_is_offense
            )
        except Exception:
            pass
    return cloned


def _worker_predict_actions_for_team(
    policy,
    base_obs: dict,
    env,
    team,
    deterministic: bool,
    strategy,
) -> dict[int, int]:
    """Predict actions for a team in worker context."""
    _np = _worker_state["np"]
    _Team = _worker_state["Team"]
    _get_policy_action_probabilities = _worker_state["get_policy_action_probabilities"]
    _resolve_illegal_actions = _worker_state["resolve_illegal_actions"]

    actions_by_player: dict[int, int] = {}

    if policy is None or base_obs is None or env is None:
        return actions_by_player

    base_obs = _ensure_set_obs(policy, env, base_obs)
    team_ids = list(env.offense_ids if team == _Team.OFFENSE else env.defense_ids)
    if not team_ids:
        return actions_by_player

    role_flag_value = _worker_role_flag_value(team)
    conditioned_obs = _worker_clone_obs_with_role_flag(base_obs, role_flag_value)

    try:
        raw_actions, _ = policy.predict(conditioned_obs, deterministic=deterministic)
    except Exception as err:
        err_count = int(_worker_state.get("predict_failure_count", 0)) + 1
        _worker_state["predict_failure_count"] = err_count
        if err_count <= 5:
            team_name = getattr(team, "name", str(team))
            print(
                f"[EvalWorker][WARN] policy.predict failed for team={team_name} "
                f"(deterministic={deterministic}) count={err_count}: {err}"
            )
        return actions_by_player

    raw_actions = _np.array(raw_actions).reshape(-1)
    action_len = raw_actions.shape[0]
    team_mask = base_obs["action_mask"][team_ids]

    if action_len == len(team_ids):
        team_pred_actions = raw_actions
    elif action_len == getattr(env, "n_players", action_len):
        team_pred_actions = raw_actions[team_ids]
    else:
        team_pred_actions = raw_actions[: len(team_ids)]

    probs = _get_policy_action_probabilities(policy, conditioned_obs)
    if probs is not None:
        probs = [_np.asarray(p, dtype=_np.float32) for p in probs]
        if len(probs) == getattr(env, "n_players", len(probs)):
            team_probs = [probs[int(pid)] for pid in team_ids]
        else:
            team_probs = probs[: len(team_ids)]
    else:
        team_probs = None

    # Deterministic policy execution should select argmax among legal actions from
    # the model's probability distribution directly. This keeps worker eval
    # behavior consistent with live/sequential inference paths.
    if deterministic and team_probs is not None and len(team_probs) == len(team_ids):
        resolved_actions = _np.zeros(len(team_ids), dtype=int)
        for idx in range(len(team_ids)):
            legal = _np.where(team_mask[idx] == 1)[0]
            if len(legal) == 0:
                resolved_actions[idx] = 0
                continue
            p = _np.asarray(team_probs[idx], dtype=_np.float32)
            if p.shape[0] <= int(_np.max(legal)):
                # Fallback to resolver if probability vector is malformed.
                resolved_actions = _resolve_illegal_actions(
                    _np.array(team_pred_actions),
                    team_mask,
                    strategy,
                    deterministic,
                    team_probs,
                )
                break
            masked = p[legal]
            resolved_actions[idx] = int(legal[int(_np.argmax(masked))])
    else:
        resolved_actions = _resolve_illegal_actions(
            _np.array(team_pred_actions),
            team_mask,
            strategy,
            deterministic,
            team_probs,
        )

    for idx, pid in enumerate(team_ids):
        actions_by_player[int(pid)] = int(resolved_actions[idx])

    return actions_by_player


def _worker_predict_policy_actions(
    policy,
    base_obs: dict,
    env,
    deterministic: bool,
    strategy,
):
    if policy is None or base_obs is None or env is None:
        return None

    _np = _worker_state["np"]
    _Team = _worker_state["Team"]

    num_players = env.n_players
    full_actions = _np.zeros(num_players, dtype=int)

    for team in (_Team.OFFENSE, _Team.DEFENSE):
        team_actions = _worker_predict_actions_for_team(
            policy,
            base_obs,
            env,
            team,
            deterministic,
            strategy,
        )
        for pid, action in team_actions.items():
            full_actions[int(pid)] = int(action)

    return full_actions


def _run_episode_batch_worker(args: tuple) -> dict:
    """Worker entrypoint to run a batch of episodes."""
    batch_specs, player_deterministic, opponent_deterministic, custom_setup, randomize_offense_permutation = args
    env = _worker_state["env"]
    unified_policy = _worker_state["unified_policy"]
    opponent_policy = _worker_state["opponent_policy"]
    user_team = _worker_state["user_team"]
    role_flag_offense = _worker_state["role_flag_offense"]
    role_flag_defense = _worker_state["role_flag_defense"]
    _np = _worker_state["np"]
    _Team = _worker_state["Team"]
    _IllegalActionStrategy = _worker_state["IllegalActionStrategy"]
    progress_queue = _worker_state.get("progress_queue")

    results = []
    shot_accumulator: dict[str, list[int]] = {}
    per_player_stats = _init_player_stats(env.n_players)
    eval_diagnostics = _init_eval_diagnostics()
    user_team_ids = list(env.offense_ids if user_team == _Team.OFFENSE else env.defense_ids)
    user_team_ids_set = {int(pid) for pid in user_team_ids}

    for ep_idx, seed in batch_specs:
        # Custom reset options per episode
        reset_opts = _build_reset_options_for_custom_setup(custom_setup, enforce_fixed_skills=True)
        if randomize_offense_permutation:
            perm = _sample_offense_permutation(env, _np.random)
            perm_dict = {env.offense_ids[i]: perm[i] for i in range(len(env.offense_ids))}
            reset_opts["offense_player_permutation"] = perm_dict

        obs, _ = env.reset(seed=seed, options=reset_opts)
        env.training_team = user_team

        # Update role_flag using worker-local values
        obs["role_flag"] = _np.array([role_flag_offense if user_team == _Team.OFFENSE else role_flag_defense], dtype=_np.float32)
        _accumulate_intent_selection(eval_diagnostics, env)

        done = False
        step_count = 0
        episode_rewards = {"offense": 0.0, "defense": 0.0}
        last_action_results = {}
        episode_shots: dict[str, list[int]] = {}

        while not done:
            player_strategy = (
                _IllegalActionStrategy.BEST_PROB if player_deterministic else _IllegalActionStrategy.SAMPLE_PROB
            )
            opponent_strategy = (
                _IllegalActionStrategy.BEST_PROB if opponent_deterministic else _IllegalActionStrategy.SAMPLE_PROB
            )
            actions_player = _worker_predict_policy_actions(
                unified_policy,
                obs,
                env,
                deterministic=player_deterministic,
                strategy=player_strategy,
            )
            actions_opponent = _worker_predict_policy_actions(
                opponent_policy if opponent_policy is not None else unified_policy,
                obs,
                env,
                deterministic=opponent_deterministic,
                strategy=opponent_strategy,
            )

            full_action = _np.zeros(env.n_players, dtype=int)
            if user_team == _Team.OFFENSE:
                full_action[env.offense_ids] = actions_player[env.offense_ids]
                full_action[env.defense_ids] = actions_opponent[env.defense_ids]
            else:
                full_action[env.defense_ids] = actions_player[env.defense_ids]
                full_action[env.offense_ids] = actions_opponent[env.offense_ids]

            _accumulate_action_mix(eval_diagnostics, full_action, user_team_ids)
            obs, reward, terminated, truncated, info = env.step(full_action)
            action_results = info.get("action_results", {}) if info else {}
            if not action_results:
                action_results = info.get("last_action_results", {}) if info else {}
            if not action_results:
                action_results = getattr(env, "last_action_results", {}) or {}
            last_action_results = action_results

            shots_for_step = action_results.get("shots", {}) if isinstance(action_results, dict) else {}
            _accumulate_assist_links(eval_diagnostics, shots_for_step, user_team_ids_set, env)
            for shooter_id, shot_res in shots_for_step.items():
                try:
                    sid = int(shooter_id)
                    pos = env.positions[sid]
                    q, r = int(pos[0]), int(pos[1])
                    key = f"{q},{r}"
                    if shot_accumulator is not None:
                        if key not in shot_accumulator:
                            shot_accumulator[key] = [0, 0]
                    if key not in episode_shots:
                        episode_shots[key] = [0, 0]
                    if shot_accumulator is not None:
                        shot_accumulator[key][0] += 1
                    episode_shots[key][0] += 1
                    if bool(shot_res.get("success", False)):
                        if shot_accumulator is not None:
                            shot_accumulator[key][1] += 1
                        episode_shots[key][1] += 1
                    _record_shot_for_stats(per_player_stats, sid, env, bool(shot_res.get("success", False)), bool(shot_res.get("assist_full", False)))
                    if "assist_passer_id" in shot_res:
                        _record_assist_for_stats(
                            per_player_stats,
                            int(shot_res.get("assist_passer_id")),
                            bool(shot_res.get("assist_full", False)),
                            bool(shot_res.get("assist_potential", False)),
                        )
                except Exception:
                    continue

            turnovers_raw_step = action_results.get("turnovers", []) if isinstance(action_results, dict) else []
            if isinstance(turnovers_raw_step, (list, tuple)):
                for t in turnovers_raw_step:
                    try:
                        _record_turnover_for_stats(per_player_stats, t.get("player_id"))
                    except Exception:
                        continue
            _accumulate_turnover_reasons(eval_diagnostics, turnovers_raw_step, user_team_ids_set)
            _accumulate_reward_breakdown(
                eval_diagnostics,
                env,
                action_results if isinstance(action_results, dict) else {},
                info if isinstance(info, dict) else {},
                reward,
                user_team,
            )

            # Update role_flag for next step
            obs["role_flag"] = _np.array([role_flag_offense if user_team == _Team.OFFENSE else role_flag_defense], dtype=_np.float32)

            episode_rewards["offense"] += float(reward[env.offense_ids].sum())
            episode_rewards["defense"] += float(reward[env.defense_ids].sum())
            step_count += 1
            done = bool(terminated or truncated)

        # Collect stats
        shot_clock = getattr(env, "shot_clock", None)
        three_point_distance = float(getattr(env, "three_point_distance", 4.0))
        # Mark participation for per-player stats
        for pid in per_player_stats:
            per_player_stats[pid]["episodes"] += 1
        if per_player_stats is not None:
            for pid in per_player_stats:
                per_player_stats[pid]["steps"] += step_count

        results.append(
            {
                "episode": ep_idx + 1,
                "steps": step_count,
                "episode_rewards": episode_rewards,
                "outcome_info": {
                    "shots": last_action_results.get("shots", {}) if isinstance(last_action_results, dict) else {},
                    "turnovers": last_action_results.get("turnovers", []) if isinstance(last_action_results, dict) else [],
                    "defensive_lane_violations": last_action_results.get("defensive_lane_violations", []) if isinstance(last_action_results, dict) else [],
                    "shot_clock": shot_clock,
                    "three_point_distance": three_point_distance,
                },
                "shot_counts": episode_shots,
            }
        )
        if progress_queue is not None:
            try:
                progress_queue.put_nowait(1)
            except queue.Full:
                pass
            except Exception:
                pass

    return {
        "results": results,
        "shot_accumulator": shot_accumulator if shot_accumulator is not None else {},
        "per_player_stats": per_player_stats,
        "eval_diagnostics": eval_diagnostics,
    }


def _init_player_stats(n_players: int) -> dict:
    stats = {}
    for pid in range(n_players):
        stats[pid] = {
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
    return stats


def _merge_player_stats(dest: dict, src: dict) -> dict:
    if dest is None:
        dest = {}
    for pid_raw, src_stats in (src or {}).items():
        pid = int(pid_raw)
        if pid not in dest:
            dest[pid] = {
                "shots": 0,
                "makes": 0,
                "shot_types": {"dunk": [0, 0], "two": [0, 0], "three": [0, 0]},
                "assists": 0,
                "potential_assists": 0,
                "turnovers": 0,
                "points": 0.0,
                "episodes": 0,
                "steps": 0,
                "shot_chart": {},
                "unassisted": {"dunk": 0, "two": 0, "three": 0},
            }
        dst_stats = dest[pid]
        dst_stats["shots"] += int(src_stats.get("shots", 0))
        dst_stats["makes"] += int(src_stats.get("makes", 0))
        dst_stats["assists"] += int(src_stats.get("assists", 0))
        dst_stats["potential_assists"] += int(src_stats.get("potential_assists", 0))
        dst_stats["turnovers"] += int(src_stats.get("turnovers", 0))
        dst_stats["points"] += float(src_stats.get("points", 0.0))
        dst_stats["episodes"] += int(src_stats.get("episodes", 0))
        dst_stats["steps"] += int(src_stats.get("steps", 0))

        for shot_type in ("dunk", "two", "three"):
            src_pair = src_stats.get("shot_types", {}).get(shot_type, [0, 0])
            dst_pair = dst_stats["shot_types"].setdefault(shot_type, [0, 0])
            dst_pair[0] += int(src_pair[0] if isinstance(src_pair, (list, tuple)) else 0)
            dst_pair[1] += int(src_pair[1] if isinstance(src_pair, (list, tuple)) else 0)
            dst_stats.setdefault("assist_full_by_type", {}).setdefault(shot_type, 0)
            dst_stats["assist_full_by_type"][shot_type] += int(
                src_stats.get("assist_full_by_type", {}).get(shot_type, 0) or 0
            )

        src_chart = src_stats.get("shot_chart", {}) or {}
        for loc, vals in src_chart.items():
            dst_pair = dst_stats["shot_chart"].setdefault(loc, [0, 0])
            try:
                att = int(vals[0]) if isinstance(vals, (list, tuple)) and len(vals) > 0 else 0
                mk = int(vals[1]) if isinstance(vals, (list, tuple)) and len(vals) > 1 else 0
            except Exception:
                att, mk = 0, 0
            dst_pair[0] += att
            dst_pair[1] += mk
        src_un = src_stats.get("unassisted", {}) or {}
        dst_un = dst_stats.setdefault("unassisted", {"dunk": 0, "two": 0, "three": 0})
        for key in ("dunk", "two", "three"):
            dst_un[key] = dst_un.get(key, 0) + int(src_un.get(key, 0) or 0)
    return dest


def _init_eval_diagnostics() -> dict:
    return {
        "intent_selection_counts": {},
        "intent_inactive_count": 0,
        "turnover_reasons": {},
        "assist_links": {},
        "assist_links_by_type": {"dunk": {}, "two": {}, "three": {}},
        "potential_assist_links": {},
        "potential_assist_links_by_type": {"dunk": {}, "two": {}, "three": {}},
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


def _merge_eval_diagnostics(dest: dict | None, src: dict | None) -> dict:
    if dest is None:
        dest = _init_eval_diagnostics()
    if not src:
        return dest

    for raw_z, count in (src.get("intent_selection_counts") or {}).items():
        key = str(raw_z)
        dest["intent_selection_counts"][key] = int(
            dest["intent_selection_counts"].get(key, 0)
        ) + int(count or 0)

    dest["intent_inactive_count"] = int(dest.get("intent_inactive_count", 0)) + int(
        src.get("intent_inactive_count", 0) or 0
    )

    for reason, count in (src.get("turnover_reasons") or {}).items():
        key = str(reason) if reason is not None else "unknown"
        dest["turnover_reasons"][key] = int(dest["turnover_reasons"].get(key, 0)) + int(
            count or 0
        )

    for link_key, count in (src.get("assist_links") or {}).items():
        key = str(link_key)
        dest["assist_links"][key] = int(dest["assist_links"].get(key, 0)) + int(
            count or 0
        )

    for link_key, count in (src.get("potential_assist_links") or {}).items():
        key = str(link_key)
        dest["potential_assist_links"][key] = int(
            dest["potential_assist_links"].get(key, 0)
        ) + int(count or 0)

    for shot_type in ("dunk", "two", "three"):
        dest.setdefault("assist_links_by_type", {}).setdefault(shot_type, {})
        src_map = (src.get("assist_links_by_type") or {}).get(shot_type, {}) or {}
        for link_key, count in src_map.items():
            key = str(link_key)
            dest["assist_links_by_type"][shot_type][key] = int(
                dest["assist_links_by_type"][shot_type].get(key, 0)
            ) + int(count or 0)

    for shot_type in ("dunk", "two", "three"):
        dest.setdefault("potential_assist_links_by_type", {}).setdefault(shot_type, {})
        src_map = (src.get("potential_assist_links_by_type") or {}).get(
            shot_type, {}
        ) or {}
        for link_key, count in src_map.items():
            key = str(link_key)
            dest["potential_assist_links_by_type"][shot_type][key] = int(
                dest["potential_assist_links_by_type"][shot_type].get(key, 0)
            ) + int(count or 0)

    for key in ("noop", "move", "shoot", "pass", "other", "total"):
        dest["action_mix"][key] = int(dest["action_mix"].get(key, 0)) + int(
            (src.get("action_mix") or {}).get(key, 0) or 0
        )

    for key in (
        "total_reward",
        "expected_points",
        "pass_reward",
        "violation_reward",
        "assist_potential",
        "assist_full_bonus",
        "phi_shaping",
        "unexplained",
    ):
        dest["reward_breakdown"][key] = float(dest["reward_breakdown"].get(key, 0.0)) + float(
            (src.get("reward_breakdown") or {}).get(key, 0.0) or 0.0
        )

    return dest


def _is_offense_team(team) -> bool:
    return str(getattr(team, "name", team)).upper() == "OFFENSE"


def _action_bucket(action_id: int) -> str:
    aid = int(action_id)
    if aid == 0:
        return "noop"
    if 1 <= aid <= 6:
        return "move"
    if aid == 7:
        return "shoot"
    if 8 <= aid <= 13:
        return "pass"
    return "other"


def _accumulate_action_mix(eval_diagnostics: dict, full_action, user_team_ids: list[int]) -> None:
    action_mix = eval_diagnostics.setdefault("action_mix", {})
    for pid in user_team_ids:
        try:
            action_id = int(full_action[int(pid)])
        except Exception:
            action_id = 0
        bucket = _action_bucket(action_id)
        action_mix[bucket] = int(action_mix.get(bucket, 0)) + 1
        action_mix["total"] = int(action_mix.get("total", 0)) + 1


def _accumulate_intent_selection(eval_diagnostics: dict, env) -> None:
    if eval_diagnostics is None or env is None:
        return
    if not bool(getattr(env, "enable_intent_learning", False)):
        return
    if not bool(getattr(env, "intent_active", False)):
        eval_diagnostics["intent_inactive_count"] = int(
            eval_diagnostics.get("intent_inactive_count", 0)
        ) + 1
        return
    try:
        z = int(getattr(env, "intent_index", 0))
    except Exception:
        z = 0
    counts = eval_diagnostics.setdefault("intent_selection_counts", {})
    key = str(z)
    counts[key] = int(counts.get(key, 0)) + 1


def _accumulate_turnover_reasons(
    eval_diagnostics: dict, turnovers_raw_step, user_team_ids_set: set[int]
) -> None:
    turnover_reasons = eval_diagnostics.setdefault("turnover_reasons", {})
    if not isinstance(turnovers_raw_step, (list, tuple)):
        return
    for turnover in turnovers_raw_step:
        if not isinstance(turnover, dict):
            continue
        pid = turnover.get("player_id")
        if pid is None:
            continue
        try:
            if int(pid) not in user_team_ids_set:
                continue
        except Exception:
            continue
        reason = str(turnover.get("reason") or "unknown")
        turnover_reasons[reason] = int(turnover_reasons.get(reason, 0)) + 1


def _accumulate_assist_links(
    eval_diagnostics: dict, shots_for_step, user_team_ids_set: set[int], env=None
) -> None:
    assist_links = eval_diagnostics.setdefault("assist_links", {})
    potential_assist_links = eval_diagnostics.setdefault("potential_assist_links", {})
    assist_links_by_type = eval_diagnostics.setdefault(
        "assist_links_by_type", {"dunk": {}, "two": {}, "three": {}}
    )
    potential_assist_links_by_type = eval_diagnostics.setdefault(
        "potential_assist_links_by_type", {"dunk": {}, "two": {}, "three": {}}
    )
    for shot_type in ("dunk", "two", "three"):
        assist_links_by_type.setdefault(shot_type, {})
        potential_assist_links_by_type.setdefault(shot_type, {})
    if not isinstance(shots_for_step, dict):
        return

    for shooter_raw, shot_res in shots_for_step.items():
        if not isinstance(shot_res, dict):
            continue
        assist_potential = bool(shot_res.get("assist_potential", False))
        assist_full = bool(shot_res.get("assist_full", False))
        if not assist_potential and not assist_full:
            continue
        passer_raw = shot_res.get("assist_passer_id")
        if passer_raw is None:
            continue
        try:
            shooter_id = int(shooter_raw)
            passer_id = int(passer_raw)
        except Exception:
            continue
        if shooter_id == passer_id:
            continue
        if (
            shooter_id not in user_team_ids_set
            or passer_id not in user_team_ids_set
        ):
            continue

        shot_type = "two"
        try:
            distance = int(shot_res.get("distance"))
        except Exception:
            distance = None
        try:
            is_three_raw = shot_res.get("is_three")
            is_three = bool(is_three_raw) if is_three_raw is not None else None
        except Exception:
            is_three = None
        if distance == 0 and bool(getattr(env, "allow_dunks", True)):
            shot_type = "dunk"
        elif is_three is True:
            shot_type = "three"
        elif is_three is False:
            shot_type = "two"
        elif distance is not None:
            three_point_distance = float(getattr(env, "three_point_distance", 4.0))
            shot_type = "three" if distance >= three_point_distance else "two"

        key = f"{passer_id}->{shooter_id}"
        if assist_full:
            assist_links[key] = int(assist_links.get(key, 0)) + 1
            assist_links_by_type[shot_type][key] = int(
                assist_links_by_type[shot_type].get(key, 0)
            ) + 1
        if assist_potential and not assist_full:
            potential_assist_links[key] = int(potential_assist_links.get(key, 0)) + 1
            potential_assist_links_by_type[shot_type][key] = int(
                potential_assist_links_by_type[shot_type].get(key, 0)
            ) + 1


def _accumulate_reward_breakdown(
    eval_diagnostics: dict,
    env,
    action_results: dict,
    info: dict,
    reward,
    user_team,
) -> None:
    reward_breakdown = eval_diagnostics.setdefault("reward_breakdown", {})
    is_offense_user = _is_offense_team(user_team)
    offense_sign = 1.0 if is_offense_user else -1.0
    user_team_ids = env.offense_ids if is_offense_user else env.defense_ids

    team_reward = float(np.sum(reward[user_team_ids]))
    reward_breakdown["total_reward"] = float(reward_breakdown.get("total_reward", 0.0)) + team_reward

    step_known = 0.0

    pass_successes = 0
    for pass_result in (action_results.get("passes", {}) or {}).values():
        if isinstance(pass_result, dict) and pass_result.get("success"):
            pass_successes += 1
    pass_amt = offense_sign * float(getattr(env, "pass_reward", 0.0)) * float(pass_successes)
    reward_breakdown["pass_reward"] = float(reward_breakdown.get("pass_reward", 0.0)) + pass_amt
    step_known += pass_amt

    shots = action_results.get("shots", {}) or {}
    for shot_result in shots.values():
        if not isinstance(shot_result, dict):
            continue
        expected_points = float(shot_result.get("expected_points", 0.0) or 0.0)
        expected_amt = offense_sign * expected_points
        reward_breakdown["expected_points"] = float(reward_breakdown.get("expected_points", 0.0)) + expected_amt
        step_known += expected_amt

        if bool(shot_result.get("assist_potential", False)):
            pot_pct = getattr(env, "potential_assist_pct", None)
            if pot_pct is not None:
                potential_amt = max(0.0, float(pot_pct) * expected_points)
            else:
                potential_amt = float(getattr(env, "potential_assist_reward", 0.0))
            potential_term = offense_sign * potential_amt
            reward_breakdown["assist_potential"] = float(
                reward_breakdown.get("assist_potential", 0.0)
            ) + potential_term
            step_known += potential_term

        if bool(shot_result.get("assist_full", False)):
            full_pct = getattr(env, "full_assist_bonus_pct", None)
            if full_pct is not None:
                full_amt = max(0.0, float(full_pct) * expected_points)
            else:
                full_amt = float(getattr(env, "full_assist_bonus", 0.0))
            full_term = offense_sign * full_amt
            reward_breakdown["assist_full_bonus"] = float(
                reward_breakdown.get("assist_full_bonus", 0.0)
            ) + full_term
            step_known += full_term

    if (action_results.get("defensive_lane_violations") or []) and not shots:
        violation_count = len(action_results.get("defensive_lane_violations") or [])
        violation_amt = offense_sign * float(getattr(env, "violation_reward", 0.0)) * float(
            violation_count
        )
        reward_breakdown["violation_reward"] = float(
            reward_breakdown.get("violation_reward", 0.0)
        ) + violation_amt
        step_known += violation_amt

    phi_team_amt = 0.0
    if isinstance(info, dict) and info.get("phi_r_shape") is not None:
        try:
            phi_per_player = float(info.get("phi_r_shape") or 0.0)
            phi_team_amt = offense_sign * phi_per_player * float(
                max(1, int(getattr(env, "players_per_side", 1)))
            )
        except Exception:
            phi_team_amt = 0.0
    reward_breakdown["phi_shaping"] = float(reward_breakdown.get("phi_shaping", 0.0)) + phi_team_amt
    step_known += phi_team_amt

    reward_breakdown["unexplained"] = float(reward_breakdown.get("unexplained", 0.0)) + (
        team_reward - step_known
    )


def _record_shot_for_stats(stats: dict, shooter_id: int, env, success: bool, assist_full: bool = False):
    if stats is None or shooter_id not in stats:
        return
    try:
        pos = env.positions[shooter_id]
        dist = env._hex_distance(pos, env.basket_position)
        is_three = bool(env._is_three_point_hex(tuple(pos)))
    except Exception:
        pos = (0, 0)
        dist = None
        is_three = False

    shot_type = "three" if is_three else "two"
    if dist == 0 and getattr(env, "allow_dunks", True):
        shot_type = "dunk"

    player_stats = stats[shooter_id]
    player_stats["shots"] += 1
    player_stats["makes"] += 1 if success else 0
    player_stats["points"] += 3 if (shot_type == "three" and success) else (2 if success else 0)
    pair = player_stats["shot_types"].setdefault(shot_type, [0, 0])
    pair[0] += 1
    if success:
        pair[1] += 1
        afbt = player_stats.setdefault("assist_full_by_type", {"dunk": 0, "two": 0, "three": 0})
        if assist_full:
            afbt[shot_type] = afbt.get(shot_type, 0) + 1
        if not assist_full:
            ua = player_stats.setdefault("unassisted", {"dunk": 0, "two": 0, "three": 0})
            ua[shot_type] = ua.get(shot_type, 0) + 1

    try:
        key = f"{int(pos[0])},{int(pos[1])}"
        chart_pair = player_stats["shot_chart"].setdefault(key, [0, 0])
        chart_pair[0] += 1
        if success:
            chart_pair[1] += 1
    except Exception:
        pass


def _record_assist_for_stats(stats: dict, passer_id: Optional[int], full: bool, potential: bool):
    if stats is None or passer_id is None or passer_id not in stats:
        return
    if full:
        stats[passer_id]["assists"] += 1
    if potential:
        stats[passer_id]["potential_assists"] += 1


def _record_turnover_for_stats(stats: dict, player_id: Optional[int]):
    if stats is None or player_id is None:
        return
    pid = int(player_id)
    if pid in stats:
        stats[pid]["turnovers"] += 1


def _build_reset_options_for_custom_setup(custom_setup: dict | None, enforce_fixed_skills: bool = False) -> dict:
    if not custom_setup:
        return {}
    opts: dict = {}
    if custom_setup.get("initial_positions"):
        opts["initial_positions"] = [tuple(p) for p in custom_setup["initial_positions"]]
    if custom_setup.get("ball_holder") is not None:
        opts["ball_holder"] = int(custom_setup["ball_holder"])
    shooting_mode = custom_setup.get("shooting_mode") or "random"
    if enforce_fixed_skills and shooting_mode == "fixed" and custom_setup.get("offense_skills"):
        opts["offense_skills"] = copy.deepcopy(custom_setup["offense_skills"])
    return opts


def _sample_offense_permutation(env, rng=None) -> list[int]:
    rng = rng or np.random
    offense_ids = list(env.offense_ids)
    rng.shuffle(offense_ids)
    return offense_ids


def _run_sequential_evaluation(
    num_episodes: int,
    player_deterministic: bool,
    opponent_deterministic: bool,
    env,
    unified_policy: PPO,
    opponent_policy: PPO | None,
    user_team: Team,
    role_flag_offense: float,
    role_flag_defense: float,
    shot_accumulator: dict[str, list[int]] | None = None,
    custom_setup: dict | None = None,
    randomize_offense_permutation: bool = False,
    progress_callback=None,
) -> dict:
    results = []
    per_player_stats = _init_player_stats(env.n_players)
    eval_diagnostics = _init_eval_diagnostics()
    user_team_ids = list(env.offense_ids if user_team == Team.OFFENSE else env.defense_ids)
    user_team_ids_set = {int(pid) for pid in user_team_ids}
    progress_start = time.time()
    progress_every = max(1, num_episodes // 50)

    for ep_idx in range(num_episodes):
        reset_opts = _build_reset_options_for_custom_setup(custom_setup, enforce_fixed_skills=True)
        if randomize_offense_permutation:
            perm = _sample_offense_permutation(env)
            perm_dict = {env.offense_ids[i]: perm[i] for i in range(len(env.offense_ids))}
            reset_opts["offense_player_permutation"] = perm_dict

        obs, _ = env.reset(options=reset_opts)
        env.training_team = user_team
        obs["role_flag"] = np.array([role_flag_offense if user_team == Team.OFFENSE else role_flag_defense], dtype=np.float32)
        _accumulate_intent_selection(eval_diagnostics, env)

        done = False
        step_count = 0
        episode_rewards = {"offense": 0.0, "defense": 0.0}
        last_action_results = {}
        episode_shots: dict[str, list[int]] = {}

        while not done:
            player_strategy = IllegalActionStrategy.BEST_PROB if player_deterministic else IllegalActionStrategy.SAMPLE_PROB
            opponent_strategy = IllegalActionStrategy.BEST_PROB if opponent_deterministic else IllegalActionStrategy.SAMPLE_PROB

            actions_player = _predict_policy_actions(
                unified_policy,
                obs,
                env,
                deterministic=player_deterministic,
                strategy=player_strategy,
            )
            actions_opponent = _predict_policy_actions(
                opponent_policy if opponent_policy is not None else unified_policy,
                obs,
                env,
                deterministic=opponent_deterministic,
                strategy=opponent_strategy,
            )

            full_action = np.zeros(env.n_players, dtype=int)
            if user_team == Team.OFFENSE:
                full_action[env.offense_ids] = actions_player[0][env.offense_ids] if isinstance(actions_player, tuple) else actions_player[env.offense_ids]
                full_action[env.defense_ids] = actions_opponent[0][env.defense_ids] if isinstance(actions_opponent, tuple) else actions_opponent[env.defense_ids]
            else:
                full_action[env.defense_ids] = actions_player[0][env.defense_ids] if isinstance(actions_player, tuple) else actions_player[env.defense_ids]
                full_action[env.offense_ids] = actions_opponent[0][env.offense_ids] if isinstance(actions_opponent, tuple) else actions_opponent[env.offense_ids]

            _accumulate_action_mix(eval_diagnostics, full_action, user_team_ids)
            obs, reward, terminated, truncated, info = env.step(full_action)
            action_results = (
                info.get("action_results", {})
                if info
                else {}
            )
            if not action_results:
                action_results = info.get("last_action_results", {}) if info else {}
            if not action_results:
                action_results = getattr(env, "last_action_results", {}) or {}
            last_action_results = action_results

            shots_for_step = action_results.get("shots", {}) if isinstance(action_results, dict) else {}
            _accumulate_assist_links(eval_diagnostics, shots_for_step, user_team_ids_set, env)
            for shooter_id, shot_res in shots_for_step.items():
                try:
                    sid = int(shooter_id)
                    pos = env.positions[sid]
                    q, r = int(pos[0]), int(pos[1])
                    key = f"{q},{r}"
                    if shot_accumulator is not None:
                        if key not in shot_accumulator:
                            shot_accumulator[key] = [0, 0]
                    if key not in episode_shots:
                        episode_shots[key] = [0, 0]
                    if shot_accumulator is not None:
                        shot_accumulator[key][0] += 1
                    episode_shots[key][0] += 1
                    if bool(shot_res.get("success", False)):
                        if shot_accumulator is not None:
                            shot_accumulator[key][1] += 1
                        episode_shots[key][1] += 1
                    _record_shot_for_stats(per_player_stats, sid, env, bool(shot_res.get("success", False)), bool(shot_res.get("assist_full", False)))
                    if "assist_passer_id" in shot_res:
                        _record_assist_for_stats(
                            per_player_stats,
                            int(shot_res.get("assist_passer_id")),
                            bool(shot_res.get("assist_full", False)),
                            bool(shot_res.get("assist_potential", False)),
                        )
                except Exception:
                    continue

            turnovers_raw_step = action_results.get("turnovers", []) if isinstance(action_results, dict) else []
            if isinstance(turnovers_raw_step, (list, tuple)):
                for t in turnovers_raw_step:
                    try:
                        _record_turnover_for_stats(per_player_stats, t.get("player_id"))
                    except Exception:
                        continue
            _accumulate_turnover_reasons(eval_diagnostics, turnovers_raw_step, user_team_ids_set)
            _accumulate_reward_breakdown(
                eval_diagnostics,
                env,
                action_results if isinstance(action_results, dict) else {},
                info if isinstance(info, dict) else {},
                reward,
                user_team,
            )

            obs["role_flag"] = np.array([role_flag_offense if user_team == Team.OFFENSE else role_flag_defense], dtype=np.float32)

            episode_rewards["offense"] += float(reward[env.offense_ids].sum())
            episode_rewards["defense"] += float(reward[env.defense_ids].sum())
            step_count += 1
            done = bool(terminated or truncated)

        shot_clock = getattr(env, "shot_clock", None)
        three_point_distance = float(getattr(env, "three_point_distance", 4.0))
        for pid in per_player_stats:
            per_player_stats[pid]["episodes"] += 1
            per_player_stats[pid]["steps"] += step_count

        results.append(
            {
                "episode": ep_idx + 1,
                "steps": step_count,
                "episode_rewards": episode_rewards,
                "outcome_info": {
                    "shots": last_action_results.get("shots", {}) if isinstance(last_action_results, dict) else {},
                    "turnovers": last_action_results.get("turnovers", []) if isinstance(last_action_results, dict) else [],
                    "defensive_lane_violations": last_action_results.get("defensive_lane_violations", []) if isinstance(last_action_results, dict) else [],
                    "shot_clock": shot_clock,
                    "three_point_distance": three_point_distance,
                },
                "shot_counts": episode_shots,
            }
        )
        if progress_callback is not None:
            try:
                progress_callback(ep_idx + 1, num_episodes)
            except Exception:
                pass
        if (ep_idx + 1) % progress_every == 0 or (ep_idx + 1) == num_episodes:
            _print_eval_progress("Evaluation", ep_idx + 1, num_episodes, progress_start)

    return {
        "results": results,
        "shot_accumulator": shot_accumulator if shot_accumulator is not None else {},
        "per_player_stats": per_player_stats,
        "eval_diagnostics": eval_diagnostics,
    }


def _run_parallel_evaluation(
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
    shot_accumulator: dict[str, list[int]] | None = None,
    custom_setup: dict | None = None,
    randomize_offense_permutation: bool = False,
    num_workers: int | None = None,
    progress_callback=None,
) -> dict:
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 16)
    num_workers = min(num_workers, num_episodes)

    episode_specs = [(i, int(np.random.randint(0, 2**31 - 1))) for i in range(num_episodes)]
    target_batches = max(1, num_workers * 4)
    batch_size = max(1, (num_episodes + target_batches - 1) // target_batches)
    batches = []
    for i in range(0, num_episodes, batch_size):
        batch = episode_specs[i : i + batch_size]
        if batch:
            batches.append((batch, player_deterministic, opponent_deterministic, custom_setup, randomize_offense_permutation))

    print(
        f"[Parallel Evaluation] Using {len(batches)} worker processes for {num_episodes} episodes ({batch_size} episodes/batch)"
    )

    ctx = mp.get_context("spawn")
    progress_queue = ctx.Queue()
    progress_start = time.time()
    completed = 0

    with ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=ctx,
        initializer=_init_evaluation_worker,
        initargs=(
            required_params,
            optional_params,
            unified_policy_path,
            opponent_policy_path,
            user_team_name,
            role_flag_offense,
            role_flag_defense,
            progress_queue,
        ),
    ) as executor:
        futures = {}
        for batch in batches:
            future = executor.submit(_run_episode_batch_worker, batch)
            futures[future] = len(batch[0])
        batch_results = []
        pending = set(futures.keys())
        done_reported = False

        def report_progress() -> None:
            nonlocal done_reported
            if completed >= num_episodes:
                if done_reported:
                    return
                done_reported = True
            if progress_callback is not None:
                try:
                    progress_callback(completed, num_episodes)
                except Exception:
                    pass
            _print_eval_progress("Parallel Eval", completed, num_episodes, progress_start)

        report_progress()
        while pending:
            done, pending = wait(pending, timeout=2.0, return_when=FIRST_COMPLETED)
            drained = False
            while True:
                try:
                    completed += int(progress_queue.get_nowait())
                    drained = True
                except queue.Empty:
                    break
                except Exception:
                    drained = True
                    break
            if drained or not done:
                report_progress()
            if not done:
                continue
            for future in done:
                payload = future.result()
                batch_results.append(payload)
                report_progress()
        while True:
            try:
                completed += int(progress_queue.get_nowait())
            except queue.Empty:
                break
            except Exception:
                break
        report_progress()

    results = []
    merged_player_stats: dict = {}
    merged_eval_diagnostics: dict = _init_eval_diagnostics()
    for payload in batch_results:
        if not payload:
            continue
        if isinstance(payload, dict):
            batch_res = payload.get("results", [])
            results.extend(batch_res)
            if shot_accumulator is not None:
                batch_shots = payload.get("shot_accumulator", {}) or {}
                for key, vals in batch_shots.items():
                    try:
                        att = int(vals[0]) if isinstance(vals, (list, tuple)) and len(vals) > 0 else 0
                        mk = int(vals[1]) if isinstance(vals, (list, tuple)) and len(vals) > 1 else 0
                    except Exception:
                        att, mk = 0, 0
                    if key not in shot_accumulator:
                        shot_accumulator[key] = [0, 0]
                    shot_accumulator[key][0] += att
                    shot_accumulator[key][1] += mk
            merged_player_stats = _merge_player_stats(merged_player_stats, payload.get("per_player_stats") or {})
            merged_eval_diagnostics = _merge_eval_diagnostics(
                merged_eval_diagnostics, payload.get("eval_diagnostics") or {}
            )
        elif isinstance(payload, list):
            results.extend(payload)

    return {
        "results": results,
        "shot_accumulator": shot_accumulator if shot_accumulator is not None else {},
        "per_player_stats": merged_player_stats,
        "eval_diagnostics": merged_eval_diagnostics,
    }


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

    normalized: dict = {}
    shooting_mode = str(setup.get("shooting_mode") or "random").lower()
    if shooting_mode not in ("random", "fixed"):
        raise HTTPException(status_code=400, detail=f"Invalid shooting_mode: {shooting_mode}")
    normalized["shooting_mode"] = shooting_mode

    if setup.get("initial_positions") is not None:
        positions_raw = setup["initial_positions"]
        if not isinstance(positions_raw, (list, tuple)):
            raise HTTPException(status_code=400, detail="initial_positions must be a list.")
        if len(positions_raw) != env.n_players:
            raise HTTPException(
                status_code=400,
                detail=f"initial_positions must have {env.n_players} entries (one per player).",
            )
        positions: list[tuple[int, int]] = []
        seen = set()
        for pos in positions_raw:
            if not isinstance(pos, (list, tuple)) or len(pos) != 2:
                raise HTTPException(status_code=400, detail=f"Invalid position entry: {pos}")
            q, r = int(pos[0]), int(pos[1])
            if not env._is_valid_position(q, r):
                raise HTTPException(status_code=400, detail=f"Position {(q, r)} is out of bounds.")
            if (q, r) in seen:
                raise HTTPException(status_code=400, detail=f"Duplicate position {(q, r)} is not allowed.")
            seen.add((q, r))
            positions.append((q, r))
        normalized["initial_positions"] = positions

    if setup.get("ball_holder") is not None:
        bh = int(setup["ball_holder"])
        if bh < 0 or bh >= env.n_players:
            raise HTTPException(status_code=400, detail=f"Invalid ball_holder id: {bh}")
        normalized["ball_holder"] = bh

    # Validate fixed offense skills if requested
    if shooting_mode == "fixed":
        offense_ids = getattr(env, "offense_ids", [])
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
    unified_policy_path: str,
    opponent_policy_path: str | None,
    user_team_name: str,
    role_flag_offense: float,
    role_flag_defense: float,
    shot_accumulator: dict[str, list[int]] | None = None,
    custom_setup: dict | None = None,
    randomize_offense_permutation: bool = False,
    num_workers: int | None = None,
    progress_callback=None,
):
    if num_workers is None or num_workers <= 1:
        env = basketworld.HexagonBasketballEnv(**required_params, **optional_params, render_mode=None)
        custom_objects = {
            "PassBiasDualCriticPolicy": PassBiasDualCriticPolicy,
            "PassBiasMultiInputPolicy": PassBiasMultiInputPolicy,
            "SetAttentionDualCriticPolicy": SetAttentionDualCriticPolicy,
            "SetAttentionExtractor": SetAttentionExtractor,
        }
        unified_policy = load_ppo_for_inference(
            unified_policy_path,
            device="cpu",
            custom_objects=custom_objects,
        )
        opponent_policy = (
            load_ppo_for_inference(
                opponent_policy_path,
                device="cpu",
                custom_objects=custom_objects,
            )
            if opponent_policy_path
            else None
        )
        pass_mode = str(optional_params.get("pass_mode", "directional"))
        for policy_obj in (unified_policy, opponent_policy):
            policy = getattr(policy_obj, "policy", None) if policy_obj is not None else None
            if policy is None:
                continue
            if hasattr(policy, "set_pass_mode"):
                try:
                    policy.set_pass_mode(pass_mode)
                except Exception:
                    pass
        user_team = Team.OFFENSE if user_team_name == "OFFENSE" else Team.DEFENSE

        return _run_sequential_evaluation(
            num_episodes,
            player_deterministic,
            opponent_deterministic,
            env,
            unified_policy,
            opponent_policy,
            user_team,
            role_flag_offense,
            role_flag_defense,
            shot_accumulator=shot_accumulator,
            custom_setup=custom_setup,
            randomize_offense_permutation=randomize_offense_permutation,
            progress_callback=progress_callback,
        )

    return _run_parallel_evaluation(
        num_episodes,
        player_deterministic,
        opponent_deterministic,
        required_params,
        optional_params,
        unified_policy_path,
        opponent_policy_path,
        user_team_name,
        role_flag_offense,
        role_flag_defense,
        shot_accumulator=shot_accumulator,
        custom_setup=custom_setup,
        randomize_offense_permutation=randomize_offense_permutation,
        num_workers=num_workers,
        progress_callback=progress_callback,
    )


def replay_last_episode():
    """Replay the most recent episode frames and return current state."""
    if not game_state.frames:
        raise ValueError("No frames available for replay")
    try:
        frame = game_state.env.render()
        if frame is not None:
            game_state.frames.append(frame)
    except Exception:
        pass
    if game_state.episode_states:
        return game_state.episode_states[-1]
    return {}


def save_episode_from_pngs(frames: list[str], durations: Optional[list[float]] = None, step_duration_ms: Optional[float] = None) -> str:
    """Save provided base64 PNGs as a GIF and return the file path."""
    import imageio
    import os
    from datetime import datetime

    base_dir = "episodes"
    if getattr(game_state, "run_id", None):
        base_dir = os.path.join(base_dir, str(game_state.run_id))
    os.makedirs(base_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gif_path = os.path.join(base_dir, f"upload_{timestamp}.gif")

    def _decode_b64_png(data: str):
        import base64
        import io
        import PIL.Image

        header = "data:image/png;base64,"
        if data.startswith(header):
            data = data[len(header) :]
        img_bytes = base64.b64decode(data)
        return PIL.Image.open(io.BytesIO(img_bytes))

    images = [_decode_b64_png(frame) for frame in frames]

    # durations: either per-frame list (seconds) or single step duration (ms)
    if durations is None:
        duration = (step_duration_ms or 100) / 1000.0
        durations = [duration] * len(images)
    imageio.mimsave(gif_path, images, duration=durations)
    return gif_path


def pass_steal_preview(env, positions: list[tuple[int, int]], ball_holder: int):
    """Return pass steal probabilities for a hypothetical placement."""
    if env is None:
        raise ValueError("Environment not initialized.")
    if len(positions) != env.n_players:
        raise ValueError(f"positions must have {env.n_players} entries (got {len(positions)})")

    seen = set()
    validated_positions: list[tuple[int, int]] = []
    for pos in positions:
        if not isinstance(pos, (list, tuple)) or len(pos) != 2:
            raise ValueError(f"Invalid position entry: {pos}")
        q, r = int(pos[0]), int(pos[1])
        if not env._is_valid_position(q, r):
            raise ValueError(f"Position {(q, r)} is out of bounds.")
        if (q, r) in seen:
            raise ValueError(f"Duplicate position {(q, r)} not allowed.")
        seen.add((q, r))
        validated_positions.append((q, r))

    ball_holder = int(ball_holder)
    if ball_holder < 0 or ball_holder >= env.n_players:
        raise ValueError(f"Invalid ball_holder id: {ball_holder}")

    orig_positions = list(env.positions)
    orig_ball_holder = env.ball_holder
    orig_obs = game_state.obs
    orig_prev_obs = game_state.prev_obs

    try:
        env.positions = list(validated_positions)
        env.ball_holder = ball_holder
        observer_is_offense = bool(env.training_team == Team.OFFENSE)
        if hasattr(env, "_build_observation_dict"):
            dummy_obs = env._build_observation_dict(observer_is_offense)
        else:
            dummy_obs = {
                "obs": env._get_observation(),
                "action_mask": env._get_action_masks(),
                "role_flag": np.array(
                    [1.0 if env.training_team == Team.OFFENSE else -1.0],
                    dtype=np.float32,
                ),
                "skills": env._get_offense_skills_array(),
            }

        steal_probs = env.calculate_pass_steal_probabilities(ball_holder)
        return {
            "steal_probabilities": {int(k): float(v) for k, v in steal_probs.items()},
            "policy_probabilities": _predict_policy_actions(
                game_state.unified_policy,
                dummy_obs,
                env,
                deterministic=False,
                strategy=IllegalActionStrategy.SAMPLE_PROB,
            )[1],
        }
    finally:
        env.positions = orig_positions
        env.ball_holder = orig_ball_holder
        game_state.obs = orig_obs
        game_state.prev_obs = orig_prev_obs
