import copy
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
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

from app.backend.mcts import _run_mcts_advisor
from app.backend.observations import _predict_policy_actions
from app.backend.state import game_state


# Worker-local storage (each process has its own copy)
_worker_state = {}


def _init_evaluation_worker(
    required_params: dict,
    optional_params: dict,
    unified_policy_path: str,
    opponent_policy_path: str | None,
    user_team_name: str,
    role_flag_offense: float,
    role_flag_defense: float,
):
    """Initialize a worker process with its own environment and policies."""
    global _worker_state

    # Import all required modules for worker functions
    import numpy as _np
    from basketworld.envs.basketworld_env_v2 import Team as _Team
    from basketworld.utils.action_resolution import (
        IllegalActionStrategy as _IllegalActionStrategy,
        get_policy_action_probabilities as _get_policy_action_probabilities,
        resolve_illegal_actions as _resolve_illegal_actions,
    )

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
    unified_policy = PPO.load(unified_policy_path, custom_objects=custom_objects)
    opponent_policy = (
        PPO.load(opponent_policy_path, custom_objects=custom_objects)
        if opponent_policy_path
        else None
    )

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
    }


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
    if "players" in obs:
        cloned["players"] = _np.copy(obs["players"])
    if "globals" in obs:
        cloned["globals"] = _np.copy(obs["globals"])
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

    team_ids = list(env.offense_ids if team == _Team.OFFENSE else env.defense_ids)
    if not team_ids:
        return actions_by_player

    role_flag_value = _worker_role_flag_value(team)
    conditioned_obs = _worker_clone_obs_with_role_flag(base_obs, role_flag_value)

    try:
        raw_actions, _ = policy.predict(conditioned_obs, deterministic=deterministic)
    except Exception:
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

    results = []
    shot_accumulator: dict[str, list[int]] = {}
    per_player_stats = _init_player_stats(env.n_players)

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

            obs, reward, terminated, truncated, info = env.step(full_action)
            action_results = info.get("action_results", {}) if info else {}
            if not action_results:
                action_results = info.get("last_action_results", {}) if info else {}
            if not action_results:
                action_results = getattr(env, "last_action_results", {}) or {}
            last_action_results = action_results

            shots_for_step = action_results.get("shots", {}) if isinstance(action_results, dict) else {}
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
                    "shot_clock": shot_clock,
                    "three_point_distance": three_point_distance,
                },
                "shot_counts": episode_shots,
            }
        )

    return {
        "results": results,
        "shot_accumulator": shot_accumulator if shot_accumulator is not None else {},
        "per_player_stats": per_player_stats,
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
) -> dict:
    results = []
    per_player_stats = _init_player_stats(env.n_players)

    for ep_idx in range(num_episodes):
        reset_opts = _build_reset_options_for_custom_setup(custom_setup, enforce_fixed_skills=True)
        if randomize_offense_permutation:
            perm = _sample_offense_permutation(env)
            perm_dict = {env.offense_ids[i]: perm[i] for i in range(len(env.offense_ids))}
            reset_opts["offense_player_permutation"] = perm_dict

        obs, _ = env.reset(options=reset_opts)
        env.training_team = user_team
        obs["role_flag"] = np.array([role_flag_offense if user_team == Team.OFFENSE else role_flag_defense], dtype=np.float32)

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
                    "shot_clock": shot_clock,
                    "three_point_distance": three_point_distance,
                },
                "shot_counts": episode_shots,
            }
        )

    return {
        "results": results,
        "shot_accumulator": shot_accumulator if shot_accumulator is not None else {},
        "per_player_stats": per_player_stats,
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
) -> dict:
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 16)
    num_workers = min(num_workers, num_episodes)

    episode_specs = [(i, int(np.random.randint(0, 2**31 - 1))) for i in range(num_episodes)]
    batch_size = (num_episodes + num_workers - 1) // num_workers
    batches = []
    for i in range(0, num_episodes, batch_size):
        batch = episode_specs[i : i + batch_size]
        if batch:
            batches.append((batch, player_deterministic, opponent_deterministic, custom_setup, randomize_offense_permutation))

    print(
        f"[Parallel Evaluation] Using {len(batches)} worker processes for {num_episodes} episodes ({batch_size} episodes/batch)"
    )

    ctx = mp.get_context("spawn")

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
        ),
    ) as executor:
        batch_results = list(executor.map(_run_episode_batch_worker, batches))

    results = []
    merged_player_stats: dict = {}
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
        elif isinstance(payload, list):
            results.extend(payload)

    return {
        "results": results,
        "shot_accumulator": shot_accumulator if shot_accumulator is not None else {},
        "per_player_stats": merged_player_stats,
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
):
    if num_workers is None or num_workers <= 1:
        env = basketworld.HexagonBasketballEnv(**required_params, **optional_params, render_mode=None)
        custom_objects = {
            "PassBiasDualCriticPolicy": PassBiasDualCriticPolicy,
            "PassBiasMultiInputPolicy": PassBiasMultiInputPolicy,
            "SetAttentionDualCriticPolicy": SetAttentionDualCriticPolicy,
            "SetAttentionExtractor": SetAttentionExtractor,
        }
        unified_policy = PPO.load(unified_policy_path, custom_objects=custom_objects)
        opponent_policy = (
            PPO.load(opponent_policy_path, custom_objects=custom_objects)
            if opponent_policy_path
            else None
        )
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
        obs_vec = env._get_observation()
        action_mask = env._get_action_masks()

        dummy_obs = {
            "obs": obs_vec,
            "action_mask": action_mask,
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
