#!/usr/bin/env python3
"""
Evaluation script to analyze the performance of trained self-play agents.

This script connects to an MLflow run, downloads the latest trained policies,
runs them against each other, computes performance metrics, and logs the
resulting analysis and visualization GIFs back to the original MLflow run.
"""
import argparse
import os
import numpy as np
import tempfile
import re
from stable_baselines3 import PPO
import basketworld
from basketworld.envs.basketworld_env_v2 import Team
import mlflow
from basketworld.utils.evaluation_helpers import (
    get_outcome_category,
    create_and_log_gif,
)
import csv
from collections import defaultdict
from tqdm import tqdm

from basketworld.utils.mlflow_params import get_mlflow_params
from basketworld.envs.basketworld_env_v2 import HexagonBasketballEnv
from basketworld.utils.action_resolution import (
    IllegalActionStrategy,
    get_policy_action_probabilities,
    resolve_illegal_actions,
)


def analyze_results(results: list, num_episodes: int):
    """Analyzes and prints the evaluation results."""
    print("\n--- Evaluation Results ---")

    outcomes = defaultdict(int)
    episode_lengths = []
    shot_distances = []
    total_pass_attempts = 0
    total_pass_completions = 0
    total_pass_intercepts = 0
    total_pass_oob = 0
    total_potential_assists = 0
    total_potential_assisted_2pt = 0
    total_potential_assisted_3pt = 0
    total_potential_assisted_dunk = 0
    total_assists = 0
    total_assisted_2pt = 0
    total_assisted_3pt = 0
    total_assisted_dunk = 0
    avg_reward_offense_total = 0.0  # per-player average reward (unused in final report)
    avg_reward_defense_total = 0.0  # per-player average reward (unused in final report)
    team_reward_offense_total = 0.0  # team-summed reward (matches training logs)
    for res in results:
        outcomes[res["outcome"]] += 1
        episode_lengths.append(res["length"])
        total_pass_attempts += res.get("pass_attempts", 0)
        total_pass_completions += res.get("pass_completions", 0)
        total_pass_intercepts += res.get("pass_intercepts", 0)
        total_pass_oob += res.get("pass_oob", 0)
        total_potential_assists += res.get("potential_assists", 0)
        total_potential_assisted_2pt += res.get("potential_assisted_2pt", 0)
        total_potential_assisted_3pt += res.get("potential_assisted_3pt", 0)
        total_potential_assisted_dunk += res.get("potential_assisted_dunk", 0)
        total_assists += res.get("assists", 0)
        total_assisted_2pt += res.get("assisted_2pt", 0)
        total_assisted_3pt += res.get("assisted_3pt", 0)
        total_assisted_dunk += res.get("assisted_dunk", 0)
        avg_reward_offense_total += float(res.get("avg_reward_offense", 0.0))
        avg_reward_defense_total += float(res.get("avg_reward_defense", 0.0))
        team_reward_offense_total += float(res.get("team_reward_offense", 0.0))
    print(f"Shot Distances Mean: {np.mean(shot_distances)}")
    print(f"Shot Distances Std: {np.std(shot_distances)}")
    print(f"Total Episodes: {num_episodes}\n")

    # --- Episode Length Statistics ---
    avg_len = np.mean(episode_lengths)
    std_len = np.std(episode_lengths)
    min_len = np.min(episode_lengths)
    max_len = np.max(episode_lengths)
    print("Episode Length Stats:")
    print(f"  - Mean: {avg_len:.2f}")
    print(f"  - Std Dev: {std_len:.2f}")
    print(f"  - Min/Max: {min_len}/{max_len}\n")

    # --- Scoring and Outcome Statistics ---
    made_2pts = outcomes.get("Made 2pt", 0)
    made_3pts = outcomes.get("Made 3pt", 0)
    missed_2pts = outcomes.get("Missed 2pt", 0)
    missed_3pts = outcomes.get("Missed 3pt", 0)
    made_dunks = outcomes.get("Made Dunk", 0)
    missed_dunks = outcomes.get("Missed Dunk", 0)
    score_rate = (made_2pts + made_3pts + made_dunks) / num_episodes
    turnovers = (
        outcomes.get("Turnover (Pressure)", 0)
        + outcomes.get("Turnover (OOB - Move)", 0)
        + outcomes.get("Turnover (OOB - Pass)", 0)
        + outcomes.get("Turnover (Intercepted)", 0)
        + outcomes.get("Turnover (Shot Clock Violation)", 0)
    )
    total_made = made_2pts + made_3pts + made_dunks
    total_missed = missed_2pts + missed_3pts + missed_dunks
    total_shots = total_made + total_missed
    print(f"Offensive Score Rate: {100.0 * score_rate:.2f}%")
    print(f"Made 2pts: {made_2pts}")
    print(f"Missed 2pts: {missed_2pts}")
    print(f"Made 3pts: {made_3pts}")
    print(f"Missed 3pts: {missed_3pts}")
    print(f"Made Dunks: {made_dunks}")
    print(f"Missed Dunks: {missed_dunks}")
    print(f"Total made shots: {total_made}")
    print(f"Total missed shots: {total_missed}")
    print(f"Total shots: {total_shots}")
    print(f"Total turnovers: {turnovers}")
    if turnovers > 0:
        print(f"  - OOB (Move): {outcomes.get('Turnover (OOB - Move)', 0)}")
        print(f"  - OOB (Pass): {outcomes.get('Turnover (OOB - Pass)', 0)}")
        print(f"  - Intercepted: {outcomes.get('Turnover (Intercepted)', 0)}")
        print(f"  - Pressure: {outcomes.get('Turnover (Pressure)', 0)}")
        print(f"  - Shot Clock: {outcomes.get('Turnover (Shot Clock Violation)', 0)}")
    # --- Passing Statistics ---
    print("\nPassing Stats:")
    print(f"Pass attempts: {total_pass_attempts}")
    print(f"Completed passes: {total_pass_completions}")
    if total_pass_attempts > 0:
        print(
            f"Pass completion%: {100.0 * total_pass_completions / total_pass_attempts:.2f}%"
        )
    else:
        print("Pass completion%: N/A")
    print(f"Intercepted passes: {total_pass_intercepts}")
    print(f"Out-of-bounds passes: {total_pass_oob}")
    # --- Assist Statistics ---
    print("\nAssist Stats:")
    print(f"Potential assists (missed): {total_potential_assists}")
    print(f"  - Potential assisted 2pt misses: {total_potential_assisted_2pt}")
    print(f"  - Potential assisted 3pt misses: {total_potential_assisted_3pt}")
    print(f"  - Potential assisted dunk misses: {total_potential_assisted_dunk}")
    print(f"Assists: {total_assists}")
    print(f"Assisted 2pt FGM: {total_assisted_2pt}")
    print(f"Assisted 3pt FGM: {total_assisted_3pt}")
    print(f"Assisted dunk FGM: {total_assisted_dunk}")
    # --- Reward Statistics ---
    if num_episodes > 0:
        print("\nAverage Team Reward (offense only; defense symmetric):")
        print(f"Offense team reward: {team_reward_offense_total / num_episodes:.4f}")
    # Traditional 2PT% excludes dunks for separate reporting
    if (made_2pts + missed_2pts) > 0:
        print(f"2PT% (non-dunk): {100.0 * made_2pts / (made_2pts + missed_2pts):.2f}%")
    else:
        print("2PT% (non-dunk): N/A")
    if (made_dunks + missed_dunks) > 0:
        print(f"Dunk%: {100.0 * made_dunks / (made_dunks + missed_dunks):.2f}%")
    else:
        print("Dunk%: N/A")
    if made_3pts + missed_3pts > 0:
        print(f"3PT%: {100.0 * made_3pts / (made_3pts + missed_3pts):.2f}%")
    else:
        print("3PT%: N/A")
    if total_shots > 0:
        print(f"FG%: {100.0 * total_made / total_shots:.2f}%")
        print(
            f"EFG%: {100.0 * (made_2pts + made_dunks + made_3pts * 1.5) / total_shots:.2f}%"
        )
    else:
        print("FG%: N/A")
        print("EFG%: N/A")
    if (total_shots + turnovers) > 0:
        print(
            f"PPP: {2.0 * (made_2pts + made_dunks + made_3pts * 1.5) / (total_shots + turnovers):.2f}"
        )
    else:
        print("PPP: N/A")
    print("\nEpisode Termination Breakdown:")
    for outcome, count in sorted(outcomes.items()):
        percentage = (count / num_episodes) * 100
        print(f"- {outcome}: {count}/{num_episodes} ({percentage:.2f}%)")


def list_models_by_alternation(client, run_id: str):
    """Return dict alt_idx -> { 'offense': path, 'defense': path } for available pairs.
    Backward-compat; unified artifacts are handled separately.
    """
    artifacts = client.list_artifacts(run_id, "models")
    offense = [
        f.path
        for f in artifacts
        if f.path.endswith(".zip")
        and ("offense_policy" in f.path or "offense" in f.path)
    ]
    defense = [
        f.path
        for f in artifacts
        if f.path.endswith(".zip")
        and ("defense_policy" in f.path or "defense" in f.path)
    ]

    def idx_of(p):
        # Prefer explicit 'alt_<n>.zip'; fall back to last '_<n>.zip'
        m = re.search(r"alt_(\d+)\.zip$", p)
        if not m:
            m = re.search(r"_(\d+)\.zip$", p)
        return int(m.group(1)) if m else None

    off_map = {idx_of(p): p for p in offense if idx_of(p) is not None}
    def_map = {idx_of(p): p for p in defense if idx_of(p) is not None}

    common_idxs = sorted(set(off_map.keys()) & set(def_map.keys()))
    result = {i: {"offense": off_map[i], "defense": def_map[i]} for i in common_idxs}
    return result


def list_unified_by_alternation(client, run_id: str):
    """Return dict alt_idx -> unified path for unified_policy artifacts."""
    artifacts = client.list_artifacts(run_id, "models")
    unified = [
        f.path
        for f in artifacts
        if f.path.endswith(".zip")
        and ("unified_policy" in f.path or "unified" in f.path)
    ]

    def idx_of(p):
        m = re.search(r"alt_(\d+)\.zip$", p)
        if not m:
            m = re.search(r"_(\d+)\.zip$", p)
        return int(m.group(1)) if m else None

    uni_map = {idx_of(p): p for p in unified if idx_of(p) is not None}
    return {i: uni_map[i] for i in sorted(uni_map.keys())}


def run_eval_for_pair(
    offense_policy_path: str,
    defense_policy_path: str,
    num_episodes: int,
    required,
    optional,
    args,
    client,
    run_id: str,
    temp_dir: str,
    role_flag_offense: float = 1.0,
    role_flag_defense: float = 0.0,
):
    env = HexagonBasketballEnv(
        **required,
        **optional,
        render_mode="rgb_array" if not args.no_render else None,
    )

    offense_policy = PPO.load(offense_policy_path)
    defense_policy = PPO.load(defense_policy_path)

    results = []
    for i in range(num_episodes):
        obs, info = env.reset()
        done = False
        offense_ids = env.offense_ids

        episode_frames = []
        pass_attempts = 0
        pass_completions = 0
        pass_intercepts = 0
        pass_oob = 0
        cumulative_rewards = np.zeros(env.n_players, dtype=float)
        if not args.no_render:
            frame = env.render()
            if frame is not None:
                episode_frames.append(frame)

        # Choose illegal-action strategies (overrides keep current defaults if not set)
        offense_strategy = IllegalActionStrategy.SAMPLE_PROB
        if getattr(args, "offense_illegal_strategy", None):
            s = str(args.offense_illegal_strategy).lower()
            if s == "noop":
                offense_strategy = IllegalActionStrategy.NOOP
            elif s == "best":
                offense_strategy = IllegalActionStrategy.BEST_PROB
            else:
                offense_strategy = IllegalActionStrategy.SAMPLE_PROB

        defense_strategy = IllegalActionStrategy.BEST_PROB
        if getattr(args, "defense_illegal_strategy", None):
            s = str(args.defense_illegal_strategy).lower()
            if s == "noop":
                defense_strategy = IllegalActionStrategy.NOOP
            elif s == "sample":
                defense_strategy = IllegalActionStrategy.SAMPLE_PROB
            else:
                defense_strategy = IllegalActionStrategy.BEST_PROB

        while not done:
            # Build role-conditional observations (match SelfPlayEnvWrapper/backend)
            offense_obs = dict(obs)
            defense_obs = dict(obs)
            try:
                offense_obs["role_flag"] = np.array([role_flag_offense], dtype=np.float32)
                defense_obs["role_flag"] = np.array([role_flag_defense], dtype=np.float32)
            except Exception:
                pass

            offense_action, _ = offense_policy.predict(
                offense_obs, deterministic=args.deterministic_offense
            )
            defense_action, _ = defense_policy.predict(
                defense_obs, deterministic=args.deterministic_defense
            )

            action_mask = obs.get("action_mask")
            offense_probs = get_policy_action_probabilities(offense_policy, offense_obs)
            defense_probs = get_policy_action_probabilities(defense_policy, defense_obs)

            offense_resolved = resolve_illegal_actions(
                np.array(offense_action),
                action_mask,
                offense_strategy,
                args.deterministic_offense,
                offense_probs,
            )
            defense_resolved = resolve_illegal_actions(
                np.array(defense_action),
                action_mask,
                defense_strategy,
                args.deterministic_defense,
                defense_probs,
            )

            full_action = np.zeros(env.n_players, dtype=int)
            for player_id in range(env.n_players):
                if player_id in offense_ids:
                    full_action[player_id] = int(offense_resolved[player_id])
                else:
                    full_action[player_id] = int(defense_resolved[player_id])

            obs, reward, done, _, info = env.step(full_action)
            cumulative_rewards += reward

            step_results = info.get("action_results", {})
            if step_results.get("passes"):
                attempts_this_step = len(step_results["passes"])
                completions_this_step = sum(
                    1
                    for _pid, pres in step_results["passes"].items()
                    if pres.get("success")
                )
                intercepts_this_step = sum(
                    1
                    for _pid, pres in step_results["passes"].items()
                    if not pres.get("success") and pres.get("reason") == "intercepted"
                )
                oob_this_step = sum(
                    1
                    for _pid, pres in step_results["passes"].items()
                    if not pres.get("success") and pres.get("reason") == "out_of_bounds"
                )
                pass_attempts += attempts_this_step
                pass_completions += completions_this_step
                pass_intercepts += intercepts_this_step
                pass_oob += oob_this_step

            if not args.no_render:
                frame = env.render()
                if frame is not None:
                    episode_frames.append(frame)

        # Determine outcome
        final_info = info
        action_results = final_info.get("action_results", {})
        outcome = "Unknown"
        three_point_distance = env.three_point_distance
        potential_assists = 0
        assists = 0
        assisted_2pt = 0
        assisted_3pt = 0
        assisted_dunk = 0
        potential_assisted_2pt = 0
        potential_assisted_3pt = 0
        potential_assisted_dunk = 0
        if action_results.get("shots"):
            shot_result = list(action_results["shots"].values())[0]
            is_dunk = shot_result.get("distance", 999) == 0
            if is_dunk:
                outcome = "Made Dunk" if shot_result["success"] else "Missed Dunk"
            elif (
                shot_result["success"]
                and shot_result["distance"] < three_point_distance
            ):
                outcome = "Made 2pt"
            elif (
                shot_result["success"]
                and shot_result["distance"] >= three_point_distance
            ):
                outcome = "Made 3pt"
            elif (
                not shot_result["success"]
                and shot_result["distance"] < three_point_distance
            ):
                outcome = "Missed 2pt"
            elif (
                not shot_result["success"]
                and shot_result["distance"] >= three_point_distance
            ):
                outcome = "Missed 3pt"
            # Assist flags
            if shot_result.get("assist_potential") and not shot_result.get("success"):
                potential_assists += 1
                if is_dunk:
                    potential_assisted_dunk += 1
                elif shot_result.get("distance", 999) >= three_point_distance:
                    potential_assisted_3pt += 1
                else:
                    potential_assisted_2pt += 1
            if shot_result.get("assist_full"):
                assists += 1
                if shot_result.get("success"):
                    if is_dunk:
                        assisted_dunk += 1
                    elif shot_result.get("distance", 999) >= three_point_distance:
                        assisted_3pt += 1
                    else:
                        assisted_2pt += 1
        elif action_results.get("turnovers"):
            turnover_reason = action_results["turnovers"][0]["reason"]
            if turnover_reason == "intercepted":
                outcome = "Turnover (Intercepted)"
            elif turnover_reason == "pass_out_of_bounds":
                outcome = "Turnover (OOB - Pass)"
            elif turnover_reason == "move_out_of_bounds":
                outcome = "Turnover (OOB - Move)"
            elif turnover_reason == "defender_pressure":
                outcome = "Turnover (Pressure)"
        elif env.unwrapped.shot_clock <= 0:
            outcome = "Turnover (Shot Clock Violation)"

        # Rewards by team for this episode
        # Per-player average (historically printed in analysis)
        avg_reward_offense = (
            float(np.mean(cumulative_rewards[env.offense_ids]))
            if env.offense_ids
            else 0.0
        )
        avg_reward_defense = (
            float(np.mean(cumulative_rewards[env.defense_ids]))
            if env.defense_ids
            else 0.0
        )
        # Team-summed reward (matches training "Mean Episode Reward" which aggregates over teammates)
        team_reward_offense = (
            float(np.sum(cumulative_rewards[env.offense_ids]))
            if env.offense_ids
            else 0.0
        )
        team_reward_defense = (
            float(np.sum(cumulative_rewards[env.defense_ids]))
            if env.defense_ids
            else 0.0
        )

        results.append(
            {
                "outcome": outcome,
                "length": env.unwrapped.step_count,
                "pass_attempts": pass_attempts,
                "pass_completions": pass_completions,
                "pass_intercepts": pass_intercepts,
                "pass_oob": pass_oob,
                "potential_assists": potential_assists,
                "potential_assisted_2pt": potential_assisted_2pt,
                "potential_assisted_3pt": potential_assisted_3pt,
                "potential_assisted_dunk": potential_assisted_dunk,
                "assists": assists,
                "assisted_2pt": assisted_2pt,
                "assisted_3pt": assisted_3pt,
                "assisted_dunk": assisted_dunk,
                "avg_reward_offense": avg_reward_offense,
                "avg_reward_defense": avg_reward_defense,
                "team_reward_offense": team_reward_offense,
                "team_reward_defense": team_reward_defense,
            }
        )

        if not args.no_render and args.log_gifs:
            # Optional per-episode logging if requested
            valid_frames = [f for f in episode_frames if f is not None]
            if valid_frames:
                create_and_log_gif(
                    frames=valid_frames,
                    episode_num=i,
                    outcome=outcome,
                    temp_dir=temp_dir,
                    artifact_path=f"gifs/{get_outcome_category(outcome)}",
                )

    return results


def run_eval_for_unified(
    unified_policy_path: str,
    num_episodes: int,
    required,
    optional,
    args,
    client,
    run_id: str,
    temp_dir: str,
):
    env = HexagonBasketballEnv(
        **required,
        **optional,
        render_mode="rgb_array" if not args.no_render else None,
    )

    policy = PPO.load(unified_policy_path)

    results = []
    for i in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_frames = []
        pass_attempts = 0
        pass_completions = 0
        pass_intercepts = 0
        pass_oob = 0
        cumulative_rewards = np.zeros(env.n_players, dtype=float)
        if not args.no_render:
            frame = env.render()
            if frame is not None:
                episode_frames.append(frame)

        # Resolve strategy for unified policy (default: sample)
        unified_strategy = IllegalActionStrategy.SAMPLE_PROB
        if getattr(args, "unified_illegal_strategy", None):
            s = str(args.unified_illegal_strategy).lower()
            if s == "noop":
                unified_strategy = IllegalActionStrategy.NOOP
            elif s == "best":
                unified_strategy = IllegalActionStrategy.BEST_PROB

        while not done:
            predicted, _ = policy.predict(obs, deterministic=args.deterministic_unified)
            action_mask = obs.get("action_mask")
            probs = get_policy_action_probabilities(policy, obs)
            resolved = resolve_illegal_actions(
                np.array(predicted),
                action_mask,
                unified_strategy,
                args.deterministic_unified,
                probs,
            )
            obs, reward, done, _, info = env.step(resolved)
            cumulative_rewards += reward

            step_results = info.get("action_results", {})
            if step_results.get("passes"):
                attempts_this_step = len(step_results["passes"])
                completions_this_step = sum(
                    1
                    for _pid, pres in step_results["passes"].items()
                    if pres.get("success")
                )
                intercepts_this_step = sum(
                    1
                    for _pid, pres in step_results["passes"].items()
                    if not pres.get("success") and pres.get("reason") == "intercepted"
                )
                oob_this_step = sum(
                    1
                    for _pid, pres in step_results["passes"].items()
                    if not pres.get("success") and pres.get("reason") == "out_of_bounds"
                )
                pass_attempts += attempts_this_step
                pass_completions += completions_this_step
                pass_intercepts += intercepts_this_step
                pass_oob += oob_this_step

            if not args.no_render:
                frame = env.render()
                if frame is not None:
                    episode_frames.append(frame)

        final_info = info
        action_results = final_info.get("action_results", {})
        outcome = "Unknown"
        three_point_distance = env.three_point_distance
        potential_assists = 0
        assists = 0
        assisted_2pt = 0
        assisted_3pt = 0
        assisted_dunk = 0
        potential_assisted_2pt = 0
        potential_assisted_3pt = 0
        potential_assisted_dunk = 0
        if action_results.get("shots"):
            shot_result = list(action_results["shots"].values())[0]
            is_dunk = shot_result.get("distance", 999) == 0
            if is_dunk:
                outcome = "Made Dunk" if shot_result["success"] else "Missed Dunk"
            elif (
                shot_result["success"]
                and shot_result["distance"] < three_point_distance
            ):
                outcome = "Made 2pt"
            elif (
                shot_result["success"]
                and shot_result["distance"] >= three_point_distance
            ):
                outcome = "Made 3pt"
            elif (
                not shot_result["success"]
                and shot_result["distance"] < three_point_distance
            ):
                outcome = "Missed 2pt"
            elif (
                not shot_result["success"]
                and shot_result["distance"] >= three_point_distance
            ):
                outcome = "Missed 3pt"
            # Assist flags
            if shot_result.get("assist_potential") and not shot_result.get("success"):
                potential_assists += 1
                if is_dunk:
                    potential_assisted_dunk += 1
                elif shot_result.get("distance", 999) >= three_point_distance:
                    potential_assisted_3pt += 1
                else:
                    potential_assisted_2pt += 1
            if shot_result.get("assist_full"):
                assists += 1
                if shot_result.get("success"):
                    if is_dunk:
                        assisted_dunk += 1
                    elif shot_result.get("distance", 999) >= three_point_distance:
                        assisted_3pt += 1
                    else:
                        assisted_2pt += 1
        elif action_results.get("turnovers"):
            turnover_reason = action_results["turnovers"][0]["reason"]
            if turnover_reason == "intercepted":
                outcome = "Turnover (Intercepted)"
            elif turnover_reason == "pass_out_of_bounds":
                outcome = "Turnover (OOB - Pass)"
            elif turnover_reason == "move_out_of_bounds":
                outcome = "Turnover (OOB - Move)"
            elif turnover_reason == "defender_pressure":
                outcome = "Turnover (Pressure)"
        elif env.unwrapped.shot_clock <= 0:
            outcome = "Turnover (Shot Clock Violation)"

        # Rewards by team for this episode
        # Per-player average (historically printed in analysis)
        avg_reward_offense = (
            float(np.mean(cumulative_rewards[env.offense_ids]))
            if env.offense_ids
            else 0.0
        )
        avg_reward_defense = (
            float(np.mean(cumulative_rewards[env.defense_ids]))
            if env.defense_ids
            else 0.0
        )
        # Team-summed reward (matches training logs)
        team_reward_offense = (
            float(np.sum(cumulative_rewards[env.offense_ids]))
            if env.offense_ids
            else 0.0
        )
        team_reward_defense = (
            float(np.sum(cumulative_rewards[env.defense_ids]))
            if env.defense_ids
            else 0.0
        )

        results.append(
            {
                "outcome": outcome,
                "length": env.unwrapped.step_count,
                "pass_attempts": pass_attempts,
                "pass_completions": pass_completions,
                "pass_intercepts": pass_intercepts,
                "pass_oob": pass_oob,
                "potential_assists": potential_assists,
                "potential_assisted_2pt": potential_assisted_2pt,
                "potential_assisted_3pt": potential_assisted_3pt,
                "potential_assisted_dunk": potential_assisted_dunk,
                "assists": assists,
                "assisted_2pt": assisted_2pt,
                "assisted_3pt": assisted_3pt,
                "assisted_dunk": assisted_dunk,
                "avg_reward_offense": avg_reward_offense,
                "avg_reward_defense": avg_reward_defense,
                "team_reward_offense": team_reward_offense,
                "team_reward_defense": team_reward_defense,
            }
        )

        if not args.no_render and args.log_gifs:
            valid_frames = [f for f in episode_frames if f is not None]
            if valid_frames:
                create_and_log_gif(
                    frames=valid_frames,
                    episode_num=i,
                    outcome=outcome,
                    temp_dir=temp_dir,
                    artifact_path=f"gifs/{get_outcome_category(outcome)}",
                )

    return results


def summarize_to_row(results: list, alternation_index: int):
    outcomes = defaultdict(int)
    episode_lengths = []
    total_pass_attempts = 0
    total_pass_completions = 0
    total_pass_intercepts = 0
    total_pass_oob = 0
    total_potential_assists = 0
    total_potential_assisted_2pt = 0
    total_potential_assisted_3pt = 0
    total_potential_assisted_dunk = 0
    total_assists = 0
    total_assisted_2pt = 0
    total_assisted_3pt = 0
    total_assisted_dunk = 0
    avg_reward_offense_total = 0.0  # per-player average (unused in final row)
    avg_reward_defense_total = 0.0  # per-player average (unused in final row)
    team_reward_offense_total = 0.0  # team-summed
    for res in results:
        outcomes[res["outcome"]] += 1
        episode_lengths.append(res["length"])
        total_pass_attempts += res.get("pass_attempts", 0)
        total_pass_completions += res.get("pass_completions", 0)
        total_pass_intercepts += res.get("pass_intercepts", 0)
        total_pass_oob += res.get("pass_oob", 0)
        total_potential_assists += res.get("potential_assists", 0)
        total_potential_assisted_2pt += res.get("potential_assisted_2pt", 0)
        total_potential_assisted_3pt += res.get("potential_assisted_3pt", 0)
        total_potential_assisted_dunk += res.get("potential_assisted_dunk", 0)
        total_assists += res.get("assists", 0)
        total_assisted_2pt += res.get("assisted_2pt", 0)
        total_assisted_3pt += res.get("assisted_3pt", 0)
        avg_reward_offense_total += float(res.get("avg_reward_offense", 0.0))
        avg_reward_defense_total += float(res.get("avg_reward_defense", 0.0))
        team_reward_offense_total += float(res.get("team_reward_offense", 0.0))
        total_assisted_dunk += res.get("assisted_dunk", 0)

    num_episodes = max(1, len(results))
    avg_len = float(np.mean(episode_lengths)) if episode_lengths else 0.0
    made_2pts = outcomes.get("Made 2pt", 0)
    made_3pts = outcomes.get("Made 3pt", 0)
    missed_2pts = outcomes.get("Missed 2pt", 0)
    missed_3pts = outcomes.get("Missed 3pt", 0)
    made_dunks = outcomes.get("Made Dunk", 0)
    missed_dunks = outcomes.get("Missed Dunk", 0)
    total_made = made_2pts + made_3pts + made_dunks
    total_missed = missed_2pts + missed_3pts + missed_dunks
    total_shots = total_made + total_missed
    turnovers = (
        outcomes.get("Turnover (Pressure)", 0)
        + outcomes.get("Turnover (OOB - Move)", 0)
        + outcomes.get("Turnover (OOB - Pass)", 0)
        + outcomes.get("Turnover (Intercepted)", 0)
        + outcomes.get("Turnover (Shot Clock Violation)", 0)
    )

    row = {
        "alternation": alternation_index,
        "episodes": num_episodes,
        "avg_len": avg_len,
        "score_rate": total_made / num_episodes if num_episodes else 0.0,
        "made_2pt": made_2pts,
        "missed_2pt": missed_2pts,
        "made_3pt": made_3pts,
        "missed_3pt": missed_3pts,
        "made_dunk": made_dunks,
        "missed_dunk": missed_dunks,
        "total_shots": total_shots,
        "turnovers": turnovers,
        "pass_attempts": total_pass_attempts,
        "pass_completions": total_pass_completions,
        "pass_intercepts": total_pass_intercepts,
        "pass_oob": total_pass_oob,
        "potential_assists": total_potential_assists,
        "potential_assisted_2pt": total_potential_assisted_2pt,
        "potential_assisted_3pt": total_potential_assisted_3pt,
        "potential_assisted_dunk": total_potential_assisted_dunk,
        "assists": total_assists,
        "assisted_2pt": total_assisted_2pt,
        "assisted_3pt": total_assisted_3pt,
        "assisted_dunk": total_assisted_dunk,
        "pct_2pt": (
            (made_2pts / (made_2pts + missed_2pts))
            if (made_2pts + missed_2pts) > 0
            else 0.0
        ),
        "pct_3pt": (
            (made_3pts / (made_3pts + missed_3pts))
            if (made_3pts + missed_3pts) > 0
            else 0.0
        ),
        "pct_dunk": (
            (made_dunks / (made_dunks + missed_dunks))
            if (made_dunks + missed_dunks) > 0
            else 0.0
        ),
        "fg_pct": (total_made / total_shots) if total_shots > 0 else 0.0,
        "efg_pct": (
            ((made_2pts + made_dunks + 1.5 * made_3pts) / total_shots)
            if total_shots > 0
            else 0.0
        ),
        "ppp": (
            (
                2.0
                * (made_2pts + made_dunks + 1.5 * made_3pts)
                / (total_shots + turnovers)
            )
            if (total_shots + turnovers) > 0
            else 0.0
        ),
        # Team-summed reward for offense only (defense symmetric)
        "avg_team_reward_offense": (
            (team_reward_offense_total / num_episodes) if num_episodes else 0.0
        ),
    }
    return row


def main(args):
    """Main evaluation function."""

    # --- Set up MLflow Tracking ---
    from basketworld.utils.mlflow_config import setup_mlflow

    setup_mlflow(verbose=False)
    client = mlflow.tracking.MlflowClient()

    try:
        run = client.get_run(args.run_id)
    except Exception as e:
        print(
            f"Error: Could not find MLflow run with ID '{args.run_id}'. Please ensure the Run ID is correct and the MLflow server is running."
        )
        print(e)
        return

    # --- Get Hyperparameters from MLflow Run ---
    print("Fetching hyperparameters from MLflow run...")
    try:
        required, optional = get_mlflow_params(client, args.run_id)

    except KeyError as e:
        print(f"Error: Run {args.run_id} is missing a required parameter: {e}")
        return

    # Extract role_flag encoding for backward compatibility (not passed to env)
    role_flag_offense = optional.pop("role_flag_offense_value")
    role_flag_defense = optional.pop("role_flag_defense_value")
    encoding_version = optional.pop("role_flag_encoding_version")
    print(f"[EVALUATE] Using role_flag encoding ({encoding_version}): offense={role_flag_offense}, defense={role_flag_defense}")

    # Re-open the original run context to log new artifacts to the correct run
    with mlflow.start_run(run_id=args.run_id):
        with tempfile.TemporaryDirectory() as temp_dir:
            if args.all_alternations:
                print("Evaluating across all alternations...")
                rows = []
                pairs = list_models_by_alternation(client, args.run_id)
                uni = list_unified_by_alternation(client, args.run_id)
                if args.use_unified or (not pairs and uni):
                    for alt_idx, uni_art in tqdm(
                        uni.items(), desc="Alternations (unified)"
                    ):
                        uni_path = client.download_artifacts(
                            args.run_id, uni_art, temp_dir
                        )
                        results = run_eval_for_unified(
                            uni_path,
                            args.episodes,
                            required,
                            optional,
                            args,
                            client,
                            args.run_id,
                            temp_dir,
                        )
                        row = summarize_to_row(results, alt_idx)
                        rows.append(row)
                elif pairs:
                    for alt_idx, pair in tqdm(
                        pairs.items(), desc="Alternations (paired)"
                    ):
                        offense_policy_path = client.download_artifacts(
                            args.run_id, pair["offense"], temp_dir
                        )
                        defense_policy_path = client.download_artifacts(
                            args.run_id, pair["defense"], temp_dir
                        )
                        results = run_eval_for_pair(
                            offense_policy_path,
                            defense_policy_path,
                            args.episodes,
                            required,
                            optional,
                            args,
                            client,
                            args.run_id,
                            temp_dir,
                            role_flag_offense,
                            role_flag_defense,
                        )
                        row = summarize_to_row(results, alt_idx)
                        rows.append(row)
                else:
                    print("No artifacts found under models/ for paired or unified.")

                # Write CSV
                csv_path = os.path.join(temp_dir, "evaluation_by_alternation.csv")
                if rows:
                    fieldnames = list(rows[0].keys())
                    with open(csv_path, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(rows)
                    mlflow.log_artifact(csv_path, artifact_path="metrics")
                    print(f"Logged CSV: {csv_path}")
                else:
                    print("No rows to write.")
            else:
                # --- Download latest matched pair (by alternation) and run single evaluation set ---
                print(f"Fetching latest models from MLflow Run ID: {args.run_id}")
                pairs = list_models_by_alternation(client, args.run_id)
                uni = list_unified_by_alternation(client, args.run_id)
                if args.use_unified or (not pairs and uni):
                    if not uni:
                        print("No unified artifacts found under 'models/'.")
                        artifacts = client.list_artifacts(args.run_id, "models")
                        print("Artifacts:")
                        for f in artifacts:
                            print(" -", f.path)
                        return
                    latest_idx = max(uni.keys())
                    uni_path = client.download_artifacts(
                        args.run_id, uni[latest_idx], temp_dir
                    )
                    print(f"Loading unified policy for alternation {latest_idx}...")
                    results = run_eval_for_unified(
                        uni_path,
                        args.episodes,
                        required,
                        optional,
                        args,
                        client,
                        args.run_id,
                        temp_dir,
                    )
                    analyze_results(results, args.episodes)
                elif pairs:
                    latest_idx = max(pairs.keys())
                    latest_pair = pairs[latest_idx]
                    offense_policy_path = client.download_artifacts(
                        args.run_id, latest_pair["offense"], temp_dir
                    )
                    defense_policy_path = client.download_artifacts(
                        args.run_id, latest_pair["defense"], temp_dir
                    )
                    print(f"Loading policies for alternation {latest_idx}...")
                    results = run_eval_for_pair(
                        offense_policy_path,
                        defense_policy_path,
                        args.episodes,
                        required,
                        optional,
                        args,
                        client,
                        args.run_id,
                        temp_dir,
                        role_flag_offense,
                        role_flag_defense,
                    )
                    analyze_results(results, args.episodes)
                else:
                    print("No paired or unified artifacts found under 'models/'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained BasketWorld agents from an MLflow run.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--run-id", type=str, required=True, help="The MLflow Run ID to evaluate."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes to run per evaluation (or per alternation if --all-alternations).",
    )
    parser.add_argument(
        "--no-render", action="store_true", help="Disable rendering a sample GIF."
    )
    parser.add_argument(
        "--log-gifs",
        action="store_true",
        help="Also log per-episode GIFs (can be large).",
    )
    parser.add_argument(
        "--use-unified",
        action="store_true",
        help="Evaluate unified policies instead of paired offense/defense.",
    )
    parser.add_argument(
        "--deterministic-offense",
        type=lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        default=False,
        help="Use deterministic offense actions.",
    )
    parser.add_argument(
        "--deterministic-defense",
        type=lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        default=False,
        help="Use deterministic defense actions.",
    )
    parser.add_argument(
        "--deterministic-unified",
        type=lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        default=False,
        help="Use deterministic actions for unified policy.",
    )
    # Illegal-action resolution strategies (noop|sample|best)
    parser.add_argument(
        "--offense-illegal-strategy",
        type=str,
        choices=["noop", "sample", "best"],
        default=None,
        help="Strategy to resolve illegal actions for offense (default: sample).",
    )
    parser.add_argument(
        "--defense-illegal-strategy",
        type=str,
        choices=["noop", "sample", "best"],
        default=None,
        help="Strategy to resolve illegal actions for defense (default: best).",
    )
    parser.add_argument(
        "--unified-illegal-strategy",
        type=str,
        choices=["noop", "sample", "best"],
        default=None,
        help="Strategy to resolve illegal actions for unified policy (default: sample).",
    )
    parser.add_argument(
        "--mask-occupied-moves",
        type=lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        default=None,
        help="If set, disallow moves into currently occupied neighboring hexes.",
    )
    parser.add_argument(
        "--all-alternations",
        action="store_true",
        help="Evaluate and aggregate metrics across all alternations and log a CSV artifact.",
    )
    # Optional overrides for dunk and spawn evaluation
    parser.add_argument(
        "--allow-dunks",
        type=lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        default=None,
        help="Override run param to enable/disable dunks during evaluation.",
    )
    parser.add_argument(
        "--dunk-pct",
        type=float,
        default=None,
        help="Override run param for dunk make probability during evaluation.",
    )
    parser.add_argument(
        "--spawn-distance",
        type=int,
        default=None,
        help="Override run param for spawn distance relative to basket (can be negative).",
    )

    args = parser.parse_args()
    main(args)
