#!/usr/bin/env python3
"""
ELO Rating Evolution Analysis Script

This script measures how ELO ratings evolve during the course of training by:
1. Loading all model checkpoints from an MLflow run
2. Running deterministic tournament matches between models
3. Computing ELO ratings based on win/loss/tie outcomes
4. Plotting ELO progression to identify overtraining or skill plateaus

The script uses a sliding window approach where each model plays against
the previous N models, making it efficient for runs with many alternations.
"""
import argparse
import os
import numpy as np
import tempfile
import re
from typing import Dict, List, Tuple
from stable_baselines3 import PPO
from basketworld.envs.basketworld_env_v2 import HexagonBasketballEnv
import mlflow
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

from basketworld.utils.mlflow_params import get_mlflow_params
from basketworld.utils.action_resolution import (
    IllegalActionStrategy,
    get_policy_action_probabilities,
    resolve_illegal_actions,
)


class EloRatingSystem:
    """Manages ELO ratings for models."""

    def __init__(self, k_factor: float = 32, initial_rating: float = 1000):
        """
        Initialize ELO rating system.

        Args:
            k_factor: Maximum rating change per game (32 = high volatility, 16 = stable)
            initial_rating: Starting ELO rating for all models
        """
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.ratings: Dict[int, float] = {}
        self.rating_history: Dict[int, List[float]] = defaultdict(list)

    def get_rating(self, model_id: int) -> float:
        """Get current ELO rating for a model."""
        if model_id not in self.ratings:
            self.ratings[model_id] = self.initial_rating
            self.rating_history[model_id].append(self.initial_rating)
        return self.ratings[model_id]

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        Calculate expected score for player A against player B.
        Returns value between 0 and 1.
        """
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def update_ratings(self, model_a: int, model_b: int, score_a: float):
        """
        Update ELO ratings after a match.

        Args:
            model_a: ID of first model
            model_b: ID of second model
            score_a: Score for model A (1.0 = win, 0.5 = tie, 0.0 = loss)
        """
        rating_a = self.get_rating(model_a)
        rating_b = self.get_rating(model_b)

        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = self.expected_score(rating_b, rating_a)

        # Update ratings
        new_rating_a = rating_a + self.k_factor * (score_a - expected_a)
        new_rating_b = rating_b + self.k_factor * ((1.0 - score_a) - expected_b)

        self.ratings[model_a] = new_rating_a
        self.ratings[model_b] = new_rating_b

        # Record history
        self.rating_history[model_a].append(new_rating_a)
        self.rating_history[model_b].append(new_rating_b)


def list_unified_by_alternation(client, run_id: str) -> Dict[int, str]:
    """
    List all unified model artifacts by alternation index.

    Returns:
        Dictionary mapping alternation index to artifact path
    """
    artifacts = client.list_artifacts(run_id, "models")
    unified = [
        f.path
        for f in artifacts
        if f.path.endswith(".zip") and "unified_policy" in f.path
    ]

    def idx_of(p):
        m = re.search(r"alt_(\d+)\.zip$", p)
        if not m:
            m = re.search(r"_(\d+)\.zip$", p)
        return int(m.group(1)) if m else None

    uni_map = {idx_of(p): p for p in unified if idx_of(p) is not None}
    return {i: uni_map[i] for i in sorted(uni_map.keys())}


def play_episode(
    env: HexagonBasketballEnv,
    policy_a: PPO,
    policy_b: PPO,
    illegal_strategy: IllegalActionStrategy,
) -> Tuple[str, Dict]:
    """
    Play one episode between two policies.

    Returns:
        Tuple of (outcome, stats_dict)
        outcome is one of: 'score', 'turnover', 'shot_clock'
    """
    obs, info = env.reset()
    done = False
    offense_ids = env.offense_ids

    # Determine which policy controls which team
    # Since teams are randomly assigned, we need to track this
    # For simplicity, we'll say policy_a controls offense, policy_b controls defense
    # (this assignment is arbitrary and randomized by env resets)

    cumulative_rewards = np.zeros(env.n_players, dtype=float)

    while not done:
        # Build role-conditional observations
        offense_obs = dict(obs)
        defense_obs = dict(obs)
        try:
            offense_obs["role_flag"] = np.array([1.0], dtype=np.float32)
            defense_obs["role_flag"] = np.array([0.0], dtype=np.float32)
        except Exception:
            pass

        # Get actions (deterministic for consistency)
        offense_action, _ = policy_a.predict(
            offense_obs, deterministic=args.deterministic_opponent
        )
        defense_action, _ = policy_b.predict(
            defense_obs, deterministic=args.deterministic_opponent
        )

        # Resolve illegal actions
        action_mask = obs.get("action_mask")
        offense_probs = get_policy_action_probabilities(policy_a, offense_obs)
        defense_probs = get_policy_action_probabilities(policy_b, defense_obs)

        offense_resolved = resolve_illegal_actions(
            np.array(offense_action),
            action_mask,
            illegal_strategy,
            deterministic=args.deterministic_opponent,
            probs_per_player=offense_probs,
        )
        defense_resolved = resolve_illegal_actions(
            np.array(defense_action),
            action_mask,
            illegal_strategy,
            deterministic=args.deterministic_opponent,
            probs_per_player=defense_probs,
        )

        # Combine actions
        full_action = np.zeros(env.n_players, dtype=int)
        for player_id in range(env.n_players):
            if player_id in offense_ids:
                full_action[player_id] = int(offense_resolved[player_id])
            else:
                full_action[player_id] = int(defense_resolved[player_id])

        obs, reward, done, _, info = env.step(full_action)
        cumulative_rewards += reward

    # Determine outcome
    final_info = info
    action_results = final_info.get("action_results", {})

    outcome_type = "unknown"
    if action_results.get("shots"):
        shot_result = list(action_results["shots"].values())[0]
        if shot_result["success"]:
            outcome_type = "score"
        else:
            outcome_type = "miss"
    elif action_results.get("turnovers"):
        outcome_type = "turnover"
    elif env.unwrapped.shot_clock <= 0:
        outcome_type = "shot_clock"

    # Calculate team rewards
    team_reward_offense = float(np.sum(cumulative_rewards[env.offense_ids]))
    team_reward_defense = float(np.sum(cumulative_rewards[env.defense_ids]))

    stats = {
        "outcome": outcome_type,
        "length": env.unwrapped.step_count,
        "reward_offense": team_reward_offense,
        "reward_defense": team_reward_defense,
    }

    return outcome_type, stats


def play_matchup(
    model_a_path: str,
    model_b_path: str,
    num_episodes: int,
    env_config: Dict,
    illegal_strategy: IllegalActionStrategy,
    verbose: bool = False,
    deterministic_opponent: bool = False,
) -> Dict:
    """
    Play a matchup between two models.

    Returns:
        Dictionary with win/loss/tie counts and statistics
    """
    # Load policies
    policy_a = PPO.load(model_a_path)
    policy_b = PPO.load(model_b_path)

    # Create environment
    env = HexagonBasketballEnv(**env_config, render_mode=None)

    results = {
        "a_wins": 0,  # Model A was on offense and scored
        "b_wins": 0,  # Model B was on offense and scored
        "ties": 0,  # No score (turnovers, missed shots, shot clock)
        "episodes": [],
    }

    for ep in range(num_episodes):
        # Alternate which model plays offense to balance the matchup
        # Episodes with even index: A=offense, B=defense
        # Episodes with odd index: B=offense, A=defense
        if ep % 2 == 0:
            outcome, stats = play_episode(env, policy_a, policy_b, illegal_strategy)
            # A was offense
            if outcome == "score":
                results["a_wins"] += 1
            elif outcome in ["turnover", "shot_clock", "miss"]:
                results["b_wins"] += 1
            stats["a_role"] = "offense"
        else:
            outcome, stats = play_episode(env, policy_b, policy_a, illegal_strategy)
            # B was offense (policy_b is first arg = offense)
            if outcome == "score":
                results["b_wins"] += 1
            elif outcome in ["turnover", "shot_clock", "miss"]:
                results["a_wins"] += 1
            stats["a_role"] = "defense"

        results["episodes"].append(stats)

    env.close()

    # Calculate win percentage for model A
    total_games = results["a_wins"] + results["b_wins"] + results["ties"]
    results["a_win_pct"] = results["a_wins"] / total_games if total_games > 0 else 0.5
    results["b_win_pct"] = results["b_wins"] / total_games if total_games > 0 else 0.5

    if verbose:
        print(
            f"    A wins: {results['a_wins']}, B wins: {results['b_wins']}, "
            f"Ties: {results['ties']} (A win%: {results['a_win_pct']:.1%})"
        )

    return results


def run_tournament(
    models: Dict[int, str],
    client,
    run_id: str,
    temp_dir: str,
    env_config: Dict,
    args,
) -> Tuple[EloRatingSystem, List[Dict]]:
    """
    Run tournament between models and compute ELO ratings.

    Args:
        models: Dict mapping alternation index to artifact path
        client: MLflow client
        run_id: MLflow run ID
        temp_dir: Temporary directory for downloads
        env_config: Environment configuration
        args: Command line arguments

    Returns:
        EloRatingSystem with updated ratings
    """
    elo = EloRatingSystem(k_factor=args.k_factor, initial_rating=args.initial_rating)

    model_ids = sorted(models.keys())
    print(f"Found {len(model_ids)} models: {model_ids}")

    # Download all models first (optional optimization)
    print("Downloading models...")
    model_paths = {}
    for alt_idx in tqdm(model_ids, desc="Downloading"):
        model_paths[alt_idx] = client.download_artifacts(
            run_id, models[alt_idx], temp_dir
        )

    # Determine matchups based on strategy
    matchups = []

    if args.tournament_mode == "sequential":
        # Each model plays against previous N models (sliding window)
        # Use list indices to ensure we only match with models that exist
        for i, alt_idx in enumerate(model_ids):
            # Look back at previous N models in the list
            for j in range(max(0, i - args.window_size), i):
                matchups.append((alt_idx, model_ids[j]))

    elif args.tournament_mode == "full":
        # Full round-robin (every model plays every other model)
        for i, alt_a in enumerate(model_ids):
            for alt_b in model_ids[i + 1 :]:
                matchups.append((alt_a, alt_b))

    elif args.tournament_mode == "sparse":
        # Each model plays against a sparse set of checkpoints
        checkpoint_interval = max(1, len(model_ids) // args.num_checkpoints)
        checkpoints = model_ids[::checkpoint_interval]
        print(f"Using checkpoint models: {checkpoints}")

        for alt_idx in model_ids:
            for checkpoint in checkpoints:
                if checkpoint != alt_idx:
                    matchups.append((alt_idx, checkpoint))

    print(f"\nRunning {len(matchups)} matchups with {args.episodes} episodes each...")
    print(f"Estimated total episodes: {len(matchups) * args.episodes}")

    # Determine illegal action strategy
    if args.illegal_strategy == "best":
        illegal_strategy = IllegalActionStrategy.BEST_PROB
    elif args.illegal_strategy == "sample":
        illegal_strategy = IllegalActionStrategy.SAMPLE_PROB
    else:
        illegal_strategy = IllegalActionStrategy.NOOP

    # Run all matchups
    matchup_results = []

    for model_a_idx, model_b_idx in tqdm(matchups, desc="Matchups"):
        print(f"\nMatchup: Model {model_a_idx} vs Model {model_b_idx}")

        results = play_matchup(
            model_paths[model_a_idx],
            model_paths[model_b_idx],
            args.episodes,
            env_config,
            illegal_strategy,
            verbose=args.verbose,
            deterministic_opponent=args.deterministic_opponent,
        )

        # Update ELO ratings
        # Convert win percentages to scores
        # A wins = 1.0, B wins = 0.0, each game contributes
        # We'll update ratings incrementally for each episode to simulate continuous evolution
        if args.incremental_elo:
            for ep_stats in results["episodes"]:
                # Determine winner of this episode
                if ep_stats["outcome"] == "score":
                    if ep_stats["a_role"] == "offense":
                        score_a = 1.0  # A won
                    else:
                        score_a = 0.0  # B won
                else:
                    # Defensive win
                    if ep_stats["a_role"] == "defense":
                        score_a = 1.0  # A won (was on defense)
                    else:
                        score_a = 0.0  # B won (was on defense)

                elo.update_ratings(model_a_idx, model_b_idx, score_a)
        else:
            # Update based on overall win percentage
            score_a = results["a_win_pct"]
            elo.update_ratings(model_a_idx, model_b_idx, score_a)

        matchup_results.append(
            {
                "model_a": model_a_idx,
                "model_b": model_b_idx,
                "a_wins": results["a_wins"],
                "b_wins": results["b_wins"],
                "ties": results["ties"],
                "a_win_pct": results["a_win_pct"],
                "elo_a": elo.get_rating(model_a_idx),
                "elo_b": elo.get_rating(model_b_idx),
            }
        )

    return elo, matchup_results


def plot_winrate_heatmap(
    matchup_results: List[Dict],
    model_ids: List[int],
    output_path: str,
):
    """
    Plot a heatmap showing win rates between all model pairs.

    This helps identify non-transitive relationships (e.g., model A beats B, B beats C, but C beats A).

    Args:
        matchup_results: List of matchup result dictionaries
        model_ids: List of model alternation indices
        output_path: Path to save the plot
    """
    n = len(model_ids)
    id_to_idx = {model_id: i for i, model_id in enumerate(model_ids)}

    # Create matrix: rows = offense, cols = defense
    # Value = offensive win rate
    win_matrix = np.full((n, n), np.nan)

    for result in matchup_results:
        a_idx = id_to_idx[result["model_a"]]
        b_idx = id_to_idx[result["model_b"]]

        # model_a win rate when playing offense vs defense
        # In our matchup, we alternate: even episodes A is offense, odd episodes B is offense
        # So a_win_pct represents mixed role performance
        # For this heatmap, we want: what's the win rate when X is offense vs Y defense?

        # Note: our current matchup structure alternates roles
        # We'll use the overall win percentage as a proxy
        win_matrix[a_idx, b_idx] = result["a_win_pct"]
        win_matrix[b_idx, a_idx] = result["b_win_pct"]

    plt.figure(figsize=(14, 12))

    # Use a diverging colormap centered at 0.5 (50% win rate)
    im = plt.imshow(
        win_matrix,
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        aspect="auto",
        interpolation="nearest",
    )

    plt.colorbar(im, label="Win Rate")
    plt.xlabel("Model Index (Defense)", fontsize=12)
    plt.ylabel("Model Index (Offense)", fontsize=12)
    plt.title(
        "Offensive Win Rate Heatmap\n(Row plays Offense vs Column plays Defense)",
        fontsize=14,
        fontweight="bold",
    )

    # Set ticks to show every Nth model
    tick_spacing = max(1, len(model_ids) // 20)
    tick_positions = list(range(0, len(model_ids), tick_spacing))
    tick_labels = [model_ids[i] for i in tick_positions]

    plt.xticks(tick_positions, tick_labels, rotation=45)
    plt.yticks(tick_positions, tick_labels)

    # Add grid
    plt.grid(False)

    # Add text annotations for surprising results (large deviations)
    # Find cases where later offense loses to much earlier defense
    for i in range(n):
        for j in range(n):
            if not np.isnan(win_matrix[i, j]):
                # If a much later model (offense) does poorly against earlier model (defense)
                if model_ids[i] > model_ids[j] + 20 and win_matrix[i, j] < 0.4:
                    plt.text(
                        j,
                        i,
                        "!",
                        ha="center",
                        va="center",
                        color="white",
                        fontweight="bold",
                        fontsize=16,
                    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved win rate heatmap to {output_path}")


def plot_elo_evolution(
    elo: EloRatingSystem,
    model_ids: List[int],
    output_path: str,
    title: str = "ELO Rating Evolution During Training",
):
    """
    Plot ELO ratings vs alternation index.

    Args:
        elo: EloRatingSystem with ratings
        model_ids: List of model alternation indices
        output_path: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(12, 7))

    # Get final ratings for each model
    ratings = [elo.get_rating(idx) for idx in model_ids]

    plt.plot(model_ids, ratings, marker="o", markersize=4, linewidth=2, alpha=0.7)
    plt.xlabel("Alternation Index", fontsize=12)
    plt.ylabel("ELO Rating", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)

    # Add horizontal line at initial rating
    initial_rating = elo.initial_rating
    plt.axhline(
        y=initial_rating,
        color="r",
        linestyle="--",
        alpha=0.5,
        label=f"Initial Rating ({initial_rating})",
    )

    # Annotate peak rating
    peak_idx = model_ids[np.argmax(ratings)]
    peak_rating = max(ratings)
    plt.annotate(
        f"Peak: Alt {peak_idx}\nELO {peak_rating:.0f}",
        xy=(peak_idx, peak_rating),
        xytext=(10, 10),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.7),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
    )

    # Add trend analysis in text box
    if len(ratings) > 10:
        early_avg = np.mean(ratings[: len(ratings) // 4])
        late_avg = np.mean(ratings[-len(ratings) // 4 :])
        trend = "improving" if late_avg > early_avg else "declining"

        textstr = (
            f"Early avg: {early_avg:.0f}\nLate avg: {late_avg:.0f}\nTrend: {trend}"
        )
        plt.text(
            0.02,
            0.98,
            textstr,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved ELO evolution plot to {output_path}")


def main(args):
    """Main function."""

    # Set up MLflow
    from basketworld.utils.mlflow_config import setup_mlflow

    setup_mlflow(verbose=False)
    client = mlflow.tracking.MlflowClient()

    try:
        client.get_run(args.run_id)
    except Exception as e:
        print(f"Error: Could not find MLflow run '{args.run_id}': {e}")
        return

    # Get environment configuration from run
    print("Fetching environment configuration from MLflow run...")
    try:
        required, optional = get_mlflow_params(client, args.run_id)
        env_config = {**required, **optional}
    except KeyError as e:
        print(f"Error: Run {args.run_id} is missing required parameter: {e}")
        return

    # List all unified models
    models = list_unified_by_alternation(client, args.run_id)

    if not models:
        print("Error: No unified models found in this run.")
        print("This script requires unified_policy artifacts.")
        return

    # Run tournament and compute ELO ratings
    with tempfile.TemporaryDirectory() as temp_dir:
        elo, matchup_results = run_tournament(
            models, client, args.run_id, temp_dir, env_config, args
        )

        # Save results
        model_ids = sorted(models.keys())

        # Create results DataFrame
        elo_df = pd.DataFrame(
            {
                "alternation": model_ids,
                "elo_rating": [elo.get_rating(idx) for idx in model_ids],
            }
        )

        matchup_df = pd.DataFrame(matchup_results)

        # Save CSVs
        elo_csv_path = os.path.join(temp_dir, "elo_ratings.csv")
        matchup_csv_path = os.path.join(temp_dir, "matchup_results.csv")

        elo_df.to_csv(elo_csv_path, index=False)
        matchup_df.to_csv(matchup_csv_path, index=False)

        print("\n--- ELO Ratings Summary ---")
        print(elo_df.to_string(index=False))
        print(
            f"\nPeak ELO: {elo_df['elo_rating'].max():.1f} at alternation {elo_df.loc[elo_df['elo_rating'].idxmax(), 'alternation']}"
        )
        print(
            f"Final ELO: {elo_df['elo_rating'].iloc[-1]:.1f} at alternation {elo_df['alternation'].iloc[-1]}"
        )

        # Detect potential overtraining
        if len(model_ids) >= 20:
            last_quarter_start = len(model_ids) * 3 // 4
            peak_idx = elo_df["elo_rating"].idxmax()

            if peak_idx < last_quarter_start:
                peak_alt = elo_df.loc[peak_idx, "alternation"]
                final_rating = elo_df["elo_rating"].iloc[-1]
                peak_rating = elo_df["elo_rating"].iloc[peak_idx]
                decline_pct = (peak_rating - final_rating) / peak_rating * 100

                print("\n⚠️  Potential overtraining detected!")
                print(
                    f"   Peak was at alternation {peak_alt}, but training continued to {model_ids[-1]}"
                )
                print(f"   Rating declined by {decline_pct:.1f}% from peak")

        # Plot ELO evolution
        plot_path = os.path.join(temp_dir, "elo_evolution.png")
        plot_elo_evolution(elo, model_ids, plot_path)

        # Optionally generate win rate heatmap
        heatmap_path = None
        if args.output_heatmap:
            heatmap_path = os.path.join(temp_dir, "winrate_heatmap.png")
            plot_winrate_heatmap(matchup_results, model_ids, heatmap_path)

        # Log artifacts to MLflow
        if not args.no_log:
            with mlflow.start_run(run_id=args.run_id):
                mlflow.log_artifact(elo_csv_path, artifact_path="elo_analysis")
                mlflow.log_artifact(matchup_csv_path, artifact_path="elo_analysis")
                mlflow.log_artifact(plot_path, artifact_path="elo_analysis")
                if heatmap_path:
                    mlflow.log_artifact(heatmap_path, artifact_path="elo_analysis")
                print(f"\nLogged artifacts to MLflow run {args.run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze ELO rating evolution during training",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="MLflow run ID to analyze",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes per matchup (default: 100)",
    )

    parser.add_argument(
        "--tournament-mode",
        type=str,
        choices=["sequential", "full", "sparse"],
        default="sequential",
        help="""Tournament structure:
  - sequential: Each model plays against previous N models (sliding window)
  - full: Full round-robin (every model plays every other model)
  - sparse: Each model plays against a sparse set of checkpoint models
(default: sequential)""",
    )

    parser.add_argument(
        "--window-size",
        type=int,
        default=2,
        help="Number of previous models to play against (sequential mode only, default: 2)",
    )

    parser.add_argument(
        "--num-checkpoints",
        type=int,
        default=10,
        help="Number of checkpoint models to use (sparse mode only, default: 10)",
    )

    parser.add_argument(
        "--k-factor",
        type=float,
        default=32,
        help="ELO K-factor (32=high volatility, 16=stable, default: 32)",
    )

    parser.add_argument(
        "--initial-rating",
        type=float,
        default=1000,
        help="Initial ELO rating for all models (default: 1000)",
    )

    parser.add_argument(
        "--illegal-strategy",
        type=str,
        choices=["best", "sample", "noop"],
        default="best",
        help="Strategy for resolving illegal actions (default: best)",
    )

    parser.add_argument(
        "--incremental-elo",
        action="store_true",
        help="Update ELO after each episode instead of after full matchup",
    )

    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Don't log results back to MLflow",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed matchup results",
    )

    parser.add_argument(
        "--output-heatmap",
        action="store_true",
        help="Generate a win-rate heatmap showing all pairwise matchups",
    )

    parser.add_argument(
        "--deterministic-opponent",
        action="store_true",
        help="Use deterministic opponent actions.",
    )

    args = parser.parse_args()
    main(args)
