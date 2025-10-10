#!/usr/bin/env python3
"""
One-off analysis: correlate potential assists with relative dunk skills.

For each episode, compute:
- delta = ball_handler_pct - teammate_pct
  - If a potential assist occurred: ball_handler is the passer, teammate is the shooter
  - Else: ball_handler is the initial ball holder; teammate is the best-dunk teammate at reset
- potential_assist: 1 if episode's shot had assist_potential, else 0
- ball_handler_pct: dunk% for the chosen ball handler in this episode
- teammate_pct: dunk% for the chosen teammate in this episode

Output a CSV and log it to MLflow.
"""

import argparse
import os
import tempfile
import re
import csv
from typing import Tuple

import numpy as np
from stable_baselines3 import PPO
import mlflow

from basketworld.envs.basketworld_env_v2 import HexagonBasketballEnv
from basketworld.utils.mlflow_params import get_mlflow_params


def _list_models_by_alternation(client, run_id: str):
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
        m = re.search(r"alt_(\d+)\.zip$", p)
        if not m:
            m = re.search(r"_(\d+)\.zip$", p)
        return int(m.group(1)) if m else None

    off_map = {idx_of(p): p for p in offense if idx_of(p) is not None}
    def_map = {idx_of(p): p for p in defense if idx_of(p) is not None}
    common_idxs = sorted(set(off_map.keys()) & set(def_map.keys()))
    return {i: {"offense": off_map[i], "defense": def_map[i]} for i in common_idxs}


def _list_unified_by_alternation(client, run_id: str):
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


def _extract_std_params_from_run(client, run_id: str) -> dict:
    """Read layup_std/three_pt_std/dunk_std from MLflow run params if present."""
    params = {}
    try:
        run = client.get_run(run_id)
        data = run.data.params or {}
        if "layup_std" in data:
            params["layup_std"] = float(data["layup_std"])  # type: ignore[arg-type]
        if "three_pt_std" in data:
            params["three_pt_std"] = float(data["three_pt_std"])  # type: ignore[arg-type]
        if "dunk_std" in data:
            params["dunk_std"] = float(data["dunk_std"])  # type: ignore[arg-type]
    except Exception:
        pass
    return params


def _pair_for_non_potential(env: HexagonBasketballEnv, initial_bh) -> Tuple[int, int]:
    """Return (ball_handler_offense_id, teammate_offense_id) using initial BH and best teammate.

    initial_bh is the ball holder captured immediately after reset.
    """
    bh = (
        initial_bh
        if initial_bh is not None
        else (env.offense_ids[0] if env.offense_ids else 0)
    )
    if bh not in env.offense_ids:
        # Fallback: choose offense[0] as ball handler
        bh = int(env.offense_ids[0])
    # Best teammate by dunk% among other offense players
    best_tid = None
    best_pct = -1.0
    for pid in env.offense_ids:
        if pid == bh:
            continue
        pct = float(env.offense_dunk_pct_by_player[pid])
        if pct > best_pct:
            best_pct = pct
            best_tid = pid
    if best_tid is None:
        # Edge case: only one offense player; pair with self (delta=0)
        best_tid = bh
    return int(bh), int(best_tid)


def run_episodes_and_log_csv(args):
    # MLflow setup
    from basketworld.utils.mlflow_config import setup_mlflow

    setup_mlflow(verbose=False)
    client = mlflow.tracking.MlflowClient()

    # Fetch env params from run
    required, optional = get_mlflow_params(client, args.run_id)
    # Inject std params from run if present
    optional.update(_extract_std_params_from_run(client, args.run_id))

    # Open run context to log artifact back to same run
    with mlflow.start_run(run_id=args.run_id):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Choose policy artifact
            pairs = _list_models_by_alternation(client, args.run_id)
            uni = _list_unified_by_alternation(client, args.run_id)

            policy = None
            policy_kind = "unified"

            if args.use_unified or (not pairs and uni):
                if not uni:
                    raise RuntimeError(
                        "No unified policy artifacts found under models/ for this run."
                    )
                latest_idx = max(uni.keys())
                uni_path = client.download_artifacts(
                    args.run_id, uni[latest_idx], temp_dir
                )
                policy = PPO.load(uni_path)
                policy_kind = "unified"
            else:
                if not pairs:
                    raise RuntimeError(
                        "No paired policy artifacts found under models/ for this run."
                    )
                latest_idx = max(pairs.keys())
                pair = pairs[latest_idx]
                offense_path = client.download_artifacts(
                    args.run_id, pair["offense"], temp_dir
                )
                defense_path = client.download_artifacts(
                    args.run_id, pair["defense"], temp_dir
                )
                offense_policy = PPO.load(offense_path)
                defense_policy = PPO.load(defense_path)
                policy = (offense_policy, defense_policy)
                policy_kind = "paired"

            # Create env
            env = HexagonBasketballEnv(
                **required,
                **optional,
            )

            rows = []
            for _ in range(args.episodes):
                obs, info = env.reset()
                done = False
                # Capture initial ball holder safely (can become None later in episode)
                initial_ball_holder = (
                    env.ball_holder
                    if env.ball_holder is not None
                    else (env.offense_ids[0] if env.offense_ids else 0)
                )
                final_info = None
                while not done:
                    if policy_kind == "unified":
                        full_action, _ = policy.predict(obs, deterministic=args.deterministic_unified)  # type: ignore[union-attr]
                    else:
                        offense_policy, defense_policy = policy  # type: ignore[misc]
                        off_action, _ = offense_policy.predict(
                            obs, deterministic=args.deterministic_offense
                        )
                        def_action, _ = defense_policy.predict(
                            obs, deterministic=args.deterministic_defense
                        )
                        full_action = np.zeros(env.n_players, dtype=int)
                        for pid in range(env.n_players):
                            full_action[pid] = (
                                off_action[pid]
                                if pid in env.offense_ids
                                else def_action[pid]
                            )

                    obs, reward, done, _, step_info = env.step(full_action)
                    final_info = step_info

                # Episode ended; compute row
                action_results = (final_info or {}).get("action_results", {})
                potential_assist = 0
                bh_pct = 0.0
                tm_pct = 0.0

                if action_results.get("shots"):
                    shooter_id = int(list(action_results["shots"].keys())[0])
                    shot_result = list(action_results["shots"].values())[0]
                    if shot_result.get("assist_potential"):
                        potential_assist = 1
                        passer_id = int(
                            shot_result.get("assist_passer_id", initial_ball_holder)
                        )
                        # Resolve percentages from offense arrays if IDs are offense
                        if (
                            passer_id in env.offense_ids
                            and shooter_id in env.offense_ids
                        ):
                            bh_pct = float(env.offense_dunk_pct_by_player[passer_id])
                            tm_pct = float(env.offense_dunk_pct_by_player[shooter_id])
                        else:
                            # Fallback to initial pairing if something odd occurred
                            bh_off, tm_off = _pair_for_non_potential(
                                env, initial_ball_holder
                            )
                            bh_pct = float(env.offense_dunk_pct_by_player[bh_off])
                            tm_pct = float(env.offense_dunk_pct_by_player[tm_off])
                    else:
                        # No potential assist on shot; use non-potential pairing
                        bh_off, tm_off = _pair_for_non_potential(
                            env, initial_ball_holder
                        )
                        bh_pct = float(env.offense_dunk_pct_by_player[bh_off])
                        tm_pct = float(env.offense_dunk_pct_by_player[tm_off])
                else:
                    # Turnover/no shot; use non-potential pairing
                    bh_off, tm_off = _pair_for_non_potential(env, initial_ball_holder)
                    bh_pct = float(env.offense_dunk_pct_by_player[bh_off])
                    tm_pct = float(env.offense_dunk_pct_by_player[tm_off])

                delta = float(bh_pct - tm_pct)
                rows.append(
                    {
                        "delta": delta,
                        "potential_assist": int(potential_assist),
                        "ball_handler_pct": float(bh_pct),
                        "teammate_pct": float(tm_pct),
                    }
                )

            # --- Logistic Regression: predict potential_assist from delta ---
            try:
                from sklearn.linear_model import LogisticRegression
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import roc_auc_score

                # Prepare data
                X = np.array([[float(r["delta"])] for r in rows], dtype=float)
                y = np.array([int(r["potential_assist"]) for r in rows], dtype=int)
                if len(np.unique(y)) >= 2 and len(y) >= 5:
                    # Stratify only if both classes present
                    strat = y if (np.sum(y == 0) > 0 and np.sum(y == 1) > 0) else None
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=strat
                    )
                    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
                    clf.fit(X_train, y_train)
                    y_prob = clf.predict_proba(X_test)[:, 1]
                    auc = float(roc_auc_score(y_test, y_prob))
                    mlflow.log_metric("assist_delta_auc", auc)
                    # Log coefficients as params for quick inspection
                    try:
                        mlflow.log_param("assist_delta_coef", float(clf.coef_[0, 0]))
                        mlflow.log_param(
                            "assist_delta_intercept", float(clf.intercept_[0])
                        )
                    except Exception:
                        pass
                    print(f"AUC (holdout 20%): {auc:.4f}")
                else:
                    print("Skipping AUC: insufficient class variety or sample size.")
            except ImportError:
                print("scikit-learn not installed; skipping AUC computation.")
            except Exception as e:
                print(f"AUC computation failed: {e}")

            # Write CSV and log
            out_path = os.path.join(temp_dir, "assist_skill_delta.csv")
            with open(out_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "delta",
                        "potential_assist",
                        "ball_handler_pct",
                        "teammate_pct",
                    ],
                )
                writer.writeheader()
                writer.writerows(rows)
            mlflow.log_artifact(out_path, artifact_path="analysis")
            print(f"Logged CSV to MLflow: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze potential assists vs relative dunk skills (one-off).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--run-id", type=str, required=True, help="MLflow Run ID to evaluate."
    )
    parser.add_argument(
        "--episodes", type=int, default=200, help="Number of episodes to simulate."
    )
    parser.add_argument(
        "--use-unified",
        action="store_true",
        help="Use unified policy instead of paired.",
        default=False,
    )
    parser.add_argument(
        "--deterministic-unified",
        type=lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        default=False,
        help="Deterministic actions for unified policy.",
    )
    parser.add_argument(
        "--deterministic-offense",
        type=lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        default=False,
        help="Deterministic actions for offense policy (paired).",
    )
    parser.add_argument(
        "--deterministic-defense",
        type=lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        default=False,
        help="Deterministic actions for defense policy (paired).",
    )
    args = parser.parse_args()

    run_episodes_and_log_csv(args)


if __name__ == "__main__":
    main()
