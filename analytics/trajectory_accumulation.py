#!/usr/bin/env python3
"""
Create cumulative player trajectory animations over N self-play episodes for a given MLflow run.

- Loads policies (latest by default) from the run's model artifacts (supports alternation selection)
- Reconstructs the environment with the run's parameters
- Accumulates per-cell occupancy counts for offense and defense over time
- Stores cumulative snapshots each episode to render animations where colors are scaled by the final max
- Saves GIFs and logs them to MLflow under trajectories/
"""
import argparse
import os
import re
import tempfile
from typing import Optional, List, Tuple

import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib import colormaps as mpl_cmaps
import math
from stable_baselines3 import PPO
from PIL import Image

import basketworld
from basketworld.utils.mlflow_params import get_mlflow_params
from tqdm import tqdm


def _select_policy_artifact(artifacts, role: str, alternation: Optional[int]) -> str:
    """Match evaluate.py/heatmap.py logic: filter by role substring and select by r'_(\d+)\.zip'."""
    role_artifacts = [f.path for f in artifacts if role in f.path]
    if not role_artifacts:
        raise RuntimeError(f"No artifacts found for role '{role}' under models/.")
    if alternation is not None:
        for p in sorted(role_artifacts):
            m = re.search(r'_(\d+)\.zip', p)
            if m and int(m.group(1)) == alternation:
                return p
        raise RuntimeError(f"No {role} policy found for alternation {alternation}. Candidates: {role_artifacts}")
    return max(role_artifacts, key=lambda p: int(re.search(r'_(\d+)\.zip', p).group(1)))


def _parse_alts(list_arg, single_arg):
    if list_arg is not None and str(list_arg).strip() != "":
        parts = re.split(r"[\s,]+", str(list_arg).strip())
        return [int(p) for p in parts if p != ""]
    if single_arg is not None:
        return [int(single_arg)]
    return [None]


def _axial_to_cartesian(q: int, r: int, size: float = 1.0) -> Tuple[float, float]:
    x = size * (math.sqrt(3) * q + (math.sqrt(3) / 2.0) * r)
    y = size * (1.5 * r)
    return x, y


def _render_hex_grid(env, arr: np.ndarray, title: str, cmap_name: str, vmax: int, show_counts: bool = False):
    fig, ax = plt.subplots(figsize=(max(8, env.court_width/1.5), max(6, env.court_height/1.5)))
    ax.set_aspect('equal')

    hex_radius = 1.0
    vmax_safe = int(vmax) if vmax and vmax > 0 else 1
    norm = Normalize(vmin=0, vmax=vmax_safe)
    cmap = mpl_cmaps.get_cmap(cmap_name)

    for r in range(env.court_height):
        for c in range(env.court_width):
            q, r_ax = env._offset_to_axial(c, r)
            x, y = _axial_to_cartesian(q, r_ax)
            val = int(arr[r, c])
            color = cmap(norm(val)) if val > 0 else (0.92, 0.92, 0.92, 0.4)
            hexagon = RegularPolygon(
                (x, y), numVertices=6, radius=hex_radius, orientation=0,
                facecolor=color, edgecolor='white', linewidth=1.0, alpha=1.0
            )
            ax.add_patch(hexagon)

            if show_counts:
                label_color = 'white' if val > 0 else '#666666'
                ax.text(x, y, str(val), color=label_color, ha='center', va='center', fontsize=7.0, zorder=7)

            # 3PT outline
            cell_distance = env._hex_distance((q, r_ax), env.basket_position)
            if cell_distance == env.three_point_distance:
                tp_outline = RegularPolygon((x, y), numVertices=6, radius=hex_radius,
                                            orientation=0, facecolor='none', edgecolor='red', linewidth=3, zorder=5)
                ax.add_patch(tp_outline)

            # Basket ring
            if (q, r_ax) == env.basket_position:
                basket_ring = plt.Circle((x, y), hex_radius * 1.05, fill=False, edgecolor='red', linewidth=3, zorder=6)
                ax.add_patch(basket_ring)

    coords = [_axial_to_cartesian(*env._offset_to_axial(c, r)) for r in range(env.court_height) for c in range(env.court_width)]
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    margin = 2.0
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('counts')

    plt.tight_layout()
    return fig


def main(args):
    tracking_uri = "http://localhost:5000"
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    required, optional = get_mlflow_params(client, args.run_id)

    artifacts = client.list_artifacts(args.run_id, "models")

    offense_alts = _parse_alts(getattr(args, "offense_alts", None), args.offense_alt)
    defense_alts = _parse_alts(getattr(args, "defense_alts", None), args.defense_alt)
    if len(offense_alts) != len(defense_alts):
        raise ValueError(f"--offense-alts and --defense-alts must have same length (got {len(offense_alts)} vs {len(defense_alts)})")

    with tempfile.TemporaryDirectory() as temp_dir:
        def _alt_of(path):
            m = re.search(r'_(\d+)\.zip', os.path.basename(path))
            return int(m.group(1)) if m else None

        env = basketworld.HexagonBasketballEnv(
            **required,
            **optional,
        )

        os.makedirs(args.output_dir, exist_ok=True)
        all_gifs = []

        for off_alt_req, def_alt_req in zip(offense_alts, defense_alts):
            offense_art_path = _select_policy_artifact(artifacts, "offense", off_alt_req)
            defense_art_path = _select_policy_artifact(artifacts, "defense", def_alt_req)

            offense_policy_path = client.download_artifacts(args.run_id, offense_art_path, temp_dir)
            defense_policy_path = client.download_artifacts(args.run_id, defense_art_path, temp_dir)

            print(f"  - Offense policy: {os.path.basename(offense_policy_path)} (alt={_alt_of(offense_policy_path)})")
            print(f"  - Defense policy: {os.path.basename(defense_policy_path)} (alt={_alt_of(defense_policy_path)})")

            offense_policy = PPO.load(offense_policy_path)
            defense_policy = PPO.load(defense_policy_path)

            offense_counts = np.zeros((env.court_height, env.court_width), dtype=np.int64)
            defense_counts = np.zeros((env.court_height, env.court_width), dtype=np.int64)

            # Per-episode cumulative snapshots
            offense_snapshots: List[np.ndarray] = []
            defense_snapshots: List[np.ndarray] = []

            def accumulate_positions():
                for pid in env.offense_ids:
                    q, r_ax = env.positions[pid]
                    col, row = env._axial_to_offset(q, r_ax)
                    if 0 <= row < env.court_height and 0 <= col < env.court_width:
                        offense_counts[row, col] += 1
                for pid in env.defense_ids:
                    q, r_ax = env.positions[pid]
                    col, row = env._axial_to_offset(q, r_ax)
                    if 0 <= row < env.court_height and 0 <= col < env.court_width:
                        defense_counts[row, col] += 1

            # Roll out episodes and capture cumulative at end of each episode
            for _ in tqdm(range(args.episodes), total=args.episodes, desc="Trajectory episodes"):
                obs, _ = env.reset()
                done = False
                accumulate_positions()
                while not done:
                    offense_action, _ = offense_policy.predict(obs, deterministic=args.deterministic_offense)
                    defense_action, _ = defense_policy.predict(obs, deterministic=args.deterministic_defense)
                    full_action = np.zeros(env.n_players, dtype=int)
                    for player_id in range(env.n_players):
                        if player_id in env.offense_ids:
                            full_action[player_id] = offense_action[player_id]
                        else:
                            full_action[player_id] = defense_action[player_id]
                    obs, _, done, _, _ = env.step(full_action)
                    accumulate_positions()
                # Store deep copies for snapshots
                offense_snapshots.append(offense_counts.copy())
                defense_snapshots.append(defense_counts.copy())

            off_alt = _alt_of(offense_policy_path)
            def_alt = _alt_of(defense_policy_path)
            off_alt_label = str(off_alt) if off_alt is not None else "latest"
            def_alt_label = str(def_alt) if def_alt is not None else "latest"

            # Fixed color scale based on final frame max
            off_vmax = int(offense_snapshots[-1].max()) if offense_snapshots else 1
            def_vmax = int(defense_snapshots[-1].max()) if defense_snapshots else 1
            off_vmax = max(off_vmax, 1)
            def_vmax = max(def_vmax, 1)

            # Render per-snapshot images then combine into GIFs
            offense_pngs = []
            defense_pngs = []

            for idx, arr in enumerate(offense_snapshots):
                fig = _render_hex_grid(env, arr, f"Offense Trajectory (frame {idx+1}/{len(offense_snapshots)})", args.cmap_name_offense, vmax=off_vmax, show_counts=args.show_counts)
                out_path = os.path.join(args.output_dir, f"trajectory_offense_{off_alt_label}_{def_alt_label}_{idx:04d}.png")
                plt.savefig(out_path, dpi=220)
                plt.close(fig)
                offense_pngs.append(out_path)

            for idx, arr in enumerate(defense_snapshots):
                fig = _render_hex_grid(env, arr, f"Defense Trajectory (frame {idx+1}/{len(defense_snapshots)})", args.cmap_name_defense, vmax=def_vmax, show_counts=args.show_counts)
                out_path = os.path.join(args.output_dir, f"trajectory_defense_{off_alt_label}_{def_alt_label}_{idx:04d}.png")
                plt.savefig(out_path, dpi=220)
                plt.close(fig)
                defense_pngs.append(out_path)

            # GIF compile
            if offense_pngs:
                offense_gif = os.path.join(args.output_dir, f"trajectory_offense_off_{off_alt_label}_def_{def_alt_label}.gif")
                frames = [Image.open(p) for p in offense_pngs]
                frames[0].save(
                    offense_gif,
                    save_all=True,
                    append_images=frames[1:],
                    duration=int(args.gif_duration * 1000),
                    loop=0,
                    optimize=False,
                )
                all_gifs.append(offense_gif)

            if defense_pngs:
                defense_gif = os.path.join(args.output_dir, f"trajectory_defense_off_{off_alt_label}_def_{def_alt_label}.gif")
                frames = [Image.open(p) for p in defense_pngs]
                frames[0].save(
                    defense_gif,
                    save_all=True,
                    append_images=frames[1:],
                    duration=int(args.gif_duration * 1000),
                    loop=0,
                    optimize=False,
                )
                all_gifs.append(defense_gif)

        # Log to MLflow
        if not args.no_log_mlflow and all_gifs:
            with mlflow.start_run(run_id=args.run_id):
                for p in all_gifs:
                    mlflow.log_artifact(p, artifact_path="trajectories")
            print("Logged trajectory animations to MLflow under 'trajectories/'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate cumulative trajectory animations from a BasketWorld MLflow run.")
    parser.add_argument("--run-id", type=str, required=True, help="MLflow Run ID")
    parser.add_argument("--episodes", type=int, default=200, help="Number of self-play episodes")
    parser.add_argument("--deterministic-offense", type=lambda v: str(v).lower() in ["1","true","yes","y","t"], default=False)
    parser.add_argument("--deterministic-defense", type=lambda v: str(v).lower() in ["1","true","yes","y","t"], default=False)
    parser.add_argument("--offense-alt", type=int, default=None, help="Use specific alternation for offense policy")
    parser.add_argument("--defense-alt", type=int, default=None, help="Use specific alternation for defense policy")
    parser.add_argument("--offense-alts", type=str, default=None, help="Comma/space separated list of offense alternations")
    parser.add_argument("--defense-alts", type=str, default=None, help="Comma/space separated list of defense alternations")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save frames and GIFs")
    parser.add_argument("--no-log-mlflow", action="store_true", help="Do not log artifacts to MLflow")
    # Optional overrides
    parser.add_argument("--allow-dunks", type=lambda v: str(v).lower() in ["1","true","yes","y","t"], default=None)
    parser.add_argument("--dunk-pct", type=float, default=None)
    parser.add_argument("--cmap-name-offense", type=str, default="winter")
    parser.add_argument("--cmap-name-defense", type=str, default="summer")
    parser.add_argument("--gif-duration", type=float, default=0.5, help="Seconds per frame in GIF")
    parser.add_argument("--show-counts", action="store_true", help="Overlay integer counts on cells")

    args = parser.parse_args()
    main(args)


