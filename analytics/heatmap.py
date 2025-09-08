#!/usr/bin/env python3
"""
Generate offense and defense heatmaps over N self-play episodes for a given MLflow run (unified-only).

- Loads unified policies (latest by default) from the run's model artifacts
- Reconstructs the environment with the run's parameters
- Accumulates positions for offense and defense separately across steps
- Saves heatmaps as PNGs and logs them to MLflow under heatmaps/
"""
import argparse
import os
import re
import tempfile
from typing import Optional

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


# moved to utils: basketworld.utils.mlflow_params


def _select_unified_artifact(artifacts, alternation: Optional[int]) -> str:
    unified_artifacts = [f.path for f in artifacts if "unified" in f.path]
    if not unified_artifacts:
        raise RuntimeError("No artifacts found for role 'unified' under models/.")
    if alternation is not None:
        for p in sorted(unified_artifacts):
            m = re.search(r'_(\d+)\.zip', p)
            if m and int(m.group(1)) == alternation:
                return p
        raise RuntimeError(f"No unified policy found for alternation {alternation}. Candidates: {unified_artifacts}")
    return max(unified_artifacts, key=lambda p: int(re.search(r'_(\d+)\.zip', p).group(1)))


def main(args):
    tracking_uri = "http://localhost:5000"
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    required, optional = get_mlflow_params(client, args.run_id)


    # Download models: list immediate files under models/
    artifacts = client.list_artifacts(args.run_id, "models")

    # Support lists of alternations; must be same length
    def _parse_alts(list_arg, single_arg):
        if list_arg is not None and str(list_arg).strip() != "":
            parts = re.split(r"[\s,]+", str(list_arg).strip())
            return [int(p) for p in parts if p != ""]
        if single_arg is not None:
            return [int(single_arg)]
        return [None]

    unified_alts = _parse_alts(getattr(args, "unified_alts", None), args.unified_alt)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Helper to read alternation from filename
        def _alt_of(path):
            m = re.search(r'_(\d+)\.zip', os.path.basename(path))
            return int(m.group(1)) if m else None

        # Setup environment (no render for speed)
        env = basketworld.HexagonBasketballEnv(
            **required,
            **optional,
        )

        # Plot helper (defined once)
        def _plot_hex_heat(arr: np.ndarray, title: str, path: str, cmap_name: str):
            fig, ax = plt.subplots(figsize=(max(8, env.court_width/1.5), max(6, env.court_height/1.5)))
            ax.set_aspect('equal')

            # Axial -> cartesian for pointy-topped hexes (match env)
            def axial_to_cartesian(q: int, r: int, size: float = 1.0):
                x = size * (math.sqrt(3) * q + (math.sqrt(3) / 2.0) * r)
                y = size * (1.5 * r)
                return x, y

            hex_radius = 1.0
            vmax = int(arr.max()) if arr.size > 0 else 0
            vmax = vmax if vmax > 0 else 1
            norm = Normalize(vmin=0, vmax=vmax)
            cmap = mpl_cmaps.get_cmap(cmap_name)

            # Draw grid of hexes colored by counts
            for r in range(env.court_height):
                for c in range(env.court_width):
                    q, r_ax = env._offset_to_axial(c, r)
                    x, y = axial_to_cartesian(q, r_ax)
                    val = int(arr[r, c])
                    color = cmap(norm(val)) if val > 0 else (0.92, 0.92, 0.92, 0.4)
                    hexagon = RegularPolygon(
                        (x, y), numVertices=6, radius=hex_radius, orientation=0,
                        facecolor=color, edgecolor='white', linewidth=1.0, alpha=1.0
                    )
                    ax.add_patch(hexagon)

                    # Count label at center
                    label_color = 'white' if val > 0 else '#666666'
                    ax.text(x, y, str(val), color=label_color, ha='center', va='center', fontsize=7.0, zorder=7)

                    # Draw 3pt outline
                    cell_distance = env._hex_distance((q, r_ax), env.basket_position)
                    if cell_distance == env.three_point_distance:
                        tp_outline = RegularPolygon((x, y), numVertices=6, radius=hex_radius,
                                                    orientation=0, facecolor='none', edgecolor='red', linewidth=3, zorder=5)
                        ax.add_patch(tp_outline)

                    # Basket ring
                    if (q, r_ax) == env.basket_position:
                        basket_ring = plt.Circle((x, y), hex_radius * 1.05, fill=False, edgecolor='red', linewidth=3, zorder=6)
                        ax.add_patch(basket_ring)

            # Axis limits
            coords = [axial_to_cartesian(*env._offset_to_axial(c, r)) for r in range(env.court_height) for c in range(env.court_width)]
            xs = [p[0] for p in coords]
            ys = [p[1] for p in coords]
            margin = 2.0
            ax.set_xlim(min(xs) - margin, max(xs) + margin)
            ax.set_ylim(min(ys) - margin, max(ys) + margin)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title)

            # Colorbar
            sm = ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('counts')

            plt.tight_layout()
            plt.savefig(path, dpi=220)
            plt.close(fig)

        all_saved_paths = []
        offense_paths = []
        defense_paths = []
        # Unified-only: iterate through alternations
        for uni_alt_req in unified_alts:
            unified_art_path = _select_unified_artifact(artifacts, uni_alt_req)
            unified_policy_path = client.download_artifacts(args.run_id, unified_art_path, temp_dir)
            print(f"  - Unified policy: {os.path.basename(unified_policy_path)} (alt={_alt_of(unified_policy_path)})")

            policy = PPO.load(unified_policy_path)

            # Heatmap accumulators (rows=height, cols=width)
            offense_heat = np.zeros((env.court_height, env.court_width), dtype=np.int64)
            defense_heat = np.zeros((env.court_height, env.court_width), dtype=np.int64)

            def accumulate_positions():
                # Count current step's positions into heatmaps
                for pid in env.offense_ids:
                    q, r = env.positions[pid]
                    col, row = env._axial_to_offset(q, r)
                    if 0 <= row < env.court_height and 0 <= col < env.court_width:
                        offense_heat[row, col] += 1
                for pid in env.defense_ids:
                    q, r = env.positions[pid]
                    col, row = env._axial_to_offset(q, r)
                    if 0 <= row < env.court_height and 0 <= col < env.court_width:
                        defense_heat[row, col] += 1

            # Run episodes
            for _ in tqdm(range(args.episodes), total=args.episodes, desc="Heatmap episodes"):
                obs, _ = env.reset()
                done = False
                accumulate_positions()
                while not done:
                    full_action, _ = policy.predict(obs, deterministic=args.deterministic)
                    obs, _, done, _, _ = env.step(full_action)
                    accumulate_positions()
            # Plot with unified label
            os.makedirs(args.output_dir, exist_ok=True)
            uni_alt = _alt_of(unified_policy_path)
            uni_alt_label = str(uni_alt) if uni_alt is not None else "latest"
            offense_path = os.path.join(args.output_dir, f"heatmap_offense_unified_{uni_alt_label}.png")
            defense_path = os.path.join(args.output_dir, f"heatmap_defense_unified_{uni_alt_label}.png")
            cmap_uni = args.cmap_name
            _plot_hex_heat(offense_heat, f"Offense Heatmap: Unified {uni_alt_label}", offense_path, cmap_name=cmap_uni)
            _plot_hex_heat(defense_heat, f"Defense Heatmap: Unified {uni_alt_label}", defense_path, cmap_name=cmap_uni)
            print(f"Saved: {offense_path}\nSaved: {defense_path}")
            all_saved_paths.extend([offense_path, defense_path])
            offense_paths.append(offense_path)
            defense_paths.append(defense_path)

            # Unified-only path does not produce paired labels; already saved above

        # Log to MLflow (all generated)
        if not args.no_log_mlflow and all_saved_paths:
            with mlflow.start_run(run_id=args.run_id):
                for p in all_saved_paths:
                    mlflow.log_artifact(p, artifact_path="heatmaps")
            print("Logged heatmaps to MLflow under 'heatmaps/'.")

        # Optional: create animated GIFs across alternations
        if getattr(args, "make_gif", False):
            try:
                if len(offense_paths) > 0:
                    offense_gif = os.path.join(args.output_dir, "heatmap_offense_anim.gif")
                    offense_frames = [Image.open(p) for p in offense_paths]
                    offense_frames[0].save(
                        offense_gif,
                        save_all=True,
                        append_images=offense_frames[1:],
                        duration=int(args.gif_duration * 1000),
                        loop=0,
                        optimize=False,
                    )
                if len(defense_paths) > 0:
                    defense_gif = os.path.join(args.output_dir, "heatmap_defense_anim.gif")
                    defense_frames = [Image.open(p) for p in defense_paths]
                    defense_frames[0].save(
                        defense_gif,
                        save_all=True,
                        append_images=defense_frames[1:],
                        duration=int(args.gif_duration * 1000),
                        loop=0,
                        optimize=False,
                    )
                print("Saved animated GIFs for heatmaps.")
                if not args.no_log_mlflow:
                    with mlflow.start_run(run_id=args.run_id):
                        if len(offense_paths) > 0:
                            mlflow.log_artifact(offense_gif, artifact_path="heatmaps")
                        if len(defense_paths) > 0:
                            mlflow.log_artifact(defense_gif, artifact_path="heatmaps")
            except Exception as e:
                print(f"[warn] Failed to create heatmap GIFs: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate heatmaps from a BasketWorld MLflow run (unified-only).")
    parser.add_argument("--run-id", type=str, required=True, help="MLflow Run ID")
    parser.add_argument("--episodes", type=int, default=200, help="Number of self-play episodes")
    parser.add_argument("--deterministic", type=lambda v: str(v).lower() in ["1","true","yes","y","t"], default=False)
    parser.add_argument("--unified-alt", type=int, default=None, help="Use specific alternation for unified policy")
    parser.add_argument("--unified-alts", type=str, default=None, help="Comma/space separated list of unified alternations")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save heatmaps")
    parser.add_argument("--no-log-mlflow", action="store_true", help="Do not log artifacts to MLflow")
    # Optional overrides
    parser.add_argument("--allow-dunks", type=lambda v: str(v).lower() in ["1","true","yes","y","t"], default=None)
    parser.add_argument("--dunk-pct", type=float, default=None)
    parser.add_argument("--cmap-name", type=str, default="winter")
    parser.add_argument("--make-gif", action="store_true", help="Also compile generated images into an animated GIF")
    parser.add_argument("--gif-duration", type=float, default=0.8, help="Seconds per frame in GIF")
    # Swallow any stray args to be robust to older invocations
    parser.add_argument("extras", nargs="*", help=argparse.SUPPRESS)

    args = parser.parse_args()
    main(args)


