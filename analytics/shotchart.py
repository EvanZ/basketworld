#!/usr/bin/env python3
"""
Generate an offense shot chart over N self-play episodes for a given MLflow run.

- Loads policies (latest by default) from the run's model artifacts
- Supports unified policy or separate offense/defense
- Reconstructs the environment with the run's parameters
- Accumulates per-cell shot attempts and makes (offense) across episodes
- Colors hexes by FG% and scales hex sizes by attempts (max attempts -> normal size)
- Saves PNG and logs to MLflow under shotcharts/
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
from matplotlib.ticker import PercentFormatter
from matplotlib import colormaps as mpl_cmaps
import math
from stable_baselines3 import PPO
from PIL import Image

import basketworld
from basketworld.utils.mlflow_params import get_mlflow_params
from basketworld.utils.action_resolution import (
    IllegalActionStrategy,
    get_policy_action_probabilities,
    resolve_illegal_actions,
)


def _tqdm(iterable, **kwargs):
    try:
        from tqdm import tqdm as _tqdm_impl  # type: ignore

        return _tqdm_impl(iterable, **kwargs)
    except Exception:
        return iterable


# moved to utils: basketworld.utils.mlflow_params


def _select_policy_artifact(artifacts, role: str, alternation: Optional[int]) -> str:
    """Match evaluate.py logic: filter by role substring and select by r'_(\d+)\.zip'."""
    role_artifacts = [f.path for f in artifacts if role in f.path]
    if not role_artifacts:
        raise RuntimeError(f"No artifacts found for role '{role}' under models/.")
    if alternation is not None:
        # Choose the one whose pattern r'_(n)\.zip' matches the requested n
        for p in sorted(role_artifacts):
            m = re.search(r"_(\d+)\.zip", p)
            idx = int(m.group(1)) if m is not None else None
            if idx == alternation:
                return p
        raise RuntimeError(
            f"No {role} policy found for alternation {alternation}. Candidates: {role_artifacts}"
        )

    # Latest by max alternation number, exactly like evaluate.py
    def _artifact_idx(path: str) -> int:
        m = re.search(r"_(\d+)\.zip", path)
        return int(m.group(1)) if m else -1

    return max(role_artifacts, key=_artifact_idx)


def _select_unified_artifact(artifacts, alternation: Optional[int]) -> str:
    unified_artifacts = [f.path for f in artifacts if "unified" in f.path]
    if not unified_artifacts:
        raise RuntimeError("No artifacts found for role 'unified' under models/.")
    if alternation is not None:
        for p in sorted(unified_artifacts):
            m = re.search(r"_(\d+)\.zip", p)
            idx = int(m.group(1)) if m is not None else None
            if idx == alternation:
                return p
        raise RuntimeError(
            f"No unified policy found for alternation {alternation}. Candidates: {unified_artifacts}"
        )

    def _artifact_idx(path: str) -> int:
        m = re.search(r"_(\d+)\.zip", path)
        return int(m.group(1)) if m else -1

    return max(unified_artifacts, key=_artifact_idx)


"""
supported values are 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 
'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 
'Grays', 'Grays_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges',
'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 
'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 
'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 
'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r',
'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 
'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 
'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'berlin', 'berlin_r', 'binary', 'binary_r',
'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 
'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r',
'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 
'gist_grey', 'gist_grey_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r',
'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 
'gist_yarg_r', 'gist_yerg', 'gist_yerg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 
'gnuplot_r', 'gray', 'gray_r', 'grey', 'grey_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 
'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'managua', 'managua_r', 
'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 
'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 
'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 
'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 
'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 
'vanimo', 'vanimo_r', 'viridis', 'viridis_r', 'winter', 'winter_r'
"""


def main(args):
    tracking_uri = "http://localhost:5000"
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    required, optional = get_mlflow_params(client, args.run_id)

    # Download models and support unified alternation list
    artifacts = client.list_artifacts(args.run_id, "models")

    def _parse_alts(list_arg, single_arg):
        if list_arg is not None and str(list_arg).strip() != "":
            parts = re.split(r"[\s,]+", str(list_arg).strip())
            return [int(p) for p in parts if p != ""]
        if single_arg is not None:
            return [int(single_arg)]
        return [None]

    unified_alts = _parse_alts(getattr(args, "unified_alts", None), args.unified_alt)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Helper to parse alternation from filename
        def _alt_of(path):
            m = re.search(r"_(\d+)\.zip", os.path.basename(path))
            return int(m.group(1)) if m else None

        # Setup environment (no render for speed)
        env = basketworld.HexagonBasketballEnv(
            **required,
            **optional,
        )

        # Plot helper (define before loop so we can call per alternation pair)
        def _plot_shot_chart(
            attempts_arr: np.ndarray,
            makes_arr: np.ndarray,
            title: str,
            path: str,
            cmap_name: str = "RdYlGn",
        ):
            fig, ax = plt.subplots(
                figsize=(max(8, env.court_width / 1.5), max(6, env.court_height / 1.5))
            )
            ax.set_aspect("equal")

            # Axial -> cartesian for pointy-topped hexes (match env)
            def axial_to_cartesian(q: int, r: int, size: float = 1.0):
                x = size * (math.sqrt(3) * q + (math.sqrt(3) / 2.0) * r)
                y = size * (1.5 * r)
                return x, y

            # Compute FG% per cell
            with np.errstate(divide="ignore", invalid="ignore"):
                fg = np.true_divide(makes_arr, attempts_arr)
                fg[~np.isfinite(fg)] = np.nan  # NaN where attempts==0

            # Size scaling by attempts
            max_attempts = int(np.nanmax(attempts_arr)) if attempts_arr.size > 0 else 0
            max_attempts = max_attempts if max_attempts > 0 else 1
            hex_radius_max = (
                args.hex_radius_max
            )  # "normal" size for the cell with most attempts
            hex_radius_min = (
                args.hex_radius_min
            )  # draw small hex even for zero attempts

            def radius_for_attempts(n: int) -> float:
                # Linear scaling; ensure zero attempts get minimum size
                frac = 0.0 if n <= 0 else min(1.0, float(n) / float(max_attempts))
                return hex_radius_min + frac * (hex_radius_max - hex_radius_min)

            # Color mapping for FG%
            norm = Normalize(vmin=0.0, vmax=1.0)
            cmap = mpl_cmaps.get_cmap(cmap_name)

            # Draw grid of hexes sized by attempts and colored by FG%
            for r in range(env.court_height):
                for c in range(env.court_width):
                    q, r_ax = env._offset_to_axial(c, r)
                    x, y = axial_to_cartesian(q, r_ax)
                    att = int(attempts_arr[r, c])
                    mk = int(makes_arr[r, c])
                    rad = radius_for_attempts(att)
                    if att > 0:
                        pct = float(mk) / float(att)
                        color = cmap(norm(pct))
                    else:
                        color = (0.92, 0.92, 0.92, 0.4)  # no-data
                    hexagon = RegularPolygon(
                        (x, y),
                        numVertices=6,
                        radius=rad,
                        orientation=0,
                        facecolor=color,
                        edgecolor="white",
                        linewidth=1.0,
                        alpha=1.0,
                    )
                    ax.add_patch(hexagon)

                    # FG% label in white text for cells with attempts
                    if att > 0:
                        label_text = f"{pct * 100:.0f}%"
                        # Scale label size with hex radius, ensure a minimum size
                        fs = max(4.0, args.label_fontsize * (rad / args.hex_radius_max))
                        ax.text(
                            x,
                            y,
                            label_text,
                            color="white",
                            ha="center",
                            va="center",
                            fontsize=fs,
                            zorder=7,
                        )

                    # 3PT outline
                    cell_distance = env._hex_distance((q, r_ax), env.basket_position)
                    if cell_distance == env.three_point_distance:
                        tp_outline = RegularPolygon(
                            (x, y),
                            numVertices=6,
                            radius=rad,
                            orientation=0,
                            facecolor="none",
                            edgecolor="red",
                            linewidth=2,
                            zorder=5,
                        )
                        ax.add_patch(tp_outline)

                    # Basket ring
                    if (q, r_ax) == env.basket_position:
                        basket_ring = plt.Circle(
                            (x, y),
                            rad * 1.05,
                            fill=False,
                            edgecolor="red",
                            linewidth=3,
                            zorder=6,
                        )
                        ax.add_patch(basket_ring)

            # Axis limits
            coords = [
                axial_to_cartesian(*env._offset_to_axial(c, r))
                for r in range(env.court_height)
                for c in range(env.court_width)
            ]
            xs = [p[0] for p in coords]
            ys = [p[1] for p in coords]
            margin = 2.0
            ax.set_xlim(min(xs) - margin, max(xs) + margin)
            ax.set_ylim(min(ys) - margin, max(ys) + margin)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title)

            # Colorbar for FG%
            sm = ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("FG%")
            cbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

            # No attempts size legend per user request

            plt.tight_layout()
            plt.savefig(path, dpi=220)
            plt.close(fig)

        all_saved_paths = []
        sequence_paths = []
        for uni_alt_req in unified_alts:
            unified_art_path = _select_unified_artifact(artifacts, uni_alt_req)
            unified_policy_path = client.download_artifacts(
                args.run_id, unified_art_path, temp_dir
            )
            print(
                f"  - Unified policy: {os.path.basename(unified_policy_path)} (alt={_alt_of(unified_policy_path)})"
            )
            policy = PPO.load(unified_policy_path)

            # Shot accumulators (rows=height, cols=width)
            attempts = np.zeros((env.court_height, env.court_width), dtype=np.int64)
            makes = np.zeros((env.court_height, env.court_width), dtype=np.int64)

            # Run episodes and collect shots
            for _ in _tqdm(
                range(args.episodes), total=args.episodes, desc="Shot chart episodes"
            ):
                obs, _ = env.reset()
                done = False
                while not done:
                    # Build role-conditioned observations for unified policy
                    action_mask = obs.get("action_mask")
                    try:
                        offense_obs = {
                            "obs": np.copy(obs["obs"]),
                            "action_mask": action_mask,
                            "role_flag": np.array([1.0], dtype=np.float32),
                            "skills": (
                                np.copy(obs.get("skills"))
                                if obs.get("skills") is not None
                                else obs.get("skills")
                            ),
                        }
                    except Exception:
                        offense_obs = obs
                    try:
                        defense_obs = {
                            "obs": np.copy(obs["obs"]),
                            "action_mask": action_mask,
                            "role_flag": np.array([0.0], dtype=np.float32),
                            "skills": (
                                np.copy(obs.get("skills"))
                                if obs.get("skills") is not None
                                else obs.get("skills")
                            ),
                        }
                    except Exception:
                        defense_obs = obs

                    off_predicted, _ = policy.predict(
                        offense_obs, deterministic=args.deterministic
                    )
                    def_predicted, _ = policy.predict(
                        defense_obs, deterministic=args.deterministic
                    )

                    strat = {
                        "noop": IllegalActionStrategy.NOOP,
                        "best": IllegalActionStrategy.BEST_PROB,
                        "sample": IllegalActionStrategy.SAMPLE_PROB,
                    }.get(
                        str(getattr(args, "unified_illegal_strategy", "noop")).lower(),
                        IllegalActionStrategy.NOOP,
                    )

                    off_probs = get_policy_action_probabilities(policy, offense_obs)
                    def_probs = get_policy_action_probabilities(policy, defense_obs)

                    off_resolved = resolve_illegal_actions(
                        np.array(off_predicted),
                        action_mask,
                        strat,
                        args.deterministic,
                        off_probs,
                    )
                    def_resolved = resolve_illegal_actions(
                        np.array(def_predicted),
                        action_mask,
                        strat,
                        args.deterministic,
                        def_probs,
                    )

                    full_action = np.zeros(env.n_players, dtype=int)
                    for pid in range(env.n_players):
                        if pid in env.offense_ids:
                            full_action[pid] = int(off_resolved[pid])
                        else:
                            full_action[pid] = int(def_resolved[pid])

                    obs, _, done, _, info = env.step(full_action)

                    action_results = info.get("action_results", {}) if info else {}
                    if action_results.get("shots"):
                        for shooter_id, shot_res in action_results["shots"].items():
                            q, r_ax = env.positions[shooter_id]
                            col, row = env._axial_to_offset(q, r_ax)
                            if (
                                0 <= row < env.court_height
                                and 0 <= col < env.court_width
                            ):
                                attempts[row, col] += 1
                                if bool(shot_res.get("success", False)):
                                    makes[row, col] += 1

            os.makedirs(args.output_dir, exist_ok=True)
            uni_alt = _alt_of(unified_policy_path)
            uni_alt_label = str(uni_alt) if uni_alt is not None else "latest"
            out_path = os.path.join(
                args.output_dir, f"shotchart_unified_{uni_alt_label}.png"
            )
            _plot_shot_chart(
                attempts,
                makes,
                f"Shot Chart: Unified {uni_alt_label}",
                out_path,
                cmap_name=args.cmap_name,
            )
            print(f"Saved: {out_path}")
            all_saved_paths.append(out_path)
            sequence_paths.append(out_path)

        # Log to MLflow all generated
        if not args.no_log_mlflow and all_saved_paths:
            with mlflow.start_run(run_id=args.run_id):
                for p in all_saved_paths:
                    mlflow.log_artifact(p, artifact_path="shotcharts")
            print("Logged shotcharts to MLflow under 'shotcharts/'.")

        # Optional: create animated GIF over the sequence
        if getattr(args, "make_gif", False):
            try:
                if len(sequence_paths) > 0:
                    gif_path = os.path.join(args.output_dir, "shotchart_anim.gif")
                    frames = [Image.open(p) for p in sequence_paths]
                    frames[0].save(
                        gif_path,
                        save_all=True,
                        append_images=frames[1:],
                        duration=int(args.gif_duration * 1000),
                        loop=0,
                        optimize=False,
                    )
                    print(f"Saved GIF: {gif_path}")
                    if not args.no_log_mlflow:
                        with mlflow.start_run(run_id=args.run_id):
                            mlflow.log_artifact(gif_path, artifact_path="shotcharts")
            except Exception as e:
                print(f"[warn] Failed to create shotchart GIF: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate an offense shot chart from a BasketWorld MLflow run (unified-only)."
    )
    parser.add_argument("--run-id", type=str, required=True, help="MLflow Run ID")
    parser.add_argument(
        "--episodes", type=int, default=200, help="Number of self-play episodes"
    )
    parser.add_argument(
        "--deterministic",
        type=lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        default=False,
    )
    parser.add_argument(
        "--unified-alt",
        type=int,
        default=None,
        help="Use specific alternation for unified policy",
    )
    parser.add_argument(
        "--unified-alts",
        type=str,
        default=None,
        help="Comma/space separated list of unified alternations",
    )
    parser.add_argument(
        "--output-dir", type=str, default=".", help="Directory to save shot chart"
    )
    parser.add_argument(
        "--no-log-mlflow", action="store_true", help="Do not log artifacts to MLflow"
    )
    # Optional overrides
    parser.add_argument(
        "--allow-dunks",
        type=lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        default=None,
    )
    parser.add_argument("--dunk-pct", type=float, default=None)
    parser.add_argument("--cmap-name", type=str, default="RdYlGn")
    parser.add_argument(
        "--hex-radius-max", type=float, default=1.0, help="Maximum hex radius"
    )
    parser.add_argument(
        "--hex-radius-min", type=float, default=0.35, help="Minimum hex radius"
    )
    parser.add_argument(
        "--label-fontsize", type=float, default=7.0, help="Base FG% label font size"
    )
    parser.add_argument(
        "--make-gif",
        action="store_true",
        help="Also compile generated images into an animated GIF",
    )
    parser.add_argument(
        "--gif-duration", type=float, default=0.8, help="Seconds per frame in GIF"
    )
    # Illegal action resolution for unified policy (align with heatmap/backend)
    parser.add_argument(
        "--unified-illegal-strategy",
        type=str,
        choices=["noop", "sample", "best"],
        default="noop",
        help="Strategy to resolve illegal actions for unified policy",
    )
    # Swallow stray args from older invocations
    parser.add_argument("extras", nargs="*", help=argparse.SUPPRESS)

    args = parser.parse_args()
    main(args)
