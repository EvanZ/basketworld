from __future__ import annotations

import math


def render_ascii(env):
    """Simple ASCII rendering for training."""
    print(f"\nShot Clock: {env.shot_clock}")
    print(f"Ball Holder: Player {env.ball_holder}")

    grid = [[" · " for _ in range(env.court_width)] for _ in range(env.court_height)]

    for r in range(env.court_height):
        for c in range(env.court_width):
            q, r_ax = env._offset_to_axial(c, r)

            if (q, r_ax) == env.basket_position:
                grid[r][c] = " B "

            for i, pos in enumerate(env.positions):
                if pos == (q, r_ax):
                    symbol = f"O{i}" if i in env.offense_ids else f"D{i}"
                    if i == env.ball_holder:
                        symbol = f"*{symbol[1]}*"
                    grid[r][c] = f" {symbol} "

    print("\nCourt Layout (O=Offense, D=Defense, *=Ball):")
    for r_idx, row in enumerate(grid):
        indent = " " if r_idx % 2 != 0 else ""
        print(indent + "".join(row))

    if env.last_action_results:
        print(f"\nLast Action Results: {env.last_action_results}")
    print("-" * 40)


def render_visual(env):
    """Visual rendering using matplotlib."""
    import io

    import matplotlib.pyplot as plt
    from matplotlib.patches import RegularPolygon
    from PIL import Image

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_aspect("equal")

    def axial_to_cartesian(q, r):
        size = 1.0
        x = size * (math.sqrt(3) * q + math.sqrt(3) / 2 * r)
        y = size * (3.0 / 2.0 * r)
        return x, y

    hex_radius = 1.0

    def _lerp(c1, c2, t):
        return (
            c1[0] + (c2[0] - c1[0]) * t,
            c1[1] + (c2[1] - c1[1]) * t,
            c1[2] + (c2[2] - c1[2]) * t,
        )

    def color_from_zscore(delta: float, std: float, zmax: float = 2.0):
        if std is None or std <= 1e-9:
            z = 0.0
        else:
            z = float(delta) / float(std)
        z = max(-zmax, min(zmax, z))
        blue = (0.20, 0.40, 0.90)
        green = (0.20, 0.70, 0.20)
        orange = (1.00, 0.60, 0.00)
        if z < 0.0:
            t = 1.0 - (abs(z) / zmax)
            return _lerp(blue, green, t)
        else:
            t = z / zmax
            return _lerp(green, orange, t)

    for c in range(env.court_width):
        for r in range(env.court_height):
            q, r_ax = env._offset_to_axial(c, r)
            x, y = axial_to_cartesian(q, r_ax)

            hexagon = RegularPolygon(
                (x, y),
                numVertices=6,
                radius=hex_radius,
                orientation=0,
                facecolor="lightgray",
                edgecolor="white",
                alpha=0.5,
                linewidth=1,
            )
            ax.add_patch(hexagon)

            if hasattr(env, "offensive_lane_hexes") and env.offensive_lane_hexes:
                is_in_lane = (q, r_ax) in env.offensive_lane_hexes
                offense_enabled = getattr(env, "offensive_three_seconds_enabled", False)
                defense_enabled = getattr(env, "illegal_defense_enabled", False)
                if is_in_lane and (offense_enabled or defense_enabled):
                    lane_hexagon = RegularPolygon(
                        (x, y),
                        numVertices=6,
                        radius=hex_radius,
                        orientation=0,
                        facecolor=(1.0, 0.39, 0.39, 0.15),
                        edgecolor=(1.0, 0.39, 0.39, 0.3),
                        linewidth=1.5,
                        zorder=2,
                    )
                    ax.add_patch(lane_hexagon)

            if (q, r_ax) == env.basket_position:
                basket_ring = plt.Circle(
                    (x, y),
                    hex_radius * 1.05,
                    fill=False,
                    edgecolor="red",
                    linewidth=4,
                    zorder=6,
                )
                ax.add_patch(basket_ring)

            if (q, r_ax) in env._three_point_line_hexes:
                tp_outline = RegularPolygon(
                    (x, y),
                    numVertices=6,
                    radius=hex_radius,
                    orientation=0,
                    facecolor="none",
                    edgecolor="red",
                    linewidth=2.5,
                    zorder=7,
                )
                ax.add_patch(tp_outline)

    for i, (q, r) in enumerate(env.positions):
        x, y = axial_to_cartesian(q, r)
        color = "blue" if i in env.offense_ids else "red"

        player_hexagon = RegularPolygon(
            (x, y),
            numVertices=6,
            radius=hex_radius,
            orientation=0,
            facecolor=color,
            edgecolor="white",
            alpha=0.9,
            zorder=10,
        )
        ax.add_patch(player_hexagon)
        ax.text(
            x,
            y,
            str(i),
            ha="center",
            va="center",
            fontsize=24,
            fontweight="bold",
            color="white",
            zorder=11,
        )

        if i == env.ball_holder:
            ball_ring = plt.Circle(
                (x, y),
                hex_radius * 0.9,
                fill=False,
                color="orange",
                linewidth=4,
                zorder=12,
            )
            ax.add_patch(ball_ring)

        try:
            if i in env.offense_ids:
                lay = float(env.offense_layup_pct_by_player[int(i)])
                three = float(env.offense_three_pt_pct_by_player[int(i)])
                dunk = float(env.offense_dunk_pct_by_player[int(i)])
                base_lay = float(env.layup_pct)
                base_three = float(env.three_pt_pct)
                base_dunk = float(env.dunk_pct)
                d_lay = lay - base_lay
                d_three = three - base_three
                d_dunk = dunk - base_dunk

                rscale = hex_radius * 1.05
                sqrt3 = math.sqrt(3.0)
                verts = [
                    (
                        x - (sqrt3 / 2.0) * rscale,
                        y - 0.5 * rscale,
                        f"{int(round(dunk * 100))}%D",
                        color_from_zscore(d_dunk, float(env.dunk_std)),
                    ),
                    (
                        x + 0.0,
                        y - 1.0 * rscale,
                        f"{int(round(lay * 100))}%L",
                        color_from_zscore(d_lay, float(env.layup_std)),
                    ),
                    (
                        x + (sqrt3 / 2.0) * rscale,
                        y - 0.5 * rscale,
                        f"{int(round(three * 100))}%3",
                        color_from_zscore(d_three, float(env.three_pt_std)),
                    ),
                ]

                for vx, vy, text, fc_col in verts:
                    ax.text(
                        vx,
                        vy,
                        text,
                        ha="center",
                        va="center",
                        fontsize=9,
                        fontweight="bold",
                        color="white",
                        bbox=dict(
                            boxstyle="round,pad=0.15",
                            fc=fc_col,
                            ec=fc_col,
                            alpha=0.85,
                        ),
                        zorder=13,
                    )
        except Exception:
            pass

    try:
        if env.last_action_results and env.last_action_results.get("passes"):
            for passer_id_str, pass_res in env.last_action_results["passes"].items():
                if pass_res.get("success") and "target" in pass_res:
                    passer_id = int(passer_id_str)
                    target_id = int(pass_res.get("target"))
                    pq, pr = env.positions[passer_id]
                    tq, tr = env.positions[target_id]
                    x1, y1 = axial_to_cartesian(pq, pr)
                    x2, y2 = axial_to_cartesian(tq, tr)
                    ax.annotate(
                        "",
                        xy=(x2, y2),
                        xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="->", color="black", linewidth=3),
                        zorder=19,
                    )
    except Exception:
        pass

    try:
        for oid in env.offense_ids:
            q, r = env.positions[oid]
            dist = env._hex_distance((q, r), env.basket_position)
            prob = float(env._calculate_shot_probability(int(oid), int(dist)))
            pct_text = f"{int(round(prob * 100))}%"
            x, y = axial_to_cartesian(q, r)
            tx = x + hex_radius * 0.0
            ty = y + hex_radius * 1.2
            ax.text(
                tx,
                ty,
                pct_text,
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="black", alpha=0.7),
                zorder=18,
            )
    except Exception:
        pass

    try:
        if env._assist_candidate is not None:
            aid = env._assist_candidate.get("passer_id", None)
            rid = env._assist_candidate.get("recipient_id", None)
            if aid is not None and rid is not None:
                aq, ar = env.positions[int(aid)]
                rq, rr = env.positions[int(rid)]
                x1, y1 = axial_to_cartesian(aq, ar)
                x2, y2 = axial_to_cartesian(rq, rr)
                ax.annotate(
                    "",
                    xy=(x2, y2),
                    xytext=(x1, y1),
                    arrowprops=dict(
                        arrowstyle="->", linestyle="dashed", color="green", linewidth=2
                    ),
                    zorder=19,
                )
    except Exception:
        pass

    all_x = []
    all_y = []
    for q, r in env.positions:
        x, y = axial_to_cartesian(q, r)
        all_x.append(x)
        all_y.append(y)
    bx, by = axial_to_cartesian(*env.basket_position)
    all_x.append(bx)
    all_y.append(by)
    padding = 2.5
    ax.set_xlim(min(all_x) - padding, max(all_x) + padding)
    ax.set_ylim(min(all_y) - padding, max(all_y) + padding)
    ax.axis("off")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)

    img = Image.open(buf)
    rgb_array = np.array(img.convert("RGB"))
    buf.close()

    return rgb_array
