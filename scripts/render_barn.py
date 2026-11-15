#!/usr/bin/env python3
"""Render the BARN navigation world (top-down) without Isaac Sim.

Draws the actual static world from data/barn_static.npz:

* Jackal at the BARN start pose (-2.25, 3.0 m, yaw 1.57 rad) as a
  0.46 x 0.42 m rectangle with a heading arrow.
* All obstacle cylinders (r = 0.075 m) for the chosen world.
* The absolute goal (-2.25, 13.0 m) and the 30 m lidar max-range circle.
* A 720-beam lidar fan: beams that hit a cylinder are colored by distance,
  misses are gray.
* The 64-bin min-pooled lidar observation the policy receives (polar bar).

Outputs PNG figures into --out-dir (default .context/renders/).

No Isaac Lab / Isaac Sim import: runs with numpy + matplotlib only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
except Exception as exc:  # pragma: no cover - environment guard
    print(f"render_barn: matplotlib required to render ({exc})", file=sys.stderr)
    sys.exit(1)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import LineCollection

# ---------------------------------------------------------------------------
# BARN constants (mirror my_robot_tasks/barn/worlds.py)
# ---------------------------------------------------------------------------
RADIUS = 0.075  # [m] cylinder radius
ROBOT_LEN, ROBOT_WID = 0.46, 0.42  # [m] Jackal chassis footprint
START_XY = np.array((-2.25, 3.0), dtype=float)  # [m] BARN static start
START_YAW = 1.57  # [rad] BARN static start yaw
GOAL_XY = np.array((-2.25, 13.0), dtype=float)  # [m] absolute goal
LIDAR_MAX = 30.0  # [m] sensor max range
LIDAR_BEAMS = 720  # beam count
LIDAR_FOV_DEG = 270.0  # [deg] horizontal field of view
LIDAR_BINS = 64  # min-pooled observation bins

REPO_ROOT = Path(__file__).resolve().parents[1]
NPZ_PATH = REPO_ROOT / "data" / "barn_static.npz"


def load_world(world_idx: int) -> np.ndarray:
    """Return (N, 2) cylinder centers [m] for a static world index."""
    data = np.load(NPZ_PATH, allow_pickle=True)
    centers = data["centers"][int(world_idx)]
    counts = data["num_cylinders"]
    return np.asarray(centers[: int(counts[int(world_idx)])], dtype=float)


def world_extent(centers: np.ndarray) -> tuple[float, float, float, float]:
    """Axis-aligned bounding box of the cylinders (world frame)."""
    xs = centers[:, 0]
    ys = centers[:, 1]
    return float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())


def lidar_scan(
    origin: np.ndarray,
    heading: float,
    centers: np.ndarray,
    max_distance: float = LIDAR_MAX,
    n_beams: int = LIDAR_BEAMS,
    fov_deg: float = LIDAR_FOV_DEG,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ray-vs-circle lidar, batched over beams.

    Returns
    -------
    angles : (B,) beam angles [rad]
    endpoints : (B, 2) hit/miss endpoints [m]
    distances : (B,) distance to nearest hit, max_distance on a miss [m]
    """
    angles = heading + np.deg2rad(np.linspace(-fov_deg / 2.0, fov_deg / 2.0, n_beams))
    dirs = np.stack((np.cos(angles), np.sin(angles)), axis=1)  # (B, 2)

    # f[b, c] = origin - center_c ; broadcast (B, C, 2)
    f = origin[None, None, :] - centers[None, :, :]
    df = np.sum(f * dirs[:, None, :], axis=-1)  # d . f   (B, C)
    ff = np.sum(f * f, axis=-1)  # |f|^2                (B, C)
    disc = df * df - ff + RADIUS * RADIUS  # discriminant (B, C)
    hit = disc >= 0.0

    root = np.sqrt(np.clip(disc, 0.0, None))
    t1 = -df - root  # near intersection
    t2 = -df + root  # far intersection
    # Nearest positive intersection along the ray.
    t = np.where(t1 > 0.0, t1, np.where(t2 > 0.0, t2, np.full_like(t1, np.inf)))
    t = np.where(hit, t, np.full_like(t, np.inf))
    t = np.clip(t, None, max_distance)
    distances = np.min(t, axis=1)  # (B,)

    endpoints = origin[None, :] + distances[:, None] * dirs
    return angles, endpoints, distances


def min_pool_ranges(ranges: np.ndarray, out_bins: int) -> np.ndarray:
    """Min-pool contiguous bins of a 1D range array (never stride slicing)."""
    n = ranges.shape[-1]
    bins = np.floor(np.arange(n, dtype=np.float32) * out_bins / n).astype(int)
    bins = np.clip(bins, 0, out_bins - 1)
    pooled = np.full(out_bins, np.inf, dtype=ranges.dtype)
    for b in range(out_bins):
        pooled[b] = np.min(ranges[bins == b])
    return pooled


def robot_patch(origin: np.ndarray, yaw: float, length: float, width: float) -> Rectangle:
    """Jackal chassis as a rotated rectangle centered at origin."""
    rect = Rectangle(
        (-length / 2.0, -width / 2.0),
        length,
        width,
        angle=np.rad2deg(yaw),
        rotation_point="center",
        fill=True,
        facecolor="#1f77b4",
        edgecolor="#0b3d66",
        linewidth=1.4,
        zorder=5,
    )
    return rect


def heading_arrow(origin: np.ndarray, yaw: float, length: float = 0.35):
    """Arrow showing the robot's forward heading."""
    dx = length * np.cos(yaw)
    dy = length * np.sin(yaw)
    return dict(x=origin[0], y=origin[1], dx=dx, dy=dy,
                width=0.02, head_width=0.09, head_length=0.14,
                fc="#0b3d66", ec="#0b3d66", zorder=6)


def plot_world(ax, centers: np.ndarray, world_idx: int, show_lidar: bool = False,
               origin: np.ndarray = START_XY, yaw: float = START_YAW,
               show_goal: bool = True,
               xlim: tuple[float, float] | None = None,
               ylim: tuple[float, float] | None = None):
    """Draw cylinders, robot, goal, max-range circle on one axes."""
    # Lidar max-range circle (reference only).
    max_circle = Circle((0.0, 0.0), LIDAR_MAX, fill=False, edgecolor="#d0d0d0",
                        linewidth=0.8, linestyle=(0, (4, 4)), zorder=1, alpha=0.6)
    ax.add_patch(max_circle)

    # Obstacle cylinders.
    for cx, cy in centers:
        ax.add_patch(Circle((cx, cy), RADIUS, fill=True, facecolor="#ff7f0e",
                            edgecolor="#b35900", linewidth=0.5, zorder=3, alpha=0.85))

    # Robot at start pose.
    ax.add_patch(robot_patch(origin, yaw, ROBOT_LEN, ROBOT_WID))
    arr = heading_arrow(origin, yaw)
    ax.annotate("", xy=(arr["x"] + arr["dx"], arr["y"] + arr["dy"]),
                xytext=(arr["x"], arr["y"]),
                arrowprops=dict(arrowstyle="->", color=arr["fc"],
                                lw=2.2, mutation_scale=24, zorder=6))

    # Goal marker.
    if show_goal:
        ax.plot(GOAL_XY[0], GOAL_XY[1], marker="*", markersize=22, color="#2ca02c",
                markeredgecolor="#1b6b1b", markeredgewidth=1.2, zorder=7, label="goal")
        ax.annotate("goal (-2.25, 13.0)", (GOAL_XY[0], GOAL_XY[1]),
                    textcoords="offset points", xytext=(10, 10), fontsize=8,
                    color="#1b6b1b")

    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"BARN static world {world_idx} — {len(centers)} cylinders\n"
                 f"robot at start (-2.25, 3.0), yaw 1.57 rad",
                 fontsize=11)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)


def plot_lidar_fan(ax, centers: np.ndarray, world_idx: int,
                   origin: np.ndarray = START_XY, yaw: float = START_YAW):
    """Draw the 720-beam lidar fan; hit beams colored by distance, misses gray."""
    angles, endpoints, distances = lidar_scan(origin, yaw, centers)
    hit = np.isfinite(distances) & (distances < LIDAR_MAX - 1e-3)

    # Draw cylinders faintly behind the fan.
    for cx, cy in centers:
        ax.add_patch(Circle((cx, cy), RADIUS, fill=True, facecolor="#ff7f0e",
                            edgecolor="#b35900", linewidth=0.4, zorder=2, alpha=0.5))

    # Robot.
    ax.add_patch(robot_patch(origin, yaw, ROBOT_LEN, ROBOT_WID))
    arr = heading_arrow(origin, yaw)
    ax.annotate("", xy=(arr["x"] + arr["dx"], arr["y"] + arr["dy"]),
                xytext=(arr["x"], arr["y"]),
                arrowprops=dict(arrowstyle="->", color=arr["fc"],
                                lw=2.0, mutation_scale=22, zorder=6))

    # Goal.
    ax.plot(GOAL_XY[0], GOAL_XY[1], marker="*", markersize=18, color="#2ca02c",
            markeredgecolor="#1b6b1b", zorder=6)

    # Beam segments from origin to endpoint.
    segments = np.stack((np.full_like(endpoints, origin), endpoints), axis=1)  # (B, 2, 2)
    colors = np.empty((len(segments), 4), dtype=float)
    cmap = plt.cm.plasma
    norm_dist = np.clip(distances / LIDAR_MAX, 0.0, 1.0)
    colors[hit] = cmap(norm_dist[hit])
    colors[~hit] = (0.75, 0.75, 0.75, 0.35)  # misses: light gray, translucent
    lw = np.where(hit, 0.9, 0.4)

    lc = LineCollection(segments, colors=colors, linewidths=lw, zorder=4)
    ax.add_collection(lc)

    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    nhit = int(hit.sum())
    ax.set_title(f"Lidar fan — world {world_idx} ({LIDAR_BEAMS} beams, {-LIDAR_FOV_DEG/2}..{LIDAR_FOV_DEG/2} deg)\n"
                 f"{nhit} beams hit a cylinder, {int((~hit).sum())} misses (30 m)",
                 fontsize=10)
    ax.grid(True, linewidth=0.4, alpha=0.5)


def plot_observation(ax, centers: np.ndarray, world_idx: int,
                     origin: np.ndarray = START_XY, yaw: float = START_YAW):
    """Polar bar chart of the 64-bin min-pooled lidar observation (normalized [0,1])."""
    _, _, distances = lidar_scan(origin, yaw, centers)
    pooled = min_pool_ranges(distances, LIDAR_BINS)
    obs = np.clip(pooled / LIDAR_MAX, 0.0, 1.0)

    theta = np.linspace(0.0, 2.0 * np.pi, LIDAR_BINS, endpoint=False)
    width = 2.0 * np.pi / LIDAR_BINS
    colors = plt.cm.plasma(np.clip(obs, 0.0, 1.0))

    bars = ax.bar(theta, obs, width=width, bottom=0.0, color=colors, edgecolor="k",
                  linewidth=0.3, alpha=0.9, zorder=3)
    ax.set_ylim(0.0, 1.05)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.0", "0.25", "0.5", "0.75", "1.0"], fontsize=8)
    ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(["0°", "45°", "90°", "135°", "180°", "225°", "270°", "315°"],
                       fontsize=8)
    ax.set_title(f"Policy observation — world {world_idx}\n"
                 f"{LIDAR_BINS} min-pooled lidar bins, normalized by {LIDAR_MAX} m (miss -> 1.0)",
                 fontsize=10, pad=16)
    ax.grid(True, linewidth=0.4, alpha=0.5, axis="y")


def render_world(world_idx: int, out_dir: Path):
    """Render one world: zoomed start view, lidar fan, and observation polar chart."""
    centers = load_world(world_idx)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: zoomed view around the start pose so individual cylinders are visible.
    sx, sy = START_XY
    fig, ax = plt.subplots(figsize=(8.2, 6.2), dpi=150)
    plot_world(ax, centers, world_idx, origin=START_XY, yaw=START_YAW,
               show_goal=False, xlim=(sx - 1.6, sx + 1.6), ylim=(sy - 1.6, sy + 1.6))
    ax.annotate("goal is 10 m north at (-2.25, 13.0)",
                (sx, sy + 1.6), textcoords="offset points", xytext=(0, 6),
                ha="center", fontsize=9, color="#1b6b1b",
                fontweight="bold")
    fig.tight_layout()
    path = out_dir / f"world_{world_idx:03d}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")

    # Figure 2: lidar fan over the full world extent.
    x0, x1, y0, y1 = world_extent(centers)
    pad = 3.0
    xmin = min(x0, START_XY[0], GOAL_XY[0]) - pad
    xmax = max(x1, START_XY[0], GOAL_XY[0]) + pad
    ymin = min(y0, START_XY[1], GOAL_XY[1]) - pad
    ymax = max(y1, START_XY[1], GOAL_XY[1]) + pad

    fig, ax = plt.subplots(figsize=(8.2, 6.2), dpi=150)
    plot_lidar_fan(ax, centers, world_idx, origin=START_XY, yaw=START_YAW)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    fig.tight_layout()
    path = out_dir / f"lidar_fan_{world_idx:03d}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")

    # Figure 3: observation polar.
    fig = plt.figure(figsize=(5.6, 5.6), dpi=150)
    ax = fig.add_subplot(111, projection="polar")
    plot_observation(ax, centers, world_idx, origin=START_XY, yaw=START_YAW)
    fig.tight_layout()
    path = out_dir / f"observation_{world_idx:03d}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Render BARN static worlds to PNG.")
    parser.add_argument("--world", type=int, default=0, help="Static world index (0-299).")
    parser.add_argument("--out-dir", type=str, default=str(REPO_ROOT / ".context" / "renders"),
                        help="Output directory for PNG figures.")
    args = parser.parse_args(argv)

    if not NPZ_PATH.exists():
        print(f"render_barn: data file not found: {NPZ_PATH}", file=sys.stderr)
        return 1

    render_world(args.world, Path(args.out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
