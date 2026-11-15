"""Offline parsing of the upstream BARN static worlds (Gazebo SDF).

Reads ``jackal_helper/worlds/BARN/world_{i}.world`` from the pinned upstream
clone and extracts:
  * the static cylinder centers (world frame, meters)
  * the planned path length (meters)

No Isaac Lab / Isaac Sim imports: fully testable without Kit.

Upstream: Daffan/the-barn-challenge@bf5a226
  run.py            start/goal, path coordinate conversion, nav metric
  worlds/BARN/path_files/path_{i}.npy  planned grid-index path (43 points)
  worlds/BARN/world_{i}.world          static cylinder poses
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

# Upstream constants (the-barn-challenge@bf5a226, run.py / report_test.py).
RADIUS = 0.075  # [m] cylinder radius
CYLINDER_LENGTH = 1.0  # [m] cylinder length
CELL = 2.0 * RADIUS  # 0.15 m grid spacing
R_SHIFT = -RADIUS - 30 * RADIUS * 2  # -4.575 m
C_SHIFT = RADIUS + 5.0  # 5.075 m

START_XY = (-2.25, 3.0)  # world frame [m]
START_YAW = 1.57  # [rad]
GOAL_OFFSET = (0.0, 10.0)  # relative to start [m]
GOAL_XY = (
    START_XY[0] + GOAL_OFFSET[0],
    START_XY[1] + GOAL_OFFSET[1],
)  # (-2.25, 13.0) absolute

NUM_STATIC_WORLDS = 300
# Upstream test set: 50 static worlds, evenly spaced, never used for training.
TEST_SPLIT = tuple(range(0, NUM_STATIC_WORLDS, 6))  # 0, 6, ..., 294


def grid_to_world(gx: float, gy: float) -> tuple[float, float]:
    """Upstream ``path_coord_to_gazebo_coord``: grid index -> world meters."""
    return (gx * CELL + R_SHIFT, gy * CELL + C_SHIFT)


@dataclass
class StaticWorld:
    index: int
    centers: np.ndarray  # (N, 2) world-frame cylinder centers [m]
    radius: float = RADIUS
    path_length: float = 0.0

    @property
    def num_cylinders(self) -> int:
        return int(self.centers.shape[0])


def _pose_centers_from_sdf(xml: str) -> list[tuple[float, float]]:
    """Extract (x, y) centers of every ``unit_cylinder_*`` model from SDF XML."""
    root = ET.fromstring(xml)
    world = root.find("world")
    if world is None:
        raise ValueError("SDF has no <world> element")
    centers: list[tuple[float, float]] = []
    for model in world.findall("model"):
        if not model.get("name", "").startswith("unit_cylinder_"):
            continue
        pose = model.find("pose")
        if pose is None or not pose.text:
            continue
        vals = [float(v) for v in pose.text.strip().split()]
        if len(vals) < 2:
            continue
        centers.append((vals[0], vals[1]))
    return centers


def parse_world(index: int, worlds_dir: str | Path) -> StaticWorld:
    """Parse a single static world file into cylinder centers."""
    path = Path(worlds_dir) / f"world_{index}.world"
    xml = path.read_text()
    centers = _pose_centers_from_sdf(xml)
    arr = np.array(centers, dtype=np.float64).reshape(-1, 2)
    return StaticWorld(index=index, centers=arr)


def path_length_from_grid(path_array: np.ndarray) -> float:
    """Path length (m) from a grid-index path, start and goal appended."""
    world_pts = np.array(
        [grid_to_world(int(g[0]), int(g[1])) for g in path_array], dtype=np.float64
    )
    start = np.array(START_XY, dtype=np.float64)
    goal = np.array(GOAL_XY, dtype=np.float64)
    seq = np.vstack([start, world_pts, goal])
    diffs = np.diff(seq, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def path_length(index: int, path_files_dir: str | Path) -> float:
    """Planned path length for a static world index."""
    path_array = np.load(Path(path_files_dir) / f"path_{index}.npy")
    return path_length_from_grid(path_array)


def world_split() -> tuple[list[int], list[int]]:
    """Train / test split over static world indices.

    Test worlds are never used for training.
    """
    test = list(TEST_SPLIT)
    test_set = set(test)
    train = [i for i in range(NUM_STATIC_WORLDS) if i not in test_set]
    return train, test
