"""Build the offline BARN static-world data table.

Parses all 300 upstream static BARN worlds (SDF) and their planned path files,
then writes ``data/barn_static.npz`` with per-world cylinder centers, path
lengths, and optimal times for use by the environment and evaluation.

No Isaac Lab / Isaac Sim imports: runs without Kit.

Outputs (``data/barn_static.npz``):
  centers        (300, 209, 2) float32  cylinder centers [m] per world
  path_lengths   (300,) float32          planned path length [m] per world
  optimal_times  (300,) float32          path_length / 2.0 [s] per world
  world_indices  (300,) int32            world indices 0..299
  upstream_hash  ()                      upstream git hash (bf5a226)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import sys

# Path resolution mirrors tests/unit/test_barn_core.py: add both the package
# root and the barn module dir so we can import the submodules directly without
# triggering my_robot_tasks/__init__.py (which pulls in Isaac Lab).
# script dir = scripts/, parents[1] = my_robot_tasks package root,
# parents[3] = robot-learning repo root (holds source/ and third_party/).
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = _REPO_ROOT / "source" / "my_robot_tasks"
_BARN_DIR = _SRC / "my_robot_tasks" / "barn"
for _p in (str(_SRC), str(_BARN_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from worlds import (  # noqa: E402
    GOAL_XY,
    NUM_STATIC_WORLDS,
    RADIUS,
    START_XY,
    grid_to_world,
    path_length_from_grid,
    parse_world,
)

WORLDS_DIR = _REPO_ROOT / "third_party" / "the-barn-challenge" / "jackal_helper" / "worlds" / "BARN"
PATH_FILES_DIR = WORLDS_DIR / "path_files"
# The project root is read-only to subprocesses, so write the build artifact to
# the writable working directory (Happy Agent sandbox cwd).
OUT_PATH = Path.cwd() / "data" / "barn_static.npz"
UPSTREAM_HASH = "bf5a226"  # the-barn-challenge@bf5a226


def main() -> None:
    centers_list: list[np.ndarray] = []
    num_cylinders = np.zeros(NUM_STATIC_WORLDS, dtype=np.int32)
    path_lengths = np.zeros(NUM_STATIC_WORLDS, dtype=np.float32)

    for i in range(NUM_STATIC_WORLDS):
        w = parse_world(i, WORLDS_DIR)
        centers_list.append(w.centers.astype(np.float32))
        num_cylinders[i] = w.num_cylinders

        path_array = np.load(PATH_FILES_DIR / f"path_{i}.npy")
        path_lengths[i] = np.float32(path_length_from_grid(path_array))

    optimal_times = (path_lengths / 2.0).astype(np.float32)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_PATH,
        centers=np.array(centers_list, dtype=object),
        num_cylinders=num_cylinders,
        path_lengths=path_lengths,
        optimal_times=optimal_times,
        world_indices=np.arange(NUM_STATIC_WORLDS, dtype=np.int32),
        upstream_hash=np.array([UPSTREAM_HASH.encode()], dtype="S7"),
    )

    print(f"Wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.1f} KB)")
    print(f"  worlds:      {NUM_STATIC_WORLDS}")
    print(f"  cylinders:   min={num_cylinders.min()} max={num_cylinders.max()} per world")
    print(f"  path_len:    min={path_lengths.min():.3f} max={path_lengths.max():.3f} mean={path_lengths.mean():.3f} m")
    print(f"  optimal_t:   min={optimal_times.min():.3f} max={optimal_times.max():.3f} s")
    print(f"  upstream:    {UPSTREAM_HASH}")


if __name__ == "__main__":
    main()
