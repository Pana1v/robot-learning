"""Unit tests for the BARN pure-Python core (``my_robot_tasks.barn``).

No Isaac Lab / Isaac Sim imports. Runs in seconds without Kit.
Covers PLAN.md Phase 2 unit tests U1-U9.

The barn modules are imported by path (not ``import my_robot_tasks``) because the
package ``__init__`` pulls in Isaac Lab, which cannot boot in this sandbox.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

# Make the barn modules importable with or without an editable install, and
# without triggering my_robot_tasks/__init__.py (which imports Isaac Lab).
_REPO_ROOT = Path(__file__).resolve().parents[4]
_SRC = _REPO_ROOT / "source" / "my_robot_tasks"
BARN_DIR = _SRC / "my_robot_tasks" / "barn"
for _p in (str(_SRC), str(BARN_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import geometry  # noqa: E402
import metric  # noqa: E402
import worlds  # noqa: E402

from geometry import analytic_lidar, footprint_hits_cylinders, min_pool_ranges  # noqa: E402
from metric import nav_metric, optimal_time  # noqa: E402
from worlds import (  # noqa: E402
    CELL,
    C_SHIFT,
    GOAL_XY,
    NUM_STATIC_WORLDS,
    RADIUS,
    R_SHIFT,
    START_XY,
    TEST_SPLIT,
    grid_to_world,
    path_length,
    parse_world,
    world_split,
)

WORLDS_DIR = _REPO_ROOT / "third_party" / "the-barn-challenge" / "jackal_helper" / "worlds" / "BARN"
PATH_FILES_DIR = WORLDS_DIR / "path_files"


class TestWorlds(unittest.TestCase):
    """U1-U4: world parsing, path lengths, axis orientation."""

    def test_u1_world_0_parse(self):
        w = parse_world(0, WORLDS_DIR)
        self.assertEqual(w.num_cylinders, 209)
        self.assertEqual(w.radius, RADIUS)
        self.assertGreaterEqual(w.centers[:, 0].min(), -4.425 - 1e-6)
        self.assertLessEqual(w.centers[:, 0].max(), -0.075 + 1e-6)
        self.assertGreaterEqual(w.centers[:, 1].min(), 0.075 - 1e-6)
        self.assertLessEqual(w.centers[:, 1].max(), 9.525 + 1e-6)

    def test_u2_all_worlds_on_grid(self):
        tol = 1e-3
        # Obstacle grid: x = -4.575 + 0.15k (offset R_SHIFT), y = 0.075 + 0.15k
        # (offset RADIUS). The path grid uses C_SHIFT=5.075 for y, which is
        # 1/3 cell off the obstacle grid (PLAN "path y-offset quirk") and must
        # not be applied to obstacle centers.
        for i in range(NUM_STATIC_WORLDS):
            with self.subTest(world=i):
                w = parse_world(i, WORLDS_DIR)
                kx = (w.centers[:, 0] - R_SHIFT) / CELL
                ky = (w.centers[:, 1] - RADIUS) / CELL
                self.assertTrue(
                    np.all(np.abs(kx - np.round(kx)) < tol), f"world {i} x off grid"
                )
                self.assertTrue(
                    np.all(np.abs(ky - np.round(ky)) < tol), f"world {i} y off grid"
                )

    def test_u3_world_0_path_length(self):
        pl = path_length(0, PATH_FILES_DIR)
        self.assertAlmostEqual(pl, 13.592, delta=1e-3)

    def test_u4_no_path_point_inside_cylinder(self):
        tol = 1e-6
        world_0 = None
        for i in range(NUM_STATIC_WORLDS):
            with self.subTest(world=i):
                w = parse_world(i, WORLDS_DIR)
                if i == 0:
                    world_0 = w
                path_array = np.load(PATH_FILES_DIR / f"path_{i}.npy")
                pts = np.array([grid_to_world(int(g[0]), int(g[1])) for g in path_array])
                seq = np.vstack([np.array(START_XY), pts, np.array(GOAL_XY)])
                diff = seq[:, None, :] - w.centers[None, :, :]
                dist = np.sqrt((diff * diff).sum(axis=-1))
                min_dist = float(dist.min())
                self.assertGreater(
                    min_dist,
                    RADIUS + tol,
                    f"world {i} path point inside cylinder (min {min_dist:.3f} m)",
                )
        self.assertIsNotNone(world_0)
        # world_0 oracle: nearest cylinder to any path point is 0.461 m.
        pts = np.array(
            [grid_to_world(int(g[0]), int(g[1])) for g in np.load(PATH_FILES_DIR / "path_0.npy")]
        )
        seq = np.vstack([np.array(START_XY), pts, np.array(GOAL_XY)])
        diff = seq[:, None, :] - world_0.centers[None, :, :]
        min_dist = float(np.sqrt((diff * diff).sum(axis=-1)).min())
        self.assertAlmostEqual(min_dist, 0.461, delta=0.01)
        # Axis quirk: swapping g0/g1 puts a world_0 path point inside a cylinder
        # (0.05 m < 0.075 m radius), so the axis order matters.
        swapped = np.array(
            [grid_to_world(int(g[1]), int(g[0])) for g in np.load(PATH_FILES_DIR / "path_0.npy")]
        )
        seq = np.vstack([np.array(START_XY), swapped, np.array(GOAL_XY)])
        diff = seq[:, None, :] - world_0.centers[None, :, :]
        swapped_min = float(np.sqrt((diff * diff).sum(axis=-1)).min())
        self.assertLess(swapped_min, RADIUS, f"swapped axis should collide, got {swapped_min:.3f} m")


class TestMetric(unittest.TestCase):
    """U5: nav metric boundary conditions."""

    def test_u5_metric_boundaries(self):
        OT = torch.tensor([10.0])  # path_length 20 m -> optimal time 10 s
        # Failure always scores 0.
        self.assertEqual(nav_metric(False, 5.0, OT).item(), 0.0)
        self.assertEqual(nav_metric(False, 100.0, OT, dynabarn_parity=True).item(), 0.0)
        # AT <= 2*OT clips to the lower bound -> 0.5 ceiling.
        self.assertAlmostEqual(nav_metric(True, 2.0 * OT, OT).item(), 0.5)
        self.assertAlmostEqual(nav_metric(True, OT, OT).item(), 0.5)
        # Mid point: AT = 5*OT -> OT / (5*OT) = 0.2.
        self.assertAlmostEqual(nav_metric(True, 5.0 * OT, OT).item(), 0.2)
        # AT >= 8*OT clips to the upper bound -> 0.125.
        self.assertAlmostEqual(nav_metric(True, 8.0 * OT, OT).item(), 0.125)
        self.assertAlmostEqual(nav_metric(True, 100.0 * OT, OT).item(), 0.125)
        # DynaBARN parity: 0.125 for any actual_time on success.
        self.assertAlmostEqual(nav_metric(True, 1000.0, OT, dynabarn_parity=True).item(), 0.125)
        self.assertAlmostEqual(nav_metric(True, 0.1, OT, dynabarn_parity=True).item(), 0.125)
        # Vector input.
        success = torch.tensor([True, False, True])
        at = torch.tensor([15.0, 1.0, 100.0])
        out = nav_metric(success, at, OT)
        self.assertEqual(out.shape, torch.Size([3]))
        self.assertAlmostEqual(out[0].item(), 0.5)  # 15 < 20 -> clipped to 20 -> OT/20 = 0.5
        self.assertEqual(out[1].item(), 0.0)
        self.assertAlmostEqual(out[2].item(), 0.125)  # 100 > 80 -> clipped to 80 -> OT/80 = 0.125


class TestGeometry(unittest.TestCase):
    """U6-U8: min-pooling, footprint test, analytic lidar."""

    def test_u6_min_pool_single_hit_survives(self):
        n_in, out_bins = 720, 64
        ranges = torch.full((n_in,), float("inf"))
        hit_beam = 100
        ranges[hit_beam] = 2.5
        pooled = min_pool_ranges(ranges, out_bins)
        self.assertEqual(pooled.shape, torch.Size([out_bins]))
        expected_bin = int((hit_beam * out_bins / n_in) // 1)
        self.assertAlmostEqual(pooled[expected_bin].item(), 2.5)
        # Every other bin is inf (no other hits).
        others = torch.cat([pooled[:expected_bin], pooled[expected_bin + 1 :]])
        self.assertTrue(torch.all(torch.isinf(others)))
        # Stride slicing on the same input loses the hit: 100 is not a multiple of 11.
        strided = ranges[::11]
        self.assertFalse(torch.any(strided == 2.5), "stride slicing must lose the single-beam hit")

    def test_u7_footprint_touch_is_hit_gap_is_not(self):
        half_length, half_width, radius = 0.23, 0.21, 0.075
        center = torch.zeros(1, 2)
        heading = torch.zeros(1)
        cylinders = torch.tensor([[0.305, 0.0]])  # touching the forward face
        self.assertTrue(
            footprint_hits_cylinders(center, heading, half_length, half_width, cylinders, radius).item()
        )
        # 1 cm gap -> no hit.
        cylinders = torch.tensor([[0.315, 0.0]])
        self.assertFalse(
            footprint_hits_cylinders(center, heading, half_length, half_width, cylinders, radius).item()
        )
        # Lateral touch.
        cylinders = torch.tensor([[0.0, 0.285]])
        self.assertTrue(
            footprint_hits_cylinders(center, heading, half_length, half_width, cylinders, radius).item()
        )
        # Rotated heading pi/4, cylinder along the forward axis at touch distance.
        heading = torch.tensor([np.pi / 4])
        forward = torch.tensor([[np.cos(np.pi / 4) * 0.305, np.sin(np.pi / 4) * 0.305]])
        self.assertTrue(
            footprint_hits_cylinders(
                center, heading, half_length, half_width, forward, radius
            ).item()
        )
        # Rotated with a 1 cm gap along the forward axis.
        forward = torch.tensor([[np.cos(np.pi / 4) * 0.315, np.sin(np.pi / 4) * 0.315]])
        self.assertFalse(
            footprint_hits_cylinders(
                center, heading, half_length, half_width, forward, radius
            ).item()
        )

    def test_u8_analytic_lidar_center_beam(self):
        origins = torch.zeros(1, 2)
        headings = torch.zeros(1)
        n_beams = 720
        beam_angles = torch.linspace(-2.35619, 2.35619, n_beams)
        beam_angles[n_beams // 2] = 0.0  # 720 is even: linspace has no exact 0, force a forward beam
        # One cylinder centered 3.0 m straight ahead; the center beam (angle 0)
        # should read 3.0 - 0.075 = 2.925 m.
        centers = torch.tensor([[3.0, 0.0]])
        dist = analytic_lidar(origins, headings, beam_angles, centers, RADIUS, 30.0)
        self.assertEqual(dist.shape, torch.Size([1, n_beams]))
        center_idx = n_beams // 2
        self.assertAlmostEqual(dist[0, center_idx].item(), 2.925, delta=1e-4)
        # The minimum over all beams equals the center-beam reading (one cylinder).
        self.assertAlmostEqual(dist.min().item(), 2.925, delta=1e-4)
        # A beam pointing away misses the cylinder to the side; misses clamp to max_distance.
        self.assertAlmostEqual(dist[0, center_idx + n_beams // 4].item(), 30.0)


class TestSplits(unittest.TestCase):
    """U9: train / test split."""

    def test_u9_splits(self):
        train, test = world_split()
        self.assertEqual(test, list(TEST_SPLIT))
        self.assertEqual(len(test), 50)
        self.assertEqual(len(train), 250)
        train_set, test_set = set(train), set(test)
        self.assertEqual(len(train_set & test_set), 0)
        self.assertEqual(train_set | test_set, set(range(NUM_STATIC_WORLDS)))


if __name__ == "__main__":
    unittest.main()
