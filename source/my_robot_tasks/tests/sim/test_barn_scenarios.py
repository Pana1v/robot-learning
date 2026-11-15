"""Sim scenario tests for the BARN navigation environment (S1-S8).

These tests exercise BarnNavEnv end-to-end once the Isaac Sim / Isaac Lab runtime
is available (Gate 1 unblocked). Until then they skip, documenting that they
cannot run in this sandbox.

Covers PLAN.md Phase 3 sim scenario tests:
  S1: env constructs; observation space 71, action space 2
  S2: reset places robot at the BARN start pose
  S3: a velocity-driven step moves the robot (body advances)
  S4: lidar observation is 64 bins, normalized to [0, 1] (inf -> 1.0)
  S5: driving into a cylinder triggers contact + terminated (collision)
  S6: reaching the goal triggers success + terminated + goal reward
  S7: reaching max_episode_length triggers time_out (truncated=True)
  S8: compute_nav_metric matches the barn/metric.py oracle
"""

from __future__ import annotations

import math
import unittest
from pathlib import Path

import torch

try:
    from isaaclab.app import AppLauncher

    _ISAACLAB_AVAILABLE = True
except Exception:  # ImportError / runtime failure without Kit
    _ISAACLAB_AVAILABLE = False

# Make the barn modules importable without triggering my_robot_tasks/__init__.py.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_SRC = _REPO_ROOT / "source" / "my_robot_tasks"
_BARN_DIR = _SRC / "my_robot_tasks" / "barn"
import sys

for _p in (str(_SRC), str(_BARN_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from metric import nav_metric as barn_nav_metric  # noqa: E402


@unittest.skipUnless(
    _ISAACLAB_AVAILABLE, "Isaac Lab / Isaac Sim runtime not available (Gate 1 blocked)"
)
class TestBarnScenarios(unittest.TestCase):
    """S1-S8: end-to-end env scenarios. Require Kit to run."""

    @classmethod
    def setUpClass(cls):
        # Boot Kit headless once for the whole test class.
        app_launcher = AppLauncher(headless=True)
        cls.simulation_app = app_launcher.app

    @classmethod
    def tearDownClass(cls):
        cls.simulation_app.close()

    def _make_env(self, num_envs: int = 4, world_idx: int = 0):
        from isaaclab.envs import DirectRLEnvCfg  # noqa: F401
        from my_robot_tasks.cfg.barn_nav_env_cfg import BarnNavEnvCfg
        from my_robot_tasks.envs.barn_nav_env import BarnNavEnv

        cfg = BarnNavEnvCfg()
        cfg.scene.num_envs = num_envs
        cfg.world_idx = world_idx
        # Shorten episodes for S3/S6/S7 without touching the shared default.
        cfg.episode_length_s = 5.0
        env = BarnNavEnv(cfg)
        return env, cfg

    # ------------------------------------------------------------------ #
    # S1: construction + space shapes
    # ------------------------------------------------------------------ #

    def test_s1_spaces(self):
        env, _ = self._make_env(num_envs=4)
        try:
            obs_space = env.observation_space
            act_space = env.action_space
            # Isaac Lab 3.0 batches the observation space to (num_envs, obs_dim).
            self.assertEqual(obs_space.shape, (4, 71))
            self.assertEqual(obs_space.shape[-1], 71)
            self.assertEqual(act_space.shape, (2,))
            # Action bounds are [-1, 1].
            low, high = act_space.low, act_space.high
            self.assertTrue(bool((low == -1.0).all()) and bool((high == 1.0).all()))
        finally:
            env.close()

    # ------------------------------------------------------------------ #
    # S2: reset places robot at start pose
    # ------------------------------------------------------------------ #

    def test_s2_reset_at_start(self):
        env, cfg = self._make_env(num_envs=2)
        try:
            obs, _ = env.reset()
            pos = env.robot.data.root_pos_w
            start = env._start_pos
            dist = torch.linalg.norm(pos[:, :2] - start.unsqueeze(0), dim=1)
            self.assertTrue(bool((dist < 0.15).all()), f"reset dist to start: {dist}")
            # Yaw near start_yaw (1.57 rad).
            yaw = env._get_yaw(env.robot.data.root_quat_w)
            self.assertTrue(bool((torch.abs(yaw - cfg.start_yaw) < 0.2).all()))
        finally:
            env.close()

    # ------------------------------------------------------------------ #
    # S3: velocity-driven step moves the robot
    # ------------------------------------------------------------------ #

    def test_s3_step_moves_robot(self):
        env, _ = self._make_env(num_envs=2)
        try:
            env.reset()
            pos_before = env.robot.data.root_pos_w[:, :2].clone()
            # Drive straight forward at max speed for several steps.
            act = torch.tensor([[1.0, 0.0], [1.0, 0.0]], device=env.device)
            for _ in range(10):
                env.step(act)
            pos_after = env.robot.data.root_pos_w[:, :2]
            moved = torch.linalg.norm(pos_after - pos_before, dim=1)
            self.assertTrue(bool((moved > 0.05).all()), f"robot did not move: {moved}")
        finally:
            env.close()

    # ------------------------------------------------------------------ #
    # S4: lidar observation 64 bins normalized
    # ------------------------------------------------------------------ #

    def test_s4_lidar_observation(self):
        env, _ = self._make_env(num_envs=2)
        try:
            obs, _ = env.reset()
            policy_obs = obs["policy"] if isinstance(obs, dict) else obs
            self.assertEqual(policy_obs.shape[-1], 71)
            lidar = policy_obs[:, :64]
            # Values normalized by max_distance (30 m): finite hits in [0,1], no-hit -> 1.0.
            self.assertTrue(bool((lidar >= 0.0).all()) and bool((lidar <= 1.0).all()))
        finally:
            env.close()

    # ------------------------------------------------------------------ #
    # S5: collision into cylinder -> terminated
    # ------------------------------------------------------------------ #

    def test_s5_collision_terminated(self):
        env, _ = self._make_env(num_envs=1, world_idx=0)
        try:
            env.reset()
            # Load world 0 cylinder centers and teleport the robot onto the first one.
            import numpy as np

            npz_path = _REPO_ROOT / "data" / "barn_static.npz"
            data = np.load(npz_path, allow_pickle=True)
            centers = data["centers"][0][: int(data["num_cylinders"][0])]
            cx, cy = float(centers[0, 0]), float(centers[0, 1])
            root_pose = env.robot.data.default_root_pose[0:1].clone()
            root_pose[0, 0] = cx
            root_pose[0, 1] = cy
            root_pose[0, 2] = 0.134
            env.robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=[0])
            env.robot.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros((1, 6), device=env.device), env_ids=[0]
            )
            # Step so the contact sensor registers the overlap.
            _, _, term, trunc, _ = env.step(torch.zeros((1, 2), device=env.device))
            self.assertTrue(bool(term[0]), "expected collision termination on cylinder")
        finally:
            env.close()

    # ------------------------------------------------------------------ #
    # S6: reaching the goal -> success + terminated
    # ------------------------------------------------------------------ #

    def test_s6_goal_reach_terminated(self):
        env, cfg = self._make_env(num_envs=1, world_idx=0)
        try:
            env.reset()
            goal = env._goal_pos
            # Teleport the robot directly on top of the goal.
            root_pose = env.robot.data.default_root_pose[0:1].clone()
            root_pose[0, 0] = goal[0]
            root_pose[0, 1] = goal[1]
            root_pose[0, 2] = 0.134
            env.robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=[0])
            env.robot.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros((1, 6), device=env.device), env_ids=[0]
            )
            # Step once so the done check observes the goal pose.
            _, _, term, trunc, _ = env.step(torch.zeros((1, 2), device=env.device))
            self.assertTrue(bool(term[0]), "expected success termination at goal")
            self.assertFalse(bool(trunc[0]))
        finally:
            env.close()

    # ------------------------------------------------------------------ #
    # S7: timeout -> truncated=True
    # ------------------------------------------------------------------ #

    def test_s7_timeout_truncated(self):
        env, cfg = self._make_env(num_envs=1, world_idx=0)
        try:
            env.reset()
            # Run with zero action until the episode length limit.
            truncated = False
            step_dt = cfg.sim.dt * cfg.decimation
            for _ in range(int(cfg.episode_length_s / step_dt) + 5):
                _, _, term, trunc, _ = env.step(torch.zeros((1, 2), device=env.device))
                if bool(term[0]) or bool(trunc[0]):
                    truncated = bool(trunc[0])
                    break
            self.assertTrue(truncated, "expected timeout truncation")
        finally:
            env.close()

    # ------------------------------------------------------------------ #
    # S8: nav metric matches barn/metric.py oracle
    # ------------------------------------------------------------------ #

    def test_s8_nav_metric_oracle(self):
        env, _ = self._make_env(num_envs=1, world_idx=0)
        try:
            optimal_time = float(env._optimal_time[0])
            # Oracle from barn/metric.py.
            self.assertAlmostEqual(
                env.compute_nav_metric(True, 2.0 * optimal_time), 0.5
            )
            self.assertAlmostEqual(
                env.compute_nav_metric(True, optimal_time), 0.5
            )
            self.assertAlmostEqual(
                env.compute_nav_metric(True, 5.0 * optimal_time), 0.2
            )
            self.assertAlmostEqual(
                env.compute_nav_metric(True, 8.0 * optimal_time), 0.125
            )
            self.assertEqual(env.compute_nav_metric(False, 1.0), 0.0)
            # Cross-check against the pure function directly.
            from metric import nav_metric

            ot = optimal_time
            self.assertAlmostEqual(
                env.compute_nav_metric(True, 3.0 * ot),
                float(nav_metric(True, 3.0 * ot, ot)),
            )
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
