#!/usr/bin/env python3
"""Construct BarnNavEnv, reset, drive forward, print diagnostics.

Boot Kit before env imports: isaaclab_physx imports carb at module load.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Smoke test for BarnNavEnv.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--steps", type=int, default=300, help="Steps to run.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Boot Kit before env imports: isaaclab_physx imports carb at module load
simulation_app = AppLauncher(args).app

import torch  # noqa: E402

from my_robot_tasks.cfg.barn_nav_env_cfg import BarnNavEnvCfg  # noqa: E402
from my_robot_tasks.envs.barn_nav_env import BarnNavEnv  # noqa: E402

ROBOT_KEY, LIDAR_KEY = "robot", "scan"  # must match the scene cfg

cfg = BarnNavEnvCfg()
cfg.scene.num_envs = args.num_envs
print(f"[smoke] constructing BarnNavEnv with num_envs={args.num_envs}")
env = BarnNavEnv(cfg=cfg)
print("[smoke] BarnNavEnv.__init__ reached")

obs, _ = env.reset(seed=0)
print(f"[smoke] obs dim {obs['policy'].shape[-1]} | cfg.observation_space {cfg.observation_space}")

robot, lidar = env.scene[ROBOT_KEY], env.scene[LIDAR_KEY]
start_xy = robot.data.root_pos_w[:, :2].clone()

actions = torch.zeros(env.num_envs, 2, device=env.device)
actions[:, 0] = 0.25  # 0.5 m/s once D6 scaling lands

for i in range(args.steps):
    obs, rew, term, trunc, _ = env.step(actions)
    if i % 30 == 0:
        # misses are inf, so inf here means the lidar saw nothing
        hit = (lidar.data.ray_hits_w - lidar.data.pos_w.unsqueeze(1)).norm(dim=-1)
        moved = (robot.data.root_pos_w[:, :2] - start_xy).norm(dim=-1)
        print(
            f"[smoke] step {i:3d} moved {moved.mean():.2f} m | min_hit {hit.min():.2f} | "
            f"rew {rew.mean():.3f} | term {int(term.sum())} | trunc {int(trunc.sum())}"
        )

env.close()
simulation_app.close()
print("[smoke] done")
