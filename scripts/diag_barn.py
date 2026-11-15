#!/usr/bin/env python3
"""Diagnostics for BarnNavEnv: confirm the open failure modes empirically.

Boots Kit headless, then reports:
  D1: robot spawn pose vs the BARN start pose
  D2: contact sensor filtered-force availability (collision detection)
  D3: lidar vs the pure-python analytic oracle
  D4: root pose write path (teleport) used by the sim tests
  D5: per-world optimal path length available in the npz
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Diagnostics for BarnNavEnv.")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

simulation_app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from my_robot_tasks.cfg.barn_nav_env_cfg import BarnNavEnvCfg  # noqa: E402
from my_robot_tasks.envs.barn_nav_env import BarnNavEnv  # noqa: E402

cfg = BarnNavEnvCfg()
cfg.scene.num_envs = args.num_envs
env = BarnNavEnv(cfg=cfg)
obs, _ = env.reset(seed=0)

robot = env.scene["robot"]
scan = env.scene["scan"]
contact = env.scene["contact"]

print("=" * 70)
print("D1: spawn pose")
print(f"  cfg.start_pos        = {tuple(cfg.start_pos)}")
print(f"  cfg.start_yaw        = {cfg.start_yaw}")
print(f"  cfg.goal_pos         = {tuple(cfg.goal_pos)}")
print(f"  robot root_pos_w     = {robot.data.root_pos_w[0].tolist()}")
yaw = env._get_yaw(robot.data.root_quat_w)
print(f"  robot yaw            = {yaw[0].item():.4f}")
d = torch.linalg.norm(robot.data.root_pos_w[:, :2] - env._start_pos.unsqueeze(0), dim=1)
print(f"  dist to start_pos    = {d.tolist()}")

print("=" * 70)
print("D2: contact sensor")
fm = contact.data.force_matrix_w
print(f"  force_matrix_w       = {None if fm is None else tuple(fm.shape)}")
nf = contact.data.net_forces_w
print(f"  net_forces_w         = {None if nf is None else tuple(nf.shape)}")
print(f"  filter_prim_paths_expr = {cfg.scene.contact.filter_prim_paths_expr}")
if nf is not None:
    print(f"  net_forces_w[0]      = {nf[0].tolist()}")

print("=" * 70)
print("D3: lidar vs analytic oracle")
from my_robot_tasks.barn.geometry import analytic_lidar  # noqa: E402
from my_robot_tasks.barn.worlds import RADIUS  # noqa: E402

npz = np.load("/home/panav/isaaclab-barn/data/barn_static.npz", allow_pickle=True)
centers = npz["centers"][0][: int(npz["num_cylinders"][0])]
print(f"  world 0 cylinders    = {len(centers)}")
print(f"  npz path_lengths[0]  = {npz['path_lengths'][0]:.4f} m")
print(f"  npz optimal_times[0] = {npz['optimal_times'][0]:.4f} s")

sensor_pos = scan.data.pos_w[0].cpu().numpy()
heading = float(yaw[0].item())
beam_angles = torch.linspace(
    np.deg2rad(-135.0), np.deg2rad(135.0), 720, dtype=torch.float32
)
oracle = (
    analytic_lidar(
        origins=torch.tensor(sensor_pos[:2], dtype=torch.float32).unsqueeze(0),
        headings=torch.tensor([heading], dtype=torch.float32),
        beam_angles=beam_angles,
        cylinder_centers=torch.tensor(centers, dtype=torch.float32),
        radius=RADIUS,
        max_distance=30.0,
    )
    .cpu()
    .numpy()[0]
)

ray_hits = scan.data.ray_hits_w[0].cpu().numpy()
sim_d = np.linalg.norm(ray_hits - sensor_pos, axis=-1)
sim_hit = np.isfinite(sim_d)
# analytic_lidar clamps misses to max_distance, so a miss is oracle == 30.0.
oracle_hit = oracle < 30.0
print(f"  sensor pos           = {sensor_pos.tolist()}")
print(f"  heading              = {heading:.4f} rad")
print(f"  sim hit rays         = {int(sim_hit.sum())}/720")
print(f"  oracle hit rays      = {int(oracle_hit.sum())}/720")
print(f"  hit/miss mismatch    = {int((sim_hit != oracle_hit).sum())} rays")
both = sim_hit & oracle_hit
if both.any():
    diff = np.abs(sim_d[both] - oracle[both])
    print(f"  |sim - oracle| max   = {diff.max():.4f} m  mean = {diff.mean():.4f} m")
else:
    print("  no overlapping hit rays to compare")

print("=" * 70)
print("D4: root pose write path")
target = env.robot.data.default_root_pose[0:1].clone()
target[0, 0] = float(cfg.start_pos[0])
target[0, 1] = float(cfg.start_pos[1])
target[0, 2] = 0.134
try:
    env.robot.write_root_pose_to_sim_index(root_pose=target, env_ids=[0])
    print("  write_root_pose_to_sim_index: OK")
except Exception as e:
    print(f"  write_root_pose_to_sim_index: FAILED -> {type(e).__name__}: {e}")
try:
    env.robot.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros((1, 6), device=env.device), env_ids=[0]
    )
    print("  write_root_velocity_to_sim_index: OK")
except Exception as e:
    print(f"  write_root_velocity_to_sim_index: FAILED -> {type(e).__name__}: {e}")

print("=" * 70)
print("D5: metric inputs")
print(f"  env._optimal_time[0]  = {env._optimal_time[0].item():.4f} s")
print(f"  cfg.optimal_path_length = {cfg.optimal_path_length}")
print(f"  npz optimal_times[0]  = {npz['optimal_times'][0]:.4f} s")

env.close()
simulation_app.close()
print("=" * 70)
print("diag done")
