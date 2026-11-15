# Copyright (c) 2026, Pana1v
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""BARN navigation environment for Isaac Lab.

Port of the ICRA BARN Navigation Challenge (ROS/Gazebo) to Isaac Lab for
training preparation. Drives a Clearpath Jackal from a start pose to a goal
pose in an obstacle-filled world, avoiding collisions.

Mirrors the original challenge:
* Static BARN worlds: indices 0-299
* Dynamic DynaBARN worlds: indices 300-359 (out of scope for Phase 0)
* Navigation metric: int(success) * optimal_time / clip(actual_time, 2*optimal_time, 8*optimal_time)

Design (D1-D9):
* D1: one static per-env obstacle mesh built from upstream world files via create_prim_from_mesh.
* D2: velocity-driven rigid body (RigidObjectCfg), no wheels/actuators.
* D3: MultiMeshRayCaster lidar, 720 beams @ 30 m, min-pooling downsample to 64 bins.
* D4: ContactSensorCfg filtered against per-env obstacle meshes.
* D5: 71-dim observation (64 lidar + 3 goal + 2 vel + 2 prev action).
* D6: 2-dim action [-1,1] scaled v=2.0·a0, ω=4.0·a1 with per-step rate limits.
* D7: (terminated, time_out) split; terminated = success | collision.
* D8: minimal rewards (progress, goal bonus, collision, time cost).
* D9: nav metric from barn/metric.py, pure torch.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

from isaaclab.envs import DirectRLEnv
from isaaclab.assets import RigidObject

from my_robot_tasks.barn.geometry import min_pool_ranges
from my_robot_tasks.barn.metric import nav_metric as barn_nav_metric
from my_robot_tasks.barn.worlds import (
    CYLINDER_LENGTH,
    NUM_STATIC_WORLDS,
    RADIUS,
    START_YAW,
)

if TYPE_CHECKING:
    from .cfg.barn_nav_env_cfg import BarnNavEnvCfg

# Candidate locations for the offline data table built by scripts/build_barn_data.py.
_BARN_STATIC_NPZ_CANDIDATES = (
    Path.cwd() / "data" / "barn_static.npz",
    Path("/home/panav/robot-learning/data/barn_static.npz"),
    Path("/home/panav/robot-learning/source/my_robot_tasks/data/barn_static.npz"),
)


def _load_barn_static_centers() -> dict[int, np.ndarray] | None:
    """Load per-world cylinder centers from barn_static.npz.

    Returns {world_idx: (N, 2) float32 centers} or None if the table is missing.
    """
    npz_path = None
    for candidate in _BARN_STATIC_NPZ_CANDIDATES:
        if candidate.exists():
            npz_path = candidate
            break
    if npz_path is None:
        return None
    data = np.load(npz_path, allow_pickle=True)
    centers_all = data["centers"]
    counts = data["num_cylinders"]
    return {int(i): centers_all[i][: int(counts[i])] for i in range(len(counts))}


class BarnNavEnv(DirectRLEnv):
    """BARN navigation environment driven by proprioceptive observations + laser scan."""

    cfg: BarnNavEnvCfg

    def __init__(self, cfg: BarnNavEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.robot: RigidObject = self.scene["robot"]
        self.scan = self.scene["scan"]
        self.contact = self.scene["contact"]

        # Cached robot state
        self._base_pos = self.robot.data.root_pos_w
        self._base_quat = self.robot.data.root_quat_w
        self._base_lin_vel = self.robot.data.root_lin_vel_w
        self._base_ang_vel = self.robot.data.root_ang_vel_w

        # Previous action for smoothness reward and rate limiting
        self._prev_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)

        # Rate-limiter state (previous commanded v, omega)
        self._prev_cmd_v = torch.zeros(self.num_envs, device=self.device)
        self._prev_cmd_omega = torch.zeros(self.num_envs, device=self.device)

        # Goal position (world frame, same for all envs)
        self._goal_pos = torch.tensor(
            self.cfg.goal_pos, device=self.device, dtype=torch.float32
        )
        self._start_pos = torch.tensor(
            self.cfg.start_pos, device=self.device, dtype=torch.float32
        )

        # Distance to goal at reset (for progress reward)
        self._initial_dist_to_goal = torch.zeros(self.num_envs, device=self.device)

        # Optimal time for metric computation (per world, loaded from data table)
        self._optimal_time = torch.zeros(self.num_envs, device=self.device)
        for env_id in range(self.num_envs):
            wid = int(self._world_of_env[env_id])
            pl = self.cfg.optimal_path_length
            if pl <= 0.0:
                pl = float(torch.linalg.norm(self._goal_pos - self._start_pos))
            self._optimal_time[env_id] = pl / 2.0

        # Navigation metric tracking (per-env, for extras logging)
        self._episode_start_step = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    # --------------------------------------------------------------------- #
    # Scene setup (D1): build per-env obstacle meshes before sim.reset()
    # so MultiMeshRayCaster finds {ENV_REGEX_NS}/Obstacles at sensor init.
    # --------------------------------------------------------------------- #

    def _setup_scene(self):
        """Isaac Lab calls this after scene instantiation but before sim.reset(),
        which is exactly when MultiMeshRayCaster needs the obstacle prims to exist."""
        self._world_of_env = torch.tensor(
            [(self.cfg.world_idx + int(i)) % NUM_STATIC_WORLDS for i in range(self.cfg.scene.num_envs)],
            device=self.device,
            dtype=torch.long,
        )
        self._create_obstacle_meshes()

    # --------------------------------------------------------------------- #
    # Action pipeline (D2, D6)
    # --------------------------------------------------------------------- #

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        """Convert body-frame (v, omega) to world-frame root velocity and write to sim.

        D2: velocity-driven rigid body. Zero lateral velocity, keep vertical velocity,
        zero roll/pitch rates. D6: rate-limit per policy step.
        """
        actions = self.actions

        # Raw command in [-1, 1] -> physical limits
        target_v = torch.clamp(actions[:, 0], -1.0, 1.0) * self.cfg.max_linear_vel
        target_omega = torch.clamp(actions[:, 1], -1.0, 1.0) * self.cfg.max_angular_vel

        # D6 rate limits: |Δv| ≤ rate_limit_v, |Δω| ≤ rate_limit_omega per policy step
        dv = target_v - self._prev_cmd_v
        dv = torch.clamp(dv, -self.cfg.rate_limit_v, self.cfg.rate_limit_v)
        v = self._prev_cmd_v + dv

        domega = target_omega - self._prev_cmd_omega
        domega = torch.clamp(domega, -self.cfg.rate_limit_omega, self.cfg.rate_limit_omega)
        omega = self._prev_cmd_omega + domega

        self._prev_cmd_v = v
        self._prev_cmd_omega = omega

        # Body-frame to world-frame velocity
        yaw = self._get_yaw(self._base_quat)
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        vx_world = cos_yaw * v  # lateral velocity is zero
        vy_world = sin_yaw * v
        vz_world = self._base_lin_vel[:, 2]  # keep simulated vertical velocity

        # Angular: zero roll/pitch rates, yaw rate = omega
        wx_world = torch.zeros_like(v)
        wy_world = torch.zeros_like(v)
        wz_world = omega

        root_vel = torch.stack(
            (vx_world, vy_world, vz_world, wx_world, wy_world, wz_world), dim=-1
        )

        self.robot.write_root_velocity_to_sim_index(root_velocity=root_vel, env_ids=None)

    # --------------------------------------------------------------------- #
    # D1: per-env obstacle meshes
    # --------------------------------------------------------------------- #

    def _create_obstacle_meshes(self) -> None:
        """Build one static obstacle mesh per env from its assigned world's cylinders.

        Each env's mesh is a single trimesh of all cylinders for that world, created
        at /World/envs/env_{i}/Obstacles via create_prim_from_mesh. The MultiMeshRayCaster
        casts against these meshes and the ContactSensor is filtered against them.
        """
        world_centers = _load_barn_static_centers()
        if world_centers is None:
            return

        # trimesh and create_prim_from_mesh are Isaac Lab / optional heavy deps;
        # import lazily so the module stays importable without Kit.
        import trimesh
        from isaaclab.terrains.utils import create_prim_from_mesh

        for env_id in range(self.cfg.scene.num_envs):
            wid = int(self._world_of_env[env_id])
            centers = world_centers[wid]  # (N, 2)

            cyl_meshes = []
            for cx, cy in centers:
                cyl = trimesh.creation.cylinder(radius=RADIUS, height=CYLINDER_LENGTH)
                cyl.apply_translation([float(cx), float(cy), CYLINDER_LENGTH / 2.0])
                cyl_meshes.append(cyl)
            mesh = trimesh.util.concatenate(cyl_meshes)

            prim_path = f"/World/envs/env_{env_id}/Obstacles"
            create_prim_from_mesh(prim_path, mesh)

    # --------------------------------------------------------------------- #
    # Observations (D5)
    # --------------------------------------------------------------------- #

    def _get_observations(self) -> dict:
        self._base_pos = self.robot.data.root_pos_w
        self._base_quat = self.robot.data.root_quat_w
        self._base_lin_vel = self.robot.data.root_lin_vel_w
        self._base_ang_vel = self.robot.data.root_ang_vel_w

        # Base linear velocity in robot frame (x, y)
        yaw = self._get_yaw(self._base_quat)
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        lin_vel_x = self._base_lin_vel[:, 0]
        lin_vel_y = self._base_lin_vel[:, 1]
        base_lin_vel_b_x = cos_yaw * lin_vel_x + sin_yaw * lin_vel_y
        base_lin_vel_b_y = -sin_yaw * lin_vel_x + cos_yaw * lin_vel_y

        # Goal relative position in robot frame: distance, sin(bearing), cos(bearing)
        goal_rel_x = self._goal_pos[0] - self._base_pos[:, 0]
        goal_rel_y = self._goal_pos[1] - self._base_pos[:, 1]
        goal_rel_dist = torch.sqrt(goal_rel_x**2 + goal_rel_y**2)
        goal_rel_angle = torch.atan2(goal_rel_y, goal_rel_x) - yaw
        goal_rel_angle = torch.atan2(torch.sin(goal_rel_angle), torch.cos(goal_rel_angle))

        # Lidar: min-pool 720 beams -> 64 bins, divided by max_distance.
        # RayCasterData.ray_hits_w holds inf for misses (no distance_to_hit field);
        # compute beam distances from ray_hits_w vs pos_w.
        scan_data = (
            self.scan.data.ray_hits_w - self.scan.data.pos_w.unsqueeze(1)
        ).norm(dim=-1)
        if scan_data is not None:
            scan_data = scan_data.reshape(self.num_envs, -1)
            # PLAN D3: rays that hit nothing return inf. Clamp to max_distance so the
            # normalized lidar stays in [0, 1] with no-hit bins reading exactly 1.0.
            scan_obs = min_pool_ranges(scan_data, 64)
            scan_obs = torch.clamp(scan_obs / self.cfg.scene.scan.max_distance, max=1.0)
        else:
            scan_obs = torch.ones((self.num_envs, 64), device=self.device)

        obs = torch.cat(
            (
                scan_obs,
                goal_rel_dist.unsqueeze(-1),
                torch.sin(goal_rel_angle).unsqueeze(-1),
                torch.cos(goal_rel_angle).unsqueeze(-1),
                base_lin_vel_b_x.unsqueeze(-1),
                base_lin_vel_b_y.unsqueeze(-1),
                self._prev_actions,
            ),
            dim=-1,
        )

        return {"policy": obs}

    # --------------------------------------------------------------------- #
    # Rewards (D8: minimal)
    # --------------------------------------------------------------------- #

    def _get_rewards(self) -> torch.Tensor:
        curr_dist = torch.linalg.norm(
            self._goal_pos.unsqueeze(0) - self._base_pos[:, :2], dim=1
        )
        progress = self._initial_dist_to_goal - curr_dist

        goal_reward = (curr_dist < self.cfg.goal_threshold).float()

        # Collision from contact sensor (filtered against obstacles only)
        contact_data = self.contact.data
        force_matrix = contact_data.force_matrix_w
        if force_matrix is not None:
            collision = force_matrix.abs().sum(dim=-1).sum(dim=-1) > 0.0
        else:
            collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Time cost (small per-step penalty)
        time_cost = torch.full((self.num_envs,), self.cfg.rew_scale_time, device=self.device)

        total_reward = (
            self.cfg.rew_scale_progress * progress
            + self.cfg.rew_scale_goal * goal_reward
            + self.cfg.rew_scale_collision * collision.float()
            + time_cost
        )

        return total_reward

    # --------------------------------------------------------------------- #
    # Terminations (D7)
    # --------------------------------------------------------------------- #

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        curr_dist = torch.linalg.norm(
            self._goal_pos.unsqueeze(0) - self._base_pos[:, :2], dim=1
        )

        # Success: within goal threshold
        success = curr_dist < self.cfg.goal_threshold

        # Collision: contact sensor reports filtered force against obstacles
        contact_data = self.contact.data
        force_matrix = contact_data.force_matrix_w
        if force_matrix is not None:
            collision = force_matrix.abs().sum(dim=-1).sum(dim=-1) > 0.0
        else:
            collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Timeout
        time_out = self.episode_length_buf >= self.max_episode_length

        terminated = collision | success
        return terminated, time_out

    # --------------------------------------------------------------------- #
    # Resets (D7)
    # --------------------------------------------------------------------- #

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        super()._reset_idx(env_ids)

        num_resets = len(env_ids)

        # --- Place robot at start pose (D7) ---
        start_x = self.cfg.start_pos[0]
        start_y = self.cfg.start_pos[1]
        start_yaw = self.cfg.start_yaw

        # Small random offset for generalization
        offset_x = torch.empty(num_resets, device=self.device).uniform_(-0.1, 0.1)
        offset_y = torch.empty(num_resets, device=self.device).uniform_(-0.1, 0.1)
        offset_yaw = torch.empty(num_resets, device=self.device).uniform_(-0.1, 0.1)

        # default_root_pose is a warp wp.array (dtype wp.transformf). Indexing it
        # ([env_ids]) returns a ProxyArray view that the PhysX kernel
        # set_root_link_pose_to_sim rejects (wp.launch refuses ProxyArrays).
        # wp.array(torch_tensor, dtype=wp.transformf) also returns a ProxyArray in
        # this warp build. Isaac Lab builds its default_root_pose from a numpy array
        # (rigid_object.py:1067-1069), which yields an owning wp.transformf array.
        # Build the pose as a float32 numpy array (7 cols: px, py, pz, qx, qy, qz, qw).
        root_pose_data = np.zeros((num_resets, 7), dtype=np.float32)
        root_pose_data[:, 0] = float(start_x) + offset_x.cpu().numpy()
        root_pose_data[:, 1] = float(start_y) + offset_y.cpu().numpy()
        root_pose_data[:, 2] = 0.134  # chassis height
        root_pose_data[:, 3:7] = self._quat_from_yaw(start_yaw + offset_yaw).cpu().numpy()
        root_pose = wp.array(root_pose_data, dtype=wp.transformf, device=self.device)

        root_vel = torch.zeros((num_resets, 6), device=self.device)
        self.robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=root_vel, env_ids=env_ids)

        # Reset rate-limiter state and previous actions (D7 trap: stale values)
        self._prev_actions[env_ids] = 0.0
        self._prev_cmd_v[env_ids] = 0.0
        self._prev_cmd_omega[env_ids] = 0.0

        # Track episode start step for metric timing
        self._episode_start_step[env_ids] = self.episode_length_buf[env_ids]

        # Update initial distance to goal
        curr_pos = self._base_pos[env_ids, :2]
        self._initial_dist_to_goal[env_ids] = torch.linalg.norm(
            self._goal_pos.unsqueeze(0) - curr_pos, dim=1
        )

    # --------------------------------------------------------------------- #
    # Navigation metric (D9)
    # --------------------------------------------------------------------- #

    def compute_nav_metric(self, success: bool, actual_time: float, env_id: int = 0) -> float:
        """Compute the BARN navigation metric for a single env.

        nav_metric = int(success) * optimal_time / clip(actual_time, 2*optimal_time, 8*optimal_time)
        """
        optimal_time = self._optimal_time[env_id].item() if torch.is_tensor(self._optimal_time) else self._optimal_time
        return barn_nav_metric(success, actual_time, optimal_time).item()

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #

    def _get_yaw(self, quat: torch.Tensor) -> torch.Tensor:
        """Extract yaw from quaternion (x, y, z, w)."""
        x = quat[:, 0]
        y = quat[:, 1]
        z = quat[:, 2]
        w = quat[:, 3]
        return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _quat_from_yaw(self, yaw: torch.Tensor) -> torch.Tensor:
        """Create quaternion (x, y, z, w) from yaw angle."""
        half = yaw / 2.0
        return torch.stack(
            (
                torch.zeros_like(half),
                torch.zeros_like(half),
                torch.sin(half),
                torch.cos(half),
            ),
            dim=-1,
        )
