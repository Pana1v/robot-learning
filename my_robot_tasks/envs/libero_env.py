# Copyright (c) 2026, Pana1v
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LIBERO-style manipulation environments for Isaac Lab.

Direct-workflow environments that run tabletop pick/place/transfer/push tasks
with a Franka Panda arm on the MuJoCo-Warp backend. Each episode records to
HDF5 in LeRobot-compatible structure for future VLA training on a remote
machine with 24+ GB VRAM.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import sample_uniform

if TYPE_CHECKING:
    from .cfg.libero_env_cfg import LiberoEnvCfg


class LiberoManipulationEnv(DirectRLEnv):
    """Tabletop manipulation environment for LIBERO-style tasks."""

    cfg: LiberoEnvCfg

    def __init__(self, cfg: LiberoEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.robot = self.scene["robot"]
        self.cube = self.scene["cube"]
        self.camera = self.scene["camera"]

        # Franka arm dof indices (first 7 joints are arm, last 2 are fingers)
        self._arm_dof_ids, _ = self.robot.find_joints(
            ["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
             "panda_joint5", "panda_joint6", "panda_joint7"]
        )
        self._gripper_dof_ids, _ = self.robot.find_joints(
            ["panda_finger_joint1", "panda_finger_joint2"]
        )
        self._arm_dof_ids = torch.tensor(self._arm_dof_ids, device=self.device)
        self._gripper_dof_ids = torch.tensor(self._gripper_dof_ids, device=self.device)

        # Default arm pose (home position)
        self._default_joint_pos = self.robot.data.default_joint_pos.clone()

        # Demo recording state
        self._episode_images: list[torch.Tensor] = []
        self._episode_states: list[torch.Tensor] = []
        self._episode_actions: list[torch.Tensor] = []
        self._episode_rewards: list[torch.Tensor] = []
        self._episode_dones: list[torch.Tensor] = []
        self._episode_index = 0

        self._hdf5_dir = Path(self.cfg.hdf5_output_dir)
        self._hdf5_dir.mkdir(parents=True, exist_ok=True)

        # Success tracking
        self._cube_grasped = False

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        # actions: (num_envs, 7) -> first 6 arm joints + gripper (open/close)
        arm_actions = self.actions[:, :6]
        gripper_action = self.actions[:, 6]

        # Arm: position targets relative to current positions
        current_arm_pos = self.robot.data.joint_pos[:, self._arm_dof_ids[:6]]
        new_arm_pos = current_arm_pos + arm_actions * 0.01  # small step size
        self.robot.set_joint_position_target(target=new_arm_pos, joint_ids=self._arm_dof_ids[:6])

        # Keep last arm joint fixed (simplify to 6-DoF control)
        self.robot.set_joint_position_target(
            target=self._default_joint_pos[:, self._arm_dof_ids[6:7]],
            joint_ids=self._arm_dof_ids[6:7],
        )

        # Gripper: binary open/close
        gripper_target = torch.where(gripper_action > 0, 0.04, 0.0)  # open or close
        self.robot.set_joint_position_target(target=gripper_target, joint_ids=self._gripper_dof_ids)

    def _get_observations(self) -> dict:
        joint_pos = self.robot.data.joint_pos
        joint_vel = self.robot.data.joint_vel

        arm_pos = joint_pos[:, self._arm_dof_ids]
        arm_vel = joint_vel[:, self._arm_dof_ids]

        cube_pos = self.cube.data.root_pos_w[:, :3]
        cube_pos = cube_pos - self.scene.env_origins  # relative to env origin

        # Gripper state (average of finger positions)
        gripper_pos = joint_pos[:, self._gripper_dof_ids].mean(dim=-1, keepdim=True)

        obs = torch.cat(
            (
                arm_pos,
                arm_vel,
                cube_pos,
                gripper_pos,
            ),
            dim=-1,
        )

        # Record for demos
        self._episode_images.append(self.camera.data.output["rgb"].clone())
        self._episode_states.append(obs.clone())
        self._episode_actions.append(self.actions.clone())

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        cube_pos = self.cube.data.root_pos_w[:, :3] - self.scene.env_origins
        ee_pos = self.robot.data.body_pos_w[:, self._ee_body_idx, :3] - self.scene.env_origins
        gripper_open = self.robot.data.joint_pos[:, self._gripper_dof_ids].mean(dim=-1)

        # Distance to cube
        dist_to_cube = torch.norm(ee_pos - cube_pos, dim=-1)

        # Cube lifted?
        cube_lifted = cube_pos[:, 2] > 0.1

        # Cube in bowl? (bowl at roughly y=0.1)
        cube_in_bowl = (torch.abs(cube_pos[:, 0] - 0.3) < 0.05) & \
                       (torch.abs(cube_pos[:, 1] - 0.1) < 0.05) & \
                       (cube_pos[:, 2] < 0.08)

        rewards = torch.zeros(self.num_envs, device=self.device)

        # Approach reward
        rewards += self.cfg.rew_scale_approach * (1.0 / (1.0 + dist_to_cube))

        # Grasp reward
        rewards += self.cfg.rew_scale_grasp * (~gripper_open.bool()).float() * (dist_to_cube < 0.05).float()

        # Lift reward
        rewards += self.cfg.rew_scale_lift * cube_lifted.float()

        # Place reward
        rewards += self.cfg.rew_scale_place * cube_in_bowl.float()

        # Action penalty
        rewards += self.cfg.rew_scale_action_penalty * torch.sum(torch.square(self.actions), dim=-1)

        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length

        # Task success depends on task_type
        cube_pos = self.cube.data.root_pos_w[:, :3] - self.scene.env_origins
        cube_in_bowl = (torch.abs(cube_pos[:, 0] - 0.3) < 0.05) & \
                       (torch.abs(cube_pos[:, 1] - 0.1) < 0.05) & \
                       (cube_pos[:, 2] < 0.08)
        cube_lifted = cube_pos[:, 2] > 0.1

        if self.cfg.task_type == "pick_cube":
            success = cube_lifted
        elif self.cfg.task_type == "place_cube":
            success = cube_in_bowl
        elif self.cfg.task_type == "transfer_cube":
            # cube moved to right side (y > 0.15)
            success = (cube_pos[:, 1] > 0.15) & cube_lifted
        elif self.cfg.task_type == "push_cube":
            success = cube_pos[:, 0] > 0.5
        else:
            success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        return success, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        # Flush completed episode
        if self._episode_images:
            self._save_episode()

        super()._reset_idx(env_ids)

        # Reset robot to home
        default_joint_pos = self._default_joint_pos[env_ids].clone()
        default_joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        self.robot.write_joint_state_to_sim(
            position=default_joint_pos, velocity=default_joint_vel, env_ids=env_ids
        )

        # Randomize cube position on table
        cube_pos_min = torch.tensor(self.cfg.cube_position_range[0], device=self.device)
        cube_pos_max = torch.tensor(self.cfg.cube_position_range[1], device=self.device)
        cube_pos = sample_uniform(cube_pos_min, cube_pos_max, (len(env_ids), 3), self.device)
        cube_pos[:, :3] += self.scene.env_origins[env_ids]
        cube_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).repeat(len(env_ids), 1)
        self.cube.write_root_pose_to_sim(
            root_pose=torch.cat([cube_pos, cube_quat], dim=-1), env_ids=env_ids
        )
        self.cube.write_root_velocity_to_sim(
            root_velocity=torch.zeros(len(env_ids), 6, device=self.device), env_ids=env_ids
        )

        self._cube_grasped = False

    def _save_episode(self):
        num_steps = len(self._episode_images)
        if num_steps == 0:
            return

        images = torch.cat(self._episode_images, dim=0).cpu().numpy()
        states = torch.cat(self._episode_states, dim=0).cpu().numpy()
        actions = torch.cat(self._episode_actions, dim=0).cpu().numpy()
        rewards = torch.cat(self._episode_rewards, dim=0).cpu().numpy()
        dones = torch.cat(self._episode_dones, dim=0).cpu().numpy()
        timestamps = np.arange(num_steps, dtype=np.float64) / 20.0

        episode_path = self._hdf5_dir / f"episode_{self._episode_index:06d}.hdf5"

        with h5py.File(episode_path, "w") as f:
            grp = f.create_group(f"episode_{self._episode_index:06d}")
            grp.create_dataset("observation.images.ego_view", data=images, compression="gzip", compression_opts=4)
            grp.create_dataset("observation.state", data=states, compression="gzip", compression_opts=4)
            grp.create_dataset("action", data=actions, compression="gzip", compression_opts=4)
            grp.create_dataset("reward", data=rewards, compression="gzip", compression_opts=4)
            grp.create_dataset("done", data=dones, compression="gzip", compression_opts=4)
            grp.create_dataset("timestamp", data=timestamps)
            grp.attrs["task_description"] = self.cfg.task_description
            grp.attrs["task_type"] = self.cfg.task_type
            grp.attrs["num_steps"] = num_steps
            grp.attrs["fps"] = 20.0

        print(f"[INFO] Saved LIBERO episode {self._episode_index} ({num_steps} steps) to {episode_path}")

        self._episode_images.clear()
        self._episode_states.clear()
        self._episode_actions.clear()
        self._episode_rewards.clear()
        self._episode_dones.clear()
        self._episode_index += 1

    @property
    def _ee_body_idx(self):
        """End-effector body index (panda_hand)."""
        body_ids, _ = self.robot.find_bodies(["panda_hand"])
        return body_ids[0]
