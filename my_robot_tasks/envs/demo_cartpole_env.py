# Copyright (c) 2026, Pana1v
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Demo-collection cartpole environment.

Records each episode to HDF5 (camera images, proprioception, actions, language
command) in a LeRobot-compatible structure. Intended for future VLA training on
a remote machine with 24+ GB VRAM. Local RL training with rsl_rl is also
supported and runs well on the RTX 5050.
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
    from .cfg.demo_cartpole_env_cfg import DemoCartpoleEnvCfg


class DemoCartpoleEnv(DirectRLEnv):
    """Cartpole balancing environment that records demonstrations to HDF5."""

    cfg: DemoCartpoleEnvCfg

    def __init__(self, cfg: DemoCartpoleEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.cartpole = self.scene["cartpole"]
        self.camera = self.scene["camera"]
        self.action_scale = self.cfg.action_scale

        # DOF indices
        self._cart_dof_idx, _ = self.cartpole.find_joints(self.cfg.cart_dof_name)
        self._pole_dof_idx, _ = self.cartpole.find_joints(self.cfg.pole_dof_name)
        self._cart_dof_idx = torch.tensor(self._cart_dof_idx, device=self.device)
        self._pole_dof_idx = torch.tensor(self._pole_dof_idx, device=self.device)

        # Demo recording state
        self._recording = True  # always record in this env
        self._episode_images: list[torch.Tensor] = []
        self._episode_states: list[torch.Tensor] = []
        self._episode_actions: list[torch.Tensor] = []
        self._episode_rewards: list[torch.Tensor] = []
        self._episode_dones: list[torch.Tensor] = []
        self._episode_infos: list[dict] = []
        self._episode_index = 0

        self._hdf5_dir = Path(self.cfg.hdf5_output_dir)
        self._hdf5_dir.mkdir(parents=True, exist_ok=True)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions

    def _apply_action(self) -> None:
        self.cartpole.set_joint_effort_target_index(target=self.actions, joint_ids=self._cart_dof_idx)

    def _get_observations(self) -> dict:
        joint_pos = self.cartpole.data.joint_pos
        joint_vel = self.cartpole.data.joint_vel

        obs = torch.cat(
            (
                joint_pos[:, self._cart_dof_idx],
                joint_vel[:, self._cart_dof_idx],
                joint_pos[:, self._pole_dof_idx],
                joint_vel[:, self._pole_dof_idx],
            ),
            dim=-1,
        )

        if self._recording:
            self._episode_images.append(self.camera.data.output["rgb"].clone())
            self._episode_states.append(obs.clone())
            self._episode_actions.append(self.actions.clone())

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        rewards = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_pole_pos,
            self.joint_pos[:, self._pole_dof_idx],
            self.reset_terminated,
            self.step_dt,
        )

        if self._recording:
            self._episode_rewards.append(rewards.clone())

        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.cartpole.data.joint_pos
        self.joint_vel = self.cartpole.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length
        out_of_bounds = torch.any(
            torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1
        )

        if self._recording:
            self._episode_dones.append((out_of_bounds | time_out).clone())
            self._episode_infos.append({})

        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.cartpole._ALL_INDICES

        # Flush the completed episode to HDF5 before resetting
        if self._recording and self._episode_images:
            self._save_episode()

        super()._reset_idx(env_ids)

        joint_pos = self.cartpole.data.default_joint_pos.torch[env_ids].clone()
        joint_pos[:, self._cart_dof_idx] += sample_uniform(
            self.cfg.initial_cart_position_range[0],
            self.cfg.initial_cart_position_range[1],
            joint_pos[:, self._cart_dof_idx].shape,
            joint_pos.device,
        )
        joint_vel = self.cartpole.data.default_joint_vel.torch[env_ids].clone()
        joint_vel[:, self._cart_dof_idx] += sample_uniform(
            self.cfg.initial_cart_velocity_range[0],
            self.cfg.initial_cart_velocity_range[1],
            joint_vel[:, self._cart_dof_idx].shape,
            joint_vel.device,
        )
        joint_pos[:, self._pole_dof_idx] += sample_uniform(
            self.cfg.initial_pole_angle_range[0],
            self.cfg.initial_pole_angle_range[1],
            joint_pos[:, self._pole_dof_idx].shape,
            joint_pos.device,
        )
        joint_vel[:, self._pole_dof_idx] += sample_uniform(
            self.cfg.initial_pole_velocity_range[0],
            self.cfg.initial_pole_velocity_range[1],
            joint_vel[:, self._pole_dof_idx].shape,
            joint_vel.device,
        )

        # Clamp to joint limits
        joint_pos_limits = self.cartpole.data.soft_joint_pos_limits.torch[env_ids]
        joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
        joint_vel_limits = self.cartpole.data.soft_joint_vel_limits.torch[env_ids]
        joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

        default_root_pose = self.cartpole.data.default_root_pose.torch[env_ids].clone()
        default_root_pose[:, :3] += self.scene.env_origins[env_ids]
        default_root_vel = self.cartpole.data.default_root_vel.torch[env_ids].clone()

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.cartpole.write_root_pose_to_sim_index(root_pose=default_root_pose, env_ids=env_ids)
        self.cartpole.write_root_velocity_to_sim_index(root_velocity=default_root_vel, env_ids=env_ids)
        self.cartpole.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.cartpole.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

    def _save_episode(self):
        """Write the buffered episode to HDF5 in LeRobot-compatible structure."""
        num_steps = len(self._episode_images)
        if num_steps == 0:
            return

        images = torch.cat(self._episode_images, dim=0).cpu().numpy()       # (T, H, W, 3)
        states = torch.cat(self._episode_states, dim=0).cpu().numpy()      # (T, state_dim)
        actions = torch.cat(self._episode_actions, dim=0).cpu().numpy()    # (T, action_dim)
        rewards = torch.cat(self._episode_rewards, dim=0).cpu().numpy()    # (T,)
        dones = torch.cat(self._episode_dones, dim=0).cpu().numpy()        # (T,)
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
            grp.attrs["num_steps"] = num_steps
            grp.attrs["fps"] = 20.0

        print(f"[INFO] Saved episode {self._episode_index} ({num_steps} steps) to {episode_path}")

        # Clear buffers and increment index
        self._episode_images.clear()
        self._episode_states.clear()
        self._episode_actions.clear()
        self._episode_rewards.clear()
        self._episode_dones.clear()
        self._episode_infos.clear()
        self._episode_index += 1


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_pole_pos: float,
    pole_pos: torch.Tensor,
    reset_terminated: torch.Tensor,
    step_dt: float,
) -> torch.Tensor:
    from isaaclab.utils.math import wrap_to_pi

    pole_pos = wrap_to_pi(pole_pos)
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos), dim=-1)
    total_reward = (rew_alive + rew_termination + rew_pole_pos) * step_dt
    return total_reward
