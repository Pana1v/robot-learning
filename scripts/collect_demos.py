# Copyright (c) 2026, Pana1v
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Demo collection script for the custom cartpole task.

Records HDF5 episodes (images, proprioception, actions, language command) in a
LeRobot-compatible structure. Run with:

    uv run --project IsaacLab python scripts/collect_demos.py --num_episodes 50 --output_dir logs/demos

The resulting dataset can be converted to LeRobot format with
IsaacLab/scripts/imitation_learning/locomanipulation_sdg/gr00t/convert_dataset.py
and then used to train a VLA (smol-vLA / OpenVLA-small) on a remote machine.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import gymnasium as gym
import isaaclab_tasks  # noqa: F401  registers isaaclab tasks
import my_robot_tasks  # noqa: F401  registers MyRobot-Cartpole-Direct
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Collect demos for VLA training.")
    parser.add_argument("--task", type=str, default="MyRobot-Cartpole-Direct", help="Gym task id.")
    parser.add_argument("--num_episodes", type=int, default=50, help="Number of episodes to record.")
    parser.add_argument("--output_dir", type=str, default="logs/demos", help="HDF5 output directory.")
    parser.add_argument("--max_steps", type=int, default=500, help="Max steps per episode.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create the env directly via gym so we can record every step.
    env = gym.make(args.task)
    obs, _ = env.reset(seed=args.seed)

    num_envs = env.num_envs
    device = env.device

    # Buffers for the current episode
    episode_images = []
    episode_states = []
    episode_actions = []
    episode_rewards = []
    episode_dones = []
    episode_infos = []

    episode_count = 0

    print(f"[INFO] Collecting {args.num_episodes} demos for task '{args.task}'")

    while episode_count < args.num_episodes:
        # Random policy for demo collection (replace with scripted/teleop policy as needed)
        action = torch.randn(num_envs, env.action_space, device=device) * 0.1

        obs, reward, terminated, truncated, info = env.step(action)

        # Record step data
        # Camera image: shape (num_envs, H, W, 3)
        if "camera" in env.scene:
            img = env.scene["camera"].data.output["rgb"]
        else:
            img = torch.zeros(num_envs, 240, 320, 3, device=device)

        episode_images.append(img.cpu())
        episode_states.append(obs["policy"].cpu())
        episode_actions.append(action.cpu())
        episode_rewards.append(reward.cpu())
        episode_dones.append((terminated | truncated).cpu())
        episode_infos.append(info)

        # Check for episode end
        if bool((terminated | truncated).any()):
            save_episode(
                output_dir,
                episode_count,
                episode_images,
                episode_states,
                episode_actions,
                episode_rewards,
                episode_dones,
                episode_infos,
                env.unwrapped.cfg.task_description,
            )
            episode_count += 1
            episode_images.clear()
            episode_states.clear()
            episode_actions.clear()
            episode_rewards.clear()
            episode_dones.clear()
            episode_infos.clear()
            env.reset()

    env.close()
    print(f"[INFO] Saved {episode_count} episodes to {output_dir}")


def save_episode(output_dir, episode_index, images, states, actions, rewards, dones, infos, task_description):
    """Write a single episode to HDF5 in LeRobot-compatible structure."""
    import h5py
    import numpy as np

    episode_path = output_dir / f"episode_{episode_index:06d}.hdf5"
    num_steps = len(images)

    # Concatenate buffers
    images_arr = torch.cat(images, dim=0).numpy()       # (T, H, W, 3)
    states_arr = torch.cat(states, dim=0).numpy()        # (T, state_dim)
    actions_arr = torch.cat(actions, dim=0).numpy()      # (T, action_dim)
    rewards_arr = torch.cat(rewards, dim=0).numpy()      # (T,)
    dones_arr = torch.cat(dones, dim=0).numpy()          # (T,)

    with h5py.File(episode_path, "w") as f:
        grp = f.create_group(f"episode_{episode_index:06d}")

        grp.create_dataset("observation.images.ego_view", data=images_arr, compression="gzip", compression_opts=4)
        grp.create_dataset("observation.state", data=states_arr, compression="gzip", compression_opts=4)
        grp.create_dataset("action", data=actions_arr, compression="gzip", compression_opts=4)
        grp.create_dataset("reward", data=rewards_arr, compression="gzip", compression_opts=4)
        grp.create_dataset("done", data=dones_arr, compression="gzip", compression_opts=4)
        grp.create_dataset("timestamp", data=np.arange(num_steps, dtype=np.float64) / 20.0)
        grp.attrs["task_description"] = task_description
        grp.attrs["num_steps"] = num_steps
        grp.attrs["fps"] = 20.0

    print(f"[INFO] Saved episode {episode_index} ({num_steps} steps) to {episode_path}")


if __name__ == "__main__":
    main()
