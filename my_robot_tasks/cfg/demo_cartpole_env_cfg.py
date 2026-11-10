# Copyright (c) 2026, Pana1v
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the demo-collection cartpole environment.

Records demonstrations (images + proprioception + actions + language command)
to HDF5 in LeRobot-compatible structure for future VLA training on a remote
machine with 24+ GB VRAM.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG


@configclass
class DemoCartpoleSceneCfg(InteractiveSceneCfg):
    """Cartpole scene with a camera for VLA observation collection."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    cartpole: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DistantLightCfg(intensity=2000.0),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.5, 0.5, 0.5, 0.5)),  # (x, y, z, w)
    )
    # Pinhole camera for VLA image observations. Placed to watch the cartpole.
    camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0.1,
        height=240,
        width=320,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(3.0, 0.0, 2.5), rot=(0.5, -0.5, 0.5, -0.5)),  # (x, y, z, w)
    )


@configclass
class DemoCartpoleEnvCfg(DirectRLEnvCfg):
    """Configuration for the demo-collection cartpole environment."""

    # env
    decimation = 2
    episode_length_s = 10.0  # longer episodes for demo trajectories
    action_scale = 100.0  # [N]
    action_space = 1
    observation_space = 5  # cart pos, cart vel, pole pos, pole vel, + padding
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    cart_dof_name = "slider_to_cart"
    pole_dof_name = "cart_to_pole"

    # scene
    scene: DemoCartpoleSceneCfg = DemoCartpoleSceneCfg(
        num_envs=1, env_spacing=4.0, replicate_physics=False, clone_in_fabric=False
    )

    # reset bounds
    max_cart_pos = 3.0  # [m]
    initial_cart_position_range = (-1.0, 1.0)  # [m]
    initial_cart_velocity_range = (-0.5, 0.5)  # [m/s]
    initial_pole_angle_range = (-0.25 * math.pi, 0.25 * math.pi)  # [rad]
    initial_pole_velocity_range = (-0.25 * math.pi, 0.25 * math.pi)  # [rad/s]

    # demo recording
    hdf5_output_dir = "logs/demos"  # directory for recorded HDF5 episodes
    task_description = "balance the cartpole"  # language command for VLA

    # reward scales (kept minimal for demo collection)
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pole_pos = -1.0

    def __post_init__(self):
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(8.0, 0.0, 5.0))
