# Copyright (c) 2026, Pana1v
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for LIBERO-style manipulation environments in Isaac Lab."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG


@configclass
class LiberoSceneCfg(InteractiveSceneCfg):
    """Tabletop manipulation scene: Franka arm, cube, bowl, camera."""

    table = AssetBaseCfg(
        prim_path="/World/table",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 0.8, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.2)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.7, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, -0.1, 0.025)),
    )
    bowl = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Bowl",
        spawn=sim_utils.CylinderCfg(
            radius=0.04, height=0.06,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.3, 0.3)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.3, 0.1, 0.03)),
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DistantLightCfg(intensity=3000.0),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.5, 0.5, 0.5, 0.5)),  # (x, y, z, w)
    )
    camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0.1,
        height=240,
        width=320,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.6, 0.0, 0.6), rot=(0.5, -0.5, 0.5, -0.5)),  # (x, y, z, w)
    )


@configclass
class LiberoEnvCfg(DirectRLEnvCfg):
    """Base configuration for LIBERO-style manipulation tasks."""

    # env
    decimation = 2
    episode_length_s = 10.0
    action_space = 7  # Franka arm joints
    observation_space = 18  # 7 joint pos + 7 joint vel + 3 cube pos + 1 gripper state

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    scene: LiberoSceneCfg = LiberoSceneCfg(
        num_envs=1, env_spacing=2.0, replicate_physics=False, clone_in_fabric=False
    )

    # task overrides (set by task-specific configs)
    task_type: str = "pick_cube"
    task_description: str = "manipulate the object"

    # demo recording
    hdf5_output_dir = "logs/demos/libero"

    # reset bounds (tabletop workspace)
    cube_position_range = ((0.2, -0.2, 0.025), (0.5, 0.2, 0.025))  # (min, max) in (x, y, z)

    # reward scales
    rew_scale_approach = 1.0
    rew_scale_grasp = 2.0
    rew_scale_lift = 5.0
    rew_scale_place = 10.0
    rew_scale_action_penalty = -0.001

    def __post_init__(self):
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(1.5, 0.0, 1.2))


@configclass
class LiberoPickCubeEnvCfg(LiberoEnvCfg):
    """Configuration for the pick-cube LIBERO-style task."""

    task_type: str = "pick_cube"
    task_description: str = "pick up the cube"


@configclass
class LiberoPlaceCubeEnvCfg(LiberoEnvCfg):
    """Configuration for the place-cube LIBERO-style task."""

    task_type: str = "place_cube"
    task_description: str = "place the cube in the bowl"


@configclass
class LiberoTransferCubeEnvCfg(LiberoEnvCfg):
    """Configuration for the transfer-cube LIBERO-style task."""

    task_type: str = "transfer_cube"
    task_description: str = "transfer the cube from left to right"


@configclass
class LiberoPushCubeEnvCfg(LiberoEnvCfg):
    """Configuration for the push-cube LIBERO-style task."""

    task_type: str = "push_cube"
    task_description: str = "push the cube to the target"
