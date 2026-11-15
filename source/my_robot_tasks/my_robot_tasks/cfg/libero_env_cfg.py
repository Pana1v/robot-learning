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
from isaaclab_tasks.core.lift.lift_env_cfg import PhysicsCfg as FrankaLiftPhysicsCfg
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg


@configclass
class LiberoSceneCfg(InteractiveSceneCfg):
    """Tabletop manipulation scene: Franka arm, cube, bowl, camera."""

    table = AssetBaseCfg(
        prim_path="/World/table",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 0.8, 0.05),
            # Static collider, no rigid body: it never moves, and Newton's shared /World cannot hold bodies.
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.2)),
        ),
        # Top surface at z=0, so the robot base, cube (z=0.025) and bowl rest on it instead of inside it.
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.025)),
    )
    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            # Without a collider and mass the cube is not a physics body: Newton cannot find it and
            # PhysX lets it fall through the table.
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.7, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, -0.1, 0.025)),
    )
    bowl = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Bowl",
        spawn=sim_utils.CylinderCfg(
            radius=0.04, height=0.06,
            # Static collider, no rigid body: it never moves, and Newton's shared /World cannot hold bodies.
            collision_props=sim_utils.CollisionPropertiesCfg(),
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
        # Front view looking back at the workspace: from (1.1, 0, 0.7) toward (0.3, 0, 0.05), ROS convention.
        # The old pose looked along +X, away from the robot, and rendered only background.
        offset=CameraCfg.OffsetCfg(pos=(1.1, 0.0, 0.7), rot=(-0.6385, -0.6385, 0.3039, 0.3039)),  # (x, y, z, w)
        # Selectable from the CLI with renderer=newton_renderer|isaacsim_rtx|ovrtx.
        # Default is the Newton Warp ray tracer: no Kit/RTX needed, much lighter on 8 GB VRAM.
        renderer_cfg=MultiBackendRendererCfg(),
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
    # Upstream Franka-lift presets (same arm + cube contact problem). Gives physics=newton_mjwarp (default),
    # physx and ovphysx; without a preset the Makefile's physics= selector is rejected.
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation, physics=FrankaLiftPhysicsCfg())

    # Training default. Every env is identical, so let the cloner replicate physics instead of
    # parsing each env separately. Demo recording forces num_envs=1 (see record_demos).
    scene: LiberoSceneCfg = LiberoSceneCfg(num_envs=1024, env_spacing=2.0)

    # task overrides (set by task-specific configs)
    task_type: str = "pick_cube"
    task_description: str = "manipulate the object"

    # demo recording. Off for RL training: the policy only sees the 18-dim state, so rendering the
    # camera and gzip-writing HDF5 on every reset is pure overhead. When False the camera is removed
    # from the scene entirely. Turn on with: env.record_demos=true env.scene.num_envs=1
    record_demos: bool = False
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
