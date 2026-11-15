# Copyright (c) 2026, Pana1v
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the BARN navigation environment in Isaac Lab.

Port of the ICRA BARN Navigation Challenge (ROS/Gazebo) to Isaac Lab for
training preparation. The task: drive a Clearpath Jackal from a start pose
to a goal pose in an obstacle-filled world, avoiding collisions.

World layout (mirrors the original challenge):
* Static BARN worlds: indices 0-299, start (-2.25, 3.0), yaw 1.57,
  goal offset (0, 10) relative -> absolute goal (-2.25, 13.0)
* Dynamic DynaBARN worlds: indices 300-359 (out of scope for Phase 0)

Navigation metric (from the original challenge):
    nav_metric = int(success) * optimal_time / clip(actual_time, 2*optimal_time, 8*optimal_time)
    where optimal_time = path_length / 2.0
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, MultiMeshRayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg

from my_robot_tasks.cfg.jackal_cfg import JACKAL_CFG, JACKAL_ROOT_Z


##
# Observation / action space constants (D5, D6). Computed from named
# constants so a hard-coded literal cannot drift from the implementation (H2).
##

LIDAR_BINS = 64  # min-pooled lidar bins
LIDAR_MAX_DISTANCE = 30.0  # [m] BARN sensor max range
GOAL_DIMS = 3  # distance, sin(bearing), cos(bearing)
VEL_DIMS = 2  # body v, omega
PREV_ACTION_DIMS = 2  # previous action
OBSERVATION_SPACE = LIDAR_BINS + GOAL_DIMS + VEL_DIMS + PREV_ACTION_DIMS  # 71
ACTION_SPACE = 2  # [linear_vel, angular_vel]

# D6 rate limits at dt = decimation / sim_dt = 4/120 = 1/30 s:
# |Δv| ≤ 20 m/s² × 1/30 s ≈ 0.667 m/s, |Δω| ≤ 25 rad/s² × 1/30 s ≈ 0.833 rad/s
RATE_LIMIT_V = 0.667  # [m/s] per policy step
RATE_LIMIT_OMEGA = 0.833  # [rad/s] per policy step


##
# Scene configuration
##

@configclass
class BarnNavSceneCfg(InteractiveSceneCfg):
    """Scene for BARN navigation: ground plane, Jackal robot, contact + lidar sensors.

    Obstacle meshes are built per-env in BarnNavEnv.__init__ (D1): one static
    collider mesh per env at /World/envs/env_{i}/Obstacles, created from the
    upstream world's cylinder centers via create_prim_from_mesh. No
    RigidObjectCollectionCfg — the robot cannot shove them and they cost
    nothing to simulate.
    """

    # Plain black floor. GroundPlaneCfg only exposes a `color` tint on the
    # theGrid shader's diffuse_tint, which leaves the grid's specular and
    # roughness pattern visible. A dedicated static-collider cuboid with a
    # black PreviewSurface material renders as a flat black floor instead.
    # No rigid_props -> static collider; the top face sits at z = 0.
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.CuboidCfg(
            size=(60.0, 60.0, 0.05),  # [m]
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.0, 0.0),
                roughness=0.9,
                metallic=0.0,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.025)),
    )

    robot: RigidObjectCfg = JACKAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DistantLightCfg(intensity=3000.0),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.5, 0.5, 0.5, 0.5)),
    )

    # Contact sensor on the robot body, filtered against per-env obstacle meshes.
    # Reads force_matrix_w (filtered), never net_forces_w (ground support fires every step).
    # create_prim_from_mesh authors the Xform at {prim_path} and the mesh+collider at
    # {prim_path}/mesh, plus a visualMaterial sibling. The filter must name exactly the
    # collider prim ({prim_path}/mesh); a wildcard like {prim_path}/* matches the
    # material sibling too (2 entries), which PhysX rejects (expects 1 collider per
    # filter pattern) and leaves force_matrix_w as None, silently disabling collision.
    contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        update_period=0.02,
        history_length=0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Obstacles/mesh"],
        debug_vis=False,
    )

    # Lidar: 720 beams from -135° to +135°, max_distance 30 m, cast against
    # per-env obstacle meshes via MultiMeshRayCaster.
    scan = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        update_period=0.02,
        max_distance=LIDAR_MAX_DISTANCE,
        mesh_prim_paths=[
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/Obstacles",
                is_shared=False,
                merge_prim_meshes=True,
                track_mesh_transforms=False,  # static meshes, no transform tracking needed
            )
        ],
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(-135.0, 135.0),
            horizontal_res=270.0 / 719.0,  # 720 beams over 270 deg
        ),
        debug_vis=False,
    )

    def __post_init__(self):
        self.num_envs = 1
        self.env_spacing = 10.0
        self.replicate_physics = False
        self.clone_in_fabric = False


##
# Environment configuration
##

@configclass
class BarnNavEnvCfg(DirectRLEnvCfg):
    """Configuration for the BARN navigation environment."""

    # env
    decimation = 4
    episode_length_s = 100.0  # matches the original 100s timeout
    action_space = ACTION_SPACE  # [linear_vel, angular_vel] in [-1, 1]
    observation_space = OBSERVATION_SPACE  # 71: 64 lidar + 3 goal + 2 vel + 2 prev action
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    scene: BarnNavSceneCfg = BarnNavSceneCfg()

    # Jackal parameters (D2: wheel_base removed — velocity-driven body, no wheels)
    wheel_radius = 0.098  # [m] Jackal wheel radius (from URDF, kept for reference)
    max_linear_vel = 2.0  # [m/s] Jackal max velocity -> v = 2.0 * a0
    max_angular_vel = 4.0  # [rad/s] Jackal max angular velocity -> omega = 4.0 * a1

    # Rate limits (D6): per policy step (dt = 1/30 s)
    rate_limit_v = RATE_LIMIT_V  # [m/s] per step = 20 m/s² * 1/30 s
    rate_limit_omega = RATE_LIMIT_OMEGA  # [rad/s] per step = 25 rad/s² * 1/30 s

    # World configuration
    world_idx = 0  # base world index; env i gets world (world_idx + i) % 300
    is_dynamic = False  # whether this is a DynaBARN world (out of scope)

    # Start and goal positions (world frame, from upstream run.py)
    start_pos = (-2.25, 3.0)  # [m] BARN static start
    start_yaw = 1.57  # [rad] BARN static start yaw
    goal_offset = (0.0, 10.0)  # [m] relative to start
    goal_pos = (start_pos[0] + goal_offset[0], start_pos[1] + goal_offset[1])  # (-2.25, 13.0)

    # Optimal path length (for metric computation, loaded from barn_static.npz)
    optimal_path_length = 0.0

    # Reward scales (D8: minimal — progress, goal bonus, collision, time cost)
    rew_scale_progress = 5.0
    rew_scale_goal = 50.0
    rew_scale_collision = -25.0
    rew_scale_time = -0.1

    # Termination thresholds
    goal_threshold = 1.0  # [m] success if within this distance of goal

    def __post_init__(self):
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(8.0, 0.0, 6.0))
