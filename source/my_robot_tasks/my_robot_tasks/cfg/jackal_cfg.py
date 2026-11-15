# Copyright (c) 2026, Pana1v
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Clearpath Jackal differential-drive mobile robot.

Phase A (D2): modeled as a single dynamic rigid body, velocity-driven.
The collision envelope comes from the Jackal URDF (chassis 0.420 × 0.310 m
plus wheels extends to about 0.46 × 0.42 m), height 0.268 m, mass 17 kg
from the URDF inertials. No joints or actuators: body-frame (v, ω) is
written directly as root velocity every physics step.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg


JACKAL_CFG = RigidObjectCfg(
    spawn=sim_utils.CuboidCfg(
        size=(0.46, 0.42, 0.268),  # [m] Jackal collision envelope (chassis + wheels)
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.02,
            rest_offset=0.0,
        ),
        activate_contact_sensors=True,
        mass_props=sim_utils.MassPropertiesCfg(mass=17.0),  # [kg] from URDF inertials
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.134),
        lin_vel=(0.0, 0.0, 0.0),
        ang_vel=(0.0, 0.0, 0.0),
    ),
)
