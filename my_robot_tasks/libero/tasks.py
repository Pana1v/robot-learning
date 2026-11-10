# Copyright (c) 2026, Pana1v
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LIBERO-style task registry.

Each entry maps a Gym task ID to its configuration overrides. The environment
class reads ``cfg.task_type`` to decide the goal condition and reward shaping.
"""

"""LIBERO-style task names for reference.

Task registration is handled in ``my_robot_tasks.__init__`` using task-specific
config subclasses (see ``cfg/libero_env_cfg.py``). This module keeps the task
descriptions documented for the dataset metadata.
"""

LIBERO_TASK_DESCRIPTIONS: dict[str, str] = {
    "Libero-PickCube-Direct": "pick up the cube",
    "Libero-PlaceCube-Direct": "place the cube in the bowl",
    "Libero-TransferCube-Direct": "transfer the cube from left to right",
    "Libero-PushCube-Direct": "push the cube to the target",
}

__all__ = ["LIBERO_TASK_DESCRIPTIONS"]
