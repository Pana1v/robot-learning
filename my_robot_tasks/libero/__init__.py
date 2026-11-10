# Copyright (c) 2026, Pana1v
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LIBERO-style manipulation task definitions for Isaac Lab.

Task IDs registered here follow the LIBERO naming convention (pick, place,
transfer, etc.) but run natively in Isaac Lab 3.0 with the MuJoCo-Warp backend,
so they are lightweight enough to run on a laptop GPU (RTX 5050, 8 GB VRAM).

These are Isaac Lab environments, not the original robosuite LIBERO benchmark.
They reuse Isaac Lab assets (Franka Panda, tabletop primitives) and record
demonstrations in LeRobot-compatible HDF5 for future VLA training on a remote
machine with 24+ GB VRAM.
"""

from .tasks import LIBERO_TASK_DESCRIPTIONS

__all__ = ["LIBERO_TASK_DESCRIPTIONS"]
