# Copyright (c) 2026, Pana1v
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""VLA inference client stubs.

These are placeholders for a future remote VLA inference server (e.g. smol-vLA
or OpenVLA-small served from a machine with 24+ GB VRAM). The Isaac Lab env on
this laptop sends camera images + language command over the network and receives
joint actions in return.

Do not use the build.nvidia.com API inside the control loop: its ~40 rpm limit
is far too slow for real-time policy execution (30–120 Hz).
"""
