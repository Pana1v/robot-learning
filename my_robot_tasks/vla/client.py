# Copyright (c) 2026, Pana1v
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stub VLA inference client.

Replace this with a real HTTP/gRPC client that talks to a remote VLA inference
server (smol-vLA or OpenVLA-small served from a machine with 24+ GB VRAM).

The client should accept an RGB image + language task description and return
joint actions. Do not call build.nvidia.com inside the control loop — its ~40 rpm
limit is far too slow for real-time policy execution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VLAAction:
    """Action returned by a VLA inference server."""

    joint_actions: np.ndarray  # shape (num_joints,), float32
    task_description: str


class VLAClient:
    """Abstract client for a remote VLA inference server.

    Subclass this and implement ``_call_server`` to talk to your actual
    inference endpoint (HTTP, gRPC, WebSocket, etc.).
    """

    def __init__(self, server_url: str | None = None, task_description: str = "balance the cartpole"):
        self.server_url = server_url
        self.task_description = task_description

    def infer(self, rgb_image: np.ndarray) -> VLAAction:
        """Run VLA inference on a single RGB image.

        Args:
            rgb_image: RGB image array of shape (H, W, 3), uint8 or float [0, 1].

        Returns:
            VLAAction with joint actions for the robot.
        """
        image = self._preprocess_image(rgb_image)
        raw_action = self._call_server(image, self.task_description)
        return self._postprocess_action(raw_action)

    def _preprocess_image(self, rgb_image: np.ndarray) -> np.ndarray:
        """Resize/normalize the image to the VLA model's expected input."""
        if rgb_image.dtype == np.uint8:
            rgb_image = rgb_image.astype(np.float32) / 255.0
        return rgb_image

    def _call_server(self, image: np.ndarray, task_description: str) -> np.ndarray:
        """Send image + text to the VLA server and receive raw action.

        Override this in a subclass with the real network call.
        """
        raise NotImplementedError(
            "VLAClient._call_server is not implemented. "
            "Point server_url at a running VLA inference server (smol-vLA / OpenVLA-small)"
            " and implement the HTTP/gRPC call here."
        )

    def _postprocess_action(self, raw_action: np.ndarray) -> VLAAction:
        """Map raw server output to joint actions in Isaac Lab's action space."""
        return VLAAction(joint_actions=np.asarray(raw_action, dtype=np.float32), task_description=self.task_description)
