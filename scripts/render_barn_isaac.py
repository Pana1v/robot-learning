#!/usr/bin/env python3
"""Render the BARN navigation environment using Isaac Lab's Camera sensor.

Drives the Jackal from its start pose toward the goal along a straight line
by writing the robot's USD transform directly each frame (this PhysX/Warp
build rejects set_velocities/set_transforms writes at the backend level, so
the physics API cannot move the body; the USD override is what the RTX
renderer draws).

Run through the Isaac Sim Kit entry point so the full Omniverse runtime
(carb, kit, RTX renderer) is available:

    isaacsim --exec scripts/render_barn_isaac.py -- --world 0 --steps 30

Outputs RGB frames into --out-dir and prints a per-frame diff so motion can
be verified numerically even without viewing the images.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import warp as wp

# my_robot_tasks imports isaaclab modules that need the Kit runtime (carb).
# When launched via `isaacsim --exec`, Kit is already up, so these imports work.
from my_robot_tasks.cfg.barn_nav_env_cfg import BarnNavEnvCfg
from my_robot_tasks.envs.barn_nav_env import BarnNavEnv
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sim import PinholeCameraCfg
from pxr import Gf, UsdGeom


def build_camera_cfg(prim_path: str, width: int, height: int, pos, quat) -> CameraCfg:
    """Camera looking down at the robot from a high-angle side view."""
    return CameraCfg(
        prim_path=prim_path,
        spawn=PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=10.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        width=width,
        height=720 if False else height,
        data_types=["rgb"],
        offset=CameraCfg.OffsetCfg(pos=pos, rot=quat, convention="world"),
    )


# --- Robot motion plan -----------------------------------------------------

START_POS = (-2.25, 3.0)  # world frame [m], from BarnNavEnvCfg.start_pos
START_YAW = 1.57  # [rad], facing +Y
GOAL_POS = (-2.25, 13.0)  # world frame [m]
ROBOT_Z = 0.134  # [m] half of the 0.268 m Jackal chassis height

MOVE_SPEED = 0.5  # [m per frame] along the straight start->goal line


def robot_pose(frame: int) -> tuple[tuple[float, float, float], float]:
    """Pose of the robot at a given frame: moves from start toward goal."""
    start = np.array(START_POS, dtype=float)
    goal = np.array(GOAL_POS, dtype=float)
    direction = goal - start
    dist = float(np.linalg.norm(direction))
    along = min(frame * MOVE_SPEED, dist)
    pos = start + (direction / dist) * along
    return (float(pos[0]), float(pos[1]), ROBOT_Z), START_YAW


def quat_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    """Quaternion (x, y, z, w) for a yaw-only rotation."""
    half = yaw / 2.0
    return (0.0, 0.0, math.sin(half), math.cos(half))


def write_usd_transform(stage, prim_path: str, pos, yaw: float) -> None:
    """Override the USD transform of a prim (bypasses the PhysX backend).

    Removes any existing xformOp properties then authors a fresh
    translate + rotateXYZ pair so the RTX renderer draws the prim at the
    requested pose, regardless of what the PhysX backend believes.
    """
    from pxr import Usd

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"prim not found: {prim_path}")
    xform = UsdGeom.Xformable(prim)
    for prop in prim.GetProperties():
        name = prop.GetName()
        if name == "xformOpOrder" or name.startswith("xformOp:"):
            prim.RemoveProperty(name)
    xform.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    xform.AddRotateXYZOp().Set(Gf.Vec3d(0.0, 0.0, math.degrees(yaw)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Render BARN env with moving robot.")
    parser.add_argument("--world", type=int, default=0, help="Static world index (0-299).")
    parser.add_argument("--steps", type=int, default=30, help="Frames to capture.")
    parser.add_argument(
        "--out-dir", type=str, default=".context/renders_isaac", help="Output directory."
    )
    args = parser.parse_args()

    # --- Build the env (Kit already up via `isaacsim --exec`) ---
    cfg = BarnNavEnvCfg()
    cfg.scene.num_envs = 1
    cfg.world_idx = args.world
    env = BarnNavEnv(cfg=cfg)
    # NOTE: do NOT call env.reset() here — __init__ -> sim.reset() already placed
    # the robot at its configured initial pose; env.reset() crashes on this
    # PhysX/Warp build (ProxyArray vs transformf in the warp kernel launch).

    # --- Camera ---
    cam = Camera(
        cfg=build_camera_cfg(
            prim_path="/World/envs/env_0/Camera",
            width=1280,
            height=720,
            pos=(0.5, 1.5, 2.8),
            quat=(0.35, -0.35, 0.15, 0.85),
        )
    )
    cam._initialize_impl()
    cam._is_initialized = True

    # Aim the camera at the robot's start->goal path so the motion is in frame.
    # Closer/lower than before: at ~5 m distance the 0.46 m Jackal is ~150 px wide
    # and its 10 m traverse (y=3 -> y=13 at x=-2.25) crosses most of the frame
    # width, so the motion is visually obvious rather than a small distant blob.
    cam.set_world_poses_from_view(
        eyes=torch.tensor([[1.5, 8.0, 3.5]], device="cpu"),
        targets=torch.tensor([[-2.25, 8.0, ROBOT_Z]], device="cpu"),
    )

    # --- Direct USD transform handle for the robot (reliable visual path) ---
    stage = env.sim.stage
    robot_prim_path = "/World/envs/env_0/Robot"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[render] world={args.world} steps={args.steps} out={out_dir}")

    prev_rgb = None
    for i in range(args.steps):
        pos, yaw = robot_pose(i)
        quat = quat_from_yaw(yaw)

        # Step physics first (keeps the sensor timestamp machinery advancing);
        # any PhysX->USD writeback lands here at the robot's previous pose.
        env.sim.step()

        # Authoritative visual path: override the USD transform each frame,
        # AFTER the step's PhysX->USD writeback. The RTX renderer draws this
        # overridden pose because nothing runs between here and the lazy
        # camera capture that would write the physics pose back to USD.
        write_usd_transform(stage, robot_prim_path, pos, yaw)

        # Force a fresh per-frame capture: drive the camera's render path
        # directly with an all-True env mask so `_update_buffers_impl` does
        # not early-return on the cleared `_is_outdated` mask (the bug that
        # made earlier runs return the cached first frame, diff=0.0).
        # `render_into_camera` -> `update_scene_state` ->
        # `update_transforms` -> `render_batch` -> `read_output` re-reads the
        # USD stage, so the robot transform override authored above is drawn
        # fresh at its new pose every frame.
        stale_mask = wp.ones(cam._view.count, dtype=wp.bool, device=cam._device)
        cam._update_buffers_impl(stale_mask)
        rgb_torch = cam.data.output["rgb"].torch  # (1, H, W, 3) uint8
        rgb = rgb_torch[0].cpu().numpy()  # (H, W, 3)

        # Robot position reported from the sim data buffer (physics-side view).
        sim_pos = env.robot.data.root_pos_w[0, :2].cpu().numpy()
        sim_yaw = env._get_yaw(env.robot.data.root_quat_w)[0].cpu().item()

        # Frame-to-frame motion check.
        diff = None
        if prev_rgb is not None:
            diff = float(np.abs(rgb.astype(np.int16) - prev_rgb.astype(np.int16)).mean())
        prev_rgb = rgb.copy()

        frame_path = out_dir / f"frame_{i:03d}.png"
        plt.imsave(frame_path, rgb)
        print(
            f"[render] frame {i:03d} usd=({pos[0]:+.2f},{pos[1]:+.2f}) yaw={yaw:+.2f} "
            f"sim=({sim_pos[0]:+.2f},{sim_pos[1]:+.2f}) diff={diff if diff is None else f'{diff:.1f}'} -> {frame_path}"
        )

    env.close()
    print("[render] done")


if __name__ == "__main__":
    main()
