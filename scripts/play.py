#!/usr/bin/env python3
"""Play an RSL-RL checkpoint for my_robot_tasks via the Isaac Lab unified entrypoint.

Boot Kit first (AppLauncher), then import my_robot_tasks so the Isaac-BARN-Nav-Direct
gym task is registered, then hand off to isaaclab_rl.entrypoints.run_play_cli.
"""

from __future__ import annotations

import sys

from cli_args import launch_app

_, simulation_app = launch_app()

from isaaclab_rl.entrypoints import run_play_cli  # noqa: E402

sys.exit(run_play_cli(sys.argv[1:]))
