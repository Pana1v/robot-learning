"""Shared command-line wiring for the my_robot_tasks scripts.

Boots the Kit simulation app through AppLauncher before importing
my_robot_tasks, because isaaclab.sim (and carb) needs a running Kit context.
"""

from __future__ import annotations

import argparse

from isaaclab.app import add_launcher_args, AppLauncher


def add_base_args(parser: argparse.ArgumentParser) -> None:
    """Add task/execution arguments common to the train, play and smoke scripts."""
    parser.add_argument("--task", type=str, default="Isaac-BARN-Nav-Direct", help="Gym task id to run.")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
    add_launcher_args(parser)


def launch_app(argv: list[str] | None = None):
    """Parse CLI args, boot Kit via AppLauncher, import my_robot_tasks.

    Returns (args, simulation_app). The import of my_robot_tasks happens after
    the AppLauncher block so that isaaclab.sim can load under carb.
    """
    parser = argparse.ArgumentParser(description="Isaac Lab script for my_robot_tasks.")
    add_base_args(parser)
    args, _ = parser.parse_known_args(argv)

    app_launcher = AppLauncher(args)
    import my_robot_tasks  # noqa: F401 (registers gym tasks at import)

    return args, app_launcher.app
