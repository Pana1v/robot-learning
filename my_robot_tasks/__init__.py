# Copyright (c) 2026, Pana1v
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom Isaac Lab tasks for the robot-learning project.

This package is intentionally kept outside the untouched IsaacLab/ submodule so
that `git pull / checkout` upgrades of the submodule never clobber custom work.
Tasks are registered via the ``isaaclab.tasks`` entry point declared in the
root ``pyproject.toml`` and discovered automatically by ``isaaclab train`` /
``isaaclab play`` / ``isaaclab list_envs`` through ``load_external_tasks()``.
"""

import gymnasium as gym

from . import agents
from . import envs

##
# Register Gym environments.
##

gym.register(
    id="MyRobot-Cartpole-Direct",
    entry_point=f"{__name__}.envs.demo_cartpole_env:DemoCartpoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.demo_cartpole_env_cfg:DemoCartpoleEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DemoCartpoleDirectPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
