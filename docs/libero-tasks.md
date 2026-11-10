# LIBERO-style manipulation tasks

The `my_robot_tasks/` package adds four tabletop manipulation environments to Isaac Lab, following the
[LIBERO](https://lifelong-robotics.github.io/libero/) naming convention (pick, place, transfer, push).
These are **Isaac Lab environments**, not the original robosuite LIBERO benchmark — they run on the
MuJoCo-Warp backend with a Franka Panda arm and record episodes to HDF5 in a LeRobot-compatible structure
for future VLA training.

## Registered tasks

| Task ID | Description | Config class |
|---------|-------------|--------------|
| `Libero-PickCube-Direct` | Pick up the cube | `LiberoPickCubeEnvCfg` |
| `Libero-PlaceCube-Direct` | Place the cube in the bowl | `LiberoPlaceCubeEnvCfg` |
| `Libero-TransferCube-Direct` | Transfer the cube from left to right | `LiberoTransferCubeEnvCfg` |
| `Libero-PushCube-Direct` | Push the cube to the target | `LiberoPushCubeEnvCfg` |

All four share the same environment class (`LiberoManipulationEnv`, a `DirectRLEnv`) and differ only
through `cfg.task_type`, which selects the goal condition and reward shaping.

## Scene

- **Robot:** Franka Panda arm (7 arm joints + 2 finger joints), 6-DoF position control + gripper open/close
- **Objects:** table, green cube, red bowl, overhead camera (320×240 RGB, 10 Hz)
- **Physics:** MuJoCo-Warp, `dt = 1/120`, decimation = 2, episode length = 10 s
- **Observation (18-D):** 7 arm joint pos + 7 arm joint vel + 3 cube pos (relative) + 1 gripper state
- **Action (7-D):** 6 arm joint position deltas + gripper binary open/close
- **Rewards:** approach, grasp, lift, place, plus a small action penalty

## Run the tasks

```bash
# Train
make train TASK=Libero-PickCube-Direct
make train TASK=Libero-PlaceCube-Direct
make train TASK=Libero-TransferCube-Direct
make train TASK=Libero-PushCube-Direct

# Play
make play TASK=Libero-PickCube-Direct CKPT=latest

# List all registered tasks (includes the Libero-* IDs)
make envs
```

Because `my_robot_tasks` declares an `isaaclab.tasks` entry point in `pyproject.toml`, the tasks are
discovered automatically by `isaaclab train` / `isaaclab play` / `isaaclab list_envs` through
`load_external_tasks()` — no Makefile changes needed.

## Demo recording (for VLA training)

Each `LiberoManipulationEnv` episode is automatically saved to HDF5 under
`logs/demos/libero/episode_XXXXXX.hdf5` with:

- `observation.images.ego_view` — camera RGB frames
- `observation.state` — 18-D observation vectors
- `action` — 7-D action vectors
- `reward`, `done`, `timestamp`
- Attributes: `task_description`, `task_type`, `num_steps`, `fps`

This matches the LeRobot dataset structure used by the conversion script at
`IsaacLab/scripts/imitation_learning/locomanipulation_sdg/gr00t/convert_dataset.py`, so the recorded
demos can be converted and used for remote VLA fine-tuning (see [VLA pipeline](vla-pipeline.md)).

## Files

```
my_robot_tasks/
  __init__.py          # Gym task registration + isaaclab.tasks entry point
  cfg/
    libero_env_cfg.py  # LiberoSceneCfg, LiberoEnvCfg, and 4 task config subclasses
    __init__.py
  envs/
    libero_env.py      # LiberoManipulationEnv (DirectRLEnv with HDF5 demo recording)
    __init__.py
  libero/
    tasks.py           # LIBERO_TASK_DESCRIPTIONS registry
    __init__.py
  agents/
    rsl_rl_ppo_cfg.py  # PPO runner config
    __init__.py
```

## Recent changes

- `0eac21a` — Add LIBERO-style manipulation tasks to `my_robot_tasks` (Libero-PickCube/PlaceCube/TransferCube/PushCube-Direct)
- `951656a` — Add `my_robot_tasks` custom task package and demo collection pipeline
