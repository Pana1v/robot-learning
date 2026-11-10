# Launch a task

The one command to remember:

```bash
make train TASK=<task-id> [NUM_ENVS=..] [PHYSICS=mjwarp|physx|ovphysx] [RL=..] [VIZ=..] [ARGS=".."]
```

## 1. Pick a task

Run `make envs` to see every registered task and which physics/renderer presets it supports.

Bundled Isaac Lab tasks (examples):

| Task | Physics | Typical use |
|------|---------|-------------|
| `Isaac-Cartpole-Direct` | mjwarp | Fast sanity check (~1 min) |
| `Isaac-Velocity-Flat-UnitreeGo2` | mjwarp | Quadruped walking, raise NUM_ENVS |
| `Libero-PickCube-Direct` | mjwarp | Pick up a cube (Franka) |
| `Libero-PlaceCube-Direct` | mjwarp | Place cube in bowl |
| `Libero-TransferCube-Direct` | mjwarp | Transfer cube left→right |
| `Libero-PushCube-Direct` | mjwarp | Push cube to target |

`Libero-*` tasks are in `my_robot_tasks/` and record demos to `logs/demos/libero/`.

## 2. Launch training

```bash
# Cartpole — defaults, headless, ~1 minute
make train

# Quadruped — 2048 parallel envs
make train TASK=Isaac-Velocity-Flat-UnitreeGo2 NUM_ENVS=2048

# LIBERO pick cube
make train TASK=Libero-PickCube-Direct

# With Hydra-style overrides
make train TASK=Isaac-Cartpole-Direct ARGS="agent.max_iterations=300 --seed 3"

# Different physics backend (physx needs make sync-sim first)
make train TASK=Isaac-Cartpole-Direct PHYSICS=physx
```

## 3. Replay (play) a trained policy

```bash
# Replay the latest checkpoint
make play TASK=Isaac-Cartpole-Direct

# Replay a specific checkpoint
make play TASK=Isaac-Cartpole-Direct CKPT=logs/rsl_rl/.../checkpoints/latest.pt

# With a viewer
make play TASK=Isaac-Cartpole-Direct VIZ=newton_gl
```

## 4. Watch training live

```bash
make tb   # TensorBoard at http://localhost:6006
```

## Common flags

| Flag | Default | What it does |
|------|---------|--------------|
| `TASK` | `Isaac-Cartpole-Direct` | Task ID (from `make envs`) |
| `PHYSICS` | `mjwarp` | `mjwarp` \| `physx` \| `ovphysx` |
| `RL` | `rsl_rl` | RL library (`rsl_rl`, `skrl`, `sb3`, `rl_games`) |
| `NUM_ENVS` | task default | Parallel environments — raise to saturate GPU |
| `VIZ` | empty | Viewer: `newton_gl`, `kit`, `viser`, `rerun` |
| `CKPT` | `latest` | Checkpoint path for `make play` |
| `ARGS` | empty | Extra flags passed straight to `isaaclab train/play` |

## Tips

- Train headless (no `VIZ`) for speed; only visualise with `make play`.
- First backend launch compiles Warp kernels (slow); later runs are cached in `~/.cache/warp`.
- `make doctor` verifies GPU, swap, Docker, and package versions before you train.
- `make clean-logs` clears all checkpoints and TensorBoard logs.

## Troubleshooting launch failures

| Symptom | Fix |
|---------|-----|
| `isaaclab: command not found` | Run `make sync` first |
| CUDA OOM | Lower `NUM_ENVS` |
| Isaac Sim missing | Run `make sync-sim` |
| EULA prompt hangs | Makefile pre-accepts it; run `make sync-sim` again |
| Quaternions look wrong | Isaac Lab 3.0 uses `(x, y, z, w)`, not 2.x's `(w, x, y, z)` |
