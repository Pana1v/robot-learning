# Quick start

Get a task running in four commands:

```bash
cd ~/robot-learning

# 1. One-time host setup (sudo): swap + Docker GPU runtime
make host

# 2. Install kit-less Isaac Lab (~8 GB download / venv)
make sync

# 3. Sanity check
make doctor

# 4. Train a task
make train
```

That trains `Isaac-Cartpole-Direct` on MuJoCo-Warp, headless, in about a minute.

## Next — launch a different task

```bash
# See every registered task
make envs

# Quadruped walking (2048 envs)
make train TASK=Isaac-Velocity-Flat-UnitreeGo2 NUM_ENVS=2048

# LIBERO-style manipulation (Franka pick cube)
make train TASK=Libero-PickCube-Direct

# Replay the latest policy
make play TASK=Isaac-Cartpole-Direct

# Watch training
make tb   # http://localhost:6006
```

For the full launch reference, see [Launch a task](launch-tasks.md).

Logs and checkpoints go to `logs/rsl_rl/<experiment>/<timestamp>/`. LIBERO demos go to `logs/demos/libero/`.
