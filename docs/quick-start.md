# Quick start

Prerequisites: Ubuntu 26.04, NVIDIA GPU with CUDA drivers, `uv` installed.

```bash
cd ~/robot-learning

# 1. One-time host setup (sudo): 20 GB swap + Docker GPU runtime
make host

# 2. Install kit-less Isaac Lab (Newton + MuJoCo-Warp, ~8 GB download, ~8 GB venv)
make sync

# 3. Sanity check
make doctor

# 4. Train Cartpole (headless, ~1 minute)
make train

# 5. Watch it in the viewer
make play

# 6. TensorBoard
make tb   # http://localhost:6061
```

Then a real robot task:

```bash
make train TASK=Isaac-Velocity-Flat-UnitreeGo2 NUM_ENVS=2048
make play  TASK=Isaac-Velocity-Flat-UnitreeGo2
```

Logs and checkpoints go to `logs/rsl_rl/<experiment>/<timestamp>/`.
