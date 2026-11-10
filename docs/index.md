# robot-learning documentation

This repo is a thin uv-based front end over [Isaac Lab 3.0](https://github.com/isaac-sim/IsaacLab) + Isaac Sim 6.2 + MuJoCo-Warp (Newton) for reinforcement learning on an HP Omen laptop (RTX 5050, 8 GB VRAM).

## Contents

- [Quick start](quick-start.md)
- [Host setup (one-time sudo)](host-setup.md)
- [Install & sync (uv)](install-sync.md)
- [Train & play](train-play.md)
- [TensorBoard](tensorboard.md)
- [Performance & power tuning](performance-tuning.md)
- [Physics backends](physics-backends.md)
- [Custom tasks (external package)](custom-tasks.md)
- [Docker fallbacks](docker.md)
- [Free cloud GPU: Colab & Kaggle](colab-kaggle.md)
- [VLA pipeline: demos → remote fine-tune → local inference](vla-pipeline.md)
- [Troubleshooting & traps](troubleshooting.md)
- [Upgrading Isaac Lab](upgrading.md)

## Repository layout

```
robot-learning/
  IsaacLab/          upstream v3.0.0 git submodule (untouched)
  Makefile           thin front end over uv / docker commands
  pyproject.toml     packaging for the robot-learning wrapper (name: robot-learning, v0.1.0)
  README.md          measured results and project overview
  host/
    setup-host.sh    one-time sudo: 20 GB swap + Docker GPU runtime
    power.sh         performance mode, Omen GPU boost, power status
  docs/              this guide
  logs/              checkpoints + TensorBoard (gitignored)
```

## Makefile knobs

| Var | Default | Meaning |
|-----|---------|---------|
| `TASK` | `Isaac-Cartpole-Direct` | Isaac Lab task ID |
| `PHYSICS` | `mjwarp` | `mjwarp` \| `physx` \| `ovphysx` |
| `RL` | `rsl_rl` | `rsl_rl` \| `skrl` \| `sb3` \| `rl_games` |
| `NUM_ENVS` | task default | parallel environments |
| `VIZ` | empty | `newton_gl` \| `kit` \| `viser` \| `rerun` |
| `CKPT` | `latest` | checkpoint for `make play` |
| `ARGS` | extra flags passed to `isaaclab train/play` |
| `EXTRAS` | uv extras for `make sync` / `sync-sim` |

Run `make help` for the full target list.
