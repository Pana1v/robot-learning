# Install & sync (uv)

The environment lives inside the `IsaacLab/` submodule and is managed by `uv`. The Makefile runs from the repo root via `uv run --project IsaacLab`.

## Kit-less install (default, recommended)

```bash
make sync
```

Installs kit-less Isaac Lab (Newton + MuJoCo-Warp, PyTorch cu130). ~8 GB download, ~8 GB venv.

Add extras: `make sync EXTRAS="all video"`

## Add Isaac Sim 6.2 (~15 GB)

```bash
make sync-sim
```

Adds `isaacsim==6.2.0.0` on top of the kit-less env. Needed for `physx` backend and the Isaac Sim GUI.

**Trap:** `uv sync` is exact. Running plain `make sync` after `make sync-sim` uninstalls Isaac Sim. Re-run `make sync-sim` (or pass the same `EXTRAS`) to restore it. `uv run` only adds packages; it never removes them.

## Available uv extras

| Extra | Purpose |
|-------|---------|
| `video` | video recording |
| `isaacsim` | Isaac Sim 6.2 (via `sync-sim`) |
| `ovphysx` | standalone PhysX without Kit |
| `all` | every extra above |

## Non-rsl RL libraries

`skrl`, `sb3`, `rl_games` need `make sync EXTRAS=all`.

## Verify

```bash
make doctor   # GPU, power, swap, docker, torch/warp/mujoco/newton, isaacsim
make envs     # list tasks + physics/renderer presets
```
