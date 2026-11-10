# robot-learning

Isaac Lab 3.0 + Isaac Sim 6.2 + MuJoCo-Warp (Newton) on an HP Omen laptop
(RTX 5050 Laptop 8 GB, Ryzen AI 7 350, 21 GB RAM, Ubuntu 26.04).

```
robot-learning/
  IsaacLab/          upstream v3.0.0 checkout, untouched (git pull / checkout to upgrade)
  Makefile           thin front end, every target prints the real uv/docker command
  host/setup-host.sh one-time sudo: 20 GB swap, Docker GPU runtime
  host/power.sh      performance mode, Omen GPU boost, power status
```

## Quick start

```bash
cd ~/robot-learning
make host                 # once: swap + docker GPU runtime (asks for sudo)
make sync                 # once: kit-less Isaac Lab, ~8 GB download, ~8 GB venv
make doctor               # sanity check
make power                # plug in AC first, then performance mode
make train                # Cartpole on MuJoCo-Warp, ~1 minute
make play                 # watch it in the Newton viewer
```

Then a real robot:

```bash
make train TASK=Isaac-Velocity-Flat-UnitreeGo2 NUM_ENVS=2048
make play  TASK=Isaac-Velocity-Flat-UnitreeGo2
make tb                   # http://localhost:6006
```

## Measured on this laptop (2026-11-03, on battery, 50 W cap, power-saver)

| Run | Envs | Speed | Peak VRAM | Notes |
|---|---|---|---|---|
| `Isaac-Cartpole-Direct`, mjwarp | 4096 | 150 iters in 63 s | small | converged (reward 4.95). Whole command 2m50s incl. first-run Warp compile |
| `Isaac-Velocity-Flat-UnitreeGo2`, mjwarp | 2048 | 0.56 s/iter (~88k steps/s) | 549 MiB | GPU only 56% busy, 41 W. A walking gait typically needs ~500 to 1500 iters, so roughly 5 to 15 min |

Takeaway: VRAM is not the limit for state-based tasks on MuJoCo-Warp. Raise
`NUM_ENVS` (4096, 8192) and plug in AC; the GPU was not even saturated here.
Logs and checkpoints go to `~/robot-learning/logs/rsl_rl/<experiment>/<timestamp>/`.

## The three physics backends

| `PHYSICS=` | Engine | Needs | Use it for |
|---|---|---|---|
| `mjwarp` (default) | Newton with the MuJoCo-Warp solver | nothing extra | Most RL. Lightest on RAM/VRAM, no Isaac Sim loaded |
| `physx` | PhysX inside full Isaac Sim (Kit) | `make sync-sim` | Tasks/assets that rely on Isaac Sim, RTX cameras, ROS bridge |
| `ovphysx` | Standalone PhysX, no Kit | auto installs `ovphysx` extra | PhysX behaviour without the Isaac Sim weight |

Not every task supports every backend. `make envs` shows the presets per task.

Why MuJoCo-Warp is the default here: Isaac Sim alone eats 3 to 4 GB of the
8 GB VRAM and wants 32 GB RAM. Kit-less Newton skips all of that, so the
whole GPU goes to parallel environments.

## Knobs

| Var | Default | Notes |
|---|---|---|
| `TASK` | `Isaac-Cartpole-Direct` | 3.0 dropped the `-v0` suffix |
| `PHYSICS` | `mjwarp` | see table above |
| `NUM_ENVS` | task default | lower it if you hit CUDA OOM |
| `VIZ` | empty (headless) | `newton_gl`, `kit` (physx only), `viser`, `rerun` |
| `RL` | `rsl_rl` | `skrl`, `sb3`, `rl_games` need `make sync EXTRAS=all` |
| `CKPT` | `latest` | path to a `.pt` for `make play` |
| `ARGS` | empty | passed straight to `isaaclab`, e.g. `ARGS="--seed 3 --max_iterations 500"` |

Hydra style overrides go in `ARGS` too, e.g. `ARGS="agent.max_iterations=300"`.

## Isaac Sim

```bash
make sync-sim             # adds isaacsim==6.2.0.0 to the same venv (~15 GB)
make sim                  # GUI
make train PHYSICS=physx  # train on PhysX inside Kit
```

If the pip Isaac Sim misbehaves on Ubuntu 26.04 (it is only tested on 22.04/24.04),
use the official container instead:

```bash
make docker-sim           # builds on nvcr.io/nvidia/isaac-sim:6.2.0
make docker-enter P=base
```

`make docker-kitless` gives the same for the kit-less stack on Ubuntu 24.04.

## Getting the most out of the laptop

- **Plug in.** On battery the GPU is capped at 50 W and GNOME drops back to
  `power-saver` by itself. On AC it can go to 80 W, or 115 W with boost.
- `make power` sets the performance profile. `sudo ./host/power.sh boost` also
  turns on the Omen custom TGP and Dynamic Boost (same as Omen Gaming Hub);
  re-run after a reboot. Check with `./host/power.sh status` while training.
- Train headless and only visualise with `make play`.
- Raise `NUM_ENVS` until `nvidia-smi` shows ~7 GB used. More envs = better GPU
  use, up to the point where VRAM runs out.
- Close Chrome/Electron apps before `make sim`; 21 GB RAM is below Isaac Sim's
  32 GB spec and swap only softens that.

## Traps

- **`uv sync` is exact.** Running plain `make sync` after `make sync-sim`
  uninstalls Isaac Sim. Use `make sync-sim` again, or pass the same `EXTRAS`.
  `uv run` (used by train/play) only adds, it never removes.
- **Quaternions are now `(x, y, z, w)`.** Isaac Lab 2.x used `(w, x, y, z)`.
  Code or tutorials written for 2.x silently break.
- **Isaac Lab 2.x does not run on Isaac Sim 6.x** and vice versa.
- First launch of each backend compiles Warp kernels and is slow (a few minutes).
  Later runs hit the cache in `~/.cache/warp`.

## Upgrading

```bash
cd IsaacLab && git fetch --tags && git checkout v3.0.1   # whatever is newest
cd .. && make sync          # or sync-sim
```

## Why uv and not pixi

Isaac Lab 3.0 ships its own `pyproject.toml` and `uv.lock` with uv specific
overrides (`[tool.uv] override-dependencies`) that reconcile Isaac Sim's older
pins with Newton/PyTorch. Pixi would re-resolve all of it into a second lock it
does not test, and its main advantage (conda-forge system packages) is not
needed because every dependency here ships as a wheel. The Makefile covers the
task-runner part pixi would add.
