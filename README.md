# robot-learning

Isaac Lab 3.0 + Isaac Sim 6.2 + MuJoCo-Warp (Newton) on an HP Omen laptop
(RTX 5050 Laptop 8 GB, Ryzen AI 7 350, 21 GB RAM, Ubuntu 26.04).

```
robot-learning/
  IsaacLab/          upstream v3.0.0 checkout, untouched
  Makefile           thin front end over uv / docker commands
  pyproject.toml     packaging + mkdocs docs extra
  mkdocs.yml         mkdocs material site config
  docs/              full documentation (15 guides)
  host/              setup-host.sh, power.sh
  my_robot_tasks/    external Isaac Lab tasks (Libero-style manipulation, demo recording)
  logs/              checkpoints + TensorBoard (gitignored)
```

## Documentation

Full guides in `docs/` (mkdocs-material):

```bash
make docs-install   # create .venv-docs + install mkdocs-material
make docs           # serve at http://127.0.0.1:8000
make docs-build     # static site into site/
```

## Quick start

```bash
cd ~/robot-learning
make host           # once: 20 GB swap + Docker GPU runtime (sudo)
make sync           # once: kit-less Isaac Lab (~8 GB download / venv)
make doctor         # sanity check
make train          # Cartpole on MuJoCo-Warp (~1 min)
make play           # watch in the Newton viewer
```

Real robot task:

```bash
make train TASK=Isaac-Velocity-Flat-UnitreeGo2 NUM_ENVS=2048
make play  TASK=Isaac-Velocity-Flat-UnitreeGo2
make tb             # TensorBoard at http://localhost:6006
```

LIBERO-style manipulation:

```bash
make train TASK=Libero-PickCube-Direct
make train TASK=Libero-PlaceCube-Direct
make train TASK=Libero-TransferCube-Direct
make train TASK=Libero-PushCube-Direct
```

Logs/checkpoints: `logs/rsl_rl/<experiment>/<timestamp>/`. LIBERO demos: `logs/demos/libero/`.

## Knobs

| Var | Default | Notes |
|-----|---------|-------|
| `TASK` | `Isaac-Cartpole-Direct` | task ID (3.0 dropped `-v0`) |
| `PHYSICS` | `mjwarp` | `mjwarp` \| `physx` \| `ovphysx` |
| `RL` | `rsl_rl` | `skrl`, `sb3`, `rl_games` need `EXTRAS=all` |
| `NUM_ENVS` | task default | lower if CUDA OOM; raise to saturate GPU |
| `VIZ` | empty | `newton_gl`, `kit`, `viser`, `rerun` |
| `CKPT` | `latest` | `.pt` for `make play` |
| `ARGS` | empty | passed to `isaaclab train/play` |
| `EXTRAS` | uv extras for `sync` / `sync-sim` |

Hydra overrides: `ARGS="agent.max_iterations=300"`.

## Physics backends

| `PHYSICS=` | Engine | Needs | Use it for |
|------------|--------|-------|------------|
| `mjwarp` | Newton + MuJoCo-Warp | nothing | Most RL; lightest on RAM/VRAM |
| `physx` | PhysX inside Isaac Sim | `make sync-sim` | RTX cameras, ROS bridge, Isaac Sim assets |
| `ovphysx` | Standalone PhysX | `ovphysx` extra | PhysX without Kit |

`mjwarp` is default because Isaac Sim alone uses 3–4 GB of 8 GB VRAM and wants 32 GB RAM. Kit-less Newton leaves the whole GPU for parallel environments.

## Custom tasks

`my_robot_tasks/` is an external package registered via the `isaaclab.tasks` entry point in `pyproject.toml`. It includes:

- `MyRobot-Cartpole-Direct` — cartpole balancing with HDF5 demo recording
- `Libero-PickCube-Direct`, `Libero-PlaceCube-Direct`, `Libero-TransferCube-Direct`, `Libero-PushCube-Direct` — LIBERO-style tabletop manipulation (Franka Panda, MuJoCo-Warp)

Install it editable into the uv env, then:

```bash
make train TASK=Libero-PickCube-Direct
make play  TASK=Libero-PickCube-Direct
```

No Makefile changes needed — `TASK`/`PHYSICS`/`RL`/`NUM_ENVS`/`VIZ`/`ARGS` map straight to `isaaclab`.

## Isaac Sim

```bash
make sync-sim      # add isaacsim==6.2.0.0 (~15 GB)
make sim           # GUI
make docker-sim    # official container fallback on Ubuntu 26.04
```

## Performance tips (RTX 5050, 8 GB VRAM)

- Plug into AC: GPU caps at 50 W on battery, 80 W / 115 W boost on AC.
- Run `make power` for the performance profile; `sudo ./host/power.sh boost` for Omen TGP.
- Train headless; visualise only with `make play`.
- Raise `NUM_ENVS` until `nvidia-smi` shows ~7 GB VRAM used.
- Close Chrome/Electron before `make sim`; 21 GB RAM is below Isaac Sim's 32 GB spec.

## Traps

- `uv sync` is exact — running it after `make sync-sim` uninstalls Isaac Sim. Re-run `make sync-sim` to restore.
- Quaternions are `(x, y, z, w)` in 3.0; 2.x used `(w, x, y, z)`. Old code silently breaks.
- Isaac Lab 2.x does not run on Isaac Sim 6.x and vice versa — match versions.
- First backend launch compiles Warp kernels (slow); later runs use `~/.cache/warp`.

## Upgrading

```bash
cd IsaacLab && git fetch --tags && git checkout v3.0.1
cd .. && make sync          # or sync-sim
```

Bump `pyproject.toml` version when you make breaking changes to the wrapper.
