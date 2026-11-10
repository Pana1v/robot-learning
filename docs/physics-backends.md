# Physics backends

Select via `PHYSICS=` (default `mjwarp`).

| Backend | Engine | Needs | Best for |
|---------|--------|-------|----------|
| `mjwarp` | Newton + MuJoCo-Warp solver | nothing extra | Most RL; lightest on RAM/VRAM; no Isaac Sim |
| `physx` | PhysX inside full Isaac Sim (Kit) | `make sync-sim` | Tasks/assets needing Isaac Sim, RTX cameras, ROS bridge |
| `ovphysx` | Standalone PhysX, no Kit | auto installs `ovphysx` extra | PhysX behaviour without Isaac Sim weight |

Not every task supports every backend. `make envs` shows presets per task.

`mjwarp` is the default because Isaac Sim alone consumes 3–4 GB of the 8 GB VRAM and wants 32 GB RAM, while kit-less Newton leaves the whole GPU for parallel environments.

```bash
make train PHYSICS=mjwarp   # default
make train PHYSICS=physx    # needs make sync-sim first
make train PHYSICS=ovphysx
```
