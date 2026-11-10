# Troubleshooting & traps

## uv sync is exact
Running plain `make sync` after `make sync-sim` uninstalls Isaac Sim. Re-run `make sync-sim` (or pass the same `EXTRAS`) to restore it.

## Quaternions
Isaac Lab 3.0 uses `(x, y, z, w)`. Isaac Lab 2.x used `(w, x, y, z)`. Code/tutorials for 2.x silently break under 3.0. Isaac Lab 2.x does not run on Isaac Sim 6.x and vice versa.

## Version matching
Isaac Lab 3.0 + Isaac Sim 6.2 are a matched pair. Do not mix 2.x with 6.x.

## First launch is slow
Each backend compiles Warp kernels on first launch (a few minutes). Later runs hit the cache in `~/.cache/warp`.

## Isaac Sim EULA
The Makefile pre-accepts the EULA (`OMNI_KIT_ACCEPT_EULA=YES`, `ACCEPT_EULA=Y`).

## Memory
21 GB RAM is below Isaac Sim's 32 GB spec. Close Chrome/Electron apps before `make sim`; swap only softens that.

## GPU on battery
Capped at 50 W. Plug into AC for 80 W / 115 W boost. Run `make power` for the performance profile.

## CUDA OOM
Lower `NUM_ENVS` if you hit out-of-memory.

## Check `make doctor`
Verifies GPU, power, swap, Docker, torch/warp/mujoco/newton, and isaacsim versions.

## Clean logs/checkpoints
```bash
make clean-logs
```
