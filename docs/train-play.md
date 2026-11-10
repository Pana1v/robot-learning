# Train & play

## Train

```bash
make train [TASK=..] [PHYSICS=mjwarp|physx|ovphysx] [NUM_ENVS=..] [VIZ=..] [RL=..] [ARGS=".."]
```

Examples:

```bash
make train                                                    # Cartpole, mjwarp, headless
make train TASK=Isaac-Velocity-Flat-UnitreeGo2 NUM_ENVS=2048   # quadruped, 2048 envs
make train PHYSICS=physx                                      # PhysX inside Isaac Sim (needs sync-sim)
make train ARGS="--seed 3 --max_iterations 500"               # extra isaaclab flags
make train ARGS="agent.max_iterations=300"                    # Hydra-style override
```

## Play (replay a policy)

```bash
make play [TASK=..] [CKPT=path] [VIZ=..] [ARGS=".."]
```

`CKPT` defaults to `latest`. Use `--viz newton_gl` for a viewer unless `VIZ` is set.

## Notes

- First launch of each backend compiles Warp kernels and is slow (a few minutes); later runs hit the cache in `~/.cache/warp`.
- Isaac Sim asks for the EULA interactively on first launch; the Makefile exports `OMNI_KIT_ACCEPT_EULA=YES` and `ACCEPT_EULA=Y` up front.
- Isaac Lab 3.0 uses quaternion order `(x, y, z, w)` (2.x used `(w, x, y, z)`). Old 2.x code/tutorials silently break.
- Train headless and only visualise with `make play`.
