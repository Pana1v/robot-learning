# Performance & power tuning

## Measured baseline (HP Omen, RTX 5050 Laptop 8 GB, Ryzen AI 7 350, 21 GB RAM, Ubuntu 26.04)

| Run | Envs | Speed | Peak VRAM | Notes |
|-----|------|-------|-----------|-------|
| Isaac-Cartpole-Direct (mjwarp) | 4096 | 150 iters / 63 s | small | converged (reward ~4.95) |
| Isaac-Velocity-Flat-UnitreeGo2 (mjwarp) | 2048 | ~0.56 s/iter (~88k steps/s) | 549 MiB | GPU ~56% busy at 41 W |

## Takeaways

- **VRAM is not the limit** for state-based MuJoCo-Warp tasks.
- **Plug into AC.** On battery the GPU is capped at 50 W and GNOME falls back to `power-saver`. On AC it reaches 80 W, or 115 W with Omen boost.
- **Raise `NUM_ENVS`** (4096, 8192) until `nvidia-smi` shows ~7 GB used, to saturate the GPU.
- Train headless; visualise only with `make play`.
- Close Chrome/Electron apps before `make sim` — 21 GB RAM is below Isaac Sim's 32 GB spec.

## Commands

```bash
make power          # performance profile + GPU status
./host/power.sh status     # live power / GPU status while training
sudo ./host/power.sh boost # Omen TGP + Dynamic Boost (re-run after reboot)
make check-cuda     # CUDA GPU runtime & memory status
```
