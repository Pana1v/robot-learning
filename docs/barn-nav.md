# BARN Navigation Challenge

Port of the ICRA BARN Navigation Challenge (ROS/Gazebo) to Isaac Lab 3.0 as a
`DirectRLEnv`. Drives a velocity-controlled Clearpath Jackal from a start pose
to a goal pose across 300 static obstacle worlds, avoiding collisions.

## Task spec

| Spec | Value |
|------|-------|
| Task ID | `Isaac-BARN-Nav-Direct` |
| Entry point | `my_robot_tasks.envs.barn_nav_env:BarnNavEnv` |
| Config | `my_robot_tasks.cfg.barn_nav_env_cfg:BarnNavEnvCfg` |
| Observation | 71-dim: 64 lidar bins + 3 goal-relative + 2 base velocity + 2 prev action |
| Action | 2-dim `[-1, 1]` → `v = 2·a0`, `ω = 4·a1` with per-step rate limits |
| Start pose | `(-2.25, 3.0, yaw 1.57)` |
| Goal | `(-2.25, 13.0)` |
| Physics | `RigidObjectCfg` cuboid (0.46×0.42×0.268 m, 17 kg), no wheels/actuators |
| Velocity write | `write_root_velocity_to_sim_index` (world frame, linear-first) |
| Lidar | `MultiMeshRayCaster`, 720 beams @ 30 m, min-pooled to 64 bins |
| Contact | `ContactSensorCfg` filtered to `{ENV_REGEX_NS}/Obstacles/mesh` |
| Rewards | Progress, goal bonus, collision, time cost (minimal, D8) |
| Termination | `terminated = success \| collision`, `time_out` split (D7) |
| Metric | `int(success) · OT / clip(AT, 2·OT, 8·OT)`, `OT = path_length / 2` (D9) |
| Worlds | Static 0-299 from `third_party/the-barn-challenge@bf5a226` |

## Design (D1-D9)

- **D1**: One static per-env obstacle mesh built from upstream world files via `create_prim_from_mesh`, authored at `/World/envs/env_{i}/Obstacles`.
- **D2**: Velocity-driven rigid body. Body-frame `(v, ω)` → world-frame root velocity written via `write_root_velocity_to_sim_index`.
- **D3**: 720-beam lidar @ 30 m, min-pooled to 64 bins, normalized by `max_distance`.
- **D4**: `ContactSensorCfg.filter_prim_paths_expr` targets `{ENV_REGEX_NS}/Obstacles/mesh` (exactly one collider prim per env; wildcard matches 2 prims and PhysX returns `None`).
- **D5**: 71-dim observation vector.
- **D6**: Per-step rate limits `|Δv| ≤ rate_limit_v`, `|Δω| ≤ rate_limit_omega`.
- **D7**: `(terminated, time_out)` split; terminated = success \| collision.
- **D8**: Minimal rewards: progress, goal bonus, collision, time cost.
- **D9**: Nav metric from `barn/metric.py`, pure torch.

## Offline data

`scripts/build_barn_data.py` parses all 300 upstream static worlds and writes
`data/barn_static.npz` containing per-world cylinder centers, planned path
lengths, and optimal times. The env loads this table once in `_setup_scene` and
caches `self._world_centers` / `self._world_path_lengths`.

## Launch

```bash
make train TASK=Isaac-BARN-Nav-Direct NUM_ENVS=256
make play  TASK=Isaac-BARN-Nav-Direct
```

Training requires the full Isaac Sim 6.2 runtime (Gate 1 blocked without it;
see `NOTES.md`).

## Diagnostics

`scripts/diag_barn.py` runs headlessly and verifies:

- **D1**: Robot spawn pose vs BARN start pose
- **D2**: Contact sensor filtered-force availability (collision detection)
- **D3**: Lidar vs pure-Python analytic oracle
- **D4**: Root pose/velocity write paths (`write_root_pose_to_sim_index`, `write_root_velocity_to_sim_index`)
- **D5**: Per-world optimal path length available in the NPZ table

```bash
python scripts/diag_barn.py --num_envs 1
```

## Reward ablation

`plot_reward_comparison.py` compares old vs new reward curves head-to-head,
parsing eval log lines and plotting mean RL cost vs training steps, plus a
JSON summary.

## Tests

- `tests/unit/test_barn_core.py` — U1-U9, pure Python (`unittest`), no Kit or Isaac Lab required. All pass.
- `tests/sim/test_barn_scenarios.py` — S1-S8, skip-until-Kit (`@unittest.skipUnless(_ISAACLAB_AVAILABLE)`). Requires Gate 1 (Isaac Sim runtime).

## Repository layout

```
source/my_robot_tasks/my_robot_tasks/
  cfg/
    jackal_cfg.py        — JACKAL_CFG RigidObjectCfg, JACKAL_ROOT_Z=0.134
    barn_nav_env_cfg.py  — BarnNavEnvCfg: obs/action spaces, sensors, start/goal
  envs/
    barn_nav_env.py      — BarnNavEnv: D1-D9 implementation
  barn/
    worlds.py            — world parser (centers, path lengths)
    metric.py            — nav_metric (pure torch)
    geometry.py          — min-pool, analytic lidar, footprint test
  agents/
    rsl_rl_ppo_cfg.py    — BarnNavDirectPPORunnerCfg
tests/
  unit/
    test_barn_core.py    — U1-U9
  sim/
    test_barn_scenarios.py — S1-S8 (skip-until-Kit)
scripts/
  build_barn_data.py     — parse 300 worlds → barn_static.npz
  diag_barn.py           — headless D1-D5 diagnostics
  train.py / play.py     — Kit boot → isaaclab_rl entry points
  smoke_barn.py          — per-env smoke test template
data/
  barn_static.npz        — 300 worlds: centers, path_lengths, optimal_times
```

## Gate status

See `NOTES.md` for the current gate table. Gate 2 (unit tests U1-U9) passes;
Gate 1 (Isaac Sim boot) is blocked in this environment until the full Isaac
Sim 6.2 runtime is installed.
