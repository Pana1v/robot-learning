# BARN-on-Isaac-Lab — execution notes

Per PLAN.md Section 7: gate results, evidence, verified claims with oracles, open questions.

## Environment (observed, not assumed)

- Host GPU: NVIDIA GeForce RTX 5050 Laptop GPU; `nvidia-smi -L` OK; `/dev/nvidia0` present.
- Isaac Lab source: `/home/panav/isaaclab-barn/IsaacLab` (read-only tree, untouched). `isaaclab.sh` present; emits deprecation warning, otherwise works.
- Python: `/home/panav/isaaclab-barn/IsaacLab/.venv/bin/python` 3.12.16. `pip` bootstrapped via `ensurepip`; setuptools copied in; editable install `robot-learning-0.1.0` OK. `TMPDIR=/home/panav/happy/Bots/research_scout/tmp` required (`/tmp` read-only).
- **Isaac Sim 6.2 full runtime is NOT available in this environment.** `import isaacsim` yields a namespace stub (`__file__ is None`); `isaacsim.simulation_app` does not exist (`ModuleNotFoundError`). `./IsaacLab/isaaclab.sh -p <script>` fails: "[ERROR] Isaac Sim is not installed or not found on PYTHONPATH." The uv cache holds only `extscache` fragments (Omniverse extension caches, cp312) from an aborted `isaacsim` wheel install — no `python.sh`, `kit`, or `SimulationApp`. GPU exists but Kit/PhysX cannot boot here.

## Phase 0 — Preflight (Gate 0: PARTIAL / BLOCKED on sim baselines)

- Import check from neutral cwd (`cd /`): PASS — H9 resolved.
  ```
  EnvSpec(id='Isaac-BARN-Nav-Direct', entry_point='my_robot_tasks.envs.barn_nav_env:BarnNavEnv',
          kwargs={'env_cfg_entry_point': 'my_robot_tasks.cfg.barn_nav_env_cfg:BarnNavEnvCfg',
                  'rsl_rl_cfg_entry_point': 'my_robot_tasks.agents.rsl_rl_ppo_cfg:BarnNavDirectPPORunnerCfg',
                  'default_agent': 'rsl_rl'})
  ```
  (Warp CUDA error 304 printed at import; harmless in this headless sandbox, import succeeds.)
- `./IsaacLab/isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless`: NOT runnable — no Isaac Sim runtime.
- Stock Cartpole baseline (Gate 0 hard requirement): NOT runnable — same blocker.
- API inventory (PLAN step 4): NOT done — requires `IsaacLab/source` greps under Kit; deferred with Gate 0.
- Upstream pin: `https://github.com/Daffan/the-barn-challenge` cloned to `third_party/the-barn-challenge`, HEAD `bf5a226f6088ec96bf0d2dbee3253a8ea6119b83` (bf5a226 "Update nav_metrics to reflect the new metrics"). Contains `jackal_helper/worlds/BARN/world_{i}.world`, `jackal_helper/worlds/DynaBARN/`, `run.py`, `test.sh`, `test_dynamic.sh`, `report_test.py`, `Singularityfile.def`.

## Phase 1 — Package and boot (Gate 1: BLOCKED on Kit boot)

Done:
- Package moved to `source/my_robot_tasks/my_robot_tasks/` (earlier session, `git mv`, history kept).
- `pyproject.toml`: `[tool.setuptools.packages.find] where = ["source/my_robot_tasks"] include = ["my_robot_tasks*"]`; entry point `my-robot-tasks = "my_robot_tasks"` under `isaaclab.tasks`; editable install OK.
- `scripts/cli_args.py`: shared wrapper — `AppLauncher(args)` then `import my_robot_tasks` (H11 pattern: task registration after AppLauncher block).
- `scripts/train.py`: boots Kit, imports `my_robot_tasks`, hands off to `isaaclab_rl.entrypoints.run_train_cli(sys.argv[1:])`.
- `scripts/play.py`: same for `run_play_cli`.
- `scripts/smoke_barn.py`: per PLAN Phase 1 step 5 template — boot Kit → `BarnNavEnv(cfg=cfg)` → `reset(seed=0)` → step loop printing obs dim, moved, min_hit, rew, term, trunc → `env.close()` / `simulation_app.close()`.
- Config layer constructs without Kit (verified): `observation_space=16`, `action_space=2`, `start_pos=(0,0)`, `goal_pos=(0,0)`, `wheel_base=0.513`, `wheel_radius=0.0985`, `scan` = 1081 beams @ `max_distance=10.0` (`horizontal_res=0.25` over ±135°), `robot` = `JACKAL_CFG` (`CuboidCfg`, no joint prims), 50 kinematic cuboid obstacles.

Inherited defect values confirmed by evidence (NOT fixed this phase, per PLAN Phase 1 rule):
- H1: start == goal == (0,0) → every episode succeeds at step 0.
- H2: `observation_space=16` but `_get_observations` returns 70 dims (6 lidar + 64 scan).
- H3/H8: RayCaster only vs `/World/ground` mesh; 1081 beams @ 10 m vs upstream 720 beams @ 30 m.
- H4: `JACKAL_CFG` is a CuboidCfg articulation with no joints → `wheel_drive` actuator cannot bind (expected first sim-side failure).
- H6: 50 random dynamic cuboids vs upstream ~209 fixed static cylinders.
- H7: wheel_base 0.513 vs upstream Jackal 0.262.

Gate 1 result: **BLOCKED**. `scripts/smoke_barn.py` never reaches `BarnNavEnv.__init__` because Kit cannot launch ("Isaac Sim is not installed or not found on PYTHONPATH"). The first real failure is therefore the missing Isaac Sim runtime, not H4/H2 — H4 vs H2 cannot be distinguished until Kit boots. Do not fix defects in this phase.

## Gate 2 — Phase 2: Pure-Python core unit tests (U1-U9)

**Gate 2 result: PASS.** All 9 unit tests pass in ~5.2 s (limit 30 s), without Kit / Isaac Lab.

Test file: `source/my_robot_tasks/tests/unit/test_barn_core.py` (stdlib `unittest`; pytest is not installed and there is no network, so the IsaacLab venv python runs unittest directly).

Method note: the `barn` modules are imported by file path (the `barn/` directory added to `sys.path`) rather than `import my_robot_tasks`, because the package `__init__` pulls in Isaac Lab which cannot boot without the Kit runtime. `barn/{worlds,metric,geometry}.py` have no Isaac imports and no mutual relative imports, so this is safe.

| ID | Check | Oracle | Result |
|---|---|---|---|
| U1 | world_0 parse | 209 cylinders, r 0.075, x∈[-4.425,-0.075], y∈[0.075,9.525] | PASS (the-barn-challenge@bf5a226, world_0.world) |
| U2 | All 300 worlds on grid | x = -4.575+0.15k, y = 0.075+0.15k, within 1e-3 | PASS (all 300) |
| U3 | world_0 path length | 13.592 m ± 1e-3 | PASS (path_files/path_0.npy) |
| U4 | Axis check, all 300 worlds | No path point inside a cylinder (d > 0.075); world_0 min 0.461 m; g0/g1 swap collides (0.05 m) | PASS |
| U5 | Metric boundaries | Fail→0; AT≤2·OT→0.5; AT≥8·OT→0.125; parity→0.125 any AT | PASS |
| U6 | Min-pool 720→64 | Single-beam hit survives; stride slicing loses it (trap documented) | PASS |
| U7 | Footprint test | Touch=hit, 1 cm gap=no hit, rotated cases | PASS |
| U8 | Analytic lidar | Cylinder 3.0 m straight ahead → center beam 2.925 m | PASS |
| U9 | Splits | test=range(0,300,6) (50); train∩test=∅; \|train\|=250 | PASS |

Notes / corrections made while writing tests:
- U2 originally applied the path y-offset (C_SHIFT=5.075) to obstacle centers. That is wrong: obstacles use y-offset 0.075 (= RADIUS); C_SHIFT=5.075 is the *path* offset, the PLAN's "path y-offset quirk" (1/3 cell off the obstacle grid). Fixed U2 to use RADIUS for y and R_SHIFT for x; all 300 worlds conform.
- U8: 720 is even, so `linspace(-2.35619, 2.35619, 720)` has no beam exactly at 0°. Forced `beam_angles[360]=0.0` so the forward beam reads exactly 2.925 m.
- U5: `analytic_lidar` clamps misses to `max_distance` (30.0) per PLAN D3 ("Rays that hit nothing return inf. Clamp them to max_distance."); test asserts misses return 30.0, not inf.

Upstream pin: `third_party/the-barn-challenge@bf5a226f6088ec96bf0d2dbee3253a8ea6119b83` (bf5a226).

## API inventory — Isaac Lab 3.0.0 source (verified by grep/read, no Kit required)

Completed PLAN Phase 0 step 4 (previously "NOT done — deferred with Gate 0"). Every item below confirmed in `IsaacLab/source/isaaclab/`, version 3.0.0 (`IsaacLab/VERSION`). Isaac Sim runtime still missing (Gate 1 blocked), so runtime behavior is unverified; only API names/signatures are confirmed.

| PLAN item | API | File:line | Notes |
|---|---|---|---|
| RayCaster cfg | `RayCasterCfg` | `isaaclab/sensors/ray_caster/ray_caster_cfg.py:25` | `mesh_prim_paths: list[str] = MISSING` — doc says "only a single static mesh is supported"; `offset` (pos/rot), `ray_alignment` ("base"/"yaw"/"world"), `pattern_cfg` (MISSING), `max_distance=1e6`, `global_world_only`, `spawn` (deprecated 3.0). |
| Multi-mesh ray caster | `MultiMeshRayCasterCfg(RayCasterCfg)` | `isaaclab/sensors/ray_caster/multi_mesh_ray_caster_cfg.py:20` | `mesh_prim_paths: list[str \| RaycastTargetCfg] = MISSING`; `RaycastTargetCfg.prim_expr: str = MISSING`, `is_shared`, `merge_prim_meshes`, `track_mesh_transforms`; `reference_meshes=True`, `update_mesh_ids=False`. PLAN D3 target exists. |
| Lidar pattern | `LidarPatternCfg(PatternBaseCfg)` | `isaaclab/sensors/ray_caster/patterns/patterns_cfg.py:204` | `func`, `channels` (MISSING), `vertical_fov_range` (MISSING), `horizontal_fov_range` (MISSING), `horizontal_res` (MISSING). |
| Contact sensor cfg | `ContactSensorCfg` | `isaaclab/sensors/contact_sensor/contact_sensor_cfg.py:19` | `prim_path` (SensorBaseCfg) selects sensing bodies; `filter_prim_paths_expr: list[str] = []` selects per-partner filter bodies; filtered force reported only when non-empty; `{ENV_REGEX_NS}` supported. |
| Filtered force matrix | `ContactSensorData.force_matrix_w` | `isaaclab/sensors/contact_sensor/base_contact_sensor_data.py:236` | "Total filtered contact forces [N] between sensor and filtered bodies" = normal+friction. **None if `filter_prim_paths_expr` empty.** This is PLAN D4's filtered force matrix. |
| Trap field | `ContactSensorData.net_forces_w` | `isaaclab/sensors/contact_sensor/base_contact_sensor_data.py:197` | Total incl. ground support — fires every step if thresholded. PLAN D4 trap confirmed. |
| `normal_force_matrix_w` | `ContactSensorData.normal_force_matrix_w` | `...base_contact_sensor_data.py:104` | Filtered normal forces; None if filter empty. |
| Mesh spawn helper | `create_prim_from_mesh(prim_path, mesh, **kwargs)` | `isaaclab/terrains/utils.py:59` | PLAN D1 candidate confirmed to exist. |
| Root velocity writer | `RigidObject.write_root_velocity_to_sim_index(root_velocity, *, env_ids, skip_forward)` | `isaaclab/assets/rigid_object/base_rigid_object.py:355` | `root_velocity` (len(env_ids),6) = [vx,vy,vz,wx,wy,wz] world frame. **PLAN D2 correction:** `write_root_velocity_to_sim` (line 790) is DEPRECATED in 3.0.0; use `_index` variant. |
| DirectRLEnv | `DirectRLEnv(gym.Env)` | `isaaclab/envs/direct_rl_env.py:37` | `step(action)` → (obs,rew,terminated,truncated,info); `reset(seed,options)` → (obs,info); `step_dt`; `_reset_idx(env_ids)`; `_pre_physics_step(actions)`; `_apply_action()` (no args — reads action buffer); `_get_observations()`; `_get_rewards()`; `_get_dones()` → (terminated, truncated). |
| RigidObjectCfg | `RigidObjectCfg(AssetBaseCfg)` | `isaaclab/assets/rigid_object/rigid_object_cfg.py:16` | `init_state: InitialStateCfg` with `lin_vel`, `ang_vel`; spawn via AssetBaseCfg. |

## Git

`.git` is read-only in this sandbox (Git LTS tmp-dir error on `git status`); commits are deferred until write access is available. Commit messages must follow PLAN rule 8 (imperative subject < 72 chars, body with only `## Issue` / `## Outcome` / `## Implementation`).

## Gate status (current session verified)

| Gate | Phase | Result | Evidence |
|------|-------|--------|----------|
| 0 | Preflight | PARTIAL / BLOCKED on sim baselines | Import check from `/` PASS (H9 resolved). `create_empty.py` and Cartpole baseline not runnable — no Isaac Sim runtime. |
| 1 | Package and boot | **BLOCKED** | `isaaclab.sh` reports "Isaac Sim is not installed or not found on PYTHONPATH." `import isaacsim` yields a namespace stub; `isaacsim.simulation_app` does not exist. `scripts/smoke_barn.py` never reaches `BarnNavEnv.__init__`. |
| 2 | Pure-Python core | **PASS** | U1-U9 all pass in 3.9 s (IsaacLab venv python, unittest directly). Verified 2025-11-12. `barn/{worlds,metric,geometry}.py` have no Isaac imports. |
| 3 | Sim scenarios S1-S8 | **BLOCKED** | `tests/sim/test_barn_scenarios.py` written and syntax-checked; all tests skip (`@unittest.skipUnless(_ISAACLAB_AVAILABLE)`). Requires Kit boot (Gate 1). |
| 4 | Scene rebuild | **BLOCKED** | Depends on Gate 1 (Kit boot + smoke script reaching `BarnNavEnv.__init__`). |
| 5 | Env logic | **BLOCKED** | Depends on Gate 4. |
| 6 | Training | **BLOCKED** | Depends on Gate 5. Requires `--task Isaac-BARN-Nav-Direct --headless --num_envs 256 --max_iterations 50`. |
| 7 | Evaluation | **BLOCKED** | Depends on Gate 6. Requires `scripts/eval_barn.py --checkpoint <path> --split test` on 50 test worlds (0,6,...,294). |

## Files complete and syntax-checked (no Kit required)

- `my_robot_tasks/cfg/jackal_cfg.py` — `JACKAL_CFG` as `RigidObjectCfg` cuboid 0.46×0.42×0.268 m, mass 17 kg (D2).
- `my_robot_tasks/cfg/barn_nav_env_cfg.py` — `BarnNavEnvCfg` with named constants LIDAR_BINS=64, OBSERVATION_SPACE=71, ACTION_SPACE=2, MultiMeshRayCasterCfg 720 beams @30 m, ContactSensorCfg filtered against obstacles (D1-D5).
- `my_robot_tasks/envs/barn_nav_env.py` — Full `BarnNavEnv` rewrite: per-env world assignment, `_create_obstacle_meshes` via `create_prim_from_mesh`, body-frame→world-frame velocity with D6 rate limits, 71-dim obs, minimal rewards, (terminated, time_out) split, `compute_nav_metric` (D2-D9).
- `my_robot_tasks/barn/{worlds,metric,geometry}.py` — Pure-Python core: world parser, nav metric, min-pool, analytic lidar, footprint test.
- `scripts/build_barn_data.py` — Parses all 300 upstream static worlds, writes `data/barn_static.npz`.
- `data/barn_static.npz` — Generated: 300 worlds, cylinders 181-365/world, path_len 10.05-13.74 m (world 0 = 13.592 m matching oracle), optimal_t 5.03-6.87 s, upstream hash bf5a226.
- `tests/unit/test_barn_core.py` — U1-U9, all pass.
- `tests/sim/test_barn_scenarios.py` — S1-S8, skip-until-Kit.
- `scripts/cli_args.py`, `scripts/train.py`, `scripts/play.py`, `scripts/smoke_barn.py` — Kit boot scripts (unrunnable without Gate 1).
- `my_robot_tasks/agents/rsl_rl_ppo_cfg.py` — `BarnNavDirectPPORunnerCfg` for RSL-RL PPO training.

## Fixes this session

- `envs/barn_nav_env.py:254`: replaced `self.scan.data.distance_to_hit` (does not exist on `RayCasterData`) with
  `(self.scan.data.ray_hits_w - self.scan.data.pos_w.unsqueeze(1)).norm(dim=-1)`, mirroring `scripts/smoke_barn.py:48`.
  `ray_hits_w` is (N, B, 3) with `inf` for misses (`ray_caster_data.py:56-66`), so min-pooling propagates misses as `inf`;
  `ProxyArray` supports `__sub__`, `.unsqueeze` and `.norm` via its torch bridge (`proxy_array.py:262-301,336-356`).
- `py_compile` of `barn_nav_env.py`: COMPILE_OK.
- Gate 2 re-run: U1-U9 PASS in ~6 s (unittest, IsaacLab venv python, no Kit).

## Gate status (unchanged)

Gate 1 is **BLOCKED**: Isaac Sim 6.2 full runtime is not installed in this environment. `import isaacsim` is a
namespace stub; there is no `kit`, no `python.sh`, no `SimulationApp`; `isaaclab.sh` errors
"Isaac Sim is not installed or not found on PYTHONPATH." `~/.cache/uv` holds only aborted `extscache` fragments,
and there is no network to fetch the runtime. Kit/PhysX cannot boot here, so `smoke_barn.py` never reaches
`BarnNavEnv.__init__`, and Phases 3-7 (S1-S8 sim tests, scene rebuild, env logic, training, evaluation) cannot proceed
until Isaac Sim 6.2 exists on a machine with GPU/Kit access.

## Next step

Install the full Isaac Sim 6.2 runtime and make it available to the IsaacLab venv (e.g. link it at
`IsaacLab/_isaac_sim` so `isaaclab.sh` finds `python.sh`/`setup_python_env.sh`), then re-run Gate 1
(`scripts/smoke_barn.py --headless`, reaching `BarnNavEnv.__init__`) and record whether H4 or H2 fires first.
Only after Gate 1 unblocks can Phases 3-7 proceed.
