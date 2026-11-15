# BARN Navigation on Isaac Lab: Execution Plan

## 0. Mission

- **Goal:** Run the static BARN benchmark natively in Isaac Lab. Build the env, reset, step, train PPO (RSL-RL), and evaluate on the 50 upstream test worlds with the upstream nav metric.
- **Environment:** Native Isaac Sim 6.2 and Isaac Lab at `~/robot-learning/IsaacLab`, RTX 5090, no sandbox, network available.
- **Starting point:** `my_robot_tasks` from the previous agent. The code parses and its configs construct, but it has never run. Treat every "verified" in the old handoff as unverified.
- **Out of scope until their gates:** DynaBARN (Phase 8), articulated Jackal (Phase 9), reward tuning.
- **Done when:** Gates 0 through 7 pass and `NOTES.md` records the evidence for each.

## 1. Operating rules

1. **Read before you call.** Confirm every Isaac Lab class, field and method in the installed source under `IsaacLab/source/` before using it. The last session guessed `activation_material`, `size=`, `distance_max=` and `attach_yaw_only=`. Every guess failed. This install splits physics into `isaaclab_physx`, so docs for older versions will mislead you.
2. **Boot Kit first.** Every script and sim test creates `AppLauncher` before importing env code. Without it, `import carb` fails even on a working install.
3. **Failing test first, then one fix.** Make one change, run the tests once, make one commit. Never batch fixes.
4. **Two strikes.** If a defect survives two fixes, stop. Write the failed hypotheses in `NOTES.md` and re-analyze from evidence.
5. **Copy a working pattern.** Before writing a new pattern (sim test, sensor, custom scene prim), find the closest stock Isaac Lab task or test and read it fully.
6. **Do not edit `IsaiahLab/`.** All changes live in the task package.
7. **Name the oracle.** Every "verified" in commits and notes states what it was checked against: an upstream file and commit, an analytic oracle, or a test ID.
8. **Commits:** imperative subject under 72 chars. The body uses only `## Issue`, `## Outcome`, `## Implementation`. No co-author or "generated with" lines.
9. **Stop and report instead of improvising** when any of these happens:
   - A Phase 0 baseline fails.
   - An API this plan names does not exist.
   - A gate cannot be met.
   - A design decision here proves wrong.

## 2. Reference facts

Source: `Daffan/the-barn-challenge@main` (`run.py`, `test.sh`, `test_dynamic.sh`, `gazebo_simulation.py`, `jackal_helper/src/collision_publisher.cpp`, `jackal_helper/worlds/`) and `jackal/jackal@noetic-devel`. Record both commit hashes in Phase 0. These values are pinned to them.

| Item | Value | Source |
|---|---|---|
| Static worlds | idx 0-299, `jackal_helper/worlds/BARN/world_{i}.world` | run.py |
| Static start / goal | Start (-2.25, 3.0), yaw 1.57. The goal offset (0, 10) is **relative to the start**, so the absolute goal is (-2.25, 13.0) | run.py |
| Dynamic worlds | idx 300-359, `DynaBARN/world_{i-300}.world`. Start (11, 0), yaw 3.14. Absolute goal (-9, 0) | run.py |
| Success | Distance to goal ≤ 1.0 m | run.py |
| Failure | Any contact between `jackal` and a non-ground entity, or 100 s elapsed | collision_publisher.cpp, run.py |
| Static obstacles | Static cylinders, r = 0.075 m, length 1.0 m. Grid: x = -4.575 + 0.15k (k = 1..30), y = 0.075 + 0.15k (k = 0..63) | world_0 |
| world_0 anchors | 209 cylinders. x ∈ [-4.425, -0.075], y ∈ [0.075, 9.525]. Model poses equal `<state>` poses | world_0 |
| Planned path | `worlds/BARN/path_files/path_{i}.npy`, (N, 2) grid indices. x_w = 0.15·g0 - 4.575, y_w = 0.15·g1 + 5.075. Prepend start, append goal, sum segment lengths | run.py |
| world_0 path | 43 points. path_length 13.592 m. Nearest cylinder center to any path point: 0.461 m. Swapping g0/g1 gives 0.05 m | computed |
| Path y-offset quirk | The path y-offset 5.075 sits 1/3 cell off the obstacle y-grid. This is upstream behavior: replicate it, do not "fix" it | computed |
| Metric | `success · OT / clip(AT, 2·OT, 8·OT)`, with OT = path_length / 2. **Ceiling is 0.5**, reached at an average speed ≥ 1 m/s along the path | run.py |
| DynaBARN quirk | path_length = -20 - 11 = -31. The clip bounds invert, so every success scores exactly 0.125 | run.py, numpy check |
| Test sets | Static: 0, 6, ..., 294 (50 worlds) × 10 trials. Dynamic: 300-359 × 10 trials | test.sh, test_dynamic.sh |
| DynaBARN content | About 11 moving cylinders (r = 0.5 m) plus 3 wall boxes. Motion lives in compiled plugins `plugins/easylibobs_N.so`. The repo has no plugin source | DynaBARN/world_0, repo tree |
| Jackal geometry | Wheelbase 0.262, track 0.37559, wheel r 0.098, chassis 0.420 × 0.310 × 0.184 m | jackal.urdf.xacro |
| Jackal limits | v ≤ 2.0 m/s, a ≤ 20 m/s². ω ≤ 4.0 rad/s, α ≤ 25 rad/s². Skid-steer separation multiplier 1.5 | control.yaml |
| BARN lidar | UST-10 Gazebo sensor: 720 samples, ±2.35619 rad, range 0.1-30 m, 50 Hz. BARN adds offset (-0.065, 0, 0.01) from the laser mount. Confirm the mount link and height in jackal_description | hokuyo_ust10.urdf.xacro, run.py |

## 3. Inherited defects

| # | Defect in the handoff | Resolved by |
|---|---|---|
| H1 | `start_pos == goal_pos == (0, 0)`, so every episode succeeds at step 0 | D7 |
| H2 | `observation_space=16`, but the lidar alone contributes 64 dims | D5 |
| H3 | Stock `RayCaster` casts only against static mesh prims. The 50 obstacles are rigid bodies, so the lidar cannot see them | D1, D3 |
| H4 | `ArticulationCfg` spawned from a `CuboidCfg` has no joints, so the `wheel_drive` actuator cannot bind | D2 |
| H5 | Collision proxy is a ray distance < 0.15 m from the lidar. The hull extends about 0.21 m, so it fires only after penetration | D4 |
| H6 | 50 random dynamic cuboids. BARN uses about 200 fixed static cylinders per world | D1 |
| H7 | `wheel_base=0.513`. The Jackal track is 0.37559 and the wheelbase is 0.262 | D2 (field removed) |
| H8 | 1081 beams at 10 m matches the physical UST-10LX datasheet. The BARN Gazebo sensor is 720 beams at 30 m | D3 |
| H9 | `PYTHONPATH` points at the package dir. Imports work only when the cwd is its parent | Phase 1 |
| H10 | Test commands skip `AppLauncher` | Phase 1 |
| H11 | Stock `train.py` never imports `my_robot_tasks`, so the task is never registered | Phase 1 |
| H12 | The nav metric was "verified" only against its own formula | D9, Phase 2 |

## 4. Target design

### D1. Worlds: static geometry from upstream files

- Parse all 300 static worlds offline into cylinder centers. Store them in `data/barn_static.npz` with per-world path lengths and the upstream commit hash.
- Assign one world per env: `world_of_env[i] = split[i % len(split)]`.
  - A world stays fixed for the whole run.
  - Diversity comes from the number of envs, not from resets, so train with num_envs ≥ 250.
- Physics: build one static collider mesh per env at `/World/envs/env_{i}/Obstacles` in `_setup_scene`.
  - Use that env's cylinders in env-local frame (BARN frame origin = env origin).
  - No rigid body API, so the robot cannot shove them and they cost nothing to simulate.
  - Candidate helper: `isaaclab.terrains.utils.create_prim_from_mesh` (confirm).
- Mesh: open cylinders (no caps), 12 sides, 1.0 m tall.
  - A ground robot and horizontal rays never touch the caps.
  - 12 sides keep the radial error at 0.0026 m and the mesh near 5k triangles per world.
- Delete the `RigidObjectCollectionCfg` and `_place_obstacles`.

### D2. Robot (phase A): one dynamic rigid body, velocity-driven

- Use a `RigidObjectCfg` cuboid sized to the Jackal collision envelope from the URDF.
  - Chassis plus wheels gives about 0.46 × 0.42 m. Fender meshes may enlarge it. Write the derivation in a comment.
  - Take the mass from the URDF inertials.
  - Set `activate_contact_sensors=True`.
- In `_apply_action`, convert body-frame (v, ω) into a world-frame root velocity and write it every physics step (confirm the writer's name).
  - Zero the lateral velocity.
  - Keep the simulated vertical velocity.
  - Zero the roll and pitch rates.
- **Keep the body dynamic, never kinematic.** PhysX filters out kinematic-static pairs by default. A kinematic robot would pass through the static obstacles with no contact report.
- Why not wheels yet: BARN's interface is `cmd_vel`. A velocity-driven body reproduces that interface with no friction or gain tuning. Wheels come in Phase 9.

### D3. Lidar

- Match the BARN sensor:
  - 720 rays from -135° to +135°.
  - One channel at zero elevation.
  - `max_distance=30.0`.
  - Mounted at the URDF laser pose plus BARN's offset.
  - Confirm the `LidarPatternCfg` fields.
- Choose the ray target from the Phase 0 inventory:
  - If `MultiMeshRayCasterCfg` exists, cast against the per-env obstacle meshes.
  - Otherwise, build one merged, collision-free copy of all tiles in world frame at `/World/ObstacleRays` and point the stock `RayCasterCfg` at it.
- Whatever target you choose, test S5 must pass. With the merged copy, that needs env_spacing ≥ 45 m. Reason: a tile including start and goal spans about 5 × 14 m, and rays reach 30 m.
- Rays that hit nothing return inf. Clamp them to max_distance.
- **Downsample by min-pooling contiguous bins, never by stride slicing.** A 0.15 m cylinder at 3 m subtends 2.9°. 64 bins over 270° are 4.2° wide, so stride slicing can step over a cylinder entirely.

### D4. Collision

- Put a `ContactSensorCfg` on the robot body with `filter_prim_paths_expr=["{ENV_REGEX_NS}/Obstacles"]`. Read the filtered force matrix, never `net_forces_w`.
- Trap: `net_forces_w` includes the ground's support force, so any threshold on it fires every step. BARN excludes ground contact explicitly.
- Fallback: this version may not report filtered contacts against static colliders. If so, use the Phase 2 analytic footprint test (oriented rectangle vs circles), which is exact for BARN geometry. Report the switch in `NOTES.md`.
- Delete the 0.15 m ray proxy.

### D5. Observations (default layout)

| Term | Dims |
|---|---|
| Lidar: 64 min-pooled bins, divided by max_distance | 64 |
| Goal in body frame: distance, sin and cos of bearing | 3 |
| Body velocity v, ω | 2 |
| Previous action | 2 |
| **Total** | **71** |

- Compute `observation_space` from named constants in the cfg, never a literal. A hard-coded literal is how H2 happened.

### D6. Actions

- 2 dims in [-1, 1]. Scale to v = 2.0·a0 m/s and ω = 4.0·a1 rad/s.
- Rate-limit at each policy step (dt = 4/120 s) using the control.yaml accelerations:
  - |Δv| ≤ 0.667 m/s
  - |Δω| ≤ 0.833 rad/s
- Without these limits, the policy learns instant velocity changes that no wheeled robot can make.

### D7. Episodes

- Reset:
  - Pose at env_origin + (-2.25, 3.0), yaw 1.57, zero velocity.
  - Clear every per-env buffer: previous goal distance, previous action, rate-limiter state.
- Trap: a stale previous distance makes the first progress reward after a reset enormous.
- `_get_dones` returns (terminated, time_out):
  - terminated = success OR collision
  - time_out = elapsed ≥ 100 s
- Trap: the timeout belongs in the time-out slot. PPO bootstraps the value on time-outs and zeroes it on terminations. Mixing them teaches the critic that running out of time is a terminal state.
- At each episode end, write these per-env values into `extras`: success, collided, timed_out, actual_time (steps × step_dt), world_idx, nav_metric.

### D8. Rewards (minimal, not tuned here)

- Progress k_p·(d_prev - d_now), a success bonus, a collision penalty, and a small per-step time cost.
- Keep existing reward terms only if they reduce to these. Reward tuning needs Pan's approval.

### D9. Metric and splits

- Put the metric in `barn/metric.py`, pure torch with no Isaac imports: `nav_metric(success, actual_time, path_length)`.
- Path length always comes from the precomputed table, never from straight-line distance.
- Add a flag `dynabarn_parity: bool`:
  - True reproduces upstream (0.125 per success).
  - False uses |path_length| = 31 m.
- Splits:
  - `test = range(0, 300, 6)` (50 worlds).
  - `train` = the other 250 worlds. Never train on test worlds.
  - Optional `val` = 25 worlds held out from train.

## 5. Package layout

```
robot-learning/
├── IsaacLab/                        # untouched
├── third_party/the-barn-challenge/  # pinned clone, data source only
└── source/my_robot_tasks/
    ├── pyproject.toml
    ├── my_robot_tasks/
    │   ├── __init__.py              # gym.register only, no heavy imports
    │   ├── barn/                    # no Isaac imports: testable without Kit
    │   │   ├── worlds.py            # .world parser, path lengths, splits
    │   │   ├── metric.py
    │   │   └── geometry.py          # min-pool, analytic lidar, footprint test
    │   ├── envs/barn_nav_env.py
    │   ├── cfg/barn_nav_env_cfg.py, jackal_cfg.py
    │   ├── agents/rsl_rl_ppo_cfg.py
    │   └── data/barn_static.npz
    ├── scripts/                     # smoke_barn, build_barn_data, train, play, cli_args, eval_barn
    └── tests/
        ├── unit/                    # plain pytest, no Kit
        └── sim/                     # AppLauncher first
```

`barn/` has no Isaac imports on purpose. Most correctness checks then run in seconds, outside the slow Kit loop.

## 6. Phases

### Phase 0: Preflight (no code changes)

1. Locate the code:
   ```bash
   find ~ -type d -name my_robot_tasks -not -path "*/.venv/*" 2>/dev/null
   ```
   If it exists only under `~/happy/Bots/research_scout`, copy it into `~/robot-learning` and commit it unchanged before touching anything.
2. Check the GPU and Kit:
   ```bash
   cd ~/robot-learning/IsaacLab
   nvidia-smi -L
   ./isaaclab.sh -p -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   ./isaaclab.sh -p -m pip show isaacsim | head -2
   cat VERSION && git log -1 --oneline
   ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless
   ```
3. Run the stock RL baseline:
   ```bash
   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
     --task Isaac-Cartpole-Direct-v0 --headless --max_iterations 5
   ```
   If a path or task name differs in this version, find it with `find scripts -name "*.py"` and `scripts/environments/list_envs.py`. Do not assume.
4. Take an API inventory. Record each item's exact name, file and line in `NOTES.md`, or mark it "missing":
   ```bash
   cd ~/robot-learning/IsaacLab/source
   grep -rn "class RayCasterCfg\|class MultiMeshRayCasterCfg\|class LidarPatternCfg" .
   grep -rn "class ContactSensorCfg\|filter_prim_paths_expr\|force_matrix_w" . | head -20
   grep -rn "def create_prim_from_mesh" .
   grep -rn "def write_root_velocity_to_sim\|def write_root_pose_to_sim" . | head
   grep -rn "activate_contact_sensors" . | head -5
   grep -n "static" $(grep -rln "class RayCaster(" .)
   cd .. && ./isaaclab.sh --help | grep -i -- "--new"
   ```
5. Pin the upstream data:
   ```bash
   git clone https://github.com/Daffan/the-barn-challenge.git ~/robot-learning/third_party/the-barn-challenge
   git -C ~/robot-learning/third_party/the-barn-challenge rev-parse HEAD   # record in NOTES.md
   ```

**Gate 0:** `create_empty.py` runs headless. Cartpole trains 5 iterations. `NOTES.md` lists every API above with file:line or "missing". The upstream hash is recorded.
**Stop if** either baseline fails. That is an install problem, not a BARN problem.

### Phase 1: Package and boot

1. Move the package with `git mv` into `source/my_robot_tasks/my_robot_tasks/`. Using `git mv` keeps the history.
2. Add `pyproject.toml`:
   ```toml
   [build-system]
   requires = ["setuptools>=64"]
   build-backend = "setuptools.build_meta"

   [project]
   name = "my_robot_tasks"
   version = "0.1.0"
   requires-python = ">=3.11"
   # torch, isaacsim and isaaclab come from the host env; listing them lets pip replace those builds
   dependencies = []

   [tool.setuptools.packages.find]
   include = ["my_robot_tasks*"]

   [tool.setuptools.package-data]
   my_robot_tasks = ["data/*.npz"]
   ```
3. Install, then test from a neutral cwd:
   ```bash
   cd ~/robot-learning
   ./IsaacLab/isaaclab.sh -p -m pip install -e source/my_robot_tasks
   cd / && ~/robot-learning/IsaacLab/isaaclab.sh -p -c \
     "import gymnasium as gym, my_robot_tasks; print(gym.spec('Isaac-BARN-Nav-Direct'))"
   ```
   Running from `/` proves the import comes from the install, not from the cwd (H9). Remove every `PYTHONPATH` export from scripts and docs.
4. Add training entry points (H11). Choose one:
   - If `isaaclab.sh --new` exists, generate a template in `/tmp/barn_template` and read its train and play scripts. Copy that pattern only, not its whole layout.
   - Otherwise, copy the stock `rsl_rl/train.py`, `play.py` and `cli_args.py` into `scripts/`. Add `import my_robot_tasks  # noqa: F401` directly after the AppLauncher block.
5. Add `scripts/smoke_barn.py`:
   ```python
   """Construct BarnNavEnv, reset, drive forward, print diagnostics."""
   import argparse

   from isaaclab.app import AppLauncher

   parser = argparse.ArgumentParser()
   parser.add_argument("--num_envs", type=int, default=1)
   parser.add_argument("--steps", type=int, default=300)
   AppLauncher.add_app_launcher_args(parser)
   args = parser.parse_args()

   # Boot Kit before env imports: isaaclab_physx imports carb at module load
   simulation_app = AppLauncher(args).app

   import torch  # noqa: E402

   from my_robot_tasks.cfg.barn_nav_env_cfg import BarnNavEnvCfg  # noqa: E402
   from my_robot_tasks.envs.barn_nav_env import BarnNavEnv  # noqa: E402

   ROBOT_KEY, LIDAR_KEY = "robot", "lidar"  # must match the scene cfg

   cfg = BarnNavEnvCfg()
   cfg.scene.num_envs = args.num_envs
   env = BarnNavEnv(cfg=cfg)
   obs, _ = env.reset(seed=0)
   print(f"obs dim {obs['policy'].shape[-1]} | cfg {cfg.observation_space}")

   robot, lidar = env.scene[ROBOT_KEY], env.scene[LIDAR_KEY]
   start_xy = robot.data.root_pos_w[:, :2].clone()
   actions = torch.zeros(env.num_envs, 2, device=env.device)
   actions[:, 0] = 0.25  # 0.5 m/s once D6 scaling lands

   for i in range(args.steps):
       obs, rew, term, trunc, _ = env.step(actions)
       if i % 30 == 0:
           # misses are inf, so inf here means the lidar saw nothing
           hit = (lidar.data.ray_hits_w - lidar.data.pos_w.unsqueeze(1)).norm(dim=-1)
           moved = (robot.data.root_pos_w[:, :2] - start_xy).norm(dim=-1)
           print(f"step {i:3d} moved {moved.mean():.2f} m | min_hit {hit.min():.2f} | "
                 f"rew {rew.mean():.3f} | term {int(term.sum())} | trunc {int(trunc.sum())}")

   env.close()
   simulation_app.close()
   ```

**Gate 1:** The import check passes from `/`. The smoke script reaches `BarnNavEnv.__init__`. Record the first real failure in `NOTES.md`; expect H4 or H2. Do not fix it in this phase.

### Phase 2: Pure-Python core (no Kit)

1. `barn/worlds.py`: parse the `unit_cylinder_*` poses and assert that model poses equal `<state>` poses. Compute path lengths with the exact upstream conversion.
2. `barn/metric.py`, as specified in D9.
3. `barn/geometry.py` with three functions:
   - `min_pool_ranges`
   - `analytic_lidar` (ray vs circle, batched torch)
   - `footprint_hits_cylinders` (oriented rectangle vs circles)
4. `scripts/build_barn_data.py` writes `data/barn_static.npz`: padded centers plus counts, radius, path_length[300], and the upstream hash.
5. Write unit tests in `tests/unit/` and run them with `./IsaacLab/isaaclab.sh -p -m pytest source/my_robot_tasks/tests/unit`:

| ID | Check | Expected |
|---|---|---|
| U1 | world_0 parse | 209 cylinders, r 0.075, x ∈ [-4.425, -0.075], y ∈ [0.075, 9.525] |
| U2 | All 300 worlds on grid | x = -4.575 + 0.15k and y = 0.075 + 0.15k, within 1e-3 |
| U3 | world_0 path length | 13.592 m ± 1e-3 |
| U4 | Axis check, all 300 worlds | No path point lies inside a cylinder (d > 0.075 m). Log the minimum per world. world_0 gives 0.461 m; a g0/g1 swap gives 0.05 m |
| U5 | Metric | Fail gives 0. AT ≤ 2·OT gives 0.5. AT ≥ 8·OT gives 0.125. Parity flag gives 0.125 for any AT on DynaBARN |
| U6 | Min-pool | A hit on a single beam survives 720 → 64. Stride slicing on the same input loses it (documents the trap) |
| U7 | Footprint test | Rectangle touching a circle is a hit. A 1 cm gap is no hit. Include rotated cases |
| U8 | Analytic lidar | One cylinder centered 3.0 m straight ahead: the center beam reads 2.925 m |
| U9 | Splits | test = range(0, 300, 6). train ∩ test = ∅. \|train\| = 250 |

**Gate 2:** U1 through U9 pass in under 30 s, without Kit.

### Phase 3: Sim scenario tests, written to fail first

Write `tests/sim/test_barn_scenarios.py` before touching the scene. Find a stock test that boots Kit (`grep -rln "AppLauncher" IsaacLab/source/*/test | head`) and copy its boot and fixture pattern.

| ID | Scenario | Pass condition | Catches |
|---|---|---|---|
| S1 | world_0. Robot 0.5 m from a cylinder with ≥ 1 m free on the approach side, facing it, v = 1 m/s | Collision terminates within 1.0 s. Success flag stays false | H3, H5 |
| S2 | Zero action from the start pose | time_out at step 3000 (100 s at 30 Hz). terminated is never set | D7 slots |
| S3 | Teleport to 0.9 m from the goal | terminated with the success flag on that step | H1 |
| S4 | world_0 start pose | RayCaster ranges match `analytic_lidar` within 0.02 m on every hit beam, with an identical hit/miss pattern | H3, H8, mount pose |
| S5 | 64 envs on 64 different worlds | Each env matches its own world's oracle (S4 tolerance) | Neighbor-tile leakage |
| S6 | Robot at rest in open space for 2 s | Collision flag never set | Ground in `net_forces_w` |
| S7 | Reset right after S1 | First-step progress reward ≤ k_p · 2.0 · step_dt | Stale previous distance |
| S8 | Observation shape | `obs.shape[-1] == cfg.observation_space` | H2 |

**Gate 3:** All eight tests run and fail for the expected reasons. Record the reasons in `NOTES.md`. If a test passes now, check that it can fail at all.

### Phase 4: Scene rebuild

Make one commit per step. Run the smoke script and the sim tests after each.

1. Robot per D2. Delete the actuator cfg and `wheel_base`. The smoke script should construct the env, and the robot should move under the forward command.
2. Obstacle physics meshes per D1. Measure scene build time and GPU memory at 64, 256 and 1024 envs, and record them.
3. Lidar per D3. S4 and S5 should turn green.
4. Contact sensor per D4. S1 and S6 should turn green. If S1 reads zero filtered force, switch to the analytic fallback and report it.

**Gate 4:** S1, S4, S5 and S6 are green. Build time and memory are recorded.
**Stop if** the 1024-env build takes over 5 minutes or leaves under 8 GB of GPU memory free. Report the numbers before optimizing.

### Phase 5: Env logic

1. Observations per D5. S8 should turn green.
2. Actions per D6.
3. Resets and dones per D7. S2, S3 and S7 should turn green.
4. Rewards per D8. Log each reward term separately in `extras`.

**Gate 5:** S1 through S8 are all green. The smoke script runs clean at 1 and 64 envs.

### Phase 6: Training

1. Short run:
   ```bash
   cd ~/robot-learning/source/my_robot_tasks
   ~/robot-learning/IsaacLab/isaaclab.sh -p scripts/train.py --task Isaac-BARN-Nav-Direct \
     --headless --num_envs 256 --max_iterations 50
   ```
   Check four things:
   - No NaN in the losses or observations.
   - All three end causes (success, collision, timeout) appear in the logs.
   - Each reward term has a plausible magnitude.
   - Record steps per second.
2. Full run with `--num_envs 1024`, so each train world gets at least 4 envs. Take the iteration count from the runner cfg.

**Gate 6:** Training completes. The success rate on train worlds rises over the run; attach the curve.
**Stop if** learning stays flat. Report the curves and per-term reward magnitudes. Do not tune rewards without approval.

### Phase 7: Evaluation with upstream parity

1. Before evaluating any policy, establish a pipeline anchor: `scripts/eval_barn.py --policy path_follower`.
   - Pure pursuit on `path_{i}.npy` at 1.0 m/s with a 0.5 m lookahead.
   - Expect high success, with successful runs scoring just under the 0.5 ceiling.
   - Failures clustered at the same spot point to a coordinate or footprint bug. Fix that before trusting any policy score.
2. Evaluate the policy with `scripts/eval_barn.py --checkpoint <path> --split test`:
   - Pin env i to test world i mod 50.
   - Use the deterministic policy (mean action).
   - Run one episode per env.
3. Write one line per episode in the upstream `out.txt` format: `world_idx success collided timeout time nav_metric`. Read upstream `report_test.py` and match its aggregation exactly.
4. Trials: a deterministic policy with a deterministic reset produces 10 identical trials. Either add documented reset noise or report 1 trial. Never report 10 identical runs as 10 samples.

**Gate 7:** The path follower and the trained policy are both evaluated on all 50 test worlds. Two eval runs reproduce the same numbers.

### Phase 8: DynaBARN (needs Pan's approval first)

- Blocker: obstacle motion is compiled into `easylibobs_N.so`, and the repo has no source for it.
- Data path:
  - Run each DynaBARN world in the upstream Gazebo setup (the repo ships a `Singularityfile.def`).
  - Log the `easylibobs_*` poses at 50 Hz for 110 s and save one `npz` per world.
  - Record each world twice to test whether the motion is time-deterministic.
- Isaac side:
  - Replay the trajectories on kinematic rigid bodies. A dynamic robot against kinematic obstacles does generate contacts.
  - Lidar: use `MultiMeshRayCaster` if it exists. Otherwise extend `analytic_lidar` to handle boxes.
- Metric: settle `dynabarn_parity` with Pan before reporting any numbers.

### Phase 9: Articulated Jackal (optional, after Gate 7)

- Convert the Jackal URDF with Isaac Lab's URDF converter (confirm the script path under `scripts/tools/`).
  - Four velocity-controlled wheels.
  - Skid-steer with an effective separation of 0.37559 × 1.5.
- Compare against the phase A body: replay the same command sequence and plot trajectories and velocity tracking error.
- Then rerun Gate 7 with the unchanged phase A policy.

## 7. Reporting

- Append to `NOTES.md` after each phase:
  - Commits.
  - Gate result with a summary of the test output.
  - Each "verified" claim with its oracle.
  - Open questions.
- A stop report contains four parts:
  - Where the work stopped.
  - The hypotheses tried and their results.
  - The evidence (logs and numbers).
  - The proposed next step.

---

## Before you hand this off

Three decisions in this plan are mine, not upstream's. They are the assumptions most likely to be wrong:

- **D1: one fixed world per env.** It makes the stock RayCaster workable and keeps physics cheap. The cost is that diversity comes only from the env count.
- **D2: a velocity-driven body.** It matches BARN's `cmd_vel` interface but skips wheel slip and tracking lag. Phase 9 is where that gap surfaces.
- **D5: the observation layout.** It is a reasonable default, not something upstream specifies.

D1 is the one most worth testing. What would you measure to tell whether fixed worlds per env hurt generalization compared with resampling worlds on every reset, and what result would change your mind?