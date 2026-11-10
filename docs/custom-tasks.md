# Custom tasks (external package)

Custom Isaac Lab tasks live in a **separate external Python package**, not inside the untouched `IsaacLab/` submodule.

## Structure

```
my-tasks/
  pyproject.toml      # declares isaaclab.tasks entry point
  src/my_tasks/
    __init__.py       # gym.register(id=..., entry_point=..., kwargs={...})
    tasks.py          # env configs, rsl_rl cfg, default agent
```

`pyproject.toml` entry point:

```toml
[project.entry-points."isaaclab.tasks"]
my-task = "my_tasks.tasks"
```

The module's `__init__.py` calls:

```python
gym.register(
    id="MyTask-v0",
    entry_point=...,
    kwargs={env_cfg_entry_point, rsl_rl_cfg_entry_point, default_agent},
)
```

## Install & run

```bash
# install the external package editable into the uv environment
uv pip install -e /path/to/my-tasks   # or add to uv sync extras

# run via the robot-learning Makefile
make train TASK=MyTask-v0
make play  TASK=MyTask-v0
```

The `isaaclab train/play` CLI calls `load_external_tasks()`, which loads all `isaaclab.tasks` entry points before resolving the task ID from the Gym registry.

The Makefile maps `TASK`/`PHYSICS`/`RL`/`NUM_ENVS`/`VIZ`/`ARGS` directly to the underlying `isaaclab` command, so no Makefile changes are needed.

The repo's `my_robot_tasks/` package is itself an example of this pattern — it registers four LIBERO-style manipulation tasks (`Libero-PickCube-Direct`, `Libero-PlaceCube-Direct`, `Libero-TransferCube-Direct`, `Libero-PushCube-Direct`) via the `isaaclab.tasks` entry point and runs them with:

```bash
make train TASK=Libero-PickCube-Direct
```
