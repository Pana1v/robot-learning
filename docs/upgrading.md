# Upgrading Isaac Lab

The `IsaacLab/` submodule is pinned to v3.0.0 and is left untouched. Upgrade via git in the submodule:

```bash
cd IsaacLab
git fetch --tags
git checkout v3.0.1   # or whatever is newest
cd ..
make sync             # or make sync-sim
```

Isaac Lab 2.x does not run on Isaac Sim 6.x and vice versa — match versions when upgrading.

## Upgrading the robot-learning wrapper

The wrapper itself (Makefile, pyproject.toml, README, host scripts) is versioned in this repo. Bump `version` in `pyproject.toml` and commit when you make breaking changes.
