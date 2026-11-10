# TensorBoard

```bash
make tb
```

Serves TensorBoard over all runs in `logs/`. Default bind address is all interfaces; open `http://localhost:6061` (or the port the Makefile prints).

Logs and checkpoints live under:

```
logs/rsl_rl/<experiment>/<timestamp>/
```

Clear everything with:

```bash
make clean-logs
```
