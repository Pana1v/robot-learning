# Host setup (one-time sudo)

`make host` runs `sudo ./host/setup-host.sh`, which:

1. Creates a 20 GB swap file (helps when 21 GB RAM is tight under Isaac Sim).
2. Installs and configures the NVIDIA Docker GPU runtime.

Run once:

```bash
make host
```

Re-run `make doctor` afterward to verify GPU, swap, Docker, and Python package versions.
