# Docker fallbacks

If the native pip Isaac Sim misbehaves on Ubuntu 26.04 (only officially tested on 22.04/24.04), use the official containers.

## Kit-less container

```bash
make docker-kitless   # build + start upstream kit-less container
make docker-enter P=kitless
make docker-stop P=kitless
```

## Isaac Sim 6.2 container

```bash
make docker-sim       # builds on nvcr.io/nvidia/isaac-sim:6.2.0
make docker-enter P=base
make docker-stop P=base
```

Commands run from `IsaacLab/docker/container.py`.
