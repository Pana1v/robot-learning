# Career switch: robotics & AI learning path

This project is a deliberate skill-building vehicle for a career switch into robotics and AI. Every artifact here maps to target roles and companies:

| Target role | Company | Skills this project builds |
|-------------|---------|---------------------------|
| Software Engineer | Applied Materials (C++, OS, networking, multi-threading) | uv packaging, Makefile task runner, Docker GPU runtime, headless simulation, Python/C++ boundary |
| Embedded firmware engineer | Analog Devices (C/C++, RTOS, Zephyr/FreeRTOS) | Real-time control loop structure, deterministic step/actuate/reset cycle, sensor I/O patterns, embedded-style resource constraints (8 GB VRAM, 21 GB RAM) |
| AI/ML Engineer | HARMAN Automotive (SLM/VLA/VLM, edge deployment, Python+C++) | Quantized VLA inference (4-bit GGUF/EXL2), LeRobot dataset pipeline, RL policy training (rsl_rl), local inference on RTX 5050, edge deployment constraints |
| Perception engineer | Qualcomm XR Research (SLAM, sensor fusion, C/C++, Isaac Sim) | Camera sensor config (PinholeCameraCfg, RGB streams), Isaac Sim backend path (physx), Isaac Lab sensor framework, multi-modal observation (vision + proprioception) |
| Simulation Platform engineer | Applied Intuition (C++/Python, 4+ yrs, autonomy) | Physics backend abstraction (mjwarp / physx / ovphysx), MuJoCo-Warp + Newton, task registry via `isaaclab.tasks` entry points, simulation config management |
| Research Scientist — World Models | Universal Robots (sim-to-real, VLA, research tier) | VLA pipeline (demo collection → LeRobot → remote fine-tune → local inference), smol-vLA / OpenVLA-small, LoRA/QLoRA fine-tuning strategy, sim-to-real transfer via domain randomization |
| Robotics engineer | UR / MiR collaborator (ROS2, Python, C++, PyTorch, Isaac Sim) | DirectRLEnv / manager-based env design, PyTorch tensor operations, Isaac Lab API, ROS2-ready action/observation contracts, Isaac Sim integration path |

## Learning priorities (in order)

1. **C++** — underlying Isaac Lab / PhysX / Newton internals, sensor plugins, actuator models
2. **CUDA** — MuJoCo-Warp kernel compilation, GPU memory budgeting (8 GB VRAM), parallel env scaling (`NUM_ENVS` → `nvidia-smi` ~7 GB)
3. **Robot learning** — RL policies (rsl_rl PPO), VLA inference (quantized smol-vLA / OpenVLA-small), LeRobot data format, sim-to-real

## Project artifacts as portfolio evidence

| Artifact | Demonstrates |
|----------|-------------|
| `my_robot_tasks/` — external task package with `isaaclab.tasks` entry point | Python packaging, plugin architecture, Gym registry integration |
| `my_robot_tasks/envs/libero_env.py` — `DirectRLEnv` with HDF5 demo recording | RL environment design, control loop, data collection pipeline |
| `my_robot_tasks/cfg/libero_env_cfg.py` — task config hierarchy (`@configclass`) | Configuration-driven simulation, reward shaping, reset logic |
| `scripts/collect_demos.py` — demo collection CLI | Automation, argument parsing, structured data output |
| VLA pipeline docs — demos → remote fine-tune → local inference | End-to-end ML systems design, edge/remote split, checkpoint staging |
| `host/power.sh`, `host/setup-host.sh` — host provisioning | Linux systems, Docker GPU runtime, swap tuning, performance profiling |
| Makefile — thin front end over `uv` / docker commands | Build automation, task runner design, reproducible environments |
| `mkdocs.yml` + `docs/` — mkdocs-material site | Technical writing, documentation architecture |

## How to use this in interviews

- **"Tell me about a project."** Walk through: custom Isaac Lab tasks → demo collection → LeRobot conversion → remote VLA fine-tuning → local quantized inference on a laptop GPU. Emphasize the hardware constraint design (8 GB VRAM → remote training, local inference).
- **"Python vs C++."** Point to the Python task/env configs and note where C++ would enter (Isaac Lab backend, sensor plugins, CUDA kernels).
- **"Edge deployment."** Discuss quantized VLA inference on RTX 5050, VRAM budgeting, and the remote/local split for training vs inference.
- **"Systems."** Discuss uv dependency management, Docker GPU runtime, swap configuration, and Makefile-based reproducibility.

## Next steps for career readiness

- [ ] Add a C++ component (e.g., a custom Isaac Lab sensor plugin or actuator model in C++).
- [ ] Profile CUDA kernels with `nvidia-smi` / Nsight and document VRAM vs `NUM_ENVS` scaling.
- [ ] Run a quantized VLA locally (smol-vLA 4-bit GGUF) and measure inference latency vs control frequency.
- [ ] Publish a fine-tuned VLA checkpoint to Hugging Face Hub from Colab/Kaggle training.
- [ ] Add ROS2 bridge documentation (Isaac Lab → ROS2 action/observation topics) for UR/MiR collaborator roles.
