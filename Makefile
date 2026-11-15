# Thin front end over upstream Isaac Lab 3.0 (uv based).
# Every target just prints and runs the real command, so you can always
# copy the underlying uv / docker call and drop the Makefile.
#
#   make help
#   make train TASK=Isaac-Velocity-Flat-UnitreeGo2 PHYSICS=mjwarp NUM_ENVS=2048

SHELL := /bin/bash
.DEFAULT_GOAL := help

LAB      := $(CURDIR)/IsaacLab
# --project lets us run from this folder without cd'ing into the checkout
UV       := uv run --project $(LAB)
# Direct interpreter, for the scripts that must not trigger a uv sync.
PY       := $(LAB)/.venv/bin/python

# ----------------------------------------------------------------- knobs
TASK     ?= Isaac-Cartpole-Direct
PHYSICS  ?= mjwarp          # mjwarp | physx | ovphysx
RL       ?= rsl_rl          # rsl_rl | skrl | sb3 | rl_games (non-rsl need: make sync EXTRAS=all)
NUM_ENVS ?=                 # empty = task default
VIZ      ?=                 # empty = headless (fastest). newton_gl | kit | viser | rerun
RENDERER ?=                 # camera renderer, empty = task default. newton_renderer | isaacsim_rtx | ovrtx
CKPT     ?= latest
WORLD    ?= 0               # BARN static world index, for 'make render'
STEPS    ?= 300             # steps to run, for 'make smoke'
ARGS     ?=                 # anything else, passed straight to isaaclab

# Map the short backend name to the Isaac Lab selector and the uv extra it needs.
# PHYSICS= (empty) passes no selector at all, so the task keeps its own default.
ifeq ($(strip $(PHYSICS)),)
  PHYS_SEL :=
  PHYS_EXTRA :=
else ifeq ($(strip $(PHYSICS)),mjwarp)
  PHYS_SEL := physics=newton_mjwarp
  PHYS_EXTRA :=
else ifeq ($(strip $(PHYSICS)),physx)
  PHYS_SEL := physics=isaacsim_physx
  PHYS_EXTRA := --extra isaacsim
else ifeq ($(strip $(PHYSICS)),ovphysx)
  PHYS_SEL := physics=ovphysx
  PHYS_EXTRA := --extra ovphysx
else
  $(error PHYSICS must be empty, mjwarp, physx or ovphysx)
endif

# A task whose cfg declares no physics preset rejects a physics= selector outright:
# "Typed preset selector(s) physics=... did not match a selectable preset of that
# type for this task." Their backend is fixed by the task itself
# (SimulationCfg.physics = None -> PhysX), so the selector is dropped and the
# matching uv extra is used instead. PHYS_NOTE keeps that visible.
NO_PHYSICS_PRESET := Isaac-BARN-Nav-Direct MyRobot-Cartpole-Direct
ifneq ($(filter $(strip $(TASK)),$(NO_PHYSICS_PRESET)),)
  PHYS_SEL :=
  PHYS_EXTRA := --extra isaacsim
  PHYS_NOTE := $(TASK) declares no physics preset, using its own default (PhysX) instead of PHYSICS=$(strip $(PHYSICS))
endif

REND_SEL := $(if $(strip $(RENDERER)),renderer=$(strip $(RENDERER)))

COMMON := --rl_library $(strip $(RL)) --task $(strip $(TASK)) \
          $(if $(strip $(NUM_ENVS)),--num_envs $(strip $(NUM_ENVS))) \
          $(if $(strip $(VIZ)),--viz $(strip $(VIZ)))

# uv sync is exact and removes packages not in IsaacLab's lockfile, including our task
# package. Without its entry point, isaaclab train cannot find Libero-*/BARN tasks. Re-add it after every sync.
TASKS_INSTALL := uv pip install --python $(LAB)/.venv/bin/python --no-deps -e $(CURDIR)

# Isaac Sim asks for the EULA interactively on first launch; accept it up front
export OMNI_KIT_ACCEPT_EULA := YES
export ACCEPT_EULA := Y

##@ Setup
.PHONY: host sync sync-sim doctor

host: ## one-time sudo setup: 20 GB swap + Docker GPU runtime
	sudo ./host/setup-host.sh

sync: ## install kit-less Isaac Lab (Newton + MuJoCo-Warp, PyTorch cu130). EXTRAS="all video" to add more
	cd $(LAB) && uv sync --extra video $(foreach e,$(EXTRAS),--extra $(e))
	$(TASKS_INSTALL)

clean-logs: ## clear all local checkpoints and tensorboard logs
	rm -rf logs/*

check-cuda: ## quick verification of CUDA GPU runtime & memory status
	@nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv

sync-sim: ## add Isaac Sim 6.2 (pip, ~15 GB) on top of the kit-less env
	cd $(LAB) && uv sync --extra video --extra isaacsim $(foreach e,$(EXTRAS),--extra $(e))
	$(TASKS_INSTALL)
	test -e host/compat-lib/libxml2.so.2 || ./host/compat-libs.sh

doctor: ## check GPU, power, swap, docker, torch/warp/mujoco versions, task package
	@./host/power.sh status; echo
	@echo "Swap          : $$(swapon --show=SIZE --noheadings | tr -d ' ')"
	@echo "Docker GPU    : $$(docker info 2>/dev/null | grep -q 'Runtimes:.*nvidia' && echo ok || echo 'not set up, run: make host')"
	@$(UV) --no-sync python -c "import torch, warp, mujoco, newton; \
	print('torch        :', torch.__version__, 'cuda ok' if torch.cuda.is_available() else 'NO CUDA', torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''); \
	print('warp         :', warp.__version__); print('mujoco       :', mujoco.__version__); print('newton       :', newton.__version__)" 2>/dev/null \
	  || echo "Python env    : not installed yet, run: make sync"
	@$(UV) --no-sync python -c "import isaacsim; print('isaacsim     : installed')" 2>/dev/null || echo "isaacsim      : not installed (make sync-sim)"
	@$(UV) --no-sync python -c "import importlib.metadata as m, importlib.util as u; \
	eps=[e for e in m.entry_points(group='isaaclab.tasks') if e.name=='my-robot-tasks']; \
	print('task package  :', 'my-robot-tasks registered' if eps and u.find_spec('my_robot_tasks') else 'NOT REGISTERED, run: make sync')" 2>/dev/null \
	  || echo "task package  : not installed yet, run: make sync"
	@test -f data/barn_static.npz && echo "barn data     : ok" || echo "barn data     : missing, run: make data"

##@ Train and evaluate
.PHONY: envs envs-all train play tb power

envs: ## list this project's tasks and which physics/renderer presets each supports
	$(UV) --no-sync isaaclab list_envs --show_presets $(ARGS)

envs-all: ## list every registered Isaac Lab task, not just this project's
	$(UV) --no-sync isaaclab list_envs --show_presets --all

train: ## train TASK with PHYSICS backend, headless unless VIZ is set
	$(if $(strip $(PHYS_NOTE)),@echo "[make] $(PHYS_NOTE)")
	$(UV) $(PHYS_EXTRA) isaaclab train $(COMMON) $(ARGS) $(PHYS_SEL) $(REND_SEL)

play: ## replay the latest (or CKPT=path) policy with a viewer
	$(if $(strip $(PHYS_NOTE)),@echo "[make] $(PHYS_NOTE)")
	$(UV) $(PHYS_EXTRA) isaaclab play $(COMMON) \
	  $(if $(strip $(VIZ)),,--viz newton_gl) --checkpoint $(strip $(CKPT)) $(ARGS) $(PHYS_SEL) $(REND_SEL)

tb: ## tensorboard over all runs
	$(UV) --no-sync tensorboard --logdir $(CURDIR)/logs --bind_all

power: ## performance profile + GPU status (sudo ./host/power.sh boost for Omen TGP)
	./host/power.sh train

##@ BARN task
.PHONY: test smoke render data asset

test: ## run the BARN unit tests (no Isaac Sim needed, ~10 s)
	$(PY) -m unittest discover -s source/my_robot_tasks/tests/unit -t source/my_robot_tasks/tests/unit -v

smoke: ## boot the BARN env headless, reset and step it (needs Isaac Sim)
	$(UV) --no-sync python scripts/smoke_barn.py --visualizer none \
	  $(if $(strip $(NUM_ENVS)),--num_envs $(strip $(NUM_ENVS))) --steps $(strip $(STEPS)) $(ARGS)

render: ## draw BARN world WORLD to PNG (numpy + matplotlib only, no Isaac Sim)
	$(PY) scripts/render_barn.py --world $(strip $(WORLD)) $(ARGS)

data: ## rebuild data/barn_static.npz from third_party/the-barn-challenge
	@test -d third_party/the-barn-challenge || { echo "missing third_party/the-barn-challenge (clone Daffan/the-barn-challenge@bf5a226)"; exit 1; }
	cd $(CURDIR) && $(PY) source/my_robot_tasks/scripts/build_barn_data.py

asset: ## rebuild the Jackal visual USD from the upstream Isaac Sim asset
	$(PY) scripts/build_jackal_asset.py

##@ Isaac Sim
.PHONY: sim compat-libs

# Experience file. isaacsim.exp.full.newton swaps PhysX for Newton physics in the GUI.
SIM_EXP ?= isaacsim.exp.full

# The desktop runs on the AMD iGPU (PRIME on-demand). Without these, the viewport is rendered on
# the RTX 5050 and then copied to the iGPU every frame. NVIDIA_only also stops Vulkan from
# enumerating the iGPU as a second device.
# host/compat-lib holds libxml2.so.2 + ICU 74 from Ubuntu 24.04 (make compat-libs), which 26.04 dropped.
SIM_ENV := LD_LIBRARY_PATH=$(CURDIR)/host/compat-lib$${LD_LIBRARY_PATH:+:$$LD_LIBRARY_PATH} __NV_PRIME_RENDER_OFFLOAD=1 __VK_LAYER_NV_optimus=NVIDIA_only __GLX_VENDOR_LIBRARY_NAME=nvidia

# Viewport performance on 8 GB VRAM. These only affect the RTX viewport, not physics. Shadows and
# direct lighting stay on so depth cues remain readable. SIM_QUALITY=1 drops all of these.
#   reflections / translucency / indirect diffuse / AO: the costly ray-traced passes
#   dlssg: DLSS frame generation (Blackwell). GUI only: Kit disables it because it breaks SDG captures
#   useFabricSceneDelegate: render straight from Fabric, skipping the USD -> Hydra sync each frame
SIM_FAST := \
	--/rtx/reflections/enabled=false \
	--/rtx/translucency/enabled=false \
	--/rtx/indirectDiffuse/enabled=false \
	--/rtx/ambientOcclusion/enabled=false \
	--/rtx/directLighting/sampledLighting/enabled=true \
	--/rtx/post/dlss/execMode=0 \
	--/rtx-transient/dlssg/enabled=true \
	--/app/useFabricSceneDelegate=true \
	--/rtx/hydra/readTransformsFromFabricInRenderDelegate=true

compat-libs: ## fetch libxml2.so.2 + ICU 74 that Isaac Sim needs on Ubuntu 26.04 (no sudo)
	./host/compat-libs.sh

sim: ## open the Isaac Sim 6.2 GUI tuned for this laptop. SIM_QUALITY=1 for stock look, SIM_EXP=isaacsim.exp.full.newton for Newton physics
	$(SIM_ENV) $(UV) --extra isaacsim isaacsim $(SIM_EXP) $(if $(SIM_QUALITY),,$(SIM_FAST)) $(ARGS)

##@ Docker (fallback if the native install misbehaves on Ubuntu 26.04)
.PHONY: docker-kitless docker-sim docker-enter docker-stop

docker-kitless: ## build + start the upstream kit-less container
	cd $(LAB) && python3 docker/container.py start kitless

docker-sim: ## build + start the upstream Isaac Sim 6.2 container (nvcr.io/nvidia/isaac-sim:6.2.0)
	cd $(LAB) && python3 docker/container.py start base

docker-enter: ## shell into a running container: make docker-enter P=kitless|base
	cd $(LAB) && python3 docker/container.py enter $(or $(P),base)

docker-stop: ## stop a container: make docker-stop P=kitless|base
	cd $(LAB) && python3 docker/container.py stop $(or $(P),base)

##@ Docs
.PHONY: docs docs-build docs-install

DOCS_VENV := $(CURDIR)/.venv-docs
DOCS_MKDOCS := $(DOCS_VENV)/bin/mkdocs

docs-install: ## install mkdocs + material theme into a dedicated venv
	test -d $(DOCS_VENV) || python3 -m venv $(DOCS_VENV)
	$(DOCS_VENV)/bin/pip install --upgrade pip
	$(DOCS_VENV)/bin/pip install mkdocs mkdocs-material

docs: ## serve the mkdocs documentation site locally (http://127.0.0.1:8000)
	$(DOCS_MKDOCS) serve

docs-build: ## build the static mkdocs site into site/
	$(DOCS_MKDOCS) build

##@ Help
.PHONY: help
help:
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage: make <target> [TASK=..] [PHYSICS=mjwarp|physx|ovphysx] [NUM_ENVS=..] [VIZ=..]\n"} \
	  /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2 } \
	  /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) }' $(MAKEFILE_LIST)
	@echo; echo "Current: TASK=$(strip $(TASK)) PHYSICS=$(strip $(PHYSICS)) RL=$(strip $(RL))"
	@echo; echo "PHYSICS : mjwarp = Newton/MuJoCo-Warp (default) | physx = Isaac Sim PhysX | ovphysx = standalone OVPhysX"
	@echo "          PHYSICS= (empty) keeps the task's own default."
	@echo "          Isaac-BARN-Nav-Direct and MyRobot-Cartpole-Direct declare no physics"
	@echo "          preset, so the selector is ignored and they run on their own default."
	@echo "TASKS   : Isaac-BARN-Nav-Direct (BARN nav) | MyRobot-Cartpole-Direct | Libero-*"
	@echo "          'make envs' lists this project's tasks, 'make envs-all' lists every task."
