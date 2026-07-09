# Thesis RL Codebase

Modular codebase for MetaDrive-based reinforcement learning experiments, with:

- thesis runtime code under `src/`
- Hydra configuration under `conf/`
- editable third-party dependencies under `third_party/`
- Docker/Compose as the primary reproducible workflow

## Repository Layout

```text
conf/
docs/
  architecture/
  archive/
  setup/
  specs/
  workflows/
scripts/
src/
tests/
third_party/
  metadrive/
  scenarionet/
  stable-baselines3/
```

`third_party/` is part of the repository layout. The expected workflow is a
clean clone plus submodule initialization, not sibling checkouts outside the
repo tree.

## Clone And Initialize

```bash
git clone --recurse-submodules <your-repo-url>
cd thesis-metadrive
```

If you already cloned the repo without submodules:

```bash
git submodule update --init --recursive
```

## Dependency Model

The project currently uses local editable dependencies for:

- `third_party/metadrive`
- `third_party/stable-baselines3`

`third_party/scenarionet` is included and prepared in the workspace layout for
future dataset integration. The current phase does not yet implement the full
ScenarioNet training pipeline inside `thesis_rl`.

For Docker builds, `torch` comes from the NVIDIA PyTorch base image and is
excluded from `uv`-managed dependencies.

## Requirements

The repository is Docker-first and currently optimized for Linux/NVIDIA hosts.
Some requirements are explicit in the repo, while hardware sizing is partly an
engineering estimate based on the default configs and submodules.

### Explicit Requirements

- Linux or WSL2 is the intended host environment.
- Docker Engine must be installed and reachable.
- Docker Compose v2 must be available as `docker compose`.
- Git submodules are required because `third_party/` is part of the repo layout.
- If you work outside the container, Python must be `>=3.10,<3.11`.
- The container image is `nvcr.io/nvidia/pytorch:24.01-py3`.

### NVIDIA Requirements

- The default container workflow expects an NVIDIA-capable Docker host.
- `compose.yaml` uses `gpus: all`, `ipc: host`, and `network_mode: host`.
- The NVIDIA PyTorch `24.01` base image uses CUDA `12.3.2`.
- According to the official NVIDIA release notes, that container requires
  NVIDIA driver `545+` in general.
- For some data center GPUs, NVIDIA documents compatibility with
  `470.57+`, `525.85+`, `535.86+`, or `545.23+`.

Reference:
- [NVIDIA PyTorch Release 24.01](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-01.html)

### Minimum Recommended Machine

- Linux or WSL2
- Docker Engine + `docker compose` v2
- Git with submodule support
- NVIDIA GPU with working `nvidia-smi`
- NVIDIA Container Toolkit / Docker GPU runtime available
- `16 GB` RAM
- `4 GB` VRAM
- `20-30 GB` free disk
- Internet access for image pulls and dependency resolution during build

### Comfortable Recommended Machine

- Linux natively
- Recent Docker Engine and Compose v2
- NVIDIA GPU with recent driver compatible with CUDA `12.3.2`
- `32 GB` RAM
- `8 GB+` VRAM
- `50+ GB` free disk
- Modern multi-core CPU

### RAM Notes

- The default config uses `obs=semantic_state`, vectorized envs with
  `num_envs=5`, and TD3/SAC presets commonly use a replay buffer of `300000`.
- The replay buffer stores both `obs` and `next_obs` in `float32`.
- With the default `semantic_state` observation budget, that is roughly
  `5.3 GiB` of RAM just for replay-buffer observation storage before accounting
  for the rest of Python, PyTorch, the simulator, logs, and the OS.
- Because of that, `16 GB` is a realistic floor and `32 GB` is much safer for
  normal development and experiments.

### Optional / Feature-Specific Notes

- The core repo path uses `use_render: false`, so on-screen rendering is not a
  base requirement for training.
- MetaDrive supports headless rendering, but its advanced CUDA image pipeline is
  a separate feature with extra OpenGL/CUDA requirements in the upstream docs.
- A CPU-only host might be adaptable with repo changes, but it is not the
  standard or documented workflow for this repository today.

## Default Runtime Paths

Inside the container, the portable defaults are:

- project root: `/workspace/thesis-metadrive`
- outputs root: `/workspace/outputs`
- data root: `/workspace/data`
- ScenarioNet data root: `/workspace/data/scenarionet`
- MetaDrive data root: `/workspace/data/metadrive`

On your remote VM, you can still point host-mounted paths to `/scratch/...`,
but `/scratch` is no longer required by default.

## Configure Local Paths

`docker compose` reads `.env` automatically. Start from the template:

```bash
cp .env.example .env
```

Or use the repository bootstrap helper:

```bash
./setup.sh
```

`setup.sh` creates `.env` if missing, ensures the host mount directories
exist, initializes submodules, checks UID/GID, Docker/Compose, and basic
Linux/NVIDIA compatibility, then runs `docker compose build`, a container
import smoke check, and `pytest` by default. It exits non-zero when it finds a
blocking issue. Use `./setup.sh --skip-build --skip-smoke-check --skip-pytest`
if you only want the bootstrap and preflight checks.

Important variables:

- `HOST_OUTPUTS_DIR`
- `HOST_DATA_DIR`
- `HOST_CONTAINER_HOME_DIR`
- `HOST_UID`
- `HOST_GID`
- `OUTPUTS_ROOT`
- `DATA_ROOT`
- `SCENARIONET_DATA_ROOT`
- `METADRIVE_DATA_ROOT`

Portable local defaults in `.env.example` use repo-relative host directories:

- `./outputs`
- `./data`
- `./.container-home`

If you want your remote VM to use `/scratch`, just change the host-side values
in `.env`, for example:

```env
HOST_OUTPUTS_DIR=/scratch/your_user/thesis-metadrive/outputs
HOST_DATA_DIR=/scratch/your_user/thesis-metadrive/data
HOST_CONTAINER_HOME_DIR=/scratch/your_user/container-home
```

Rule of thumb:

- customize host-side variables in `.env`
- keep container-side `/workspace/...` paths stable
- never commit `.env`, only `.env.example`

For the full per-machine setup procedure, see
[docs/setup/environment_setup.md](docs/setup/environment_setup.md).
For a short clone-to-usable-machine checklist, see the
`New Machine Bring-Up Checklist` section in that document.

## Docker Workflow

This repository is optimized for Docker-based development and runs.

### 1. Build And Start

```bash
docker compose up -d --build
```

### 2. Enter The Container

```bash
docker compose exec dev bash
```

### 3. Install / Refresh The Environment

Inside the container:

```bash
uv sync --extra dev
```

### 4. Run Commands

Examples:

```bash
uv run --no-sync python -m pytest -q
uv run --no-sync python -m thesis_rl.cli.train
uv run --no-sync python -m thesis_rl.cli.evaluate
```

### 5. Verify End-To-End Training

After `./setup.sh` has passed, the quickest manual confirmation that the repo
really works end-to-end is a smoke training run:

```bash
docker compose up -d
docker compose exec dev bash
uv run --no-sync python -m thesis_rl.cli.train --config-name presets/test/smoke_train
```

After the run, check that a new run directory appeared under `outputs/` and
contains at least:

- `logs/`
- `csv/`
- `checkpoints/`
- `artifacts/`

That is the practical "it really runs on this machine" confirmation after the
build/import/tests phase.

## Parallel Runs

For one-pane-per-seed tmux launches:

```bash
scripts/tmux_seed_grid.sh \
  --session smoke_alg_sac \
  --docker-container thesis-metadrive-dev -- \
  uv run --no-sync python -m thesis_rl.cli.train \
    --config-name presets/agent/sac_sb3 \
    run_profile=smoke \
    reward=monitor_only \
    curriculum=disabled
```

Helper scripts under `scripts/` now use `OUTPUTS_ROOT` inside the container
instead of assuming `/scratch/...`.

## Notes On Portability

- `compose.yaml` keeps Linux/NVIDIA-oriented settings such as `gpus: all`,
  `ipc: host`, and `network_mode: host` intentionally.
- Those settings are good defaults for the target research environment, but
  they are not meant to imply universal compatibility on every host OS.
- Host-side paths should always be changed through `.env`, not by hardcoding
  machine-specific paths in source files.

## Documentation

This `README.md` is the canonical entry point for cloning, configuring, and
running the repository. `docs/README.md` is only a lightweight map of the
documentation folder.

Useful follow-up documents:

- [docs/README.md](docs/README.md)
- [docs/setup/comparison_run_commands.md](docs/setup/comparison_run_commands.md)
- [docs/workflows/algorithm_selection_playbook.md](docs/workflows/algorithm_selection_playbook.md)
- [docs/setup/validation_commands.md](docs/setup/validation_commands.md)
- [docs/architecture/sb3_fork_migration_plan.md](docs/architecture/sb3_fork_migration_plan.md)
