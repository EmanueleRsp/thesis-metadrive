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

Important variables:

- `HOST_OUTPUTS_DIR`
- `HOST_DATA_DIR`
- `HOST_CONTAINER_HOME_DIR`
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

Useful entry points:

- [docs/README.md](docs/README.md)
- [docs/comparison_run_commands.md](docs/comparison_run_commands.md)
- [docs/algorithm_selection_playbook.md](docs/algorithm_selection_playbook.md)
- [docs/validation_commands.md](docs/validation_commands.md)
- [docs/sb3_fork_migration_plan.md](docs/sb3_fork_migration_plan.md)
