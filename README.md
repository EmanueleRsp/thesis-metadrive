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

For Docker builds, the repository starts from the official minimal Python
3.10 image. PyTorch 2.8 is installed from the backend selected per machine by
`TORCH_BACKEND` and excluded from the platform-neutral uv lock.

Dataset preparation uses the separate `dataset-pipeline` Compose service. It
shares the pre-PyTorch Docker layers with `dev` and does not install PyTorch or
CUDA; the Waymo conversion remains isolated in the TensorFlow-only
`waymo-converter` service.

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
- The base image is the digest-pinned official `python:3.10.20-slim-bookworm`.

### NVIDIA Requirements

- CPU-only checks use `compose.yaml`; GPU runs add `compose.gpu.yaml`.
- Host networking is opt-in through `compose.linux-host.yaml`.
- `.env` selects `TORCH_BACKEND=cpu`, `cu126`, or `cu128`.
- `cu128` is the default for modern NVIDIA GPUs and is required for Blackwell
  (`sm_120`, including RTX 50xx); `cu126` is the legacy NVIDIA option.
- GPU execution requires a compatible NVIDIA driver and the NVIDIA Container
  Toolkit. CPU-only checks can use `TORCH_BACKEND=cpu`.

Choose before building:

| Machine | `.env` value | Compose command |
|---|---|---|
| CPU / CI | `TORCH_BACKEND=cpu` | `docker compose ...` |
| Legacy NVIDIA | `TORCH_BACKEND=cu126` | add `compose.gpu.yaml` |
| Modern NVIDIA / RTX 50xx | `TORCH_BACKEND=cu128` | add `compose.gpu.yaml` |

References:
- [Official Python Docker image](https://hub.docker.com/_/python)
- [PyTorch installation guide](https://pytorch.org/get-started/locally/)

### Minimum Recommended Machine

- Linux or WSL2
- Docker Engine + `docker compose` v2
- Git with submodule support
- NVIDIA GPU with working `nvidia-smi`
- NVIDIA Container Toolkit / Docker GPU runtime available
- `16 GB` RAM
- `4 GB` VRAM
- `12-20 GB` free disk
- Internet access for image pulls and dependency resolution during build

### Comfortable Recommended Machine

- Linux natively
- Recent Docker Engine and Compose v2
- NVIDIA GPU with a driver compatible with the selected Torch backend
- `32 GB` RAM
- `8 GB+` VRAM
- `30+ GB` free disk
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

`setup.sh` creates `.env` if missing, fills UID/GID from the current Linux
user, creates host mount directories, initializes submodules, and runs the
portable preflight checks. Use `./setup.sh --verify` for build, imports and
pytest, or `./setup.sh --verify --gpu` to additionally verify CUDA inside the
container.

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

CPU-only development/checks:

```bash
docker compose up -d --build
```

Linux/NVIDIA development and experiments:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml up -d --build
```

The equivalent shortcuts are `make up` and `make up-gpu`. Use `make gpu-check`
to execute a real CUDA tensor operation, not just detect the device.

### 2. Enter The Container

```bash
docker compose exec dev bash
```

### 3. Dependencies

The image already contains the frozen runtime and development dependencies.
Do not run `uv sync` interactively: after changing `pyproject.toml` or
`uv.lock`, rebuild with `docker compose build` (or `make build`). This keeps
the container environment immutable and reproducible.

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
    --docker-compose-service dev -- \
  uv run --no-sync python -m thesis_rl.cli.train \
    --config-name presets/agent/sac_sb3 \
    run_profile=smoke \
    reward=monitor_only \
    curriculum=disabled
```

Helper scripts under `scripts/` now use `OUTPUTS_ROOT` inside the container
instead of assuming `/scratch/...`.

## Notes On Portability

- `compose.yaml` is the portable CPU-capable base.
- `compose.gpu.yaml` adds NVIDIA GPU access and host IPC.
- `compose.linux-host.yaml` separately enables host networking when needed.
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
