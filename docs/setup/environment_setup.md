# Environment Setup

This note explains how to use `.env.example` and `.env` in a portable way
across different machines.

## Core Rule

The project should work like this:

```text
Each machine has its own .env.
The code always sees /workspace/outputs and /workspace/data.
Only .env decides where those files really live on the host.
```

So:

- commit only `.env.example`
- never commit `.env`
- on each new machine, copy `.env.example` to `.env`
- then edit only the host-specific variables

## `.env` Is Not `.venv`

- `.env` = project configuration and Docker mount configuration
- `.venv` = local Python environment

With the Docker workflow, `.venv` is not required.

## What You Usually Change

On a new machine, you usually only need to check:

```env
USER_NAME=appuser
HOST_UID=1000
HOST_GID=1000
TORCH_VERSION=2.9.1
TORCH_BACKEND=cu128

HOST_OUTPUTS_DIR=./outputs
HOST_DATA_DIR=./data
HOST_CONTAINER_HOME_DIR=./.container-home
```

Select `TORCH_BACKEND` before the first build:

- `cpu` for CI and machines without NVIDIA;
- `cu126` for legacy NVIDIA hosts;
- `cu128` for modern NVIDIA GPUs and Blackwell/RTX 50xx.

Changing it requires `docker compose build` because the backend is baked into
the image.

### `HOST_OUTPUTS_DIR`

Where runs, checkpoints, videos, logs, and analysis outputs live.

Local:

```env
HOST_OUTPUTS_DIR=./outputs
```

Remote VM:

```env
HOST_OUTPUTS_DIR=/scratch/your_user/thesis-metadrive/outputs
```

### `HOST_DATA_DIR`

Where datasets, ScenarioNet data, exported scenarios, and local data live.

Local:

```env
HOST_DATA_DIR=./data
```

Remote VM:

```env
HOST_DATA_DIR=/scratch/your_user/thesis-metadrive/data
```

### `HOST_CONTAINER_HOME_DIR`

Container home/cache directory for temporary files, `uv` cache, and local
configuration.

Local:

```env
HOST_CONTAINER_HOME_DIR=./.container-home
```

Remote VM:

```env
HOST_CONTAINER_HOME_DIR=/scratch/your_user/thesis-metadrive/.container-home
```

## What You Almost Never Change

These variables should remain stable:

```env
CONTAINER_HOME_DIR=/workspace/.container-home
CONTAINER_OUTPUTS_DIR=/workspace/outputs
CONTAINER_DATA_DIR=/workspace/data
OUTPUTS_ROOT=/workspace/outputs
DATA_ROOT=/workspace/data
SCENARIONET_DATA_ROOT=/workspace/data/scenarionet
METADRIVE_DATA_ROOT=/workspace/data/metadrive
```

These are in-container paths. The host may use `./outputs`, `/scratch/...`,
`/mnt/...`, or something else, but the code inside the container should always
see:

- `/workspace/outputs`
- `/workspace/data`

## UID And GID

On Linux/WSL:

```bash
id -u
id -g
```

If you get `1000` and `1000`, you can keep the defaults.

`USER_NAME` can remain `appuser`. UID/GID matter much more than the symbolic
username for file permissions.

## Machine Requirements

This repository is Docker-first and currently assumes a Linux/NVIDIA-oriented
host. Some requirements are explicit in the repo, while memory sizing is an
engineering estimate based on the default configs.

### Minimum Recommended Machine

- Linux or WSL2
- Docker Engine installed and reachable
- Docker Compose v2 available as `docker compose`
- Git with submodule support
- NVIDIA GPU with working `nvidia-smi`
- NVIDIA Container Toolkit / Docker GPU runtime
- `16 GB` RAM
- `4 GB` VRAM
- `12-20 GB` free disk

### Comfortable Recommended Machine

- Linux natively
- Recent Docker Engine and Compose v2
- NVIDIA GPU with a driver compatible with the selected Torch backend
- `32 GB` RAM
- `8 GB+` VRAM
- `30+ GB` free disk
- Modern multi-core CPU

### Runtime Image And Driver Note

The repository uses the digest-pinned official
`python:3.10.20-slim-bookworm` image and installs `torch==2.9.1` using the
backend selected in `.env`. Official Python 3.10 CUDA wheels are available for
Linux x86_64 and aarch64. GPU runs require a compatible host NVIDIA driver plus
the NVIDIA Container Toolkit. `cu128` supports Blackwell/RTX 50xx; CPU-only
setup uses `TORCH_BACKEND=cpu` and does not require an NVIDIA GPU. Keep the
backend explicit: a GPU compute capability alone cannot establish driver,
Python ABI, and wheel compatibility.

References:
- [Official Python Docker image](https://hub.docker.com/_/python)
- [PyTorch installation guide](https://pytorch.org/get-started/locally/)

### RAM Note

The default configs are not especially light on memory. In particular, the
standard TD3/SAC paths use `obs=semantic_state` together with a replay buffer
of `300000` transitions, which is roughly `5.3 GiB` of RAM just for replay
buffer observation storage before accounting for the simulator, PyTorch, Python,
logs, and the OS. That is why `16 GB` is a realistic floor and `32 GB` is much
safer for day-to-day use.

## Standard Procedure On A New Machine

```bash
git clone --recurse-submodules <repo-url>
cd thesis-metadrive
cp .env.example .env
```

Shortcut:

```bash
./setup.sh
```

The helper bootstraps `.env`, fills Linux UID/GID automatically, creates the
configured host directories, initializes submodules, and performs preflight
checks. Full verification is explicit:

```bash
./setup.sh --verify         # CPU-capable build, imports and tests
./setup.sh --verify --gpu   # also require and test NVIDIA/CUDA access
```

If you want only the bootstrap and preflight checks:

```bash
./setup.sh
```

Then check `.env`.

On Linux/WSL:

```bash
id -u
id -g
```

Then create the host directories.

Local:

```bash
mkdir -p outputs data .container-home
```

Remote VM:

```bash
mkdir -p /scratch/your_user/thesis-metadrive/outputs
mkdir -p /scratch/your_user/thesis-metadrive/data
mkdir -p /scratch/your_user/thesis-metadrive/.container-home
```

Then:

```bash
docker compose config
docker compose build
docker compose run --rm dev bash

# NVIDIA run
docker compose -f compose.yaml -f compose.gpu.yaml up -d
```

## New Machine Bring-Up Checklist

Use this checklist immediately after cloning the repository on any new machine.
It is intentionally short and focuses on setup and portability issues rather
than long-running experiments.

### 0. Clone And Initialize Submodules

```bash
git clone --recurse-submodules <repo-url>
cd thesis-metadrive
```

If the repository was already cloned without submodules:

```bash
git submodule update --init --recursive
```

### 1. Create And Review `.env`

```bash
cp .env.example .env
```

Review only the host-specific values:

- `HOST_OUTPUTS_DIR`
- `HOST_DATA_DIR`
- `HOST_CONTAINER_HOME_DIR`
- `HOST_UID`
- `HOST_GID`

On Linux/WSL, check UID/GID with:

```bash
id -u
id -g
```

### 2. Create Host Directories

Local defaults:

```bash
mkdir -p outputs data .container-home
```

Remote VM example:

```bash
mkdir -p /scratch/your_user/thesis-metadrive/outputs
mkdir -p /scratch/your_user/thesis-metadrive/data
mkdir -p /scratch/your_user/thesis-metadrive/.container-home
```

### 3. Validate Compose Expansion

```bash
docker compose config
```

What this catches:

- missing `.env` variables
- invalid mount paths
- invalid Compose syntax

### 4. Build The Image

```bash
docker compose build
```

What this catches:

- broken Dockerfile changes
- unresolved submodule-based dependencies
- dependency installation problems

### 5. Run A Container Import Smoke Check

```bash
docker compose run --rm dev bash -lc "
  uv run --no-sync python -c 'import thesis_rl, metadrive, stable_baselines3; print(\"imports ok\")'
"
```

What this catches:

- project import failures
- broken editable dependency wiring
- missing package metadata resolution

### 6. Optional GPU Check

On the host:

```bash
nvidia-smi
```

Inside the container:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml run --rm dev bash -lc "
  python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')
PY
"
```

### 7. Optional Targeted Test Check

If you want one fast repo-local check before heavier validation:

```bash
docker compose run --rm dev bash -lc "
  uv run --no-sync python -m pytest -q tests/test_common_paths.py
"
```

### 8. Optional Deeper Validation

For a broader but still structured validation flow, continue with:

- [validation_commands.md](validation_commands.md)

## Recommended Minimal Success Criteria

Consider the machine setup usable when all of the following pass:

- `git submodule update --init --recursive`
- `docker compose config`
- `docker compose build`
- container import smoke check for `thesis_rl`, `metadrive`, and `stable_baselines3`
- GPU check, if the machine is supposed to provide CUDA

## NVIDIA GPU

The `.env` file does not necessarily need to change.

You do need to verify that the machine has:

- NVIDIA drivers
- Docker
- NVIDIA Container Toolkit

Then you can check:

```bash
nvidia-smi
```

and inside the container:

```bash
python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

If you have multiple GPUs, you may want to add locally to your `.env`:

```env
CUDA_VISIBLE_DEVICES=0
```

or:

```env
NVIDIA_VISIBLE_DEVICES=0
```

But that is not necessary in the generic template.
