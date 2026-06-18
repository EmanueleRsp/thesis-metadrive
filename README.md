# Thesis RL Codebase

Modular codebase for MetaDrive experiments with:

- MetaDrive for training/evaluation
- modular pipeline: `preprocessor -> planner -> adapter`
- Hydra configs
- `uv` environment management
- Stage 1 baseline with **TD3**
- future rulebook-based **violation vector**
- future curriculum over scenario complexity

## Current scope

This scaffold implements the **repository skeleton** and the **initial configuration strategy** for:

1. baseline training with MetaDrive
2. later curriculum integration
3. later rulebook / violation-vector integration

## Fixed initial decisions

- baseline backend: **custom PyTorch planners (TD3/SAC/PPO)**
- observation: **LidarStateObservation**
- preprocessor: **IdentityPreprocessor**
- planner output: direct low-level MetaDrive action
- adapter: **DirectActionAdapter**
- rule signal semantics: **violation scores**
- curriculum: performance-based, with optional manual mode

## Layout

```text
conf/
src/thesis_rl/
tests/
docs/
```

## Recommended workflow

The recommended and supported workflow for this repository is:

- `docker compose` for development and runs
- the NVIDIA PyTorch base image for `torch`
- `/scratch/$USER/...` for large outputs and container home/cache data

Host-side `uv` usage outside Docker is currently best treated as a legacy/advanced path, not the primary setup path.

## Current portability status

This repository is still optimized for a specific research environment and is not yet fully plug-and-play for an arbitrary GitHub clone.

In particular, the current recommended workflow assumes:

- Docker with recent `docker compose` support
- BuildKit support for Compose `additional_contexts`
- NVIDIA GPU support in Docker
- a local sibling MetaDrive checkout at `../third-party/metadrive`
- an output-capable path such as `/scratch/$USER/...`

If those assumptions do not hold on a target machine, the project may still be usable, but the local setup and/or `compose.yaml` will likely need adaptation.

## External prerequisites

Before using the current Docker workflow, make sure the host machine provides:

- Docker Engine running and accessible to your user
- Docker Compose v2
- NVIDIA container runtime / Docker GPU support
- enough disk space to build `nvcr.io/nvidia/pytorch:24.01-py3`
- a writable scratch-like area for large outputs

For local Python/`uv` workflows outside Docker, the current environment still expects a compatible preinstalled `torch` runtime. That path remains available, but it is not the recommended default.

## Required directory layout

The current repository expects this layout on disk:

```text
<parent>/
  thesis-metadrive/
  third-party/
    metadrive/
    stable-baselines3/
```

Concretely, in the current setup:

- the repo lives at `.../thesis-metadrive`
- the local MetaDrive checkout lives at `../third-party/metadrive`
- the local SB3 fork checkout lives at `../third-party/stable-baselines3`

This is required because `pyproject.toml` currently pins:

- `metadrive-simulator` from `../third-party/metadrive`
- `stable-baselines3` from `../third-party/stable-baselines3`

Without those sibling checkouts, `uv sync` and Docker image builds will fail.

For Docker specifically, `torch` now comes from the NVIDIA base image (`nvcr.io/nvidia/pytorch:24.01-py3`) and is excluded from uv-managed dependencies.

At the moment this means:

- Docker path: `torch` is provided by the base image and `uv` does not try to resolve or install it
- host path: `uv` likewise excludes `torch`, so a host-native workflow needs a compatible preinstalled `torch`

`metadrive-simulator` still resolves from the sibling local checkout, matching the MetaDrive development workflow.

## Output storage assumptions

The current default output root is:

- `/scratch/$USER/thesis-metadrive/outputs`

This is a deliberate choice to avoid filling the home directory with checkpoints, replay buffers, logs, videos, and analysis artifacts.

If a machine does not provide `/scratch`, you will need to override:

- `paths.outputs_root`

and, for Docker Compose, likely adapt:

- the scratch bind mount in `compose.yaml`
- `HOME` inside the container
- any docs/examples that refer to `/scratch/$USER/...`

## Intended commands

```bash
uv sync --extra dev
uv run --no-sync python -m thesis_rl.cli.train experiment=baseline device=cuda
uv run --no-sync python -m thesis_rl.cli.evaluate checkpoint_path=checkpoints/baseline_td3.zip device=cuda
```

These commands are primarily relevant inside the Docker container.
Running them directly on the host may still require extra machine-specific setup because `torch` must already be available outside uv.

For the current test and smoke-run workflow, see
[`docs/validation_commands.md`](docs/validation_commands.md).

For the current fork-backed baseline presets, prefer:

- `presets/agent/sac_sb3`, `presets/agent/ppo_sb3`, `presets/agent/td3_sb3`
- `presets/agent/sac_mlp_sb3`, `presets/agent/ppo_mlp_sb3`, `presets/agent/td3_mlp_sb3`
- `presets/agent/sac_lq_sb3`, `presets/agent/ppo_lq_sb3`, `presets/agent/td3_lq_sb3`

The first group keeps the vanilla SB3-style flattened-observation path. The
second group uses thesis `MLPEncoder` with fork-backed SB3 algorithms. The
third group uses thesis `LQEncoder` with the same fork-backed path.

Important note:

- `presets/agent/*_lq_sb3` assume `obs=semantic_state`
- avoid overriding those presets with `obs=lidar_state`, because `LQEncoder`
  is wired for the semantic-state observation path

Legacy agent presets remain in the repo temporarily for migration support, but
new runs should prefer the fork-backed presets above.

For a curated overview of the documentation set, see
[`docs/README.md`](docs/README.md).

## Notes for container usage

- Run commands from `/workspace/thesis-metadrive`.
- The Compose service mounts:
  - `../third-party/metadrive` at `/workspace/third-party/metadrive`
  - `../third-party/stable-baselines3` at `/workspace/third-party/stable-baselines3`
  so a container-side `uv sync` can resolve both local path dependencies and
  write build metadata when needed.
- Prefer `uv run --no-sync` for repeated train/evaluate runs after `uv sync --extra dev`.
- Inside the container, prefer `uv sync --extra dev` as the standard way to
  refresh project installs. Avoid a plain `uv pip install -e .` when the repo
  also contains a host-side `.venv`, because `uv pip` may try to target that
  environment instead of `/opt/venv`.
- By default run artifacts now go under `/scratch/$USER/thesis-metadrive/outputs`.
- Override the output root per run with `paths.outputs_root=/some/other/path` if needed.
- Treat this container workflow as the source of truth unless you intentionally want to debug host-native installation issues.

## Container workflow (shared and reproducible)

This repository includes:
- `Dockerfile`: pinned base image + `uv` + locked deps via `uv.lock`
- `compose.yaml`: standard dev container (GPU, non-root user, repo mounted from `home`, outputs written to `/scratch`)

### 1) Prepare scratch directories

```bash
mkdir -p /scratch/$USER/container-home
mkdir -p /scratch/$USER/thesis-metadrive/outputs
```

### 2) Create local Compose environment file

`docker compose` reads `.env` automatically. Create a local copy and fill in your host values:

```bash
cp .env.example .env
```

Then edit `.env` so it matches your account:

```env
USER_NAME=your_unix_username
HOST_UID=your_numeric_uid
HOST_GID=your_numeric_gid
```

You can get the numeric values with:

```bash
id -u
id -g
```

`.env` is ignored by git, so this stays local to your machine.
These values are also used at build time to create a named non-root user inside the image, so the shell prompt does not fall back to `I have no name!`.
The container virtual environment lives at `/opt/venv`, so it is not shadowed by any host-side `.venv` inside the mounted repo.
That virtual environment is created with access to the NVIDIA base image system site-packages, so GPU-enabled `torch` comes from the base image while the rest of the project dependencies are synced by `uv`.

### 3) Build and start container

```bash
docker compose up -d --build
```

If you change `.env` later, rebuild so the image user is regenerated with the new UID/GID:

```bash
docker compose up -d --build
```

### 4) Enter container shell

```bash
docker compose exec dev bash
```

### 5) Run training / evaluation

```bash
uv run --no-sync python -m thesis_rl.cli.train
uv run --no-sync python -m thesis_rl.cli.evaluate
```

Examples with Hydra preset/overrides:

```bash
uv run --no-sync python -m thesis_rl.cli.train preset=td3/td3_scalar_def_curr
uv run --no-sync python -m thesis_rl.cli.evaluate preset=td3/td3_scalar_def_curr
uv run --no-sync python -m thesis_rl.cli.train preset=td3/td3_scalar_rulebook_scale_tuning_no_curr run_profile=fast
```

Runs now write by default to:

- `/scratch/$USER/thesis-metadrive/outputs`

while the code stays mounted from:

- `/home/.../thesis-metadrive`

### 6) Parallel runs in same container

Attach from multiple terminals and run concurrent processes:

```bash
docker compose exec dev bash
tmux new -s thesis
```

In another host terminal:

```bash
docker compose exec dev bash
tmux attach -t thesis
```

### 7) Launch one pane per seed with tmux

To create a tmux session with one tiled pane per seed and automatically append
`seed=<seed>` to each command:

```bash
scripts/tmux_seed_grid.sh --session sac_semantic_obs -- \
  uv run --no-sync python -m thesis_rl.cli.train \
    --config-name config \
    run_profile=thesis \
    reward=monitor_only \
    curriculum=disabled \
    obs=semantic_state \
    agent/planner/encoder=mlp \
    agent/planner/decoder=mlp_encoded \
    agent/planner/algorithm=sac
```

By default it launches seeds `0..9`. Useful options:

```bash
scripts/tmux_seed_grid.sh --attach --seed-list 0,1,2 -- <command ...>
scripts/tmux_seed_grid.sh --seed-count 4 --seed-start 10 -- <command ...>
```

To run each pane inside an already running Docker container and enter the
repository workspace first, you can just pass `--docker-container`; the script
defaults to `/workspace/thesis-metadrive` inside the container:

```bash
scripts/tmux_seed_grid.sh \
  --session sac_semantic_obs \
  --docker-container thesis-metadrive-dev \
  --attach -- \
  uv run --no-sync python -m thesis_rl.cli.train \
    --config-name config \
    run_profile=thesis \
    reward=monitor_only \
    curriculum=disabled \
    obs=semantic_state \
    agent/planner/encoder=mlp \
    agent/planner/decoder=mlp_encoded \
    agent/planner/algorithm=sac
```

If needed, you can still override the container workdir explicitly with
`--docker-workdir`.

### 8) Stop the container

```bash
docker compose down
```

## Current status

This is still a **starter scaffold**, not the full implementation.
The main value at this stage is:
- stable repository structure
- stable interfaces
- stable Hydra organization

## TODO

- Add startup checks for CUDA visibility and MetaDrive runtime dependencies.
- Add optional non-interactive job profiles in `compose.yaml` for CI-style runs.
