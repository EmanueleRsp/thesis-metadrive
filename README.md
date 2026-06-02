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

## Intended commands

```bash
uv pip install -e .
uv sync
uv run --no-sync python -m thesis_rl.cli.train experiment=baseline device=cuda
uv run --no-sync python -m thesis_rl.cli.evaluate checkpoint_path=checkpoints/baseline_td3.zip device=cuda
```

For the current test and smoke-run workflow, see
[`docs/validation_commands.md`](docs/validation_commands.md).

## Notes for container usage

- Run commands from `/workspace/thesis/thesis-metadrive`.
- Prefer `uv run --no-sync` for repeated train/evaluate runs after `uv sync`.
- If the module `thesis_rl` is not found, refresh editable install with `uv pip install -e .`.

## Container workflow (shared and reproducible)

This repository includes:
- `Dockerfile`: pinned base image + `uv` + locked deps via `uv.lock`
- `compose.yaml`: standard dev container (GPU, interactive shell, persisted `outputs/`)

### 1) Build and start container

```bash
docker compose up -d --build
```

### 2) Enter container shell

```bash
docker compose exec dev bash
```

### 3) Run training / evaluation

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

### 4) Parallel runs in same container

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

### 5) Launch one pane per seed with tmux

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
defaults to `/workspace/thesis/thesis-metadrive` inside the container:

```bash
scripts/tmux_seed_grid.sh \
  --session sac_semantic_obs \
  --docker-container e.respino \
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

## Current status

This is still a **starter scaffold**, not the full implementation.
The main value at this stage is:
- stable repository structure
- stable interfaces
- stable Hydra organization

## TODO

- Add startup checks for CUDA visibility and MetaDrive runtime dependencies.
- Add optional non-interactive job profiles in `compose.yaml` for CI-style runs.
