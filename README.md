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

- baseline backend: **Stable-Baselines3 TD3**
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
uv run --no-sync python -m thesis_rl.train experiment=baseline device=cuda
uv run --no-sync python -m thesis_rl.evaluate checkpoint_path=checkpoints/baseline_td3.zip device=cuda
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
uv run --no-sync python -m thesis_rl.train
uv run --no-sync python -m thesis_rl.evaluate
```

Examples with Hydra preset/overrides:

```bash
uv run --no-sync python -m thesis_rl.train preset=td3/td3_scalar_def_curr
uv run --no-sync python -m thesis_rl.evaluate preset=td3/td3_scalar_def_curr
uv run --no-sync python -m thesis_rl.train preset=td3/td3_scalar_rulebook_scale_tuning_no_curr run_profile=fast
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

## Current status

This is still a **starter scaffold**, not the full implementation.
The main value at this stage is:
- stable repository structure
- stable interfaces
- stable Hydra organization

## TODO

- Add startup checks for CUDA visibility and MetaDrive runtime dependencies.
- Add optional non-interactive job profiles in `compose.yaml` for CI-style runs.
