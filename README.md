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

## Notes for container usage

- Run commands from `/workspace/thesis/thesis-metadrive`.
- Prefer `uv run --no-sync` for repeated train/evaluate runs after `uv sync`.
- If the module `thesis_rl` is not found, refresh editable install with `uv pip install -e .`.

## Current status

This is still a **starter scaffold**, not the full implementation.
The main value at this stage is:
- stable repository structure
- stable interfaces
- stable Hydra organization

## TODO

- Add a fully container-first setup for reproducibility and portability:
	- define a project Dockerfile with pinned system and Python dependencies
	- remove local machine-specific wheel/path assumptions
	- keep `uv run` workflow consistent inside the container
	- document host GPU driver prerequisites and startup checks
