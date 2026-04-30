# Validation Commands

Run all commands from the project root inside the container:

```bash
cd /workspace/thesis/thesis-metadrive
```

## After Code Changes

Refresh the editable install, then run the full test suite:

```bash
uv pip install -e .
uv run --no-sync python -m pytest -q
```

If `pytest` is missing, sync the dev extra once:

```bash
uv sync --extra dev
```

## Smoke Test: Baseline, No Curriculum

This short run is intended to validate train/eval/test flow, seed windows,
checkpoint creation, logging, CSV output, and TD3 update plumbing.

```bash
uv run --no-sync python -m thesis_rl.train --config-name td3_scalar_def_no_curr \
  experiment.total_timesteps=2000 \
  experiment.eval_interval=1000 \
  experiment.log_interval=100 \
  experiment.eval_episodes=2 \
  experiment.final_eval_episodes=2 \
  planner.learning_starts=100 \
  planner.batch_size=32
```

Expected smoke-test signs:

- `n_updates > 0`
- `actor_loss`, `critic_loss`, and `learning_rate` are not `NaN`
- intermediate evaluation uses the validation split
- final evaluation uses the test split
- `checkpoints/final.zip` is produced
- run artifacts are under `outputs/.../{logs,csv,checkpoints,artifacts}`

## Smoke Test: Curriculum

Use the same short-run overrides with the curriculum config:

```bash
uv run --no-sync python -m thesis_rl.train --config-name td3_scalar_def_curr \
  experiment.total_timesteps=2000 \
  experiment.eval_interval=1000 \
  experiment.log_interval=100 \
  experiment.eval_episodes=2 \
  experiment.final_eval_episodes=2 \
  planner.learning_starts=100 \
  planner.batch_size=32
```

## Notes

- The smoke-test TD3 overrides are intentionally not production defaults.
- Production-style TD3 keeps `planner.learning_starts=10000`, so runs shorter than that collect replay but do not update.
- MetaDrive treats `reset(seed=...)` as a scenario index. The project keeps train, validation, and test scenario pools disjoint through `scenario_splits` in `conf/config.yaml`.
