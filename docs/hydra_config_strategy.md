# Hydra Config Strategy

## Why Hydra here

Hydra is a good fit because this project will need to switch frequently between:

- environments
- scenario stages
- preprocessors
- planners
- adapters
- reward modes
- rulebooks
- hyperparameters

The main requirement is not just flexibility, but **controlled combinability**.

## Recommended configuration principle

Use Hydra groups for each interchangeable subsystem:

- `env`
- `agent`
- `planner`
- `preprocessor`
- `adapter`
- `reward`
- `experiment`

Avoid stuffing everything into a single huge YAML file.

## Suggested top-level config

```yaml
defaults:
  - env: metadrive_base
  - agent: baseline
  - planner: baseline_mlp
  - preprocessor: identity
  - adapter: direct_action
  - reward: scalar_default
  - experiment: baseline
  - _self_

seed: 42
device: cpu
run_name: ${experiment.name}_${now:%Y%m%d_%H%M%S}
```

## Suggested responsibilities

### env
Contains:
- observation mode
- map/scenario parameters
- traffic settings
- randomization flags
- seeds
- episode horizon

### planner
Contains:
- network sizes
- optimizer
- learning rates
- gamma
- tau
- update frequencies
- exploration parameters

### preprocessor
Contains:
- flattening
- normalization
- optional future encoder params

### adapter
Contains:
- adapter type
- clipping toggles
- optional future controller params

### reward
Contains:
- scalar vs violation-vector mode
- rule order
- enabled rules
- logging toggles
- external rulebook bridge parameters

### experiment
Contains:
- training steps
- eval frequency
- checkpoint frequency
- curriculum enable/disable
- promotion metric
- promotion threshold
- promotion consecutive evals

## Strong recommendation

Keep a dedicated experiment config for each milestone:

- `experiment/baseline.yaml`
- `experiment/curriculum.yaml`
- `experiment/rulebook.yaml`

This will make thesis experiments reproducible and easy to relaunch.
