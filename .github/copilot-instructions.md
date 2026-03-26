# Copilot Instructions for This Repository

This repository is a research codebase for reinforcement learning in MetaDrive.

## Mandatory design decisions

Copilot should assume the following decisions are already fixed unless explicitly changed in the docs:

- The initial training backend may use **Stable-Baselines3**
- The initial observation is **MetaDrive `LidarStateObservation`**
- The first preprocessor is an **identity/dummy preprocessor**
- The first planner outputs the **final low-level continuous action**
- The first adapter is a **thin direct-action adapter**
- The rule signal is a **vector of violation scores**
- Curriculum progression is **performance-based**, with optional manual stage selection
- The initial codebase must remain executable at every incremental step

## Core design constraints

- Keep the code **modular**
- Keep the code **incrementally executable**
- Prefer **simple abstractions** over premature generalization
- Avoid hidden coupling between environment, planner, reward, and curriculum
- All important behavior must be **configurable through Hydra**
- Preserve backward compatibility of the training entrypoint as features are added

## Agent architecture

The agent is structured as:

1. `preprocessor`
   - input: raw observation from MetaDrive
   - output: state representation / embedding for the planner

2. `planner`
   - input: preprocessed state
   - output: planner prediction
   - initially: direct low-level action

3. `adapter`
   - input: planner prediction
   - output: final action expected by MetaDrive

This decomposition must stay explicit in the code.

## Reward architecture

- MetaDrive returns a scalar reward through the normal environment interface
- We also need a **vector of violation scores** based on a rulebook
- The violation vector must be computed in a separate reward module / wrapper
- Rule components should be available individually
- Do not entangle rule evaluation with planner logic

## Curriculum architecture

Scenario complexity must be staged:

- Stage 1: fixed simple maps, no traffic
- Stage 2: small procedural maps, no traffic, gradual randomization
- Stage 3: procedural maps with light traffic
- Stage 4: train/test disjoint scenario seeds for generalization

The curriculum logic should live in dedicated classes/modules, not inside the main training loop.

## Implementation style

- Use Python type hints
- Prefer dataclasses or explicit config objects when useful
- Write concise docstrings
- Log important shapes, config choices, and environment metadata
- Add small smoke tests for each major module
- Avoid overengineering the first baseline

## Priority order

When implementing, optimize for this order:

1. correctness
2. modularity
3. reproducibility
4. extensibility
5. performance

## What to avoid

- giant monolithic trainer classes
- planner-specific logic inside environment wrappers
- hard-coded reward rules in training scripts
- hard-coded curriculum thresholds in multiple places
- ad-hoc config parsing outside Hydra
