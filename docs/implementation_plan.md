# Implementation Plan

## Guiding principle

Proceed in small steps, but keep the code runnable after every step.

---

## Phase 0 — Repository bootstrap

### Goal
Prepare the repository skeleton and tooling.

### Tasks
- initialize `uv`
- define `pyproject.toml`
- create `src/` layout
- set up Hydra config tree
- create baseline entrypoints
- add logging/checkpoint directories
- add smoke tests

### Exit criterion
A dummy train script runs with Hydra config resolution.

---

## Phase 1 — MetaDrive baseline with external trainer

### Goal
Run training/evaluation on MetaDrive with minimal customization.

### Locked decisions for this phase
- external backend: Stable-Baselines3 TD3
- planner integration strategy: full `PlannerAgent` wrapper from the first implementation
- evaluation protocol for curriculum promotion: 10 stochastic episodes

### Tasks
- create env factory for MetaDrive
- inspect observation and action spaces
- add `IdentityPreprocessor`
- add `DirectActionAdapter`
- integrate TD3 behind the `PlannerAgent` API
- create train and evaluate scripts
- save checkpoints and episode metrics

### Key implementation choice
In this phase, planner output should already match MetaDrive continuous action format.

### Exit criterion
You can train and evaluate a simple agent end-to-end on the default scalar reward.

### Milestone M1 — First executable baseline
M1 is reached when:

- `uv run python -m thesis_rl.train experiment=baseline` runs
- a checkpoint is saved
- `uv run python -m thesis_rl.evaluate ...` runs
- observation and action shapes are logged

---

## Phase 2 — Curriculum over scenario complexity

### Goal
Introduce staged environment complexity while preserving baseline compatibility.

### Tasks
- define stage configs
- implement `CurriculumStage`
- implement `CurriculumManager`
- support fixed stage runs
- support automatic stage progression from performance thresholds
  - metric family: weighted composite metric
  - promotion protocol: 10 stochastic eval episodes
- separate training and evaluation seeds for the last stage

### Exit criterion
The same training script can run:
- fixed-stage experiments
- curriculum-enabled experiments

---

## Phase 3 — Reward manager and violation-vector scaffolding

### Goal
Add violation-vector computation without coupling it yet to planner logic.

### Tasks
- define `TransitionContext`
- implement `RewardManager`
- attach violation data to `info`
- define per-component logging
- write tests for violation-vector assembly order
- stub the rulebook bridge

### Locked logging schema
- `info["violation_vector"]`
- `info["violation_components"]`
- `info["rule_metadata"]`

### Exit criterion
Training still works with scalar reward, but violation components are also computed and logged.

---

## Phase 4 — Rulebook integration

### Goal
Connect the reward manager to the existing rule implementation.

### Tasks
- create a thin adapter around the current rule module
- standardize rule I/O
- ensure rule ordering is config-driven
- verify step-based and transition-based rules
- test reset behavior and stateful rules

### Exit criterion
The violation vector is computed from the real rulebook implementation.

---

## Phase 5 — Planner integration with violation vector

### Goal
Allow the planner to consume and optimize the ordered violation vector.

### Tasks
- decide replay format for violation vectors
- define planner training interface for multi-objective returns
- add lexicographic placeholder implementation
- maintain compatibility with scalar baseline experiments

### Exit criterion
You can switch between scalar baseline and vector-violation planner through config.

---

## Practical advice for VSCode + Copilot

Use Copilot on one narrow task at a time. A good sequence is:

1. repo skeleton
2. env factory
3. identity preprocessor
4. direct adapter
5. baseline trainer integration
6. evaluator
7. curriculum manager
8. transition context
9. reward manager
10. rulebook bridge

This reduces the chance of Copilot generating a monolithic design too early.

## Execution checklist (merged from next_implementation_steps_v3)

Immediate coding sequence:

1. implement MetaDrive env factory
2. implement identity preprocessor
3. implement direct adapter
4. implement TD3 `PlannerAgent` wrapper
5. implement evaluator
6. add stage configs for curriculum
7. add transition context
8. add reward manager
9. add rulebook bridge
