# Initial Design Decisions

This file records the concrete design decisions that are considered fixed before implementing the first executable version of the codebase.

## 1. Baseline training backend

The first executable baseline may rely on an **external RL implementation** for continuous-control training and evaluation.

Initial practical choice:
- use **Stable-Baselines3** for Step 1
- preserve an internal modular repository structure around env, preprocessing, adapter, reward, and curriculum
- later integrate the thesis planner without breaking the surrounding infrastructure

Rationale:
the first milestone is infrastructure + executability, not planner novelty.

## 2. Observation and preprocessor

The initial policy input is **MetaDrive `LidarStateObservation`**.

Decision:
- use a `DummyPreprocessor` / `IdentityPreprocessor`
- return the observation essentially unchanged
- allow only minimal conversion if required by the backend:
  - dtype conversion
  - NumPy to tensor conversion
  - shape normalization if needed

Non-goal for Step 1:
- no learned state encoder
- no AR1-like embedding yet
- no additional feature engineering unless strictly necessary

## 3. Planner output contract

The initial planner outputs the final action expected by MetaDrive.

Decision:
- the planner output is the raw continuous action
- no intermediate waypoint or high-level command is used in the first baseline

Expected shape:
```python
action = [steering, throttle_or_brake]
```

## 4. Adapter behavior

The initial adapter is a **thin direct-action adapter**.

Responsibilities:
- verify action dimensionality
- clip values to the valid range
- convert to the exact type / container expected by the environment

Non-goal:
- no controller logic
- no policy-level abstraction beyond thin validation

Future extension:
if the planner later produces higher-level references, the adapter may evolve toward a MetaDrive `Policy`-compatible controller.

## 5. Reward semantics

The rulebook signal is modeled as a **vector of violation scores**.

Convention:
- `0` = no violation
- `> 0` = violation
- larger value = more severe violation

Implication:
the future lexicographic planner should interpret the objective as **minimize the ordered violation vector**, not maximize a conventional reward vector.

Terminology rule for the repository:
- use `scalar_reward` for the environment reward
- use `violation_vector` for the rulebook output
- use `violation_components` for named rule values

## 6. Curriculum policy

The scenario curriculum is **performance-based**.

The codebase must support both:
- `manual_stage_mode`
- `auto_promotion_mode`

Initial auto-promotion policy:
- evaluation happens every `eval_interval`
- the promotion metric is initially `success_rate`
- promotion occurs only if the metric exceeds a threshold for `K` consecutive evaluations

Recommended initial defaults:
- `promotion_metric = success_rate`
- `promotion_threshold = 0.80`
- `promotion_consecutive_evals = 3`

These are defaults, not hard-coded constants.

## 7. Rulebook integration timing

The existing rule module is **not** refactored immediately.

Decision:
- define the interfaces now
- integrate the real module later
- decide wrapper vs refactor only when the actual module is plugged in

Required abstractions to prepare now:
- `TransitionContext`
- `BaseRule` or equivalent protocol
- `RulebookEvaluator`
- reward manager output schema

## 8. Design philosophy

The repository should be **modular and extensible**, but not prematurely generalized.

Interpretation:
- define stable interfaces
- implement only the simplest useful concrete versions first
- avoid building a generic framework for every possible future planner

Priority:
- working code first
- extensibility second
- abstraction only when it pays off
