# Remaining Open Decisions

This file tracks only decisions that are still genuinely open.

## Already locked (for alignment)

- baseline backend for Step 1: Stable-Baselines3 TD3
- integration strategy: full `PlannerAgent` wrapper from the first implementation
- promotion evaluation protocol: 10 stochastic episodes
- violation logging schema:
  - `info["violation_vector"]`
  - `info["violation_components"]`
  - `info["rule_metadata"]`

## 1. Composite metric definition for curriculum promotion

Question:
What exact weighted formula should be used as the official promotion metric?

Open points:
- which components are included (e.g., success, collision-free, progress)
- exact weights
- threshold value and consecutive-evaluation policy

Proposed starting point (to validate):
- `promotion_score = 0.5 * success_rate + 0.3 * collision_free_rate + 0.2 * progress_score`

## 2. Stochastic evaluation reproducibility policy

Question:
How should stochastic evaluation be seeded so promotion remains stable and reproducible?

Open points:
- fixed seed set reused across evaluations vs rotating seeds
- smoothing policy (single eval window vs moving average)
- minimum number of evaluation windows before allowing stage promotion

## 3. Rulebook bridge contract (pending external module review)

Question:
What thin adapter shape best maps the existing rule module to the internal interfaces?

Status:
- deferred until the existing module is reviewed

Expected output of this decision:
- normalized rule I/O contract
- stateful-rule reset semantics
- final rule ordering source of truth (config-driven)

## 4. PlannerAgent boundary details

Question:
What is the minimum stable interface for the full wrapper in Phase 1?

Open points:
- required methods (`train`, `predict`, `save`, `load`, `evaluate`)
- ownership of replay buffer and logging hooks
- which trainer details remain internal vs exposed for diagnostics
