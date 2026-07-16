# Copilot Instructions For This Repository

Follow the durable repository rules in `../AGENTS.md`.

Before proposing or changing behavior:

1. consult `../docs/project_index.md` for the selected authoritative
   specification;
2. read applicable approved ADRs under `../docs/decisions/`;
3. inspect the current Hydra composition under `../conf/` and the current code;
4. use or update the relevant ExecPlan under `../docs/implementation/`.

Do not treat this file as a scientific specification or freeze configuration
defaults here. The current base composition in `conf/config.yaml` selects the
semantic-state observation and an SB3-backed SAC planner, but presets may
override those choices. Verify the resolved configuration for the task.

Preserve the explicit preprocessor, planner, and adapter boundaries; keep reward,
rule evaluation, curriculum, environment, and planner responsibilities separate;
use Hydra for important behavior; and avoid unrelated cleanup or speculative
abstractions.

Never change formulas, semantics, thresholds, public interfaces, dataset policy,
experimental behavior, or protected test expectations without the approval
required by `AGENTS.md`.
