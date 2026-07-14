# Scenario ACL Implementation Plan

## Objective

Implement the curriculum described in
`curriculum_learning_specification.md` as a new curriculum kind,
without blocking on the unfinished rulebook redesign.

## Current status

Current repository status after the replay smoke validation:

- `v1` completed:
  - config integration
  - dedicated runtime branch
  - generator arms
  - MAB generate-only flow
  - scenario export
  - temporary usefulness proxy
- `v2` completed:
  - scenario buffer
  - replay sampling with usefulness + staleness
  - replay execution through `ScenarioEnv`
  - smoke validation of `generate -> replay` in a single run
- `v3` not started intentionally:
  - mutation
  - structural/semantic validation for children
- `v4` intentionally deferred:
  - finalized rulebook-aware usefulness / criticality

Important:

- replay currently works, but it carries explicit compatibility debt with the
  installed MetaDrive `ScenarioEnv` API and with some replay-time rulebook
  inputs; these gaps are documented below and in
  `curriculum_learning_specification.md`
- because the rulebook/scalarization redesign is still open, the recommended
  next major implementation step is not mutation, but returning to rulebook
  work first

### Generator arms versus ScenarioNet semantic arms

The ACL MAB supports two explicit arm spaces. `arm_space=generator` samples
seven procedural profiles whose names (`broad_random`, `simple_low_risk`, and
so on) describe parameter distributions. `arm_space=scenario` samples the six
canonical ScenarioNet arms A0–A5 and delegates selection to the strict catalog
provider. The two namespaces remain distinct; in particular, PG generation
does not provide the VRU context required by A4.

The ScenarioNet provider now accepts an optional `arm` filter, the staged
ScenarioNet preset uses it for the six A0–A5 stages, and
`curriculum=scenario_acl_scenarionet` uses the same filter under MAB control.
Semantic ACL is catalog-based: it does not perform procedural export or
mutation, but its replay buffer stores the exact selected catalog record and
replays it through the ScenarioNet runtime. The legacy generator ACL retains
its existing export/buffer/replay path.

ScenarioNet ACL replay uses `ThesisScenarioEnv` as well as the sampled path.
The replay environment inherits the canonical ScenarioNet configuration:
native `horizon` and `allowed_more_steps` remain disabled, while the effective
episode limit is `scenario_length + env.episode_control.extra_steps_after_scenario`.
Legacy generator replay continues to use the native `ScenarioEnv` path.

The default warm-up is 100 scenarios in the ACL buffer, matching the original
ACL design; completed catalog records are collected episode by episode and
mutation remains disabled by contract.

Example invocation:

```bash
python -m thesis_rl.cli.train \
  env=scenarionet \
  curriculum=scenario_acl_scenarionet \
  env.vectorized.enabled=false
```

The target integration model is:

- selectable via Hydra config like the other components
- isolated from the existing `staged` curriculum
- resumable and checkpoint-friendly
- incrementally shippable in phases

---

## Locked Decisions

### 1. New curriculum kind

The new curriculum should be implemented as:

```text
curriculum.kind = scenario_acl
```

and not as an extension of `staged`.

Reason:

- `staged` selects coarse env pools and promotion thresholds
- `scenario_acl` selects concrete scenarios episode-by-episode
- the runtime control flow is materially different

### 2. Separate v1 from full specification

The full specification is larger than a single safe implementation step.
We should ship it in phases.

Version targets:

- `v1`: generate-only + MAB + ScenarioDescription export + basic usefulness + persistence
- `v2`: scenario buffer + replay sampling
- `v3`: mutation + validation + replay via `ScenarioEnv`
- `v4`: exact rulebook criticality and algorithm-specific learning potential

### 3. Single-env only in v1

`scenario_acl` should initially require:

```text
env.vectorized.enabled = false
```

Reason:

- the curriculum acts at episode/scenario granularity
- vectorized training complicates per-episode scenario selection, export,
  replay accounting, and MAB feedback
- we can reintroduce multi-env support later with an explicit design

### 4. Rulebook dependency stays behind an interface

The unfinished rulebook work should be isolated behind a scorer interface:

```python
compute_rule_criticality(...)
```

`scenario_acl` must be implementable before the final rulebook definition,
with a temporary fallback scorer.

### 5. Learning potential is phased

The exact formulas in the specification for PPO/TD3/SAC/distributional RL
should not block the first implementation.

### 6. Fresh sampling and replay are disjoint

Once a catalog scenario has been accepted into the ACL buffer, fresh sampling
must exclude it. It can subsequently be selected only through replay, whose
usefulness/staleness distribution is the single source of replay decisions.
If an arm has no fresh candidates after this exclusion, the driver falls back
to buffer replay and records the origin as `fallback_replay_exhausted_arm`.
Rejected records are not excluded and may be sampled again as fresh scenarios.

We should start with a pluggable learning-potential API and provide:

- `proxy` scorer in v1
- exact backend-specific scorers later

---

## Current Architecture Fit

The existing curriculum stack already supports adding a new strategy:

- `src/thesis_rl/curriculum/config.py`
- `src/thesis_rl/curriculum/interfaces/strategy.py`
- `src/thesis_rl/curriculum/registry.py`
- `src/thesis_rl/curriculum/manager.py`

However, the current strategy protocol is shaped around stage progression.
`scenario_acl` needs more than:

- current env overrides
- eval metrics
- promotion

It also needs:

- next-scenario selection
- curriculum iteration mode selection (`generate` vs `exploit`)
- scenario export and storage
- post-episode feedback updates
- replay/mutation bookkeeping

Therefore the implementation should add a dedicated runtime path for
`scenario_acl`, while still using the curriculum registry/config machinery.

---

## Proposed File Layout

### New config files

- `conf/curriculum/scenario_acl.yaml`
- `conf/curriculum/scenario_acl_generate_only.yaml`
- `conf/curriculum/scenario_acl_mab_plus_replay.yaml`

Optional later split if the config grows too much:

- `conf/curriculum/scenario_acl/mab.yaml`
- `conf/curriculum/scenario_acl/replay.yaml`
- `conf/curriculum/scenario_acl/mutation.yaml`
- `conf/curriculum/scenario_acl/validation.yaml`

### New Python package

- `src/thesis_rl/curriculum/scenario_acl/__init__.py`
- `src/thesis_rl/curriculum/scenario_acl/config.py`
- `src/thesis_rl/curriculum/scenario_acl/strategy.py`
- `src/thesis_rl/curriculum/scenario_acl/driver.py`
- `src/thesis_rl/curriculum/scenario_acl/arms.py`
- `src/thesis_rl/curriculum/scenario_acl/mab.py`
- `src/thesis_rl/curriculum/scenario_acl/record.py`
- `src/thesis_rl/curriculum/scenario_acl/buffer.py`
- `src/thesis_rl/curriculum/scenario_acl/ranking.py`
- `src/thesis_rl/curriculum/scenario_acl/usefulness.py`
- `src/thesis_rl/curriculum/scenario_acl/scenario_store.py`
- `src/thesis_rl/curriculum/scenario_acl/scenario_env.py`
- `src/thesis_rl/curriculum/scenario_acl/mutation.py`
- `src/thesis_rl/curriculum/scenario_acl/validation.py`

### Existing files likely to change

- `src/thesis_rl/curriculum/config.py`
- `src/thesis_rl/curriculum/registry.py`
- `src/thesis_rl/runtime/loops/train_loop.py`
- `src/thesis_rl/runtime/loops/eval_loop.py`
- `src/thesis_rl/runtime/wiring/builders.py`
- `src/thesis_rl/runtime/io/metadata.py`
- `src/thesis_rl/runtime/io/csv_recorder.py`

### New tests

- `tests/test_scenario_acl_config.py`
- `tests/test_scenario_acl_mab.py`
- `tests/test_scenario_acl_buffer.py`
- `tests/test_scenario_acl_ranking.py`
- `tests/test_scenario_acl_usefulness.py`
- `tests/test_scenario_acl_store.py`
- `tests/test_scenario_acl_validation.py`
- `tests/test_scenario_acl_runtime_wiring.py`

---

## Runtime Design

## 1. Top-level integration

At runtime the training loop should branch on curriculum kind:

- `disabled`: current baseline path
- `staged`: current staged path
- `scenario_acl`: dedicated scenario-level curriculum path

This branch should happen early in `run_training(...)`, not deep inside
the staged flow.

## 2. Scenario ACL driver

`scenario_acl` should introduce a small runtime driver responsible for:

- deciding curriculum iteration mode
- building the next training environment
- exporting generated scenarios
- updating MAB/buffer/state after episodes
- requesting replay or mutation when enabled

The driver should own curriculum-specific state, while the training loop
remains responsible for:

- planner/agent lifecycle
- checkpoints
- evaluation cadence
- global logging

## 3. Environment modes

The driver needs two environment families:

- `MetaDriveEnv` for procedural generation
- `ScenarioEnv` for replayed or mutated scenarios

This implies a dedicated builder helper instead of reusing only
`merge_env_config_with_overrides(...)`.

## 4. Persistence

The curriculum state saved in training checkpoints must grow beyond the
current staged counters. At minimum it should include:

- MAB weights
- recent window state
- scenario buffer metadata
- scenario store paths
- counters for replay/mutation usage

Large artifacts must stay on disk; checkpoint payloads should store
metadata and references, not full scenario blobs.

---

## Phase Plan

## Phase 1 - Config and skeleton

### Goal

Make `scenario_acl` selectable and validated through config.

### Tasks

- extend `CurriculumConfig` to parse `kind=scenario_acl`
- add dedicated dataclasses for scenario ACL config
- register the new strategy in the curriculum registry
- add fail-fast validation:
  - vectorized env disabled
  - supported reward modes only
  - supported ablation mode names

### Exit criterion

Hydra composes `curriculum=scenario_acl` and a `CurriculumManager`
can be constructed without touching the runtime yet.

## Phase 2 - Generate-only driver

### Goal

Run a scenario-level curriculum loop using only procedural generation.

### Tasks

- implement generator arm definitions A0-A6
- implement arm sampling from config
- implement MAB weights, target weights, and sync logic
- implement a minimal scenario ACL driver with:
  - warmup
  - `generate` mode only
  - per-episode feedback update
- export generated scenarios via MetaDrive `export_scenarios(...)`
- persist ScenarioDescription datasets to a curriculum-owned directory

### Deferred in this phase

- replay from buffer
- mutation
- exact rulebook criticality

### Exit criterion

A training run can execute with:

```text
curriculum=scenario_acl_generate_only
```

and produces:

- selected arm history
- exported scenario files
- updated MAB state

## Phase 3 - Usefulness and ranking

### Goal

Introduce a stable usefulness API that works now and can be upgraded later.

### Tasks

- define `ScenarioUsefulness` representation
- define lexicographic comparison helpers
- add `RuleCriticalityScorer` interface
- add `LearningPotentialScorer` interface
- implement initial scorers:
  - rule criticality fallback from already available evaluation metrics
  - learning-potential proxy from available runtime/backend metrics
- implement rank normalization for MAB feedback

### Important note

The fallback implementation must be explicitly marked temporary and should
be easy to replace once rulebook and backend hooks are finalized.

### Exit criterion

Generated scenarios get a comparable usefulness value and a normalized score
usable by MAB.

## Phase 4 - Scenario buffer and replay

### Goal

Add `Lambda` and exploitation without mutation.

### Tasks

- implement `ScenarioRecord`
- implement buffer capacity and lexicographic replacement
- implement duplicate rejection by scenario hash
- implement recent-window ranking set
- implement replay sampling with usefulness + staleness
- implement replay execution through `ScenarioEnv`

### Exit criterion

Runs can alternate between:

- generating new scenarios
- replaying stored scenarios

with ablation support for:

- `uniform_pg`
- `mab_generate_only`
- `mab_plus_replay`

## Phase 5 - Mutation and validation

### Goal

Enable controlled scenario mutation in exploit mode.

### Tasks

- implement mutation operators
- implement structural validation
- implement custom semantic validation
- implement save/reload/smoke-test path for mutated scenarios
- enforce per-parent child limits and failed-attempt caps

### Exit criterion

`full_curriculum` can:

- replay parent scenarios
- produce children
- validate them
- evaluate them
- insert them into the buffer

## Phase 6 - Exact rulebook integration

### Goal

Replace the placeholder rule criticality scorer.

### Tasks

- connect to finalized rulebook outputs
- compute final `C_rule_s`
- finalize rulebook-dependent diagnostics

### Exit criterion

`C_rule_s` is derived from the finalized rulebook definitions rather than
from temporary proxies.

## Phase 7 - Exact learning potential

### Goal

Replace the proxy learning-potential scorer with backend-specific logic.

### Tasks

- add planner/runtime hooks needed to compute per-scenario signals
- implement PPO scorer
- implement TD3 scorer
- implement SAC scorer
- design future hooks for lexicographic/distributional planners

### Exit criterion

Backend-specific `LP_alg_s` is available for the currently supported
training algorithms in the repo.

---

## Suggested v1 Scope

To keep momentum high, the first coding pass should stop at:

- Phase 1
- Phase 2
- enough of Phase 3 to support a temporary usefulness score

That gives us:

- a real new curriculum kind
- real arm adaptation
- real scenario export
- no premature entanglement with replay/mutation/rulebook redesign

This is the best balance between progress and risk.

---

## Metrics and Artifacts

The new curriculum should create dedicated artifacts under the run directory,
for example:

- `artifacts/curriculum/scenario_acl_state.json`
- `artifacts/curriculum/scenario_buffer.jsonl`
- `artifacts/curriculum/mab_history.jsonl`
- `artifacts/curriculum/scenarios/`

Minimum CSV/log visibility for v1:

- selected generator arm
- arm probability distribution
- normalized usefulness
- generate vs exploit mode
- scenario id / parent id
- export path

---

## Testing Strategy

### Unit tests first

Write unit tests for:

- config parsing
- MAB probability/update logic
- lexicographic ranking
- buffer replacement
- duplicate rejection
- normalization

### Integration tests second

Add smoke-style tests for:

- `scenario_acl` config composition
- runtime branch selection
- scenario export dataset creation
- replay loading through `ScenarioEnv`

### Guardrails

The initial runtime should fail fast when:

- vectorized training is enabled
- replay/mutation is requested before required config is present
- usefulness scorer is missing
- scenario export fails

---

## Risks to Control

### 1. Training loop complexity

Do not try to bend the current staged loop into supporting scenario ACL.
Keep a separate driver and a clear runtime branch.

### 2. Artifact volume

Scenario export can generate many files.
The implementation must define a stable on-disk layout early.

### 3. Hidden MetaDrive assumptions

Before enabling replay/mutation by default, verify:

- ScenarioDescription export fidelity
- ScenarioEnv reload behavior
- `reactive_traffic=True` under agent control
- exact env flag compatibility

Current tracked compatibility note:

- in the MetaDrive version currently used by the Docker/container workflow,
  `ScenarioEnv` replay rejected these config keys at construction time:
  `out_of_road_done`, `on_continuous_line_done`,
  `on_broken_line_done`
- the implementation currently works around this by filtering replay-time
  config to the subset of keys supported by the installed version
- this is intentionally tracked as technical debt / compatibility debt, not
  treated as a final semantic decision
- when replay semantics are finalized, revisit whether these flags:
  - become supported after a MetaDrive upgrade
  - need an alternative implementation path
  - should be removed from the design surface entirely

Additional tracked compatibility note:

- during replay with `ScenarioEnv`, the reward/rulebook wrapper may emit
  warnings about missing derived inputs such as `opposite_carriageway` and
  `target_region`
- these warnings do not block replay execution, but they mean some rules can
  become neutral or fall back to weaker geometry/goal signals
- this should be treated as semantic compatibility debt between replayed
  scenarios and the full rulebook runtime, especially before relying on
  replay-time rule metrics for final comparisons

### 4. Planner hook debt

Do not overpromise exact `LP_alg_s` in v1.
Treat it as a scheduled second-step integration.

---

## Recommended Immediate Coding Sequence

1. add config dataclasses and Hydra group for `scenario_acl`
2. register the new curriculum kind
3. add runtime branching in `train_loop.py`
4. implement generator arms and MAB
5. implement scenario export/store
6. implement temporary usefulness scorer
7. run a generate-only smoke experiment
8. only then move to replay/buffer

---

## Definition of Done for First Merge

The first merge should satisfy all of the following:

- `curriculum=scenario_acl_generate_only` composes successfully
- training starts through a dedicated scenario ACL runtime path
- generator arms are sampled and logged
- MAB weights update over time
- generated scenarios are exported and saved to disk
- curriculum state is persisted and reloadable
- tests cover config, MAB, and runtime wiring

If all of that is true, the project has a solid base for the second wave:
buffer, replay, mutation, and final rulebook coupling.

## Definition of Done for Second Merge

The second merge is considered achieved when all of the following are true:

- `curriculum=scenario_acl_mab_plus_replay` composes successfully
- the runtime alternates between `generate` and `exploit_replay`
- generated scenarios are inserted into the scenario buffer
- replayed scenarios are loaded back through `ScenarioEnv`
- replay updates `num_seen`, `last_seen_step`, and usefulness of stored records
- smoke validation confirms a run can execute:
  - one generated chunk
  - one replayed chunk
- known replay compatibility gaps are documented explicitly

Current assessment:

- this repository has reached that second-merge threshold
- mutation and final rulebook coupling remain separate future phases
