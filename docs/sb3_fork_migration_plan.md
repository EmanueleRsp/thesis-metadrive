# SB3 Fork Migration Plan

## Status

This document is the proposed migration plan from the current mixed setup
(custom backends plus direct SB3 wrappers) toward a local SB3 fork used as the
algorithmic core for future scalar, lexicographic, and distributional work.

It is intended to become the operational reference for this migration once the
open decisions below are confirmed.

Current closure snapshot for the scalar baseline migration:

- fork-backed `PPO`, `SAC`, and `TD3` are the preferred baseline path
- thesis encoder bridges are wired and smoke-tested on the fork-backed path
- current closure gate is:
  tests + smoke runs + medium training runs
- standalone `load+eval` and `resume` validation remain deferred
- legacy backends remain temporarily for compatibility and historical
  comparison, but should not be used as the default path for new runs

## Confirmed decisions

The following decisions are now fixed unless later revised explicitly:

- SB3 will live as a sibling checkout at
  `../third-party/stable-baselines3`
- Docker integration should mount only the specific external libraries needed by
  the project, not the whole sibling `third-party/` tree
- the immediate scope is not lexicographic or distributional research yet
- the immediate goal is to consolidate correct fork-backed baseline
  implementations for `PPO`, `SAC`, and `TD3`
- custom encoders, decoders, and future custom buffer variants remain
  requirements for the target architecture

## Why change direction

The current repo already exposes two planner families behind the same runtime
contracts:

- internal custom backends (`td3`, `sac`, `ppo`)
- direct SB3-backed wrappers (`td3_sb3`, `sac_sb3`, `ppo_sb3`)

That has been useful for parity work, but it is not the best long-term base for
research extensions.

For the next phase of the thesis, the codebase needs all of the following:

- future lexicographic objectives
- future distributional variants
- reuse of existing thesis-specific encoder/decoder modules
- stable orchestration around reward, rulebook, curriculum, evaluation, and
  analysis

If deep algorithmic changes remain outside SB3, the repo risks maintaining a
second unofficial SB3 clone inside `thesis_rl`. That would make future changes
to replay semantics, targets, losses, and update loops harder to reason about
and validate.

## Recommendation

Recommended direction:

1. adopt a local SB3 fork as the algorithmic core
2. keep `thesis_rl` as the thesis-specific research and orchestration layer
3. migrate gradually, preserving current custom backends as temporary `legacy`
   implementations
4. prioritize `TD3` and `SAC`
5. defer `PPO` until the off-policy migration pattern is stable

This is a "fork early, patch late" strategy:

- create and integrate the fork now
- keep the initial diff from upstream small
- move only truly algorithmic changes into the fork when they become necessary

## Current repo constraints

The current project assumes a sibling `third-party/` directory on disk:

```text
<parent>/
  thesis-metadrive/
  third-party/
    metadrive/
```

At the moment:

- `pyproject.toml` resolves MetaDrive from `../third-party/metadrive`
- `pyproject.toml` resolves SB3 from `../third-party/stable-baselines3`
- `compose.yaml` mounts both sibling checkouts:
  - `../third-party/metadrive`
  - `../third-party/stable-baselines3`

The intended SB3 fork location is:

- `/home/e.respino/main/thesis/third-party/stable-baselines3`

## Target architecture

The migration target is a 3-layer layout.

### 1. SB3 fork layer

Location:

- preferred if keeping sibling external checkout:
  - `../third-party/stable-baselines3`
- alternative if vendoring inside repo:
  - `third_party/stable-baselines3`

Responsibilities:

- scalar reference implementations
- replay / rollout buffers
- target computation
- Bellman updates
- loss logic
- entropy tuning
- distributional support
- lexicographic objective support

This layer should own changes that alter the semantics of learning.

### 2. Thesis SB3 extension layer

Proposed package:

- `src/thesis_rl/sb3_extensions/`

Responsibilities:

- custom feature extractors
- custom policy builders
- custom actor / critic heads
- config adapters from Hydra to SB3 `policy_kwargs`
- helpers for wiring thesis modules into forked SB3 algorithms

This layer bridges thesis-specific networks to the SB3 fork without copying the
same model code into the fork itself.

### 3. Thesis runtime layer

Existing package:

- `src/thesis_rl/`

Responsibilities:

- environment factory and wrappers
- reward manager and rulebook integration
- curriculum management
- agent runtime and lifecycle contracts
- evaluation, logging, checkpoint ranking, analysis
- experiment configuration

This layer should remain agnostic to whether the underlying algorithm uses
stock SB3, the local fork, or a legacy custom backend.

## Ownership rule

Use this rule to decide where new code belongs.

Place code in `thesis_rl` when the change concerns:

- environment wiring
- reward / rulebook logic
- curriculum logic
- evaluation behavior
- checkpoint ranking and metadata
- experiment configs
- reusable thesis-specific neural modules
- custom feature extractors or policy builders that can plug into SB3

Place code in the SB3 fork when the change concerns:

- replay buffer semantics
- rollout storage semantics
- target computation
- Bellman updates
- training losses
- entropy tuning
- critic output parameterization for distributional RL
- lexicographic objective handling inside optimization
- any algorithm step whose mathematical meaning differs from upstream SB3

## Operational decision rule

When a change is proposed, classify it with this sequence:

1. does it only decide how existing modules are connected?
2. does it only change network architecture while keeping the same SB3 training
   meaning?
3. does it change targets, losses, buffers, rollout semantics, or optimization
   meaning?

If the answer is:

- `1`, it belongs in `thesis_rl`
- `2`, it usually belongs in `thesis_rl.sb3_extensions`
- `3`, it belongs in the SB3 fork

This rule exists to avoid a common failure mode: growing thicker and thicker
wrappers in `thesis_rl` until the project accidentally maintains a second SB3
implementation outside the fork.

### Wiring vs semantics examples

Examples that belong in `thesis_rl`:

- translate Hydra config into SB3 `policy_kwargs`
- select encoder/decoder presets
- add a thesis `features_extractor_class`
- add a custom policy class that still plugs into SB3's normal update logic
- pass a custom replay-buffer class into an unchanged SB3 algorithm
- add runtime logging, metadata, evaluation, checkpoint ranking, curriculum, or
  reward wiring

Examples that belong in the SB3 fork:

- change replay sampling semantics
- change rollout collection semantics
- change Bellman targets
- change actor or critic loss definitions
- change entropy tuning logic
- add quantile / categorical / support-based value targets
- add lexicographic optimization or lexicographic target logic
- change how multiple heads are interpreted during training

### Multi-head rule

The number of heads alone is not the deciding factor.

Keep multi-head work in `thesis_rl` when:

- the heads are architectural only
- SB3 still sees normal feature extraction plus policy/value modules
- training semantics stay equivalent to upstream scalar SB3

Move multi-head work into the fork when:

- the heads require new targets
- the heads require new losses
- the heads require new aggregation or ordering rules during optimization
- the heads change what a critic output mathematically means

In short:

- multi-head architecture only: usually `thesis_rl`
- multi-head training semantics: fork SB3

### Escalation ladder for network customization

Prefer the smallest sufficient integration step:

1. `policy_kwargs` only
2. `features_extractor_class` bridge in `thesis_rl.sb3_extensions`
3. custom SB3 policy class in `thesis_rl.sb3_extensions`
4. SB3 fork changes

Escalate only when the previous layer is no longer expressive enough.

Practical interpretation:

- custom encoder with standard SB3-style head: stay at step 2
- custom head with standard SB3 update logic: usually step 3
- distributional or lexicographic update logic: step 4

## What should not happen

Avoid this pattern:

- keep SB3 from pip as a black box
- grow thicker wrappers in `src/thesis_rl/agent/planners/algorithms/*_sb3.py`
- reimplement more and more of SB3 training logic in `thesis_rl`

That path recreates a second RL framework inside the thesis codebase.

## Configuration refactor target

The current setup uses planner names that mix algorithm family and
implementation, such as:

- `td3`
- `td3_sb3`
- `sac`
- `sac_sb3`

This should be replaced with four explicit axes.

### Proposed planner config axes

- `agent.planner.family`
  - `td3 | sac | ppo`
- `agent.planner.impl`
  - `legacy | sb3_fork`
- `agent.planner.objective`
  - `scalar | lexicographic | distributional | lexicographic_distributional`
- `agent.planner.network`
  - `sb3_mlp | lq | mlp_encoded | none | ...`

Benefits:

- avoids exploding algorithm names
- cleanly separates research objective from implementation backend
- makes future experiment matrices easier to compose and analyze

## Current operational presets

Before that config refactor is done, the repo should treat the following preset
families as the practical migration surface:

- vanilla fork-backed baselines:
  - `presets/agent/td3_sb3`
  - `presets/agent/sac_sb3`
  - `presets/agent/ppo_sb3`
- fork-backed baselines with thesis MLP encoder:
  - `presets/agent/td3_mlp_sb3`
  - `presets/agent/sac_mlp_sb3`
  - `presets/agent/ppo_mlp_sb3`
- fork-backed baselines with thesis encoders/decoders:
  - `presets/agent/td3_lq_sb3`
  - `presets/agent/sac_lq_sb3`
  - `presets/agent/ppo_lq_sb3`

Observation note:

- the `*_lq_sb3` presets assume `obs=semantic_state`
- they should not be paired with `obs=lidar_state` unless the encoder contract
  is intentionally reworked

This gives the project three immediately usable paths:

- reproduce plain SB3-style baselines with the local fork
- use thesis `MLPEncoder` on the same fork-backed algorithm path
- keep thesis-specific network modules while already exercising the fork-backed
  training stack

## Network integration strategy

Existing thesis encoder / decoder modules should remain in `thesis_rl`.

Recommended strategy:

- keep reusable encoder/decoder code in thesis packages
- expose them to SB3 through custom `features_extractor_class`,
  custom policy classes, and custom critic/value heads
- avoid copying those modules into `third-party/stable-baselines3`

This preserves modularity and keeps the fork focused on algorithmic semantics.

## Legacy backend policy

The current custom backends should not be deleted immediately.

Recommended transition policy:

- keep `td3`, `sac`, and `ppo` custom backends available as `legacy`
- stop treating "parity with SB3 inside thesis custom code" as a permanent end
  state
- after the fork-based path is stable, decide whether legacy stays only for
  archival comparison or is removed

## Legacy removal gate

The legacy backends should be removable only when all of the following are true:

- fork-backed `TD3`, `SAC`, and `PPO` cover train, load, and evaluation paths
- the current canonical presets use the fork-backed path
- the required thesis encoder bridge is validated on the fork-backed path
- any required custom replay-buffer hook has a stable integration point
- smoke runs and relevant tests no longer depend on legacy-only behavior
- docs and run commands identify the fork-backed path as the source of truth

Optional but recommended before deletion:

- one final comparison pass between legacy scalar runs and fork-backed scalar
  runs for the experiments that matter most

Until that gate is met, legacy should be treated as:

- temporary compatibility
- historical comparison support
- not the preferred implementation path

## Phased migration plan

### Phase 0 - confirm decisions

Goal:

- lock the integration shape before any nontrivial code moves

Deliverables:

- confirm fork location
- confirm how the fork is mounted into Docker
- confirm naming conventions for config refactor

Exit criterion:

- migration assumptions are written and accepted

### Phase 1 - add the SB3 fork with zero semantic changes

Goal:

- integrate a local fork without changing algorithm behavior

Tasks:

- create the SB3 fork
- add it to local development layout
- make the container able to see it
- point dependency resolution at the local fork instead of only at the pip
  package

Important note:

- dependency switching in `pyproject.toml` should happen only after the sibling
  fork checkout actually exists
- until then, the project should continue to resolve SB3 from the currently
  pinned pip dependency to avoid breaking the active workflow

Exit criterion:

- the project can import SB3 from the local fork inside the Docker container

### Phase 2 - thin the current SB3 wrappers

Goal:

- make the thesis runtime depend on stable contracts, not on ad hoc SB3 details

Tasks:

- keep the planner factory and lifecycle contracts stable
- refactor direct SB3 wrappers into thin adapters over the local fork
- keep train/eval orchestration unchanged at first

Exit criterion:

- runtime behavior is unchanged and wrappers are thinner than today

### Phase 3 - parity stabilization on forked SB3

Goal:

- preserve current validated scalar behavior on the new integration path

Tasks:

- run and maintain current SB3 direct-backend tests
- preserve checkpoint, replay-buffer, and timeout semantics already tested
- verify train/eval still work under current Docker workflow

Exit criterion:

- current scalar fork-backed backends pass the relevant tests and smoke runs

### Phase 4 - refactor config axes

Goal:

- stop encoding implementation details in planner names

Tasks:

- introduce `family`, `impl`, `objective`, and `network`
- preserve backward-compatible aliases during migration
- update presets and analysis labels

Exit criterion:

- new experiments can be expressed without names like `td3_sb3`

### Phase 5 - integrate thesis-specific networks cleanly

Goal:

- reuse existing thesis encoder/decoder code without cloning the whole training
  stack

Tasks:

- create `thesis_rl.sb3_extensions`
- wire custom feature extractors / policies / critic heads
- standardize how Hydra configs map to SB3 policy kwargs

Exit criterion:

- thesis networks can be used by fork-based scalar algorithms cleanly

Additional guardrail:

- if a new network need is expressible through `policy_kwargs`,
  `features_extractor_class`, or a custom thesis-side SB3 policy, do not patch
  the fork yet

### Phase 6 - implement first research variant

Goal:

- establish the first non-upstream algorithm on the new architecture

Recommended starting order:

1. `TD3 + lexicographic`
2. `SAC + lexicographic` or `SAC + distributional`
3. combined lexicographic-distributional variants

Rationale:

- `TD3` has the smallest surface area for the first deep objective change
- `SAC` is likely the stronger long-term off-policy base for richer extensions

Exit criterion:

- one research variant trains on the fork-based stack without duplicating the
  full algorithm in `thesis_rl`

### Phase 7 - legacy cleanup and comparison

Goal:

- decide what remains from the old path

Tasks:

- compare fork-based scalar runs vs legacy runs where still useful
- decide whether legacy stays for archival comparison only
- document the new source of truth
- remove legacy only after the legacy removal gate above is satisfied

Exit criterion:

- the repo has one clearly preferred algorithmic core

## Decision points to confirm

These choices should be confirmed before Phase 1 starts.

### Decision A - where the fork lives

Option 1:

- keep SB3 as a sibling checkout in `../third-party/stable-baselines3`

Pros:

- consistent with current MetaDrive sibling layout
- avoids placing a large third-party repo inside this repo tree
- easy to inspect side by side with MetaDrive

Cons:

- not tracked inside this repo
- requires explicit Compose mounts
- local setup depends on external directory discipline

Option 2:

- vendor SB3 inside this repo, e.g. `third_party/stable-baselines3`

Pros:

- can be tracked directly by this repo
- easier to reason about from the repo alone
- can be managed as submodule or subtree

Cons:

- diverges from current sibling `third-party` workflow
- larger repo working tree

Recommendation:

- if the priority is minimum disruption to the current machine layout, choose
  Option 1
- if the priority is self-contained repo history and cleaner Git ownership,
  choose Option 2

Chosen option:

- Option 1

### Decision B - Git integration mode

Relevant only if SB3 is vendored inside this repo.

Options:

- `submodule`
- `subtree`
- plain committed vendor copy

Recommendation:

- prefer `submodule` if you want explicit upstream/fork separation
- prefer `subtree` only if the team strongly dislikes submodule workflow
- avoid a plain unstructured vendor copy unless there is a strong reason

### Decision C - first research target

Options:

- `TD3 + lexicographic`
- `SAC + lexicographic`
- `SAC + distributional`

Recommendation:

- first implementation target: `TD3 + lexicographic`
- first robust long-term off-policy target after that: `SAC + distributional`

Current near-term scope:

- do not start with lexicographic or distributional variants yet
- first consolidate fork-backed baseline `PPO`, `SAC`, and `TD3`

## Recommended immediate next steps

1. confirm the fork location strategy
2. confirm whether Docker should mount only SB3 or the whole sibling
   `third-party/` directory
3. create the fork and local checkout
4. update Compose and dependency wiring
5. add the first `thesis_rl.sb3_extensions` skeleton

## Concrete recommendation

If choosing based on the current project state alone, the recommended path is:

1. use a local SB3 fork
2. keep thesis-specific modules in `thesis_rl`
3. keep deep learning semantics in the fork
4. migrate `TD3` and `SAC` first
5. keep current custom backends only as temporary legacy references

This gives the cleanest path toward lexicographic and distributional research
without permanently maintaining a second hidden SB3 implementation inside the
thesis codebase.
