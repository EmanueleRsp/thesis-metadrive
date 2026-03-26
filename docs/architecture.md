# Architecture

## 1. High-level objective

The codebase must support research on modular autonomous-driving RL in MetaDrive, with a future lexicographic multi-objective planner. The system should allow fast replacement of:

- observation preprocessing
- planner core
- action adaptation
- reward / violation computation
- scenario curriculum

## 2. End-to-end pipeline

```text
MetaDrive observation
    ↓
Preprocessor
    ↓
State representation
    ↓
Planner
    ↓
Planner output
    ↓
Adapter
    ↓
MetaDrive action
    ↓
Environment step
    ↓
Scalar reward + info
    ↓
Reward manager / rulebook evaluator
    ↓
Violation vector
```

## 3. Main modules

### 3.1 Environment layer
Responsible for:
- creating MetaDrive envs
- applying wrappers
- exposing train/eval environments
- managing seeds and scenario configs

Suggested files:
- `src/thesis_rl/envs/factory.py`
- `src/thesis_rl/envs/wrappers.py`
- `src/thesis_rl/envs/metadrive_env.py`

### 3.2 Preprocessor layer
Responsible for:
- converting raw observation to planner input
- optional state embedding
- optional normalization / flattening

Initial implementation:
- `IdentityPreprocessor`

Suggested interface:
```python
class BasePreprocessor(Protocol):
    def reset(self) -> None: ...
    def __call__(self, observation) -> Any: ...
```

### 3.3 Planner layer
Responsible for:
- core decision logic
- training-time forward pass
- action prediction
- later lexicographic and distributional reasoning

Initial contract:
- output the final low-level action expected by MetaDrive

Suggested distinction:
- `PlannerModule`: neural module / decision core
- `PlannerAgent`: owns optimizer, replay buffer, update logic

### 3.4 Adapter layer
Responsible for:
- mapping planner output to MetaDrive action space

Initial implementation:
- `DirectActionAdapter`

Responsibilities:
- shape validation
- clipping
- exact formatting

Later:
- waypoint/controller adapter
- MetaDrive `Policy`-compatible action controller

### 3.5 Reward layer
Responsible for:
- collecting default scalar reward
- computing violation-vector components
- keeping MetaDrive compatibility

Important design rule:
the violation vector should not replace the environment API prematurely. Instead, it should be attached to `info` and optionally stored by the training pipeline.

### 3.6 Rulebook layer
Responsible for:
- declaring rule order
- evaluating each rule
- returning ordered violation components
- bridging to the existing rule module later

Suggested interface:
```python
class BaseRule(Protocol):
    name: str
    def reset(self) -> None: ...
    def evaluate(self, transition_context) -> float: ...
```

```python
class RulebookEvaluator:
    def __init__(self, rules: list[BaseRule]): ...
    def evaluate(self, transition_context) -> dict[str, float]: ...
```

### 3.7 Curriculum layer
Responsible for:
- stage configuration
- stage switching
- progression criteria
- train/eval seed ranges

Suggested files:
- `src/thesis_rl/curriculum/stages.py`
- `src/thesis_rl/curriculum/manager.py`

The manager should support:
- fixed stage selection
- performance-based auto-promotion

## 4. Recommended early abstractions

Use these abstractions from the beginning:

- `EnvFactory`
- `Agent`
- `Preprocessor`
- `Planner`
- `Adapter`
- `RewardManager`
- `RulebookEvaluator`
- `CurriculumManager`

Do **not** fully generalize them yet; just define stable interfaces.

## 5. Transition context

The reward/rule system will need more than raw current observation. Introduce a dedicated object early:

```python
@dataclass
class TransitionContext:
    obs: Any
    next_obs: Any
    action: Any
    scalar_reward: float
    done: bool
    truncated: bool
    info: dict
    prev_info: dict | None = None
    raw_env_state: dict | None = None
    prev_raw_env_state: dict | None = None
```

Why this matters:
- some rules are state-based
- some rules are transition-based
- collision severity may need previous-step vehicle states
- future planners may need richer logging and diagnostics

## 6. Baseline path

For the first working version:

- observation -> identity preprocessor
- planner -> simple continuous-control baseline
- adapter -> direct action adapter
- reward -> default scalar only
- curriculum -> off

This gives the minimal baseline without constraining later design.
