from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from thesis_rl.agents.planner_lifecycle import BasePlannerLifecycle



class BasePlanner(Protocol):
    '''Protocol for planner backends used by PlannerAgent.
     Planners must implement a stable interface for training lifecycle and action prediction,
     allowing the PlannerAgent to orchestrate training without coupling to specific RL libraries.
     This enables flexible composition of algorithms, curricula, and interventions while
     maintaining clean separation of concerns between agent orchestration and planner internals.
     Methods:
         `get_lifecycle()`: Return a `BasePlannerLifecycle` instance for granular training control.
         `predict(observation, deterministic)`: Compute action from observation.
         `save(checkpoint_path)`: Save model state to checkpoint.
         `set_env(env)`: Update the planner's environment reference (if applicable).
     '''

    def get_lifecycle(self) -> BasePlannerLifecycle: ...

    def predict(self, observation: Any, deterministic: bool = False): ...

    def save(self, checkpoint_path: str | Path) -> None: ...

    def set_env(self, env: Any) -> None: ...

