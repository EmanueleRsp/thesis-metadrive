from __future__ import annotations

from collections.abc import Callable

from thesis_rl.curriculum.config import CurriculumConfig
from thesis_rl.curriculum.interfaces import CurriculumStrategy
from thesis_rl.curriculum.strategies import StagedCurriculum


StrategyFactory = Callable[[CurriculumConfig], CurriculumStrategy]


_STRATEGY_FACTORIES: dict[str, StrategyFactory] = {
    "staged": lambda config: StagedCurriculum(config.staged),
}


def register_curriculum_strategy(kind: str, factory: StrategyFactory) -> None:
    key = str(kind).strip().lower()
    if not key:
        raise ValueError("Curriculum strategy kind must be a non-empty string.")
    _STRATEGY_FACTORIES[key] = factory


def build_curriculum_strategy(config: CurriculumConfig) -> CurriculumStrategy:
    kind = str(config.kind).strip().lower()
    factory = _STRATEGY_FACTORIES.get(kind)
    if factory is None:
        raise ValueError(f"Unsupported curriculum kind '{kind}'.")
    return factory(config)
