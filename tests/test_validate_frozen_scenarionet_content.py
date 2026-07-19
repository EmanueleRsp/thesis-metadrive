from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_validator_module():
    path = Path(__file__).parents[1] / "scripts" / "validate_frozen_scenarionet_content.py"
    spec = importlib.util.spec_from_file_location("frozen_content_validator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validator_accepts_array_like_dataset_sequences() -> None:
    """Regression for numpy-backed ScenarioDescription state arrays."""

    class ArrayLike:
        def __init__(self, values: list[float]) -> None:
            self._values = values

        def __len__(self) -> int:
            return len(self._values)

        def __getitem__(self, index: int) -> float:
            return self._values[index]

    validator = _load_validator_module()

    assert validator._sequence_has_length(ArrayLike([1.0, 2.0]), 2)
    assert validator._is_finite_number(ArrayLike([3.0])[0])
