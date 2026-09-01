"""Guard the one duplicated formula in the expert-cost measurement tool.

`scripts/measure_expert_rulebook_costs.py` reuses the runtime's own candidate
selection, so measurement and runtime cannot diverge there. The single exception
is `parametric_safe_distance_m`, which re-derives the RSS longitudinal safe
distance with the response time left free, because the production formula freezes
it as a module constant and a measurement must not mutate it.

`TEST-RSEC-001` pins that duplication to production at the frozen response time.
Without it, a future edit to `components/rss.py` would silently invalidate every
measurement the tool has produced, including the one that decides `DEC-RSEC-001`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from thesis_rl.rulebook.v2.components.rss import RESPONSE_TIME_S, safe_distance_m


def load_measurement_module() -> ModuleType:
    module_path = Path(__file__).parents[1] / "scripts" / "measure_expert_rulebook_costs.py"
    spec = importlib.util.spec_from_file_location("measure_expert_rulebook_costs", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # `@dataclass` resolves `sys.modules[cls.__module__]` while processing the
    # class body, so a module executed outside `sys.modules` raises
    # `AttributeError: 'NoneType' object has no attribute '__dict__'`. Registering
    # it first is required for any script that defines a dataclass.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("ego_speed_mps", [0.0, 0.1, 3.0, 7.5, 10.0, 15.0, 22.2])
@pytest.mark.parametrize("front_speed_mps", [0.0, 5.0, 12.0])
@pytest.mark.parametrize("ego_brake_mps2", [2.7, 8.0])
def test_parametric_safe_distance_matches_production_at_the_frozen_response_time(
    ego_speed_mps: float, front_speed_mps: float, ego_brake_mps2: float
) -> None:
    """TEST-RSEC-001: the sweep formula is production at `rho = RESPONSE_TIME_S`."""

    module = load_measurement_module()
    measured = module.parametric_safe_distance_m(
        ego_speed_mps=ego_speed_mps,
        front_speed_mps=front_speed_mps,
        ego_brake_mps2=ego_brake_mps2,
        response_time_s=RESPONSE_TIME_S,
    )
    production = safe_distance_m(
        ego_speed_mps=ego_speed_mps,
        front_speed_mps=front_speed_mps,
        ego_brake_mps2=ego_brake_mps2,
    )
    assert measured == pytest.approx(production, rel=0.0, abs=1e-12)


def test_parametric_safe_distance_is_monotone_in_the_response_time() -> None:
    """A shorter response time can never require a longer safe distance."""

    module = load_measurement_module()
    distances = [
        module.parametric_safe_distance_m(
            ego_speed_mps=10.0,
            front_speed_mps=10.0,
            ego_brake_mps2=8.0,
            response_time_s=rho,
        )
        for rho in (0.3, 0.4, 0.5, 0.75, 1.0)
    ]
    assert distances == sorted(distances)


def test_selected_records_filters_by_split_source_and_eligibility() -> None:
    """The measurement must not leak validation or test records into the sample."""

    module = load_measurement_module()
    payload = {
        "records": [
            {"scenario_uid": "w-train", "source": "waymo", "split": "train"},
            {"scenario_uid": "w-test", "source": "waymo", "split": "test"},
            {"scenario_uid": "pg-train", "source": "pg", "split": "train"},
            {
                "scenario_uid": "w-train-ineligible",
                "source": "waymo",
                "split": "train",
                "rulebook_eligible": False,
            },
        ]
    }
    selected = module.selected_records(payload, split="train", source="waymo", limit=None)
    assert [record["scenario_uid"] for record in selected] == ["w-train"]
