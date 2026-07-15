from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from math import inf

import pytest
from shapely.geometry import Polygon

from thesis_rl.rulebook.v2 import (
    RulebookEvaluationError,
    RulebookV2Config,
    TaskRouteRecord,
    load_rulebook_v2_config,
)
from thesis_rl.rulebook.v2.config import ExecutionConfig, RULEBOOK_V2_VERSION
from thesis_rl.rulebook.v2.errors import EvaluationFailure
from thesis_rl.rulebook.v2.registry import (
    DEFAULT_RULEBOOK_V2_REGISTRY,
    ComponentDefinition,
    RulebookV2Registry,
)
from thesis_rl.rulebook.v2.types import (
    ActorClass,
    ActorSnapshot,
    ComponentStatus,
    MacroRule,
    RuleComponentResult,
    RulebookResult,
)


def test_task_route_is_immutable_and_has_no_future_trajectory_fields() -> None:
    route = TaskRouteRecord("scenario-1", ("lane-a", "lane-b"), "waymo_offline", "v2", "abc")
    with pytest.raises(FrozenInstanceError):
        route.lane_ids = ()  # type: ignore[misc]
    assert "timestamp" not in route.__dataclass_fields__
    assert "future" not in " ".join(route.__dataclass_fields__)


def test_v2_config_rejects_nonconformant_execution_or_order() -> None:
    RulebookV2Config().validate()
    with pytest.raises(ValueError, match="fail-fast"):
        RulebookV2Config(execution=ExecutionConfig(silent_fallbacks=True)).validate()
    bad = RulebookV2Config(order=(MacroRule.ROUTE_PROGRESS,))
    with pytest.raises(ValueError, match="order"):
        bad.validate()
    with pytest.raises(ValueError, match="Unsupported"):
        load_rulebook_v2_config({"version": RULEBOOK_V2_VERSION, "unexpected": True})


def test_registry_rejects_memory_double_writer() -> None:
    duplicate = DEFAULT_RULEBOOK_V2_REGISTRY.components + (
        ComponentDefinition(
            "collision", MacroRule.COLLISION_IMPACT, None, frozenset({"previous_contact_ids"})
        ),
    )
    with pytest.raises(ValueError, match="duplicate"):
        RulebookV2Registry(duplicate)


def test_evaluation_error_keeps_typed_context() -> None:
    failure = EvaluationFailure("scenario-7", 12, "signal", "invalid state")
    error = RulebookEvaluationError(failure)
    assert error.failure == failure
    assert "scenario-7" in str(error)


def test_shapely_fixture_is_available_for_canonical_contracts() -> None:
    assert Polygon(((0, 0), (1, 0), (1, 1), (0, 0))).is_valid


def test_result_diagnostics_are_json_serializable() -> None:
    component = RuleComponentResult(
        "collision", 0.0, {"closing_speed_mps": 0.0}, True, True,
        ComponentStatus.SATISFIED, {"actors": ()},
    )
    result = RulebookResult(
        (0.0, 0.0, 0.0, 0.1), (0.0, 0.0, 0.0), 0.2, {"collision": component}, True
    )
    payload = result.to_dict()
    assert json.loads(json.dumps(payload))["components"]["collision"]["status"] == "satisfied"


def test_snapshot_and_result_reject_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        ActorSnapshot(
            "ego", ActorClass.VEHICLE, (inf, 0.0), 0.0, 0.0, (0.0, 0.0), Polygon(), None, 20.0
        )
    with pytest.raises(ValueError, match="finite"):
        RulebookResult((0.0, 0.0, 0.0, inf), (0.0, 0.0, 0.0), 0.0, {}, True)
