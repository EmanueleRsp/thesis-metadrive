import pytest
from shapely.geometry import Polygon
from thesis_rl.rulebook.v2.aggregation import aggregate_rulebook_result
from thesis_rl.rulebook.v2.errors import RulebookEvaluationError
from thesis_rl.rulebook.v2.monitor import evaluate_monitor_transition, evaluate_registered_transition
from thesis_rl.rulebook.v2.types import CacheDelta, ComponentStatus, ConflictZoneRecord, MemoryDelta, MovementKey, RuleComponentResult, RulebookMemory

def _component(name, cost, applicable=True):
    return RuleComponentResult(name, cost, {}, applicable, True, ComponentStatus.VIOLATED if cost else ComponentStatus.SATISFIED, {})

def test_aggregation_produces_ordered_four_margins_and_preserves_components():
    result = aggregate_rulebook_result(components=(_component("collision", 1.0), _component("ttc", 0.4), _component("offroad", 0.2)), raw_progress_m=0.5, progress_margin=0.3)
    assert result.margins == (-1.0, -0.4, -0.2, 0.3)
    assert "ttc" in result.components and "road_traffic_compliance" in result.components

def test_monitor_merges_memory_only_after_valid_result():
    output = (_component("offroad", 0.0), MemoryDelta(), CacheDelta())
    result, memory, cache = evaluate_monitor_transition(memory=RulebookMemory(), component_outputs=(output,), raw_progress_m=1.0, progress_margin=1.0)
    assert result.complete_evaluation and memory == RulebookMemory() and cache.new_conflict_zones == ()

def test_aggregation_rejects_incomplete_component():
    component = RuleComponentResult("offroad", 0.0, {}, True, False, ComponentStatus.NOT_EVALUABLE, {})
    with pytest.raises(ValueError):
        aggregate_rulebook_result(components=(component,), raw_progress_m=0.0, progress_margin=0.0)


def test_registered_transition_rejects_partial_component_inputs():
    with pytest.raises(ValueError, match="Missing evaluator inputs"):
        evaluate_registered_transition(
            memory=RulebookMemory(), component_inputs={}, raw_progress_m=0.0,
            progress_margin=0.0,
        )


def test_monitor_commit_is_atomic_when_cache_validation_fails_after_memory_proposal():
    zone = ConflictZoneRecord(
        "zone",
        Polygon(((0, 0), (1, 0), (1, 1), (0, 1))),
        MovementKey("a", "n", "e"),
        MovementKey("b", "n", "f"),
        0.0,
        1.0,
        0.0,
    )
    conflicting_zone = ConflictZoneRecord(
        "zone",
        Polygon(((2, 0), (3, 0), (3, 1), (2, 1))),
        zone.ego_movement_key,
        zone.other_movement_key,
        zone.route_entry_s_m,
        zone.route_exit_s_m,
        zone.elevation_m,
    )
    memory = RulebookMemory()
    output = (
        _component("offroad", 0.0),
        MemoryDelta("progress", (("previous_route_s_m", 3.0),)),
        CacheDelta((zone,)),
    )
    with pytest.raises(RulebookEvaluationError):
        evaluate_monitor_transition(
            memory=memory,
            component_outputs=(output,),
            raw_progress_m=1.0,
            progress_margin=0.0,
            pending_cache_delta=CacheDelta((conflicting_zone,)),
        )
    assert memory == RulebookMemory()
