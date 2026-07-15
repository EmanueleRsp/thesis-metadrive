import pytest
from thesis_rl.rulebook.v2.aggregation import aggregate_rulebook_result
from thesis_rl.rulebook.v2.monitor import evaluate_monitor_transition
from thesis_rl.rulebook.v2.types import CacheDelta, ComponentStatus, MemoryDelta, RuleComponentResult, RulebookMemory

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
