import pytest
from shapely.geometry import Polygon
from thesis_rl.rulebook.v2.aggregation import aggregate_rulebook_result
from thesis_rl.rulebook.v2.errors import RulebookEvaluationError
from thesis_rl.rulebook.v2.monitor import (
    evaluate_monitor_transition,
    evaluate_registered_transition,
)
from thesis_rl.rulebook.v2.types import (
    ActorClass,
    ActorSnapshot,
    CacheDelta,
    ComponentStatus,
    ConflictZoneRecord,
    EnvSnapshot,
    MemoryDelta,
    MovementKey,
    RuleComponentResult,
    RulebookMemory,
)


def _component(name, cost, applicable=True):
    return RuleComponentResult(
        name,
        cost,
        {},
        applicable,
        True,
        ComponentStatus.VIOLATED if cost else ComponentStatus.SATISFIED,
        {},
    )


def test_aggregation_produces_ordered_six_margins_and_preserves_components():
    """RULEBOOK-V5.1 §3. One margin per level, in `MACRO_RULE_ORDER`.

    Cost levels are carried negated so that larger is better on every entry and
    a lexicographic consumer can compare the vector without being told which
    entries are costs. Levels with no applicable sub-rule contribute `+0.0`, not
    `-0.0`.
    """

    result = aggregate_rulebook_result(
        components=(
            _component("collision", 1.0),
            _component("ttc", 0.4),
            _component("offroad", 0.2),
            _component("solid_line", 0.9),
            _component("advance_shortfall", 0.6),
        ),
        raw_progress_m=0.5,
        progress_margin=0.3,
    )
    # L5 divides by a declared three even though one lane sub-rule applied.
    assert result.margins == (-1.0, -0.4, -0.2, 0.3, pytest.approx(-0.3), -0.6)
    assert "ttc" in result.components and "non_relaxable_compliance" in result.components
    # The atomic sub-rule survives alongside its level.
    assert "advance_shortfall" in result.components and "progress_rate" in result.components


def test_monitor_merges_memory_only_after_valid_result():
    output = (_component("offroad", 0.0), MemoryDelta(), CacheDelta())
    result, memory, cache = evaluate_monitor_transition(
        memory=RulebookMemory(),
        component_outputs=(output,),
        raw_progress_m=1.0,
        progress_margin=1.0,
    )
    assert (
        result.complete_evaluation and memory == RulebookMemory() and cache.new_conflict_zones == ()
    )


def test_monitor_builds_and_commits_history_preview_from_post_snapshot():
    footprint = Polygon(((-1, -0.5), (1, -0.5), (1, 0.5), (-1, 0.5)))
    ego = ActorSnapshot(
        "ego", ActorClass.VEHICLE, (0.0, 0.0), 0.0, 0.0, (1.0, 0.0), footprint, "lane", 20.0
    )
    post = EnvSnapshot("scenario", 1, 0.1, ego, (), (), frozenset(), {})
    output = (_component("offroad", 0.0), MemoryDelta(), CacheDelta())
    _, next_memory, _ = evaluate_monitor_transition(
        memory=RulebookMemory(previous_sim_time_s=0.0),
        component_outputs=(output,),
        raw_progress_m=0.0,
        progress_margin=0.0,
        post_state=post,
    )
    assert next_memory.previous_sim_time_s == 0.1
    assert tuple(history.actor_id for history in next_memory.actor_motion_histories) == ("ego",)


def test_aggregation_rejects_incomplete_component():
    component = RuleComponentResult(
        "offroad", 0.0, {}, True, False, ComponentStatus.NOT_EVALUABLE, {}
    )
    with pytest.raises(ValueError):
        aggregate_rulebook_result(components=(component,), raw_progress_m=0.0, progress_margin=0.0)


def test_registered_transition_rejects_partial_component_inputs():
    with pytest.raises(ValueError, match="Missing evaluator inputs"):
        evaluate_registered_transition(
            memory=RulebookMemory(),
            component_inputs={},
            raw_progress_m=0.0,
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
        MemoryDelta("progress"),
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
