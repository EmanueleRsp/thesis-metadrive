"""Transactional Rulebook v2 monitor orchestration."""

from __future__ import annotations

from thesis_rl.rulebook.v2.aggregation import aggregate_rulebook_result
from thesis_rl.rulebook.v2.errors import EvaluationFailure, RulebookEvaluationError
from thesis_rl.rulebook.v2.memory import merge_cache_deltas, merge_memory_deltas
from collections.abc import Mapping
from typing import cast

from thesis_rl.rulebook.v2.registry import DEFAULT_RULEBOOK_V2_REGISTRY, RulebookV2Registry
from thesis_rl.rulebook.v2.types import CacheDelta, MemoryDelta, RulebookMemory, RuleComponentResult


def evaluate_monitor_transition(*, memory: RulebookMemory, component_outputs: tuple[tuple[RuleComponentResult, MemoryDelta, CacheDelta], ...],
                                raw_progress_m: float, progress_margin: float, pending_cache_delta: CacheDelta = CacheDelta(),
                                progress_output: tuple[RuleComponentResult, MemoryDelta, CacheDelta] | None = None):
    """Commit all evaluator deltas only after complete result validation."""
    if not isinstance(memory, RulebookMemory):
        raise TypeError("memory must be a RulebookMemory")
    results = tuple(output[0] for output in component_outputs)
    progress_deltas = () if progress_output is None else (progress_output,)
    if progress_output is not None:
        results += (progress_output[0],)
        raw_progress_m = cast(float, progress_output[0].raw["route_delta_m"])
        progress_margin = progress_output[0].cost
    memory_deltas = tuple(output[1] for output in component_outputs + progress_deltas if output[1].writes)
    cache_deltas = (pending_cache_delta,) + tuple(output[2] for output in component_outputs + progress_deltas)
    try:
        result = aggregate_rulebook_result(components=results, raw_progress_m=raw_progress_m, progress_margin=progress_margin)
        if not result.complete_evaluation:
            raise ValueError("Monitor cannot return incomplete evaluation")
        # Validate both append-only cache and memory ownership before exposing
        # either result.  Both operations are immutable, so a failure cannot
        # leave a partially committed transition behind.
        cache_delta = merge_cache_deltas(cache_deltas)
        next_memory = merge_memory_deltas(memory, memory_deltas)
    except (ValueError, TypeError) as exc:
        raise RulebookEvaluationError(
            EvaluationFailure("unknown", -1, "monitor", str(exc))
        ) from exc
    return result, next_memory, cache_delta


def evaluate_registered_transition(
    *,
    memory: RulebookMemory,
    component_inputs: Mapping[str, Mapping[str, object]],
    raw_progress_m: float,
    progress_margin: float,
    pending_cache_delta: CacheDelta = CacheDelta(),
    registry: RulebookV2Registry = DEFAULT_RULEBOOK_V2_REGISTRY,
):
    """Invoke every normative evaluator through the fixed registry.

    The adapter supplies canonical keyword arguments for each component.  A
    missing input is a contract error: callers must invoke the evaluator even
    when its domain is absent so it can return an explicit
    ``NOT_APPLICABLE`` result.
    """
    normative_names = tuple(
        component.name
        for component in registry.components
        if component.normative_output and component.name != "progress"
    )
    supplied = set(component_inputs)
    missing = tuple(name for name in normative_names if name not in supplied)
    if "progress" not in supplied:
        missing += ("progress",)
    unknown = tuple(sorted(supplied.difference(normative_names).difference({"progress"})))
    if missing:
        raise ValueError(f"Missing evaluator inputs: {missing}")
    if unknown:
        raise ValueError(f"Unknown evaluator inputs: {unknown}")
    outputs = tuple(
        registry.evaluate(name, **dict(component_inputs[name]))
        for name in normative_names
    )
    progress_input = component_inputs["progress"]
    progress_output = registry.evaluate("progress", **dict(progress_input))
    return evaluate_monitor_transition(
        memory=memory,
        component_outputs=outputs,
        raw_progress_m=raw_progress_m,
        progress_margin=progress_margin,
        pending_cache_delta=pending_cache_delta,
        progress_output=progress_output,
    )
