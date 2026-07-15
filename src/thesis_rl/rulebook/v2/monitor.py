"""Transactional Rulebook v2 monitor orchestration."""

from __future__ import annotations

from thesis_rl.rulebook.v2.aggregation import aggregate_rulebook_result
from thesis_rl.rulebook.v2.memory import merge_cache_deltas, merge_memory_deltas
from thesis_rl.rulebook.v2.types import CacheDelta, MemoryDelta, RulebookMemory, RuleComponentResult


def evaluate_monitor_transition(*, memory: RulebookMemory, component_outputs: tuple[tuple[RuleComponentResult, MemoryDelta, CacheDelta], ...],
                                raw_progress_m: float, progress_margin: float, pending_cache_delta: CacheDelta = CacheDelta()):
    """Commit all evaluator deltas only after complete result validation."""
    results = tuple(output[0] for output in component_outputs)
    memory_deltas = tuple(output[1] for output in component_outputs if output[1].writes)
    cache_deltas = (pending_cache_delta,) + tuple(output[2] for output in component_outputs)
    result = aggregate_rulebook_result(components=results, raw_progress_m=raw_progress_m, progress_margin=progress_margin)
    if not result.complete_evaluation:
        raise ValueError("Monitor cannot return incomplete evaluation")
    next_memory = merge_memory_deltas(memory, memory_deltas)
    cache_delta = merge_cache_deltas(cache_deltas)
    return result, next_memory, cache_delta
