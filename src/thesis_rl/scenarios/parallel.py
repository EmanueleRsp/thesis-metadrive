"""Bounded deterministic process-pool helpers for offline ScenarioNet work."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
import multiprocessing
from collections.abc import Callable, Sequence
from typing import TypeVar, cast


T = TypeVar("T")
R = TypeVar("R")
ProgressCallback = Callable[[int, int], None]
_MISSING = object()


def ordered_process_map(
    items: Sequence[T],
    worker: Callable[[T], R],
    *,
    workers: int = 1,
    progress_callback: ProgressCallback | None = None,
) -> tuple[R, ...]:
    """Map top-level worker calls in bounded processes and preserve input order."""

    if workers < 1:
        raise ValueError("workers must be positive")
    total = len(items)
    if not total:
        return ()

    if workers == 1:
        results = []
        for completed, item in enumerate(items, start=1):
            results.append(worker(item))
            if progress_callback is not None:
                progress_callback(completed, total)
        return tuple(results)

    worker_count = min(workers, total)
    max_in_flight = worker_count * 2
    results: list[R | object] = [_MISSING] * total
    pending: dict[Future[R], int] = {}
    next_index = 0

    def submit_until_full(executor: ProcessPoolExecutor) -> None:
        nonlocal next_index
        while next_index < total and len(pending) < max_in_flight:
            future = executor.submit(worker, items[next_index])
            pending[future] = next_index
            next_index += 1

    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
        submit_until_full(executor)
        completed = 0
        while pending:
            done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for future in done:
                index = pending.pop(future)
                results[index] = future.result()
                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, total)
            submit_until_full(executor)

    if any(result is _MISSING for result in results):
        raise RuntimeError("parallel map completed with missing results")
    return tuple(cast(R, result) for result in results)


__all__ = ["ProgressCallback", "ordered_process_map"]
