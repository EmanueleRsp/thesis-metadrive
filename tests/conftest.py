"""Collection-order hook that keeps the parallel gate short.

Two `integration` cases of `tests/test_reward_return_ordering_runtime.py` take
about 585 s each and dominate the suite. `pytest-xdist` hands out *contiguous*
runs of the collection in every distribution mode (`load` sends chunks,
`worksteal` splits the collection evenly and never steals the last queued item
behind a running test), so adjacent long cases land on one worker and run back
to back: measured 19m50s (`--dist load`) and 19m36s (`--dist worksteal`) at
`-n 16`, against 28m00s sequential, with the two Waymo cases accounting for
almost the whole run in both.

The fix is the position of those items, not the scheduler. Marked items are
spread evenly through the collection, so no chunk holds two of them, and since
xdist only offers a worker more work when it completes a test, a worker running
a long case is left alone until it finishes. The hook changes execution order
only; every test still runs exactly once, and the relative order of the marked
items and of the unmarked items is preserved. With one worker (or without
xdist) the same order is used and is harmless.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import islice
from typing import TypeVar

import pytest

T = TypeVar("T")

SPREAD_MARKER = "integration"


def spread_marked_items(items: Sequence[T], is_marked: Callable[[T], bool]) -> list[T]:
    """Return ``items`` with the marked ones spaced evenly, each leading a run.

    Marked items keep their relative order, unmarked items keep theirs, and the
    result is a permutation of the input: the same length, the same elements.
    With ``m`` marked and ``n`` unmarked items, consecutive marked items are
    separated by ``n // m`` unmarked ones and the remainder trails at the end.
    """

    marked = [item for item in items if is_marked(item)]
    unmarked = [item for item in items if not is_marked(item)]
    if not marked or not unmarked:
        return list(items)
    gap = len(unmarked) // len(marked)
    remaining = iter(unmarked)
    spread: list[T] = []
    for item in marked:
        spread.append(item)
        spread.extend(islice(remaining, gap))
    spread.extend(remaining)
    return spread


def pytest_collection_modifyitems(
    session: pytest.Session, config: pytest.Config, items: list[pytest.Item]
) -> None:
    del session, config
    items[:] = spread_marked_items(
        items, lambda item: item.get_closest_marker(SPREAD_MARKER) is not None
    )
