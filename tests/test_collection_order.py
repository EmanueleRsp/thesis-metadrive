"""The collection-order hook spreads `integration` items without losing any."""

from __future__ import annotations

import pytest
from conftest import spread_marked_items


def _is_slow(item: str) -> bool:
    return item.startswith("slow")


def test_spread_is_a_permutation_preserving_both_relative_orders() -> None:
    items = [f"fast{i}" for i in range(20)]
    items[7:7] = ["slow0", "slow1", "slow2"]
    items.append("slow3")

    spread = spread_marked_items(items, _is_slow)

    assert sorted(spread) == sorted(items)
    assert len(spread) == len(items)
    assert [item for item in spread if _is_slow(item)] == ["slow0", "slow1", "slow2", "slow3"]
    assert [item for item in spread if not _is_slow(item)] == [f"fast{i}" for i in range(20)]


def test_marked_items_are_separated_by_the_floor_of_the_ratio() -> None:
    items = ["slow0", "slow1", "slow2", "slow3"] + [f"fast{i}" for i in range(22)]

    spread = spread_marked_items(items, _is_slow)

    positions = [index for index, item in enumerate(spread) if _is_slow(item)]
    # 22 unmarked / 4 marked -> a gap of 5 unmarked items between marked ones,
    # and the 2 left over trail at the end.
    assert positions == [0, 6, 12, 18]
    assert spread[-2:] == ["fast20", "fast21"]


@pytest.mark.parametrize(
    "items",
    [[], ["fast0", "fast1"], ["slow0", "slow1"], ["slow0"], ["fast0"]],
)
def test_nothing_to_spread_returns_the_input_order(items: list[str]) -> None:
    assert spread_marked_items(items, _is_slow) == items


def test_marked_items_are_never_adjacent_in_this_suite(request: pytest.FixtureRequest) -> None:
    """Live check on the real collection: two spread items never touch.

    The stride that makes the parallel gate short is the one the hook computed
    for the session this test runs in, so the property is checked on the actual
    item list rather than on a synthetic one. Skips when the marker is absent
    or deselected, because then there is nothing to spread.
    """

    items = request.session.items
    positions = [
        index for index, item in enumerate(items) if item.get_closest_marker("integration")
    ]
    if len(positions) < 2:
        pytest.skip("fewer than two `integration` items collected in this session")
    gaps = [later - earlier for earlier, later in zip(positions, positions[1:])]
    assert min(gaps) > 1, f"integration items collected adjacently: positions {positions}"
