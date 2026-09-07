"""Audit 2026-09-07, C23 and C25: two training statistics that asserted a falsehood.

C23 -- an aborted episode has no success, no collision and no off-road, so the
trailing ``else`` in the reason chain filed it under ``timeout``. The timeout
rate is the statistic used to judge how much of the training distribution is
reaching the horizon, and the abort rate sat at exactly zero beside it, so both
numbers were wrong in the same direction and neither showed it.

C25 -- ``train_reset_seed_unique_count`` reported ``0`` whenever no reset-seed
function drives resets, which is the normal ScenarioNet case. A reproducibility
column reading "zero distinct seeds" is worse than an empty one: it looks like
evidence.
"""

from __future__ import annotations

import inspect

import pytest

from thesis_rl.agent.agent import _episode_termination_reason, _unique_reset_seed_count


@pytest.mark.parametrize(
    ("success", "collision", "out_of_road", "expected"),
    [
        (True, False, False, "success"),
        (False, True, False, "collision"),
        (False, False, True, "out_of_road"),
        (False, False, False, "timeout"),
        # Priority is preserved when several flags are set at once.
        (True, True, True, "success"),
        (False, True, True, "collision"),
    ],
)
def test_unaborted_outcomes_keep_their_previous_classification(
    success: bool, collision: bool, out_of_road: bool, expected: str
) -> None:
    assert (
        _episode_termination_reason(success=success, collision=collision, out_of_road=out_of_road)
        == expected
    )


def test_an_abort_is_reported_as_aborted_not_as_a_timeout() -> None:
    assert (
        _episode_termination_reason(success=False, collision=False, out_of_road=False, aborted=True)
        == "aborted"
    )


def test_abort_outranks_every_other_outcome() -> None:
    """A slot that aborted produced no observable outcome to report.

    Whatever the accumulators happen to hold for that slot describes the episode
    the abort interrupted, not one that finished, so reporting it as a success or
    a collision would put a fabricated outcome into the statistics.
    """

    for success, collision, out_of_road in (
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, True, True),
    ):
        assert (
            _episode_termination_reason(
                success=success,
                collision=collision,
                out_of_road=out_of_road,
                aborted=True,
            )
            == "aborted"
        )


def test_the_reason_chain_has_exactly_one_implementation() -> None:
    """Three inline copies is how a fourth outcome gets forgotten in two of them."""

    parameters = inspect.signature(_episode_termination_reason).parameters
    assert set(parameters) == {"success", "collision", "out_of_road", "aborted"}
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY for parameter in parameters.values()
    )


def test_no_reset_seeds_reports_not_applicable_rather_than_zero() -> None:
    """ScenarioNet selects records by UID, so MetaDrive's seed never enters the choice."""

    assert _unique_reset_seed_count([]) is None


def test_reset_seed_count_still_counts_distinct_seeds_when_they_exist() -> None:
    assert _unique_reset_seed_count([7]) == 1
    assert _unique_reset_seed_count([7, 7, 7]) == 1
    assert _unique_reset_seed_count([3, 1, 4, 1, 5]) == 4
