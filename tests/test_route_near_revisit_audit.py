"""Acceptance tests for the near-revisit re-audit instrument.

`scripts/audit_route_near_revisits.py` makes executable the standing constraint
in `docs/implementation/route_coordinate_driving_mission_v1.1_exec_plan.md`
§11.1, which requires the near-revisit audit to be re-run on any regenerated or
extended frozen index. An audit that reports a clean population is only worth
believing if it is also shown to report a dirty one, so the positive control
here is the geometry that `docs/audits/rulebook_architecture_2026-09-09/
g6_ratchet.py` actually executes the ratchet on.

Three properties are asserted, and they are the instrument's contract:

* on a straight route the fold excess is identically zero and no on-route
  under-charge is positive, so ordinary geometry does not trip it;
* on the `g6` hairpin it recovers the arc separation that script's return jump
  charges `-1`, at the lateral separation of the two legs;
* the original 15 m criterion is reproduced, so a run against a future index can
  be compared with the 2026-09-05 figures rather than only with itself.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest

# The lateral separation of the two hairpin legs in `g6_ratchet.py`, which is
# also the lane width both static adapters fall back to.
HAIRPIN_LEG_SEPARATION_M = 3.5
# `g6_ratchet.py` reports this route length, and the ratchet gain it measures is
# one clip width less than the arc separation of the two legs.
HAIRPIN_LENGTH_M = 84.03


def load_audit_module() -> ModuleType:
    module_path = Path(__file__).parents[1] / "scripts" / "audit_route_near_revisits.py"
    spec = importlib.util.spec_from_file_location("audit_route_near_revisits", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # `@dataclass` resolves `sys.modules[cls.__module__]` while processing the
    # class body, so a module executed outside `sys.modules` fails.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def audit() -> ModuleType:
    return load_audit_module()


def _straight_points(length_m: float = 60.0, spacing_m: float = 0.5):
    count = int(length_m / spacing_m) + 1
    return [(index * spacing_m, 0.0, 0.0) for index in range(count)]


def _hairpin_points():
    """The `g6_ratchet.py` hairpin, restated so the two stay comparable."""

    points = [(float(x), 0.0, 0.0) for x in range(0, 41, 2)]
    points.append((41.0, 1.75, 0.0))
    points.extend((float(x), HAIRPIN_LEG_SEPARATION_M, 0.0) for x in range(40, -1, -2))
    return points


def _run(audit: ModuleType, points):
    return audit._audit_route(
        scenario_uid="fixture",
        source="fixture",
        split="fixture",
        route_length_m=float("nan"),
        points_xyz=points,
    )


def test_the_clip_width_is_one_step_of_travel_at_the_reference_speed(audit: ModuleType) -> None:
    """TEST-V3-01: `D_REF` is derived from production, never restated."""

    from thesis_rl.rulebook.v2.components.progress import MISSION_PROGRESS_REFERENCE_SPEED_MPS

    assert audit.D_REF_M == pytest.approx(MISSION_PROGRESS_REFERENCE_SPEED_MPS * 0.1)
    # The value `g6_ratchet.py` and `RULEBOOK-V5.1` §4.1 both use.
    assert audit.D_REF_M == pytest.approx(2.2222, abs=1.0e-4)


def test_a_straight_route_has_no_fold_and_no_positive_under_charge(audit: ModuleType) -> None:
    """TEST-V3-02: the instrument is silent on geometry that cannot fold.

    This is the property the naive threshold substitution would fail. A criterion
    of "two portions more than one clip width apart in arc length, within one
    lane width" fires on every straight route, because arc length equals the
    chord there; pairing the arc separation with the planar separation is what
    makes the criterion mean something.
    """

    result = _run(audit, _straight_points())

    assert result.error == ""
    for index, bound in enumerate(audit.PROXIMITY_BANDS_M):
        if bound == math.inf:
            continue
        assert result.band_max_fold_excess_m[index] == pytest.approx(0.0, abs=1.0e-9), bound
        assert result.band_max_undercharge[index] <= audit.UNDERCHARGE_EPSILON, bound
    assert result.one_step_arc_multiple_counts[0] == 0
    # Nothing on a straight route is ever further apart in arc length than in
    # plane, so the original criterion finds no near-revisit at any separation.
    assert result.legacy_min_planar_m > audit.LEGACY_PROXIMITY_BANDS_M[-1]


def test_the_hairpin_the_ratchet_runs_on_is_detected(audit: ModuleType) -> None:
    """TEST-V3-03: the positive control, tied to the executed exploit.

    `g6_ratchet.py` drives this route and banks `+36.00` channel units per lap at
    zero net displacement, because the `-84.03 m` return jump is charged `-1`.
    The audit must see that from the geometry alone: two portions one lane apart
    whose arc separation is the whole route.
    """

    result = _run(audit, _hairpin_points())

    assert result.error == ""
    lane_band = audit.PROXIMITY_BANDS_M.index(HAIRPIN_LEG_SEPARATION_M)

    # The two legs are one lane apart and nearly the whole route apart in arc
    # length, so the fold excess at that band is the route minus the lane.
    assert result.band_max_fold_excess_m[lane_band] > HAIRPIN_LENGTH_M - 10.0
    arc_at_worst = result.band_fold_excess_arc_m[lane_band]
    assert arc_at_worst > HAIRPIN_LENGTH_M - 10.0
    assert result.band_fold_excess_planar_m[lane_band] <= HAIRPIN_LEG_SEPARATION_M

    # And the branch-switch under-charge it implies is the gain `g6` measures:
    # the jump is worth `arc / D_REF` and is charged one.
    switch_under_charge = arc_at_worst / audit.D_REF_M - 1.0
    assert switch_under_charge > 30.0

    # The original criterion sees it too, which is why it was the right audit for
    # its own question even though its threshold is too coarse for this one.
    assert result.legacy_min_planar_m <= HAIRPIN_LEG_SEPARATION_M


def test_the_hairpin_is_invisible_at_one_step_of_travel(audit: ModuleType) -> None:
    """TEST-V3-04: the two regimes are distinct, and the report must show both.

    The hairpin legs are 3.5 m apart, further than one step of travel, so an ego
    that stays on the route cannot realise the jump in a single step. Reporting
    only the one-step figure would therefore call this route clean while `g6`
    executes the ratchet on it — which is why the fold-excess table exists.
    """

    result = _run(audit, _hairpin_points())

    one_step_band = 0
    assert audit.PROXIMITY_BANDS_M[one_step_band] == audit.D_REF_M
    assert result.band_max_undercharge[one_step_band] <= audit.UNDERCHARGE_EPSILON
    assert result.one_step_arc_multiple_counts[0] == 0
    # But the fold is visible one band out, at a 1.75 m lateral excursion.
    assert result.band_max_fold_excess_m[one_step_band] < 1.0
    assert result.band_max_fold_excess_m[1] > HAIRPIN_LENGTH_M - 10.0


def test_a_gentle_curve_produces_curvature_not_a_fold(audit: ModuleType) -> None:
    """TEST-V3-05: ordinary curvature stays below the ADR-035 continuity factor.

    Cutting the inside of a bend genuinely advances the centerline coordinate
    faster than the ego moves, which is why ADR-035 sized its continuity bound at
    a factor of `2.0` rather than `1.0`. A quarter circle of a radius a road
    actually turns at must land well inside that factor, or the bound would be
    rejecting legitimate motion.
    """

    radius_m = 12.0
    steps = 200
    points = [
        (
            radius_m * math.cos(math.pi / 2 * index / steps),
            radius_m * math.sin(math.pi / 2 * index / steps),
            0.0,
        )
        for index in range(steps + 1)
    ]

    result = _run(audit, points)

    assert result.error == ""
    # No arc separation above one clip width is reachable in one step of travel.
    assert result.one_step_arc_multiple_counts[0] == 0
    assert result.band_max_undercharge[0] <= audit.UNDERCHARGE_EPSILON
    # And the worst arc separation within one step of travel is inside twice the
    # clip width, i.e. inside the factor ADR-035 chose.
    assert result.one_step_max_arc_m < 2.0 * audit.D_REF_M
