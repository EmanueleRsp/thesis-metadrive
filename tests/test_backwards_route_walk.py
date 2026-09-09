"""Acceptance tests for `D14`'s backwards route walk.

The walk lives in `scripts/measure_final_gate_carriageway_coverage.py` as its
`--walk-spacing-m` mode. Its **geometry** needs no tests here: it calls the same
`_decompose` the instrument already reconciles against the frozen final gate on
every record, at a maximum residual of 3.5e-12 m over all 3,500. What is new and
therefore untested is the **selection and continuity logic** built on top of it —
which corridor at the goal counts as an exposure, and when a corridor is still
the same corridor one station back.

That logic is what the `D14` conclusion rests on, and each of the three
properties below was got wrong first and fixed:

* only the *nearest* qualifying corridor was walked, so a short one masked a long
  one further out;
* a corridor qualified on its *centre* missing the gate, which under-reports a
  wide component whose centre sweeps the gate and whose end does not;
* "reachable" was cut at the lane-merge tolerance, which classified the 1-2 cm
  representation artefacts of `C48` as unreachable surface.

So `_decompose` is replaced here by scripted cross-sections. That is the point:
these tests are about the walk, and a geometry fixture would only re-test the
decomposition while making the continuity rules hard to state.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

# The gate used throughout: symmetric about the route, 4 m wide, so an ego
# centred beyond +/-(2 + one ego width) misses it.
GATE_LO = -2.0
GATE_HI = +2.0
SPACING_M = 5.0
ROUTE_LENGTH_M = 20.0


def load_module() -> ModuleType:
    module_path = (
        Path(__file__).parents[1] / "scripts" / "measure_final_gate_carriageway_coverage.py"
    )
    spec = importlib.util.spec_from_file_location("carriageway_coverage", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def coverage() -> ModuleType:
    return load_module()


class _Projection:
    tangent_xy = (1.0, 0.0)


class _Route:
    """The straight route the scripted cross-sections are attached to."""

    length_m = ROUTE_LENGTH_M

    def point_at(self, s_m: float):
        return (float(s_m), 0.0, 0.0)

    def project(self, point_xy, *, position_z=None, previous_s_m=None):
        return _Projection()


def _component(lo: float, hi: float, lane_id: str = "L", kind: str = "LANE_SURFACE_STREET"):
    return [lo, hi, [lane_id], [kind]]


def _host():
    """The route's own carriageway: one lane either side of the centreline."""

    return _component(-2.0, +2.0, lane_id="route")


def _walk(coverage: ModuleType, monkeypatch, stations: dict[float, list]):
    """Run the walk against scripted cross-sections, keyed by station."""

    def fake_decompose(*, goal_xyz, **_):
        station = round(float(goal_xyz[0]), 6)
        return stations[station]

    monkeypatch.setattr(coverage, "_decompose", fake_decompose)
    return coverage._walk_backwards(
        route=_Route(),
        s_goal_m=ROUTE_LENGTH_M,
        spacing_m=SPACING_M,
        gate_lo=GATE_LO,
        gate_hi=GATE_HI,
        lanes={},
        lane_types={},
        route_lane_ids={"route"},
    )


def test_a_route_with_no_parallel_surface_reports_no_corridor(coverage, monkeypatch) -> None:
    """TEST-D14-01: the host component alone is not an exposure."""

    stations = {float(s): [_host()] for s in (20.0, 15.0, 10.0, 5.0, 0.0)}
    result = _walk(coverage, monkeypatch, stations)

    assert result["walk_stations"] == 1
    assert result["walk_unusable_stations"] == 0
    assert result.get("corridor_to_goal_m") is None


def test_a_corridor_is_followed_while_components_overlap(coverage, monkeypatch) -> None:
    """TEST-D14-02: the corridor's length is the arc actually walked."""

    corridor = _component(+6.0, +10.0)
    stations = {
        20.0: [_host(), corridor],
        15.0: [_host(), _component(+6.0, +10.0)],
        10.0: [_host(), _component(+5.5, +9.5)],
        # The surface ends here, so the corridor is 20 -> 10 = 10 m.
        5.0: [_host()],
        0.0: [_host()],
    }
    result = _walk(coverage, monkeypatch, stations)

    assert result["corridor_to_goal_m"] == pytest.approx(10.0)
    assert result["corridor_to_goal_offset_m"] == pytest.approx(8.0)
    assert result["corridor_to_goal_clears_gate"] is True
    assert result["corridor_to_goal_is_route_lane"] is False


def test_a_component_the_ego_does_not_fit_in_breaks_the_corridor(coverage, monkeypatch) -> None:
    """TEST-D14-03: a corridor narrower than the ego is not one it can drive."""

    narrow = coverage.EGO_WIDTH_M / 2.0
    stations = {
        20.0: [_host(), _component(+6.0, +10.0)],
        15.0: [_host(), _component(+6.0, 6.0 + narrow)],
        10.0: [_host(), _component(+6.0, +10.0)],
        5.0: [_host()],
        0.0: [_host()],
    }
    result = _walk(coverage, monkeypatch, stations)

    # It stops at the narrow station rather than stepping over it.
    assert result["corridor_to_goal_m"] == pytest.approx(0.0)


def test_a_component_that_does_not_overlap_breaks_the_corridor(coverage, monkeypatch) -> None:
    """TEST-D14-04: two strips at similar distances are not one carriageway.

    Overlap is what makes consecutive components the same physical surface.
    Without it a corridor could be assembled out of unrelated strips that merely
    happen to sit at comparable lateral offsets on successive cross-sections.
    """

    stations = {
        20.0: [_host(), _component(+6.0, +10.0)],
        15.0: [_host(), _component(+14.0, +18.0)],
        10.0: [_host(), _component(+6.0, +10.0)],
        5.0: [_host()],
        0.0: [_host()],
    }
    result = _walk(coverage, monkeypatch, stations)

    assert result["corridor_to_goal_m"] == pytest.approx(0.0)


def test_a_corridor_qualifies_on_containing_a_missing_position(coverage, monkeypatch) -> None:
    """TEST-D14-05: containment, not the centre. Regression for an under-report.

    The configuration needs a host narrower than the gate, which is the ordinary
    case — the 2026-09-08 run measured a median gate length of 10.26 m, about
    three lanes. Here the route's own lane is `[-1, +1]` inside a gate of
    `[-2, +2]`, and the corridor is `[+1.2, +4.5]`.

    Its centre sits at `+2.85`, from which an ego still sweeps the gate
    (`2.85 - 0.926 = 1.92 < 2.0`), so a centre test would discard it. But its far
    end at `+4.5` is more than an ego width past `gate_hi`, so a position inside
    it exists from which the ego misses the gate — and that is what makes it an
    exposure.
    """

    host = _component(-1.0, +1.0, lane_id="route")
    corridor = _component(+1.2, +4.5)
    assert (corridor[0] + corridor[1]) / 2.0 == pytest.approx(2.85)
    assert coverage._crosses(2.85, GATE_LO, GATE_HI) is True
    assert corridor[1] > GATE_HI + coverage.EGO_WIDTH_M

    stations = {
        20.0: [host, corridor],
        15.0: [host, _component(+1.2, +4.5)],
        10.0: [host],
        5.0: [host],
        0.0: [host],
    }
    result = _walk(coverage, monkeypatch, stations)

    assert result["corridor_to_goal_m"] == pytest.approx(5.0)
    assert result["corridor_to_goal_offset_m"] == pytest.approx(2.85)


def test_a_component_entirely_inside_the_gate_is_not_an_exposure(coverage, monkeypatch) -> None:
    """TEST-D14-06: the other side of the same rule.

    A component the ego cannot miss the gate from is not an exposure however long
    it runs, because the mission still succeeds from it.
    """

    inside = _component(-1.9, +1.9)
    stations = {float(s): [_host(), inside] for s in (20.0, 15.0, 10.0, 5.0, 0.0)}
    result = _walk(coverage, monkeypatch, stations)

    assert result.get("corridor_to_goal_m") is None


def test_every_candidate_is_walked_and_the_longest_wins(coverage, monkeypatch) -> None:
    """TEST-D14-07: regression for walking only the nearest corridor.

    The near corridor at `+6..+10` dies one station back; the far one at
    `+14..+18` runs to the end of the route. Taking the nearest would have
    reported 0 m and hidden a 20 m exposure.
    """

    stations = {
        20.0: [_host(), _component(+6.0, +10.0), _component(+14.0, +18.0)],
        15.0: [_host(), _component(+14.0, +18.0)],
        10.0: [_host(), _component(+14.0, +18.0)],
        5.0: [_host(), _component(+14.0, +18.0)],
        0.0: [_host(), _component(+14.0, +18.0)],
    }
    result = _walk(coverage, monkeypatch, stations)

    assert result["corridor_to_goal_m"] == pytest.approx(20.0)
    assert result["corridor_to_goal_offset_m"] == pytest.approx(16.0)


def test_the_entry_gap_is_read_where_the_corridor_ends(coverage, monkeypatch) -> None:
    """TEST-D14-08: the gap is where the ego would have to enter, not at the goal.

    It decides whether the exposure costs `offroad` or is free, so reading it at
    the goal — where the corridor is often further from the host than at its far
    end — would price the wrong crossing.
    """

    stations = {
        20.0: [_host(), _component(+9.0, +13.0)],
        # One station back the corridor sits directly against the host.
        15.0: [_host(), _component(+2.0, +9.5)],
        10.0: [_host()],
        5.0: [_host()],
        0.0: [_host()],
    }
    result = _walk(coverage, monkeypatch, stations)

    assert result["corridor_to_goal_m"] == pytest.approx(5.0)
    # Host is [-2, +2] and the corridor there starts at +2.0, so nothing to cross.
    assert result["corridor_entry_gap_m"] == pytest.approx(0.0)


def test_an_ambiguous_host_is_counted_rather_than_silently_skipped(coverage, monkeypatch) -> None:
    """TEST-D14-09: a cross-section with no unique host stops the walk loudly.

    A station where offset 0 falls in two components, or in none, is geometry the
    walk cannot interpret. Continuing past it would invent continuity; dropping it
    without a count would make an interrupted corridor look whole.
    """

    stations = {
        20.0: [_host(), _component(+6.0, +10.0)],
        15.0: [_host(), _component(+6.0, +10.0)],
        # Offset 0 is in no component here.
        10.0: [_component(+3.0, +5.0), _component(+6.0, +10.0)],
        5.0: [_host(), _component(+6.0, +10.0)],
        0.0: [_host(), _component(+6.0, +10.0)],
    }
    result = _walk(coverage, monkeypatch, stations)

    assert result["walk_unusable_stations"] == 1
    assert result["corridor_to_goal_m"] == pytest.approx(5.0)
