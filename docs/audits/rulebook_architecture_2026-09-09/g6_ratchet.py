"""G6 - the negative-clip ratchet on the mission-progress channel.

`Delta q = clip(Delta s / D_REF, -1, +1)` is symmetric, but the magnitudes it
bounds are not. Forward, vehicle physics already caps `Delta s` at
`v_max * dt = 2.2222 m`, so the positive clip protects against nothing real.
Backward, a route-projection branch jump is unbounded and the clip charges it
`-1` instead of its true magnitude. A trajectory that walks out along one leg of
a hairpin and returns along the other therefore banks the outward leg honestly
and pays only `-1` for the return.

This is the one script in this directory that imports the repository. Run it
from the repository root:

    PYTHONPATH=src uv run --no-sync python \
        docs/audits/rulebook_architecture_2026-09-09/g6_ratchet.py

What blocks this today is the geometry of the frozen panels, not the reward: the
2026-09-05 read-only audit over all 3,500 frozen missions found zero routes with
two portions of `|Delta s| > 15 m` closer than 6 m in plane
(`docs/implementation/route_coordinate_driving_mission_v1.1_exec_plan.md:262-268`).
That is a property of the population, its standing constraint requires a re-audit
on any regenerated index, the re-audit script does not exist, and its 15 m
threshold does not cover revisits between 2.2222 m and 15 m -- which is all the
ratchet needs. See `docs/open_items.md`.
"""

from __future__ import annotations

from thesis_rl.rulebook.v2.geometry.route import RoutePolyline

D_REF_M = 2.2222  # v_ref * dt, RULEBOOK-V5.1 section 4.1


def hairpin() -> RoutePolyline:
    """Two straight legs one lane apart, joined by a hairpin. 84.03 m."""
    points = [(float(x), 0.0, 0.0) for x in range(0, 41, 2)]
    points.append((41.0, 1.75, 0.0))
    points.extend((float(x), 3.5, 0.0) for x in range(40, -1, -2))
    return RoutePolyline(points_xyz=tuple(points))


def walk(route: RoutePolyline, poses: list[tuple[float, float]], label: str) -> float:
    """Drive `poses` through the route, projecting sequentially as the tracker does."""
    previous_s_m = 0.0
    paid = 0.0
    first_s_m: float | None = None
    worst_delta_s_m = 0.0
    for position_xy in poses:
        s_m = route.project(position_xy, position_z=0.0, previous_s_m=previous_s_m).s_m
        if first_s_m is None:
            first_s_m = s_m
        delta_s_m = s_m - previous_s_m
        worst_delta_s_m = min(worst_delta_s_m, delta_s_m)
        paid += min(max(delta_s_m / D_REF_M, -1.0), 1.0)
        previous_s_m = s_m
    telescoped = (previous_s_m - (first_s_m or 0.0)) / D_REF_M
    print(
        f"{label:<38} paid={paid:8.2f}  telescoped={telescoped:8.2f}  "
        f"worst delta_s={worst_delta_s_m:8.2f} m"
    )
    return paid


def main() -> None:
    route = hairpin()
    length_m = route._segment_starts_m[-1] + route._segment_lengths_m[-1]
    print(f"hairpin route: {length_m:.2f} m over {len(route.points_xyz)} points")
    print(f"one clip width = D_REF = {D_REF_M} m\n")

    outward = [(float(x), 0.0) for x in range(0, 41)]
    ret = [(float(x), 3.5) for x in range(40, -1, -1)]
    walk(route, outward + [(41.0, 1.75)] + ret, "honest: the whole route once")

    # The exploit: cross one lane at the far end instead of taking the hairpin,
    # drive back to the start, and repeat. Net displacement per lap is zero.
    lap = outward + [(40.0, 3.5)] + [(float(x), 3.5) for x in range(39, -1, -1)] + [(0.0, 0.0)]
    for laps in (1, 2, 3):
        walk(route, lap * laps, f"closed loop x{laps} (net displacement 0)")

    print(
        "\nThe return jump is charged -1 instead of its true magnitude, so each lap\n"
        "banks the outward leg for free. The gain is linear in the number of laps and\n"
        "bounded by nothing in the reward. At lambda4 = 2.0 each lap is worth twice the\n"
        "channel figure in reward units."
    )


if __name__ == "__main__":
    main()
