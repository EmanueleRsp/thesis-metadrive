"""Per-episode route-adherence and clip diagnostics, reported and never priced.

`RULEBOOK-V5.1` §7 requires two counters that describe a run rather than score
it, and `REQ-EF-15` adds a third quantity that was computed on every step and
read by nothing. This module is the reporting end of all three: it names the CSV
columns and turns one episode's metadata into those columns.

**Why the corridor diagnostic needed a home.** `R4` credits arc-length advance of
a nearest-point projection onto the assigned route, and mission success is a
crossing of one frozen finite gate; nothing links them. An ego on a legal,
same-direction carriageway beside its route therefore keeps earning progress
while the gate stays uncrossed, and no cost or termination objects — the
out-of-road surface is the union of *every* map lane, and `wrong_carriageway`
classifies a same-direction road as aligned. `route_outside_fraction` is the only
quantity in the pipeline that measures this, and open item `D14` records that it
reached no report. Aggregating it is what turns "the reward could be paid off
corridor" from an argument into a measurement.

**Why the longest run and not the mean.** A mean cannot separate clipping the
inside of four corners from driving eighty consecutive steps on the wrong road,
and only the second is the failure. `route_fully_outside_max_run` is the
statistic that distinguishes them, so it is the one to read first.

**Why the evaluated-step count is a column and not an implementation detail.**
The progress evaluator omits the diagnostic when it has no ego footprint or no
assigned corridor. Without `route_outside_evaluated_steps` a zero would be
ambiguous between "never left the corridor" and "never measured", and a check
that inspected nothing must not be recorded as a check that passed.
"""

from __future__ import annotations

from typing import Any, Mapping


ROUTE_ADHERENCE_EPISODE_COLUMNS: tuple[str, ...] = (
    # `REQ-EF-15` / `D14`: does the policy drive off its assigned corridor?
    "route_outside_evaluated_steps",
    "route_outside_steps",
    "route_fully_outside_steps",
    "route_fully_outside_max_run",
    "mean_route_outside_fraction",
    "mean_route_adherence",
    # RULEBOOK-V5.1 §7. `l4_clip_binding_steps` is what `AC-RB5.1-05` needs to
    # stop being `NOT ESTABLISHED`: the engine-force cap bounds the ego's
    # travel, not its projection, so whether the clip binds on an agent
    # trajectory is an open measurement rather than a settled deduction.
    "l4_clip_binding_steps",
    "l5_reached_steps",
    "mean_ego_speed_mps",
)


def route_adherence_episode_fields(scenario_metadata: Any) -> dict[str, Any]:
    """Return one episode's columns, keyed by `ROUTE_ADHERENCE_EPISODE_COLUMNS`.

    An absent key yields ``None`` rather than an error, so a run recorded before
    these counters existed, or one whose environment stack has no Rulebook
    wrapper, still writes a well-formed row.
    """

    metadata: Mapping[str, Any] = scenario_metadata if isinstance(scenario_metadata, Mapping) else {}
    return {name: metadata.get(name) for name in ROUTE_ADHERENCE_EPISODE_COLUMNS}
