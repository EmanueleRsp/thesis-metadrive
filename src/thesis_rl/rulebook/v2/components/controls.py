"""Deterministic traffic-control catalog helpers for R3."""

from __future__ import annotations

from collections.abc import Mapping

from thesis_rl.rulebook.v2.geometry.conflict_zones import worst_case_temporal_gap_violation
from thesis_rl.rulebook.v2.geometry.continuous_sat import OccupancyInterval
from thesis_rl.rulebook.v2.types import (
    ComponentStatus,
    MemoryDelta,
    CacheDelta,
    RuleComponentResult,
    TrafficControlRecord,
    ApproachControl,
)
from thesis_rl.rulebook.v2.errors import (
    RuntimeScenarioNotEvaluableError,
    RuntimeScenarioNotEvaluableReason,
)

VALID_SIGNAL_STATES = frozenset({"GREEN", "YELLOW", "RED", "FLASHING_YELLOW", "UNKNOWN"})
STOP_ZONE_M = 1.0
STOP_SPEED_MPS = 0.1
STOP_DEADBAND_M = 0.05
STOP_MIN_DWELL_S = 1.0
CROSSWALK_GAP_S = 1.0


def select_active_signal_group(
    *,
    controls: tuple[TrafficControlRecord, ...],
    ego_front_s_m: float,
    resolved_group_ids: frozenset[str],
) -> TrafficControlRecord | None:
    """Select the first unresolved signal control ahead, strictly by route coordinate."""
    if not isinstance(ego_front_s_m, (int, float)):
        raise ValueError("ego_front_s_m must be numeric")
    candidates = [
        control
        for control in controls
        if control.control_type is ApproachControl.SIGNAL
        and control.control_group_id not in resolved_group_ids
        and control.route_s_m >= ego_front_s_m
    ]
    return min(
        candidates, key=lambda control: (control.route_s_m, control.control_group_id), default=None
    )


def signal_group_state(
    *, control: TrafficControlRecord, signal_states_by_physical_id: Mapping[str, str]
) -> str:
    """Return a concordant physical signal state; disagreement is fail-fast."""
    states = [
        signal_states_by_physical_id.get(identifier) for identifier in control.physical_control_ids
    ]
    if not states or any(state not in VALID_SIGNAL_STATES for state in states):
        raise ValueError(f"Missing or invalid state for signal group {control.control_group_id!r}")
    if len(set(states)) != 1:
        raise ValueError(
            f"Discordant physical states for signal group {control.control_group_id!r}"
        )
    state = states[0]
    if state is None:
        raise ValueError(f"Missing state for signal group {control.control_group_id!r}")
    return state


def evaluate_signal_state(
    *,
    control: TrafficControlRecord | None,
    ego_front_s_m: float,
    signal_states_by_physical_id: Mapping[str, str],
    resolved_group_ids: frozenset[str],
) -> tuple[RuleComponentResult, MemoryDelta, CacheDelta]:
    if control is None:
        result = RuleComponentResult(
            "signal", 0.0, {"state": None}, False, True, ComponentStatus.NOT_APPLICABLE, {}
        )
        return result, MemoryDelta(), CacheDelta()
    state = signal_group_state(
        control=control, signal_states_by_physical_id=signal_states_by_physical_id
    )
    result = RuleComponentResult(
        "signal",
        0.0,
        {"group_id": control.control_group_id, "state": state},
        True,
        True,
        ComponentStatus.SATISFIED,
        {"route_s_m": control.route_s_m, "ego_front_s_m": ego_front_s_m},
    )
    delta = MemoryDelta(
        writer="signal",
        writes=(
            ("active_signal_group_id", control.control_group_id),
            ("previous_signal_state", state),
        ),
    )
    return result, delta, CacheDelta()


def select_active_stop_group(
    *,
    controls: tuple[TrafficControlRecord, ...],
    ego_front_s_m: float,
    resolved_group_ids: frozenset[str],
    signal_group_ids: frozenset[str] = frozenset(),
) -> TrafficControlRecord | None:
    candidates = [
        control
        for control in controls
        if control.control_type is ApproachControl.STOP
        and control.control_group_id not in resolved_group_ids
        and control.control_group_id not in signal_group_ids
        and control.route_s_m >= ego_front_s_m
    ]
    return min(
        candidates, key=lambda control: (control.route_s_m, control.control_group_id), default=None
    )


def evaluate_stop(
    *,
    control: TrafficControlRecord | None,
    pre_delta_m: float | None,
    post_delta_m: float | None,
    speed_mps: float,
    previous_continuous_s: float,
    previous_best_s: float,
    delta_t_s: float,
    previous_group_id: str | None,
    resolved_group_ids: frozenset[str],
    crossing: bool,
) -> tuple[RuleComponentResult, MemoryDelta, CacheDelta]:
    if not all(
        isinstance(value, (int, float))
        for value in (speed_mps, previous_continuous_s, previous_best_s, delta_t_s)
    ):
        raise ValueError("Stop evaluator numeric inputs are required")
    if speed_mps < 0.0 or previous_continuous_s < 0.0 or previous_best_s < 0.0 or delta_t_s <= 0.0:
        raise ValueError("Stop evaluator inputs out of range")
    if control is None:
        result = RuleComponentResult(
            "stop", 0.0, {"crossing": False}, False, True, ComponentStatus.NOT_APPLICABLE, {}
        )
        return result, MemoryDelta(), CacheDelta()
    if pre_delta_m is None or post_delta_m is None:
        raise ValueError("Stop evaluator requires pre/post signed distances")
    # REQ-EF-09: the dwell state belongs to the control group it was accrued
    # at.  Carrying it across a group change let a second stop sign inherit a
    # standstill performed at the first one and be crossed at speed for zero
    # cost (v4.7 §7.7.5-6 scope both timers to the active group).
    same_group = previous_group_id == control.control_group_id
    carried_continuous_s = previous_continuous_s if same_group else 0.0
    carried_best_s = previous_best_s if same_group else 0.0
    in_zone = 0.0 <= post_delta_m <= STOP_ZONE_M
    continuous = (
        carried_continuous_s + delta_t_s if in_zone and speed_mps <= STOP_SPEED_MPS else 0.0
    )
    best = max(carried_best_s, continuous)
    illegal = crossing and best < STOP_MIN_DWELL_S
    cost = max(0.0, 1.0 - best / STOP_MIN_DWELL_S) if illegal else 0.0
    next_resolved = frozenset(
        set(resolved_group_ids) | ({control.control_group_id} if crossing else set())
    )
    result = RuleComponentResult(
        "stop",
        cost,
        {"continuous_timer_s": continuous, "best_timer_s": best, "crossing": crossing},
        True,
        True,
        ComponentStatus.VIOLATED if illegal else ComponentStatus.SATISFIED,
        {"group_id": control.control_group_id, "in_stop_zone": in_zone},
    )
    delta = MemoryDelta(
        writer="stop",
        writes=(
            ("active_stop_group_id", control.control_group_id),
            ("stop_continuous_timer_s", continuous),
            ("stop_best_timer_s", best),
            ("previous_stop_delta_m", post_delta_m),
            ("resolved_stop_group_ids", next_resolved),
        ),
    )
    return result, delta, CacheDelta()


def evaluate_signal_transition(
    *,
    control: TrafficControlRecord | None,
    pre_state: str | None,
    post_state: str | None,
    pre_delta_m: float | None,
    post_delta_m: float | None,
    speed_mps: float,
    route_tangent_xy: tuple[float, float],
    delta_t_s: float,
    previous_yellow_must_stop: bool,
    previous_signal_delta_m: float | None,
    previous_group_id: str | None,
    resolved_group_ids: frozenset[str],
    crossing: bool,
    ego_brake_mps2: float | None,
) -> tuple[RuleComponentResult, MemoryDelta, CacheDelta]:
    """Evaluate signal crossing with pre-action color and post-action approach cost."""
    if control is None:
        return (
            RuleComponentResult(
                "signal", 0.0, {"crossing": False}, False, True, ComponentStatus.NOT_APPLICABLE, {}
            ),
            MemoryDelta(),
            CacheDelta(),
        )
    if (
        pre_state not in VALID_SIGNAL_STATES
        or post_state not in VALID_SIGNAL_STATES
        or "UNKNOWN" in (pre_state, post_state)
    ):
        cause = ValueError("Signal transition requires valid, known pre/post states")
        raise RuntimeScenarioNotEvaluableError(
            RuntimeScenarioNotEvaluableReason.INVALID_SIGNAL_TRANSITION,
            str(cause),
            diagnostics={
                "active_signal_group_id": control.control_group_id,
                "physical_signal_ids": list(control.physical_control_ids),
                "raw_pre_state": pre_state,
                "normalized_pre_state": pre_state,
                "raw_post_state": post_state,
                "normalized_post_state": post_state,
                "mapping_status": "invalid_or_unknown_transition",
            },
        ) from cause
    if pre_delta_m is None or post_delta_m is None or speed_mps < 0.0 or delta_t_s <= 0.0:
        raise ValueError("Signal transition requires valid distances, speed and timestep")
    if ego_brake_mps2 is None or ego_brake_mps2 <= 0.0:
        raise ValueError("Signal transition requires calibrated positive ego braking")
    # REQ-EF-10: the frozen yellow commitment belongs to the signal group it
    # was decided at (v4.7 §7.6.3 freezes it "per la stessa fase").  Carrying
    # it into a different group made a newly encountered yellow inherit the
    # previous intersection's decision.
    same_group = previous_group_id == control.control_group_id
    carried_yellow_must_stop = previous_yellow_must_stop if same_group else False
    if pre_state == "FLASHING_YELLOW" or post_state == "FLASHING_YELLOW":
        must_stop = False
        cost = 0.0
        status = ComponentStatus.NOT_APPLICABLE
    else:
        d_req = speed_mps * delta_t_s + speed_mps * speed_mps / (2.0 * ego_brake_mps2)
        must_stop = carried_yellow_must_stop
        if post_state == "YELLOW" and pre_state != "YELLOW":
            # REQ-EF-14: v4.7 §7.6.3-4 evaluates delta_sig(t) and v_app(t) at
            # the same instant t, and Y_must_stop^+ is by definition the
            # current (post-action) value.  Pairing the pre-state distance
            # with the post-state approach speed was temporally incoherent and
            # misclassified cases near the threshold.
            must_stop = post_delta_m >= d_req
        cross_must_stop = pre_state == "RED" or (pre_state == "YELLOW" and carried_yellow_must_stop)
        approach_must_stop = post_state == "RED" or (post_state == "YELLOW" and must_stop)
        if crossing and cross_must_stop:
            cost = 1.0
        elif not crossing and approach_must_stop and post_delta_m > 0.0 and d_req > 0.0:
            cost = min(max(1.0 - post_delta_m / d_req, 0.0), 1.0)
        else:
            cost = 0.0
        status = ComponentStatus.VIOLATED if cost > 0.0 else ComponentStatus.SATISFIED
    next_resolved = frozenset(
        set(resolved_group_ids) | ({control.control_group_id} if crossing else set())
    )
    result = RuleComponentResult(
        "signal",
        cost,
        {
            "pre_state": pre_state,
            "post_state": post_state,
            "crossing": crossing,
            "yellow_must_stop": must_stop,
        },
        True,
        True,
        status,
        {
            "group_id": control.control_group_id,
            "pre_delta_m": pre_delta_m,
            "post_delta_m": post_delta_m,
        },
    )
    delta = MemoryDelta(
        writer="signal",
        writes=(
            ("active_signal_group_id", control.control_group_id),
            ("previous_signal_state", post_state),
            ("yellow_must_stop", must_stop),
            ("previous_signal_delta_m", post_delta_m),
            ("resolved_signal_group_ids", next_resolved),
        ),
    )
    return result, delta, CacheDelta()


def evaluate_crosswalk_yield(
    *,
    zone_id: str,
    ego_interval: OccupancyInterval,
    vru_intervals: tuple[tuple[str, OccupancyInterval], ...],
    distance_to_entry_m: float,
    approach_speed_mps: float,
    delta_t_s: float,
    ego_occupied: bool,
    ego_entered: bool,
    preexisting_zone_ids: frozenset[str],
    previous_illegal_entries: frozenset[tuple[str, str]],
    vertical_applicable: bool = True,
    ego_brake_mps2: float | None = None,
) -> tuple[RuleComponentResult, MemoryDelta, CacheDelta]:
    """Evaluate crosswalk temporal gap and commit gate for live VRU intervals."""
    if not vertical_applicable:
        # The illegal-entry latch must still be persisted here (already
        # pruned upstream against live zone geometry, independent of this
        # step's selection): an empty MemoryDelta would silently keep any
        # stale entry from a previous step forever, since no other branch
        # revisits a zone once it stops being selected (same bug class as
        # ADR-025 for vehicle_yield).
        return (
            RuleComponentResult(
                "crosswalk",
                0.0,
                {"zone_id": zone_id},
                False,
                True,
                ComponentStatus.NOT_APPLICABLE,
                {},
            ),
            MemoryDelta(
                writer="crosswalk",
                writes=(("crosswalk_illegal_entries", previous_illegal_entries),),
            ),
            CacheDelta(),
        )
    if ego_interval is None:
        raise ValueError("Crosswalk requires an evaluable ego occupancy interval")
    if delta_t_s <= 0.0 or approach_speed_mps < 0.0:
        raise ValueError("Crosswalk kinematic inputs out of range")
    if ego_brake_mps2 is None or ego_brake_mps2 <= 0.0:
        raise ValueError("Crosswalk requires calibrated positive ego braking")
    ego_start, ego_end = ego_interval.start_s, ego_interval.end_s
    worst = 0.0
    candidate_gaps: dict[str, float | None] = {}
    for actor_id, interval in vru_intervals:
        start, end = interval.start_s, interval.end_s
        if ego_end is not None and start >= ego_end:
            gap = start - ego_end
        elif end is not None and end <= ego_start:
            gap = ego_start - end
        else:
            gap = None
        candidate_gaps[actor_id] = gap
        risk = 1.0 if gap is None else min(max((CROSSWALK_GAP_S - gap) / CROSSWALK_GAP_S, 0.0), 1.0)
        worst = max(worst, risk)
    d_stop = approach_speed_mps * delta_t_s + approach_speed_mps * approach_speed_mps / (
        2.0 * ego_brake_mps2
    )
    commit = min(max(1.0 - distance_to_entry_m / d_stop, 0.0), 1.0) if d_stop > 0.0 else 0.0
    before = distance_to_entry_m > 0.05 and not ego_occupied and zone_id not in preexisting_zone_ids
    approach_cost = worst * commit if before else 0.0
    illegal = ego_entered and worst > 0.0
    active = set(previous_illegal_entries)
    if illegal:
        active.update((actor_id, zone_id) for actor_id, _ in vru_intervals)
    elif not ego_occupied:
        active = {entry for entry in active if entry[1] != zone_id}
    # ADR-064: the persistence latch is removed from the cost. It pinned the cost
    # at 1.0 for as long as the ego occupied the zone, which is both a
    # credit-assignment defect -- inside the zone no action changes the cost, so
    # the agent is charged repeatedly for a decision already taken, with no
    # gradient toward better behaviour -- and a Test B failure, because the latch
    # is a memory the agent cannot see and is unbounded in duration, so it cannot
    # be made a plausible perception feature either. The memoryless approach
    # term, which prices that same decision while it is still being made, is
    # retained. The illegal-entry set is still tracked and reported, following
    # the convention of `rss` and the not-at-fault collisions: measured, never
    # priced. REQ-EF-08's applicability coupling goes with it -- with no latched
    # cost there is nothing for `aggregate_max_component` to discard.
    active_latch_for_zone = bool(ego_occupied and any(entry[1] == zone_id for entry in active))
    cost = approach_cost
    applicable = bool(vru_intervals)
    result = RuleComponentResult(
        "crosswalk",
        cost,
        {"zone_id": zone_id, "candidate_gaps_s": candidate_gaps, "commit": commit},
        applicable,
        True,
        ComponentStatus.VIOLATED
        if cost > 0.0
        else (ComponentStatus.SATISFIED if applicable else ComponentStatus.NOT_APPLICABLE),
        {"before_gate": before, "latched_illegal_entry": active_latch_for_zone},
    )
    delta = MemoryDelta(
        writer="crosswalk", writes=(("crosswalk_illegal_entries", frozenset(active)),)
    )
    return result, delta, CacheDelta()


def evaluate_vehicle_yield(
    *,
    zone_id: str,
    ego_interval: OccupancyInterval,
    prioritized_intervals: tuple[tuple[str, OccupancyInterval], ...],
    distance_to_entry_m: float,
    approach_speed_mps: float,
    delta_t_s: float,
    ego_occupied: bool,
    entered_actor_ids: frozenset[str],
    previous_illegal_entries: frozenset[tuple[str, str]],
    preexisting: bool = False,
    pre_state_entered_actor_ids: frozenset[str] | None = None,
    pre_state_gap_violation: float | None = None,
    actor_movement_keys: tuple[tuple[str, object], ...] = (),
    previous_frozen_movement_keys: tuple[tuple[str, object], ...] = (),
    exited_actor_ids: frozenset[str] = frozenset(),
    ego_brake_mps2: float | None = None,
) -> tuple[RuleComponentResult, MemoryDelta, CacheDelta]:
    """Evaluate the scoped vehicle-yield predicates.

    ``entered_actor_ids`` is the set computed from the pre-state occupancy
    view (DEC-005).  The explicit alias is accepted for adapters that expose
    the temporal origin in their field name; supplying both must agree.
    Approach intervals may be computed from the post-state, but an actor that
    leaves during the same control step must still be present in the pre-state
    entry set to create an illegal-entry flag.

    ``pre_state_gap_violation``, when supplied, gates illegal-entry latch
    creation (DEC-005 Fase A/B: the entry-legality judgment uses the
    pre-state gap) independently of ``prioritized_intervals``, which remains
    the post-state view used for the continuous approach cost (Fase C).
    Callers that only evaluate a single snapshot (e.g. unit tests exercising
    the historical post-state-only formula) may omit it, in which case the
    post-state ``worst`` gap is reused for the latch gate as before.
    """
    if ego_interval is None:
        raise ValueError("Vehicle-yield requires an evaluable ego interval")
    if delta_t_s <= 0.0 or approach_speed_mps < 0.0:
        raise ValueError("Vehicle-yield kinematic inputs out of range")
    if pre_state_entered_actor_ids is not None:
        if entered_actor_ids and entered_actor_ids != pre_state_entered_actor_ids:
            raise ValueError("DEC-005 entered actor sets disagree")
        entered_actor_ids = pre_state_entered_actor_ids
    if prioritized_intervals and (ego_brake_mps2 is None or ego_brake_mps2 <= 0.0):
        raise ValueError("Vehicle-yield requires calibrated positive ego braking")

    worst = worst_case_temporal_gap_violation(
        ego_interval=ego_interval,
        other_intervals=prioritized_intervals,
        gap_scale_s=CROSSWALK_GAP_S,
    )
    # DEC-005 Fase B: the illegal-entry latch gates on the pre-state gap when
    # supplied; the post-state ``worst`` remains the fallback for callers
    # that never separated the two snapshots.
    gap_gate = worst if pre_state_gap_violation is None else pre_state_gap_violation

    illegal_keys = set(previous_illegal_entries)
    for actor_id in entered_actor_ids:
        if gap_gate > 0.0:
            illegal_keys.add((actor_id, zone_id))
    if not ego_occupied:
        illegal_keys = {key for key in illegal_keys if key[1] != zone_id}

    # MovementKey is frozen at entry and released only after complete exit.
    # Preserve insertion-independent ordering for deterministic memory deltas.
    frozen = dict(previous_frozen_movement_keys)
    for actor_id, movement_key in actor_movement_keys:
        if actor_id in entered_actor_ids and actor_id not in frozen:
            frozen[actor_id] = movement_key
    for actor_id in exited_actor_ids:
        frozen.pop(actor_id, None)

    if prioritized_intervals:
        d_stop = approach_speed_mps * delta_t_s + approach_speed_mps * approach_speed_mps / (
            2.0 * ego_brake_mps2
        )
        commit = min(max(1.0 - distance_to_entry_m / d_stop, 0.0), 1.0) if d_stop > 0.0 else 0.0
        before = distance_to_entry_m > 0.05 and not ego_occupied and not preexisting
        approach = worst * commit if before else 0.0
    else:
        commit, before, approach = 0.0, False, 0.0

    # ADR-064 removes the latch from the cost, for the reasons given in
    # `evaluate_crosswalk_yield`. This is where 353 of the 364 latched expert
    # steps were charged -- 86.9 % of all traffic-control cost -- so the change
    # is measurable here even though it is justified on observability and credit
    # assignment rather than on magnitude. DEC-005 Fase D's dominance rule
    # (REQ-VY-04) is superseded: the latch is reported, not priced.
    active_latch_for_zone = ego_occupied and any(key[1] == zone_id for key in illegal_keys)
    cost = approach
    applicable = bool(prioritized_intervals)
    result = RuleComponentResult(
        "vehicle_yield",
        cost,
        {
            "zone_id": zone_id,
            "prioritized_actor_count": len(prioritized_intervals),
            "commit": commit,
        },
        applicable,
        True,
        ComponentStatus.VIOLATED
        if cost > 0.0
        else (ComponentStatus.SATISFIED if applicable else ComponentStatus.NOT_APPLICABLE),
        {"before_gate": before, "latched_illegal_entry": active_latch_for_zone},
    )
    delta = MemoryDelta(
        writer="vehicle_yield",
        writes=(
            ("vehicle_yield_illegal_entries", frozenset(illegal_keys)),
            ("frozen_actor_movement_keys", tuple(sorted(frozen.items(), key=lambda item: item[0]))),
        ),
    )
    return result, delta, CacheDelta()
