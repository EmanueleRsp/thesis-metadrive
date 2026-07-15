"""Deterministic traffic-control catalog helpers for R3."""

from __future__ import annotations

from thesis_rl.rulebook.v2.types import ComponentStatus, MemoryDelta, CacheDelta, RuleComponentResult, TrafficControlRecord, ApproachControl

VALID_SIGNAL_STATES = frozenset({"GREEN", "YELLOW", "RED", "FLASHING_YELLOW", "UNKNOWN"})
STOP_ZONE_M = 1.0
STOP_SPEED_MPS = 0.1
STOP_DEADBAND_M = 0.05
STOP_MIN_DWELL_S = 1.0
SIGNAL_BRAKE_MPS2 = 8.0
CROSSWALK_GAP_S = 1.0


def select_active_signal_group(*, controls: tuple[TrafficControlRecord, ...], ego_front_s_m: float,
                               resolved_group_ids: frozenset[str]) -> TrafficControlRecord | None:
    """Select the first unresolved signal control ahead, strictly by route coordinate."""
    if not isinstance(ego_front_s_m, (int, float)):
        raise ValueError("ego_front_s_m must be numeric")
    candidates = [control for control in controls
                  if control.control_type is ApproachControl.SIGNAL
                  and control.control_group_id not in resolved_group_ids
                  and control.route_s_m >= ego_front_s_m]
    return min(candidates, key=lambda control: (control.route_s_m, control.control_group_id), default=None)


def signal_group_state(*, control: TrafficControlRecord, signal_states_by_physical_id: dict[str, str] | object) -> str:
    """Return a concordant physical signal state; disagreement is fail-fast."""
    states = [signal_states_by_physical_id.get(identifier) for identifier in control.physical_control_ids]
    if not states or any(state not in VALID_SIGNAL_STATES for state in states):
        raise ValueError(f"Missing or invalid state for signal group {control.control_group_id!r}")
    if len(set(states)) != 1:
        raise ValueError(f"Discordant physical states for signal group {control.control_group_id!r}")
    return states[0]


def evaluate_signal_state(*, control: TrafficControlRecord | None, ego_front_s_m: float,
                          signal_states_by_physical_id: dict[str, str] | object,
                          resolved_group_ids: frozenset[str]) -> tuple[RuleComponentResult, MemoryDelta, CacheDelta]:
    if control is None:
        result = RuleComponentResult("signal", 0.0, {"state": None}, False, True, ComponentStatus.NOT_APPLICABLE, {})
        return result, MemoryDelta(), CacheDelta()
    state = signal_group_state(control=control, signal_states_by_physical_id=signal_states_by_physical_id)
    result = RuleComponentResult("signal", 0.0, {"group_id": control.control_group_id, "state": state}, True, True, ComponentStatus.SATISFIED, {"route_s_m": control.route_s_m, "ego_front_s_m": ego_front_s_m})
    delta = MemoryDelta(writer="signal", writes=(("active_signal_group_id", control.control_group_id), ("previous_signal_state", state)))
    return result, delta, CacheDelta()


def select_active_stop_group(*, controls: tuple[TrafficControlRecord, ...], ego_front_s_m: float,
                            resolved_group_ids: frozenset[str], signal_group_ids: frozenset[str] = frozenset()) -> TrafficControlRecord | None:
    candidates = [control for control in controls if control.control_type is ApproachControl.STOP
                  and control.control_group_id not in resolved_group_ids
                  and control.control_group_id not in signal_group_ids
                  and control.route_s_m >= ego_front_s_m]
    return min(candidates, key=lambda control: (control.route_s_m, control.control_group_id), default=None)


def evaluate_stop(*, control: TrafficControlRecord | None, pre_delta_m: float | None, post_delta_m: float | None,
                  speed_mps: float, previous_continuous_s: float, previous_best_s: float,
                  delta_t_s: float, previous_group_id: str | None, resolved_group_ids: frozenset[str],
                  crossing: bool) -> tuple[RuleComponentResult, MemoryDelta, CacheDelta]:
    if not all(isinstance(value, (int, float)) for value in (speed_mps, previous_continuous_s, previous_best_s, delta_t_s)):
        raise ValueError("Stop evaluator numeric inputs are required")
    if speed_mps < 0.0 or previous_continuous_s < 0.0 or previous_best_s < 0.0 or delta_t_s <= 0.0:
        raise ValueError("Stop evaluator inputs out of range")
    if control is None:
        result = RuleComponentResult("stop", 0.0, {"crossing": False}, False, True, ComponentStatus.NOT_APPLICABLE, {})
        return result, MemoryDelta(), CacheDelta()
    if pre_delta_m is None or post_delta_m is None:
        raise ValueError("Stop evaluator requires pre/post signed distances")
    in_zone = 0.0 <= post_delta_m <= STOP_ZONE_M
    continuous = previous_continuous_s + delta_t_s if in_zone and speed_mps <= STOP_SPEED_MPS else 0.0
    best = max(previous_best_s, continuous)
    illegal = crossing and best < STOP_MIN_DWELL_S
    cost = max(0.0, 1.0 - best / STOP_MIN_DWELL_S) if illegal else 0.0
    next_resolved = frozenset(set(resolved_group_ids) | ({control.control_group_id} if crossing else set()))
    result = RuleComponentResult("stop", cost, {"continuous_timer_s": continuous, "best_timer_s": best, "crossing": crossing}, True, True, ComponentStatus.VIOLATED if illegal else ComponentStatus.SATISFIED, {"group_id": control.control_group_id, "in_stop_zone": in_zone})
    delta = MemoryDelta(writer="stop", writes=(("active_stop_group_id", control.control_group_id), ("stop_continuous_timer_s", continuous), ("stop_best_timer_s", best), ("previous_stop_delta_m", post_delta_m), ("resolved_stop_group_ids", next_resolved)))
    return result, delta, CacheDelta()


def evaluate_signal_transition(*, control: TrafficControlRecord | None, pre_state: str | None, post_state: str | None,
                               pre_delta_m: float | None, post_delta_m: float | None, speed_mps: float,
                               route_tangent_xy: tuple[float, float], delta_t_s: float, previous_yellow_must_stop: bool,
                               previous_signal_delta_m: float | None, previous_group_id: str | None,
                               resolved_group_ids: frozenset[str], crossing: bool) -> tuple[RuleComponentResult, MemoryDelta, CacheDelta]:
    """Evaluate signal crossing with pre-action color and post-action approach cost."""
    if control is None:
        return RuleComponentResult("signal", 0.0, {"crossing": False}, False, True, ComponentStatus.NOT_APPLICABLE, {}), MemoryDelta(), CacheDelta()
    if pre_state not in VALID_SIGNAL_STATES or post_state not in VALID_SIGNAL_STATES or "UNKNOWN" in (pre_state, post_state):
        raise ValueError("Signal transition requires valid, known pre/post states")
    if pre_delta_m is None or post_delta_m is None or speed_mps < 0.0 or delta_t_s <= 0.0:
        raise ValueError("Signal transition requires valid distances, speed and timestep")
    if pre_state == "FLASHING_YELLOW" or post_state == "FLASHING_YELLOW":
        must_stop = False
        cost = 0.0
        status = ComponentStatus.NOT_APPLICABLE
    else:
        d_req = speed_mps * delta_t_s + speed_mps * speed_mps / (2.0 * SIGNAL_BRAKE_MPS2)
        must_stop = previous_yellow_must_stop
        if post_state == "YELLOW" and pre_state != "YELLOW":
            must_stop = pre_delta_m >= d_req
        cross_must_stop = pre_state == "RED" or (pre_state == "YELLOW" and previous_yellow_must_stop)
        approach_must_stop = post_state == "RED" or (post_state == "YELLOW" and must_stop)
        if crossing and cross_must_stop:
            cost = 1.0
        elif not crossing and approach_must_stop and post_delta_m > 0.0 and d_req > 0.0:
            cost = min(max(1.0 - post_delta_m / d_req, 0.0), 1.0)
        else:
            cost = 0.0
        status = ComponentStatus.VIOLATED if cost > 0.0 else ComponentStatus.SATISFIED
    next_resolved = frozenset(set(resolved_group_ids) | ({control.control_group_id} if crossing else set()))
    result = RuleComponentResult("signal", cost, {"pre_state": pre_state, "post_state": post_state, "crossing": crossing, "yellow_must_stop": must_stop}, True, True, status, {"group_id": control.control_group_id, "pre_delta_m": pre_delta_m, "post_delta_m": post_delta_m})
    delta = MemoryDelta(writer="signal", writes=(("active_signal_group_id", control.control_group_id), ("previous_signal_state", post_state), ("yellow_must_stop", must_stop), ("previous_signal_delta_m", post_delta_m), ("resolved_signal_group_ids", next_resolved)))
    return result, delta, CacheDelta()


def evaluate_crosswalk_yield(*, zone_id: str, ego_interval, vru_intervals: tuple[tuple[str, object], ...],
                             distance_to_entry_m: float, approach_speed_mps: float, delta_t_s: float,
                             ego_occupied: bool, ego_entered: bool, preexisting_zone_ids: frozenset[str],
                             previous_illegal_entries: frozenset[tuple[str, str]], vertical_applicable: bool = True) -> tuple[RuleComponentResult, MemoryDelta, CacheDelta]:
    """Evaluate crosswalk temporal gap and commit gate for live VRU intervals."""
    if not vertical_applicable:
        return RuleComponentResult("crosswalk", 0.0, {"zone_id": zone_id}, False, True, ComponentStatus.NOT_APPLICABLE, {}), MemoryDelta(), CacheDelta()
    if ego_interval is None:
        raise ValueError("Crosswalk requires an evaluable ego occupancy interval")
    if delta_t_s <= 0.0 or approach_speed_mps < 0.0:
        raise ValueError("Crosswalk kinematic inputs out of range")
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
    d_stop = approach_speed_mps * delta_t_s + approach_speed_mps * approach_speed_mps / (2.0 * SIGNAL_BRAKE_MPS2)
    commit = min(max(1.0 - distance_to_entry_m / d_stop, 0.0), 1.0) if d_stop > 0.0 else 0.0
    before = distance_to_entry_m > 0.05 and not ego_occupied and zone_id not in preexisting_zone_ids
    approach_cost = worst * commit if before else 0.0
    illegal = ego_entered and worst > 0.0
    active = set(previous_illegal_entries)
    if illegal:
        active.update((actor_id, zone_id) for actor_id, _ in vru_intervals)
    elif not ego_occupied:
        active = {entry for entry in active if entry[1] != zone_id}
    cost = 1.0 if ego_occupied and active else approach_cost
    result = RuleComponentResult("crosswalk", cost, {"zone_id": zone_id, "candidate_gaps_s": candidate_gaps, "commit": commit}, bool(vru_intervals), True, ComponentStatus.VIOLATED if cost > 0.0 else (ComponentStatus.SATISFIED if vru_intervals else ComponentStatus.NOT_APPLICABLE), {"before_gate": before})
    delta = MemoryDelta(writer="crosswalk", writes=(("crosswalk_illegal_entries", frozenset(active)),))
    return result, delta, CacheDelta()


def evaluate_vehicle_yield(*, zone_id: str, ego_interval, prioritized_intervals: tuple[tuple[str, object], ...],
                           distance_to_entry_m: float, approach_speed_mps: float, delta_t_s: float,
                           ego_occupied: bool, entered_actor_ids: frozenset[str], previous_illegal_entries: frozenset[tuple[str, str]],
                           preexisting: bool = False,
                           pre_state_entered_actor_ids: frozenset[str] | None = None) -> tuple[RuleComponentResult, MemoryDelta, CacheDelta]:
    """Evaluate the scoped vehicle-yield predicates.

    ``entered_actor_ids`` is the set computed from the pre-state occupancy
    view (DEC-005).  The explicit alias is accepted for adapters that expose
    the temporal origin in their field name; supplying both must agree.
    Approach intervals may be computed from the post-state, but an actor that
    leaves during the same control step must still be present in the pre-state
    entry set to create an illegal-entry flag.
    """
    if ego_interval is None:
        raise ValueError("Vehicle-yield requires an evaluable ego interval")
    if delta_t_s <= 0.0 or approach_speed_mps < 0.0:
        raise ValueError("Vehicle-yield kinematic inputs out of range")
    if pre_state_entered_actor_ids is not None:
        if entered_actor_ids and entered_actor_ids != pre_state_entered_actor_ids:
            raise ValueError("DEC-005 entered actor sets disagree")
        entered_actor_ids = pre_state_entered_actor_ids
    worst = 0.0
    for _, interval in prioritized_intervals:
        if ego_interval.end_s is not None and interval.start_s >= ego_interval.end_s:
            gap = interval.start_s - ego_interval.end_s
        elif interval.end_s is not None and interval.end_s <= ego_interval.start_s:
            gap = ego_interval.start_s - interval.end_s
        else:
            gap = None
        worst = max(worst, 1.0 if gap is None else min(max((CROSSWALK_GAP_S - gap) / CROSSWALK_GAP_S, 0.0), 1.0))
    d_stop = approach_speed_mps * delta_t_s + approach_speed_mps * approach_speed_mps / (2.0 * SIGNAL_BRAKE_MPS2)
    commit = min(max(1.0 - distance_to_entry_m / d_stop, 0.0), 1.0) if d_stop > 0.0 else 0.0
    before = distance_to_entry_m > 0.05 and not ego_occupied and not preexisting
    approach = worst * commit if before else 0.0
    illegal_keys = set(previous_illegal_entries)
    for actor_id in entered_actor_ids:
        if worst > 0.0:
            illegal_keys.add((actor_id, zone_id))
    if not ego_occupied:
        illegal_keys = {key for key in illegal_keys if key[1] != zone_id}
    cost = 1.0 if ego_occupied and any(key[1] == zone_id for key in illegal_keys) else approach
    result = RuleComponentResult("vehicle_yield", cost, {"zone_id": zone_id, "prioritized_actor_count": len(prioritized_intervals), "commit": commit}, bool(prioritized_intervals), True, ComponentStatus.VIOLATED if cost > 0.0 else (ComponentStatus.SATISFIED if prioritized_intervals else ComponentStatus.NOT_APPLICABLE), {"before_gate": before})
    delta = MemoryDelta(writer="vehicle_yield", writes=(("vehicle_yield_illegal_entries", frozenset(illegal_keys)),))
    return result, delta, CacheDelta()
