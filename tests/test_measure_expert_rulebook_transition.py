"""Guard the three places the transition replay can silently diverge from runtime.

`scripts/measure_expert_rulebook_transition.py` drives production's own
``evaluate_transition``, so almost nothing can drift. Three things can:

* the offline signal-state colour map, duplicated from the live MetaDrive reader;
* the component-name list, which decides which costs the report even looks at;
* the causality of the signal read, which must never consult a future step.

Each is cheap to break and expensive to notice: a silently dropped component
reads as a satisfied rule, which is exactly the reading that would misinform the
rulebook decisions this measurement exists to settle.
"""

from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path
from types import ModuleType

import pytest

from thesis_rl.reward.scalarization import ScalarizationConfig, scalarize_rulebook_margins
from thesis_rl.rulebook.v2.components.rss import RESPONSE_TIME_S, safe_distance_m
from thesis_rl.rulebook.v2.context.metadrive_live import _LIVE_SIGNAL_STATE_MAP
from thesis_rl.rulebook.v2.registry import DEFAULT_RULEBOOK_V2_REGISTRY


# Registry names and result names coincide now that `wrong_way` -- whose
# evaluator named its own result `wrongway` -- has been deregistered by ADR-066.
# The map is kept rather than deleted: it is the seam where a component's
# registry key and its result key can diverge, and that divergence reads as a
# satisfied rule rather than as an error.
_RESULT_NAME_BY_REGISTRY_NAME: dict[str, str] = {}


def load_measurement_module() -> ModuleType:
    module_path = Path(__file__).parents[1] / "scripts" / "measure_expert_rulebook_transition.py"
    spec = importlib.util.spec_from_file_location("measure_expert_rulebook_transition", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # `@dataclass` resolves `sys.modules[cls.__module__]` while processing the
    # class body, so a module executed outside `sys.modules` raises
    # `AttributeError: 'NoneType' object has no attribute '__dict__'`.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_offline_signal_state_map_matches_the_live_reader() -> None:
    """TEST-RSEC-014: offline and live must resolve identical signal colours."""

    module = load_measurement_module()
    assert module._SIGNAL_STATE_MAP == _LIVE_SIGNAL_STATE_MAP


def test_measured_components_cover_every_normative_registry_component() -> None:
    """TEST-RSEC-015: a component missing from the list reads as satisfied.

    ``progress`` is deliberately excluded: it carries the R4 margin, which the
    measurement tracks on its own channel rather than as a cost.
    """

    module = load_measurement_module()
    expected = {
        _RESULT_NAME_BY_REGISTRY_NAME.get(component.name, component.name)
        for component in DEFAULT_RULEBOOK_V2_REGISTRY.components
        if component.normative_output and component.name != "progress"
    }
    assert set(module._NORMATIVE_COMPONENTS) == expected


def test_signal_states_read_only_the_requested_step() -> None:
    """A measurement that peeked ahead would leak future signal state into R3."""

    module = load_measurement_module()
    scenario = {
        "dynamic_map_states": {
            "light-a": {
                "type": "TRAFFIC_LIGHT",
                "state": {"object_state": ["LANE_STATE_GO", "LANE_STATE_STOP"]},
            }
        }
    }
    assert module.signal_states_at(scenario, 0) == {"light-a": "GREEN"}
    assert module.signal_states_at(scenario, 1) == {"light-a": "RED"}


def test_signal_states_fail_closed_past_the_logged_sequence() -> None:
    """A step beyond the logged sequence must read UNKNOWN, never a stale colour."""

    module = load_measurement_module()
    scenario = {
        "dynamic_map_states": {
            "light-a": {"type": "TRAFFIC_LIGHT", "state": {"object_state": ["LANE_STATE_GO"]}}
        }
    }
    assert module.signal_states_at(scenario, 5) == {"light-a": "UNKNOWN"}


def test_non_traffic_light_dynamic_states_are_ignored() -> None:
    module = load_measurement_module()
    scenario = {"dynamic_map_states": {"other": {"type": "SOMETHING_ELSE", "state": {}}}}
    assert module.signal_states_at(scenario, 0) == {}


def test_selected_records_filters_by_split_source_and_eligibility() -> None:
    """The measurement must not leak validation or test records into the sample."""

    module = load_measurement_module()
    payload = {
        "records": [
            {"scenario_uid": "w-train", "source": "waymo", "split": "train"},
            {"scenario_uid": "w-test", "source": "waymo", "split": "test"},
            {"scenario_uid": "pg-train", "source": "pg", "split": "train"},
            {
                "scenario_uid": "w-train-ineligible",
                "source": "waymo",
                "split": "train",
                "rulebook_eligible": False,
            },
        ]
    }
    selected = module.selected_records(payload, split="train", source="waymo", limit=None)
    assert [record["scenario_uid"] for record in selected] == ["w-train"]


def test_measurement_merge_is_order_independent() -> None:
    """Parallel and sequential runs must agree, so merging must be additive."""

    module = load_measurement_module()
    crawl = module.EGO_CRAWL_THRESHOLDS_MPS[1]
    first = module.Measurement()
    first.total_steps = 2
    first.components["ttc"].add(0.5, applicable=True)
    first.episode_returns.append(-10.0)
    first.latch_attribution["vehicle_yield:latch"] += 1
    first.final_blame_total["clearance"] += 3.0
    first.final_blame_dominant_when_negative["clearance"] += 1
    first.final_negative_episodes += 1
    first.final_negative_mass_total[crawl] += 3.0
    first.final_negative_mass_at_crawl[crawl] += 1.0

    second = module.Measurement()
    second.total_steps = 3
    second.components["ttc"].add(0.25, applicable=True)
    second.components["ttc"].add(0.0, applicable=True)
    second.episode_returns.append(4.0)
    second.final_blame_total["rss_lateral"] += 7.0
    second.final_negative_episodes += 1
    second.final_negative_mass_total[crawl] += 7.0
    second.final_negative_mass_at_crawl[crawl] += 4.0

    forward, backward = module.Measurement(), module.Measurement()
    forward.merge(first)
    forward.merge(second)
    backward.merge(second)
    backward.merge(first)

    assert forward.summary() == backward.summary()
    assert forward.total_steps == 5
    assert forward.components["ttc"].violated_steps == 2
    assert sorted(forward.episode_returns) == [-10.0, 4.0]
    assert forward.final_negative_episodes == 2
    assert forward.final_negative_mass_total[crawl] == pytest.approx(10.0)
    assert forward.final_negative_mass_at_crawl[crawl] == pytest.approx(5.0)


def test_time_headway_gate_excludes_slow_steps_instead_of_scoring_them_zero() -> None:
    """TEST-RSEC-016: a gated-out step must be inapplicable, not a satisfied step.

    The gate sweep exists to tell crawl-speed queueing apart from tailgating. If a
    step below the gate were recorded as an applicable zero-cost step, the higher
    gates would report a violation rate diluted by exactly the steps they were
    meant to exclude, and every gate would look equally good.
    """

    module = load_measurement_module()

    class _Candidate:
        def __init__(self, ego_speed_mps: float, gap_m: float) -> None:
            self.ego_speed_mps = ego_speed_mps
            self.gap_m = gap_m

    # 3 m at 3 m/s is a 1.0 s headway: violating below the 1.5 s threshold,
    # satisfied at 0.5 s, and out of scope entirely once the gate exceeds 3 m/s.
    resolved = module.time_headway_costs((_Candidate(3.0, 3.0),))

    assert resolved[(0.5, 1.5)] == (pytest.approx(1.0 / 3.0), True)
    assert resolved[(0.5, 0.5)] == (0.0, True)
    assert resolved[(2.0, 1.5)] == (pytest.approx(1.0 / 3.0), True)
    assert resolved[(5.0, 1.5)] == (0.0, False)
    assert resolved[(8.0, 1.5)] == (0.0, False)

    # No candidate at all is inapplicable at every gate: an ego with no leader is
    # not a compliant follower, it is simply not following anything.
    empty = module.time_headway_costs(())
    assert all(entry == (0.0, False) for entry in empty.values())
    assert set(empty) == {
        (gate, threshold)
        for gate in module.THW_MIN_EGO_SPEED_GATES_MPS
        for threshold in module.THW_THRESHOLDS_S
    }


def test_lateral_rss_variant_is_a_strict_relaxation_of_production() -> None:
    """TEST-RSEC-017: the scoped lateral rule may never charge more than production.

    ``d_safe^lat`` adds the neighbour's worst-case inward displacement, and that
    displacement goes *negative* for a neighbour moving away — so simply zeroing
    the actor term removes a credit as well as a debit and can charge the ego
    more than production did. A variant that is not a relaxation cannot be read
    as evidence that the rule was over-scoped.
    """

    module = load_measurement_module()

    class _Candidate:
        longitudinal_unsafe = True

        def __init__(self, gap_m: float, ego_inward: float, actor_inward: float) -> None:
            self.lateral_gap_m = gap_m
            self.ego_inward_speed_mps = ego_inward
            self.actor_inward_speed_mps = actor_inward

    # An approaching neighbour: production charges more, the variant relaxes.
    full, scoped = module.lateral_rss_costs((_Candidate(0.5, 0.2, 0.8),))
    assert full > scoped

    # A receding neighbour: production already charges less than the ego-alone
    # requirement would, and that credit must be kept rather than reversed.
    full, scoped = module.lateral_rss_costs((_Candidate(0.2, 0.6, -0.6),))
    assert scoped == pytest.approx(full)

    for gap in (0.05, 0.2, 0.5, 1.0):
        for ego_inward in (0.0, 0.3, 0.9):
            for actor_inward in (-0.9, -0.3, 0.0, 0.3, 0.9):
                full, scoped = module.lateral_rss_costs(
                    (_Candidate(gap, ego_inward, actor_inward),)
                )
                assert scoped <= full + 1e-12

    # An unscoped pair contributes nothing under either definition.
    assert module.lateral_rss_costs(()) == (0.0, 0.0)


def test_scalarization_family_contains_production_exactly() -> None:
    """TEST-RSEC-018: the family must reproduce SCAL-V1.1 at (a=3, sigma=1, lambda=1).

    Every grid number is compared against the production column, so a family that
    only approximates production would make the whole comparison meaningless.
    """

    module = load_measurement_module()
    config = ScalarizationConfig()

    for margins in (
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.87),
        (0.0, -0.35, 0.0, 0.2),
        (0.0, -1.0, -1.0, -1.0),
        (-0.5, -0.25, -0.75, 0.5),
        (0.0, 0.0, -0.001, 1.0),
    ):
        expected = scalarize_rulebook_margins(margins, config).reward
        produced = module.family_reward(
            margins, base=3.0, severity=1.0, flat=0.0, progress_weight=1.0
        )
        assert produced == pytest.approx(expected, abs=1e-12)


def test_rank_preservation_predicate_matches_a_brute_force_search() -> None:
    """TEST-RSEC-019: whatever the predicate admits must survive direct enumeration.

    The predicate is the only thing standing between a cheaper scalarization and
    a silently non-lexicographic one, so it is checked against an explicit search
    for a counterexample rather than trusted as algebra. Only *soundness* is
    asserted: the predicate compares against the supremum of the violating case,
    which is approached as the margin tends to zero but never attained, so it
    rejects knife-edge members that enumeration cannot break. Refusing a member
    that would have been admissible costs nothing; admitting one that is not
    would silently destroy the hierarchy.
    """

    module = load_measurement_module()
    # -1e-6 is above `family_reward`'s canonicalization tolerance, so it is a
    # genuine violation; -1e-9 would be snapped to satisfied and the
    # enumeration would disagree with the predicate for the wrong reason.
    grid = (-1.0, -0.6, -0.3, -1e-6, 0.0)
    progress = (-1.0, 0.0, 1.0)

    def has_counterexample(base: float, severity: float, flat: float, weight: float) -> bool:
        for level in range(3):
            best_violating = -float("inf")
            worst_satisfied = float("inf")
            for lower in itertools.product(grid, repeat=2 - level):
                for progress_margin in progress:
                    for own in grid:
                        margins = [0.0, 0.0, 0.0, progress_margin]
                        margins[level] = own
                        for offset, value in enumerate(lower):
                            margins[level + 1 + offset] = value
                        reward = module.family_reward(
                            tuple(margins),
                            base=base,
                            severity=severity,
                            flat=flat,
                            progress_weight=weight,
                        )
                        if own == 0.0:
                            worst_satisfied = min(worst_satisfied, reward)
                        else:
                            best_violating = max(best_violating, reward)
            if best_violating >= worst_satisfied:
                return True
        return False

    # Every member the predicate rejects outright must also be reported, so the
    # search below covers bases well outside the grid in both directions.
    for base in (1.5, 1.9, 2.0, 2.01, 2.2, 2.5, 3.0, 4.0):
        for severity in (0.0, 0.25, 0.5, 1.0):
            for flat in (0.0, 0.25):
                if module.is_rank_preserving(base, severity, flat, 1.0):
                    assert not has_counterexample(base, severity, flat, 1.0), (
                        f"predicate admitted a non-lexicographic member at base={base}, "
                        f"severity={severity}, flat={flat}"
                    )

    # A member well below the bound must be rejected, and enumeration must agree.
    assert not module.is_rank_preserving(1.5, 0.0, 0.0, 1.0)
    assert has_counterexample(1.5, 0.0, 0.0, 1.0)

    # The two published anchors: Veer et al. need a > 2 with severity outside the
    # priority weight, and SCAL-V1.1's severity-inside form needs the larger base
    # it re-derived.
    assert module.is_rank_preserving(2.01, 0.0, 0.0, 1.0)
    assert not module.is_rank_preserving(2.0, 0.0, 0.0, 1.0)
    assert module.is_rank_preserving(3.0, 1.0, 0.0, 1.0)
    assert not module.is_rank_preserving(2.9, 1.0, 0.0, 1.0)
    # Veer et al.'s own 1/N tie-breaker is affordable at a = 2.2 but not at 2.01.
    assert module.is_rank_preserving(2.2, 0.0, 0.25, 1.0)
    assert not module.is_rank_preserving(2.01, 0.0, 0.25, 1.0)


def test_family_grid_admits_only_rank_preserving_members() -> None:
    """TEST-RSEC-020: a non-lexicographic member must never reach the report."""

    module = load_measurement_module()
    grid = module.family_grid()
    assert grid, "the grid must not be empty"
    labels = [label for label, _, _, _ in grid]
    assert len(labels) == len(set(labels))
    for _, base, severity, flat in grid:
        assert module.is_rank_preserving(
            base, severity, flat, module.SCALARIZATION_FAMILY_PROGRESS_WEIGHT
        )
    assert ("a3_sev1_flat0", 3.0, 1.0, 0.0) in grid


def test_responsive_rss_matches_production_at_the_published_response_time() -> None:
    """TEST-RSEC-021: the swept safe distance must equal production's at rho = 1 s.

    `components/rss.py` fixes rho in a module constant, so the sweep carries its
    own copy of the formula. If the copy drifted, every responsive-RSS rate would
    describe a rule nobody implemented.
    """

    module = load_measurement_module()

    for ego_speed in (0.0, 3.0, 9.0, 20.0):
        for front_speed in (0.0, 5.0, 15.0):
            expected = safe_distance_m(
                ego_speed_mps=ego_speed, front_speed_mps=front_speed, ego_brake_mps2=8.0
            )
            produced = module.rss_safe_distance_at(
                ego_speed_mps=ego_speed,
                front_speed_mps=front_speed,
                ego_brake_mps2=8.0,
                response_time_s=RESPONSE_TIME_S,
            )
            assert produced == pytest.approx(expected, abs=1e-9)


def test_responsive_rss_is_zero_whenever_the_ego_is_braking_hard_enough() -> None:
    """TEST-RSEC-022: braking must zero the cost, or the region is not invariant.

    The whole justification for this variant is that from every state inside the
    unsafe envelope an action exists that removes the cost. If a braking ego were
    still charged, the variant would inherit exactly the defect it exists to fix.
    """

    module = load_measurement_module()

    class _Candidate:
        actor_id = "lead"
        gap_m = 1.0
        ego_speed_mps = 10.0
        front_speed_mps = 0.0

    hard_brake = module.responsive_rss_costs(
        (_Candidate(),), ego_brake_mps2=8.0, ego_accel_mps2=-4.0
    )
    coasting = module.responsive_rss_costs((_Candidate(),), ego_brake_mps2=8.0, ego_accel_mps2=0.0)

    assert all(cost == 0.0 for cost, _ in hard_brake.values())
    assert all(applicable for _, applicable in hard_brake.values())
    # A 1 m gap at 10 m/s behind a stopped leader is unsafe at every swept rho.
    assert all(cost > 0.0 for cost, _ in coasting.values())

    # A standstill ego has no available response and is never charged.
    class _Stopped(_Candidate):
        ego_speed_mps = 0.0

    stopped = module.responsive_rss_costs((_Stopped(),), ego_brake_mps2=8.0, ego_accel_mps2=0.0)
    assert all(cost == 0.0 and not applicable for cost, applicable in stopped.values())


def test_mission_snapshot_keeps_route_completion_within_range() -> None:
    """MissionSnapshot validates its own ranges; the replay must not trip them."""

    module = load_measurement_module()

    class _Projection:
        s_m = 25.0

    class _Route:
        length_m = 20.0

        def project(self, *args, **kwargs):
            return _Projection()

    class _Ego:
        position_xy = (0.0, 0.0)
        position_z = 0.0

    snapshot, s_m = module.mission_snapshot_at(
        route=_Route(), ego=_Ego(), step=3, mission_hash="uid", previous_s_m=None
    )
    assert s_m == pytest.approx(25.0)
    assert snapshot.route_completion == pytest.approx(1.0)
    assert snapshot.remaining_distance_m == pytest.approx(0.0)
    assert snapshot.step_index == 3


def test_speed_limit_is_admitted_only_from_real_map_provenance() -> None:
    """TEST-RSEC-023: a generator default must never be read as a traffic norm.

    Every ScenarioNet producer writes ``speed_limit_kmh``, but only a real-map
    converter fills it from a posted limit and records the ``speed_limit_mph``
    datum beside it. MetaDrive's PG exporter writes out whichever default the
    lane constructor held -- ``1000`` from ``abs_lane``, ``20`` from
    ``create_pg_block_utils`` -- under the same key and without converting the
    unit its own blocks document in m/s. Admitting either would price the
    expert against a constructor default, so the gate is provenance.
    """

    module = load_measurement_module()
    scenario = {
        "map_features": {
            # Waymo: posted limit with its source datum.
            "waymo_15mph": {"speed_limit_kmh": 24.14, "speed_limit_mph": 15.0},
            "waymo_45mph": {"speed_limit_kmh": 72.42, "speed_limit_mph": 45.0},
            # PG: the two constructor defaults, no source datum.
            "pg_straight": {"speed_limit_kmh": 1000.0},
            "pg_curve": {"speed_limit_kmh": 20.0},
            # Provenance present but the value itself unusable.
            "unrecorded": {"speed_limit_kmh": 0.0, "speed_limit_mph": 0.0},
            "sentinel_with_provenance": {"speed_limit_kmh": 1000.0, "speed_limit_mph": 621.4},
            "not_a_lane": {"type": "ROAD_EDGE_BOUNDARY"},
        }
    }

    limits = module.lane_speed_limits_mps(scenario)

    assert set(limits) == {"waymo_15mph", "waymo_45mph"}
    assert limits["waymo_15mph"] == pytest.approx(24.14 / 3.6)
    assert limits["waymo_45mph"] == pytest.approx(72.42 / 3.6)


def test_worst_named_agrees_with_max_and_breaks_ties_by_order() -> None:
    """TEST-RSEC-024: naming the blame must not change the macro cost."""

    module = load_measurement_module()

    assert module.worst_named(()) == (0.0, None)
    # An all-satisfied macro has no attributable sub-rule.
    assert module.worst_named((("a", 0.0), ("b", 0.0))) == (0.0, None)
    assert module.worst_named((("a", 0.2), ("b", 0.7), ("c", 0.4))) == (0.7, "b")
    # Ties keep the first candidate, matching ``macro_cost_with_blame``.
    assert module.worst_named((("a", 0.5), ("b", 0.5))) == (0.5, "a")

    costs = [("r%d" % index, value) for index, value in enumerate((0.0, 0.31, 0.9, 0.9, 0.12))]
    worst, blame = module.worst_named(costs)
    assert worst == pytest.approx(max(value for _, value in costs))
    assert blame == "r2"


def test_crawl_split_of_the_residual_is_a_monotone_partition() -> None:
    """TEST-RSEC-025: the not-avoidable share must be a fraction, and monotone.

    The split exists to answer whether the below-standstill residual is penalty
    the expert could have avoided by slowing. Mass charged at a crawl is a
    subset of all charged mass, and raising the crawl threshold can only move
    mass into that subset, never out of it. A violation of either would make the
    reported fraction meaningless.
    """

    module = load_measurement_module()
    measurement = module.Measurement()
    measurement.final_negative_episodes = 2
    # 100 units charged in the tail; progressively more of it at low ego speed.
    for crawl, at_crawl in zip(module.EGO_CRAWL_THRESHOLDS_MPS, (5.0, 30.0, 55.0, 80.0)):
        measurement.final_negative_mass_total[crawl] = 100.0
        measurement.final_negative_mass_at_crawl[crawl] = at_crawl

    reported = measurement.summary()["final_rulebook_blame"][
        "penalty_mass_not_avoidable_by_slowing"
    ]
    fractions = [
        reported[f"ego_speed_le_{crawl:g}mps"]["fraction_of_penalty_mass"]
        for crawl in module.EGO_CRAWL_THRESHOLDS_MPS
    ]
    assert all(0.0 <= value <= 1.0 for value in fractions)
    assert fractions == sorted(fractions)
    assert fractions[-1] == pytest.approx(0.8)

    # An empty tail must report absence, not a fabricated zero share.
    empty = module.Measurement().summary()["final_rulebook_blame"]
    assert empty["episodes_below_standstill"] == 0
    assert all(
        entry["fraction_of_penalty_mass"] is None
        for entry in empty["penalty_mass_not_avoidable_by_slowing"].values()
    )
    assert empty["penalty_mass_by_sub_rule_at_crawl"] == {}


def test_per_sub_rule_crawl_mass_never_exceeds_that_sub_rule_total() -> None:
    """TEST-RSEC-026: the cross-tab must refine the aggregate, not contradict it.

    The aggregate split says how much of the below-standstill tail was charged
    to an already-stopped ego. It cannot say whether that is the expert stopping
    in a bad place (a position rule, controlled-invariant) or the rulebook
    charging a stationary ego for another agent's motion (an interaction rule,
    not controlled-invariant). Only the per-sub-rule cross-tab separates those,
    and it is trustworthy only if each cell stays inside its own sub-rule's
    total and grows monotonically with the crawl threshold.
    """

    module = load_measurement_module()
    measurement = module.Measurement()
    measurement.final_negative_episodes = 3
    thresholds = module.EGO_CRAWL_THRESHOLDS_MPS
    # `clearance` charged almost entirely at a crawl; `solid_line` almost none.
    measurement.final_blame_mass_in_negative.update({"clearance": 40.0, "solid_line": 60.0})
    for crawl, mass in zip(thresholds, (30.0, 32.0, 35.0, 38.0)):
        measurement.final_blame_mass_at_crawl[("clearance", crawl)] = mass
    for crawl, mass in zip(thresholds, (1.0, 2.0, 4.0, 9.0)):
        measurement.final_blame_mass_at_crawl[("solid_line", crawl)] = mass

    reported = measurement.summary()["final_rulebook_blame"]["penalty_mass_by_sub_rule_at_crawl"]

    # Most-expensive sub-rule first, so the table reads without re-sorting.
    assert list(reported) == ["solid_line", "clearance"]
    for name, total in (("clearance", 40.0), ("solid_line", 60.0)):
        entry = reported[name]
        assert entry["reward_units_in_negative_episodes"] == pytest.approx(total)
        cells = [entry[f"at_ego_speed_le_{crawl:g}mps"] for crawl in thresholds]
        assert cells == sorted(cells), "crawl mass must be monotone in the threshold"
        assert cells[-1] <= total, "a cell may never exceed its own sub-rule total"
        # The reported share is rounded to 4 decimals for the report.
        assert entry["share_at_lowest_crawl"] == pytest.approx(cells[0] / total, abs=5e-5)

    # The discriminating quantity: a sub-rule charging a stopped ego is separated
    # from one charging a moving ego, which the aggregate split cannot show.
    assert reported["clearance"]["share_at_lowest_crawl"] == pytest.approx(0.75, abs=5e-5)
    assert reported["solid_line"]["share_at_lowest_crawl"] == pytest.approx(1.0 / 60.0, abs=5e-5)


def test_at_fault_gate_covers_only_the_interaction_sub_rules() -> None:
    """TEST-RSEC-027: the gate must be a blame rule, not a blanket amnesty.

    ADR-070 gates the sub-rules whose cost is set by another agent's state,
    because from a stopped ego no action avoids them -- the controlled-invariance
    test that rejected `rss`. Position sub-rules must stay ungated: from a state
    stopped astride a lane marking an action that leaves it exists, so charging
    it is a normative disagreement rather than an unavoidable cost. A gate that
    silently covered them would make stopping anywhere free.
    """

    module = load_measurement_module()

    assert module.AT_FAULT_GATED_SUB_RULES == {"clearance", "rss_lateral", "ttc"}
    for position_rule in ("offroad", "solid_line", "dashed_line", "wrong_carriageway"):
        assert position_rule not in module.AT_FAULT_GATED_SUB_RULES
    for control_rule in ("signal", "stop", "crosswalk", "vehicle_yield", "speed_limit"):
        assert control_rule not in module.AT_FAULT_GATED_SUB_RULES

    # The adopted threshold must be one of the two published nuPlan values, and
    # small enough that crawling under it buys immunity at zero R4 progress.
    assert module.FINAL_AT_FAULT_GATE_MPS in module.AT_FAULT_GATE_THRESHOLDS_MPS
    assert module.AT_FAULT_GATE_THRESHOLDS_MPS == (0.005, 0.05)
    from thesis_rl.rulebook.v2.components.progress import MISSION_PROGRESS_REFERENCE_SPEED_MPS

    assert module.FINAL_AT_FAULT_GATE_MPS / MISSION_PROGRESS_REFERENCE_SPEED_MPS < 0.005

    # Gating is a relaxation: it may only ever lower the macro cost, and only
    # through a gated sub-rule.
    costs = (("clearance", 0.8), ("rss_lateral", 0.3), ("ttc", 0.5))
    ungated, ungated_blame = module.worst_named(costs)
    gated, _ = module.worst_named(tuple((name, 0.0) for name, _ in costs))
    assert ungated == pytest.approx(0.8) and ungated_blame == "clearance"
    assert gated == 0.0

    # A position rule at the same cost survives the gate untouched.
    mixed = (("clearance", 0.0), ("solid_line", 0.8))
    assert module.worst_named(mixed) == (0.8, "solid_line")


def test_every_priced_variant_is_registered_by_the_accumulator() -> None:
    """TEST-RSEC-028: regression for the at-fault gate's first run.

    The per-episode totals and the accumulator listed the variant names
    independently. Adding `final_gate*` to the accumulator alone raised
    ``KeyError`` on all 1100 records -- loudly, but only after a full replay.
    One source of names removes the failure mode; this pins that they agree,
    and that the accumulator is complete before any record is merged.
    """

    module = load_measurement_module()
    names = module.all_variant_names()

    assert len(names) == len(set(names)), "variant names must be unique"
    assert set(module.Measurement().variant_returns) == set(names)
    # The three at-fault columns the decision is read from must be present.
    for required in ("final", "final_ungated", "final_gate0.05", "final_gate0.005"):
        assert required in names

    # A summary built before any record must expose every variant, so a missing
    # column is a structural error rather than an empty percentile.
    reported = module.Measurement().summary()["counterfactual_rulebooks"]
    assert set(reported) == set(names)


def test_the_telescoping_residual_keeps_both_signs_apart() -> None:
    """TEST-RSEC-029: regression for `C51`.

    The residual was accumulated as ``abs(...)`` into one maximum, so an
    over-payment and an under-payment were indistinguishable in the report. The
    two have opposite meanings: a deficit is the forward clip truncating advance
    the ego earned, a surplus is advance credited that the ego never made — the
    ratchet `C50` executes. On the expert panel the deficit is 62.294, so any
    reduction to a single number, absolute or by magnitude, would have hidden a
    surplus behind it. This pins that both survive, through the merge as well,
    because the parallel replay reaches the report only through `merge`.
    """

    module = load_measurement_module()

    measurement = module.Measurement()
    assert measurement.v51_telescoping_max_surplus == 0.0
    assert measurement.v51_telescoping_max_deficit == 0.0

    # A large deficit must not conceal a smaller surplus, which is the failure
    # the absolute form had.
    measurement.observe_telescoping_residual(-62.294)
    measurement.observe_telescoping_residual(+0.75)
    measurement.observe_telescoping_residual(-1.5)
    assert measurement.v51_telescoping_max_surplus == pytest.approx(0.75)
    assert measurement.v51_telescoping_max_deficit == pytest.approx(-62.294)

    # Each worker reports its own extremes and the merge must keep both sides.
    worker = module.Measurement()
    worker.observe_telescoping_residual(+9.5)
    worker.observe_telescoping_residual(-3.0)
    measurement.merge(worker)
    assert measurement.v51_telescoping_max_surplus == pytest.approx(9.5)
    assert measurement.v51_telescoping_max_deficit == pytest.approx(-62.294)

    reported = measurement.summary()["rulebook_v51"]
    assert reported["telescoping_max_surplus"] == pytest.approx(9.5)
    assert reported["telescoping_max_deficit"] == pytest.approx(-62.294)
    # The absolute figure earlier revisions published is recoverable, so a run
    # against this report stays comparable with the 2026-09-07 calibration.
    assert max(
        reported["telescoping_max_surplus"], -reported["telescoping_max_deficit"]
    ) == pytest.approx(62.294)


def test_the_telescoping_residual_publishes_its_per_episode_distribution() -> None:
    """TEST-RSEC-030: `F14`'s "only the maximum is recorded" is closed.

    Two extremes say how large the worst episode is and nothing about how many
    episodes carry the defect. That distinction is not decorative: the expert
    panel's mean deficit is a fraction of a channel unit against a maximum of
    62.294, so the distribution is violently skewed, and a bottom-tail statistic
    such as `fraction_below_standstill` is measured exactly where such skew
    bites. `concentration_in_worst_episode` is the scalar that separates one
    outlier from a spread defect, and this pins both readings of it.
    """

    module = load_measurement_module()

    empty = module.Measurement().summary()["rulebook_v51"]["telescoping_residual"]
    assert empty == {"episodes": 0}

    # One outlier against many clean episodes: concentration near 1.
    outlier = module.Measurement()
    for _ in range(99):
        outlier.observe_telescoping_residual(0.0)
    outlier.observe_telescoping_residual(-62.294)
    reported = outlier.summary()["rulebook_v51"]["telescoping_residual"]
    assert reported["episodes"] == 100
    assert reported["episodes_with_residual"] == 1
    assert reported["p50"] == pytest.approx(0.0)
    assert reported["concentration_in_worst_episode"] == pytest.approx(1.0)

    # The same total spread over every episode: concentration near zero, and the
    # two extremes alone would not have told these two populations apart.
    spread = module.Measurement()
    for _ in range(100):
        spread.observe_telescoping_residual(-0.62294)
    reported = spread.summary()["rulebook_v51"]["telescoping_residual"]
    assert reported["episodes"] == 100
    assert reported["episodes_with_residual"] == 100
    assert reported["total_absolute"] == pytest.approx(62.294)
    assert reported["concentration_in_worst_episode"] == pytest.approx(0.01)

    # The merge must carry episodes, not just extremes, or the parallel replay
    # would report the distribution of one worker.
    combined = module.Measurement()
    combined.merge(outlier)
    combined.merge(spread)
    reported = combined.summary()["rulebook_v51"]["telescoping_residual"]
    assert reported["episodes"] == 200
    assert reported["total_absolute"] == pytest.approx(124.588)
    assert combined.v51_telescoping_max_deficit == pytest.approx(-62.294)


def _default_scalarization_yaml_values() -> dict[str, float]:
    """Plain-line parse of `conf/scalarization/default.yaml`, the same style
    `test_rulebook_v51_orderings.py`'s `_shipped_gamma` uses: a fixture that
    hardcoded these values would keep asserting the old configuration after
    it moved, which is exactly the failure being guarded against.
    """

    config_path = Path(__file__).parents[1] / "conf" / "scalarization" / "default.yaml"
    values: dict[str, float] = {}
    for line in config_path.read_text(encoding="utf-8").splitlines():
        stripped = line.split("#", 1)[0].strip()
        if not stripped or ":" not in stripped:
            continue
        key, _, raw_value = stripped.partition(":")
        key = key.strip()
        raw_value = raw_value.strip()
        if key in {
            "priority_base",
            "severity",
            "flat_tie_breaker",
            "progress_weight",
            "relaxable_weight",
            "progress_rate_weight",
        }:
            values[key] = float(raw_value)
    return values


def test_production_scalarization_config_matches_the_shipped_configuration() -> None:
    """Regression: `production_scalarization_config()` must construct without
    raising and must match `conf/scalarization/default.yaml`, read directly
    rather than duplicated as a second hardcoded literal.

    `production_scalarization`'s `priority_base` carried the literal `2.2` --
    correct when written 2026-09-01 -- until ADR-081 moved the six-level
    mode's required base to `2.5` on 2026-09-07 (`SIX_LEVEL_PRIORITY_BASE`)
    without this call site following it. `ScalarizationConfig.__post_init__`
    then raised `ScalarizationConfigurationError` on construction, for every
    one of the 1100 `train` records, and the replay's own handler counted
    each as a skipped scenario: `scenarios_measured` read `0`, silently,
    exactly the failure shape a comment already on this line documented for
    an earlier, different cause. Confirmed pre-fix: a full run against the
    unmodified tree reported `scenarios_measured: 0` and
    `scenarios_skipped: {"error:ScalarizationConfigurationError": 1100}`,
    with every sampled message reading
    "Mode 'six_level_priority_weighted_rank' requires priority_base=2.5,
    got 2.2." (`outputs/a7_m1/a7-m1-run.log`, 2026-09-10).
    """

    module = load_measurement_module()
    config = module.production_scalarization_config()
    assert config.mode == "six_level_priority_weighted_rank"

    shipped = _default_scalarization_yaml_values()
    assert config.priority_base == pytest.approx(shipped["priority_base"])
    assert config.severity == pytest.approx(shipped["severity"])
    assert config.flat_tie_breaker == pytest.approx(shipped["flat_tie_breaker"])
    assert config.progress_weight == pytest.approx(shipped["progress_weight"])
    assert config.relaxable_weight == pytest.approx(shipped["relaxable_weight"])
    assert config.progress_rate_weight == pytest.approx(shipped["progress_rate_weight"])


def test_a7_reward_matches_the_plan_ss5_1_term_by_term() -> None:
    """A7 ExecPlan SS5.1: `a7_reward` is transcribed independently, so it is
    checked against a from-scratch reimplementation rather than against
    remembered numbers -- the same discipline `v51_reward` is held to.
    """

    module = load_measurement_module()

    def reference(l1, l2, l3, l4, delta_q, w5, phi, lam4, base, severity):
        total = 0.0
        for weight, cost in zip((base**3, base**2, base), (l1, l2, l3)):
            margin = -cost
            satisfied = 1.0 if margin == 0.0 else 0.0
            total += weight * ((satisfied - 1.0) + severity * margin) + phi * margin
        m5 = -l4
        satisfied5 = 1.0 if m5 == 0.0 else 0.0
        total += w5 * ((satisfied5 - 1.0) + severity * m5)
        total += lam4 * delta_q
        return total

    cases = [
        dict(
            l1=0.0,
            l2=0.0,
            l3=0.0,
            l4=0.0,
            delta_q=0.0,
            w5=0.15,
            phi=0.0,
            lam4=2.0,
            base=2.5,
            severity=0.30,
        ),
        dict(
            l1=0.3,
            l2=0.0,
            l3=0.0,
            l4=0.0,
            delta_q=0.2,
            w5=0.15,
            phi=0.25,
            lam4=2.0,
            base=2.5,
            severity=0.30,
        ),
        dict(
            l1=0.0,
            l2=1.0,
            l3=0.0,
            l4=1.0,
            delta_q=-1.0,
            w5=0.25,
            phi=0.0,
            lam4=1.9,
            base=2.5,
            severity=0.30,
        ),
    ]
    for case in cases:
        assert module.a7_reward(**case) == pytest.approx(reference(**case))

    # ExecPlan SS6.3's own worked figure: a fully-violated K2 step costs
    # exactly 8.125 at `phi=0` and 8.375 at `phi=0.25` retained, at the
    # production pair -- the number the plan derives `w5`'s upper bound from.
    for phi, expected in ((0.0, -8.125), (0.25, -8.375)):
        reward = module.a7_reward(
            l1=0.0,
            l2=1.0,
            l3=0.0,
            l4=0.0,
            delta_q=0.0,
            w5=0.15,
            phi=phi,
            lam4=2.0,
            base=2.5,
            severity=0.30,
        )
        assert reward == pytest.approx(expected)


def test_a7_rank_preservation_reproduces_the_plans_worked_margins() -> None:
    """ExecPlan SS5.1: `phi` must reach only K1-K3, not K4 (`w5`).

    The plan records one transcription that got this backwards: letting
    `phi` count K4's own level as one more "lower priority level" moves the
    thinnest margin from k=2 to k=3 and changes every ratio. The correct
    implementation is checked against the plan's own worked figures at
    `a=2.5, sigma=0.30, lam4=2.0, w5=0.15`; the wrong one is shown to
    disagree, reproducing the plan's own regression figures exactly.
    """

    module = load_measurement_module()
    base, severity, w5, lam4 = 2.5, 0.30, 0.15, 2.0
    weights = module.family_weights(base)

    def ratio(phi: float, index: int) -> float:
        lower = weights[index + 1 :]
        tail = w5 * (1.0 + severity) + lam4 * module.A7_DELTA_Q_MAX
        bound = (1.0 + severity) * sum(lower) + phi * len(lower) + tail
        return weights[index] / bound

    for phi, expected in ((0.0, (1.1514, 1.1478, 1.1390)), (0.25, (1.1105, 1.0975, 1.1390))):
        for index, value in enumerate(expected):
            assert ratio(phi, index) == pytest.approx(value, abs=5e-5)
        assert module.a7_is_rank_preserving(base, severity, phi, w5, lam4) is True

    def wrong_ratio(phi: float, index: int) -> float:
        # The trap: `w5` folded into the "lower" sum subject to `(1 + sigma)`,
        # and `phi` counting K4 as one more lower priority level.
        lower = weights[index + 1 :]
        tail = lam4 * module.A7_DELTA_Q_MAX
        bound = (1.0 + severity) * (sum(lower) + w5) + phi * (len(lower) + 1) + tail
        return weights[index] / bound

    wrong = [round(wrong_ratio(0.25, index), 4) for index in range(3)]
    assert wrong == [1.0911, 1.0513, 1.0225]
    assert min(range(3), key=lambda index: wrong[index]) == 2, (
        "the trap moves the thinnest margin to k=3"
    )

    # A member below the bound must be rejected: `w5` alone above its cap.
    assert not module.a7_is_rank_preserving(base, severity, 0.0, 0.384616, lam4)
    assert module.a7_is_rank_preserving(base, severity, 0.0, 0.38460, lam4)


def test_a7_grid_has_six_admissible_members_at_the_production_pair() -> None:
    """ExecPlan SS10/`M1` task 3: `w5 x phi` at the approved `lam4 = 2.0`
    (four members), plus `DEC-A7-010`'s two pre-registered `lam4`
    alternatives at the approved `(w5, phi)` (two more) -- six members,
    every one rank-preserving, none silently dropped or duplicated.
    """

    module = load_measurement_module()
    grid = module.a7_grid()
    labels = [label for label, _, _, _ in grid]
    assert len(labels) == 6
    assert len(set(labels)) == 6
    for _, w5, phi, lam4 in grid:
        assert module.a7_is_rank_preserving(module.A7_BASE, module.A7_SEVERITY, phi, w5, lam4)
    assert {(w5, phi, lam4) for _, w5, phi, lam4 in grid} == {
        (0.15, 0.0, 2.0),
        (0.15, 0.25, 2.0),
        (0.25, 0.0, 2.0),
        (0.25, 0.25, 2.0),
        (0.15, 0.0, 1.25),
        (0.15, 0.0, 1.9),
    }
    names = module.all_variant_names()
    assert all(f"a7_{label}" in names for label in labels)


def test_a7_standstill_baseline_is_exactly_zero() -> None:
    """The trap the plan records at ExecPlan SS10/`M1` task 3: A7 deletes
    `L6` entirely, so a stopped in-lane ego violates nothing under any A7
    grid member -- unlike every six-level variant, whose baseline is
    `-lambda6 * (dt / T_REF) * T`. Checked explicitly rather than trusting
    this function's trailing fallback, which exists for unmatched names, not
    because it happens to be the right value here.
    """

    module = load_measurement_module()
    for label, _w5, _phi, _lam4 in module.a7_grid():
        for steps in (0, 1, 500):
            assert module.v51_standstill_return(f"a7_{label}", steps) == 0.0


def test_a7_grid_variants_fraction_below_standstill_uses_the_zero_baseline() -> None:
    """Integration point for the standstill trap: the generic
    `counterfactual_rulebooks` report must compare A7 variants against 0,
    the baseline above pins directly, not against the six-level
    `-lambda6 * (dt / T_REF) * T` form every `v51_*`/`v51cal_*` variant uses.
    """

    module = load_measurement_module()
    measurement = module.Measurement()
    label = next(iter(module.a7_grid()))[0]
    variant = f"a7_{label}"
    measurement.episode_returns = [-1.0, 0.5, -0.2, 3.0]
    measurement.episode_steps = [10, 10, 10, 10]
    measurement.variant_returns[variant] = [-1.0, 0.5, -0.2, 3.0]

    reported = measurement.summary()["counterfactual_rulebooks"][variant]
    assert reported["fraction_below_standstill"] == pytest.approx(2 / 4)


def test_a7_exposure_reports_three_objects_and_the_budget_rule() -> None:
    """`M1` task 1: raw, per-span and discounted exposure must be reported
    separately, and SS6.4's `tau_i` rule -- the undiscounted realized
    maximum, minus any DECLARED panel-defect exclusion, with the exclusions
    listed per record -- must be computed on `raw` alone, not silently mixed
    with the other two objects `D1`'s eventual choice may still need.
    """

    module = load_measurement_module()
    measurement = module.Measurement()

    # "e1": the higher-exposure record, with a positive route length.
    measurement.observe_a7_exposure(
        "e1",
        {"X_imp": 0.0, "X_int": 4.0, "X_hard": 1.0, "X_soft": 0.6},
        {"X_imp": 0.0, "X_int": 2.0, "X_hard": 0.5, "X_soft": 0.3},
        {"X_imp": 0.0, "X_int": 3.5, "X_hard": 0.9, "X_soft": 0.55},
    )
    # "e2": a zero-length route, which must contribute no `per_span` sample
    # rather than a division artifact -- the same guard `v51_q` already uses.
    measurement.observe_a7_exposure(
        "e2",
        {"X_imp": 1.0, "X_int": 1.0, "X_hard": 0.2, "X_soft": 0.1},
        None,
        {"X_imp": 0.9, "X_int": 0.9, "X_hard": 0.18, "X_soft": 0.09},
    )

    reported = measurement.summary()["a7_measurement"]["exposure_by_channel"]

    raw_int = reported["raw"]["X_int"]
    assert raw_int["episodes"] == 2
    assert raw_int["max"] == pytest.approx(4.0)
    assert raw_int["max_scenario_uid"] == "e1"
    assert raw_int["tau_i"] == pytest.approx(4.0)
    assert raw_int["declared_panel_defect_exclusions"] == []
    assert raw_int["top_1pct_records"] == [{"scenario_uid": "e1", "value": 4.0}]

    per_span_int = reported["per_span"]["X_int"]
    assert per_span_int["episodes"] == 1
    assert per_span_int["max"] == pytest.approx(2.0)
    assert "tau_i" not in per_span_int

    discounted_imp = reported["discounted_gamma_0_9982"]["X_imp"]
    assert discounted_imp["episodes"] == 2
    assert discounted_imp["max"] == pytest.approx(0.9)
    assert "tau_i" not in discounted_imp

    # `X_imp` is `tau_1`'s evidence (ExecPlan SS6.4): a nonzero raw maximum
    # falsifies `tau_1 = 0` directly rather than by assumption.
    assert reported["raw"]["X_imp"]["max"] == pytest.approx(1.0)

    empty = module.Measurement().summary()["a7_measurement"]["exposure_by_channel"]
    assert empty["raw"]["X_int"] == {"episodes": 0}


def test_a7_measurement_fields_merge_order_independently() -> None:
    """The parallel replay reaches the report only through `merge`, so every
    A7 `M1` accumulator must combine additively regardless of worker order --
    the same guarantee `test_measurement_merge_is_order_independent` pins for
    the pre-existing accumulators.
    """

    module = load_measurement_module()

    first = module.Measurement()
    first.observe_a7_exposure(
        "e1",
        {"X_imp": 0.0, "X_int": 1.0, "X_hard": 0.0, "X_soft": 0.05},
        {"X_imp": 0.0, "X_int": 0.5, "X_hard": 0.0, "X_soft": 0.02},
        {"X_imp": 0.0, "X_int": 0.9, "X_hard": 0.0, "X_soft": 0.04},
    )
    first.a7_k2_argmax["ttc"] += 5
    first.a7_k3_argmax["offroad"] += 2
    first.a7_k2_k3_cooccurring_steps = 1
    first.a7_k2_k3_cooccurring_episodes = 1
    first.a7_k2_k3_joint_samples.append((0.3, 0.4))

    second = module.Measurement()
    second.observe_a7_exposure(
        "e2",
        {"X_imp": 0.2, "X_int": 4.0, "X_hard": 0.1, "X_soft": 0.0},
        {"X_imp": 0.1, "X_int": 2.0, "X_hard": 0.05, "X_soft": 0.0},
        {"X_imp": 0.18, "X_int": 3.6, "X_hard": 0.09, "X_soft": 0.0},
    )
    second.a7_k2_argmax["clearance"] += 3
    second.a7_k2_k3_cooccurring_steps = 2
    second.a7_k2_k3_cooccurring_episodes = 1
    second.a7_k2_k3_joint_samples.append((0.6, 0.1))

    forward, backward = module.Measurement(), module.Measurement()
    forward.merge(first)
    forward.merge(second)
    backward.merge(second)
    backward.merge(first)

    assert forward.summary()["a7_measurement"] == backward.summary()["a7_measurement"]
    reported = forward.summary()["a7_measurement"]["exposure_by_channel"]["raw"]["X_int"]
    assert reported["episodes"] == 2
    assert reported["max"] == pytest.approx(4.0)
    assert forward.a7_k2_k3_cooccurring_steps == 3
    assert forward.a7_k2_k3_cooccurring_episodes == 2
    assert sum(forward.a7_k2_argmax.values()) == 8


def test_a7_argmax_counts_frequency_not_mass() -> None:
    """`M1` task 4, `F12`: whether `clearance` is ever the argmax at all is a
    frequency question reward mass cannot answer -- a rarely-winning,
    low-cost sub-rule and a never-winning one both contribute negligible
    mass, and only a frequency count tells them apart.
    """

    module = load_measurement_module()
    measurement = module.Measurement()
    measurement.a7_k2_argmax.update({"ttc": 97, "rss_lateral": 2, "none": 900})

    reported = measurement.summary()["a7_measurement"]["argmax_within_level"]["k2_interaction_risk"]
    assert reported["applicable_steps"] == 999
    assert "clearance" not in reported["by_sub_rule"]
    assert reported["by_sub_rule"]["ttc"]["steps"] == 97
    assert reported["by_sub_rule"]["ttc"]["frequency"] == pytest.approx(97 / 999, abs=1e-6)
    assert reported["by_sub_rule"]["none"]["frequency"] == pytest.approx(900 / 999, abs=1e-6)

    empty = module.Measurement().summary()["a7_measurement"]["argmax_within_level"][
        "k3_non_negotiable_compliance"
    ]
    assert empty == {"applicable_steps": 0, "by_sub_rule": {}}


def test_a7_k2_k3_cooccurrence_measures_joint_steps_and_episodes() -> None:
    """`DEC-A7-013`: prices the hierarchy's least-supported adjacency
    directly, as the steps and episodes where K2 and K3 are non-zero
    together, plus their joint distribution on exactly those steps.
    """

    module = load_measurement_module()
    measurement = module.Measurement()
    measurement.total_steps = 1000
    measurement.a7_k2_k3_cooccurring_steps = 3
    measurement.a7_k2_k3_cooccurring_episodes = 2
    measurement.a7_k2_k3_joint_samples.extend([(0.2, 0.5), (0.4, 0.5), (0.9, 0.1)])

    reported = measurement.summary()["a7_measurement"]["k2_k3_cooccurrence"]
    assert reported["steps_both_nonzero"] == 3
    assert reported["steps_both_nonzero_pct_of_all_steps"] == pytest.approx(0.3)
    assert reported["episodes_both_nonzero"] == 2
    joint = reported["joint_distribution_on_those_steps"]
    assert joint["k2_p50"] == pytest.approx(0.4)
    assert joint["k3_p50"] == pytest.approx(0.5)

    empty = module.Measurement().summary()["a7_measurement"]["k2_k3_cooccurrence"]
    assert empty["steps_both_nonzero"] == 0
    assert empty["joint_distribution_on_those_steps"] is None
