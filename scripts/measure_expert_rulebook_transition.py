"""Production-equivalent replay of the logged expert through the Rulebook v2 transition.

`measure_expert_rulebook_costs.py` answers a narrower question: it calls five
sub-rule evaluators directly, which is enough for the RSS response-time sweep but
leaves seven normative sub-rules unmeasured. Because both macro rules aggregate
by ``max``, that made its reported R2/R3 costs a strict *lower bound* on what the
production reward charges the expert.

This script closes that gap by driving the logged Waymo SDC track through
``evaluate_transition`` itself — the same entry point the live wrapper calls —
with the real ``RulebookMemory`` threaded across steps. Every normative sub-rule
is therefore evaluated exactly as in training, including the four traffic-control
rules and their persistence latches, which no measurement has covered so far.

Two things are deliberately *not* simulated:

* R1 ``collision`` reads physics contact records, which do not exist offline. The
  replay passes an empty contact set, so R1 is zero by construction and is
  reported as ``NOT_MEASURED`` rather than as a measurement. This is harmless for
  the question at hand: the logged human expert does not crash.
* Ego pose and velocity come from the recorded track, never from a policy. Every
  measured cost is therefore a property of the *metric definition*, not of any
  learned behaviour.

The headline output is the per-episode scalarized return under the production
``SCAL-V1.1`` configuration, decomposed by channel. If a competent human driver
scores below a policy that never moves, the reward cannot rank policies by
driving quality, and no amount of training will fix that.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import pickle
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from thesis_rl.mission.types import MissionSnapshot
from thesis_rl.reward.scalarization import (
    SIX_LEVEL_VECTOR_SCHEMA_ID,
    ScalarizationConfig,
    scalarize_rulebook_margins,
)
from thesis_rl.rulebook.v2.components.at_fault_gate import ego_is_stopped
from thesis_rl.rulebook.v2.components.clearance import CLEARANCE_THRESHOLDS_M
from thesis_rl.rulebook.v2.components.progress import MISSION_PROGRESS_REFERENCE_SPEED_MPS
from thesis_rl.rulebook.v2.components.road import dashed_lateral_penetration
from thesis_rl.rulebook.v2.components.rss import (
    FRONT_MAX_BRAKE_MPS2,
    MAX_RESPONSE_ACCEL_MPS2,
    RSS_STANDSTILL_SPEED_MPS,
    RSSCalibrationArtifact,
)
from thesis_rl.rulebook.v2.components.rss_lateral import lateral_safe_distance_m
from thesis_rl.rulebook.v2.context.pg_static_adapter import build_pg_static_adapter_result
from thesis_rl.rulebook.v2.context.waymo_static_adapter import build_waymo_static_adapter_result
from thesis_rl.rulebook.v2.geometry.drivable import (
    DrivableLaneRecord,
    carriageway_surfaces_for_ego,
    drivable_surface_for_ego,
)
from thesis_rl.rulebook.v2.geometry.footprint import oriented_bounding_box
from thesis_rl.rulebook.v2.geometry.vertical import VERTICAL_COMPATIBILITY_TOLERANCE_M
from thesis_rl.rulebook.v2.memory import apply_cache_delta
from thesis_rl.rulebook.v2.geometry.lanes import associate_route_lane
from thesis_rl.rulebook.v2.transition import (
    control_line_diagnostics,
    RulebookTransitionConfig,
    _rss_candidates,
    _rss_lateral_candidates,
    build_episode_cache,
    evaluate_transition,
    initial_memory_for_snapshot,
)
from thesis_rl.rulebook.v2.wrapper import ego_kinematics_payload
from thesis_rl.runtime.comfort_diagnostics import (
    COMFORT_STATISTICS,
    NUPLAN_COMFORT_BOUNDS,
    ComfortEpisodeAccumulator,
)
from thesis_rl.rulebook.v2.types import (
    ActorClass,
    ActorSnapshot,
    EnvSnapshot,
    MapFeatureClass,
    RulebookMemory,
)

# MetaDrive's per-vehicle-type constant (`pg_space.py`: max_speed_km_h=80).
DEFAULT_SPEED_CAP_MPS = 80.0 / 3.6
# ADR-047: b_meas ~= 10.7 m/s^2 clamped by the physical bound to 8.0.
DEFAULT_EGO_BRAKE_MPS2 = 8.0
DELTA_T_S = 0.1
_CALIBRATION_HASH = "expert-transition-measurement"
# Enough to tell one systematic failure from a handful of unusable records.
_ERROR_SAMPLE_LIMIT = 5

_ACTOR_CLASSES = {
    "VEHICLE": ActorClass.VEHICLE,
    "PEDESTRIAN": ActorClass.PEDESTRIAN,
    "CYCLIST": ActorClass.CYCLIST,
    "TRAFFIC_CONE": ActorClass.STATIC_COLLIDABLE,
    "TRAFFIC_BARRIER": ActorClass.STATIC_COLLIDABLE,
    "TRAFFIC_OBJECT": ActorClass.STATIC_COLLIDABLE,
}

# Mirrors `context.metadrive_live._LIVE_SIGNAL_STATE_MAP`. The live reader maps
# MetaDrive object states; offline the same colours come straight from the
# logged `dynamic_map_states` sequence the light manager itself replays.
_SIGNAL_STATE_MAP = {
    "TRAFFIC_LIGHT_GREEN": "GREEN",
    "TRAFFIC_LIGHT_YELLOW": "YELLOW",
    "TRAFFIC_LIGHT_RED": "RED",
    "LANE_STATE_GO": "GREEN",
    "LANE_STATE_ARROW_GO": "GREEN",
    "LANE_STATE_CAUTION": "YELLOW",
    "LANE_STATE_ARROW_CAUTION": "YELLOW",
    "LANE_STATE_FLASHING_CAUTION": "FLASHING_YELLOW",
    "LANE_STATE_STOP": "RED",
    "LANE_STATE_ARROW_STOP": "RED",
    "LANE_STATE_FLASHING_STOP": "RED",
    "TRAFFIC_LIGHT_UNKNOWN": "UNKNOWN",
    "LANE_STATE_UNKNOWN": "UNKNOWN",
}

# Every normative sub-rule in the fixed registry, in registry order. R1 is listed
# so its NOT_MEASURED status stays visible in the report instead of silently
# reading as a satisfied rule.
_NORMATIVE_COMPONENTS = (
    "collision",
    "rss",
    "rss_lateral",
    "ttc",
    "clearance",
    "offroad",
    # ``wrongway`` is gone: ADR-066 deregistered it, and the aggregation no
    # longer accepts either spelling.
    "wrong_carriageway",
    "solid_line",
    "dashed_line",
    "signal",
    "stop",
    "crosswalk",
    "vehicle_yield",
    # L3, ADR-068. Now a registered production sub-rule rather than a variant
    # this script computed for itself; it is inapplicable throughout the PG
    # panel, by provenance.
    "speed_limit",
    # L6, ADR-076. Named for what it measures rather than for its level: a
    # sub-rule sharing its level's name shadowed the atomic result in the
    # component map.
    "advance_shortfall",
)
_MACRO_COMPONENTS = (
    "collision_safety",
    "interaction_risk",
    "non_relaxable_compliance",
    "relaxable_lane_compliance",
    "progress_rate",
)

_R2_COMPONENTS = ("rss", "rss_lateral", "ttc", "clearance")
_R3_COMPONENTS = (
    "offroad",
    "wrongway",
    "wrong_carriageway",
    "solid_line",
    "dashed_line",
    "signal",
    "stop",
    "crosswalk",
    "vehicle_yield",
)

# Counterfactual rulebooks, evaluated exactly from the same per-step component
# costs as production — no re-simulation, no approximation. Each names the
# sub-rules it *drops* from a macro rule; `drop_latch` additionally zeroes the
# traffic-control steps whose cost came from the persistence latch rather than
# from the memoryless approach term.
#
# Only definition changes expressible as "ignore this component's cost" can be
# measured this way. Geometric-tolerance variants (a 0.3 m off-road band, solid
# line as normalized overlap, a centre-entry criterion for wrong carriageway)
# change the cost itself and need their own geometric pass; they are not here.
_VARIANTS: dict[str, dict[str, Any]] = {
    "production": {"drop_r2": frozenset(), "drop_r3": frozenset(), "drop_latch": False},
    # Longitudinal RSS demoted to diagnostic: it charges a state the environment
    # imposes on the ego rather than a choice the ego made.
    "no_rss": {"drop_r2": frozenset({"rss"}), "drop_r3": frozenset(), "drop_latch": False},
    # Additionally: dashed_line and wrongway removed, traffic-control latches
    # removed. This is the redesign minus the geometric tolerances.
    "proposed_no_tolerances": {
        "drop_r2": frozenset({"rss"}),
        "drop_r3": frozenset({"dashed_line", "wrongway"}),
        "drop_latch": True,
    },
    # Isolates the latch alone, to separate it from every other change.
    "latch_only": {"drop_r2": frozenset(), "drop_r3": frozenset(), "drop_latch": True},
}

# nuPlan's declared geometric tolerance for drivable-area compliance. It exists
# because the bounding box over-approximates the vehicle, which is a measurement
# artifact, not permissiveness about leaving the road.
OFFROAD_TOLERANCE_M = 0.3
# A solid lane marking is a painted stripe of real width, not the 1 cm numerical
# epsilon `evaluate_solid_line` currently buffers by to test intersection.
SOLID_LINE_HALF_WIDTH_M = 0.075
# Below this penetration the marking only grazes the footprint edge: the same
# bounding-box over-approximation as above, in the lateral direction. Penetration
# is dimensionless (1.0 = marking through the centroid, 0.0 = tangent to the
# edge), so for a ~1.85 m wide vehicle a tolerance of 0.3 corresponds to roughly
# the same 0.3 m of lateral slack the off-road band allows. Swept rather than
# chosen, because a wider detection buffer than production's 1 cm epsilon finds
# more contacts and the tolerance has to absorb them.
SOLID_LINE_PENETRATION_TOLERANCES = (0.1, 0.2, 0.3, 0.4)
SOLID_LINE_PROPOSED_TOLERANCE = 0.3
# Time headway is the standard surrogate safety measure, and unlike the RSS safe
# distance its thresholds come from naturalistic driving data rather than from a
# worst-case guarantee. Swept, not chosen: the expert decides which survive.
THW_THRESHOLDS_S = (0.3, 0.5, 0.7, 1.0, 1.5, 2.0)
# ``THW = gap / v_ego`` degrades as a criticality measure at crawl speed: queueing
# 1.5 m behind a slow leader reads as a 0.5 s headway although the leader's own
# speed makes the situation harmless. The minimum-ego-speed gate is therefore
# swept alongside the threshold, so one pass reports the expert's violation rate
# under every candidate low-speed exclusion instead of fixing one in advance.
THW_MIN_EGO_SPEED_GATES_MPS = (0.5, 2.0, 5.0, 8.0)
# nuPlan's speed-limit tolerance is 2.23 m/s (5 mph). Swept around it so the
# expert decides whether the rule is admissible at the published value or only
# at a looser one.
SPEED_LIMIT_TOLERANCES_MPS = (0.0, 1.0, 2.23, 4.47)
SPEED_LIMIT_PROPOSED_TOLERANCE_MPS = 2.23
# ScenarioNet writes 1000 km/h on PG lanes to mean "unlimited", and 0.0 on Waymo
# lanes whose limit the source did not record. Neither is a legal limit.
SPEED_LIMIT_SENTINEL_KMH = 999.0
# TTC is the one surviving R2 sub-rule with no tolerance of its own. Swept
# uniformly across actor classes so the values are directly comparable with
# nuPlan's single 0.95 s threshold; production keeps 0.8 s for vehicles and
# 1.0 s for VRU.
TTC_THRESHOLD_SWEEP_S = (0.4, 0.6, 0.8, 0.95, 1.2)
# The specified rulebook: nuPlan's uniform TTC bound, and the scalarization
# member the family grid selected.
FINAL_TTC_THRESHOLD_S = 0.95
FINAL_PRIORITY_BASE = 2.2
FINAL_SEVERITY = 0.0
FINAL_FLAT_TIE_BREAKER = 0.25

# Ego speeds under which a penalty cannot be blamed on the ego's own motion.
# The below-standstill comparison asks whether standing still beats the expert;
# a penalty the expert incurred while already at a crawl would not have been
# avoided by slowing further, so it does not explain that comparison. Swept
# rather than fixed: the split must not rest on one arbitrary threshold.
EGO_CRAWL_THRESHOLDS_MPS = (0.1, 0.5, 1.0, 2.0)

# ADR-070. At or below this ego speed the interaction sub-rules are inapplicable,
# because no ego action can avoid what another agent brings to a stationary
# vehicle -- the controlled-invariance test that rejected `rss`. Both published
# nuPlan thresholds are measured so the choice rests on the delta rather than on
# an assertion: 5e-02 is the at-fault collision threshold
# (`no_ego_at_fault_collisions._get_collision_type`, docstring "Threshold for 0
# speed due to noise"), 5e-03 the TTC one
# (`time_to_collision_within_bound`, which returns no value at all below it).
# 5e-02 is adopted because blame, not numerical guarding, is the principle being
# imported, and because 0.05 m/s = 0.18 km/h leaves no exploitable band: R4's
# margin there is ~0, so crawling under the gate buys immunity at zero progress.
AT_FAULT_GATE_THRESHOLDS_MPS = (0.005, 0.05)
FINAL_AT_FAULT_GATE_MPS = 0.05
# The interaction sub-rules, i.e. those whose cost is set by another agent's
# state. Position sub-rules are deliberately excluded: from a state stopped
# astride a lane marking an action that leaves it exists, so the region stays
# controlled-invariant and the charge is a normative disagreement, not a defect.
AT_FAULT_GATED_SUB_RULES = frozenset({"clearance", "rss_lateral", "ttc"})

# --- RULEBOOK-V5.1: the five-level liveness-aware hierarchy -------------------
# v5.0 placed every road rule above progress, so a policy that never moves was
# unbeatable. v5.1 splits that band: rules a competent driver may relax to
# complete a mission go *below* progress, the rest stay above it.
V51_L2_SUB_RULES = ("ttc", "clearance", "rss_lateral")
V51_L3_SUB_RULES = ("offroad", "signal", "stop", "crosswalk", "vehicle_yield", "speed_limit")
V51_L5_SUB_RULES = ("solid_line", "wrong_carriageway", "dashed_line")
# Fixed denominator. Recomputing it over the applicable sub-rules only would
# change the reward scale mid-episode as sub-rules become applicable, so an
# inapplicable sub-rule contributes 0 and is reported through its own mask.
V51_L5_DENOMINATOR = float(len(V51_L5_SUB_RULES))
# L4 is the *monotone* route advance, normalized by a FIXED reference distance
# rather than by each record's own route length.
#
# Dividing by `L_route` was tried first and measured: it caps every mission at
# `lambda4` regardless of length, while penalties stay per-step and grow with it,
# and -- worse -- the rank-preservation bound is then set by the SHORTEST route in
# the panel (19.96 m against a 171.98 m mean), so one degenerate record dictates
# `lambda4` for every other. Measured ceiling 32.7 against v5.0's 40.3.
#
# With a fixed reference the binding case becomes the longest single STEP instead
# of the shortest route, which is far less extreme, and a long mission earns
# proportionally more -- which is right, since a long mission plausibly needs more
# relaxation. `L_ref` itself is a UNIT, not a calibration: it cancels out of
# `ceiling = a * distance / longest step`, so changing it only rescales `lambda4`
# inversely. `v_ref * dt` is chosen so that `Delta q = 1` reads as "advanced one
# reference-speed step", making `lambda4` directly comparable to the priority
# weights. Route completion is NOT lost -- it is reported as an episodic metric,
# which is where it belongs.
V51_REFERENCE_ADVANCE_M = MISSION_PROGRESS_REFERENCE_SPEED_MPS * 0.1
# Clipping at one reference step is normative, not a fitted percentile: advancing
# further means exceeding the reference, which L3 `speed_limit` already prices,
# and paying it again as progress would credit the same conduct twice in opposite
# directions. A single declared speed is used rather than the per-lane posted
# limit because ADR-068's provenance gate leaves PG with no posted limit at all,
# so a per-lane clip would give R4 two different scales across a 50/50 mixture.
V51_DELTA_Q_MAX = 1.0
V51_T_REF_S = 1.0
V51_STEP_DT_S = 0.1
# `lambda4` is NOT a convention, and assuming it was is what the first
# measurement falsified: at `lambda4 = 1` under the old normalization the expert
# scored -8.49 against v5.0's +22.05, i.e. the mission was worth less than the
# violations incurred completing it. The SS5.4 tail bounds it, `lambda4 *
# DELTA_Q_MAX + lambda5 * (dt / T_REF) < a`, so at `a = 2.2` and `DELTA_Q_MAX = 1`
# roughly `lambda4 + 0.1 * eta < 2.2`.
# ADR-076. L6 `progress_rate`: the sixth level, BELOW relaxable lane compliance.
#
# `gamma = 1` (ADR-075) makes the mission channel telescope, so two trajectories
# reaching the same place tie at L4 however long they take. Measured on the
# frozen panel, that indifference band is wide: the median mission needs only
# 3.40 m/s to arrive inside its horizon while the agent is capped at 22.22 m/s,
# a slack of 6.5x. Inside the band, slower driving reduces exposure to the L2
# interaction sub-rules, so without L6 the optimum is to crawl at the slowest
# speed that still arrives.
#
# A per-step time cost folded into L4 was measured and REJECTED: it makes speed
# compete with progress *before* L5 is consulted, so every unit of preference for
# arriving sooner is also a unit that pays for an illegal shortcut. That converts
# O3 from a theorem back into a calibration, which is exactly what `gamma = 1`
# was bought to avoid.
#
# Placed at L6 instead, an illegal shortcut carries `c_L5 > 0` and therefore
# loses BEFORE the level is ever reached: in the lexicographic arms the weight on
# L6 is unconstrained by O3. Only the scalar arm, which sums everything, keeps
# the coupling.
#
# `1 - clip(Delta q, 0, 1)` rather than a flat time counter: summed over two
# completing trajectories it equals `T - Q`, so it ranks by duration exactly,
# while still giving a per-step gradient. Standing still and reversing both cost
# the maximum, 1.
V51_L6_SUB_RULES = ("progress_rate",)
V51_L6_DENOMINATOR = float(len(V51_L6_SUB_RULES))
# Bounded by O3 against the reference shortcut of RULEBOOK-V5.1 SS4.6 (saves 40
# steps, rides ONE marking for 30 at `c_L5 = 1/3`):
# `lambda6 * 40 * (dt / T_REF) < eta * (1/3) * (dt / T_REF) * 30` => `lambda6 < 0.25`.
V51_LAMBDA6_GRID = (0.0, 0.1, 0.2, 0.25)
V51_LAMBDA4_GRID = (0.5, 1.0, 1.5, 2.0, 2.15)
V51_ETA_GRID = (0.0, 0.5, 1.0, 2.0, 5.0)


def v51_weight_grid() -> tuple[tuple[str, float, float, float], ...]:
    """Admissible ``(lambda4, eta, lambda6)`` triples, labelled. Inadmissible ones are
    never priced, exactly as `family_grid` refuses non-rank-preserving members."""

    return tuple(
        (f"l4{lam:g}_eta{eta:g}_l6{lam6:g}", lam, eta, lam6)
        for lam in V51_LAMBDA4_GRID
        for eta in V51_ETA_GRID
        for lam6 in V51_LAMBDA6_GRID
        if v51_is_rank_preserving(
            FINAL_PRIORITY_BASE, FINAL_SEVERITY, FINAL_FLAT_TIE_BREAKER, lam, eta, lam6
        )
    )


def v51_standstill_return(variant: str, episode_steps: int) -> float:
    """The return standing still would have earned, for this variant.

    Before ADR-076 this was 0 for every rulebook: a stopped in-lane ego violates
    nothing and makes no progress. L6 charges a stopped ego the maximum
    ``c_L6 = 1`` on every step, so standing still now returns
    ``-lambda6 * (dt / T_REF) * T`` and the below-standstill diagnostic has to
    compare against that instead of against zero. Comparing against zero would
    silently overstate the improvement, since the baseline it is measured
    against got worse.
    """

    for label, _lam, _eta, lam6 in v51_weight_grid():
        if variant == f"v51_{label}":
            # A stopped ego advances nothing, so `c_L6 = 1` on every step. L4
            # itself is exactly 0, as it was before ADR-076.
            return -lam6 * (V51_STEP_DT_S / V51_T_REF_S) * float(episode_steps)
    return 0.0


def lane_speed_limits_mps(scenario: Mapping[str, Any]) -> dict[str, float]:
    """Posted limit per lane id, admitted only from a real-map provenance.

    ``speed_limit_kmh`` is written by every ScenarioNet producer, but only a
    real-map converter fills it from a posted limit. Waymo lanes carry the
    source datum ``speed_limit_mph`` alongside it (24.14 km/h = 15 mph, 40.23 =
    25, 72.42 = 45); the PG exporter has no posted limit to read and writes
    whichever default the lane constructor happened to hold:

    * ``1000`` for lanes built directly as ``StraightLane``/``CircularLane``
      (``abs_lane.py``: ``self.speed_limit = 1000  # should be set manually``),
      which is straights and the first block;
    * ``20`` for lanes built through ``create_pg_block_utils`` (its own
      ``speed_limit: float = 20`` default), which is curves and intersections.

    Neither is a posted limit, and the unit of the second is not even the one
    the key claims. MetaDrive's PG blocks document their limits in m/s
    (``ramp.py``: ``SPEED_LIMIT = 12  # 12 m/s ~= 40 km/h``; ``tollgate.py``:
    ``SPEED_LIMIT = 3  # m/s ~= 5 miles per hour``) while the exporter writes
    ``lane.speed_limit`` out verbatim under a ``_kmh`` key with no conversion
    (``node_road_network.py`` and ``edge_road_network.py``). Reading the PG
    value as km/h therefore understates it by 3.6x, and reading it as m/s still
    reports a constructor default as a traffic norm.

    So the admission test is provenance, not value: a lane speed limit is
    normative only where the record also carries the real-map ``speed_limit_mph``
    datum it was derived from. This keeps the existing rejections of the
    unrecorded ``0.0`` and of the ``>= 999`` sentinel, and additionally rejects
    every PG default. See ADR-068 and RULEBOOK-V5.0 SS11.
    """

    limits: dict[str, float] = {}
    for feature_id, feature in (scenario.get("map_features") or {}).items():
        if not isinstance(feature, Mapping):
            continue
        if not isinstance(feature.get("speed_limit_mph"), (int, float)):
            continue
        limit_kmh = feature.get("speed_limit_kmh")
        if not isinstance(limit_kmh, (int, float)):
            continue
        limit_kmh = float(limit_kmh)
        if (
            not math.isfinite(limit_kmh)
            or limit_kmh <= 0.0
            or limit_kmh >= SPEED_LIMIT_SENTINEL_KMH
        ):
            continue
        limits[str(feature_id)] = limit_kmh / 3.6
    return limits


def speed_limit_costs(*, ego_speed_mps: float, limit_mps: float) -> dict[float, float]:
    """Overspeed cost at each swept tolerance, normalized by the posted limit.

    The excess is divided by the limit so the cost is dimensionless and a given
    cost means the same relative overspeed on a 15 mph street and a 45 mph road.
    """

    costs: dict[float, float] = {}
    for tolerance in SPEED_LIMIT_TOLERANCES_MPS:
        excess = ego_speed_mps - (limit_mps + tolerance)
        costs[tolerance] = min(max(excess / limit_mps, 0.0), 1.0) if excess > 0.0 else 0.0
    return costs


def clearance_costs(
    *, ego_footprint, actors: Iterable[Any], ego_position_z: float, drivable_surface
) -> tuple[float, float]:
    """Reproduced production VRU clearance, and the same rule scoped to the roadway.

    ``evaluate_clearance`` charges any pedestrian or cyclist within 1 m of the ego
    footprint with no test of where that VRU is standing. A person waiting on the
    kerb of a narrow street is inside 1 m of every passing vehicle and is not in
    conflict with any of them, so the unscoped rule prices normal urban driving.
    The scoped variant keeps only VRUs whose centroid lies on the drivable
    surface, the same centre-entry criterion the wrong-carriageway variant uses.

    Both are returned from one pass so the reproduced value can be checked
    against production's own cost before the scoped one is believed.
    """

    unscoped = 0.0
    scoped = 0.0
    for actor in actors:
        threshold = CLEARANCE_THRESHOLDS_M.get(actor.actor_class)
        if threshold is None:
            continue
        if abs(float(actor.position_z) - ego_position_z) > VERTICAL_COMPATIBILITY_TOLERANCE_M:
            continue
        distance = ego_footprint.distance(actor.footprint)
        if not math.isfinite(distance):
            continue
        cost = max(0.0, 1.0 - distance / threshold)
        unscoped = max(unscoped, cost)
        on_road = (
            drivable_surface is not None
            and not drivable_surface.is_empty
            and drivable_surface.contains(actor.footprint.centroid)
        )
        if on_road:
            scoped = max(scoped, cost)
    return unscoped, scoped


def lateral_rss_costs(candidates: Iterable[Any]) -> tuple[float, float]:
    """Reproduced production lateral RSS, and the same rule without the actor term.

    ``d_safe^lat`` adds the other vehicle's worst-case inward displacement to the
    ego's. A neighbour drifting toward the ego at 0.5 m/s therefore raises the
    gap the ego is required to hold by half a metre, which inside a lane of fixed
    width the ego cannot always produce: the requirement is a function of another
    agent's action. The ego-only variant charges the ego for the displacement it
    brings itself, which is the controlled-invariant part of the same rule.
    """

    full = 0.0
    ego_only = 0.0
    for candidate in candidates:
        if not candidate.longitudinal_unsafe:
            continue
        gap = float(candidate.lateral_gap_m)
        ego_inward = float(candidate.ego_inward_speed_mps)
        safe_full = lateral_safe_distance_m(
            ego_inward_speed_mps=ego_inward,
            actor_inward_speed_mps=float(candidate.actor_inward_speed_mps),
        )
        # A receding neighbour lowers the requirement and that credit is kept;
        # an approaching one may not raise it. Taking the minimum makes the
        # variant a strict relaxation of production, which TEST-RSEC-017 pins.
        safe_scoped = min(
            safe_full,
            lateral_safe_distance_m(ego_inward_speed_mps=ego_inward, actor_inward_speed_mps=0.0),
        )
        for safe, sink in ((safe_full, "full"), (safe_scoped, "ego_only")):
            cost = 0.0 if safe == 0.0 else min(max(1.0 - gap / safe, 0.0), 1.0)
            if sink == "full":
                full = max(full, cost)
            else:
                ego_only = max(ego_only, cost)
    return full, ego_only


def ttc_costs_at(raw: Mapping[str, Any]) -> dict[float, float]:
    """Worst TTC cost at each swept threshold, from the component's own raw TTC values.

    ``evaluate_ttc`` writes ``ttc_s = -1.0`` where no collision time exists, which
    is not a small TTC but the absence of one.
    """

    worst = {threshold: 0.0 for threshold in TTC_THRESHOLD_SWEEP_S}
    for actor in raw.get("actors", ()):
        ttc_s = float(actor.get("ttc_s", -1.0))
        if ttc_s < 0.0:
            continue
        for threshold in TTC_THRESHOLD_SWEEP_S:
            worst[threshold] = max(worst[threshold], max(0.0, 1.0 - ttc_s / threshold))
    return worst


def graded_reward(margins: tuple[float, float, float, float], priority_base: float) -> float:
    """The same priority weights without SCAL-V1.1's satisfaction indicator.

    ``bounded_priority_weighted_rank`` charges ``base^e * ((satisfied - 1) + m)``,
    so any violation however small pays a full ``base^e`` before its severity is
    counted. Dropping that term leaves ``base^e * m``: still priority-weighted and
    still bounded, but continuous at the satisfaction boundary. Measured, not
    proposed - the production scalarization is a frozen user decision.
    """

    exponents = (3, 2, 1, 0)
    return float(
        sum(priority_base**exponent * margin for exponent, margin in zip(exponents, margins))
    )


# --- scalarization family -------------------------------------------------
#
# Veer et al. (ICRA 2023) Thm. 1 ranks whole *trajectories* with
# ``R = sum_i a^(N-i+1) step(rho_i) + (1/N) rho_i``. Two adaptations are forced
# by using it as a per-*step* reward that is then summed over an episode:
#
# 1. the step term must be shifted so a satisfied step scores 0 rather than
#    ``+a^e``. Unshifted, a stopped ego collects the full priority stack every
#    step forever, which is the standing-still degeneracy in its purest form.
#    For trajectory ranking the shift is a harmless constant; for a per-step sum
#    it is not, because episodes differ in length.
# 2. progress cannot stay a ``1/N`` tie-breaker. For Veer et al. progress breaks
#    ties among equally rule-compliant trajectories; here it is the task.
#
# The family below spans both conventions with one severity parameter:
#
#     r = sum_{k=1..3} a^(4-k) * ((step(m_k) - 1) + sigma * m_k) + lambda * m_4
#
# ``sigma = 1, a = 3, lambda = 1`` is SCAL-V1.1 exactly (asserted at runtime);
# ``sigma = 0`` puts severity outside the priority weight as the published form
# does. Lower ``a`` and lower ``sigma`` both shrink what one violation costs.
SCALARIZATION_FAMILY_BASES = (2.01, 2.2, 2.5, 3.0)
# ``severity`` scales the margin *inside* the priority weight (SCAL-V1.1's form);
# ``flat`` is Veer et al.'s own ``1/N`` tie-breaker, outside it. They buy the same
# thing - a gradient toward compliance within a violation - at very different
# prices in base.
SCALARIZATION_FAMILY_SEVERITIES = (0.0, 0.25, 0.5, 1.0)
SCALARIZATION_FAMILY_FLAT_TERMS = (0.0, 0.25)
SCALARIZATION_FAMILY_PROGRESS_WEIGHT = 1.0
# `scalarize_rulebook_margins` snaps near-zero margins to exactly 0 before doing
# anything else, so a margin of -9e-9 is a satisfied rule there. The family must
# canonicalize identically or it charges violations production never saw.
SCALARIZATION_FAMILY_TOLERANCE = ScalarizationConfig().numerical_tolerance


def family_weights(base: float) -> tuple[float, float, float]:
    """Priority weights for the three cost channels, highest priority first."""

    return (base**3, base**2, base)


def is_rank_preserving(base: float, severity: float, flat: float, progress_weight: float) -> bool:
    """Whether one step's ordering is lexicographic in the three cost channels.

    A violation at level ``k`` scores at best ``-w_k + lambda``; the same level
    satisfied scores at worst ``-(1+sigma) sum_{j>k} w_j - flat*(3-k) - lambda``,
    because a violated lower level ranges over ``[-w_j(1+sigma) - flat, -w_j)``.
    Dominance at every level is therefore exactly

        w_k > (1 + sigma) * sum_{j>k} w_j + flat * (3 - k) + 2 * lambda.

    At ``sigma = flat = 0, lambda = 1`` this reduces to ``a > 2``, which is Veer
    et al.'s own condition; at ``sigma = 1`` it forces ``a >= 2.92``, which is why
    SCAL-V1.1 had to re-derive the base as 3.
    """

    weights = family_weights(base)
    for index, weight in enumerate(weights):
        lower = weights[index + 1 :]
        bound = (1.0 + severity) * sum(lower) + flat * len(lower) + 2.0 * progress_weight
        if weight <= bound:
            return False
    return True


def family_reward(
    margins: tuple[float, float, float, float],
    *,
    base: float,
    severity: float,
    flat: float,
    progress_weight: float,
) -> float:
    """One step's reward under the family member ``(base, severity, flat, lambda)``."""

    canonical = tuple(
        0.0 if abs(value) <= SCALARIZATION_FAMILY_TOLERANCE else value for value in margins
    )
    total = 0.0
    for weight, margin in zip(family_weights(base), canonical[:3]):
        satisfied = 1.0 if margin == 0.0 else 0.0
        total += weight * ((satisfied - 1.0) + severity * margin) + flat * margin
    return total + progress_weight * canonical[3]


def v51_reward(
    *,
    l1: float,
    l2: float,
    l3: float,
    delta_q: float,
    l5: float,
    l6: float = 0.0,
    lam: float,
    eta: float,
    lam6: float = 0.0,
) -> float:
    """One step's reward under ``SCAL-V1.4`` (RULEBOOK-V5.1 §5.1).

    ``delta_q`` is the bare advance again: ADR-076 rejected folding a time cost
    into L4 and put it at L6 instead, so two trajectories reaching the same place
    tie at L4 **exactly** and O3 is decided by L5, as a theorem rather than as a
    calibration.

    L1-L3 are ``SCAL-V1.2`` verbatim, so their per-step dominance is inherited
    rather than re-argued. L4 and L5 form a *finite* exchange because no finite
    weight can make a continuous progress increment dominate a bounded cost as
    the increment tends to zero (§5.2). The alternative -- an indicator on L4 --
    would restore uniform dominance at the price of rewarding infinitesimal
    creep, i.e. it converts "stop forever" into "creep forever along the
    marking", which is the worse degeneracy.
    """

    costs = (l1, l2, l3)
    canonical = tuple(0.0 if abs(c) <= SCALARIZATION_FAMILY_TOLERANCE else -c for c in costs)
    total = 0.0
    for weight, margin in zip(family_weights(FINAL_PRIORITY_BASE), canonical):
        satisfied = 1.0 if margin == 0.0 else 0.0
        total += weight * ((satisfied - 1.0) + FINAL_SEVERITY * margin)
        total += FINAL_FLAT_TIE_BREAKER * margin
    total += lam * delta_q
    total -= eta * l5 * (V51_STEP_DT_S / V51_T_REF_S)
    return total - lam6 * l6 * (V51_STEP_DT_S / V51_T_REF_S)


def v51_is_rank_preserving(
    base: float,
    severity: float,
    flat: float,
    progress_weight: float,
    eta: float,
    lam6: float = 0.0,
    delta_q_max: float = V51_DELTA_Q_MAX,
) -> bool:
    """RULEBOOK-V5.1 §5.4: does L1-L3 dominance survive the new utility tail?

    Same derivation as ``is_rank_preserving``; only the tail differs. v5.0's
    progress channel swung over ``[-1, 1]`` and contributed ``2 * lambda``, while
    here the tail is ``lambda4 * DELTA_Q_MAX + eta * (dt / T_REF)`` -- roughly
    twenty times smaller, which is exactly why eta comes out unconstrained.
    """

    weights = family_weights(base)
    tail = (
        progress_weight * delta_q_max
        + eta * (V51_STEP_DT_S / V51_T_REF_S)
        + lam6 * (V51_STEP_DT_S / V51_T_REF_S)
    )
    for index, weight in enumerate(weights):
        lower = weights[index + 1 :]
        if weight <= (1.0 + severity) * sum(lower) + flat * len(lower) + tail:
            return False
    return True


def family_grid() -> tuple[tuple[str, float, float, float], ...]:
    """Every rank-preserving ``(base, severity, flat)`` triple, plus its label."""

    grid = []
    for base in SCALARIZATION_FAMILY_BASES:
        for severity in SCALARIZATION_FAMILY_SEVERITIES:
            for flat in SCALARIZATION_FAMILY_FLAT_TERMS:
                if is_rank_preserving(base, severity, flat, SCALARIZATION_FAMILY_PROGRESS_WEIGHT):
                    grid.append((f"a{base:g}_sev{severity:g}_flat{flat:g}", base, severity, flat))
    return tuple(grid)


# A distance envelope is not controlled-invariant: a cut-in puts the ego inside
# it with no action that restores the gap this step, which is why the expert
# fails RSS on 18.29% of applicable steps. RSS's own answer is the *proper
# response* - the ego must brake, not maintain a distance - but the literal
# proper response carries an unbounded per-actor latch and is not Markovian.
#
# The variant below is its memoryless projection: charge only when the gap is
# unsafe *and* the ego is not decelerating. From every state inside the envelope
# the action "brake" is available and zeroes the cost, so the region becomes
# controlled-invariant, and the rule prices a decision rather than a state.
RESPONSIVE_RSS_RESPONSE_TIMES_S = (0.3, 0.5, 1.0)
RESPONSIVE_RSS_BRAKE_THRESHOLDS_MPS2 = (0.0, 0.5, 1.0, 2.0)


def rss_safe_distance_at(
    *, ego_speed_mps: float, front_speed_mps: float, ego_brake_mps2: float, response_time_s: float
) -> float:
    """RSS longitudinal safe distance with the response time as a parameter.

    `components/rss.py` fixes rho at 1.0 s in a module constant, so the sweep
    needs its own copy of the formula; `TEST-RSEC-021` pins the two together at
    rho = 1.0 so this cannot drift from production.
    """

    ego_speed = max(0.0, ego_speed_mps)
    front_speed = max(0.0, front_speed_mps)
    response_speed = ego_speed + response_time_s * MAX_RESPONSE_ACCEL_MPS2
    reaction = ego_speed * response_time_s + 0.5 * MAX_RESPONSE_ACCEL_MPS2 * response_time_s**2
    ego_braking = response_speed**2 / (2.0 * ego_brake_mps2)
    front_braking = front_speed**2 / (2.0 * FRONT_MAX_BRAKE_MPS2)
    return max(0.0, reaction + ego_braking - front_braking)


def responsive_rss_costs(
    candidates: Iterable[Any], *, ego_brake_mps2: float, ego_accel_mps2: float
) -> dict[tuple[float, float], tuple[float, bool]]:
    """Unsafe-gap severity, discounted by how hard the ego is actually braking."""

    resolved: dict[tuple[float, float], tuple[float, bool]] = {}
    scoped = [
        candidate
        for candidate in candidates
        if float(candidate.ego_speed_mps) > RSS_STANDSTILL_SPEED_MPS
    ]
    for response_time in RESPONSIVE_RSS_RESPONSE_TIMES_S:
        worst = 0.0
        for candidate in scoped:
            safe = rss_safe_distance_at(
                ego_speed_mps=float(candidate.ego_speed_mps),
                front_speed_mps=float(candidate.front_speed_mps),
                ego_brake_mps2=ego_brake_mps2,
                response_time_s=response_time,
            )
            if safe <= 0.0:
                continue
            gap = max(0.0, float(candidate.gap_m))
            worst = max(worst, min(max(1.0 - gap / safe, 0.0), 1.0))
        for brake_threshold in RESPONSIVE_RSS_BRAKE_THRESHOLDS_MPS2:
            # Strict, so a coasting ego (exactly zero acceleration) never counts
            # as responding at the 0 threshold.
            responding = ego_accel_mps2 < -brake_threshold
            resolved[(response_time, brake_threshold)] = (
                0.0 if responding else worst,
                bool(scoped),
            )
    return resolved


def variant_offroad_cost(*, ego_footprint, drivable_surface) -> float:
    """Off-road area fraction outside the drivable surface widened by the tolerance."""

    tolerant = drivable_surface.buffer(OFFROAD_TOLERANCE_M)
    outside = ego_footprint.difference(tolerant).area
    if not math.isfinite(outside) or outside <= 0.0:
        return 0.0
    return min(max(outside / ego_footprint.area, 0.0), 1.0)


def variant_solid_line_costs(
    *, ego_footprint, solid_boundaries: Iterable[Any]
) -> dict[float, float]:
    """Graded solid-line occupancy: how deeply the marking sits in the footprint.

    Production charges a flat 1.0 for any contact with a 1 cm buffer, so a
    tangency and a vehicle centred on the line are indistinguishable. This reuses
    the repository's existing `dashed_lateral_penetration`, which is 1.0 when the
    marking passes through the footprint centroid and 0.0 when it is tangent to
    the edge, so solid and dashed markings finally grade on the same scale.
    """

    worst = {tolerance: 0.0 for tolerance in SOLID_LINE_PENETRATION_TOLERANCES}
    for boundary in solid_boundaries:
        geometry = getattr(boundary, "geometry", boundary)
        if geometry.is_empty or not geometry.is_valid:
            continue
        if not ego_footprint.intersects(geometry.buffer(SOLID_LINE_HALF_WIDTH_M)):
            continue
        penetration = dashed_lateral_penetration(ego_footprint, geometry)
        for tolerance in SOLID_LINE_PENETRATION_TOLERANCES:
            if penetration <= tolerance:
                continue
            scaled = (penetration - tolerance) / (1.0 - tolerance)
            worst[tolerance] = max(worst[tolerance], min(max(scaled, 0.0), 1.0))
    return worst


def variant_wrong_carriageway_cost(*, ego_footprint, aligned_surface, opposing_surface) -> float:
    """Opposing-carriageway occupancy gated on the ego *centre* having entered.

    Production charges any overlap down to a bumper corner, which is what a
    junction's lane-polygon seams produce. Requiring the centroid inside makes
    the rule fire on an actual carriageway entry, then grades it by footprint
    fraction exactly as production does.
    """

    if opposing_surface is None or opposing_surface.is_empty:
        return 0.0
    exclusive = (
        opposing_surface.difference(aligned_surface)
        if aligned_surface is not None and not aligned_surface.is_empty
        else opposing_surface
    )
    if exclusive.is_empty or not exclusive.contains(ego_footprint.centroid):
        return 0.0
    invaded = ego_footprint.intersection(exclusive).area
    if not math.isfinite(invaded) or invaded <= 0.0:
        return 0.0
    return min(max(invaded / ego_footprint.area, 0.0), 1.0)


def time_headway_costs(
    candidates: Iterable[Any],
) -> dict[tuple[float, float], tuple[float, bool]]:
    """Worst time-headway cost per (minimum-ego-speed gate, threshold) pair.

    ``THW = gap / v_ego`` is undefined at standstill, so a stopped ego is never
    charged: queueing behind a stopped vehicle is correct driving, not tailgating.
    Above standstill the measure still overstates criticality at crawl speed, so
    the gate is swept; the second element of each entry reports whether the step
    is applicable under that gate, which is what separates the conditional
    violation rate from the unconditional one.
    """

    resolved: dict[tuple[float, float], tuple[float, bool]] = {}
    for gate in THW_MIN_EGO_SPEED_GATES_MPS:
        worst = {threshold: 0.0 for threshold in THW_THRESHOLDS_S}
        applicable = False
        for candidate in candidates:
            ego_speed = float(candidate.ego_speed_mps)
            if ego_speed <= gate:
                continue
            applicable = True
            headway_s = max(0.0, float(candidate.gap_m)) / ego_speed
            for threshold in THW_THRESHOLDS_S:
                worst[threshold] = max(
                    worst[threshold], min(max(1.0 - headway_s / threshold, 0.0), 1.0)
                )
        for threshold in THW_THRESHOLDS_S:
            resolved[(gate, threshold)] = (worst[threshold], applicable)
    return resolved


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    if not sorted_values:
        return float("nan")
    index = min(len(sorted_values) - 1, max(0, int(round(quantile * (len(sorted_values) - 1)))))
    return float(sorted_values[index])


@dataclass
class CostSamples:
    """Streaming accumulator for one component's per-step cost distribution."""

    applicable_steps: int = 0
    violated_steps: int = 0
    positive_costs: list[float] = field(default_factory=list)

    def add(self, cost: float, *, applicable: bool) -> None:
        if not applicable:
            return
        self.applicable_steps += 1
        if cost > 0.0:
            self.violated_steps += 1
            self.positive_costs.append(float(cost))

    def merge(self, other: "CostSamples") -> None:
        self.applicable_steps += other.applicable_steps
        self.violated_steps += other.violated_steps
        self.positive_costs.extend(other.positive_costs)

    def summary(self, total_steps: int) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "applicable_steps": self.applicable_steps,
            "violated_steps": self.violated_steps,
            "violated_fraction_of_applicable": (
                round(self.violated_steps / self.applicable_steps, 6)
                if self.applicable_steps
                else None
            ),
            "violated_fraction_of_all_steps": (
                round(self.violated_steps / total_steps, 6) if total_steps else None
            ),
        }
        if self.positive_costs:
            values = sorted(self.positive_costs)
            payload["positive_cost_percentiles"] = {
                label: round(_percentile(values, quantile), 4)
                for label, quantile in (
                    ("p50", 0.50),
                    ("p75", 0.75),
                    ("p90", 0.90),
                    ("p99", 0.99),
                    ("max", 1.0),
                )
            }
            payload["positive_cost_mean"] = round(statistics.fmean(values), 4)
        return payload


def _sdc_z_origin(scenario: Mapping[str, Any]) -> float:
    """Mirror `waymo_static_adapter._sdc_z_origin` so both frames agree."""

    metadata = scenario.get("metadata", {})
    tracks = scenario.get("tracks", {})
    sdc_id = str(metadata.get("sdc_id", ""))
    if not isinstance(tracks, Mapping) or sdc_id not in tracks:
        return 0.0
    state = tracks[sdc_id].get("state")
    positions = state.get("position") if isinstance(state, Mapping) else None
    if positions is None:
        return 0.0
    values = np.asarray(positions, dtype=float)
    if values.ndim != 2 or values.shape[1] < 3 or len(values) == 0 or not np.isfinite(values[0, 2]):
        return 0.0
    return float(values[0, 2])


def _snapshot_at(
    track_id: str, track: Mapping[str, Any], step: int, *, z_origin_m: float
) -> ActorSnapshot | None:
    """Build one canonical actor snapshot from a logged track sample."""

    actor_class = _ACTOR_CLASSES.get(str(track.get("type", "")))
    if actor_class is None:
        return None
    state = track.get("state")
    if not isinstance(state, Mapping):
        return None
    valid = np.asarray(state.get("valid"), dtype=bool)
    if step >= len(valid) or not bool(valid[step]):
        return None
    position = np.asarray(state["position"], dtype=float)[step]
    heading = float(np.asarray(state["heading"], dtype=float)[step])
    velocity = np.asarray(state["velocity"], dtype=float)[step]
    length = float(np.asarray(state["length"], dtype=float)[step])
    width = float(np.asarray(state["width"], dtype=float)[step])
    if length <= 0.0 or width <= 0.0:
        return None
    values = (position[0], position[1], position[2], heading, velocity[0], velocity[1])
    if not all(math.isfinite(float(value)) for value in values):
        return None
    center = (float(position[0]), float(position[1]))
    return ActorSnapshot(
        actor_id=str(track_id),
        actor_class=actor_class,
        position_xy=center,
        position_z=float(position[2]) - z_origin_m,
        heading_rad=heading,
        velocity_xy=(float(velocity[0]), float(velocity[1])),
        footprint=oriented_bounding_box(
            center_xy=center, heading_rad=heading, length_m=length, width_m=width
        ),
        live_lane_id=None,
        configured_speed_cap_mps=(
            DEFAULT_SPEED_CAP_MPS if actor_class is ActorClass.VEHICLE else None
        ),
    )


def signal_states_at(scenario: Mapping[str, Any], step: int) -> dict[str, str]:
    """Read the logged signal colour of every traffic light at one step.

    The live path reads MetaDrive's light manager, which itself replays this same
    ``dynamic_map_states`` sequence, so offline and online agree by construction.
    Only the current step is read: no future state is consulted.
    """

    dynamic_states = scenario.get("dynamic_map_states", {})
    if not isinstance(dynamic_states, Mapping):
        return {}
    states: dict[str, str] = {}
    for physical_id, dynamic in sorted(dynamic_states.items(), key=lambda item: str(item[0])):
        if not isinstance(dynamic, Mapping) or dynamic.get("type") != "TRAFFIC_LIGHT":
            continue
        state = dynamic.get("state")
        sequence = state.get("object_state") if isinstance(state, Mapping) else None
        if not isinstance(sequence, (list, tuple, np.ndarray)) or step >= len(sequence):
            states[str(physical_id)] = "UNKNOWN"
            continue
        states[str(physical_id)] = _SIGNAL_STATE_MAP.get(str(sequence[step]), "UNKNOWN")
    return states


def mission_snapshot_at(
    *, route, ego: ActorSnapshot, step: int, mission_hash: str, previous_s_m: float | None
) -> tuple[MissionSnapshot, float]:
    """Build the minimal route-derived mission context R4 needs.

    R4 reads only ``s_m`` and the step index (``components/progress.py``), so the
    replay supplies the exact canonical route station and derives the remaining
    fields consistently rather than instantiating the full mission runtime.
    """

    projection = route.project(
        ego.position_xy, position_z=ego.position_z, previous_s_m=previous_s_m
    )
    s_m = float(projection.s_m)
    total_m = float(route.length_m)
    remaining = max(0.0, total_m - s_m)
    completion = 0.0 if total_m <= 0.0 else min(max(s_m / total_m, 0.0), 1.0)
    snapshot = MissionSnapshot(
        mission_hash=mission_hash,
        step_index=step,
        pending_gate_index=0,
        remaining_distance_m=remaining,
        route_completion=completion,
        reachable=True,
        mission_success=False,
        mission_unreachable=False,
        s_m=s_m,
    )
    return snapshot, s_m


def comfort_test_a(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the repository's own admissibility test to the nuPlan comfort bounds.

    `RULEBOOK-V5.0` §2.1: a rule that carries a satisfaction indicator must be
    satisfiable by a competent driver, so the expert's violation rate is what
    decides whether a threshold charges for driving badly or merely for driving.
    Every rulebook sub-rule was falsified this way; the comfort bounds were
    adopted on nuPlan's published authority alone, and this closes that gap.

    Reported per channel: the share of expert episodes breaching the bound, and
    the distribution of the channel itself, so a failure can be read as "the
    bound is slightly tight" or "the instrument is measuring something else".
    """

    defined = [summary for summary in summaries if isinstance(summary, Mapping)]
    verdicts = [
        bool(summary["is_comfortable"])
        for summary in defined
        if summary.get("is_comfortable") is not None
    ]
    channels: dict[str, Any] = {}
    for name in COMFORT_STATISTICS:
        values = sorted(
            float(summary[name])
            for summary in defined
            if summary.get(name) is not None and math.isfinite(float(summary[name]))
        )
        if not values:
            channels[name] = {"episodes": 0}
            continue
        bound = float(getattr(NUPLAN_COMFORT_BOUNDS, name))
        breaching = (
            sum(1 for value in values if value < bound)
            if name == "min_lon_accel"
            else sum(1 for value in values if value > bound)
        )
        channels[name] = {
            "episodes": len(values),
            "bound": bound,
            "expert_violation_rate": breaching / len(values),
            "p50": _percentile(values, 0.50),
            "p95": _percentile(values, 0.95),
            "p99": _percentile(values, 0.99),
            "worst": values[0] if name == "min_lon_accel" else values[-1],
        }
    return {
        "episodes_with_verdict": len(verdicts),
        "episodes_without_verdict": len(defined) - len(verdicts),
        "expert_comfort_rate": (sum(verdicts) / len(verdicts)) if verdicts else None,
        "channels": channels,
    }


@dataclass
class Measurement:
    """Aggregate over every replayed scenario."""

    scenarios_measured: int = 0
    scenarios_skipped: Counter = field(default_factory=Counter)
    total_steps: int = 0
    components: dict[str, CostSamples] = field(default_factory=dict)
    macros: dict[str, CostSamples] = field(default_factory=dict)
    progress_margins: list[float] = field(default_factory=list)
    episode_returns: list[float] = field(default_factory=list)
    episode_steps: list[int] = field(default_factory=list)
    channel_returns: dict[str, list[float]] = field(default_factory=dict)
    # Separates a traffic-control cost produced by the memoryless approach term
    # from one produced purely by the persistence latch, which is the quantity
    # that decides whether removing the latch changes the expert's bill.
    latch_attribution: Counter = field(default_factory=Counter)
    # Per-episode scalarized return under each counterfactual rulebook, computed
    # in the same pass from the same per-step costs, so the comparison against
    # production is exact rather than extrapolated.
    variant_returns: dict[str, list[float]] = field(default_factory=dict)
    variant_r2_violated: Counter = field(default_factory=Counter)
    variant_r3_violated: Counter = field(default_factory=Counter)
    # Geometric redefinitions of the three suspected-artifact R3 sub-rules, and
    # the time-headway sweep. Unlike the counterfactuals above these change the
    # cost itself, so they are recomputed from geometry rather than reweighted.
    variant_components: dict[str, CostSamples] = field(default_factory=dict)
    thw_samples: dict[tuple[float, float], CostSamples] = field(default_factory=dict)
    # Which sub-rule the proposed rulebook's residual penalty is owed to, in
    # reward units. The dominant-when-negative counter answers whether the
    # remaining below-standstill episodes share one cause worth fixing or are an
    # irreducible spread; the total answers how much each sub-rule costs overall.
    error_samples: list[str] = field(default_factory=list)
    proposed_blame_total: Counter = field(default_factory=Counter)
    proposed_blame_dominant_when_negative: Counter = field(default_factory=Counter)
    # The same accounting for the rulebook actually specified, which differs
    # from `proposed` in four sub-rules. Attributing the specified rulebook's
    # residual from the proposed rulebook's table would be an extrapolation
    # across those four, so it is measured separately here.
    final_blame_total: Counter = field(default_factory=Counter)
    final_blame_dominant_when_negative: Counter = field(default_factory=Counter)
    # Penalty mass in the episodes the specified rulebook leaves below
    # standstill, split by whether the ego was already at a crawl when it was
    # charged. Mass charged at a crawl was not avoidable by slowing down, so it
    # cannot be what makes standing still preferable; mass charged while moving
    # was. Keyed by crawl threshold.
    final_negative_mass_total: dict[float, float] = field(default_factory=dict)
    final_negative_mass_at_crawl: dict[float, float] = field(default_factory=dict)
    final_negative_episodes: int = 0
    # The same mass, crossed with the sub-rule that charged it. The aggregate
    # split says how much of the tail was charged to an already-stopped ego; it
    # cannot say whether that is the expert stopping in a bad place or the
    # rulebook charging a stopped ego for something it cannot avoid. Those have
    # opposite remedies, and only the sub-rule identity separates them: a
    # position rule charges where the ego chose to stop, an interaction rule
    # charges what another agent did to a stationary ego -- which is the
    # controlled-invariance failure that rejected `rss` in SS4.7.
    final_blame_mass_in_negative: Counter = field(default_factory=Counter)
    final_blame_mass_at_crawl: Counter = field(default_factory=Counter)
    # RULEBOOK-V5.1. The channel counters are what shows the restructure reached
    # only what it was scoped to reach: L3 must be unchanged by `eta`, and L5 is
    # new mass rather than moved mass only if its violation rate matches the
    # relaxable sub-rules' rate under v5.0. `telescoping_max_error` is
    # `AC-RB5.1-07`: if the monotone construction is right, the undiscounted
    # per-episode sum of `delta_q` equals `q_T - q_0` exactly.
    v51_channel_steps: int = 0
    v51_channel_violated: Counter = field(default_factory=Counter)
    v51_delta_q_total: float = 0.0
    v51_delta_q_clip_binding: int = 0
    v51_delta_q_max_observed: float = 0.0
    # Where the missing completion sits. The route is map-matched from the SDC
    # track, so it cannot extend past the ego in lane count; if the expert still
    # fails to reach 1.0, the loss must be the unused head of the first lane and
    # tail of the last, which the lane sequence includes in full.
    v51_delta_s_max_observed: float = 0.0
    v51_route_lengths_m: list[float] = field(default_factory=list)
    v51_delta_s_samples: list[float] = field(default_factory=list)
    # REQ-RB5.1-OBS-01. `sigma - s` is how far the ego sits behind its own route
    # high-water mark: it is exactly the state that decides whether forward
    # motion earns L4 reward, and it appears in no observation field. Counting
    # where it is non-zero bounds how much hidden state the requirement is
    # actually about, rather than settling it on the definition alone.
    v51_behind_peak_steps: int = 0
    v51_behind_peak_steps_1m: int = 0
    v51_behind_peak_max_m: float = 0.0
    v51_behind_peak_episodes: int = 0
    v51_q_start: list[float] = field(default_factory=list)
    v51_q_end: list[float] = field(default_factory=list)
    v51_telescoping_max_error: float = 0.0
    # `T-RB51-12`. Until `RB51`/`M3`-`M5` this script *reimplemented* the
    # redefined sub-rules as variants, because production still charged the old
    # ones. Production now implements them natively, so the two must agree to
    # numerical tolerance on every step -- and if they do, the specification's
    # published numbers become a claim about production rather than about a
    # script. The divergence is accumulated on every run rather than gated behind
    # a flag: a check that has to be remembered is a check that will be forgotten,
    # and a silent drift here would invalidate every figure in §5.5.
    oracle_divergence: dict[str, float] = field(default_factory=dict)
    # `M8b`, closing `C3` and answering `C2`. `control_line_diagnostics` is static
    # per scenario and has existed since ADR-051; what was missing was any
    # cross-episode aggregation of it, so the 209 records where `SIGNAL` is never
    # selectable stayed an undecomposed headline. These counters separate the
    # controls lost at adapter construction from those excluded by route
    # reachability, which is the split that says how much of `C2` is a defect at
    # all.
    control_line_totals: Counter[str] = field(default_factory=Counter)
    scenarios_with_controls: int = 0
    scenarios_with_no_selectable_signal: int = 0
    # `M8c`. Test A can only reject, so a rule that is never *applicable* passes
    # it perfectly. Per-step applicability is already reported; this is the
    # per-scenario view -- on how many records a sub-rule never applies at all --
    # which is the granularity `C2` needs and the one the method is blind to.
    scenarios_where_applicable: Counter[str] = field(default_factory=Counter)
    # EP-COMFORT-DIAG Test A for ride comfort. Every rulebook sub-rule is
    # falsified against the logged expert before it is trusted; the nuPlan
    # comfort bounds entered this repository unfalsified, on their published
    # authority alone. One `ComfortEpisodeAccumulator` summary per replayed
    # record, so the same question can be asked of them: does a competent human
    # driver satisfy these bounds in *this* simulator's state representation?
    comfort_summaries: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        for name in (
            "offroad_tolerant",
            "wrong_carriageway_entry",
            "clearance_on_road",
            "rss_lateral_ego_only",
        ):
            self.variant_components.setdefault(name, CostSamples())
        for tolerance in SPEED_LIMIT_TOLERANCES_MPS:
            self.variant_components.setdefault(f"speed_limit_tol{tolerance:g}", CostSamples())
        for threshold in TTC_THRESHOLD_SWEEP_S:
            self.variant_components.setdefault(f"ttc_thr{threshold:g}s", CostSamples())
        for response_time in RESPONSIVE_RSS_RESPONSE_TIMES_S:
            for brake in RESPONSIVE_RSS_BRAKE_THRESHOLDS_MPS2:
                self.variant_components.setdefault(
                    f"rss_responsive_rho{response_time:g}_brake{brake:g}", CostSamples()
                )
        for tolerance in SOLID_LINE_PENETRATION_TOLERANCES:
            self.variant_components.setdefault(f"solid_line_graded_tol{tolerance:g}", CostSamples())
        for gate in THW_MIN_EGO_SPEED_GATES_MPS:
            for threshold in THW_THRESHOLDS_S:
                self.thw_samples.setdefault((gate, threshold), CostSamples())
        for crawl in EGO_CRAWL_THRESHOLDS_MPS:
            self.final_negative_mass_total.setdefault(crawl, 0.0)
            self.final_negative_mass_at_crawl.setdefault(crawl, 0.0)
        for variant in all_variant_names():
            self.variant_returns.setdefault(variant, [])
        for name in _NORMATIVE_COMPONENTS:
            self.components.setdefault(name, CostSamples())
        for name in _MACRO_COMPONENTS:
            self.macros.setdefault(name, CostSamples())
        # `RULEBOOK-V5.1` §3's six levels, replacing v4.7's `r1..r4`. The names
        # are an output contract: they reach the report, so renaming them is the
        # same class of change `DEC-RB51-005` approved for `MacroRule`.
        for channel in ("l1", "l2", "l3", "l4", "l5", "l6"):
            self.channel_returns.setdefault(channel, [])

    def observe_oracle(self, name: str, production: float, variant: float) -> None:
        """Record the worst disagreement seen between production and the variant."""

        # `setdefault` first: an absent key and a zero divergence must not read
        # the same way. An empty report would say "perfect agreement" and "never
        # compared" with the same evidence, which is the failure mode this whole
        # check exists to catch.
        divergence = abs(float(production) - float(variant))
        self.oracle_divergence[name] = max(divergence, self.oracle_divergence.setdefault(name, 0.0))

    def merge(self, other: "Measurement") -> None:
        """Absorb one worker's partial accumulator; every field is additive.

        Percentiles come from the merged raw sample lists at report time, never
        from per-worker percentiles, so the parallel result is identical to the
        sequential one regardless of completion order.
        """

        self.scenarios_measured += other.scenarios_measured
        self.scenarios_skipped.update(other.scenarios_skipped)
        self.control_line_totals.update(other.control_line_totals)
        self.scenarios_with_controls += other.scenarios_with_controls
        self.scenarios_with_no_selectable_signal += other.scenarios_with_no_selectable_signal
        self.scenarios_where_applicable.update(other.scenarios_where_applicable)
        for name, divergence in other.oracle_divergence.items():
            # Same `setdefault`-first rule as `observe_oracle`: the key must
            # survive the merge even when every worker saw exact agreement.
            self.oracle_divergence[name] = max(
                divergence, self.oracle_divergence.setdefault(name, 0.0)
            )
        self.total_steps += other.total_steps
        for name, samples in other.components.items():
            self.components.setdefault(name, CostSamples()).merge(samples)
        for name, samples in other.macros.items():
            self.macros.setdefault(name, CostSamples()).merge(samples)
        self.progress_margins.extend(other.progress_margins)
        self.episode_returns.extend(other.episode_returns)
        self.episode_steps.extend(other.episode_steps)
        for channel, values in other.channel_returns.items():
            self.channel_returns.setdefault(channel, []).extend(values)
        self.latch_attribution.update(other.latch_attribution)
        for variant, values in other.variant_returns.items():
            self.variant_returns.setdefault(variant, []).extend(values)
        self.variant_r2_violated.update(other.variant_r2_violated)
        self.variant_r3_violated.update(other.variant_r3_violated)
        for name, samples in other.variant_components.items():
            self.variant_components.setdefault(name, CostSamples()).merge(samples)
        for key, samples in other.thw_samples.items():
            self.thw_samples.setdefault(key, CostSamples()).merge(samples)
        self.error_samples.extend(
            other.error_samples[: _ERROR_SAMPLE_LIMIT - len(self.error_samples)]
        )
        self.proposed_blame_total.update(other.proposed_blame_total)
        self.proposed_blame_dominant_when_negative.update(
            other.proposed_blame_dominant_when_negative
        )
        self.final_blame_total.update(other.final_blame_total)
        self.final_blame_dominant_when_negative.update(other.final_blame_dominant_when_negative)
        self.v51_channel_steps += other.v51_channel_steps
        self.v51_channel_violated.update(other.v51_channel_violated)
        self.v51_delta_q_total += other.v51_delta_q_total
        self.v51_delta_q_clip_binding += other.v51_delta_q_clip_binding
        self.v51_delta_q_max_observed = max(
            self.v51_delta_q_max_observed, other.v51_delta_q_max_observed
        )
        self.v51_delta_s_max_observed = max(
            self.v51_delta_s_max_observed, other.v51_delta_s_max_observed
        )
        self.v51_route_lengths_m.extend(other.v51_route_lengths_m)
        self.v51_delta_s_samples.extend(other.v51_delta_s_samples)
        self.v51_behind_peak_steps += other.v51_behind_peak_steps
        self.v51_behind_peak_steps_1m += other.v51_behind_peak_steps_1m
        self.v51_behind_peak_max_m = max(self.v51_behind_peak_max_m, other.v51_behind_peak_max_m)
        self.v51_behind_peak_episodes += other.v51_behind_peak_episodes
        self.v51_q_start.extend(other.v51_q_start)
        self.v51_q_end.extend(other.v51_q_end)
        self.v51_telescoping_max_error = max(
            self.v51_telescoping_max_error, other.v51_telescoping_max_error
        )
        self.final_negative_episodes += other.final_negative_episodes
        for crawl, mass in other.final_negative_mass_total.items():
            self.final_negative_mass_total[crawl] = (
                self.final_negative_mass_total.get(crawl, 0.0) + mass
            )
        for crawl, mass in other.final_negative_mass_at_crawl.items():
            self.final_negative_mass_at_crawl[crawl] = (
                self.final_negative_mass_at_crawl.get(crawl, 0.0) + mass
            )
        self.final_blame_mass_in_negative.update(other.final_blame_mass_in_negative)
        self.final_blame_mass_at_crawl.update(other.final_blame_mass_at_crawl)
        self.comfort_summaries.extend(other.comfort_summaries)

    def summary(self) -> dict[str, Any]:
        returns = sorted(self.episode_returns)
        steps = sorted(self.episode_steps)
        margins = sorted(self.progress_margins)
        channels = {
            channel: sorted(values) for channel, values in sorted(self.channel_returns.items())
        }
        return {
            "scenarios_measured": self.scenarios_measured,
            "scenarios_skipped": dict(self.scenarios_skipped),
            # EP-COMFORT-DIAG: Test A applied to the nuPlan comfort bounds.
            "comfort_test_a": comfort_test_a(self.comfort_summaries),
            "error_samples": list(self.error_samples),
            "measured_steps": self.total_steps,
            "r1_status": "NOT_MEASURED: offline replay has no physics contact records",
            # `T-RB51-12`. Every entry must be 0 (or below float tolerance): a
            # non-zero value means production and the variant this script used to
            # derive §5.5's numbers disagree, and every figure downstream of it is
            # then a claim about the script rather than about production. Placed
            # near the top of the report because it conditions everything below.
            "control_line_coverage": {
                "scenarios_with_controls": self.scenarios_with_controls,
                "scenarios_with_no_selectable_signal": self.scenarios_with_no_selectable_signal,
                **{name: int(value) for name, value in sorted(self.control_line_totals.items())},
            },
            "scenarios_where_sub_rule_applies": {
                name: int(self.scenarios_where_applicable.get(name, 0))
                for name in sorted(_NORMATIVE_COMPONENTS)
            },
            "oracle_max_divergence": {
                name: round(value, 12) for name, value in sorted(self.oracle_divergence.items())
            },
            "expert_episode_return": {
                "p10": round(_percentile(returns, 0.10), 2) if returns else None,
                "p50": round(_percentile(returns, 0.50), 2) if returns else None,
                "p90": round(_percentile(returns, 0.90), 2) if returns else None,
                "mean": round(statistics.fmean(returns), 2) if returns else None,
                "fraction_below_standstill": (
                    round(sum(1 for value in returns if value < 0.0) / len(returns), 4)
                    if returns
                    else None
                ),
            },
            "expert_episode_steps": {
                "p50": _percentile(steps, 0.50) if steps else None,
            },
            "expert_episode_return_by_channel": {
                channel: {
                    "p50": round(_percentile(values, 0.50), 2) if values else None,
                    "mean": round(statistics.fmean(values), 2) if values else None,
                }
                for channel, values in channels.items()
            },
            "progress_margin": {
                "p50": round(_percentile(margins, 0.50), 4) if margins else None,
                "mean": round(statistics.fmean(margins), 4) if margins else None,
            },
            "sub_rules": {
                name: self.components[name].summary(self.total_steps)
                for name in _NORMATIVE_COMPONENTS
            },
            "macro_rules": {
                name: self.macros[name].summary(self.total_steps) for name in _MACRO_COMPONENTS
            },
            "traffic_control_latch_attribution": dict(sorted(self.latch_attribution.items())),
            "geometric_variants": {
                name: self.variant_components[name].summary(self.total_steps)
                for name in sorted(self.variant_components)
            },
            "time_headway_sweep": {
                f"gate_{gate:g}mps": {
                    f"thw_{threshold:g}s": self.thw_samples[(gate, threshold)].summary(
                        self.total_steps
                    )
                    for threshold in THW_THRESHOLDS_S
                }
                for gate in THW_MIN_EGO_SPEED_GATES_MPS
            },
            "proposed_rulebook_blame": {
                "reward_units_by_sub_rule": {
                    name: round(mass, 1)
                    for name, mass in sorted(
                        self.proposed_blame_total.items(), key=lambda item: -item[1]
                    )
                },
                "dominant_sub_rule_when_episode_below_standstill": dict(
                    sorted(
                        self.proposed_blame_dominant_when_negative.items(),
                        key=lambda item: -item[1],
                    )
                ),
            },
            "rulebook_v51": {
                "channel_steps": self.v51_channel_steps,
                # L3 must be identical across every `eta`, because `eta` reaches
                # only L5; if it is not, the restructure leaked.
                "channel_violation_rate_pct": {
                    channel: (
                        round(
                            100.0 * self.v51_channel_violated[channel] / self.v51_channel_steps, 4
                        )
                        if self.v51_channel_steps
                        else None
                    )
                    for channel in ("l2", "l3", "l5", "l6")
                },
                "mean_delta_q_per_step": (
                    round(self.v51_delta_q_total / self.v51_channel_steps, 6)
                    if self.v51_channel_steps
                    else None
                ),
                # AC-RB5.1-05 and AC-RB5.1-07: both must be exactly zero.
                "delta_q_clip_binding_steps": self.v51_delta_q_clip_binding,
                # Sets how tight DELTA_Q_MAX can honestly be, and therefore how
                # much progress weight the SS5.4 tail can afford.
                "delta_q_max_observed": round(self.v51_delta_q_max_observed, 6),
                "delta_s_max_observed_m": round(self.v51_delta_s_max_observed, 3),
                "reference_advance_m": round(V51_REFERENCE_ADVANCE_M, 4),
                # REQ-RB5.1-OBS-01: the hidden state's actual size.
                "behind_peak": {
                    "steps": self.v51_behind_peak_steps,
                    "steps_pct": (
                        round(100.0 * self.v51_behind_peak_steps / self.v51_channel_steps, 4)
                        if self.v51_channel_steps
                        else None
                    ),
                    "steps_beyond_1m": self.v51_behind_peak_steps_1m,
                    "max_m": round(self.v51_behind_peak_max_m, 3),
                    "episodes": self.v51_behind_peak_episodes,
                },
                # Is the largest step real motion or an arc-length projection
                # jump? If p99.9 sits far below the max, the max is an artifact
                # and clipping removes an error rather than costing fidelity.
                "delta_s_percentiles_m": (
                    {
                        f"p{q}": round(
                            sorted(self.v51_delta_s_samples)[
                                min(
                                    len(self.v51_delta_s_samples) - 1,
                                    int(len(self.v51_delta_s_samples) * q / 100.0),
                                )
                            ],
                            3,
                        )
                        for q in (50, 90, 99, 99.9)
                    }
                    if self.v51_delta_s_samples
                    else None
                ),
                # The global bound the SS5.4 condition needs is the cap divided by
                # the SHORTEST route in the panel: that record is where one step
                # buys the largest fraction of a mission.
                "route_length_min_m": (
                    round(min(self.v51_route_lengths_m), 2) if self.v51_route_lengths_m else None
                ),
                "route_length_mean_m": (
                    round(sum(self.v51_route_lengths_m) / len(self.v51_route_lengths_m), 2)
                    if self.v51_route_lengths_m
                    else None
                ),
                "admissible_weight_triples": [label for label, _, _, _ in v51_weight_grid()],
                "q_start_mean": (
                    round(sum(self.v51_q_start) / len(self.v51_q_start), 4)
                    if self.v51_q_start
                    else None
                ),
                "q_end_mean": (
                    round(sum(self.v51_q_end) / len(self.v51_q_end), 4) if self.v51_q_end else None
                ),
                "telescoping_max_error": round(self.v51_telescoping_max_error, 12),
            },
            "final_rulebook_blame": {
                "reward_units_by_sub_rule": {
                    name: round(mass, 1)
                    for name, mass in sorted(
                        self.final_blame_total.items(), key=lambda item: -item[1]
                    )
                },
                "dominant_sub_rule_when_episode_below_standstill": dict(
                    sorted(
                        self.final_blame_dominant_when_negative.items(),
                        key=lambda item: -item[1],
                    )
                ),
                "episodes_below_standstill": self.final_negative_episodes,
                # Of the penalty charged inside those episodes, the share the
                # expert incurred while already at or below a crawl. That share
                # was not avoidable by driving more slowly, so it cannot be what
                # makes standing still preferable; the complement was avoidable
                # and is the part that genuinely favours standstill.
                "penalty_mass_not_avoidable_by_slowing": {
                    f"ego_speed_le_{crawl:g}mps": {
                        "reward_units": round(self.final_negative_mass_at_crawl[crawl], 1),
                        "fraction_of_penalty_mass": (
                            round(
                                self.final_negative_mass_at_crawl[crawl]
                                / self.final_negative_mass_total[crawl],
                                4,
                            )
                            if self.final_negative_mass_total[crawl] > 0.0
                            else None
                        ),
                    }
                    for crawl in EGO_CRAWL_THRESHOLDS_MPS
                },
                # Which sub-rule charged the mass a stopped ego was billed for.
                # A position rule (offroad, solid_line, dashed_line) charges
                # where the ego chose to stop and stays controlled-invariant: a
                # legal place to stand exists. An interaction rule (clearance,
                # rss_lateral, ttc) charges what another agent did to a
                # stationary ego, which no ego action can avoid -- the same
                # controlled-invariance failure that rejected `rss`. The two
                # findings have opposite remedies, so they are reported apart.
                "penalty_mass_by_sub_rule_at_crawl": {
                    name: {
                        "reward_units_in_negative_episodes": round(total, 1),
                        **{
                            f"at_ego_speed_le_{crawl:g}mps": round(
                                self.final_blame_mass_at_crawl.get((name, crawl), 0.0), 1
                            )
                            for crawl in EGO_CRAWL_THRESHOLDS_MPS
                        },
                        "share_at_lowest_crawl": (
                            round(
                                self.final_blame_mass_at_crawl.get(
                                    (name, EGO_CRAWL_THRESHOLDS_MPS[0]), 0.0
                                )
                                / total,
                                4,
                            )
                            if total > 0.0
                            else None
                        ),
                    }
                    for name, total in sorted(
                        self.final_blame_mass_in_negative.items(), key=lambda item: -item[1]
                    )
                },
                "total_penalty_mass_in_those_episodes": round(
                    self.final_negative_mass_total[EGO_CRAWL_THRESHOLDS_MPS[0]], 1
                ),
            },
            "counterfactual_rulebooks": {
                variant: {
                    # The below-standstill tail is reported down to p1: whether the
                    # residual negative episodes sit just under zero or far below it
                    # decides whether they are a remaining defect or ordinary spread.
                    **{
                        f"episode_return_p{int(quantile * 100)}": (
                            round(_percentile(sorted(values), quantile), 2) if values else None
                        )
                        for quantile in (0.01, 0.05, 0.10, 0.25, 0.50, 0.90)
                    },
                    "episode_return_mean": (round(statistics.fmean(values), 2) if values else None),
                    "fraction_below_standstill": (
                        round(
                            sum(
                                1
                                for value, steps in zip(values, self.episode_steps)
                                if value < v51_standstill_return(variant, steps)
                            )
                            / len(values),
                            4,
                        )
                        if values
                        else None
                    ),
                    "r2_violated_fraction_of_all_steps": (
                        round(self.variant_r2_violated[variant] / self.total_steps, 6)
                        if self.total_steps
                        else None
                    ),
                    "r3_violated_fraction_of_all_steps": (
                        round(self.variant_r3_violated[variant] / self.total_steps, 6)
                        if self.total_steps
                        else None
                    ),
                }
                for variant, values in sorted(self.variant_returns.items())
            },
        }


def _is_latch_cost(name: str, component) -> bool:
    """True when a traffic-control cost came from the persistence latch.

    ``crosswalk`` and ``vehicle_yield`` both compute ``cost = 1.0 if
    active_latch_for_zone else approach_cost``. A positive cost whose
    ``before_gate`` diagnostic is False is therefore the latch branch: the ego is
    inside the zone being charged for an entry decision already made, and nothing
    it does this step changes the cost.
    """

    if name not in {"crosswalk", "vehicle_yield"} or component.cost <= 0.0:
        return False
    return not bool(component.diagnostics.get("before_gate", False))


def macro_cost(
    names: Iterable[str], components: Mapping[str, Any], *, drop: frozenset[str], drop_latch: bool
) -> tuple[float, bool]:
    """Recompute one macro rule by max over applicable sub-rules, as production does."""

    worst, applicable, _ = macro_cost_with_blame(
        names, components, drop=drop, drop_latch=drop_latch
    )
    return worst, applicable


def macro_cost_with_blame(
    names: Iterable[str], components: Mapping[str, Any], *, drop: frozenset[str], drop_latch: bool
) -> tuple[float, bool, str | None]:
    """As :func:`macro_cost`, also naming the sub-rule that attained the maximum.

    Under max aggregation exactly one sub-rule sets the macro cost, so the whole
    step's penalty on that channel is attributable to it. Ties keep the first in
    registry order, which is deterministic and matches the order production
    itself evaluates.
    """

    worst = 0.0
    applicable = False
    blame: str | None = None
    for name in names:
        if name in drop:
            continue
        component = components.get(name)
        if component is None or not component.applicable:
            continue
        applicable = True
        cost = 0.0 if (drop_latch and _is_latch_cost(name, component)) else float(component.cost)
        if cost > worst:
            worst, blame = cost, name
    return (worst if applicable else 0.0), applicable, blame


def all_variant_names() -> tuple[str, ...]:
    """Every counterfactual rulebook the replay prices, in report order.

    The accumulator and the per-episode totals must agree exactly: a name known
    to one and not the other raises ``KeyError`` on every record, which is how
    the at-fault gate was first added. One source removes the failure mode.
    """

    return (
        *_VARIANTS,
        "proposed",
        "proposed_scoped",
        "proposed_scoped_graded",
        "proposed_scoped_pavone",
        "final",
        "final_ungated",
        *(f"final_gate{gate:g}" for gate in AT_FAULT_GATE_THRESHOLDS_MPS),
        *(f"family_{label}" for label, _, _, _ in family_grid()),
        *(f"v51_{label}" for label, _, _, _ in v51_weight_grid()),
    )


def worst_named(candidates: Iterable[tuple[str, float]]) -> tuple[float, str | None]:
    """Max aggregation over already-resolved ``(name, cost)`` pairs.

    :func:`macro_cost_with_blame` reads costs off the production components; the
    specified rulebook's macros mix those with costs recomputed here from
    geometry, so they cannot go through it. Tie handling matches: the first
    candidate in the given order wins, and the order is fixed by the caller.
    """

    worst = 0.0
    blame: str | None = None
    for name, cost in candidates:
        if cost > worst:
            worst, blame = cost, name
    return worst, blame


def _latch_attribution(name: str, component, measurement: Measurement) -> None:
    """Record whether a positive control cost came from the latch or the approach.

    ``crosswalk`` and ``vehicle_yield`` both compute ``cost = 1.0 if
    active_latch_for_zone else approach_cost``. A cost of exactly 1.0 whose
    ``before_gate`` diagnostic is False is therefore the latch branch: the ego is
    inside the zone and is being charged for an entry decision already made.
    """

    if component.cost <= 0.0:
        return
    branch = "latch" if _is_latch_cost(name, component) else "approach"
    measurement.latch_attribution[f"{name}:{branch}"] += 1


def replay_scenario(
    scenario: Mapping[str, Any],
    *,
    scenario_uid: str,
    measurement: Measurement,
    ego_brake_mps2: float,
    scalarization: ScalarizationConfig,
    source: str = "waymo",
) -> None:
    """Replay one scenario through the full transition and accumulate its costs.

    ``source`` selects the static adapter. On PG the result is a **coverage**
    measurement -- applicability rates and geometric sanity -- and explicitly
    **not Test A**: a PG record's logged ego is `IDMPolicy`, which makes the
    headway rules circular and the positional rules vacuous, so replay can
    establish nothing there about whether a rule is satisfiable. What it does
    establish is whether the rulebook is *applicable* on PG at all, and a
    sub-rule never applicable on half the training distribution is a sub-rule
    silently absent from it.
    """

    pavone_scalarization = ScalarizationConfig(mode="bounded_satisfaction_rank", priority_base=2.01)
    # The production baseline is now `SCAL-V1.4` on the six-level vector, while
    # the counterfactual variants below keep the four-margin family they were
    # derived under. Conflating the two is what broke this script silently: since
    # `RB51`/`M1` production emits six margins, and feeding them to the
    # four-margin config raised `ScalarizationEvaluationError` on **every**
    # record. The failure was invisible because it is a `ValueError` subclass and
    # the replay's own handler counted it as a skipped scenario -- the report said
    # "0 measured", which nobody read because the script had not been run since.
    production_scalarization = ScalarizationConfig(
        mode="six_level_priority_weighted_rank",
        priority_base=2.2,
        vector_schema_id=SIX_LEVEL_VECTOR_SCHEMA_ID,
    )

    static = (
        build_pg_static_adapter_result(scenario, scenario_uid=scenario_uid)
        if source.lower() == "pg"
        else build_waymo_static_adapter_result(scenario, scenario_uid=scenario_uid)
    )
    if static.validation_errors:
        measurement.scenarios_skipped[
            f"validation:{static.validation_errors[0].split(':')[0]}"
        ] += 1
        return
    cache = build_episode_cache(static)
    route = cache.route_polyline
    if route is None:
        measurement.scenarios_skipped["no_route_polyline"] += 1
        return
    z_origin_m = _sdc_z_origin(scenario)
    speed_limits = lane_speed_limits_mps(scenario)
    tracks = scenario["tracks"]
    sdc_id = str(scenario["metadata"].get("sdc_id", ""))
    if sdc_id not in tracks:
        measurement.scenarios_skipped["no_sdc_track"] += 1
        return
    config = RulebookTransitionConfig(
        rss_calibration=RSSCalibrationArtifact(
            config_hash=_CALIBRATION_HASH, ego_min_brake_mps2=ego_brake_mps2
        ),
        expected_config_hash=_CALIBRATION_HASH,
    )
    # Surfaces for the geometric variants. `evaluate_transition` builds its own
    # internally but does not expose them, so these are rebuilt with the same
    # helpers; the production replay has already shown the two paths agree on all
    # three components to within 0.01 percentage points.
    drivable_lanes = tuple(
        DrivableLaneRecord(lane.lane_id, lane.centerline, lane.polygon_xy, None)
        for lane in cache.route_lanes
    )

    def build_snapshot(step: int, previous_s_m: float | None) -> tuple[EnvSnapshot, float] | None:
        ego = _snapshot_at(sdc_id, tracks[sdc_id], step, z_origin_m=z_origin_m)
        if ego is None:
            return None
        actors = tuple(
            snapshot
            for track_id, track in tracks.items()
            if str(track_id) != sdc_id
            and (snapshot := _snapshot_at(track_id, track, step, z_origin_m=z_origin_m)) is not None
        )
        mission, s_m = mission_snapshot_at(
            route=route,
            ego=ego,
            step=step,
            mission_hash=scenario_uid,
            previous_s_m=previous_s_m,
        )
        snapshot = EnvSnapshot(
            scenario_id=scenario_uid,
            step_index=step,
            sim_time_s=float(step) * DELTA_T_S,
            ego=ego,
            actors=actors,
            contact_onset_records=(),
            active_contact_ids=frozenset(),
            signal_states_by_physical_id=signal_states_at(scenario, step),
            mission_snapshot=mission,
        )
        return snapshot, s_m

    length = int(scenario.get("length", 0))
    initial = build_snapshot(0, None)
    if initial is None:
        measurement.scenarios_skipped["no_valid_sdc_step"] += 1
        return
    pre_state, previous_s_m = initial
    # EP-COMFORT-DIAG: the same accumulator production runs, fed the same
    # `ego_kinematics_payload` the online wrapper publishes, so the expert
    # reference and the agent measurements come from one definition.
    comfort = ComfortEpisodeAccumulator()
    comfort.observe({"ego_kinematics": ego_kinematics_payload(pre_state)})
    memory: RulebookMemory = initial_memory_for_snapshot(pre_state, cache)
    # RULEBOOK-V5.1 L4. ``q`` is the running maximum of the completion fraction,
    # so ground already credited earns nothing when re-covered and the episode
    # total telescopes to ``q_T - q_0``.
    route_length_m = float(route.length_m)
    v51_s_max = previous_s_m
    v51_prev_s_m = previous_s_m
    v51_q = 0.0 if route_length_m <= 0.0 else min(max(previous_s_m / route_length_m, 0.0), 1.0)
    v51_q_start = v51_q
    measurement.v51_route_lengths_m.append(route_length_m)
    v51_episode_delta_q = 0.0
    v51_episode_delta_s = 0.0
    v51_episode_behind_peak = False
    episode_return = 0.0
    # `M8c`: the per-scenario view of applicability. A rule that is never
    # applicable on a record was never tested there, and Test A -- which can only
    # reject -- reads that as a clean pass.
    episode_applicable_sub_rules: set[str] = set()
    episode_channels = {"l1": 0.0, "l2": 0.0, "l3": 0.0, "l4": 0.0, "l5": 0.0, "l6": 0.0}
    episode_steps = 0
    variant_totals = {variant: 0.0 for variant in all_variant_names()}
    # Blame accounting for the proposed rulebook, in reward units. Under max
    # aggregation one sub-rule sets each channel's cost, so the penalty that
    # channel contributes is wholly attributable to it. Measured by differencing
    # the scalarized reward against the same step with that channel satisfied,
    # so the weights never have to be restated here.
    episode_blame: Counter = Counter()
    # The same, for the specified rulebook, plus the crawl split of its penalty
    # mass. Both are only reported over the episodes it leaves below standstill,
    # but they must be accumulated per step because the classification is a
    # property of the step, not of the episode.
    final_episode_blame: Counter = Counter()
    final_mass_total = dict.fromkeys(EGO_CRAWL_THRESHOLDS_MPS, 0.0)
    final_mass_at_crawl = dict.fromkeys(EGO_CRAWL_THRESHOLDS_MPS, 0.0)
    final_blame_mass_at_crawl: Counter = Counter()

    for step in range(1, length):
        built = build_snapshot(step, previous_s_m)
        if built is None:
            # Signal the gap explicitly. Skipping the observation silently would
            # let a Savitzky-Golay window span the missing steps and treat the
            # jump across them as ordinary motion; the online path gets this for
            # free because an unusable step still reaches `observe`.
            comfort.observe(None)
            continue
        post_state, previous_s_m = built
        comfort.observe({"ego_kinematics": ego_kinematics_payload(post_state)})
        result, memory, cache_delta = evaluate_transition(
            pre_state=pre_state,
            post_state=post_state,
            memory=memory,
            cache=cache,
            config=config,
        )
        cache = apply_cache_delta(cache, cache_delta)
        # Both RSS rules are scoped on the *pre*-transition state while the
        # geometric rules use the post state (`transition.py` component_inputs).
        # The variants must be built from the same state production used, so the
        # pre state is kept before the loop advances.
        rss_state = pre_state
        pre_state = post_state

        measurement.total_steps += 1
        episode_steps += 1
        for name in _NORMATIVE_COMPONENTS:
            component = result.components.get(name)
            if component is None:
                continue
            if component.applicable:
                episode_applicable_sub_rules.add(name)
            measurement.components[name].add(component.cost, applicable=component.applicable)
            if name in {"crosswalk", "vehicle_yield"}:
                _latch_attribution(name, component, measurement)
        for name in _MACRO_COMPONENTS:
            component = result.components.get(name)
            if component is not None:
                measurement.macros[name].add(component.cost, applicable=component.applicable)
        measurement.progress_margins.append(float(result.margins[3]))

        scalarized = scalarize_rulebook_margins(result.margins, production_scalarization)
        episode_return += float(scalarized.reward)
        # The per-channel decomposition mirrors the `SCAL-V1.4` formula so the
        # deficit can be attributed without re-deriving it downstream. L1-L3 keep
        # the priority-weighted indicator form; L4 is a utility and L5/L6 are
        # costs scaled by `dt / T_REF`.
        base = float(production_scalarization.priority_base)
        dt_ratio = float(production_scalarization.step_dt_s) / float(
            production_scalarization.reference_time_s
        )
        for index, (channel, weight) in enumerate((("l1", base**3), ("l2", base**2), ("l3", base))):
            margin = float(result.margins[index])
            indicator = -1.0 if margin < 0.0 else 0.0
            episode_channels[channel] += weight * (indicator + margin) + (
                float(production_scalarization.flat_tie_breaker) * margin
            )
        episode_channels["l4"] += float(production_scalarization.progress_weight) * float(
            result.margins[3]
        )
        episode_channels["l5"] += (
            float(production_scalarization.relaxable_weight) * float(result.margins[4]) * dt_ratio
        )
        episode_channels["l6"] += (
            float(production_scalarization.progress_rate_weight)
            * float(result.margins[5])
            * dt_ratio
        )

        progress_margin = float(result.margins[3])
        for variant, spec in _VARIANTS.items():
            r2_cost, _ = macro_cost(
                _R2_COMPONENTS,
                result.components,
                drop=spec["drop_r2"],
                drop_latch=spec["drop_latch"],
            )
            r3_cost, _ = macro_cost(
                _R3_COMPONENTS,
                result.components,
                drop=spec["drop_r3"],
                drop_latch=spec["drop_latch"],
            )
            if variant == "production":
                # The counterfactuals are only trustworthy if the same code path
                # reproduces production exactly when nothing is dropped. Checking
                # it on every real step is a far stronger guarantee than a unit
                # test, and makes a divergence fail the run instead of silently
                # reporting a wrong comparison.
                # The counterfactual family is v5.0-era: its `r2`/`r3` are the
                # *old* macro rules, and production no longer computes either.
                # `rss` left L2 (ADR-063) and the relaxable lane rules left R3
                # for L5 (ADR-072), so the self-check is re-pointed at the v5.1
                # channels and recomputed from the v5.1 memberships. Comparing
                # the old aggregates against the new channels would fail on every
                # step where `rss` is the worst L2 candidate -- 18.29 % of
                # applicable steps -- and would say the instrument was broken
                # when it was only measuring a rulebook that no longer exists.
                for macro_name, members in (
                    ("interaction_risk", V51_L2_SUB_RULES),
                    ("non_relaxable_compliance", V51_L3_SUB_RULES),
                    ("relaxable_lane_compliance", V51_L5_SUB_RULES),
                ):
                    recomputed, _ = macro_cost(
                        members, result.components, drop=frozenset(), drop_latch=False
                    )
                    produced = result.components[macro_name].cost
                    if macro_name == "relaxable_lane_compliance":
                        # L5 aggregates by normalized sum with a *fixed*
                        # denominator of 3, not by max (§3.3), so it needs its own
                        # recomputation rather than `macro_cost`'s worst-of.
                        recomputed = (
                            sum(
                                result.components[name].cost
                                for name in members
                                if name in result.components and result.components[name].applicable
                            )
                            / 3.0
                        )
                    if abs(produced - recomputed) > 1e-9:
                        raise ValueError(
                            f"Counterfactual recomputation diverged from production "
                            f"{macro_name}: {produced!r} vs {recomputed!r}"
                        )
            if r2_cost > 0.0:
                measurement.variant_r2_violated[variant] += 1
            if r3_cost > 0.0:
                measurement.variant_r3_violated[variant] += 1
            variant_totals[variant] += float(
                scalarize_rulebook_margins(
                    (float(result.margins[0]), -r2_cost, -r3_cost, progress_margin), scalarization
                ).reward
            )

        # --- geometric variants and the time-headway sweep -------------------
        ego = post_state.ego
        drivable = drivable_surface_for_ego(
            ego_footprint=ego.footprint,
            ego_position_xy=ego.position_xy,
            ego_position_z=ego.position_z,
            lanes=drivable_lanes,
        )
        variant_offroad = 0.0
        variant_carriageway = 0.0
        if not drivable.is_empty and drivable.is_valid:
            variant_offroad = variant_offroad_cost(
                ego_footprint=ego.footprint, drivable_surface=drivable
            )
            measurement.variant_components["offroad_tolerant"].add(variant_offroad, applicable=True)
            measurement.observe_oracle(
                "offroad", result.components["offroad"].cost, variant_offroad
            )
            surfaces = carriageway_surfaces_for_ego(
                ego_position_xy=ego.position_xy,
                ego_position_z=ego.position_z,
                route_tangent_xy=route.project(
                    ego.position_xy, position_z=ego.position_z
                ).tangent_xy,
                lanes=drivable_lanes,
            )
            variant_carriageway = variant_wrong_carriageway_cost(
                ego_footprint=ego.footprint,
                aligned_surface=surfaces.aligned,
                opposing_surface=surfaces.opposing,
            )
            measurement.variant_components["wrong_carriageway_entry"].add(
                variant_carriageway, applicable=True
            )
            measurement.observe_oracle(
                "wrong_carriageway",
                result.components["wrong_carriageway"].cost,
                variant_carriageway,
            )
        solid_boundaries = tuple(
            feature
            for feature in cache.map_feature_catalog.values()
            if feature.feature_class is MapFeatureClass.LANE_MARKING_SOLID
            and feature.elevation_m is not None
            and abs(feature.elevation_m - ego.position_z) <= VERTICAL_COMPATIBILITY_TOLERANCE_M
        )
        solid_costs = variant_solid_line_costs(
            ego_footprint=ego.footprint, solid_boundaries=solid_boundaries
        )
        for tolerance, cost in solid_costs.items():
            measurement.variant_components[f"solid_line_graded_tol{tolerance:g}"].add(
                cost, applicable=bool(solid_boundaries)
            )
        variant_solid = solid_costs[SOLID_LINE_PROPOSED_TOLERANCE]
        measurement.observe_oracle(
            "solid_line", result.components["solid_line"].cost, variant_solid
        )
        candidates = _rss_candidates(
            ego=rss_state.ego,
            actors=rss_state.actors,
            route=route,
            route_lanes=cache.route_lanes,
        )
        for key, (cost, applicable) in time_headway_costs(candidates).items():
            measurement.thw_samples[key].add(cost, applicable=applicable)
        ego_accel_mps2 = (
            math.hypot(*post_state.ego.velocity_xy) - math.hypot(*rss_state.ego.velocity_xy)
        ) / DELTA_T_S
        for (response_time, brake), (cost, applicable) in responsive_rss_costs(
            candidates,
            ego_brake_mps2=config.rss_calibration.ego_min_brake_mps2,
            ego_accel_mps2=ego_accel_mps2,
        ).items():
            measurement.variant_components[
                f"rss_responsive_rho{response_time:g}_brake{brake:g}"
            ].add(cost, applicable=applicable)

        # Blame accounting named clearance and rss_lateral as the two largest
        # residual costs of the proposed rulebook, so both are re-derived here
        # under a scoping that keeps only what the ego itself controls.
        # `T-RB51-12` for the two gated sub-rules. Compared only above the
        # at-fault gate: below it production reports NOT_APPLICABLE by design
        # (ADR-070) while these variants still price the geometry, so a
        # disagreement there is the gate working, not a drift.
        if not ego_is_stopped(post_state.ego.velocity_xy):
            measurement.observe_oracle(
                "clearance",
                result.components["clearance"].cost,
                clearance_costs(
                    ego_footprint=ego.footprint,
                    actors=post_state.actors,
                    ego_position_z=ego.position_z,
                    drivable_surface=drivable,
                )[1],
            )
            measurement.observe_oracle(
                "ttc",
                result.components["ttc"].cost,
                ttc_costs_at(result.components["ttc"].raw)[FINAL_TTC_THRESHOLD_S],
            )
        clearance_full, clearance_scoped = clearance_costs(
            ego_footprint=ego.footprint,
            actors=post_state.actors,
            ego_position_z=ego.position_z,
            drivable_surface=drivable,
        )
        lateral_candidates = _rss_lateral_candidates(
            ego=rss_state.ego,
            actors=rss_state.actors,
            route=route,
            route_lanes=cache.route_lanes,
            ego_brake_mps2=config.rss_calibration.ego_min_brake_mps2,
        )
        lateral_full, lateral_ego_only = lateral_rss_costs(lateral_candidates)
        # A variant is only worth reading if the reproduction it is derived from
        # matches what production charged on the same step.
        #
        # What production charges changed under `RB51`: `clearance` is now scoped
        # to VRU on the roadway (ADR-067), so the *scoped* reproduction is the one
        # that must match, not the unscoped one this check was written against;
        # and both sub-rules are inapplicable below the at-fault gate (ADR-070),
        # where production has nothing to reproduce. Leaving the old comparison in
        # place would have reported the rulebook's own approved redefinitions as
        # an instrument defect.
        if not ego_is_stopped(post_state.ego.velocity_xy):
            for name, reproduced in (
                ("clearance", clearance_scoped),
                ("rss_lateral", lateral_full),
            ):
                produced = result.components[name]
                expected = float(produced.cost) if produced.applicable else 0.0
                if abs(expected - reproduced) > 1e-6:
                    raise ValueError(
                        f"Variant reproduction diverged from production {name}: "
                        f"{expected!r} vs {reproduced!r}"
                    )
        measurement.variant_components["clearance_on_road"].add(
            clearance_scoped, applicable=result.components["clearance"].applicable
        )
        measurement.variant_components["rss_lateral_ego_only"].add(
            lateral_ego_only, applicable=result.components["rss_lateral"].applicable
        )

        ego_lane = associate_route_lane(
            position_xy=ego.position_xy,
            position_z=ego.position_z,
            heading_rad=ego.heading_rad,
            route_lanes=cache.route_lanes,
        )
        limit_mps = speed_limits.get(ego_lane.lane_id) if ego_lane is not None else None
        ego_speed_mps = math.hypot(*ego.velocity_xy)
        for tolerance in SPEED_LIMIT_TOLERANCES_MPS:
            cost = (
                speed_limit_costs(ego_speed_mps=ego_speed_mps, limit_mps=limit_mps)[tolerance]
                if limit_mps is not None
                else 0.0
            )
            measurement.variant_components[f"speed_limit_tol{tolerance:g}"].add(
                cost, applicable=limit_mps is not None
            )

        # The full proposed rulebook: RSS demoted, wrongway and dashed_line
        # removed, traffic-control latches removed, and the three geometric
        # sub-rules redefined.
        proposed_r2, _, blame_r2 = macro_cost_with_blame(
            _R2_COMPONENTS, result.components, drop=frozenset({"rss"}), drop_latch=False
        )
        proposed_r3, _, blame_r3 = macro_cost_with_blame(
            ("signal", "stop", "crosswalk", "vehicle_yield"),
            result.components,
            drop=frozenset(),
            drop_latch=True,
        )
        for variant_name, variant_cost in (
            ("offroad", variant_offroad),
            ("solid_line", variant_solid),
            ("wrong_carriageway", variant_carriageway),
        ):
            if variant_cost > proposed_r3:
                proposed_r3, blame_r3 = variant_cost, variant_name
        proposed_reward = float(
            scalarize_rulebook_margins(
                (float(result.margins[0]), -proposed_r2, -proposed_r3, progress_margin),
                scalarization,
            ).reward
        )
        variant_totals["proposed"] += proposed_reward
        for channel_cost, blame_name, satisfied in (
            (proposed_r2, blame_r2, (float(result.margins[0]), 0.0, -proposed_r3, progress_margin)),
            (proposed_r3, blame_r3, (float(result.margins[0]), -proposed_r2, 0.0, progress_margin)),
        ):
            if channel_cost <= 0.0 or blame_name is None:
                continue
            episode_blame[blame_name] += (
                float(scalarize_rulebook_margins(satisfied, scalarization).reward) - proposed_reward
            )
        if proposed_r2 > 0.0:
            measurement.variant_r2_violated["proposed"] += 1
        if proposed_r3 > 0.0:
            measurement.variant_r3_violated["proposed"] += 1

        # `proposed_scoped` additionally restricts the two sub-rules the blame
        # accounting named: VRU clearance to VRUs on the roadway, and lateral RSS
        # to the displacement the ego itself brings. R3 is unchanged.
        scoped_ttc = result.components["ttc"]
        scoped_r2 = max(
            clearance_scoped,
            lateral_ego_only,
            float(scoped_ttc.cost) if scoped_ttc.applicable else 0.0,
        )
        scoped_reward = float(
            scalarize_rulebook_margins(
                (float(result.margins[0]), -scoped_r2, -proposed_r3, progress_margin),
                scalarization,
            ).reward
        )
        variant_totals["proposed_scoped"] += scoped_reward
        # The same rulebook priced without SCAL-V1.1's satisfaction indicator,
        # which is what makes a rare sub-rule expensive regardless of severity.
        # Veer et al. (ICRA 2023) Theorem 1 keeps severity *outside* the
        # priority-weighted term, as a shared 1/N tie-breaker; SCAL-V1.1 SS7.6
        # deliberately moved it inside, multiplying it by the priority weight.
        # `bounded_satisfaction_rank` at a=2.01 is the published construction, so
        # the deviation can be priced instead of argued.
        variant_totals["proposed_scoped_pavone"] += float(
            scalarize_rulebook_margins(
                (float(result.margins[0]), -scoped_r2, -proposed_r3, progress_margin),
                pavone_scalarization,
            ).reward
        )
        # The rulebook as specified: clearance scoped to the roadway, rss_lateral
        # left as production computes it, ttc at nuPlan's uniform 0.95 s, R3 with
        # the three geometric redefinitions, dashed_line kept, wrongway dropped,
        # traffic-control latches dropped, speed_limit added, and the three
        # interaction sub-rules gated on a stopped ego (ADR-070). Scalarized with
        # a = 2.2, severity outside the priority weight, flat 1/N tie-breaker.
        final_rss_lateral = result.components["rss_lateral"]
        final_dashed = result.components["dashed_line"]
        interaction_costs = (
            ("clearance", clearance_scoped),
            (
                "rss_lateral",
                float(final_rss_lateral.cost) if final_rss_lateral.applicable else 0.0,
            ),
            ("ttc", ttc_costs_at(result.components["ttc"].raw)[FINAL_TTC_THRESHOLD_S]),
        )
        # Both published thresholds are priced, alongside the ungated rulebook, so
        # the decision rests on the measured delta between them.
        gated_interaction_costs = {
            gate: tuple(
                (name, 0.0 if name in AT_FAULT_GATED_SUB_RULES and ego_stopped else cost)
                for name, cost in interaction_costs
            )
            for gate, ego_stopped in (
                (gate, ego_speed_mps <= gate) for gate in AT_FAULT_GATE_THRESHOLDS_MPS
            )
        }
        final_r2, final_blame_r2 = worst_named(gated_interaction_costs[FINAL_AT_FAULT_GATE_MPS])
        ungated_r2, _ = worst_named(interaction_costs)
        final_control_r3, _, final_blame_control = macro_cost_with_blame(
            ("signal", "stop", "crosswalk", "vehicle_yield"),
            result.components,
            drop=frozenset(),
            drop_latch=True,
        )
        final_dashed_cost = float(final_dashed.cost) if final_dashed.applicable else 0.0
        final_speed_limit_cost = (
            speed_limit_costs(ego_speed_mps=ego_speed_mps, limit_mps=limit_mps)[
                SPEED_LIMIT_PROPOSED_TOLERANCE_MPS
            ]
            if limit_mps is not None
            else 0.0
        )
        final_r3, final_blame_r3 = worst_named(
            (
                (final_blame_control or "traffic_control", final_control_r3),
                ("offroad", variant_offroad),
                ("solid_line", variant_solid),
                ("wrong_carriageway", variant_carriageway),
                ("dashed_line", final_dashed_cost),
                ("speed_limit", final_speed_limit_cost),
            )
        )
        # --- RULEBOOK-V5.1 five-level aggregation ---------------------------
        # Same atomic costs, redistributed: the relaxable lane rules leave the
        # band above progress and become L5. L1-L3 keep `max`; L5 takes a
        # normalized sum, because there the quantity of interest is the *total*
        # amount of relaxation -- `max` would make a second concurrent violation
        # free, which breaks ordering O6.
        v51_behind_peak_m = max(0.0, v51_s_max - previous_s_m)  # diagnostic only
        if v51_behind_peak_m > 1e-6:
            measurement.v51_behind_peak_steps += 1
            v51_episode_behind_peak = True
        if v51_behind_peak_m > 1.0:
            measurement.v51_behind_peak_steps_1m += 1
        measurement.v51_behind_peak_max_m = max(
            measurement.v51_behind_peak_max_m, v51_behind_peak_m
        )
        # SIGNED, not monotone. Reversing debits exactly what re-advancing
        # credits, so the return is invariant to how many times a stretch is
        # covered -- the same invariance the running maximum bought, obtained by
        # compensation instead of by memory, and therefore with no hidden state.
        v51_delta_s = previous_s_m - v51_prev_s_m
        measurement.v51_delta_s_max_observed = max(
            measurement.v51_delta_s_max_observed, abs(v51_delta_s)
        )
        if v51_delta_s > 0.0:
            measurement.v51_delta_s_samples.append(v51_delta_s)
        v51_s_max = max(v51_s_max, previous_s_m)
        v51_prev_s_m = previous_s_m
        v51_delta_q_raw = v51_delta_s / V51_REFERENCE_ADVANCE_M
        v51_delta_q = max(-V51_DELTA_Q_MAX, min(v51_delta_q_raw, V51_DELTA_Q_MAX))
        if abs(v51_delta_q_raw) > V51_DELTA_Q_MAX:
            measurement.v51_delta_q_clip_binding += 1
        # ADR-076: L4 is the bare advance again -- the time cost lives at L6,
        # below relaxable lane compliance, so an illegal shortcut loses at L5
        # before the level is reached and O3 stays a theorem.
        v51_l4 = v51_delta_q
        # `1 - clip(Delta q, 0, 1)`: summed over a completing trajectory this is
        # `T - Q`, so L6 ranks by duration exactly, while still giving a per-step
        # gradient. Reverse motion clips to 0 advance and therefore costs the
        # maximum, consistent with L4's own signed penalty.
        v51_l6 = (1.0 - max(0.0, min(v51_delta_q, 1.0))) / V51_L6_DENOMINATOR
        v51_q = 0.0 if route_length_m <= 0.0 else min(v51_s_max / route_length_m, 1.0)
        v51_l1 = -float(result.margins[0])
        v51_l2 = final_r2
        v51_l3 = max(final_control_r3, variant_offroad, final_speed_limit_cost)
        v51_l5 = (variant_solid + variant_carriageway + final_dashed_cost) / V51_L5_DENOMINATOR
        measurement.v51_channel_steps += 1
        if v51_l2 > 0.0:
            measurement.v51_channel_violated["l2"] += 1
        if v51_l3 > 0.0:
            measurement.v51_channel_violated["l3"] += 1
        if v51_l5 > 0.0:
            measurement.v51_channel_violated["l5"] += 1
        if v51_l6 > 0.0:
            measurement.v51_channel_violated["l6"] += 1
        measurement.v51_delta_q_total += v51_delta_q
        measurement.v51_delta_q_max_observed = max(
            measurement.v51_delta_q_max_observed, v51_delta_q
        )
        v51_episode_delta_q += v51_delta_q
        v51_episode_delta_s += v51_delta_s
        for label, lam, eta, lam6 in v51_weight_grid():
            variant_totals[f"v51_{label}"] += v51_reward(
                l1=v51_l1,
                l2=v51_l2,
                l3=v51_l3,
                delta_q=v51_l4,
                l5=v51_l5,
                l6=v51_l6,
                lam=lam,
                eta=eta,
                lam6=lam6,
            )
        final_margins = (float(result.margins[0]), -final_r2, -final_r3, progress_margin)
        final_reward = family_reward(
            final_margins,
            base=FINAL_PRIORITY_BASE,
            severity=FINAL_SEVERITY,
            flat=FINAL_FLAT_TIE_BREAKER,
            progress_weight=SCALARIZATION_FAMILY_PROGRESS_WEIGHT,
        )
        variant_totals["final"] += final_reward
        # The same rulebook without the gate, and at the alternative published
        # threshold. R3 is identical in all three, so any difference is the
        # gate's alone.
        for gate, gated_costs in gated_interaction_costs.items():
            gate_r2, _ = worst_named(gated_costs)
            variant_totals[f"final_gate{gate:g}"] += family_reward(
                (float(result.margins[0]), -gate_r2, -final_r3, progress_margin),
                base=FINAL_PRIORITY_BASE,
                severity=FINAL_SEVERITY,
                flat=FINAL_FLAT_TIE_BREAKER,
                progress_weight=SCALARIZATION_FAMILY_PROGRESS_WEIGHT,
            )
            if gate_r2 > 0.0:
                measurement.variant_r2_violated[f"final_gate{gate:g}"] += 1
        variant_totals["final_ungated"] += family_reward(
            (float(result.margins[0]), -ungated_r2, -final_r3, progress_margin),
            base=FINAL_PRIORITY_BASE,
            severity=FINAL_SEVERITY,
            flat=FINAL_FLAT_TIE_BREAKER,
            progress_weight=SCALARIZATION_FAMILY_PROGRESS_WEIGHT,
        )
        if ungated_r2 > 0.0:
            measurement.variant_r2_violated["final_ungated"] += 1
        if final_r3 > 0.0:
            measurement.variant_r3_violated["final_ungated"] += 1
            for gate in AT_FAULT_GATE_THRESHOLDS_MPS:
                measurement.variant_r3_violated[f"final_gate{gate:g}"] += 1
        # Blame and the crawl split are both differences against the same step
        # with one channel satisfied, so the scalarization weights are never
        # restated here and the two accountings stay mutually consistent.
        step_penalty = 0.0
        for channel_cost, blame_name, satisfied in (
            (final_r2, final_blame_r2, (final_margins[0], 0.0, final_margins[2], final_margins[3])),
            (final_r3, final_blame_r3, (final_margins[0], final_margins[1], 0.0, final_margins[3])),
        ):
            if channel_cost <= 0.0 or blame_name is None:
                continue
            relief = (
                family_reward(
                    satisfied,
                    base=FINAL_PRIORITY_BASE,
                    severity=FINAL_SEVERITY,
                    flat=FINAL_FLAT_TIE_BREAKER,
                    progress_weight=SCALARIZATION_FAMILY_PROGRESS_WEIGHT,
                )
                - final_reward
            )
            final_episode_blame[blame_name] += relief
            step_penalty += relief
            for crawl in EGO_CRAWL_THRESHOLDS_MPS:
                if ego_speed_mps <= crawl:
                    final_blame_mass_at_crawl[(blame_name, crawl)] += relief
        if step_penalty > 0.0:
            for crawl in EGO_CRAWL_THRESHOLDS_MPS:
                final_mass_total[crawl] += step_penalty
                if ego_speed_mps <= crawl:
                    final_mass_at_crawl[crawl] += step_penalty
        if final_r2 > 0.0:
            measurement.variant_r2_violated["final"] += 1
        if final_r3 > 0.0:
            measurement.variant_r3_violated["final"] += 1

        scoped_margins = (
            float(result.margins[0]),
            -scoped_r2,
            -proposed_r3,
            progress_margin,
        )
        for label, base, severity, flat in family_grid():
            variant_totals[f"family_{label}"] += family_reward(
                scoped_margins,
                base=base,
                severity=severity,
                flat=flat,
                progress_weight=SCALARIZATION_FAMILY_PROGRESS_WEIGHT,
            )
        # The family must contain production exactly, or none of the grid's
        # numbers are comparable with the production column above.
        reproduced = family_reward(
            (float(result.margins[0]), -proposed_r2, -proposed_r3, progress_margin),
            base=3.0,
            severity=1.0,
            flat=0.0,
            progress_weight=1.0,
        )
        if abs(reproduced - proposed_reward) > 1e-9:
            raise ValueError(
                f"Scalarization family failed to reproduce SCAL-V1.1: "
                f"{reproduced!r} vs {proposed_reward!r}"
            )
        variant_totals["proposed_scoped_graded"] += graded_reward(
            (float(result.margins[0]), -scoped_r2, -proposed_r3, progress_margin),
            float(scalarization.priority_base),
        )
        ttc_component = result.components["ttc"]
        for threshold, cost in ttc_costs_at(ttc_component.raw).items():
            measurement.variant_components[f"ttc_thr{threshold:g}s"].add(
                cost, applicable=ttc_component.applicable
            )
        if scoped_r2 > 0.0:
            measurement.variant_r2_violated["proposed_scoped"] += 1
        if proposed_r3 > 0.0:
            measurement.variant_r3_violated["proposed_scoped"] += 1

    if episode_steps == 0:
        measurement.scenarios_skipped["no_valid_transition"] += 1
        return
    measurement.scenarios_measured += 1
    measurement.comfort_summaries.append(comfort.finalize())
    for name in episode_applicable_sub_rules:
        measurement.scenarios_where_applicable[name] += 1
    # `M8b`. Static per scenario, so it is read once here rather than per step.
    control_lines = control_line_diagnostics(cache)
    if control_lines:
        measurement.control_line_totals.update(control_lines)
        signal_total = int(control_lines.get("signal_controls_total", 0))
        if signal_total:
            measurement.scenarios_with_controls += 1
            dropped = int(control_lines.get("signal_controls_approach_filter_dropped", 0))
            # ADR-051 reports 209 records where no `SIGNAL` is ever selectable and
            # splits them into 52 lost at adapter construction and 157 that are
            # "genuinely unrelated approaches **or** route ends more than one lane
            # short". This counts the second kind directly, which is what makes
            # the 25.2 % headline decomposable instead of an upper bound.
            if dropped >= signal_total:
                measurement.scenarios_with_no_selectable_signal += 1
    # AC-RB5.1-07. The monotone construction makes the undiscounted sum of the
    # per-step increments equal the episode's net completion exactly; any error
    # means `q` was not monotone or the clip bound.
    measurement.v51_telescoping_max_error = max(
        measurement.v51_telescoping_max_error,
        abs(v51_episode_delta_q - v51_episode_delta_s / V51_REFERENCE_ADVANCE_M),
    )
    if v51_episode_behind_peak:
        measurement.v51_behind_peak_episodes += 1
    measurement.v51_q_start.append(v51_q_start)
    measurement.v51_q_end.append(v51_q)
    measurement.episode_returns.append(episode_return)
    measurement.episode_steps.append(episode_steps)
    for channel, value in episode_channels.items():
        measurement.channel_returns[channel].append(value)
    for variant, total in variant_totals.items():
        measurement.variant_returns[variant].append(total)
    measurement.proposed_blame_total.update(episode_blame)
    if variant_totals["proposed"] < 0.0 and episode_blame:
        dominant = max(episode_blame.items(), key=lambda item: (item[1], item[0]))[0]
        measurement.proposed_blame_dominant_when_negative[dominant] += 1
    measurement.final_blame_total.update(final_episode_blame)
    # The crawl split answers a question about the below-standstill tail only, so
    # it is accumulated over exactly those episodes. Mass from episodes that
    # already beat standstill would dilute it with penalties the expert absorbed
    # and still came out ahead of standing still.
    if variant_totals["final"] < 0.0:
        measurement.final_negative_episodes += 1
        if final_episode_blame:
            dominant = max(final_episode_blame.items(), key=lambda item: (item[1], item[0]))[0]
            measurement.final_blame_dominant_when_negative[dominant] += 1
        for crawl in EGO_CRAWL_THRESHOLDS_MPS:
            measurement.final_negative_mass_total[crawl] += final_mass_total[crawl]
            measurement.final_negative_mass_at_crawl[crawl] += final_mass_at_crawl[crawl]
        measurement.final_blame_mass_in_negative.update(final_episode_blame)
        measurement.final_blame_mass_at_crawl.update(final_blame_mass_at_crawl)


@dataclass(frozen=True)
class WorkItem:
    """One record's work unit, sized to survive process serialization."""

    relative_path: str
    scenario_uid: str
    ego_brake_mps2: float
    # `M8a`. The source decides which static adapter runs. It used to be absent
    # because the script called the Waymo adapter unconditionally, which is why
    # PG coverage was never measured -- not because a PG adapter was missing.
    source: str = "waymo"


_WORKER_DATA_ROOT: Path | None = None


def _init_worker(data_root: Path) -> None:
    global _WORKER_DATA_ROOT
    _WORKER_DATA_ROOT = data_root


def replay_work_item(item: WorkItem) -> Measurement:
    """Replay one record in isolation and return its partial accumulator.

    Records share no state — adapter, cache and memory are all per-episode — so
    the work parallelizes exactly and ``Measurement.merge`` recombines the
    partials without changing the result.
    """

    assert _WORKER_DATA_ROOT is not None, "worker was not initialized with a data root"
    measurement = Measurement()
    path = _WORKER_DATA_ROOT / item.relative_path
    if not path.exists():
        measurement.scenarios_skipped["missing_source_file"] += 1
        return measurement
    with path.open("rb") as handle:
        scenario = pickle.load(handle)
    try:
        replay_scenario(
            scenario,
            scenario_uid=item.scenario_uid,
            measurement=measurement,
            ego_brake_mps2=item.ego_brake_mps2,
            scalarization=ScalarizationConfig(),
            source=item.source,
        )
    except (ValueError, KeyError, IndexError) as error:
        measurement.scenarios_skipped[f"error:{type(error).__name__}"] += 1
        # A bare exception-type count cannot distinguish an unusable record from
        # a self-check the measurement itself tripped, which are opposite
        # findings. Keep a bounded sample of the messages so the report says
        # which one happened.
        if len(measurement.error_samples) < _ERROR_SAMPLE_LIMIT:
            measurement.error_samples.append(f"{item.scenario_uid}: {str(error)[:200]}")
    return measurement


def selected_records(
    payload: Mapping[str, Any], *, split: str, source: str, limit: int | None
) -> list[Mapping[str, Any]]:
    records = [
        record
        for record in payload["records"]
        if record.get("source") == source
        and (split == "all" or record.get("split") == split)
        and record.get("rulebook_eligible", True)
    ]
    records.sort(key=lambda record: str(record.get("scenario_uid")))
    return records if limit is None else records[:limit]


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--frozen-index", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--source", default="waymo")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ego-brake-mps2", type=float, default=DEFAULT_EGO_BRAKE_MPS2)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(32, os.cpu_count() or 1),
        help="Worker processes; 1 runs in-process. Records are independent.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    payload = json.loads(args.frozen_index.read_text(encoding="utf-8"))
    records = selected_records(payload, split=args.split, source=args.source, limit=args.limit)
    measurement = Measurement()
    items = [
        WorkItem(
            relative_path=str(record["relative_path"]),
            scenario_uid=str(record["scenario_uid"]),
            ego_brake_mps2=float(args.ego_brake_mps2),
            source=str(record.get("source", args.source)),
        )
        for record in records
    ]
    workers = max(1, int(args.workers))
    if workers == 1:
        _init_worker(args.data_root)
        for index, item in enumerate(items, start=1):
            measurement.merge(replay_work_item(item))
            if index % 25 == 0:
                print(f"... {index}/{len(items)} records", flush=True)
    else:
        # Shapely is single-threaded and each record is independent, so the work
        # scales with processes. Cap the per-process BLAS/OpenMP pools first, or
        # every worker spawns its own and oversubscribes the host.
        for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(variable, "1")
        with multiprocessing.get_context("fork").Pool(
            processes=workers, initializer=_init_worker, initargs=(args.data_root,)
        ) as pool:
            for index, partial in enumerate(
                pool.imap_unordered(replay_work_item, items, chunksize=2), start=1
            ):
                measurement.merge(partial)
                if index % 25 == 0:
                    print(f"... {index}/{len(items)} records", flush=True)

    summary = {
        "configuration": {
            "split": args.split,
            "source": args.source,
            "records_selected": len(records),
            "ego_brake_mps2": float(args.ego_brake_mps2),
            "scalarization_mode": ScalarizationConfig().mode,
            "priority_base": ScalarizationConfig().priority_base,
            "workers": workers,
        },
        "result": measurement.summary(),
    }
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
