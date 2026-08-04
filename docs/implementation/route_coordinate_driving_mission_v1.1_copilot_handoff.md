# Copilot Handoff: Route-Coordinate Driving Mission v1.1

Use the following prompt from repository root on branch
`codex/route-coordinate-mission`.

```text
Implement the approved Route-Coordinate Driving Mission v1.1 completely and
verify it for experimental use.

Repository workflow is mandatory. Read AGENTS.md, then read these files in full
before changing production code:

- docs/project_index.md
- docs/specifications/driving_mission_v1.1_specification.md
- docs/decisions/ADR-054-route-coordinate-mission.md
- docs/decisions/ADR-052-unified-driving-mission-contract.md
- docs/decisions/ADR-053-geometric-gate-only-mission-boundaries.md
- docs/implementation/route_coordinate_driving_mission_v1.1_exec_plan.md
- .agent/PLANS.md

DRIVING-MISSION-V1.1 is authoritative. ADR-054 is accepted and every
DEC-RCM-001..009 is approved. Do not reopen or reinterpret those decisions.
Keep the ExecPlan current as implementation evidence and execute M1 through M8
in order. Add/freeze the mandatory acceptance tests before each production
milestone, implement the smallest coherent change, and update tests,
documentation, traceability, progress, file lists, validation results, and final
reconciliation together.

The required behavior, without substitutions, is:

1. Preserve PG- and Waymo-specific offline derivation of ordered assigned-route
   lane occurrences, but normalize both into the same versioned mission record.
   Runtime route construction and tracking must be identical for both sources.
   Reuse the v1 frozen offline final-goal evidence only after the specified
   complete audit. Preserve all 3,500 UID/split/path/source identities and write
   new candidate artifacts without overwriting historical data.

2. Build one deterministic canonical 3D polyline from ordered lane occurrences,
   parameterized by XY arc length. Preserve occurrence identity and Z. Freeze
   s_start from the causal reset pose and s_goal from the offline goal. Reject
   invalid topology, geometry, goals, and non-positive task intervals; never
   invent bridges or fall back to native MetaDrive trajectory authority.

3. Implement exactly one environment-owned stateful route association update
   per committed transition. Let L be accumulated planar ego-center travel since
   the last successful association and search only:

       B = 2.0 * L + canonical_projection_epsilon
       I = [max(s_start, s_previous - B),
            min(s_goal,  s_previous + B)]

   Project onto the finite, vertically compatible geometry of r(I), including
   endpoints. Select minimum planar distance. For genuine numeric near-ties,
   use absolute route-tangent alignment with the ego motion axis, then minimum
   abs(s_candidate - s_previous), then occurrence/segment order. If motion is
   below the canonical epsilon, use the ego longitudinal heading axis only for
   that tie-break. Heading must never gate or scale progress. Endpoint selection
   is CLAMPED. Only missing finite/vertically compatible bounded geometry is
   FROZEN: retain s, produce delta_s=0, accumulate travel, and resume by the same
   ordinary bounded algorithm later. Do not add an unbounded fallback, HMM,
   candidate score, ambiguity state, graph recovery, or tunable alternative.

4. Make one immutable MissionSnapshot the sole authority for every route
   consumer. R4, completion, semantic observation, LiDAR observation, Rulebook
   route relevance, success, metrics, and video must not independently project
   the ego. Repeated reads in one step must be bit-identical.

5. Define R4 exactly as:

       clip((s_post - s_pre) / (22.2222222222 * delta_t), -1, 1)

   It remains signed and is independent of configured speed cap, lateral
   offset, heading, off-route/wrong-way/wrong-carriageway status, R1-R3 values,
   final-span compatibility, and scalarization. Only an explicit projection
   freeze yields zero through delta_s=0. Do not add lateral penalties, heading
   factors, checkpoint sums, graph distance, terminal bonus, or shaping.

6. Expose instantaneous completion from current s and public route_completion as
   max-so-far longitudinal completion. Success sets public completion to one,
   but longitudinal completion reaching one does not itself imply success or
   terminate the episode.

7. Remove all intermediate gates/checkpoint counters from runtime mission
   authority. Keep the existing observation tensor layouts: exactly ten local
   route samples at s+5, s+10, ..., s+50 metres, using the existing goal
   crop/mask/padding semantics. Recompute them every step from tracker s. They
   are navigation information only: no persistent reached state, progress,
   success, or termination role. Do not add maneuver commands, multiscale
   samples, or change observation width/horizon. Bump schema/checkpoint/replay
   identities as required because the source semantics change.

8. Retain only one final swept-front-bumper gate at s_goal. Require crossing in
   positive route direction, vertical compatibility, and intersection with
   frozen legal same-direction compatible final lane spans. Accept a compatible
   legal parallel destination lane; reject opposing carriageways,
   perpendicular roads, unrelated service roads, reverse crossing, touch-only,
   spawn-beyond, and wrong-level cases. Do not require stop, minimum speed, or
   ego-heading threshold. Compatibility affects success only, never s or R4.

9. Preserve episode boundaries: collision, physical out-of-road, and final
   mission success terminate; time limit truncates. Clamp/freeze, route
   departure, missing lane association, reversal, wrong-way travel, and bypass
   of former intermediate gates do not terminate or truncate. Preserve all
   existing R1-R3 and scalarization behavior.

10. Preserve causality: runtime may use only the frozen mission, static map, and
    current/past committed ego states. Future SDC states and native MetaDrive
    completion/success/checkpoints remain prohibited except explicit diagnostic
    comparison where the specification allows it.

Treat factor 2.0 and the 10x5 m observation layout as fixed approved contracts,
not parameters to optimize. M1 validation must test them on the required
straight, curved, inner/outer-offset, reverse, folded, self-intersecting,
close-parallel, roundabout, and vertically overlapping fixtures plus PG/Waymo
replays. If validation shows systematic clamp/freeze or a contract defect, stop
and report exact evidence as a new approval gate; do not silently tune behavior.

Implement every TEST-RCM-001..032 and satisfy AC-RCM-001..010. Add a regression
test for every bug discovered. Run the exact focused commands recorded in the
ExecPlan, then applicable Rulebook tests, full tests, config checks, PG and Waymo
smokes, and git diff --check. Never claim an unexecuted check passed. Do not add
dependencies, overwrite frozen historical artifacts, weaken mandatory tests, or
perform unrelated cleanup.

At completion, reconcile every REQ/AC/TEST in the ExecPlan, update
docs/project_index.md to the verified implementation status, review the final
diff, and report in Italian: resulting behavior, changed files, checks and exact
results, approved decisions followed, deviations, unresolved issues, known
limitations, deferred optional work, experimental readiness, and which ChatGPT
project source files must be replaced.
```
