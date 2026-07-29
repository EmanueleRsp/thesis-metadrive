# ADR-034: PG static-obstacle generation and arm classification

- Status: APPROVED
- Date: 2026-07-29
- Approval evidence: explicit user instruction to apply the previously proposed bounded profile schedule and integrate static obstacles in dataset arm classification.
- Scope: ScenarioNet Integration v1.1, PG generation and catalog classification only.

## Context

MetaDrive `accident_prob` creates static accident/construction scenes on each eligible road block. ScenarioNet PG profiles previously fixed it to zero. The semantic observation already exposes visible static collidable objects, while the catalog already supports `has_static_obstacle` as a tag but not as an arm-classification input.

## Decision

Future PG generation uses the following per-eligible-block probabilities: P0 `0.00`, P1 `0.03`, P2 `0.08`, P3 `0.08`, P5 `0.15`.

The catalog records `has_static_obstacle` only when a static accident-scene object was actually realized. Static obstacles are an orthogonal scenario property. They make `A0_simple_low_traffic` ineligible, causing an otherwise simple scenario to use the existing A1 fallback. No static-obstacle-only rule promotes a scenario to A5; A2--A5 retain their existing precedence conditions.

## Consequences

Future rebuilt PG datasets have static-obstacle diversity and retain a clean A0 reference. Frozen artifacts remain unchanged and cannot be mixed with rebuilt artifacts without recording a new dataset identity. A live generation smoke must verify that the local MetaDrive exporter preserves realized obstacles in the runtime scenario.
