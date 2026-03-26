# MetaDrive Assumptions and Notes

This file summarizes the assumptions that should shape the first implementation.

## 1. Observation choice

The initial implementation should use **LidarStateObservation**.

Reason:
- it is already structured
- it is already normalized enough for the first baseline
- it avoids image preprocessing
- it fits the first modular baseline well

The thesis notes describe it as a compact state vector composed of ego-state features, navigation features, nearby vehicle descriptors, and LiDAR ray measurements. In the standard configuration, the notes report a decomposition of `9 + 10 + 16 + 240` dimensions.

## 2. Action interface

MetaDrive uses a normalized **2D continuous action**:

- `a[0]`: steering
- `a[1]`: acceleration / braking

with both dimensions normalized in `[-1, 1]`.

This strongly supports the initial design choice:
- planner outputs the final low-level action
- adapter is initially a thin pass-through component

## 3. Gym compatibility and violation vector

The notes explicitly point out that MetaDrive still follows a standard Gym-style interface and returns a **scalar reward**. Therefore, rulebook-based components should initially be computed externally and exposed through `info`, instead of replacing the scalar reward at the environment interface.

Design consequence:
- keep env API standard
- compute violation vector in wrapper / reward manager
- let planner integration with vector violations happen later

## 4. Curriculum stages

The notes propose four stages:

1. fixed simple maps with no traffic
2. small procedural maps without traffic, plus gradual randomization
3. procedural maps with light traffic
4. generalization setting with disjoint train/test seeds

Concretely, the notes mention examples such as:
- Stage 1: `map="S"`, `"C"`, `"SC"`, `"ST"` and `traffic_density=0`
- Stage 2: `map=3 or 4`, `traffic_density=0`, then enable `random_lane_width`, `random_lane_num`, optionally `random_agent_model`
- Stage 3: `map=5`, `traffic_density=0.05 to 0.1`, with similar gradual randomization
- Stage 4: larger train scenario range via `num_scenarios` and `start_seed`, plus a disjoint evaluation seed range

## 5. Selected rulebook order

The thesis notes define the initial ordered signal as:

- no collisions with vehicles
- staying within the drivable area
- staying on the correct side of road
- goal / progress

with priority:

`collision ≻ drivable_area ≻ correct_side ≻ goal/progress`

## 6. Rule implementation consequences

The notes distinguish between:
- state-based rules
- transition-based rules

In particular:
- collision severity based on kinetic energy loss needs previous-step information
- drivable-area and correct-side rules are naturally state-based
- progress can initially be implemented as current goal proximity rather than differential progress

Design consequence:
introduce a `TransitionContext` early in the codebase, even before the planner uses violation vectors.
