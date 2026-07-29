"""Parent-owned Scenario ACL slot decision.

The vectorized driver owns episode identity, generation counters, and mode
counters; this module owns the decision itself so the Generate/Replay contract
of the ACL v1.3 specification is reachable from tests without a live simulator.

The random-number call order is part of the observable contract: the replay
coin flip precedes the bandit arm draw, which precedes the fresh-record draw.
Any change to that order changes seeded run reproducibility. ACL `v1.3`
(`DEC-008`) preserves the order but changes the values consumed, because the
Generate candidate set is no longer filtered by scenario-buffer membership;
`v1.1`/`v1.2` seeds therefore do not reproduce their runs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from thesis_rl.curriculum.scenario_acl.arms import ScenarioArm, SCENARIO_ARM_NAMES
from thesis_rl.curriculum.scenario_acl.buffer import ScenarioBuffer
from thesis_rl.curriculum.scenario_acl.catalog_state import ScenarioCatalogVisitState
from thesis_rl.curriculum.scenario_acl.mab import ScenarioArmBandit
from thesis_rl.curriculum.scenario_acl.record import ScenarioRecord
from thesis_rl.curriculum.scenario_acl.vectorized import AclSlotSelection


@dataclass(frozen=True)
class AclSlotDecision:
    """One slot decision plus the diagnostics the driver accounts for.

    ``sampled_arm_index`` is the arm the MAB actually drew.  It differs from
    ``selection.arm_index`` whenever a Generate draw could not be honoured and
    degraded into a Replay, which is exactly the case the MAB never observes.

    ``eligible_arm_mask`` is the Generate eligibility of every arm at decision
    time (`v1.3` REQ-009, ADR-028): `True` where the arm still has at least one
    catalog record that is neither quarantined nor in flight. It is `None` when
    the coin flip resolved to Replay before any arm eligibility was computed.
    Under `DEC-008` every arm is normally eligible; a `False` entry now means a
    genuinely unavailable pool, not a pool the scenario buffer happens to hold.

    ``coverage_cycle_closed`` reports whether this Generate draw exhausted the
    drawn arm's coverage cycle and restarted it, and ``coverage_cycle_id`` is
    the arm's cycle index the draw belongs to (`DEC-009`).
    """

    selection: AclSlotSelection
    sampled_arm_index: int | None = None
    sampled_arm_name: str | None = None
    generate_arm_exhausted: bool = False
    eligible_arm_mask: np.ndarray | None = None
    coverage_cycle_closed: bool = False
    coverage_cycle_id: int | None = None


def fresh_generate_candidates(
    train_records: Sequence[Any],
    *,
    arm_name: str,
    excluded_scenario_uids: frozenset[str] | set[str],
    visit_state: ScenarioCatalogVisitState | None = None,
) -> list[Any]:
    """Return the frozen catalog records eligible for a Generate on ``arm_name``.

    `v1.3` REQ-009 (`DEC-008`, `ADR-032`): a record is Generate-admissible iff
    it is neither quarantined (`ADR-024`) nor in flight in the current batch
    (`ADR-016`). Membership in the scenario buffer is irrelevant — the buffer is
    a replay index over the catalog, not a consumption marker. The previous
    rule caused `FIND-001` (an arm absorbed by the buffer left the curriculum)
    and `FIND-005` (per-arm bandit feedback drawn from each arm's low-learning-
    potential residual).

    Passing ``visit_state`` additionally restricts the result to the arm's
    current coverage cycle by dropping already-visited records (`DEC-009`).
    Omitting it yields the unrestricted admissible set, which is what arm
    eligibility is computed over.
    """

    candidates = [
        record
        for record in train_records
        if record.primary_arm == arm_name and record.scenario_uid not in excluded_scenario_uids
    ]
    if visit_state is None:
        return candidates
    return [
        record
        for record in candidates
        if not visit_state.is_visited(record.scenario_uid, arm_name=arm_name)
    ]


def eligible_generate_arm_mask(
    train_records: Sequence[Any],
    *,
    arms: Sequence[ScenarioArm],
    excluded_scenario_uids: frozenset[str] | set[str],
) -> np.ndarray:
    """Return, per arm, whether it currently has at least one admissible record.

    Coverage cycles are deliberately ignored here: an exhausted cycle closes and
    restarts (`DEC-009`), so it can never make an arm ineligible. An arm is
    ineligible only when every one of its frozen catalog records is
    simultaneously quarantined or in flight, which is the residual case
    `ADR-028`'s renormalization still exists to handle.
    """

    return np.array(
        [
            bool(
                fresh_generate_candidates(
                    train_records,
                    arm_name=arm.name,
                    excluded_scenario_uids=excluded_scenario_uids,
                )
            )
            for arm in arms
        ],
        dtype=bool,
    )


def _replay_selection(
    *,
    slot: int,
    episode_id: int,
    generation: int,
    record: ScenarioRecord,
) -> AclSlotSelection:
    arm_name = str(record.scenario_arm or record.generator_arm or record.source)
    return AclSlotSelection(
        slot_id=slot,
        episode_id=episode_id,
        generation=generation,
        mode="replay",
        arm_index=SCENARIO_ARM_NAMES.index(arm_name),
        arm_name=arm_name,
        reset_seed=int(record.reset_seed),
        scenario_uid=str(record.scenario_id),
        runtime_index=int(record.scenario_index),
        source=str(record.source),
    )


def select_acl_slot_decision(
    *,
    slot: int,
    episode_id: int,
    generation: int,
    excluded_scenario_uids: frozenset[str],
    buffer: ScenarioBuffer,
    bandit: ScenarioArmBandit,
    arms: Sequence[ScenarioArm],
    train_records: Sequence[Any],
    scenario_cfg: Any,
    rng: np.random.Generator,
    visit_state: ScenarioCatalogVisitState,
) -> AclSlotDecision:
    """Decide Generate or Replay for one worker slot."""

    # `v1.3` REQ-009 / `DEC-008`: `excluded_scenario_uids` carries exactly the
    # quarantined and in-flight UIDs supplied by the caller. The scenario
    # buffer is *not* unioned into it: a buffered record stays Generate-
    # admissible, so an arm can never be absorbed out of the curriculum
    # (`FIND-001`) and the bandit no longer observes only each arm's low-LP
    # residual (`FIND-005`).
    effective_excluded = frozenset(excluded_scenario_uids)
    can_replay = bool(
        scenario_cfg.use_replay and len(buffer) >= int(scenario_cfg.warmup_buffer_size)
    )
    if can_replay and float(rng.random()) < float(scenario_cfg.exploit_probability):
        replay = buffer.sample_replay(
            rng=rng,
            current_step=episode_id,
            cfg=scenario_cfg.replay_sampling,
            use_staleness=bool(scenario_cfg.use_staleness),
        )
        return AclSlotDecision(
            selection=_replay_selection(
                slot=slot,
                episode_id=episode_id,
                generation=generation,
                record=replay.record,
            )
        )

    # ADR-028, narrowed by `ADR-032`: restrict the arm draw to arms that still
    # have an admissible record instead of drawing blind and discovering the
    # unavailability afterwards (the pre-fix behaviour let such an arm's EMA
    # score freeze forever, because the degraded draw never reached
    # `bandit.update`). After `DEC-008` this is a safety net for the residual
    # all-quarantined / all-in-flight case, not a routine condition.
    eligible_mask = eligible_generate_arm_mask(
        train_records,
        arms=arms,
        excluded_scenario_uids=effective_excluded,
    )
    if not eligible_mask.any():
        if not len(buffer):
            raise RuntimeError("No ACL arm has a fresh catalog record and replay is empty.")
        replay = buffer.sample_replay(
            rng=rng,
            current_step=episode_id,
            cfg=scenario_cfg.replay_sampling,
            use_staleness=bool(scenario_cfg.use_staleness),
        )
        return AclSlotDecision(
            selection=_replay_selection(
                slot=slot,
                episode_id=episode_id,
                generation=generation,
                record=replay.record,
            ),
            generate_arm_exhausted=True,
            eligible_arm_mask=eligible_mask,
        )

    if bool(scenario_cfg.use_mab):
        arm_index, probabilities = bandit.sample_arm(rng, eligible_mask=eligible_mask)
    else:
        eligible_indices = np.flatnonzero(eligible_mask)
        arm_index = int(eligible_indices[int(rng.integers(0, len(eligible_indices)))])
        probabilities = np.where(eligible_mask, 1.0 / int(eligible_mask.sum()), 0.0)
    arm_name = arms[arm_index].name

    # `DEC-009`: sample without replacement inside the arm's coverage cycle.
    # An exhausted cycle closes and restarts *before* the draw, so cycle
    # exhaustion never degrades a Generate into a Replay. Closing consumes no
    # random numbers, so the declared RNG call order is preserved.
    candidates = fresh_generate_candidates(
        train_records,
        arm_name=arm_name,
        excluded_scenario_uids=effective_excluded,
        visit_state=visit_state,
    )
    coverage_cycle_closed = False
    if not candidates:
        visit_state.close_cycle(arm_name)
        coverage_cycle_closed = True
        candidates = fresh_generate_candidates(
            train_records,
            arm_name=arm_name,
            excluded_scenario_uids=effective_excluded,
        )
    if not candidates:
        # Defensive: eligibility was computed from the same exclusion snapshot
        # immediately above and the cycle has just been cleared, so this is
        # unreachable.
        raise RuntimeError(
            f"ACL arm {arm_name!r} was reported eligible but has no fresh catalog record."
        )

    # RAT-010: uniform within the cycle candidate set. No intra-arm
    # prioritization by learning potential exists, by design.
    record = candidates[int(rng.integers(0, len(candidates)))]
    # Marked at selection time, not at commit time, so a record in flight
    # cannot be drawn twice inside the same cycle (REQ-009).
    visit_state.mark_visited(
        str(record.scenario_uid), arm_name=arm_name, episode_id=int(episode_id)
    )
    return AclSlotDecision(
        selection=AclSlotSelection(
            slot_id=slot,
            episode_id=episode_id,
            generation=generation,
            mode="generate",
            arm_index=int(arm_index),
            arm_name=arm_name,
            reset_seed=int(record.runtime_index),
            scenario_uid=str(record.scenario_uid),
            runtime_index=int(record.runtime_index),
            source=str(record.source),
            selection_probability=float(probabilities[arm_index]),
        ),
        sampled_arm_index=int(arm_index),
        sampled_arm_name=arm_name,
        eligible_arm_mask=eligible_mask,
        coverage_cycle_closed=coverage_cycle_closed,
        coverage_cycle_id=visit_state.cycle_id(arm_name),
    )
