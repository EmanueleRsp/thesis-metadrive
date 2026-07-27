"""Parent-owned Scenario ACL slot decision.

The vectorized driver owns episode identity, generation counters, and mode
counters; this module owns the decision itself so the Generate/Replay contract
of the ACL v1.1 specification is reachable from tests without a live simulator.

The random-number call order is part of the observable contract: the replay
coin flip precedes the bandit arm draw, which precedes the fresh-record draw.
Any change to that order changes seeded run reproducibility.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from thesis_rl.curriculum.scenario_acl.arms import ScenarioArm, SCENARIO_ARM_NAMES
from thesis_rl.curriculum.scenario_acl.buffer import ScenarioBuffer
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
    time (ACL-SN-EXH-001, ADR-028): `True` where the arm still has at least one
    frozen catalog record the scenario buffer does not hold. It is `None` when
    the coin flip resolved to Replay before any arm eligibility was computed.
    """

    selection: AclSlotSelection
    sampled_arm_index: int | None = None
    sampled_arm_name: str | None = None
    generate_arm_exhausted: bool = False
    eligible_arm_mask: np.ndarray | None = None


def fresh_generate_candidates(
    train_records: Sequence[Any],
    *,
    arm_name: str,
    excluded_scenario_uids: frozenset[str] | set[str],
) -> list[Any]:
    """Return the frozen catalog records still eligible for a Generate on ``arm_name``.

    A record already held by the scenario buffer is not eligible: Generate means
    a scenario the buffer has never scored, Replay means one it already holds.
    """

    return [
        record
        for record in train_records
        if record.primary_arm == arm_name and record.scenario_uid not in excluded_scenario_uids
    ]


def eligible_generate_arm_mask(
    train_records: Sequence[Any],
    *,
    arms: Sequence[ScenarioArm],
    excluded_scenario_uids: frozenset[str] | set[str],
) -> np.ndarray:
    """Return, per arm, whether it currently has at least one fresh Generate record.

    An arm becomes ineligible once every one of its frozen catalog records has
    been absorbed into the scenario buffer (ACL-SN-EXH-001): Generate can no
    longer draw a scenario the buffer has not already scored for that arm.
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
) -> AclSlotDecision:
    """Decide Generate or Replay for one worker slot."""

    effective_excluded = set(excluded_scenario_uids)
    effective_excluded.update(str(record.scenario_id) for record in buffer.records())
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

    # ACL-SN-EXH-001 / ADR-028: an arm whose whole frozen pool the buffer
    # already holds cannot honour a Generate draw. Restrict the arm draw to
    # arms that still have a fresh record instead of drawing blind and
    # discovering the exhaustion afterwards (the pre-fix behaviour, which let
    # an exhausted arm's EMA score freeze forever because the degraded draw
    # never reached `bandit.update`).
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
    candidates = fresh_generate_candidates(
        train_records,
        arm_name=arm_name,
        excluded_scenario_uids=effective_excluded,
    )
    if not candidates:
        # Defensive: eligibility was computed from the same buffer and
        # exclusion snapshot immediately above, so this is unreachable.
        raise RuntimeError(
            f"ACL arm {arm_name!r} was reported eligible but has no fresh catalog record."
        )

    record = candidates[int(rng.integers(0, len(candidates)))]
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
    )
