"""Regression tests for FIND-001: a Generate arm whose fresh pool is exhausted.

See `docs/implementation/scenario_acl_exhausted_arm_mab_starvation_exec_plan.md`
and `docs/decisions/ADR-028-scenario-acl-generate-eligibility-renormalization.md`.

Originally observed live in
`EXP_thesis_.../td3_sb3/seed_0/20260726_055617`: the EMA score of `A4_vru`
stayed bit-identical at 0.715461969165923 from MAB update 1420 to 3118 while
the arm still held the highest Generate probability (~0.336). The arm's entire
frozen train pool (333 Waymo records; PG is a structurally empty cell) had been
absorbed by the scenario buffer, so every Generate draw on `A4_vru` silently
degraded into a Replay, and Replay never feeds the MAB — an absorbing state
that grew, never shrank, the exhausted arm's Generate share.

The approved correction (`DEC-EXH-001` option A, `DEC-EXH-002`) restricts the
Generate arm draw to arms that currently have at least one admissible catalog
record, renormalizing the softmax and the `eta/K` exploration floor over that
eligible subset (`ScenarioArmBandit.probabilities(eligible_mask=...)`). These
tests pin the corrected behaviour; `test_exhausted_arm_no_longer_starves_the_mab`
used to pin the defect itself before the fix and is inverted here, not deleted.

**Updated for ACL v1.3 (`ADR-032`, `DEC-008`).** The *root cause* of `FIND-001`
is gone: scenario-buffer membership no longer removes a record from the Generate
pool, so an absorbed arm can no longer become ineligible at all -- pinned by
`tests/test_scenario_acl_catalog_coverage.py` and, here, by
`test_buffer_absorption_alone_no_longer_makes_an_arm_ineligible`. The `ADR-028`
renormalization is retained as a safety net for the residual case in which an
arm's whole pool is simultaneously quarantined (`ADR-024`) or in flight
(`ADR-016`). The fixtures below therefore construct unavailability through that
exclusion set rather than through the buffer; the protections being asserted are
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest

from thesis_rl.curriculum.config import (
    ScenarioAclConfig,
    ScenarioAclMabConfig,
)
from thesis_rl.curriculum.scenario_acl.arms import build_default_scenario_arms
from thesis_rl.curriculum.scenario_acl.buffer import ScenarioBuffer
from thesis_rl.curriculum.scenario_acl.catalog_state import ScenarioCatalogVisitState
from thesis_rl.curriculum.scenario_acl.mab import ScenarioArmBandit
from thesis_rl.curriculum.scenario_acl.record import ScenarioRecord
from thesis_rl.curriculum.scenario_acl.selection import (
    eligible_generate_arm_mask,
    fresh_generate_candidates,
    select_acl_slot_decision,
)


ARMS = build_default_scenario_arms()
ARM_NAMES = tuple(arm.name for arm in ARMS)
EXHAUSTED_ARM = "A4_vru"
EXHAUSTED_ARM_INDEX = ARM_NAMES.index(EXHAUSTED_ARM)


@dataclass(frozen=True)
class _CatalogRecord:
    """Minimal stand-in for a frozen ScenarioNet catalog record."""

    scenario_uid: str
    primary_arm: str
    runtime_index: int
    source: str


def _catalog_record(arm: str, index: int, *, source: str = "waymo") -> _CatalogRecord:
    return _CatalogRecord(
        scenario_uid=f"{source}:train:{arm}:{index:04d}",
        primary_arm=arm,
        runtime_index=index,
        source=source,
    )


def _buffer_record(catalog_record: _CatalogRecord, *, usefulness: float) -> ScenarioRecord:
    return ScenarioRecord(
        scenario_id=catalog_record.scenario_uid,
        source=catalog_record.source,
        parent_id=None,
        scenario_description_path=f"/tmp/{catalog_record.scenario_uid}.pkl",
        scenario_description_hash=catalog_record.scenario_uid,
        dataset_directory="/tmp",
        scenario_index=catalog_record.runtime_index,
        env_config={"start_seed": catalog_record.runtime_index},
        reset_seed=catalog_record.runtime_index,
        generator_arm=None,
        mutation_type=None,
        mutation_params=None,
        validation_status="valid",
        rule_criticality=0.0,
        learning_potential=usefulness,
        usefulness=usefulness,
        usefulness_norm=1.0,
        rank=0,
        num_seen=1,
        last_seen_step=0,
        num_children=0,
        metrics_summary={},
        scenario_arm=catalog_record.primary_arm,
    )


def _config() -> ScenarioAclConfig:
    """The frozen ACL v1.1 defaults, with replay enabled as in the live profile."""

    return ScenarioAclConfig(
        buffer_capacity=1000,
        warmup_buffer_size=100,
        generate_probability=0.4,
        exploit_probability=0.6,
        use_mab=True,
        use_replay=True,
        use_staleness=True,
        mab=ScenarioAclMabConfig(),
    )


def _exhausted_fixture(
    *, pool_per_arm: int = 6
) -> tuple[tuple[_CatalogRecord, ...], ScenarioBuffer, frozenset[str]]:
    """Build a catalog where only ``EXHAUSTED_ARM`` has no admissible record.

    Every arm owns ``pool_per_arm`` frozen records. Since ACL v1.3 (`DEC-008`)
    unavailability can only come from the exclusion set the driver supplies --
    quarantined (`ADR-024`) and in-flight (`ADR-016`) UIDs -- so the whole pool
    of ``EXHAUSTED_ARM`` is excluded there. The buffer additionally holds that
    arm's whole pool plus one record of every other arm, which under `DEC-008`
    must have no effect whatsoever on eligibility; keeping it populated is what
    makes these fixtures a regression test for `FIND-001`'s root cause.
    """

    train_records = tuple(
        _catalog_record(arm, index) for arm in ARM_NAMES for index in range(pool_per_arm)
    )
    buffer = ScenarioBuffer(capacity=1000)
    for record in train_records:
        if record.primary_arm == EXHAUSTED_ARM:
            # A4 scores highest, which is exactly why the bandit favours it.
            buffer.insert(_buffer_record(record, usefulness=0.9))
        elif record.runtime_index == 0:
            buffer.insert(_buffer_record(record, usefulness=0.1))
    blocked = frozenset(
        record.scenario_uid for record in train_records if record.primary_arm == EXHAUSTED_ARM
    )
    return train_records, buffer, blocked


def _visit_state() -> ScenarioCatalogVisitState:
    return ScenarioCatalogVisitState(ARM_NAMES)


def test_generate_candidates_empty_once_the_whole_arm_pool_is_blocked() -> None:
    """The residual exhaustion trigger: every record of the arm is unavailable."""

    train_records, _buffer, blocked = _exhausted_fixture()

    assert (
        fresh_generate_candidates(
            train_records, arm_name=EXHAUSTED_ARM, excluded_scenario_uids=blocked
        )
        == []
    )
    for arm in ARM_NAMES:
        if arm == EXHAUSTED_ARM:
            continue
        assert fresh_generate_candidates(
            train_records, arm_name=arm, excluded_scenario_uids=blocked
        ), f"arm {arm} must keep candidates for this fixture to isolate the defect"


def test_eligible_generate_arm_mask_excludes_only_the_exhausted_arm() -> None:
    train_records, _buffer, blocked = _exhausted_fixture()

    mask = eligible_generate_arm_mask(train_records, arms=ARMS, excluded_scenario_uids=blocked)

    assert mask.tolist() == [name != EXHAUSTED_ARM for name in ARM_NAMES]


def test_buffer_absorption_alone_no_longer_makes_an_arm_ineligible() -> None:
    """`FIND-001` root cause removed (ACL v1.3 `DEC-008`, `ADR-032`).

    The same fixture that made `EXHAUSTED_ARM` ineligible under `v1.1`/`v1.2` --
    its entire pool held by the scenario buffer -- now leaves every arm
    eligible, because the buffer is not consulted at all. `ADR-028` only fires
    when the exclusion set itself blocks the pool.
    """

    train_records, buffer, _blocked = _exhausted_fixture()
    assert len(buffer) > 0

    mask = eligible_generate_arm_mask(train_records, arms=ARMS, excluded_scenario_uids=frozenset())

    assert mask.all()
    assert fresh_generate_candidates(
        train_records, arm_name=EXHAUSTED_ARM, excluded_scenario_uids=frozenset()
    )


def test_exhausted_arm_generate_draw_is_reassigned_to_an_eligible_arm() -> None:
    """The corrected behaviour: the exhausted arm is never drawn at all.

    A bandit that (pre-fix) would sample the exhausted arm with near-certainty
    instead produces a Generate on one of the still-eligible arms, and the
    decision records that the exhausted arm was excluded before sampling.
    """

    train_records, buffer, blocked = _exhausted_fixture()
    scenario_cfg = _config()
    bandit = ScenarioArmBandit(replace(scenario_cfg.mab, eta=1e-12, temperature=0.01))
    bandit.scores = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)

    decision = select_acl_slot_decision(
        slot=0,
        episode_id=0,
        generation=0,
        excluded_scenario_uids=blocked,
        buffer=buffer,
        bandit=bandit,
        arms=ARMS,
        train_records=train_records,
        # Force the Generate branch: no replay coin flip may intercept the draw.
        scenario_cfg=replace(scenario_cfg, exploit_probability=0.0),
        rng=np.random.default_rng(20260726),
        visit_state=_visit_state(),
    )

    assert decision.eligible_arm_mask is not None
    assert decision.eligible_arm_mask[EXHAUSTED_ARM_INDEX] == np.False_
    assert decision.selection.mode == "generate"
    assert decision.selection.arm_name != EXHAUSTED_ARM
    assert decision.sampled_arm_name != EXHAUSTED_ARM
    assert decision.generate_arm_exhausted is False
    assert decision.selection.selection_probability is not None


def test_exhausted_arm_no_longer_starves_the_mab() -> None:
    """FIND-001, corrected: the exhausted arm's frozen score can no longer self-block.

    This used to reproduce the live auto-block: the exhausted arm was sampled
    with its (frozen, highest) score, degraded silently to Replay, and its
    Generate probability only grew as the other arms decayed. With the fix,
    the arm is excluded from the draw entirely, so it is never sampled and its
    selection probability under the live eligibility mask is exactly zero
    instead of growing without bound.
    """

    train_records, buffer, blocked = _exhausted_fixture()
    scenario_cfg = replace(_config(), exploit_probability=0.0)
    bandit = ScenarioArmBandit(scenario_cfg.mab)
    bandit.scores = np.array([0.10, 0.10, 0.10, 0.10, 0.72, 0.10], dtype=np.float64)
    frozen_score = float(bandit.scores[EXHAUSTED_ARM_INDEX])

    rng = np.random.default_rng(20260726)
    visit_state = _visit_state()
    sampled_exhausted = 0
    generated_exhausted = 0
    for episode_id in range(400):
        decision = select_acl_slot_decision(
            slot=0,
            episode_id=episode_id,
            generation=episode_id,
            excluded_scenario_uids=blocked,
            buffer=buffer,
            bandit=bandit,
            arms=ARMS,
            train_records=train_records,
            scenario_cfg=scenario_cfg,
            rng=rng,
            visit_state=visit_state,
        )
        assert decision.eligible_arm_mask is not None
        assert decision.eligible_arm_mask[EXHAUSTED_ARM_INDEX] == np.False_
        if decision.sampled_arm_name == EXHAUSTED_ARM:
            sampled_exhausted += 1
        selection = decision.selection
        if selection.mode != "generate":
            continue
        if selection.arm_name == EXHAUSTED_ARM:
            generated_exhausted += 1
        # Mirror driver.py commit_event: only a committed Generate feeds the MAB.
        bandit.update(
            arm_index=selection.arm_index,
            normalized_usefulness=0.05,
            selection_probability=selection.selection_probability,
        )

    # The exhausted arm is excluded before sampling: it is drawn zero times,
    # not just zero times committed.
    assert sampled_exhausted == 0
    assert generated_exhausted == 0
    assert bandit.update_count > 0, "the other arms must still receive feedback"
    # The score cannot move because the arm is never selected -- that is
    # expected and harmless once the arm can no longer be drawn.
    assert float(bandit.scores[EXHAUSTED_ARM_INDEX]) == frozen_score
    # The key correction: under the live eligibility mask, the frozen score no
    # longer converts into a growing selection probability.
    eligible_mask = eligible_generate_arm_mask(
        train_records,
        arms=ARMS,
        excluded_scenario_uids=blocked,
    )
    assert bandit.probabilities(eligible_mask)[EXHAUSTED_ARM_INDEX] == 0.0


def test_generate_on_a_non_exhausted_arm_still_feeds_the_mab() -> None:
    """Control case: an arm with fresh records still updates its own score."""

    train_records, buffer, blocked = _exhausted_fixture()
    scenario_cfg = replace(_config(), exploit_probability=0.0)
    bandit = ScenarioArmBandit(replace(scenario_cfg.mab, eta=1e-12, temperature=0.01))
    bandit.scores = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    decision = select_acl_slot_decision(
        slot=0,
        episode_id=0,
        generation=0,
        excluded_scenario_uids=blocked,
        buffer=buffer,
        bandit=bandit,
        arms=ARMS,
        train_records=train_records,
        scenario_cfg=scenario_cfg,
        rng=np.random.default_rng(20260726),
        visit_state=_visit_state(),
    )

    assert decision.generate_arm_exhausted is False
    assert decision.selection.mode == "generate"
    assert decision.selection.arm_name == ARM_NAMES[0]
    assert decision.selection.selection_probability is not None

    before = float(bandit.scores[0])
    bandit.update(
        arm_index=decision.selection.arm_index,
        normalized_usefulness=0.0,
        selection_probability=decision.selection.selection_probability,
    )
    assert float(bandit.scores[0]) < before


def test_all_arms_exhausted_falls_back_to_replay_and_flags_exhaustion() -> None:
    """When every arm lacks an admissible record, Generate degrades to Replay.

    This is the only case that still degrades silently into a Replay. Since ACL
    v1.3 (`DEC-008`) it requires every record of every arm to be simultaneously
    quarantined or in flight -- buffer absorption, which used to be sufficient,
    now has no effect. The buffer is non-empty by construction, so Replay has
    something to fall back on. `generate_arm_exhausted` flags exactly this case.
    """

    train_records, buffer, _blocked = _exhausted_fixture()
    # Keep the rest of the catalog in the buffer too, to show that absorption
    # is not what produces the degradation any more.
    for record in train_records:
        if record.primary_arm != EXHAUSTED_ARM:
            buffer.insert(_buffer_record(record, usefulness=0.2))
    all_blocked = frozenset(record.scenario_uid for record in train_records)
    scenario_cfg = replace(_config(), exploit_probability=0.0)
    bandit = ScenarioArmBandit(scenario_cfg.mab)

    decision = select_acl_slot_decision(
        slot=0,
        episode_id=0,
        generation=0,
        excluded_scenario_uids=all_blocked,
        buffer=buffer,
        bandit=bandit,
        arms=ARMS,
        train_records=train_records,
        scenario_cfg=scenario_cfg,
        rng=np.random.default_rng(20260726),
        visit_state=_visit_state(),
    )

    assert decision.eligible_arm_mask is not None
    assert not decision.eligible_arm_mask.any()
    assert decision.generate_arm_exhausted is True
    assert decision.selection.mode == "replay"
    assert decision.selection.selection_probability is None
    assert decision.sampled_arm_index is None


def test_no_eligible_arm_with_an_empty_buffer_is_fatal() -> None:
    """The spec's `missing frozen record is fatal` path is preserved.

    Every arm must be ineligible (not just the one the bandit favours) and the
    buffer must be empty for Replay to have nothing left to fall back on.
    """

    only_arm = ARM_NAMES[0]
    train_records = tuple(_catalog_record(only_arm, index) for index in range(2))
    scenario_cfg = replace(_config(), exploit_probability=0.0, warmup_buffer_size=0)
    bandit = ScenarioArmBandit(scenario_cfg.mab)

    with pytest.raises(RuntimeError, match="No ACL arm has a fresh catalog record"):
        select_acl_slot_decision(
            slot=0,
            episode_id=0,
            generation=0,
            # The only arm with any catalog records at all is excluded here,
            # simulating both of its records already claimed elsewhere in the
            # same vectorized batch; every other arm has zero records ever.
            excluded_scenario_uids=frozenset(record.scenario_uid for record in train_records),
            buffer=ScenarioBuffer(capacity=10),
            bandit=bandit,
            arms=ARMS,
            train_records=train_records,
            scenario_cfg=scenario_cfg,
            rng=np.random.default_rng(20260726),
            visit_state=_visit_state(),
        )
