"""ACL v1.3 REQ-009/REQ-005: Generate/catalog decoupling and coverage cycles.

Plan `ACL-SN-CAT-003`
(`docs/implementation/automatic_curriculum_learning_v1.3_exec_plan.md`),
decision record `docs/decisions/ADR-032-acl-generate-catalog-decoupling.md`.

`v1.1`/`v1.2` defined a *fresh* Generate candidate as a catalog record the
scenario buffer does not hold. On a frozen finite catalog that rule produced
two defects:

- `FIND-001`: an arm whose whole pool was absorbed by the buffer left the
  curriculum permanently (`tests/test_scenario_acl_arm_exhaustion.py` pins the
  `ADR-028` mitigation of the symptom);
- `FIND-005`: because the buffer retains high-learning-potential records by
  construction, the residual Generate pool of every arm converged to records
  the buffer had already rejected, so `q_i` estimated each arm's learning
  potential *conditioned on being below the eviction threshold*.

`DEC-008` removes the rule: Generate excludes only quarantined (`ADR-024`) and
in-flight (`ADR-016`) records. `DEC-009` adds per-arm coverage cycles so the
curated finite catalog is covered systematically instead of by coupon-collector
chance. These tests pin both.

Test IDs: `TEST-CAT-001`, `TEST-CAT-002`, `TEST-CAT-003`, `TEST-CAT-005`,
`TEST-CAT-007`, `TEST-CAT-009`, `TEST-CAT-011`, `TEST-CAT-012`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest

from thesis_rl.curriculum.config import ScenarioAclConfig, ScenarioAclMabConfig
from thesis_rl.curriculum.scenario_acl.arms import build_default_scenario_arms
from thesis_rl.curriculum.scenario_acl.buffer import ScenarioBuffer
from thesis_rl.curriculum.scenario_acl.catalog_state import (
    COVERAGE_STATE_SCHEMA,
    ScenarioCatalogVisitState,
)
from thesis_rl.curriculum.scenario_acl.mab import ScenarioArmBandit
from thesis_rl.curriculum.scenario_acl.record import ScenarioRecord
from thesis_rl.curriculum.scenario_acl.selection import (
    eligible_generate_arm_mask,
    fresh_generate_candidates,
    select_acl_slot_decision,
)

ARMS = build_default_scenario_arms()
ARM_NAMES = tuple(arm.name for arm in ARMS)
POOL_PER_ARM = 5


@dataclass(frozen=True)
class _CatalogRecord:
    """Minimal stand-in for a frozen ScenarioNet catalog record."""

    scenario_uid: str
    primary_arm: str
    runtime_index: int
    source: str


def _catalog(pool_per_arm: int = POOL_PER_ARM) -> tuple[_CatalogRecord, ...]:
    return tuple(
        _CatalogRecord(
            scenario_uid=f"waymo:train:{arm}:{index:04d}",
            primary_arm=arm,
            runtime_index=index,
            source="waymo",
        )
        for arm in ARM_NAMES
        for index in range(pool_per_arm)
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


def _config(**overrides: object) -> ScenarioAclConfig:
    """The approved v1.3 defaults, with the Generate branch forced by default."""

    base = ScenarioAclConfig(
        buffer_capacity=250,
        warmup_buffer_size=100,
        generate_probability=0.4,
        exploit_probability=0.0,
        use_mab=True,
        use_replay=True,
        use_staleness=True,
        mab=ScenarioAclMabConfig(),
    )
    return replace(base, **overrides) if overrides else base


def _visit_state() -> ScenarioCatalogVisitState:
    return ScenarioCatalogVisitState(ARM_NAMES)


def _decide(
    *,
    train_records: tuple[_CatalogRecord, ...],
    buffer: ScenarioBuffer,
    bandit: ScenarioArmBandit,
    visit_state: ScenarioCatalogVisitState,
    rng: np.random.Generator,
    episode_id: int = 0,
    scenario_cfg: ScenarioAclConfig | None = None,
    excluded: frozenset[str] = frozenset(),
):
    return select_acl_slot_decision(
        slot=0,
        episode_id=episode_id,
        generation=episode_id,
        excluded_scenario_uids=excluded,
        buffer=buffer,
        bandit=bandit,
        arms=ARMS,
        train_records=train_records,
        scenario_cfg=scenario_cfg or _config(),
        rng=rng,
        visit_state=visit_state,
    )


# --- TEST-CAT-001 -----------------------------------------------------------


def test_generate_candidates_ignore_buffer_membership() -> None:
    """`TEST-CAT-001` / `AC-009`: candidates and eligibility are buffer-invariant.

    The same catalog is evaluated against an empty buffer, a half-filled one,
    and one holding an arm's entire pool. `DEC-008` requires all three to give
    identical candidate sets and an identical eligibility mask, because buffer
    membership is no longer part of Generate admissibility.
    """

    train_records = _catalog()
    arm = ARM_NAMES[0]
    empty = ScenarioBuffer(capacity=250)
    half = ScenarioBuffer(capacity=250)
    absorbed = ScenarioBuffer(capacity=250)
    for record in train_records:
        if record.primary_arm != arm:
            continue
        absorbed.insert(_buffer_record(record, usefulness=0.9))
        if record.runtime_index < POOL_PER_ARM // 2:
            half.insert(_buffer_record(record, usefulness=0.9))
    assert len(absorbed) == POOL_PER_ARM

    expected_uids = sorted(
        record.scenario_uid for record in train_records if record.primary_arm == arm
    )
    expected_mask = [True] * len(ARM_NAMES)
    for buffer in (empty, half, absorbed):
        # The exclusion set is exactly what the driver supplies: quarantine and
        # in-flight UIDs. The buffer is deliberately not consulted.
        del buffer
        candidates = fresh_generate_candidates(
            train_records, arm_name=arm, excluded_scenario_uids=frozenset()
        )
        assert sorted(record.scenario_uid for record in candidates) == expected_uids
        mask = eligible_generate_arm_mask(
            train_records, arms=ARMS, excluded_scenario_uids=frozenset()
        )
        assert mask.tolist() == expected_mask


# --- TEST-CAT-002 -----------------------------------------------------------


def test_fully_buffered_arm_still_generates_and_updates_ema() -> None:
    """`TEST-CAT-002` / `AC-010`: `FIND-001`'s absorbing state cannot form.

    The buffer holds every record of the arm the bandit favours. Pre-`DEC-008`
    that arm was ineligible and its EMA froze; now it is drawn, generates, and
    its score moves on commit.
    """

    train_records = _catalog()
    arm = ARM_NAMES[4]
    arm_index = ARM_NAMES.index(arm)
    buffer = ScenarioBuffer(capacity=250)
    for record in train_records:
        if record.primary_arm == arm:
            buffer.insert(_buffer_record(record, usefulness=0.9))

    scenario_cfg = _config()
    # Near-deterministic bandit: the favoured arm is the fully buffered one.
    bandit = ScenarioArmBandit(replace(scenario_cfg.mab, eta=1e-12, temperature=0.01))
    bandit.scores = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)
    before = float(bandit.scores[arm_index])

    decision = _decide(
        train_records=train_records,
        buffer=buffer,
        bandit=bandit,
        visit_state=_visit_state(),
        rng=np.random.default_rng(20260729),
    )

    assert decision.selection.mode == "generate"
    assert decision.selection.arm_name == arm
    assert decision.eligible_arm_mask is not None
    assert bool(decision.eligible_arm_mask[arm_index]) is True
    assert decision.generate_arm_exhausted is False
    # The drawn record is one the buffer already holds -- which is exactly the
    # case `v1.2` made impossible.
    assert any(item.scenario_id == decision.selection.scenario_uid for item in buffer.records())

    bandit.update(
        arm_index=decision.selection.arm_index,
        normalized_usefulness=0.0,
        selection_probability=decision.selection.selection_probability,
    )
    assert float(bandit.scores[arm_index]) < before


# --- TEST-CAT-003 -----------------------------------------------------------


def test_coverage_cycle_visits_each_record_once_then_restarts() -> None:
    """`TEST-CAT-003` / `AC-011`: sampling without replacement inside a cycle.

    With a single eligible arm, the first `N` Generate draws must be a
    permutation of that arm's pool. Draw `N+1` closes the cycle, increments the
    arm's counter, clears its visited set, and starts a fresh pass. Other arms'
    counters are untouched: cycles are per arm, with no global cycle.
    """

    arm = ARM_NAMES[2]
    train_records = tuple(record for record in _catalog() if record.primary_arm == arm)
    scenario_cfg = _config()
    bandit = ScenarioArmBandit(scenario_cfg.mab)
    visit_state = _visit_state()
    buffer = ScenarioBuffer(capacity=250)
    rng = np.random.default_rng(7)

    drawn: list[str] = []
    closures: list[bool] = []
    for episode_id in range(POOL_PER_ARM + 2):
        decision = _decide(
            train_records=train_records,
            buffer=buffer,
            bandit=bandit,
            visit_state=visit_state,
            rng=rng,
            episode_id=episode_id,
        )
        assert decision.selection.mode == "generate"
        assert decision.selection.arm_name == arm
        drawn.append(str(decision.selection.scenario_uid))
        closures.append(decision.coverage_cycle_closed)

    first_cycle = drawn[:POOL_PER_ARM]
    assert sorted(first_cycle) == sorted(record.scenario_uid for record in train_records)
    assert len(set(first_cycle)) == POOL_PER_ARM
    assert closures[:POOL_PER_ARM] == [False] * POOL_PER_ARM
    # The first draw of the second cycle is the one that closes the first.
    assert closures[POOL_PER_ARM] is True
    assert closures[POOL_PER_ARM + 1] is False
    assert visit_state.cycle_id(arm) == 1
    assert visit_state.visited_count(arm) == 2
    for other in ARM_NAMES:
        if other == arm:
            continue
        assert visit_state.cycle_id(other) == 0
        assert visit_state.visited_count(other) == 0


def test_close_cycle_affects_only_the_given_arm() -> None:
    """`TEST-CAT-003` (state level): `close_cycle` is per arm."""

    visit_state = _visit_state()
    visit_state.mark_visited("uid-a", arm_name=ARM_NAMES[0], episode_id=1)
    visit_state.mark_visited("uid-b", arm_name=ARM_NAMES[1], episode_id=2)

    assert visit_state.close_cycle(ARM_NAMES[0]) == 1
    assert visit_state.visited_count(ARM_NAMES[0]) == 0
    assert visit_state.cycle_id(ARM_NAMES[1]) == 0
    assert visit_state.is_visited("uid-b", arm_name=ARM_NAMES[1]) is True
    # Lifetime diagnostics survive the cycle boundary; only `V_i` is cleared.
    assert visit_state.num_generate_visits("uid-a") == 1
    assert visit_state.last_generate_episode_id("uid-a") == 1


# --- TEST-CAT-005 -----------------------------------------------------------


def test_quarantined_and_in_flight_records_are_excluded_without_blocking_closure() -> None:
    """`TEST-CAT-005` / REQ-009: the only two admissible-set exclusions.

    Quarantined and in-flight records never appear as candidates, and the
    coverage cycle closes over the *remaining* admissible records rather than
    waiting for records it can never draw.
    """

    arm = ARM_NAMES[1]
    train_records = tuple(record for record in _catalog() if record.primary_arm == arm)
    blocked = frozenset(
        {train_records[0].scenario_uid, train_records[1].scenario_uid}
    )  # one quarantined, one in flight -- indistinguishable at this interface
    admissible = sorted(
        record.scenario_uid for record in train_records if record.scenario_uid not in blocked
    )

    scenario_cfg = _config()
    bandit = ScenarioArmBandit(scenario_cfg.mab)
    visit_state = _visit_state()
    buffer = ScenarioBuffer(capacity=250)
    rng = np.random.default_rng(11)

    drawn: list[str] = []
    for episode_id in range(len(admissible) + 1):
        decision = _decide(
            train_records=train_records,
            buffer=buffer,
            bandit=bandit,
            visit_state=visit_state,
            rng=rng,
            episode_id=episode_id,
            excluded=blocked,
        )
        drawn.append(str(decision.selection.scenario_uid))

    assert not blocked.intersection(drawn)
    assert sorted(set(drawn[: len(admissible)])) == admissible
    # The cycle closed on the admissible subset, not on the full pool.
    assert visit_state.cycle_id(arm) == 1


def test_records_blocked_at_closure_stay_visited_in_the_next_cycle() -> None:
    """`TEST-CAT-005`: closing a cycle clears `V_i` for the whole arm.

    A record that was in flight while the cycle closed becomes drawable again
    in the new cycle, which is correct: the new cycle is a fresh pass over the
    arm's admissible pool.
    """

    arm = ARM_NAMES[3]
    visit_state = _visit_state()
    visit_state.mark_visited("uid-x", arm_name=arm, episode_id=0)
    visit_state.close_cycle(arm)
    assert visit_state.is_visited("uid-x", arm_name=arm) is False


# --- TEST-CAT-006 -----------------------------------------------------------


def test_data_abort_keeps_the_record_visited_and_removes_only_the_buffer_reference() -> None:
    """`TEST-CAT-006` / REQ-009 + `ADR-024`.

    A typed runtime data-abort produces no learning potential, no MAB update
    and no buffer scoring. It quarantines the record and drops its buffer
    reference, but the record stays marked as visited in the arm's current
    cycle -- so the cycle can still close and the unusable record is not
    re-drawn before the next cycle.

    This reproduces the driver's abort handling (`buffer.remove_scenario_id`
    plus the quarantine UID entering the exclusion set) at the level the
    `commit_event` closure is not reachable from (`LIM-004`).
    """

    arm = ARM_NAMES[0]
    train_records = tuple(record for record in _catalog() if record.primary_arm == arm)
    scenario_cfg = _config()
    bandit = ScenarioArmBandit(scenario_cfg.mab)
    visit_state = _visit_state()
    buffer = ScenarioBuffer(capacity=250)
    for record in train_records:
        buffer.insert(_buffer_record(record, usefulness=0.5))

    decision = _decide(
        train_records=train_records,
        buffer=buffer,
        bandit=bandit,
        visit_state=visit_state,
        rng=np.random.default_rng(17),
    )
    aborted_uid = str(decision.selection.scenario_uid)
    assert visit_state.is_visited(aborted_uid, arm_name=arm) is True

    # Driver abort handling: quarantine the UID and drop the buffer reference.
    assert buffer.remove_scenario_id(aborted_uid) is True
    quarantined = frozenset({aborted_uid})
    assert visit_state.is_visited(aborted_uid, arm_name=arm) is True
    assert not any(item.scenario_id == aborted_uid for item in buffer.records())

    # The record is never drawn again, and the cycle still closes over the rest:
    # the remaining admissible records are covered, then the next draw finds an
    # empty candidate set and restarts the cycle instead of deadlocking on the
    # quarantined record it can never draw.
    drawn = []
    for episode_id in range(1, len(train_records) + 1):
        drawn.append(
            str(
                _decide(
                    train_records=train_records,
                    buffer=buffer,
                    bandit=bandit,
                    visit_state=visit_state,
                    rng=np.random.default_rng(100 + episode_id),
                    episode_id=episode_id,
                    excluded=quarantined,
                ).selection.scenario_uid
            )
        )
    assert aborted_uid not in drawn
    assert visit_state.cycle_id(arm) == 1


# --- TEST-CAT-007 -----------------------------------------------------------


def test_coverage_state_round_trip() -> None:
    """`TEST-CAT-007` / `AC-005`: persistence round-trip is exact."""

    visit_state = _visit_state()
    visit_state.mark_visited("uid-a", arm_name=ARM_NAMES[0], episode_id=3)
    visit_state.mark_visited("uid-b", arm_name=ARM_NAMES[0], episode_id=4)
    visit_state.mark_visited("uid-c", arm_name=ARM_NAMES[5], episode_id=5)
    visit_state.close_cycle(ARM_NAMES[5])
    visit_state.mark_visited("uid-c", arm_name=ARM_NAMES[5], episode_id=6)

    payload = visit_state.state_dict()
    assert payload["schema"] == COVERAGE_STATE_SCHEMA
    restored = ScenarioCatalogVisitState.from_state_dict(payload)

    assert restored.state_dict() == payload
    assert restored.arm_names == visit_state.arm_names
    for arm in ARM_NAMES:
        assert restored.cycle_id(arm) == visit_state.cycle_id(arm)
        assert restored.visited_uids(arm) == visit_state.visited_uids(arm)
    assert restored.num_generate_visits("uid-c") == 2
    assert restored.last_generate_episode_id("uid-c") == 6


def test_coverage_state_rejects_foreign_schema() -> None:
    """`TEST-CAT-007` / REQ-005: no silent reinterpretation of older state."""

    with pytest.raises(ValueError, match="acl_coverage_v1"):
        ScenarioCatalogVisitState.from_state_dict({"schema": "acl_coverage_v0", "arm_names": []})


def test_coverage_state_rejects_unknown_arm() -> None:
    visit_state = _visit_state()
    with pytest.raises(KeyError):
        visit_state.mark_visited("uid-a", arm_name="A9_nonexistent", episode_id=0)


# --- TEST-CAT-009 -----------------------------------------------------------


def test_no_admissible_record_degrades_to_replay_then_fails_when_buffer_is_empty() -> None:
    """`TEST-CAT-009` / REQ-003: the residual degradation path is preserved.

    After `DEC-008` this can only happen when every record of every arm is
    simultaneously quarantined or in flight -- not merely buffered.
    """

    train_records = _catalog(pool_per_arm=1)
    all_blocked = frozenset(record.scenario_uid for record in train_records)
    scenario_cfg = _config(warmup_buffer_size=0)
    bandit = ScenarioArmBandit(scenario_cfg.mab)

    populated = ScenarioBuffer(capacity=250)
    populated.insert(_buffer_record(train_records[0], usefulness=0.5))
    decision = _decide(
        train_records=train_records,
        buffer=populated,
        bandit=bandit,
        visit_state=_visit_state(),
        rng=np.random.default_rng(3),
        scenario_cfg=scenario_cfg,
        excluded=all_blocked,
    )
    assert decision.generate_arm_exhausted is True
    assert decision.selection.mode == "replay"

    with pytest.raises(RuntimeError, match="No ACL arm has a fresh catalog record"):
        _decide(
            train_records=train_records,
            buffer=ScenarioBuffer(capacity=250),
            bandit=bandit,
            visit_state=_visit_state(),
            rng=np.random.default_rng(3),
            scenario_cfg=scenario_cfg,
            excluded=all_blocked,
        )


# --- TEST-CAT-011 -----------------------------------------------------------


def test_seeded_selection_sequence_is_deterministic() -> None:
    """`TEST-CAT-011` / REQ-005, `ADR-016`: same seed, same sequence and counters."""

    def run() -> tuple[list[tuple[str, str, str]], dict[str, dict[str, int]]]:
        train_records = _catalog()
        scenario_cfg = _config()
        bandit = ScenarioArmBandit(scenario_cfg.mab)
        visit_state = _visit_state()
        buffer = ScenarioBuffer(capacity=250)
        rng = np.random.default_rng(20260729)
        trace: list[tuple[str, str, str]] = []
        for episode_id in range(40):
            decision = _decide(
                train_records=train_records,
                buffer=buffer,
                bandit=bandit,
                visit_state=visit_state,
                rng=rng,
                episode_id=episode_id,
            )
            selection = decision.selection
            trace.append((selection.mode, str(selection.arm_name), str(selection.scenario_uid)))
            if selection.mode == "generate":
                bandit.update(
                    arm_index=selection.arm_index,
                    normalized_usefulness=0.5,
                    selection_probability=selection.selection_probability,
                )
        return trace, visit_state.coverage_summary()

    first_trace, first_coverage = run()
    second_trace, second_coverage = run()
    assert first_trace == second_trace
    assert first_coverage == second_coverage
    # The run must actually exercise at least one cycle closure to be meaningful.
    assert any(entry["cycle_id"] > 0 for entry in first_coverage.values())


# --- TEST-CAT-012 -----------------------------------------------------------


def test_adr_028_renormalization_still_applies_to_a_genuinely_unavailable_arm() -> None:
    """`TEST-CAT-012` / REQ-002: `ADR-028` survives as a safety net.

    `DEC-008` narrows the condition that makes an arm ineligible but does not
    remove the renormalization. With one arm entirely quarantined, that arm
    must get exactly zero probability and the `eta/K` floor must be
    renormalized over the remaining five.
    """

    train_records = _catalog()
    unavailable = ARM_NAMES[3]
    unavailable_index = ARM_NAMES.index(unavailable)
    quarantined = frozenset(
        record.scenario_uid for record in train_records if record.primary_arm == unavailable
    )

    mask = eligible_generate_arm_mask(train_records, arms=ARMS, excluded_scenario_uids=quarantined)
    assert mask.tolist() == [name != unavailable for name in ARM_NAMES]

    scenario_cfg = _config()
    bandit = ScenarioArmBandit(scenario_cfg.mab)
    probabilities = bandit.probabilities(mask)
    eta = float(scenario_cfg.mab.eta)
    assert probabilities[unavailable_index] == 0.0
    assert np.isclose(probabilities.sum(), 1.0)
    for index, name in enumerate(ARM_NAMES):
        if name == unavailable:
            continue
        assert probabilities[index] >= eta / 5.0 - 1e-12

    decision = _decide(
        train_records=train_records,
        buffer=ScenarioBuffer(capacity=250),
        bandit=bandit,
        visit_state=_visit_state(),
        rng=np.random.default_rng(5),
        excluded=quarantined,
    )
    assert decision.selection.mode == "generate"
    assert decision.selection.arm_name != unavailable
    assert decision.selection.scenario_uid not in quarantined
