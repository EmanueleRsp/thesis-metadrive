from __future__ import annotations

import json

import numpy as np
import pytest

from thesis_rl.curriculum.scenario_acl.vectorized import (
    VECTOR_STATE_VERSION,
    AclCompletion,
    AclEpisodeAccumulator,
    AclSlotSelection,
    AclVectorState,
    AclVectorSelectionCoordinator,
    AclVectorTransaction,
    derive_worker_rng,
    load_acl_vector_state,
    order_acl_completions,
    retain_unresolved_acl_completions,
    save_acl_vector_state,
    validate_fresh_batch_unique,
)
from thesis_rl.agent.planners.core.lifecycle import _DelegatingLifecycle


def _selection(slot_id: int, *, uid: str, mode: str = "generate") -> AclSlotSelection:
    return AclSlotSelection(
        slot_id=slot_id,
        episode_id=slot_id,
        generation=slot_id,
        mode=mode,
        arm_index=slot_id,
        arm_name=f"A{slot_id}",
        reset_seed=100 + slot_id,
        scenario_uid=uid,
    )


def test_acl_vector_state_round_trip_and_env_count_guard(tmp_path) -> None:
    selection = _selection(0, uid="s0")
    state = AclVectorState(
        n_envs=2,
        collection_tick=4,
        active_selections={0: selection},
        accumulators={0: AclEpisodeAccumulator(slot_id=0, episode_id=0)},
        pending_completions=[
            AclCompletion(
                collection_tick=3,
                worker_id=0,
                episode_id=0,
                selection=selection,
                metrics={"reward": 1.0},
            )
        ],
        last_observations=np.asarray([[1.0, 2.0]], dtype=np.float32),
        rng_state={"bit_generator": "PCG64"},
    )
    path = tmp_path / "acl_vector_state.json"
    save_acl_vector_state(path, state)
    restored = load_acl_vector_state(path, expected_n_envs=2)
    assert restored.to_dict() == state.to_dict()
    assert np.array_equal(restored.last_observations, state.last_observations)
    with pytest.raises(ValueError, match="n_envs"):
        load_acl_vector_state(path, expected_n_envs=3)


def test_acl_vector_state_persists_runtime_quarantine_across_resume(tmp_path) -> None:
    state = AclVectorState(
        n_envs=2,
        quarantined_scenario_uids=["waymo:broken", "pg:broken"],
    )
    path = tmp_path / "acl_vector_state.json"
    save_acl_vector_state(path, state)

    restored = load_acl_vector_state(path, expected_n_envs=2)
    assert restored.quarantined_scenario_uids == ["pg:broken", "waymo:broken"]


def test_acl_vector_state_rejects_a_mismatched_version(tmp_path) -> None:
    state = AclVectorState(n_envs=2)
    path = tmp_path / "acl_vector_state.json"
    save_acl_vector_state(path, state)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["version"] = VECTOR_STATE_VERSION - 1
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="incompatible"):
        load_acl_vector_state(path, expected_n_envs=2)


def test_fresh_duplicates_are_rejected_but_replay_duplicates_are_allowed() -> None:
    with pytest.raises(ValueError, match="Duplicate fresh"):
        validate_fresh_batch_unique([_selection(0, uid="same"), _selection(1, uid="same")])
    validate_fresh_batch_unique(
        [_selection(0, uid="same", mode="replay"), _selection(1, uid="same", mode="replay")]
    )


def test_completion_order_does_not_depend_on_process_arrival_order() -> None:
    completions = [
        {"collection_tick": 2, "worker_id": 1},
        {"collection_tick": 1, "worker_id": 1},
        {"collection_tick": 1, "worker_id": 0},
    ]
    assert [(x["collection_tick"], x["worker_id"]) for x in order_acl_completions(completions)] == [
        (1, 0),
        (1, 1),
        (2, 1),
    ]


def test_completion_order_uses_episode_id_only_for_same_slot_tick_ties() -> None:
    completions = [
        {"collection_tick": 1, "worker_id": 0, "episode_id": 3},
        {"collection_tick": 1, "worker_id": 0, "episode_id": 2},
    ]
    assert [item["episode_id"] for item in order_acl_completions(completions)] == [2, 3]


def test_unresolved_completion_is_not_duplicated_when_lp_arrives_late() -> None:
    selection = _selection(0, uid="s0")
    completion = AclCompletion(0, 0, 0, selection)
    assert retain_unresolved_acl_completions(
        [completion],
        [completion],
        resolved_keys=(),
    ) == [completion]


def test_parent_transaction_commits_per_episode_lp_in_stable_order() -> None:
    state = AclVectorState(n_envs=2)
    transaction = AclVectorTransaction(state)
    first = _selection(0, uid="s0")
    second = _selection(1, uid="s1")
    completions = [
        AclCompletion(2, 1, 1, second),
        AclCompletion(2, 0, 0, first),
    ]
    committed: list[tuple[int, float]] = []
    events = transaction.commit_tick(
        completions,
        learning_potentials={(0, 0): 1.0, (1, 1): 9.0},
        commit=lambda event: committed.append((event["worker_id"], event["learning_potential"])),
    )
    assert committed == [(0, 1.0), (1, 9.0)]
    assert [event["learning_potential"] for event in events] == [1.0, 9.0]
    assert state.collection_tick == 3
    assert [
        (x.worker_id, x.selection.scenario_uid) for x in order_acl_completions(completions)
    ] == [
        (0, "s0"),
        (1, "s1"),
    ]


def test_selection_coordinator_sorts_slots_and_rejects_fresh_duplicates() -> None:
    state = AclVectorState(n_envs=2)
    coordinator = AclVectorSelectionCoordinator(state)
    calls: list[tuple[int, frozenset[str]]] = []

    def selector(slot: int, excluded: frozenset[str]) -> AclSlotSelection:
        calls.append((slot, excluded))
        return _selection(slot, uid=f"scenario-{slot}")

    selected = coordinator.select_batch([1, 0], selector=selector)
    assert list(selected) == [0, 1]
    assert calls == [(0, frozenset()), (1, frozenset({"scenario-0"}))]

    with pytest.raises(ValueError, match="Duplicate fresh"):
        coordinator.select_batch(
            [0, 1],
            selector=lambda slot, _excluded: AclSlotSelection(
                slot_id=slot,
                episode_id=slot + 2,
                generation=2,
                mode="generate",
                arm_index=slot,
                arm_name=f"A{slot}",
                reset_seed=200 + slot,
                scenario_uid="same",
            ),
        )


def test_resume_restarts_persisted_slots_before_the_first_step() -> None:
    state = AclVectorState(
        n_envs=2,
        active_selections={0: _selection(0, uid="s0"), 1: _selection(1, uid="s1")},
    )
    coordinator = AclVectorSelectionCoordinator(state)

    class Env:
        acl_mode = True

        def __init__(self) -> None:
            self.configured: list[tuple[int, dict]] = []
            self.reset_calls: list[tuple[list[int], dict[int, int]]] = []

        def configure_acl_selection(self, slot: int, selection: dict) -> None:
            self.configured.append((slot, selection))

        def reset_slots(self, slots: list[int], *, seeds: dict[int, int]):
            self.reset_calls.append((slots, seeds))
            return {
                slot: (np.asarray([seeds[slot]], dtype=np.float32), {"reset_seed": seeds[slot]})
                for slot in slots
            }

    env = Env()
    results = coordinator.restart_active_slots_for_resume(env)

    assert [slot for slot, _selection in env.configured] == [0, 1]
    assert env.reset_calls == [([0, 1], {0: 100, 1: 101})]
    assert [int(results[slot][0][0]) for slot in (0, 1)] == [100, 101]


def test_td3_and_sac_lp_stays_with_episode_accumulator() -> None:
    first = AclEpisodeAccumulator(slot_id=0, episode_id=1)
    second = AclEpisodeAccumulator(slot_id=1, episode_id=1)
    first.add_transition(reward=0.0, done=False, td3_residual=2.0, sac_residual=-4.0)
    second.add_transition(reward=0.0, done=True, td3_residual=10.0, sac_residual=1.0)
    assert first.learning_potential("td3_sb3") == pytest.approx(2.0)
    assert second.learning_potential("sac_sb3") == pytest.approx(1.0)


def test_ppo_partial_episode_is_carried_until_completion() -> None:
    accumulator = AclEpisodeAccumulator(slot_id=0, episode_id=1)
    accumulator.add_transition(reward=1.0, done=False, value=0.0, next_value=0.0)
    partial = AclEpisodeAccumulator.from_dict(accumulator.to_dict())
    partial.add_transition(reward=2.0, done=True, value=0.0, next_value=0.0)
    assert partial.learning_potential("ppo_sb3") > 0.0


def test_lifecycle_assigns_collection_residuals_to_slot_and_episode() -> None:
    class Backend:
        def observe_transition_batch(self, **_kwargs):
            return None

        def collection_learning_potential_batch(self, **_kwargs):
            return np.asarray([2.0, 10.0], dtype=np.float32)

    lifecycle = _DelegatingLifecycle(Backend())
    lifecycle.observe_transition_batch(
        observations=np.zeros((2, 1), dtype=np.float32),
        buffer_actions=np.zeros((2, 1), dtype=np.float32),
        rewards=np.zeros(2, dtype=np.float32),
        dones=np.asarray([False, True]),
        next_observations=np.zeros((2, 1), dtype=np.float32),
        infos=[
            {"acl_slot_id": 0, "acl_episode_id": 7},
            {"acl_slot_id": 1, "acl_episode_id": 3},
        ],
    )
    assert lifecycle.acl_learning_potential(0, 7) == pytest.approx(2.0)
    assert lifecycle.acl_learning_potential(1, 3) == pytest.approx(10.0)
    assert lifecycle.acl_learning_potential(0, 3) is None


def test_worker_rng_derivation_is_stable() -> None:
    left = derive_worker_rng(7, 1).random(4)
    right = derive_worker_rng(7, 1).random(4)
    assert np.array_equal(left, right)
