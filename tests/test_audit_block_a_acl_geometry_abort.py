"""Audit 2026-09-06, block A2: a geometry abort must not crash the ACL driver.

The vectorized ACL episode-end callback used to recognise only the scenario
data abort; a geometry abort carries no ``acl_episode_id`` either, so
``int(payload["episode_id"])`` raised ``TypeError`` and the run died.
"""

from __future__ import annotations

from thesis_rl.curriculum.scenario_acl.driver import aborted_acl_slots


def test_geometry_abort_is_classified_like_a_data_abort() -> None:
    payloads = [
        {"worker_id": 0, "episode_id": 7, "info": {"acl_episode_id": 7}},
        {"worker_id": 1, "episode_id": None, "info": {"runtime_scenario_data_abort": True}},
        {"worker_id": 2, "episode_id": None, "info": {"runtime_geometry_abort": True}},
    ]

    assert aborted_acl_slots(payloads) == {1, 2}


def test_any_payload_without_an_episode_id_is_treated_as_aborted() -> None:
    payloads = [{"worker_id": 3, "episode_id": None, "info": {}}]

    assert aborted_acl_slots(payloads) == {3}


def test_ordinary_completions_are_not_aborted() -> None:
    payloads = [{"worker_id": 4, "episode_id": 1, "info": {"acl_episode_id": 1}}]

    assert aborted_acl_slots(payloads) == set()
