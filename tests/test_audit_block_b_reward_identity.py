"""Audit 2026-09-07, C29: the six-level weights belong to the checkpoint identity.

`build_reward_semantics_identity` carried `priority_base` but none of the weights
that sit beside it in the same formula. That cost nothing while every one of them
was frozen. ADR-081 makes `severity` a live parameter, and a checkpoint trained
at one severity would then resume against another without any complaint -- one
set of curves carrying two different rewards, which is the failure the sidecar
exists to prevent.
"""

from __future__ import annotations

import pytest

from thesis_rl.contracts.reward_semantics import build_reward_semantics_identity

_SIX_LEVEL_WEIGHTS = (
    "severity",
    "flat_tie_breaker",
    "progress_weight",
    "relaxable_weight",
    "progress_rate_weight",
    "step_dt_s",
    "reference_time_s",
)


def _config(**scalarization_overrides: object) -> dict:
    scalarization = {
        "specification_id": "SCAL-V1.4",
        "version": "1.4",
        "mode": "six_level_priority_weighted_rank",
        "vector_schema_id": "rulebook_v5_1_six_level_v1",
        "priority_base": 2.5,
        "severity": 0.30,
        "flat_tie_breaker": 0.25,
        "progress_weight": 2.0,
        "relaxable_weight": 1.0,
        "progress_rate_weight": 0.2,
        "step_dt_s": 0.1,
        "reference_time_s": 1.0,
        "numerical_tolerance": 1.0e-8,
        "native_environment_reward_weight": 0.0,
        "legacy": {"vector_schema_id": None, "rule_scales": None},
    }
    scalarization.update(scalarization_overrides)
    return {
        "reward": {"behavior": "scalar_reward"},
        "rulebook": {
            "implementation_family": "v2",
            "specification_id": "RULEBOOK-V5.1",
            "version": "v5.1",
        },
        "scalarization": scalarization,
    }


def test_every_six_level_weight_reaches_the_identity() -> None:
    identity = build_reward_semantics_identity(_config())
    assert identity is not None

    recorded = identity["scalarization"]
    for field in _SIX_LEVEL_WEIGHTS:
        assert field in recorded, f"{field} is in the reward formula but not in the identity"


@pytest.mark.parametrize("field", _SIX_LEVEL_WEIGHTS)
def test_changing_any_weight_changes_the_identity(field: str) -> None:
    """Otherwise a resume silently mixes two rewards in one run."""

    baseline = build_reward_semantics_identity(_config())
    changed = build_reward_semantics_identity(_config(**{field: 0.123}))

    assert baseline != changed, f"{field} does not reach the checkpoint identity"


def test_severity_is_the_field_that_made_this_urgent() -> None:
    """ADR-081 turned `sigma` from a frozen 0 into a selected 0.30.

    Called out separately from the parametrized case because the parametrization
    would still pass if someone later removed `severity` and left the other six.
    """

    at_zero = build_reward_semantics_identity(_config(severity=0.0))
    at_selected = build_reward_semantics_identity(_config(severity=0.30))

    assert at_zero is not None and at_selected is not None
    assert at_zero["scalarization"]["severity"] == 0.0
    assert at_selected["scalarization"]["severity"] == 0.30
    assert at_zero != at_selected


def test_an_undeclared_weight_is_recorded_as_undeclared_not_defaulted() -> None:
    """The `C8` rule: a missing key must not silently become a plausible value."""

    config = _config()
    del config["scalarization"]["severity"]

    identity = build_reward_semantics_identity(config)
    assert identity is not None
    assert identity["scalarization"]["severity"] is None
