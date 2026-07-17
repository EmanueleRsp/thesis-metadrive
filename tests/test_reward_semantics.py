"""Runtime checkpoint compatibility tests for reward semantics."""

from __future__ import annotations

from pathlib import Path

import pytest

from thesis_rl.agent.agent import Agent
from thesis_rl.contracts.reward_semantics import (
    RewardSemanticsCompatibilityError,
    assert_reward_semantics_compatible,
    build_reward_semantics_identity,
    reward_semantics_sidecar_path,
)


def _config(mode: str = "bounded_satisfaction_rank") -> dict[str, object]:
    return {
        "reward": {"behavior": "scalar_reward"},
        "rulebook": {
            "implementation_family": "v2",
            "specification_id": "RULEBOOK-V4.7",
            "version": "4.7-final-implementation-complete",
        },
        "scalarization": {
            "specification_id": "SCAL-V1.0",
            "version": "1.0",
            "mode": mode,
            "vector_schema_id": "rulebook_v2_macro_v4",
            "priority_base": 2.01,
            "numerical_tolerance": 1.0e-8,
            "native_environment_reward_weight": 0.0,
            "sigmoid": {"sharpness": 30.0},
            "legacy": {"vector_schema_id": None, "rule_scales": None},
        },
    }


def test_agent_save_writes_reward_semantics_before_resume_validation(tmp_path: Path) -> None:
    class Planner:
        def save(self, path: str | Path) -> None:
            Path(path).write_bytes(b"model")

    identity = build_reward_semantics_identity(_config())
    agent = Agent(preprocessor=object(), planner=Planner(), adapter=object())
    agent.set_checkpoint_identity(identity)
    checkpoint = tmp_path / "latest"
    agent.save(checkpoint)

    sidecar = reward_semantics_sidecar_path(checkpoint)
    assert sidecar.exists()
    assert_reward_semantics_compatible(checkpoint, identity)


def test_resume_rejects_changed_scalarization_before_loading(tmp_path: Path) -> None:
    class Planner:
        def save(self, path: str | Path) -> None:
            Path(path).write_bytes(b"model")

    checkpoint = tmp_path / "latest"
    identity = build_reward_semantics_identity(_config())
    changed = build_reward_semantics_identity(_config("bounded_centered_sigmoid"))
    agent = Agent(preprocessor=object(), planner=Planner(), adapter=object())
    agent.set_checkpoint_identity(identity)
    agent.save(checkpoint)

    with pytest.raises(RewardSemanticsCompatibilityError, match="mismatch"):
        assert_reward_semantics_compatible(checkpoint, changed)


def test_non_scalar_runs_do_not_require_scalarization_sidecars(tmp_path: Path) -> None:
    checkpoint = tmp_path / "monitor_only"
    assert build_reward_semantics_identity({"reward": {"behavior": "monitor_only"}}) is None
    assert_reward_semantics_compatible(checkpoint, None)
