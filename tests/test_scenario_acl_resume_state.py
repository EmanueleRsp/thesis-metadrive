"""The Scenario ACL resume state: what a chunk boundary writes, and what a resume reads.

The two chunk-end writers had diverged. The vectorized one -- the production
path, since every serious profile runs 20 environments -- omitted
`recent_usefulness`, and the loader's `.get(..., [])` turned the omission into an
empty window without complaint. These tests pin the contract from both ends: the
payload carries every field the loader reads, and the window is one of them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from thesis_rl.curriculum.config import ScenarioAclConfig
from thesis_rl.curriculum.scenario_acl.arms import SCENARIO_ARM_NAMES
from thesis_rl.curriculum.scenario_acl.buffer import ScenarioBuffer
from thesis_rl.curriculum.scenario_acl.catalog_state import ScenarioCatalogVisitState
from thesis_rl.curriculum.scenario_acl.driver import (
    _load_scenario_acl_resume_state,
    _normalize_learning_potential,
    _scenario_acl_resume_state,
)
from thesis_rl.curriculum.scenario_acl.mab import ScenarioArmBandit

_WINDOW = (0.125, 0.5, 0.875)


def _write_resume_artifacts(run_dir: Path, state: dict[str, object]) -> None:
    curriculum_dir = run_dir / "artifacts" / "curriculum"
    curriculum_dir.mkdir(parents=True, exist_ok=True)
    (curriculum_dir / "scenario_acl_state.json").write_text(
        json.dumps(state, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    (curriculum_dir / "scenario_buffer.json").write_text(
        json.dumps(ScenarioBuffer(capacity=250).state_dict(), ensure_ascii=True),
        encoding="utf-8",
    )
    (curriculum_dir / "scenario_coverage_state.json").write_text(
        json.dumps(ScenarioCatalogVisitState(SCENARIO_ARM_NAMES).state_dict(), ensure_ascii=True),
        encoding="utf-8",
    )


def _resume_cfg(run_dir: Path):
    return OmegaConf.create({"checkpoint": {"resume": {"enabled": True, "run_dir": str(run_dir)}}})


def test_resume_state_round_trips_every_field_the_loader_reads(tmp_path: Path) -> None:
    """One payload builder, so a writer cannot drop a field the loader needs.

    Each value here is distinctive, so a field silently lost between the two
    sides shows up as that field's default rather than as a passing test.
    """

    scenario_cfg = ScenarioAclConfig()
    bandit = ScenarioArmBandit(scenario_cfg.mab)
    bandit.update(arm_index=2, normalized_usefulness=0.75)
    written_rng = np.random.default_rng(20260907)

    state = _scenario_acl_resume_state(
        global_step=17_000,
        chunk_id=5,
        eval_id=3,
        episode_id=41,
        buffer=ScenarioBuffer(capacity=250),
        bandit=bandit,
        visit_state=ScenarioCatalogVisitState(SCENARIO_ARM_NAMES),
        rng=written_rng,
        recent_usefulness=_WINDOW,
    )
    _write_resume_artifacts(tmp_path, state)

    resumed_rng = np.random.default_rng(1)
    _, resumed_bandit, _, global_step, chunk_id, eval_id, episode_id, window = (
        _load_scenario_acl_resume_state(
            cfg=_resume_cfg(tmp_path),
            artifact_paths={"root": tmp_path},
            scenario_cfg=scenario_cfg,
            arm_names=SCENARIO_ARM_NAMES,
            rng=resumed_rng,
        )
    )

    assert window == pytest.approx(list(_WINDOW))
    assert (global_step, chunk_id, eval_id, episode_id) == (17_000, 5, 3, 41)
    assert resumed_bandit.scores.tolist() == pytest.approx(bandit.scores.tolist())
    assert resumed_rng.bit_generator.state == written_rng.bit_generator.state


def test_an_empty_window_scores_the_first_episode_as_a_maximum() -> None:
    """Why the window has to survive a resume, rather than being rebuilt.

    With nothing to rank against, the normalization returns the maximum by
    construction, whatever the episode's true learning potential. The bandit's
    EMA then takes a spurious 1.0 for whichever arm happens to be pulled first
    after the resume, and `recent_window_size = 100` episodes are needed before
    the ranking grid is dense again.
    """

    assert _normalize_learning_potential(0.001, []) == 1.0
    assert _normalize_learning_potential(0.001, list(_WINDOW)) < 1.0
