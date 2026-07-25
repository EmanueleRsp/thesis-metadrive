from __future__ import annotations

from pathlib import Path

import pytest

from thesis_rl.envs.factory import _resolve_agent_policy, _resolve_frozen_panel_uids
from thesis_rl.scenarios.panel_manifest import build_balanced_panel, save_panel_manifest
from thesis_rl.scenarios.records import ScenarioRecord


def test_resolve_agent_policy_accepts_canonical_class_names() -> None:
    env_input_cls = _resolve_agent_policy("EnvInputPolicy")
    expert_cls = _resolve_agent_policy("ExpertPolicy")
    idm_cls = _resolve_agent_policy("IDMPolicy")

    assert env_input_cls.__name__ == "EnvInputPolicy"
    assert expert_cls.__name__ == "ExpertPolicy"
    assert idm_cls.__name__ == "IDMPolicy"


def _record(index: int, arm: str) -> ScenarioRecord:
    return ScenarioRecord(
        scenario_uid=f"waymo:v1:{arm}:{index}",
        scenario_id=str(index),
        source="waymo",
        relative_path=f"waymo/database/{index}.pkl",
        official_split="training_20s",
        source_log_id=f"log-{index}",
        source_scenario_id=str(index),
        dataset_version="v1",
        converter_version="converter",
        split="validation",
        runtime_index=index,
        length=100,
        pg_profile=None,
        pg_seed=None,
        map_id="S",
        primary_arm=arm,
        tags=(),
        signal_reliability="not_applicable",
        validation_status="valid",
        validation_warnings=(),
    )


def test_resolve_frozen_panel_uids_returns_none_when_not_configured() -> None:
    assert (
        _resolve_frozen_panel_uids({}, split="validation", scenario_uids_file_eligible=None)
        is None
    )


def test_resolve_frozen_panel_uids_loads_frozen_manifest(tmp_path: Path) -> None:
    from thesis_rl.scenarios.arms import ARMS

    records = [_record(i, arm) for arm in ARMS for i in range(4)]
    manifest = build_balanced_panel(records, split="validation", size=6, seed=1)
    path = tmp_path / "validation_panel_manifest_v1.json"
    save_panel_manifest(manifest, path)

    resolved = _resolve_frozen_panel_uids(
        {"panel_manifest_path": str(path)},
        split="validation",
        scenario_uids_file_eligible=None,
    )
    assert resolved == manifest.scenario_uids


def test_resolve_frozen_panel_uids_fails_closed_on_split_mismatch(tmp_path: Path) -> None:
    from thesis_rl.scenarios.arms import ARMS

    records = [_record(i, arm) for arm in ARMS for i in range(4)]
    manifest = build_balanced_panel(records, split="validation", size=6, seed=1)
    path = tmp_path / "validation_panel_manifest_v1.json"
    save_panel_manifest(manifest, path)

    with pytest.raises(ValueError, match="frozen for split"):
        _resolve_frozen_panel_uids(
            {"panel_manifest_path": str(path)},
            split="test",
            scenario_uids_file_eligible=None,
        )


def test_resolve_frozen_panel_uids_fails_closed_on_golden_uid_set_mismatch(tmp_path: Path) -> None:
    from thesis_rl.scenarios.arms import ARMS

    records = [_record(i, arm) for arm in ARMS for i in range(4)]
    manifest = build_balanced_panel(records, split="validation", size=6, seed=1)
    path = tmp_path / "validation_panel_manifest_v1.json"
    save_panel_manifest(manifest, path)

    with pytest.raises(ValueError, match="disagree on"):
        _resolve_frozen_panel_uids(
            {"panel_manifest_path": str(path)},
            split="validation",
            scenario_uids_file_eligible=("some:other:uid",),
        )
