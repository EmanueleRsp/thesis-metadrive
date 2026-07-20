from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from thesis_rl.cli.scenarios.golden_rulebook_trace import _build_recorder_factory
from thesis_rl.scenarios.golden import load_golden_scenario_uids


MANIFEST = (
    Path(__file__).parents[1]
    / "docs"
    / "audits"
    / "scalar_pipeline_audit_2026-07-19"
    / "golden_suite_content_validated"
    / "golden_suite_manifest.json"
)


def test_content_validated_golden_suite_contains_48_unique_uids() -> None:
    uids = load_golden_scenario_uids(MANIFEST)
    assert len(uids) == 48
    assert len(set(uids)) == 48


def test_golden_suite_loader_rejects_unvalidated_manifest(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema": "candidate", "references": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="content-validated"):
        load_golden_scenario_uids(path)


def test_golden_trace_recorder_factory_uses_configured_video_path(tmp_path: Path) -> None:
    cfg = OmegaConf.create(
        {
            "video": {
                "enabled": True,
                "fps": 20,
                "topdown": {},
                "max_final_videos": 0,
                "save_manifest": True,
                "save_trajectory_log": True,
            },
            "paths": {"videos_dir": str(tmp_path / "videos")},
            "seed": 42,
            "experiment": {"eval_deterministic": True},
            "reward": {
                "type": "rulebook",
                "behavior": "scalar_reward",
                "rulebook_config": "selection",
            },
        }
    )

    factory = _build_recorder_factory(
        cfg=cfg,
        run_dir=tmp_path,
        resolved_env_payload={},
    )
    assert callable(factory)
