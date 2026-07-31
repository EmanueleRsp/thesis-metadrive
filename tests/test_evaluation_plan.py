from __future__ import annotations

import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf

from thesis_rl.runtime.evaluation_plan import resolve_scenarionet_evaluation_panels
from thesis_rl.scenarios.panel_manifest import (
    NAMED_PANELS,
    PROFILE_SUBSET_SEEDS,
    PROFILE_SUBSET_SIZES,
    PanelManifest,
    build_profile_subset_panel,
    save_panel_manifest,
)
from thesis_rl.scenarios.runtime_database import sha256_file


def _manifest(name: str, size: int) -> PanelManifest:
    specification = NAMED_PANELS[name]
    uids = tuple(f"{name}:{index}" for index in range(size))
    return PanelManifest(
        schema_version="v1",
        split=str(specification["split"]),
        seed=7,
        size=size,
        arms=("A0_simple_low_traffic",),
        per_arm_counts=(size,),
        scenario_uids=uids,
        sha256=hashlib.sha256("\n".join(uids).encode("utf-8")).hexdigest(),
        draw_policy=str(specification["draw_policy"]),
        source=str(specification["source"]),
    )


def _write_frozen_index(root: Path) -> None:
    panels: dict[str, dict[str, object]] = {}
    smoke: dict[str, dict[str, object]] = {}
    records = []
    for name, specification in NAMED_PANELS.items():
        parent = _manifest(name, int(specification["full_size"]))
        path = root / "panels" / f"{name}_panel_manifest_v1.json"
        save_panel_manifest(parent, path)
        panels[name] = {
            "artifact": {"relative_path": str(path.relative_to(root)), "sha256": sha256_file(path)},
            "manifest": json.loads(path.read_text(encoding="utf-8")),
        }
        records.extend(
            [
                type("Record", (), {"scenario_uid": uid, "primary_arm": "A0_simple_low_traffic"})()
                for uid in parent.scenario_uids
            ]
        )
        child = build_profile_subset_panel(
            parent,
            records,
            parent_name=name,
            scope="smoke",
            size=PROFILE_SUBSET_SIZES["smoke"][name],
            seed=PROFILE_SUBSET_SEEDS["smoke"],
        )
        child_path = root / "panels" / f"smoke_{name}_panel_manifest_v1.json"
        save_panel_manifest(child, child_path)
        smoke[name] = {
            "artifact": {
                "relative_path": str(child_path.relative_to(root)),
                "sha256": sha256_file(child_path),
            },
            "manifest": json.loads(child_path.read_text(encoding="utf-8")),
        }
    index = {
        "schema": "scenarionet_frozen_selection_v1",
        "selection_hash": "selection-hash",
        "records": [{"scenario_uid": "placeholder"}],
        "panel_manifests": panels,
        "profile_panel_manifests": {"smoke": smoke},
    }
    target = root / "frozen" / "scenario_selection_index.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(index), encoding="utf-8")


def _cfg(root: Path, profile: str):
    return OmegaConf.create(
        {
            "env": {"name": "scenarionet"},
            "run_profile": {"name": profile},
            "paths": {"scenarionet_data_root": str(root)},
            "evaluation": {"scenarionet": {"profiles": {profile: {"scope": profile if profile == "smoke" else "full"}}}},
            # Must be ignored for official ScenarioNet cardinality.
            "experiment": {"eval_episodes": 1, "final_eval_episodes": 1},
        }
    )


def test_full_profile_resolves_complete_validation_panels_ignoring_scalar_overrides(tmp_path: Path) -> None:
    root = tmp_path / "scenarionet"
    _write_frozen_index(root)
    panels = resolve_scenarionet_evaluation_panels(_cfg(root, "default"), final=False)
    assert [(panel.name, panel.episode_count, panel.scope) for panel in panels] == [
        ("validation_waymo_empirical", 150, "full"),
        ("validation_pg", 150, "full"),
    ]


def test_smoke_profile_resolves_frozen_diagnostic_final_panels(tmp_path: Path) -> None:
    root = tmp_path / "scenarionet"
    _write_frozen_index(root)
    panels = resolve_scenarionet_evaluation_panels(_cfg(root, "smoke"), final=True)
    assert [(panel.name, panel.episode_count, panel.scope) for panel in panels] == [
        ("test_waymo_empirical", 20, "smoke"),
        ("test_pg", 20, "smoke"),
        ("test_arm_stratified", 12, "smoke"),
    ]
