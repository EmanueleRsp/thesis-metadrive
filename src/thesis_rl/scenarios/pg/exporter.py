from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from thesis_rl.scenarios.arms import assign_primary_arm, derive_scenario_tags
from thesis_rl.scenarios.catalog import ScenarioCatalogEntry
from thesis_rl.scenarios.features import extract_scenario_features
from thesis_rl.scenarios.quality import apply_catalog_quality_policy
from thesis_rl.scenarios.pg.profiles import GenerationSpec
from thesis_rl.scenarios.pg.validation import PGValidationResult, validate_exported_scenario
from thesis_rl.scenarios.records import ScenarioRecord


def _block_ids_from_map_metadata(map_metadata: dict[str, Any]) -> tuple[str, ...]:
    sequence = map_metadata.get("block_sequence", [])
    result: list[str] = []
    for block in sequence:
        if isinstance(block, dict):
            block_id = block.get("block_ID") or block.get("block_id") or block.get("id")
            if block_id is not None:
                result.append(str(block_id))
    return tuple(result)


def _realized_metadata(spec: GenerationSpec, map_metadata: dict[str, Any]) -> dict[str, Any]:
    ids = _block_ids_from_map_metadata(map_metadata)
    merge_ids = {"y", "r", "R", "O"}
    intersection_ids = {"X", "T"}
    return {
        "profile": spec.profile,
        "seed": spec.seed,
        "realized_topology": {
            "has_merge_or_roundabout": bool(set(ids).intersection(merge_ids)),
            "has_intersection": bool(set(ids).intersection(intersection_ids)),
        },
        "route_traffic_controls": {
            "has_traffic_light": False,
            "traffic_light_states_complete": True,
            "has_stop_sign": False,
            "has_crosswalk": False,
        },
        "block_sequence": list(ids),
        "static_obstacle": map_metadata.get(
            "static_obstacle", {"realized": False, "object_types": []}
        ),
    }


def _has_realized_static_obstacle(realized: dict[str, Any]) -> bool:
    metadata = realized.get("static_obstacle", {})
    return isinstance(metadata, dict) and metadata.get("realized") is True


def _write_yaml(path: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite generation manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    temporary.replace(path)


def _scenario_record(
    scenario: dict[str, Any],
    *,
    data_root: Path,
    scenario_path: Path,
    spec: GenerationSpec,
    validation: PGValidationResult,
    entry_features: Any,
    has_static_obstacle: bool,
) -> ScenarioRecord:
    scenario_id = validation.scenario_id
    relative_path = scenario_path.resolve().relative_to(data_root.resolve()).as_posix()
    return ScenarioRecord(
        scenario_uid=f"pg:scenarionet_v1:{scenario_id}",
        scenario_id=scenario_id,
        source="pg",
        relative_path=relative_path,
        official_split=None,
        source_log_id=None,
        source_scenario_id=scenario_id,
        dataset_version="scenarionet_v1",
        converter_version=None,
        split="train",
        runtime_index=None,
        length=validation.scenario_length,
        pg_profile=spec.profile,
        pg_seed=spec.seed,
        map_id="".join(spec.block_sequence),
        primary_arm=assign_primary_arm(
            entry_features, has_static_obstacle=has_static_obstacle
        ),
        tags=derive_scenario_tags(
            entry_features, has_static_obstacle=has_static_obstacle
        ),
        signal_reliability=entry_features.signal_reliability,
        validation_status=validation.status,  # type: ignore[arg-type]
        validation_warnings=validation.warnings,
    )


def export_pg_scenario(
    scenario: dict[str, Any],
    *,
    spec: GenerationSpec,
    realized_map_metadata: dict[str, Any] | None = None,
    data_root: str | Path,
    output_directory: str | Path,
    generator_commit: str | None,
    exporter_commit: str | None,
    overwrite: bool = False,
) -> tuple[ScenarioCatalogEntry, PGValidationResult, Path]:
    """Persist one exported ScenarioDescription and its reproducibility manifest."""
    from metadrive.scenario.utils import save_dataset  # type: ignore[import-not-found]

    root = Path(data_root).expanduser().resolve()
    output = Path(output_directory).expanduser().resolve()
    validation = validate_exported_scenario(scenario)
    if validation.status == "invalid":
        raise ValueError("cannot export invalid ScenarioDescription")
    map_metadata = realized_map_metadata or {}
    if not isinstance(map_metadata, dict):
        map_metadata = {}
    realized = _realized_metadata(spec, map_metadata)
    features = extract_scenario_features(
        copy.deepcopy(scenario), "pg", realized_generation_metadata=realized
    )
    scenario_copy = copy.deepcopy(scenario)
    dataset_dir = output / spec.profile / str(spec.seed)
    save_dataset(
        [scenario_copy],
        dataset_name="pg",
        dataset_version="scenarionet_v1",
        dataset_dir=str(dataset_dir),
    )
    scenario_files = sorted(
        path
        for path in dataset_dir.glob("*.pkl")
        if path.name not in {"dataset_summary.pkl", "dataset_mapping.pkl"}
    )
    if len(scenario_files) != 1:
        raise RuntimeError(f"expected one exported ScenarioDescription, found {scenario_files}")
    scenario_path = scenario_files[0]
    record = _scenario_record(
        scenario_copy,
        data_root=root,
        scenario_path=scenario_path,
        spec=spec,
        validation=validation,
        entry_features=features,
        has_static_obstacle=_has_realized_static_obstacle(realized),
    )
    record = apply_catalog_quality_policy(record, features)
    entry = ScenarioCatalogEntry(record=record, features=features)
    generation_manifest = {
        "scenario_id": validation.scenario_id,
        "generation": {
            "profile": spec.profile,
            "seed": spec.seed,
            "block_sequence": list(spec.block_sequence),
            "traffic_density": spec.traffic_density,
            "lane_num": spec.lane_num,
            "lane_width": spec.lane_width,
            "accident_prob": spec.accident_prob,
            "max_episode_length": spec.max_episode_length,
        },
        "software": {"generator_commit": generator_commit, "exporter_commit": exporter_commit},
        "output": {
            "scenario_path": scenario_path.relative_to(root).as_posix(),
            "scenario_length": validation.scenario_length,
            "validation_status": validation.status,
        },
        "realized": realized,
    }
    manifest_path = root / "pg" / "generation_manifests" / f"{validation.scenario_id}.yaml"
    _write_yaml(manifest_path, generation_manifest, overwrite=overwrite)
    return entry, validation, manifest_path
