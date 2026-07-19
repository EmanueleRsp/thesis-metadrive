#!/usr/bin/env python3
"""Validate candidate golden-suite references against immutable source content."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _feature_types(description: dict[str, Any]) -> list[str]:
    features = description.get("map_features", {})
    if not isinstance(features, dict):
        raise ValueError("ScenarioDescription map_features must be a mapping.")
    return sorted(
        {
            str(feature.get("type", "unknown"))
            for feature in features.values()
            if isinstance(feature, dict)
        }
    )


def _validate_reference(
    reference: dict[str, Any], catalog_record: dict[str, Any], data_root: Path
) -> dict[str, Any]:
    if reference["split"] != "train" or catalog_record["split"] != "train":
        raise ValueError(f"Golden reference is not train-only: {reference['scenario_uid']}")
    path = data_root / str(reference["relative_path"])
    with path.open("rb") as handle:
        description = pickle.load(handle)
    if not isinstance(description, dict):
        raise ValueError(f"ScenarioDescription is not a mapping: {path}")
    if int(description.get("length", -1)) != int(catalog_record["length"]):
        raise ValueError(f"Length mismatch for {reference['scenario_uid']}")
    tracks = description.get("tracks")
    metadata = description.get("metadata")
    features = description.get("map_features")
    if not isinstance(tracks, dict) or not tracks:
        raise ValueError(f"ScenarioDescription tracks are missing: {reference['scenario_uid']}")
    if not isinstance(metadata, dict):
        raise ValueError(f"ScenarioDescription metadata is missing: {reference['scenario_uid']}")
    if not isinstance(features, dict) or not features:
        raise ValueError(
            f"ScenarioDescription map features are missing: {reference['scenario_uid']}"
        )
    sdc_id = metadata.get("sdc_id")
    if sdc_id not in tracks:
        raise ValueError(f"SDC track is missing: {reference['scenario_uid']}")
    route_lanes = [str(lane) for lane in catalog_record.get("assigned_route_lane_ids", [])]
    map_keys = {str(key) for key in features}
    missing_route_lanes = sorted(lane for lane in route_lanes if lane not in map_keys)
    if missing_route_lanes:
        raise ValueError(
            f"Assigned route lanes are absent from map features for {reference['scenario_uid']}: "
            f"{missing_route_lanes[:3]}"
        )
    types = _feature_types(description)
    controls = [
        feature_type
        for feature_type in types
        if any(token in feature_type for token in ("TRAFFIC_LIGHT", "STOP_SIGN", "CROSSWALK"))
    ]
    markings = [feature_type for feature_type in types if "ROAD_LINE" in feature_type]
    return {
        "scenario_uid": str(reference["scenario_uid"]),
        "source": str(reference["source"]),
        "primary_arm": str(reference["primary_arm"]),
        "relative_path": str(reference["relative_path"]),
        "content_validation": "pass",
        "description_length": int(description["length"]),
        "track_count": len(tracks),
        "map_feature_count": len(features),
        "sdc_id": str(sdc_id),
        "route_lane_count": len(route_lanes),
        "map_feature_types": ";".join(types),
        "actual_traffic_controls": ";".join(controls) or "none_in_map_features",
        "actual_lane_markings": ";".join(markings) or "none_in_map_features",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--frozen-index", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite golden-suite output: {args.output_dir}")

    candidate = _load_json(args.candidate_manifest)
    if candidate.get("status") != "CANDIDATE_METADATA_ONLY_NOT_FINAL":
        raise ValueError("Expected the unfinalized candidate golden manifest.")
    references = candidate.get("references")
    if not isinstance(references, list) or len(references) != 48:
        raise ValueError("Golden candidate must contain exactly 48 references.")
    if len({str(reference.get("scenario_uid")) for reference in references}) != 48:
        raise ValueError("Golden candidate references must be unique.")
    frozen = _load_json(args.frozen_index)
    catalog = {str(row["scenario_uid"]): row for row in frozen.get("records", [])}
    if len(catalog) != len(frozen.get("records", [])):
        raise ValueError("Frozen catalog contains duplicate scenario UIDs.")

    evidence = []
    for reference in references:
        scenario_uid = str(reference["scenario_uid"])
        try:
            catalog_record = catalog[scenario_uid]
        except KeyError as exc:
            raise ValueError(
                f"Golden reference is absent from the frozen catalog: {scenario_uid}"
            ) from exc
        evidence.append(_validate_reference(reference, catalog_record, args.data_root))

    arm_counts = {
        arm: sum(row["primary_arm"] == arm for row in evidence)
        for arm in candidate["selection"]["source_quotas"]
    }
    if set(arm_counts.values()) != {8}:
        raise ValueError(f"Golden suite must retain eight references per arm: {arm_counts}")
    args.output_dir.mkdir(parents=True)
    final_manifest = dict(candidate)
    final_manifest["schema"] = "scenarionet_golden_suite_content_validated_v1"
    final_manifest["status"] = "CONTENT_VALIDATED_REFERENCE_SUITE"
    final_manifest["content_validation"] = {
        "source_root": str(args.data_root.resolve()),
        "validated_references": len(evidence),
        "checks": [
            "source_file_exists",
            "pickle_mapping",
            "catalog_length_match",
            "nonempty_tracks",
            "nonempty_map_features",
            "sdc_track_present",
            "assigned_route_lanes_present_in_map_features",
        ],
        "not_validated": [
            "ScenarioEnv_reset_step",
            "live_rulebook_evaluation",
            "policy_termination_or_truncation",
        ],
    }
    (args.output_dir / "golden_suite_manifest.json").write_text(
        json.dumps(final_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "golden_suite_content_evidence.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(evidence[0]))
        writer.writeheader()
        writer.writerows(evidence)


if __name__ == "__main__":
    main()
