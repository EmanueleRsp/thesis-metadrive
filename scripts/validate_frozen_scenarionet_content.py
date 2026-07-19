#!/usr/bin/env python3
"""Read-only structural validation of every frozen ScenarioNet source file."""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import pickle
from pathlib import Path
from typing import Any


def _load_index(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "scenarionet_frozen_selection_v1":
        raise ValueError(f"Unsupported frozen index schema: {payload.get('schema')!r}")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Frozen index must contain a non-empty records list.")
    return records


def _sequence_has_length(value: Any, expected_length: int) -> bool:
    try:
        return len(value) == expected_length
    except TypeError:
        return False


def _is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _validate_record(record: dict[str, Any], data_root: Path) -> tuple[str, str | None]:
    path = data_root / str(record["relative_path"])
    if not path.is_file():
        return "missing_file", f"Source file does not exist: {path}"
    try:
        with path.open("rb") as handle:
            description = pickle.load(handle)
    except Exception as exc:  # Dataset audit must report an unreadable input, not mutate it.
        return "unloadable", f"{type(exc).__name__}: {exc}"
    if not isinstance(description, dict):
        return "invalid_description", "ScenarioDescription is not a mapping"

    length = record.get("length")
    if not isinstance(length, int) or length <= 0:
        return "invalid_catalog_horizon", f"Catalog length is invalid: {length!r}"
    if description.get("length") != length:
        return "horizon_mismatch", f"Catalog={length}, description={description.get('length')!r}"

    tracks = description.get("tracks")
    metadata = description.get("metadata")
    features = description.get("map_features")
    if not isinstance(tracks, dict) or not tracks:
        return "missing_tracks", "ScenarioDescription tracks are absent or empty"
    if not isinstance(metadata, dict):
        return "missing_metadata", "ScenarioDescription metadata is not a mapping"
    if not isinstance(features, dict) or not features:
        return "missing_map_features", "ScenarioDescription map_features are absent or empty"
    sdc_id = metadata.get("sdc_id")
    sdc_track = tracks.get(sdc_id)
    if not isinstance(sdc_track, dict):
        return "missing_sdc_track", f"SDC track is absent: {sdc_id!r}"
    state = sdc_track.get("state")
    if not isinstance(state, dict):
        return "missing_sdc_state", "SDC state is not a mapping"
    for key in ("position", "heading", "valid"):
        if not _sequence_has_length(state.get(key), length):
            actual = state.get(key)
            actual_length = len(actual) if isinstance(actual, (list, tuple)) else None
            return "invalid_sdc_trajectory", f"{key} length={actual_length}, expected={length}"
    valid = state["valid"]
    if not any(bool(value) for value in valid):
        return "invalid_sdc_trajectory", "SDC valid mask has no valid timestep"
    for index, position in enumerate(state["position"]):
        try:
            is_2d = len(position) >= 2
        except TypeError:
            is_2d = False
        if not is_2d:
            return "invalid_sdc_trajectory", f"position[{index}] is not at least 2D"
        if not all(_is_finite_number(value) for value in position[:2]):
            return "invalid_sdc_trajectory", f"position[{index}] contains a non-finite coordinate"
    if not all(_is_finite_number(value) for value in state["heading"]):
        return "invalid_sdc_trajectory", "SDC heading contains a non-finite value"

    route_lanes = record.get("assigned_route_lane_ids")
    if not isinstance(route_lanes, list) or not route_lanes:
        return "missing_assigned_route", "Catalog assigned_route_lane_ids are absent or empty"
    missing_lanes = [str(lane) for lane in route_lanes if str(lane) not in features]
    if missing_lanes:
        return "invalid_assigned_route", f"Route lanes absent from map: {missing_lanes[:3]}"
    return "pass", None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-index", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite validation output: {args.output_dir}")

    records = _load_index(args.frozen_index)
    outcomes: collections.Counter[str] = collections.Counter()
    issues: list[dict[str, str]] = []
    for record in records:
        outcome, detail = _validate_record(record, args.data_root)
        outcomes[outcome] += 1
        if detail is not None:
            issues.append(
                {
                    "scenario_uid": str(record["scenario_uid"]),
                    "source": str(record["source"]),
                    "split": str(record["split"]),
                    "relative_path": str(record["relative_path"]),
                    "outcome": outcome,
                    "detail": detail,
                }
            )

    args.output_dir.mkdir(parents=True)
    report = [
        "# Frozen ScenarioNet Live Content Validation",
        "",
        "Status: `PASS`" if not issues else "Status: `FAIL`",
        "",
        "## Scope",
        "",
        "All checks opened immutable source files read-only. No source file, catalog, ",
        "ScenarioDescription, split, tag, or route annotation was changed.",
        "",
        "## Results",
        "",
        f"- Records checked: `{len(records)}`",
        f"- Outcome counts: `{dict(sorted(outcomes.items()))}`",
        "- Structural checks: source file existence, pickle loadability, mapping shape, ",
        "catalog/description horizon equality, nonempty tracks/map, SDC state, position/heading ",
        "finite values and lengths, nonempty SDC valid mask, and assigned-route lane membership.",
        "- Not evaluated: simulator reset/step, Rulebook runtime eligibility, controls semantics, ",
        "policy termination/truncation, and learning behavior.",
    ]
    (args.output_dir / "live_content_validation_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "live_content_validation_issues.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["scenario_uid", "source", "split", "relative_path", "outcome", "detail"],
        )
        writer.writeheader()
        writer.writerows(issues)
    if issues:
        raise SystemExit(f"Live content validation found {len(issues)} issue(s).")


if __name__ == "__main__":
    main()
