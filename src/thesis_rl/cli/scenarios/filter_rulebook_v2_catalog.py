"""Build the audited Rulebook v2-eligible catalog before ScenarioNet splitting."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from thesis_rl.rulebook.v2.calibration import load_calibration_artifact
from thesis_rl.rulebook.v2.config import RULEBOOK_V2_VERSION, geometry_config_hash
from thesis_rl.rulebook.v2.context.catalog_eligibility import evaluate_catalog_entries
from thesis_rl.rulebook.v2.context.task_route import TaskRouteEligibility
from thesis_rl.scenarios.catalog import (
    ScenarioCatalogEntry,
    read_scenario_catalog,
    write_scenario_catalog,
)
from thesis_rl.scenarios.runtime_database import sha256_file
from thesis_rl.scenarios.mission_eligibility import (
    MISSION_BUILDER_IDENTITY,
    MISSION_ELIGIBILITY_SCHEMA,
    evaluate_driving_mission_entries,
)


def _canonical_json_hash(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"ego config is not readable JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("ego config must be a JSON object")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _scenario_fingerprint(entry: ScenarioCatalogEntry, data_root: Path) -> str:
    """Return a cheap identity for an immutable scenario payload."""

    try:
        stat = (data_root / entry.record.relative_path).stat()
    except OSError:
        return "missing"
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def _load_incremental_cache(
    path: Path,
    *,
    entries: tuple[ScenarioCatalogEntry, ...],
    data_root: Path,
    geometry_hash: str,
    calibration_hash: str,
) -> dict[str, TaskRouteEligibility]:
    """Load cache records proven compatible with the current inputs."""

    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "rulebook-v2-catalog-eligibility-v2"
    ):
        return {}
    if payload.get("geometry_config_hash") != geometry_hash:
        return {}
    if payload.get("calibration_hash") != calibration_hash:
        return {}
    records = payload.get("records")
    if not isinstance(records, list):
        return {}
    by_uid = {item.get("scenario_uid"): item for item in records if isinstance(item, dict)}
    cached: dict[str, TaskRouteEligibility] = {}
    for entry in entries:
        item = by_uid.get(entry.record.scenario_uid)
        if not isinstance(item, dict):
            continue
        if item.get("relative_path") != entry.record.relative_path:
            continue
        if item.get("scenario_fingerprint") != _scenario_fingerprint(entry, data_root):
            continue
        try:
            cached[entry.record.scenario_uid] = TaskRouteEligibility(
                scenario_uid=str(item["scenario_uid"]),
                rulebook_version=str(item["rulebook_version"]),
                adapter_version=str(item["adapter_version"]),
                geometry_config_hash=str(item["geometry_config_hash"]),
                calibration_hash=str(item["calibration_hash"]),
                rulebook_eligible=bool(item["rulebook_eligible"]),
                validation_errors=tuple(str(value) for value in item.get("validation_errors", ())),
                assigned_route_lane_ids=tuple(
                    str(value) for value in item.get("assigned_route_lane_ids", ())
                ),
                assigned_route_source=str(
                    item.get("assigned_route_source", "offline_task_annotation")
                ),
            )
        except (KeyError, TypeError, ValueError):
            continue
    return cached


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-catalog", type=Path, required=True)
    parser.add_argument("--eligibility-output", type=Path, required=True)
    parser.add_argument(
        "--mission-eligibility-output",
        type=Path,
        help="JSON report for the mandatory pre-split driving-mission validation.",
    )
    parser.add_argument("--ego-config", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of spawned worker processes used for static eligibility evaluation.",
    )
    parser.add_argument(
        "--no-incremental",
        action="store_true",
        help="Re-evaluate every record instead of reusing compatible eligibility results.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    catalog_path = args.catalog.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    output_catalog = args.output_catalog.expanduser().resolve()
    eligibility_output = args.eligibility_output.expanduser().resolve()
    mission_eligibility_output = (
        (
            args.mission_eligibility_output
            or output_catalog.with_name("driving_mission_eligibility.json")
        )
        .expanduser()
        .resolve()
    )
    ego_hash = _canonical_json_hash(args.ego_config.expanduser().resolve())
    calibration = load_calibration_artifact(
        args.calibration.expanduser().resolve(),
        expected_config_hash=ego_hash,
    )
    geometry_hash = geometry_config_hash()
    catalog = read_scenario_catalog(catalog_path)
    console = Console(stderr=True)
    entries = tuple(catalog.entries)
    cached = (
        {}
        if args.no_incremental
        else _load_incremental_cache(
            eligibility_output,
            entries=entries,
            data_root=data_root,
            geometry_hash=geometry_hash,
            calibration_hash=calibration.config_hash,
        )
    )
    pending = tuple(entry for entry in entries if entry.record.scenario_uid not in cached)
    console.log(
        f"Evaluating {len(pending)} of {len(entries)} catalog entries with "
        f"{args.workers} worker process(es); reusing {len(cached)} compatible results"
    )
    evaluated: dict[str, TaskRouteEligibility] = dict(cached)
    if pending:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        ) as progress:
            task_id = progress.add_task("Rulebook v2 static eligibility", total=len(pending))

            def report_progress(completed: int, total: int) -> None:
                progress.update(task_id, completed=completed, total=total)

            evaluated.update(
                {
                    result.scenario_uid: result
                    for result in evaluate_catalog_entries(
                        pending,
                        data_root=data_root,
                        geometry_config_hash=geometry_hash,
                        calibration_hash=calibration.config_hash,
                        workers=args.workers,
                        progress_callback=report_progress,
                    )
                }
            )
    eligibility = tuple(evaluated[entry.record.scenario_uid] for entry in entries)
    by_uid = {record.scenario_uid: record for record in eligibility}
    entries_by_uid = {entry.record.scenario_uid: entry for entry in entries}
    annotated_entries = tuple(
        ScenarioCatalogEntry(
            record=replace(
                entry.record,
                rulebook_eligible=by_uid[entry.record.scenario_uid].rulebook_eligible,
                rulebook_validation_errors=by_uid[entry.record.scenario_uid].validation_errors,
                assigned_route_lane_ids=by_uid[entry.record.scenario_uid].assigned_route_lane_ids,
                assigned_route_source=by_uid[entry.record.scenario_uid].assigned_route_source,
            ),
            features=entry.features,
        )
        for entry in catalog.entries
    )
    rulebook_selected = tuple(
        entry for entry in annotated_entries if entry.record.rulebook_eligible is True
    )
    console.log(f"Building driving missions for {len(rulebook_selected)} Rulebook-eligible entries")
    mission_results = evaluate_driving_mission_entries(
        rulebook_selected,
        data_root=data_root,
        workers=args.workers,
    )
    mission_by_uid = {result.scenario_uid: result for result in mission_results}
    selected = tuple(
        ScenarioCatalogEntry(
            record=replace(
                entry.record,
                driving_mission=mission_by_uid[entry.record.scenario_uid].mission,
            ),
            features=entry.features,
        )
        for entry in rulebook_selected
        if mission_by_uid[entry.record.scenario_uid].eligible
    )
    mission_cause_counts = Counter(
        error for result in mission_results for error in result.validation_errors
    )
    mission_source_counts = {
        source: {
            "eligible": sum(
                result.eligible for result in mission_results if result.source == source
            ),
            "excluded": sum(
                not result.eligible for result in mission_results if result.source == source
            ),
        }
        for source in ("pg", "waymo")
    }
    mission_payload = {
        "schema": MISSION_ELIGIBILITY_SCHEMA,
        "builder_identity": MISSION_BUILDER_IDENTITY,
        "input_catalog": str(catalog_path),
        "input_catalog_hash": sha256_file(catalog_path),
        "total_rulebook_eligible_records": len(mission_results),
        "mission_eligible_records": len(selected),
        "mission_excluded_records": len(mission_results) - len(selected),
        "counts_by_source": mission_source_counts,
        "excluded_by_cause": dict(sorted(mission_cause_counts.items())),
        "records": [result.to_dict() for result in mission_results],
    }
    mission_eligibility_output.parent.mkdir(parents=True, exist_ok=True)
    mission_eligibility_output.write_text(
        json.dumps(mission_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    console.log(
        f"Driving-mission filtering complete: eligible={len(selected)}, "
        f"excluded={len(mission_results) - len(selected)}"
    )
    if not selected:
        raise ValueError("driving-mission validation rejected every Rulebook-eligible entry")
    if output_catalog.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite Rulebook catalog: {output_catalog}")
    if eligibility_output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite eligibility artifact: {eligibility_output}")
    if mission_eligibility_output.exists() and not args.overwrite:
        raise FileExistsError(
            "refusing to overwrite driving-mission eligibility artifact: "
            f"{mission_eligibility_output}"
        )
    write_scenario_catalog(selected, output_catalog, overwrite=True)

    cause_counts = Counter(error for record in eligibility for error in record.validation_errors)
    source_counts = {
        source: {
            "eligible": sum(
                record.rulebook_eligible
                and entries_by_uid[record.scenario_uid].record.source == source
                for record in eligibility
            ),
            "excluded": sum(
                not record.rulebook_eligible
                and entries_by_uid[record.scenario_uid].record.source == source
                for record in eligibility
            ),
        }
        for source in ("pg", "waymo")
    }
    payload = {
        "schema": "rulebook-v2-catalog-eligibility-v2",
        "rulebook_version": RULEBOOK_V2_VERSION,
        "input_catalog": str(catalog_path),
        "input_catalog_hash": sha256_file(catalog_path),
        "geometry_config_hash": geometry_hash,
        "calibration_hash": calibration.config_hash,
        "total_records": len(eligibility),
        "eligible_records": len(selected),
        "excluded_records": len(eligibility) - len(selected),
        "counts_by_source": source_counts,
        "excluded_by_cause": dict(sorted(cause_counts.items())),
        "records": [
            {
                **asdict(record),
                "source": entries_by_uid[record.scenario_uid].record.source,
                "relative_path": entries_by_uid[record.scenario_uid].record.relative_path,
                "scenario_fingerprint": _scenario_fingerprint(
                    entries_by_uid[record.scenario_uid], data_root
                ),
            }
            for record in eligibility
        ],
    }
    eligibility_output.parent.mkdir(parents=True, exist_ok=True)
    eligibility_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    console.log(
        f"Rulebook v2 filtering complete: eligible={len(selected)}, "
        f"excluded={len(eligibility) - len(selected)}"
    )
    print(
        json.dumps(
            {
                "catalog": str(output_catalog),
                "eligibility": str(eligibility_output),
                "mission_eligibility": str(mission_eligibility_output),
                "eligible_records": len(selected),
                "excluded_records": len(eligibility) - len(selected),
                "mission_excluded_records": len(mission_results) - len(selected),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
