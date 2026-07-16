"""Build the audited Rulebook v2-eligible catalog before ScenarioNet splitting."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
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
from thesis_rl.scenarios.catalog import read_scenario_catalog, write_scenario_catalog
from thesis_rl.scenarios.runtime_database import sha256_file


def _canonical_json_hash(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"ego config is not readable JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("ego config must be a JSON object")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-catalog", type=Path, required=True)
    parser.add_argument("--eligibility-output", type=Path, required=True)
    parser.add_argument("--ego-config", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of spawned worker processes used for static eligibility evaluation.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    catalog_path = args.catalog.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    output_catalog = args.output_catalog.expanduser().resolve()
    eligibility_output = args.eligibility_output.expanduser().resolve()
    ego_hash = _canonical_json_hash(args.ego_config.expanduser().resolve())
    calibration = load_calibration_artifact(
        args.calibration.expanduser().resolve(),
        expected_config_hash=ego_hash,
    )
    geometry_hash = geometry_config_hash()
    catalog = read_scenario_catalog(catalog_path)
    console = Console(stderr=True)
    console.log(
        f"Evaluating {len(catalog.entries)} catalog entries with {args.workers} worker process(es)"
    )
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
        task_id = progress.add_task("Rulebook v2 static eligibility", total=len(catalog.entries))

        def report_progress(completed: int, total: int) -> None:
            progress.update(task_id, completed=completed, total=total)

        eligibility = evaluate_catalog_entries(
            catalog.entries,
            data_root=data_root,
            geometry_config_hash=geometry_hash,
            calibration_hash=calibration.config_hash,
            workers=args.workers,
            progress_callback=report_progress,
        )
    by_uid = {record.scenario_uid: record for record in eligibility}
    entries_by_uid = {entry.record.scenario_uid: entry for entry in catalog.entries}
    selected = tuple(
        entry for entry in catalog.entries if by_uid[entry.record.scenario_uid].rulebook_eligible
    )
    if not selected:
        raise ValueError("Rulebook v2 eligibility rejected every catalog entry")
    if output_catalog.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite Rulebook catalog: {output_catalog}")
    if eligibility_output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite eligibility artifact: {eligibility_output}")
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
        "schema": "rulebook-v2-catalog-eligibility-v1",
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
                "eligible_records": len(selected),
                "excluded_records": len(eligibility) - len(selected),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
