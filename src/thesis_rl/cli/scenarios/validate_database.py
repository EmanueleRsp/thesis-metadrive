from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from thesis_rl.scenarios.catalog import read_scenario_catalog
from thesis_rl.scenarios.runtime_database import verify_runtime_mapping
from thesis_rl.scenarios.runtime_database import sha256_file
from thesis_rl.scenarios.validation import validate_records, write_validation_summary
from thesis_rl.cli.scenarios.ui import console, make_progress, print_key_value_table, print_panel


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a ScenarioNet runtime mapping.")
    parser.add_argument("runtime_directory")
    parser.add_argument("--data-root", default=os.environ.get("SCENARIONET_DATA_ROOT"))
    parser.add_argument(
        "--catalog",
        help="Optional scenario_catalog.parquet to run catalog-driven validation.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        help="Validate only records belonging to this runtime split.",
    )
    parser.add_argument(
        "--validation-summary",
        help="Output JSON path (default: <runtime_directory>/validation_summary.json).",
    )
    parser.add_argument("--skip-feature-extraction", action="store_true")
    args = parser.parse_args()
    if not args.data_root:
        raise SystemExit("--data-root or SCENARIONET_DATA_ROOT is required")
    with console.status("Checking runtime database mapping", spinner="dots"):
        filenames = verify_runtime_mapping(args.runtime_directory)
    payload: dict[str, object] = {
        "runtime_files": len(filenames),
        "feature_extraction_skipped": bool(args.skip_feature_extraction),
    }
    if args.catalog:
        catalog_path = Path(args.catalog).expanduser().resolve()
        catalog = read_scenario_catalog(catalog_path)
        records = tuple(
            record for record in catalog.records if args.split is None or record.split == args.split
        )
        if not records:
            raise ValueError(
                f"catalog has no records for split={args.split!r}" if args.split else "catalog is empty"
            )
        progress = make_progress()
        task_id = progress.add_task(
            "Validating ScenarioDescription files", total=len(records)
        )

        def on_progress(index, _total, record, result) -> None:
            progress.update(
                task_id,
                completed=index,
                description=f"Validating {record.source} • {result.status} • {record.scenario_uid}",
            )

        with progress:
            results = validate_records(
                records,
                data_root=args.data_root,
                run_feature_extraction=not args.skip_feature_extraction,
                progress_callback=on_progress,
            )
        summary_path = Path(args.validation_summary or (Path(args.runtime_directory) / "validation_summary.json"))
        write_validation_summary(
            results,
            summary_path,
            catalog_hash=sha256_file(catalog_path),
            overwrite=True,
        )
        payload["catalog_records"] = len(records)
        payload["catalog_total_records"] = len(catalog.records)
        payload["validation_summary"] = str(summary_path.expanduser().resolve())
        validation_counts = {
            status: sum(result.status == status for result in results)
            for status in ("valid", "warning", "invalid")
        }
        payload["validation_counts"] = validation_counts
        invalid = int(validation_counts["invalid"])
        split_suffix = f" (split={args.split})" if args.split else ""
        validation_message = (
            f"Runtime files: {len(filenames)}\n"
            f"Catalog records checked: {len(records)}{split_suffix}\n"
            f"Valid: {validation_counts['valid']} | "
            f"Warnings: {validation_counts['warning']} | Invalid: {invalid}\n"
            f"Summary: {summary_path}"
        )
        print_panel(
            "Runtime validation passed" if invalid == 0 else "Runtime validation found invalid scenarios",
            validation_message,
            style="green" if invalid == 0 else "red",
        )
        print_key_value_table("Validation counts", list(validation_counts.items()))
    else:
        print_panel(
            "Runtime mapping valid",
            f"Runtime files: {len(filenames)}\nDirectory: {args.runtime_directory}",
        )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
