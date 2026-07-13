"""Build deterministic, source-aware ScenarioNet train/validation/test splits."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml  # type: ignore[import-untyped]

from thesis_rl.scenarios.catalog import read_scenario_catalog, write_scenario_catalog
from thesis_rl.scenarios.manifests import validate_split_manifest
from thesis_rl.scenarios.pipeline import (
    SPLITS,
    SOURCES,
    assign_source_splits,
    assign_source_splits_to_targets,
    arm_selection_diagnostics,
)
from thesis_rl.scenarios.reports import write_json_report
from thesis_rl.scenarios.records import SIGNAL_RELIABILITIES
from thesis_rl.scenarios.runtime_database import sha256_file
from thesis_rl.cli.scenarios.ui import console, print_key_value_table, print_panel


def _counts_from_args(args: argparse.Namespace) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for source in SOURCES:
        counts[source] = {
            split: (
                -1
                if getattr(args, f"{source}_{split}") is None
                else int(getattr(args, f"{source}_{split}"))
            )
            for split in SPLITS
        }
    if any(value < 0 for source in counts.values() for value in source.values()):
        raise ValueError(
            "all split counts are required; pass --waymo-train/validation/test "
            "and --pg-train/validation/test explicitly"
        )
    return counts


def _read_split_policy(path: str | None) -> tuple[dict, dict]:
    if path is None:
        return {}, {}
    payload = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("arm minimums config must be a mapping")
    split = payload.get("split", {})
    if not isinstance(split, dict):
        raise ValueError("pipeline split config must be a mapping")
    minimums = split.get("arm_minimums", {})
    if not isinstance(minimums, dict):
        raise ValueError("split.arm_minimums must be a mapping")
    signal_policy = split.get("allowed_signal_reliabilities", {})
    if not isinstance(signal_policy, dict):
        raise ValueError("split.allowed_signal_reliabilities must be a mapping")
    return minimums, signal_policy


def main() -> int:
    parser = argparse.ArgumentParser(description="Build deterministic ScenarioNet splits.")
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split-manifest")
    parser.add_argument("--groups", help="Accepted for compatibility; groups are derived from records.")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument(
        "--arm-minimums-config",
        help="Pipeline YAML containing optional split.arm_minimums.",
    )
    for source in SOURCES:
        for split in SPLITS:
            parser.add_argument(f"--{source}-{split}", dest=f"{source}_{split}", type=int)
            parser.add_argument(
                f"--{source}-target-{split}",
                dest=f"{source}_target_{split}",
                type=int,
            )
    parser.add_argument(
        "--auto-targets",
        action="store_true",
        help="Assign whole groups toward target counts and record actual counts.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    catalog_path = Path(args.catalog).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    manifest_path = Path(
        args.split_manifest
        or output_path.parent.parent / "splits" / "split_manifest.json"
    ).expanduser().resolve()
    with console.status("Reading catalog and assigning leakage-free splits", spinner="dots"):
        catalog = read_scenario_catalog(catalog_path)
        arm_minimums, signal_policy = _read_split_policy(args.arm_minimums_config)
        if args.auto_targets:
            targets = {
                source: {
                    split: (
                        0
                        if getattr(args, f"{source}_target_{split}") is None
                        else int(getattr(args, f"{source}_target_{split}"))
                    )
                    for split in SPLITS
                }
                for source in SOURCES
            }
            if not any(value > 0 for source in targets.values() for value in source.values()):
                raise ValueError("auto-targets requires at least one positive split target")
            entries = assign_source_splits_to_targets(
                catalog.entries,
                targets=targets,
                seed=int(args.split_seed),
                arm_minimums=arm_minimums,
                allowed_signal_reliabilities=signal_policy,
            )
            selection_mode = "grouped_target"
        else:
            counts = _counts_from_args(args)
            entries = assign_source_splits(
                catalog.entries,
                counts=counts,
                seed=int(args.split_seed),
            )
            selection_mode = "exact"
        write_scenario_catalog(entries, output_path, overwrite=args.overwrite)

    split_counts = {
        split: {
            source: sum(
                entry.record.split == split and entry.record.source == source
                for entry in entries
            )
            for source in SOURCES
        }
        for split in SPLITS
    }
    arm_diagnostics = arm_selection_diagnostics(entries, arm_minimums)
    signal_excluded_records = sum(
        entry.features.signal_reliability
        not in set(signal_policy.get(entry.record.source, SIGNAL_RELIABILITIES))
        for entry in catalog.entries
    )
    total_arm_deficit = sum(
        values["deficit"]
        for source in arm_diagnostics.values()
        for split in source.values()
        for values in split.values()
    )
    manifest = validate_split_manifest(
        {
            "split_seed": int(args.split_seed),
            "source_policy": {
                "waymo": "grouped",
                "pg": "seed_disjoint",
            },
            "grouping": {
                "waymo": "source_log_id_or_scenario_uid",
                "pg": "pg_seed",
            },
            "selection": {
                "mode": selection_mode,
                "input_records": len(catalog.entries),
                "selected_records": len(entries),
                "excluded_records": len(catalog.entries) - len(entries),
                "arm_minimums": arm_minimums,
                "arm_diagnostics": arm_diagnostics,
                "total_arm_deficit": total_arm_deficit,
                "allowed_signal_reliabilities": signal_policy,
                "signal_excluded_records": signal_excluded_records,
            },
            "counts": split_counts,
            "catalog_hash": sha256_file(output_path),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite split manifest: {manifest_path}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = {
        "catalog": str(output_path),
        "split_manifest": str(manifest_path),
        "counts": split_counts,
        "split_seed": int(args.split_seed),
        "input_records": len(catalog.entries),
        "selected_records": len(entries),
        "excluded_records": len(catalog.entries) - len(entries),
        "arm_diagnostics": arm_diagnostics,
        "total_arm_deficit": total_arm_deficit,
        "allowed_signal_reliabilities": signal_policy,
        "signal_excluded_records": signal_excluded_records,
    }
    write_json_report(
        report,
        output_path.parent / "split_report.json",
        overwrite=args.overwrite,
    )
    print_panel(
        "Splits built",
        f"Selected {len(entries)}/{len(catalog.entries)} catalog records\n"
        f"Mode: {selection_mode}\n"
        f"Manifest: {manifest_path}",
    )
    if total_arm_deficit:
        print_panel(
            "Arm coverage deficit",
            f"{total_arm_deficit} requested source/split/arm slots remain "
            "unfilled; inspect split_report.json before freezing the dataset.",
            style="yellow",
        )
    print_key_value_table(
        "Effective split counts",
        [
            (
                split,
                ", ".join(f"{source}={split_counts[split][source]}" for source in SOURCES),
            )
            for split in SPLITS
        ],
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
