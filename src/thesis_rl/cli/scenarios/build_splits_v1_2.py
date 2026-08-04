"""Build ScenarioNet splits under the v1.2 holdout-first policy.

`SCENARIONET-INTEGRATION` v1.2 §3.5: empirical (label-blind) validation and
test pools are reserved first, an arm-stratified challenge test pool is
reserved from the residual, and a source-balanced, near-uniform-by-arm
training pool is selected from what remains. This is a separate entry point from
`build_splits.py` (the v1.1 `exact`/`grouped_target`/`balanced_arm_source`
policies), not a replacement: `build_splits.py` remains available for
reproducing v1.1 datasets. Every other pipeline stage (catalog build,
Rulebook filtering, threshold computation, runtime database build, freeze)
is unchanged and consumes this script's output exactly as it does
`build_splits.py`'s, because `record.split` remains one of `train`/
`validation`/`test`; the empirical/stratified distinction lives only in
`record.holdout_pool`.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml  # type: ignore[import-untyped]

from thesis_rl.scenarios.arms import ARMS
from thesis_rl.scenarios.catalog import (
    ScenarioCatalogEntry,
    read_scenario_catalog,
    write_scenario_catalog,
)
from thesis_rl.scenarios.manifests import validate_split_manifest
from thesis_rl.mission.types import MISSION_SCHEMA_VERSION
from thesis_rl.scenarios.pipeline import (
    SOURCES,
    SPLITS,
    assign_holdout_first_splits,
)
from thesis_rl.scenarios.pg.profiles import PG_HOLDOUT_EQUIPROBABLE_MIXTURE, PG_PROFILES
from thesis_rl.scenarios.records import SIGNAL_RELIABILITIES
from thesis_rl.scenarios.reports import compute_pg_replenishment_report, write_json_report
from thesis_rl.scenarios.runtime_database import sha256_file
from thesis_rl.cli.scenarios.ui import console, print_key_value_table, print_panel

SPLIT_POLICY = "holdout_first_empirical_then_stratified_then_balanced_train"


def _read_train_arm_minimums(path: str | None) -> dict[str, dict[str, int]]:
    """Read `{source: {arm: minimum}}` from a pipeline YAML's `split.train_arm_minimums`."""
    result: dict[str, dict[str, int]] = {source: {} for source in SOURCES}
    if path is None:
        return result
    payload = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("train-arm-minimums config must be a mapping")
    split_section = payload.get("split", {})
    if not isinstance(split_section, dict):
        raise ValueError("pipeline split config must be a mapping")
    minimums = split_section.get("train_arm_minimums", {})
    if not isinstance(minimums, dict):
        raise ValueError("split.train_arm_minimums must be a mapping")
    for source, arm_payload in minimums.items():
        if source not in SOURCES:
            raise ValueError(f"unknown arm-minimum source: {source!r}")
        if not isinstance(arm_payload, dict):
            raise ValueError(f"split.train_arm_minimums.{source} must be a mapping")
        for arm, value in arm_payload.items():
            if arm not in ARMS:
                raise ValueError(f"unknown arm in train_arm_minimums: {arm!r}")
            result[source][arm] = int(value)
    return result


def _detect_pg_holdout_count_per_profile(
    entries: tuple[ScenarioCatalogEntry, ...],
    *,
    seed_start: int,
    batch_size: int | None,
) -> int:
    """Infer the requested PG window from the latest observed profile seed.

    Individual generation tasks can fail, so a complete contiguous range of
    successful files is neither expected nor required. The declared range
    covers every attempted seed through the last observed one and is rounded
    to the configured equal-profile batch size.
    """
    highest_offset = -1
    for profile_index, profile in enumerate(PG_PROFILES):
        profile_start = seed_start + profile_index * 1_000_000
        offsets = [
            int(entry.record.pg_seed) - profile_start
            for entry in entries
            if entry.record.source == "pg"
            and entry.record.pg_profile == profile.name
            and entry.record.pg_seed is not None
            and profile_start <= int(entry.record.pg_seed) < profile_start + 1_000_000
        ]
        if not offsets:
            raise ValueError(
                "could not auto-detect a PG holdout seed for profile "
                f"{profile.name!r} at seed_start={seed_start}"
            )
        highest_offset = max(highest_offset, max(offsets))
    count = highest_offset + 1
    if batch_size is not None:
        count = ((count + batch_size - 1) // batch_size) * batch_size
    if count < 1:
        raise ValueError(f"could not auto-detect any PG holdout batch at seed_start={seed_start}")
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split-manifest")
    parser.add_argument(
        "--pg-replenishment-report",
        help="JSON report describing whether the filtered PG pool needs more offline seeds.",
    )
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--waymo-test-empirical", type=int, required=True)
    parser.add_argument("--pg-test-empirical", type=int, required=True)
    parser.add_argument("--waymo-validation", type=int, required=True)
    parser.add_argument("--pg-validation", type=int, required=True)
    parser.add_argument("--waymo-train", type=int, required=True)
    parser.add_argument("--pg-train", type=int, required=True)
    parser.add_argument("--stratified-total", type=int, required=True)
    parser.add_argument(
        "--pg-holdout-seed-start",
        type=int,
        help="Base seed of the declared PG empirical-holdout generation range.",
    )
    parser.add_argument(
        "--pg-holdout-count-per-profile",
        help="Number of generated PG holdout scenarios per profile, or 'auto'.",
    )
    parser.add_argument(
        "--pg-holdout-batch-size-per-profile",
        type=int,
        help="Equal per-profile batch size, recorded with per-batch eligible counts.",
    )
    parser.add_argument(
        "--train-arm-minimums-config",
        help="Pipeline YAML containing split.train_arm_minimums.<source>.<arm>.",
    )
    parser.add_argument(
        "--pg-holdout-mixture-tolerance",
        type=float,
        default=0.05,
        help="Max allowed deviation from the equiprobable PG holdout mixture before failing.",
    )
    parser.add_argument(
        "--require-driving-mission",
        action="store_true",
        help="Fail unless every Rulebook-eligible candidate has a v1.1.1 driving mission.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if (args.pg_holdout_seed_start is None) != (args.pg_holdout_count_per_profile is None):
        parser.error(
            "--pg-holdout-seed-start and --pg-holdout-count-per-profile must be supplied together"
        )
    if args.pg_holdout_count_per_profile not in {None, "auto"}:
        try:
            args.pg_holdout_count_per_profile = int(args.pg_holdout_count_per_profile)
        except ValueError:
            parser.error("--pg-holdout-count-per-profile must be a positive integer or 'auto'")
        if args.pg_holdout_count_per_profile < 1:
            parser.error("--pg-holdout-count-per-profile must be positive")
    if args.pg_holdout_batch_size_per_profile is not None and (
        args.pg_holdout_batch_size_per_profile < 1
        or (
            args.pg_holdout_count_per_profile not in {None, "auto"}
            and args.pg_holdout_count_per_profile % args.pg_holdout_batch_size_per_profile != 0
        )
    ):
        parser.error("--pg-holdout-batch-size-per-profile must divide the declared count")

    catalog_path = Path(args.catalog).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    manifest_path = (
        Path(args.split_manifest or output_path.parent.parent / "splits" / "split_manifest.yaml")
        .expanduser()
        .resolve()
    )
    pg_replenishment_path = (
        Path(args.pg_replenishment_report or output_path.parent / "pg_replenishment_report.json")
        .expanduser()
        .resolve()
    )
    train_arm_minimums = _read_train_arm_minimums(args.train_arm_minimums_config)

    test_empirical_counts = {
        "waymo": int(args.waymo_test_empirical),
        "pg": int(args.pg_test_empirical),
    }
    validation_counts = {"waymo": int(args.waymo_validation), "pg": int(args.pg_validation)}
    train_counts = {"waymo": int(args.waymo_train), "pg": int(args.pg_train)}
    if any(
        value < 0
        for value in (
            *test_empirical_counts.values(),
            *validation_counts.values(),
            *train_counts.values(),
        )
    ):
        parser.error("all pool counts must be non-negative")

    def is_declared_pg_holdout_seed(seed: int | None) -> bool:
        if args.pg_holdout_seed_start is None:
            return True
        if seed is None:
            return False
        return any(
            args.pg_holdout_seed_start + profile_index * 1_000_000
            <= seed
            < args.pg_holdout_seed_start
            + profile_index * 1_000_000
            + args.pg_holdout_count_per_profile
            for profile_index in range(len(PG_PROFILES))
        )

    with console.status("Reading catalog and assigning holdout-first splits", spinner="dots"):
        catalog = read_scenario_catalog(catalog_path)
        if args.pg_holdout_count_per_profile == "auto":
            args.pg_holdout_count_per_profile = _detect_pg_holdout_count_per_profile(
                catalog.entries,
                seed_start=int(args.pg_holdout_seed_start),
                batch_size=args.pg_holdout_batch_size_per_profile,
            )
        if args.pg_holdout_batch_size_per_profile is not None and (
            args.pg_holdout_count_per_profile is None
            or args.pg_holdout_count_per_profile % args.pg_holdout_batch_size_per_profile != 0
        ):
            parser.error("--pg-holdout-batch-size-per-profile must divide the declared count")
        valid_or_warning_entries = tuple(
            entry
            for entry in catalog.entries
            if entry.record.validation_status in {"valid", "warning"}
        )
        rulebook_eligible_entries = tuple(
            entry for entry in valid_or_warning_entries if entry.record.rulebook_eligible is True
        )
        if args.require_driving_mission:
            missing_mission = tuple(
                entry.record.scenario_uid
                for entry in rulebook_eligible_entries
                if (
                    not isinstance(entry.record.driving_mission, dict)
                    or entry.record.driving_mission.get("schema_version") != MISSION_SCHEMA_VERSION
                )
            )
            if missing_mission:
                raise ValueError(
                    "catalog is not mission-ready; missing driving_mission for "
                    f"{missing_mission[:5]}"
                )
        empirical_candidate_entries = tuple(
            entry
            for entry in rulebook_eligible_entries
            if entry.record.source == "waymo" or is_declared_pg_holdout_seed(entry.record.pg_seed)
        )
        try:
            entries = assign_holdout_first_splits(
                rulebook_eligible_entries,
                test_empirical_counts=test_empirical_counts,
                validation_counts=validation_counts,
                stratified_total=int(args.stratified_total),
                train_arm_minimums=train_arm_minimums,
                train_source_counts=train_counts,
                seed=int(args.split_seed),
                empirical_candidate_entries=empirical_candidate_entries,
            )
        except ValueError as exc:
            replenishment_report = compute_pg_replenishment_report(
                catalog.entries,
                targets={
                    source: {
                        "train": 0,
                        "validation": validation_counts[source],
                        "test": test_empirical_counts[source],
                    }
                    for source in SOURCES
                },
                allowed_signal_reliabilities=SIGNAL_RELIABILITIES,
                selection_error=str(exc),
            )
            write_json_report(replenishment_report, pg_replenishment_path, overwrite=True)
            raise
        write_scenario_catalog(entries, output_path, overwrite=args.overwrite)

    # The 20% x 5 contract applies to generation batches. Rulebook eligibility
    # may differ by profile, so the post-filter distribution is report-only.
    selected_pg_holdout = [
        entry
        for entry in entries
        if entry.record.source == "pg" and entry.record.holdout_pool == "empirical"
    ]
    selected_pg_total = len(selected_pg_holdout)
    pg_mixture_observed = {
        profile: sum(entry.record.pg_profile == profile for entry in selected_pg_holdout)
        / selected_pg_total
        for profile in PG_HOLDOUT_EQUIPROBABLE_MIXTURE
    }
    pg_holdout_batches = []
    if (
        args.pg_holdout_seed_start is not None
        and args.pg_holdout_batch_size_per_profile is not None
    ):
        for offset in range(
            0, args.pg_holdout_count_per_profile, args.pg_holdout_batch_size_per_profile
        ):
            pg_holdout_batches.append(
                {
                    "seed_start": args.pg_holdout_seed_start + offset,
                    "count_per_profile": args.pg_holdout_batch_size_per_profile,
                    "rulebook_eligible_count": sum(
                        entry.record.source == "pg"
                        and entry.record.pg_seed is not None
                        and any(
                            args.pg_holdout_seed_start + profile_index * 1_000_000 + offset
                            <= entry.record.pg_seed
                            < args.pg_holdout_seed_start
                            + profile_index * 1_000_000
                            + offset
                            + args.pg_holdout_batch_size_per_profile
                            for profile_index in range(len(PG_PROFILES))
                        )
                        for entry in entries
                    ),
                }
            )

    split_counts = {
        split: {
            source: sum(
                entry.record.split == split and entry.record.source == source for entry in entries
            )
            for source in SOURCES
        }
        for split in SPLITS
    }
    observed_arm_distribution_by_pool: dict[str, dict[str, dict[str, int]]] = {}
    for pool_name, predicate in {
        "train": lambda e: e.record.split == "train",
        "validation": lambda e: e.record.split == "validation",
        "test_empirical": lambda e: (
            e.record.split == "test" and e.record.holdout_pool == "empirical"
        ),
        "test_stratified": lambda e: (
            e.record.split == "test" and e.record.holdout_pool == "stratified"
        ),
    }.items():
        pool_entries = [entry for entry in entries if predicate(entry)]
        observed_arm_distribution_by_pool[pool_name] = {
            source: {
                arm: sum(
                    entry.record.source == source and entry.record.primary_arm == arm
                    for entry in pool_entries
                )
                for arm in ARMS
            }
            for source in SOURCES
        }

    manifest = validate_split_manifest(
        {
            "split_seed": int(args.split_seed),
            "split_policy": SPLIT_POLICY,
            "source_policy": {
                "waymo": "grouped",
                "pg": "seed_disjoint",
            },
            "grouping": {
                "waymo": "map_identity_fingerprint",
                "pg": "pg_seed",
            },
            "targets": {
                source: {
                    "train": train_counts[source],
                    "validation": validation_counts[source],
                    "test": split_counts["test"][source],
                }
                for source in SOURCES
            },
            "balancing": {
                "arm_targets": "near_uniform_train_and_stratified",
                "max_arm_count_difference": 1,
                "source_target_within_arm": "best_effort_50_50_with_exact_train_source_totals",
                "preserve_exact_source_totals": True,
                "structural_empty_cells": {"A4_vru": {"pg": True}},
                "allow_cross_source_fill_within_same_arm": True,
                "allow_relabeling": False,
                "allow_duplicate_records": False,
                "allow_quality_filter_relaxation": False,
            },
            "waymo_acquisition": {
                "ordering_seed": int(args.split_seed),
                "batch_size_shards": 64,
                "max_new_shards": 256,
            },
            "counts": split_counts,
            "catalog_hash": sha256_file(output_path),
            "created_at": datetime.now(timezone.utc).isoformat(),
            # v1.2-specific extension keys: additive, not required by the
            # count-only validator above (`validate_split_manifest`), but
            # required by SCENARIONET-INTEGRATION v1.2 SS3.4/SS6.4.
            "holdout_policy": {
                "order": ["test_empirical", "validation_empirical", "test_stratified", "train"],
                "empirical_draw": {"permutation_seed": int(args.split_seed), "label_blind": True},
                "pg_holdout_mixture": PG_HOLDOUT_EQUIPROBABLE_MIXTURE,
                "pg_holdout_seed_range": (
                    None
                    if args.pg_holdout_seed_start is None
                    else {
                        "seed_start": int(args.pg_holdout_seed_start),
                        "count_per_profile": int(args.pg_holdout_count_per_profile),
                    }
                ),
                "pg_holdout_batches": pg_holdout_batches,
                "pg_holdout_mixture_observed": pg_mixture_observed,
                "pg_holdout_mixture_error": None,
            },
            "selection_report": {
                "observed_arm_distribution_by_pool": observed_arm_distribution_by_pool,
                "train_arm_minimums": train_arm_minimums,
                "train_source_targets": train_counts,
            },
        }
    )
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite split manifest: {manifest_path}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=True), encoding="utf-8")

    write_json_report(
        compute_pg_replenishment_report(
            catalog.entries,
            targets={
                source: {
                    "train": 0,
                    "validation": validation_counts[source],
                    "test": test_empirical_counts[source],
                }
                for source in SOURCES
            },
            allowed_signal_reliabilities=SIGNAL_RELIABILITIES,
            selected_entries=entries,
        ),
        pg_replenishment_path,
        overwrite=args.overwrite,
    )

    print_panel(
        "Holdout-first splits built",
        f"Selected {len(entries)}/{len(catalog.entries)} catalog records\n"
        f"Policy: {SPLIT_POLICY}\n"
        f"Manifest: {manifest_path}",
        style="green",
    )
    print_key_value_table(
        "Effective split counts",
        [
            (split, ", ".join(f"{source}={split_counts[split][source]}" for source in SOURCES))
            for split in SPLITS
        ],
    )
    print(
        json.dumps(
            {"manifest": str(manifest_path), "counts": split_counts}, indent=2, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
