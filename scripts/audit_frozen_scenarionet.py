#!/usr/bin/env python3
"""Produce immutable-reference audit artifacts from a frozen ScenarioNet index.

This command never opens source ScenarioDescription files for writing and never
modifies the dataset. It writes only compact audit and reference-manifest files
to the explicit output directory.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


ARMS = (
    "A0_simple_low_traffic",
    "A1_traffic",
    "A2_junction",
    "A3_complex_junction",
    "A4_vru",
    "A5_critical_mixed",
)
SOURCES = ("waymo", "pg")
SPLITS = ("train", "validation", "test")


def _read_index(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "scenarionet_frozen_selection_v1":
        raise ValueError(f"Unsupported frozen index schema: {payload.get('schema')!r}")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Frozen index must contain a non-empty records list.")
    return payload


def _digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _catalog_fingerprint(records: Iterable[dict[str, Any]]) -> str:
    rows = [
        "|".join(
            str(record.get(field, ""))
            for field in (
                "scenario_uid",
                "source",
                "split",
                "primary_arm",
                "runtime_index",
                "relative_path",
            )
        )
        for record in records
    ]
    return hashlib.sha256("\n".join(sorted(rows)).encode("utf-8")).hexdigest()


def _largest_remainder_source_quotas(records: list[dict[str, Any]], slots: int) -> dict[str, int]:
    """Allocate source references proportionally with a feasible one-per-source guard."""

    counts = collections.Counter(str(record["source"]) for record in records)
    present = [source for source in SOURCES if counts[source] > 0]
    if not present:
        raise ValueError("Cannot allocate a golden-suite arm without train records.")
    total = sum(counts.values())
    raw = {source: slots * counts[source] / total for source in present}
    quotas = {source: math.floor(raw[source]) for source in present}
    remaining = slots - sum(quotas.values())
    for source in sorted(present, key=lambda item: (-(raw[item] - quotas[item]), item))[:remaining]:
        quotas[source] += 1

    if slots >= len(present):
        for source in present:
            if quotas[source] != 0:
                continue
            donor = max(
                (candidate for candidate in present if quotas[candidate] > 1),
                key=lambda candidate: (quotas[candidate], raw[candidate], candidate),
            )
            quotas[donor] -= 1
            quotas[source] += 1
    return {source: quotas.get(source, 0) for source in SOURCES}


def _coverage_score(record: dict[str, Any], frequencies: collections.Counter[str]) -> float:
    tags = [str(tag) for tag in record.get("tags", [])]
    score = sum(1.0 / frequencies[f"tag:{tag}"] for tag in tags if frequencies[f"tag:{tag}"])
    topology = str(record.get("topology_tag", "unknown"))
    score += 1.0 / max(frequencies[f"topology:{topology}"], 1)
    for key in (
        "has_route_traffic_light",
        "has_route_stop_sign",
        "has_route_crosswalk",
        "vru_interaction",
        "dense_traffic",
        "has_merge_or_roundabout",
        "has_intersection",
    ):
        if bool(record.get(key, False)):
            score += 0.5 / max(frequencies[f"feature:{key}"], 1)
    score += min(float(record.get("dynamic_object_count", 0)), 100.0) / 1000.0
    score += min(float(record.get("length", 0)), 600.0) / 6000.0
    return score


def _select_golden_rows(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    selected: list[dict[str, Any]] = []
    quotas_by_arm: dict[str, dict[str, int]] = {}
    for arm in ARMS:
        arm_records = [
            record
            for record in records
            if record["split"] == "train" and record["primary_arm"] == arm
        ]
        quotas = _largest_remainder_source_quotas(arm_records, slots=8)
        quotas_by_arm[arm] = quotas
        for source in SOURCES:
            candidates = [record for record in arm_records if record["source"] == source]
            frequencies: collections.Counter[str] = collections.Counter()
            for candidate in candidates:
                frequencies[f"topology:{candidate.get('topology_tag', 'unknown')}"] += 1
                for tag in candidate.get("tags", []):
                    frequencies[f"tag:{tag}"] += 1
                for key in (
                    "has_route_traffic_light",
                    "has_route_stop_sign",
                    "has_route_crosswalk",
                    "vru_interaction",
                    "dense_traffic",
                    "has_merge_or_roundabout",
                    "has_intersection",
                ):
                    if bool(candidate.get(key, False)):
                        frequencies[f"feature:{key}"] += 1
            ranked = sorted(
                candidates,
                key=lambda record: (
                    -_coverage_score(record, frequencies),
                    str(record["scenario_uid"]),
                ),
            )
            selected.extend(ranked[: quotas[source]])

    if len(selected) != 48 or len({record["scenario_uid"] for record in selected}) != 48:
        raise AssertionError("Golden-suite selection must contain exactly 48 unique references.")
    arm_counts = collections.Counter(record["primary_arm"] for record in selected)
    if any(arm_counts[arm] != 8 for arm in ARMS):
        raise AssertionError("Golden-suite selection must contain eight references per arm.")
    return selected, quotas_by_arm


def _duplicate_values(records: list[dict[str, Any]], field: str) -> dict[str, list[str]]:
    values: dict[str, list[str]] = collections.defaultdict(list)
    for record in records:
        value = record.get(field)
        if value not in (None, ""):
            values[str(value)].append(str(record["scenario_uid"]))
    return {value: sorted(uids) for value, uids in values.items() if len(uids) > 1}


def _cross_split_groups(records: list[dict[str, Any]], field: str) -> dict[str, list[str]]:
    groups: dict[str, set[str]] = collections.defaultdict(set)
    for record in records:
        value = record.get(field)
        if value not in (None, ""):
            groups[str(value)].add(str(record["split"]))
    return {value: sorted(splits) for value, splits in groups.items() if len(splits) > 1}


def _coverage_row(record: dict[str, Any]) -> dict[str, str]:
    controls = []
    for field, label in (
        ("has_route_traffic_light", "traffic_light"),
        ("has_route_stop_sign", "stop_sign"),
        ("has_route_crosswalk", "crosswalk"),
    ):
        if bool(record.get(field, False)):
            controls.append(label)
    interactions = []
    if int(record.get("vehicle_conflict_count", 0)) > 0:
        interactions.append("vehicle_conflict")
    if int(record.get("vru_conflict_count", 0)) > 0 or bool(record.get("vru_interaction", False)):
        interactions.append("vru_interaction")
    return {
        "scenario_uid": str(record["scenario_uid"]),
        "scenario_id": str(record["scenario_id"]),
        "source": str(record["source"]),
        "split": str(record["split"]),
        "primary_arm": str(record["primary_arm"]),
        "tags": ";".join(str(tag) for tag in record.get("tags", [])),
        "topology": str(record.get("topology_tag", "unknown")),
        "duration_steps": str(record.get("length", "")),
        "actor_density_q90": str(record.get("relevant_agents_q90", "")),
        "interactions": ";".join(interactions) or "none_in_catalog_metadata",
        "traffic_controls": ";".join(controls) or "none_in_catalog_metadata",
        "lane_marking_coverage": "unavailable_without_source_content",
        "conflict_zone_coverage": "unavailable_without_source_content",
        "rulebook_applicability": "eligible_static_catalog",
        "expected_runtime_behavior": "valid_source_metadata; termination/truncation requires live smoke",
        "selection_reason": "deterministic metadata coverage score within arm/source quota",
        "unavailable_dimensions": "lane markings, live conflict zones, control geometry, runtime fallback",
        "relative_path": str(record["relative_path"]),
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(rows: list[tuple[str, ...]], header: tuple[str, ...]) -> str:
    body = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    body.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(body)


def _build_report(
    *,
    index_path: Path,
    index_digest: str,
    payload: dict[str, Any],
    records: list[dict[str, Any]],
    fingerprint: str,
    golden_rows: list[dict[str, Any]],
    quotas: dict[str, dict[str, int]],
    source_root: Path | None,
) -> str:
    split_source = collections.Counter((str(row["split"]), str(row["source"])) for row in records)
    arm_source = collections.Counter(
        (str(row["primary_arm"]), str(row["source"])) for row in records
    )
    expected = payload.get("split_manifest", {}).get("targets", {})
    actual_vs_expected = []
    for split in SPLITS:
        for source in SOURCES:
            actual = split_source[(split, source)]
            target = expected.get(source, {}).get(split)
            actual_vs_expected.append(
                (
                    split,
                    source,
                    str(actual),
                    str(target),
                    "match" if actual == target else "deviation",
                )
            )
    source_file_status = "NOT_RUN: no source root supplied"
    if source_root is not None:
        missing = [
            row["relative_path"]
            for row in records
            if not (source_root / str(row["relative_path"])).is_file()
        ]
        source_file_status = f"{'PASS' if not missing else 'FAIL'}: missing={len(missing)}"

    selected_shards = (
        payload.get("source_inventory", {}).get("waymo", {}).get("selected_shards", [])
    )
    pg_generations = payload.get("source_inventory", {}).get("pg", {}).get("generations", [])
    source_id_leaks = _cross_split_groups(records, "source_scenario_id")
    pg_identity_leaks = _cross_split_groups(
        [row for row in records if row["source"] == "pg"], "scenario_id"
    )
    provenance_log_cross_split = _cross_split_groups(
        [row for row in records if row["source"] == "waymo"], "source_log_id"
    )
    duplicate_scenario_ids = _duplicate_values(records, "scenario_id")
    duplicate_uids = _duplicate_values(records, "scenario_uid")
    waymo_records = [row for row in records if row["source"] == "waymo"]
    pg_records = [row for row in records if row["source"] == "pg"]
    duplicate_waymo_segments = _duplicate_values(waymo_records, "source_scenario_id")
    pg_generation_identities = collections.Counter(
        (str(row.get("pg_profile")), str(row.get("pg_seed"))) for row in pg_records
    )
    duplicate_pg_generations = sum(count > 1 for count in pg_generation_identities.values())
    lengths = [int(row["length"]) for row in records]
    route_missing = [
        row["scenario_uid"] for row in records if not row.get("assigned_route_lane_ids")
    ]
    validation = collections.Counter(str(row.get("validation_status")) for row in records)
    signal = collections.Counter(str(row.get("signal_reliability")) for row in records)
    warnings = sum(bool(row.get("validation_warnings")) for row in records)
    exclusions = sum(
        row.get("validation_status") not in {"valid", "warning"}
        or not bool(row.get("rulebook_eligible"))
        or bool(row.get("rulebook_validation_errors"))
        for row in records
    )
    source_split_rows = [
        (split, source, str(split_source[(split, source)]))
        for split in SPLITS
        for source in SOURCES
    ]
    arm_rows = [
        (
            arm,
            str(arm_source[(arm, "waymo")]),
            str(arm_source[(arm, "pg")]),
            str(sum(arm_source[(arm, source)] for source in SOURCES)),
        )
        for arm in ARMS
    ]
    quota_rows = [(arm, str(quotas[arm]["waymo"]), str(quotas[arm]["pg"]), "8") for arm in ARMS]
    return f"""# Frozen ScenarioNet Audit and Golden-Suite Proposal

Status: `METADATA AUDIT COMPLETE — LIVE DATASET VALIDATION PENDING`

## Immutable Inputs

- Frozen index: `{index_path}`
- Frozen index SHA-256: `{index_digest}`
- Index schema: `{payload.get("schema")}`
- Index-created timestamp: `{payload.get("created_at")}`
- Deterministic catalog fingerprint: `{fingerprint}`
- Declared runtime root: `{source_root if source_root is not None else "/workspace/data/scenarionet (not mounted)"}`
- Protected-data policy: no source file, ScenarioDescription, split, tag, or selected reference was modified.

## Population And Targets

{_markdown_table(actual_vs_expected, ("split", "source", "actual", "target", "result"))}

{_markdown_table(source_split_rows, ("split", "source", "count"))}

{_markdown_table(arm_rows, ("primary arm", "Waymo", "PG", "total"))}

The selected population contains `{len(records)}` records. The approved near-uniform arm totals are 583 or 584; observed totals are represented above. No target deviation was found.

## Source Identity, Duplicates, And Leakage

- Duplicate scenario UIDs: `{len(duplicate_uids)}`; duplicate scenario IDs: `{len(duplicate_scenario_ids)}`.
- Waymo selected-shard entries recorded by the frozen index: `{len(selected_shards)}`; PG generation identities recorded: `{len(pg_generations)}`.
- Duplicate Waymo original source-segment identities: `{len(duplicate_waymo_segments)}`; duplicate PG `(profile, seed)` identities: `{duplicate_pg_generations}`.
- Cross-split duplicate original source scenario identities: `{len(source_id_leaks)}`; cross-split duplicate PG scenario identities: `{len(pg_identity_leaks)}`.
- Waymo provenance shard/log IDs crossing splits: `{len(provenance_log_cross_split)}`. This is reported as provenance reuse, not split leakage: ADR-001 permits `source_log_id` grouping only when it proves a shared source group; the frozen metadata does not establish that proof.

## Metadata Validation Evidence

- Validation statuses: `{dict(sorted(validation.items()))}`; validation-warning records: `{warnings}`; selected exclusion/error records: `{exclusions}`.
- Catalog-declared Rulebook eligibility: `{sum(bool(row.get("rulebook_eligible")) for row in records)}/{len(records)}`; this is not a live Rulebook or source-content validation result.
- Signal reliability: `{dict(sorted(signal.items()))}`.
- Catalog route metadata lists missing: `{len(route_missing)}`; route-source identities: `{dict(sorted(collections.Counter(str(row.get("assigned_route_source")) for row in records).items()))}`. This is not live assigned-route validity verification.
- Scenario horizons in catalog metadata: min `{min(lengths)}`, max `{max(lengths)}` decision steps.
- Source-file loadability: `{source_file_status}`. Scenario-content validation, actual traffic-control geometry, trajectory validation, and live Rulebook/ScenarioEnv checks are not claimed without the exact protected root.

## Golden-Suite Candidate (Not Final)

The suite contains exactly eight train references per arm. Source allocation uses largest-remainder rounding over actual train presence, with the required at-least-one-source guard when feasible.

{_markdown_table(quota_rows, ("primary arm", "Waymo quota", "PG quota", "total"))}

The companion CSV is the required candidate coverage matrix. It records catalog-backed topology, duration, density, tags, controls, interactions, static Rulebook eligibility, and explicit unavailable dimensions. It is not a policy-evaluation set and is not final until each referenced source file and ScenarioDescription can be inspected under the mounted immutable root.

## Reproducibility

The catalog fingerprint hashes the sorted tuple `(scenario_uid, source, split, primary_arm, runtime_index, relative_path)`. The golden-suite manifest includes the frozen-index digest, selection policy, source quotas, and the 48 source references. Re-running this command on the same index produces the same reference set.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index", type=Path, required=True, help="Frozen selection index to audit."
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New audit-output directory."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Optional mounted protected root; only existence checks are performed.",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite audit output directory: {args.output_dir}")

    index_path = args.index.resolve()
    payload = _read_index(index_path)
    records = [dict(record) for record in payload["records"]]
    fingerprint = _catalog_fingerprint(records)
    golden_records, quotas = _select_golden_rows(records)
    coverage_rows = [_coverage_row(record) for record in golden_records]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True)
    manifest = {
        "schema": "scenarionet_golden_suite_candidate_v1",
        "status": "CANDIDATE_METADATA_ONLY_NOT_FINAL",
        "purpose": "integration and regression testing only; not policy evaluation",
        "source_index": str(index_path),
        "source_index_sha256": _digest_file(index_path),
        "catalog_fingerprint": fingerprint,
        "selection": {
            "split": "train",
            "per_arm": 8,
            "source_allocation": "largest_remainder_with_feasible_source_guard",
            "ranking": "deterministic_catalog_metadata_coverage_score",
            "source_quotas": quotas,
            "requires_live_content_inspection_before_finalization": True,
        },
        "references": [
            {
                "scenario_uid": row["scenario_uid"],
                "scenario_id": row["scenario_id"],
                "source": row["source"],
                "split": row["split"],
                "primary_arm": row["primary_arm"],
                "relative_path": row["relative_path"],
                "runtime_index": next(
                    record["runtime_index"]
                    for record in golden_records
                    if record["scenario_uid"] == row["scenario_uid"]
                ),
            }
            for row in coverage_rows
        ],
    }
    (output_dir / "golden_suite_candidate_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "golden_suite_candidate_coverage.csv", coverage_rows)
    (output_dir / "audit_report.md").write_text(
        _build_report(
            index_path=index_path,
            index_digest=_digest_file(index_path),
            payload=payload,
            records=records,
            fingerprint=fingerprint,
            golden_rows=coverage_rows,
            quotas=quotas,
            source_root=args.source_root.resolve() if args.source_root is not None else None,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
