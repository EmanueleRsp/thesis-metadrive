#!/usr/bin/env python3
"""Test the Waymo train/holdout split for geographic co-location.

ADR-038 adopted a `map_identity_fingerprint` as the Waymo grouping key so that
"two 20-second windows recorded over the same road geometry" cannot land in
different splits. Measured on the frozen artifacts, that key merges **406 of
54,104** catalogued records (0.8 %) and **zero** of the 1,805 selected ones:
every selected record is its own group, so group-disjointness is vacuously
satisfied and proves nothing.

That is expected from the implementation
(`src/thesis_rl/scenarios/waymo.py:150`): the fingerprint hashes an exact,
sorted sample of metre-quantized polyline points, so it collapses only
*identically cropped* maps. Two overlapping windows over the same intersection
carry different map crops and therefore different fingerprints.

The converted scenarios keep WOMD's shared metric frame -- map extents run to
several kilometres from the origin and are not recentred per scenario -- so
co-location can be measured directly instead of being hashed for. This script
measures it: bounding-box overlap between every train and every holdout record,
refined by the Jaccard overlap of their metre-quantized map-point sets.

Read-only.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

QUANTIZATION_M = 2.0
# Report thresholds on the Jaccard overlap of quantized map points.
JACCARD_THRESHOLDS = (0.01, 0.05, 0.10, 0.25, 0.50, 0.90)


def _footprint(path: Path) -> tuple[frozenset[tuple[int, int]], tuple[float, float, float, float]] | None:
    try:
        with path.open("rb") as handle:
            scenario = pickle.load(handle)
    except Exception:  # noqa: BLE001 - diagnostic script
        return None
    features = scenario.get("map_features")
    if not isinstance(features, dict):
        return None
    cells: set[tuple[int, int]] = set()
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    for feature in features.values():
        if not isinstance(feature, dict):
            continue
        polyline = feature.get("polyline")
        if polyline is None:
            continue
        for point in polyline:
            try:
                x, y = float(point[0]), float(point[1])
            except (TypeError, IndexError, ValueError):
                continue
            cells.add((int(x // QUANTIZATION_M), int(y // QUANTIZATION_M)))
            min_x, max_x = min(min_x, x), max(max_x, x)
            min_y, max_y = min(min_y, y), max(max_y, y)
    if not cells:
        return None
    return frozenset(cells), (min_x, min_y, max_x, max_y)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--index", required=True, help="frozen scenario_selection_index.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    index = json.loads(Path(args.index).expanduser().read_text(encoding="utf-8"))
    records = [r for r in index["records"] if r["source"] == "waymo"]

    footprints: dict[str, frozenset[tuple[int, int]]] = {}
    split_of: dict[str, str] = {}
    unreadable = 0
    for position, record in enumerate(records):
        result = _footprint(data_root / record["relative_path"])
        if result is None:
            unreadable += 1
            continue
        cells, _bbox = result
        uid = record["scenario_uid"]
        footprints[uid] = cells
        split_of[uid] = record["split"]
        if (position + 1) % 250 == 0:
            print(f"loaded {position + 1}/{len(records)}", flush=True)

    # Invert to a cell -> uids index so only scenarios sharing at least one cell
    # are ever compared; everything else has Jaccard 0 by construction.
    by_cell: dict[tuple[int, int], list[str]] = defaultdict(list)
    for uid, cells in footprints.items():
        for cell in cells:
            by_cell[cell].append(uid)

    candidate_pairs: set[tuple[str, str]] = set()
    for uids in by_cell.values():
        if len(uids) < 2:
            continue
        ordered = sorted(uids)
        for i, first in enumerate(ordered):
            for second in ordered[i + 1 :]:
                candidate_pairs.add((first, second))

    cross: list[dict[str, Any]] = []
    within_train = 0
    for first, second in candidate_pairs:
        split_a, split_b = split_of[first], split_of[second]
        cells_a, cells_b = footprints[first], footprints[second]
        intersection = len(cells_a & cells_b)
        if not intersection:
            continue
        jaccard = intersection / len(cells_a | cells_b)
        containment = intersection / min(len(cells_a), len(cells_b))
        if split_a == split_b:
            if split_a == "train":
                within_train += 1
            continue
        cross.append(
            {
                "a": first,
                "a_split": split_a,
                "b": second,
                "b_split": split_b,
                "jaccard": jaccard,
                "containment": containment,
                "shared_cells": intersection,
            }
        )

    cross.sort(key=lambda item: item["containment"], reverse=True)

    # Containment is the measure that matters for leakage and Jaccard is not:
    # WOMD crops each scenario's map to its own extent, so a small holdout crop
    # fully contained in a large training crop scores a low Jaccard while being
    # completely seen during training. Per holdout record, keep the largest
    # fraction of its own map that also appears in some *training* record.
    max_containment: dict[str, float] = {}
    for item in cross:
        pairing = {item["a_split"]: item["a"], item["b_split"]: item["b"]}
        if "train" not in pairing:
            continue
        for split, uid in pairing.items():
            if split == "train":
                continue
            own = footprints[uid]
            other = footprints[pairing["train"]]
            fraction = len(own & other) / len(own)
            max_containment[uid] = max(max_containment.get(uid, 0.0), fraction)
    holdout_total = sum(1 for value in split_of.values() if value != "train")
    report = {
        "schema": "waymo-split-colocation-v1",
        "quantization_m": QUANTIZATION_M,
        "records": len(records),
        "measured": len(footprints),
        "unreadable": unreadable,
        "split_counts": {
            split: sum(1 for value in split_of.values() if value == split)
            for split in sorted(set(split_of.values()))
        },
        "cross_split_overlapping_pairs": len(cross),
        "within_train_overlapping_pairs": within_train,
        "cross_split_pairs_above_jaccard": {
            f"{threshold:g}": sum(1 for item in cross if item["jaccard"] >= threshold)
            for threshold in JACCARD_THRESHOLDS
        },
        "holdout_records": holdout_total,
        "holdout_records_by_max_train_containment": {
            f">={threshold:g}": sum(
                1 for value in max_containment.values() if value >= threshold
            )
            for threshold in JACCARD_THRESHOLDS
        },
        "holdout_max_containment_quantiles": {
            label: sorted(max_containment.values())[
                min(len(max_containment) - 1, int(fraction * (len(max_containment) - 1)))
            ]
            for label, fraction in (("p50", 0.5), ("p90", 0.9), ("p99", 0.99), ("max", 1.0))
        }
        if max_containment
        else {},
        "holdout_records_touching_train": len(
            {
                (item["a"] if item["a_split"] != "train" else item["b"])
                for item in cross
                if "train" in (item["a_split"], item["b_split"])
            }
        ),
        "worst_pairs": cross[:25],
    }
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "worst_pairs"}, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
