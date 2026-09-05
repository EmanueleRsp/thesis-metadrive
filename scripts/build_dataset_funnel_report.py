#!/usr/bin/env python3
"""Reconstruct the ScenarioNet dataset-construction funnel as thesis-ready tables and figures.

The dataset chapter needs one reproducible account of how 66,854 converted
scenarios became the 3,500 frozen records, and the pipeline never wrote that
account in a single place: each stage left its own artifact, and
`catalog/split_report.json` is a stale `SCENARIONET-INTEGRATION` v1.1 report
that must not be used for the v1.2 dataset (it still claims 1,000 training
records per source against the frozen index's 1,100).

This command reads only immutable pipeline artifacts, never opens a
ScenarioDescription file, and writes only to the explicit output directory. It
records the sha256 of every input it consumed so a published table can be traced
back to the exact artifact that produced it.

Stages, and the artifact that establishes each one:

    S0 converted pool          catalog/catalog_report.json
    S1 Rulebook eligibility    rulebook_v2/catalog_eligibility.json
    S2 driving-mission build   rulebook_v2/driving_mission_eligibility.json
    S3 split candidate pool    catalog/scenario_catalog_rulebook_v2.parquet
    S4 frozen selection        frozen/scenario_selection_index.json

S3 needs `pyarrow`, which is present in the `dataset-pipeline` image and absent
from a bare host interpreter. Without it the stage is reported as NOT_COMPUTED
rather than guessed, and every other stage still runs.

Figures are emitted as standalone SVG so that no plotting dependency is added to
the project. They are vector output and can be included directly in the thesis
or converted with any SVG toolchain.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

ARMS = (
    "A0_simple_low_traffic",
    "A1_traffic",
    "A2_junction",
    "A3_complex_junction",
    "A4_vru",
    "A5_critical_mixed",
)
SOURCES = ("waymo", "pg")
POOLS = (
    ("train", None),
    ("validation", "empirical"),
    ("test", "empirical"),
    ("test", "stratified"),
)
POOL_LABELS = {
    ("train", None): "train",
    ("validation", "empirical"): "validation (empirical)",
    ("test", "empirical"): "test (empirical)",
    ("test", "stratified"): "test (arm-stratified)",
}
PG_PROFILES = (
    "P0_simple",
    "P1_vehicle_interaction",
    "P2_merge_or_roundabout",
    "P3_intersection",
    "P5_complex_mixed",
)
DESCRIPTIVE_FEATURES = (
    ("length", "scenario length (control steps)"),
    ("route_length_m", "assigned route length (m)"),
    ("relevant_agents_q90", "relevant agents (q90)"),
    ("relevant_vehicles_q90", "relevant vehicles (q90)"),
    ("relevant_vrus_q90", "relevant VRUs (q90)"),
    ("vehicle_conflict_count", "vehicle conflicts"),
    ("vru_conflict_count", "VRU conflicts"),
    ("map_feature_count", "map features"),
    ("dynamic_object_count", "dynamic objects"),
)
FROZEN_SCHEMAS = frozenset(
    {
        "scenarionet_frozen_selection_v1",
        "scenarionet_frozen_selection_mission_v1_1_1",
    }
)

SOURCE_COLOR = {"waymo": "#2f6f9f", "pg": "#c9702a"}
SOURCE_LABEL = {"waymo": "Waymo", "pg": "PG"}
AXIS_COLOR = "#333333"
GRID_COLOR = "#d8d8d8"
FONT = "Helvetica, Arial, sans-serif"


# --------------------------------------------------------------------------- io


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


# ------------------------------------------------------------------ statistics


def percentile(sorted_values: Sequence[float], fraction: float) -> float:
    """Linear-interpolated percentile over an already sorted, non-empty sequence."""

    if not sorted_values:
        raise ValueError("percentile requires at least one value")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = fraction * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower]) * (1.0 - weight) + float(sorted_values[upper]) * weight


def describe(values: Sequence[float]) -> dict[str, float] | None:
    """Return the descriptive summary published for a dataset feature."""

    finite = sorted(float(value) for value in values if value is not None)
    if not finite:
        return None
    return {
        "n": float(len(finite)),
        "min": finite[0],
        "p05": percentile(finite, 0.05),
        "p50": percentile(finite, 0.50),
        "mean": sum(finite) / len(finite),
        "p95": percentile(finite, 0.95),
        "max": finite[-1],
    }


def error_category(message: str) -> str:
    """Collapse a validation message to the stable category before its first colon."""

    return message.split(":", 1)[0].strip() or "unspecified"


def records_by_error_category(
    records: Iterable[dict[str, Any]],
    *,
    eligible_key: str,
) -> dict[str, dict[str, int]]:
    """Count *records* (not messages) rejected under each error category, per source.

    `excluded_by_cause` in the pipeline artifacts counts messages, and one record
    can carry several, so those numbers do not sum to the excluded-record count.
    This aggregation is per record and does sum correctly once a record is
    attributed to every category it triggered.
    """

    counts: dict[str, dict[str, int]] = {}
    for record in records:
        if record.get(eligible_key) is not False:
            continue
        source = str(record.get("source", "unknown"))
        categories = {error_category(str(error)) for error in record.get("validation_errors", ())}
        if not categories:
            categories = {"unspecified"}
        for category in categories:
            bucket = counts.setdefault(category, {"waymo": 0, "pg": 0, "total": 0})
            bucket[source] = bucket.get(source, 0) + 1
            bucket["total"] += 1
    return counts


# ----------------------------------------------------------------- funnel model


@dataclass(frozen=True)
class FunnelStage:
    key: str
    label: str
    artifact: str
    counts: dict[str, int] | None
    note: str = ""

    @property
    def total(self) -> int | None:
        if self.counts is None:
            return None
        return sum(self.counts.values())


@dataclass
class FunnelReport:
    stages: list[FunnelStage] = field(default_factory=list)
    inputs: dict[str, str] = field(default_factory=dict)

    def rows(self) -> list[list[Any]]:
        first_total = next((stage.total for stage in self.stages if stage.total), None)
        previous_total: int | None = None
        rows: list[list[Any]] = []
        for stage in self.stages:
            total = stage.total
            if total is None:
                rows.append(
                    [stage.key, stage.label, "", "", "", "", "", stage.artifact, stage.note]
                )
                continue
            retained_overall = 100.0 * total / first_total if first_total else float("nan")
            retained_stage = (
                100.0 * total / previous_total if previous_total not in (None, 0) else float("nan")
            )
            rows.append(
                [
                    stage.key,
                    stage.label,
                    stage.counts.get("waymo", 0),
                    stage.counts.get("pg", 0),
                    total,
                    f"{retained_stage:.1f}" if not math.isnan(retained_stage) else "",
                    f"{retained_overall:.1f}",
                    stage.artifact,
                    stage.note,
                ]
            )
            previous_total = total
        return rows


def build_funnel(
    *,
    catalog_report: dict[str, Any],
    eligibility: dict[str, Any],
    mission: dict[str, Any],
    candidate_counts: dict[str, int] | None,
    frozen_records: Sequence[dict[str, Any]],
) -> list[FunnelStage]:
    """Assemble the five funnel stages from the pipeline artifacts."""

    converted = {source: int(catalog_report["by_source"].get(source, 0)) for source in SOURCES}
    rulebook = {
        source: int(eligibility["counts_by_source"][source]["eligible"]) for source in SOURCES
    }
    mission_eligible = {
        source: int(mission["counts_by_source"][source]["eligible"]) for source in SOURCES
    }
    selected: dict[str, int] = {source: 0 for source in SOURCES}
    for record in frozen_records:
        source = str(record.get("source"))
        if source in selected:
            selected[source] += 1

    return [
        FunnelStage(
            key="S0",
            label="Converted scenarios in the raw catalog",
            artifact="catalog/catalog_report.json",
            counts=converted,
            note="Waymo training_20s shards converted by ScenarioNet, plus every exported PG seed.",
        ),
        FunnelStage(
            key="S1",
            label="Rulebook-eligible",
            artifact="rulebook_v2/catalog_eligibility.json",
            counts=rulebook,
            note="Rulebook v4.7 fail-closed eligibility over the frozen ego and calibration hashes.",
        ),
        FunnelStage(
            key="S2",
            label="Driving-mission buildable",
            artifact="rulebook_v2/driving_mission_eligibility.json",
            counts=mission_eligible,
            note="DRIVING-MISSION v1.1.1 offline route build; correction-first, no runtime fallback.",
        ),
        FunnelStage(
            key="S3",
            label="Split candidate pool",
            artifact="catalog/scenario_catalog_rulebook_v2.parquet",
            counts=candidate_counts,
            note=(
                "validation_status in {valid, warning} and rulebook_eligible; v1.2 applies no "
                "signal-reliability filter, unlike v1.1."
            ),
        ),
        FunnelStage(
            key="S4",
            label="Frozen selection",
            artifact="frozen/scenario_selection_index.json",
            counts=selected,
            note="Holdout-first allocation under split_seed=0; the dataset used by every run.",
        ),
    ]


# ------------------------------------------------------------------ svg drawing


def _escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _text(
    x: float,
    y: float,
    content: str,
    *,
    size: float = 11.0,
    anchor: str = "start",
    color: str = AXIS_COLOR,
    weight: str = "normal",
) -> str:
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="{FONT}" font-size="{size:.1f}" '
        f'fill="{color}" text-anchor="{anchor}" font-weight="{weight}">{_escape(content)}</text>'
    )


def _rect(
    x: float, y: float, width: float, height: float, fill: str, *, opacity: float = 1.0
) -> str:
    return (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(width, 0.0):.2f}" '
        f'height="{max(height, 0.0):.2f}" fill="{fill}" fill-opacity="{opacity:.2f}"/>'
    )


def _line(
    x1: float, y1: float, x2: float, y2: float, *, color: str = GRID_COLOR, width: float = 1.0
) -> str:
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{color}" stroke-width="{width:.2f}"/>'
    )


def _svg(width: float, height: float, body: Sequence[str], *, title: str) -> str:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
        f'viewBox="0 0 {width:.0f} {height:.0f}" role="img">',
        f"<title>{_escape(title)}</title>",
        f'<rect width="{width:.0f}" height="{height:.0f}" fill="#ffffff"/>',
        *body,
        "</svg>",
    ]
    return "\n".join(parts) + "\n"


def _source_legend(x: float, y: float) -> list[str]:
    body: list[str] = []
    offset = 0.0
    for source in SOURCES:
        body.append(_rect(x + offset, y - 9, 11, 11, SOURCE_COLOR[source]))
        body.append(_text(x + offset + 16, y, SOURCE_LABEL[source], size=11))
        offset += 78
    return body


def figure_funnel(stages: Sequence[FunnelStage]) -> str:
    """Horizontal stacked funnel: one bar per stage, split by source."""

    drawn = [stage for stage in stages if stage.counts is not None]
    width, left, right = 900.0, 250.0, 120.0
    row_height, top = 58.0, 70.0
    height = top + row_height * len(drawn) + 60.0
    span = width - left - right
    maximum = max(stage.total or 0 for stage in drawn) or 1

    body = [_text(24, 34, "Dataset construction funnel", size=16, weight="bold")]
    body += _source_legend(left, height - 28)
    for index, stage in enumerate(drawn):
        y = top + index * row_height
        body.append(_text(24, y + 18, f"{stage.key}  {stage.label}", size=11.5))
        offset = left
        for source in SOURCES:
            value = int(stage.counts.get(source, 0))
            bar = span * value / maximum
            body.append(_rect(offset, y, bar, 26, SOURCE_COLOR[source]))
            if bar > 46:
                body.append(
                    _text(
                        offset + bar / 2,
                        y + 18,
                        f"{value:,}",
                        size=10.5,
                        anchor="middle",
                        color="#ffffff",
                    )
                )
            offset += bar
        total = stage.total or 0
        share = 100.0 * total / (drawn[0].total or 1)
        body.append(_text(offset + 10, y + 18, f"{total:,}  ({share:.1f}%)", size=11))
    return _svg(width, height, body, title="Dataset construction funnel")


def figure_arm_distribution(distribution: dict[tuple[str, str | None, str, str], int]) -> str:
    """Grouped bars: one panel per pool, one stacked bar per arm."""

    panel_width, panel_height = 400.0, 190.0
    gap_x, gap_y = 40.0, 66.0
    left, top = 56.0, 76.0
    width = left + 2 * panel_width + gap_x + 24.0
    height = top + 2 * panel_height + gap_y + 66.0
    maximum = 1
    for pool in POOLS:
        for arm in ARMS:
            stacked = sum(distribution.get((*pool, source, arm), 0) for source in SOURCES)
            maximum = max(maximum, stacked)
    step = max(1, 10 ** (len(str(maximum)) - 1) // 2)
    ticks = list(range(0, maximum + step, step))[:8]

    body = [
        _text(24, 34, "Arm distribution by pool and source", size=16, weight="bold"),
        _text(24, 54, "Bars are stacked: Waymo below, PG above.", size=11, color="#666666"),
    ]
    body += _source_legend(left, height - 26)
    for index, pool in enumerate(POOLS):
        column, row = index % 2, index // 2
        origin_x = left + column * (panel_width + gap_x)
        origin_y = top + row * (panel_height + gap_y)
        base = origin_y + panel_height
        body.append(_text(origin_x, origin_y - 10, POOL_LABELS[pool], size=12.5, weight="bold"))
        for tick in ticks:
            y = base - panel_height * tick / maximum
            body.append(_line(origin_x, y, origin_x + panel_width, y))
            body.append(
                _text(origin_x - 8, y + 4, f"{tick}", size=9.5, anchor="end", color="#777777")
            )
        slot = panel_width / len(ARMS)
        for arm_index, arm in enumerate(ARMS):
            x = origin_x + arm_index * slot + slot * 0.18
            bar_width = slot * 0.64
            offset = base
            for source in SOURCES:
                value = distribution.get((*pool, source, arm), 0)
                bar = panel_height * value / maximum
                body.append(_rect(x, offset - bar, bar_width, bar, SOURCE_COLOR[source]))
                offset -= bar
            stacked = sum(distribution.get((*pool, source, arm), 0) for source in SOURCES)
            body.append(
                _text(x + bar_width / 2, offset - 5, f"{stacked}", size=9.5, anchor="middle")
            )
            body.append(
                _text(x + bar_width / 2, base + 15, arm.split("_")[0], size=10, anchor="middle")
            )
        body.append(_line(origin_x, base, origin_x + panel_width, base, color=AXIS_COLOR))
    return _svg(width, height, body, title="Arm distribution by pool and source")


def figure_cdf(
    series: dict[str, Sequence[float]],
    *,
    title: str,
    x_label: str,
    clip_percentile: float = 0.99,
) -> str:
    """Empirical CDF, one curve per source, clipped at a high percentile for legibility."""

    width, height = 640.0, 380.0
    left, right, top, bottom = 72.0, 28.0, 66.0, 62.0
    plot_w = width - left - right
    plot_h = height - top - bottom
    base = top + plot_h
    upper = 0.0
    prepared: dict[str, list[float]] = {}
    for source, values in series.items():
        ordered = sorted(float(value) for value in values if value is not None)
        prepared[source] = ordered
        if ordered:
            upper = max(upper, percentile(ordered, clip_percentile))
    upper = upper or 1.0

    body = [
        _text(24, 30, title, size=15, weight="bold"),
        _text(
            24, 48, f"Clipped at the {clip_percentile:.0%} percentile.", size=10.5, color="#666666"
        ),
    ]
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = base - plot_h * fraction
        body.append(_line(left, y, left + plot_w, y))
        body.append(
            _text(left - 8, y + 4, f"{fraction:.0%}", size=9.5, anchor="end", color="#777777")
        )
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = left + plot_w * fraction
        body.append(
            _text(
                x, base + 20, f"{upper * fraction:,.0f}", size=9.5, anchor="middle", color="#777777"
            )
        )
    body.append(_line(left, base, left + plot_w, base, color=AXIS_COLOR))
    body.append(_line(left, top, left, base, color=AXIS_COLOR))
    body.append(_text(left + plot_w / 2, base + 42, x_label, size=11, anchor="middle"))

    for source, ordered in prepared.items():
        if not ordered:
            continue
        points: list[str] = []
        total = len(ordered)
        for index, value in enumerate(ordered, start=1):
            x = left + plot_w * min(value, upper) / upper
            y = base - plot_h * index / total
            points.append(f"{x:.2f},{y:.2f}")
            if value >= upper:
                break
        body.append(
            f'<polyline fill="none" stroke="{SOURCE_COLOR[source]}" stroke-width="2" '
            f'points="{" ".join(points)}"/>'
        )
    body += _source_legend(left + 12, top + 22)
    return _svg(width, height, body, title=title)


def figure_pg_mixture(declared: dict[str, float], observed: dict[str, float]) -> str:
    """Declared versus observed PG profile mixture in the empirical holdouts."""

    width, height = 700.0, 360.0
    left, right, top, bottom = 62.0, 24.0, 78.0, 96.0
    plot_w = width - left - right
    plot_h = height - top - bottom
    base = top + plot_h
    maximum = max([*declared.values(), *observed.values(), 0.05]) * 1.15

    body = [
        _text(24, 30, "PG generation mixture: declared versus observed", size=15, weight="bold"),
        _text(
            24,
            50,
            "The 20% x 5 contract binds generation batches; the post-filter mixture is an observed"
            " consequence.",
            size=10.5,
            color="#666666",
        ),
    ]
    for fraction in (0.0, 0.1, 0.2, 0.3, 0.4):
        if fraction > maximum:
            continue
        y = base - plot_h * fraction / maximum
        body.append(_line(left, y, left + plot_w, y))
        body.append(
            _text(left - 8, y + 4, f"{fraction:.0%}", size=9.5, anchor="end", color="#777777")
        )
    slot = plot_w / len(PG_PROFILES)
    for index, profile in enumerate(PG_PROFILES):
        x = left + index * slot
        for offset, (value, color, label) in enumerate(
            (
                (declared.get(profile, 0.0), "#8a8a8a", "declared"),
                (observed.get(profile, 0.0), SOURCE_COLOR["pg"], "observed"),
            )
        ):
            bar = plot_h * value / maximum
            bar_x = x + slot * (0.18 + offset * 0.33)
            bar_w = slot * 0.30
            body.append(_rect(bar_x, base - bar, bar_w, bar, color))
            body.append(
                _text(bar_x + bar_w / 2, base - bar - 6, f"{value:.1%}", size=9.5, anchor="middle")
            )
        body.append(
            _text(x + slot / 2, base + 18, profile.split("_")[0], size=10.5, anchor="middle")
        )
        body.append(
            _text(
                x + slot / 2,
                base + 32,
                profile.split("_", 1)[1].replace("_", " "),
                size=9,
                anchor="middle",
                color="#777777",
            )
        )
    body.append(_line(left, base, left + plot_w, base, color=AXIS_COLOR))
    body.append(_rect(left, height - 26, 11, 11, "#8a8a8a"))
    body.append(_text(left + 16, height - 17, "declared (20% each)", size=10.5))
    body.append(_rect(left + 170, height - 26, 11, 11, SOURCE_COLOR["pg"]))
    body.append(_text(left + 186, height - 17, "observed after filtering", size=10.5))
    return _svg(width, height, body, title="PG generation mixture")


# ------------------------------------------------------------------- reporting


def _markdown_table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(str(cell) for cell in header) + " |"]
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for row in rows:
        lines.append("| " + " | ".join("" if cell is None else str(cell) for cell in row) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    default_root = os.environ.get("SCENARIONET_DATA_ROOT", "data/scenarionet")
    parser.add_argument("--data-root", default=default_root, help="ScenarioNet data root.")
    parser.add_argument(
        "--frozen-index", default=None, help="Override the frozen selection index path."
    )
    parser.add_argument(
        "--filtered-catalog", default=None, help="Override the Rulebook-filtered catalog path."
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory receiving the report, tables and figures."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Allow writing into a non-empty output directory."
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    frozen_path = Path(args.frozen_index or data_root / "frozen" / "scenario_selection_index.json")
    filtered_catalog = Path(
        args.filtered_catalog or data_root / "catalog" / "scenario_catalog_rulebook_v2.parquet"
    )
    catalog_report_path = data_root / "catalog" / "catalog_report.json"
    arm_report_path = data_root / "catalog" / "arm_report.json"
    thresholds_path = data_root / "splits" / "arm_thresholds.json"
    eligibility_path = data_root / "rulebook_v2" / "catalog_eligibility.json"
    mission_path = data_root / "rulebook_v2" / "driving_mission_eligibility.json"
    shard_ledger = data_root / "waymo" / "acquisition" / "converted_shards.txt"

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        parser.error(
            f"refusing to write into a non-empty directory without --overwrite: {output_dir}"
        )
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    for directory in (output_dir, tables_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    required = {
        "catalog_report": catalog_report_path,
        "arm_report": arm_report_path,
        "arm_thresholds": thresholds_path,
        "catalog_eligibility": eligibility_path,
        "driving_mission_eligibility": mission_path,
        "frozen_index": frozen_path,
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        parser.error("missing pipeline artifacts: " + ", ".join(missing))

    catalog_report = _load_json(catalog_report_path)
    arm_report = _load_json(arm_report_path)
    thresholds = _load_json(thresholds_path)
    eligibility = _load_json(eligibility_path)
    mission = _load_json(mission_path)
    frozen = _load_json(frozen_path)
    if frozen.get("schema") not in FROZEN_SCHEMAS:
        parser.error(f"unsupported frozen index schema: {frozen.get('schema')!r}")
    frozen_records = frozen["records"]
    split_manifest = frozen["split_manifest"]

    # S3 -- the only stage that needs the columnar catalog.
    candidate_counts: dict[str, int] | None = None
    candidate_detail: list[list[Any]] = []
    s3_reject_causes: dict[str, dict[str, int]] = {}
    parquet_note = ""
    try:
        import pyarrow.parquet as pq  # noqa: PLC0415 -- optional, container-only
    except ImportError:
        parquet_note = (
            "pyarrow is unavailable in this interpreter; run inside the dataset-pipeline image."
        )
    else:
        if not filtered_catalog.is_file():
            parquet_note = f"filtered catalog not found: {filtered_catalog}"
        else:
            wanted = [
                "source",
                "validation_status",
                "signal_reliability",
                "rulebook_eligible",
                "validation_warnings",
            ]
            schema_names = set(pq.read_schema(filtered_catalog).names)
            columns = [name for name in wanted if name in schema_names]
            table = pq.read_table(filtered_catalog, columns=columns).to_pylist()
            candidate_counts = {source: 0 for source in SOURCES}
            breakdown: dict[tuple[str, str, str], int] = {}
            for row in table:
                status = str(row.get("validation_status"))
                source = str(row.get("source"))
                if status in {"valid", "warning"} and row.get("rulebook_eligible") is True:
                    if source in candidate_counts:
                        candidate_counts[source] += 1
                    key = (source, status, str(row.get("signal_reliability")))
                    breakdown[key] = breakdown.get(key, 0) + 1
                    continue
                # Rejected at S3. `mark_hard_quality_failures` sets `invalid` exactly when an
                # offline quality filter fires, so the record's warnings are its rejection reasons.
                categories = {
                    error_category(str(warning))
                    for warning in (row.get("validation_warnings") or ())
                } or {"unspecified"}
                for category in categories:
                    bucket = s3_reject_causes.setdefault(
                        category, {"waymo": 0, "pg": 0, "total": 0}
                    )
                    bucket[source] = bucket.get(source, 0) + 1
                    bucket["total"] += 1
            candidate_detail = [
                [source, status, reliability, count]
                for (source, status, reliability), count in sorted(breakdown.items())
            ]

    stages = build_funnel(
        catalog_report=catalog_report,
        eligibility=eligibility,
        mission=mission,
        candidate_counts=candidate_counts,
        frozen_records=frozen_records,
    )
    report = FunnelReport(stages=list(stages))

    # ------------------------------------------------------------------ tables
    funnel_header = [
        "stage",
        "description",
        "waymo",
        "pg",
        "total",
        "retained_vs_previous_pct",
        "retained_vs_converted_pct",
        "artifact",
        "note",
    ]
    _write_csv(tables_dir / "funnel.csv", funnel_header, report.rows())

    rulebook_causes = records_by_error_category(
        eligibility["records"], eligible_key="rulebook_eligible"
    )
    mission_causes = records_by_error_category(mission["records"], eligible_key="eligible")
    cause_rows = (
        [
            ["S1_rulebook", category, counts["waymo"], counts["pg"], counts["total"]]
            for category, counts in sorted(
                rulebook_causes.items(), key=lambda item: -item[1]["total"]
            )
        ]
        + [
            ["S2_driving_mission", category, counts["waymo"], counts["pg"], counts["total"]]
            for category, counts in sorted(
                mission_causes.items(), key=lambda item: -item[1]["total"]
            )
        ]
        + [
            ["S3_offline_quality", category, counts["waymo"], counts["pg"], counts["total"]]
            for category, counts in sorted(
                s3_reject_causes.items(), key=lambda item: -item[1]["total"]
            )
        ]
    )
    _write_csv(
        tables_dir / "exclusion_causes.csv",
        ["stage", "cause_category", "waymo_records", "pg_records", "total_records"],
        cause_rows,
    )

    distribution: dict[tuple[str, str | None, str, str], int] = {}
    for record in frozen_records:
        key = (
            str(record.get("split")),
            record.get("holdout_pool"),
            str(record.get("source")),
            str(record.get("primary_arm")),
        )
        distribution[key] = distribution.get(key, 0) + 1
    distribution_rows = [
        [split, pool or "", source, arm, distribution.get((split, pool, source, arm), 0)]
        for split, pool in POOLS
        for source in SOURCES
        for arm in ARMS
    ]
    _write_csv(
        tables_dir / "split_source_arm.csv",
        ["split", "holdout_pool", "source", "primary_arm", "count"],
        distribution_rows,
    )

    stats_rows: list[list[Any]] = []
    for feature, label in DESCRIPTIVE_FEATURES:
        for source in SOURCES:
            summary = describe(
                [record.get(feature) for record in frozen_records if record.get("source") == source]
            )
            if summary is None:
                continue
            stats_rows.append(
                [
                    feature,
                    label,
                    source,
                    int(summary["n"]),
                    f"{summary['min']:.2f}",
                    f"{summary['p05']:.2f}",
                    f"{summary['p50']:.2f}",
                    f"{summary['mean']:.2f}",
                    f"{summary['p95']:.2f}",
                    f"{summary['max']:.2f}",
                ]
            )
    _write_csv(
        tables_dir / "descriptive_stats.csv",
        ["feature", "label", "source", "n", "min", "p05", "p50", "mean", "p95", "max"],
        stats_rows,
    )

    panel_rows: list[list[Any]] = []
    for name, payload in sorted(frozen.get("panel_manifests", {}).items()):
        manifest = payload["manifest"]
        panel_rows.append(
            [
                name,
                manifest.get("scope"),
                manifest.get("split"),
                manifest.get("source"),
                manifest.get("draw_policy"),
                manifest.get("size"),
                manifest.get("seed"),
                "",
                manifest.get("sha256"),
            ]
        )
    for scope, panels in sorted(frozen.get("profile_panel_manifests", {}).items()):
        for name, payload in sorted(panels.items()):
            manifest = payload["manifest"]
            panel_rows.append(
                [
                    f"{scope}_{name}",
                    manifest.get("scope"),
                    manifest.get("split"),
                    manifest.get("source"),
                    manifest.get("draw_policy"),
                    manifest.get("size"),
                    manifest.get("seed"),
                    manifest.get("parent_panel") or "",
                    manifest.get("sha256"),
                ]
            )
    _write_csv(
        tables_dir / "panels.csv",
        [
            "panel",
            "scope",
            "split",
            "source",
            "draw_policy",
            "size",
            "seed",
            "parent_panel",
            "sha256",
        ],
        panel_rows,
    )

    holdout_policy = split_manifest.get("holdout_policy", {})
    declared_mixture = holdout_policy.get("pg_holdout_mixture", {})
    observed_mixture = holdout_policy.get("pg_holdout_mixture_observed", {})
    profile_selected: dict[str, int] = {}
    profile_holdout: dict[str, int] = {}
    for record in frozen_records:
        profile = record.get("pg_profile")
        if not profile:
            continue
        profile_selected[profile] = profile_selected.get(profile, 0) + 1
        if record.get("holdout_pool") == "empirical":
            profile_holdout[profile] = profile_holdout.get(profile, 0) + 1
    _write_csv(
        tables_dir / "pg_profile_mixture.csv",
        [
            "pg_profile",
            "declared_share",
            "observed_holdout_share",
            "empirical_holdout_records",
            "selected_records",
        ],
        [
            [
                profile,
                f"{declared_mixture.get(profile, 0.0):.4f}",
                f"{observed_mixture.get(profile, 0.0):.4f}",
                profile_holdout.get(profile, 0),
                profile_selected.get(profile, 0),
            ]
            for profile in PG_PROFILES
        ],
    )

    _write_csv(
        tables_dir / "arm_thresholds.csv",
        ["parameter", "value"],
        [[key, value] for key, value in sorted(thresholds.items())],
    )

    if candidate_detail:
        _write_csv(
            tables_dir / "candidate_pool_quality.csv",
            ["source", "validation_status", "signal_reliability", "count"],
            candidate_detail,
        )

    # ----------------------------------------------------------------- figures
    (figures_dir / "fig_funnel.svg").write_text(figure_funnel(stages), encoding="utf-8")
    (figures_dir / "fig_arm_distribution.svg").write_text(
        figure_arm_distribution(distribution), encoding="utf-8"
    )
    for feature, label, filename in (
        ("route_length_m", "assigned route length (m)", "fig_route_length_cdf.svg"),
        ("relevant_agents_q90", "relevant agents (q90)", "fig_agent_density_cdf.svg"),
    ):
        series = {
            source: [
                record[feature]
                for record in frozen_records
                if record.get("source") == source and record.get(feature) is not None
            ]
            for source in SOURCES
        }
        (figures_dir / filename).write_text(
            figure_cdf(series, title=f"Distribution of {label}", x_label=label),
            encoding="utf-8",
        )
    (figures_dir / "fig_pg_mixture.svg").write_text(
        figure_pg_mixture(declared_mixture, observed_mixture), encoding="utf-8"
    )

    # -------------------------------------------------------------- provenance
    provenance = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "scripts/build_dataset_funnel_report.py",
        "data_root": str(data_root),
        "frozen_selection_hash": frozen.get("selection_hash"),
        "frozen_created_at": frozen.get("created_at"),
        "split_policy": split_manifest.get("split_policy"),
        "split_seed": split_manifest.get("split_seed"),
        "rulebook_version": eligibility.get("rulebook_version"),
        "mission_builder_identity": mission.get("builder_identity"),
        "waymo_dataset_version": frozen.get("data_policy", {}).get("waymo_dataset_version"),
        "converted_shards": (
            sum(1 for line in shard_ledger.read_text(encoding="utf-8").splitlines() if line.strip())
            if shard_ledger.is_file()
            else None
        ),
        "inputs": {
            name: {"path": str(path), "sha256": _digest_file(path)}
            for name, path in required.items()
        },
    }
    if filtered_catalog.is_file() and candidate_counts is not None:
        provenance["inputs"]["filtered_catalog"] = {
            "path": str(filtered_catalog),
            "sha256": _digest_file(filtered_catalog),
        }
    if parquet_note:
        provenance["stage_s3_note"] = parquet_note
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # ------------------------------------------------------------------ report
    lines = [
        "# Dataset construction funnel",
        "",
        f"Generated {provenance['generated_at']} by `scripts/build_dataset_funnel_report.py`.",
        "Every number below is read from a pipeline artifact whose sha256 is recorded in",
        "`provenance.json`. `catalog/split_report.json` is deliberately NOT a source: it is a",
        "stale `SCENARIONET-INTEGRATION` v1.1 report and disagrees with the frozen v1.2 dataset.",
        "",
        "## 1. Funnel",
        "",
        _markdown_table(funnel_header[:7], [row[:7] for row in report.rows()]),
        "",
    ]
    if parquet_note:
        lines += [f"> Stage S3 NOT_COMPUTED: {parquet_note}", ""]
    lines += [
        "## 2. Exclusion causes, counted per record",
        "",
        "The pipeline's own `excluded_by_cause` counts *messages*; a record carrying several",
        "messages is counted several times. The table below counts records, attributing each",
        "record to every category it triggered.",
        "",
        _markdown_table(
            ["stage", "cause category", "waymo", "pg", "total"],
            cause_rows,
        ),
        "",
        "## 3. Frozen selection by pool, source and arm",
        "",
        _markdown_table(
            ["pool", "source", *[arm.split("_")[0] for arm in ARMS], "total"],
            [
                [
                    POOL_LABELS[(split, pool)],
                    SOURCE_LABEL[source],
                    *[distribution.get((split, pool, source, arm), 0) for arm in ARMS],
                    sum(distribution.get((split, pool, source, arm), 0) for arm in ARMS),
                ]
                for split, pool in POOLS
                for source in SOURCES
            ],
        ),
        "",
        "## 4. Arm thresholds in force",
        "",
        _markdown_table(
            ["parameter", "value"], [[key, value] for key, value in sorted(thresholds.items())]
        ),
        "",
        "## 5. Artifacts written",
        "",
        "- `tables/funnel.csv`, `tables/exclusion_causes.csv`, `tables/split_source_arm.csv`",
        "- `tables/descriptive_stats.csv`, `tables/panels.csv`, `tables/pg_profile_mixture.csv`",
        "- `tables/arm_thresholds.csv`"
        + (", `tables/candidate_pool_quality.csv`" if candidate_detail else ""),
        "- `figures/fig_funnel.svg`, `figures/fig_arm_distribution.svg`,"
        " `figures/fig_route_length_cdf.svg`, `figures/fig_agent_density_cdf.svg`,"
        " `figures/fig_pg_mixture.svg`",
        "- `provenance.json`",
        "",
    ]
    (output_dir / "funnel.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote dataset funnel report to {output_dir}")
    for stage in stages:
        total = stage.total
        print(f"  {stage.key} {stage.label}: {total if total is not None else 'NOT_COMPUTED'}")
    print(f"  arm_report cross-check total: {arm_report['arms']['total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
