"""Tests for the dataset-construction funnel reporter (`DFR` plan).

The reporter publishes numbers that go straight into the thesis, so its
aggregation must be pinned: percentiles, per-record (not per-message) exclusion
attribution, funnel retention arithmetic, and the not-computed path taken when
`pyarrow` is unavailable. The figures are checked for well-formed XML rather
than appearance.
"""

from __future__ import annotations

import importlib.util
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_dataset_funnel_report.py"


def _load_reporter() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_dataset_funnel_report", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


reporter = _load_reporter()


CATALOG_REPORT = {"total": 30, "by_source": {"waymo": 20, "pg": 10}}
ELIGIBILITY = {"counts_by_source": {"waymo": {"eligible": 12}, "pg": {"eligible": 8}}}
MISSION = {"counts_by_source": {"waymo": {"eligible": 9}, "pg": {"eligible": 7}}}
FROZEN_RECORDS = [
    {"source": "waymo", "split": "train", "holdout_pool": None},
    {"source": "waymo", "split": "test", "holdout_pool": "empirical"},
    {"source": "pg", "split": "train", "holdout_pool": None},
]


def _funnel(candidate_counts: dict[str, int] | None):
    return reporter.build_funnel(
        catalog_report=CATALOG_REPORT,
        eligibility=ELIGIBILITY,
        mission=MISSION,
        candidate_counts=candidate_counts,
        frozen_records=FROZEN_RECORDS,
    )


def test_percentile_interpolates_between_neighbours() -> None:
    """TEST-DFR-001: percentiles interpolate linearly and hit the exact endpoints."""

    values = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert reporter.percentile(values, 0.0) == 0.0
    assert reporter.percentile(values, 1.0) == 4.0
    assert reporter.percentile(values, 0.5) == 2.0
    assert reporter.percentile(values, 0.125) == pytest.approx(0.5)
    assert reporter.percentile([7.0], 0.9) == 7.0
    with pytest.raises(ValueError):
        reporter.percentile([], 0.5)


def test_describe_summarizes_and_skips_missing_values() -> None:
    """TEST-DFR-002: `describe` ignores `None` and returns `None` for an empty feature."""

    summary = reporter.describe([3.0, None, 1.0, 2.0])
    assert summary is not None
    assert summary["n"] == 3.0
    assert summary["min"] == 1.0
    assert summary["max"] == 3.0
    assert summary["p50"] == 2.0
    assert summary["mean"] == pytest.approx(2.0)
    assert reporter.describe([]) is None
    assert reporter.describe([None]) is None


def test_error_category_keeps_the_prefix_before_the_first_colon() -> None:
    """TEST-DFR-003: message detail after the first colon never creates a new category."""

    assert reporter.error_category("assigned_route_invalid: lanes 3->4 not contiguous")
    assert (
        reporter.error_category("assigned_route_invalid: lanes 3->4 not contiguous")
        == "assigned_route_invalid"
    )
    assert (
        reporter.error_category("mission_build_error:ValueError:trim end") == "mission_build_error"
    )
    assert reporter.error_category("no_colon_here") == "no_colon_here"
    assert reporter.error_category("") == "unspecified"


def test_exclusion_causes_are_counted_per_record_not_per_message() -> None:
    """TEST-DFR-004: a record with two messages of one category is counted once.

    This is the whole reason the reporter exists rather than republishing the
    pipeline's own `excluded_by_cause`, which counts messages.
    """

    records = [
        {
            "source": "waymo",
            "rulebook_eligible": False,
            "validation_errors": ["route_invalid: a", "route_invalid: b", "signal_unknown: c"],
        },
        {"source": "pg", "rulebook_eligible": False, "validation_errors": ["route_invalid: d"]},
        {"source": "waymo", "rulebook_eligible": True, "validation_errors": ["route_invalid: e"]},
        {"source": "waymo", "rulebook_eligible": False, "validation_errors": []},
    ]
    counts = reporter.records_by_error_category(records, eligible_key="rulebook_eligible")

    assert counts["route_invalid"] == {"waymo": 1, "pg": 1, "total": 2}
    assert counts["signal_unknown"] == {"waymo": 1, "pg": 0, "total": 1}
    assert counts["unspecified"]["total"] == 1
    assert "route_invalid" in counts and counts["route_invalid"]["waymo"] == 1


def test_build_funnel_orders_stages_and_counts_each_source() -> None:
    """TEST-DFR-005: the five stages carry per-source counts and their source artifact."""

    stages = _funnel({"waymo": 5, "pg": 6})

    assert [stage.key for stage in stages] == ["S0", "S1", "S2", "S3", "S4"]
    assert [stage.total for stage in stages] == [30, 20, 16, 11, 3]
    assert stages[0].counts == {"waymo": 20, "pg": 10}
    assert stages[4].counts == {"waymo": 2, "pg": 1}
    assert stages[3].artifact.endswith("scenario_catalog_rulebook_v2.parquet")


def test_missing_pyarrow_stage_is_reported_not_guessed() -> None:
    """TEST-DFR-006: with no candidate counts, S3 stays empty instead of being interpolated."""

    rows = reporter.FunnelReport(stages=list(_funnel(None))).rows()
    s3 = next(row for row in rows if row[0] == "S3")

    assert s3[2:7] == ["", "", "", "", ""]
    assert [row[0] for row in rows] == ["S0", "S1", "S2", "S3", "S4"]


def test_retention_percentages_chain_across_computed_stages() -> None:
    """TEST-DFR-007: retention is stage-over-previous and stage-over-converted."""

    rows = reporter.FunnelReport(stages=list(_funnel({"waymo": 5, "pg": 5}))).rows()
    by_key = {row[0]: row for row in rows}

    assert by_key["S0"][6] == "100.0"
    assert by_key["S1"][5] == f"{100.0 * 20 / 30:.1f}"
    assert by_key["S2"][5] == f"{100.0 * 16 / 20:.1f}"
    assert by_key["S4"][6] == f"{100.0 * 3 / 30:.1f}"


def test_figures_are_well_formed_svg() -> None:
    """TEST-DFR-008: every figure parses as XML and declares an SVG root."""

    distribution = {
        ("train", None, "waymo", "A0_simple_low_traffic"): 20,
        ("train", None, "pg", "A0_simple_low_traffic"): 346,
        ("test", "stratified", "waymo", "A4_vru"): 50,
    }
    documents = [
        reporter.figure_funnel(_funnel({"waymo": 5, "pg": 6})),
        reporter.figure_arm_distribution(distribution),
        reporter.figure_cdf({"waymo": [1.0, 2.0, 3.0], "pg": [2.0, 4.0]}, title="t", x_label="x"),
        reporter.figure_pg_mixture({"P0_simple": 0.2}, {"P0_simple": 0.35}),
    ]

    for document in documents:
        root = ET.fromstring(document)
        assert root.tag.endswith("svg")
        assert root.attrib["width"] and root.attrib["height"]


def test_figure_text_is_xml_escaped() -> None:
    """TEST-DFR-009: a label containing markup characters cannot break the document."""

    document = reporter.figure_cdf({"waymo": [1.0]}, title="a & b <c>", x_label="d > e")
    root = ET.fromstring(document)

    assert "a & b <c>" in {node.text for node in root.iter() if node.text}
