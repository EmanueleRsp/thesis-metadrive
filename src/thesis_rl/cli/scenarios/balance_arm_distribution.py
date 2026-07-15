"""Balance the classified ScenarioNet catalog across semantic arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from thesis_rl.scenarios.arms import ARMS
from thesis_rl.scenarios.catalog import read_scenario_catalog, write_scenario_catalog
from thesis_rl.scenarios.pipeline import balance_arm_distribution
from thesis_rl.scenarios.reports import (
    compute_arm_distribution,
    compute_feature_statistics,
    write_json_report,
)
from thesis_rl.cli.scenarios.ui import print_key_value_table, print_panel


def _read_threshold_values(path: str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--output-catalog", required=True)
    parser.add_argument("--target-total", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prefer-source", choices=("waymo", "pg"), default="waymo")
    parser.add_argument("--thresholds")
    parser.add_argument("--report")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    catalog = read_scenario_catalog(args.catalog)
    balanced, balance_report = balance_arm_distribution(
        catalog.entries,
        target_total=int(args.target_total),
        seed=int(args.seed),
        prefer_source=str(args.prefer_source),
    )
    write_scenario_catalog(
        balanced,
        args.output_catalog,
        overwrite=args.overwrite,
    )
    report = {
        "catalog": str(Path(args.output_catalog).expanduser().resolve()),
        "thresholds": str(Path(args.thresholds).expanduser().resolve())
        if args.thresholds
        else None,
        "threshold_values": _read_threshold_values(args.thresholds),
        "balance": balance_report,
        "features": compute_feature_statistics(balanced),
        "arms": compute_arm_distribution(balanced),
    }
    report_path = Path(
        args.report
        or Path(args.output_catalog).expanduser().resolve().parent / "arm_report.json"
    )
    write_json_report(report, report_path, overwrite=args.overwrite)
    print_panel(
        "Arm-balanced catalog ready",
        f"Selected {balance_report['selected_records']}/"
        f"{balance_report['input_records']} records\n"
        f"Target per arm: {balance_report['target_per_arm']}\n"
        f"Total arm deficit: {balance_report['total_deficit']}\n"
        f"Catalog: {args.output_catalog}",
        style="green" if balance_report["total_deficit"] == 0 else "yellow",
    )
    print_key_value_table(
        "Balanced arm counts",
        [
            (
                arm,
                f"{balance_report['diagnostics'][arm]['after']}"
                f"/{balance_report['diagnostics'][arm]['target']}",
            )
            for arm in ARMS
        ],
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
