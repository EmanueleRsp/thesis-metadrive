from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Iterative loop helper: aggregate margin logs and enforce strict scale coverage."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input JSONL logs/patterns for aggregation (normal + forced).",
    )
    parser.add_argument(
        "--aggregated-output",
        default="outputs/scale_calibration/aggregated_rule_margins.jsonl",
        help="Aggregated JSONL output path.",
    )
    parser.add_argument(
        "--report-json",
        default="outputs/scale_calibration/scale_report.json",
        help="Scale tuning JSON report path.",
    )
    parser.add_argument("--percentile", type=float, default=90.0)
    parser.add_argument("--min-scale", type=float, default=1e-6)
    parser.add_argument("--min-active-margin", type=float, default=1e-9)
    parser.add_argument("--min-samples", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    aggregate_cmd = [
        "python",
        "-m",
        "thesis_rl.tools.calibration.aggregate_rule_margins",
        "--output",
        str(args.aggregated_output),
        "--input",
        *list(args.inputs),
    ]
    subprocess.run(aggregate_cmd, check=True)

    tune_cmd = [
        "python",
        "-m",
        "thesis_rl.tools.calibration.scale_tuning",
        "--input",
        str(args.aggregated_output),
        "--percentile",
        str(args.percentile),
        "--min-scale",
        str(args.min_scale),
        "--min-active-margin",
        str(args.min_active_margin),
        "--min-samples",
        str(args.min_samples),
        "--strict",
        "--output-json",
        str(args.report_json),
    ]

    completed = subprocess.run(tune_cmd, check=False)
    if completed.returncode != 0:
        print("\nStrict tuning failed due to insufficient coverage.")
        print("Action: run additional normal/forced diagnostics, then rerun this command.")
        raise SystemExit(completed.returncode)

    report = json.loads(Path(args.report_json).read_text(encoding="utf-8"))
    print("\nScale calibration complete.")
    print("Suggested scales:")
    for name, value in report.get("scales", {}).items():
        print(f"  {name}: {value}")
    print("\nNext action: copy `scales` into conf/reward/base_rulebook.yaml")


if __name__ == "__main__":
    main()
