from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Suggest rule scales from logged rule margins (JSONL)."
    )
    parser.add_argument("--input", required=True, help="Path to JSONL margin log file.")
    parser.add_argument(
        "--percentile",
        type=float,
        default=90.0,
        help="Absolute-margin percentile used as suggested scale (default: 90).",
    )
    parser.add_argument(
        "--min-scale",
        type=float,
        default=1e-6,
        help="Lower bound for suggested scales (default: 1e-6).",
    )
    return parser.parse_args()


def suggest_scales(
    *,
    input_path: Path,
    percentile: float,
    min_scale: float,
) -> dict[str, float]:
    if percentile <= 0.0 or percentile >= 100.0:
        raise ValueError("`percentile` must be in (0, 100).")
    if min_scale <= 0.0:
        raise ValueError("`min_scale` must be > 0.")
    if not input_path.exists():
        raise FileNotFoundError(f"Margin log file not found: {input_path}")

    by_rule: dict[str, list[float]] = defaultdict(list)

    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            item = json.loads(raw)
            components = item.get("rule_components")
            if not isinstance(components, dict):
                continue
            for name, margin in components.items():
                by_rule[str(name)].append(abs(float(margin)))

    if not by_rule:
        raise ValueError("No `rule_components` found in margin log.")

    suggestions: dict[str, float] = {}
    for name, values in sorted(by_rule.items()):
        arr = np.asarray(values, dtype=np.float64)
        q = float(np.percentile(arr, percentile))
        suggestions[name] = max(q, min_scale)
    return suggestions


def main() -> None:
    args = _parse_args()
    suggestions = suggest_scales(
        input_path=Path(args.input),
        percentile=float(args.percentile),
        min_scale=float(args.min_scale),
    )

    print("scales:")
    for name, value in suggestions.items():
        print(f"  {name}: {value:.6g}")


if __name__ == "__main__":
    main()
