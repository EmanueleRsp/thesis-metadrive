from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Suggest rule scales from logged rule margins (JSONL)."
    )
    parser.add_argument(
        "--input",
        required=True,
        nargs="+",
        help="One or more JSONL margin logs (supports glob patterns).",
    )
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
    parser.add_argument(
        "--min-active-margin",
        type=float,
        default=1e-9,
        help="Threshold to consider a margin as active: abs(margin) > threshold.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=0,
        help="Minimum active samples required per rule.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any rule has fewer active samples than `--min-samples`.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write JSON report (scales + coverage).",
    )
    return parser.parse_args()


def _resolve_inputs(inputs: list[str]) -> list[Path]:
    resolved: list[Path] = []
    for item in inputs:
        pattern = str(item).strip()
        if not pattern:
            continue
        matches = sorted(Path().glob(pattern))
        if matches:
            resolved.extend(path for path in matches if path.is_file())
            continue
        candidate = Path(pattern)
        if candidate.is_file():
            resolved.append(candidate)
    # Preserve order but remove duplicates.
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in resolved:
        norm = path.resolve()
        if norm in seen:
            continue
        seen.add(norm)
        unique.append(path)
    return unique


def suggest_scales(
    *,
    input_paths: list[Path],
    percentile: float,
    min_scale: float,
    min_active_margin: float,
    min_samples: int,
    strict: bool,
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    if percentile <= 0.0 or percentile >= 100.0:
        raise ValueError("`percentile` must be in (0, 100).")
    if min_scale <= 0.0:
        raise ValueError("`min_scale` must be > 0.")
    if min_active_margin < 0.0:
        raise ValueError("`min_active_margin` must be >= 0.")
    if min_samples < 0:
        raise ValueError("`min_samples` must be >= 0.")
    if not input_paths:
        raise ValueError("No valid input files resolved from `--input`.")

    by_rule: dict[str, list[float]] = defaultdict(list)

    for input_path in input_paths:
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
    coverage: dict[str, dict[str, Any]] = {}
    insufficient_rules: list[str] = []

    for name, values in sorted(by_rule.items()):
        arr = np.asarray(values, dtype=np.float64)
        active_mask = arr > float(min_active_margin)
        active = arr[active_mask]
        total_samples = int(arr.size)
        active_samples = int(active.size)
        sufficient = active_samples >= int(min_samples)
        if not sufficient:
            insufficient_rules.append(name)

        source = active if active_samples > 0 else arr
        q = float(np.percentile(source, percentile))
        suggestions[name] = max(q, min_scale)
        coverage[name] = {
            "total_samples": total_samples,
            "active_samples": active_samples,
            "min_samples_required": int(min_samples),
            "sufficient_samples": bool(sufficient),
            "min_active_margin": float(min_active_margin),
        }

    if strict and insufficient_rules:
        details = ", ".join(
            f"{name}({coverage[name]['active_samples']}/{min_samples})"
            for name in insufficient_rules
        )
        raise ValueError(
            "Insufficient active samples for strict scale tuning: "
            f"{details}. Run more normal/forced scenarios."
        )
    return suggestions, coverage


def main() -> None:
    args = _parse_args()
    inputs = _resolve_inputs(list(args.input))
    suggestions, coverage = suggest_scales(
        input_paths=inputs,
        percentile=float(args.percentile),
        min_scale=float(args.min_scale),
        min_active_margin=float(args.min_active_margin),
        min_samples=int(args.min_samples),
        strict=bool(args.strict),
    )

    print("inputs:")
    for path in inputs:
        print(f"  - {path}")

    print("coverage:")
    for name, stats in coverage.items():
        print(
            f"  {name}: active={stats['active_samples']} total={stats['total_samples']} "
            f"required>={stats['min_samples_required']} sufficient={stats['sufficient_samples']}"
        )

    print("scales:")
    for name, value in suggestions.items():
        print(f"  {name}: {value:.6g}")

    if args.output_json:
        output_path = Path(str(args.output_json))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "inputs": [str(path) for path in inputs],
            "percentile": float(args.percentile),
            "min_scale": float(args.min_scale),
            "min_active_margin": float(args.min_active_margin),
            "min_samples": int(args.min_samples),
            "strict": bool(args.strict),
            "coverage": coverage,
            "scales": suggestions,
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
