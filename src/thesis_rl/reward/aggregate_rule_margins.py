from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate multiple rule margin JSONL files into one JSONL file."
    )
    parser.add_argument(
        "--input",
        required=True,
        nargs="+",
        help="One or more input JSONL files (supports glob patterns).",
    )
    parser.add_argument("--output", required=True, help="Output JSONL file path.")
    return parser.parse_args()


def _resolve_inputs(patterns: list[str]) -> list[Path]:
    resolved: list[Path] = []
    for pattern in patterns:
        s = str(pattern).strip()
        if not s:
            continue
        matches = sorted(Path().glob(s))
        if matches:
            resolved.extend(path for path in matches if path.is_file())
            continue
        path = Path(s)
        if path.is_file():
            resolved.append(path)

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in resolved:
        real = path.resolve()
        if real in seen:
            continue
        seen.add(real)
        unique.append(path)
    return unique


def main() -> None:
    args = _parse_args()
    inputs = _resolve_inputs(list(args.input))
    if not inputs:
        raise ValueError("No valid input files resolved from `--input`.")

    output = Path(str(args.output))
    output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with output.open("w", encoding="utf-8") as out:
        for path in inputs:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    raw = line.strip()
                    if not raw:
                        continue
                    try:
                        item = json.loads(raw)
                    except Exception:
                        continue
                    components = item.get("rule_components")
                    if not isinstance(components, dict):
                        continue
                    out.write(raw)
                    out.write("\n")
                    written += 1

    print(f"inputs: {len(inputs)}")
    for path in inputs:
        print(f"  - {path}")
    print(f"written_rows: {written}")
    print(f"output: {output}")


if __name__ == "__main__":
    main()

