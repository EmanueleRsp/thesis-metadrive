"""Manually record the EVAL-PROTOCOL REQ-011 three-way disposition for a run.

Disposition is never inferred automatically from metric values (REQ-011
invariant): a human reviews the run and records one of three dispositions
plus a rationale, patched additively into ``run_metadata.yaml``.

Usage:
    python scripts/set_run_disposition.py <artifacts_dir> <disposition> "<rationale>"

Where <disposition> is one of:
    infrastructure_failure
    non_convergent
    condition_attributable_failure
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thesis_rl.runtime.io.metadata import update_run_metadata

VALID_DISPOSITIONS = (
    "infrastructure_failure",
    "non_convergent",
    "condition_attributable_failure",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts_dir", type=Path)
    parser.add_argument("disposition", choices=VALID_DISPOSITIONS)
    parser.add_argument("rationale", type=str)
    args = parser.parse_args()

    if not args.rationale.strip():
        parser.error("rationale must not be empty (EVAL-PROTOCOL REQ-011)")

    path = update_run_metadata(
        args.artifacts_dir,
        {"disposition": args.disposition, "disposition_rationale": args.rationale},
    )
    print(f"Recorded disposition={args.disposition!r} in {path}")


if __name__ == "__main__":
    main()
