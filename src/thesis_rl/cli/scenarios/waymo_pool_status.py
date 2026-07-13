"""Report whether the converted Waymo pool satisfies the eligible target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from thesis_rl.scenarios.waymo import load_converted_waymo_entries
from thesis_rl.scenarios.waymo_pool import (
    WAYMO_POOL_POLICY_VERSION,
    fingerprint_waymo_database,
    summarize_waymo_pool,
)


def _cached_payload(
    report: Path,
    *,
    database_fingerprint: str,
    required: int,
    allowed: list[str],
) -> dict[str, Any] | None:
    if not report.is_file():
        return None
    payload = json.loads(report.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    expected = {
        "policy_version": WAYMO_POOL_POLICY_VERSION,
        "database_fingerprint": database_fingerprint,
        "required": required,
        "allowed_signal_reliabilities": sorted(allowed),
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        return None
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--required", type=int, required=True)
    parser.add_argument(
        "--allowed-signal-reliability",
        action="append",
        dest="allowed",
        required=True,
    )
    parser.add_argument("--report")
    parser.add_argument("--shards-output")
    parser.add_argument("--reuse-report-if-current", action="store_true")
    parser.add_argument("--format", choices=("json", "env"), default="json")
    args = parser.parse_args()

    database = Path(args.database).expanduser().resolve()
    report = Path(args.report).expanduser() if args.report else None
    database_fingerprint = fingerprint_waymo_database(database)
    payload = (
        _cached_payload(
            report,
            database_fingerprint=database_fingerprint,
            required=args.required,
            allowed=args.allowed,
        )
        if args.reuse_report_if_current and report is not None
        else None
    )
    if payload is None:
        if database.is_dir() and any(database.rglob("sd_*.pkl")):
            entries, _ = load_converted_waymo_entries(
                database,
                data_root=Path(args.data_root).expanduser().resolve(),
            )
        else:
            entries = ()
        status = summarize_waymo_pool(
            entries,
            allowed_signal_reliabilities=args.allowed,
            required=args.required,
        )
        payload = {
            "policy_version": WAYMO_POOL_POLICY_VERSION,
            "database_fingerprint": database_fingerprint,
            "allowed_signal_reliabilities": sorted(args.allowed),
            "total": status.total,
            "eligible": status.eligible,
            "required": status.required,
            "deficit": status.deficit,
            "complete": status.complete,
            "by_signal_reliability": status.by_signal_reliability,
            "eligible_by_arm": status.eligible_by_arm,
            "source_shards": list(status.source_shards),
        }
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.shards_output:
        shards_output = Path(args.shards_output).expanduser()
        shards_output.parent.mkdir(parents=True, exist_ok=True)
        shards_output.write_text(
            "\n".join(str(value) for value in payload["source_shards"]) + "\n",
            encoding="utf-8",
        )

    if args.format == "env":
        print(f"SCENARIONET_WAYMO_POOL_TOTAL\t{payload['total']}")
        print(f"SCENARIONET_WAYMO_ELIGIBLE_COUNT\t{payload['eligible']}")
        print(f"SCENARIONET_WAYMO_ELIGIBLE_DEFICIT\t{payload['deficit']}")
        complete = "true" if payload["complete"] else "false"
        print(f"SCENARIONET_WAYMO_POOL_COMPLETE\t{complete}")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
