#!/usr/bin/env python3
"""Re-evaluate frozen Rulebook eligibility under the current code and report divergences.

The frozen dataset's S1 gate was computed once, on 2026-08-04, under Rulebook
`4.7-final-implementation-complete`. `RULEBOOK-V5.1` has since landed and touched
both static adapters (commit `69e3402`, the `speed_limit` sub-rule). Reading the
diff suggests the change is purely additive — a new optional
`posted_speed_limit_mps` field, `None` for PG unconditionally and `None` for
Waymo without the `speed_limit_mph` provenance datum, with no new error path —
but a thesis cannot rest a 66,854-to-20,733 selection gate on a reading of a
diff.

This command re-runs the *same* evaluator (`evaluate_catalog_entries`) over a
deterministic stratified sample of the raw catalog and compares, per record,
both the boolean verdict and the exact validation-error tuple against the frozen
artifact. It writes nothing except its own report, and it never modifies the
dataset or the eligibility artifact.

A divergence does not necessarily mean the dataset is wrong — it means the
selection gate is no longer reproducible from the current code, which is a
statement the dataset chapter has to make explicitly either way.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from thesis_rl.rulebook.v2.calibration import load_calibration_artifact
from thesis_rl.rulebook.v2.config import RULEBOOK_V2_VERSION, geometry_config_hash
from thesis_rl.rulebook.v2.context.catalog_eligibility import evaluate_catalog_entries
from thesis_rl.scenarios.catalog import read_scenario_catalog

SOURCES = ("waymo", "pg")


def _canonical_json_hash(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _stratified_sample(
    uids_by_stratum: dict[tuple[str, bool], list[str]],
    *,
    per_stratum: int,
    seed: int,
) -> list[str]:
    """Take up to `per_stratum` uids from each (source, frozen verdict) stratum.

    Sampling both verdicts matters: a regression that made the gate *stricter*
    would only show up on records the freeze accepted, and one that made it
    *looser* only on records the freeze rejected.
    """

    rng = random.Random(seed)
    sample: list[str] = []
    for stratum in sorted(uids_by_stratum):
        pool = sorted(uids_by_stratum[stratum])
        sample.extend(pool if len(pool) <= per_stratum else rng.sample(pool, per_stratum))
    return sorted(sample)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", required=True, help="ScenarioNet data root.")
    parser.add_argument(
        "--per-stratum",
        type=int,
        default=150,
        help="Records per (source, frozen verdict) stratum; four strata.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output", required=True, help="JSON report path.")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    raw_catalog = data_root / "catalog" / "scenario_catalog_raw.parquet"
    eligibility_path = data_root / "rulebook_v2" / "catalog_eligibility.json"
    ego_config = data_root / "rulebook_v2" / "ego_config.json"
    calibration_path = data_root / "rulebook_v2" / "calibration_b_e.json"
    for path in (raw_catalog, eligibility_path, ego_config, calibration_path):
        if not path.is_file():
            parser.error(f"missing artifact: {path}")

    frozen = json.loads(eligibility_path.read_text(encoding="utf-8"))
    frozen_by_uid: dict[str, dict[str, Any]] = {
        record["scenario_uid"]: record for record in frozen["records"]
    }

    ego_hash = _canonical_json_hash(ego_config)
    calibration = load_calibration_artifact(calibration_path, expected_config_hash=ego_hash)
    geometry_hash = geometry_config_hash()

    catalog = read_scenario_catalog(raw_catalog)
    entries_by_uid = {entry.record.scenario_uid: entry for entry in catalog.entries}

    uids_by_stratum: dict[tuple[str, bool], list[str]] = {}
    for uid, record in frozen_by_uid.items():
        if uid not in entries_by_uid:
            continue
        key = (str(record["source"]), bool(record["rulebook_eligible"]))
        uids_by_stratum.setdefault(key, []).append(uid)
    sample_uids = _stratified_sample(
        uids_by_stratum, per_stratum=args.per_stratum, seed=args.seed
    )

    results = evaluate_catalog_entries(
        [entries_by_uid[uid] for uid in sample_uids],
        data_root=data_root,
        geometry_config_hash=geometry_hash,
        calibration_hash=calibration.config_hash,
        workers=args.workers,
    )

    verdict_divergences: list[dict[str, Any]] = []
    error_divergences: list[dict[str, Any]] = []
    for result in results:
        reference = frozen_by_uid[result.scenario_uid]
        if bool(result.rulebook_eligible) != bool(reference["rulebook_eligible"]):
            verdict_divergences.append(
                {
                    "scenario_uid": result.scenario_uid,
                    "source": reference["source"],
                    "frozen": reference["rulebook_eligible"],
                    "current": result.rulebook_eligible,
                }
            )
        if sorted(result.validation_errors) != sorted(reference.get("validation_errors", [])):
            error_divergences.append(
                {
                    "scenario_uid": result.scenario_uid,
                    "source": reference["source"],
                    "frozen": sorted(reference.get("validation_errors", []))[:5],
                    "current": sorted(result.validation_errors)[:5],
                }
            )

    report = {
        "schema": "rulebook-eligibility-stability-v1",
        "frozen_rulebook_version": frozen.get("rulebook_version"),
        "current_rulebook_version": RULEBOOK_V2_VERSION,
        "frozen_geometry_config_hash": frozen.get("geometry_config_hash"),
        "current_geometry_config_hash": geometry_hash,
        "geometry_config_hash_matches": frozen.get("geometry_config_hash") == geometry_hash,
        "frozen_calibration_hash": frozen.get("calibration_hash"),
        "current_calibration_hash": calibration.config_hash,
        "calibration_hash_matches": frozen.get("calibration_hash") == calibration.config_hash,
        "sampling": {
            "seed": args.seed,
            "per_stratum": args.per_stratum,
            "strata": {f"{source}:{verdict}": len(uids) for (source, verdict), uids in sorted(uids_by_stratum.items())},
            "evaluated_records": len(results),
        },
        "verdict_divergences": verdict_divergences,
        "error_divergences": error_divergences,
        "verdict_divergence_count": len(verdict_divergences),
        "error_divergence_count": len(error_divergences),
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Evaluated {len(results)} records under {RULEBOOK_V2_VERSION}")
    print(f"  geometry_config_hash matches frozen: {report['geometry_config_hash_matches']}")
    print(f"  calibration_hash matches frozen:     {report['calibration_hash_matches']}")
    print(f"  verdict divergences: {len(verdict_divergences)}")
    print(f"  error-tuple divergences: {len(error_divergences)}")
    print(f"Report written to {output}")
    return 1 if verdict_divergences or error_divergences else 0


if __name__ == "__main__":
    raise SystemExit(main())
