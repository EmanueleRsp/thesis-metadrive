from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from thesis_rl.common.paths import default_outputs_glob_str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate final_eval.csv files and print a sanity-check comparison table."
    )
    parser.add_argument(
        "--pattern",
        default=default_outputs_glob_str("**/csv/final_eval.csv"),
        help="Glob pattern used to discover final_eval.csv files.",
    )
    return parser.parse_args()


def _nested_get(mapping: dict[str, Any], path: list[str], default: Any = None) -> Any:
    cur: Any = mapping
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _read_run_meta(final_eval_path: str) -> dict[str, Any]:
    run_dir = Path(final_eval_path).resolve().parent.parent
    hydra_cfg = run_dir / "hydra" / "config.yaml"
    meta = {
        "run_dir": str(run_dir),
        "config_name": None,
        "curriculum_enabled": None,
    }
    if not hydra_cfg.is_file():
        return meta
    try:
        cfg = yaml.safe_load(hydra_cfg.read_text(encoding="utf-8")) or {}
    except Exception:
        return meta

    meta["config_name"] = _nested_get(cfg, ["metadata", "config_name"], None)
    meta["curriculum_enabled"] = _nested_get(cfg, ["curriculum", "enabled"], None)
    return meta


def main() -> None:
    args = _parse_args()
    files = sorted(glob.glob(args.pattern, recursive=True))
    if not files:
        raise SystemExit("No final_eval.csv found")

    parts: list[pd.DataFrame] = []
    for file_path in files:
        row = pd.read_csv(file_path).assign(_src=file_path)
        meta = _read_run_meta(file_path)
        row = row.assign(
            run_dir=meta["run_dir"],
            config_name=meta["config_name"],
            curriculum_enabled=meta["curriculum_enabled"],
        )
        parts.append(row)

    df = pd.concat(parts, ignore_index=True)

    cols = [
        c
        for c in [
            "config_name",
            "curriculum_enabled",
            "reward_mode",
            "seed",
            "success_rate",
            "collision_rate",
            "out_of_road_rate",
            "route_completion",
            "top_rule_violation_rate",
            "avg_error_value",
            "max_error_value",
        ]
        if c in df.columns
    ]
    print(df[cols].sort_values(["curriculum_enabled", "reward_mode", "seed"]).to_string(index=False))


if __name__ == "__main__":
    main()
