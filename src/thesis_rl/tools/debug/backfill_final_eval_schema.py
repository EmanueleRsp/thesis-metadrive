from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import yaml


RULEBOOK_MODE_BY_REWARD_MODE = {
    "scalar_native": "none",
    "scalar_default": "monitor_only",
    "scalar_rulebook": "scalar_reward",
}


def _must_get(mapping: dict[str, Any], path: list[str], cfg_path: Path) -> Any:
    cur: Any = mapping
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            raise ValueError(f"Missing key {'.'.join(path)} in {cfg_path}")
        cur = cur[key]
    return cur


def _optional_get(mapping: dict[str, Any], path: list[str], default: Any = None) -> Any:
    cur: Any = mapping
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        return list(reader.fieldnames), list(reader)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _ensure_fieldnames(fieldnames: list[str], required: list[str]) -> list[str]:
    out = list(fieldnames)
    for key in required:
        if key not in out:
            out.append(key)
    return out


def _infer_eval_fields(global_step: int | None, final_step: int | None) -> tuple[str, str]:
    is_final = global_step is not None and final_step is not None and global_step == final_step
    if is_final:
        return "final", "test"
    return "intermediate", "curriculum_eval"


def _patch_final_eval(final_eval_csv: Path, cfg: dict[str, Any], force: bool) -> int | None:
    reward_mode = str(_must_get(cfg, ["reward", "mode"], final_eval_csv)).strip()
    curriculum_enabled = bool(_must_get(cfg, ["curriculum", "enabled"], final_eval_csv))
    rulebook_name = str(_optional_get(cfg, ["reward", "rulebook"], "base_rulebook")).strip() or "base_rulebook"

    if reward_mode not in RULEBOOK_MODE_BY_REWARD_MODE:
        raise ValueError(
            f"Unsupported reward.mode='{reward_mode}' in {final_eval_csv}. "
            f"Expected one of: {sorted(RULEBOOK_MODE_BY_REWARD_MODE.keys())}"
        )
    rulebook_mode = RULEBOOK_MODE_BY_REWARD_MODE[reward_mode]

    fieldnames, rows = _read_csv(final_eval_csv)
    fieldnames = _ensure_fieldnames(
        fieldnames,
        [
            "curriculum_enabled",
            "rulebook_name",
            "rulebook_mode",
            "eval_type",
            "scenario_set",
            "steps_to_final_stage",
        ],
    )

    final_step_detected: int | None = None
    for row in rows:
        if force or str(row.get("curriculum_enabled", "")).strip() == "":
            row["curriculum_enabled"] = "true" if curriculum_enabled else "false"
        if force or str(row.get("rulebook_name", "")).strip() == "":
            row["rulebook_name"] = rulebook_name
        if force or str(row.get("rulebook_mode", "")).strip() == "":
            row["rulebook_mode"] = rulebook_mode
        if force or str(row.get("eval_type", "")).strip() == "":
            row["eval_type"] = "final"
        if force or str(row.get("scenario_set", "")).strip() == "":
            row["scenario_set"] = "test"
        if force or str(row.get("steps_to_final_stage", "")).strip() == "":
            reached = str(row.get("final_stage_reached", "")).strip().lower() in {"true", "1", "yes"}
            if curriculum_enabled:
                row["steps_to_final_stage"] = "0" if reached else "-1"
            else:
                row["steps_to_final_stage"] = "0"

        final_step_detected = _to_int(row.get("total_timesteps"))

    _write_csv(final_eval_csv, fieldnames, rows)
    print(f"patched: {final_eval_csv}")
    return final_step_detected


def _patch_eval_like_csv(path: Path, force: bool, final_step: int | None) -> None:
    if not path.exists():
        return
    fieldnames, rows = _read_csv(path)
    fieldnames = _ensure_fieldnames(fieldnames, ["eval_type", "scenario_set"])

    for row in rows:
        eval_type_default, scenario_set_default = _infer_eval_fields(_to_int(row.get("global_step")), final_step)
        if force or str(row.get("eval_type", "")).strip() == "":
            row["eval_type"] = eval_type_default
        if force or str(row.get("scenario_set", "")).strip() == "":
            row["scenario_set"] = scenario_set_default

    _write_csv(path, fieldnames, rows)
    print(f"patched: {path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill strict analysis columns across outputs/**/csv/*.csv "
            "using each run's hydra/config.yaml."
        )
    )
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--force", action="store_true", help="Overwrite existing values too.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outputs_root = Path(args.outputs_root)

    csv_dirs = sorted(path for path in outputs_root.glob("**/csv") if path.is_dir())
    if not csv_dirs:
        raise SystemExit(f"No csv directories found under {outputs_root}")

    patched_runs = 0
    for csv_dir in csv_dirs:
        run_dir = csv_dir.parent
        cfg_path = run_dir / "hydra" / "config.yaml"
        final_eval_csv = csv_dir / "final_eval.csv"

        if not cfg_path.exists() or not final_eval_csv.exists():
            continue

        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        if not isinstance(cfg, dict):
            raise ValueError(f"Invalid YAML structure in {cfg_path}")

        final_step = _patch_final_eval(final_eval_csv, cfg, force=bool(args.force))
        _patch_eval_like_csv(csv_dir / "evals.csv", force=bool(args.force), final_step=final_step)
        _patch_eval_like_csv(csv_dir / "eval_episodes.csv", force=bool(args.force), final_step=final_step)
        _patch_eval_like_csv(csv_dir / "rule_metrics.csv", force=bool(args.force), final_step=final_step)
        patched_runs += 1

    print(f"\nDone. Patched {patched_runs} runs.")


if __name__ == "__main__":
    main()
