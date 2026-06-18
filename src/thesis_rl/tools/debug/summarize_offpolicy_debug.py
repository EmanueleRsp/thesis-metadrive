from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _fmt_list(value: Any) -> str:
    if not isinstance(value, list):
        return str(value)
    return "[" + ", ".join(f"{float(item):.3f}" for item in value) + "]"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize off-policy debug JSONL logs.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--tail", type=int, default=3)
    args = parser.parse_args()

    rows = _read_jsonl(args.path)
    if not rows:
        raise SystemExit(f"No rows found in {args.path}")

    collect_rows = [row for row in rows if row.get("event") == "collect"]
    update_rows = [row for row in rows if row.get("event") == "update"]

    print(f"log={args.path}")
    print(f"collect_rows={len(collect_rows)} update_rows={len(update_rows)}")

    if collect_rows:
        print("\nLast collect rows:")
        for row in collect_rows[-int(args.tail) :]:
            print(
                "  "
                f"step={row.get('total_steps')} replay={row.get('replay_size')} "
                f"reward_mean={float(row.get('reward_mean', 0.0)):.4f} "
                f"route_completion_mean={float(row.get('route_completion_mean', 0.0)):.4f} "
                f"action_abs_mean={_fmt_list(row.get('action_abs_mean'))} "
                f"near_zero_dim_frac={float(row.get('action_frac_near_zero_dimwise', 0.0)):.3f} "
                f"timeout_rate={float(row.get('timeout_rate', 0.0)):.3f}"
            )

    if update_rows:
        print("\nLast update rows:")
        for row in update_rows[-int(args.tail) :]:
            print(
                "  "
                f"step={row.get('total_steps')} replay={row.get('replay_size')} update_call={row.get('update_call')} "
                f"batch_reward_mean={float(row.get('batch_reward_mean', 0.0)):.4f} "
                f"batch_done_rate={float(row.get('batch_done_rate', 0.0)):.4f} "
                f"policy_action_abs_mean={float(row.get('policy_action_abs_mean', 0.0)):.4f} "
                f"q_target_mean={float(row.get('q_target_mean', 0.0)):.4f} "
                f"q1_mean={float(row.get('q1_mean', 0.0)):.4f} "
                f"q2_mean={float(row.get('q2_mean', 0.0)):.4f} "
                f"actor_loss={float(row.get('actor_loss', 0.0)):.4f} "
                f"critic_loss={float(row.get('critic_loss', 0.0)):.4f}"
                + (
                    f" alpha={float(row.get('alpha', 0.0)):.4f}"
                    if "alpha" in row
                    else ""
                )
                + (
                    f" logp={float(row.get('policy_log_prob_mean', 0.0)):.4f}"
                    if "policy_log_prob_mean" in row
                    else ""
                )
            )


if __name__ == "__main__":
    main()
