"""Did the policy improve at all, per arm, in a completed ACL run?

The teacher-replay analysis found SNR ~ 0.01 on every arm and every dimension.
That has two very different explanations:

  (a) the ACL v2.0 windows are too short to see a real but slow improvement, or
  (b) the policy did not measurably improve on these outcomes at all.

They are distinguished by looking at the level, not the local slope: bin each
arm's committed episodes by training progress and print the mean outcome per bin.
A linear fit cannot tell them apart if learning is front-loaded or absent.

Both Generate and Replay episodes are used here, because the question is about
the policy, not about the teacher's sampling.

Usage:
    python learning_curve_check.py <events.jsonl> [<events.jsonl> ...]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np

N_BINS = 8


def load(path):
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if '"scenario_acl_episode_ended"' not in line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            m = ev.get("metrics") or {}
            ctx = m.get("live_event_context") or {}
            if ctx.get("arm") is None or m.get("route_completion") is None:
                continue
            rows.append({
                "episode_id": int(ev.get("episode_id", 0)),
                "arm": ctx["arm"],
                "origin": ctx.get("origin"),
                "success": 1.0 if m.get("success") else 0.0,
                "rc": float(m["route_completion"]),
                "collision": 1.0 if m.get("collision") else 0.0,
                "oor": 1.0 if m.get("out_of_road") else 0.0,
                "reward": float(m.get("reward", 0.0)),
            })
    rows.sort(key=lambda r: r["episode_id"])
    return rows


def table(label, rows, key):
    if not rows:
        return
    per_arm = defaultdict(list)
    for r in rows:
        per_arm[r["arm"]].append(r)
    print(f"\n  === {label} : mean over {N_BINS} equal bins of the arm's episode sequence ===")
    header = "  " + f"{'arm':<5}{'n':>7}" + "".join(f"{f'b{i+1}':>9}" for i in range(N_BINS))
    header += f"{'last-first':>12}"
    print(header)
    for arm, recs in sorted(per_arm.items()):
        vals = np.asarray([r[key] for r in recs], dtype=float)
        if len(vals) < N_BINS * 5:
            continue
        chunks = np.array_split(vals, N_BINS)
        means = [float(c.mean()) for c in chunks]
        delta = means[-1] - means[0]
        print("  " + f"{arm:<5}{len(vals):>7}" + "".join(f"{m:>9.3f}" for m in means)
              + f"{delta:>12.3f}")
    allv = np.asarray([r[key] for r in rows], dtype=float)
    chunks = np.array_split(allv, N_BINS)
    means = [float(c.mean()) for c in chunks]
    print("  " + f"{'ALL':<5}{len(allv):>7}" + "".join(f"{m:>9.3f}" for m in means)
          + f"{means[-1]-means[0]:>12.3f}")


def analyse(path):
    print("=" * 110)
    print(f"RUN: {path}")
    rows = load(path)
    if not rows:
        print("  no usable episodes")
        return
    gen = [r for r in rows if r["origin"] == "new"]
    print(f"  episodes {len(rows)} (generate {len(gen)}, replay {len(rows)-len(gen)}), "
          f"episode_id {rows[0]['episode_id']}..{rows[-1]['episode_id']}")
    for label, key in (("route_completion", "rc"), ("success", "success"),
                       ("collision", "collision"), ("out_of_road", "oor"),
                       ("episode reward", "reward")):
        table(label, rows, key)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for p in sys.argv[1:]:
        analyse(p)
        print()
