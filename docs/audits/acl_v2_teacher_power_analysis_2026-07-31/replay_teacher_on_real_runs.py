"""Replay the proposed ACL v2.0 teacher over per-episode records of completed runs.

Reads `scenario_acl_episode_ended` events from an existing run's `logs/events.jsonl`,
keeps only Generate episodes, groups them per arm in commit order, and then runs the
*actual* proposed statistics over those real sequences:

  * the 2H sliding windows,
  * the Vargha-Delaney G,
  * the exact conditional permutation deadband,
  * the Goldilocks gate and the signed feedback A,
  * the EMA score.

It reports, per arm and per dimension, the empirical SNR and how often the teacher
would have fired. This replaces the assumed SNR of teacher_dynamics_simulation.py
with a measured one.

Dimensions available from these logs:
  task            (-success, -route_completion)      lexicographic, always observed
  route           -route_completion                  continuous component alone
  R1_proxy        collision indicator                stands in for v_R1; the max
                                                     impact cost was never logged
  R3_proxy        out_of_road indicator              R3's offroad subcomponent only

R2 cannot be reconstructed: per-macro-rule episodic costs are not in these logs.

Usage:
    python replay_teacher_on_real_runs.py <events.jsonl> [<events.jsonl> ...]
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict

import numpy as np

H = 10
E2 = H * (2 * H + 1)
TOTAL_SPLITS = math.comb(2 * H, H)
ALPHA = 0.05


# --------------------------------------------------------- teacher internals --

def doubled_midranks(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    out = [0] * len(vals)
    pos = 0
    while pos < len(order):
        end = pos
        while end + 1 < len(order) and vals[order[end + 1]] == vals[order[pos]]:
            end += 1
        size = end - pos + 1
        for k in range(pos, end + 1):
            out[order[k]] = 2 * pos + size + 1
        pos = end + 1
    return out


_CACHE: dict = {}


def exact_two_sided_p(recent, older):
    ranks = doubled_midranks(list(recent) + list(older))
    s_obs = sum(ranks[:H])
    uniq = sorted(set(ranks))
    counts = [ranks.count(r) for r in uniq]
    key = (tuple(uniq), tuple(counts), s_obs)
    if key in _CACHE:
        return _CACHE[key]

    max_s = sum(sorted(ranks, reverse=True)[:H])
    ways = np.zeros((H + 1, max_s + 1), dtype=np.int64)
    ways[0, 0] = 1
    for r, m in zip(uniq, counts):
        new = np.zeros_like(ways)
        for k in range(0, min(m, H) + 1):
            shift = k * r
            if shift > max_s:
                break
            new[k:, shift:] += ways[: H + 1 - k, : max_s + 1 - shift] * math.comb(m, k)
        ways = new
    idx = np.arange(max_s + 1)
    p = float(ways[H][np.abs(idx - E2) >= abs(s_obs - E2)].sum()) / TOTAL_SPLITS
    _CACHE[key] = p
    return p


def vargha_delaney(recent, older):
    n = 0.0
    for x in recent:
        for y in older:
            if x < y:
                n += 1.0
            elif x == y:
                n += 0.5
    return n / (H * H)


# ------------------------------------------------------------------- loading --

def load_generate_episodes(path):
    """Return {arm: [record, ...]} in commit order, Generate episodes only."""
    per_arm = defaultdict(list)
    total = replay = aborted = 0
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
            arm = ctx.get("arm")
            if arm is None:
                continue
            total += 1
            if ctx.get("origin") != "new":
                replay += 1
                continue
            rc = m.get("route_completion")
            if rc is None:
                aborted += 1
                continue
            per_arm[arm].append({
                "episode_id": int(ev.get("episode_id", 0)),
                "success": 1.0 if m.get("success") else 0.0,
                "route_completion": float(rc),
                "collision": 1.0 if m.get("collision") else 0.0,
                "out_of_road": 1.0 if m.get("out_of_road") else 0.0,
            })
    for arm in per_arm:
        per_arm[arm].sort(key=lambda r: r["episode_id"])
    return per_arm, total, replay, aborted


DIMENSIONS = {
    # name -> (key function, lower-is-better already, gate function over recent window)
    "task":     lambda r: (-r["success"], -r["route_completion"]),
    "route":    lambda r: -r["route_completion"],
    "R1_proxy": lambda r: r["collision"],
    "R3_proxy": lambda r: r["out_of_road"],
}

GATES = {
    "task":     lambda win: float(np.mean([r["route_completion"] for r in win])),
    "route":    lambda win: float(np.mean([r["route_completion"] for r in win])),
    "R1_proxy": lambda win: float(np.mean([r["collision"] for r in win])),
    "R3_proxy": lambda win: float(np.mean([r["out_of_road"] for r in win])),
}


def empirical_snr(records, keyfn):
    """delta over H arm-local episodes from a linear trend, over residual sd."""
    vals = [keyfn(r) for r in records]
    if isinstance(vals[0], tuple):  # lexicographic: use the scalarised surrogate
        vals = [v[0] * 2.0 + v[1] for v in vals]
    y = np.asarray(vals, dtype=float)
    if len(y) < 40 or np.allclose(y, y[0]):
        return float("nan"), float("nan"), float("nan")
    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    sd = float(resid.std(ddof=2))
    delta = abs(float(slope)) * H
    return delta, sd, (delta / sd if sd > 0 else float("nan"))


def replay_teacher(records, dim):
    """Run the proposed deadband over one arm's real Generate sequence."""
    keyfn, gatefn = DIMENSIONS[dim], GATES[dim]
    fired = updates = 0
    gs, a_vals = [], []
    for end in range(2 * H, len(records) + 1):
        win = records[end - 2 * H:end]
        keys = [keyfn(r) for r in win]
        older, recent = keys[:H], keys[H:]
        g = vargha_delaney(recent, older)
        p = exact_two_sided_p(recent, older)
        updates += 1
        gs.append(g)
        if p <= ALPHA:
            fired += 1
            x = gatefn(win[H:])
            a_vals.append(0.5 + 4 * x * (1 - x) * (g - 0.5))
        else:
            a_vals.append(0.5)
    if updates == 0:
        return None
    return {
        "updates": updates,
        "fire_rate": fired / updates,
        "mean_G": float(np.mean(gs)),
        "sd_G": float(np.std(gs)),
        "mean_A": float(np.mean(a_vals)),
        "q_inf": float(np.mean(a_vals)),
    }


def analyse(path):
    print("=" * 100)
    print(f"RUN: {path}")
    per_arm, total, replay, aborted = load_generate_episodes(path)
    n_gen = sum(len(v) for v in per_arm.values())
    print(f"  committed ACL episodes {total} | replay {replay} | "
          f"generate {n_gen} | dropped {aborted}")
    if not per_arm:
        print("  no Generate episodes with arm provenance; skipping")
        return
    print(f"  generate per arm: " +
          ", ".join(f"{a}={len(v)}" for a, v in sorted(per_arm.items())))

    for dim in ("route", "task", "R1_proxy", "R3_proxy"):
        print(f"\n  --- dimension: {dim} ---")
        print(f"  {'arm':<5} {'n':>6} {'delta/H':>9} {'sd':>9} {'SNR':>7} "
              f"{'mean G':>8} {'sd G':>7} {'fire@.05':>9} {'mean A':>8}")
        for arm, recs in sorted(per_arm.items()):
            if len(recs) < 2 * H + 5:
                print(f"  {arm:<5} {len(recs):>6}   (too few Generate episodes)")
                continue
            d, sd, snr = empirical_snr(recs, DIMENSIONS[dim])
            res = replay_teacher(recs, dim)
            if res is None:
                continue
            print(f"  {arm:<5} {len(recs):>6} {d:>9.4f} {sd:>9.4f} {snr:>7.2f} "
                  f"{res['mean_G']:>8.3f} {res['sd_G']:>7.3f} "
                  f"{res['fire_rate']:>9.3f} {res['mean_A']:>8.3f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for p in sys.argv[1:]:
        analyse(p)
        print()
