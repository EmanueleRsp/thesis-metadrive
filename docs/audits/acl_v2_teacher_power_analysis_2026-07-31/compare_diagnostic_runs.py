"""Compare the training curves of the 120k-step diagnostic runs.

Answers, from `csv/train_chunks.csv` only (so it works for ACL and non-ACL runs
alike): does the policy's *driving competence* improve, and does it depend on the
reward construction rather than on the curriculum?

The relevant contrast is
    monitor_only   -> native MetaDrive reward, rulebook observed but not used
    scalar_reward  -> rulebook scalarisation drives the policy
    r1road variant -> offroad promoted into R1 alongside collision

Usage:
    python compare_diagnostic_runs.py <outputs_root>
"""

from __future__ import annotations

import csv
import glob
import os
import sys

COLS = [
    ("steps", "steps_end", "{:>8.0f}"),
    ("reward", "ep_rew_mean", "{:>9.1f}"),
    ("success", "ep_success_rate", "{:>8.3f}"),
    ("route", "ep_route_completion_mean", "{:>7.3f}"),
    ("collis", "ep_collision_rate", "{:>7.3f}"),
    ("offroad", "ep_out_of_road_rate", "{:>8.3f}"),
    ("ep_len", "ep_len_mean", "{:>7.1f}"),
]


def load(path):
    with open(path, newline="") as fh:
        return [r for r in csv.DictReader(fh)]


def fnum(row, key):
    v = row.get(key)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def show(label, path):
    rows = load(path)
    if not rows:
        return None
    print(f"\n### {label}")
    print(f"    {path.split('outputs/')[-1]}")
    header = "    " + "".join(f"{name:>9}" for name, _, _ in COLS)
    print(header)
    for r in rows:
        line = "    "
        for _name, key, fmt in COLS:
            line += fmt.format(fnum(r, key)).rjust(9)
        print(line)
    first, last = rows[0], rows[-1]
    return {
        "label": label,
        "steps": fnum(last, "steps_end"),
        "d_reward": fnum(last, "ep_rew_mean") - fnum(first, "ep_rew_mean"),
        "d_success": fnum(last, "ep_success_rate") - fnum(first, "ep_success_rate"),
        "d_route": fnum(last, "ep_route_completion_mean") - fnum(first, "ep_route_completion_mean"),
        "d_collision": fnum(last, "ep_collision_rate") - fnum(first, "ep_collision_rate"),
        "d_offroad": fnum(last, "ep_out_of_road_rate") - fnum(first, "ep_out_of_road_rate"),
        "final_success": fnum(last, "ep_success_rate"),
        "final_route": fnum(last, "ep_route_completion_mean"),
        "final_offroad": fnum(last, "ep_out_of_road_rate"),
        "final_collision": fnum(last, "ep_collision_rate"),
    }


LABELS = {
    "EXP_diag-sac-lite-native-noacl-fast": "native reward, ACL off",
    "EXP_diag-sac-lite-monitor-noacl-fast": "monitor_only, ACL off",
    "EXP_diag-sac-lite-monitor-acl-fast": "monitor_only, ACL on",
    "EXP_diag-sac-lite-scalar-noacl-fast": "scalar rulebook, ACL off",
    "EXP_diag-sac-lite-scalar-acl-fast": "scalar rulebook, ACL on",
    "EXP_diag-sac-lite-scalar-noacl-fast-r1road": "scalar rulebook + offroad in R1, ACL off",
}


def main(root):
    summaries = []
    for pattern, label in sorted(LABELS.items(), key=lambda kv: kv[1]):
        hits = sorted(glob.glob(os.path.join(root, pattern + "_*", "*", "*", "*",
                                             "csv", "train_chunks.csv")))
        hits = [h for h in hits if len(load(h)) > 3]
        if not hits:
            print(f"\n### {label}: no run instance with more than 3 chunks")
            continue
        best = max(hits, key=lambda h: len(load(h)))
        s = show(label, best)
        if s:
            summaries.append(s)

    print("\n" + "=" * 104)
    print("SUMMARY  (delta = last chunk - first chunk)")
    print("=" * 104)
    print(f"{'configuration':<44}{'steps':>8}{'dRew':>8}{'dSucc':>8}{'dRoute':>8}"
          f"{'dColl':>8}{'dOffrd':>8}{'succ_f':>8}{'offr_f':>8}")
    for s in summaries:
        print(f"{s['label']:<44}{s['steps']:>8.0f}{s['d_reward']:>8.1f}"
              f"{s['d_success']:>8.3f}{s['d_route']:>8.3f}{s['d_collision']:>8.3f}"
              f"{s['d_offroad']:>8.3f}{s['final_success']:>8.3f}{s['final_offroad']:>8.3f}")

    pooled(root)
    level_or_trend(root)


def _paths(root):
    out = {}
    for pattern, label in LABELS.items():
        hits = sorted(glob.glob(os.path.join(root, pattern + "_*", "*", "*", "*",
                                             "csv", "train_chunks.csv")))
        hits = [h for h in hits if len(load(h)) > 3]
        if hits:
            out[label] = max(hits, key=lambda h: len(load(h)))
    return out


def _wmean(rows, key):
    n = sum(int(float(r["episodes"])) for r in rows)
    if n == 0:
        return float("nan"), 0
    return sum(fnum(r, key) * int(float(r["episodes"])) for r in rows) / n, n


def pooled(root):
    """Episode-weighted means over chunks 2..end, with binomial 95% CIs.

    Chunk 1 is dropped: it contains the initial exploration transient and is not
    comparable across reward constructions.
    """
    import math
    print("\n" + "=" * 104)
    print("POOLED over chunks 2..end, episode-weighted, +-95% binomial CI on the pooled count")
    print("=" * 104)
    print(f"{'configuration':<34}{'N_ep':>6}{'offroad':>18}{'collision':>18}"
          f"{'success':>17}{'route':>8}{'ep_len':>8}")
    for label, path in sorted(_paths(root).items()):
        rows = load(path)[1:]
        n = sum(int(float(r["episodes"])) for r in rows)
        if n == 0:
            continue

        def ci(p):
            return 1.96 * math.sqrt(max(p * (1 - p), 1e-9) / n)

        o, _ = _wmean(rows, "ep_out_of_road_rate")
        c, _ = _wmean(rows, "ep_collision_rate")
        s, _ = _wmean(rows, "ep_success_rate")
        r, _ = _wmean(rows, "ep_route_completion_mean")
        el, _ = _wmean(rows, "ep_len_mean")
        print(f"{label:<34}{n:>6}{o:>11.3f} +-{ci(o):<5.3f}{c:>11.3f} +-{ci(c):<5.3f}"
              f"{s:>10.3f} +-{ci(s):<5.3f}{r:>8.3f}{el:>8.1f}")


def level_or_trend(root):
    """Separate a level shift (present from the start) from a learning trend."""
    print("\n" + "=" * 104)
    print("LEVEL SHIFT vs LEARNING TREND: early (chunks 2-5) against late (chunks 9-12)")
    print("=" * 104)
    print(f"{'configuration':<34}{'metric':<9}{'early':>10}{'late':>10}{'trend':>9}")
    for label, path in sorted(_paths(root).items()):
        rows = load(path)
        if len(rows) < 12:
            continue
        early, late = rows[1:5], rows[8:12]
        for i, (name, key) in enumerate((("offroad", "ep_out_of_road_rate"),
                                         ("route", "ep_route_completion_mean"),
                                         ("success", "ep_success_rate"))):
            e, _ = _wmean(early, key)
            l, _ = _wmean(late, key)
            print(f"{label if i == 0 else '':<34}{name:<9}{e:>10.3f}{l:>10.3f}{l-e:>+9.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "/scratch/e.respino/thesis-metadrive/outputs")
