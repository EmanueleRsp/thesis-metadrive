"""G4 - reward-hacking probes and the summary table, at both discounts."""
from __future__ import annotations
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from g2_bench import *   # noqa

W = dict(SHIPPED); W["w5"] = 0.25
T_EP = 199          # Waymo median episode


def probes():
    """Each probe: (label, exploit trajectory, honest trajectory, 'honest must win')."""
    P = {}
    honest = completing_run(REF_LEGAL)                    # 200 steps, legal, completes
    # H1 sprint then stop: cover 40 % of the mission at the clip, then idle out the episode
    sprint_steps = int(round(0.40 * Q))                   # dq = 1 per step
    P["H1 sprint 40% of the mission then idle to the horizon"] = (
        [(0.0, 0.0, 0.0, 1.0, 0.0)] * sprint_steps
        + [(0.0, 0.0, 0.0, 0.0, 0.0)] * (REF_LEGAL - sprint_steps), honest)
    # H2 overshoot: complete, then keep advancing along the polyline past the goal
    P["H2 complete then overshoot 16% of the polyline past the goal"] = (
        completing_run(REF_SHORT) + [(0.0, 0.0, 0.0, Q * 0.16 / 40, 0.0)] * 40,
        completing_run(REF_SHORT))
    # H3 spend the whole relaxation budget for nothing (same route, same duration)
    P["H3 relax for 30 steps with no progress benefit"] = (
        completing_run(REF_LEGAL, l5_steps=30, severity=REF_L5_SEV), honest)
    # H4 escape: collide at fault on step 20 (episode terminates there)
    P["H4 collide at fault on step 20 to end the episode"] = (
        completing_run(REF_LEGAL)[:20] + [(0.6, 0.0, 0.0, Q / REF_LEGAL, 0.0)], honest)
    # H4b escape from a losing episode: same, but the honest run is a bad one
    bad = [(0.0, 0.9, 0.0, Q / REF_LEGAL, 0.0)] * REF_LEGAL
    P["H4b collide at fault on step 20 instead of 200 steps of L2 violation"] = (
        bad[:20] + [(0.6, 0.9, 0.0, Q / REF_LEGAL, 0.0)], bad)
    # H5 projection jump: one clipped +1 advance with no real motion
    P["H5 one clipped route-projection jump (+1) on an otherwise idle run"] = (
        [(0.0, 0.0, 0.0, 0.0, 0.0)] * 99 + [(0.0, 0.0, 0.0, 1.0, 0.0)]
        + [(0.0, 0.0, 0.0, 0.0, 0.0)] * 100, standing_still(REF_LEGAL))
    # H6 creep forever along a marking
    P["H6 creep along a marking for the whole episode vs standing still"] = (
        [(0.0, 0.0, 0.0, 0.01, REF_L5_SEV)] * REF_LEGAL, standing_still(REF_LEGAL))
    return P


def run(gamma):
    print("#" * 92)
    print(f"#  REWARD-HACKING PROBES at gamma = {gamma}   "
          f"(exploit must LOSE except where marked)")
    print("#" * 92)
    P = probes()
    for label, (bad_traj, good_traj) in P.items():
        print(f"\n  {label}")
        for cand in CANDIDATES:
            s_bad = cand.scalar(bad_traj, gamma, **W)
            s_good = cand.scalar(good_traj, gamma, **W)
            gap = s_bad - s_good
            verdict = "EXPLOIT WINS" if gap > 0 else "ok"
            print(f"      {cand.key:<4} scalar exploit {s_bad:9.3f} vs honest {s_good:9.3f}"
                  f"  gap {gap:+9.3f}  {verdict}")


def summary(gamma):
    from g3_battery import battery
    bat = battery()
    print()
    print("#" * 92)
    print(f"#  SUMMARY at gamma = {gamma}")
    print("#" * 92)
    names = list(bat)
    print(f"{'candidate':<6}{'ch':>3}{'w':>3}{'tau':>4}  " + "".join(f"{n.split()[0]:>6}" for n in names))
    for cand in CANDIDATES:
        sc, lx = [], []
        for n in names:
            A, B = bat[n]
            sc.append("P" if cand.scalar(A, gamma, **W) > cand.scalar(B, gamma, **W) else "F")
            v = lex_prefers(cand, A, B, gamma)
            lx.append("P" if v else ("F" if v is False else "-"))
        print(f"{cand.key:<6}{cand.n_channels:>3}{len(cand.params):>3}{len(cand.thresholds):>4}  "
              + "".join(f"{s:>6}" for s in sc) + "   scalar")
        print(f"{'':<6}{'':>3}{'':>3}{'':>4}  " + "".join(f"{s:>6}" for s in lx) + "   strict-lex")


if __name__ == "__main__":
    for g in (GAMMA_NOW, GAMMA_NEW):
        summary(g)
    run(GAMMA_NOW)
