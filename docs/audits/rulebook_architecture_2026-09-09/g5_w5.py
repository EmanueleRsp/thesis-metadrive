"""G5 - derive w5, the one new parameter A7 introduces, from a physical criterion.

The criterion, stated before any value is picked:

  The relaxable-compliance charge must sit between two manoeuvres the ego can
  actually perform.

  LOWER BOUND - the shortcut must not pay.  Riding a marking at severity 1/3 for
  3 s to arrive 4 s sooner must cost more than the 4 s are worth.  (This is O3
  against the section 4.6 reference shortcut, which is the cheapest illegal
  shortcut saving the most time, so a more aggressive one is easier.)

  UPPER BOUND - passing must stay cheaper than stopping.  An obstruction that
  forces 1 s of marking contact must cost less than stopping and restarting.
  Stopping from urban speed v and returning to it at a comfortable a_c costs
  v/a_c seconds of delay: at v = 10 m/s and a_c = 2 m/s that is 5.0 s.  If the
  charge exceeded that, the ego would rather wait for ever than pass, which is
  the v5.0 pathology this whole line of work exists to remove.

Both bounds are expressed in the same currency - seconds of arrival time - and
the exchange rate between reward units and seconds is read off the reward itself.
"""
from __future__ import annotations
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import g2_bench as g2
from g2_bench import *   # noqa

A, SIG, LAM4 = SHIPPED["a"], SHIPPED["sig"], SHIPPED["lam4"]
DREF = 2.2222


def time_value_per_second(gamma, lam4=LAM4):
    """Reward units gained by completing the reference mission one second sooner.

    Measured as the discounted L4 difference between two completing runs whose
    durations differ by ten steps, divided by one second.
    """
    hi = lam4 * g2._ch(completing_run(REF_LEGAL - 10), gamma, 3)
    lo = lam4 * g2._ch(completing_run(REF_LEGAL), gamma, 3)
    return (hi - lo) / 1.0        # 10 steps = 1 s


def charge(w5, severity, steps):
    """Indicator charge for `steps` steps at c_L5 = severity."""
    return steps * w5 * (1.0 + SIG * severity)


for gamma in (0.996, 0.9982):
    tv = time_value_per_second(gamma)
    print("=" * 78)
    print(f"gamma = {gamma}")
    print("=" * 78)
    print(f"  time value of arriving 1 s sooner        = {tv:.4f} reward units")
    # lower bound: 3 s at 1/3 must beat 4 s of arrival time
    lo = 4.0 * tv / charge(1.0, 1.0 / 3.0, 30)
    # upper bound: 1 s at 1/3 must cost less than a 5 s stop-and-restart
    hi = 5.0 * tv / charge(1.0, 1.0 / 3.0, 10)
    # section 5.4 admissibility, with lam6 = 0 and no eta term
    cap = (A - LAM4 * DQMAX) / (1.0 + SIG)
    print(f"  LOWER bound  (O3, shortcut must not pay)  w5 > {lo:.4f}")
    print(f"  UPPER bound  (passing beats waiting)      w5 < {hi:.4f}")
    print(f"  section 5.4 admissibility cap             w5 < {cap:.4f}")
    top = min(hi, cap)
    print(f"  admissible window                         [{lo:.4f}, {top:.4f})"
          f"   width {100*(top-lo)/top:.1f}% of the cap")
    print(f"  geometric centre of the window            w5 = {(lo*top)**0.5:.4f}")

print()
print("=" * 78)
print("Consequences at w5 = 0.25 (the value F3 priced) and at the centre")
print("=" * 78)
for w5 in (0.15, 0.20, 0.25, 0.30, 0.3846):
    W = dict(SHIPPED); W["w5"] = w5
    a7 = [c for c in CANDIDATES if c.key == "A7"][0]
    legal = completing_run(REF_LEGAL)
    short = completing_run(REF_SHORT, l5_steps=REF_L5_STEPS, severity=REF_L5_SEV)
    o3 = a7.scalar(legal, 0.996, **W) - a7.scalar(short, 0.996, **W)
    o3b = a7.scalar(legal, 0.9982, **W) - a7.scalar(short, 0.9982, **W)
    stop = standing_still()
    detour = with_relaxation(legal_drive(), first=10, count=20, severity=0.5)
    o2 = a7.scalar(detour, 0.996, **W) - a7.scalar(stop, 0.996, **W)
    creep = [(0.0, 0.0, 0.0, 0.01, REF_L5_SEV)] * REF_LEGAL
    h6 = a7.scalar(creep, 0.996, **W) - a7.scalar(standing_still(REF_LEGAL), 0.996, **W)
    tail = w5 * (1.0 + SIG) + LAM4 * DQMAX
    # expert cost: 1897 violated L5 steps over 1100 episodes, mean severity 0.0876
    expert = (1897 / 1100) * w5 * (1.0 + SIG * 0.0876)
    # distance a marking crossing must buy, per second, for the scalar arm to take it
    metres = w5 * (1.0 + SIG / 3.0) * DREF / LAM4 * 10.0
    print(f"  w5={w5:<7} O3(0.996)={o3:+7.3f} O3(0.9982)={o3b:+7.3f} O2={o2:+7.2f} "
          f"creep={h6:+8.2f}  a/tail={A/tail:.3f}  expert cost={-expert:+.3f}  "
          f"buys>={metres:.2f} m/s")
