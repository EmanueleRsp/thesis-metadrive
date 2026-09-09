"""G3 - run the behavioural battery over every candidate, three comparison rules.

For the thresholded rule the useful output is not a verdict at one arbitrary tau
but the *critical threshold* per channel: a constrained channel i ties (and the
comparison descends) exactly when tau_i <= min(ca_i, cb_i), and decides otherwise.
So each pair yields, per channel, one number and the verdict if that channel
decides. That is the whole answer, with no tau invented.
"""
from __future__ import annotations
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from g2_bench import *   # noqa

W = dict(SHIPPED)
W["w5"] = 0.25

# ------------------------------------------------------------------ battery
def battery():
    stop = standing_still()
    B = {}
    B["B1 liveness: legal drive in traffic > standing still"] = (
        legal_drive(l2=TRAFFIC_L2), stop)
    B["B2 necessary relaxation to complete > standing still"] = (
        with_relaxation(legal_drive(), first=10, count=20, severity=0.5), stop)
    B["B2' same, with the traffic residual present"] = (
        with_relaxation(legal_drive(l2=TRAFFIC_L2), first=10, count=20, severity=0.5), stop)
    B["B3 legal completion > faster completion riding a marking"] = (
        completing_run(REF_LEGAL),
        completing_run(REF_SHORT, l5_steps=REF_L5_STEPS, severity=REF_L5_SEV))
    B["B4 heavy relaxation > at-fault collision"] = (
        with_relaxation(legal_drive(), first=0, count=20, severity=1.0),
        [(0.4, 0.0, 0.0, CRUISE_DQ, 0.0) if i == 12 else (0.0, 0.0, 0.0, CRUISE_DQ, 0.0)
         for i in range(STEPS)])
    B["B5 waiting at red > running it, with less progress"] = (
        [(0.0, 0.0, 0.0, 0.0, 0.0)] * 15 + [(0.0, 0.0, 0.0, CRUISE_DQ, 0.0)] * 25,
        [(0.0, 0.0, 0.8, CRUISE_DQ, 0.0)] * 5 + [(0.0, 0.0, 0.0, CRUISE_DQ, 0.0)] * 35)
    B["B6 minimum violation: 5 relaxed steps > 25"] = (
        with_relaxation(legal_drive(), first=10, count=5, severity=0.5),
        with_relaxation(legal_drive(), first=10, count=25, severity=0.5))
    B["B9 mitigated impact > full impact (same everything else)"] = (
        [(0.3, 0.0, 0.0, CRUISE_DQ, 0.0) if i == 12 else (0.0, 0.0, 0.0, CRUISE_DQ, 0.0)
         for i in range(STEPS)],
        [(0.9, 0.0, 0.0, CRUISE_DQ, 0.0) if i == 12 else (0.0, 0.0, 0.0, CRUISE_DQ, 0.0)
         for i in range(STEPS)])
    B["B10 promptness: same mission, 160 steps > 200 steps, both legal"] = (
        completing_run(REF_SHORT), completing_run(REF_LEGAL))
    B["B11 no reward for oscillation"] = (
        [(0.0, 0.0, 0.0, 0.4, 0.0)] * 10,
        ([(0.0, 0.0, 0.0, 0.4, 0.0)] * 5 + [(0.0, 0.0, 0.0, -0.4, 0.0)] * 5) * 3
        + [(0.0, 0.0, 0.0, 0.4, 0.0)] * 10)
    return B


def report(gamma: float) -> None:
    print("#" * 90)
    print(f"#  BATTERY at gamma = {gamma}")
    print("#" * 90)
    bat = battery()
    for cand in CANDIDATES:
        print()
        print("=" * 90)
        print(f"{cand.key}  {cand.name}   ({cand.n_channels} channels, "
              f"{len(cand.params)} weights, {len(cand.thresholds)} thresholds)")
        print("=" * 90)
        for label, (A, Bt) in bat.items():
            s_a = cand.scalar(A, gamma, **W)
            s_b = cand.scalar(Bt, gamma, **W)
            scalar_ok = s_a > s_b
            lex = lex_prefers(cand, A, Bt, gamma)
            dec = deciding(cand, A, Bt, gamma)
            ca, cb = cand.channels(A, gamma), cand.channels(Bt, gamma)
            # thresholded: critical tau per constrained channel
            crit = []
            for nm, kind, thr, who in critical_taus(cand, A, Bt, gamma):
                if kind == "cost":
                    crit.append(f"{nm}(cost) gives {who} iff tau<{thr:.4g}")
                else:
                    crit.append(f"{nm}(value) gives {who} iff tau>{thr:.4g}")
            hi = cand.sense[-1]
            last = "A" if ((ca[-1] > cb[-1]) if hi else (ca[-1] < cb[-1])) else "B"
            if abs(ca[-1] - cb[-1]) < 1e-12:
                last = "tie"
            print(f"  {label}")
            print(f"      scalar {'PASS' if scalar_ok else 'FAIL'} (margin {s_a - s_b:+.4f})"
                  f"   strict-lex {'PASS' if lex else ('FAIL' if lex is False else 'TIE')}"
                  f" at {dec}")
            print(f"      thresholded: {' | '.join(crit) if crit else 'all constrained channels tie'}"
                  f"  -> if all pass through, last channel gives {last}")


if __name__ == "__main__":
    report(GAMMA_NOW)
