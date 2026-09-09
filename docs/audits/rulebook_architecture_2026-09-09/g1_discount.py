"""G1 - the discount decision: verify F9, price every exit.

Standard library only. Reads the frozen selection index by relative path from the
repository root (pass the root as argv[1] if running elsewhere).
"""
import json, math, sys
from collections import defaultdict

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/home/e.respino/main/thesis/thesis-metadrive"
IDX = f"{ROOT}/data/scenarionet/frozen/scenario_selection_index.json"

A, SIG, PHI, LAM4, ETA, LAM6 = 2.5, 0.30, 0.25, 2.0, 1.0, 0.2
DT, DQMAX = 0.1, 1.0
Q = 90.17 / 2.2222                 # mean mission span, D_REF units
T_LEGAL, T_SHORT = 200, 160        # section 4.6 reference shortcut
L5_STEPS, L5_SEV = 30, 1.0 / 3.0

def disc(n, g):
    return n if g == 1.0 else (1.0 - g ** n) / (1.0 - g)

def run(T, g, l5_steps=0, sev=0.0, lam6=LAM6, eta=ETA, lam4=LAM4):
    dq = Q / T
    c6 = 1.0 - max(0.0, min(dq, 1.0))
    return dict(
        r_l4=lam4 * dq * disc(T, g),
        r_l5=eta * sev * DT * disc(l5_steps, g),
        r_l6=lam6 * c6 * DT * disc(T, g),
        ch_l4=dq * disc(T, g),
        ch_l5=sev * disc(l5_steps, g),
    )

def o3_margin(g):
    L = run(T_LEGAL, g)
    S = run(T_SHORT, g, L5_STEPS, L5_SEV)
    ml = L["r_l4"] - L["r_l5"] - L["r_l6"]
    ms = S["r_l4"] - S["r_l5"] - S["r_l6"]
    dec = "L4" if abs(S["ch_l4"] - L["ch_l4"]) > 1e-12 else "L5"
    return ml - ms, dec

print("=" * 78)
print("G1.1  F9 re-verification: the true training horizon")
print("=" * 78)
d = json.load(open(IDX))
by = defaultdict(list)
for r in d["records"]:
    by[(r["split"], r["source"])].append(int(r["length"]))
D = math.log(A) / -math.log(0.996)
print(f"break-even Delta at a={A}, gamma=0.996 : {D:.4f} steps")
print(f"{'split/source':<18}{'n':>6}{'p50':>6}{'p95':>6}{'max':>6}{'>Delta':>9}")
for k in sorted(by):
    v = sorted(by[k]); n = len(v)
    over = sum(1 for x in v if x > D)
    print(f"{'/'.join(k):<18}{n:>6}{v[n//2]:>6}{v[int(.95*n)]:>6}{max(v):>6}{100*over/n:>8.1f}%")
tr = [x for k, v in by.items() if k[0] == "train" for x in v]
print(f"\nTRAIN: {sum(1 for x in tr if x > D)}/{len(tr)} = "
      f"{100*sum(1 for x in tr if x > D)/len(tr):.2f}% exceed the break-even step")
LMAX_SCEN = max(tr)
LMAX_EP = LMAX_SCEN - 1     # thesis_scenario_env.py:47  episode_steps >= scenario_length - 1
print(f"longest training scenario  = {LMAX_SCEN} logged steps")
print(f"longest training EPISODE   = {LMAX_EP} control steps (truncation at scenario_length-1)")

print()
print("=" * 78)
print("G1.2  The criterion, restated: Delta > L  <=>  gamma^L >= 1/a")
print("=" * 78)
for g in (0.99, 0.995, 0.996, 0.998, 0.9982, 0.999):
    print(f"  gamma={g:<8} gamma^500 = {g**500:.4f}   1/a = {1/A:.4f}   "
          f"{'OK ' if g**500 >= 1/A else 'FAIL'}  Delta = {math.log(A)/-math.log(g):7.1f}")
print("  (identity: requiring Delta>L is exactly requiring the whole-episode damping")
print("   gamma^L not to fall below the one-level priority ratio 1/a)")

print()
print("=" * 78)
print("G1.3  Exit A - raise gamma. One value, and its cost.")
print("=" * 78)
for L in (199, 241, 500, 501):
    g = math.exp(-math.log(A) / L)
    print(f"  L={L:<5} gamma_min = {g:.6f}   1/(1-g) = {1/(1-g):6.1f} steps = {0.1/(1-g):5.1f} s")
CAND = 0.9982
print(f"\n  proposal gamma = {CAND}")
print(f"    Delta            = {math.log(A)/-math.log(CAND):.1f} steps  vs L = {LMAX_EP}  "
      f"(margin {100*(math.log(A)/-math.log(CAND)/LMAX_EP - 1):.1f}%)")
print(f"    1/(1-gamma)      = {1/(1-CAND):.0f} steps = {0.1/(1-CAND):.1f} s   "
      f"(was {1/(1-0.996):.0f} steps = {0.1/(1-0.996):.1f} s)  -> x{(1-0.996)/(1-CAND):.2f}")
print(f"    gamma^500        = {CAND**500:.4f}   (was {0.996**500:.4f})")
print(f"    gamma^199        = {CAND**199:.4f}   (was {0.996**199:.4f})")
m0, d0 = o3_margin(0.996); m1, d1 = o3_margin(CAND); m2, d2 = o3_margin(1.0)
print(f"    O3 margin        = {m1:+.4f} decided at {d1}  (was {m0:+.4f} at {d0}; "
      f"gamma=1 gives {m2:+.4f} at {d2})")

print()
print("=" * 78)
print("G1.4  Exit B - raise a instead, at gamma = 0.996")
print("=" * 78)
for L in (199, 241, LMAX_EP):
    aa = math.exp(L * -math.log(0.996))
    lam4_max = aa - (ETA + LAM6) * DT
    print(f"  L={L:<5} a >= {aa:8.3f}  a^3 = {aa**3:10.1f} "
          f"(x{(aa/A)**3:6.1f} the shipped critic range)   lambda4_max = {lam4_max:.2f}")
print("  cost: a drags sigma (the section 5.4 window scales with a), lambda4, and the")
print("        217,189-transition calibration grid; ADR-081 already priced a=3.0 as")
print("        rejected in both directions.")

print()
print("=" * 78)
print("G1.5  Exit C - cap the horizon")
print("=" * 78)
pg = sorted(x for k, v in by.items() if k == ("train", "pg") for x in v)
for cap in (229, 300, 400):
    cut = sum(1 for x in pg if x > cap)
    print(f"  cap at {cap:>4}: truncates {cut}/{len(pg)} = {100*cut/len(pg):.1f}% of PG train records")
print("  PG mean route_length_m = 183.81 m; at the expert's 4.535 m/s that is 405 steps,")
print("  so a cap below ~410 makes the median PG mission structurally incompletable.")

print()
print("=" * 78)
print("G1.6  Does the 45-minute calibration have to be re-run?  NO.")
print("=" * 78)
print("  measure_expert_rulebook_transition.py:2038  episode_return += scalarized.reward")
print("  -> the audit's returns are UNDISCOUNTED sums; every calibration column")
print("     (mean, p1, p5, p50, fraction_below_standstill) is gamma-free.")
print("  section 5.4's rank-preservation inequality is per-step and gamma-free too.")
print("  So a, sigma, lambda4, eta, lambda6, phi are all untouched by this decision.")
