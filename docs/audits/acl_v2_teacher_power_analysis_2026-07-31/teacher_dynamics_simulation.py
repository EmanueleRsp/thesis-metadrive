"""ACL v2.0 teacher: closed-loop dynamics parameterised by signal-to-noise.

The decisive quantity is

    SNR = (mean episodic key of the older window - mean of the recent window) / sd

i.e. the policy improvement on that arm across the H-episode gap between the two
window centres, measured in units of the between-episode (between-scenario)
standard deviation inside the arm. True Vargha-Delaney A = Phi(SNR / sqrt(2)).

Nothing in the repository currently measures SNR, so it is swept explicitly
rather than assumed. Everything else about the teacher's responsiveness follows
from it.
"""

from __future__ import annotations

import math
import time

import numpy as np

H = 10
N_POOL = 2 * H
E2 = H * (2 * H + 1)
TOTAL_SPLITS = math.comb(N_POOL, H)


# ------------------------------------------------------- exact permutation ---

def doubled_midranks(vals: np.ndarray) -> np.ndarray:
    order = np.argsort(vals, kind="stable")
    s = vals[order]
    out = np.empty(len(vals), dtype=np.int64)
    pos = 0
    while pos < len(s):
        end = pos
        while end + 1 < len(s) and s[end + 1] == s[pos]:
            end += 1
        size = end - pos + 1
        out[order[pos:end + 1]] = 2 * pos + size + 1
        pos = end + 1
    return out


_CACHE: dict[tuple, float] = {}


def exact_two_sided_p(recent: np.ndarray, older: np.ndarray) -> float:
    ranks = doubled_midranks(np.concatenate([recent, older]))
    s_obs = int(ranks[:H].sum())
    uniq, counts = np.unique(ranks, return_counts=True)
    key = (tuple(uniq.tolist()), tuple(counts.tolist()), s_obs)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit

    max_s = int(np.sort(ranks)[-H:].sum())
    ways = np.zeros((H + 1, max_s + 1), dtype=np.int64)
    ways[0, 0] = 1
    for r, m in zip(uniq.tolist(), counts.tolist()):
        new = np.zeros_like(ways)
        for k in range(0, min(m, H) + 1):
            shift = k * r
            if shift > max_s:
                break
            new[k:, shift:] += ways[: H + 1 - k, : max_s + 1 - shift] * math.comb(m, k)
        ways = new

    row = ways[H]
    idx = np.arange(max_s + 1)
    p = float(row[np.abs(idx - E2) >= abs(s_obs - E2)].sum()) / TOTAL_SPLITS
    _CACHE[key] = p
    return p


def brute_force_p(recent, older):
    """Reference implementation for validating the DP."""
    from itertools import combinations
    ranks = doubled_midranks(np.concatenate([recent, older]))
    s_obs = int(ranks[:H].sum())
    dev = abs(s_obs - E2)
    tail = sum(1 for c in combinations(range(N_POOL), H)
               if abs(int(ranks[list(c)].sum()) - E2) >= dev)
    return tail / TOTAL_SPLITS


def vargha_delaney(recent: np.ndarray, older: np.ndarray) -> float:
    d = recent[:, None] - older[None, :]
    return float(((d < 0).sum() + 0.5 * (d == 0).sum()) / (H * H))


def probabilities(q, tau=0.5, eta=0.2):
    z = np.asarray(q, dtype=float) / tau
    z = z - z.max()
    sm = np.exp(z)
    sm /= sm.sum()
    return (1 - eta) * sm + eta / len(q)


# --------------------------------------------------------------- validation --

def validate():
    rng = np.random.default_rng(0)
    print("=" * 88)
    print("0.  DP validated against exhaustive enumeration of all C(20,10)=184756 splits")
    print("=" * 88)
    ok = True
    for trial in range(6):
        if trial < 3:
            v = rng.normal(0, 1, N_POOL)
        else:  # tie-heavy
            v = np.where(rng.random(N_POOL) < 0.6, 0.0, rng.uniform(0.1, 0.9, N_POOL))
        _CACHE.clear()
        a, b = exact_two_sided_p(v[:H], v[H:]), brute_force_p(v[:H], v[H:])
        match = abs(a - b) < 1e-12
        ok &= match
        print(f"  trial {trial}: DP={a:.9f}  brute={b:.9f}  {'OK' if match else 'MISMATCH'}")
    print(f"  => {'all match' if ok else 'FAILURE'}")
    return ok


# ------------------------------------------------------------- SNR sweep -----

def snr_sweep(reps=4000):
    rng = np.random.default_rng(11)
    print("\n" + "=" * 88)
    print("A.  detection rate of the permutation deadband vs SNR, H=10")
    print("=" * 88)
    print(f"{'SNR':>6} {'true A':>8} {'mean G':>8} {'det@.05':>10} {'det@.0125':>11}")
    for snr in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0):
        h05 = h0125 = 0
        gs = []
        for _ in range(reps):
            older = rng.normal(0.0, 1.0, H)
            recent = rng.normal(-snr, 1.0, H)
            gs.append(vargha_delaney(recent, older))
            p = exact_two_sided_p(recent, older)
            h05 += p <= 0.05
            h0125 += p <= 0.0125
        a = 0.5 * math.erfc(-snr / 2.0)
        print(f"{snr:>6.2f} {a:>8.3f} {np.mean(gs):>8.3f} "
              f"{h05/reps:>10.3f} {h0125/reps:>11.3f}")


# --------------------------------------------------------- closed-loop sim ---

class Arm:
    def __init__(self, name, start, end, t0, span, noise_mult):
        self.name, self.start, self.end = name, start, end
        self.t0, self.span, self.noise_mult = t0, span, noise_mult
        self.win: list[float] = []
        self.n = 0
        self.updates = self.fired = 0
        self.a_sum = 0.0
        self.sd = 0.0

    def level(self, t):          # t = arm-local episode index
        if t <= self.t0:
            return self.start
        if t >= self.t0 + self.span:
            return self.end
        return self.start + (t - self.t0) / self.span * (self.end - self.start)


def simulate(use_holm, snr, total=9000, seed=5, snap_every=750):
    rng = np.random.default_rng(seed)
    arms = [
        #      name          start  end   t0  span  noise
        Arm("A0_easy",       0.45, 0.10,   0,  150, 1.0),
        Arm("A1_traffic",    0.55, 0.22, 120,  180, 1.0),
        Arm("A2_junction",   0.60, 0.32, 280,  200, 1.0),
        Arm("A3_complex",    0.70, 0.50, 450,  200, 1.2),
        Arm("A4_noisy",      0.65, 0.65,   0,    1, 4.0),   # stationary, very noisy
        Arm("A5_hard",       0.80, 0.79,   0,    1, 1.5),   # stationary, hard
    ]
    # sd set so the H-episode gap between window centres equals `snr` sds.
    for a in arms:
        delta_total = abs(a.start - a.end)
        gap = delta_total * min(1.0, H / max(a.span, 1))
        a.sd = (gap / snr if gap > 0 else 0.04) * a.noise_mult

    q = np.full(6, 0.5)
    t = 0
    while min(a.n for a in arms) < 2 * H:
        a = arms[int(np.argmin([x.n for x in arms]))]
        a.win.append(float(rng.normal(a.level(a.n), a.sd)))
        a.n += 1
        t += 1
    calib = t

    snaps, peak = [], np.full(6, 1 / 6)
    for step in range(total):
        p = probabilities(q)
        peak = np.maximum(peak, p)
        i = int(rng.choice(6, p=p))
        a = arms[i]
        a.win.append(float(rng.normal(a.level(a.n), a.sd)))
        a.win = a.win[-2 * H:]
        a.n += 1

        w = np.asarray(a.win)
        g = vargha_delaney(w[H:], w[:H])
        pv = exact_two_sided_p(w[H:], w[:H])
        a.updates += 1
        if pv <= (0.0125 if use_holm else 0.05):
            a.fired += 1
            # gate: the violated-step fraction proxy is the level itself
            fbar = float(np.clip(np.mean(w[H:]), 0, 1))
            fb = 0.5 + 4 * fbar * (1 - fbar) * (g - 0.5)
        else:
            fb = 0.5
        a.a_sum += fb
        q[i] = 0.9 * q[i] + 0.1 * fb

        if step % snap_every == 0:
            snaps.append((step, probabilities(q).copy()))
    return arms, snaps, calib, peak


def report(label, arms, snaps, calib, peak):
    print("\n" + "-" * 88)
    print(f"{label}   (calibration = {calib} generate episodes)")
    print("-" * 88)
    print("          " + "".join(f"{a.name[:10]:>11}" for a in arms))
    for step, p in snaps:
        print(f"ep{step:>6}  " + "".join(f"{v:>11.3f}" for v in p))
    print("peak p    " + "".join(f"{v:>11.3f}" for v in peak))
    print("fire rate " + "".join(f"{a.fired/max(a.updates,1):>11.3f}" for a in arms))
    print("mean A    " + "".join(f"{a.a_sum/max(a.updates,1):>11.3f}" for a in arms))
    print("n_gen     " + "".join(f"{a.n:>11d}" for a in arms))


def bench():
    rng = np.random.default_rng(2)
    _CACHE.clear()
    s = [rng.normal(0, 1, N_POOL) for _ in range(400)]
    t0 = time.perf_counter()
    for v in s:
        exact_two_sided_p(v[:H], v[H:])
    cold = (time.perf_counter() - t0) / len(s)
    t0 = time.perf_counter()
    for v in s:
        exact_two_sided_p(v[:H], v[H:])
    warm = (time.perf_counter() - t0) / len(s)
    print("\n" + "=" * 88)
    print("C.  cost of one exact permutation p-value (H=10)")
    print("=" * 88)
    print(f"  int64 DP, cold cache : {cold*1e3:>7.3f} ms/call")
    print(f"  int64 DP, warm cache : {warm*1e3:>7.3f} ms/call")
    print(f"  4 dimensions per Generate commit: {4*cold*1e3:.2f} ms worst case")


if __name__ == "__main__":
    if not validate():
        raise SystemExit("DP validation failed")
    snr_sweep()
    print("\n" + "=" * 88)
    print("B.  closed-loop teacher dynamics")
    print("    A0->A1->A2->A3 improve in sequence; A4 stationary+very noisy; A5 stationary+hard")
    print("=" * 88)
    for snr in (0.75, 1.25, 2.0):
        print(f"\n########## SNR per window gap = {snr}  (true A = "
              f"{0.5*math.erfc(-snr/2.0):.3f}) ##########")
        for holm in (False, True):
            a, s, c, pk = simulate(use_holm=holm, snr=snr)
            report(f"{'WITH' if holm else 'NO'} Holm, SNR={snr}", a, s, c, pk)
    bench()
