"""Detection power of the ACL v2.0 neutrality band as a function of H and SNR.

The 2026-07-31 analysis measured detection rates at H=10 only, and the choice of
H=20 as the specification default rested on an extrapolation from the argument
that power grows as H^1.5 (sample size contributes sqrt(H); the separation
between window centres contributes another factor of H, because SNR itself is
proportional to H for locally linear improvement).

This script measures that claim instead of extrapolating it. It sweeps H over
the candidate values and, for each, reports the detection rate of the exact
conditional permutation band at the specification level of 0.05.

Two SNR conventions are reported, and the distinction is the whole point:

  snr_fixed  - SNR held constant as H varies. Isolates the sample-size effect
               alone. This is NOT the operating situation; it is the control.
  snr_at_10  - SNR quoted at H=10 and scaled linearly with H, i.e. the arm has a
               fixed improvement rate per episode and a larger window therefore
               spans proportionally more training. This is the operating case
               and the one that justifies the default.

Self-contained; numpy only; fixed seeds; consumes no RNG inside the p-value.
"""

from __future__ import annotations

import math

import numpy as np

BAND_LEVEL = 0.05
REPS = 4000


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


def exact_two_sided_p(recent: np.ndarray, older: np.ndarray, h: int) -> float:
    """Exact conditional two-sided permutation p-value, integer DP over tie groups."""
    total_splits = math.comb(2 * h, h)
    if total_splits >= 2 ** 63:
        raise ValueError(f"C(2H,H) overflows int64 at H={h}")
    e2 = h * (2 * h + 1)

    ranks = doubled_midranks(np.concatenate([recent, older]))
    s_obs = int(ranks[:h].sum())
    uniq, counts = np.unique(ranks, return_counts=True)
    key = (h, tuple(uniq.tolist()), tuple(counts.tolist()), s_obs)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit

    max_s = int(np.sort(ranks)[-h:].sum())
    ways = np.zeros((h + 1, max_s + 1), dtype=np.int64)
    ways[0, 0] = 1
    for r, m in zip(uniq.tolist(), counts.tolist()):
        new = np.zeros_like(ways)
        for k in range(0, min(m, h) + 1):
            shift = k * r
            if shift > max_s:
                break
            new[k:, shift:] += ways[: h + 1 - k, : max_s + 1 - shift] * math.comb(m, k)
        ways = new

    row = ways[h]
    if int(row.sum()) != total_splits:
        raise AssertionError(f"DP mass {row.sum()} != C(2H,H) {total_splits} at H={h}")
    idx = np.arange(max_s + 1)
    p = float(row[np.abs(idx - e2) >= abs(s_obs - e2)].sum()) / total_splits
    _CACHE[key] = p
    return p


def vargha_delaney(recent: np.ndarray, older: np.ndarray) -> float:
    d = recent[:, None] - older[None, :]
    return float(((d < 0).sum() + 0.5 * (d == 0).sum()) / (len(recent) * len(older)))


def detection(h: int, snr: float, seed: int) -> tuple[float, float]:
    """Return (mean G, detection rate) over REPS seeded repetitions."""
    rng = np.random.default_rng(seed)
    fired = 0
    gs = np.empty(REPS)
    for j in range(REPS):
        older = rng.normal(0.0, 1.0, h)
        recent = rng.normal(-snr, 1.0, h)          # lower key is better
        gs[j] = vargha_delaney(recent, older)
        fired += exact_two_sided_p(recent, older, h) <= BAND_LEVEL
    return float(gs.mean()), fired / REPS


def main() -> None:
    h_values = (10, 15, 20, 30)
    snr_values = (0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)

    print("=" * 92)
    print("Exact-integer sanity of the DP across the candidate H values")
    print("=" * 92)
    for h in h_values:
        c = math.comb(2 * h, h)
        print(f"  H={h:>3}  C(2H,H)={c:>22,}  fits int64: {c < 2 ** 63}")

    print("\n" + "=" * 92)
    print("A.  CONTROL: SNR held fixed as H grows (isolates the sample-size effect only)")
    print(f"    detection rate of the neutrality band at level {BAND_LEVEL}, {REPS} reps/cell")
    print("=" * 92)
    print(f"{'SNR':>6} {'true A':>8} " + "".join(f"{'H='+str(h):>10}" for h in h_values))
    for snr in snr_values:
        row = [detection(h, snr, seed=1000 + h)[1] for h in h_values]
        a = 0.5 * math.erfc(-snr / 2.0)
        print(f"{snr:>6.2f} {a:>8.3f} " + "".join(f"{v:>10.3f}" for v in row))

    print("\n" + "=" * 92)
    print("B.  OPERATING CASE: SNR quoted at H=10 and scaled linearly with H")
    print("    (fixed improvement rate per arm-episode => a wider window spans more training)")
    print(f"    detection rate at level {BAND_LEVEL}, {REPS} reps/cell")
    print("=" * 92)
    print(f"{'SNR@H=10':>9} " + "".join(f"{'H='+str(h):>10}" for h in h_values))
    for snr in snr_values:
        row = [detection(h, snr * h / 10.0, seed=2000 + h)[1] for h in h_values]
        print(f"{snr:>9.2f} " + "".join(f"{v:>10.3f}" for v in row))

    print("\n" + "=" * 92)
    print("C.  Calibration cost, K=6 arms")
    print("=" * 92)
    for h in h_values:
        print(f"  H={h:>3}  calibration = K*2H = {6 * 2 * h:>4} valid Generate episodes")

    print("\n" + "=" * 92)
    print("D.  False-fire rate under exchangeability (SNR=0), must stay at the band level")
    print("=" * 92)
    for h in h_values:
        g, det = detection(h, 0.0, seed=3000 + h)
        print(f"  H={h:>3}  mean G={g:.4f}  fire rate={det:.4f}  (nominal {BAND_LEVEL})")


if __name__ == "__main__":
    main()
