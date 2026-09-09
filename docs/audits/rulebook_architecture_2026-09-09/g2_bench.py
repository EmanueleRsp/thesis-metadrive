"""G2 - candidate-architecture bench.

Every candidate is scored on the same trajectory battery under three comparison
rules: scalar sum, strict lexicographic over episodic channel returns, and
thresholded lexicographic over episodic channel returns.

Trajectories are per-step atomic vectors, the RULEBOOK-V5.1 section 3.4 contract:
    (c_coll, c_l2, c_l3, delta_q, c_l5)
which is exactly the shape tests/test_rulebook_v51_orderings.py uses, so the
figures here are comparable with the repository's own fixtures.

Standard library only.
"""
from __future__ import annotations
import math
from dataclasses import dataclass

# ---------------------------------------------------------------- constants
DT_RATIO = 0.1                     # V51_STEP_DT_S / V51_T_REF_S
DQMAX = 1.0
Q = 90.17 / 2.2222                 # mean mission span in D_REF units = 40.58
STEPS, CRUISE_DQ, TRAFFIC_L2 = 40, 0.25, 0.05
REF_LEGAL, REF_SHORT, REF_L5_STEPS, REF_L5_SEV = 200, 160, 30, 1.0 / 3.0

SHIPPED = dict(a=2.5, sig=0.30, phi=0.25, lam4=2.0, eta=1.0, lam6=0.2)
GAMMA_NOW, GAMMA_NEW = 0.996, 0.9982

Step = tuple[float, float, float, float, float]   # coll, l2, l3, dq, l5


def l6_cost(dq: float) -> float:
    return 1.0 - max(0.0, min(dq, 1.0))


def disc(n: int, g: float) -> float:
    return float(n) if g == 1.0 else (1.0 - g ** n) / (1.0 - g)


# ---------------------------------------------------------------- fixtures
def standing_still(n: int = STEPS) -> list[Step]:
    return [(0.0, 0.0, 0.0, 0.0, 0.0)] * n


def legal_drive(*, l2: float = 0.0, l2_steps: int = 1, dq: float = CRUISE_DQ,
                n: int = STEPS) -> list[Step]:
    steps = [(0.0, 0.0, 0.0, dq, 0.0)] * n
    for i in range(min(l2_steps, n)):
        steps[i] = (0.0, l2, 0.0, dq, 0.0)
    return steps


def with_relaxation(base: list[Step], *, first: int, count: int, severity: float) -> list[Step]:
    out = list(base)
    for i in range(first, min(first + count, len(out))):
        c, l2, l3, dq, _ = out[i]
        out[i] = (c, l2, l3, dq, severity)
    return out


def completing_run(steps: int, *, l5_steps: int = 0, severity: float = 0.0,
                   l2: float = 0.0, l2_steps: int = 0) -> list[Step]:
    adv = Q / steps
    assert adv <= 1.0
    return [(0.0, l2 if i < l2_steps else 0.0, 0.0, adv,
             severity if i < l5_steps else 0.0) for i in range(steps)]


# ---------------------------------------------------------------- candidates
@dataclass
class Candidate:
    key: str
    name: str
    n_channels: int
    # channels(traj, gamma) -> tuple of episodic channel returns, in comparison order
    # sense: True where higher is better
    sense: tuple[bool, ...]
    params: tuple[str, ...]
    thresholds: tuple[str, ...]

    def channels(self, traj, gamma):  # pragma: no cover - overridden
        raise NotImplementedError

    def scalar(self, traj, gamma, **w):  # pragma: no cover - overridden
        raise NotImplementedError


def _ch(traj, gamma, idx):
    return sum(gamma ** t * s[idx] for t, s in enumerate(traj))


def _ch_l6(traj, gamma):
    return sum(gamma ** t * l6_cost(s[3]) for t, s in enumerate(traj))


def _priority_terms(c1, c2, c3, a, sig, phi):
    """SCAL-V1.4's three priority levels, verbatim from v51_reward."""
    total = 0.0
    for w, c in zip((a ** 3, a ** 2, a), (c1, c2, c3)):
        m = 0.0 if abs(c) <= 1e-12 else -c
        sat = 1.0 if m == 0.0 else 0.0
        total += w * ((sat - 1.0) + sig * m) + phi * m
    return total


# ---- A0: status quo, six levels ------------------------------------------
class A0(Candidate):
    def channels(self, traj, gamma):
        return (_ch(traj, gamma, 0), _ch(traj, gamma, 1), _ch(traj, gamma, 2),
                _ch(traj, gamma, 3), _ch(traj, gamma, 4), _ch_l6(traj, gamma))

    def scalar(self, traj, gamma, **w):
        a, sig, phi = w["a"], w["sig"], w["phi"]
        lam4, eta, lam6 = w["lam4"], w["eta"], w["lam6"]
        tot = 0.0
        for t, (c1, c2, c3, dq, c5) in enumerate(traj):
            r = _priority_terms(c1, c2, c3, a, sig, phi)
            r += lam4 * dq - eta * c5 * DT_RATIO - lam6 * l6_cost(dq) * DT_RATIO
            tot += gamma ** t * r
        return tot


# ---- A1: progress last, five channels, no L6 -----------------------------
class A1(Candidate):
    """K1 coll > K2 interaction > K3 non-relaxable > K4 relaxable > K5 progress.

    The scalar adapter keeps relaxable compliance in the *priority* block, with
    its own indicator at weight a^0 = 1, so rank preservation binds at k = 4:
    a^0 > lam4 * DQMAX  ->  lam4 < 1 / (1 + sigma) after the severity slope.
    """
    def channels(self, traj, gamma):
        return (_ch(traj, gamma, 0), _ch(traj, gamma, 1), _ch(traj, gamma, 2),
                _ch(traj, gamma, 4), _ch(traj, gamma, 3))

    def scalar(self, traj, gamma, **w):
        a, sig, phi, lam4 = w["a"], w["sig"], w["phi"], w["lam4"]
        tot = 0.0
        for t, (c1, c2, c3, dq, c5) in enumerate(traj):
            r = 0.0
            for wt, c in zip((a ** 3, a ** 2, a, 1.0), (c1, c2, c3, c5)):
                m = 0.0 if abs(c) <= 1e-12 else -c
                sat = 1.0 if m == 0.0 else 0.0
                r += wt * ((sat - 1.0) + sig * m) + phi * m
            r += lam4 * dq
            tot += gamma ** t * r
        return tot


# ---- A1c: progress last, five channels, relaxable in the CONTINUOUS tail --
class A1c(Candidate):
    """Same channel order as A1, but the scalar adapter keeps the relaxable
    channel as a continuous finite exchange (eta * c5 * dt/T_REF), i.e. it does
    NOT try to make the scalar arm dominate. lam4 is then free up to a - eta*0.1.
    """
    def channels(self, traj, gamma):
        return (_ch(traj, gamma, 0), _ch(traj, gamma, 1), _ch(traj, gamma, 2),
                _ch(traj, gamma, 4), _ch(traj, gamma, 3))

    def scalar(self, traj, gamma, **w):
        a, sig, phi, lam4, eta = w["a"], w["sig"], w["phi"], w["lam4"], w["eta"]
        tot = 0.0
        for t, (c1, c2, c3, dq, c5) in enumerate(traj):
            r = _priority_terms(c1, c2, c3, a, sig, phi)
            r += lam4 * dq - eta * c5 * DT_RATIO
            tot += gamma ** t * r
        return tot


# ---- A2a: four channels, collision and interaction merged ----------------
class A2a(Candidate):
    def channels(self, traj, gamma):
        safety = sum(gamma ** t * max(s[0], s[1]) for t, s in enumerate(traj))
        return (safety, _ch(traj, gamma, 2), _ch(traj, gamma, 4), _ch(traj, gamma, 3))

    def scalar(self, traj, gamma, **w):
        a, sig, phi, lam4, eta = w["a"], w["sig"], w["phi"], w["lam4"], w["eta"]
        tot = 0.0
        for t, (c1, c2, c3, dq, c5) in enumerate(traj):
            r = 0.0
            for wt, c in zip((a ** 2, a), (max(c1, c2), c3)):
                m = 0.0 if abs(c) <= 1e-12 else -c
                sat = 1.0 if m == 0.0 else 0.0
                r += wt * ((sat - 1.0) + sig * m) + phi * m
            r += lam4 * dq - eta * c5 * DT_RATIO
            tot += gamma ** t * r
        return tot


# ---- A2b: four channels, compliance merged with a budget -----------------
class A2b(Candidate):
    def channels(self, traj, gamma):
        comp = sum(gamma ** t * max(s[2], s[4]) for t, s in enumerate(traj))
        return (_ch(traj, gamma, 0), _ch(traj, gamma, 1), comp, _ch(traj, gamma, 3))

    def scalar(self, traj, gamma, **w):
        a, sig, phi, lam4 = w["a"], w["sig"], w["phi"], w["lam4"]
        tot = 0.0
        for t, (c1, c2, c3, dq, c5) in enumerate(traj):
            r = 0.0
            for wt, c in zip((a ** 2, a, 1.0), (c1, c2, max(c3, c5))):
                m = 0.0 if abs(c) <= 1e-12 else -c
                sat = 1.0 if m == 0.0 else 0.0
                r += wt * ((sat - 1.0) + sig * m) + phi * m
            r += lam4 * dq
            tot += gamma ** t * r
        return tot


# ---- A6: six levels, relaxable charged by a satisfaction indicator -------
class A6(Candidate):
    def channels(self, traj, gamma):
        return A0.channels(self, traj, gamma)

    def scalar(self, traj, gamma, **w):
        a, sig, phi = w["a"], w["sig"], w["phi"]
        lam4, w5, lam6 = w["lam4"], w.get("w5", 0.25), w["lam6"]
        tot = 0.0
        for t, (c1, c2, c3, dq, c5) in enumerate(traj):
            r = _priority_terms(c1, c2, c3, a, sig, phi)
            m = 0.0 if abs(c5) <= 1e-12 else -c5
            sat = 1.0 if m == 0.0 else 0.0
            r += w5 * ((sat - 1.0) + sig * m)
            r += lam4 * dq - lam6 * l6_cost(dq) * DT_RATIO
            tot += gamma ** t * r
        return tot


CANDIDATES: list[Candidate] = [
    A0("A0", "status quo, six levels", 6, (False,) * 3 + (True,) + (False,) * 2,
       ("a", "sigma", "phi", "lam4", "eta", "lam6"), ("t1", "t2", "t3", "t4", "t5")),
    A1("A1", "progress last, 5ch, relaxable as priority level", 5,
       (False, False, False, False, True),
       ("a", "sigma", "phi", "lam4"), ("t1", "t2", "t3", "t4")),
    A1c("A1c", "progress last, 5ch, relaxable in the continuous tail", 5,
        (False, False, False, False, True),
        ("a", "sigma", "phi", "lam4", "eta"), ("t1", "t2", "t3", "t4")),
    A2a("A2a", "progress last, 4ch, safety merged", 4,
        (False, False, False, True),
        ("a", "sigma", "phi", "lam4", "eta"), ("t1", "t2", "t3")),
    A2b("A2b", "progress last, 4ch, compliance merged", 4,
        (False, False, False, True),
        ("a", "sigma", "phi", "lam4"), ("t1", "t2", "t3")),
    A6("A6", "six levels, relaxable by indicator (F3)", 6,
       (False,) * 3 + (True,) + (False,) * 2,
       ("a", "sigma", "phi", "lam4", "w5", "lam6"), ("t1", "t2", "t3", "t4", "t5")),
]


# ---------------------------------------------------------------- comparisons
def strict_lex(cand, A, B, gamma):
    ca, cb = cand.channels(A, gamma), cand.channels(B, gamma)
    for v_a, v_b, hi in zip(ca, cb, cand.sense):
        if abs(v_a - v_b) < 1e-12:
            continue
        return (v_a > v_b) if hi else (v_a < v_b), _name(cand, ca.index(v_a) if False else _pos(ca, v_a))
    return None, None


def _pos(seq, val):
    for i, v in enumerate(seq):
        if v == val:
            return i
    return -1


def _name(cand, i):
    return f"K{i+1}"


def deciding(cand, A, B, gamma):
    ca, cb = cand.channels(A, gamma), cand.channels(B, gamma)
    for i, (v_a, v_b) in enumerate(zip(ca, cb)):
        if abs(v_a - v_b) >= 1e-12:
            return f"K{i+1}"
    return None


def lex_prefers(cand, A, B, gamma):
    ca, cb = cand.channels(A, gamma), cand.channels(B, gamma)
    for (v_a, v_b, hi) in zip(ca, cb, cand.sense):
        if abs(v_a - v_b) < 1e-12:
            continue
        return (v_a > v_b) if hi else (v_a < v_b)
    return None


def thresholded_prefers(cand, A, B, gamma, taus):
    """Thresholded lexicographic over episodic channel returns.

    Gabor/Vamplew threshold a *value* objective at tau by comparing min(v, tau):
    both above budget tie and the comparison descends. A cost channel c is the
    value -c with threshold -tau, so the comparison is on max(c, tau), lower
    better. Consequences, and they are the ones that matter:

      * tau = 0 is NOT "clip everything to zero"; it is zero tolerance, i.e.
        exactly strict lexicographic on that channel;
      * constrained channel i ties (and the comparison descends) exactly when
        tau_i >= max(ca_i, cb_i);
      * whenever it decides, the lower cost wins, as in strict lex.

    The last channel is unthresholded and open-ended.
    """
    ca, cb = cand.channels(A, gamma), cand.channels(B, gamma)
    n = len(ca)
    for i in range(n - 1):
        tau = taus[i]
        v_a, v_b = max(ca[i], tau), max(cb[i], tau)
        if abs(v_a - v_b) < 1e-12:
            continue
        return (v_a < v_b), f"K{i+1}"
    hi = cand.sense[n - 1]
    if abs(ca[-1] - cb[-1]) < 1e-12:
        return None, None
    return ((ca[-1] > cb[-1]) if hi else (ca[-1] < cb[-1])), f"K{n}"


def critical_taus(cand, A, B, gamma):
    """Per constrained channel: the tau interval on which it ties, and who wins otherwise.

    Cost channel (lower better): comparison on max(c, tau); ties iff tau >= max(ca, cb).
    Value channel (higher better): comparison on min(v, tau); ties iff tau <= min(ca, cb).
    Returns (name, sense, boundary, winner-if-it-decides).
    """
    ca, cb = cand.channels(A, gamma), cand.channels(B, gamma)
    out = []
    for i in range(len(ca) - 1):
        if abs(ca[i] - cb[i]) < 1e-12:
            continue
        if cand.sense[i]:          # value channel, higher better
            out.append((f"K{i+1}", "value", min(ca[i], cb[i]),
                        "A" if ca[i] > cb[i] else "B"))
        else:                      # cost channel, lower better
            out.append((f"K{i+1}", "cost", max(ca[i], cb[i]),
                        "A" if ca[i] < cb[i] else "B"))
    return out


# ---- A7: progress last, five channels, relaxable by a bounded indicator ---
class A7(Candidate):
    """K1 coll > K2 interaction > K3 non-relaxable > K4 relaxable > K5 progress.

    Scalar adapter: the three priority levels unchanged, plus a *bounded*
    satisfaction indicator w5 on the relaxable channel (F3's counterfactual with
    lam6 = 0), plus lam4 * delta_q. No L6.

    Section 5.4 then caps w5 < (a - lam4*DQMAX) / (1 + sigma) = 0.3846 at the
    shipped a = 2.5, lam4 = 2.0, sigma = 0.30.
    """
    def channels(self, traj, gamma):
        return (_ch(traj, gamma, 0), _ch(traj, gamma, 1), _ch(traj, gamma, 2),
                _ch(traj, gamma, 4), _ch(traj, gamma, 3))

    def scalar(self, traj, gamma, **w):
        a, sig, phi, lam4, w5 = w["a"], w["sig"], w["phi"], w["lam4"], w.get("w5", 0.25)
        tot = 0.0
        for t, (c1, c2, c3, dq, c5) in enumerate(traj):
            r = _priority_terms(c1, c2, c3, a, sig, phi)
            m = 0.0 if abs(c5) <= 1e-12 else -c5
            sat = 1.0 if m == 0.0 else 0.0
            r += w5 * ((sat - 1.0) + sig * m)
            r += lam4 * dq
            tot += gamma ** t * r
        return tot


CANDIDATES.append(
    A7("A7", "progress last, 5ch, relaxable by bounded indicator, no L6", 5,
       (False, False, False, False, True),
       ("a", "sigma", "phi", "lam4", "w5"), ("t1", "t2", "t3", "t4")))
