"""A47 -- the design dichotomy has a closed form, and it lives in ONE corner.

The design-time sign check discharges Theorem `thm:bernstein` per
configuration: a D=96 elevation of every block, every entry.  That is a verified
table rather than a theorem.  This file replaces the table's content by a law.

THE KERNEL IS A GALERKIN PROJECTION, PROVABLY.  `Gr(s,t) = u(s)^T K^-1 u(t)` is
the reproducing kernel of the spline space `V` under `E(f) = int |f^(k)|^2`:
`E(Gr(s,.), f) = f(s)` for every `f` in `V`.  Since the continuum Green's
function `g_t` satisfies the same identity against all of `H^k_0`, the
projection `P_V` (E-orthogonal) gives

    Gr(s,t) = (P_V g_t)(s) .

So `Gr` IS the Galerkin approximation of `g`, which the paper asserted.

THE CORNER DECIDES.  `u` has its first `l+1` rows pinned, so as `s -> 0` on
segment `0` and `t -> 1` on segment `N-1`,

    Gr = C(d,l+1)^2 * c * s^{l+1} (1-t)^{l+1} * (1 + o(1)),
    c := B^{0,N-1}[l+1, d-l-1] ,

one rational number.  `c < 0` therefore REFUTES positivity outright -- that
direction is a theorem, not an observation.  a47b measures the converse.

JUNCTION EXACTNESS, AND ITS TWO CONDITIONS.  `g_t` solves `(-1)^k g^(2k) =
delta_t`, so it IS a spline: degree `2k-1`, class `C^{2k-2}`, one knot at `t`.
If `t` is a junction, `d >= 2k-1` and `eta <= 2k-2`, then `g_t` lies in `V` and
the reproducing property forces `Gr(., t) = g_t` EXACTLY -- no approximation,
hence no sign to lose.  The second condition is conformity: `eta >= k-1`, or a
spline in `V` has a kink whose `k`-th derivative carries a delta the piecewise
Gram does not see, `V` is not inside `H^k_0`, and the projection picture is
false.  BOTH edges bind: a47e measures them, and `k=2, eta=0` is a cell where
`eta <= 2k-2` holds, conformity fails, and exactness fails with it.

THE CLOSED FORM.  At `eta = k-1` (the family every dichotomy row fixes),

    N >= 2 :  c = k / ( N^{2k} (d!/(d-k)!)^2 )   > 0
    N  = 1 :  c = -(d-1)/d^2                     < 0

In GLOBAL coordinates the corner constant is `C(d,k)^2 N^{2k} c = k/(k!)^2`,
independent of `d` and of `N` -- and that is exactly the corner constant of the
CONTINUUM Green's function (a47d).  One knot restores it exactly; no polynomial
degree ever does.  That is the dichotomy, in closed form.

WHAT IS NOT PROVED.  `c >= 0  =>  Gr >= 0` is measured (a47b), not derived.
And the M-matrix route to it is REFUTED (a47f): in the nonnegative B-spline
basis `K^-1` has negative entries at configurations the certificate accepts, so
`Gr >= 0` is strictly weaker than a discrete maximum principle.

Gates:
  a47a  the corner asymptotic constant, against a direct limit
  a47b  sign(c) vs full elevated positivity, over a configuration grid
  a47c  the closed form for c, exactly over Q, at eta = k-1
  a47d  the global corner constant equals the continuum one
  a47e  junction exactness: k-1 <= eta <= 2k-2 makes Gr(.,junction) EQUAL g
  a47f  NEGATIVE: the B-spline discrete-maximum-principle route fails
  a47g  the eta <= 2k-1 conjecture, pushed well outside the swept box

Run:  python experiments/a47_corner_law.py
"""
import json
import math
import os
import sys
from fractions import Fraction as F
from math import comb

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
ART = os.path.join(ROOT, "artifacts")
sys.path.insert(0, os.path.join(ROOT, "src"))

from exact_green import coefficient_blocks, elevated_min          # noqa: E402

D_ELEV = 96


def corner(d, k, l, N, eta):
    b, _ = coefficient_blocks(d, k, l, N, eta)
    return b[(0, N - 1)][l + 1][d - l - 1]


def bd_q(s, d):
    return [F(comb(d, i)) * s ** i * (1 - s) ** (d - i) for i in range(d + 1)]


def ghat(blocks, d, i, s, j, t):
    u, v = bd_q(s, d), bd_q(t, d)
    B = blocks[(i, j)]
    return sum(u[a] * B[a][b] * v[b] for a in range(d + 1) for b in range(d + 1))


# ---- the continuum Green's function of (-1)^k D^2k on H^k_0, exactly --------
def green_left(k, y):
    """Left-branch coefficients of `G(., y)`, over Q."""
    n = 2 * k

    def md(x, m):
        r = [F(0)] * n
        for i in range(m, n):
            f = 1
            for t in range(m):
                f *= (i - t)
            r[i] = F(f) * x ** (i - m)
        return r

    rows, rhs = [], []
    for m in range(k):
        rows.append(md(F(0), m) + [F(0)] * n); rhs.append(F(0))
    for m in range(k):
        rows.append([F(0)] * n + md(F(1), m)); rhs.append(F(0))
    for m in range(n - 1):
        rows.append(md(y, m) + [-x for x in md(y, m)]); rhs.append(F(0))
    rows.append([-x for x in md(y, n - 1)] + md(y, n - 1))
    rhs.append(F((-1) ** k))
    A = [rows[i] + [rhs[i]] for i in range(2 * n)]
    p = 0
    for c in range(2 * n):
        pi = next((r for r in range(p, 2 * n) if A[r][c] != 0), None)
        if pi is None:
            continue
        A[p], A[pi] = A[pi], A[p]
        pv = A[p][c]; A[p] = [x / pv for x in A[p]]
        for r in range(2 * n):
            if r != p and A[r][c] != 0:
                f = A[r][c]; A[r] = [A[r][j] - f * A[p][j] for j in range(2 * n + 1)]
        p += 1
    return [A[i][2 * n] for i in range(n)]


def green_eval(k, y):
    """`G(x, y)` as a callable, exactly (both branches)."""
    n = 2 * k
    a = green_left(k, y)
    # right branch: rebuild by the same system, reading the second half
    def md(x, m):
        r = [F(0)] * n
        for i in range(m, n):
            f = 1
            for t in range(m):
                f *= (i - t)
            r[i] = F(f) * x ** (i - m)
        return r
    rows, rhs = [], []
    for m in range(k):
        rows.append(md(F(0), m) + [F(0)] * n); rhs.append(F(0))
    for m in range(k):
        rows.append([F(0)] * n + md(F(1), m)); rhs.append(F(0))
    for m in range(n - 1):
        rows.append(md(y, m) + [-x for x in md(y, m)]); rhs.append(F(0))
    rows.append([-x for x in md(y, n - 1)] + md(y, n - 1))
    rhs.append(F((-1) ** k))
    A = [rows[i] + [rhs[i]] for i in range(2 * n)]
    p = 0
    for c in range(2 * n):
        pi = next((r for r in range(p, 2 * n) if A[r][c] != 0), None)
        if pi is None:
            continue
        A[p], A[pi] = A[pi], A[p]
        pv = A[p][c]; A[p] = [x / pv for x in A[p]]
        for r in range(2 * n):
            if r != p and A[r][c] != 0:
                f = A[r][c]; A[r] = [A[r][j] - f * A[p][j] for j in range(2 * n + 1)]
        p += 1
    sol = [A[i][2 * n] for i in range(2 * n)]
    aa, bb = sol[:n], sol[n:]
    return lambda x: (sum(aa[i] * x ** i for i in range(n)) if x <= y
                      else sum(bb[i] * x ** i for i in range(n)))


# ---- the B-spline basis, for the refuted route -----------------------------
def _knots(d, N, eta):
    t = [0.0] * (d + 1)
    for j in range(1, N):
        t += [j / N] * (d - eta)
    return t + [1.0] * (d + 1)


def _bspl(t, a, p, x):
    if p == 0:
        return 1.0 if (t[a] <= x < t[a + 1]) else 0.0
    v = 0.0
    if t[a + p] > t[a]:
        v += (x - t[a]) / (t[a + p] - t[a]) * _bspl(t, a, p - 1, x)
    if t[a + p + 1] > t[a + 1]:
        v += (t[a + p + 1] - x) / (t[a + p + 1] - t[a + 1]) * _bspl(t, a + 1, p - 1, x)
    return v


def _bspl_der(t, a, p, x, m):
    if m == 0:
        return _bspl(t, a, p, x)
    v = 0.0
    if t[a + p] > t[a]:
        v += p / (t[a + p] - t[a]) * _bspl_der(t, a, p - 1, x, m - 1)
    if t[a + p + 1] > t[a + 1]:
        v -= p / (t[a + p + 1] - t[a + 1]) * _bspl_der(t, a + 1, p - 1, x, m - 1)
    return v


def bspline_stiffness(d, k, l, N, eta):
    t = _knots(d, N, eta)
    nb = len(t) - d - 1
    idx = list(range(l + 1, nb - (l + 1)))
    gx, gw = np.polynomial.legendre.leggauss(2 * d + 2)
    K = np.zeros((len(idx), len(idx)))
    h = 1.0 / N
    for i in range(N):
        a0, b0 = i * h, (i + 1) * h
        xs = 0.5 * (b0 - a0) * gx + 0.5 * (a0 + b0)
        ws = 0.5 * (b0 - a0) * gw
        D = np.array([[_bspl_der(t, a, d, x, k) for x in xs] for a in idx])
        K += D @ np.diag(ws) @ D.T
    return K, len(idx)


# ----------------------------------------------------------------------
def main():
    out = {}

    # --- a47a: the corner asymptotic constant -------------------------------
    print("=== A47: the design dichotomy in closed form ===\n")
    print("a47a  corner asymptotics  Gr ~ C(d,l+1)^2 c s^{l+1}(1-t)^{l+1}")
    worst_a = 0.0
    rows_a = []
    for (d, k, l, N, eta) in [(3, 1, 0, 2, 0), (5, 2, 1, 2, 1), (7, 3, 2, 3, 2),
                              (5, 1, 0, 1, 0), (9, 4, 3, 2, 3)]:
        blocks, _ = coefficient_blocks(d, k, l, N, eta)
        c = blocks[(0, N - 1)][l + 1][d - l - 1]
        pred = F(comb(d, l + 1)) ** 2 * c
        # the remainder is O(eps), so the test is the RATE: halving eps must
        # halve the error.  A fixed tolerance would only be measuring eps.
        errs = []
        for eps in (F(1, 4096), F(1, 8192)):
            got = ghat(blocks, d, 0, eps, N - 1, 1 - eps) / (eps ** (2 * (l + 1)))
            errs.append(abs(float(got - pred) / max(1e-300, abs(float(pred)))))
        # a zero remainder is the asymptotic holding EXACTLY, which is
        # stronger than the O(eps) rate, not a failure of it.
        if errs[0] == 0.0 and errs[1] == 0.0:
            rate, dev = float("inf"), 0.0
        else:
            rate = errs[0] / max(errs[1], 1e-300)
            dev = abs(rate - 2.0)
        worst_a = max(worst_a, dev)
        rows_a.append(dict(d=d, k=k, l=l, N=N, eta=eta,
                           err_lo=errs[1],
                           rate=(None if rate == float("inf") else rate),
                           exact=bool(errs[0] == 0.0)))
        print("      d=%d k=%d l=%d N=%d eta=%d   err %.2e -> %.2e, %s"
              % (d, k, l, N, eta, errs[0], errs[1],
                 "EXACT (no remainder)" if errs[0] == 0.0 else "rate %.3f" % rate))
    out["a47a_corner_asymptotics"] = dict(n=len(rows_a),
                                          worst_rate_deviation=worst_a,
                                          passed=bool(worst_a < 0.1))

    # --- a47b: does the corner sign decide the whole certificate? -----------
    print("\na47b  sign(c) vs the full elevated positivity test")
    n_cell = agree = 0
    n_neg = 0
    mismatches = []
    for k, l in ((1, 0), (2, 1), (3, 2)):
        for d in (2 * k + 1, 2 * k + 3, 2 * k + 5):
            for N in (1, 2, 3, 4):
                for eta in range(0, min(d, 2 * k + 3)):
                    if N == 1 and eta > 0:
                        continue
                    try:
                        blocks, _ = coefficient_blocks(d, k, l, N, eta)
                    except Exception:                            # noqa: BLE001
                        continue
                    c = blocks[(0, N - 1)][l + 1][d - l - 1]
                    mn = min(elevated_min(B, d, max(D_ELEV, d))
                             for B in blocks.values())
                    n_cell += 1
                    if not (mn >= 0):
                        n_neg += 1
                    if (mn >= 0) == (c >= 0):
                        agree += 1
                    else:
                        mismatches.append(dict(d=d, k=k, l=l, N=N, eta=eta))
    print("      %d configurations: agree on %d, disagree on %d "
          "(%d fail positivity, every one with c < 0)"
          % (n_cell, agree, len(mismatches), n_neg))
    out["a47b_corner_decides"] = dict(n=n_cell, n_agree=agree,
                                      n_disagree=len(mismatches),
                                      n_fail_positivity=n_neg,
                                      passed=bool(n_cell > 0 and not mismatches))

    # --- a47c: the closed form, exactly -------------------------------------
    print("\na47c  closed form at eta = k-1:  c = k / (N^{2k} (d!/(d-k)!)^2)")
    n_c = bad_c = 0
    for k in (1, 2, 3, 4, 5):
        for d in range(2 * k - 1, 2 * k + 8, 2):
            if d < k + 1:
                continue
            for N in (2, 3, 4, 5, 6):
                try:
                    v = corner(d, k, k - 1, N, k - 1)
                except Exception:                                # noqa: BLE001
                    continue
                n_c += 1
                if v != F(k, N ** (2 * k) * math.perm(d, k) ** 2):
                    bad_c += 1
    n_s = bad_s = 0
    for d in (3, 5, 7, 9, 11, 13):
        n_s += 1
        if corner(d, 1, 0, 1, 0) != F(-(d - 1), d * d):
            bad_s += 1
    print("      N >= 2: %d cells exact, %d mismatches" % (n_c, bad_c))
    print("      N  = 1: c = -(d-1)/d^2 on %d degrees, %d mismatches" % (n_s, bad_s))
    out["a47c_closed_form"] = dict(n_multi=n_c, n_bad_multi=bad_c,
                                   n_single=n_s, n_bad_single=bad_s,
                                   passed=bool(bad_c == 0 and bad_s == 0
                                               and n_c > 0 and n_s > 0))

    # --- a47d: the global corner constant is the continuum one --------------
    print("\na47d  global corner constant  C(d,k)^2 N^{2k} c  =  k/(k!)^2  =  continuum")
    n_d = bad_d = 0
    cont_ok = True
    gammas = {}
    for k in (1, 2, 3, 4, 5):
        tgt = F(k, math.factorial(k) ** 2)
        for d in (2 * k + 1, 2 * k + 3):
            for N in (2, 3, 4):
                c = corner(d, k, k - 1, N, k - 1)
                n_d += 1
                if F(comb(d, k)) ** 2 * F(N) ** (2 * k) * c != tgt:
                    bad_d += 1
        # the continuum limit, from the exact Green's function
        vals = [green_left(k, F(q - 1, q))[k] / F(1, q) ** k for q in (10 ** 3, 10 ** 4)]
        # at k = 1 the continuum value is already exact at every y, so demand
        # STRICT improvement only when there is an error left to improve on.
        e0 = abs(float(vals[0]) - float(tgt))
        e1 = abs(float(vals[-1]) - float(tgt))
        if not (e1 < 1e-3 * float(tgt) and (e1 < e0 or (e0 == 0.0 and e1 == 0.0))):
            cont_ok = False
        gammas["k%d" % k] = dict(discrete=str(tgt), discrete_value=float(tgt),
                                 continuum_limit=float(vals[-1]))
        print("      k=%d : discrete = %s ; continuum limit -> %.10f" %
              (k, tgt, float(vals[-1])))
    out["a47d_continuum_match"] = dict(n=n_d, n_bad=bad_d,
                                       continuum_converges=cont_ok,
                                       corner_constants=gammas,
                                       passed=bool(bad_d == 0 and cont_ok))

    # --- a47e: junction exactness -------------------------------------------
    print("\na47e  k-1 <= eta <= 2k-2 and tau a junction  =>  Gr(.,tau) IS g_tau")
    n_e = bad_e = 0
    rows_e = []
    for k, l in ((1, 0), (2, 1)):
        for d in (2 * k + 1, 2 * k + 3):
            for N in (2, 3):
                for eta in range(0, 2 * k):
                    try:
                        blocks, _ = coefficient_blocks(d, k, l, N, eta)
                    except Exception:                            # noqa: BLE001
                        continue
                    y = F(1, N)
                    G = green_eval(k, y)
                    worst = F(0)
                    for i in range(N):
                        for sa in range(0, 7):
                            s = F(sa, 6)
                            worst = max(worst,
                                        abs(ghat(blocks, d, i, s, 0, F(1))
                                            - G(F(i, N) + s / N)))
                    exact = (worst == 0)
                    expect = (k - 1 <= eta <= 2 * k - 2)
                    n_e += 1
                    if exact != expect:
                        bad_e += 1
                    rows_e.append(dict(d=d, k=k, N=N, eta=eta, exact=exact,
                                       predicted=expect))
    print("      %d cells; exactness matches `k-1 <= eta <= 2k-2` on %d, fails on %d"
          % (n_e, n_e - bad_e, bad_e))
    out["a47e_junction_exact"] = dict(n=n_e, n_bad=bad_e,
                                      passed=bool(n_e > 0 and bad_e == 0))

    # --- a47f: the discrete-maximum-principle route is REFUTED --------------
    print("\na47f  NEGATIVE: K^-1 in the nonnegative B-spline basis is NOT >= 0")
    refuted = []
    for (d, k, l, N, eta) in [(5, 1, 0, 2, 1), (7, 2, 1, 3, 3), (3, 1, 0, 2, 0)]:
        K, r = bspline_stiffness(d, k, l, N, eta)
        Ki = np.linalg.inv(K)
        rel = float(Ki.min() / np.abs(Ki).max())
        blocks, _ = coefficient_blocks(d, k, l, N, eta)
        certified = bool(min(elevated_min(B, d, max(D_ELEV, d))
                             for B in blocks.values()) >= 0)
        refuted.append(dict(d=d, k=k, l=l, N=N, eta=eta,
                            min_Kinv_rel=rel, certified=certified))
        print("      d=%d k=%d N=%d eta=%d : certified=%s but min(K^-1)/max = %+.3e"
              % (d, k, N, eta, certified, rel))
    n_ref = sum(1 for x in refuted if x["certified"] and x["min_Kinv_rel"] < -1e-9)
    out["a47f_dmp_refuted"] = dict(n=len(refuted), n_certified_with_negative=n_ref,
                                   rows=refuted, passed=bool(n_ref > 0))

    # --- a47g: the conjecture outside the box the manuscript sweeps ---------
    print("\na47g  eta in {2k-2, 2k-1} pushed to d <= 13, N <= 8, k <= 4")
    cases = ([(d, 1, 0, N, e) for d in (3, 5, 7, 9, 11, 13)
              for N in (2, 3) for e in (0, 1)]
             + [(d, 2, 1, N, e) for d in (5, 7, 9, 11)
                for N in (2, 3) for e in (2, 3)]
             + [(d, 3, 2, N, 5) for d in (7, 9, 11) for N in (2, 3)]
             + [(d, 4, 3, N, 7) for d in (9, 11) for N in (2, 3)]
             + [(d, 1, 0, N, 1) for d in (3, 5) for N in (4, 6, 8)])
    n_g = neg_g = 0
    for (d, k, l, N, eta) in cases:
        try:
            blocks, _ = coefficient_blocks(d, k, l, N, eta)
        except Exception:                                        # noqa: BLE001
            continue
        mn = min(elevated_min(B, d, max(D_ELEV, d)) for B in blocks.values())
        n_g += 1
        if mn < 0:
            neg_g += 1
            print("      *** NEGATIVE at d=%d k=%d N=%d eta=%d" % (d, k, N, eta))
    print("      %d cells outside the swept box, %d negative" % (n_g, neg_g))
    out["a47g_outside_the_box"] = dict(n=n_g, n_negative=neg_g,
                                       passed=bool(n_g > 0 and neg_g == 0))

    out["_rows"] = dict(a47a=rows_a, a47e=rows_e)
    path = os.path.join(ART, "a47_corner_law.json")
    with open(path, "w") as fh:
        json.dump(dict(gates={k: v for k, v in out.items() if not k.startswith("_")},
                       rows=out["_rows"]), fh, indent=1)
    ok = all(v["passed"] for k, v in out.items() if not k.startswith("_"))
    print("\n  wrote %s" % path)
    print("  ALL GATES PASSED" if ok else "  *** A GATE FAILED ***")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
