"""A14 -- the conditions that were assumed rather than checked, decided over Q.

Four things the paper leaned on without establishing them.

(1) `K > 0` WAS ASSUMED.  The text wrote `K := G^(k)[F,F] > 0` as if obvious.
    It is not: `G^(k)` has nullity `k`, because every polynomial of degree
    `<= k-1` is killed by the `k`-th derivative.  After facial reduction the
    surviving question is whether a nonzero polynomial of degree `<= k-1` can
    have a zero of multiplicity `>= l+1` at BOTH endpoints, and since such a
    polynomial has at most `k-1` zeros with multiplicity,

        K > 0   <=>   2(l+1) > k-1   <=>   k <= 2l+2 ,

    with nullity `max(0, k - 2(l+1))`.  With `l = k-1` -- the named population
    and the benchmark of the source paper -- this always holds, so nothing run
    is wrong; but `k = 3, l = 0` is singular (`s(1-s)` is in the kernel), and
    Theorem "for every k, d, n" was false exactly where it advertised
    generality.  Multisegment is tighter, because interior pieces only see
    `eta+1` constraints per end:

        nullity(K_multi) = max(0, N k - (N-1) min(eta+1, k) - 2 min(l+1, k)) .

    a14a checks both formulas exactly, on every swept cell and every row of the
    source paper's Table I, and on configurations chosen to VIOLATE them so the
    condition is not vacuous.

(2) `a_0` WAS BISECTED.  Phase A3 found the nodal point by bisection and quoted
    `0.1127016528695822`.  It has a closed form.  For `f = 2` the condition
    `Gr_12(sigma, 1-sigma) = 0` is a polynomial equation over `Q`, and its
    relevant factor is `2(2d-1)s^2 - 2(2d-1)s + (d-2)`, so

        a_0(d) = (1 - sqrt(3/(2d-1))) / 2 ,        a_0(3) = (1-sqrt(3/5))/2 .

    a14b verifies the divisibility exactly for `d = 3..13` and reports the
    agreement with the bisected value -- an independent check of the whole `Gr`
    pipeline, since the two are computed by completely different routes.

(3) THE BAND'S BOUNDARY WAS CALLED MEASURED.  It is measured for the contacts
    the solver realises, but for the atom family the analysis is about it is
    provable: with atoms at `(sigma, 1/2, 1-sigma)`, the middle weight of
    `K = sum_a w_a u_a u_a^T` is a rational function of `sigma` whose numerator
    is divisible by the minimal polynomial of `a_0`.  So the weight vanishes
    exactly at `a_0`, is positive below it and negative above.  a14c.

(4) MEMBERSHIP IN THE MOMENT CONE WAS ONLY EVER REFUTED.  Theorem "membership
    is decidable" exhibited the separating functional and never the other
    direction.  a14d exhibits it: exact rational atoms and strictly positive
    rational weights reproducing `K`.  a14e then makes the refuting direction
    exact too -- a rational `Y`, verified nonnegative on `[0,1]` by the
    Bernstein coefficients of `p_Y` rather than by sampling.

Writes artifacts/a14_exact_conditions.json.
"""
import argparse
import json
import math
import os
import sys
from fractions import Fraction as F
from math import comb

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
ART = os.path.join(HERE, "..", "artifacts")

from exact_green import (energy_nullity, gr12_polynomial,        # noqa: E402
                         poly_divmod, poly_add, poly_mul, poly_trim,
                         bernstein_poly, reflect, solve as qsolve,
                         q_gram_deriv)

CONFIGS = ((3, 1, 0), (5, 1, 0), (7, 1, 0), (9, 1, 0),
           (5, 2, 1), (7, 2, 1), (7, 3, 2))
NS = (1, 2, 3, 4, 5, 6)
TABLE1 = [("velocity", 1, 1, 0, 12, 0), ("acceleration", 2, 3, 2, 6, 2),
          ("jerk", 3, 5, 3, 4, 3), ("snap", 4, 7, 4, 3, 4)]
# chosen to BREAK the condition, so a14a cannot pass vacuously
CONTROLS = [(5, 3, 0, 1, 0), (7, 3, 0, 1, 0), (9, 5, 1, 1, 0), (5, 3, 2, 3, 0)]


def _ev(p, x):
    t = F(0)
    for c in reversed(p):
        t = t * x + c
    return t


# ---------------------------------------------------------------- a14a
def gate_a14a():
    rows, bad = [], []
    for N in NS:
        for (d, k, l) in CONFIGS:
            eta = max(0, k - 1) if N > 1 else 0
            r = energy_nullity(d, k, l, N, eta)
            r.update(d=d, k=k, l=l, N=N, eta=eta, kind="swept")
            rows.append(r)
            if not r["definite"]:
                bad.append(r)
    for nm, k, d, l, N, eta in TABLE1:
        r = energy_nullity(d, k, l, N, eta)
        r.update(d=d, k=k, l=l, N=N, eta=eta, kind="table1", name=nm)
        rows.append(r)
        if not r["definite"]:
            bad.append(r)
    print("    swept cells and Table I: %d configurations, %d singular"
          % (len(rows), len(bad)))

    ctrl = []
    for (d, k, l, N, eta) in CONTROLS:
        r = energy_nullity(d, k, l, N, eta)
        r.update(d=d, k=k, l=l, N=N, eta=eta, kind="control")
        ctrl.append(r)
        print("    control (d,k,l,N,eta)=%-16s nullity %d (formula %d)  %s"
              % (str((d, k, l, N, eta)), r["nullity"], r["predicted"],
                 "SINGULAR, as intended" if not r["definite"] else "definite"))
    formula_ok = all(x["nullity"] == x["predicted"] for x in rows + ctrl)
    print("    the nullity formula reproduces every case: %s" % formula_ok)
    return dict(rows=rows, controls=ctrl, n=len(rows), n_singular=len(bad),
                singular=[dict((kk, x[kk]) for kk in ("d", "k", "l", "N", "eta"))
                          for x in bad],
                n_controls=len(ctrl),
                n_controls_singular=sum(not x["definite"] for x in ctrl),
                formula_matches_everywhere=bool(formula_ok),
                passed=bool(not bad and formula_ok
                            and all(not x["definite"] for x in ctrl)))


# ---------------------------------------------------------------- a14b
def gate_a14b():
    rows = []
    for d in (3, 5, 7, 9, 11, 13):
        l = (d - 3) // 2
        P, K, Ki = gr12_polynomial(d, 1, l)
        mp = [F(d - 2), F(-2 * (2 * d - 1)), F(2 * (2 * d - 1))]
        _, rem = poly_divmod(P, mp)
        a0 = (1.0 - math.sqrt(3.0 / (2 * d - 1))) / 2.0
        # Divisibility exhibits A root.  The proposition asserts a_0 IS the
        # nodal point, which needs it to be the only root in (0, 1/2) -- else
        # the certificate could fail somewhere else first and the closed form
        # would be naming the wrong crossing.
        rts = np.roots([float(c) for c in P[::-1]])
        inner = sorted({round(float(z.real), 10) for z in rts
                        if abs(z.imag) < 1e-9 and 1e-8 < z.real < 0.5 - 1e-8})
        rows.append(dict(d=d, l=l, divides=bool(rem == [F(0)]),
                         minimal_poly=[str(x) for x in mp], a0=a0,
                         n_roots_in_open_half=len(inner),
                         unique_root=bool(len(inner) == 1
                                          and abs(inner[0] - a0) < 1e-7)))
        print("    d=%-3d l=%-2d  %ds^2-%ds+%d divides Gr_12: %-5s  "
              "a0 = (1-sqrt(3/%d))/2 = %.12f"
              % (d, l, 2 * (2 * d - 1), 2 * (2 * d - 1), d - 2,
                 rem == [F(0)], 2 * d - 1, a0))
    # The per-d input, isolated.  Symmetry reduces the nodal condition to the
    # single equation A = q/(2(p+q)) with A = sigma(1-sigma), so uniqueness of
    # the root in (0,1/2) is free (A is increasing there) and every dependence
    # on d sits in one Gram ratio.  Checking THAT is checking the proposition,
    # and it reaches much further than the degree-96 divisibility sweep does.
    ratio = []
    for dd in range(3, 42, 2):
        ll = (dd - 3) // 2
        Fi = [ll + 1, ll + 2]
        G = q_gram_deriv(dd, 1)
        pp, qq = G[Fi[0]][Fi[0]], G[Fi[0]][Fi[1]]
        ratio.append(dict(d=dd, holds=bool(qq / pp == F(dd - 2, dd + 1)
                                           and G[Fi[1]][Fi[1]] == pp)))
    n_ratio = sum(x["holds"] for x in ratio)
    print("    Gram identity q/p = (d-2)/(d+1): %d of %d odd d in 3..41"
          % (n_ratio, len(ratio)))

    a0_exact = (1.0 - math.sqrt(0.6)) / 2.0
    bisected = 0.1127016528695822
    print("    closed form %.16f vs A3's bisected %.16f -> agree to %.2e"
          % (a0_exact, bisected, abs(a0_exact - bisected)))
    # Is a_0 a Gauss-Legendre node?  `Z = 0` means `K = sum_a w_a u u^T`, which
    # is quadrature-shaped, so the question is worth asking -- and worth
    # ANSWERING, because two hits out of three would otherwise become folklore.
    gauss = []
    for r in rows:
        best = None
        for n in range(2, 9):
            x, _ = np.polynomial.legendre.leggauss(n)
            nodes = sorted((x + 1) / 2)
            e = min(abs(r["a0"] - y) for y in nodes)
            if best is None or e < best[0]:
                best = (e, n)
        gauss.append(dict(d=r["d"], nearest_err=best[0], order=best[1],
                          is_node=bool(best[0] < 1e-12)))
        print("    d=%-3d a0=%.12f  %s"
              % (r["d"], r["a0"],
                 "exact %d-point Gauss node" % best[1] if best[0] < 1e-12
                 else "no Gauss node of order 2-8 (nearest miss %.1e)" % best[0]))
    n_hits = sum(g["is_node"] for g in gauss)
    print("    -> %d of %d are Gauss nodes: a coincidence, not stated in the "
          "paper" % (n_hits, len(gauss)))
    return dict(rows=rows, n=len(rows),
                n_divides=sum(x["divides"] for x in rows),
                a0_exact=a0_exact, a0_bisected=bisected,
                agreement=abs(a0_exact - bisected),
                gauss=gauss, n_gauss_hits=n_hits,
                worst_gauss_miss=max(g["nearest_err"] for g in gauss),
                n_unique=sum(x["unique_root"] for x in rows),
                gram_ratio=ratio, n_gram_ratio=n_ratio,
                max_d_gram_ratio=max(x["d"] for x in ratio),
                passed=bool(all(x["divides"] for x in rows)
                            and all(x["unique_root"] for x in rows)
                            and n_ratio == len(ratio)
                            and abs(a0_exact - bisected) < 1e-7))


# ---------------------------------------------------------------- a14c
def _middle_weight_numerator():
    """Numerator of the middle weight, as a polynomial in `sigma` over `Q`."""
    d, k, l = 3, 1, 0
    Fi = list(range(l + 1, d - l))
    G = q_gram_deriv(d, k)
    K = [[G[i][j] for j in Fi] for i in Fi]
    u = [bernstein_poly(i, d) for i in Fi]
    v = [reflect(p) for p in u]
    half = [F(3, 8), F(3, 8)]
    c1 = [poly_mul(u[0], u[0]), poly_mul(u[0], u[1]), poly_mul(u[1], u[1])]
    c3 = [poly_mul(v[0], v[0]), poly_mul(v[0], v[1]), poly_mul(v[1], v[1])]
    c2 = [[half[0] * half[0]], [half[0] * half[1]], [half[1] * half[1]]]
    rhs = [[K[0][0]], [K[0][1]], [K[1][1]]]

    def det3(M):
        def m(a, b):
            return poly_mul(a, b)

        t = m(M[0][0], poly_add(m(M[1][1], M[2][2]),
                                [-x for x in m(M[1][2], M[2][1])]))
        t = poly_add(t, [-x for x in m(M[0][1],
                                       poly_add(m(M[1][0], M[2][2]),
                                                [-x for x in m(M[1][2], M[2][0])]))])
        t = poly_add(t, m(M[0][2], poly_add(m(M[1][0], M[2][1]),
                                            [-x for x in m(M[1][1], M[2][0])])))
        return poly_trim(t)

    D = det3([[c1[i], c2[i], c3[i]] for i in range(3)])
    Nm = det3([[c1[i], rhs[i], c3[i]] for i in range(3)])
    return Nm, D


def gate_a14c():
    Nm, D = _middle_weight_numerator()
    _, rem = poly_divmod(Nm, [F(1), F(-10), F(10)])
    print("    minimal polynomial of a0 divides the middle weight's numerator:",
          rem == [F(0)])
    samples = []
    for s in (F(1, 20), F(1, 10), F(11, 100), F(1127, 10000),
              F(1128, 10000), F(1, 8), F(1, 5)):
        val = _ev(Nm, s) / _ev(D, s)
        samples.append(dict(sigma=float(s), w_middle=float(val),
                            sign=int(np.sign(float(val)))))
        print("      sigma=%-10s w_middle=%+.6g" % (s, float(val)))
    a0 = (1.0 - math.sqrt(0.6)) / 2.0
    below = [x for x in samples if x["sigma"] < a0]
    above = [x for x in samples if x["sigma"] > a0]
    return dict(divides=bool(rem == [F(0)]), a0=a0, samples=samples,
                n_below=len(below), n_below_positive=sum(x["sign"] > 0 for x in below),
                n_above=len(above), n_above_negative=sum(x["sign"] < 0 for x in above),
                passed=bool(rem == [F(0)]
                            and all(x["sign"] > 0 for x in below)
                            and all(x["sign"] < 0 for x in above)))


# ---------------------------------------------------------------- a14d
def gate_a14d():
    """The membership direction, over `Q`: atoms and strictly positive weights."""
    d, k, l = 3, 1, 0
    Fi = list(range(l + 1, d - l))
    G = q_gram_deriv(d, k)
    K = [[G[i][j] for j in Fi] for i in Fi]

    def uvec(s):
        return [F(comb(d, i)) * s ** i * (1 - s) ** (d - i) for i in Fi]

    sigma = F(1, 10)
    ss = [sigma, F(1, 2), 1 - sigma]
    U = [uvec(s) for s in ss]
    A = [[U[a][0] ** 2 for a in range(3)],
         [U[a][0] * U[a][1] for a in range(3)],
         [U[a][1] ** 2 for a in range(3)]]
    w = [r[0] for r in qsolve(A, [[K[0][0]], [K[0][1]], [K[1][1]]])]
    rebuilt = [[sum(w[a] * U[a][i] * U[a][j] for a in range(3))
                for j in range(2)] for i in range(2)]
    exact = all(rebuilt[i][j] == K[i][j] for i in range(2) for j in range(2))
    print("    atoms s = %s" % ", ".join(str(x) for x in ss))
    print("    weights  = %s   all > 0: %s" % (", ".join(str(x) for x in w),
                                               all(x > 0 for x in w)))
    print("    reproduces K exactly: %s" % exact)
    return dict(atoms=[str(x) for x in ss], weights=[str(x) for x in w],
                weights_float=[float(x) for x in w],
                all_positive=bool(all(x > 0 for x in w)),
                reproduces_K_exactly=bool(exact),
                passed=bool(exact and all(x > 0 for x in w)))


# ---------------------------------------------------------------- a14e
def _bernstein_coeffs_of_pY(Y, d, Fi):
    """Degree-`2d` Bernstein coefficients of `p_Y(s) = u(s)^T Y u(s)`, exactly.

    `B_{i,d}B_{j,d} = [C(d,i)C(d,j)/C(2d,i+j)] B_{i+j,2d}`, so this is the
    `S_d` map of the paper applied to `Y` -- a finite exact rational formula,
    no sampling.  Nonnegative coefficients certify `p_Y >= 0` on `[0,1]` by the
    convex-hull property, which is the same argument Section V-C uses.
    """
    out = [F(0)] * (2 * d + 1)
    for a, i in enumerate(Fi):
        for b, j in enumerate(Fi):
            out[i + j] += Y[a][b] * F(comb(d, i) * comb(d, j), comb(2 * d, i + j))
    return out


def gate_a14e():
    """Make the refuting direction exact too, on a configuration where `K` is
    NOT in the cone.  A rational `Y` is produced from the numerical projection
    and then verified over `Q`; how it was found does not matter."""
    from scipy.optimize import nnls
    rows = []
    for (d, k, l) in ((5, 1, 0), (7, 1, 0), (9, 1, 0)):
        Fi = list(range(l + 1, d - l))
        G = q_gram_deriv(d, k)
        K = [[G[i][j] for j in Fi] for i in Fi]
        Kf = np.array([[float(x) for x in r] for r in K])
        f = len(Fi)
        ss = np.linspace(0.0, 1.0, 401)
        U = np.array([[comb(d, i) * s ** i * (1 - s) ** (d - i) for i in Fi]
                      for s in ss])
        A = np.array([np.outer(u, u).ravel() for u in U]).T
        w, _ = nnls(A, Kf.ravel())
        R = Kf - (A @ w).reshape(f, f)          # sign convention of a7
        Yf = -0.5 * (R + R.T)
        # Rationalise, then push Y strictly INTO the dual cone.  p_Y touches
        # zero at the support atoms, so a rounded Y can dip just below; adding
        # a small multiple of the rank-one 11^T -- whose p is
        # (sum_{i in F} B_{i,d})^2, with strictly positive Bernstein
        # coefficients at every attainable index -- restores nonnegativity
        # without changing the sign of <K,Y>, which is very negative.  How Y is
        # found does not matter; only the exact checks below do.
        Yq = [[F(int(round(Yf[i][j] * 10 ** 9)), 10 ** 9) for j in range(f)]
              for i in range(f)]
        ones = [[F(1)] * f for _ in range(f)]
        qc = _bernstein_coeffs_of_pY(ones, d, Fi)
        eps = F(0)
        c0 = _bernstein_coeffs_of_pY(Yq, d, Fi)
        deficit = [(-c, q) for c, q in zip(c0, qc) if c < 0 and q > 0]
        if deficit:
            eps = max(dc / q for dc, q in deficit) * F(11, 10)
        Yq = [[Yq[i][j] + eps for j in range(f)] for i in range(f)]
        coeffs = _bernstein_coeffs_of_pY(Yq, d, Fi)
        nonneg = all(c >= 0 for c in coeffs)
        pair = sum(K[i][j] * Yq[i][j] for i in range(f) for j in range(f))
        rows.append(dict(d=d, k=k, l=l, f=f, eps=float(eps),
                         pY_bernstein_nonneg=bool(nonneg),
                         min_coeff=float(min(coeffs)),
                         inner_KY=float(pair), separates=bool(nonneg and pair < 0)))
        print("    d=%-2d f=%-2d  p_Y Bernstein coefficients >= 0: %-5s  "
              "<K,Y> = %+.4e  ->  %s"
              % (d, f, nonneg, float(pair),
                 "K not in the cone, exactly" if nonneg and pair < 0 else "no verdict"))
    return dict(rows=rows, n=len(rows),
                n_separated=sum(x["separates"] for x in rows),
                passed=bool(all(x["separates"] for x in rows)))


# ---------------------------------------------------------------- a14f
def gate_a14f():
    """The corner of `K^{-1}` and the lobe's onset, in closed form over `Q`.

    A draft appendix claimed the anti-diagonal section
    `h(sigma) = B(sigma, 1-sigma)` of the single-segment Green block has
    `h'(0) < 0`.  That is FALSE: `b_d|_F` vanishes at both endpoints, so
    `h(0) = h'(0) = 0` and the lobe's onset is of order `2(l+1)`, not 1.
    The truth is sharper and exact.  Since `B_{i,d}(sigma) = O(sigma^i)` and
    `B_{j,d}(1-sigma) = O(sigma^{d-j})`, the least order `i + d - j` over
    `F = {l+1, ..., d-l-1}` is `2(l+1)`, attained only at
    `(i, j) = (l+1, d-l-1)`, so

        h(sigma) = C(d,l+1)^2 (K^{-1})_{first,last} sigma^{2(l+1)} + ...

    and for `k = 1, l = 0` the corner entry itself has the closed form

        (K^{-1})_{1,d-1} = -(d-1)/d^2   =>   h = -(d-1) sigma^2 + O(sigma^3).

    So the negative lobe exists at EVERY degree with quadratic onset, and its
    leading coefficient -(d-1) DEEPENS with `d` even as the corner radius
    `a_0(d)` shrinks.  Checked exactly: the full polynomial `h` is expanded
    over `Q` and every coefficient below the predicted order must vanish."""
    rows = []
    for (d, k, l) in ((3, 1, 0), (5, 1, 0), (7, 1, 0), (9, 1, 0), (11, 1, 0),
                      (5, 2, 1), (7, 2, 1), (7, 3, 2)):
        Fi = list(range(l + 1, d - l))
        f = len(Fi)
        G = q_gram_deriv(d, k)
        K = [[G[i][j] for j in Fi] for i in Fi]
        ident = [[F(1) if i == j else F(0) for j in range(f)] for i in range(f)]
        Kinv = qsolve(K, ident)
        corner = Kinv[0][f - 1]

        # h(sigma) = sum_{i,j} B_i(sigma) Kinv_ij B_j(1-sigma), exactly.
        h = [F(0)]
        for a, i in enumerate(Fi):
            Bi = bernstein_poly(i, d)
            for b, j in enumerate(Fi):
                if Kinv[a][b] == 0:
                    continue
                term = poly_mul(Bi, reflect(bernstein_poly(j, d)))
                h = poly_add(h, [Kinv[a][b] * c for c in term])
        h = poly_trim(h)

        onset = 2 * (l + 1)
        low_zero = all(h[m] == 0 for m in range(min(onset, len(h))))
        lead = h[onset] if len(h) > onset else F(0)
        lead_pred = F(comb(d, l + 1)) ** 2 * corner
        closed = F(-(d - 1), d * d)
        is_k1 = (k == 1 and l == 0)
        ok = (low_zero and lead == lead_pred and lead < 0
              and (corner == closed if is_k1 else corner != closed))
        rows.append(dict(d=d, k=k, l=l, f=f, onset=onset,
                         corner=float(corner), corner_num=str(corner),
                         corner_closed_form=bool(corner == closed),
                         lead=float(lead), lead_matches=bool(lead == lead_pred),
                         low_orders_vanish=bool(low_zero),
                         lobe_negative=bool(lead < 0), ok=bool(ok)))
        print("    (d,k,l)=%-9s corner %-12s %s  h ~ %+g s^%d  %s"
              % (str((d, k, l)), str(corner),
                 "= -(d-1)/d^2" if corner == closed else "(no closed form)",
                 float(lead), onset, "ok" if ok else "MISMATCH"))
    n_k1 = sum(1 for x in rows if x["corner_closed_form"])
    print("    closed form holds on all k=1,l=0 rows (%d) and on no other" % n_k1)
    return dict(rows=rows, n=len(rows),
                n_closed_form=n_k1,
                passed=bool(all(x["ok"] for x in rows) and n_k1 == 5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ART, "a14_exact_conditions.json"))
    a = ap.parse_args()

    print("\n[a14a] K > 0 and K_multi > 0: the condition, and where it bites")
    a14a = gate_a14a()
    print("\n[a14b] a_0 in closed form")
    a14b = gate_a14b()
    print("\n[a14c] the middle atom's weight changes sign exactly at a_0")
    a14c = gate_a14c()
    print("\n[a14d] moment-cone MEMBERSHIP, exhibited over Q")
    a14d = gate_a14d()
    print("\n[a14e] moment-cone NON-membership, certified over Q")
    a14e = gate_a14e()
    print("\n[a14f] the corner of K^{-1} and the lobe's onset, in closed form")
    a14f = gate_a14f()

    gates = dict(a14a_invertibility=a14a, a14b_a0_closed_form=a14b,
                 a14c_middle_weight_sign=a14c, a14d_cone_membership=a14d,
                 a14e_cone_separation_exact=a14e,
                 a14f_corner_closed_form=a14f)
    print("\n  --- gates ---")
    for nm, g in gates.items():
        print("  %s: %s" % (nm, "PASS" if g["passed"] else "FAIL"))
    with open(a.out, "w") as fh:
        json.dump(dict(gates=gates), fh, indent=1, default=float)
    print("\nwrote %s" % a.out)
    if not all(g["passed"] for g in gates.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
