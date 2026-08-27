"""The one claim in the paper that was never decided over `Q`.

Everything positive is exact: the a priori certificate is a finite sign check on
rational data, verified at degree 96 in exact arithmetic.  The single NEGATIVE
result -- an instance with `rho = 2`, which is what makes `rho <= 1` false at
finite degree -- was read numerically and arbitrated three ways, because a
counterexample needs an optimum and an optimum of (5) is algebraic, not
rational.  The Conclusion said so in as many words.

It does not have to be that way.  Nothing forces the INSTANCE to be given first.

THE CONSTRUCTION.  Fix `d = 3`, `k = 1`, `l = 0`, so `f = 2`, and build a
rational KKT point, then read off the obstacles it is optimal for.

  1. `Z = 0`.  Theorem 14's membership witness already gives `K = sum_a w_a u_a
     u_a^T` over `Q`, with atoms `s = 1/10, 1/2, 9/10` and weights
     `3125/162, 1/3, 3125/162`, all strictly positive.  That is `X_F`
     stationarity, exactly.
  2. Impose the instance's mirror symmetry: `Gamma_F = [[a, -a], [b, b]]` and
     `S = [[sg, ta], [ta, sg]]`.  The middle obstacle's tangency then holds
     identically, and `Gamma_F` stationarity collapses to ONE scalar equation,
     which places the corner obstacles at `x = +-8/5` -- a rational number.
  3. The remaining tangency is linear in `ta`, so `ta` is rational too.
  4. The radii follow from the contact conditions and are rational.

WHAT IS THEN CHECKED, ALL OVER Q, NOTHING SAMPLED:

  * `K = M_lambda` exactly, so `Z = 0` and the dual is feasible;
  * `Gamma_F` stationarity, as an identity between rationals;
  * `(s - s_j)^2` divides `q_j` EXACTLY -- the tangency is algebraic, not a
    small residual -- and the quotient is strictly positive on `[0,1]`, decided
    by a Sturm sequence.  So `q_j >= 0` with equality exactly at `s_j`, which is
    primal feasibility and complementary slackness together;
  * `S > 0`, from `det S > 0` and `tr S > 0` in exact arithmetic.

`S > 0` with `f = 2` gives `rho = rank S = 2` at this optimum, and `rho <= f = 2`
everywhere, so the maximum-rank optimum has `rho = 2`.  No tolerance is used
anywhere in the chain.

Gates:
  a41a  the witness satisfies every KKT condition exactly over Q
  a41b  S is positive definite over Q, hence rho = 2
  a41c  two independent solvers reproduce the witness's objective value, so the
        exact construction and the numerical reading agree
  a41e  the construction is not delicate, and the chosen witness is not on an
        edge: the same box of rational (Gamma_F, S) is scanned exhaustively, the
        witnesses counted, and the chosen one's two margins reported
  a41d  Hypothesis 1 holds for this instance, over Q.  KKT is sufficient for
        optimality only under Slater, and Slater is Hypothesis 1 by the lemma,
        so a RATIONAL strictly collision-free curve is exhibited and its
        clearance decided by the same Sturm test.  With a41d the whole chain --
        Slater, KKT, optimality, rank -- is decided over Q with no tolerance in
        it anywhere.

Run:  python experiments/a41_exact_rho2.py
"""
import json
import os
import sys
from fractions import Fraction as F
from math import comb

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
ART = os.path.join(ROOT, "artifacts")
sys.path.insert(0, os.path.join(ROOT, "src"))

D, K_ORDER, ELL = 3, 1, 0
FIDX = [1, 2]
ATOMS = [F(1, 10), F(1, 2), F(9, 10)]
WEIGHTS = [F(3125, 162), F(1, 3), F(3125, 162)]
# The three free rational choices.  They are not tuned to make the construction
# work -- 3946 rational triples in the scanned box give a valid witness -- but to
# make it ROBUST: this one leaves the start point 0.15 clear of the nearest
# obstacle and puts S's eigenvalue ratio at 0.82, so neither the geometry nor
# the rank reading is anywhere near a boundary.
ALPHA, BETA, SIGMA = F(-3, 4), F(1, 20), F(3, 20)


# ---------- rational univariate polynomials, as coefficient lists ----------
def pmul(p, q):
    r = [F(0)] * (len(p) + len(q) - 1)
    for i, x in enumerate(p):
        for j, y in enumerate(q):
            r[i + j] += x * y
    return r


def padd(*ps):
    n = max(len(p) for p in ps)
    r = [F(0)] * n
    for p in ps:
        for i, x in enumerate(p):
            r[i] += x
    return r


def pscale(p, c):
    return [c * x for x in p]


def pder(p):
    return [p[i] * i for i in range(1, len(p))] or [F(0)]


def peval(p, t):
    return sum(c * t ** i for i, c in enumerate(p))


def pint01(p):
    return sum(c / F(i + 1) for i, c in enumerate(p))


def bern(dd, a):
    out = [F(0)] * (dd + 1)
    for j in range(dd - a + 1):
        out[a + j] += F(comb(dd, a)) * comb(dd - a, j) * (-1) ** j
    return out


def ptrim(p):
    p = list(p)
    while p and p[-1] == 0:
        p.pop()
    return p


def pdivmod(a, b):
    a = list(a)
    q = [F(0)] * max(1, len(a) - len(b) + 1)
    while True:
        a = ptrim(a)
        if len(a) < len(b):
            break
        k = len(a) - len(b)
        c = a[-1] / b[-1]
        q[k] = c
        for i, x in enumerate(b):
            a[k + i] -= c * x
    return q, (a or [F(0)])


def pgcd(a, b):
    a, b = ptrim(a), ptrim(b)
    while b:
        _, r = pdivmod(a, b)
        a, b = b, ptrim(r)
    return [c / a[-1] for c in a] if a else [F(1)]


def sturm_chain(p):
    seq = [ptrim(p), ptrim(pder(p))]
    while True:
        _, r = pdivmod(seq[-2], seq[-1])
        r = ptrim(r)
        if not r:
            break
        seq.append([-c for c in r])
    return seq


def _changes(seq, t):
    vs = [peval(s, t) for s in seq]
    vs = [v for v in vs if v != 0]
    return sum(1 for i in range(len(vs) - 1) if (vs[i] > 0) != (vs[i + 1] > 0))


def positive_on_unit_interval(p):
    """`p > 0` everywhere on `[0,1]`, decided exactly.  No tolerance."""
    p = ptrim(p)
    if not p or peval(p, F(0)) <= 0 or peval(p, F(1)) <= 0:
        return False
    sf, _ = pdivmod(p, pgcd(p, pder(p)))
    sf = ptrim(sf)
    if len(sf) <= 1:
        return True
    ch = sturm_chain(sf)
    return _changes(ch, F(0)) - _changes(ch, F(1)) == 0


def divide_double_root(q, t):
    """`q / (s - t)^2` exactly, or None if the division is not exact."""
    lin = [-t, F(1)]
    for _ in range(2):
        h, r = pdivmod(q, lin)
        if ptrim(r):
            return None
        q = ptrim(h)
    return q


def rational_clear_curve(centres, r2, B, gKx, gKy, U):
    """A rational curve with the prescribed ends and `q_j > 0` on [0,1].

    This is Hypothesis 1 exhibited, not sampled: the clearance polynomials are
    rational and their positivity is decided by the same Sturm test as the rest
    of this file.  The search is over a small grid of rational control points and
    stops at the first witness, so the answer does not depend on how long it ran.
    """
    for hn in range(1, 41):
        for pn in range(-20, 21):
            h, pp = F(hn, 10), F(pn, 10)
            gx = padd(gKx, pscale(U[0], pp), pscale(U[1], -pp))
            gy = padd(gKy, pscale(U[0], h), pscale(U[1], h))
            ok = True
            for (cx, cy), rr in zip(centres, r2):
                dx, dy = padd(gx, [-cx]), padd(gy, [-cy])
                q = ptrim(padd(pmul(dx, dx), pmul(dy, dy), [-rr]))
                if not positive_on_unit_interval(list(q)):
                    ok = False
                    break
            if ok:
                return dict(p=pp, h=h)
    return None


def scan_box(build_one):
    """Every rational (a, b, sg) on the published grid that yields a witness.

    Reported as a count, not as a search that stopped when it found one: a
    construction that works at 3946 points of a grid is a family, and a
    construction that works at one is a coincidence.  The grid is fixed here so
    the count is reproducible.
    """
    found = []
    for an in range(-40, 41):
        for bn in range(1, 41):
            for sn in range(1, 41):
                r = build_one(F(an, 20), F(bn, 20), F(sn, 20))
                if r is not None:
                    found.append(r)
    return found


def main():
    B = [bern(D, a) for a in range(D + 1)]
    dB = [pder(b) for b in B]
    G = [[pint01(pmul(dB[a], dB[b])) for b in range(D + 1)] for a in range(D + 1)]
    K = [[G[a][b] for b in FIDX] for a in FIDX]

    def u_at(t):
        return [F(comb(D, a)) * t ** a * (1 - t) ** (D - a) for a in FIDX]

    M = [[sum(w * u_at(t)[i] * u_at(t)[j] for t, w in zip(ATOMS, WEIGHTS))
          for j in (0, 1)] for i in (0, 1)]
    pencil_exact = (M == K)
    print("    K = %s" % K)
    print("    Z = K - M_lambda = 0 exactly: %s   (weights all > 0: %s)"
          % (pencil_exact, all(w > 0 for w in WEIGHTS)))

    # Gamma_F stationarity places the corner obstacles, exactly
    lam = 4 * (G[3][1] - G[0][1])
    w = WEIGHTS[0]
    u1, u3 = u_at(ATOMS[0]), u_at(ATOMS[2])
    shift = u3[0] - u1[0]
    eps = lam / (2 * w * shift)
    A, Bp = (F(-2), F(0)), (F(2), F(0))
    gKx = padd(pscale(B[0], A[0]), pscale(B[3], Bp[0]))
    gKy = padd(pscale(B[0], A[1]), pscale(B[3], Bp[1]))
    xc = peval(gKx, ATOMS[2]) - eps
    centres = [(-xc, F(0)), (F(0), F(0)), (xc, F(0))]
    print("    corner obstacles at x = +-%s  (exactly rational)" % xc)

    Gf = [[ALPHA, -ALPHA], [BETA, BETA]]
    U = [B[1], B[2]]
    gx = padd(gKx, pscale(U[0], Gf[0][0]), pscale(U[1], Gf[0][1]))
    gy = padd(gKy, pscale(U[0], Gf[1][0]), pscale(U[1], Gf[1][1]))

    # the remaining tangency is linear in tau
    t = ATOMS[0]
    u, up = u_at(t), [peval(pder(U[0]), t), peval(pder(U[1]), t)]
    e1 = (peval(gKx, t) - centres[0][0], peval(gKy, t) - centres[0][1])
    pt = (e1[0] + Gf[0][0] * u[0] + Gf[0][1] * u[1],
          e1[1] + Gf[1][0] * u[0] + Gf[1][1] * u[1])
    gp = (peval(pder(gx), t), peval(pder(gy), t))
    a_sg = u[0] * up[0] + u[1] * up[1]
    a_ta = u[0] * up[1] + u[1] * up[0]
    tau = -(2 * (pt[0] * gp[0] + pt[1] * gp[1]) + 2 * SIGMA * a_sg) / (2 * a_ta)
    S = [[SIGMA, tau], [tau, SIGMA]]
    detS, trS = S[0][0] * S[1][1] - S[0][1] * S[1][0], S[0][0] + S[1][1]
    print("    S = %s   det = %s   tr = %s   positive definite: %s"
          % (S, detS, trS, detS > 0 and trS > 0))

    v = padd(pscale(pmul(U[0], U[0]), S[0][0]),
             pscale(pmul(U[0], U[1]), 2 * S[0][1]),
             pscale(pmul(U[1], U[1]), S[1][1]))
    qs, r2 = [], []
    for i, (cx, cy) in enumerate(centres):
        dx, dy = padd(gx, [-cx]), padd(gy, [-cy])
        raw = padd(pmul(dx, dx), pmul(dy, dy), v)
        rr = peval(raw, ATOMS[i])
        r2.append(rr)
        qs.append(ptrim(padd(raw, [-rr])))

    # stationarity, re-checked as an identity between rationals
    stat = [[F(0), F(0)], [F(0), F(0)]]
    for a in range(3):
        ua = u_at(ATOMS[a])
        ea = (peval(gKx, ATOMS[a]) - centres[a][0],
              peval(gKy, ATOMS[a]) - centres[a][1])
        for i in (0, 1):
            for j in (0, 1):
                stat[i][j] += 2 * WEIGHTS[a] * ea[i] * ua[j]
    L = [[lam, -lam], [F(0), F(0)]]
    stat_exact = (stat == L)
    print("    Gamma_F stationarity holds exactly: %s" % stat_exact)

    tang, pos = [], []
    for i, q in enumerate(qs):
        g = divide_double_root(list(q), ATOMS[i])
        tang.append(g is not None)
        pos.append(bool(g is not None and positive_on_unit_interval(list(g))))
        print("    obstacle %d: r^2 = %-22s (s-s_%d)^2 | q_j: %s   quotient > 0 "
              "on [0,1]: %s" % (i, r2[i], i, tang[-1], pos[-1]))

    def build_one(al, be, sg):
        Gf2 = [[al, -al], [be, be]]
        gx2 = padd(gKx, pscale(U[0], Gf2[0][0]), pscale(U[1], Gf2[0][1]))
        gy2 = padd(gKy, pscale(U[0], Gf2[1][0]), pscale(U[1], Gf2[1][1]))
        tt = ATOMS[0]
        uu, uup = u_at(tt), [peval(pder(U[0]), tt), peval(pder(U[1]), tt)]
        ee = (peval(gKx, tt) - centres[0][0], peval(gKy, tt) - centres[0][1])
        pp = (ee[0] + Gf2[0][0] * uu[0] + Gf2[0][1] * uu[1],
              ee[1] + Gf2[1][0] * uu[0] + Gf2[1][1] * uu[1])
        gg = (peval(pder(gx2), tt), peval(pder(gy2), tt))
        asg = uu[0] * uup[0] + uu[1] * uup[1]
        ata = uu[0] * uup[1] + uu[1] * uup[0]
        if ata == 0:
            return None
        tt2 = -(2 * (pp[0] * gg[0] + pp[1] * gg[1]) + 2 * sg * asg) / (2 * ata)
        lo, hi = sg - abs(tt2), sg + abs(tt2)
        if lo <= 0:
            return None
        vv = padd(pscale(pmul(U[0], U[0]), sg),
                  pscale(pmul(U[0], U[1]), 2 * tt2),
                  pscale(pmul(U[1], U[1]), sg))
        rr2 = []
        qq = []
        for i2, (cx2, cy2) in enumerate(centres):
            dx2, dy2 = padd(gx2, [-cx2]), padd(gy2, [-cy2])
            raw2 = padd(pmul(dx2, dx2), pmul(dy2, dy2), vv)
            v2 = peval(raw2, ATOMS[i2])
            rr2.append(v2)
            qq.append(ptrim(padd(raw2, [-v2])))
        if any(x <= 0 for x in rr2):
            return None
        d0, d1 = (F(-2) - centres[0][0]) ** 2, F(4)
        if not (d0 > rr2[0] and d1 > rr2[1]):
            return None
        for i2, q2 in enumerate(qq):
            g2 = divide_double_root(list(q2), ATOMS[i2])
            if g2 is None or not positive_on_unit_interval(list(g2)):
                return None
        return dict(margin=min(d0 - rr2[0], d1 - rr2[1]), ratio=lo / hi,
                    chosen=(al == ALPHA and be == BETA and sg == SIGMA))

    box = scan_box(build_one)
    mine = [r for r in box if r["chosen"]]
    print("    witnesses on the published rational grid: %d   (the chosen one "
          "is among them: %s)" % (len(box), bool(mine)))
    if mine:
        print("    chosen margins: start clearance %.4f, S eigenvalue ratio %.4f"
              % (float(mine[0]["margin"]), float(mine[0]["ratio"])))

    clear = rational_clear_curve(centres, r2, B, gKx, gKy, U)
    print("    Hypothesis 1, over Q: rational strictly clear curve at "
          "(p, h) = %s" % (None if clear is None
                           else "(%s, %s)" % (clear["p"], clear["h"])))

    # ---- the numerical cross-check ------------------------------------
    from relaxation import Segment
    from node import Node, build as nbuild
    obs = [((float(c[0]), float(c[1])), float(rr) ** 0.5)
           for c, rr in zip(centres, r2)]
    seg = Segment(n=2, d=D, k=K_ORDER, l=ELL, obstacles=obs,
                  bc0=[[-2.0, 0.0]], bc1=[[2.0, 0.0]])
    Gfn = np.array([[float(Gf[0][0]), float(Gf[0][1])],
                    [float(Gf[1][0]), float(Gf[1][1])]])
    Sn = np.array([[float(S[0][0]), float(S[0][1])],
                   [float(S[1][0]), float(S[1][1])]])
    wit_val = float(np.sum(seg.Gk * seg.full_X(Gfn, Gfn.T @ Gfn + Sn)))
    reads = []
    for solver, tol in (("CLARABEL", 1e-11), ("SCS", 1e-10)):
        h = nbuild(seg, Node(lo=None, hi=None), use_rlt=False, use_lift=False)
        kw = (dict(tol_gap_abs=tol, tol_gap_rel=tol, tol_feas=tol, max_iter=4000)
              if solver == "CLARABEL" else dict(eps=tol, max_iters=300000))
        h["prob"].solve(solver=solver, verbose=False, **kw)
        G2 = np.asarray(h["Gfree"].value)
        X2 = 0.5 * (np.asarray(h["Xfree"].value)
                    + np.asarray(h["Xfree"].value).T)
        ev = np.linalg.eigvalsh(X2 - G2.T @ G2)[::-1]
        rho = int((ev > 1e-7 * max(1.0, abs(ev).max())).sum())
        reads.append(dict(solver=solver, tol=tol, status=h["prob"].status,
                          value=float(h["prob"].value), rho=rho))
        print("    %-9s value %.12f   rho = %d" % (solver, h["prob"].value, rho))
    print("    witness   value %.12f   rho = 2 by exact arithmetic" % wit_val)
    gapv = max(abs(r["value"] - wit_val) for r in reads)

    a41a = dict(pencil_exact=bool(pencil_exact),
                stationarity_exact=bool(stat_exact),
                tangency_exact=[bool(x) for x in tang],
                quotient_positive=[bool(x) for x in pos],
                weights_positive=bool(all(w > 0 for w in WEIGHTS)),
                centre=str(xc), radii_sq=[str(x) for x in r2],
                passed=bool(pencil_exact and stat_exact and all(tang)
                            and all(pos) and all(w > 0 for w in WEIGHTS)))
    a41b = dict(S=[[str(S[0][0]), str(S[0][1])], [str(S[1][0]), str(S[1][1])]],
                det=str(detS), trace=str(trS), rho=2, f=2,
                passed=bool(detS > 0 and trS > 0))
    a41c = dict(readings=reads, witness_value=wit_val, worst_gap=gapv,
                passed=bool(all(r["rho"] == 2 for r in reads) and gapv < 1e-8))
    a41e = dict(n_witnesses=len(box), chosen_in_box=bool(mine),
                margin=float(mine[0]["margin"]) if mine else None,
                ratio=float(mine[0]["ratio"]) if mine else None,
                passed=bool(mine and len(box) == 3946))
    a41d = dict(found=bool(clear is not None),
                control_points=(None if clear is None
                                else dict(p=str(clear["p"]), h=str(clear["h"]))),
                passed=bool(clear is not None))
    with open(os.path.join(ART, "a41_exact_rho2.json"), "w") as fh:
        json.dump(dict(gates=dict(a41a_kkt_over_Q=a41a,
                                  a41b_S_definite_over_Q=a41b,
                                  a41c_numerical_agreement=a41c,
                                  a41d_hypothesis1_over_Q=a41d,
                                  a41e_not_delicate=a41e)), fh, indent=1)
    print("\n  gates: a41a %s  a41b %s  a41c %s  a41d %s  a41e %s"
          % (a41a["passed"], a41b["passed"], a41c["passed"], a41d["passed"],
             a41e["passed"]))
    return 0 if all(g["passed"] for g in (a41a, a41b, a41c, a41d, a41e)) else 1


if __name__ == "__main__":
    sys.exit(main())
