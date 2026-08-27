"""A2 -- `rho <= 1` for EVERY `k`, via the Green's function and Perron-Frobenius.

Phase 12 could prove `rho <= 1` only at `k = 1` (a Wronskian argument on a
second-order ODE) and left the constant unexplained for `k >= 2`; that was
`docs/rho_le_1.md`'s main open question, and it is the source paper's open
problem #2.  This experiment tests a different route that has no `k` dependence.

THE ARGUMENT.  In the continuum the lifted components `v_i` satisfy

    (-1)^k v_i^(2k) = lambda v_i     (distributionally),

with `v_i` vanishing to order `l+1` at BOTH ends (`V` is supported on the free
columns), and with `lambda = sum_j lambda_j` a nonnegative measure supported on
the contact set -- FINITELY MANY ATOMS `{s_a}` with weights `w_a > 0` (Phase 11
verified this identification to `1.1e-11`).  Inverting the operator through its
Green's function `G`:

    v_i = sum_a w_a v_i(s_a) G(., s_a)      =>      y_i = (Gr W) y_i,

where `y_i = (v_i(s_a))_a`, `Gr_{ba} = G(s_b, s_a)` and `W = diag(w)`.  So every
`y_i` is an eigenvector of the `m x m` matrix `Gr W` for eigenvalue **1**.

Two facts then close it, for every `k`:

  (i)  eigenvalue 1 is the SPECTRAL RADIUS.  Dual feasibility `Z = K - M_lambda
       >= 0` says exactly that the quadratic form `int |v^(k)|^2 - int v^2
       dlambda` is nonnegative, i.e. `lambda` sits at or below the first
       eigenvalue -- so an attained eigenvalue 1 is the largest.  Measured:
       `mu_1 = 1.000000` on 59/59 instances (`a1_contact_rank.json`).
  (ii) `Gr W` is an ENTRYWISE POSITIVE matrix, because the Green's function of
       `(-1)^k D^(2k)` with clamped separated boundary conditions is positive on
       the interior -- classical in 1-D (the 2-D counterexamples for the clamped
       plate do not apply here).  Perron-Frobenius then makes the spectral
       radius a SIMPLE eigenvalue.

Simple eigenvalue 1  =>  the `y_i` are all proportional  =>  the `v_i` are all
proportional  =>  **rho <= 1**, with no restriction on `k`.

This supersedes the Wronskian argument rather than extending it, and it explains
why Phase 12's order-counting prediction (`rho = 2` reachable at `k = 2`) was
refuted: order counting ignores that the contact measure is finitely atomic and
that the resulting eigenvalue problem has a Perron structure.

Gates:
  a2a  the continuum Green's function of the clamped `2k`-order operator is
       strictly positive on the interior, for `k = 1..4`   (fact (ii))
  a2b  the fixed-point identity `y_i = Gr W y_i` holds at the solved optimum
  a2c  `Gr W` is entrywise positive AT THE CONTACT POINTS, and its top
       eigenvalue is simple with a measured margin
  a2d  the finite-`d` Bernstein `Gr` is NOT positive for arbitrary point tuples
       -- so the hypothesis is about contact configurations, not about the basis,
       and that limitation is recorded rather than hidden

Writes artifacts/a2_perron_route.json.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
ART = os.path.join(os.path.dirname(__file__), "..", "artifacts")

from relaxation import Segment                              # noqa: E402
from node import Node, build                                # noqa: E402
from bernstein import bd                                    # noqa: E402
from instances import hard_instances                        # noqa: E402


def bcs(l):
    b0 = [[-2.0, 0.0], [4.0, 0.0]] + [[0.0, 0.0]] * max(0, l - 1)
    b1 = [[2.0, 0.0], [4.0, 0.0]] + [[0.0, 0.0]] * max(0, l - 1)
    return b0[:l + 1], b1[:l + 1]


# ----------------------------------------------------------------------
# (a2a) the continuum Green's function of (-1)^k D^{2k}, clamped both ends
# ----------------------------------------------------------------------
def green_clamped(t, k, npts=None):
    """G(., t) for (-1)^k u^(2k) = delta_t on [0,1] with u^(i)(0)=u^(i)(1)=0,
    i = 0..k-1.  Piecewise polynomial of degree 2k-1; solved as a linear system.

    Unknowns: coefficients of the left piece p (deg <= 2k-1) and the right piece
    q (deg <= 2k-1) -- 4k unknowns.  Equations: k left BCs, k right BCs, 2k-1
    continuity conditions at t, and the jump  (-1)^k [u^(2k-1)] = 1.
    """
    m = 2 * k                                     # coefficients per piece
    N = 2 * m
    A = np.zeros((N, N))
    b = np.zeros(N)
    row = 0

    def dmono(x, deg, order):
        """d^order/dx^order of x^j for j = 0..deg-1, as a row vector."""
        out = np.zeros(deg)
        for j in range(deg):
            if j >= order:
                c = 1.0
                for i in range(order):
                    c *= (j - i)
                out[j] = c * x ** (j - order)
        return out

    for i in range(k):                            # left BCs at 0
        A[row, :m] = dmono(0.0, m, i)
        row += 1
    for i in range(k):                            # right BCs at 1
        A[row, m:] = dmono(1.0, m, i)
        row += 1
    for i in range(2 * k - 1):                    # continuity at t
        A[row, :m] = dmono(t, m, i)
        A[row, m:] = -dmono(t, m, i)
        row += 1
    # jump: (-1)^k ( q^(2k-1)(t) - p^(2k-1)(t) ) = 1
    A[row, :m] = -((-1.0) ** k) * dmono(t, m, 2 * k - 1)
    A[row, m:] = ((-1.0) ** k) * dmono(t, m, 2 * k - 1)
    b[row] = 1.0
    row += 1
    assert row == N, (row, N)

    coef = np.linalg.solve(A, b)
    p, q = coef[:m], coef[m:]

    ss = np.linspace(0.0, 1.0, npts or 401)
    vals = np.where(ss <= t,
                    np.polyval(p[::-1], ss), np.polyval(q[::-1], ss))
    return ss, vals


def green_positivity(ks=(1, 2, 3, 4), nt=25, npts=401, edge=0.05):
    """Minimum of the normalised Green's function over a STRICT interior.

    `edge` matters: `G` vanishes to order `k` at each end, so a min taken right
    up to the boundary reports `O(edge^k)` -- a decay, not a near sign change.
    Both are recorded so the number cannot be misread.
    """
    rows = []
    for k in ks:
        worst = np.inf
        worst_at = None
        for t in np.linspace(0.08, 0.92, nt):
            ss, v = green_clamped(float(t), k, npts)
            interior = (ss > edge) & (ss < 1.0 - edge)
            scale = max(1e-300, float(np.max(np.abs(v))))
            mn = float(np.min(v[interior]) / scale)
            if mn < worst:
                worst, worst_at = mn, float(t)
        rows.append(dict(k=k, min_normalised_G=worst, at_t=worst_at,
                         edge=float(edge), positive=bool(worst > 0)))
    return rows


def green_gram(ss, k):
    """[G(s_b, s_a)] from the continuum Green's function."""
    m = len(ss)
    Gr = np.zeros((m, m))
    for a, t in enumerate(ss):
        grid, v = green_clamped(float(t), k, 20001)
        Gr[:, a] = np.interp(ss, grid, v)
    return 0.5 * (Gr + Gr.T)


# ----------------------------------------------------------------------
# (a2b, a2c) at the solved optimum
# ----------------------------------------------------------------------
def analyse(seg, contact_tol=1e-4, ns=4001):
    from scipy.optimize import nnls
    nd = Node(lo=None, hi=None)
    h = build(seg, nd, use_rlt=False, use_lift=False)
    h["prob"].solve(solver="CLARABEL", tol_gap_abs=1e-10, tol_gap_rel=1e-10,
                    tol_feas=1e-10, max_iter=2000, verbose=False)
    if h["prob"].status != "optimal" or h["Gfree"].value is None:
        return dict(status=h["prob"].status)
    Gf = np.asarray(h["Gfree"].value)
    Xf = 0.5 * (np.asarray(h["Xfree"].value) + np.asarray(h["Xfree"].value).T)
    S = Xf - Gf.T @ Gf
    wS = np.linalg.eigvalsh(S)
    rho = int(np.sum(wS > 1e-6 * max(1.0, float(np.abs(wS).max()))))
    if rho < 1:
        return dict(status="tight")

    Zf = np.asarray(h["cons"][0].dual_value)
    Z = 0.5 * (Zf + Zf.T)[seg.n:, seg.n:]
    K = seg.Gk[np.ix_(seg.Fidx, seg.Fidx)]
    M = K - Z

    res = seg._package(Gf, Xf, float(h["prob"].value))
    V = seg.lifted_V(res)                                   # (rho, d+1)
    ss = np.linspace(0.0, 1.0, ns)
    B = np.array([bd(s, seg.d) for s in ss])
    P = B @ res["Gamma"].T
    vv = np.sum((B @ V.T) ** 2, axis=1) if V.size else np.zeros(ns)
    zeta = np.full(ns, np.inf)
    for c, r in seg.obs:
        zeta = np.minimum(zeta, np.sum((P - c) ** 2, axis=1) + vv - r * r)
    contacts = [float(ss[i]) for i in range(1, ns - 1)
                if zeta[i] <= zeta[i - 1] and zeta[i] <= zeta[i + 1]
                and zeta[i] < contact_tol]
    if not contacts:
        return dict(status="no contact detected")

    U = np.array([bd(s, seg.d)[seg.Fidx] for s in contacts]).T   # (f, m)
    A = np.array([np.outer(U[:, a], U[:, a]).ravel()
                  for a in range(U.shape[1])]).T
    w, rn = nnls(A, M.ravel())
    fit = float(rn / max(1e-12, np.linalg.norm(M)))

    Ki = np.linalg.inv(K)
    Gr = U.T @ Ki @ U
    GW = Gr @ np.diag(w)
    # entrywise positivity, normalised so the scale of w cannot flatter it
    dgg = np.sqrt(np.abs(np.diag(Gr)))
    Grn = Gr / np.outer(dgg, dgg)
    ev = np.sort(np.abs(np.linalg.eigvals(GW)))[::-1]
    top_simple_margin = float((ev[0] - ev[1]) / ev[0]) if ev.size > 1 else None

    # (a2b) the fixed-point identity y = (Gr W) y for every lifted component
    Y = np.array([[float(V[i] @ bd(s, seg.d)) for s in contacts]
                  for i in range(V.shape[0])])                # (rho, m)
    fp_err = 0.0
    for i in range(Y.shape[0]):
        yi = Y[i]
        sc = max(1e-12, float(np.max(np.abs(yi))))
        fp_err = max(fp_err, float(np.max(np.abs(GW @ yi - yi)) / sc))

    return dict(status="optimal", d=int(seg.d), k=int(seg.k), l=int(seg.l),
                f=int(seg.f), rho=rho, m=len(contacts), contacts=contacts,
                atom_fit_residual=fit,
                min_Gr_entry_norm=float(Grn.min()),
                min_GW_entry=float(GW.min()),
                top_eig=float(ev[0]), second_eig=float(ev[1]) if ev.size > 1 else 0.0,
                top_simple_margin=top_simple_margin,
                fixed_point_err=fp_err)


def population():
    out = []
    H = hard_instances()
    for nm in ("symmetric_blocker", "staggered_pair", "three_in_a_row"):
        out.append((f"{nm} k=1 d=3", H[nm]))
    for k in (1, 2, 3, 4):
        l = k - 1
        for extra in (0, 2):
            d = 2 * k + 1 + extra
            b0, b1 = bcs(l)
            for nm in ("symmetric_blocker", "staggered_pair", "three_in_a_row"):
                out.append((f"{nm} k={k} d={d}",
                            dict(hard_instances(d=d, k=k, l=l)[nm],
                                 bc0=b0, bc1=b1)))
    recs = json.load(open(os.path.join(ART, "gate1.json")))["records"]
    for r in recs:
        if r.get("skipped") or r["rho"] < 1:
            continue
        obs = [(np.array([o[0], o[1]]), float(o[2])) for o in r["obstacles"]]
        out.append((f"gate1[{r['i']}]",
                    dict(n=2, d=3, k=1, l=0, obstacles=obs,
                         bc0=[[-2.0, 0.0]], bc1=[[2.0, 0.0]])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ART, "a2_perron_route.json"))
    args = ap.parse_args()

    print("=== A2: rho <= 1 for every k, via the Green's function + "
          "Perron-Frobenius ===")

    print("\n  --- a2a: is the continuum Green's function positive? ---")
    print("  (-1)^k u^(2k) = delta_t on [0,1], u^(i)(0)=u^(i)(1)=0 for i<k)")
    gp = green_positivity(edge=0.05)
    gp_edge = green_positivity(edge=1e-3)
    for r, re_ in zip(gp, gp_edge):
        print("    k=%d  min normalised G on [0.05, 0.95]: %+.4e  "
              "(worst t=%.2f);  up to 1e-3 of the boundary: %+.3e  %s"
              % (r["k"], r["min_normalised_G"], r["at_t"],
                 re_["min_normalised_G"],
                 "POSITIVE" if r["positive"] else "*** NOT POSITIVE ***"))
    a2a = dict(rows=gp, rows_to_boundary=gp_edge,
               passed=bool(all(r["positive"] for r in gp)
                           and all(r["positive"] for r in gp_edge)),
               min_over_k=float(min(r["min_normalised_G"] for r in gp)),
               min_over_k_to_boundary=float(min(r["min_normalised_G"]
                                                for r in gp_edge)),
               note="G vanishes to order k at each end, so the near-boundary "
                    "minimum is O(edge^k) decay, not a near sign change")

    print("\n  --- a2b/a2c: at the solved optimum ---")
    print("  %-24s %2s %2s %2s %3s %2s %10s %10s %9s %11s"
          % ("instance", "k", "d", "f", "rho", "m", "minGr_norm", "top_eig",
             "simple", "fixedpt_err"))
    rows = []
    for name, kw in population():
        r = analyse(Segment(**kw))
        if r.get("status") != "optimal":
            continue
        rows.append(dict(name=name, **{k: v for k, v in r.items()
                                       if k != "status"}))
        print("  %-24s %2d %2d %2d %3d %2d %+10.4f %10.6f %9s %11.2e"
              % (name, r["k"], r["d"], r["f"], r["rho"], r["m"],
                 r["min_Gr_entry_norm"], r["top_eig"],
                 "-" if r["top_simple_margin"] is None
                 else "%.4f" % r["top_simple_margin"], r["fixed_point_err"]))

    fp = [r["fixed_point_err"] for r in rows]
    pos = [r["min_Gr_entry_norm"] for r in rows]
    gw = [r["min_GW_entry"] for r in rows]
    marg = [r["top_simple_margin"] for r in rows
            if r["top_simple_margin"] is not None]
    tops = [r["top_eig"] for r in rows]
    a2b = dict(n=len(rows), max_fixed_point_err=float(max(fp)) if fp else None,
               median_fixed_point_err=float(np.median(fp)) if fp else None,
               passed=bool(fp and max(fp) < 5e-3))
    a2c = dict(n=len(rows),
               min_Gr_entry_norm=float(min(pos)) if pos else None,
               min_GW_entry=float(min(gw)) if gw else None,
               min_top_simple_margin=float(min(marg)) if marg else None,
               median_top_simple_margin=float(np.median(marg)) if marg else None,
               top_eig_range=[float(min(tops)), float(max(tops))] if tops else None,
               n_by_k={str(k): sum(1 for r in rows if r["k"] == k)
                       for k in sorted({r["k"] for r in rows})},
               passed=bool(pos and min(pos) > 0 and min(gw) > 0
                           and marg and min(marg) > 0))

    with open(os.path.join(ART, "a1_contact_rank.json")) as fh:
        a1 = json.load(fh)["gates"]
    a2d = dict(
        note="the hypothesis is about CONTACT configurations, not the basis: "
             "for arbitrary point tuples the finite-d Bernstein Gr does go "
             "negative, and that is measured in a1_contact_rank.json",
        worst_Gr_entry_random_points=a1["a1d_mechanisms_refuted"]
                                       ["worst_Gr_entry_norm"],
        worst_Gr_entry_at_contacts=a2c["min_Gr_entry_norm"],
        passed=bool(a1["a1d_mechanisms_refuted"]["worst_Gr_entry_norm"] < 0
                    and (a2c["min_Gr_entry_norm"] or -1) > 0))

    gates = dict(a2a_green_positive=a2a, a2b_fixed_point=a2b,
                 a2c_perron_at_contacts=a2c, a2d_scope_of_hypothesis=a2d)

    print("\n  --- gates ---")
    for nm in ("a2a_green_positive", "a2b_fixed_point",
               "a2c_perron_at_contacts", "a2d_scope_of_hypothesis"):
        print(f"  {nm}: {'PASS' if gates[nm]['passed'] else 'FAIL'}")
    print(f"\n  Green's function positivity, worst over k=1..4: "
          f"{a2a['min_over_k']:+.3e} on [0.05, 0.95] "
          f"({a2a['min_over_k_to_boundary']:+.2e} up to 1e-3 of the boundary, "
          f"which is the O(edge^k) decay)")
    print(f"  fixed point y = (Gr W) y: max relative error "
          f"{a2b['max_fixed_point_err']:.2e} over {a2b['n']} instances "
          f"(by k: {a2c['n_by_k']})")
    print(f"  entrywise positivity at contacts: min normalised Gr entry "
          f"{a2c['min_Gr_entry_norm']:+.4f}, min (Gr W) entry "
          f"{a2c['min_GW_entry']:+.3e}")
    print(f"  top eigenvalue of Gr W: range {a2c['top_eig_range']}, "
          f"simplicity margin min {a2c['min_top_simple_margin']:.4f}, "
          f"median {a2c['median_top_simple_margin']:.4f}")
    print(f"  scope: Gr goes to {a2d['worst_Gr_entry_random_points']:+.3e} at "
          f"ARBITRARY points but {a2d['worst_Gr_entry_at_contacts']:+.4f} at "
          f"contact points")

    with open(args.out, "w") as fh:
        json.dump(dict(gates=gates, rows=rows), fh, indent=1, default=float)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
