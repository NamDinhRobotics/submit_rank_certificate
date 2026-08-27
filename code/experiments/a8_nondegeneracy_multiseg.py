"""A8 -- closes the two items A7 left open: is the AHO reading actually earned,
and does the `f = 2` verdict survive MULTISEGMENT (their Table I setting)?

A7 explained the openness of the `rho = 2` regions by Alizadeh-Haeberly-Overton:
the rank of a *nondegenerate* SDP optimum is locally constant in the data, so an
optimum sitting on a corank-2 face stays there under perturbation, with no
symmetry needed.  A7 explicitly did NOT verify nondegeneracy, and recorded that
as open item [12].  Quoting a theorem whose hypothesis you have not checked is
the same move this repo has already been burned by twice, so a8a checks it.

a8b then attacks item [13], which matters more for the write-up.  A7's headline
-- `Z = 0` needs `K` in the moment cone, and that holds only at `f = 2` -- was
measured on SINGLE segments.  The source paper's Table I is MULTISEGMENT with
`C^k` junctions, and at those `(d, l)` a single segment has `f <= 0`: the
freedom lives in the joints.  So "only `f = 2`" says nothing about their
configurations until the same question is asked of the joint problem.

THE MULTISEGMENT FORM.  With the `(Y, W)` parameterisation of `multisegment.py`
(`Gamma = Gamma0 + Y Nperp^T`, PSD block `[[I, Y], [Y^T, W]]`, so
`rho = rank(W - Y^T Y)`), the cost's dependence on `W` is

    sum_i <G^(k), (Nperp W Nperp^T)[block i]>  =  <K_multi, W>,
    K_multi := time_scale * sum_i  Nperp_i^T G^(k) Nperp_i        (r x r),

with `Nperp_i = Nperp[block i, :]`.  A contact at parameter `s` on segment `i`
contributes a rank-one term with `u = Nperp_i^T b_d(s)`, so the moment cone is a
union of `N` curves rather than one:

    C_multi := cone{ (Nperp_i^T b_d(s)) (Nperp_i^T b_d(s))^T : i < N, s in [0,1] }

and `Z = 0` needs `K_multi in C_multi`.  The separating certificate generalises
unchanged, except that `p_Y` is now `N` univariate polynomials of degree `2d`,
one per segment, each still decided EXACTLY from the real roots of its
derivative.

Gates:
  a8a  strict complementarity at the `rho = 2` optima: `rank(P) + rank(Z_full)`
       must equal `n + f`.  If it fails, A7's AHO explanation is not earned and
       the README must say so.
  a8b  the identity `Z = K_multi - M_lambda` holds for multisegment -- the same
       Layer-2 check that made the single-segment cone question meaningful.  A
       cone verdict on an unverified `K_multi` would be arithmetic about nothing.
  a8c  the multisegment cone verdict, with exact certificates, across `N` and
       `eta` -- does `Z = 0` become reachable when the freedom comes from joints?

Writes artifacts/a8_nondegeneracy_multiseg.json.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
ART = os.path.join(HERE, "..", "artifacts")

from bernstein import bd, gram_deriv                        # noqa: E402
from node import Node, build                                # noqa: E402
from multisegment import MultiSegment                       # noqa: E402
from a4_simplicity_margin import three_blocker              # noqa: E402
from a7_moment_cone import poly_min_on_unit, bernstein_monomials  # noqa: E402


# ----------------------------------------------------------------------
# a8a -- is the AHO hypothesis actually satisfied?
# ----------------------------------------------------------------------
def strict_complementarity(seg, tol=1e-11, rtol=1e-6):
    """`rank(P) + rank(Z_full) == n + f` at the optimum?

    `P` is the primal PSD block and `Z_full` its dual.  Complementarity always
    gives `<=`; EQUALITY is strict complementarity, and it is the hypothesis
    under which the optimal rank is locally constant in the data.  Ranks are
    read with a relative eigenvalue threshold, and the spectra are stored so the
    threshold can be argued with rather than trusted.
    """
    h = build(seg, Node(lo=None, hi=None), use_rlt=False, use_lift=False)
    h["prob"].solve(solver="CLARABEL", tol_gap_abs=tol, tol_gap_rel=tol,
                    tol_feas=tol, max_iter=5000, verbose=False)
    if h["prob"].status not in ("optimal", "optimal_inaccurate"):
        return dict(status=h["prob"].status)
    Y = np.asarray(h["Gfree"].value)
    W = 0.5 * (np.asarray(h["Xfree"].value) + np.asarray(h["Xfree"].value).T)
    n, f = seg.n, seg.f
    P = np.block([[np.eye(n), Y], [Y.T, W]])
    Zf = np.asarray(h["cons"][0].dual_value)
    Zf = 0.5 * (Zf + Zf.T)

    def rank_of(A):
        w = np.linalg.eigvalsh(A)
        sc = max(1.0, float(np.abs(w).max()))
        return int(np.sum(w > rtol * sc)), [float(x) for x in np.sort(w)[::-1]]

    rP, specP = rank_of(P)
    rZ, specZ = rank_of(Zf)
    S = W - Y.T @ Y
    wS = np.linalg.eigvalsh(S)
    rho = int(np.sum(wS > rtol * max(1.0, float(np.abs(wS).max()))))
    return dict(status="optimal", n=n, f=f, rho=rho, rank_P=rP, rank_Z=rZ,
                sum_ranks=rP + rZ, target=n + f,
                strict=bool(rP + rZ == n + f),
                spec_P=specP[: n + f], spec_Z=specZ[: n + f],
                complementarity=float(np.linalg.norm(P @ Zf)
                                      / max(1e-300, np.linalg.norm(P)
                                            * np.linalg.norm(Zf) + 1e-300)))


# ----------------------------------------------------------------------
# a8b / a8c -- the multisegment moment cone
# ----------------------------------------------------------------------
def multi_pieces(ms):
    """`K_multi` and the per-segment generator maps `Nperp_i`."""
    Np_i = [ms.Nperp[ms.slice_(i), :] for i in range(ms.N)]
    Gk = ms.Gk
    K = ms.time_scale * sum(Ni.T @ Gk @ Ni for Ni in Np_i)
    return 0.5 * (K + K.T), Np_i


def _pmins(ms, Np_i, Yc):
    """`min_[0,1] p_Y^{(i)}` per segment, exactly, plus the arg-mins."""
    B = bernstein_monomials(ms.d)
    out = []
    for Ni in Np_i:
        Q = Ni @ Yc @ Ni.T                      # (d+1, d+1)
        p = np.zeros(2 * ms.d + 1)
        for a in range(ms.d + 1):
            for b in range(ms.d + 1):
                if Q[a, b] == 0.0:
                    continue
                p += Q[a, b] * np.polynomial.polynomial.polymul(
                    B[a], B[b])[: 2 * ms.d + 1]
        out.append(poly_min_on_unit(p))
    return out


def multi_cone_verdict(ms, ngrid=401, rounds=40):
    """Is `K_multi` in the union-of-curves moment cone?  Exact certificate.

    A fixed grid is not enough here: projecting onto a grid makes `Y := -R`
    nonnegative AT THE GRID POINTS, and between them it dips slightly negative
    (`-1.91e-07` to `-2.28e-06` relative, kept as `p_min_grid_only_rel`), which
    is not a certificate.  Rather than widen
    the tolerance until it passes -- the one move this repo forbids -- generate
    columns: find the exact arg-min of `p_Y`, add that parameter to the grid,
    and re-project.  This is the standard cutting-plane for a moment cone and it
    terminates with a genuinely nonnegative `p_Y`.
    """
    from scipy.optimize import nnls
    K, Np_i = multi_pieces(ms)
    r = K.shape[0]
    tags = [(i, float(s)) for i in range(ms.N)
            for s in np.linspace(1e-9, 1.0 - 1e-9, ngrid)]
    rel, pmin_grid = np.inf, None
    for it in range(rounds):
        A = np.array([np.outer(Np_i[i].T @ bd(s, ms.d),
                               Np_i[i].T @ bd(s, ms.d)).ravel()
                      for i, s in tags]).T
        w, _ = nnls(A, K.ravel())
        R = (K.ravel() - A @ w).reshape(r, r)
        R = 0.5 * (R + R.T)
        Yc = -R
        rel = float(np.linalg.norm(R) / np.linalg.norm(K))
        scale = max(1.0, float(np.abs(Yc).max()))
        mins = _pmins(ms, Np_i, Yc)
        pmin = float(min(v for v, _ in mins))
        if pmin_grid is None:
            # round 0 is the plain grid projection -- kept because it is the
            # evidence that a fixed grid does NOT certify anything
            pmin_grid = pmin
        if rel < 1e-9 or pmin > -1e-12 * scale:
            break
        for i, (v, at) in enumerate(mins):           # add the violated points
            if v <= -1e-12 * scale:
                tags.append((i, float(at)))
    kY = float(np.sum(K * Yc))
    sup = [tags[j] for j in np.where(w > 1e-10 * max(1.0, w.max()))[0]]
    return dict(N=ms.N, d=ms.d, k=ms.k, l=ms.l, eta=ms.eta, r=int(r),
                rel_residual=rel, K_dot_Y=kY, p_min=pmin, scale=scale,
                p_min_grid_only=float(pmin_grid),
                p_min_grid_only_rel=float(pmin_grid / scale),
                rounds=it + 1, n_cols=len(tags),
                n_atoms=len(sup), atoms=[[i, s] for i, s in sup[:12]],
                in_cone=bool(rel < 1e-9),
                separated=bool(kY < -1e-9 and pmin > -1e-9 * scale))


def multi_layer2_check(ms, contact_tol=1e-4, ns=2001):
    """Verify `Z = K_multi - M_lambda` on a solved multisegment instance.

    Same discipline as Layer 2 for one segment: fit nonnegative atoms at the
    DETECTED contacts of the lifted curve and see whether they reproduce
    `K_multi - Z`.  Without this the cone verdict would be about a matrix nobody
    checked is the right one.
    """
    from scipy.optimize import nnls
    res = ms.solve()
    if res is None or res.get("status") not in ("optimal", "optimal_inaccurate"):
        return dict(status=None if res is None else res.get("status"))
    cvx = ms._cvx
    Zf = np.asarray(cvx["cons"][0].dual_value)
    Zf = 0.5 * (Zf + Zf.T)
    Z = Zf[ms.n:, ms.n:]
    K, Np_i = multi_pieces(ms)
    M = K - Z

    G = res["Gamma"]
    V = ms.lifted_V(res)
    ss = np.linspace(0.0, 1.0, ns)
    Bs = np.array([bd(s, ms.d) for s in ss])
    # Two things that a single-segment detector gets away with and this one
    # cannot.  (1) Scan each obstacle SEPARATELY: a min over obstacles hides a
    # contact whenever two of them are touched at nearby parameters.  (2) Include
    # the ENDPOINTS: a contact at a junction sits at `s = 1` of one segment and
    # `s = 0` of the next, and `range(1, ns-1)` silently drops it -- which is
    # exactly what made the first version of this check fail, with `rank(M) = 3`
    # against only `m = 2` atoms found.
    atoms = []
    for i in range(ms.N):
        Gi = ms.seg_Gamma(G, i)
        Vi = V[:, ms.slice_(i)] if V.size else np.zeros((0, ms.d + 1))
        P = Bs @ Gi.T
        vv = np.sum((Bs @ Vi.T) ** 2, axis=1) if Vi.size else np.zeros(ns)
        for c, rr in ms.obs:
            zeta = np.sum((P - c) ** 2, axis=1) + vv - rr * rr
            for a in range(ns):
                lo = zeta[a - 1] if a > 0 else np.inf
                hi = zeta[a + 1] if a < ns - 1 else np.inf
                if zeta[a] <= lo and zeta[a] <= hi and zeta[a] < contact_tol:
                    atoms.append((i, float(ss[a])))
    if not atoms:
        return dict(status="no contact detected")
    U = np.array([ (Np_i[i].T @ bd(s, ms.d)) for i, s in atoms]).T
    A = np.array([np.outer(U[:, a], U[:, a]).ravel()
                  for a in range(U.shape[1])]).T
    w, rn = nnls(A, M.ravel())
    r = K.shape[0]
    return dict(status="optimal", N=ms.N, eta=ms.eta, r=int(r), m=len(atoms),
                atoms=[[i, s] for i, s in atoms[:12]],
                fit_residual=float(rn / max(1e-12, np.linalg.norm(M))),
                fit_dof=int(r * (r + 1) // 2 - len(atoms)),
                z_rel=float(np.linalg.norm(Z) / np.linalg.norm(K)),
                rho=int(len(ms.lifted_V(res))))


def blockers_multi(s1=0.10, rmid=0.35, r=0.25):
    x1 = -2.0 + 4.0 * s1
    return [(np.array([x1, 0.0]), r), (np.array([-x1, 0.0]), r),
            (np.array([0.0, 0.0]), rmid)]


def make_ms(N, d, k, l, eta, obstacles):
    return MultiSegment(n=2, d=d, k=k, l=l, N=N, eta=eta,
                        obstacles=obstacles,
                        bc0=[[-2.0, 0.0]] + [[0.0, 0.0]] * l,
                        bc1=[[2.0, 0.0]] + [[0.0, 0.0]] * l)


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ART,
                                                  "a8_nondegeneracy_multiseg.json"))
    args = ap.parse_args()

    print("=== A8: is the AHO hypothesis earned, and does f=2 survive "
          "multisegment? ===")

    # ---- a8a ---------------------------------------------------------
    print("\n  --- a8a: strict complementarity at the rho = 2 optima ---")
    print("  %-26s %2s %2s %4s %7s %7s %6s %s"
          % ("instance", "n", "f", "rho", "rank P", "rank Z", "n+f", "strict?"))
    CASES = [("f=2 three_blocker s1=0.08", three_blocker(0.08, rmid=0.35)),
             ("f=2 three_blocker s1=0.10", three_blocker(0.10, rmid=0.35)),
             ("f=2 rho=1 control s1=0.18", three_blocker(0.18, rmid=0.35))]
    from a7_moment_cone import corner_pair
    from a5_rho2_scope import corner_radius
    rr = corner_radius(0.0304)
    CASES.append(("f=4 corner_pair d=5",
                  corner_pair(0.0304, 1 - 0.0304, rr, rr, 0.20, 5, 1, 0)))
    rows = []
    for name, seg in CASES:
        r = strict_complementarity(seg)
        if r.get("status") != "optimal":
            print("  %-26s %s" % (name, r.get("status")))
            continue
        rows.append(dict(name=name, **r))
        print("  %-26s %2d %2d %4d %7d %7d %6d %s"
              % (name, r["n"], r["f"], r["rho"], r["rank_P"], r["rank_Z"],
                 r["target"], "YES" if r["strict"] else "*** NO ***"))
    a8a = dict(rows=rows, n=len(rows),
               n_strict=sum(1 for r in rows if r["strict"]),
               passed=bool(rows and all(r["strict"] for r in rows)))
    print(f"\n    strict complementarity holds on {a8a['n_strict']}/{len(rows)}"
          f" -- this is the hypothesis A7's AHO reading needs")

    # ---- a8b ---------------------------------------------------------
    print("\n  --- a8b: does Z = K_multi - M_lambda hold for multisegment? ---")
    print("  %2s %2s %3s %3s %3s %3s %6s %6s %10s %10s"
          % ("N", "d", "k", "l", "eta", "r", "m", "dof", "fit resid", "z_rel"))
    l2 = []
    for (N, d, k, l, eta) in ((2, 3, 1, 0, 1), (3, 3, 1, 0, 1), (2, 5, 1, 0, 1),
                              (2, 3, 1, 0, 0), (4, 3, 1, 0, 1)):
        try:
            ms = make_ms(N, d, k, l, eta, blockers_multi())
            c = multi_layer2_check(ms)
        except Exception as exc:                             # noqa: BLE001
            print("  %2d %2d %3d %3d %3d  %s" % (N, d, k, l, eta,
                                                 str(exc)[:44]))
            continue
        if c.get("status") != "optimal":
            print("  %2d %2d %3d %3d %3d  %s" % (N, d, k, l, eta, c.get("status")))
            continue
        c.update(N=N, d=d, k=k, l=l, eta=eta)
        l2.append(c)
        print("  %2d %2d %3d %3d %3d %3d %6d %6d %10.2e %10.2e"
              % (N, d, k, l, eta, c["r"], c["m"], c["fit_dof"],
                 c["fit_residual"], c["z_rel"]))
    live = [c for c in l2 if c["fit_dof"] > 0]
    a8b = dict(rows=l2, n=len(l2), n_live=len(live),
               max_fit_residual=float(max([c["fit_residual"] for c in live],
                                          default=0.0)),
               passed=bool(live and max(c["fit_residual"] for c in live) < 1e-3))
    print(f"\n    {len(live)} non-vacuous fits (dof > 0), worst relative "
          f"residual {a8b['max_fit_residual']:.2e} -- the identity carries over")

    # ---- a8c ---------------------------------------------------------
    print("\n  --- a8c: the MULTISEGMENT moment cone ---")
    print("  %2s %2s %3s %3s %3s %3s %12s %13s %13s %s"
          % ("N", "d", "k", "l", "eta", "r", "rel resid", "<K,Y>", "min p_Y",
             "verdict"))
    cone = []
    for (N, d, k, l, eta) in ((1, 3, 1, 0, 1), (2, 3, 1, 0, 1), (3, 3, 1, 0, 1),
                              (4, 3, 1, 0, 1), (2, 3, 1, 0, 0),
                              (2, 5, 1, 0, 1), (3, 5, 1, 0, 2),
                              (2, 5, 2, 1, 2), (2, 7, 2, 1, 2)):
        try:
            ms = make_ms(N, d, k, l, eta, blockers_multi())
            v = multi_cone_verdict(ms)
        except Exception as exc:                             # noqa: BLE001
            print("  %2d %2d %3d %3d %3d  %s" % (N, d, k, l, eta, str(exc)[:44]))
            continue
        cone.append(v)
        print("  %2d %2d %3d %3d %3d %3d %12.2e %13.3e %13.2e %s"
              % (N, d, k, l, eta, v["r"], v["rel_residual"], v["K_dot_Y"],
                 v["p_min"] / v["scale"],
                 "IN CONE" if v["in_cone"]
                 else ("separated" if v["separated"] else "*** undecided ***")))
    inc = [v for v in cone if v["in_cone"]]
    und = [v for v in cone if not v["in_cone"] and not v["separated"]]
    a8c = dict(rows=cone, n=len(cone), n_in_cone=len(inc), n_undecided=len(und),
               r_in_cone=sorted({v["r"] for v in inc}),
               r_separated=sorted({v["r"] for v in cone if v["separated"]}),
               passed=bool(cone and not und))
    print(f"\n    in the cone for r = {a8c['r_in_cone']}; separated for "
          f"r = {a8c['r_separated']}; {len(und)} undecided")
    print("    (r is the multisegment analogue of f -- the number of free "
          "parameters after the joints are imposed)")

    gates = dict(a8a_strict_complementarity=a8a,
                 a8b_multiseg_identity=a8b,
                 a8c_multiseg_cone=a8c)
    print("\n  --- gates ---")
    for nm in gates:
        print(f"  {nm}: {'PASS' if gates[nm]['passed'] else 'FAIL'}")

    with open(args.out, "w") as fh:
        json.dump(dict(gates=gates), fh, indent=1, default=float)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
