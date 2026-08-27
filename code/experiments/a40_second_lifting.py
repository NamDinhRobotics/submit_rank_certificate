"""Section 6, exercised rather than asserted.

Section 6 reads three hypotheses off the proofs -- (H1) one PSD block, (H2) a
definite reduced cost `K` fixed by the parameterization, (H3) every constraint
reaching `X_F` only through `X_F -> u(s)^T X_F u(s)` -- and observes that Lemma
2, `Z = K - M_lambda`, Theorems 5 and 6, Corollary 7 and Theorem 12 use nothing
else.  That is an argument from reading the proofs.  A referee is entitled to
ask for it to be run.

So run it.  Keep the Bezier parameterization and the ball obstacles, which fix
(H1) and (H3), and change the ONE thing the structural claim says may change:
the reduced cost `K`.  Two replacements, neither a derivative energy:

  weighted   K = int_0^1 (1 + 4s) b'(s) b'(s)^T ds     a position-dependent
                                                       effort penalty
  tikhonov   K = G^(1)[F,F] + lambda I                 energy plus a control-
                                                       point regulariser
  mass       K = int_0^1 b(s) b(s)^T ds                the cost int ||gamma||^2

The first two are perturbations of the cost they replace, which is a fair
objection to them: they stay in a neighbourhood of `G^(k)`.  The mass matrix is
not.  Its Green's function is that of the IDENTITY, not of `D^(2k)`, so
Proposition 1's nullity count and the whole disconjugacy reading of Section 4
have nothing whatever to say about it -- which is the point.

Both are symmetric positive definite, so (H2) holds, and neither is `G^(k)` for
any `k`, so Proposition 1's nullity count and the disconjugacy reading of
Section 4 have nothing to say about them.  What should survive is the pencil and
everything downstream of it.

Measured per instance:
  * the pencil identity `Z = K - sum_a w_a u_a u_a^T`, as a relative residual
  * Theorem 6: does the number of unit eigenvalues of `W^{1/2} Gr W^{1/2}`
    equal `dim ker Z`, and does that equal `rho`
  * Theorem 5: `rho <= m`
  * Theorem 13's elevated-Bernstein hypotheses on the NEW `K^{-1}` -- the
    particular half, which has to be re-checked and is not entitled to hold

TWO THINGS THIS DID NOT SHOW AT FIRST, AND NOW DOES.

a40c originally reported the elevated Bernstein minimum at `d = 3`, `N = 1`,
where it is NEGATIVE for the derivative cost too.  All three costs fail there,
so the numbers demonstrated that a value moves, not that a POSITIVE verdict
fails to transfer -- which is the claim that matters.  a40d asks the question
where it can be answered: at `N = 2, 3, 4`, where the derivative cost certifies.

And every instance a40a/a40b touched had `rho = 1` and `m <= 2`, the regime
where the certificate is the two-scalar criterion and Perron-Frobenius is never
used.  a40e adds a three-contact instance, so the eigenvalue machinery whose
transfer is being reported on is actually exercised.

Gates:
  a40a  the pencil and Theorem 6 survive both replacement costs
  a40b  Theorem 5's bound survives both
  a40c  the a priori Bernstein test is re-checked, not inherited: its verdict is
        reported for each cost rather than assumed
  a40d  at N = 2,3,4 the derivative cost certifies -- does a replacement cost?
  a40e  the same, on an instance with three live contacts
  a40g  Corollary 14 quantifies over every ambient dimension and Theorem 5's
        proof says n never enters, so the replacement costs are run in R^3 too
  a40f  the mass cost produces rho = 2 on an ORDINARY instance, and the paper's
        own standard for any rho = 2 reading is three arbitrated solves -- so
        this one gets them too rather than being quoted from a single solve.

Run:  python experiments/a40_second_lifting.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
ART = os.path.join(ROOT, "artifacts")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from relaxation import Segment                                   # noqa: E402
from node import Node, build                                     # noqa: E402
from bernstein import bd, bd_derivs, gram_deriv                  # noqa: E402
from instances import hard_instances                             # noqa: E402
from a3_finite_d_certificate import refine_contact, free_idx, bcs  # noqa: E402
from a5_rho2_scope import three_blocker_cfg                      # noqa: E402


def weighted_gram(d, weight, ns=20001):
    """`int_0^1 weight(s) b'(s) b'(s)^T ds` by Simpson on a fine grid.

    Simpson rather than a quadrature rule tuned to the degree: the integrand is
    a polynomial of degree `2(d-1)` times a weight, and the point here is the
    matrix, not the last digit.  The residual test below is relative, so a
    quadrature error of `1e-12` cannot manufacture a passing pencil.
    """
    s = np.linspace(0.0, 1.0, ns)
    dB = np.array([bd_derivs(si, d, 1)[1] for si in s])          # (ns, d+1)
    w = np.asarray([weight(si) for si in s])
    coef = np.ones(ns)
    coef[1:-1:2] = 4.0
    coef[2:-1:2] = 2.0
    coef *= (s[1] - s[0]) / 3.0
    return np.einsum("i,ia,ib->ab", coef * w, dB, dB)


def mass_gram(d, ns=20001):
    """`int_0^1 b(s) b(s)^T ds`: the cost `int ||gamma||^2`, not a derivative
    energy at all.  Positive definite because the Bernstein basis is."""
    s = np.linspace(0.0, 1.0, ns)
    B = np.array([bd(si, d) for si in s])
    coef = np.ones(ns)
    coef[1:-1:2] = 4.0
    coef[2:-1:2] = 2.0
    coef *= (s[1] - s[0]) / 3.0
    return np.einsum("i,ia,ib->ab", coef, B, B)


def analyse(seg, tol=1e-10, ns=4001, contact_tol=1e-4):
    """`a4.analyse`, but with `K` read off the segment's own cost matrix."""
    from scipy.optimize import nnls
    h = build(seg, Node(lo=None, hi=None), use_rlt=False, use_lift=False)
    h["prob"].solve(solver="CLARABEL", tol_gap_abs=tol, tol_gap_rel=tol,
                    tol_feas=tol, max_iter=2000, verbose=False)
    if h["prob"].status not in ("optimal", "optimal_inaccurate"):
        return dict(status=h["prob"].status)
    Gf = np.asarray(h["Gfree"].value)
    Xf = 0.5 * (np.asarray(h["Xfree"].value) + np.asarray(h["Xfree"].value).T)
    res = seg._package(Gf, Xf, float(h["prob"].value))
    rho = res["rho"]
    if rho < 1:
        return dict(status="tight", rho=0)

    Zfull = 0.5 * (np.asarray(h["cons"][0].dual_value)
                   + np.asarray(h["cons"][0].dual_value).T)
    Z = Zfull[seg.n:, seg.n:]
    F = free_idx(seg.d, seg.l)
    K = seg.Gk[np.ix_(F, F)]
    Ki = np.linalg.inv(K)
    M = K - Z

    V = seg.lifted_V(res)
    ss = np.linspace(0.0, 1.0, ns)
    B = np.array([bd(s, seg.d) for s in ss])
    P = B @ res["Gamma"].T
    vv = np.sum((B @ V.T) ** 2, axis=1) if V.size else np.zeros(ns)
    zeta = np.full(ns, np.inf)
    for c, r in seg.obs:
        zeta = np.minimum(zeta, np.sum((P - c) ** 2, axis=1) + vv - r * r)
    grid = [float(ss[i]) for i in range(1, ns - 1)
            if zeta[i] <= zeta[i - 1] and zeta[i] <= zeta[i + 1]
            and zeta[i] < contact_tol]
    if not grid:
        return dict(status="no contact detected", rho=rho)
    fine = [refine_contact(seg, res["Gamma"], V, s0, 3.0 / (ns - 1))[0]
            for s0 in grid]
    # endpoint contacts carry u = 0 and are not live
    fine = [s for s in fine if np.linalg.norm(bd(s, seg.d)[F]) > 1e-12]
    if not fine:
        return dict(status="no live contact", rho=rho)

    U = np.array([bd(s, seg.d)[F] for s in fine]).T
    A = np.array([np.outer(U[:, a], U[:, a]).ravel()
                  for a in range(U.shape[1])]).T
    w, _ = nnls(A, M.ravel())
    resid = float(np.linalg.norm(A @ w - M.ravel())
                  / max(1e-300, np.linalg.norm(M.ravel())))

    Gr = U.T @ Ki @ U
    Wh = np.diag(np.sqrt(np.maximum(w, 0.0)))
    mu = np.sort(np.linalg.eigvalsh(Wh @ Gr @ Wh))[::-1]
    n_unit = int(np.sum(np.abs(mu - 1.0) < 1e-4))
    wZ = np.linalg.eigvalsh(Z)
    ker = int(np.sum(np.abs(wZ) < 1e-6 * max(1.0, float(np.abs(wZ).max()))))
    return dict(status="ok", rho=rho, m=int(U.shape[1]), ker_dim=ker,
                n_unit=n_unit, pencil_residual=resid,
                mu1=float(mu[0]), mu2=float(mu[1]) if mu.size > 1 else None)


def bernstein_hypotheses(K, d, ell, D=48):
    """Theorem 13's elevated array for a single segment with this `K`.

    Returns the minimum entry of `E B E^T` relative to its largest, where
    `B = K^{-1}` padded back to the full index set.  Nonneg means the a priori
    route is available for this cost; negative means it is not, and the
    structural half is on its own.
    """
    from math import comb
    F = free_idx(d, ell)
    B = np.zeros((d + 1, d + 1))
    B[np.ix_(F, F)] = np.linalg.inv(K[np.ix_(F, F)] if K.shape[0] == d + 1
                                    else K)
    E = np.array([[comb(d, a) * comb(D - d, A - a) / comb(D, A)
                   if 0 <= A - a <= D - d else 0.0
                   for a in range(d + 1)] for A in range(D + 1)])
    Bt = E @ B @ E.T
    scale = max(1e-300, float(np.abs(Bt).max()))
    return float(Bt.min() / scale)


def multiseg_bernstein(d, k, ell, N, eta, Gfull, D=48):
    """Theorem 13's elevated array for a multisegment configuration and cost.

    `K_multi = tau * sum_i Nperp_i^T G Nperp_i`, blocks `Nperp_i K^-1 Nperp_j^T`,
    elevated to degree `D`.  Returns the smallest entry relative to the largest,
    so the verdict is scale-free and comparable across costs.
    """
    from math import comb
    from multisegment import MultiSegment
    b0, b1 = bcs(ell)
    ms = MultiSegment(n=2, d=d, k=k, l=ell, N=N, eta=eta,
                      obstacles=[(np.array([0.0, 0.0]), 0.2)], bc0=b0, bc1=b1)
    Np, r = ms.Nperp, ms.r
    K = np.zeros((r, r))
    for i in range(N):
        Bi = Np[ms.slice_(i), :]
        K += ms.time_scale * Bi.T @ Gfull @ Bi
    Ki = np.linalg.inv(K)
    E = np.array([[comb(d, a) * comb(D - d, A - a) / comb(D, A)
                   if 0 <= A - a <= D - d else 0.0
                   for a in range(d + 1)] for A in range(D + 1)])
    worst, scale = np.inf, 0.0
    for i in range(N):
        for j in range(N):
            B = Np[ms.slice_(i), :] @ Ki @ Np[ms.slice_(j), :].T
            Bt = E @ B @ E.T
            worst = min(worst, float(Bt.min()))
            scale = max(scale, float(np.abs(Bt).max()))
    return worst / max(1e-300, scale), r


def arbitrate(seg, tol_list=(("CLARABEL", 1e-11), ("CLARABEL", 1e-9),
                            ("SCS", 1e-10))):
    """Read the gap's spectrum three ways, as Section 4 requires of any rho=2.

    Returns one row per reading: the two largest eigenvalues of `S`, their
    ratio, and the optimal value.  A genuine rank-two gap agrees across all
    three; a degeneracy read off one solver does not.
    """
    out = []
    for solver, tol in tol_list:
        h = build(seg, Node(lo=None, hi=None), use_rlt=False, use_lift=False)
        kw = (dict(tol_gap_abs=tol, tol_gap_rel=tol, tol_feas=tol, max_iter=4000)
              if solver == "CLARABEL" else dict(eps=tol, max_iters=200000))
        h["prob"].solve(solver=solver, verbose=False, **kw)
        if h["prob"].status not in ("optimal", "optimal_inaccurate"):
            out.append(dict(solver=solver, tol=tol, status=h["prob"].status))
            continue
        Gf = np.asarray(h["Gfree"].value)
        Xf = 0.5 * (np.asarray(h["Xfree"].value)
                    + np.asarray(h["Xfree"].value).T)
        w = np.linalg.eigvalsh(Xf - Gf.T @ Gf)[::-1]
        out.append(dict(solver=solver, tol=tol, status=h["prob"].status,
                        lam1=float(w[0]), lam2=float(w[1]),
                        ratio=float(w[1] / w[0]), cost=float(h["prob"].value)))
    return out


def main():
    d, k, ell = 3, 1, 0
    base = Segment(**hard_instances(d=d, k=k, l=ell)["three_in_a_row"])
    Gk0 = base.Gk.copy()
    lam = 0.25 * float(np.trace(Gk0)) / (d + 1)

    costs = {
        "derivative (control)": Gk0,
        "weighted 1+4s": weighted_gram(d, lambda s: 1.0 + 4.0 * s),
        "tikhonov +lambda I": Gk0 + lam * np.eye(d + 1),
        "mass (k=0)": mass_gram(d),
    }

    # A claim about "any K > 0" evidenced at one degree is evidence about that
    # degree.  The solve half runs at d = 3 and d = 5, with each cost rebuilt
    # for the degree rather than reused.
    rows = []
    for dd in (3, 5):
        cd = {"derivative (control)": gram_deriv(dd, k),
              "weighted 1+4s": weighted_gram(dd, lambda s: 1.0 + 4.0 * s),
              "tikhonov +lambda I": (gram_deriv(dd, k)
                                     + 0.25 * float(np.trace(gram_deriv(dd, k)))
                                     / (dd + 1) * np.eye(dd + 1)),
              "mass (k=0)": mass_gram(dd)}
        for label, Kfull in cd.items():
            for nm in ("symmetric_blocker", "staggered_pair", "three_in_a_row"):
                seg = Segment(**hard_instances(d=dd, k=k, l=ell)[nm])
                seg.Gk = Kfull
                r = analyse(seg)
                r.update(cost=label, instance=nm, d=dd)
                rows.append(r)
                print("    d=%d %-22s %-20s %s"
                      % (dd, label, nm,
                         ("rho=%d m=%d ker=%d unit=%d resid=%.1e"
                          % (r["rho"], r["m"], r["ker_dim"], r["n_unit"],
                             r["pencil_residual"]))
                         if r["status"] == "ok" else r["status"]))


    # ---- a40d: N >= 2, where the derivative cost CERTIFIES ---------------
    print("\n    multisegment, where the derivative cost certifies:")
    multi = []
    for N in (2, 3, 4):
        row = dict(N=N)
        for label, Kfull in costs.items():
            m, r = multiseg_bernstein(d, k, ell, N, 0, Kfull)
            row[label] = m
            row["r"] = r
        multi.append(row)
        print("      N=%d r=%d  " % (N, row["r"])
              + "  ".join("%s %+.3e" % (lab.split()[0], row[lab])
                          for lab in costs))

    # ---- a40e: three live contacts, so Perron-Frobenius is used ----------
    print("\n    three-contact instance (m = 3), where the m<=2 shortcut does not apply:")
    three = []
    for label, Kfull in costs.items():
        F = free_idx(d, ell)
        assert np.linalg.eigvalsh(Kfull[np.ix_(F, F)]).min() > 0, label
        seg = three_blocker_cfg(0.12, 0.60, d, k, ell)
        seg.Gk = Kfull
        r = analyse(seg)
        r.update(cost=label, instance="three_blocker s1=0.12")
        three.append(r)
        print("      %-22s %s"
              % (label, ("rho=%d m=%d ker=%d unit=%d resid=%.1e"
                         % (r["rho"], r["m"], r["ker_dim"], r["n_unit"],
                            r["pencil_residual"]))
                 if r["status"] == "ok" else r["status"]))
    rows += three

    # ---- a40g: the same costs in R^3, since n is claimed not to enter -----
    print("\n    the same costs in R^3 (n enters only through the rows of Gamma):")
    r3 = []
    for label, Kfull in costs.items():
        seg = three_blocker_cfg(0.12, 0.60, d, k, ell, n=3)
        seg.Gk = Kfull
        r = analyse(seg)
        r.update(cost=label, instance="three_blocker R^3", d=d, n=3)
        r3.append(r)
        print("      %-22s %s"
              % (label, ("rho=%d m=%d ker=%d unit=%d resid=%.1e"
                         % (r["rho"], r["m"], r["ker_dim"], r["n_unit"],
                            r["pencil_residual"]))
                 if r["status"] == "ok" else r["status"]))
    rows += r3

    # ---- a40f: the mass cost's rho = 2, read three ways -------------------
    print("\n    the mass cost's rho = 2, arbitrated as Section 4 requires:")
    seg = three_blocker_cfg(0.12, 0.60, d, k, ell)
    seg.Gk = costs["mass (k=0)"]
    arb = arbitrate(seg)
    for a in arb:
        print("      %-9s tol %.0e  lam2/lam1 = %.4g   cost %.9f"
              % (a["solver"], a["tol"], a.get("ratio", float("nan")),
                 a.get("cost", float("nan"))))

    ok = [r for r in rows if r["status"] == "ok"]
    new = [r for r in ok if r["cost"] != "derivative (control)"]
    tight = [r for r in rows if r["status"] == "tight"]
    a40a = dict(n=len(ok), n_new=len(new), n_costs=len(costs),
                degrees=sorted({r.get("d", d) for r in rows}),
                dims=sorted({r.get("n", 2) for r in rows}),
                n_posed=len(rows), n_tight=len(tight),
                tight_costs=sorted(r["cost"] for r in tight),
                n_thm6=sum(r["n_unit"] == r["ker_dim"] == r["rho"] for r in ok),
                worst_pencil_residual=max(r["pencil_residual"] for r in ok),
                passed=bool(new and all(r["n_unit"] == r["ker_dim"] == r["rho"]
                                        for r in ok)
                            and max(r["pencil_residual"] for r in ok) < 1e-3))
    a40b = dict(n=len(ok), n_bound=sum(r["rho"] <= r["m"] for r in ok),
                passed=bool(all(r["rho"] <= r["m"] for r in ok)))
    bern = {lab: bernstein_hypotheses(K, d, ell) for lab, K in costs.items()}
    a40c = dict(elevated_min=bern,
                n_costs=len(bern),
                n_negative=sum(v < 0 for v in bern.values()),
                n_costs_total=len(costs),
                passed=bool(len(bern) == len(costs)
                            and all(v < 0 for v in bern.values())))
    # a40d: the control must CERTIFY (>= 0) at every N, or the cell says nothing
    ctrl = "derivative (control)"
    tol = -1e-9
    a40d = dict(rows=multi, n=len(multi), tol=abs(tol),
                worst_abs=max(abs(r[lab]) for r in multi for lab in costs),
                n_control_certified=sum(r[ctrl] >= tol for r in multi),
                n_weighted_certified=sum(r["weighted 1+4s"] >= tol
                                         for r in multi),
                n_tikhonov_certified=sum(r["tikhonov +lambda I"] >= tol
                                         for r in multi),
                n_mass_certified=sum(r["mass (k=0)"] >= tol for r in multi),
                n_all_certified=sum(all(r[lab] >= tol for lab in costs)
                                    for r in multi),
                passed=bool(len(multi) == 3
                            and all(r[ctrl] >= tol for r in multi)))
    tok = [r for r in three if r["status"] == "ok"]
    a40e = dict(n=len(tok), max_m=max((r["m"] for r in tok), default=0),
                costs_reaching_m3=sorted(r["cost"] for r in tok if r["m"] >= 3),
                n_thm6=sum(r["n_unit"] == r["ker_dim"] == r["rho"]
                           for r in tok),
                worst_pencil_residual=max((r["pencil_residual"] for r in tok),
                                          default=None),
                passed=bool(len(tok) == len(costs)
                            and max(r["m"] for r in tok) >= 3
                            and all(r["n_unit"] == r["ker_dim"] == r["rho"]
                                    for r in tok)))
    r3ok = [r for r in r3 if r["status"] == "ok"]
    a40g = dict(n=len(r3ok), n_posed=len(r3),
                n_thm6=sum(r["n_unit"] == r["ker_dim"] == r["rho"]
                           for r in r3ok),
                passed=bool(r3ok and all(r["n_unit"] == r["ker_dim"] == r["rho"]
                                         for r in r3ok)))
    good = [a for a in arb if "ratio" in a]
    a40f = dict(readings=arb, n=len(good),
                ratio_spread=(max(a["ratio"] for a in good)
                              - min(a["ratio"] for a in good)) if good else None,
                ratio=min((a["ratio"] for a in good), default=None),
                cost_spread=(max(a["cost"] for a in good)
                             - min(a["cost"] for a in good)) if good else None,
                passed=bool(len(good) == 3
                            and max(a["ratio"] for a in good)
                            - min(a["ratio"] for a in good) < 1e-4
                            and min(a["ratio"] for a in good) > 0.09))
    with open(os.path.join(ART, "a40_second_lifting.json"), "w") as fh:
        json.dump(dict(gates=dict(a40a_pencil_survives=a40a,
                                  a40b_contact_bound_survives=a40b,
                                  a40c_bernstein_recheck=a40c,
                                  a40d_multiseg_transfer=a40d,
                                  a40e_three_contacts=a40e,
                                  a40f_mass_rho2_arbitrated=a40f,
                                  a40g_ambient_dimension=a40g),
                       rows=rows), fh, indent=1)
    print("\n  gates: a40a %s  a40b %s  a40c %s  a40d %s  a40e %s  a40f %s  "
          "a40g %s"
          % (a40a["passed"], a40b["passed"], a40c["passed"], a40d["passed"],
             a40e["passed"], a40f["passed"], a40g["passed"]))
    return 0 if all(g["passed"] for g in (a40a, a40b, a40c, a40d, a40e, a40f,
                                          a40g)) else 1


if __name__ == "__main__":
    sys.exit(main())
