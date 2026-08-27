"""(H3), exercised.  Section 6 varies the cost and says so; this varies the
constraint family, which is the hypothesis the cost cannot reach.

Section 6 separates what the argument needs -- (H1) one PSD block, (H2) a
definite reduced cost, (H3) every constraint reaching `X_F` through ONE
nonnegative rank-one family `u(s)^T X_F u(s)` -- from what is particular to
Bezier curves and ball obstacles.  Its experiment replaces `K`, which tests
(H2).  `u` never moves, so (H1) and (H3) stay untested, and Section 6 records that.

A FLEET MOVES `u`.  Stack `R` robots into one lift.  A robot-obstacle contact
on robot `i` still reads `X_F` through `e_i (x) b(s)|_F`; a robot-robot contact
between `i` and `k` reads it through `(e_i - e_k) (x) b(s)|_F`.  Same shape,
different map -- exactly what (H3) quantifies over, and a constraint no change
of cost can produce.  It is also a robot problem rather than a curve problem:
several vehicles that must avoid the obstacles and each other.

WHAT SHOULD HAPPEN, IF SECTION 6 IS RIGHT

  structural   the pencil `Z = K - M_lambda`, Theorem 8's unit-eigenvalue
               count, Theorem 7's `rho <= m`: all should survive, because none
               of their proofs asks what `u` is.
  particular   the a priori route should NOT survive, and for a reason that has
               nothing to do with the one in Section 6.  There the Green kernel
               lost its sign because the cost operator changed.  Here `K` is
               block diagonal over robots and the Green matrix factors
               entrywise as `Gr = L * B`, with `B` the single-robot kernel and
               `L` the Gram of the contacts' incidence vectors.  `L` carries
               `-1` whenever two contacts share a robot with opposite sign, so
               `Gr` has negative entries even where `B` is strictly positive --
               even on a spline family the single-robot certificate certifies.

So the a priori route closes twice over, analytically in Section 6 and
combinatorially here, while the pencil does not notice either.  The
combinatorial side is analysed in a companion study; this file claims only what
it measures, which is that (H3) is where the split lives.

Gates:
  a43a  the pencil survives the new `u`: residual, and Theorem 8's count equal
        to `dim ker Z` and to `rho` on every instance that is loose
  a43b  Theorem 7's bound `rho <= m` survives
  a43c  `Gr = L * B` entrywise, to numerical precision -- the factorisation, not
        just the negativity
  a43d  `Gr` has negative entries wherever a contact pattern makes `L` carry a
        `-1`, so the a priori route is closed by the contact graph

Run:  python experiments/a43_h3_multirobot.py
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

from multirobot import MultiRobot                                # noqa: E402
from bernstein import bd, gram_deriv                             # noqa: E402
from a22_multirobot_rank import (instances, incidence_gram,      # noqa: E402
                                 realised_contacts)

D, K_ORDER, ELL, NDIM = 3, 1, 0, 2


def free_idx(d, ell):
    return list(range(ell + 1, d - ell))


def contact_vectors(mr, res, contacts, tol=1e-4):
    """`u_a` for each realised contact, in the STACKED free coordinates.

    Obstacle contact on robot `i`  ->  `e_i (x) b(s)|_F`.
    Pair contact between `i`, `k`  ->  `(e_i - e_k) (x) b(s)|_F`.
    The parameter `s` is where the corresponding clearance is minimised.
    """
    Fi = free_idx(mr.d, mr.l)
    ss = np.linspace(0.0, 1.0, 2001)
    Bs = np.array([bd(s, mr.d) for s in ss])
    pts = [G @ Bs.T for G in res["Gamma"]]
    out = []
    for c in contacts:
        if c[0] == "obs":
            i, jj = c[1], c[2]
            cen, rad = mr.obs[i][jj]
            v = np.sum((pts[i] - cen[:, None]) ** 2, axis=0) - rad ** 2
            a = int(np.argmin(v))
            inc = np.zeros(mr.R)
            inc[i] = 1.0
        else:
            i, kk = c[1], c[2]
            v = np.sum((pts[i] - pts[kk]) ** 2, axis=0) - (2 * mr.rr) ** 2
            a = int(np.argmin(v))
            inc = np.zeros(mr.R)
            inc[i], inc[kk] = 1.0, -1.0
        u = np.kron(inc, Bs[a][Fi])
        out.append(dict(u=u, s=float(ss[a]), inc=inc, kind=c[0]))
    return out


def analyse(name, kw):
    from scipy.optimize import nnls
    mr = MultiRobot(n=NDIM, d=D, k=K_ORDER, l=ELL, **kw)
    c = mr.build()
    res = mr.solve()
    if not res.get("converged"):
        return dict(name=name, status=res.get("status", "?"))
    Zfull = np.asarray(c["cons"][0].dual_value if "cons" in c
                       else c["prob"].constraints[0].dual_value)
    Zfull = 0.5 * (Zfull + Zfull.T)
    Z = Zfull[mr.n:, mr.n:]
    Fi = free_idx(D, ELL)
    G1 = gram_deriv(D, K_ORDER)[np.ix_(Fi, Fi)]
    K = np.kron(np.eye(mr.R), G1)                 # block diagonal over robots
    rho = int(res["rho"])
    contacts = realised_contacts(mr, res)
    if not contacts:
        return dict(name=name, status="no contact", rho=rho)
    cv = contact_vectors(mr, res, contacts)
    U = np.array([x["u"] for x in cv]).T
    M = K - Z
    A = np.array([np.outer(U[:, a], U[:, a]).ravel()
                  for a in range(U.shape[1])]).T
    w, _ = nnls(A, M.ravel())
    resid = float(np.linalg.norm(A @ w - M.ravel())
                  / max(1e-300, np.linalg.norm(M.ravel())))
    Ki = np.linalg.inv(K)
    Gr = U.T @ Ki @ U
    Wh = np.diag(np.sqrt(np.maximum(w, 0.0)))
    mu = np.sort(np.linalg.eigvalsh(Wh @ Gr @ Wh))[::-1]
    n_unit = int(np.sum(np.abs(mu - 1.0) < 1e-4))
    wZ = np.linalg.eigvalsh(Z)
    ker = int(np.sum(np.abs(wZ) < 1e-6 * max(1.0, float(np.abs(wZ).max()))))

    # the factorisation Gr = L * B, entrywise
    L = incidence_gram(mr.R, contacts)
    Bk = np.array([[float(bd(x["s"], D)[Fi] @ np.linalg.inv(G1)
                          @ bd(y["s"], D)[Fi]) for y in cv] for x in cv])
    fac = float(np.max(np.abs(Gr - L * Bk))
                / max(1e-300, float(np.max(np.abs(Gr)))))
    return dict(name=name, status="ok", rho=rho, m=int(U.shape[1]),
                Gamma=[np.asarray(G, float).tolist() for G in res["Gamma"]],
                obstacles=[[[list(map(float, c)), float(r)] for c, r in ob]
                           for ob in mr.obs],
                robot_radius=float(mr.rr),
                contact_s=[x["s"] for x in cv],
                ker_dim=ker, n_unit=n_unit, pencil_residual=resid,
                R=int(mr.R), kinds=[x["kind"] for x in cv],
                gr_min=float(Gr.min()), b_min=float(Bk.min()),
                L_has_minus=bool((L < -0.5).any()), factor_residual=fac)


def main():
    rows = []
    for name, kw, _why in instances():
        try:
            r = analyse(name, kw)
        except Exception as exc:                                 # noqa: BLE001
            r = dict(name=name, status="error: %s" % type(exc).__name__)
        rows.append(r)
        if r["status"] == "ok":
            print("    %-11s R=%d rho=%d m=%d ker=%d unit=%d resid=%.1e  "
                  "min Gr=%+.3f  min B=%+.3f  L has -1: %s  Gr=L*B to %.1e"
                  % (r["name"], r["R"], r["rho"], r["m"], r["ker_dim"],
                     r["n_unit"], r["pencil_residual"], r["gr_min"],
                     r["b_min"], r["L_has_minus"], r["factor_residual"]))
        else:
            print("    %-11s %s" % (r["name"], r["status"]))

    ok = [r for r in rows if r["status"] == "ok" and r["rho"] >= 1]
    a43a = dict(n=len(ok), n_thm8=sum(r["n_unit"] == r["ker_dim"] == r["rho"]
                                      for r in ok),
                worst_residual=max((r["pencil_residual"] for r in ok),
                                   default=None),
                passed=bool(ok and all(r["n_unit"] == r["ker_dim"] == r["rho"]
                                       for r in ok)
                            and max(r["pencil_residual"] for r in ok) < 1e-3))
    a43b = dict(n=len(ok), n_bound=sum(r["rho"] <= r["m"] for r in ok),
                passed=bool(ok and all(r["rho"] <= r["m"] for r in ok)))
    a43c = dict(n=len(ok),
                worst=max((r["factor_residual"] for r in ok), default=None),
                passed=bool(ok and max(r["factor_residual"]
                                       for r in ok) < 1e-6))
    neg = [r for r in ok if r["L_has_minus"]]
    a43d = dict(n_with_minus=len(neg),
                n_negative_Gr=sum(1 for r in neg if r["gr_min"] < -1e-9),
                n_B_positive=sum(1 for r in neg if r["b_min"] > 0),
                passed=bool(neg and all(r["gr_min"] < -1e-9 for r in neg)))
    with open(os.path.join(ART, "a43_h3_multirobot.json"), "w") as fh:
        json.dump(dict(gates=dict(a43a_pencil_survives_new_u=a43a,
                                  a43b_contact_bound_survives=a43b,
                                  a43c_factorisation=a43c,
                                  a43d_contact_graph_closes_it=a43d),
                       rows=rows), fh, indent=1)
    print("\n  gates: a43a %s  a43b %s  a43c %s  a43d %s"
          % (a43a["passed"], a43b["passed"], a43c["passed"], a43d["passed"]))
    return 0 if all(g["passed"] for g in (a43a, a43b, a43c, a43d)) else 1


if __name__ == "__main__":
    sys.exit(main())
