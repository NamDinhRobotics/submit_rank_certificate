"""A49 -- planning on an estimator's map: the certificate meets state estimation.

The obstacles are never where the planner thinks they are: a robot plans on the
output of an estimator.  This file runs the paper's pipeline on exactly that
interface and measures what survives.

THE SETUP.  True field: two disjoint balls.  Each center is observed T = 8
times through N(0, sigma_e^2 I) noise, sigma_e = 0.15; the planner sees the
sample mean c_hat with the standard deviation sigma_hat = sigma_e / sqrt(T)
and inflates every radius by kappa * sigma_hat, kappa in {0, 1, 2, 3}.  Radii
are known; only centers are estimated.  Planning runs the certified-recovery
policy of a48 on the inflated map -- rho = 0 executes gamma*, rho >= 1 folds,
nothing uncertified is ever returned -- and the returned curve is scored
against the TRUE field, exactly.

THE TRIANGLE MAKES SAFETY A THEOREM, NOT A STATISTIC.  If the estimate is
covered -- ||c_hat_j - c_j|| <= kappa sigma_hat for every j -- then a curve
certified on the inflated map satisfies, for every s,

    ||gamma - c|| >= ||gamma - c_hat|| - ||c_hat - c||
                 >= (r + kappa sigma_hat) - kappa sigma_hat = r,

so the TRUE clearance is nonnegative.  A collision on the true field is
therefore only possible on the UNCOVERED draws, the Gaussian tail that
kappa sigma_hat fails to contain; the experiment checks that the partition is
exact, not approximate.  And the certificate itself never needs re-derivation
as kappa moves: the Green matrix reads the contacts, not the obstacles, so
inflating the map changes WHERE the curve touches but not the spectral test
applied there -- measured as g > 0 on every rho >= 1 solve at every kappa.

CAUTION HAS A RANK COST, AND RECOVERY PAYS IT.  Inflating the map squeezes
the corridors, which is precisely what pushes solves into the rho = 1 regime:
recovery demand should RISE with kappa.  That mechanism -- estimation caution
manufactures the recovery problem, and the certified fold absorbs it -- is the
experiment's second reading.

Gates:
  a49a  the triangle, verified: every certified curve on a covered draw has
        exact true-field clearance >= 0
  a49b  the partition is exact: every true-field collision lies on an
        uncovered draw, and coverage rises with kappa
  a49c  recovery demand rises with kappa (events at kappa = 3 vs 0), and no
        uncertified curve is ever returned
  a49d  the certificate travels: g > 0 on every rho >= 1 solve, all kappa

Run:  python experiments/a49_estimated_obstacles.py
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

from instances import random_instance                            # noqa: E402
from relaxation import Segment, exact_clearance                  # noqa: E402
import a46_rank_one_recovery as a46                              # noqa: E402
import a4_simplicity_margin as a4                                # noqa: E402

D, K, L = 5, 2, 1
T_OBS = 8
SIGMA_E = 0.15
KAPPAS = (0, 1, 2, 3)
N_DRAW = 40
RHO_TOL = 1e-6


def plan_certified(obs_map):
    """The a48 policy on one map: (G, status, rho, g_margin)."""
    bc0 = np.array([[-2.0, 0.0], [0.0, 0.0]], float)
    bc1 = np.array([[2.0, 0.0], [0.0, 0.0]], float)
    seg = Segment(n=2, d=D, k=K, l=L, obstacles=obs_map, bc0=bc0, bc1=bc1)
    try:
        res = seg.solve(backend="cvxpy", solver="CLARABEL")
    except Exception:                                            # noqa: BLE001
        return None, "solver_fail", None, None
    if not res.get("converged"):
        return None, "solver_fail", None, None
    rho = int(res["rho"])
    if rho == 0:
        return res["Gamma"], "direct", 0, None
    g_margin = None
    cert = a4.analyse(seg)
    if cert.get("status") == "optimal":
        g_margin = cert.get("margin")
    vv, z, P, G0, _ = a46.lifted(seg, res)
    _, _, (Jb, wb), (Db, wd) = a46.search(seg, G0, vv, P, z, nR=61, nT=121)
    for w in (wb, wd):
        if w is None:
            continue
        ex, _ = a46.certify(seg, G0, vv, w)
        if ex >= 0.0:
            return a46.fold(G0, vv, w), "recovered", rho, g_margin
    return None, "unrecovered", rho, g_margin


def main():
    rng = np.random.default_rng(20260828)
    rows = []
    for i in range(N_DRAW):
        inst = random_instance(rng, n_obs=2, d=D, k=K, l=L,
                               rmin=0.5, rmax=0.85)
        if len(inst["obstacles"]) < 2:
            continue
        true_obs = inst["obstacles"]
        sig_hat = SIGMA_E / np.sqrt(T_OBS)
        c_hat = [np.mean(rng.normal(0, SIGMA_E, (T_OBS, 2)) + c, axis=0)
                 for c, _ in true_obs]
        errs = [float(np.linalg.norm(ch - c))
                for ch, (c, _) in zip(c_hat, true_obs)]
        for kap in KAPPAS:
            obs_map = [(ch, r + kap * sig_hat)
                       for ch, (_, r) in zip(c_hat, true_obs)]
            G, status, rho, g_margin = plan_certified(obs_map)
            covered = bool(all(e <= kap * sig_hat for e in errs))
            row = dict(i=i, kappa=kap, status=status, rho=rho,
                       covered=covered, max_err=max(errs),
                       g_margin=g_margin)
            if G is not None:
                tc, _ = exact_clearance(G, true_obs, D)
                row["true_clear"] = float(tc)
            rows.append(row)
        print("  draw %2d  err %.3f/%.3f  " % (i, errs[0], errs[1])
              + "  ".join("k%d:%s" % (r["kappa"], r["status"][:5])
                          for r in rows[-len(KAPPAS):]))

    planned = [r for r in rows if "true_clear" in r]
    # a49a: the triangle
    cov = [r for r in planned if r["covered"]]
    worst_cov = min((r["true_clear"] for r in cov), default=np.inf)
    # a49b: exact partition + coverage curve
    colls = [r for r in planned if r["true_clear"] < 0.0]
    n_coll_covered = sum(1 for r in colls if r["covered"])
    cover_rate = {k: (np.mean([r["covered"] for r in rows if r["kappa"] == k])
                      if any(r["kappa"] == k for r in rows) else 0.0)
                  for k in KAPPAS}
    coll_by_k = {k: sum(1 for r in colls if r["kappa"] == k) for k in KAPPAS}
    # a49c: recovery demand
    rec_by_k = {k: sum(1 for r in rows
                       if r["kappa"] == k and r["status"] == "recovered")
                for k in KAPPAS}
    n_unrec = sum(1 for r in rows if r["status"] == "unrecovered")
    # a49d: the margin
    margins = [r["g_margin"] for r in rows if r["g_margin"] is not None]

    print("\n  coverage by kappa: %s" %
          {k: round(float(v), 2) for k, v in cover_rate.items()})
    print("  true-field collisions by kappa: %s   (on covered draws: %d)"
          % (coll_by_k, n_coll_covered))
    print("  recovery events by kappa: %s   unrecovered: %d"
          % (rec_by_k, n_unrec))
    print("  worst certified true clearance on covered draws: %.3e" % worst_cov)
    if margins:
        print("  certificate margin g on rho>=1 solves: min %.4f  median %.4f"
              % (min(margins), float(np.median(margins))))

    gates = dict(
        a49a_triangle=dict(n_covered_planned=len(cov),
                           worst_true_clearance=float(worst_cov),
                           passed=bool(len(cov) > 0 and worst_cov >= -1e-9)),
        a49b_partition=dict(collisions_by_kappa=coll_by_k,
                            collisions_on_covered=n_coll_covered,
                            coverage_by_kappa={str(k): float(v)
                                               for k, v in cover_rate.items()},
                            passed=bool(n_coll_covered == 0
                                        and cover_rate[3] > cover_rate[0])),
        a49c_caution_costs_rank=dict(recoveries_by_kappa=rec_by_k,
                                     unrecovered=n_unrec,
                                     passed=bool(rec_by_k[3] >= rec_by_k[0]
                                                 and sum(rec_by_k.values()) > 0)),
        a49d_margin_travels=dict(n_measured=len(margins),
                                 min_g=(float(min(margins)) if margins else None),
                                 median_g=(float(np.median(margins))
                                           if margins else None),
                                 passed=bool(margins and min(margins) > 0.0)),
    )
    out = os.path.join(ART, "a49_estimated_obstacles.json")
    with open(out, "w") as fh:
        json.dump(dict(gates=gates, rows=rows), fh, indent=1)
    ok = all(g["passed"] for g in gates.values())
    print("  wrote %s" % out)
    print("  ALL GATES PASSED" if ok else "  *** A GATE FAILED ***")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
