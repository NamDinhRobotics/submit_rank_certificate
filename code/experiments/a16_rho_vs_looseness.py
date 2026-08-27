"""A16 -- what `rho` is worth operationally, and where reading it misleads.

`rho` read at the maximum-rank point of the optimal face does not by itself
establish that the relaxation is loose: the face may also carry a lower-rank
point, and the value may be exact there.  That is a logical caveat; this phase
measures it, and the measurement is more useful than the caveat.

For each instance the optimal value is bracketed from both sides:

    c*_SDP                        a valid lower bound
    a STRICTLY feasible curve     an upper bound, clearance >= eps > 0

The gap between them bounds the true looseness, without appealing to a local
solver's claim to have found the optimum.  A strictly feasible point is used
rather than a boundary one because a point that merely touches the obstacle can
report a cost slightly BELOW the true optimum, which then appears to violate the
lower bound.

  a16a  `rho = 0` means the value is exact.  Not "close": the bracket closes to
        solver tolerance on every such instance.  This is what makes `rho` worth
        reading at all -- it is a usable certificate of exactness.
  a16b  `rho = 1` usually IS loose, by a few percent.  So the deficiency is not
        cosmetic, and the certificate that bounds it is not measuring nothing.
  a16c  THE LIMIT, and the reason a16 exists: `rho = 1` does NOT imply loose.  A
        witness is exhibited where the maximum-rank reading is `rho = 1` and the
        bracket still closes -- so the optimal face carries a lower-rank point
        and the interior-point method simply did not land on it.  Anyone reading
        `rho` as a looseness score would be wrong on this instance.

Writes artifacts/a16_rho_vs_looseness.json.
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
ART = os.path.join(HERE, "..", "artifacts")

from relaxation import Segment                               # noqa: E402
from bernstein import bd, gram_deriv                         # noqa: E402
from a4_simplicity_margin import analyse                     # noqa: E402

A, B = np.array([-2.0, 0.0]), np.array([2.0, 0.0])
D, K, L = 5, 2, 1                       # f = 2: two free control points
SEED = 11
N_WANT = 40
EPS = 1e-6                              # the upper bound must be STRICTLY feasible

_GK = gram_deriv(D, K)
_BB = np.array([bd(s, D) for s in np.linspace(0.0, 1.0, 801)])


def _pack(p):
    return np.array([[-2, -2, p[0], p[2], 2, 2],
                     [0, 0, p[1], p[3], 0, 0]], float)


def _cost(x):
    G = _pack(x)
    return float(np.trace(G @ _GK @ G.T))


def _clearance(x, obs):
    P = _BB @ _pack(x).T
    return min(np.linalg.norm(P - c, axis=1).min() - r for c, r in obs)


def upper_bound(obs, eps=EPS):
    """Cheapest STRICTLY feasible curve we can find, from lifted starts.

    Lifted starts, not the straight line: the straight line sits inside the
    obstacles, and a local solver started there fails on about an eighth of this
    population -- a property of the initial guess, not of the method, and not
    something this phase is measuring.
    """
    best = None
    for y in (1.0, 2.0, -1.0, -2.0):
        r = minimize(_cost, np.array([-0.9, y, 0.9, y], float), method="SLSQP",
                     constraints=[{"type": "ineq",
                                   "fun": lambda x: _clearance(x, obs) - eps}],
                     options=dict(maxiter=400, ftol=1e-15))
        if _clearance(r.x, obs) >= 0.5 * eps and (best is None or r.fun < best[0]):
            best = (float(r.fun), float(_clearance(r.x, obs)),
                    [float(v) for v in r.x])
    return best


def population(n_want=N_WANT, seed=SEED):
    """Two disjoint spheres, both cutting the straight line from A to B."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(600):
        if len(rows) >= n_want:
            break
        cx = rng.uniform(-1.35, 1.35, 2)
        cy = rng.uniform(-0.3, 0.3, 2)
        rr = rng.uniform(0.35, 0.62, 2)
        if abs(cx[0] - cx[1]) < rr[0] + rr[1]:
            continue
        if min(abs(cy[j]) - rr[j] for j in range(2)) > 0:
            continue                     # must actually block the straight line
        obs = [(np.array([cx[j], cy[j]]), float(rr[j])) for j in range(2)]
        try:
            seg = Segment(n=2, d=D, k=K, l=L, obstacles=obs,
                          bc0=[A.tolist(), [0.0, 0.0]],
                          bc1=[B.tolist(), [0.0, 0.0]])
            res = seg.solve(backend="cvxpy", solver="CLARABEL")
            if res.get("cost") is None:
                continue
            r = analyse(seg, ns=2001)
        except Exception:                                    # noqa: BLE001
            continue
        ub = upper_bound(obs)
        if ub is None:
            continue
        rho = r.get("rho")
        rows.append(dict(
            obstacles=[[float(c[0]), float(c[1]), float(rad)] for c, rad in obs],
            c_sdp=float(res["cost"]), rho=rho, m=r.get("m"),
            margin=None if r.get("margin") is None else float(r["margin"]),
            c_upper=ub[0], upper_clearance=ub[1], upper_x=ub[2],
            rel_gap=(ub[0] - float(res["cost"])) / max(ub[0], 1e-12),
            proj_clearance=float(seg.min_clearance(res["Gamma"])),
            Gamma=[[float(v) for v in row] for row in res["Gamma"]]))
    return rows


def gate_a16a(rows):
    """`rho = 0` (reported as a tight solve) must mean the value is exact."""
    z = [x for x in rows if x["rho"] in (0, None)]
    worst = max(abs(x["rel_gap"]) for x in z) if z else None
    print("    rho = 0 on %d instances; bracket closes to <= %.1e relative"
          % (len(z), worst))
    return dict(n=len(z), worst_abs_rel_gap=worst,
                median_rel_gap=float(np.median([x["rel_gap"] for x in z])),
                passed=bool(z and worst < 1e-3))


def gate_a16b(rows):
    """`rho = 1` is usually genuinely loose, so the bound is not vacuous."""
    o = [x for x in rows if x["rho"] == 1]
    g = [x["rel_gap"] for x in o]
    print("    rho = 1 on %d instances; relative looseness median %.2e, max %.2e"
          % (len(o), float(np.median(g)), max(g)))
    return dict(n=len(o), median_rel_gap=float(np.median(g)),
                max_rel_gap=float(max(g)), min_rel_gap=float(min(g)),
                n_loose_above_1pct=sum(1 for v in g if v > 1e-2),
                passed=bool(o and np.median(g) > 1e-2))


def gate_a16c(rows):
    """THE LIMIT.  A witness with `rho = 1` whose value is nonetheless tight.

    Without this the paper's caveat about the maximum-rank reading would be a
    logical possibility with no instance behind it; with it, reading `rho` as a
    looseness score is demonstrably wrong somewhere.
    """
    o = [x for x in rows if x["rho"] == 1]
    tight = sorted(o, key=lambda x: x["rel_gap"])[:1]
    if not tight:
        print("    no rho = 1 instance found")
        return dict(n_rho1=len(o), passed=False)
    w = tight[0]
    print("    witness: rho = 1, m = %s, margin %.4f, yet the bracket closes to "
          "%.1e relative" % (w["m"], w["margin"], w["rel_gap"]))
    print("      c*_SDP = %.5f   strictly feasible upper = %.5f (clearance %+.1e)"
          % (w["c_sdp"], w["c_upper"], w["upper_clearance"]))
    print("      the projection into R^2 cuts through by %+.4f"
          % w["proj_clearance"])
    return dict(n_rho1=len(o),
                n_tight_below_1e3=sum(1 for x in o if x["rel_gap"] < 1e-3),
                witness=w,
                passed=bool(w["rel_gap"] < 1e-3 and w["proj_clearance"] < 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ART, "a16_rho_vs_looseness.json"))
    a = ap.parse_args()

    print("\nbracketing %d two-sphere instances from both sides" % N_WANT)
    rows = population()
    print("    %d instances solved and bracketed" % len(rows))

    print("\n[a16a] rho = 0 means the value is exact")
    a16a = gate_a16a(rows)
    print("\n[a16b] rho = 1 is usually loose, by a few percent")
    a16b = gate_a16b(rows)
    print("\n[a16c] LIMIT: rho = 1 does not imply loose")
    a16c = gate_a16c(rows)

    gates = dict(a16a_rho0_is_exact=a16a, a16b_rho1_usually_loose=a16b,
                 a16c_rho1_not_always_loose=a16c)
    print("\n  --- gates ---")
    for nm, g in gates.items():
        print("  %s: %s" % (nm, "PASS" if g["passed"] else "FAIL"))
    with open(a.out, "w") as fh:
        json.dump(dict(gates=gates, rows=rows, seed=SEED, eps=EPS,
                       config=dict(d=D, k=K, l=L)), fh, indent=1, default=float)
    print("\nwrote %s" % a.out)
    if not all(g["passed"] for g in gates.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
