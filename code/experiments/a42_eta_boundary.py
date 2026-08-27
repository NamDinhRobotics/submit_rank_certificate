"""The dichotomy is not "one polynomial versus knots".  The junction has to be
loose enough, and the paper's sweep never varied the axis that decides it.

Section 5 reports that a single polynomial fails and knotted curves certify,
42 configurations, `N = 1..6` over seven `(d, k, l)` settings.  Every one of
those 42 fixes `eta = max(0, k-1)`.  The welded control `eta = d` fails, and the
manuscript reads that as the extreme case.  It is not extreme: between `k-1` and
`d` there is a boundary, and it is much closer to `k-1` than to `d`.

WHAT THIS MEASURES

  * the safe region.  `eta <= 2k-1` certifies in every configuration tried.
  * that the bound cannot be raised.  `eta = 2k` fails in some configurations --
    3 of the 9 at `N = 3`, including the `C^2` cubic -- so `2k-1` is the largest
    uniform bound the data supports.  It does NOT fail everywhere at `2k`, which
    is why the safe region is stated as a sufficient condition and not as a
    characterisation.
  * that the failures are real.  A negative elevated Bernstein minimum only says
    the sufficient test failed; `Gr` is evaluated directly and returns a
    genuinely negative value, at the same opposite-ends geometry as the
    single-polynomial counterexample of Section 4.
  * that the failure set is NOT an interval in `eta`.  `d = 7, k = 1, N = 2`
    fails at `eta = 3` and certifies again at `eta = 4`, so no threshold theorem
    of the form "certified iff eta <= E(d,k,N)" can be true.
  * that the paper's own claims survive.  All four rows of Table I of [1] lie
    inside `eta <= 2k-1`, so nothing certified in the manuscript is affected.

WHY IT MATTERS OUTSIDE THIS PAPER.  A `C^2` cubic spline -- `d = 3`, `k = 1`,
`eta = 2`, the standard smoothness choice in trajectory generation -- sits
OUTSIDE the safe region, and `Gr` is negative for it.  The a priori route is
closed for the most ordinary spline in the field, and only a per-configuration
check finds that out.

Gates:
  a42a  eta <= 2k-1 certifies everywhere tried, and the count of configurations
  a42b  eta = 2k fails somewhere, so 2k-1 cannot be raised
  a42c  the failures are failures of `Gr`, not of the test: a directly evaluated
        negative value for the C^2 cubic spline
  a42d  the failure set is not an interval in eta
  a42e  Table I of [1] lies inside the safe region and still certifies

Run:  python experiments/a42_eta_boundary.py
"""
import json
import os
import sys
from math import comb

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
ART = os.path.join(ROOT, "artifacts")
sys.path.insert(0, os.path.join(ROOT, "src"))

from exact_green import certify_exact, coefficient_blocks, energy_nullity  # noqa: E402

CONFIGS = ((3, 1, 0), (5, 1, 0), (7, 1, 0), (9, 1, 0),
           (5, 2, 1), (7, 2, 1), (9, 2, 1), (7, 3, 2), (9, 3, 2))
NS = (2, 3, 4, 5)
D_ELEV = 64
TABLE1 = (("velocity", 1, 1, 0, 12, 0), ("acceleration", 2, 3, 2, 6, 2),
          ("jerk", 3, 5, 3, 4, 3), ("snap", 4, 7, 4, 3, 4))


def verdict(d, k, ell, N, eta):
    if energy_nullity(d, k, ell, N, eta)["nullity"] != 0:
        return None
    try:
        return certify_exact(d, k, ell, N, eta, D=D_ELEV)
    except Exception:                                        # noqa: BLE001
        return None


def gr_direct_min(d, k, ell, N, eta, ns=801):
    """The smallest value `Gr` actually takes, relative to its largest.

    The elevated test is one-sided, so a negative minimum there proves nothing.
    This evaluates the function.
    """
    blocks, _ = coefficient_blocks(d, k, ell, N, eta)
    bb = np.array([[comb(d, a) * s ** a * (1 - s) ** (d - a)
                    for a in range(d + 1)]
                   for s in np.linspace(0.0, 1.0, ns)])
    ss = np.linspace(0.0, 1.0, ns)
    worst, scale, where = np.inf, 0.0, None
    for (i, j), Bm in blocks.items():
        Bn = np.array([[float(x) for x in row] for row in Bm])
        Gm = bb @ Bn @ bb.T
        scale = max(scale, float(np.abs(Gm).max()))
        if float(Gm.min()) < worst:
            worst = float(Gm.min())
            a, b = np.unravel_index(np.argmin(Gm), Gm.shape)
            where = (int(i), int(j), float(ss[a]), float(ss[b]))
    return worst / max(scale, 1e-300), where


def main():
    safe, unsafe_2k, rows = [], [], []
    for (d, k, ell) in CONFIGS:
        for N in NS:
            certified = []
            for eta in range(0, d + 1):
                r = verdict(d, k, ell, N, eta)
                if r is None:
                    continue
                rows.append(dict(d=d, k=k, l=ell, N=N, eta=eta,
                                 nonneg=bool(r["nonneg"]),
                                 elev=float(r["elev_min_rel"])))
                if r["nonneg"]:
                    certified.append(eta)
                if eta <= 2 * k - 1:
                    safe.append((d, k, N, eta, bool(r["nonneg"])))
                if eta == 2 * k:
                    unsafe_2k.append((d, k, N, bool(r["nonneg"])))
            print("    d=%d k=%d N=%d : certified eta = %s"
                  % (d, k, N, certified))

    n_safe = len(safe)
    safe_fail = [s for s in safe if not s[4]]
    print("\n    eta <= 2k-1 : %d configurations, %d failures"
          % (n_safe, len(safe_fail)))
    at2k_N3 = [u for u in unsafe_2k if u[2] == 3]
    n_fail_2k = sum(1 for u in at2k_N3 if not u[3])
    print("    eta =  2k   : fails at N=3 in %d of %d cases, so 2k-1 is the "
          "largest uniform bound" % (n_fail_2k, len(at2k_N3)))

    g, where = gr_direct_min(3, 1, 0, 3, 2)
    print("    C^2 cubic spline (d=3,k=1,N=3,eta=2): min Gr = %.3e relative, "
          "at segments (%d,%d), s=%.3f, t=%.3f" % (g, where[0], where[1],
                                                   where[2], where[3]))

    # non-monotone: a configuration that fails and then certifies again
    gap = None
    for (d, k, ell) in CONFIGS:
        for N in NS:
            cert = [r["eta"] for r in rows
                    if (r["d"], r["k"], r["N"]) == (d, k, N) and r["nonneg"]]
            fail = [r["eta"] for r in rows
                    if (r["d"], r["k"], r["N"]) == (d, k, N) and not r["nonneg"]]
            for e in fail:
                if any(c > e for c in cert):
                    gap = dict(d=d, k=k, N=N, fails_at=e,
                               certifies_again_at=min(c for c in cert if c > e))
                    break
            if gap:
                break
        if gap:
            break
    print("    not an interval: %s" % gap)

    t1 = []
    for nm, k, d, ell, N, eta in TABLE1:
        r = certify_exact(d, k, ell, N, eta, D=96)
        t1.append(dict(name=nm, k=k, d=d, eta=eta, inside=bool(eta <= 2 * k - 1),
                       certified=bool(r["nonneg"])))
    print("    Table I of [1]: inside the safe region and certified: %s"
          % all(x["inside"] and x["certified"] for x in t1))

    a42a = dict(n=n_safe, n_fail=len(safe_fail), passed=bool(not safe_fail))
    a42b = dict(n=len(at2k_N3), n_fail=n_fail_2k,
                passed=bool(n_fail_2k > 0))
    a42c = dict(min_rel=g, where=where,
                passed=bool(g < -1e-6 and where[0] != where[1]))
    a42d = dict(gap=gap, passed=bool(gap is not None))
    a42e = dict(rows=t1, passed=bool(all(x["inside"] and x["certified"]
                                         for x in t1)))
    with open(os.path.join(ART, "a42_eta_boundary.json"), "w") as fh:
        json.dump(dict(gates=dict(a42a_safe_region=a42a,
                                  a42b_boundary_tight=a42b,
                                  a42c_Gr_really_negative=a42c,
                                  a42d_not_an_interval=a42d,
                                  a42e_table1_unaffected=a42e),
                       rows=rows), fh, indent=1)
    print("\n  gates: a42a %s  a42b %s  a42c %s  a42d %s  a42e %s"
          % (a42a["passed"], a42b["passed"], a42c["passed"], a42d["passed"],
             a42e["passed"]))
    return 0 if all(g_["passed"] for g_ in (a42a, a42b, a42c, a42d,
                                            a42e)) else 1


if __name__ == "__main__":
    sys.exit(main())
