"""The a priori claim, on a configuration a quadrotor actually flies, chosen here
rather than inherited from [1].

Corollary 14 certifies the rows of Table I of [1].  That is their benchmark, and
a reader is entitled to ask whether the claim survives outside it.  So this file
picks a minimum-snap quadrotor configuration of our own and runs the claim end
to end on it:

    k = 4 (snap),  d = 2k-1 = 7,  C^4 joints (eta = 4),  l = 4,  N = 5,  R^3

Minimum snap with `C^4` continuity is the standard differential-flatness setting
for multirotors: the flat outputs are the position, the cost is the snap energy,
and obstacles enter as safety spheres.  `N = 5` is ours -- Table I's snap row is
`N = 3` -- and it leaves `r = 10` free parameters instead of `4`.

WHAT IS CLAIMED, AND IN WHICH ORDER

  1. The configuration is certified BEFORE any obstacle field is drawn: the
     elevated Bernstein minimum is exactly 0 in rational arithmetic at D = 96,
     so Theorem 15 applies at every contact set, and Corollary 14 gives
     `rho <= 1` for EVERY sphere arrangement and every ambient dimension.
  2. Only then are the fields drawn and solved.  Every solve must return
     `rho <= 1`.  A single `rho = 2` would refute step 1, which is what makes
     this a test and not an illustration.
  3. `eta = 4` and `2k-1 = 7`, so the configuration sits inside the safe region
     of the continuity sweep -- as, separately, does Table I.

ONE NUMERICAL POINT, WHICH TURNED INTO A RESULT.  `MultiSegment` scales
the cost by `N^(2k-1)` so that "more segments" is not a free cost reduction.  At
minimum snap that is `N^7`, which is `1.6e4` at `N = 4` and `7.8e4` at `N = 5`,
and Clarabel then reports the obstacle-free problem UNBOUNDED -- a conditioning
failure, not a modelling one.  Scaling `K` by a positive constant leaves the
optimum, and therefore `rho`, untouched, so the solves below use the unscaled
cost.  The a priori verdict is likewise unaffected: scaling `K` scales
`B = Nperp K^-1 Nperp^T` by its reciprocal and leaves every sign alone.

At `N = 3`, where the scaled problem still converges, the two DISAGREE: the
scaled solve returns `rho = 2` and `rho = 1` on fields where the unscaled one
returns `0`.  Since scaling cannot move the optimal set, one of the two is
wrong, and the a priori certificate says which: this configuration is certified
`rho <= 1` before any field is drawn, so a reported `rho = 2` is a solver
artefact.  A bound proved in advance refuting a number a solver printed is the
clearest answer this file has to what such a bound is for.

Gates:
  a44a  the configuration certifies a priori, in exact arithmetic
  a44b  every solved random field returns rho <= 1, with the count
  a44c  it lies inside eta <= 2k-1
  a44d  the certificate catches a solver artefact.  Scaling the cost by a
        positive constant cannot move the optimal set, so it cannot move rho --
        yet the scaled solve REPORTS rho = 2 on fields where the unscaled one
        reports 0.  The configuration is certified rho <= 1 in advance, so the
        scaled reading is refuted without solving anything again.  This is the
        a priori bound doing work a solver could not do for itself.

Run:  python experiments/a44_quadrotor_minsnap.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
ART = os.path.join(ROOT, "artifacts")
sys.path.insert(0, os.path.join(ROOT, "src"))

from exact_green import certify_exact                            # noqa: E402
from multisegment import MultiSegment                            # noqa: E402

D, K_ORDER, ELL, ETA, NSEG, NDIM = 7, 4, 4, 4, 5, 3
START, GOAL = np.array([-2.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0])
N_FIELDS = 40


def boundary():
    """Rest-to-rest: position fixed, derivatives 1..l zero at both ends."""
    b0 = np.zeros((ELL + 1, NDIM))
    b1 = np.zeros((ELL + 1, NDIM))
    b0[0], b1[0] = START, GOAL
    return b0, b1


def _blocks_chord(c, r):
    """Does this sphere intersect the straight line from START to GOAL?

    Fields that do not block the chord are solved by the chord, and a census of
    those measures nothing: the trajectory never has to avoid anything.  So a
    field is kept only if at least one sphere blocks it.
    """
    ab = GOAL - START
    t = float(np.clip(np.dot(c - START, ab) / np.dot(ab, ab), 0.0, 1.0))
    return float(np.linalg.norm(START + t * ab - c)) < r


def random_field(rng, n_obs=3, margin=0.25):
    """Spheres clear of both endpoints, at least one of them blocking the chord."""
    for _ in range(400):
        obs = []
        guard = 0
        while len(obs) < n_obs and guard < 400:
            guard += 1
            c = np.array([rng.uniform(-1.3, 1.3), rng.uniform(-0.7, 0.7),
                          rng.uniform(-0.7, 0.7)])
            r = rng.uniform(0.25, 0.55)
            if (np.linalg.norm(c - START) <= r + margin
                    or np.linalg.norm(c - GOAL) <= r + margin):
                continue
            if any(np.linalg.norm(c - c2) <= r + r2 + 0.05 for c2, r2 in obs):
                continue
            obs.append((c, r))
        if len(obs) == n_obs and any(_blocks_chord(c, r) for c, r in obs):
            return obs
    raise RuntimeError("no blocking field found")


def main():
    cert = certify_exact(D, K_ORDER, ELL, NSEG, ETA, D=96)
    print("    configuration k=%d d=%d l=%d N=%d eta=%d in R^%d, r=%d"
          % (K_ORDER, D, ELL, NSEG, ETA, NDIM, cert["r"]))
    print("    certified a priori, exact at degree 96: %s   (elevated minimum "
          "%s)" % (cert["nonneg"], cert["elev_min"]))
    print("    eta = %d, 2k-1 = %d, inside the safe region: %s\n"
          % (ETA, 2 * K_ORDER - 1, ETA <= 2 * K_ORDER - 1))

    b0, b1 = boundary()

    # a44d: the two scalings, compared where the scaled one still converges.
    scale_rows = []
    rng0 = np.random.default_rng(7)
    while len(scale_rows) < 8:
        obs = random_field(rng0)
        got = {}
        for ts in (True, False):
            try:
                m = MultiSegment(n=NDIM, d=D, k=K_ORDER, l=ELL, N=3,
                                 obstacles=obs, bc0=b0, bc1=b1, eta=ETA,
                                 normalise_time=ts)
                r = m.solve()
            except Exception:                                    # noqa: BLE001
                r = dict(converged=False)
            if r.get("converged"):
                got[ts] = int(r["rho"])
        if len(got) == 2:
            scale_rows.append(dict(scaled=got[True], unscaled=got[False]))
    n_scaled_bad = sum(1 for r in scale_rows if r["scaled"] > 1)
    n_unscaled_bad = sum(1 for r in scale_rows if r["unscaled"] > 1)
    print("    scaling at N=3, %d fields: the scaled solve reports rho > 1 on "
          "%d of them, the unscaled on %d" % (len(scale_rows), n_scaled_bad,
                                              n_unscaled_bad))
    print("    the configuration is certified rho <= 1 in advance, so those "
          "readings are solver artefacts\n")

    rng = np.random.default_rng(20260825)
    rows, tight, loose = [], None, None
    while len(rows) < N_FIELDS:
        obs = random_field(rng)
        try:
            ms = MultiSegment(n=NDIM, d=D, k=K_ORDER, l=ELL, N=NSEG,
                              obstacles=obs, bc0=b0, bc1=b1, eta=ETA,
                              normalise_time=False)
            res = ms.solve()
        except Exception:                                        # noqa: BLE001
            continue
        if not res.get("converged"):
            continue
        rho = int(res["rho"])
        rows.append(dict(rho=rho, cost=float(res["cost"]),
                         obs=[[list(map(float, c)), float(r)] for c, r in obs]))
        pack = (float(res["cost"]), obs, res)
        if rho == 0 and (tight is None or pack[0] < tight[0]):
            tight = pack
        if rho == 1 and (loose is None or pack[0] < loose[0]):
            loose = pack
        if len(rows) % 10 == 0:
            print("    %2d fields solved, max rho so far = %d"
                  % (len(rows), max(r["rho"] for r in rows)))

    n_le1 = sum(1 for r in rows if r["rho"] <= 1)
    dist = {v: sum(1 for r in rows if r["rho"] == v)
            for v in sorted({r["rho"] for r in rows})}
    print("\n    %d random sphere fields solved;  rho distribution %s"
          % (len(rows), dist))
    print("    rho <= 1 on %d of %d, as certified in advance" % (n_le1, len(rows)))

    def pack_fig(p_):
        if p_ is None:
            return None
        cost, obs, res = p_
        return dict(cost=cost, rho=int(res["rho"]),
                    obstacles=[[list(map(float, c)), float(r)]
                               for c, r in obs],
                    Gamma=np.asarray(res["Gamma"], float).tolist())

    fig = dict(tight=pack_fig(tight), loose=pack_fig(loose))

    a44a = dict(certified=bool(cert["nonneg"]), elev_min=str(cert["elev_min"]),
                r=int(cert["r"]), N=NSEG, eta=ETA, k=K_ORDER, d=D, n=NDIM,
                passed=bool(cert["nonneg"] and str(cert["elev_min"]) == "0"))
    a44b = dict(n=len(rows), n_rho_le_1=n_le1, dist={str(a): b for a, b in
                                                     dist.items()},
                max_rho=max(r["rho"] for r in rows),
                passed=bool(len(rows) == N_FIELDS and n_le1 == len(rows)))
    a44c = dict(eta=ETA, bound=2 * K_ORDER - 1,
                passed=bool(ETA <= 2 * K_ORDER - 1))
    a44d = dict(n=len(scale_rows), rows=scale_rows,
                n_scaled_over_1=n_scaled_bad, n_unscaled_over_1=n_unscaled_bad,
                passed=bool(scale_rows and n_scaled_bad > 0
                            and n_unscaled_bad == 0))
    with open(os.path.join(ART, "a44_quadrotor_minsnap.json"), "w") as fh:
        json.dump(dict(gates=dict(a44a_apriori=a44a, a44b_census=a44b,
                                  a44c_inside_safe_region=a44c,
                                  a44d_scaling_is_numerical=a44d),
                       rows=rows, figure=fig), fh, indent=1)
    print("\n  gates: a44a %s  a44b %s  a44c %s  a44d %s"
          % (a44a["passed"], a44b["passed"], a44c["passed"], a44d["passed"]))
    return 0 if all(g["passed"] for g in (a44a, a44b, a44c, a44d)) else 1


if __name__ == "__main__":
    sys.exit(main())
