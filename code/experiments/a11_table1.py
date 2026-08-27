"""A11 -- run the certificate on the SOURCE PAPER'S OWN Table I, and note that
doing so settles every instance of it at once.

A9 and A10 close both routes to `rho >= 2` for multisegment curves, but on
configurations of our choosing.  The paper's claim is about THEIR regime, so
the honest check is their Table I, read off the paper itself rather than off a
second-hand note (`paper sdp.pdf`, p. 8):

    Degree  Cost                Continuity  BC   Segments
    1       velocity  (k = 1)   C^0         0    12
    3       acceleration (k=2)  C^2         2    6
    5       jerk      (k = 3)   C^3         3    4
    7       snap      (k = 4)   C^4         4    3

with the caption stating degree `= 2k-1`, continuity `C^k` (`C^0` at `k = 1`
"to avoid overconstraining the problem"), and BC order the highest derivative
fixed to zero at the endpoints.  Note every row has single-segment
`f = d+1-2(l+1) <= 0`: in their benchmark ALL the freedom is in the joints,
which is exactly why A7's single-segment verdict says nothing about them and
this phase is needed.

THE POINT THAT MAKES THIS MORE THAN A SPOT CHECK.  The certificate tests the
entries of `N_{perp,i} K_multi^{-1} N_{perp,j}^T`, and that block is built from
the cost Gram and the boundary/continuity null space only.  It does not involve
the obstacles, and it does not involve the ambient dimension `n` -- both enter
the SDP elsewhere.  a11b verifies this invariance numerically rather than
asserting it (identical to `0.00e+00` across obstacle count, obstacle placement,
and `n = 2, 3, 5`).  Hence certifying a Table I ROW certifies `rho <= 1` for
every instance of that row: any obstacles, any number of them, any `n` --
including the 100 random `R^3` (15 obstacles) and `R^5` (30 obstacles) instances
the paper evaluates on.

Gates:
  a11a  each Table I row is certified: the elevated Bernstein coefficients of
        the discrete Green matrix are nonnegative, hence `Gr >= 0`, hence
        `rho <= 1` by Perron-Frobenius
  a11b  the certificate is invariant to obstacles and to `n`, so a11a is a
        statement about the configuration and not about one instance
  a11c  the other route is shut too: every row has `r > 2`, so by the
        moment-cone obstruction `Z = 0` is impossible there

Writes artifacts/a11_table1.json.
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

from multisegment import MultiSegment                       # noqa: E402
from a8_nondegeneracy_multiseg import multi_pieces          # noqa: E402
from a9_multiseg_rho2 import _elevation                     # noqa: E402


# (name, k, degree, BC order l, segments N, continuity eta) -- paper's Table I
TABLE1 = [("velocity", 1, 1, 0, 12, 0),
          ("acceleration", 2, 3, 2, 6, 2),
          ("jerk", 3, 5, 3, 4, 3),
          ("snap", 4, 7, 4, 3, 4)]

D_ELEV = 96


def build(k, d, l, N, eta, obstacles=None, n=2):
    if obstacles is None:
        obstacles = [(np.zeros(n), 0.4)]
    b0 = [[-2.0] + [0.0] * (n - 1)] + [[0.0] * n] * l
    b1 = [[2.0] + [0.0] * (n - 1)] + [[0.0] * n] * l
    return MultiSegment(n=n, d=d, k=k, l=l, N=N, eta=eta,
                        obstacles=obstacles, bc0=b0, bc1=b1)


def coef_block(ms):
    K, Np = multi_pieces(ms)
    Ki = np.linalg.inv(K)
    return K, [Np[i] @ Ki @ Np[j].T for i in range(ms.N) for j in range(ms.N)]


def certify(k, d, l, N, eta, D=D_ELEV):
    ms = build(k, d, l, N, eta)
    K, blocks = coef_block(ms)
    scale = max(1.0, max(float(np.abs(B).max()) for B in blocks))
    raw = float(min(B.min() for B in blocks))
    E = _elevation(d, D)
    ele = float(min((E @ B @ E.T).min() for B in blocks))
    return dict(k=k, d=d, l=l, N=N, eta=eta, r=int(K.shape[0]),
                f_single=int(d + 1 - 2 * (l + 1)), scale=scale,
                raw_min=raw, raw_min_rel=raw / scale,
                elev_min=ele, elev_min_rel=ele / scale,
                certified=bool(ele >= -1e-12 * scale))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ART, "a11_table1.json"))
    args = ap.parse_args()

    print("=== A11: the certificate on the source paper's own Table I ===")

    # ---- a11a --------------------------------------------------------
    print("\n  --- a11a: each Table I row ---")
    print("  %-13s %2s %2s %2s %3s %4s %4s %6s | %13s %s"
          % ("row", "k", "d", "l", "N", "eta", "r", "f_1seg",
             "elev min coef", "verdict"))
    rows = []
    for (nm, k, d, l, N, eta) in TABLE1:
        r = certify(k, d, l, N, eta)
        r["name"] = nm
        rows.append(r)
        print("  %-13s %2d %2d %2d %3d %4d %4d %6d | %13.3e %s"
              % (nm, k, d, l, N, eta, r["r"], r["f_single"], r["elev_min_rel"],
                 "CERTIFIED" if r["certified"] else "*** NOT certified ***"))
    a11a = dict(rows=rows, n=len(rows),
                n_certified=sum(1 for r in rows if r["certified"]),
                worst_elev_rel=float(min(r["elev_min_rel"] for r in rows)),
                all_single_seg_degenerate=bool(all(r["f_single"] <= 0
                                                   for r in rows)),
                passed=bool(rows and all(r["certified"] for r in rows)))
    print(f"\n    certified {a11a['n_certified']}/{len(rows)}; every row has "
          f"single-segment f <= 0, i.e. all freedom is in the joints")

    # ---- a11b --------------------------------------------------------
    print("\n  --- a11b: is the certificate a property of the CONFIGURATION? ---")
    rng = np.random.default_rng(0)
    probes = [("1 obstacle, R^2", 2, [(np.zeros(2), 0.4)]),
              ("2 obstacles, R^2", 2, [(np.array([0.9, -0.3]), 0.7),
                                       (np.array([-1.1, 0.5]), 0.35)]),
              ("15 obstacles, R^3", 3, [(rng.uniform(-2, 2, 3), 0.6)
                                        for _ in range(15)]),
              ("30 obstacles, R^5", 5, [(rng.uniform(-2, 2, 5), 0.5)
                                        for _ in range(30)])]
    inv = []
    for (nm, k, d, l, N, eta) in TABLE1:
        ref = None
        worst = 0.0
        for (pn, n, obs) in probes:
            _, blocks = coef_block(build(k, d, l, N, eta, obstacles=obs, n=n))
            flat = np.concatenate([B.ravel() for B in blocks])
            if ref is None:
                ref = flat
            else:
                worst = max(worst, float(np.abs(flat - ref).max()))
        inv.append(dict(name=nm, max_abs_diff=worst))
        print("  %-13s max |difference| across obstacles and n = 2,3,5: %.2e"
              % (nm, worst))
    a11b = dict(rows=inv, probes=[p[0] for p in probes],
                worst=float(max(r["max_abs_diff"] for r in inv)),
                passed=bool(all(r["max_abs_diff"] == 0.0 for r in inv)))
    print(f"\n    worst difference {a11b['worst']:.2e} -- the block is a "
          f"function of (N, d, k, l, eta) alone, so a11a settles EVERY instance "
          f"of each row: any obstacles, any n")

    # ---- a11c --------------------------------------------------------
    print("\n  --- a11c: the other route (Z = 0) needs r = 2 ---")
    r_ok = [r for r in rows if r["r"] > 2]
    a11c = dict(r_values={r["name"]: r["r"] for r in rows},
                n_with_r_gt_2=len(r_ok),
                passed=bool(len(r_ok) == len(rows)))
    print("    r = %s -- all > 2, so by the moment-cone obstruction Z = 0 is "
          "impossible on every Table I row" % a11c["r_values"])

    gates = dict(a11a_table1_certified=a11a, a11b_configuration_only=a11b,
                 a11c_cone_route_shut=a11c)
    print("\n  --- gates ---")
    for nm in gates:
        print(f"  {nm}: {'PASS' if gates[nm]['passed'] else 'FAIL'}")

    with open(args.out, "w") as fh:
        json.dump(dict(gates=gates, table1=TABLE1), fh, indent=1, default=float)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
