"""A50 -- the certificate does not grow with the problem.

Theorem `thm:contacts` says the recovery-dimension bound is charged to the
CONTACTS: no term in it is the ambient dimension, the curve degree, the segment
count, or the order of the semidefinite program.  Every other file measures the
certificate on one problem size.  This one grows the problem deliberately and
watches the certificate stay where it is.

TWO AXES, EACH GROWING THE LIFTED PROGRAM AND NOTHING ELSE.

  degree   k = 2, l = 1, R^2, d = 5, 7, 9, 11, 13, so the lifted free block is
           f x f with f = d + 1 - 2(l + 1) = 2, 4, 6, 8, 10.  The obstacle
           field is redrawn per degree but from the same seed and the same law.

  ambient  d = 9, k = 2, l = 1 fixed (f = 6), n = 2, 3, 4, 5.  The obstacles
           are the same discs, embedded in R^n; the curve genuinely has n rows.

In both, the reported quantity is `m`, the number of live contacts the optimal
dual certificate charges -- the order of the matrix Corollary `cor:cert` runs
its eigendecomposition on.  The a priori surrogate is m <= Jd (Lemma
`lem:atomic`); what is measured is much smaller, and flat.

WHY THIS FILE EXISTS AT ALL.  The recovery figure's own instance has d = 5,
l = 1, hence f = 2, and its contact count is also 2 -- so on that instance the
certificate and the lifted block are the SAME SIZE and the claim has no visible
content.  A reader shown only that instance could reasonably conclude the
opposite of the theorem.

Gates:
  a50a  along the degree axis f grows and max m does not
  a50b  along the ambient axis n grows and max m does not
  a50c  the size ratio f/m grows monotonically in d over the measured cells
  a50d  every solved cell is certified: g > 0 wherever rho >= 1

Run:  python experiments/a50_size_contrast.py
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

from relaxation import Segment                               # noqa: E402
import a4_simplicity_margin as a4                            # noqa: E402

K, L = 2, 1
DEGREES = (5, 7, 9, 11, 13)
AMBIENTS = (2, 3, 4, 5)
N_DRAW = 14
DISCS = [((-0.75, 0.30), 0.55), ((0.70, -0.28), 0.60), ((0.05, 0.85), 0.45)]


def embed(n, jitter):
    """The same discs, embedded in R^n, nudged by `jitter` in the plane."""
    out = []
    for (cx, cy), r in DISCS:
        c = np.zeros(n)
        c[0], c[1] = cx + jitter[0], cy + jitter[1]
        out.append((c, r))
    return out


def one_cell(n, d, rng):
    """Draw until a recovery-positive instance turns up; measure it."""
    bc0 = np.zeros((L + 1, n)); bc1 = np.zeros((L + 1, n))
    bc0[0, 0], bc1[0, 0] = -2.0, 2.0
    rows = []
    for _ in range(N_DRAW):
        obs = embed(n, rng.normal(0, 0.16, 2))
        try:
            seg = Segment(n=n, d=d, k=K, l=L, obstacles=obs, bc0=bc0, bc1=bc1)
            res = seg.solve(backend="cvxpy", solver="CLARABEL")
        except Exception:                                    # noqa: BLE001
            continue
        if not res.get("converged") or int(res["rho"]) < 1:
            continue
        cert = a4.analyse(seg)
        if cert.get("status") != "optimal":
            continue
        rows.append(dict(n=n, d=d, f=seg.f, rho=int(res["rho"]),
                         m=int(cert["m"]), margin=float(cert["margin"])))
    return rows


def main():
    rng = np.random.default_rng(20260829)
    deg_rows, amb_rows = [], []

    print("=== A50: the certificate does not grow with the problem ===\n")
    print("degree axis  (n = 2, k = 2, l = 1)")
    print("   d    f      cells   max m    f/m   min g")
    for d in DEGREES:
        rows = one_cell(2, d, rng)
        deg_rows += rows
        if rows:
            mm = max(r["m"] for r in rows)
            print("  %2d   %2d      %3d      %2d    %4.1f   %.4f"
                  % (d, rows[0]["f"], len(rows), mm, rows[0]["f"] / mm,
                     min(r["margin"] for r in rows)))

    print("\nambient axis  (d = 9, f = 6 fixed)")
    print("   n    f      cells   max m   min g")
    for n in AMBIENTS:
        rows = one_cell(n, 9, rng)
        amb_rows += rows
        if rows:
            print("  %2d   %2d      %3d      %2d    %.4f"
                  % (n, rows[0]["f"], len(rows),
                     max(r["m"] for r in rows),
                     min(r["margin"] for r in rows)))

    by_d = {d: [r for r in deg_rows if r["d"] == d] for d in DEGREES}
    by_n = {n: [r for r in amb_rows if r["n"] == n] for n in AMBIENTS}
    mmax_d = {d: (max(r["m"] for r in v) if v else None) for d, v in by_d.items()}
    mmax_n = {n: (max(r["m"] for r in v) if v else None) for n, v in by_n.items()}
    solved_d = [d for d in DEGREES if by_d[d]]
    solved_n = [n for n in AMBIENTS if by_n[n]]
    ratios = [by_d[d][0]["f"] / mmax_d[d] for d in solved_d]
    allrows = deg_rows + amb_rows
    min_g = min((r["margin"] for r in allrows), default=None)

    print("\n  m never exceeds %d anywhere, while f runs %d -> %d and n runs %d -> %d"
          % (max(r["m"] for r in allrows),
             by_d[solved_d[0]][0]["f"], by_d[solved_d[-1]][0]["f"],
             solved_n[0], solved_n[-1]))

    gates = dict(
        a50a_degree=dict(
            f_by_d={str(d): by_d[d][0]["f"] for d in solved_d},
            max_m_by_d={str(d): mmax_d[d] for d in solved_d},
            passed=bool(len(solved_d) >= 4
                        and by_d[solved_d[-1]][0]["f"] > by_d[solved_d[0]][0]["f"]
                        and mmax_d[solved_d[-1]] <= mmax_d[solved_d[0]])),
        a50b_ambient=dict(
            max_m_by_n={str(n): mmax_n[n] for n in solved_n},
            passed=bool(len(solved_n) >= 3
                        and max(mmax_n[n] for n in solved_n)
                        == min(mmax_n[n] for n in solved_n))),
        a50c_ratio=dict(
            ratios=[float(x) for x in ratios],
            passed=bool(len(ratios) >= 4
                        and all(b >= a for a, b in zip(ratios, ratios[1:])))),
        a50d_certified=dict(
            n_cells=len(allrows), min_margin=min_g,
            passed=bool(allrows and min_g > 0.0)),
    )
    out = os.path.join(ART, "a50_size_contrast.json")
    with open(out, "w") as fh:
        json.dump(dict(gates=gates, rows=allrows), fh, indent=1)
    ok = all(g["passed"] for g in gates.values())
    print("  wrote %s" % out)
    print("  ALL GATES PASSED" if ok else "  *** A GATE FAILED ***")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
