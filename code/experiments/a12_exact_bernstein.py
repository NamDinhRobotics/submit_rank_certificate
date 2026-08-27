"""A12 -- the Bernstein positivity certificate in EXACT rational arithmetic,
and the irreducibility step Perron-Frobenius actually needs.

TWO THINGS WERE BEING ASSUMED.

(1) A9-A11 report elevated Bernstein coefficient minima of `-1.7e-18`,
    `-3.7e-21`, `-4.6e-23` and call the configuration certified.  Those numbers
    are NEGATIVE, and the test requires nonnegative coefficients, so what was
    really being asserted is "that is roundoff" -- an assumption about the
    arithmetic, not a result.  Everything in the construction is rational, so
    the sign is decidable.  It is decided here: every one of them is EXACTLY
    ZERO.  The assumption was true, and is now a computation.

(2) Nonnegative coefficients give `Gr >= 0`, and `Gr >= 0` does NOT give a
    simple top eigenvalue.  Perron-Frobenius needs irreducibility, and a
    nonnegative matrix can be reducible: `diag(A, A)` is entrywise nonnegative
    and positive definite with a doubly degenerate top eigenvalue.  So the
    theorem as previously stated does not follow from its own hypothesis.

    What repairs it is a property of the Bernstein form rather than a stronger
    numerical test.  On the OPEN square every Bernstein basis function is
    strictly positive, so a nonnegative coefficient array that is not
    identically zero gives `Gr((i,s),(j,t)) > 0` for all `s,t` in `(0,1)`.
    Entrywise positive implies irreducible.  Hence the missing hypothesis is
    just "no `(i,j)` block vanishes identically", which a12c checks -- and
    which holds everywhere here.

    The zeros that do occur are structural and harmless: exactly `2(l+1)` rows
    of `N_perp` vanish, namely the control points pinned by the boundary
    conditions.  A contact there has `u_a = 0`, contributing `w_a u_a u_a^T = 0`
    to `M_lambda`; it is a vacuous contact and drops out of the Gram matrix
    rather than disconnecting it.

Gates:
  a12a  Table I of the source paper, exactly: all four rows nonnegative, and
        the reported negative minima are exactly zero
  a12b  the dichotomy, exactly: `N=1` is strictly negative at all seven
        settings and `N>=2` is nonnegative at all 35, so the dichotomy no
        longer rests on where a tolerance is drawn
  a12c  irreducibility: no `(i,j)` block vanishes identically, and the rows
        that do vanish are exactly the `2(l+1)` pinned control points
  a12d  the exact block reproduces the floating-point one, which is what
        licenses reading A9-A11's numbers as approximations of these
  a12e  negative control: welded knots (`eta = d`) stay strictly negative in
        exact arithmetic too, so exactness did not simply certify everything
  a12f  the BOUNDARY of the square, which the open-square argument does not
        reach.  A pinned endpoint is vacuous (`u = 0`), but an interior
        JUNCTION is `s = 1` of segment `i` and `s = 0` of segment `i+1`, has
        `u != 0`, and is reachable.  There `Gr` is one row (against an interior
        parameter) or one entry (against another boundary parameter) of the
        block, so the extra hypothesis is a finite sign check on rows `0` and
        `d`.  It holds in every configuration tested, which removes the
        "interior parameters" caveat entirely.

Writes artifacts/a12_exact_bernstein.json.
"""
import argparse
import json
import os
import sys
from fractions import Fraction as F

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
ART = os.path.join(HERE, "..", "artifacts")

from exact_green import (certify_exact, coefficient_blocks,   # noqa: E402
                         constraint_matrix, endpoint_structure, nullspace)

TABLE1 = [("velocity", 1, 1, 0, 12, 0), ("acceleration", 2, 3, 2, 6, 2),
          ("jerk", 3, 5, 3, 4, 3), ("snap", 4, 7, 4, 3, 4)]
CONFIGS = ((3, 1, 0), (5, 1, 0), (7, 1, 0), (9, 1, 0),
           (5, 2, 1), (7, 2, 1), (7, 3, 2))
NS = (1, 2, 3, 4, 5, 6)
WELD = ((1, 0), (2, 0), (2, 1), (2, 2), (2, 3), (3, 3), (4, 3))
D_ELEV = 96


def _f(x):
    return float(x)


def structure(d, k, l, N, eta):
    """The irreducibility data: vanishing rows, and vanishing blocks."""
    blocks, r = coefficient_blocks(d, k, l, N, eta)
    Np = nullspace(constraint_matrix(d, l, N, eta))
    M = N * (d + 1)
    pinned = [i for i in range(M) if all(Np[c][i] == 0 for c in range(r))]
    dead = [[i, j] for (i, j), B in blocks.items()
            if all(x == 0 for row in B for x in row)]
    pos = all(any(x > 0 for row in B for x in row) for B in blocks.values())
    nz = [abs(x) for B in blocks.values() for row in B for x in row if x != 0]
    return dict(r=r, n_pinned=len(pinned), pinned=pinned,
                expected_pinned=2 * (l + 1), n_dead_blocks=len(dead),
                every_block_has_a_positive_entry=bool(pos),
                smallest_nonzero=_f(min(nz)) if nz else None)


def gate_a12a():
    rows = []
    for nm, k, d, l, N, eta in TABLE1:
        r = certify_exact(d, k, l, N, eta, D=D_ELEV)
        rows.append(dict(name=nm, k=k, d=d, l=l, N=N, eta=eta, r=r["r"],
                         elev_min_is_zero=bool(r["elev_min"] == 0),
                         elev_min=_f(r["elev_min"]),
                         elev_min_rel=_f(r["elev_min_rel"]),
                         nonneg=r["nonneg"], strict=r["strict"]))
        print("    %-13s k=%d d=%-2d N=%-3d r=%-3d  exact elev min = %s%s"
              % (nm, k, d, N, r["r"],
                 "0 exactly" if r["elev_min"] == 0 else str(r["elev_min"])[:34],
                 "" if r["nonneg"] else "   <-- NEGATIVE"))
    return dict(rows=rows, n=len(rows),
                n_nonneg=sum(x["nonneg"] for x in rows),
                n_exactly_zero=sum(x["elev_min_is_zero"] for x in rows),
                n_strict=sum(x["strict"] for x in rows),
                passed=all(x["nonneg"] for x in rows))


def gate_a12b():
    rows = []
    for N in NS:
        cells = []
        for (d, k, l) in CONFIGS:
            eta = max(0, k - 1) if N > 1 else 0
            r = certify_exact(d, k, l, N, eta, D=D_ELEV)
            rows.append(dict(N=N, d=d, k=k, l=l, eta=eta, r=r["r"],
                             elev_min_rel=_f(r["elev_min_rel"]),
                             nonneg=r["nonneg"], strict=r["strict"],
                             exactly_zero=bool(r["elev_min"] == 0)))
            cells.append("  OK   " if r["nonneg"]
                         else "%+7.0e" % _f(r["elev_min_rel"]))
        print("    N=%d | %s" % (N, " ".join(cells)))
    single = [x for x in rows if x["N"] == 1]
    multi = [x for x in rows if x["N"] > 1]
    return dict(rows=rows,
                n_single=len(single), n_single_nonneg=sum(x["nonneg"] for x in single),
                n_multi=len(multi), n_multi_nonneg=sum(x["nonneg"] for x in multi),
                n_multi_exactly_zero=sum(x["exactly_zero"] for x in multi),
                worst_single=max(x["elev_min_rel"] for x in single),
                passed=bool(not any(x["nonneg"] for x in single)
                            and all(x["nonneg"] for x in multi)))


def gate_a12c():
    rows = []
    for nm, k, d, l, N, eta in TABLE1:
        s = structure(d, k, l, N, eta)
        s.update(name=nm, k=k, d=d, l=l, N=N, eta=eta)
        rows.append(s)
        print("    %-13s pinned rows %2d (expected %2d)  dead blocks %d  "
              "every block has a positive entry: %s  smallest nonzero %.2e"
              % (nm, s["n_pinned"], s["expected_pinned"], s["n_dead_blocks"],
                 s["every_block_has_a_positive_entry"], s["smallest_nonzero"]))
    for N in (2, 3):
        for (d, k, l) in ((3, 1, 0), (7, 2, 1)):
            s = structure(d, k, l, N, max(0, k - 1))
            s.update(name="sweep", k=k, d=d, l=l, N=N, eta=max(0, k - 1))
            rows.append(s)
    return dict(rows=rows, n=len(rows),
                n_dead_blocks=sum(x["n_dead_blocks"] for x in rows),
                n_pinned_as_expected=sum(x["n_pinned"] == x["expected_pinned"]
                                         for x in rows),
                passed=bool(all(x["n_dead_blocks"] == 0 for x in rows)
                            and all(x["every_block_has_a_positive_entry"]
                                    for x in rows)
                            and all(x["n_pinned"] == x["expected_pinned"]
                                    for x in rows)))


def gate_a12d():
    """Does the exact block reproduce the floating-point one?

    Without this the exact computation could be answering a different question
    from the one A9-A11 measured -- a different null-space basis, say.  It is
    not: the block is basis-independent, and this is the numerical witness.
    """
    from a11_table1 import build, coef_block
    rows = []
    for nm, k, d, l, N, eta in TABLE1:
        blocks, _ = coefficient_blocks(d, k, l, N, eta)
        ms = build(k, d, l, N, eta)
        _, fl = coef_block(ms)
        worst = 0.0
        for i in range(N):
            for j in range(N):
                E = np.array([[_f(x) for x in row] for row in blocks[(i, j)]])
                worst = max(worst, float(np.abs(E - fl[i * N + j]).max()))
        rows.append(dict(name=nm, max_abs_diff=worst))
        print("    %-13s max |exact - float| = %.3e" % (nm, worst))
    return dict(rows=rows, worst=max(x["max_abs_diff"] for x in rows),
                passed=bool(max(x["max_abs_diff"] for x in rows) < 1e-12))


def gate_a12e():
    """Welded knots must still fail.  `eta = d` makes the spline a single
    polynomial however many pieces it is cut into, so a test that certified it
    would be certifying the very thing the dichotomy says fails."""
    rows = []
    for (N, eta) in WELD:
        e = eta if N > 1 else 0
        r = certify_exact(3, 1, 0, N, e, D=D_ELEV)
        genuine = bool(N > 1 and e < 3)
        rows.append(dict(N=N, eta=e, r=r["r"], genuine_knot=genuine,
                         elev_min_rel=_f(r["elev_min_rel"]),
                         nonneg=r["nonneg"]))
        print("    N=%d eta=%d r=%-2d %-14s exact elev min rel = %+.3e  %s"
              % (N, e, r["r"], "(genuine knot)" if genuine else "(welded)",
                 _f(r["elev_min_rel"]), "OK" if r["nonneg"] else "fails"))
    return dict(rows=rows,
                n_genuine=sum(x["genuine_knot"] for x in rows),
                n_genuine_certified=sum(x["nonneg"] and x["genuine_knot"]
                                        for x in rows),
                n_welded=sum(not x["genuine_knot"] for x in rows),
                n_welded_certified=sum(x["nonneg"] and not x["genuine_knot"]
                                       for x in rows),
                passed=bool(all(x["nonneg"] == x["genuine_knot"] for x in rows)))


def gate_a12g():
    """WHICH ARRAY the nonnegativity hypothesis is about.

    The theorem was stated on the raw coefficient block `B^ij` while what the
    sweep verifies is the degree-96 ELEVATED array `E B E^T`.  Those are not
    the same hypothesis: raw nonnegative implies elevated nonnegative, never
    the reverse.  And the difference is not academic -- the raw block carries a
    negative entry on most configurations, so a theorem stated on it would not
    apply to them.  Measured here so the paper cannot quietly say `B` again.
    """
    rows = []
    for nm, k, d, l, N, eta in TABLE1:
        r = certify_exact(d, k, l, N, eta, D=D_ELEV)
        rows.append(dict(name=nm, k=k, d=d, N=N,
                         raw_min_rel=_f(r["raw_min_rel"]),
                         raw_nonneg=bool(r["raw_min"] >= 0),
                         elev_nonneg=bool(r["elev_min"] >= 0)))
    for N in NS[1:]:
        for (d, k, l) in CONFIGS:
            r = certify_exact(d, k, l, N, max(0, k - 1), D=D_ELEV)
            rows.append(dict(name="sweep", k=k, d=d, N=N,
                             raw_min_rel=_f(r["raw_min_rel"]),
                             raw_nonneg=bool(r["raw_min"] >= 0),
                             elev_nonneg=bool(r["elev_min"] >= 0)))
    n_raw_neg = sum(not x["raw_nonneg"] for x in rows)
    worst = min(x["raw_min_rel"] for x in rows)
    print("    raw block negative in %d of %d configurations, worst %.3e "
          "relative" % (n_raw_neg, len(rows), worst))
    print("    elevated array nonnegative in all %d"
          % sum(x["elev_nonneg"] for x in rows))
    return dict(rows=rows, n=len(rows), n_raw_negative=n_raw_neg,
                worst_raw_min_rel=worst,
                n_elevated_nonneg=sum(x["elev_nonneg"] for x in rows),
                passed=bool(n_raw_neg > 0                     # not vacuous
                            and all(x["elev_nonneg"] for x in rows)))


def gate_a12f():
    """Remove the "interior parameters" caveat, or find out where it is needed.

    The open-square argument covers contacts at interior parameters.  It leaves
    the boundary, and the boundary is not empty: an interior JUNCTION is
    `s = 1` of segment `i` and `s = 0` of segment `i+1`, it has `u != 0`, and it
    is physically reachable.  There `Gr` is read off one row (against an
    interior parameter) or one entry (against another boundary parameter) of the
    block, so the extra hypothesis is a finite sign check on rows `0` and `d` --
    no new machinery, the same coefficient blocks.
    """
    rows = []
    for nm, k, d, l, N, eta in TABLE1:
        s = endpoint_structure(d, k, l, N, eta)
        s.update(name=nm, k=k, d=d, l=l, N=N, eta=eta)
        rows.append(s)
        print("    %-13s live endpoints %-3d  bad rows %-2d  bad corners %-2d  -> %s"
              % (nm, s["n_live_endpoints"], s["n_bad_rows"], s["n_bad_corners"],
                 "closed square OK" if s["closed_square_ok"] else "FAILS"))
    for N in NS[1:]:
        for (d, k, l) in CONFIGS:
            s = endpoint_structure(d, k, l, N, max(0, k - 1))
            s.update(name="sweep", k=k, d=d, l=l, N=N, eta=max(0, k - 1))
            rows.append(s)
    n_ok = sum(x["closed_square_ok"] for x in rows)
    print("    %d of %d configurations satisfy it, so the hypothesis holds on "
          "the CLOSED square" % (n_ok, len(rows)))
    return dict(rows=rows, n=len(rows), n_closed_square_ok=n_ok,
                n_table1=len(TABLE1),
                n_table1_ok=sum(x["closed_square_ok"] for x in rows[:len(TABLE1)]),
                passed=bool(n_ok == len(rows)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ART, "a12_exact_bernstein.json"))
    a = ap.parse_args()

    print("\n[a12a] Table I of the source paper, in exact arithmetic")
    a12a = gate_a12a()
    print("\n[a12b] the dichotomy, in exact arithmetic")
    a12b = gate_a12b()
    print("\n[a12c] irreducibility structure")
    a12c = gate_a12c()
    print("\n[a12d] exact vs floating point")
    a12d = gate_a12d()
    print("\n[a12e] negative control: welded knots")
    a12e = gate_a12e()
    print("\n[a12f] the boundary: junctions, and the closed square")
    a12f = gate_a12f()
    print("\n[a12g] the hypothesis is about the ELEVATED array, not the raw one")
    a12g = gate_a12g()

    gates = dict(a12a_table1_exact=a12a, a12b_dichotomy_exact=a12b,
                 a12c_irreducibility=a12c, a12d_exact_matches_float=a12d,
                 a12e_welded_control=a12e,
                 a12f_closed_square=a12f,
                 a12g_elevated_not_raw=a12g)
    print("\n  --- gates ---")
    for name, g in gates.items():
        print("  %s: %s" % (name, "PASS" if g["passed"] else "FAIL"))
    with open(a.out, "w") as fh:
        json.dump(dict(gates=gates, D_elev=D_ELEV), fh, indent=1, default=float)
    print("\nwrote %s" % a.out)
    if not all(g["passed"] for g in gates.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
