"""A13 -- what the nodal point `a0` actually has to do with the counterexample.

THE INCOHERENCE THIS RESOLVES.  Section IV motivated the search with the `m = 2`
remark (the certificate can fail only on the nodal set of `Gr`), located
`a0 = 0.1127017` from it, and then reported that the cells it found have `m = 3`
and `Z = 0`.  Those are two different routes, and as written the agreement
between `a0` and the observed band reads as a numerical coincidence.  It is not
one, but the connection is narrower than "the theory predicts the location", so
it is worth pinning exactly.

  a13a  `a0` IS a moment-cone condition, not merely a nodal one.  For `f = 2`
        and TWO contacts, `Z = 0` means `K = w_1u_1u_1^T + w_2u_2u_2^T`.  With
        `U = [u_1 u_2]` invertible that is `W = U^{-1}KU^{-T}` diagonal, and
        since `Gr^{-1} = U^{-1}KU^{-T}` and a `2x2` inverse has off-diagonal
        `-Gr_12/det`, the condition is exactly `Gr_12 = 0`.  So the nodal point
        of `Gr` is precisely the unique symmetric two-contact configuration at
        which `Z = 0` is available at all.

  a13b  why the region is OPEN rather than a point.  Two atoms span a rank-2
        subspace of the 3-dimensional `S^2`, so cone membership is codimension
        one -- a single parameter value.  Three atoms span all of `S^2`, so the
        cone is full-dimensional and membership is an OPEN condition.  The band
        is the third atom turning an equation into an inequality.

  a13c  and the honest limit: cone membership is NECESSARY for `Z = 0`, not
        sufficient.  A configuration is exhibited where all three weights are
        strictly positive -- so `K` is in the cone -- and yet `rho = 1`, because
        the rest of the KKT system does not agree.  This is why the extent of
        the band is measured rather than predicted, and why a13 does not claim
        `a0` bounds it.

  a13d  the transition, measured along a line through `a0`: with the corner
        contacts inside the negative lobe the middle obstacle is touched too
        (`m = 3`, `Z = 0`, `rho = 2`); past `a0` the off-diagonal changes sign,
        the middle contact disappears (`m = 2`) and the certificate clears the
        instance.

Writes artifacts/a13_nodal_mechanism.json.
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

from bernstein import bd                                     # noqa: E402
from a3_finite_d_certificate import green_matrix             # noqa: E402
from a4_simplicity_margin import three_blocker, analyse      # noqa: E402

A0 = 0.1127016528695822          # a3's bisected nodal point, recomputed there
D, KK, LL = 3, 1, 0              # the counterexample family: f = 2


def _setup():
    K, Ki, F = green_matrix(D, KK, LL)
    return K, Ki, (lambda s: bd(s, D)[F])


def _svec(M):
    return np.array([M[0, 0], np.sqrt(2.0) * M[0, 1], M[1, 1]])


def gate_a13a():
    """`Gr_12 = 0` and "the two-atom representation exists" are the same test,
    and both single out `a0`."""
    K, Ki, u = _setup()
    rows = []
    for s in (0.08, 0.10, A0, 0.12, 0.14):
        U = np.array([u(s), u(1 - s)]).T
        Gr = U.T @ Ki @ U
        g12 = float(Gr[0, 1] / np.sqrt(Gr[0, 0] * Gr[1, 1]))
        W = np.linalg.solve(U, np.linalg.solve(U, K.T).T)      # U^-1 K U^-T
        rows.append(dict(sigma=float(s), gr12_norm=g12,
                         W_offdiag=float(W[0, 1]),
                         w1=float(W[0, 0]), w2=float(W[1, 1])))
        print("    sigma=%-9.6f Gr12=%-11.3e  offdiag(W)=%-11.3e  w=(%.3f,%.3f)"
              % (s, g12, W[0, 1], W[0, 0], W[1, 1]))
    at_a0 = [r for r in rows if abs(r["sigma"] - A0) < 1e-12][0]
    # the two vanish together, and they vanish at a0
    both_small = abs(at_a0["gr12_norm"]) < 1e-6 and abs(at_a0["W_offdiag"]) < 1e-5
    signs_agree = all(np.sign(r["gr12_norm"]) == -np.sign(r["W_offdiag"])
                      for r in rows if abs(r["sigma"] - A0) > 1e-6)
    return dict(a0=A0, rows=rows, vanish_together_at_a0=bool(both_small),
                signs_track=bool(signs_agree),
                weights_positive_at_a0=bool(at_a0["w1"] > 0 and at_a0["w2"] > 0),
                passed=bool(both_small and signs_agree
                            and at_a0["w1"] > 0 and at_a0["w2"] > 0))


def gate_a13b():
    """Two atoms: codimension one.  Three atoms: full-dimensional, hence open."""
    _, _, u = _setup()
    out = {}
    for label, ss in (("two_corner_atoms", (0.08, 0.92)),
                      ("three_atoms", (0.08, 0.5, 0.92))):
        A = np.array([_svec(np.outer(u(s), u(s))) for s in ss]).T
        out[label] = dict(n_atoms=len(ss), rank=int(np.linalg.matrix_rank(A)),
                          ambient=3)
        print("    %-18s atoms=%d rank=%d of %d -> %s"
              % (label, len(ss), out[label]["rank"], 3,
                 "codimension 1" if out[label]["rank"] < 3
                 else "full-dimensional (open)"))
    return dict(**out,
                passed=bool(out["two_corner_atoms"]["rank"] == 2
                            and out["three_atoms"]["rank"] == 3))


def gate_a13c():
    """Necessary, not sufficient: a positive representation with rho = 1."""
    K, _, u = _setup()
    seg = three_blocker(0.12, rmid=0.60)
    r = analyse(seg, ns=4001)
    cs = np.array(r["contacts"])
    A = np.array([_svec(np.outer(u(s), u(s))) for s in cs]).T
    w, *_ = np.linalg.lstsq(A, _svec(K), rcond=None)
    resid = float(np.linalg.norm(A @ w - _svec(K)))
    print("    s1=0.12 rmid=0.60: contacts %s"
          % np.array2string(cs, precision=4, separator=","))
    print("    weights %s (all > 0: %s), residual %.1e, yet rho = %s"
          % (np.array2string(w, precision=3, separator=","),
             bool((w > 0).all()), resid, r.get("rho")))
    return dict(s1=0.12, rmid=0.60, contacts=[float(x) for x in cs],
                weights=[float(x) for x in w], residual=resid,
                all_weights_positive=bool((w > 0).all()), rho=r.get("rho"),
                passed=bool((w > 0).all() and resid < 1e-9 and r.get("rho") == 1))


def gate_a13d():
    """The transition along a line through `a0`, at a middle radius inside the
    band rather than on its edge."""
    _, Ki, u = _setup()
    rows = []
    for s1 in (0.08, 0.09, 0.10, 0.105, 0.11, 0.1127, 0.115, 0.12, 0.13):
        seg = three_blocker(s1, rmid=0.25)
        r = analyse(seg, ns=4001)
        if r.get("status") != "optimal":
            rows.append(dict(s1=float(s1), status=r.get("status")))
            continue
        cs = np.array(r["contacts"])
        U = np.array([u(s) for s in cs]).T
        Gr = U.T @ Ki @ U
        g12 = float(Gr[0, -1] / np.sqrt(Gr[0, 0] * Gr[-1, -1])) if len(cs) > 1 \
            else float("nan")
        rows.append(dict(s1=float(s1), sigma=float(cs.min()), m=r.get("m"),
                         rho=r.get("rho"), gr12_at_contacts=g12,
                         margin=r.get("margin")))
        print("    s1=%-7.4f sigma=%-8.4f m=%-2s rho=%-2s Gr12=%+.3e"
              % (s1, cs.min(), r.get("m"), r.get("rho"), g12))
    ok = [r for r in rows if "rho" in r]
    below = [r for r in ok if r["s1"] <= A0]
    above = [r for r in ok if r["s1"] > A0]
    # `a_0` is a CONTACT parameter and `s_1` a design parameter, so comparing
    # them needs them to be the same number.  Inside the band they are: the
    # corner contact lands on the design point.  Above it the contact slides,
    # which is why the gap is reported for both sides rather than once.
    gap_below = max(abs(r["s1"] - r["sigma"]) for r in below) if below else None
    gap_above = max(abs(r["s1"] - r["sigma"]) for r in above) if above else None
    print("    |s1 - sigma|: %.2e below a0, %.2e above" % (gap_below, gap_above))
    return dict(rmid=0.25, a0=A0, rows=rows,
                max_s1_sigma_gap_below=gap_below,
                max_s1_sigma_gap_above=gap_above,
                n_below=len(below), n_below_rho2=sum(r["rho"] == 2 for r in below),
                n_above=len(above), n_above_rho1=sum(r["rho"] == 1 for r in above),
                m_drops_above=bool(all(r["m"] == 2 for r in above)
                                   and all(r["m"] == 3 for r in below)),
                sign_flips=bool(all(r["gr12_at_contacts"] < 0 for r in below)
                                and all(r["gr12_at_contacts"] > 0 for r in above)),
                passed=bool(below and above
                            and all(r["rho"] == 2 for r in below)
                            and all(r["rho"] == 1 for r in above)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ART, "a13_nodal_mechanism.json"))
    a = ap.parse_args()

    print("\n[a13a] a0 is the unique two-contact configuration admitting Z = 0")
    a13a = gate_a13a()
    print("\n[a13b] two atoms: codimension 1;  three atoms: full-dimensional")
    a13b = gate_a13b()
    print("\n[a13c] cone membership is necessary, NOT sufficient")
    a13c = gate_a13c()
    print("\n[a13d] the transition across a0 at rmid = 0.25")
    a13d = gate_a13d()

    gates = dict(a13a_a0_is_the_two_atom_solution=a13a,
                 a13b_third_atom_opens_the_cone=a13b,
                 a13c_membership_not_sufficient=a13c,
                 a13d_transition_across_a0=a13d)
    print("\n  --- gates ---")
    for nm, g in gates.items():
        print("  %s: %s" % (nm, "PASS" if g["passed"] else "FAIL"))
    with open(a.out, "w") as fh:
        json.dump(dict(gates=gates), fh, indent=1, default=float)
    print("\nwrote %s" % a.out)
    if not all(g["passed"] for g in gates.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
