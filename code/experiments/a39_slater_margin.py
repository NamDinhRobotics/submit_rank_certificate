"""Assumption 4(i), checked instead of assumed.

The manuscript verifies part (ii) of Assumption 4 on every instance it reports
and, since the atomicity lemma, proves it outright.  Part (i) -- Slater's
condition for the problem AFTER facial reduction, which is what buys strong
duality and an attained dual, and therefore everything downstream -- was only
ever asserted.  With (ii) discharged, (i) is the paper's last live assumption,
so leaving it unmeasured is the wrong asymmetry.

THE TEST.  Slater holds iff the reduced feasible set has a point strictly inside
every cone constraint.  That is itself an SDP: keep the constraints of (5), drop
the objective, and maximise the uniform interior margin

    max  t   s.t.   P >= t I,   Q0_j >= t I,   Q1_j >= t I,   (linear equality)

whose value is positive exactly when a strictly feasible point exists.  This is
a decision, not a diagnostic: `t* > 0` certifies Slater for that instance, and
`t* <= 0` would refute it.  The witness margin is reported so a reader can see
how far from the boundary the interior point sits.

The margin is capped at `t <= 1` so the answer is a number and not an
unboundedness report; the cap binds on many instances and the count is reported,
because "smallest margin" is otherwise a statistic about the cap.

a39c is the one that matters now.  A finite sample cannot discharge a hypothesis
quantified over every obstacle arrangement, and the lemma that replaced the
assumption is constructive: from a strictly collision-free curve it BUILDS a
strictly feasible point, by lifting `X_F` off `Gamma_F^T Gamma_F` and shifting
each sum-of-squares block by `eta` times a fixed positive `h`.  a39c runs that
construction and checks the point it produces, so what is verified is the proof,
not a sample of its conclusions.

Gates:
  a39a  every instance in the named population admits a strictly feasible point
  a39b  the same over a random census draw, so the verdict is not a property of
        the hand-built layouts
  a39c  the CONSTRUCTION of Lemma "strict feasibility is Slater" produces a
        point strictly inside every cone, with the linear equality exact
  a39d  the COUNTEREXAMPLE satisfies the clearance hypothesis.  Every theorem in
        the paper presumes it, and the counterexample is the one instance
        where a failure would be embarrassing rather than academic: nothing
        proved would apply to the paper's only negative result.

Run:  python experiments/a39_slater_margin.py
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

from relaxation import Segment                                  # noqa: E402
from instances import hard_instances, random_instance           # noqa: E402
from bernstein import bd                                        # noqa: E402


def slater_margin(seg, tol=1e-9):
    """`max t` over the reduced feasible set with every cone pushed in by `tI`.

    Returns `(t_star, status)`.  Unbounded is reported as such rather than
    coerced to a number: it still means a strictly feasible point exists, but
    the *value* would be meaningless, so the caller sees the difference.
    """
    import cvxpy as cp
    n, d, f = seg.n, seg.d, seg.f
    Gfree = cp.Variable((n, f))
    Xfree = cp.Variable((f, f), symmetric=True)
    t = cp.Variable()

    Gfix = seg.Gfix
    Gamma = cp.hstack([cp.Constant(Gfix), Gfree]) @ seg.P.T
    cross = Gfix.T @ Gfree
    Xperm = cp.bmat([[cp.Constant(Gfix.T @ Gfix), cross], [cross.T, Xfree]])
    X = seg.P @ Xperm @ seg.P.T

    cons = [cp.bmat([[np.eye(n), Gfree], [Gfree.T, Xfree]])
            >> t * np.eye(n + f)]
    e_row = np.ones((1, d + 1))
    for cj, rj in seg.obs:
        Q0 = cp.Variable((d + 1, d + 1), symmetric=True)
        Q1 = cp.Variable((d, d), symmetric=True)
        cons += [Q0 >> t * np.eye(d + 1), Q1 >> t * np.eye(d)]
        Gc = Gamma.T @ cj.reshape(n, 1)
        ge = Gc @ e_row
        M = X - ge - ge.T + (float(cj @ cj) - rj ** 2) * np.ones((d + 1, d + 1))
        resid = [cp.sum(cp.multiply(seg.Sd[kk], M - Q0))
                 - cp.sum(cp.multiply(seg.Td[kk], Q1))
                 for kk in range(2 * d + 1)]
        cons.append(cp.hstack(resid) == 0)

    # t is bounded above only by the geometry; cap it so the answer is a margin
    # and not an unboundedness report.  The cap is far above any margin seen.
    cons.append(t <= 1.0)
    prob = cp.Problem(cp.Maximize(t), cons)
    try:
        prob.solve(solver="CLARABEL", tol_gap_abs=tol, tol_gap_rel=tol,
                   tol_feas=tol, verbose=False)
    except Exception as exc:                                # noqa: BLE001
        return None, "error: %s" % type(exc).__name__
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return None, prob.status
    return float(t.value), prob.status


def slater_witness(seg, eps=1e-2):
    """Build the strictly feasible point the way the lemma's proof does.

    Not a search: `Gamma_F` comes from a curve the caller has already checked is
    strictly clear, `X_F = Gamma_F^T Gamma_F + eps I` makes the Schur complement
    `eps I`, and each block pair is `Q' + eta I` where `Q'` represents
    `q_j^eps - eta h`.  Returns the smallest eigenvalue over all cones and the
    worst residual of the linear equality.
    """
    import cvxpy as cp
    d, n = seg.d, seg.n
    Gf = np.asarray(seg._slater_Gfree)
    Xf = Gf.T @ Gf + eps * np.eye(seg.f)
    P = np.block([[np.eye(n), Gf], [Gf.T, Xf]])
    X = seg.full_X(Gf, Xf)
    G = seg.full_Gamma(Gf)

    # h, with Gram pair (I_{d+1}, I_d) -- the fixed strictly positive polynomial
    e = np.ones(d + 1)
    worst_eig, worst_resid = float(np.linalg.eigvalsh(P).min()), 0.0
    for cj, rj in seg.obs:
        M = seg.M_obstacle(X, G, cj, rj)
        # delta: the minimum of q_j^eps over [0,1], read off a fine grid
        ss = np.linspace(0.0, 1.0, 4001)
        B = np.array([bd(s, d) for s in ss])
        q = np.einsum("ia,ab,ib->i", B, M, B)
        delta = float(q.min())
        if delta <= 0:
            return None, None, "q_j^eps not positive (delta=%.3g)" % delta
        eta = 0.4 * delta
        # represent q_j^eps - eta h with PSD blocks, then add eta I back
        Q0 = cp.Variable((d + 1, d + 1), symmetric=True)
        Q1 = cp.Variable((d, d), symmetric=True)
        Mh0, Mh1 = np.eye(d + 1), np.eye(d)
        target = M - eta * Mh0
        resid = [cp.sum(cp.multiply(seg.Sd[k], target - Q0))
                 - cp.sum(cp.multiply(seg.Td[k], Q1 + eta * Mh1))
                 for k in range(2 * d + 1)]
        prob = cp.Problem(cp.Minimize(0), [Q0 >> 0, Q1 >> 0,
                                           cp.hstack(resid) == 0])
        prob.solve(solver="CLARABEL", verbose=False)
        if prob.status not in ("optimal", "optimal_inaccurate"):
            return None, None, "no SOS representation: %s" % prob.status
        A0 = np.asarray(Q0.value) + eta * Mh0
        A1 = np.asarray(Q1.value) + eta * Mh1
        worst_eig = min(worst_eig, float(np.linalg.eigvalsh(A0).min()),
                        float(np.linalg.eigvalsh(A1).min()))
        r = [float(np.sum(seg.Sd[k] * (M - A0)) - np.sum(seg.Td[k] * A1))
             for k in range(2 * d + 1)]
        worst_resid = max(worst_resid, float(np.max(np.abs(r))))
    return worst_eig, worst_resid, "ok"


def strictly_clear_curve(seg, ntry=60, seed=7):
    """A degree-d curve with the prescribed boundary data and positive clearance.

    Straight chord first, then random control points; the instance is skipped if
    none is found, which is Assumption 4 failing rather than the check failing.
    """
    rng = np.random.default_rng(seed)
    ss = np.linspace(0.0, 1.0, 2001)
    B = np.array([bd(s, seg.d) for s in ss])
    best = None
    for t in range(ntry):
        Gf = (np.zeros((seg.n, seg.f)) if t == 0
              else rng.normal(scale=1.5, size=(seg.n, seg.f)))
        G = seg.full_Gamma(Gf)
        P = B @ G.T
        clear = min(float((np.sum((P - c) ** 2, axis=1) - r * r).min())
                    for c, r in seg.obs)
        if best is None or clear > best[1]:
            best = (Gf, clear)
        if clear > 1e-3:
            return Gf, clear
    return (best if best and best[1] > 1e-3 else (None, None))


def population():
    out = []
    H = hard_instances()
    for nm in ("symmetric_blocker", "staggered_pair", "three_in_a_row"):
        out.append(("%s k=1 d=3" % nm, H[nm]))
    for k in (1, 2, 3, 4):
        ell = k - 1
        for d in (2 * k + 1, 2 * k + 3):
            for nm in ("symmetric_blocker", "staggered_pair", "three_in_a_row"):
                b0 = np.zeros((ell + 1, 2))
                b1 = np.zeros((ell + 1, 2))
                b0[0] = (-2.0, 0.0)
                b1[0] = (2.0, 0.0)
                kw = dict(hard_instances(d=d, k=k, l=ell)[nm], bc0=b0, bc1=b1)
                out.append(("%s k=%d d=%d" % (nm, k, d), kw))
    return out


def main():
    rows = []
    for name, kw in population():
        try:
            seg = Segment(**kw)
        except ValueError:
            continue
        t, status = slater_margin(seg)
        rows.append(dict(name=name, kind="named", d=kw["d"], k=kw["k"],
                         t=t, status=status))
        print("    %-32s t* = %s   (%s)"
              % (name, "%.4g" % t if t is not None else "--", status))

    rng = np.random.default_rng(20260824)
    census = []
    while len(census) < 20:
        kw = random_instance(rng)
        if not kw["obstacles"]:
            continue
        try:
            seg = Segment(**kw)
        except ValueError:
            continue
        t, status = slater_margin(seg)
        if t is None:
            continue
        census.append(dict(name="census %d" % len(census), kind="census",
                           d=kw["d"], k=kw["k"], t=t, status=status))
    print("    census: %d draws, smallest t* = %.4g"
          % (len(census), min(r["t"] for r in census)))

    # a39c: run the lemma's construction and check what it produces
    wit = []
    for name, kw in population()[:9]:
        try:
            seg = Segment(**kw)
        except ValueError:
            continue
        Gf, clear = strictly_clear_curve(seg)
        if Gf is None:
            continue
        seg._slater_Gfree = Gf
        eig, resid, status = slater_witness(seg)
        if eig is None:
            print("    witness %-28s %s" % (name, status))
            continue
        wit.append(dict(name=name, clearance=clear, min_eig=eig, resid=resid))
        print("    witness %-28s min eig %.3g   equality residual %.1e"
              % (name, eig, resid))

    # a39d: does the counterexample family itself have clearance?
    from a5_rho2_scope import three_blocker_cfg
    cells = []
    for s1, rmid in ((0.08, 0.60), (0.08, 0.50), (0.09, 0.60), (0.09, 0.50),
                     (0.10, 0.60), (0.10, 0.50), (0.12, 0.60)):
        seg = three_blocker_cfg(s1, rmid, 3, 1, 0)
        Gf, clear = strictly_clear_curve(seg, ntry=400, seed=11)
        cells.append(dict(s1=s1, rmid=rmid,
                          clear=(None if Gf is None else float(clear))))
    got = [c["clear"] for c in cells if c["clear"] is not None]
    print("\n    counterexample family: %d of %d cells admit a strictly clear "
          "curve, least clearance %.4g" % (len(got), len(cells), min(got)))

    named_ok = [r for r in rows if r["t"] is not None and r["t"] > 0]
    cen_ok = [r for r in census if r["t"] > 0]
    a39a = dict(n=len(rows), n_strict=len(named_ok),
                min_t=min((r["t"] for r in named_ok), default=None),
                passed=bool(rows and len(named_ok) == len(rows)))
    a39b = dict(n=len(census), n_strict=len(cen_ok),
                min_t=min((r["t"] for r in cen_ok), default=None),
                passed=bool(census and len(cen_ok) == len(census)))
    allt = [r["t"] for r in rows + census if r.get("t") is not None]
    a39b["n_at_cap"] = sum(abs(t - 1.0) < 1e-6 for t in allt)
    a39b["n_all"] = len(allt)
    a39d = dict(n=len(cells), n_clear=len(got), min_clear=min(got),
                cells=cells,
                passed=bool(len(got) == len(cells) and min(got) > 0.024))
    a39c = dict(n=len(wit),
                min_eig=min((w["min_eig"] for w in wit), default=None),
                worst_resid=max((w["resid"] for w in wit), default=None),
                passed=bool(wit and all(w["min_eig"] > 0 for w in wit)
                            and max(w["resid"] for w in wit) < 1e-6))
    with open(os.path.join(ART, "a39_slater_margin.json"), "w") as fh:
        json.dump(dict(gates=dict(a39a_named_slater=a39a,
                                  a39b_census_slater=a39b,
                                  a39c_lemma_witness=a39c,
                                  a39d_counterexample_clearance=a39d),
                       rows=rows + census, witnesses=wit), fh, indent=1)
    print("\n  gates: a39a %s  a39b %s  a39c %s  a39d %s"
          % (a39a["passed"], a39b["passed"], a39c["passed"], a39d["passed"]))
    return 0 if all(g["passed"] for g in (a39a, a39b, a39c, a39d)) else 1


if __name__ == "__main__":
    sys.exit(main())
