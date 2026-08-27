"""A4 -- the certificate never needed positivity, and the corner is covered.

Phase A3 stated Layer 3c as "`Gr W` entrywise positive `=>` (Perron-Frobenius)
`=>` top eigenvalue simple `=>` `rho <= 1`", and then measured the hypothesis
FAILING when two contacts sit near opposite ends of `[0,1]` -- with `rho = 1`
surviving there anyway.  That gap was an artefact of how the certificate was
phrased, not a gap in the mathematics.

THE SHARPENING.  Layers 1-2 already give, with no hypotheses beyond `K > 0`,

    rho  <=  dim ker Z  =  multiplicity of the top eigenvalue of  W^{1/2} Gr W^{1/2}

(A3's `a3b`, verified 53/53).  So

    simplicity margin  g := (mu_1 - mu_2) / mu_1  >  0     =>     rho <= 1,

and contrapositively `rho >= 2` forces `g = 0`.  That is a **complete**
certificate, not merely a sufficient one, and it is computable: `Gr` is `m x m`
with `m <= 3`, so `g` costs one tiny eigenvalue problem.  Perron-Frobenius was
one *route* to proving `g > 0` without computing it -- and positivity is what
that route needs, not what the conclusion needs.

Consequences, all measured here:

  * **the corner is certified after all.**  Inside it `Gr_12 < 0`, which
    separates the two eigenvalues exactly as well as `Gr_12 > 0` does.  A3's
    "uncovered" instances were never uncovered; they were only un-provable by
    the particular sufficient condition A3 quoted.
  * **for `m <= 2` no eigenvalue problem is needed either.**  A `2 x 2` symmetric
    matrix has a degenerate top eigenvalue iff it is a multiple of `I`, so

        m = 1:  g > 0 always                                    (unconditional)
        m = 2:  g = 0  iff  Gr_12 = 0  AND  w_1 Gr_11 = w_2 Gr_22

    -- two scalar equations, i.e. codimension 2, stated without any positivity.
    The census below reports how much of the population that covers.
  * **what is actually still open** is narrower than "the corner": it is proving
    `g > 0` *a priori*, without solving the instance.  For `m <= 2` that is the
    codim-2 statement; for `m >= 3` positivity is still the only a priori route
    we have, so the honest residual is `m >= 3` inside the corner -- and a4d
    goes looking for it.

>>> AND FINDS IT.  a4e turns the residual into a COUNTEREXAMPLE. <<<

Three blockers -- two placed inside the corner at `s1, 1-s1` and one at the
middle -- give `m = 3` with a contact pair the certificate cannot clear, and
there `rho = 2`.  Not a knife edge: an OPEN region, `s1 in [0.08, 0.10]` across
every middle radius tried, and its boundary sits at the nodal point
`a0 = 0.1127` that Phase A3 computed.  Verified three ways (Clarabel, SCS and
the maximum-rank face probe agree to 4 digits, second eigenvalue ratio `6.3%`
to `29.4%` -- not a noise floor), stable across conic tolerances `1e-8..1e-12`,
Theorem 3's lifted curve exact to `4e-14`, and the ORIGINAL problem is feasible
(`c*_P = 18.528` from 121 successful starts) with a real gap, so this is a
loose instance and not a relaxation escaping an infeasible one.

What it overturns, and what it does not:

  * `rho > 1` is NOT unreachable.  Phase 8 hunted it over 32 engineered probes
    and 300 random instances and concluded it was unreachable; that conclusion
    was a limit of the search, not of the geometry.  The difference here is that
    A3's nodal analysis said exactly where to look.
  * their **Lemma 4 (`rho <= f`) is TIGHT**: here `f = 2` and `rho = 2`.
  * Phase 12's `rho <= 2k-(l+1)` gives `1` at `k=1, l=0` and is **exceeded**.
    No contradiction -- that bound is a CONTINUUM statement and this is finite
    `d` -- but it does settle that the continuum bound does not transfer, which
    until now was an open reading.
  * the certificate is **not** damaged: it REFUSES every one of these instances
    (margin `<= 2e-10`).  Sound, complete, and its negative region is exactly
    where the counterexamples live.  The mechanism is visible in the dual: there
    `M_lambda = K`, i.e. `Z = 0`, so complementarity constrains the rank not at
    all and the relaxation uses all `f` dimensions.

Gates:
  a4a  `g > 0  =>  rho <= 1` on the whole population, with the margin reported;
       and `g` is checked against `dim ker Z` so the two agree instance by instance
  a4b  the `m = 2` criterion is EQUIVALENT to `g = 0` (tested on constructed
       matrices, both directions, including the near-degenerate ones)
  a4c  census of `m` over a wide random population: how far does `m <= 2` reach?
  a4d  the corner is certified by `g` for `m <= 2` -- the two-blocker family --
       even where Perron-Frobenius is inapplicable
  a4e  the residual is non-empty: an open region of `rho = 2`, every cell of it
       inside the nodal region and every cell REFUSED by the certificate (an
       instance cleared by the certificate yet carrying `rho >= 2` would be the
       one fatal outcome, and there are none).  The witness cell additionally
       has to survive the two ways this could still be an artifact: the rank
       reading must not move when the conic tolerance sweeps `1e-8` to `1e-12`
       (a stopping floor would), and the Theorem-3 lift must close (clearance
       `>= 0`), or the `rho = 2` reading describes nothing.

Writes artifacts/a4_simplicity_margin.json.
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

from relaxation import Segment                              # noqa: E402
from node import Node, build                                # noqa: E402
from bernstein import bd                                    # noqa: E402
from instances import hard_instances, random_instance       # noqa: E402
from a3_finite_d_certificate import (green_matrix, refine_contact,  # noqa: E402
                                     gr_entry_norm, corner_box, bcs,
                                     two_blocker)


# ----------------------------------------------------------------------
def margins(Gr, w):
    """`(g, off, bal)` -- the simplicity margin and, for `m = 2`, the two
    scalars whose simultaneous vanishing is the ONLY way `g` can vanish."""
    Wh = np.diag(np.sqrt(np.maximum(w, 0.0)))
    mu = np.sort(np.linalg.eigvalsh(Wh @ Gr @ Wh))[::-1]
    g = float((mu[0] - mu[1]) / mu[0]) if mu.size > 1 and mu[0] > 0 else 1.0
    off = bal = None
    if Gr.shape[0] == 2:
        off = float(abs(Gr[0, 1]) / np.sqrt(abs(Gr[0, 0] * Gr[1, 1])))
        a, b = w[0] * Gr[0, 0], w[1] * Gr[1, 1]
        bal = float(abs(a - b) / max(abs(a), abs(b), 1e-300))
    return g, off, bal


def analyse(seg, contact_tol=1e-4, ns=4001, tol=1e-10):
    from scipy.optimize import nnls
    h = build(seg, Node(lo=None, hi=None), use_rlt=False, use_lift=False)
    h["prob"].solve(solver="CLARABEL", tol_gap_abs=tol, tol_gap_rel=tol,
                    tol_feas=tol, max_iter=2000, verbose=False)
    if h["prob"].status != "optimal" or h["Gfree"].value is None:
        return dict(status=h["prob"].status)
    Gf = np.asarray(h["Gfree"].value)
    Xf = 0.5 * (np.asarray(h["Xfree"].value) + np.asarray(h["Xfree"].value).T)
    wS = np.linalg.eigvalsh(Xf - Gf.T @ Gf)
    rho = int(np.sum(wS > 1e-6 * max(1.0, float(np.abs(wS).max()))))
    if rho < 1:
        return dict(status="tight")

    Z = 0.5 * (np.asarray(h["cons"][0].dual_value)
               + np.asarray(h["cons"][0].dual_value).T)[seg.n:, seg.n:]
    K, Ki, F = green_matrix(seg.d, seg.k, seg.l)
    M = K - Z

    res = seg._package(Gf, Xf, float(h["prob"].value))
    V = seg.lifted_V(res)
    ss = np.linspace(0.0, 1.0, ns)
    B = np.array([bd(s, seg.d) for s in ss])
    P = B @ res["Gamma"].T
    vv = np.sum((B @ V.T) ** 2, axis=1) if V.size else np.zeros(ns)
    zeta = np.full(ns, np.inf)
    for c, r in seg.obs:
        zeta = np.minimum(zeta, np.sum((P - c) ** 2, axis=1) + vv - r * r)
    grid = [float(ss[i]) for i in range(1, ns - 1)
            if zeta[i] <= zeta[i - 1] and zeta[i] <= zeta[i + 1]
            and zeta[i] < contact_tol]
    if not grid:
        return dict(status="no contact detected")
    fine = [refine_contact(seg, res["Gamma"], V, s0, 3.0 / (ns - 1))[0]
            for s0 in grid]

    U = np.array([bd(s, seg.d)[F] for s in fine]).T
    A = np.array([np.outer(U[:, a], U[:, a]).ravel()
                  for a in range(U.shape[1])]).T
    w, rn = nnls(A, M.ravel())
    Gr = U.T @ Ki @ U
    g, off, bal = margins(Gr, w)

    wZ = np.linalg.eigvalsh(Z)
    ker = int(np.sum(np.abs(wZ) < 1e-6 * max(1.0, float(np.abs(wZ).max()))))
    GW = Gr @ np.diag(w)
    return dict(status="optimal", d=int(seg.d), k=int(seg.k), l=int(seg.l),
                f=int(seg.f), rho=rho, m=len(fine), contacts=fine,
                atom_fit=float(rn / max(1e-12, np.linalg.norm(M))),
                ker_dim=ker, margin=g, off_diag=off, weight_balance=bal,
                min_GW_entry=float(GW.min()),
                pf_applies=bool(GW.min() > 0),
                weights=[float(x) for x in w])


# ----------------------------------------------------------------------
def population_named():
    out = []
    H = hard_instances()
    for nm in ("symmetric_blocker", "staggered_pair", "three_in_a_row"):
        out.append((f"{nm} k=1 d=3", H[nm]))
    for k in (1, 2, 3, 4):
        l = k - 1
        for extra in (0, 2):
            d = 2 * k + 1 + extra
            b0, b1 = bcs(l)
            for nm in ("symmetric_blocker", "staggered_pair", "three_in_a_row"):
                out.append((f"{nm} k={k} d={d}",
                            dict(hard_instances(d=d, k=k, l=l)[nm],
                                 bc0=b0, bc1=b1)))
    recs = json.load(open(os.path.join(ART, "gate1.json")))["records"]
    for r in recs:
        if r.get("skipped") or r["rho"] < 1:
            continue
        obs = [(np.array([o[0], o[1]]), float(o[2])) for o in r["obstacles"]]
        out.append((f"gate1[{r['i']}]",
                    dict(n=2, d=3, k=1, l=0, obstacles=obs,
                         bc0=[[-2.0, 0.0]], bc1=[[2.0, 0.0]])))
    return out


def census(n_draw, seed=20260806):
    """How far does `m <= 2` reach?  Loose instances only -- `rho = 0` has no
    contact structure to speak of, so it would pad the count for free."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_draw):
        kw = random_instance(rng)
        if not kw["obstacles"]:
            continue
        try:
            r = analyse(Segment(**kw))
        except Exception:                                   # noqa: BLE001
            continue
        if r.get("status") != "optimal":
            continue
        rows.append(dict(i=i, m=r["m"], rho=r["rho"], margin=r["margin"],
                         off_diag=r["off_diag"], pf_applies=r["pf_applies"],
                         min_GW_entry=r["min_GW_entry"],
                         contacts=r["contacts"]))
    return rows


S1S = (0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.18, 0.25, 0.35)
RMIDS = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60)


def _arbitrate(seg):
    """Three independent rank readings; see `a3_finite_d_certificate`."""
    from a3_finite_d_certificate import arbitrate_rank
    out = arbitrate_rank(seg)
    out["cost"] = out["clarabel"]["cost"]
    return out


def _ground_truth(s1, rmid, r=0.25, ntry=200, seed=3):
    """`c*_P` by multi-start SLSQP -- is the ORIGINAL problem even feasible?

    Without this the whole finding could be "the relaxation escapes into extra
    dimensions because no degree-`d` curve exists at all", which would be a
    different and much weaker claim.
    """
    from groundtruth import GroundTruth
    x1 = -2.0 + 4.0 * float(s1)
    gt = GroundTruth(n=2, d=3, k=1, l=0,
                     obstacles=[(np.array([x1, 0.0]), float(r)),
                                (np.array([-x1, 0.0]), float(r)),
                                (np.array([0.0, 0.0]), float(rmid))],
                     bc0=[[-2.0, 0.0]], bc1=[[2.0, 0.0]])
    res = gt.solve(ntry=ntry, seed=seed)
    # the RUN COUNT is part of the claim, not bookkeeping: "c*_P = 19.5889"
    # means "the best of this many independent local solves", and a reader
    # cannot judge that without knowing how many succeeded.
    return dict(cost=float(res["cost"]) if res.get("ok") else None,
                n_try=int(ntry), n_success=int(res.get("n_success", 0)),
                n_distinct=len(res.get("minima", [])))


def _solve_cell(seg, tol=1e-11):
    """One Clarabel solve of a cell, packaged so `Segment` can be asked about
    the lift.  Same call as `arbitrate_rank`'s first reading."""
    h = build(seg, Node(lo=None, hi=None), use_rlt=False, use_lift=False)
    h["prob"].solve(solver="CLARABEL", tol_gap_abs=tol, tol_gap_rel=tol,
                    tol_feas=tol, max_iter=5000, verbose=False)
    Gf = np.asarray(h["Gfree"].value)
    Xf = np.asarray(h["Xfree"].value)
    return seg._package(Gf, 0.5 * (Xf + Xf.T), float(h["prob"].value)), h


def _tol_sweep(seg, tols=(1e-8, 1e-9, 1e-10, 1e-11, 1e-12)):
    """Is the second eigenvalue a RANK or a stopping artifact?

    The distinction is decidable and this is how: a ratio that TRACKS the conic
    tolerance is a floor -- it is wherever the solver happened to stop -- while
    one that sits still as the tolerance moves four decades is a property of
    the optimum.  A3 ran this test on a cell that turned out to be spurious and
    watched the ratio fall with the tolerance; the same test is owed to the
    cell the counterexample rests on.
    """
    out = []
    for tol in tols:
        res, h = _solve_cell(seg, tol=tol)
        w = np.sort(np.linalg.eigvalsh(res["gap"]))[::-1]
        out.append(dict(tol=float(tol), status=h["prob"].status,
                        ratio=float(w[1] / w[0]) if w.size > 1 else 0.0))
    return out


def _lift_clearance(seg):
    """Theorem 3 of the source paper is a CHECKABLE claim about this instance:
    the relaxed solution, lifted into `rho` extra dimensions, is an exact
    optimum -- so the lifted curve must clear every obstacle.  Report the worst
    clearance over the curve and the obstacles; negative would mean the lift
    does not close and the whole reading is void."""
    res, _ = _solve_cell(seg)
    return float(seg.lifted_clearance(res))


def three_blocker(s1, r=0.25, rmid=0.25, d=3, k=1, l=0):
    """Three blockers: two in the corners, one in the middle -- the shape that
    would put `m >= 3` INSIDE the corner, where neither the `m <= 2` criterion
    nor Perron-Frobenius has anything to say a priori."""
    x1 = -2.0 + 4.0 * float(s1)
    return Segment(n=2, d=d, k=k, l=l,
                   obstacles=[(np.array([x1, 0.0]), float(r)),
                              (np.array([-x1, 0.0]), float(r)),
                              (np.array([0.0, 0.0]), float(rmid))],
                   bc0=[[-2.0, 0.0]], bc1=[[2.0, 0.0]])


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=600)
    ap.add_argument("--out", default=os.path.join(ART, "a4_simplicity_margin.json"))
    args = ap.parse_args()

    print("=== A4: the simplicity margin is the certificate; positivity is a "
          "route to it ===")

    # ---- a4a ---------------------------------------------------------
    print("\n  --- a4a: g > 0 => rho <= 1, on the named population ---")
    print("  %-24s %2s %2s %3s %2s %9s %9s %9s %5s"
          % ("instance", "k", "d", "rho", "m", "margin g", "off-diag",
             "balance", "PF?"))
    rows = []
    for name, kw in population_named():
        r = analyse(Segment(**kw))
        if r.get("status") != "optimal":
            continue
        rows.append(dict(name=name, **{k: v for k, v in r.items()
                                       if k != "status"}))
        print("  %-24s %2d %2d %3d %2d %9.4f %9s %9s %5s"
              % (name, r["k"], r["d"], r["rho"], r["m"], r["margin"],
                 "-" if r["off_diag"] is None else "%.3e" % r["off_diag"],
                 "-" if r["weight_balance"] is None
                 else "%.3e" % r["weight_balance"],
                 "yes" if r["pf_applies"] else "NO"))

    bad = [r["name"] for r in rows if r["margin"] > 0 and r["rho"] > 1]
    disagree = [r["name"] for r in rows
                if (r["ker_dim"] == 1) != (r["margin"] > 1e-6)]
    # Where the named population comes from, since the paper states the
    # recipe: fixed layouts swept over (k, d) plus the loose instances of the
    # reproduction benchmark.  Recorded so the split is auditable, not asserted.
    n_gate1 = sum(1 for r in rows if r["name"].startswith("gate1["))
    # Where the layout half of the population comes from, measured rather than
    # asserted -- the text quotes this arithmetic and it has been wrong twice.
    # NOTE the sweep's (k=1, extra=0) cell IS (k,d)=(1,3), so the three layouts
    # entered explicitly at (1,3) are byte-identical repeats of three swept
    # ones: 24 DISTINCT configurations, 27 entries.  Repeats cannot move a min
    # but can shift a median, so the split is recorded, not glossed.
    named = population_named()
    layout = [nm for nm, _ in named if not nm.startswith("gate1[")]
    n_distinct = len(set(layout))
    a4a = dict(n=len(rows), n_layout_sweep=len(rows) - n_gate1,
               n_from_gate1=n_gate1,
               n_layout_entries=len(layout),
               n_layout_distinct=n_distinct,
               n_duplicate_entries=len(layout) - n_distinct,
               n_violations=len(bad), violations=bad[:10],
               n_ker_disagreements=len(disagree), disagreements=disagree[:10],
               min_margin=float(min(r["margin"] for r in rows)),
               median_margin=float(np.median([r["margin"] for r in rows])),
               n_pf_applies=sum(1 for r in rows if r["pf_applies"]),
               passed=bool(not bad and not disagree))
    print(f"\n    margin > 0 on {sum(1 for r in rows if r['margin'] > 0)}"
          f"/{len(rows)}, min {a4a['min_margin']:.4f}, median "
          f"{a4a['median_margin']:.4f}; it agrees with dim ker Z on "
          f"{len(rows) - len(disagree)}/{len(rows)}")

    # ---- a4b ---------------------------------------------------------
    print("\n  --- a4b: for m = 2, g = 0 <=> (off-diagonal = 0 AND balanced) ---")
    rng = np.random.default_rng(7)
    checks, mism = 0, 0
    for _ in range(4000):
        a, b = rng.uniform(0.2, 3.0, 2)
        c = rng.choice([0.0, rng.uniform(-1.0, 1.0) * np.sqrt(a * b) * 0.95])
        if rng.random() < 0.25:
            b = a                                   # force the balanced case
        Gr = np.array([[a, c], [c, b]])
        if np.min(np.linalg.eigvalsh(Gr)) <= 0:
            continue
        w = np.array([1.0 / Gr[0, 0], 1.0 / Gr[1, 1]])
        if rng.random() < 0.5:
            w[1] *= rng.uniform(0.5, 1.5)           # and the unbalanced case
        g, off, bal = margins(Gr, w)
        degenerate = g < 1e-12
        criterion = (off < 1e-12) and (bal < 1e-12)
        checks += 1
        mism += int(degenerate != criterion)
    a4b = dict(n=checks, n_mismatch=mism, passed=bool(mism == 0))
    print(f"    {checks} constructed 2x2 cases, {mism} mismatches between "
          f"'g = 0' and 'off-diagonal = 0 and balanced'")

    # ---- a4c ---------------------------------------------------------
    print(f"\n  --- a4c: census of m over {args.draws} random draws ---")
    cen = census(args.draws)
    from collections import Counter
    cm = Counter(r["m"] for r in cen)
    n_le2 = sum(v for kk, v in cm.items() if kk <= 2)
    named_m = Counter(r["m"] for r in rows)
    a4c = dict(n_loose=len(cen), n_draws=int(args.draws),
               n_m_le_2=int(n_le2),
               by_m={str(kk): int(v) for kk, v in sorted(cm.items())},
               share_m_le_2=n_le2 / max(1, len(cen)),
               named_by_m={str(kk): int(v) for kk, v in sorted(named_m.items())},
               min_margin=float(min([r["margin"] for r in cen], default=1.0)),
               n_pf_fails=sum(1 for r in cen if not r["pf_applies"]),
               passed=bool(cen))
    print(f"    {len(cen)} loose instances; m distribution {dict(sorted(cm.items()))}"
          f" => m <= 2 on {100 * a4c['share_m_le_2']:.1f}%")
    print(f"    min margin over the census: {a4c['min_margin']:.4f}; "
          f"Perron-Frobenius inapplicable on {a4c['n_pf_fails']}")

    # ---- a4d ---------------------------------------------------------
    print("\n  --- a4d: the corner, and the hunt for m >= 3 inside it ---")
    A = corner_box(3, 1, 0)
    corner_rows = []
    for s1 in (0.080, 0.100, 0.110, 0.11270165):
        r = analyse(two_blocker(s1), ns=8001)
        if r.get("status") != "optimal":
            continue
        pair = (min(r["contacts"]), max(r["contacts"]))
        corner_rows.append(dict(kind="two_blocker", s1=float(s1), m=r["m"],
                                rho=r["rho"], margin=r["margin"],
                                off_diag=r["off_diag"],
                                gr12=gr_entry_norm(pair[0], pair[1], 3,
                                                   green_matrix(3, 1, 0)[1],
                                                   green_matrix(3, 1, 0)[2]),
                                pf_applies=r["pf_applies"],
                                contacts=r["contacts"]))
    print("    two blockers driven into the corner (A = %.4f):" % A)
    print("      %12s %2s %3s %11s %11s %6s" % ("s1", "m", "rho", "Gr12(norm)",
                                                "margin g", "PF?"))
    for r in corner_rows:
        print("      %12.8f %2d %3d %+11.3e %11.4f %6s"
              % (r["s1"], r["m"], r["rho"], r["gr12"], r["margin"],
                 "yes" if r["pf_applies"] else "NO"))

    # Soundness, plus the point of the phase: the margin must clear instances
    # that Perron-Frobenius cannot.  The last row is A3's exact nodal point,
    # where `Gr_12 -> 0` and the certificate REFUSES -- correctly, and a3g's
    # arbitration puts `rho = 1` there regardless.  Refusing is not a failure;
    # clearing an instance that carries `rho >= 2` would be.
    cleared = [r for r in corner_rows if r["margin"] > 1e-6]
    unsound_corner = [r for r in cleared if r["rho"] > 1]
    pf_fails_but_cleared = [r for r in cleared if not r["pf_applies"]]
    a4d = dict(A=float(A), corner=corner_rows,
               n_cleared=len(cleared), n_unsound=len(unsound_corner),
               n_pf_fails_but_cleared=len(pf_fails_but_cleared),
               corner_pf_fails=sum(1 for r in corner_rows
                                   if not r["pf_applies"]),
               refused=[r["s1"] for r in corner_rows if r["margin"] <= 1e-6],
               passed=bool(not unsound_corner and pf_fails_but_cleared))
    print(f"    {len(pf_fails_but_cleared)} of {len(corner_rows)} corner rows "
          f"are cleared by the MARGIN while Perron-Frobenius is inapplicable "
          f"-- that is the whole point; {len(unsound_corner)} unsound")
    print(f"    refused (margin = 0, the codim-2 nodal point itself): "
          f"{a4d['refused']} -- a3g arbitrates rho = 1 there")

    # ---- a4e: the residual is NOT empty -- a natural rho = 2 -------
    print("\n  --- a4e: mapping the residual (m = 3 with two corner contacts) ---")
    print("    %6s %s" % ("s1", "".join("%8.2f" % rm for rm in RMIDS)))
    phase, a0 = [], 0.11270165
    for s1 in S1S:
        row, cells = "", []
        for rm in RMIDS:
            try:
                r = analyse(three_blocker(s1, rmid=rm), ns=4001)
            except Exception:                                # noqa: BLE001
                r = dict(status="error")
            st = r.get("status")
            cells.append(dict(s1=float(s1), rmid=float(rm), status=st,
                              rho=r.get("rho"), m=r.get("m"),
                              margin=r.get("margin"),
                              pf_applies=r.get("pf_applies")))
            row += "%8s" % (("rho=%d" % r["rho"]) if st == "optimal"
                            else str(st)[:7])
        phase += cells
        print("    %6.2f %s" % (s1, row))
    solved = [c for c in phase if c["status"] == "optimal"]
    rho2 = [c for c in solved if c["rho"] and c["rho"] >= 2]

    # every rho >= 2 cell must (i) sit inside the nodal region s1 < a0 and
    # (ii) have been REFUSED by the certificate -- a certificate that cleared
    # one of these would be unsound, which is the only fatal outcome here
    unsound = [c for c in rho2 if c["margin"] is not None and c["margin"] > 1e-6]
    outside = [c for c in rho2 if c["s1"] > a0]

    verified = None
    if rho2:
        c0 = max(rho2, key=lambda c: c["rmid"])
        seg = three_blocker(c0["s1"], rmid=c0["rmid"])
        arb = _arbitrate(seg)
        gt = _ground_truth(c0["s1"], c0["rmid"])
        gtc = gt["cost"]
        verified = dict(s1=c0["s1"], rmid=c0["rmid"], arbitration=arb,
                        c_p=gtc, c_sdp=arb["cost"],
                        gap=(gtc - arb["cost"]) if gtc else None,
                        thm2_ok=bool(gtc and arb["cost"] <= gtc + 1e-6),
                        gt_n_try=gt["n_try"], gt_n_success=gt["n_success"],
                        gt_n_distinct=gt["n_distinct"],
                        tol_sweep=_tol_sweep(seg),
                        lift_clearance=_lift_clearance(seg))
        print(f"\n    VERIFIED at s1={c0['s1']:.2f}, rmid={c0['rmid']:.2f}: "
              f"three independent rank readings "
              f"{arb['clarabel']['ratio']:.3e} (Clarabel) / "
              f"{arb['scs']['ratio']:.3e} (SCS) / "
              f"{arb['max_rank_face']['ratio']:.3e} (max-rank face)")
        print(f"    c*_SDP = {arb['cost']:.8f}  <  c*_P = {gtc:.8f}  "
              f"(gap {gtc - arb['cost']:.2e}) -- feasible, loose, and rho = 2 = f")
        rr = [p["ratio"] for p in verified["tol_sweep"]]
        print(f"    tolerance sweep 1e-8..1e-12: ratio in "
              f"[{min(rr):.3e}, {max(rr):.3e}], spread {max(rr)/min(rr):.2f}x "
              f"against 1e4x of tolerance; lifted clearance "
              f"{verified['lift_clearance']:.2e}; c*_P is the best of "
              f"{verified['gt_n_success']}/{verified['gt_n_try']} successful "
              f"local solves ({verified['gt_n_distinct']} distinct minima)")

    # The counterexample stands on the second eigenvalue being a RANK and not a
    # place the solver stopped.  Criterion fixed before running: the tolerance
    # moves 1e4x, so a floor would move with it; a rank must not.  "Not" is
    # taken to be a spread below 10x -- two decades clear of the tolerance
    # sweep, and the a3g cell that WAS a floor moved five decades.  The lift
    # must also close (Thm 3), so the clearance may not be negative beyond the
    # conic tolerance it was solved to.
    sweep_ok = lift_ok = True
    if verified:
        rr = [p["ratio"] for p in verified["tol_sweep"]]
        sweep_ok = bool(rr and min(rr) > 0 and max(rr) / min(rr) < 10.0)
        lift_ok = bool(verified["lift_clearance"] >= -1e-9)
        verified["tol_spread"] = float(max(rr) / min(rr)) if rr else None
        verified["sweep_ok"] = sweep_ok
        verified["lift_ok"] = lift_ok

    a4e = dict(phase=phase, n_solved=len(solved), n_rho2=len(rho2),
               a0=a0, n_unsound=len(unsound), n_outside_nodal=len(outside),
               rho2_cells=[dict(s1=c["s1"], rmid=c["rmid"], rho=c["rho"],
                                m=c["m"], margin=c["margin"]) for c in rho2],
               verified=verified,
               max_margin_on_rho2=float(max([c["margin"] for c in rho2
                                             if c["margin"] is not None],
                                            default=0.0)),
               passed=bool(rho2 and not unsound and not outside
                           and sweep_ok and lift_ok))
    print(f"\n    rho >= 2 on {len(rho2)}/{len(solved)} solved cells, ALL with "
          f"s1 < a0 = {a0:.6f} and ALL refused by the certificate "
          f"(max margin on them {a4e['max_margin_on_rho2']:.2e})")

    gates = dict(a4a_margin_is_the_certificate=a4a,
                 a4b_m2_criterion_equivalent=a4b,
                 a4c_census_of_m=a4c,
                 a4d_corner_certified=a4d,
                 a4e_natural_rho2=a4e)
    print("\n  --- gates ---")
    for nm in gates:
        print(f"  {nm}: {'PASS' if gates[nm]['passed'] else 'FAIL'}")

    with open(args.out, "w") as fh:
        json.dump(dict(gates=gates, rows=rows, census=cen), fh, indent=1,
                  default=float)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
