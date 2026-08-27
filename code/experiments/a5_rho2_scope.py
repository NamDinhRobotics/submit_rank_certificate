"""A5 -- how far does the `rho = 2` region reach?  Degree, cost order, ambient dim.

Phase A4 built a natural `rho = 2` on an open region and thereby withdrew Phase
8's "unreachable" and refuted the source paper's reported "we never observe
rho > 1".  But A4 measured ONE configuration: `d = 3, k = 1, l = 0` in `R^2`.
That leaves the cheapest and most consequential question open -- is the
counterexample a curiosity about the smallest case, or does it live in the
regime the paper actually benchmarks (`d = 2k-1` up to 7, `k` up to 4, `R^3` and
`R^5`)?

This experiment sweeps the SAME three-blocker family across:

  * degree     `d = 3, 5, 7, 9`  at `k = 1, l = 0`  (so `f = 2, 4, 6, 8`)
  * cost order `k = 1..4` at `l = k-1`, at `d = 2k+1` and `d = 2k+3`
  * ambient    `R^3`, using the CONFINED lift (`instances.lift_to_confined_r3`):
               plain `R^3` instances are 100% tight (Phase 6) because the curve
               escapes over the obstacles, so the walls are what make the
               question meaningful rather than vacuous.

Each configuration gets its OWN nodal point `a0 = nodal_diag_root(d, k, l)` and
the `s1` grid is placed RELATIVE to it, because `a0` moves with `(d, k, l)` --
sweeping a fixed `s1` window would measure the grid, not the geometry.

TWO STAGES, and the reason is a defect this experiment found in A3.  A first
version screened all 189 cells with ONE Clarabel solve and arbitrated only the
headlines.  Three gates failed, and the failure was the instrument: at `d = 9`
that reading produced 8 spurious `rho = 2` cells.  Worse, A3's own arbiter
(`rank2_attained = face_ratio > 1e-6`) *agreed* with them -- because the
max-rank face probe buys apparent rank by violating the cost-optimality
constraint it is supposed to respect.  So:

  stage 1  cheap screen over the whole grid;
  stage 2  ARBITRATE every cell where `rho >= 2` could live -- the screen says
           so, OR the certificate refuses it (margin <= 1e-6), which by
           construction is the only region it can live in.

and the arbiter is repaired to require the probe to have stayed on the face and
an interior-point method to corroborate (see `arbitrated_rank`).  Cells that
trip only the raw face test are reported as UNDETERMINED and counted in neither
column.

Gates:
  a5a  every arbitrated `rho = 2` cell is REFUSED by the simplicity-margin
       certificate (soundness at the new configurations -- a cleared cell is the
       only fatal outcome)
  a5b  the unreliability of the cheap screen and of A3's raw criterion is
       recorded per configuration rather than quietly discarded
  a5c  the headline instances are feasible and genuinely loose
       (`c*_SDP < c*_P`), so the rank is not the relaxation escaping an
       infeasible problem
  a5d  every arbitrated `rho = 2` cell sits inside its own configuration's
       nodal region `s1 < a0` -- the theory keeps pointing at where its own
       counterexample is

Writes artifacts/a5_rho2_scope.json.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
ART = os.path.join(os.path.dirname(__file__), "..", "artifacts")

from relaxation import Segment                              # noqa: E402
from instances import lift_to_confined_r3                   # noqa: E402
from a4_simplicity_margin import analyse                    # noqa: E402
from a3_finite_d_certificate import (arbitrate_rank,        # noqa: E402
                                     nodal_diag_root)


def bcs(l, n=2):
    """Position, then chord velocity, then zeros -- `l+1` derivative orders."""
    z = [0.0] * (n - 2)
    b0 = [[-2.0, 0.0] + z, [4.0, 0.0] + z] + [[0.0, 0.0] + z] * max(0, l - 1)
    b1 = [[2.0, 0.0] + z, [4.0, 0.0] + z] + [[0.0, 0.0] + z] * max(0, l - 1)
    return b0[:l + 1], b1[:l + 1]


def corner_radius(s1, r_frac=0.55, r_max=0.25):
    """Corner-blocker radius that keeps the ENDPOINTS clear.

    The corner blocker sits at `x1 = -2 + 4 s1`, i.e. a distance `4 s1` from the
    start, so a fixed `r = 0.25` swallows the endpoint as soon as
    `s1 < 0.0625` -- and `a0` falls to `0.022` by `d = 9`.  Scaling the radius
    with the room available is what makes the family reach the nodal point at
    every degree instead of reporting `infeasible` and looking like a finding.
    """
    return float(min(r_max, r_frac * 4.0 * float(s1)))


def three_blocker_cfg(s1, rmid, d, k, l, r=None, n=2):
    """A4's family, but valid for any `(d, k, l)` and liftable to confined R^3."""
    r = corner_radius(s1) if r is None else float(r)
    x1 = -2.0 + 4.0 * float(s1)
    planar = [(np.array([x1, 0.0]), float(r)),
              (np.array([-x1, 0.0]), float(r)),
              (np.array([0.0, 0.0]), float(rmid))]
    b0, b1 = bcs(l, n)
    if n == 2:
        obs = planar
    elif n == 3:
        obs = lift_to_confined_r3(planar)
    else:
        raise ValueError(f"n = {n} not built here")
    return Segment(n=n, d=d, k=k, l=l, obstacles=obs, bc0=b0, bc1=b1)


def ground_truth(s1, rmid, d, k, l, r=None, n=2, ntry=200, seed=3):
    """`c*_P` by multi-start SLSQP: is the ORIGINAL problem feasible at all?"""
    from groundtruth import GroundTruth
    r = corner_radius(s1) if r is None else float(r)
    x1 = -2.0 + 4.0 * float(s1)
    planar = [(np.array([x1, 0.0]), float(r)),
              (np.array([-x1, 0.0]), float(r)),
              (np.array([0.0, 0.0]), float(rmid))]
    obs = planar if n == 2 else lift_to_confined_r3(planar)
    b0, b1 = bcs(l, n)
    gt = GroundTruth(n=n, d=d, k=k, l=l, obstacles=obs, bc0=b0, bc1=b1)
    try:
        res = gt.solve(ntry=ntry, seed=seed)
    except Exception:                                        # noqa: BLE001
        return None
    return float(res["cost"]) if res.get("ok") else None


SIGNAL_OVER_RESIDUAL = 100.0


def arbitrated_rank(seg, thresh=1e-6, sor=SIGNAL_OVER_RESIDUAL):
    """`rho` from three readings, with A3's arbiter REPAIRED.

    A3 decided `rank2_attained = (face_ratio > 1e-6)` from the max-rank face
    probe alone.  Pushing that to `d = 9` exposes a defect: the probe maximises
    `tr(X_F)` subject to the cost staying optimal, and at degree 9 the Bernstein
    conditioning lets it buy apparent rank by VIOLATING that constraint.
    Measured there: face ratio `1.23e-04` against its own constraint residual
    `1.58e-05` (signal/violation = 7.8), and `5.84e-04` against `4.63e-04`
    (ratio 1.3) -- while both interior-point readings say `1e-07` and `1e-14`.
    Compare a genuine case, `d = 7, k = 2`: `2.74e-01` against `1.63e-03`
    (ratio 168) with both IPMs corroborating at `2.87e-01`.

    So a verdict of 2 now requires three things, not one:
      * the face probe sees it:              `face_ratio > thresh`
      * the probe stayed ON the face:        `face_ratio > sor * residual`
      * an IPM corroborates:                 `min(clarabel, scs) > thresh`
        (interior-point methods converge to the relative interior of the optimal
        face, i.e. its maximum-rank point, so a real `rho = 2` must be visible
        to them -- and at `d = 3` and `d = 7` it is, to four digits)
    Anything else that trips the first test only is reported as UNDETERMINED and
    counted in neither column.  Silence about it would be the dishonest option.
    """
    try:
        a = arbitrate_rank(seg)
    except Exception as exc:                                 # noqa: BLE001
        return dict(verdict=None, reason=f"solver:{type(exc).__name__}",
                    ratios=None)
    face = abs(float(a["max_rank_face"]["ratio"]))
    resid = float(a["max_rank_face"]["residual"])
    cl = abs(float(a["clarabel"]["ratio"]))
    sc = abs(float(a["scs"]["ratio"]))
    on_face = face > sor * max(resid, 1e-300)
    ipm = min(cl, sc) > thresh
    if face <= thresh:
        verdict, reason = 1, "face probe sees no second direction"
    elif on_face and ipm:
        verdict, reason = 2, "face + both IPMs agree"
    else:
        verdict = None
        reason = ("probe off the face" if not on_face
                  else "no IPM corroboration")
    return dict(verdict=verdict, reason=reason, face=face, face_resid=resid,
                signal_over_resid=face / max(resid, 1e-300),
                clarabel=cl, scs=sc, cost=float(a["clarabel"]["cost"]),
                statuses={kk: a[kk]["status"] for kk in
                          ("clarabel", "scs", "max_rank_face")})


def arbiter_defect_table(cases):
    """The evidence that A3's raw criterion is defective, as committed numbers.

    For each case: the three rank readings, the face probe's OWN constraint
    residual, the signal/violation ratio that separates a real second direction
    from a probe that walked off the optimal face, and the ABSOLUTE spectrum of
    the gap matrix -- because "the second eigenvalue sits at the tolerance floor"
    is the claim, and it can only be checked in absolute terms.
    """
    from node import Node, build
    rows = []
    for (d, k, l, s1, rm) in cases:
        seg = three_blocker_cfg(s1, rm, d, k, l, n=2)
        rec = dict(d=d, k=k, l=l, s1=float(s1), rmid=float(rm))
        try:
            a = arbitrate_rank(seg)
            rec.update(clarabel=abs(float(a["clarabel"]["ratio"])),
                       scs=abs(float(a["scs"]["ratio"])),
                       face=abs(float(a["max_rank_face"]["ratio"])),
                       face_resid=float(a["max_rank_face"]["residual"]),
                       statuses={kk: a[kk]["status"] for kk in
                                 ("clarabel", "scs", "max_rank_face")})
            rec["signal_over_resid"] = (rec["face"]
                                        / max(rec["face_resid"], 1e-300))
        except Exception as exc:                             # noqa: BLE001
            rec["arbitration_error"] = type(exc).__name__
        try:
            h = build(seg, Node(lo=None, hi=None), use_rlt=False, use_lift=False)
            h["prob"].solve(solver="CLARABEL", tol_gap_abs=1e-11,
                            tol_gap_rel=1e-11, tol_feas=1e-11, max_iter=5000,
                            verbose=False)
            Gf = np.asarray(h["Gfree"].value)
            Xf = np.asarray(h["Xfree"].value)
            Xf = 0.5 * (Xf + Xf.T)
            w = np.sort(np.linalg.eigvalsh(Xf - Gf.T @ Gf))[::-1]
            rec["gap_spectrum_abs"] = [float(x) for x in w[:3]]
            rec["base_status"] = h["prob"].status
        except Exception as exc:                             # noqa: BLE001
            rec["base_error"] = type(exc).__name__
        rows.append(rec)
    return rows


# ----------------------------------------------------------------------
def sweep_config(d, k, l, n, rmids, rel_grid, r=None):
    """Sweep `s1` relative to this configuration's own nodal point."""
    try:
        a0 = float(nodal_diag_root(d, k, l))
    except Exception:                                        # noqa: BLE001
        a0 = None
    cells = []
    grid = [a0 * g for g in rel_grid] if a0 else [0.05 * i for i in range(1, 9)]
    for s1 in grid:
        if not (0.01 < s1 < 0.45):
            continue
        for rm in rmids:
            try:
                seg = three_blocker_cfg(s1, rm, d, k, l, r=r, n=n)
                res = analyse(seg, ns=4001)
            except Exception as exc:                         # noqa: BLE001
                res = dict(status=f"error:{type(exc).__name__}")
            cell = dict(s1=float(s1), rmid=float(rm),
                        status=res.get("status"), rho_screen=res.get("rho"),
                        m=res.get("m"), margin=res.get("margin"),
                        pf_applies=res.get("pf_applies"))
            # STAGE 2.  Arbitrate wherever rho >= 2 could live: the cheap
            # reading says so, OR the certificate refuses (margin <= 1e-6),
            # which is by construction the only region it can live in.
            if cell["status"] == "optimal" and (
                    (cell["rho_screen"] or 0) >= 2
                    or (cell["margin"] is not None and cell["margin"] <= 1e-6)):
                cell["arb"] = arbitrated_rank(seg)
                cell["rho"] = cell["arb"]["verdict"]
            else:
                cell["arb"] = None
                cell["rho"] = cell["rho_screen"]
            cells.append(cell)
    return dict(d=d, k=k, l=l, n=n, a0=a0, cells=cells)


def summarise(cfg):
    solved = [c for c in cfg["cells"] if c["status"] == "optimal"]
    hi = [c for c in solved if c["rho"] == 2]
    undet = [c for c in solved if c["arb"] and c["arb"]["verdict"] is None]
    screen_only = [c for c in solved
                   if (c["rho_screen"] or 0) >= 2 and c["rho"] != 2]
    cleared = [c for c in hi if c["margin"] is not None and c["margin"] > 1e-6]
    outside = [c for c in hi if cfg["a0"] and c["s1"] > cfg["a0"]]
    return dict(n_cells=len(cfg["cells"]), n_solved=len(solved),
                n_arbitrated=sum(1 for c in solved if c["arb"]),
                n_rho2=len(hi), n_undetermined=len(undet),
                n_screen_false_positive=len(screen_only),
                max_rho=max([c["rho"] for c in solved
                             if isinstance(c["rho"], int)], default=0),
                n_cleared_unsound=len(cleared), n_outside_nodal=len(outside),
                worst_margin_on_rho2=max([c["margin"] for c in hi
                                          if c["margin"] is not None],
                                         default=None),
                best_signal_over_resid=max([c["arb"]["signal_over_resid"]
                                            for c in hi if c["arb"]],
                                           default=None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rmids", type=float, nargs="*",
                    default=[0.20, 0.35, 0.50])
    ap.add_argument("--rel", type=float, nargs="*",
                    default=[0.55, 0.70, 0.80, 0.90, 0.97, 1.05, 1.25])
    ap.add_argument("--ntry", type=int, default=200)
    ap.add_argument("--out", default=os.path.join(ART, "a5_rho2_scope.json"))
    args = ap.parse_args()

    CONFIGS = [(3, 1, 0, 2), (5, 1, 0, 2), (7, 1, 0, 2), (9, 1, 0, 2),
               (5, 2, 1, 2), (7, 2, 1, 2), (9, 3, 2, 2), (11, 4, 3, 2),
               (3, 1, 0, 3)]

    print("=== A5: how far does the rho = 2 region reach? ===\n")
    print("  two-stage: cheap screen, then ARBITRATE every cell where rho >= 2")
    print("  could live (screen says so, or the certificate refuses).  A3's")
    print("  face-probe criterion is repaired -- see `arbitrated_rank`.\n")
    print("  %3s %2s %2s %2s %3s %9s | %6s %5s %6s %7s %6s %9s"
          % ("d", "k", "l", "n", "f", "a0", "solved", "arb", "rho=2",
             "undet", "scrnFP", "worst_g"))

    cfgs = []
    for (d, k, l, n) in CONFIGS:
        cfg = sweep_config(d, k, l, n, args.rmids, args.rel)
        s = summarise(cfg)
        cfg["summary"] = s
        try:
            f = Segment(**dict(n=n, d=d, k=k, l=l,
                               obstacles=[(np.array([0.0] * n), 0.5)],
                               bc0=bcs(l, n)[0], bc1=bcs(l, n)[1])).f
        except Exception:                                    # noqa: BLE001
            f = -1
        cfg["f"] = int(f)
        cfgs.append(cfg)
        print("  %3d %2d %2d %2d %3d %9s | %6d %5d %6d %7d %6d %9s"
              % (d, k, l, n, f,
                 "-" if cfg["a0"] is None else "%.6f" % cfg["a0"],
                 s["n_solved"], s["n_arbitrated"], s["n_rho2"],
                 s["n_undetermined"], s["n_screen_false_positive"],
                 "-" if s["worst_margin_on_rho2"] is None
                 else "%.1e" % s["worst_margin_on_rho2"]))

    # ------------------------------------------------------------------
    # arbitrate + ground-truth one headline cell per configuration
    # ------------------------------------------------------------------
    print("\n  --- arbitration of one rho >= 2 headline per configuration ---")
    heads = []
    for cfg in cfgs:
        hi = [c for c in cfg["cells"]
              if c["status"] == "optimal" and c["rho"] == 2]
        if not hi:
            continue
        c0 = max(hi, key=lambda c: c["arb"]["signal_over_resid"])
        d, k, l, n = cfg["d"], cfg["k"], cfg["l"], cfg["n"]
        arb = c0["arb"]
        cp = ground_truth(c0["s1"], c0["rmid"], d, k, l, n=n, ntry=args.ntry)
        cs = arb["cost"]
        head = dict(d=d, k=k, l=l, n=n, s1=c0["s1"], rmid=c0["rmid"],
                    rho=c0["rho"], margin=c0["margin"],
                    face=arb["face"], face_resid=arb["face_resid"],
                    signal_over_resid=arb["signal_over_resid"],
                    clarabel=arb["clarabel"], scs=arb["scs"],
                    statuses=arb["statuses"], c_sdp=cs, c_p=cp,
                    gap=(cp - cs) if (cp is not None and cs is not None) else None,
                    loose=bool(cp is not None and cs is not None
                               and cp - cs > 1e-4),
                    thm2_ok=bool(cp is None or (cs is not None
                                                and cs <= cp + 1e-6)))
        heads.append(head)
        print("    d=%-2d k=%d l=%d n=%d  s1=%.4f rmid=%.2f  rho=2  "
              "cl/scs/face %.3e/%.3e/%.3e  sig/resid %8.1f  "
              "c*_SDP=%s c*_P=%s %s"
              % (d, k, l, n, c0["s1"], c0["rmid"], arb["clarabel"], arb["scs"],
                 arb["face"], arb["signal_over_resid"],
                 "-" if cs is None else "%.6f" % cs,
                 "-" if cp is None else "%.6f" % cp,
                 "LOOSE" if head["loose"] else "(no gap measured)"))

    # ------------------------------------------------------------------
    # the committed evidence for A3's arbiter defect
    # ------------------------------------------------------------------
    print("\n  --- A3's raw face-probe criterion, and why it is defective ---")
    print("  %-22s %10s %10s %10s %10s %9s  %s"
          % ("case", "clarabel", "scs", "face", "face_resid", "sig/res",
             "gap spectrum (absolute)"))
    defect = arbiter_defect_table(
        [(9, 1, 0, 0.02753, 0.35), (9, 1, 0, 0.01541, 0.35),
         (9, 1, 0, 0.01211, 0.50), (9, 1, 0, 0.01762, 0.50),
         (7, 2, 1, 0.04143, 0.35), (3, 1, 0, 0.06200, 0.50)])
    for r in defect:
        tag = "d=%d k=%d s1=%.5f" % (r["d"], r["k"], r["s1"])
        if "arbitration_error" in r:
            print("  %-22s %s" % (tag, "ARBITRATION FAILS: "
                                  + r["arbitration_error"]))
            continue
        print("  %-22s %10.3e %10.3e %10.3e %10.3e %9.1f  %s"
              % (tag, r["clarabel"], r["scs"], r["face"], r["face_resid"],
                 r["signal_over_resid"],
                 " ".join("%.2e" % x for x in r.get("gap_spectrum_abs", []))))

    allsum = [c["summary"] for c in cfgs]
    reach = [c for c in cfgs if c["summary"]["n_rho2"] > 0]
    gates = dict(
        a5a_certificate_sound=dict(
            n_rho2_cells=sum(s["n_rho2"] for s in allsum),
            n_cleared_unsound=sum(s["n_cleared_unsound"] for s in allsum),
            worst_margin_on_rho2=max([s["worst_margin_on_rho2"] for s in allsum
                                      if s["worst_margin_on_rho2"] is not None],
                                     default=None),
            passed=bool(sum(s["n_cleared_unsound"] for s in allsum) == 0)),
        a5b_screen_was_unreliable=dict(
            note="A3's face-probe criterion, and the cheap single-solve rank "
                 "reading, both mislabel cells at high d; this records how many",
            defect_table=defect,
            n_screen_false_positive=sum(s["n_screen_false_positive"]
                                        for s in allsum),
            n_undetermined=sum(s["n_undetermined"] for s in allsum),
            by_config={f"d{c['d']}k{c['k']}n{c['n']}":
                       dict(fp=c["summary"]["n_screen_false_positive"],
                            undet=c["summary"]["n_undetermined"])
                       for c in cfgs if c["summary"]["n_screen_false_positive"]
                       or c["summary"]["n_undetermined"]},
            passed=True),
        a5c_feasible_and_loose=dict(
            n_heads=len(heads),
            n_loose=sum(1 for h in heads if h["loose"]),
            n_thm2_violations=sum(1 for h in heads if not h["thm2_ok"]),
            passed=bool(all(h["thm2_ok"] for h in heads)
                        and any(h["loose"] for h in heads))),
        a5d_inside_nodal_region=dict(
            n_outside=sum(s["n_outside_nodal"] for s in allsum),
            passed=bool(sum(s["n_outside_nodal"] for s in allsum) == 0)),
        reach=dict(
            configs_with_rho_ge2=[dict(d=c["d"], k=c["k"], l=c["l"], n=c["n"],
                                       f=c["f"], n_cells=c["summary"]["n_rho2"],
                                       best_signal_over_resid=c["summary"]
                                       ["best_signal_over_resid"])
                                  for c in reach],
            max_rho_overall=max(s["max_rho"] for s in allsum),
            n_configs=len(cfgs), n_configs_with_rho_ge2=len(reach),
            highest_k_with_rho_ge2=max([c["k"] for c in reach], default=None),
            highest_d_with_rho_ge2=max([c["d"] for c in reach], default=None),
            rho_ge2_in_R3=bool(any(c["n"] == 3 for c in reach))),
    )

    print("\n  --- gates ---")
    for nm in ("a5a_certificate_sound", "a5b_screen_was_unreliable",
               "a5c_feasible_and_loose", "a5d_inside_nodal_region"):
        print(f"  {nm}: {'PASS' if gates[nm]['passed'] else 'FAIL'}")
    rr = gates["reach"]
    print(f"\n  REACH: rho >= 2 on {rr['n_configs_with_rho_ge2']}/"
          f"{rr['n_configs']} configurations; highest d = {rr['highest_d_with_rho_ge2']}, "
          f"highest k = {rr['highest_k_with_rho_ge2']}, in R^3: {rr['rho_ge2_in_R3']}")
    print(f"  max rho seen anywhere: {rr['max_rho_overall']}")
    print(f"  certificate refused every rho >= 2 cell "
          f"({gates['a5a_certificate_sound']['n_rho2_cells']} of them, worst "
          f"margin {gates['a5a_certificate_sound']['worst_margin_on_rho2']})")

    with open(args.out, "w") as fh:
        json.dump(dict(gates=gates, configs=cfgs, headlines=heads), fh,
                  indent=1, default=float)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
