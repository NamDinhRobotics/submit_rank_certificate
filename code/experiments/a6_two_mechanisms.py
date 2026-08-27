"""A6 -- `rho = 2` has TWO mechanisms, and which one you get is decided by
whether the contact count reaches the number of free control points.

Phase A4 explained its `rho = 2` region by `Z = 0`: the PSD block's multiplier
vanishes, complementarity constrains no rank at all, hence an OPEN region rather
than a codimension-2 coincidence.  Phase A5 then found `rho = 2` at `f = 4` as
well -- where that explanation cannot hold, and A1's bound says why:

    M_lambda = sum_a w_a u_a u_a^T   ==>   rank(M_lambda) <= m = #contacts,
    Z = K - M_lambda  with  K > 0  of size f
    ==>  m < f  forces  Z != 0  (rank Z >= f - m),   and   Z = 0  needs  m >= f.

Measured, on A5's own `rho = 2` cells: at `f = 2, m = 3` the ratio `||Z||/||K||`
is `1e-12`-`1e-9`; at `f = 4, m = 3` it is `0.43`-`0.46`.  So there are two
regimes and they are cleanly separated, not a spectrum:

    m >= f :  Z can vanish  ->  rho = f on an OPEN region      (A4's mechanism)
    m <  f :  Z != 0 always ->  rho = 2 needs a DOUBLE EIGENVALUE of the m x m
                                Gram, i.e. a degeneracy of positive codimension

The second regime raises its own question, and it is the point of this
experiment.  A codimension-2 set in a two-parameter sweep is isolated points,
which a `7 x 3` grid should essentially never hit -- yet `7/42 = 16.7%` of the
`f = 4` cells are `rho = 2`.  The suspect is the family's own MIRROR SYMMETRY
(blockers at `s1` and `1-s1` plus one in the middle, symmetric under `s -> 1-s`).
Under a `Z_2` the Gram splits into symmetric and antisymmetric blocks, and a
crossing BETWEEN blocks is codimension 1 -- a curve in the sweep, which a grid
does hit.  Note this refines, and does not contradict, `rho_le_1.md`: `Z_2` has
no two-dimensional irrep so it cannot FORCE a double eigenvalue, but it can drop
the codimension from 2 to 1, which changes reachability entirely.

FALSIFIABLE PREDICTION, and the reason this is worth running: break the mirror
symmetry and the two regimes must react DIFFERENTLY.  `Z = 0` is an open
condition, so the `f = 2` cells should survive; a codimension-1 crossing that
exists only because of the symmetry should disappear at `f = 4`.

And a second prediction, which would finish the answer to their open problem #2:
if `Z = 0` needs `m >= f`, then giving the curve FOUR contacts at `f = 4` should
put `Z = 0` back within reach and take `rho` to `4`, not `2`.

Gates:
  a6a  the two-mechanism split is real: `||Z||/||K||` separates the `m >= f`
       cells from the `m < f` cells by many orders of magnitude
  a6b  `m < f` implies `Z != 0` on every measured cell (this one is a theorem;
       a violation would mean the contact identification is wrong)
  a6c  symmetry breaking discriminates as predicted: the `Z = 0` cells survive
       and the double-eigenvalue cells do not
  a6d  the `rho >= 3` hunt is reported either way, with the arbitrated verdict
       and the reason -- a null result here is a real result

Writes artifacts/a6_two_mechanisms.json.
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
from node import Node, build                                # noqa: E402
from a4_simplicity_margin import analyse                    # noqa: E402
from a5_rho2_scope import (arbitrated_rank, bcs,            # noqa: E402
                           corner_radius, three_blocker_cfg)
from a3_finite_d_certificate import nodal_diag_root         # noqa: E402


def z_norm(seg, tol=1e-11):
    """`||Z||_F / ||K||_F` at the optimum -- 0 means the PSD multiplier died."""
    h = build(seg, Node(lo=None, hi=None), use_rlt=False, use_lift=False)
    h["prob"].solve(solver="CLARABEL", tol_gap_abs=tol, tol_gap_rel=tol,
                    tol_feas=tol, max_iter=5000, verbose=False)
    if h["prob"].status not in ("optimal", "optimal_inaccurate"):
        return None
    Z = np.asarray(h["cons"][0].dual_value)
    Z = 0.5 * (Z + Z.T)[seg.n:, seg.n:]
    K = seg.Gk[np.ix_(seg.Fidx, seg.Fidx)]
    return float(np.linalg.norm(Z) / max(1e-300, np.linalg.norm(K)))


def blocker_seg(xs_r, d, k, l, n=2, rmid=None, xmid=0.0):
    """Blockers at given (x, r) pairs, plus optionally one at (xmid, 0)."""
    obs = [(np.array([float(x)] + [0.0] * (n - 1)), float(r)) for x, r in xs_r]
    if rmid is not None:
        obs.append((np.array([float(xmid)] + [0.0] * (n - 1)), float(rmid)))
    b0, b1 = bcs(l, n)
    return Segment(n=n, d=d, k=k, l=l, obstacles=obs, bc0=b0, bc1=b1)


def three_blocker_asym(s1, rmid, d, k, l, xmid, n=2):
    """A4's family with the mirror symmetry BROKEN by moving the middle blocker."""
    r = corner_radius(s1)
    x1 = -2.0 + 4.0 * float(s1)
    return blocker_seg([(x1, r), (-x1, r)], d, k, l, n=n, rmid=rmid, xmid=xmid)


def four_blocker(s1, s2, d, k, l, n=2):
    """Two mirrored corner PAIRS -- four contacts, to reach `m >= f` at f = 4."""
    r1, r2 = corner_radius(s1), corner_radius(s2)
    x1, x2 = -2.0 + 4.0 * float(s1), -2.0 + 4.0 * float(s2)
    return blocker_seg([(x1, r1), (-x1, r1), (x2, r2), (-x2, r2)], d, k, l, n=n)


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ART, "a6_two_mechanisms.json"))
    args = ap.parse_args()
    print("=== A6: two mechanisms for rho = 2, decided by m vs f ===")

    with open(os.path.join(ART, "a5_rho2_scope.json")) as fh:
        a5 = json.load(fh)

    # ---- a6a / a6b: the split, on A5's own rho = 2 cells ---------------
    print("\n  --- a6a/a6b: ||Z||/||K|| on every A5 rho = 2 cell ---")
    print("  %-20s %2s %2s %12s  %s" % ("config", "f", "m", "||Z||/||K||",
                                        "mechanism"))
    split = []
    for cfg in a5["configs"]:
        hi = [c for c in cfg["cells"]
              if c["status"] == "optimal" and c["rho"] == 2]
        for c in hi:
            seg = three_blocker_cfg(c["s1"], c["rmid"], cfg["d"], cfg["k"],
                                    cfg["l"], n=cfg["n"])
            zn = z_norm(seg)
            rec = dict(d=cfg["d"], k=cfg["k"], l=cfg["l"], n=cfg["n"],
                       f=cfg["f"], m=c["m"], s1=c["s1"], rmid=c["rmid"],
                       z_rel=zn, m_ge_f=bool(c["m"] >= cfg["f"]))
            split.append(rec)
    for rec in split:
        if rec["z_rel"] is None:
            continue
        print("  d=%-2d k=%d n=%d %8.4f %2d %2d %12.3e  %s"
              % (rec["d"], rec["k"], rec["n"], rec["s1"], rec["f"], rec["m"],
                 rec["z_rel"],
                 "Z = 0 (slack constraint)" if rec["z_rel"] < 1e-6
                 else "Z != 0 (double eigenvalue)"))
    ok = [r for r in split if r["z_rel"] is not None]
    zf = [r["z_rel"] for r in ok if r["m_ge_f"]]
    zl = [r["z_rel"] for r in ok if not r["m_ge_f"]]
    a6a = dict(n=len(ok), n_m_ge_f=len(zf), n_m_lt_f=len(zl),
               max_z_when_m_ge_f=float(max(zf)) if zf else None,
               min_z_when_m_lt_f=float(min(zl)) if zl else None,
               separation=float(min(zl) / max(zf)) if (zf and zl) else None,
               passed=bool(zf and zl and min(zl) > 1e4 * max(zf)))
    a6b = dict(n_m_lt_f=len(zl),
               n_violations=sum(1 for z in zl if z < 1e-6),
               passed=bool(all(z >= 1e-6 for z in zl)))
    print(f"\n    m >= f: max ||Z||/||K|| = {a6a['max_z_when_m_ge_f']}")
    print(f"    m <  f: min ||Z||/||K|| = {a6a['min_z_when_m_lt_f']}")
    print(f"    separation factor: {a6a['separation']}")

    # ---- a6c: break the mirror symmetry -------------------------------
    print("\n  --- a6c: break the mirror symmetry (move the middle blocker) ---")
    print("  %-18s %2s %6s | %s" % ("config", "f", "xmid",
                                    "rho (arbitrated) / ||Z||/||K||"))
    sym = []
    CASES = [(3, 1, 0, 0.0620, 0.50), (5, 2, 1, 0.0542, 0.50),      # f = 2
             (5, 1, 0, 0.0304, 0.20), (7, 2, 1, 0.04143, 0.35)]     # f = 4
    XMIDS = [0.0, 0.10, 0.25, 0.50]
    for (d, k, l, s1, rm) in CASES:
        f = Segment(**dict(n=2, d=d, k=k, l=l,
                           obstacles=[(np.array([0.0, 0.0]), 0.5)],
                           bc0=bcs(l)[0], bc1=bcs(l)[1])).f
        row = []
        for xm in XMIDS:
            seg = three_blocker_asym(s1, rm, d, k, l, xmid=xm)
            res = analyse(seg, ns=4001)
            if res.get("status") != "optimal":
                row.append(dict(xmid=xm, status=res.get("status")))
                continue
            arb = arbitrated_rank(seg) if (res.get("rho") or 0) >= 2 \
                or (res.get("margin") is not None
                    and res["margin"] <= 1e-6) else dict(verdict=res.get("rho"),
                                                         reason="screen")
            row.append(dict(xmid=xm, status="optimal", m=res.get("m"),
                            rho=arb["verdict"], reason=arb.get("reason"),
                            margin=res.get("margin"), z_rel=z_norm(seg)))
        sym.append(dict(d=d, k=k, l=l, f=int(f), s1=s1, rmid=rm, row=row))
        cells = "  ".join(
            ("%s: %s/%s" % (("x=%.2f" % c["xmid"]),
                            ("rho=%s" % c["rho"]) if "rho" in c else c["status"],
                            ("%.1e" % c["z_rel"]) if c.get("z_rel") is not None
                            else "-"))
            for c in row)
        print("  d=%-2d k=%d s1=%.4f %2d | %s" % (d, k, s1, f, cells))

    def surv(entry):
        base = [c for c in entry["row"] if c["xmid"] == 0.0]
        pert = [c for c in entry["row"] if c["xmid"] > 0.0]
        b2 = bool(base and base[0].get("rho") == 2)
        p2 = sum(1 for c in pert if c.get("rho") == 2)
        return b2, p2, len(pert)
    surv_f2 = [surv(e) for e in sym if e["f"] == 2]
    surv_f4 = [surv(e) for e in sym if e["f"] == 4]
    frac2 = (sum(p for _, p, _ in surv_f2)
             / max(1, sum(t for _, _, t in surv_f2)))
    frac4 = (sum(p for _, p, _ in surv_f4)
             / max(1, sum(t for _, _, t in surv_f4)))
    a6c = dict(entries=sym, survival_rate_m_ge_f=float(frac2),
               survival_rate_m_lt_f=float(frac4),
               prediction="Z=0 cells survive asymmetry; double-eigenvalue "
                          "cells do not",
               passed=bool(frac2 > frac4))
    print(f"\n    rho = 2 survival under asymmetry: f = 2 (Z=0) {100*frac2:.0f}%"
          f"   vs   f = 4 (double eigenvalue) {100*frac4:.0f}%")
    print(f"    prediction {'CONFIRMED' if a6c['passed'] else 'REFUTED'}")

    # ---- a6d: hunt rho >= 3 with m >= f -------------------------------
    print("\n  --- a6d: four blockers at f = 4, so m >= f and Z = 0 is back "
          "in reach ---")
    print("  %-18s %2s %2s %6s %8s %12s  %s"
          % ("config", "f", "m", "rho", "margin", "||Z||/||K||", "reason"))
    hunt = []
    for (d, k, l) in ((5, 1, 0), (7, 2, 1)):
        a0 = float(nodal_diag_root(d, k, l))
        for g1 in (0.60, 0.80, 0.97, 1.20):
            for g2 in (1.8, 2.6, 3.6, 5.0):
                s1, s2 = a0 * g1, a0 * g1 * g2
                if not (0.01 < s1 < s2 < 0.42):
                    continue
                try:
                    seg = four_blocker(s1, s2, d, k, l)
                    res = analyse(seg, ns=4001)
                except Exception as exc:                     # noqa: BLE001
                    res = dict(status=f"error:{type(exc).__name__}")
                rec = dict(d=d, k=k, l=l, s1=float(s1), s2=float(s2),
                           status=res.get("status"), m=res.get("m"),
                           margin=res.get("margin"), f=int(seg.f)
                           if res.get("status") else None)
                if res.get("status") == "optimal":
                    arb = arbitrated_rank(seg)
                    rec.update(rho=arb["verdict"], reason=arb["reason"],
                               z_rel=z_norm(seg), face=arb.get("face"),
                               sor=arb.get("signal_over_resid"))
                hunt.append(rec)
                if res.get("status") == "optimal":
                    print("  d=%-2d k=%d %.3f/%.3f %2d %2s %6s %8s %12s  %s"
                          % (d, k, s1, s2, rec["f"] or -1,
                             str(rec.get("m")), str(rec.get("rho")),
                             "%.1e" % rec["margin"] if rec["margin"]
                             is not None else "-",
                             "%.3e" % rec["z_rel"] if rec.get("z_rel")
                             is not None else "-", rec.get("reason")))
    solved = [h for h in hunt if h.get("status") == "optimal"]
    hi3 = [h for h in solved if isinstance(h.get("rho"), int) and h["rho"] >= 3]
    mgef = [h for h in solved if h.get("m") and h["f"] and h["m"] >= h["f"]]
    a6d = dict(n_cells=len(hunt), n_solved=len(solved),
               n_rho_ge3=len(hi3), n_m_ge_f=len(mgef),
               max_rho=max([h["rho"] for h in solved
                            if isinstance(h.get("rho"), int)], default=0),
               max_m=max([h["m"] for h in solved if h.get("m")], default=0),
               examples=[{kk: h[kk] for kk in ("d", "k", "s1", "s2", "m", "rho",
                                              "z_rel", "sor")}
                         for h in hi3[:6]],
               unsound=[h for h in hi3
                        if h.get("margin") is not None and h["margin"] > 1e-6],
               passed=True)
    print(f"\n    solved {len(solved)} four-blocker cells; max m = "
          f"{a6d['max_m']}, cells with m >= f: {a6d['n_m_ge_f']}, "
          f"rho >= 3: {a6d['n_rho_ge3']}, max rho = {a6d['max_rho']}")
    if a6d["unsound"]:
        print("    *** certificate CLEARED a rho >= 3 cell -- unsound ***")
    a6d["passed"] = bool(not a6d["unsound"])

    gates = dict(a6a_two_mechanism_split=a6a, a6b_m_lt_f_forces_Z_nonzero=a6b,
                 a6c_symmetry_breaking=a6c, a6d_rho3_hunt=a6d)
    print("\n  --- gates ---")
    for nm in gates:
        print(f"  {nm}: {'PASS' if gates[nm]['passed'] else 'FAIL'}")

    with open(args.out, "w") as fh:
        json.dump(dict(gates=gates, split=split, hunt=hunt), fh, indent=1,
                  default=float)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
