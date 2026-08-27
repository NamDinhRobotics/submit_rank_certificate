"""A10 -- the whole `rho = 2` story reduces to one dichotomy: a single
polynomial has a negative lobe in its discrete Green's function, and one knot
removes it.

A9 certified `Gr >= 0` -- hence `rho <= 1` -- on 8 multisegment configurations
and deliberately refused to generalise: "the certificate is per-configuration
and cheap, check it for yours".  This phase asks whether that caution was
warranted, by sweeping the certificate across `N = 1..6` and seven `(d, k, l)`
settings.  The answer is a clean dichotomy, and it is not the one dimension
counting would predict:

    N = 1  (one polynomial)      : the certificate FAILS at every (d, k, l)
    N >= 2 (any knots at all)    : the certificate PASSES at every one

It is emphatically **not** about how many free parameters there are.  `N = 1,
d = 9` has `f = 8` free coefficients and fails; `N = 2, d = 3` has `r = 4` and
passes.  Fewer parameters, positive Green's function.  What matters is the KNOT.

WHY THIS IS THE WHOLE STORY.  Every route to `rho >= 2` found in this repo needs
the Green's function to go negative somewhere:

  * A3 located the corner where it does, and A4 drove an instance into it to get
    the counterexample;
  * A7 showed the OTHER route (`Z = 0`) needs `K` in the moment cone, which
    happens only at `f = 2`;
  * A9 showed both routes are shut for multisegment.

So "why is `rho <= 1` almost always true" and "why did the counterexample need a
single segment with two free control points" have the same one-line answer: a
single polynomial's Green's function undershoots, and a knot stops it.

THE DECAY, which reconciles several earlier measurements.  The single-segment
negative lobe does not merely exist, it SHRINKS geometrically with degree --
roughly a decade per two degrees.  That is the same trend, seen from the dual
side, as A3's corner radius `A` shrinking (`0.200` at `d = 3` to `0.054` at
`d = 7`) and A5's `rho = 2` cells getting harder to find as `d` grows.  So the
counterexample is not just a single-segment phenomenon, it is a LOW-DEGREE
single-segment phenomenon, and this quantifies how fast it fades.

Gates:
  a10a  the dichotomy: `N = 1` fails and `N >= 2` certifies, across the sweep,
        with the failure magnitudes recorded so "fails" is not just a label
  a10b  the negative lobe decays geometrically in `d` at `N = 1`, and the decay
        rate is reported rather than asserted
  a10c  it is NOT dimension: exhibit configurations where the single segment has
        MORE free parameters than a passing multisegment one and still fails

Writes artifacts/a10_knot_dichotomy.json.
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

from a8_nondegeneracy_multiseg import multi_pieces, make_ms, blockers_multi  # noqa: E402
from a9_multiseg_rho2 import _elevation                     # noqa: E402


CONFIGS = ((3, 1, 0), (5, 1, 0), (7, 1, 0), (9, 1, 0),
           (5, 2, 1), (7, 2, 1), (7, 3, 2))
NS = (1, 2, 3, 4, 5, 6)
D_ELEV = 96


def certificate(N, d, k, l, eta, D=D_ELEV):
    """Min elevated Bernstein coefficient of `Gr`, relative to the block scale.

    `>= 0` proves `Gr >= 0` on the whole parameter square (A9), hence `rho <= 1`
    by Perron-Frobenius.  Negative does NOT prove the converse -- the bound is
    conservative -- so the elevation degree is fixed and reported.
    """
    ms = make_ms(N, d, k, l, eta, blockers_multi())
    K, Np = multi_pieces(ms)
    Ki = np.linalg.inv(K)
    blocks = [Np[i] @ Ki @ Np[j].T for i in range(N) for j in range(N)]
    scale = max(1.0, max(float(np.abs(B).max()) for B in blocks))
    E = _elevation(d, D)
    lo = float(min((E @ B @ E.T).min() for B in blocks))
    return dict(N=N, d=d, k=k, l=l, eta=eta, r=int(K.shape[0]),
                min_coef=lo, scale=scale, min_coef_rel=lo / scale,
                certified=bool(lo >= -1e-12 * scale))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ART, "a10_knot_dichotomy.json"))
    args = ap.parse_args()

    print("=== A10: one knot removes the negative lobe ===")

    # ---- a10a --------------------------------------------------------
    print(f"\n  --- a10a: the dichotomy (elevated to D = {D_ELEV}) ---")
    print("  %3s | %s" % ("N", " ".join("d=%d k=%d" % (d, k)
                                        for (d, k, _) in CONFIGS)))
    rows = []
    for N in NS:
        cells = []
        for (d, k, l) in CONFIGS:
            eta = max(0, k - 1) if N > 1 else 0
            try:
                r = certificate(N, d, k, l, eta)
            except Exception as exc:                         # noqa: BLE001
                cells.append("  err  ")
                rows.append(dict(N=N, d=d, k=k, l=l, error=str(exc)[:40]))
                continue
            rows.append(r)
            cells.append("  OK   " if r["certified"]
                         else "%+7.0e" % r["min_coef_rel"])
        print("  %3d | %s" % (N, " ".join(cells)))

    ok = [r for r in rows if "min_coef" in r]
    single = [r for r in ok if r["N"] == 1]
    multi = [r for r in ok if r["N"] > 1]
    a10a = dict(rows=rows, n=len(ok), n_single=len(single), n_multi=len(multi),
                n_single_certified=sum(1 for r in single if r["certified"]),
                n_multi_certified=sum(1 for r in multi if r["certified"]),
                worst_single=float(max(r["min_coef_rel"] for r in single)),
                worst_multi=float(min(r["min_coef_rel"] for r in multi)),
                passed=bool(single and multi
                            and not any(r["certified"] for r in single)
                            and all(r["certified"] for r in multi)))
    print(f"\n    N = 1: certified {a10a['n_single_certified']}/{len(single)} "
          f"(least-negative {a10a['worst_single']:+.1e})")
    print(f"    N >= 2: certified {a10a['n_multi_certified']}/{len(multi)} "
          f"(worst coefficient {a10a['worst_multi']:+.1e})")

    # ---- a10b --------------------------------------------------------
    print("\n  --- a10b: the single-segment lobe decays with degree ---")
    print("  %3s %3s %3s %3s %14s %12s"
          % ("d", "k", "l", "f", "min coef (rel)", "ratio vs d-2"))
    decay, prev = [], {}
    for (d, k, l) in CONFIGS:
        r = certificate(1, d, k, l, 0)
        key = (k, l)
        ratio = (prev[key] / r["min_coef_rel"]) if key in prev else None
        prev[key] = r["min_coef_rel"]
        decay.append(dict(d=d, k=k, l=l, f=r["r"],
                          min_coef_rel=r["min_coef_rel"], ratio=ratio))
        print("  %3d %3d %3d %3d %14.3e %12s"
              % (d, k, l, r["r"], r["min_coef_rel"],
                 "-" if ratio is None else "%.1fx" % ratio))
    k1 = [x for x in decay if (x["k"], x["l"]) == (1, 0)]
    ratios = [x["ratio"] for x in k1 if x["ratio"]]
    a10b = dict(rows=decay, k1_ratios=ratios,
                median_ratio_per_2deg=float(np.median(ratios)) if ratios else None,
                all_negative=bool(all(x["min_coef_rel"] < 0 for x in decay)),
                passed=bool(decay and all(x["min_coef_rel"] < 0 for x in decay)
                            and ratios and min(ratios) > 1.0))
    print(f"\n    every single-segment configuration is negative, and along "
          f"k = 1 the lobe shrinks by a median factor "
          f"{a10b['median_ratio_per_2deg']:.1f} per two degrees -- the dual-side "
          f"view of A3's corner radius shrinking and A5's rho = 2 getting rarer")

    # ---- a10c --------------------------------------------------------
    print("\n  --- a10c: it is the KNOT, not the dimension ---")
    print("  %-28s %3s %3s %14s %s" % ("configuration", "r", "N",
                                       "min coef (rel)", "verdict"))
    pairs = []
    for (N, d, k, l, eta) in ((1, 9, 1, 0, 0), (2, 3, 1, 0, 1),
                              (1, 7, 1, 0, 0), (2, 3, 1, 0, 0),
                              (1, 5, 1, 0, 0), (3, 3, 1, 0, 1)):
        r = certificate(N, d, k, l, eta)
        pairs.append(r)
        print("  %-28s %3d %3d %14.3e %s"
              % ("N=%d d=%d k=%d eta=%d" % (N, d, k, eta), r["r"], N,
                 r["min_coef_rel"], "certified" if r["certified"] else "FAILS"))
    bigger_single = [p for p in pairs if p["N"] == 1]
    small_multi = [p for p in pairs if p["N"] > 1]
    witness = [(s, m) for s in bigger_single for m in small_multi
               if s["r"] > m["r"] and not s["certified"] and m["certified"]]
    a10c = dict(rows=pairs, n_witnesses=len(witness),
                examples=[dict(single_r=s["r"], single_d=s["d"],
                               multi_r=m["r"], multi_N=m["N"], multi_d=m["d"])
                          for s, m in witness[:6]],
                passed=bool(witness))
    print(f"\n    {len(witness)} pairs where the SINGLE segment has strictly more "
          f"free parameters than a certified multisegment one and still fails "
          f"-- so dimension is not the driver")

    # ---- a10d: a WELDED knot is not a knot --------------------------
    print("\n  --- a10d: negative control -- weld the knot shut ---")
    print("  A degree-d spline with C^d continuity IS a single polynomial, so if")
    print("  the dichotomy really tracks the freedom at the joint rather than the")
    print("  label N, then eta = d must FAIL however many segments there are.")
    print("  Without this control, 'N >= 2 certifies' would be a claim about a")
    print("  test that merely responds to N.")
    print()
    print("  %3s %4s %4s | %14s %s" % ("N", "eta", "r", "elev min coef",
                                       "verdict"))
    weld = []
    for (N, eta) in ((1, 0), (2, 0), (2, 1), (2, 2), (2, 3), (3, 3), (4, 3)):
        r = certificate(N, 3, 1, 0, eta if N > 1 else 0)
        genuine = (N > 1 and eta < 3)
        weld.append(dict(N=N, eta=eta, r=r["r"], genuine_knot=genuine,
                         min_coef_rel=r["min_coef_rel"],
                         certified=r["certified"]))
        print("  %3d %4d %4d | %14.3e %s"
              % (N, eta, r["r"], r["min_coef_rel"],
                 "CERTIFIED" if r["certified"] else "FAILS"))
    welded = [w for w in weld if not w["genuine_knot"]]
    genuine = [w for w in weld if w["genuine_knot"]]
    a10d = dict(rows=weld,
                n_welded=len(welded), n_welded_certified=sum(
                    1 for w in welded if w["certified"]),
                n_genuine=len(genuine), n_genuine_certified=sum(
                    1 for w in genuine if w["certified"]),
                welded_r_values=sorted({w["r"] for w in welded}),
                passed=bool(welded and genuine
                            and not any(w["certified"] for w in welded)
                            and all(w["certified"] for w in genuine)))
    print(f"\n    welded knots (eta = d) certify "
          f"{a10d['n_welded_certified']}/{len(welded)}; genuine knots certify "
          f"{a10d['n_genuine_certified']}/{len(genuine)}")
    print(f"    every welded case collapses to r = {a10d['welded_r_values']}, "
          f"the single polynomial's own value -- so the dichotomy is about the "
          f"freedom the joint LEAVES, not about N")

    gates = dict(a10a_knot_dichotomy=a10a, a10b_lobe_decay=a10b,
                 a10c_not_dimension=a10c, a10d_welded_knot_control=a10d)
    print("\n  --- gates ---")
    for nm in gates:
        print(f"  {nm}: {'PASS' if gates[nm]['passed'] else 'FAIL'}")

    with open(args.out, "w") as fh:
        json.dump(dict(gates=gates), fh, indent=1, default=float)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
