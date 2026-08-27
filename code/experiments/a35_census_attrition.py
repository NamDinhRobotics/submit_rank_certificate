"""Where the other 540 draws went, and whether losing them biased the census.

The census in A4 keeps 60 instances out of 600 random draws.  The manuscript
says it keeps "those that solve and are loose", which is true and is not enough:
a filter that discards instances the SOLVER could not finish is a filter whose
selectivity may correlate with the very degeneracy the paper measures.  If
near-degenerate instances are disproportionately the ones Clarabel gives up on,
then the reported margin floor is a floor over the easy half of the population
and means less than it appears to.

So: replay the same 600 draws under the same seed, classify every one, and then
RESCUE the solver-failure bucket with a second solver.  If SCS finishes them and
they turn out loose with margins below the census floor, the filter was hiding
exactly what it should not have been, and the paper has to say so.  If they are
tight, or their margins sit inside the reported range, the attrition is benign.

The test is designed to be able to fail.  A rescue pass that could only confirm
the original number would not be worth running.

Gates:
  a35a  the 600 draws are fully accounted for, bucket by bucket
  a35b  the solver-failure bucket, re-solved with SCS, contains no loose
        instance whose margin falls below the census floor reported in A4

Run:  python experiments/a35_census_attrition.py
"""
import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from relaxation import Segment                                # noqa: E402
from node import Node, build                                  # noqa: E402
from instances import random_instance                         # noqa: E402
from a4_simplicity_margin import analyse, census              # noqa: E402

ART = os.path.join(HERE, "..", "artifacts")
SEED = 20260806
N_DRAW = 600
CENSUS_FLOOR = 0.5028157959876302        # a4c min_margin, the number at issue


def classify(n_draw=N_DRAW, seed=SEED):
    """Replay A4's census draw for draw, recording why each one left."""
    rng = np.random.default_rng(seed)
    buckets, failed = [], []
    for i in range(n_draw):
        kw = random_instance(rng)
        if not kw["obstacles"]:
            buckets.append(("no obstacle drawn", i, None)); continue
        try:
            r = analyse(Segment(**kw))
        except Exception as exc:                              # noqa: BLE001
            buckets.append(("analysis raised", i, type(exc).__name__)); continue
        st = r.get("status")
        if st == "optimal":
            buckets.append(("kept: solved and loose", i, r.get("margin")))
        elif st == "tight":
            buckets.append(("dropped: tight (rho = 0)", i, None))
        elif st == "no contact detected":
            buckets.append(("dropped: no contact detected", i, None))
        else:
            buckets.append(("dropped: solver did not finish", i, st))
            failed.append((i, kw, st))
    return buckets, failed


def rescue(failed, tol=1e-9):
    """Re-solve the solver-failure bucket with SCS instead of Clarabel."""
    out = []
    for i, kw, st in failed:
        seg = Segment(**kw)
        try:
            h = build(seg, Node(lo=None, hi=None), use_rlt=False, use_lift=False)
            h["prob"].solve(solver="SCS", eps_abs=tol, eps_rel=tol,
                            max_iters=20000, verbose=False)
            if h["prob"].status not in ("optimal", "optimal_inaccurate") \
                    or h["Gfree"].value is None:
                out.append(dict(i=i, clarabel=st, scs=h["prob"].status,
                                rescued=False)); continue
            Gf = np.asarray(h["Gfree"].value)
            Xf = np.asarray(h["Xfree"].value); Xf = 0.5 * (Xf + Xf.T)
            wS = np.linalg.eigvalsh(Xf - Gf.T @ Gf)
            rho = int(np.sum(wS > 1e-6 * max(1.0, float(np.abs(wS).max()))))
            rec = dict(i=i, clarabel=st, scs=h["prob"].status, rescued=True,
                       rho=rho, loose=bool(rho >= 1), margin=None)
            if rho >= 1:                       # only a loose one has a margin
                r2 = analyse(seg)
                rec["margin"] = (float(r2["margin"])
                                 if r2.get("status") == "optimal" else None)
            out.append(rec)
        except Exception as exc:                              # noqa: BLE001
            out.append(dict(i=i, clarabel=st, scs=type(exc).__name__,
                            rescued=False))
    return out


def main():
    buckets, failed = classify()
    counts = collections.Counter(b[0] for b in buckets)
    print("    attrition over %d draws:" % N_DRAW)
    for k, v in counts.most_common():
        print("      %-32s %4d  (%4.1f%%)" % (k, v, 100.0 * v / N_DRAW))
    total = sum(counts.values())

    print("\n    re-solving the %d solver failures with SCS..." % len(failed))
    resc = rescue(failed)
    n_res = sum(r["rescued"] for r in resc)
    loose = [r for r in resc if r.get("loose")]
    margins = [r["margin"] for r in loose if r.get("margin") is not None]
    below = [m for m in margins if m < CENSUS_FLOOR]
    print("      SCS finished       %d of %d" % (n_res, len(resc)))
    print("      of those, loose    %d" % len(loose))
    print("      margins recovered  %d, min %s"
          % (len(margins), ("%.4f" % min(margins)) if margins else "n/a"))
    print("      BELOW the census floor %.4f: %d" % (CENSUS_FLOOR, len(below)))

    a35a = dict(n_draws=N_DRAW, accounted=total, counts=dict(counts),
                n_kept=counts.get("kept: solved and loose", 0),
                passed=bool(total == N_DRAW))
    a35b = dict(n_solver_failures=len(failed), n_rescued_by_scs=n_res,
                n_rescued_loose=len(loose), n_margins=len(margins),
                min_rescued_margin=(min(margins) if margins else None),
                census_floor=CENSUS_FLOOR, n_below_floor=len(below),
                rescued=resc, passed=bool(not below))
    blob = dict(gates=dict(a35a_attrition=a35a, a35b_no_hidden_low_margin=a35b),
                seed=SEED)
    with open(os.path.join(ART, "a35_census_attrition.json"), "w") as fh:
        json.dump(blob, fh, indent=1)
    print("\n  gates: a35a %s  a35b %s" % (a35a["passed"], a35b["passed"]))
    return 0 if (a35a["passed"] and a35b["passed"]) else 1


if __name__ == "__main__":
    sys.exit(main())
