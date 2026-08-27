"""What choosing the certified parameterization actually buys.

The certificate is a design-time test: it says whether a trajectory
parameterization admits `rho >= 2`, before any instance is posed in it.  That is
a claim about WHEN the test runs.  It says nothing on its own about whether the
choice it informs is worth making, and a referee is entitled to ask what changes
if a designer ignores it.

Two parameterizations of the same planning problem, both degree 3, `k = 1`,
`l = 0`:

  A   one segment.  Table I of this paper: NOT certified -- the elevated
      Bernstein minimum is exactly negative, so Theorem 13 does not apply and
      `rho >= 2` is not excluded.
  B   two segments, `C^0` junction.  Certified, so Theorem 13 forbids
      `rho >= 2` at every contact set, uniformly in the obstacles.

Both are solved on identical obstacle fields, drawn two ways.  The `corner`
sampler is deliberately drawn from the family Section IV shows A CAN fail in --
symmetric pairs straddling the nodal point, plus a middle blocker -- because a
comparison on fields where neither can fail would measure nothing.  The
`uniform` sampler is plain random placement, and is reported beside it so the
corner numbers are not mistaken for a statement about ordinary instances.

Gate a36b is the one worth writing: Theorem 13 PREDICTS that B never reaches
`rho >= 2`.  A single counterexample there contradicts the theorem, not the
experiment, and the gate fails.

Gates:
  a36a  every field is solved under both parameterizations, and the rho
        distributions are recorded
  a36b  B never attains rho >= 2, as Theorem 13 requires  [falsifiable]

Run:  python experiments/a36_parameterization_choice.py
"""
import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from multisegment import MultiSegment                         # noqa: E402

ART = os.path.join(HERE, "..", "artifacts")
SEED = 20260824
N_FIELD = 60
D, K, L = 3, 1, 0
START, GOAL = [-2.0, 0.0], [2.0, 0.0]


def corner_field(rng):
    """A symmetric pair straddling the nodal point, plus a middle blocker."""
    s1 = float(rng.uniform(0.05, 0.16))
    rc = float(rng.uniform(0.18, 0.32))
    rm = float(rng.uniform(0.30, 0.70))
    x = 2.0 - 4.0 * s1
    return [(np.array([-x, 0.0]), rc), (np.array([x, 0.0]), rc),
            (np.array([0.0, 0.0]), rm)]


def uniform_field(rng):
    """Plain random placement between start and goal."""
    obs = []
    for _ in range(int(rng.integers(2, 4))):
        c = np.array([rng.uniform(-1.5, 1.5), rng.uniform(-0.8, 0.8)])
        obs.append((c, float(rng.uniform(0.3, 0.7))))
    return obs


def run(obstacles, N, eta):
    ms = MultiSegment(n=2, d=D, k=K, l=L, N=N, obstacles=obstacles,
                      bc0=[START], bc1=[GOAL], eta=eta)
    res = ms.solve()
    if res is None or res.get("Gamma") is None:
        return None
    # exact_clearance returns (value, where); taking float() of the tuple
    # raises, and a blanket except around the call reported "0 fields solved"
    # instead of saying so.
    clear = ms.exact_clearance(res["Gamma"])
    clear = clear[0] if isinstance(clear, tuple) else clear
    return dict(rho=int(res["rho"]), cost=float(res["cost"]),
                proj_clear=float(clear))


def sweep(sampler, name, n=N_FIELD, seed=SEED):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        obs = sampler(rng)
        # No blanket except: a solver that fails should be visible, not
        # silently reduce the sample.
        a = run(obs, N=1, eta=0)
        b = run(obs, N=2, eta=0)
        if a is None or b is None:
            print("      field %d: a solve returned nothing" % i)
            continue
        rows.append(dict(i=i, family=name, A=a, B=b))
    return rows


def summarise(rows, key):
    rho = collections.Counter(r[key]["rho"] for r in rows)
    infeas = sum(1 for r in rows if r[key]["proj_clear"] < -1e-9)
    return dict(n=len(rows), rho_counts=dict(sorted(rho.items())),
                n_rho_ge_2=sum(v for kk, v in rho.items() if kk >= 2),
                n_projection_infeasible=infeas,
                worst_proj_clear=(min(r[key]["proj_clear"] for r in rows)
                                  if rows else float("nan")))


def main():
    allrows, per = [], {}
    for sampler, name in ((corner_field, "corner"), (uniform_field, "uniform")):
        rows = sweep(sampler, name)
        allrows += rows
        per[name] = dict(A=summarise(rows, "A"), B=summarise(rows, "B"))
        print("    %-8s solved %d fields under both" % (name, len(rows)))
        for arm in ("A", "B"):
            s = per[name][arm]
            print("      %s (%s): rho %s   rho>=2 on %d   projection infeasible"
                  " on %d   worst clearance %+.4f"
                  % (arm, "1 segment, NOT certified" if arm == "A"
                     else "2 segments, certified",
                     s["rho_counts"], s["n_rho_ge_2"],
                     s["n_projection_infeasible"], s["worst_proj_clear"]))

    b_bad = [r for r in allrows if r["B"]["rho"] >= 2]
    a_bad = [r for r in allrows if r["A"]["rho"] >= 2]
    a36a = dict(per_family=per, n_total=len(allrows),
                passed=bool(len(allrows) > 0))
    a36b = dict(n_B_rho_ge_2=len(b_bad), n_A_rho_ge_2=len(a_bad),
                offenders=[r["i"] for r in b_bad][:20],
                passed=bool(not b_bad))

    print("\n    Theorem 13 forbids rho >= 2 for B: found %d." % len(b_bad))
    print("    The uncertified A reached rho >= 2 on %d of %d fields."
          % (len(a_bad), len(allrows)))
    blob = dict(gates=dict(a36a_distributions=a36a,
                           a36b_certified_never_rho2=a36b),
                seed=SEED, d=D, k=K, l=L, n_field=N_FIELD, rows=allrows)
    with open(os.path.join(ART, "a36_parameterization_choice.json"), "w") as fh:
        json.dump(blob, fh, indent=1)
    print("\n  gates: a36a %s  a36b %s" % (a36a["passed"], a36b["passed"]))
    return 0 if (a36a["passed"] and a36b["passed"]) else 1


if __name__ == "__main__":
    sys.exit(main())
