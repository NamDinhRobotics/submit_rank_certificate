"""Three numbers a referee asked for that the paper had but never reported.

FIRST.  Eq. (8) is a chain, `rho <= f - rank Z_full <= dim ker Z`, and the
manuscript says so.  Theorem 6 then computes `dim ker Z` exactly -- so what the
paper computes exactly is an UPPER BOUND on `rho`, not `rho`.  How loose that
bound is was never measured, although `a1` and `a4` have stored both numbers on
every instance since they were written.

SECOND.  The atomicity lemma bounds the number of live contacts by `d`, the
degree, because the multiplier is supported on the zero set of a sum of squares
of polynomials of degree at most `d`.  That bound should hold on every instance
already solved, and it is a falsification test: one instance with `m > d` would
refute the lemma.

THIRD.  Fig. 2(c) reports "15 of 56 solved cells" and never says how many cells
were posed.  The census reports its own attrition carefully; this sweep should
be held to the same standard.  The number is in the figure's own artifact.

Everything here is computed from committed artifacts -- nothing is re-solved --
so these numbers cannot disagree with the ones already in the paper.

FOURTH.  The comparison and named populations are reported as "112", which is
their ROW count and not their instance count: the comparison set contains every
named instance, so the same instance is read twice, once by `a1` and once by
`a4`.  A referee is entitled to the distinct count and to know that the two
readings agree, so both are computed here rather than left implicit.

Gates:
  a38a  rho == dim ker Z on every instance where both were recorded
  a38b  m <= d on every instance, and the largest m seen
  a38c  the counterexample sweep's attrition: posed, solved, rho = 2
  a38d  how many DISTINCT instances the 112 rows are, and how far the two
        independent solves of each disagree

Run:  python experiments/a38_chain_tightness.py
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "..", "artifacts")


def load(name):
    with open(os.path.join(ART, name + ".json")) as fh:
        return json.load(fh)


def _instance_key(row):
    """Identify an instance across the two harnesses.

    `a1` names the k-sweep entry `symmetric_blocker k=1 d=5` and the same
    instance in its loose list `symmetric_blocker d=5`; `a4` names it
    `symmetric_blocker k=1 d=5`.  The configuration is carried by (d, k, l)
    anyway, so the layout name with those suffixes stripped is the rest of it.
    """
    name = re.sub(r"\s*k=\d+\s*", "", row["name"])
    name = re.sub(r"\s*d=\d+\s*", "", name)
    return (name.strip(), row["d"], row["k"], row["l"])


def population_overlap(pool, named):
    """(distinct instances, worst disagreement between two solves of one)."""
    by = {}
    for r in pool + named:
        by.setdefault(_instance_key(r), []).append(r)
    worst = 0.0
    for rows in by.values():
        cs = [sorted(r["contacts"]) for r in rows if r.get("contacts")]
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                if len(cs[i]) != len(cs[j]):        # a real disagreement
                    return len(by), float("inf")
                worst = max(worst, max(abs(a - b)
                                       for a, b in zip(cs[i], cs[j])))
    return len(by), worst


def main():
    a1 = load("a1_contact_rank")
    a4 = load("a4_simplicity_margin")

    pool = [r for r in a1["loose"] + a1["k_sweep"]
            if r.get("rho") is not None and r.get("ker_dim") is not None]
    named = [r for r in a4["rows"]
             if r.get("rho") is not None and r.get("ker_dim") is not None]
    tight = [r for r in pool + named if r["rho"] == r["ker_dim"]]
    print("    chain (8), rho <= dim ker Z, over %d instances (%d comparison "
          "+ %d named):" % (len(pool) + len(named), len(pool), len(named)))
    print("      attained (rho == dim ker Z)   %3d" % len(tight))
    print("      slack    (rho <  dim ker Z)   %3d" % (len(pool) + len(named)
                                                       - len(tight)))

    # m <= d.  a1 records the contact count as `n_contacts`; a4 as `m`.
    mrows = ([(r.get("n_contacts"), r["d"]) for r in pool]
             + [(r.get("m"), r["d"]) for r in named])
    mrows = [(m, d) for m, d in mrows if m is not None]
    over = [(m, d) for m, d in mrows if m > d]
    print("\n    atomicity bound m <= d over %d instances:" % len(mrows))
    print("      violations                    %3d" % len(over))
    print("      largest m seen                %3d  (largest d %d)"
          % (max(m for m, _ in mrows), max(d for _, d in mrows)))

    # census carries m but not d; report its m separately as corroboration
    cen = [r["m"] for r in a4.get("census", []) if r.get("m") is not None]
    print("      largest m in the census       %3d  over %d instances"
          % (max(cen), len(cen)))

    panel = load("fig_counterexample")["panel_c"]
    print("\n    counterexample sweep: %d cells posed, %d solved, %d with rho=2"
          % (panel["n_cells"], panel["n_solved"], panel["n_rho2"]))

    a38a = dict(n=len(pool) + len(named), n_comparison=len(pool),
                n_named=len(named), n_attained=len(tight),
                n_slack=len(pool) + len(named) - len(tight),
                passed=bool(len(tight) == len(pool) + len(named)
                            and len(pool) == 59 and len(named) == 53))
    a38b = dict(n=len(mrows), n_violations=len(over),
                max_m=max(m for m, _ in mrows), max_d=max(d for _, d in mrows),
                max_m_census=max(cen), n_census=len(cen),
                passed=bool(not over and max(m for m, _ in mrows) == 3
                            and max(cen) == 3))
    a38c = dict(n_cells=panel["n_cells"], n_solved=panel["n_solved"],
                n_rho2=panel["n_rho2"],
                passed=bool(panel["n_cells"] == 72 and panel["n_solved"] == 56
                            and panel["n_rho2"] == 15))

    n_distinct, worst = population_overlap(pool, named)
    print("\n    the %d rows are %d DISTINCT instances (%d read twice); the two"
          % (len(pool) + len(named), n_distinct,
             len(pool) + len(named) - n_distinct))
    print("    independent solves of a repeated instance place its contacts "
          "within %.3g" % worst)
    a38d = dict(n_rows=len(pool) + len(named), n_distinct=n_distinct,
                n_named_not_in_comparison=len(
                    {_instance_key(r) for r in named}
                    - {_instance_key(r) for r in pool}),
                worst_contact_disagreement=worst,
                passed=bool(n_distinct == 54 and worst < 1.3e-4))

    with open(os.path.join(ART, "a38_chain_tightness.json"), "w") as fh:
        json.dump(dict(gates=dict(a38a_chain_attained=a38a,
                                  a38b_atom_bound=a38b,
                                  a38c_sweep_attrition=a38c,
                                  a38d_population_overlap=a38d)), fh, indent=1)
    print("\n  gates: a38a %s  a38b %s  a38c %s  a38d %s"
          % (a38a["passed"], a38b["passed"], a38c["passed"], a38d["passed"]))
    return 0 if all(g["passed"] for g in (a38a, a38b, a38c, a38d)) else 1


if __name__ == "__main__":
    sys.exit(main())
