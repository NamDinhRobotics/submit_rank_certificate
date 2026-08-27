"""Two numbers the paper reported only from the side that flattered them.

FIRST.  Theorem 5 gives `rho <= rank M_lambda <= m` and the manuscript said it
is "not comparable with rho <= f and strictly tighter on 22 of 59".  Both halves
are true and together they are not the whole picture: a bound described as
incomparable owes the reader the count on BOTH sides, and 22 of 59 leaves 37
instances unaccounted for.  They are not all ties.

SECOND.  Corollary 7 is sufficient and not necessary -- `g > 0` implies
`rho <= 1` and the converse fails.  The manuscript measures the direction that
looks good (the 15 known `rho = 2` cells are all refused) and never the one a
user cares about: how often does the test REFUSE an instance that was fine?  A
sufficient test with a high false-refusal rate is a test nobody can plan around.

Both are computed from the committed artifacts of A1 and A4 rather than
re-solved, so they cannot disagree with the numbers already in the paper.

Gates:
  a37a  the contact bound against rho <= f, split three ways, summing to 59
  a37b  the certificate's false-refusal count over the named and census
        populations -- instances with rho <= 1 that the test declines to certify

Run:  python experiments/a37_bound_and_refusals.py
"""
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "..", "artifacts")


def load(name):
    with open(os.path.join(ART, name + ".json")) as fh:
        return json.load(fh)


def main():
    a1 = load("a1_contact_rank")
    rows = a1["loose"] + a1["k_sweep"]
    split = collections.Counter()
    for r in rows:
        m = r.get("m")
        if m is None:
            ct = r.get("contacts")
            m = len(ct) if isinstance(ct, list) else ct
        f = r["f"]
        split["tighter" if m < f else ("equal" if m == f else "looser")] += 1
    print("    Theorem 5 (rho <= m) against Lemma 4 (rho <= f), %d instances:"
          % len(rows))
    for k in ("tighter", "equal", "looser"):
        print("      %-8s %3d" % (k, split[k]))

    a4 = load("a4_simplicity_margin")
    named = [r for r in a4.get("rows", [])
             if isinstance(r.get("margin"), (int, float))]
    census = [r for r in a4.get("census", [])
              if isinstance(r.get("margin"), (int, float))]
    pool = named + census
    # A refusal is g = 0 on an instance that does satisfy rho <= 1: the test
    # declining something it need not have declined.
    refused = [r for r in pool if r["margin"] <= 0.0 and r.get("rho", 1) <= 1]
    certifiable = [r for r in pool if r.get("rho", 1) <= 1]
    print("\n    certificate over %d instances (%d named + %d census):"
          % (len(pool), len(named), len(census)))
    print("      with rho <= 1          %3d" % len(certifiable))
    print("      of those, REFUSED      %3d" % len(refused))
    print("      smallest margin        %.4f" % min(r["margin"] for r in pool))

    a37a = dict(n=len(rows), tighter=split["tighter"], equal=split["equal"],
                looser=split["looser"],
                passed=bool(sum(split.values()) == len(rows) == 59))
    a37b = dict(n_pool=len(pool), n_named=len(named), n_census=len(census),
                n_certifiable=len(certifiable), n_false_refusals=len(refused),
                min_margin=min(r["margin"] for r in pool),
                passed=bool(len(refused) == 0))
    with open(os.path.join(ART, "a37_bound_and_refusals.json"), "w") as fh:
        json.dump(dict(gates=dict(a37a_three_way_split=a37a,
                                  a37b_false_refusals=a37b)), fh, indent=1)
    print("\n  gates: a37a %s  a37b %s" % (a37a["passed"], a37b["passed"]))
    return 0 if (a37a["passed"] and a37b["passed"]) else 1


if __name__ == "__main__":
    sys.exit(main())
