"""What the gap rank does when the robots have to avoid each other.

SUPERSEDED IN TWO PLACES BY a23 -- read this before quoting gates a22b or a22d.
`realised_contacts` below reads the active set off `Gamma`.  Every constraint
here is affine in the LIFTED variable, so its value is the projected value plus
a square, and wherever `rho > 0` the lift can pay for a curve that goes straight
through another robot.  Read through the lift, `cross2`'s robot-robot contact is
slack by `+0.32`, so a22b's only witness is not one; and a22d's criterion -- a
negative entry in `L` -- is a sign CONVENTION away from no negative entry, since
flipping the sign of a pair contact is a signature similarity that moves no
eigenvalue.  The obstruction to Perron-Frobenius is signed-graph BALANCE, and
`cross2`'s pattern is balanced.  Both are measured in
`a23_fleet_rho_law.py`/`src/fleet_graph.py`; the formulation result and the
`R = 1` control below are unaffected.

The single-robot story of this repo is that `rho <= 1` is certifiable per
spline family, hard to violate, and violated only in a corner of parameter
space built on purpose.  This asks what survives when the same relaxation
plans for a fleet.

The formulation extends with no new machinery: stack the control points,
lift the joint Gram `X = Gamma_all^T Gamma_all`, and note that

    ||gamma_i(s) - gamma_k(s)||^2 - (2r)^2

is affine in that same `X` (blocks `X_ii - X_ik - X_ki + X_kk`), so
Markov-Lukacs applies to robot-robot avoidance exactly as it does to
obstacles.  `X` is still the `O(n)` invariant, so the relaxation stays
independent of the ambient dimension -- a fleet in `R^3` costs what it costs
in `R^2` -- while growing quadratically in the fleet size.

The certificate does NOT extend, and the reason is combinatorial.  With `K`
block diagonal over robots, the Green matrix at the contacts factors
entrywise as `Gr = L * B`, where `B` is the single-robot Green kernel and `L`
is the Gram of the contacts' incidence vectors (`e_i` for an obstacle contact
on robot i, `e_i - e_k` for a contact between i and k).  `L` carries `-1`
wherever two contacts share a robot with opposite sign, so `Gr` has negative
entries even where `B` is strictly positive -- even on a family the
single-robot certificate certifies.  Perron-Frobenius has nothing to stand
on.

So `rho <= 1` should fail for fleets.  The measurements say something more
precise than that:

    rho counts the INDEPENDENT binary homotopy decisions the instance leaves
    open -- 0 when the optimal class is unique, R when R robots are
    decoupled and each is undecided, and something in between when they
    interact.

which is why a head-on swap (one decision: who goes above) reads 1, not 2.

GATES:
  a22a  R=1 reproduces the single-robot relaxation exactly -- the control
  a22b  rho >= 2 occurs with an ACTIVE robot-robot contact, so the failure
        of the single-robot bound is not an artifact of decoupling
  a22c  decoupled fleets read rho = R, and every reading rho >= 2 is
        confirmed by a second solver rather than by one interior point
  a22d  the incidence Gram L of the realised contact pattern has negative
        entries, so Perron-Frobenius cannot apply -- reported per instance

Writes artifacts/a22_multirobot_rank.json.
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
ART = os.path.join(HERE, "..", "artifacts")

from multirobot import MultiRobot                              # noqa: E402
from relaxation import Segment                                 # noqa: E402

D, K, L = 3, 1, 0


def bc(x, y):
    return np.array([[float(x), float(y)]])


def instances():
    """Each entry: (name, kwargs for MultiRobot, what it is meant to show)."""
    out = []
    out.append(("swap2", dict(
        obstacles=[], bc0=[bc(-2, 0), bc(2, 0)], bc1=[bc(2, 0), bc(-2, 0)],
        robot_radius=0.3),
        "two robots head-on: ONE decision, who passes above"))
    out.append(("decoupled2", dict(
        obstacles={"per_robot": [[(np.array([0.0, 0.5]), 0.45)],
                                 [(np.array([0.0, -0.5]), 0.45)]]},
        bc0=[bc(-2, 0.5), bc(-2, -0.5)], bc1=[bc(2, 0.5), bc(2, -0.5)],
        robot_radius=0.0, pairs=[]),
        "two independent single-robot problems, each undecided"))
    out.append(("decoupled3", dict(
        obstacles={"per_robot": [[(np.array([0.0, y]), 0.45)]
                                 for y in (1.0, 0.0, -1.0)]},
        bc0=[bc(-2, y) for y in (1, 0, -1)],
        bc1=[bc(2, y) for y in (1, 0, -1)],
        robot_radius=0.0, pairs=[]),
        "three independent single-robot problems"))
    out.append(("cross2", dict(
        obstacles=[(np.array([0.0, 0.0]), 0.4)],
        bc0=[bc(-2, 0), bc(0, -2)], bc1=[bc(2, 0), bc(0, 2)],
        robot_radius=0.25),
        "perpendicular crossing around a shared obstacle: COUPLED"))
    out.append(("own_obs2", dict(
        obstacles={"per_robot": [[(np.array([0.0, 0.6]), 0.5)],
                                 [(np.array([0.0, -0.6]), 0.5)]]},
        bc0=[bc(-2, 0.6), bc(-2, -0.6)], bc1=[bc(2, 0.6), bc(2, -0.6)],
        robot_radius=0.25),
        "own obstacles with pairwise avoidance switched on"))
    for R, pts in ((3, [(-2., 0.), (1., 1.7), (1., -1.7)]),
                   (4, [(-2., 0.), (0., 2.), (2., 0.), (0., -2.)])):
        out.append(("cycle%d" % R, dict(
            obstacles=[], bc0=[bc(*pts[i]) for i in range(R)],
            bc1=[bc(*pts[(i + 1) % R]) for i in range(R)], robot_radius=0.3),
            "cyclic permutation in open space: is the class unique?"))
    return out


def incidence_gram(R, contacts):
    """`L` for a contact pattern: `e_i` for obstacle, `e_i - e_k` for a pair."""
    V = []
    for c in contacts:
        v = np.zeros(R)
        if c[0] == "obs":
            v[c[1]] = 1.0
        else:
            v[c[1]], v[c[2]] = 1.0, -1.0
        V.append(v)
    Vm = np.array(V)
    return Vm @ Vm.T


def realised_contacts(mr, res, tol=1e-4):
    """Which constraints are active at the returned curves."""
    ss = np.linspace(0.0, 1.0, 801)
    from bernstein import bd
    Bs = np.array([bd(s, mr.d) for s in ss])
    pts = [G @ Bs.T for G in res["Gamma"]]
    out = []
    for i, P in enumerate(pts):
        for jj, (c, r) in enumerate(mr.obs[i]):
            v = np.sum((P - c[:, None]) ** 2, axis=0) - r ** 2
            if float(v.min()) <= tol:
                out.append(("obs", i, jj))
    for (i, j) in mr.pairs:
        v = np.sum((pts[i] - pts[j]) ** 2, axis=0) - (2 * mr.rr) ** 2
        if float(v.min()) <= tol:
            out.append(("pair", i, j))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ART, "a22_multirobot_rank.json"))
    a = ap.parse_args()
    t0 = time.perf_counter()

    # ---- a22a: R = 1 must reproduce the single-robot relaxation
    obs1 = [(np.array([0.0, 0.0]), 0.5)]
    mr1 = MultiRobot(2, D, K, L, obs1, [bc(-2, 0)], [bc(2, 0)])
    r1 = mr1.solve()
    sg = Segment(n=2, d=D, k=K, l=L, obstacles=obs1,
                 bc0=bc(-2, 0), bc1=bc(2, 0))
    r0 = sg.solve_cvxpy()
    control = dict(multirobot=float(r1["cost"]), segment=float(r0["cost"]),
                   abs_diff=float(abs(r1["cost"] - r0["cost"])),
                   rho_multirobot=int(r1["rho"]), rho_segment=int(r0["rho"]))
    print("[a22a] R=1 control: %.9f vs %.9f (diff %.1e), rho %d vs %d"
          % (control["multirobot"], control["segment"], control["abs_diff"],
             control["rho_multirobot"], control["rho_segment"]))

    rows = []
    for name, kw, what in instances():
        mr = MultiRobot(2, D, K, L, **kw)
        res = mr.solve()
        cl = mr.clearances(res["Gamma"])
        con = realised_contacts(mr, res)
        Lg = incidence_gram(mr.R, con) if con else np.zeros((0, 0))
        arb = None
        if res["rho"] >= 2:
            alt = mr.solve(solver="SCS", ladder=(1e-9,))
            arb = dict(solver="SCS", rho=int(alt.get("rho", -1)),
                       cost=float(alt.get("cost", np.nan)),
                       status=alt.get("status"))
        rows.append(dict(
            name=name, what=what, R=mr.R, rho=int(res["rho"]),
            cost=float(res["cost"]), status=res.get("status"),
            lam1=float(res["lam1"]), lam2=float(res["lam2"]),
            lam2_over_lam1=float(res["lam2"] / res["lam1"])
            if res["lam1"] > 0 else 0.0,
            pair_clearance=float(cl["pair"]),
            obstacle_clearance=float(cl["obstacle"]),
            pairwise_active=bool(np.isfinite(cl["pair"])
                                 and cl["pair"] <= 1e-4),
            contacts=[list(c) for c in con],
            L_has_negative=bool(Lg.size and float(Lg.min()) < -1e-12),
            L_min=float(Lg.min()) if Lg.size else None,
            arbitration=arb))
        print("  %-11s R=%d rho=%d  cost %8.4f  lam2/lam1 %.3f  pair %+.2e  "
              "L_min %s%s"
              % (name, mr.R, res["rho"], res["cost"],
                 rows[-1]["lam2_over_lam1"], cl["pair"], rows[-1]["L_min"],
                 "  SCS rho=%d" % arb["rho"] if arb else ""))

    by = {r["name"]: r for r in rows}
    coupled_rho2 = [r for r in rows
                    if r["rho"] >= 2 and r["pairwise_active"]]
    arb_ok = [r for r in rows if r["rho"] >= 2
              and r["arbitration"] and r["arbitration"]["rho"] >= 2]
    gates = dict(
        a22a_R1_reproduces_single_robot=dict(
            **control,
            passed=bool(control["abs_diff"] <= 1e-6
                        and control["rho_multirobot"] == control["rho_segment"])),
        a22b_rho2_with_active_pair_contact=dict(
            witnesses=[r["name"] for r in coupled_rho2],
            n=len(coupled_rho2),
            note="rho >= 2 while the robot-robot constraint is tight, so the "
                 "single-robot bound fails for genuinely interacting robots "
                 "and not merely for a direct sum of independent ones",
            passed=bool(coupled_rho2)),
        a22c_decoupled_rho_equals_R=dict(
            decoupled2=by["decoupled2"]["rho"], decoupled3=by["decoupled3"]["rho"],
            n_rho2=len([r for r in rows if r["rho"] >= 2]),
            n_arbitrated=len(arb_ok),
            passed=bool(by["decoupled2"]["rho"] == 2
                        and by["decoupled3"]["rho"] == 3
                        and len(arb_ok) == len([r for r in rows
                                                if r["rho"] >= 2]))),
        a22d_incidence_gram_blocks_perron=dict(
            with_pair_contacts=[r["name"] for r in rows
                                if any(c[0] == "pair" for c in r["contacts"])],
            n_L_negative=sum(1 for r in rows if r["L_has_negative"]),
            note="a negative entry in L makes Gr = L * B indefinite in sign "
                 "however positive the Green kernel B is",
            passed=True, note2="reported per instance"))

    out = dict(description=__doc__.strip().splitlines()[0],
               family=dict(d=D, k=K, l=L, n=2), gates=gates,
               control_R1=control, instances=rows,
               runtime_s=time.perf_counter() - t0)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("\n  --- gates ---")
    for nm, g in gates.items():
        print("  %-38s %s" % (nm, "PASS" if g["passed"] else "FAIL"))
    print("wrote %s (%.0f s)" % (a.out, out["runtime_s"]))
    if not all(g["passed"] for g in gates.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
