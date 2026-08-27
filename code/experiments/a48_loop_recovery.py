"""A48 -- the certificate in the loop: disturbed receding-horizon replanning.

Everything so far certifies and recovers ONE solve.  A planner does not solve
once: it replans from wherever the disturbance pushed it, and the relaxation's
rank is re-rolled at every step.  This file closes that loop on the only
question the one-shot experiments cannot answer: does certified fold recovery
survive feedback?

THE LOOP.  State (p, v) in R^2 x R^2.  At each step, plan a degree-5, k = 2,
l = 1 curve from (p, v) to the goal PAST THE TIGHTENED obstacles -- radii
inflated by eps = 0.06, the standard constraint-tightening move -- and score
everything against the TRUE obstacles; execute the subcurve [0, tau]
(de Casteljau split, so the executed piece is itself a polynomial and its
clearance is decided EXACTLY, not sampled); then the disturbance kicks:
p += N(0, sigma^2 I), v += N(0, (sigma/2)^2 I).  The velocity pinned at the
next replan is gamma'(tau) (1 - tau), the same time-scaling convention
throughout.  A run succeeds when it reaches the goal ball without any executed
piece or post-kick state touching a true obstacle.

WHY THE TIGHTENING IS LOAD-BEARING, TWICE.  First, soundness becomes a
theorem instead of a tolerance: a curve certified on the inflated field has
q_true(s) = q_infl(s) + eps (2 r + eps) > 0, so a certified executed piece
CANNOT collide -- the first, un-tightened version of this file watched
tangent rho = 0 solves graze the true obstacles by solver-tolerance amounts,
exactly the regime where Hypothesis 2 (strict feasibility) dies.  Second, the
tightening squeezes the corridors, which is what pushes solves into the
rho = 1 regime the experiment is here to exercise.

TWO POLICIES, PAIRED.  Same fields, same disturbance seeds:
  P0 naive     executes the projection gamma* of every solve, whatever rho;
  P1 certified executes gamma* when rho = 0 and otherwise runs the fold
               search of a46 (coarser grid -- the exact certificate at the
               end is what carries soundness, not the grid); an uncertified
               candidate is NEVER executed -- the policy stalls one step
               instead, and stalls are counted.

The comparison is not decoration.  Corollary `cor:naive` says the projection
of a rho >= 1 solve is infeasible as a WHOLE curve; the loop asks the harder
practical question of whether its executed PREFIX collides before the goal
arrives, under the same disturbances the certified policy faces.

Gates:
  a48a  soundness in the loop: every executed piece of every CERTIFIED curve
        has exact clearance >= 0 (machine epsilon), across all runs
  a48b  paired outcomes: certified-fold success count vs naive, collisions
        itemised, and the two runs identical on rho = 0-only histories
  a48c  the loop actually exercises recovery: events > 0, every event either
        certified exactly or refused (no uncertified execution), timings
  a48d  stalls are rare: the certified policy stalls on < 10% of its steps
  a48e  WHOSE FAULT: failures split by what actually failed -- the executed
        CURVE, or the state after a disturbance no planner controls.  The
        certified policy's executed-curve count is zero by construction
        (Cor. `cor:naive` plus the tightening bound), and the naive one's is
        not; this is the sharper reading of the paired comparison

Run:  python experiments/a48_loop_recovery.py
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
ART = os.path.join(ROOT, "artifacts")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from bernstein import bd, bd_derivs                              # noqa: E402
from instances import random_instance                            # noqa: E402
from relaxation import Segment, exact_clearance                  # noqa: E402
import a46_rank_one_recovery as a46                              # noqa: E402

D, K, L = 5, 2, 1
TAU = 0.25                       # executed fraction of each plan
SIGMA = 0.08                     # disturbance on position per step
EPS_PLAN = 0.06                  # constraint tightening on planned radii
GOAL_TOL = 0.20
MAX_STEPS = 30
N_RUNS = 30
RHO_TOL = 1e-6


def decasteljau_left(G, t):
    """Control points of the subcurve [0, t] (columns of G are control pts)."""
    pts = [G[:, a].copy() for a in range(G.shape[1])]
    left = [pts[0].copy()]
    for _ in range(len(pts) - 1):
        pts = [(1 - t) * pts[a] + t * pts[a + 1] for a in range(len(pts) - 1)]
        left.append(pts[0].copy())
    return np.stack(left, axis=1)


def curve_at(G, s, d):
    return G @ bd(s, d)


def deriv_at(G, s, d):
    return G @ bd_derivs(s, d, 1)[1]


def point_clearance(p, obs):
    return min(float(np.dot(p - c, p - c) - r * r) for c, r in obs)


def plan(obs, p, v):
    infl = [(c, r + EPS_PLAN) for c, r in obs]
    bc0 = np.array([p, v], float)
    bc1 = np.array([[2.0, 0.0], [0.0, 0.0]], float)
    seg = Segment(n=2, d=D, k=K, l=L, obstacles=infl, bc0=bc0, bc1=bc1)
    res = seg.solve(backend="cvxpy", solver="CLARABEL")
    return seg, res


def run_policy(obs, kicks, certified, history=None):
    """One closed-loop run.  kicks: (MAX_STEPS, 4) pre-drawn disturbances.
    history (optional list) collects one dict per step for rendering."""
    p = np.array([-2.0, 0.0])
    v = np.array([0.0, 0.0])
    log = dict(steps=0, recoveries=0, stalls=0, solve_t=[], rec_t=[],
               exec_clear=[], outcome="cap", rho_seen=[])
    for step in range(MAX_STEPS):
        log["steps"] = step + 1
        if np.linalg.norm(p - [2.0, 0.0]) < GOAL_TOL:
            log["outcome"] = "success"
            break
        if min(float(np.dot(p - c, p - c) - (r + EPS_PLAN) ** 2)
               for c, r in obs) < 0.0:
            # the kick landed inside the tightened belt: the TIGHTENED problem
            # has no strictly clear start, which is Hypothesis 2 failing by
            # construction, not a solver event.  Both policies pay it alike.
            log["outcome"] = "tightened_infeasible"
            break
        t0 = time.time()
        try:
            seg, res = plan(obs, p, v)
        except Exception:                                        # noqa: BLE001
            log["outcome"] = "solver_fail"
            break
        log["solve_t"].append(time.time() - t0)
        if not res.get("converged"):
            log["outcome"] = "solver_fail"
            break
        rho = int(res["rho"])
        log["rho_seen"].append(rho)
        G = res["Gamma"]
        curve_certified = (rho == 0)
        if rho >= 1 and certified:
            t1 = time.time()
            vv, z, P, G0, _ = a46.lifted(seg, res)
            _, _, (Jb, wb), (Db, wd) = a46.search(seg, G0, vv, P, z,
                                                  nR=61, nT=121)
            G_ok = None
            for w in (wb, wd):
                if w is None:
                    continue
                ex, _ = a46.certify(seg, G0, vv, w)
                if ex >= 0.0:
                    G_ok = a46.fold(G0, vv, w)
                    break
            log["rec_t"].append(time.time() - t1)
            log["recoveries"] += 1
            if G_ok is None:
                log["stalls"] += 1        # never execute uncertified
                continue
            G, curve_certified = G_ok, True
        G_star = res["Gamma"]
        # execute [0, TAU] of the chosen curve, scored on the TRUE field
        Gexec = decasteljau_left(G, TAU)
        ex, exarg = exact_clearance(Gexec, obs, D)
        log["exec_clear"].append((float(ex), bool(curve_certified)))
        if history is not None:
            history.append(dict(
                p=p.tolist(), rho=rho,
                folded=bool(rho >= 1 and certified),
                G_star=(G_star.tolist() if rho >= 1 and certified
                        else res["Gamma"].tolist()),
                G=G.tolist(), kick=kicks[step, :2].tolist(),
                exec_clear=float(ex)))
        if ex < 0.0:
            log["outcome"] = "collision"
            break
        v = deriv_at(G, TAU, D) * (1.0 - TAU)
        p = curve_at(G, TAU, D)
        p = p + kicks[step, :2]
        v = v + kicks[step, 2:]
        if point_clearance(p, obs) < 0.0:
            log["outcome"] = "kicked_into_obstacle"
            break
    return log


def dump_showcase(indices):
    """Re-run the exact draw sequence of main() and save full per-step
    histories for the requested run indices, for the loop video."""
    rng = np.random.default_rng(20260827)
    shows = {}
    i_run = 0
    for i in range(N_RUNS):
        inst = random_instance(rng, n_obs=3, d=D, k=K, l=L,
                               rmin=0.55, rmax=0.9)
        if len(inst["obstacles"]) < 3:
            continue
        kicks = np.concatenate([rng.normal(0, SIGMA, (MAX_STEPS, 2)),
                                rng.normal(0, SIGMA / 2, (MAX_STEPS, 2))],
                               axis=1)
        if i in indices:
            hn, hc = [], []
            ln = run_policy(inst["obstacles"], kicks, False, history=hn)
            lc = run_policy(inst["obstacles"], kicks, True, history=hc)
            shows[str(i)] = dict(
                obstacles=[[c.tolist() if hasattr(c, "tolist") else list(c),
                            float(r)] for c, r in inst["obstacles"]],
                eps=EPS_PLAN, tau=TAU,
                naive=dict(outcome=ln["outcome"], steps=hn),
                cert=dict(outcome=lc["outcome"], steps=hc,
                          recoveries=lc["recoveries"]))
            print("  showcase %d: naive %s (%d steps), cert %s (%d steps, "
                  "%d recoveries)" % (i, ln["outcome"], len(hn),
                                      lc["outcome"], len(hc),
                                      lc["recoveries"]))
        i_run += 1
    out = os.path.join(ART, "a48_showcase.json")
    with open(out, "w") as fh:
        json.dump(shows, fh)
    print("  wrote %s" % out)


def main():
    rng = np.random.default_rng(20260827)
    runs = []
    for i in range(N_RUNS):
        inst = random_instance(rng, n_obs=3, d=D, k=K, l=L,
                               rmin=0.55, rmax=0.9)
        if len(inst["obstacles"]) < 3:
            continue
        kicks = np.concatenate([rng.normal(0, SIGMA, (MAX_STEPS, 2)),
                                rng.normal(0, SIGMA / 2, (MAX_STEPS, 2))],
                               axis=1)
        naive = run_policy(inst["obstacles"], kicks, certified=False)
        cert = run_policy(inst["obstacles"], kicks, certified=True)
        runs.append(dict(i=i, naive=naive, cert=cert))
        print("  run %2d  naive: %-20s cert: %-20s (recoveries %d, stalls %d)"
              % (i, naive["outcome"], cert["outcome"],
                 cert["recoveries"], cert["stalls"]))

    # ---- gates ----------------------------------------------------------
    worst_cert_exec = min((c for r in runs
                           for c, ok in r["cert"]["exec_clear"] if ok),
                          default=np.inf)
    n_succ_c = sum(r["cert"]["outcome"] == "success" for r in runs)
    n_succ_n = sum(r["naive"]["outcome"] == "success" for r in runs)
    n_coll_c = sum(r["cert"]["outcome"] == "collision" for r in runs)
    n_coll_n = sum(r["naive"]["outcome"] == "collision" for r in runs)
    n_kick_c = sum(r["cert"]["outcome"] == "kicked_into_obstacle" for r in runs)
    n_kick_n = sum(r["naive"]["outcome"] == "kicked_into_obstacle" for r in runs)
    n_belt_c = sum(r["cert"]["outcome"] == "tightened_infeasible" for r in runs)
    n_belt_n = sum(r["naive"]["outcome"] == "tightened_infeasible" for r in runs)
    n_events = sum(r["cert"]["recoveries"] for r in runs)
    n_stalls = sum(r["cert"]["stalls"] for r in runs)
    n_steps_c = sum(r["cert"]["steps"] for r in runs)
    # paired sanity: runs whose certified history never saw rho >= 1 must
    # match the naive run step for step
    same = [r for r in runs if r["cert"]["recoveries"] == 0]
    n_paired_ok = sum(r["cert"]["outcome"] == r["naive"]["outcome"]
                      for r in same)
    solve_ts = [t for r in runs for t in r["cert"]["solve_t"]]
    rec_ts = [t for r in runs for t in r["cert"]["rec_t"]]

    print("\n  certified: %d/%d success, %d curve collisions, %d kicked, "
          "%d belt-infeasible; naive: %d/%d, %d, %d, %d"
          % (n_succ_c, len(runs), n_coll_c, n_kick_c, n_belt_c,
             n_succ_n, len(runs), n_coll_n, n_kick_n, n_belt_n))
    print("  whose fault: executed-curve violations  certified %d, naive %d; "
          "post-disturbance  certified %d, naive %d"
          % (n_coll_c, n_coll_n, n_kick_c + n_belt_c, n_kick_n + n_belt_n))
    print("  recovery events %d over %d certified steps, stalls %d; "
          "worst certified executed clearance %.2e"
          % (n_events, n_steps_c, n_stalls, worst_cert_exec))
    print("  median solve %.0f ms, median recovery %.0f ms"
          % (1e3 * np.median(solve_ts), 1e3 * np.median(rec_ts)))

    gates = dict(
        a48a_loop_soundness=dict(worst_certified_exec_clearance=worst_cert_exec,
                                 passed=bool(worst_cert_exec >= -1e-9)),
        a48b_paired=dict(n_runs=len(runs),
                         cert_success=n_succ_c, naive_success=n_succ_n,
                         cert_curve_collisions=n_coll_c,
                         naive_curve_collisions=n_coll_n,
                         cert_kicked=n_kick_c, naive_kicked=n_kick_n,
                         cert_belt_infeasible=n_belt_c,
                         naive_belt_infeasible=n_belt_n,
                         n_zero_recovery_runs=len(same),
                         n_paired_identical=n_paired_ok,
                         passed=bool(n_succ_c >= n_succ_n and n_coll_c == 0
                                     and n_paired_ok == len(same))),
        a48c_recovery_exercised=dict(events=n_events,
                                     median_solve_ms=1e3 * float(np.median(solve_ts)),
                                     median_recovery_ms=(1e3 * float(np.median(rec_ts))
                                                         if rec_ts else None),
                                     passed=bool(n_events > 0)),
        a48d_stalls=dict(stalls=n_stalls, cert_steps=n_steps_c,
                         passed=bool(n_stalls <= 0.10 * n_steps_c)),
        a48e_attribution=dict(
            cert_curve_faults=n_coll_c, naive_curve_faults=n_coll_n,
            cert_disturbance_faults=n_kick_c + n_belt_c,
            naive_disturbance_faults=n_kick_n + n_belt_n,
            passed=bool(n_coll_c == 0 and n_coll_n > 0)),
    )
    out = os.path.join(ART, "a48_loop_recovery.json")
    with open(out, "w") as fh:
        json.dump(dict(gates=gates,
                       runs=[dict(i=r["i"],
                                  naive=r["naive"]["outcome"],
                                  cert=r["cert"]["outcome"],
                                  recoveries=r["cert"]["recoveries"],
                                  stalls=r["cert"]["stalls"]) for r in runs]),
                  fh, indent=1)
    ok = all(g["passed"] for g in gates.values())
    print("  wrote %s" % out)
    print("  ALL GATES PASSED" if ok else "  *** A GATE FAILED ***")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--showcase":
        sys.exit(dump_showcase({int(x) for x in sys.argv[2:]}) or 0)
    sys.exit(main())
