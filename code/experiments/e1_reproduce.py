"""Phase 1 -- reproduce the relaxation.  Gates 1b, 1c, 1d, 1f.

Gate 1b  obstacle-free min-energy cost is exactly 16, equispaced, rho = 0.
Gate 1c  Theorem 3 on four named instances.
Gate 1d  on >= 200 random instances:  c*_SDP <= c*_P + 1e-6 always, and
         c*_SDP == c*_P to 1e-5 whenever rho == 0.
Gate 1f  the spectrum of S = X - Gamma^T Gamma (lam_1 and lam_2) over all
         instances -- this sets the solver accuracy the rank test needs.

Ground truth is verified for *continuous-time* feasibility (exact polynomial
minimisation), not by sampling: a sampled ground truth cuts corners between
samples, lands below the true c*_P, and fires the Theorem-2 tripwire on a
correct relaxation.

Writes artifacts/gate1.json.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
ART = os.path.join(os.path.dirname(__file__), "..", "artifacts")

from relaxation import Segment, exact_clearance      # noqa: E402
from groundtruth import GroundTruth                  # noqa: E402
from instances import random_instance, START, GOAL   # noqa: E402

N_INSTANCES = 220
N_STARTS = 60


# ---------------------------------------------------------------- Gate 1b
def gate1b():
    seg = Segment(n=2, d=3, k=1, l=0, obstacles=[], bc0=[START], bc1=[GOAL])
    res = seg.solve()
    expect = np.array([[-2.0, -2.0 / 3.0, 2.0 / 3.0, 2.0], [0.0] * 4])
    out = dict(cost=res["cost"], rho=res["rho"],
               cost_err=abs(res["cost"] - 16.0),
               ctrl_pts=res["Gamma"].tolist(),
               ctrl_err=float(np.max(np.abs(res["Gamma"] - expect))),
               passed=bool(abs(res["cost"] - 16.0) < 1e-8 and res["rho"] == 0
                           and np.max(np.abs(res["Gamma"] - expect)) < 1e-6))
    print(f"[Gate 1b] cost={res['cost']:.12f} (err {out['cost_err']:.2e})  "
          f"rho={res['rho']}  ctrl-pt err={out['ctrl_err']:.2e}  "
          f"-> {'PASS' if out['passed'] else 'FAIL'}")
    return out


# ---------------------------------------------------------------- Gate 1c
NAMED = {
    "symmetric": [(np.array([0.0, 0.0]), 0.8)],
    "offset":    [(np.array([0.0, 0.62]), 0.8)],
    "two_obs":   [(np.array([-0.8, 0.35]), 0.5), (np.array([0.8, -0.35]), 0.5)],
    "corridor":  [(np.array([0.0, 1.05]), 0.8), (np.array([0.0, -1.05]), 0.8)],
}


def gate1c():
    rows, ok = {}, True
    for name, obs in NAMED.items():
        seg = Segment(n=2, d=3, k=1, l=0, obstacles=obs, bc0=[START], bc1=[GOAL])
        res = seg.solve()
        lift = seg.lifted_clearance(res)
        proj, _ = exact_clearance(res["Gamma"], seg.obs, seg.d)
        good = lift >= -1e-8 and ((proj >= -1e-7) == (res["rho"] == 0))
        ok &= good
        rows[name] = dict(rho=res["rho"], cost=res["cost"],
                          lifted_clearance=lift, projected_p_min=proj,
                          projected_dist=seg.min_clearance(res["Gamma"]),
                          lam1=res["lam1"], lam2=res["lam2"], passed=bool(good))
        print(f"[Gate 1c] {name:10s} rho={res['rho']}  cost={res['cost']:.6f}  "
              f"lifted={lift:+.2e}  projected p_min={proj:+.3e}  "
              f"-> {'PASS' if good else 'FAIL'}")
    return dict(cases=rows, passed=bool(ok))


# ----------------------------------------------------------- Gates 1d/1f
def gate1d_1f(n_instances=N_INSTANCES, seed=12345):
    rng = np.random.default_rng(seed)
    recs = []
    t0 = time.perf_counter()
    for i in range(n_instances):
        inst = random_instance(rng)
        seg = Segment(**inst)
        res = seg.solve()
        if not res.get("converged"):
            recs.append(dict(i=i, sdp_status=res["status"], skipped=True))
            continue
        gt = GroundTruth(**inst)
        g = gt.solve(ntry=N_STARTS, seed=i)
        lift = seg.lifted_clearance(res)
        proj, _ = exact_clearance(res["Gamma"], seg.obs, seg.d)
        rec = dict(i=i, skipped=False, n_obs=len(inst["obstacles"]),
                   obstacles=[[float(c[0]), float(c[1]), float(r)]
                              for c, r in inst["obstacles"]],
                   rho=res["rho"], c_sdp=res["cost"],
                   lam1=res["lam1"], lam2=res["lam2"],
                   lifted_clearance=lift, projected_p_min=proj,
                   gt_ok=bool(g["ok"]),
                   c_p=g["cost"] if g["ok"] else None,
                   n_distinct=g.get("n_distinct", 0))
        if g["ok"]:
            rec["gap"] = g["cost"] - res["cost"]
            rec["rel_gap"] = rec["gap"] / max(1.0, abs(g["cost"]))
        recs.append(rec)
        if (i + 1) % 25 == 0:
            print(f"    ... {i+1}/{n_instances}  "
                  f"({time.perf_counter()-t0:.0f}s elapsed)")

    live = [r for r in recs if not r["skipped"] and r["gt_ok"]]
    n_skipped = sum(1 for r in recs if r["skipped"])
    n_nogt = sum(1 for r in recs if not r["skipped"] and not r["gt_ok"])

    # --- Gate 1d
    viol = [r for r in live if r["gap"] < -1e-6]
    tight = [r for r in live if r["rho"] == 0]
    tight_bad = [r for r in tight if abs(r["gap"]) > 1e-5]
    lifted_bad = [r for r in recs
                  if not r["skipped"] and r["lifted_clearance"] < -1e-8]

    d1 = dict(n_total=n_instances, n_evaluated=len(live),
              n_sdp_failed=n_skipped, n_gt_failed=n_nogt,
              n_theorem2_violations=len(viol),
              worst_theorem2_gap=float(min((r["gap"] for r in live), default=0.0)),
              n_rho0=len(tight),
              n_rho0_mismatch=len(tight_bad),
              worst_rho0_err=float(max((abs(r["gap"]) for r in tight), default=0.0)),
              n_lifted_violations=len(lifted_bad),
              violations=[r["i"] for r in viol],
              rho0_mismatches=[r["i"] for r in tight_bad],
              passed=bool(not viol and not tight_bad and not lifted_bad))
    print(f"\n[Gate 1d] evaluated {len(live)}/{n_instances}  "
          f"(SDP failed {n_skipped}, GT failed {n_nogt})")
    print(f"          Thm 2 violations (c_SDP > c_P + 1e-6): {len(viol)}   "
          f"worst gap {d1['worst_theorem2_gap']:+.3e}")
    print(f"          rho=0 instances: {len(tight)}/{len(live)}  "
          f"({100*len(tight)/max(1,len(live)):.1f}% tight at the root);  "
          f"mismatches {len(tight_bad)}, worst |c_SDP-c_P| {d1['worst_rho0_err']:.3e}")
    print(f"          lifted-curve violations (Thm 3): {len(lifted_bad)}  "
          f"-> {'PASS' if d1['passed'] else 'FAIL'}")

    # --- rho distribution (Gate 1e's numeric half)
    rho_hist = {}
    for r in recs:
        if not r["skipped"]:
            rho_hist[r["rho"]] = rho_hist.get(r["rho"], 0) + 1
    print(f"\n[rho distribution] {dict(sorted(rho_hist.items()))}   "
          f"(Lemma 4 bound: rho <= f = 2)")

    # --- Gate 1f
    l1 = np.array([r["lam1"] for r in recs if not r["skipped"]])
    l2 = np.array([r["lam2"] for r in recs if not r["skipped"]])
    pos = l1 > 1e-6
    d6 = dict(
        lam1=dict(min=float(l1.min()), median=float(np.median(l1)),
                  max=float(l1.max())),
        lam2=dict(min=float(l2.min()), median=float(np.median(l2)),
                  max=float(l2.max())),
        lam1_when_rho_ge_1=dict(
            n=int(pos.sum()),
            min=float(l1[pos].min()) if pos.any() else None,
            median=float(np.median(l1[pos])) if pos.any() else None),
        lam2_when_rho_ge_1=dict(
            max=float(l2[pos].max()) if pos.any() else None),
        max_abs_lam2_when_rho0=float(np.abs(l2[~pos]).max()) if (~pos).any() else None,
        rho_hist={int(k): int(v) for k, v in sorted(rho_hist.items())},
    )
    print(f"[Gate 1f] lam_1(S): min {l1.min():+.3e}  med {np.median(l1):+.3e}  "
          f"max {l1.max():+.3e}")
    print(f"          lam_2(S): min {l2.min():+.3e}  med {np.median(l2):+.3e}  "
          f"max {l2.max():+.3e}")
    if pos.any():
        print(f"          when rho>=1: smallest lam_1 = {l1[pos].min():.3e}, "
              f"largest lam_2 = {l2[pos].max():.3e}  "
              f"-> spectral gap spans {l1[pos].min()/max(abs(l2[pos].max()),1e-16):.1e}x")
    return d1, d6, recs


def main():
    os.makedirs(ART, exist_ok=True)
    print("=== Phase 1: reproduce the relaxation ===\n")
    r1b = gate1b()
    print()
    r1c = gate1c()
    print()
    r1d, r1f, recs = gate1d_1f()

    report = dict(gate1b=r1b, gate1c=r1c, gate1d=r1d, gate1f=r1f,
                  n_starts_groundtruth=N_STARTS, records=recs)
    path = os.path.join(ART, "gate1.json")
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {path}")

    allpass = r1b["passed"] and r1c["passed"] and r1d["passed"]
    print(f"\n=== PHASE 1 {'PASS' if allpass else 'FAIL'} ===")
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
