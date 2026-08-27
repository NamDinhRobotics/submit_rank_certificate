"""A1 -- `rho` is bounded by the NUMBER OF CONTACTS, and two mechanisms for
`rho <= 1` are refuted on the way.

Their open problem #2 asks for tighter bounds on `rho` (docs/open_problems_map.md).
Phase 11 established `rho <= dim ker(K - M_lambda)` with `M_lambda` the moment
matrix of the contact measure.  Since `M_lambda = sum_a w_a u_a u_a^T` with
`u_a = b_d(s_a)|_F` one atom per contact, its rank is at most the number of
atoms, and `K > 0`, so

    rho  <=  dim ker(K - M_lambda)  <=  rank(M_lambda)  <=  #contacts.        (*)

Proof of the middle step: `Zx = 0` means `Kx = M_lambda x in range(M_lambda)`,
and `K` is invertible, so `ker(Z) -> range(M_lambda)`, `x |-> Kx`, is injective.

(*) is rigorous for EVERY `k`, `d`, `n` and for multisegment curves -- unlike the
Wronskian sharpening of Phase 12, which is `k = 1` only.  It also explains
tightness at a stroke: no contact means `M_lambda = 0`, hence `Z = K > 0` and
`rho = 0`.

REDUCTION.  Writing `U = [u_1 ... u_m]`, `W = diag(w)`, the nonzero eigenvalues
of `K^{-1/2} M_lambda K^{-1/2}` are those of the `m x m` Green's-function Gram

    Gr = U^T K^{-1} U,      spectrum of  W^{1/2} Gr W^{1/2},

and dual feasibility `Z >= 0` forces all of them `<= 1`.  So `dim ker(Z)` is the
multiplicity of the **largest** eigenvalue of an `m x m` matrix -- which is why
two classical mechanisms suggest themselves, and this experiment refutes both:

  * total positivity of `Gr` (Gantmacher-Krein: TP => simple eigenvalues) --
    REFUTED, minors of order >= 2 go negative;
  * entrywise positivity of `Gr` (Perron-Frobenius: a positive matrix has a
    simple spectral radius) -- REFUTED, `K^{-1}` is not entrywise positive and
    `Gr` inherits negative entries.

What survives is a measured margin: the relative separation of the top two
eigenvalues never collapses.  Reported, not promoted to a theorem.

Gates:
  a1a  rho <= rank(M_lambda) on every loose instance     (theorem; fail=numerics)
  a1b  rank(M_lambda) <= #contacts detected              (the atom structure)
  a1c  the k-sweep: (*) holds at k >= 2, where the Wronskian argument does not
  a1d  the two refuted mechanisms are recorded with counterexample magnitudes
  a1e  (*) is strictly tighter than their Lemma 4 (rho <= f) on some instances,
       and the share is reported rather than asserted

Writes artifacts/a1_contact_rank.json.
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
ART = os.path.join(os.path.dirname(__file__), "..", "artifacts")

from relaxation import Segment                              # noqa: E402
from node import Node, build                                # noqa: E402
from bernstein import bd                                    # noqa: E402
from instances import hard_instances                        # noqa: E402


def bcs(l):
    b0 = [[-2.0, 0.0], [4.0, 0.0]] + [[0.0, 0.0]] * max(0, l - 1)
    b1 = [[2.0, 0.0], [4.0, 0.0]] + [[0.0, 0.0]] * max(0, l - 1)
    return b0[:l + 1], b1[:l + 1]


def num_rank(A, rtol=1e-7):
    w = np.linalg.eigvalsh(0.5 * (A + A.T))
    scale = max(1.0, float(np.abs(w).max()))
    return int(np.sum(w > rtol * scale)), [float(x) for x in np.sort(w)[::-1]]


# ----------------------------------------------------------------------
# part 1 -- structural: are the two classical mechanisms available?
# ----------------------------------------------------------------------
def minors_by_order(A):
    """min over minors of each order, on the diagonally normalised matrix."""
    m = A.shape[0]
    dg = np.sqrt(np.abs(np.diag(A)))
    dg[dg == 0] = 1.0
    N = A / np.outer(dg, dg)
    out = {}
    for r in range(1, m + 1):
        worst = np.inf
        for R in itertools.combinations(range(m), r):
            for C in itertools.combinations(range(m), r):
                worst = min(worst, float(np.linalg.det(N[np.ix_(R, C)])))
        out[r] = worst
    return out


def structural_probe(configs, ms=(2, 3, 4), trials=300, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for (d, k, l) in configs:
        b0, b1 = bcs(l)
        seg = Segment(**dict(hard_instances(d=d, k=k, l=l)["symmetric_blocker"],
                             bc0=b0, bc1=b1))
        K = seg.Gk[np.ix_(seg.Fidx, seg.Fidx)]
        Ki = np.linalg.inv(K)
        rec = dict(d=d, k=k, l=l, f=int(seg.f),
                   min_Kinv_entry=float(Ki.min()),
                   min_minor_order={}, min_Gr_entry_norm=np.inf,
                   min_rel_eig_gap=np.inf)
        for m in ms:
            if m > seg.f:
                continue
            worst_ord = {r: np.inf for r in range(1, m + 1)}
            for _ in range(trials):
                ss = np.sort(rng.uniform(0.0, 1.0, m))
                if np.min(np.diff(ss)) < 1e-3:
                    continue
                U = np.array([bd(s, d)[seg.Fidx] for s in ss]).T
                Gr = U.T @ Ki @ U
                mo = minors_by_order(Gr)
                for r, v in mo.items():
                    worst_ord[r] = min(worst_ord[r], v)
                rec["min_Gr_entry_norm"] = min(rec["min_Gr_entry_norm"], mo[1])
                w = np.sort(np.linalg.eigvalsh(0.5 * (Gr + Gr.T)))[::-1]
                if w.size > 1 and w[0] > 0:
                    rec["min_rel_eig_gap"] = min(
                        rec["min_rel_eig_gap"], float((w[0] - w[1]) / w[0]))
            rec["min_minor_order"][str(m)] = {str(r): float(v)
                                              for r, v in worst_ord.items()}
        rec["min_Gr_entry_norm"] = float(rec["min_Gr_entry_norm"])
        rec["min_rel_eig_gap"] = float(rec["min_rel_eig_gap"])
        rows.append(rec)
    return rows


# ----------------------------------------------------------------------
# part 2 -- the population: does (*) hold, and how tight is it?
# ----------------------------------------------------------------------
def analyse(seg, contact_tol=1e-4, ns=4001):
    from scipy.optimize import nnls
    nd = Node(lo=None, hi=None)
    h = build(seg, nd, use_rlt=False, use_lift=False)
    h["prob"].solve(solver="CLARABEL", tol_gap_abs=1e-10, tol_gap_rel=1e-10,
                    tol_feas=1e-10, max_iter=2000, verbose=False)
    if h["prob"].status != "optimal" or h["Gfree"].value is None:
        return dict(status=h["prob"].status)
    Gf = np.asarray(h["Gfree"].value)
    Xf = 0.5 * (np.asarray(h["Xfree"].value) + np.asarray(h["Xfree"].value).T)
    S = Xf - Gf.T @ Gf
    wS = np.linalg.eigvalsh(S)
    rho = int(np.sum(wS > 1e-6 * max(1.0, float(np.abs(wS).max()))))

    Zf = np.asarray(h["cons"][0].dual_value)
    Zf = 0.5 * (Zf + Zf.T)
    Z = Zf[seg.n:, seg.n:]
    K = seg.Gk[np.ix_(seg.Fidx, seg.Fidx)]
    M = K - Z
    rank_M, spec_M = num_rank(M)
    wZ = np.sort(np.linalg.eigvalsh(Z))
    kerdim = int(np.sum(np.abs(wZ) < 1e-6 * max(1.0, float(np.abs(wZ).max()))))

    # eigenvalues of K^{-1/2} M K^{-1/2}: dual feasibility puts them all <= 1,
    # and dim ker(Z) is the multiplicity of the largest.
    Lw, Lv = np.linalg.eigh(K)
    Kmh = Lv @ np.diag(1.0 / np.sqrt(np.maximum(Lw, 1e-300))) @ Lv.T
    mu = np.sort(np.linalg.eigvalsh(Kmh @ M @ Kmh))[::-1]
    mu1 = float(mu[0])
    mu2 = float(mu[1]) if mu.size > 1 else 0.0

    res = seg._package(Gf, Xf, float(h["prob"].value))
    V = seg.lifted_V(res)
    ss = np.linspace(0.0, 1.0, ns)
    B = np.array([bd(s, seg.d) for s in ss])
    P = B @ res["Gamma"].T
    vv = np.sum((B @ V.T) ** 2, axis=1) if V.size else np.zeros(ns)
    zeta = np.full(ns, np.inf)
    for c, r in seg.obs:
        zeta = np.minimum(zeta, np.sum((P - c) ** 2, axis=1) + vv - r * r)
    contacts = [float(ss[i]) for i in range(1, ns - 1)
                if zeta[i] <= zeta[i - 1] and zeta[i] <= zeta[i + 1]
                and zeta[i] < contact_tol]

    fit_res, n_pos, gr_min, gr_gap = None, None, None, None
    if contacts:
        U = np.array([bd(s, seg.d)[seg.Fidx] for s in contacts]).T
        A = np.array([np.outer(U[:, a], U[:, a]).ravel()
                      for a in range(U.shape[1])]).T
        w, rn = nnls(A, M.ravel())
        fit_res = float(rn / max(1e-12, np.linalg.norm(M)))
        n_pos = int(np.sum(w > 1e-8 * max(1.0, float(w.max()))))
        Ki = np.linalg.inv(K)
        Gr = U.T @ Ki @ U
        dgg = np.sqrt(np.abs(np.diag(Gr)))
        gr_min = float((Gr / np.outer(dgg, dgg)).min())
        ev = np.sort(np.linalg.eigvalsh(0.5 * (Gr + Gr.T)))[::-1]
        gr_gap = float((ev[0] - ev[1]) / ev[0]) if ev.size > 1 and ev[0] > 0 else None

    return dict(status="optimal", f=int(seg.f), k=int(seg.k), d=int(seg.d),
                l=int(seg.l), rho=rho, ker_dim=kerdim, rank_M=rank_M,
                spec_M=spec_M[:4], mu1=mu1, mu2=mu2, mu_gap=float(mu1 - mu2),
                n_contacts=len(contacts), contacts=contacts,
                n_atoms_positive=n_pos, atom_fit_residual=fit_res,
                gr_min_entry_norm=gr_min, gr_rel_eig_gap=gr_gap)


def loose_population():
    out = []
    recs = json.load(open(os.path.join(ART, "gate1.json")))["records"]
    for r in recs:
        if r.get("skipped") or r["rho"] < 1:
            continue
        obs = [(np.array([o[0], o[1]]), float(o[2])) for o in r["obstacles"]]
        out.append((f"gate1[{r['i']}]",
                    dict(n=2, d=3, k=1, l=0, obstacles=obs,
                         bc0=[[-2.0, 0.0]], bc1=[[2.0, 0.0]])))
    H = hard_instances()
    for nm in ("symmetric_blocker", "staggered_pair", "three_in_a_row"):
        out.append((nm, H[nm]))
    for nm in ("symmetric_blocker", "staggered_pair"):
        for d in (5, 7, 9):
            out.append((f"{nm} d={d}", dict(H[nm], d=d)))
    return out


def k_sweep_population():
    """k >= 2, where the Phase-12 Wronskian sharpening does NOT apply."""
    out = []
    for k in (1, 2, 3, 4):
        l = k - 1
        for extra in (0, 2):
            d = 2 * k + 1 + extra
            b0, b1 = bcs(l)
            for nm in ("symmetric_blocker", "staggered_pair", "three_in_a_row"):
                out.append((f"{nm} k={k} d={d}",
                            dict(hard_instances(d=d, k=k, l=l)[nm],
                                 bc0=b0, bc1=b1)))
    return out


# ----------------------------------------------------------------------
def run_pop(pop, label):
    rows = []
    print(f"\n  --- {label} ---")
    print("  %-24s %2s %2s %2s %3s %4s %6s %5s %7s %7s %9s"
          % ("instance", "k", "d", "f", "rho", "rkM", "#cts", "atoms",
             "mu1", "gap", "fit"))
    for name, kw in pop:
        seg = Segment(**kw)
        r = analyse(seg)
        if r.get("status") != "optimal" or r["rho"] < 1:
            continue
        rows.append(dict(name=name, **{k: v for k, v in r.items()
                                       if k != "status"}))
        print("  %-24s %2d %2d %2d %3d %4d %6d %5s %7.4f %7.4f %9s"
              % (name, r["k"], r["d"], r["f"], r["rho"], r["rank_M"],
                 r["n_contacts"], str(r["n_atoms_positive"]), r["mu1"],
                 r["mu_gap"],
                 "-" if r["atom_fit_residual"] is None
                 else "%.1e" % r["atom_fit_residual"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--out", default=os.path.join(ART, "a1_contact_rank.json"))
    args = ap.parse_args()

    print("=== A1: rho is bounded by the number of contacts ===")

    CONFIGS = ((3, 1, 0), (5, 1, 0), (7, 1, 0), (9, 1, 0), (11, 1, 0),
               (5, 2, 1), (7, 2, 1), (9, 2, 1), (11, 2, 1),
               (7, 3, 2), (9, 3, 2), (11, 3, 2), (9, 4, 3), (11, 4, 3))
    print("\n  --- part 1: are the classical simplicity mechanisms available? ---")
    print("  %3s %3s %3s %3s %14s %13s %13s %11s"
          % ("d", "k", "l", "f", "min K^-1 entry", "min Gr entry",
             "min minor >=2", "min eig gap"))
    struct = structural_probe(CONFIGS, trials=args.trials)
    for r in struct:
        deep = [v for mo in r["min_minor_order"].values()
                for rr, v in mo.items() if int(rr) >= 2]
        r["min_minor_order2plus"] = float(min(deep)) if deep else None
        print("  %3d %3d %3d %3d %14.2e %13.2e %13.2e %11.4f"
              % (r["d"], r["k"], r["l"], r["f"], r["min_Kinv_entry"],
                 r["min_Gr_entry_norm"],
                 r["min_minor_order2plus"] if r["min_minor_order2plus"]
                 is not None else float("nan"), r["min_rel_eig_gap"]))

    tp_ok = all((r["min_minor_order2plus"] or 0) > 0 for r in struct)
    pf_ok = all(r["min_Kinv_entry"] > 0 and r["min_Gr_entry_norm"] > 0
                for r in struct)
    min_gap_struct = float(min(r["min_rel_eig_gap"] for r in struct))
    print(f"\n    total positivity of Gr (Gantmacher-Krein route): "
          f"{'AVAILABLE' if tp_ok else 'REFUTED'}")
    print(f"    entrywise positivity of Gr (Perron-Frobenius route): "
          f"{'AVAILABLE' if pf_ok else 'REFUTED'}")
    print(f"    measured relative gap of the top two eigenvalues of Gr, "
          f"worst over all configs and random contact tuples: {min_gap_struct:.4f}")

    loose = run_pop(loose_population(), "part 2: the loose population (k = 1)")
    ksw = run_pop(k_sweep_population(),
                  "part 3: the k-sweep (k >= 2: no Wronskian argument)")

    allrows = loose + ksw
    a1a = [r for r in allrows if not r["rho"] <= r["rank_M"]]
    a1b = [r for r in allrows
           if r["n_contacts"] and not r["rank_M"] <= r["n_contacts"]]
    a1c = [r for r in ksw if not (r["rho"] <= r["rank_M"]
                                  and (not r["n_contacts"]
                                       or r["rank_M"] <= r["n_contacts"]))]
    tighter = [r for r in allrows if r["n_contacts"] and r["n_contacts"] < r["f"]]
    gaps = [r["mu_gap"] for r in allrows]
    grgaps = [r["gr_rel_eig_gap"] for r in allrows
              if r["gr_rel_eig_gap"] is not None]
    grmin = [r["gr_min_entry_norm"] for r in allrows
             if r["gr_min_entry_norm"] is not None]

    gates = dict(
        a1a_rho_le_rank_M=dict(n=len(allrows), n_violations=len(a1a),
                               passed=not a1a),
        a1b_rank_M_le_contacts=dict(n=len(allrows), n_violations=len(a1b),
                                    passed=not a1b),
        a1c_holds_at_k_ge_2=dict(n=len(ksw), n_violations=len(a1c),
                                 passed=not a1c),
        a1d_mechanisms_refuted=dict(
            total_positivity_available=bool(tp_ok),
            perron_frobenius_available=bool(pf_ok),
            worst_minor_order2plus=float(min((r["min_minor_order2plus"] or 0)
                                             for r in struct)),
            worst_Kinv_entry=float(min(r["min_Kinv_entry"] for r in struct)),
            worst_Gr_entry_norm=float(min(r["min_Gr_entry_norm"] for r in struct)),
            min_rel_eig_gap_random=min_gap_struct,
            passed=bool((not tp_ok) and (not pf_ok))),
        a1e_tighter_than_lemma4=dict(
            # the population is a1's own: the k=1 loose set plus the k-sweep.
            # Recorded so the text cannot describe it as something else.
            n=len(allrows), n_loose_k1=len(loose), n_ksweep=len(ksw),
            n_strictly_tighter=len(tighter),
            share=len(tighter) / max(1, len(allrows)),
            examples=[dict(name=r["name"], f=r["f"], n_contacts=r["n_contacts"],
                           rho=r["rho"]) for r in tighter[:12]]),
        margins=dict(
            min_mu_gap=float(min(gaps)) if gaps else None,
            median_mu_gap=float(np.median(gaps)) if gaps else None,
            min_mu1=float(min(r["mu1"] for r in allrows)) if allrows else None,
            max_mu1=float(max(r["mu1"] for r in allrows)) if allrows else None,
            min_Gr_rel_eig_gap_at_contacts=float(min(grgaps)) if grgaps else None,
            min_Gr_entry_at_contacts=float(min(grmin)) if grmin else None),
    )

    print("\n  --- gates ---")
    for nm in ("a1a_rho_le_rank_M", "a1b_rank_M_le_contacts",
               "a1c_holds_at_k_ge_2", "a1d_mechanisms_refuted"):
        print(f"  {nm}: {'PASS' if gates[nm]['passed'] else 'FAIL'}")
    e = gates["a1e_tighter_than_lemma4"]
    print(f"  a1e: #contacts < f (strictly tighter than their Lemma 4) on "
          f"{e['n_strictly_tighter']}/{e['n']} = {100*e['share']:.1f}%")
    mg = gates["margins"]
    print(f"\n  mu_1 (largest eigenvalue of K^-1/2 M K^-1/2, must be <= 1): "
          f"[{mg['min_mu1']:.6f}, {mg['max_mu1']:.6f}]")
    print(f"  distance from mu_2 to mu_1: min {mg['min_mu_gap']:.4f}, "
          f"median {mg['median_mu_gap']:.4f}")
    print(f"  at the ACTUAL contact points: min normalised Gr entry "
          f"{mg['min_Gr_entry_at_contacts']:+.4f}, "
          f"min relative top-two eigen gap {mg['min_Gr_rel_eig_gap_at_contacts']:.4f}")

    with open(args.out, "w") as fh:
        json.dump(dict(gates=gates, structural=struct, loose=loose,
                       k_sweep=ksw), fh, indent=1, default=float)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
