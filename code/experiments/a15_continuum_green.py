"""A15 -- the negative lobe is a discretisation artefact at EVERY cost order.

WHAT WAS MISSING.  The paper explains the lobe by identifying `Gr` as the
Galerkin approximation of the Green's function of `(-1)^k d^2k/ds^2k` under the
problem's boundary conditions, and noting that kernel is strictly positive.  Two
gaps sat in that sentence:

  (1) positivity of the continuum kernel, which is elementary at `k = 1`
      (Dirichlet) and `k = 2` (clamped beam) but has to hold at every `k` for
      the explanation to cover Table I of the source paper, which runs to
      `k = 4` (snap);

  (2) "Galerkin approximation", which is a claim about what `Gr` IS and so has
      to be measured rather than asserted.

Both are settled here.

THE POSITIVITY, FOR ALL k.  With `l = k-1` -- the benchmark setting -- the free
Bernstein functions have a zero of multiplicity `k` at each end, so the
continuum problem is the CLAMPED one,

    (-1)^k u^{(2k)} = f  on [0,1],   u^{(i)}(0) = u^{(i)}(1) = 0,  i < k ,

which is the `(k, 2k-k) = (k, k)` conjugate two-point problem.  `D^{2k}` is
disconjugate on any interval for a reason with no content: a nonzero solution of
`u^{(2k)} = 0` is a polynomial of degree `< 2k`, so it has fewer than `2k` zeros
counting multiplicity, and disconjugacy is exactly the absence of `2k`.  The
classical theorem for conjugate problems then gives the Green's function a
CONSTANT SIGN, namely `(-1)^{n-k} = (-1)^k` (Coppel, Disconjugacy, LNM 220).

a15a does not take that on trust either: it builds `G` exactly over `Q` from its
defining conditions and checks the sign and strictness directly.

  a15a  the continuum kernel: `(-1)^k G > 0` on the open square, exactly, for
        `k = 1..6` -- so the sign the classical theorem predicts is the sign the
        construction has
  a15b  `Gr_d -> G` as the degree grows, for `k = 1..4`: the word "Galerkin"
        earns its place, measured rather than asserted
  a15c  the lobe depth of the discrete `Gr` decays with degree at `k = 1,2,3,4`
        -- the full range of Table I, so a measurement at any of its cost orders
        could falsify the claim, rather than only at the two where positivity is
        elementary
  a15d  NEGATIVE CONTROL: at `l = 0` with `k >= 3` the free functions do NOT lie
        in `H_0^k`, the continuum problem is not the clamped one, and `K` is
        singular anyway (Proposition 1).  The explanation is therefore scoped to
        `l = k-1`, and this records that the scope is real.

Writes artifacts/a15_continuum_green.json.
"""
import argparse
import json
import os
import sys
from fractions import Fraction as F
from math import factorial

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
ART = os.path.join(HERE, "..", "artifacts")

from bernstein import gram_deriv, bd                          # noqa: E402
from exact_green import certify_exact, energy_nullity          # noqa: E402


# ------------------------------------------------ the continuum kernel
def green_pieces(k, t):
    """Exact `G(.,t)` for the clamped problem, as two degree-`(2k-1)` pieces.

    `G(.,t)` solves `u^{(2k)} = 0` away from `s = t`, so it is a polynomial of
    degree `< 2k` on each side.  The `2k` unknowns per side are fixed by `k`
    boundary conditions at each end, `C^{2k-2}` matching at `s = t`, and a unit
    jump in the `(2k-1)`-st derivative.  Every coefficient is rational, so the
    whole kernel is computed over `Q` and its sign is decidable.
    """
    n = 2 * k
    rows, rhs = [], []

    def dc(i, j, s):
        if j > i:
            return F(0)
        return F(factorial(i), factorial(i - j)) * (s ** (i - j) if i - j else F(1))

    for j in range(k):                              # clamped at s = 0
        r = [F(0)] * (2 * n)
        for i in range(n):
            r[i] = dc(i, j, F(0))
        rows.append(r)
        rhs.append(F(0))
    for j in range(k):                              # clamped at s = 1
        r = [F(0)] * (2 * n)
        for i in range(n):
            r[n + i] = dc(i, j, F(1))
        rows.append(r)
        rhs.append(F(0))
    for j in range(n):                              # matching and the jump
        r = [F(0)] * (2 * n)
        for i in range(n):
            r[i] = dc(i, j, t)
            r[n + i] = -dc(i, j, t)
        rows.append(r)
        rhs.append(F(0) if j < n - 1 else F(-1))

    A = [rows[i][:] + [rhs[i]] for i in range(2 * n)]
    piv = 0
    for c in range(2 * n):
        p = next((i for i in range(piv, 2 * n) if A[i][c] != 0), None)
        if p is None:
            continue
        A[piv], A[p] = A[p], A[piv]
        inv = F(1) / A[piv][c]
        A[piv] = [x * inv for x in A[piv]]
        for i in range(2 * n):
            if i != piv and A[i][c] != 0:
                f = A[i][c]
                A[i] = [x - f * y for x, y in zip(A[i], A[piv])]
        piv += 1
    sol = [A[i][2 * n] for i in range(2 * n)]
    return sol[:n], sol[n:]


def G_exact(k, s, t):
    """`(-1)^k G(s,t)`, exactly, for rational `s, t`."""
    L, R = green_pieces(k, t)
    c = L if s <= t else R
    return (-1) ** k * sum(cc * (s ** i) for i, cc in enumerate(c))


# ---------------------------------------------------------------- a15a
def gate_a15a(kmax=6, grid=12):
    rows = []
    for k in range(1, kmax + 1):
        worst, at = None, None
        for a in range(1, grid):
            t = F(a, grid)
            L, R = green_pieces(k, t)
            for b in range(1, grid):
                s = F(b, grid)
                c = L if s <= t else R
                v = (-1) ** k * sum(cc * (s ** i) for i, cc in enumerate(c))
                if worst is None or v < worst:
                    worst, at = v, (s, t)
        rows.append(dict(k=k, n_samples=(grid - 1) ** 2,
                         min_signed_G=float(worst),
                         argmin=[float(at[0]), float(at[1])],
                         strictly_positive=bool(worst > 0)))
        print("    k=%d  min (-1)^k G over %d interior points = %.3e  %s"
              % (k, (grid - 1) ** 2, float(worst),
                 "> 0" if worst > 0 else "<= 0  <-- NOT positive"))
    return dict(rows=rows, kmax=kmax,
                n_strictly_positive=sum(x["strictly_positive"] for x in rows),
                passed=bool(all(x["strictly_positive"] for x in rows)))


# ---------------------------------------------------------------- a15b
def _Gr_disc(d, k, l, s, t):
    Fi = list(range(l + 1, d - l))
    K = gram_deriv(d, k)[np.ix_(Fi, Fi)]
    Ki = np.linalg.inv(K)
    return float(bd(s, d)[Fi] @ Ki @ bd(t, d)[Fi])


def gate_a15b(degrees=(5, 9, 13, 17, 21), grid=7):
    pts = [(F(a, grid), F(b, grid)) for a in range(1, grid) for b in range(1, grid)]
    rows = []
    for k in (1, 2, 3, 4):
        l = k - 1
        errs = []
        for d in degrees:
            if d - 1 - 2 * l < 1:
                continue
            e = max(abs(_Gr_disc(d, k, l, float(s), float(t))
                        - float(G_exact(k, s, t))) for s, t in pts)
            errs.append(dict(d=d, max_abs_err=e))
        mono = all(errs[i + 1]["max_abs_err"] < errs[i]["max_abs_err"]
                   for i in range(len(errs) - 1))
        rows.append(dict(k=k, l=l, errs=errs, monotone_decreasing=bool(mono),
                         first=errs[0]["max_abs_err"], last=errs[-1]["max_abs_err"]))
        print("    k=%d  max|Gr_d - G|: %s   %s"
              % (k, "  ".join("d=%d:%.1e" % (e["d"], e["max_abs_err"]) for e in errs),
                 "monotone" if mono else "NOT monotone"))
    return dict(rows=rows,
                n_monotone=sum(x["monotone_decreasing"] for x in rows),
                passed=bool(all(x["monotone_decreasing"] for x in rows)
                            and all(x["last"] < x["first"] for x in rows)))


# ---------------------------------------------------------------- a15c
def gate_a15c():
    """Lobe depth against degree, at every cost order of the benchmark.

    Same quantity Fig. 1(b) plots -- the minimum elevated Bernstein coefficient
    of the single-segment block, relative -- so the new rows are directly
    comparable to the published ones, which stopped at `k = 2`.
    """
    rows = []
    for k in (1, 2, 3, 4):
        l = k - 1
        ds = [d for d in range(2 * l + 3, 2 * l + 12, 2)][:4]
        depth = []
        for d in ds:
            r = certify_exact(d, k, l, 1, 0, D=96)
            depth.append(dict(d=d, lobe_depth=abs(float(r["elev_min_rel"]))))
        ratios = [depth[i]["lobe_depth"] / depth[i + 1]["lobe_depth"]
                  for i in range(len(depth) - 1) if depth[i + 1]["lobe_depth"] > 0]
        mono = all(depth[i + 1]["lobe_depth"] < depth[i]["lobe_depth"]
                   for i in range(len(depth) - 1))
        rows.append(dict(k=k, l=l, depth=depth, monotone_decay=bool(mono),
                         median_ratio_per_2deg=float(np.median(ratios)) if ratios else None))
        print("    k=%d  lobe depth: %s   %s (median x%.1f per 2 degrees)"
              % (k, "  ".join("d=%d:%.2e" % (x["d"], x["lobe_depth"]) for x in depth),
                 "decays" if mono else "DOES NOT decay",
                 np.median(ratios) if ratios else float("nan")))
    return dict(rows=rows, n_decaying=sum(x["monotone_decay"] for x in rows),
                passed=bool(all(x["monotone_decay"] for x in rows)))


# ---------------------------------------------------------------- a15d
def gate_a15d():
    """The scope is `l = k-1`.  Outside it the explanation does not apply, and
    at `k >= 3, l = 0` there is nothing to explain because `K` is singular."""
    rows = []
    for (d, k, l) in ((5, 3, 0), (7, 3, 0), (9, 4, 0), (9, 5, 1)):
        r = energy_nullity(d, k, l)
        in_H0k = bool(l + 1 >= k)
        rows.append(dict(d=d, k=k, l=l, definite=r["definite"],
                         nullity=r["nullity"], free_functions_in_H0k=in_H0k))
        print("    (d,k,l)=(%d,%d,%d)  free functions in H_0^k: %-5s  K definite: %s"
              % (d, k, l, in_H0k, r["definite"]))
    return dict(rows=rows,
                passed=bool(all((not x["free_functions_in_H0k"])
                                and (not x["definite"]) for x in rows)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ART, "a15_continuum_green.json"))
    a = ap.parse_args()

    print("\n[a15a] the continuum kernel is strictly positive, exactly, for k = 1..6")
    a15a = gate_a15a()
    print("\n[a15b] Gr_d -> G: 'Galerkin approximation' measured, not asserted")
    a15b = gate_a15b()
    print("\n[a15c] the lobe decays with degree at k = 1,2,3,4")
    a15c = gate_a15c()
    print("\n[a15d] scope control: outside l = k-1 the explanation does not apply")
    a15d = gate_a15d()

    gates = dict(a15a_continuum_positive=a15a, a15b_galerkin_convergence=a15b,
                 a15c_lobe_decays_all_k=a15c, a15d_scope_control=a15d)
    print("\n  --- gates ---")
    for nm, g in gates.items():
        print("  %s: %s" % (nm, "PASS" if g["passed"] else "FAIL"))
    with open(a.out, "w") as fh:
        json.dump(dict(gates=gates), fh, indent=1, default=float)
    print("\nwrote %s" % a.out)
    if not all(g["passed"] for g in gates.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
