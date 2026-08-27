"""Pin the conditions the paper's theorems depend on.

Three of these guard a claim a theorem makes; one guards the LIMIT -- that the
definiteness condition is not vacuous.  Without the last, `a14a` would pass on a
population that simply never violates it, and the proposition would be untested
rather than true.
"""
import math
import os
import sys

from fractions import Fraction as F

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from exact_green import (energy_nullity, gr12_polynomial,        # noqa: E402
                         poly_divmod, q_gram_deriv, solve as qsolve)

CONFIGS = ((3, 1, 0), (5, 1, 0), (7, 1, 0), (9, 1, 0),
           (5, 2, 1), (7, 2, 1), (7, 3, 2))
TABLE1 = [("velocity", 1, 1, 0, 12, 0), ("acceleration", 2, 3, 2, 6, 2),
          ("jerk", 3, 5, 3, 4, 3), ("snap", 4, 7, 4, 3, 4)]


# ------------------------------------------------- K > 0 (the P0 bug)
@pytest.mark.parametrize("N", (1, 2, 3, 4, 5, 6))
@pytest.mark.parametrize("d,k,l", CONFIGS)
def test_swept_configurations_are_definite(N, d, k, l):
    """Every configuration the paper sweeps must have an invertible energy,
    or `K^{-1}` -- and with it `Gr` and `B^{ij}` -- does not exist."""
    eta = max(0, k - 1) if N > 1 else 0
    r = energy_nullity(d, k, l, N, eta)
    assert r["definite"], (N, d, k, l, eta)
    assert r["nullity"] == r["predicted"] == 0


@pytest.mark.parametrize("name,k,d,l,N,eta", TABLE1)
def test_benchmark_rows_are_definite(name, k, d, l, N, eta):
    assert energy_nullity(d, k, l, N, eta)["definite"]


@pytest.mark.parametrize("d,k,l,N,eta,why", [
    (5, 3, 0, 1, 0, "k=3 > 2l+2=2: s(1-s) is in the kernel"),
    (7, 3, 0, 1, 0, "same, at higher degree"),
    (9, 5, 1, 1, 0, "k=5 > 2l+2=4"),
    (5, 3, 2, 3, 0, "multisegment: the middle piece can be a free parabola"),
])
def test_the_condition_is_not_vacuous(d, k, l, N, eta, why):
    """THE LIMIT.  If these ever became definite, `k <= 2l+2` would be
    describing nothing and Theorem 4's dropped "for every k" would have been
    dropped for no reason."""
    r = energy_nullity(d, k, l, N, eta)
    assert not r["definite"], why
    assert r["nullity"] == r["predicted"] > 0


@pytest.mark.parametrize("k,l", [(1, 0), (2, 0), (2, 1), (3, 1), (4, 1),
                                 (3, 2), (5, 2), (6, 2)])
def test_nullity_formula_matches_the_rank(k, l):
    """`k <= 2l+2` and the nullity count are one statement; check they agree
    on both sides of the boundary rather than only where the paper works."""
    d = max(2 * k + 1, 2 * l + 3)
    r = energy_nullity(d, k, l)
    assert r["nullity"] == max(0, k - 2 * (l + 1))
    assert r["definite"] == (k <= 2 * l + 2)


# ------------------------------------------------- a_0 in closed form
@pytest.mark.parametrize("d", (3, 5, 7, 9, 11, 13))
def test_a0_minimal_polynomial_divides_gr12(d):
    """`Gr_12(sigma, 1-sigma)` has rational coefficients, so the nodal point is
    an algebraic number we can name instead of bisecting for."""
    l = (d - 3) // 2
    P, _, _ = gr12_polynomial(d, 1, l)
    mp = [F(d - 2), F(-2 * (2 * d - 1)), F(2 * (2 * d - 1))]
    _, rem = poly_divmod(P, mp)
    assert rem == [F(0)]


def test_a0_agrees_with_the_earlier_bisection():
    """An independent check of the whole `Gr` pipeline: the two routes share no
    code, so agreement to the bisection's own accuracy is evidence for both."""
    a0 = (1 - math.sqrt(3 / 5)) / 2
    assert abs(a0 - 0.1127016528695822) < 3e-8


def test_K_is_the_matrix_the_closed_form_assumes():
    """The closed form is derived from `K_11 = K_22 = 6/5`, `K_12 = 3/10`; if
    the Gram convention ever changed, the derivation would be stale while the
    divisibility test above still passed on the new polynomial."""
    _, K, Ki = gr12_polynomial(3, 1, 0)
    assert K == [[F(6, 5), F(3, 10)], [F(3, 10), F(6, 5)]]
    assert Ki[0][1] / Ki[0][0] == F(-1, 4)


# ------------------------------------------------- the cone, both ways
def test_moment_cone_membership_is_exhibited_not_just_refuted():
    """Theorem 10 claims membership is decidable; a paper that only ever shows
    the separating functional has shown one direction."""
    d, k, l = 3, 1, 0
    Fi = list(range(l + 1, d - l))
    from math import comb
    G = q_gram_deriv(d, k)
    K = [[G[i][j] for j in Fi] for i in Fi]

    def u(s):
        return [F(comb(d, i)) * s ** i * (1 - s) ** (d - i) for i in Fi]

    ss = [F(1, 10), F(1, 2), F(9, 10)]
    U = [u(s) for s in ss]
    A = [[U[a][0] ** 2 for a in range(3)],
         [U[a][0] * U[a][1] for a in range(3)],
         [U[a][1] ** 2 for a in range(3)]]
    w = [r[0] for r in qsolve(A, [[K[0][0]], [K[0][1]], [K[1][1]]])]
    assert all(x > 0 for x in w)
    rebuilt = [[sum(w[a] * U[a][i] * U[a][j] for a in range(3))
                for j in range(2)] for i in range(2)]
    assert rebuilt == K


# ------------------- which array the hypothesis is about
@pytest.mark.parametrize("d,k,l,N,eta", [(3, 1, 0, 2, 0), (5, 1, 0, 3, 0),
                                         (7, 2, 1, 2, 1), (7, 3, 2, 4, 2)])
def test_elevation_preserves_the_boundary_data(d, k, l, N, eta):
    """Theorem 12 states its nonnegativity hypothesis on the ELEVATED array and
    not the raw block, because the raw block is negative on most configurations.
    That is only free if the OTHER three hypotheses mean the same thing on both
    arrays -- they read row 0, row d and the corners, and elevation must leave
    those alone.  If this failed, stating the first hypothesis on the elevated
    array would silently change the other three."""
    from math import comb
    from exact_green import coefficient_blocks
    D = 24
    E = [[F(comb(d, a) * comb(D - d, A - a), comb(D, A))
          if 0 <= A - a <= D - d else F(0) for a in range(d + 1)]
         for A in range(D + 1)]
    blocks, _ = coefficient_blocks(d, k, l, N, eta)
    for B in blocks.values():
        EB = [[sum(E[A][a] * B[a][b] for a in range(d + 1))
               for b in range(d + 1)] for A in range(D + 1)]
        T = [[sum(EB[A][b] * E[C][b] for b in range(d + 1))
              for C in range(D + 1)] for A in range(D + 1)]
        assert all(x == 0 for x in B[0]) == all(x == 0 for x in T[0])
        assert all(x == 0 for x in B[d]) == all(x == 0 for x in T[D])
        for (a, b), (A, C) in (((0, 0), (0, 0)), ((0, d), (0, D)),
                               ((d, 0), (D, 0)), ((d, d), (D, D))):
            assert B[a][b] == T[A][C]


def test_the_raw_block_really_is_negative_somewhere():
    """THE LIMIT on the fix above.  If the raw block were nonnegative anyway,
    stating the hypothesis on the elevated array would be a distinction without
    a difference and the theorem could go back to the simpler form."""
    from exact_green import certify_exact
    neg = [(d, k, N) for N in (2, 3)
           for (d, k, l) in ((3, 1, 0), (5, 1, 0), (7, 1, 0))
           if certify_exact(d, k, l, N, max(0, k - 1), D=96)["raw_min"] < 0]
    assert neg, "raw blocks are all nonnegative -- the fix is unnecessary"


def test_the_nodal_point_reduces_to_one_gram_ratio():
    """Proposition 2 is not a range check over `d`: symmetry collapses the nodal
    condition to `A = q/(2(p+q))` with `A = sigma(1-sigma)`, so uniqueness in
    `(0,1/2)` is free and the whole dependence on `d` is one Gram ratio.
    Checking that ratio checks the proposition, and reaches much further than a
    divisibility sweep."""
    from exact_green import q_gram_deriv
    for d in range(3, 42, 2):
        l = (d - 3) // 2
        Fi = [l + 1, l + 2]
        G = q_gram_deriv(d, 1)
        p, q = G[Fi[0]][Fi[0]], G[Fi[0]][Fi[1]]
        assert G[Fi[1]][Fi[1]] == p, d          # K is symmetric-persymmetric
        assert q / p == F(d - 2, d + 1), d
        # and the ratio is exactly what puts A at the claimed value
        assert q / (2 * (p + q)) == F(d - 2, 2 * (2 * d - 1)), d
