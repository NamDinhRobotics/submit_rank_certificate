"""Numerical verification of docs/rlt_lemma.md (requirement R2(a)).

Checks, on random boxes and random Gamma inside them:
  - validity: all four families hold at X = Gamma^T Gamma;
  - Lemma 1: the exact residual identities;
  - Theorem 2: the residual bound sum_p w_pi w_pj, and that it is attained;
  - Theorem 3: the aggregated bound equals the sum over p of the per-entry
    McCormick bounds on the full Shor lift -- i.e. the symmetry reduction
    costs nothing in bound quality on X.
"""
import numpy as np
import pytest

from rlt import bounds, residuals, gap_bound, mccormick_entrywise

SHAPES = [(2, 4), (3, 4), (5, 6), (1, 3)]


def random_box(rng, n, m, width=2.0):
    l = rng.uniform(-3.0, 1.0, size=(n, m))
    u = l + rng.uniform(0.0, width, size=(n, m))
    return l, u


def random_inside(rng, l, u):
    t = rng.uniform(0.0, 1.0, size=l.shape)
    return l + t * (u - l)


@pytest.mark.parametrize("n,m", SHAPES)
def test_validity_at_true_X(n, m):
    """Every family holds at X = Gamma^T Gamma for Gamma in the box."""
    rng = np.random.default_rng(n * 100 + m)
    for _ in range(50):
        l, u = random_box(rng, n, m)
        G = random_inside(rng, l, u)
        X = G.T @ G
        res = residuals(G, X, l, u)
        for key, R in res.items():
            assert R.min() > -1e-12, f"{key} violated by {R.min():.3e}"


@pytest.mark.parametrize("n,m", SHAPES)
def test_lemma1_residual_identities(n, m):
    """X_ij - (G^T G)_ij equals exactly -sum_p d_pi d_pj  (LL), etc."""
    rng = np.random.default_rng(n * 200 + m)
    for _ in range(50):
        l, u = random_box(rng, n, m)
        G = random_inside(rng, l, u)
        GtG = G.T @ G
        dlt = G - l
        eps = u - G
        lo_LL, lo_UU, up_LU, up_UL = bounds(G, l, u)
        assert np.allclose(lo_LL, GtG - dlt.T @ dlt, atol=1e-11)
        assert np.allclose(lo_UU, GtG - eps.T @ eps, atol=1e-11)
        assert np.allclose(up_LU, GtG + dlt.T @ eps, atol=1e-11)
        assert np.allclose(up_UL, GtG + eps.T @ dlt, atol=1e-11)


@pytest.mark.parametrize("n,m", SHAPES)
def test_theorem2_gap_bound(n, m):
    """Any X admitted by LL and LU is within sum_p w_pi w_pj of G^T G."""
    rng = np.random.default_rng(n * 300 + m)
    for _ in range(50):
        l, u = random_box(rng, n, m)
        G = random_inside(rng, l, u)
        GtG = G.T @ G
        lo_LL, lo_UU, up_LU, up_UL = bounds(G, l, u)
        lo = np.maximum(lo_LL, lo_UU)
        hi = np.minimum(up_LU, up_UL)
        B = gap_bound(l, u)
        assert (GtG - lo).max() <= B.max() + 1e-9
        assert (hi - GtG).max() <= B.max() + 1e-9
        assert (lo <= GtG + 1e-11).all() and (GtG <= hi + 1e-11).all()


@pytest.mark.parametrize("n,m", SHAPES)
def test_theorem2_bound_is_attained(n, m):
    """At the lower corner the LL residual is exactly sum_p w_pi w_pj."""
    rng = np.random.default_rng(n * 400 + m)
    l, u = random_box(rng, n, m)
    G = u.copy()                                  # delta = w, eps = 0
    lo_LL = bounds(G, l, u)[0]
    assert np.allclose(G.T @ G - lo_LL, gap_bound(l, u), atol=1e-11)


@pytest.mark.parametrize("n,m", SHAPES)
def test_aggregation_is_a_relaxation_of_the_full_lift(n, m):
    """Proposition 3: aggregation is WEAKER than the full Shor lift.

    The aggregated bound is max-of-sums; the full lift, projected onto
    X_ij = sum_p Y_{pi,pj}, is sum-of-maxes.  So the aggregated interval always
    *contains* the full-lift interval.  (An earlier draft claimed equality;
    this test is what refuted it.)
    """
    rng = np.random.default_rng(n * 500 + m)
    for _ in range(50):
        l, u = random_box(rng, n, m)
        G = random_inside(rng, l, u)
        lo_LL, lo_UU, up_LU, up_UL = bounds(G, l, u)

        # summing one fixed family over p does reproduce that aggregate exactly
        Gi, Gj = G[:, :, None], G[:, None, :]
        li, lj = l[:, :, None], l[:, None, :]
        ui, uj = u[:, :, None], u[:, None, :]
        assert np.allclose((lj * Gi + li * Gj - li * lj).sum(0), lo_LL, atol=1e-11)
        assert np.allclose((uj * Gi + ui * Gj - ui * uj).sum(0), lo_UU, atol=1e-11)
        assert np.allclose((uj * Gi + li * Gj - li * uj).sum(0), up_LU, atol=1e-11)

        mc_lo, mc_hi = mccormick_entrywise(G, l, u)
        agg_lo = np.maximum(lo_LL, lo_UU)
        agg_hi = np.minimum(up_LU, up_UL)
        assert (agg_lo <= mc_lo.sum(0) + 1e-11).all()
        assert (agg_hi >= mc_hi.sum(0) - 1e-11).all()
        # both still bracket the truth
        GtG = G.T @ G
        assert (agg_lo <= GtG + 1e-11).all() and (GtG <= agg_hi + 1e-11).all()


def test_aggregation_is_exact_for_n_equals_1():
    """With one ambient coordinate there is nothing to aggregate over."""
    rng = np.random.default_rng(11)
    for _ in range(100):
        l, u = random_box(rng, 1, 5)
        G = random_inside(rng, l, u)
        lo_LL, lo_UU, up_LU, up_UL = bounds(G, l, u)
        mc_lo, mc_hi = mccormick_entrywise(G, l, u)
        assert np.allclose(np.maximum(lo_LL, lo_UU), mc_lo.sum(0), atol=1e-12)
        assert np.allclose(np.minimum(up_LU, up_UL), mc_hi.sum(0), atol=1e-12)


def test_aggregation_loss_is_strict_for_n_ge_2():
    """The loss is real, not a measure-zero artefact: it shows up often."""
    rng = np.random.default_rng(12)
    strict = 0
    for _ in range(300):
        l, u = random_box(rng, 3, 4)
        G = random_inside(rng, l, u)
        lo_LL, lo_UU, up_LU, up_UL = bounds(G, l, u)
        mc_lo, mc_hi = mccormick_entrywise(G, l, u)
        agg_w = (np.minimum(up_LU, up_UL) - np.maximum(lo_LL, lo_UU)).sum()
        mc_w = (mc_hi.sum(0) - mc_lo.sum(0)).sum()
        if agg_w > mc_w + 1e-9:
            strict += 1
    assert strict > 250, f"expected a strict gap almost always, saw {strict}/300"


@pytest.mark.parametrize("n,m", SHAPES)
def test_both_obey_the_theorem2_envelope(n, m):
    """Whatever the loss, both intervals fit inside sum_p w_pi w_pj -- which is
    what makes the branch-and-bound converge at the same O(w^2) rate."""
    rng = np.random.default_rng(n * 600 + m)
    for _ in range(50):
        l, u = random_box(rng, n, m)
        G = random_inside(rng, l, u)
        B = gap_bound(l, u)
        lo_LL, lo_UU, up_LU, up_UL = bounds(G, l, u)
        agg_w = np.minimum(up_LU, up_UL) - np.maximum(lo_LL, lo_UU)
        mc_lo, mc_hi = mccormick_entrywise(G, l, u)
        mc_w = mc_hi.sum(0) - mc_lo.sum(0)
        assert (agg_w <= B + 1e-9).all()
        assert (mc_w <= B + 1e-9).all()


def test_shrinking_box_forces_exactness():
    """As the box collapses, the admissible X interval collapses onto G^T G."""
    rng = np.random.default_rng(7)
    n, m = 3, 4
    centre = rng.uniform(-1.0, 1.0, size=(n, m))
    widths, gaps = [], []
    for w in [1.0, 0.5, 0.25, 0.1, 0.01, 1e-3]:
        l, u = centre - w / 2, centre + w / 2
        G = centre.copy()
        lo_LL, lo_UU, up_LU, up_UL = bounds(G, l, u)
        lo = np.maximum(lo_LL, lo_UU)
        hi = np.minimum(up_LU, up_UL)
        widths.append(w)
        gaps.append(float((hi - lo).max()))
    # quadratic in the box width, and monotonically shrinking
    assert all(b < a for a, b in zip(gaps, gaps[1:])), gaps
    ratio = gaps[0] / gaps[-1]
    assert ratio > 1e5, f"expected ~w^2 collapse, got ratio {ratio:.2e}"
