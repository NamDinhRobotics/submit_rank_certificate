"""Multi-segment relaxation: N segments with C^eta continuity.

The load-bearing check is that the generalised facial reduction is right.
Every linear constraint Gamma A = B (boundary AND continuity) forces
S A = 0 for S = X - Gamma^T Gamma, so with N segments there are
2(l+1) + (N-1)(eta+1) constraint columns and the un-reduced LMI is very far
from strictly feasible.  If the parameterisation were wrong we would see it as
solver failures, broken continuity, or a Theorem-2 violation.
"""
import numpy as np
import pytest

from multisegment import MultiSegment, MultiGroundTruth
from relaxation import Segment

START, GOAL = [[-2.0, 0.0]], [[2.0, 0.0]]
OBS = {
    "none": [],
    "symmetric": [(np.array([0.0, 0.0]), 0.8)],
    "offset": [(np.array([0.0, 0.62]), 0.8)],
    "two": [(np.array([-0.8, 0.35]), 0.5), (np.array([0.8, -0.35]), 0.5)],
}


def make(N, obs, eta=2, d=3):
    return MultiSegment(n=2, d=d, k=1, l=0, N=N, obstacles=OBS[obs],
                        bc0=START, bc1=GOAL, eta=eta)


# --------------------------------------------------- parameterisation
@pytest.mark.parametrize("N", [1, 2, 3, 4])
@pytest.mark.parametrize("eta", [1, 2])
def test_parameterisation_satisfies_constraints_identically(N, eta):
    """Any Y at all must give a Gamma meeting the boundary and continuity
    conditions -- that is what makes the SDP strictly feasible."""
    ms = make(N, "symmetric", eta=eta)
    rng = np.random.default_rng(N * 10 + eta)
    for _ in range(20):
        Y = rng.standard_normal((2, ms.r)) * 3.0
        G = ms.full_Gamma(Y)
        assert np.allclose(G @ ms.A, ms.B, atol=1e-9), "constraints violated"
        assert ms.continuity_residual(G) < 1e-9


@pytest.mark.parametrize("N", [1, 2, 3])
def test_free_dimension_matches_the_count(N):
    """r = N(d+1) - rank(A), and rank(A) = 2(l+1) + (N-1)(eta+1) when the
    constraints are independent."""
    for eta in (1, 2):
        ms = make(N, "none", eta=eta)
        expect = N * (ms.d + 1) - (2 * (ms.l + 1) + (N - 1) * (eta + 1))
        assert ms.r == expect, f"N={N} eta={eta}: r={ms.r} != {expect}"


# ------------------------------------------------------- consistency
@pytest.mark.parametrize("obs", list(OBS))
def test_N1_reduces_to_the_single_segment_code(obs):
    ms = make(1, obs)
    sg = Segment(n=2, d=3, k=1, l=0, obstacles=OBS[obs],
                 bc0=START, bc1=GOAL)
    a, b = ms.solve(), sg.solve()
    assert a["converged"], a["status"]
    assert b["converged"], b["status"]
    assert abs(a["cost"] - b["cost"]) < 1e-7, f"{a['cost']!r} vs {b['cost']!r}"
    assert a["rho"] == b["rho"]
    assert ms.r == sg.f


@pytest.mark.parametrize("N", [1, 2, 3, 4, 6])
def test_obstacle_free_cost_is_invariant_in_N(N):
    """With the time normalisation the obstacle-free optimum is 16 for every
    N; without it the cost would read 16/N and more segments would look like a
    free improvement."""
    ms = make(N, "none")
    r = ms.solve()
    assert r["converged"], r["status"]
    assert abs(r["cost"] - 16.0) < 1e-6, r["cost"]
    assert r["rho"] == 0
    assert ms.continuity_residual(r["Gamma"]) < 1e-9


# ---------------------------------------------------------- theorems
@pytest.mark.parametrize("N", [1, 2, 3])
@pytest.mark.parametrize("obs", ["symmetric", "offset", "two"])
def test_theorem3_lifted_curve_is_collision_free(N, obs):
    ms = make(N, obs)
    r = ms.solve()
    assert r["converged"], r["status"]
    assert ms.lifted_clearance(r) >= -1e-7, ms.lifted_clearance(r)


@pytest.mark.parametrize("N", [1, 2, 3])
@pytest.mark.parametrize("obs", ["symmetric", "two"])
def test_theorem2_relaxation_lower_bounds_ground_truth(N, obs):
    ms = make(N, obs)
    r = ms.solve()
    g = MultiGroundTruth(ms).solve(ntry=40, seed=0)
    assert g["ok"], "ground truth found nothing"
    assert r["cost"] <= g["cost"] + 1e-6, \
        f"N={N} {obs}: c_SDP={r['cost']!r} > c_P={g['cost']!r}"
    if r["rho"] == 0:
        assert abs(r["cost"] - g["cost"]) < 1e-5


@pytest.mark.parametrize("N", [2, 3])
def test_projection_violates_iff_rho_positive(N):
    for obs in ("symmetric", "offset", "two"):
        ms = make(N, obs)
        r = ms.solve()
        assert r["converged"], r["status"]
        proj, _ = ms.exact_clearance(r["Gamma"])
        if r["rho"] == 0:
            assert proj >= -1e-7, f"N={N} {obs}: rho=0 but projection violates"
        else:
            assert proj < -1e-7, f"N={N} {obs}: rho>=1 but projection feasible"
