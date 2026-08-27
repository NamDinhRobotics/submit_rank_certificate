"""Gates 1b and 1c, plus a backend cross-check.

Gate 1b  obstacle-free min-energy from (-2,0) to (2,0) has cost exactly 16,
         equispaced control points, rho = 0.
Gate 1c  Theorem 3: the *lifted* curve is collision-free for every instance;
         the *projected* curve may violate the constraint precisely when rho>=1.
"""
import numpy as np
import pytest

from relaxation import Segment

START, GOAL = [-2.0, 0.0], [2.0, 0.0]
D, K, L = 3, 1, 0

# name -> (obstacles, expected rho)
CASES = {
    "symmetric":  ([(np.array([0.0, 0.0]), 0.8)], 1),
    "offset":     ([(np.array([0.0, 0.62]), 0.8)], 0),
    "two_obs":    ([(np.array([-0.8, 0.35]), 0.5),
                    (np.array([0.8, -0.35]), 0.5)], None),
    "corridor":   ([(np.array([0.0, 1.05]), 0.8),
                    (np.array([0.0, -1.05]), 0.8)], None),
}


def make(obs):
    return Segment(n=2, d=D, k=K, l=L, obstacles=obs, bc0=[START], bc1=[GOAL])


# ----------------------------------------------------------------- Gate 1b
def test_gate1b_obstacle_free():
    seg = make([])
    res = seg.solve()
    assert res["converged"], res["status"]
    assert abs(res["cost"] - 16.0) < 1e-8, f"cost={res['cost']!r}"
    assert res["rho"] == 0, f"rho={res['rho']}"
    expect = np.array([[-2.0, -2.0 / 3.0, 2.0 / 3.0, 2.0],
                       [0.0, 0.0, 0.0, 0.0]])
    assert np.allclose(res["Gamma"], expect, atol=1e-6), res["Gamma"]


# ----------------------------------------------------------------- Gate 1c
@pytest.mark.parametrize("name", list(CASES))
def test_gate1c_theorem3_lifted_is_feasible(name):
    """The lifted curve must be collision-free to 1e-8, always."""
    obs, _ = CASES[name]
    seg = make(obs)
    res = seg.solve()
    assert res["converged"], f"{name}: {res['status']}"
    lifted = seg.lifted_clearance(res)
    assert lifted >= -1e-8, f"{name}: lifted clearance {lifted:.3e} < -1e-8"


@pytest.mark.parametrize("name", list(CASES))
def test_gate1c_projection_violates_iff_rho_positive(name):
    """Projected curve is feasible when rho==0 and infeasible when rho>=1."""
    obs, expect_rho = CASES[name]
    seg = make(obs)
    res = seg.solve()
    assert res["converged"], f"{name}: {res['status']}"
    if expect_rho is not None:
        assert res["rho"] == expect_rho, f"{name}: rho={res['rho']} != {expect_rho}"

    proj = seg.projected_violation(res)          # min_s,j |g-c|^2 - r^2
    if res["rho"] == 0:
        assert proj >= -1e-7, f"{name}: rho=0 but projected violation {proj:.3e}"
    else:
        assert proj < -1e-7, f"{name}: rho={res['rho']} but projection is feasible"


def test_gate1c_symmetric_projected_clearance_is_minus_r():
    """The rho=1 symmetric case collapses onto the obstacle centre line:
    the projected curve passes straight through, clearance exactly -r."""
    seg = make(CASES["symmetric"][0])
    res = seg.solve()
    assert res["rho"] == 1
    assert abs(seg.min_clearance(res["Gamma"]) + 0.8) < 1e-6


# --------------------------------------------------- backend cross-check
@pytest.mark.parametrize("name", list(CASES))
def test_backends_agree(name):
    """Clarabel and the barrier method must find the same optimal value.

    The barrier solver is slow and occasionally fails to centre; when it does
    we skip rather than weaken the comparison.
    """
    obs, _ = CASES[name]
    a = make(obs).solve(backend="cvxpy")
    b = make(obs).solve(backend="barrier")
    assert a["converged"], a["status"]
    if not b["converged"]:
        pytest.skip(f"{name}: barrier backend did not converge "
                    f"({b['n_centering_failures']} centering failures)")
    rel = abs(a["cost"] - b["cost"]) / max(1.0, abs(a["cost"]))
    assert rel < 1e-7, f"{name}: Clarabel {a['cost']!r} vs barrier {b['cost']!r}"
    assert a["rho"] == b["rho"], f"{name}: rho {a['rho']} vs {b['rho']}"


# ----------------------------------------------------- Theorem 2 tripwire
def test_relaxation_lower_bounds_ground_truth():
    """c*_SDP <= c*_P on the named cases (Thm 2). Any violation is a bug."""
    from groundtruth import GroundTruth
    for name, (obs, _) in CASES.items():
        seg = make(obs)
        res = seg.solve()
        gt = GroundTruth(2, D, K, L, obs, [START], [GOAL])
        g = gt.solve(ntry=60, seed=0)
        assert g["ok"], f"{name}: ground truth found nothing"
        assert res["cost"] <= g["cost"] + 1e-6, \
            f"{name}: c*_SDP={res['cost']!r} > c*_P={g['cost']!r}"
        if res["rho"] == 0:
            assert abs(res["cost"] - g["cost"]) < 1e-5, \
                f"{name}: rho=0 but SDP {res['cost']!r} != P {g['cost']!r}"
