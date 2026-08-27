"""Pin the link between the nodal point `a0` and the `Z = 0` route.

Section IV used to motivate the search with the `m = 2` remark and then report
cells with `m = 3` and `Z = 0`.  Two different routes, and the agreement read as
a coincidence.  These tests pin the three facts that make it one argument, and
the one that limits it --- the limit matters most, because without it the
section would be claiming to predict a band whose extent it does not predict.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

from bernstein import bd                                        # noqa: E402
from a3_finite_d_certificate import green_matrix                # noqa: E402
from a4_simplicity_margin import three_blocker, analyse         # noqa: E402

A0 = 0.1127016528695822


def _setup():
    K, Ki, F = green_matrix(3, 1, 0)
    return K, Ki, (lambda s: bd(s, 3)[F])


def _svec(M):
    return np.array([M[0, 0], np.sqrt(2.0) * M[0, 1], M[1, 1]])


def test_two_atom_representation_exists_exactly_at_a0():
    """`Z = 0` at two contacts means `U^{-1}KU^{-T}` is diagonal, and since
    `Gr^{-1} = U^{-1}KU^{-T}`, for a 2x2 matrix that is `Gr_12 = 0` --- the
    nodal condition defining `a0`.  So `a0` is not merely where the certificate
    stops working."""
    K, Ki, u = _setup()
    U = np.array([u(A0), u(1 - A0)]).T
    Gr = U.T @ Ki @ U
    W = np.linalg.solve(U, np.linalg.solve(U, K.T).T)
    assert abs(Gr[0, 1]) < 1e-6
    assert abs(W[0, 1]) < 1e-5
    assert W[0, 0] > 0 and W[1, 1] > 0


@pytest.mark.parametrize("sigma", [0.08, 0.10, 0.12, 0.14])
def test_offdiagonal_and_weight_defect_vanish_together(sigma):
    """Away from `a0` both are nonzero, and with opposite signs: they are two
    readings of one quantity, not two coincidences."""
    K, Ki, u = _setup()
    U = np.array([u(sigma), u(1 - sigma)]).T
    Gr = U.T @ Ki @ U
    W = np.linalg.solve(U, np.linalg.solve(U, K.T).T)
    assert abs(Gr[0, 1]) > 1e-3
    assert np.sign(Gr[0, 1]) == -np.sign(W[0, 1])


def test_third_atom_makes_the_cone_full_dimensional():
    """Why the region is open rather than a point: two atoms span a plane in
    the 3-dimensional S^2, three span all of it."""
    _, _, u = _setup()
    two = np.array([_svec(np.outer(u(s), u(s))) for s in (0.08, 0.92)]).T
    three = np.array([_svec(np.outer(u(s), u(s)))
                      for s in (0.08, 0.5, 0.92)]).T
    assert np.linalg.matrix_rank(two) == 2
    assert np.linalg.matrix_rank(three) == 3


def test_cone_membership_is_not_sufficient():
    """THE LIMIT.  A configuration whose three weights are all strictly
    positive --- so `K` is in the cone --- and yet `rho = 1`.  This is why the
    paper says the extent of the band is measured, not predicted, and why the
    a0 framing must stay one-directional."""
    K, _, u = _setup()
    r = analyse(three_blocker(0.12, rmid=0.60), ns=4001)
    assert r.get("status") == "optimal"
    cs = np.array(r["contacts"])
    A = np.array([_svec(np.outer(u(s), u(s))) for s in cs]).T
    w, *_ = np.linalg.lstsq(A, _svec(K), rcond=None)
    assert np.linalg.norm(A @ w - _svec(K)) < 1e-9
    assert (w > 0).all()
    assert r["rho"] == 1


@pytest.mark.parametrize("s1,expected", [(0.08, 2), (0.10, 2), (0.11, 2),
                                         (0.115, 1), (0.12, 1), (0.13, 1)])
def test_transition_across_a0_at_rmid_025(s1, expected):
    """Along a line through the band rather than along its edge, `rho` flips at
    `a0` and the middle contact disappears with it."""
    r = analyse(three_blocker(s1, rmid=0.25), ns=4001)
    assert r.get("status") == "optimal"
    assert r["rho"] == expected
    assert r["m"] == (3 if s1 <= A0 else 2)
