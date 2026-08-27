"""The Green's-function certificate for `rho <= 1` (Phases A1-A3).

`docs/rho_le_1.md` Layer 3c says: with `Z = K - M_lambda >= 0` the dual PSD
block, `U` the Bernstein values at the contact atoms and `Gr = U^T K^{-1} U`,

    Gr W entrywise positive  =>  top eigenvalue of W^{1/2} Gr W^{1/2} simple
                             =>  dim ker Z = 1  =>  rho <= 1,

for every `k`, at finite `d`.  What is pinned here is the *linear algebra* of
that chain plus the geometry of where it stops -- the population-level numbers
live in `artifacts/a{1,2,3}_*.json` and their experiments.

The one thing these tests must not do is re-assert `rho == 1` and call that a
test of the mechanism: `rho == 1` holds on this population for reasons the
certificate does not exhaust (A3 drove an instance into the corner, where the
certificate does NOT apply and `rho` is 1 regardless).  So the chain is tested
step by step instead, including the step where it fails.
"""
import os

import numpy as np
import pytest

from relaxation import Segment
from bernstein import bd, gram_deriv
from instances import hard_instances


def free_idx(d, l):
    return list(range(l + 1, d - l))


def green(d, k, l):
    F = free_idx(d, l)
    K = gram_deriv(d, k)[np.ix_(F, F)]
    return K, np.linalg.inv(K), F


def gr_norm(s, t, d, Ki, F):
    g_st = float(bd(s, d)[F] @ Ki @ bd(t, d)[F])
    g_ss = float(bd(s, d)[F] @ Ki @ bd(s, d)[F])
    g_tt = float(bd(t, d)[F] @ Ki @ bd(t, d)[F])
    return g_st / np.sqrt(g_ss * g_tt)


CONFIGS = [(3, 1, 0), (5, 1, 0), (7, 1, 0), (5, 2, 1), (7, 2, 1), (7, 3, 2),
           (9, 4, 3)]


# ----------------------------------------------------------------------
# the algebra of the chain (no solver involved)
# ----------------------------------------------------------------------
@pytest.mark.parametrize("d,k,l", CONFIGS)
def test_kernel_is_multiplicity_of_eigenvalue_one(d, k, l):
    """Step 1-2: `dim ker(K - M)` = multiplicity of the top eigenvalue of
    `W^{1/2} Gr W^{1/2}`, and the two spectra agree on their nonzero parts."""
    K, Ki, F = green(d, k, l)
    ss = [0.3, 0.65]
    U = np.array([bd(s, d)[F] for s in ss]).T
    Gr = U.T @ Ki @ U
    # `dim ker Z >= 1` needs the top eigenvalue to be exactly 1, so normalise
    # the weights by it -- picking `w_a = c/Gr_aa` does NOT do that once the
    # off-diagonal is present, which is how the first draft of this test failed.
    w0 = np.array([1.0 / Gr[0, 0], 0.5 / Gr[1, 1]])
    Wh0 = np.diag(np.sqrt(w0))
    w = w0 / float(np.max(np.linalg.eigvalsh(Wh0 @ Gr @ Wh0)))
    M = U @ np.diag(w) @ U.T
    Z = K - M

    Lw, Lv = np.linalg.eigh(K)
    Kmh = Lv @ np.diag(1.0 / np.sqrt(Lw)) @ Lv.T
    mu_pencil = np.sort(np.linalg.eigvalsh(Kmh @ M @ Kmh))[::-1]
    Wh = np.diag(np.sqrt(w))
    mu_gram = np.sort(np.linalg.eigvalsh(Wh @ Gr @ Wh))[::-1]
    assert np.allclose(mu_pencil[:2], mu_gram, atol=1e-9)

    wZ = np.linalg.eigvalsh(Z)
    ker = int(np.sum(np.abs(wZ) < 1e-9 * max(1.0, np.abs(wZ).max())))
    mult = int(np.sum(np.abs(mu_gram - mu_gram[0]) < 1e-9))
    assert ker == mult == 1


@pytest.mark.parametrize("d,k,l", CONFIGS)
def test_dual_feasibility_caps_the_spectrum_at_one(d, k, l):
    """Step 3: `Z >= 0` is exactly "every eigenvalue of `W^{1/2}GrW^{1/2}` <= 1",
    so an attained 1 is the TOP eigenvalue and Perron-Frobenius is about the
    right end of the spectrum."""
    K, Ki, F = green(d, k, l)
    ss = [0.25, 0.7]
    U = np.array([bd(s, d)[F] for s in ss]).T
    Gr = U.T @ Ki @ U
    for scale in (0.5, 1.0, 1.3):
        w = scale * np.array([1.0 / Gr[0, 0], 0.8 / Gr[1, 1]])
        M = U @ np.diag(w) @ U.T
        Wh = np.diag(np.sqrt(w))
        mu = float(np.max(np.linalg.eigvalsh(Wh @ Gr @ Wh)))
        z_psd = float(np.min(np.linalg.eigvalsh(K - M))) > -1e-9
        assert z_psd == (mu <= 1.0 + 1e-9)


# ----------------------------------------------------------------------
# the geometry: where the certificate holds, and where it stops
# ----------------------------------------------------------------------
@pytest.mark.parametrize("d,k,l", CONFIGS)
def test_green_matrix_positive_away_from_the_corner(d, k, l):
    """Interior pairs that do not straddle both ends have `Gr > 0`, which is
    what makes the certificate apply on essentially the whole population."""
    _, Ki, F = green(d, k, l)
    for s, t in ((0.3, 0.6), (0.25, 0.75), (0.4, 0.5), (0.05, 0.5),
                 (0.2, 0.45)):
        assert gr_norm(s, t, d, Ki, F) > 0.0


def test_green_matrix_goes_negative_in_the_corner():
    """...and it is NOT positive everywhere: opposite corners break it.

    This is the hypothesis of Layer 3c failing, and it must stay visible -- the
    certificate is sufficient, not necessary, and a test suite that only ever
    exercised the good region would quietly promote it to a theorem.
    """
    _, Ki, F = green(3, 1, 0)
    assert gr_norm(0.05, 0.95, 3, Ki, F) < 0.0
    assert gr_norm(0.10, 0.90, 3, Ki, F) < 0.0
    assert gr_norm(0.02, 0.50, 3, Ki, F) > 0.0      # ONE point near an end is fine


@pytest.mark.parametrize("d,k,l", CONFIGS)
def test_degenerate_dual_is_constructible_on_the_nodal_set(d, k, l):
    """On the nodal set the failure mode is real: an honest `Z >= 0` with
    `dim ker Z = 2` exists at finite `d`.  (Whether any INSTANCE realises it is
    a different question -- A3 tried and failed, see a3g.)"""
    _, Ki, F = green(d, k, l)
    lo, hi = 1e-9, 0.5
    if not (gr_norm(lo, 1 - lo, d, Ki, F) < 0 < gr_norm(hi, 1 - hi, d, Ki, F)):
        pytest.skip("no sign change on the diagonal for this configuration")
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if gr_norm(mid, 1 - mid, d, Ki, F) <= 0:
            lo = mid
        else:
            hi = mid
    a0 = 0.5 * (lo + hi)

    K, Ki, F = green(d, k, l)
    U = np.array([bd(a0, d)[F], bd(1 - a0, d)[F]]).T
    Gr = U.T @ Ki @ U
    assert abs(Gr[0, 1]) < 1e-12 * abs(Gr[0, 0])
    w = np.array([1.0 / Gr[0, 0], 1.0 / Gr[1, 1]])
    Z = K - U @ np.diag(w) @ U.T
    wZ = np.sort(np.linalg.eigvalsh(0.5 * (Z + Z.T)))
    scale = max(1.0, float(np.abs(wZ).max()))
    # `Z = K - U diag(w) U^T` cancels two `||K||`-scale terms down to `O(1)`,
    # so its eigenvalues carry an ABSOLUTE error of order `||K|| * eps`.  With
    # `||K||` reaching `1.7e7` at `d=9,k=4` that floor is `~4e-9`, above the
    # fixed `-1e-9` this used to demand.  Compare against the larger of the
    # two; for small `d` the cancellation floor is negligible and nothing
    # changes.
    floor = 100.0 * float(np.abs(K).max()) * float(np.finfo(float).eps)
    assert wZ[0] > -max(1e-9 * scale, floor)           # still dual feasible
    assert int(np.sum(np.abs(wZ) < 1e-6 * scale)) == 2  # ...with a 2-dim kernel


# ----------------------------------------------------------------------
# the identity that ties the dual to the contacts, on a real solve
# ----------------------------------------------------------------------
def test_natural_rho_2_exists_and_the_certificate_refuses_it():
    """A NATURAL `rho = 2` (Phase A4) -- not injected, not synthetic.

    Three blockers, two of them inside the corner where the discrete Green
    matrix goes negative, one in the middle: `m = 3`, and the top eigenvalue of
    `W^{1/2} Gr W^{1/2}` is degenerate on an OPEN region.  This is the instance
    that refutes "rho > 1 is unreachable" (Phase 8's conclusion, reached by a
    search that did not know where to look).

    Two things are pinned, and the second matters more than the first:
      * `rho = 2` really is attained (`f = 2`, so their Lemma 4 is TIGHT here);
      * the certificate REFUSES it -- margin `~ 0`.  A certificate that cleared
        this instance would be unsound, and that is the failure this test is
        really guarding against.
    """
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                     "experiments"))
    from a4_simplicity_margin import three_blocker, analyse

    r = analyse(three_blocker(0.10, rmid=0.35), ns=8001)
    assert r["status"] == "optimal"
    assert r["m"] == 3
    assert r["rho"] == 2                    # the counterexample itself
    assert r["rho"] == r["f"]               # ...and it attains Lemma 4
    assert r["margin"] < 1e-6               # the certificate refuses: SOUND
    assert not r["pf_applies"]              # Perron-Frobenius inapplicable too


def test_rho_2_region_is_open_not_a_knife_edge():
    """It is not fine tuning: `rho = 2` survives a range of middle radii, and
    it turns off outside the nodal region `s1 < a0`."""
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                     "experiments"))
    from a4_simplicity_margin import three_blocker, analyse

    for rmid in (0.20, 0.35, 0.50):
        r = analyse(three_blocker(0.08, rmid=rmid), ns=4001)
        assert r["status"] == "optimal" and r["rho"] == 2, rmid
    # outside the nodal region the same family is back to rho = 1
    r = analyse(three_blocker(0.18, rmid=0.35), ns=4001)
    assert r["status"] == "optimal" and r["rho"] == 1


@pytest.mark.parametrize("d,k,l,expect_in_cone", [
    (3, 1, 0, True), (5, 2, 1, True), (7, 3, 2, True), (9, 4, 3, True),   # f = 2
    (5, 1, 0, False), (7, 1, 0, False), (9, 1, 0, False),                 # f >= 4
    (7, 2, 1, False), (9, 2, 1, False),
])
def test_moment_cone_obstruction(d, k, l, expect_in_cone):
    """`Z = 0` needs `K` to be a moment matrix of a nonnegative measure on
    `[0,1]` (Phase A7), and that holds exactly when `f = 2`.

    This is what kills A6's counting law: at `f >= 4` the obstruction is not
    "not enough contacts", it is that NO measure works, at any `m`.  The
    certificate is exact -- `p_Y >= 0` is decided from the real roots of
    `p_Y'`, not from a grid.
    """
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                     "experiments"))
    from a7_moment_cone import moment_cone_verdict

    v = moment_cone_verdict(d, k, l)
    assert v["in_cone"] == expect_in_cone
    if expect_in_cone:
        assert v["rel_residual"] < 1e-9
        assert v["f"] == 2
    else:
        assert v["f"] >= 4
        assert v["separated"]                       # a certificate exists
        assert v["K_dot_Y"] < -1e-9                 # it separates
        assert v["p_min"] > -1e-9 * v["scale"]      # and it is nonnegative


def test_strict_complementarity_at_the_rho2_optima():
    """The hypothesis A7's AHO reading needs (Phase A8): `rank(P) + rank(Z) =
    n + f`.  Complementarity always gives `<=`; equality is what makes the
    optimal rank locally constant in the data, hence the OPEN regions."""
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                     "experiments"))
    from a4_simplicity_margin import three_blocker
    from a8_nondegeneracy_multiseg import strict_complementarity

    r = strict_complementarity(three_blocker(0.10, rmid=0.35))
    assert r["status"] == "optimal"
    assert r["rho"] == 2
    assert r["strict"]
    assert r["rank_Z"] == r["f"] - r["rho"]     # = 0 here, i.e. Z vanishes


@pytest.mark.parametrize("N,d,k,l,eta,expect_in_cone", [
    (1, 3, 1, 0, 1, True),      # r = 2
    (2, 3, 1, 0, 1, False),     # r = 4
    (2, 3, 1, 0, 0, False),     # r = 5
])
def test_multisegment_moment_cone(N, d, k, l, eta, expect_in_cone):
    """A7's `f = 2` verdict survives MULTISEGMENT with `r` in place of `f`
    (Phase A8) -- which is what makes it say anything about the source paper's
    Table I, where the freedom comes from the joints rather than one segment.

    The certificate is produced by column generation, not by a fixed grid: a
    grid projection is nonnegative only AT the grid points and dips below zero
    between them.
    """
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                     "experiments"))
    from a8_nondegeneracy_multiseg import (multi_cone_verdict, make_ms,
                                           blockers_multi)

    v = multi_cone_verdict(make_ms(N, d, k, l, eta, blockers_multi()))
    assert v["in_cone"] == expect_in_cone
    if expect_in_cone:
        assert v["r"] == 2
    else:
        assert v["r"] > 2
        assert v["separated"]
        assert v["K_dot_Y"] < -1e-9
        assert v["p_min"] > -1e-9 * v["scale"]


@pytest.mark.parametrize("N,d,k,l,eta,expect_certified", [
    (1, 3, 1, 0, 1, False),     # one segment: K^-1 has a NEGATIVE entry
    (2, 3, 1, 0, 1, True),      # joints restore nonnegativity...
    (3, 3, 1, 0, 1, True),
    (2, 5, 2, 1, 2, True),
    (2, 5, 1, 0, 1, False),     # ...but not everywhere (measured >0, uncertified)
])
def test_bernstein_coefficients_of_the_green_matrix(N, d, k, l, eta,
                                                    expect_certified):
    """Joints destroy the negative corner that `rho = 2` needs (Phase A9).

    `Gr((i,s),(j,t)) = b_d(s)^T [Nperp_i K^-1 Nperp_j^T] b_d(t)`, so those block
    entries ARE the tensor-product Bernstein coefficients.  All nonnegative =>
    `Gr >= 0` on the whole parameter square, with no sampling at all -- and then
    Perron-Frobenius gives `rho <= 1`.  The test is SUFFICIENT only, which is
    why two of these configurations are pinned as *not* certified even though
    their `Gr` is positive on a grid: reporting them as proved would be the
    overclaim.
    """
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                     "experiments"))
    from a8_nondegeneracy_multiseg import multi_pieces, make_ms, blockers_multi

    ms = make_ms(N, d, k, l, eta, blockers_multi())
    K, Np = multi_pieces(ms)
    Ki = np.linalg.inv(K)
    lo = min((Np[i] @ Ki @ Np[j].T).min() for i in range(N) for j in range(N))
    assert (lo >= -1e-12) == expect_certified
    if N == 1:
        assert lo < 0            # the corner, and it is where A4's rho = 2 is


@pytest.mark.parametrize("N,d,k,l,eta,expect_certified", [
    (1, 3, 1, 0, 1, False),     # the CONTROL: one segment must stay negative
    (1, 5, 1, 0, 1, False),
    (2, 3, 1, 0, 0, True),      # elevation closes the ones the raw test missed
    (2, 5, 1, 0, 1, True),
    (3, 5, 1, 0, 2, True),
    (2, 7, 2, 1, 2, True),
])
def test_degree_elevation_certifies_every_multisegment_config(N, d, k, l, eta,
                                                              expect_certified):
    """The raw coefficient test is conservative; elevating the degree tightens
    it to the true minimum (Phase A9e), and that certifies `Gr >= 0` on every
    multisegment configuration measured -- while the single segment stays
    negative at every degree tried, which is what stops this being vacuous."""
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                     "experiments"))
    from a8_nondegeneracy_multiseg import multi_pieces, make_ms, blockers_multi
    from a9_multiseg_rho2 import _elevation

    ms = make_ms(N, d, k, l, eta, blockers_multi())
    K, Np = multi_pieces(ms)
    Ki = np.linalg.inv(K)
    blocks = [Np[i] @ Ki @ Np[j].T for i in range(N) for j in range(N)]
    scale = max(1.0, max(float(np.abs(B).max()) for B in blocks))
    E = _elevation(d, 96)
    lo = min(float((E @ B @ E.T).min()) for B in blocks)
    assert (lo >= -1e-12 * scale) == expect_certified
    if not expect_certified:
        assert lo < -1e-4        # firmly negative, not a rounding artefact


@pytest.mark.parametrize("N,d,k,l,eta,expect_certified", [
    (1, 3, 1, 0, 0, False), (1, 9, 1, 0, 0, False),   # one polynomial: always fails
    (1, 7, 3, 2, 0, False),
    (2, 3, 1, 0, 1, True), (6, 3, 1, 0, 1, True),     # any knot: always passes
    (2, 9, 1, 0, 1, True), (3, 7, 3, 2, 2, True),
])
def test_knot_dichotomy(N, d, k, l, eta, expect_certified):
    """The whole `rho = 2` story in one line (Phase A10): a single polynomial's
    discrete Green's function has a negative lobe at every `(d, k, l)`, and one
    knot removes it.

    Note `N = 1, d = 9` has `f = 8` free coefficients and fails while
    `N = 2, d = 3` has `r = 4` and passes -- fewer parameters, positive Green's
    function -- so this is about the KNOT, not about dimension.
    """
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                     "experiments"))
    from a10_knot_dichotomy import certificate

    r = certificate(N, d, k, l, eta)
    assert r["certified"] == expect_certified
    if not expect_certified:
        assert r["min_coef_rel"] < 0


@pytest.mark.parametrize("name,k,d,l,N,eta", [
    ("velocity", 1, 1, 0, 12, 0),
    ("acceleration", 2, 3, 2, 6, 2),
    ("jerk", 3, 5, 3, 4, 3),
    ("snap", 4, 7, 4, 3, 4),
])
def test_source_paper_table1_is_certified(name, k, d, l, N, eta):
    """Every row of the source paper's Table I certifies `rho <= 1` (Phase A11).

    These are read off the paper itself (`paper sdp.pdf` p. 8), not off a
    second-hand note. Each row has single-segment `f <= 0` -- all the freedom
    is in the joints -- which is why the single-segment verdict of A7 says
    nothing about them and this check is the one that speaks to their regime.
    """
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                     "experiments"))
    from a11_table1 import certify

    r = certify(k, d, l, N, eta)
    assert r["certified"]
    assert r["f_single"] <= 0        # the freedom really is in the joints
    assert r["r"] > 2                # so the Z = 0 route is shut as well


def test_certificate_is_a_property_of_the_configuration():
    """The coefficient block does not involve the obstacles or `n` (Phase A11).

    This is what upgrades the Table I check from a spot check to a statement
    about every instance of those rows -- including the 100 random `R^3` and
    `R^5` instances the paper evaluates on.  If this ever stops holding, the
    Table I claim silently narrows, so it is pinned separately.
    """
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                     "experiments"))
    from a11_table1 import build, coef_block

    def flat(obs, n):
        _, blocks = coef_block(build(2, 3, 2, 6, 2, obstacles=obs, n=n))
        return np.concatenate([B.ravel() for B in blocks])

    rng = np.random.default_rng(0)
    ref = flat([(np.zeros(2), 0.4)], 2)
    for obs, n in (([(np.array([0.9, -0.3]), 0.7),
                     (np.array([-1.1, 0.5]), 0.35)], 2),
                   ([(rng.uniform(-2, 2, 3), 0.6) for _ in range(15)], 3),
                   ([(rng.uniform(-2, 2, 5), 0.5) for _ in range(30)], 5)):
        assert np.array_equal(flat(obs, n), ref)


@pytest.mark.parametrize("N,eta,expect_certified", [
    (1, 0, False),          # one cubic
    (2, 0, True), (2, 1, True), (2, 2, True),    # genuine knots
    (2, 3, False), (3, 3, False), (4, 3, False),  # WELDED: C^3 between cubics
])
def test_a_welded_knot_is_not_a_knot(N, eta, expect_certified):
    """The negative control for the whole dichotomy (Phase A10d).

    A degree-`d` spline with `C^d` continuity *is* a single polynomial, so if
    the dichotomy tracked the label `N` rather than the freedom the joint
    leaves, welding the joint shut would still certify. It does not: `eta = 3`
    between cubics fails at `N = 2, 3, 4` alike, each collapsing to `r = 2`.
    The earlier phrasing "any number of knots certifies" was wrong for exactly
    this reason.
    """
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                     "experiments"))
    from a10_knot_dichotomy import certificate

    r = certificate(N, 3, 1, 0, eta if N > 1 else 0)
    assert r["certified"] == expect_certified
    if N > 1 and eta == 3:
        assert r["r"] == 2       # collapsed to the single polynomial's count


def test_fixed_point_identity_on_a_solved_instance():
    """`y_i = (Gr W) y_i`: the lifted components, sampled at the contacts, are
    Perron vectors of the same small matrix.  Checked on the symmetric blocker,
    whose single atom sits at `s = 0.5` and makes the fit exact."""
    from node import Node, build
    from scipy.optimize import nnls

    seg = Segment(**hard_instances()["symmetric_blocker"])
    h = build(seg, Node(lo=None, hi=None), use_rlt=False, use_lift=False)
    h["prob"].solve(solver="CLARABEL", tol_gap_abs=1e-10, tol_gap_rel=1e-10,
                    tol_feas=1e-10, max_iter=2000, verbose=False)
    assert h["prob"].status == "optimal"

    Gf = np.asarray(h["Gfree"].value)
    Xf = 0.5 * (np.asarray(h["Xfree"].value) + np.asarray(h["Xfree"].value).T)
    res = seg._package(Gf, Xf, float(h["prob"].value))
    V = seg.lifted_V(res)
    assert V.shape[0] == 1                              # rho = 1 here

    Zf = 0.5 * (np.asarray(h["cons"][0].dual_value)
                + np.asarray(h["cons"][0].dual_value).T)
    K, Ki, F = green(seg.d, seg.k, seg.l)
    M = K - Zf[seg.n:, seg.n:]

    contacts = [0.5]
    U = np.array([bd(s, seg.d)[F] for s in contacts]).T
    A = np.array([np.outer(U[:, a], U[:, a]).ravel()
                  for a in range(U.shape[1])]).T
    w, rn = nnls(A, M.ravel())
    assert rn / np.linalg.norm(M) < 1e-6                # the atom identification
    assert w[0] > 0

    Gr = U.T @ Ki @ U
    GW = Gr @ np.diag(w)
    y = np.array([float(V[0] @ bd(s, seg.d)) for s in contacts])
    assert np.max(np.abs(GW @ y - y)) < 1e-6 * max(1e-12, np.max(np.abs(y)))
    assert GW.min() > 0                                 # the certificate applies
