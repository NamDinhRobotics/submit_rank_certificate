"""The continuum kernel behind the negative lobe.

The paper explains the lobe by identifying `Gr` as the Galerkin approximation of
a Green's function that is positive.  Two parts of that sentence need backing,
and these tests supply it:

  - positivity has to hold at every `k`, not only at `k <= 2` where it is
    elementary, or the explanation covers half the benchmark;
  - "Galerkin approximation" is a claim about what `Gr` IS, so it is measured.

The scope test at the bottom matters most: the argument holds at `l = k-1`, and
if it silently applied everywhere it would be proving something false.
"""
import os
import sys
from fractions import Fraction as F

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

from a15_continuum_green import G_exact, green_pieces, _Gr_disc   # noqa: E402
from exact_green import energy_nullity                            # noqa: E402


# ---------------------------------------------- the kernel is what it claims
@pytest.mark.parametrize("k", (1, 2, 3, 4, 5, 6))
def test_continuum_kernel_strictly_positive(k):
    """`(-1)^k G > 0` on the open square, exactly.  This is the sign the
    classical disconjugacy theorem predicts for the `(k,k)` conjugate problem;
    computing it independently means the paper is not leaning on the citation
    alone."""
    worst = None
    for a in range(1, 10):
        t = F(a, 10)
        for b in range(1, 10):
            v = G_exact(k, F(b, 10), t)
            if worst is None or v < worst:
                worst = v
    assert worst > 0, (k, float(worst))


@pytest.mark.parametrize("k", (1, 2, 3))
def test_kernel_satisfies_its_own_boundary_conditions(k):
    """A sign check on a wrongly-built kernel would be meaningless, so verify
    the construction: `G(.,t)` must vanish to order `k` at both ends."""
    from math import factorial
    t = F(1, 3)
    L, R = green_pieces(k, t)

    def deriv_at(coef, j, s):
        return sum(coef[i] * F(factorial(i), factorial(i - j))
                   * (s ** (i - j) if i - j else F(1))
                   for i in range(j, len(coef)))

    for j in range(k):
        assert deriv_at(L, j, F(0)) == 0, ("left", k, j)
        assert deriv_at(R, j, F(1)) == 0, ("right", k, j)


def test_kernel_is_symmetric():
    """`G` is the Green's function of a self-adjoint problem, so `G(s,t) =
    G(t,s)`.  The construction does not impose this, which makes it a real
    check on the linear system rather than a restatement of it."""
    for k in (1, 2, 3):
        for (a, b) in ((2, 7), (3, 8), (1, 9)):
            s, t = F(a, 10), F(b, 10)
            assert G_exact(k, s, t) == G_exact(k, t, s), (k, a, b)


# ---------------------------------------------- "Galerkin" earns its place
@pytest.mark.parametrize("k,degrees", [(1, (9, 21)), (2, (9, 21)),
                                       (3, (9, 21)), (4, (9, 21))])
def test_discrete_kernel_approaches_the_continuum_one(k, degrees):
    """The word 'Galerkin' asserts `Gr_d -> G`.  Measure it."""
    l = k - 1
    pts = [(F(a, 7), F(b, 7)) for a in range(1, 7) for b in range(1, 7)]
    errs = []
    for d in degrees:
        errs.append(max(abs(_Gr_disc(d, k, l, float(s), float(t))
                            - float(G_exact(k, s, t))) for s, t in pts))
    assert errs[-1] < errs[0], (k, errs)


# ---------------------------------------------- THE SCOPE
@pytest.mark.parametrize("d,k,l", [(5, 3, 0), (7, 3, 0), (9, 4, 0), (9, 5, 1)])
def test_the_explanation_is_scoped_to_l_equals_k_minus_1(d, k, l):
    """Outside `l = k-1` the free functions are not in `H_0^k`, so the continuum
    problem is not the clamped one and the positivity argument does not apply.
    At these settings `K` is singular anyway, so there is nothing to explain --
    but if that ever changed, the paper would be citing a theorem about a
    different boundary value problem."""
    assert l + 1 < k
    assert not energy_nullity(d, k, l)["definite"]


# ---------------------------------------------- what rho is worth
def test_rho_is_not_a_looseness_score():
    """The paper's caveat about the maximum-rank reading has to be exercised by
    an instance, or it is a logical possibility with nothing behind it.

    Pinned from the committed bracket rather than recomputed: the bracket needs
    a conic solve plus a constrained local solve per instance, which does not
    belong in a unit test.  What the test guards is that the artifact still says
    the three things the paper quotes.
    """
    import json
    art = os.path.join(os.path.dirname(__file__), "..", "artifacts",
                       "a16_rho_vs_looseness.json")
    g = json.load(open(art))["gates"]
    z, o, w = (g["a16a_rho0_is_exact"], g["a16b_rho1_usually_loose"],
               g["a16c_rho1_not_always_loose"])
    assert z["n"] + o["n"] == 40
    assert z["worst_abs_rel_gap"] < 1e-3, "rho = 0 must mean the value is exact"
    assert o["median_rel_gap"] > 1e-2, "rho = 1 must usually be genuinely loose"
    # THE LIMIT: and yet one instance has rho = 1 with the value exact
    assert w["witness"]["rel_gap"] < 1e-3
    assert w["witness"]["proj_clearance"] < 0, \
        "the witness must be one whose projection is infeasible"


def test_the_lift_figure_shows_a_real_clearance():
    """The 3-D figure claims the curve clears spheres of radius 0.45.  If the
    excursion on the borrowed axis were smaller than the radius, the picture
    would be drawing something the numbers do not support."""
    import json
    art = os.path.join(os.path.dirname(__file__), "..", "artifacts",
                       "fig_liftpath.json")
    d = json.load(open(art))
    assert d["rho"] == 1
    assert d["projection_min_clearance"] < 0        # the shadow cuts through
    assert d["lift_min_clearance"] > 0              # the lift does not
    assert d["max_borrowed"] > d["sphere_radius"]
    assert d["projection_max_abs_y"] == 0.0         # exactly planar, as captioned
