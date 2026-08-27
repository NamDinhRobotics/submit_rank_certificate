# `rho <= 1`: a certificate, and where it fails

Minimal reproduction package for the paper. Assembled by
`paper/make_release.py` from the working repository: it is the paper's import
closure, not a curated subset, so what is here is what the claims rest on.

## Layout

    src/           the 10 modules the paper's phases import
    experiments/   the phases, one per section of the paper
    paper/         figure scripts, the audit tool, the manuscript
    tests/         11 test modules, including the ones that pin the FAILURES
    artifacts/     committed JSON for every number, plus console logs

## Reproduce

    conda create -y -n sdpbnb -c conda-forge python=3.12 numpy scipy matplotlib                    pytest sympy
    conda activate sdpbnb
    pip install clarabel cvxpy scs

    pytest tests/ -q                     # the suite
    bash run_all.sh                      # regenerate every artifact from scratch

`run_all.sh` re-solves several thousand SDPs and takes on the order of an hour.

**`paper/` is in the submission package and not in the public mirror.** It holds
the manuscript sources, the figure and video makers, `verify_prop9_proof.py`
(the nodal point of Prop. 23, over Q) and `audit_paper.py`, which checks every
number in the text against its stored artifact in both directions and, under
`--control`, requires every check to FAIL on perturbed data. Those build the
paper rather than the science it reports, so the public mirror carries the
experiments, the artifacts and the tests, and `run_all.sh` skips the paper
steps when the directory is absent. Where `paper/` is present:

    python paper/verify_prop9_proof.py
    python paper/audit_paper.py
    python paper/audit_paper.py --control

## Claim -> script -> gate

| Paper | Claim | Script | Gate |
|---|---|---|---|
| Sec. 3 | `rho <= f`, and `rho <=` #contacts is strictly tighter on 22/59 | `a1_contact_rank.py` | `a1e` |
| Sec. 7.3 | Prop. 23's proof over `Q` for `d = 3..17`, plus the Gram identity and the middle weight symbolically | `verify_prop9_proof.py` | `p9_proof` |
| Sec. 4 | the Perron-Frobenius route | `a2_perron_route.py` | `a2a`-`a2c` |
| Sec. 5 | the certificate is exact at finite `d` | `a3_finite_d_certificate.py` | `a3a`-`a3g` |
| Sec. 4 | `g > 0 <=> rho <= 1`, on 53 named + 60 random instances | `a4_simplicity_margin.py` | `a4a`-`a4d` |
| Sec. 7.3 | the `rho = 2` region, and the certificate refusing all of it | `a4_simplicity_margin.py` | `a4e` |
| Sec. 7.3 | how rank must be read (three readings) | `a5_rho2_scope.py` | `a5a`-`a5d` |
| Sec. 5 | the two mechanisms | `a6_two_mechanisms.py` | `a6a` |
| Sec. 5 | the moment cone shuts the `Z = 0` route (Thm. 12) | `a7_moment_cone.py` | `a7a`-`a7c` |
| Sec. 5 | multisegment: nondegeneracy and the pencil identity | `a8_nondegeneracy_multiseg.py` | `a8a`-`a8c` |
| Sec. 5 | no `rho = 2` with knots, and degree elevation certifies | `a9_multiseg_rho2.py` | `a9a`-`a9e` |
| Sec. 5 | the knot dichotomy, with the welded-knot negative control | `a10_knot_dichotomy.py` | `a10a`-`a10d` |
| Sec. 5 | the source paper's own Table I, certified uniformly (Cor. 15) | `a11_table1.py` | `a11a`-`a11c` |
| Sec. 5 | the same, in EXACT rational arithmetic, plus irreducibility | `a12_exact_bernstein.py` | `a12a`-`a12e` |
| Sec. 3 | the chain (11) attained, `m <= d`, and the sweep's attrition | `a38_chain_tightness.py` | `a38a`-`a38c` |
| Sec. 3 | Hypothesis 2: a strictly feasible point, measured | `a39_slater_margin.py` | `a39a`-`a39b` |
| Sec. 7.3 | the structural half under three replacement costs | `a40_second_lifting.py` | `a40a`-`a40g` |
| Sec. 7.1 | a quadrotor configuration chosen here, certified a priori then censused | `a44_quadrotor_minsnap.py` | `a44a`-`a44d` |
| Sec. 6, Sec. 7.2, Fig. 1 | rank-one fold recovery: realization, margins, cost, the certificate on the same instances, and why the quadrotor census cannot host it | `a46_rank_one_recovery.py`, `paper/make_figure6.py` | `a46a`-`a46g` |
| Sec. 7.4 | the certificate in the loop: disturbed receding-horizon replanning, constraint tightening as a safety theorem, paired naive-vs-certified outcomes | `a48_loop_recovery.py` | `a48a`-`a48d` |
| Sec. 7.4 | planning on an estimator's map: the triangle partition of collisions, caution's rank cost, the margin across kappa | `a49_estimated_obstacles.py` | `a49a`-`a49d` |
| Sec. 5.4 | the corner law: the Galerkin reading, the corner asymptotics, the closed form, junction exactness, and the refuted maximum-principle route (Prop. 14) | `a47_corner_law.py` | `a47a`-`a47f` |
| Sec. 7.3 | (H3) exercised on a fleet: the pencil survives a new `u`, the contact graph closes the a priori route | `a43_h3_multirobot.py` | `a43a`-`a43d` |
| Secs. 3, 7.3 | the two steps of (11) apart, the classical rank bound's number, and the arbitration cut's headroom | `a45_panel10.py` | `a45a`-`a45d` |
| Sec. 5 | the continuity order decides too: `eta <= 2k-1` safe, a `C^2` cubic outside | `a42_eta_boundary.py` | `a42a`-`a42e` |
| Sec. 7.3 | `rho = 2` decided in EXACT arithmetic: a rational KKT point and the instance it is optimal for (Prop. 25) | `a41_exact_rho2.py` | `a41a`-`a41d` |
| supplement | the Green negative lobe and its decay | `paper/make_figure.py` (submission package) | audited |
| Fig. 2 | the counterexample and its lift | `paper/make_figure2.py` (submission package) | audited |
| supplement | the lift in 3-D, two spheres | `paper/make_figure4.py` (submission package) | audited |
| supplement | the triangle partition and caution's rank cost, drawn from `a49`'s artifact | `paper/make_figure7.py` (submission package) | audited |
| video | whose fault: the naive projection's own curve cuts an obstacle, the certified fold never does | `paper/make_video_loop.py` (submission package) | -- |
| video | the certificate in four acts: contacts, the `m x m` spectrum and its gap, the fold set, the exact verdict | `paper/make_video_cert.py` (submission package) | -- |
| Sec. 3 | the certificate does not grow: `f` runs 2 to 10 and `n` runs 2 to 5 while `m` stays at 2 | `a50_size_contrast.py` | `a50a`-`a50d` |
| video | the five clips cut into one 83 s argument, in the paper's order | `paper/make_video_submission.py` (submission package) | -- |
| video | the corner law as a surface: the negative lobe of one segment, and the knot that restores the continuum constant | `paper/make_video_corner.py` (submission package) | -- |
| video | the same, as sizes: the lifted block grows, the certificate does not | `paper/make_video_size.py` (submission package) | -- |
| video | the same lift, rotated: the shadow inside the spheres from every azimuth | `paper/make_video.py` (submission package) | -- |

## What the tests pin

`tests/test_rho_certificate.py` asserts the chain INCLUDING where it fails:
the welded-knot cases that are *not* certified, the cells the certificate
*refuses*, and the single-segment control that stays negative at every degree
tried. `tests/test_exact_green.py` does the same for the exact arithmetic, and
asserts the boundary case explicitly: the Table I rows are nonnegative and
*not* strictly positive, which is why the theorem needs its irreducibility
hypothesis. A change that quietly promoted a measured statement to a certified
one would break them.

`python paper/audit_paper.py --control` perturbs every artifact number and
requires every check to fail; a check that survives is not testing anything.
It ships with the submission package (see above), not with the public mirror.

## Not included

The working repository also holds a branch-and-bound track (`bnb`, `shor`,
`lasserre`, `bm`, `oracle`, `rrt` and their phases). No claim in the paper
depends on it, so it is not here.
