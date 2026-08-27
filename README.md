# A Spectral Certificate for Low-Dimensional Recovery in Convex Motion Planning

Reproduction code and supplementary video.

## What the work says

A semidefinite relaxation of collision-free motion planning is exact when the
rank `rho` of the gap between the relaxed matrix variable and the outer product
of the recovered control points is zero. When `rho > 0`, the relaxed solution is
an exact optimum of the same problem posed in `rho` EXTRA dimensions: it routes
the trajectory around obstacles through dimensions the robot does not have, and
the projection a planner actually recovers can pass straight through them.
`video/` shows precisely that happening.

The sharpest published bound is `rho <= f`, the number of free control points;
experiments in the literature never report `rho > 1`. This work closes the gap
between the two, and then recovers a trajectory when the gap is one:

- `dim ker Z` **equals** the number of unit eigenvalues of an `m x m` Gram
  matrix built from the discrete Green's function of the cost operator at the
  `m` active contacts -- an identity, not a bound, and one small eigenvalue
  problem in place of a problem-sized rank condition. It yields `rho <= m` at
  every cost order, degree and ambient dimension.
- The same object certifies a published benchmark **a priori** and in exact
  rational arithmetic: every configuration gives `rho <= 1` for all obstacle
  arrangements and all ambient dimensions.
- At `rho = 1` the gap factors as `v v^T` and the borrowed coordinate folds
  back along a spatial direction, `gamma_w = gamma* + z w`, giving an exact
  clearance identity and a recovered trajectory certified by exact
  interval-polynomial nonnegativity.
- A rational instance attaining `rho = 2`, decided in exact arithmetic, shows
  the certificate is sufficient and not necessary.

## Contents

    code/     the reproduction package: 9 modules, 17 phases, 11 test modules,
              and a committed JSON artifact for every number in the paper
    video/    the projection seen alone, the borrowed axis put back, a full
              revolution -- because a single view of a 3-D scene cannot settle
              whether a curve clears a sphere -- and then the fold that spends
              the borrowed axis and brings the curve back into the plane

## Reproducing

    cd code
    conda create -y -n sdpbnb -c conda-forge python=3.12 numpy scipy \
                   matplotlib pytest sympy
    conda activate sdpbnb
    pip install clarabel cvxpy scs

    python -m pytest tests/ -q            # the suite, including the tests that
                                          # pin the certificate's FAILURES
    python paper/audit_paper.py           # every artifact against the constant
                                          # its script computed
    python paper/verify_prop9_proof.py    # the nodal point's closed form,
                                          # re-derived exactly over Q
    bash run_all.sh                       # regenerate every artifact, then audit

`run_all.sh` re-solves several thousand semidefinite programs and takes about an
hour. The audit reads the committed artifacts and answers in seconds. All conic
solves use Clarabel and SCS; no commercial solver is required.

### One check runs short here

`audit_paper.py` normally verifies three hops: artifact -> the constant a script
computed -> the number printed in the manuscript. The third needs the
manuscript, which this repository does not carry, so a clone runs the first two
and states that it skipped the third. It does not pass quietly as though the hop
had been made.

`verify_prop9_proof.py` is unaffected and worth running on its own: it re-derives
every identity in the proof of the nodal point symbolically over the rationals,
for degrees 3 to 17, trusting no stored number at all.

## License

[CC BY 4.0](LICENSE). Use it, change it, build on it -- give credit.
