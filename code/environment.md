# Environment

Recreate with:

```bash
conda create -y -n sdpbnb -c conda-forge python=3.12 numpy scipy matplotlib pytest
conda activate sdpbnb
pip install clarabel cvxpy scs
```

Resolved on 2026-08-04 (macOS 15.6, arm64):

| package | version |
|---|---|
| python | 3.12 |
| numpy | 2.5.1 |
| scipy | 1.18.0 |
| matplotlib | 3.11.1 |
| cvxpy | 1.9.2 |
| clarabel | 0.11.1 |
| scs | 3.2.11 |

`cvxpy.installed_solvers()` → `['CLARABEL', 'SCS', 'SCIPY', 'HIGHS', 'OSQP']`.

## Solver choice (Gate 0)

Solvers were tried in the order mosek → clarabel → cvxpy+SCS/Clarabel → Drake.

- **mosek**: not installed, and no licence is present on this machine
  (`~/mosek/` absent, `MOSEKLM_LICENSE_FILE` unset). Installing the wheel
  without a licence would give a solver that fails at solve time, so it was
  not installed. If a licence appears, add `pip install mosek` and pass
  `solver="MOSEK"` to `Segment.solve_cvxpy` — no other change is needed.
- **Clarabel** is therefore the primary solver.
- **Drake** (`pydrake`) was not needed once Clarabel was available.

All reported timings are open-source interior-point timings and are **not**
directly comparable to the paper's commercial-solver numbers.

## Note on IDE diagnostics

The repo lives inside a larger project (`cmc_opt`) whose `pyproject.toml`
anchors the editor's type checker at the wrong root, so the IDE reports
"cannot find module `bernstein`" for the flat intra-`src/` imports.
`pyrightconfig.json` in this directory sets the correct `venvPath`/`extraPaths`;
editors that honour the nearest config resolve cleanly. The imports are correct
at runtime — `tests/conftest.py` and each experiment prepend `src/` to
`sys.path`.
