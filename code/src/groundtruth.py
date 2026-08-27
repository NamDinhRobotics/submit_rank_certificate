"""Ground truth for the nonconvex problem (P): multi-start SLSQP over the free
control points, with collision enforced by dense sampling in s.

This is the reference for c*_P in Gate 1d and the source of the local minima
that Phase 2 clusters by homotopy class.  It returns *all* distinct minima
found, not just the best one, because Phase 2 needs the two best.

Analytic gradients throughout: SLSQP's finite differences are not accurate
enough for the 1e-5 agreement Gate 1d asks for.
"""
import numpy as np
from scipy.optimize import minimize

from bernstein import bd
from relaxation import Segment, obstacle_poly, poly_min_on_unit, exact_clearance


class GroundTruth:
    """Brute-force solver for (P) on one segment."""

    def __init__(self, n, d, k, l, obstacles, bc0, bc1, ns=200):
        self.seg = Segment(n, d, k, l, obstacles, bc0, bc1)
        self.n, self.d, self.f = n, d, self.seg.f
        self.obs = self.seg.obs
        self.Gk = self.seg.Gk
        self.bc0, self.bc1 = np.asarray(bc0, float), np.asarray(bc1, float)
        self._set_points(np.linspace(0.0, 1.0, ns))

    def _set_points(self, ss):
        """Collocation points at which collision is enforced in the NLP."""
        self.ss = np.asarray(ss, float)
        self.ns = self.ss.size
        self.Bs = np.array([bd(s, self.d) for s in self.ss])     # (ns, d+1)
        self.Bfree = self.Bs[:, self.seg.Fidx]                   # (ns, f)

    # ------------------------------------------------------------------
    def _G(self, x):
        return self.seg.full_Gamma(x.reshape(self.n, self.f))

    def cost(self, x):
        G = self._G(x)
        return float(np.trace(self.Gk @ G.T @ G))

    def cost_grad(self, x):
        G = self._G(x)
        return (2.0 * G @ self.Gk)[:, self.seg.Fidx].ravel()

    def cons(self, x):
        """Stacked  ||gamma(s_t)-c_j||^2 - r_j^2 >= 0  over samples and obstacles."""
        P = self.Bs @ self._G(x).T                        # (ns, n)
        return np.concatenate([np.sum((P - c) ** 2, axis=1) - r ** 2
                               for c, r in self.obs])

    def cons_jac(self, x):
        """d cons / d x, shape (ns*m, n*f)."""
        P = self.Bs @ self._G(x).T
        blocks = []
        for c, _r in self.obs:
            R = P - c                                     # (ns, n)
            # d/dGamma_free[a,b] of |R_t|^2 = 2 R_t[a] * Bfree[t,b]
            J = 2.0 * np.einsum('ta,tb->tab', R, self.Bfree)
            blocks.append(J.reshape(self.ns, self.n * self.f))
        return np.vstack(blocks)

    def min_slack(self, x):
        """Sampled min slack -- optimistic, only used inside the NLP."""
        return float(self.cons(x).min())

    def exact_slack(self, x):
        """True min_{s in [0,1]} min_j p_j(s), from the roots of p'. No sampling."""
        v, _ = exact_clearance(self._G(x), self.obs, self.d)
        return v

    def _touch_points(self, x):
        """The exact argmin of p_j over [0,1], one per obstacle."""
        G = self._G(x)
        return [poly_min_on_unit(obstacle_poly(G, c, r, self.d))[1]
                for c, r in self.obs]

    def _polish(self, x0, cons_pts, feas_tol, max_rounds=12, abort_above=None,
                ftol=1e-14, diag=None):
        """Semi-infinite cutting plane: solve, find the exact worst-case s,
        add it as a collocation point, repeat.

        A fixed grid lets the curve cut the corner between samples, so the
        resulting cost is below the true c*_P.  Adding the exact minimiser of
        p_j as a constraint point each round drives that error to zero; this
        is the same mechanism as the paper's section VIII-B post-processing.

        `abort_above` is a dominance cut-off used by the branch-and-bound,
        where the only purpose of a polish is to beat the incumbent.  Each
        round *adds* collocation points, so the round objectives are the
        values of a shrinking feasible set and would be non-decreasing if
        every round were solved globally.  They are not -- SLSQP is local, and
        we measured the sequence non-monotone on 41 of 187 multi-round calls
        -- so this is a priced heuristic, not a theorem.  See bnb.BnB's
        `polish_margin` for the measured price.

        Left at None (the default) the method is bit-for-bit what it was, so
        the Gate 1d ground truth is unaffected.

        `ftol` is SLSQP's convergence tolerance, 1e-14 by default because that
        is the value every number in this repo was produced with.  It is below
        what double-precision SLSQP can generally certify, and Phase 6b found
        the polish failing outright on most starting points with `Positive
        directional derivative for linesearch` -- the classic symptom of a
        line search asked for more decimals than the arithmetic has.  Phase 6e
        sweeps it; the default does not move without a measurement.

        `diag`, if a list, receives one record per SLSQP round: status,
        message, round objective, and -- for a FAILED round -- whether the
        rejected iterate is nonetheless exactly collision-free and what it
        costs.  That last one matters because a polish only ever *proposes* an
        incumbent (bnb verifies every one with exact_clearance), so a rejected
        iterate is not automatically worthless.  Appending to `diag` cannot
        change the numerics.
        """
        cons = [{'type': 'ineq', 'fun': self.cons, 'jac': self.cons_jac}]
        x = x0
        for rnd in range(max_rounds):
            r = minimize(self.cost, x, jac=self.cost_grad, constraints=cons,
                         method='SLSQP', options={'maxiter': 500, 'ftol': ftol})
            if diag is not None:
                rec = dict(round=rnd, success=bool(r.success),
                           status=int(r.status), message=str(r.message),
                           fun=float(r.fun), nit=int(r.nit),
                           n_pts=int(self.ns))
                if not r.success:
                    # is the rejected iterate usable at all?
                    rec["rejected_exact_slack"] = float(self.exact_slack(r.x))
                    rec["rejected_cost"] = float(self.cost(r.x))
                diag.append(rec)
            if not r.success:
                return None
            if abort_above is not None and float(r.fun) >= abort_above:
                return None
            x = r.x
            if self.exact_slack(x) >= -feas_tol:
                return float(r.fun), x.copy()
            new = [s for s in self._touch_points(x)
                   if np.min(np.abs(self.ss - s)) > 1e-9]
            if not new:
                return None                      # cannot refine further
            cons_pts = np.sort(np.concatenate([self.ss, new]))
            self._set_points(cons_pts)
        return None

    # ------------------------------------------------------------------
    def _starts(self, ntry, rng):
        """Diverse initial guesses: straight line, then detours of growing scale.

        Purely Gaussian perturbations around the straight line under-sample
        the 'go around the other side' basin, which is exactly the basin the
        looseness analysis needs, so we add explicit lateral detours.
        """
        f, n = self.f, self.n
        t = np.linspace(0.0, 1.0, f + 2)[1:-1]
        base = np.array([(1 - ti) * self.bc0[0] + ti * self.bc1[0] for ti in t]).T  # (n,f)
        out = [base.copy()]

        # deterministic lateral detours, both signs, several amplitudes
        chord = self.bc1[0] - self.bc0[0]
        nrm = np.linalg.norm(chord)
        if nrm > 1e-12 and n >= 2:
            perp = np.zeros((n, 1))
            perp[0, 0], perp[1, 0] = -chord[1] / nrm, chord[0] / nrm
            bump = np.sin(np.pi * t)[None, :]
            for amp in (0.5, 1.0, 1.75, 2.75, 4.0):
                for sgn in (+1.0, -1.0):
                    out.append(base + sgn * amp * perp * bump)

        # obstacle-aware starts: push the polygon to either side of each sphere
        for c, r in self.obs:
            for sgn in (+1.0, -1.0):
                off = base - c[:, None]
                d_ = np.linalg.norm(off, axis=0, keepdims=True)
                dirn = np.divide(off, np.maximum(d_, 1e-9))
                out.append(base + sgn * 1.5 * r * dirn)

        while len(out) < ntry:
            out.append(base + rng.standard_normal((n, f)) * rng.uniform(0.5, 3.0))
        return [g.ravel() for g in out[:ntry]]

    # ------------------------------------------------------------------
    def solve(self, ntry=60, seed=0, feas_tol=1e-9, dedup_tol=1e-4,
              ftol=1e-14):
        """Return dict with best cost/x and the list of distinct local minima.

        Every returned minimum is feasible for the *continuous-time* problem to
        `feas_tol`, verified exactly rather than by sampling.

        `ftol` is passed through to _polish; the default is the value the Gate
        1d ground truth was computed with, and Phase 6e measures what moving it
        does to c*_P before anything is changed.
        """
        rng = np.random.default_rng(seed)
        base_pts = self.ss.copy()
        minima = []
        for x0 in self._starts(ntry, rng):
            self._set_points(base_pts)           # each start gets a clean grid
            got = self._polish(x0, base_pts, feas_tol, ftol=ftol)
            if got is not None:
                minima.append(got)
        self._set_points(base_pts)

        minima.sort(key=lambda t: t[0])
        distinct = []
        for c_, x_ in minima:
            G = self._G(x_)
            if all(np.linalg.norm(G - self._G(xd)) > dedup_tol for _, xd in distinct):
                distinct.append((c_, x_))

        if not minima:
            return dict(ok=False, cost=None, x=None, minima=[], n_success=0)
        return dict(ok=True, cost=minima[0][0], x=minima[0][1],
                    minima=distinct, n_success=len(minima),
                    n_distinct=len(distinct))


def solve_P(n, d, k, l, obs, bc0, bc1, ntry=60, seed=0, ns=200):
    """Seed-compatible entry point: returns (cost, x) or None."""
    gt = GroundTruth(n, d, k, l, obs, bc0, bc1, ns=ns)
    r = gt.solve(ntry=ntry, seed=seed)
    return (r['cost'], r['x']) if r['ok'] else None
