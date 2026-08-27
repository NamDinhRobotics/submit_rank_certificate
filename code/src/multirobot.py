"""The same relaxation for R robots that must also avoid each other.

Everything the single-robot formulation needs survives the extension, which
is why this file is short.  Stack the control points of all robots into one
`Gamma_all` and lift its Gram once:

    X = Gamma_all^T Gamma_all,   block (i,k) = Gamma_i^T Gamma_k

Robot-obstacle avoidance is the single-robot constraint on the diagonal
block.  Robot-robot avoidance is

    ||gamma_i(s) - gamma_k(s)||^2 - (2 r_rob)^2 >= 0  on [0,1],

whose Bernstein coefficient matrix is `X_ii - X_ik - X_ki + X_kk` minus the
radius term -- AFFINE in the same lifted variable.  So the Markov-Lukacs
representation applies verbatim and no new machinery appears.

Two consequences are worth stating because they are what the extension is
for.

`X` is still exactly the `O(n)` invariant of `Gamma_all`, so the relaxation
is still independent of the ambient dimension: R robots in `R^3` cost what
they cost in `R^2`.  It is NOT independent of R -- the lifted block is
`R f x R f` -- so the size grows quadratically in the fleet.

And the certificate does not come along.  With `K` block diagonal over
robots, the Green matrix at the contacts factors as `Gr = L * B` (entrywise),
where `B` is the single-robot Green kernel and `L` is the Gram of the
contacts' robot-incidence vectors: `e_i` for an obstacle contact on robot i,
`e_i - e_k` for a contact between i and k.  `L` carries `-1` entries whenever
two contacts share a robot with opposite sign, so `Gr` has negative entries
even where `B` is strictly positive -- i.e. even on a spline family the
single-robot certificate certifies.  The obstruction is combinatorial, in
the contact graph, not analytic in the Green function.
"""
import numpy as np

from bernstein import S_map, T_map, bd, bd_derivs, gram_deriv


class MultiRobot:
    """R robots, one shared lift, pairwise avoidance included.

    `bc0`/`bc1` are lists of R arrays of shape (l+1, n): each robot has its
    own endpoints, which is what makes a swap a swap.
    """

    def __init__(self, n, d, k, l, obstacles, bc0, bc1, robot_radius=0.0,
                 pairs=None, trace_bound=1e4, moving=None):
        self.n, self.d, self.k, self.l = n, d, k, l
        self.R = len(bc0)
        if len(bc1) != self.R:
            raise ValueError("bc0 and bc1 must list the same robot count")
        self.rr = float(robot_radius)
        self.trace_bound = None if trace_bound is None else float(trace_bound)
        # Obstacles are either ONE shared list of (centre, radius), or one
        # such list per robot.  Sniffing the difference is how a shared list
        # of R entries silently becomes per-robot, so it is not sniffed:
        # per-robot is opted into by wrapping in `per_robot=[...]`.
        self.obs = self._parse_obstacles(obstacles)
        # A MOVING obstacle is a robot whose trajectory is DATA, not a variable.
        # `moving` is a list of `(C, radius)` with `C` an `(n, d+1)` Bernstein
        # control-point matrix, so the avoidance constraint is
        #
        #     ||gamma_i(s) - c(s)||^2 - r^2  >=  0   on [0, 1]
        #
        # whose Bernstein matrix is `X_ii - Gamma_i^T C - C^T Gamma_i + C^T C`
        # minus the radius term -- STILL AFFINE in the same lifted variable,
        # because `C` is constant.  A static obstacle is the special case
        # `C = c e^T`, so nothing about the machinery changes and the SDP does
        # not grow by a single row.
        #
        # This is why a moving obstacle costs nothing while asynchronous robot
        # timing costs everything: the obstacle's motion is KNOWN, so it is a
        # function of the same curve parameter, and the constraint stays
        # univariate.  Two robots on independent clocks would need `b(s)^T X
        # b(s')`, which Markov-Lukacs does not cover.
        # The RADIUS may be a polynomial too, and that is the whole of the
        # robust extension.  A constant radius enters the block as
        # `r^2 * e e^T`, and `e` is the Bernstein coefficient vector of the
        # constant 1 -- so `r^2 e e^T` is exactly `rho rho^T` for
        # `rho = r * e`.  Give the radius its own Bernstein coefficients
        # `rho` and the block becomes `rho rho^T`, representing
        #
        #     r(s) = sum_i rho_i b_i(s),   r(s)^2 = b(s)^T (rho rho^T) b(s)
        #
        # still affine in the lifted variable, still ONE SOS block, still
        # independent of `n`.  So an obstacle whose uncertainty GROWS with
        # lookahead -- a funnel rather than a disc -- costs nothing here,
        # while for a method that carries the tube explicitly it costs
        # structure.  `tube()` builds the usual linear profile.
        self.moving = [] if moving is None else [
            (np.asarray(C, float), self._radius_coeffs(r, d))
            for C, r in moving]
        for C, _r in self.moving:
            if C.shape != (n, d + 1):
                raise ValueError(
                    "moving obstacle control points must have shape "
                    "(n, d+1) = %s, got %s" % ((n, d + 1), C.shape))

        self.f = d + 1 - 2 * (l + 1)
        if self.f < 1:
            raise ValueError("no free control points")
        self.Kidx = list(range(l + 1)) + list(range(d - l, d + 1))
        self.Fidx = list(range(l + 1, d - l))

        db0, db1 = bd_derivs(0.0, d, l), bd_derivs(1.0, d, l)
        A = np.zeros((d + 1, 2 * (l + 1)))
        for i in range(l + 1):
            A[:, 2 * i] = db0[i]
            A[:, 2 * i + 1] = db1[i]
        assert np.allclose(A[self.Fidx, :], 0.0)
        Ainv = np.linalg.inv(A[self.Kidx, :])

        self.Gfix = []
        for rb in range(self.R):
            for nm, bc in (("bc0", bc0[rb]), ("bc1", bc1[rb])):
                arr = np.asarray(bc, float)
                if arr.shape != (l + 1, n):
                    raise ValueError(
                        "%s[%d] must have shape (l+1, n) = %s, got %s; for "
                        "l=0 pass [[x, y]] not [x, y]"
                        % (nm, rb, (l + 1, n), arr.shape))
            B = np.zeros((n, 2 * (l + 1)))
            for i in range(l + 1):
                B[:, 2 * i] = np.asarray(bc0[rb][i], float)
                B[:, 2 * i + 1] = np.asarray(bc1[rb][i], float)
            self.Gfix.append(B @ Ainv)

        self.perm = list(self.Kidx) + list(self.Fidx)
        self.P = np.zeros((d + 1, d + 1))
        for a, p in enumerate(self.perm):
            self.P[p, a] = 1.0

        self.Gk = gram_deriv(d, k)
        self.Sd, self.Td = S_map(d), T_map(d)
        self.pairs = ([(i, j) for i in range(self.R) for j in range(i + 1, self.R)]
                      if pairs is None else [tuple(p) for p in pairs])
        self._cvx = None

    @staticmethod
    def _radius_coeffs(r, d):
        """Scalar radius, or `d+1` Bernstein coefficients of `r(s)`."""
        arr = np.atleast_1d(np.asarray(r, float))
        if arr.size == 1:
            return float(arr[0]) * np.ones(d + 1)
        if arr.shape != (d + 1,):
            raise ValueError(
                "a radius profile needs d+1 = %d Bernstein coefficients, "
                "got shape %s; pass a scalar for a constant radius"
                % (d + 1, arr.shape))
        if np.any(arr < 0.0):
            raise ValueError("a radius profile must be non-negative")
        return arr

    def _parse_obstacles(self, obstacles):
        """Shared list, or `{'per_robot': [...]}` opted into explicitly.

        Sniffing which one was passed is how a shared list of exactly R
        entries silently becomes per-robot, so nothing is sniffed.
        """
        if isinstance(obstacles, dict):
            if set(obstacles) != {"per_robot"}:
                raise ValueError("obstacle dict must be {'per_robot': [...]}")
            per = obstacles["per_robot"]
            if len(per) != self.R:
                raise ValueError("per_robot needs %d lists, got %d"
                                 % (self.R, len(per)))
            return [[(np.asarray(c, float), float(r)) for c, r in ob]
                    for ob in per]
        shared = [(np.asarray(c, float), float(r)) for c, r in obstacles]
        return [list(shared) for _ in range(self.R)]

    # ---------------------------------------------------------------- build
    def _slice(self, rb):
        return slice(rb * self.f, (rb + 1) * self.f)

    def build(self):
        import cvxpy as cp
        n, d, f, R = self.n, self.d, self.f, self.R

        Gfree = cp.Variable((n, R * f), name="Gfree")
        Xfree = cp.Variable((R * f, R * f), symmetric=True, name="Xfree")
        cons = [cp.bmat([[np.eye(n), Gfree], [Gfree.T, Xfree]]) >> 0]
        if self.trace_bound is not None:
            cons.append(cp.trace(Xfree) <= self.trace_bound)

        Gam, Xb = [], {}
        for i in range(R):
            gi = Gfree[:, self._slice(i)]
            Gam.append(cp.hstack([cp.Constant(self.Gfix[i]), gi]) @ self.P.T)
        for i in range(R):
            for j in range(R):
                # X_ij in the natural (d+1) ordering, exact wherever the
                # control points are fixed by the boundary conditions.
                ff = Xfree[self._slice(i), self._slice(j)]
                kk = cp.Constant(self.Gfix[i].T @ self.Gfix[j])
                kf = cp.Constant(self.Gfix[i].T) @ Gfree[:, self._slice(j)]
                fk = Gfree[:, self._slice(i)].T @ cp.Constant(self.Gfix[j])
                Xb[(i, j)] = self.P @ cp.bmat([[kk, kf], [fk, ff]]) @ self.P.T

        e = np.ones(d + 1)
        Qvars = []

        def sos(M):
            Q0 = cp.Variable((d + 1, d + 1), symmetric=True)
            Q1 = cp.Variable((d, d), symmetric=True)
            cons.extend([Q0 >> 0, Q1 >> 0])
            Qvars.append((Q0, Q1))
            resid = [cp.sum(cp.multiply(self.Sd[t], M - Q0))
                     - cp.sum(cp.multiply(self.Td[t], Q1))
                     for t in range(2 * d + 1)]
            cons.append(cp.hstack(resid) == 0)

        for i in range(R):
            for (c, r) in self.obs[i]:
                Gc = Gam[i].T @ c
                sos(Xb[(i, i)] - cp.outer(Gc, e) - cp.outer(e, Gc)
                    + float(c @ c - r ** 2) * np.outer(e, e))
        for (i, j) in self.pairs:
            sos(Xb[(i, i)] - Xb[(i, j)] - Xb[(j, i)] + Xb[(j, j)]
                - float((2.0 * self.rr) ** 2) * np.outer(e, e))
        # Moving obstacles: the same block with one side frozen to data.
        for i in range(R):
            for (C, r) in self.moving:
                GC = Gam[i].T @ cp.Constant(C)
                # radius `r` alone, exactly the static-obstacle convention
                # of this class (the robot is a point against obstacles, a
                # disc only against other robots), so a frozen moving obstacle
                # reproduces the static one for ANY robot_radius.
                sos(Xb[(i, i)] - GC - GC.T
                    + cp.Constant(C.T @ C - np.outer(r, r)))

        cost = sum(cp.sum(cp.multiply(self.Gk, Xb[(i, i)])) for i in range(R))
        prob = cp.Problem(cp.Minimize(cost), cons)
        self._cvx = dict(prob=prob, Gfree=Gfree, Xfree=Xfree, Qvars=Qvars)
        return self._cvx

    # ---------------------------------------------------------------- solve
    LADDER = (1e-11, 1e-9, 1e-8)

    def solve(self, solver="CLARABEL", ladder=None):
        import warnings
        c = self._cvx or self.build()
        last = dict(converged=False, status="not attempted")
        for tol in (ladder or self.LADDER):
            opts = dict(verbose=False)
            if solver == "CLARABEL":
                opts.update(tol_gap_abs=tol, tol_gap_rel=tol, tol_feas=tol,
                            max_iter=800)
            elif solver == "SCS":
                opts.update(eps=tol, max_iters=200000)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    c["prob"].solve(solver=solver, **opts)
            except Exception as exc:                            # noqa: BLE001
                last = dict(converged=False, status="error: %s" % exc)
                continue
            st = c["prob"].status
            if st not in ("optimal", "optimal_inaccurate") \
                    or c["Gfree"].value is None:
                last = dict(converged=False, status=st)
                continue
            out = self._package(np.asarray(c["Gfree"].value, float),
                                0.5 * (np.asarray(c["Xfree"].value, float)
                                       + np.asarray(c["Xfree"].value, float).T),
                                float(c["prob"].value))
            out.update(status=st, converged=(st == "optimal"), tol=tol)
            if st == "optimal":
                return out
            last = out
        return last

    def _package(self, Gfree, Xfree, cost):
        """The JOINT gap and its rank: one lift shared by the whole fleet."""
        S = Xfree - Gfree.T @ Gfree
        S = 0.5 * (S + S.T)
        w = np.linalg.eigvalsh(S)
        scale = max(1.0, float(np.max(np.abs(Xfree))))
        rho = int(np.sum(w > 1e-6 * scale))
        wdesc = w[::-1]
        return dict(Gfree=Gfree, Xfree=Xfree, gap=S, gap_eigs=w,
                    lam1=float(wdesc[0]) if wdesc.size else 0.0,
                    lam2=float(wdesc[1]) if wdesc.size > 1 else 0.0,
                    rho=rho, cost=float(cost), rank_scale=scale,
                    Gamma=[self.full_Gamma(Gfree, i) for i in range(self.R)])

    def full_Gamma(self, Gfree, i):
        G = np.zeros((self.n, self.d + 1))
        G[:, self.Kidx] = self.Gfix[i]
        G[:, self.Fidx] = Gfree[:, self._slice(i)]
        return G

    # ------------------------------------------------------------ clearance
    def clearances(self, Gammas, ns=2001):
        """Continuous-time-ish check on a dense grid, both kinds of contact."""
        ss = np.linspace(0.0, 1.0, ns)
        Bs = np.array([bd(s, self.d) for s in ss])
        pts = [G @ Bs.T for G in Gammas]                     # (n, ns) each
        out = dict(obstacle=np.inf, pair=np.inf)
        for i, P in enumerate(pts):
            for (c, r) in self.obs[i]:
                v = np.sum((P - c[:, None]) ** 2, axis=0) - r ** 2
                out["obstacle"] = min(out["obstacle"], float(v.min()))
        for (i, j) in self.pairs:
            v = np.sum((pts[i] - pts[j]) ** 2, axis=0) - (2 * self.rr) ** 2
            out["pair"] = min(out["pair"], float(v.min()))
        return out
