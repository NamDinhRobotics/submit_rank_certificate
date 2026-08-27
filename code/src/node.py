"""The node relaxation for the branch-and-bound: (SDP) + R1 cuts + R2 + R3.

A node carries

  cuts        list of Cut(s_bar, a, beta, sense) -- the R1 dichotomies
  box         (l, u) on the control points        -- feeds R2(a) and R4
  directions  the distinct unit vectors a appearing in the cuts, each of which
              gets an R2(b) lift block of size d+2 (independent of n)

and the relaxation is the paper's (SDP) plus

  R1  sense * (a^T Gamma b_d(s_bar) - beta) >= 0            linear in Gamma
  R2a aggregated bound-factor RLT on (Gamma, X)             linear, no new cone
  R2b for each accumulated direction a:
        z_a = Gamma^T a,   [[1, z_a^T],[z_a, Z_a]] >= 0,   X >= Z_a
  R3  the implied products, without which branching shrinks the Gamma domain
      while X floats free and the gap never closes:
        z_a^T A = a^T B                 (boundary conditions, seen through a)
        Z_a A   = z_a (a^T B)           (the analogue of X A = Gamma^T B)
        pairwise products of cuts sharing a direction a

Sign convention for a cut: sense = -1 means a^T gamma(s_bar) <= beta,
sense = +1 means >=.  Only the pair matters; flipping `a` and `beta` together
swaps the two children.
"""
from dataclasses import dataclass, field

import numpy as np

from bernstein import bd
import rlt


@dataclass(frozen=True)
class Cut:
    s_bar: float
    a: tuple              # unit vector in R^n, hashable
    beta: float
    sense: int            # -1: <= beta,  +1: >= beta

    @property
    def a_vec(self):
        return np.asarray(self.a, float)

    def key(self):
        """Direction identity, up to sign (a and -a span the same axis)."""
        v = self.a_vec
        nz = np.flatnonzero(np.abs(v) > 1e-12)
        if nz.size and v[nz[0]] < 0:
            v = -v
        return tuple(np.round(v, 9))


@dataclass
class Node:
    cuts: list = field(default_factory=list)
    lo: np.ndarray = None            # (n, d+1) box lower bound, or None
    hi: np.ndarray = None
    depth: int = 0
    parent_lb: float = -np.inf
    tag: str = "root"

    def child(self, cut=None, lo=None, hi=None, tag=""):
        return Node(cuts=self.cuts + ([cut] if cut is not None else []),
                    lo=self.lo if lo is None else lo,
                    hi=self.hi if hi is None else hi,
                    depth=self.depth + 1, parent_lb=self.parent_lb,
                    tag=tag)

    def directions(self):
        """Distinct branching axes, in insertion order."""
        seen, out = set(), []
        for c in self.cuts:
            k = c.key()
            if k not in seen:
                seen.add(k)
                out.append((k, np.asarray(k, float)))
        return out


# ----------------------------------------------------------------------
def build(seg, node, use_rlt=True, use_lift=True, use_implied=True,
          use_XZ=True, vars=None):
    """Build the node relaxation in CVXPY.  Returns a dict of handles.

    `vars`, if given, is a pair of cvxpy EXPRESSIONS (Gfree, Xfree) to build
    the relaxation on instead of fresh Variables.  This is how the Shor
    diagnostic of Phase 6d reuses the exact same certificate/cut/RLT assembly
    on its lifted variables -- one source of truth for the constraints, so the
    diagnostic cannot silently solve a different node than the B&B does.  The
    [[I, G],[G^T, X]] block is kept even then: redundant under a Shor lift,
    but harmless, and it guarantees the caller gets a superset of this model.
    """
    import cvxpy as cp

    n, d, f = seg.n, seg.d, seg.f
    Gfix = seg.Gfix

    if vars is None:
        Gfree = cp.Variable((n, f), name="Gfree")
        Xfree = cp.Variable((f, f), symmetric=True, name="Xfree")
    else:
        Gfree, Xfree = vars

    Gamma = cp.hstack([cp.Constant(Gfix), Gfree]) @ seg.P.T
    cross = Gfix.T @ Gfree
    Xperm = cp.bmat([[cp.Constant(Gfix.T @ Gfix), cross],
                     [cross.T, Xfree]])
    X = seg.P @ Xperm @ seg.P.T

    cons = [cp.bmat([[np.eye(n), Gfree], [Gfree.T, Xfree]]) >> 0]

    # ---- the paper's obstacle certificates (unchanged) -----------------
    e_row = np.ones((1, d + 1))
    for cj, rj in seg.obs:
        Q0 = cp.Variable((d + 1, d + 1), symmetric=True)
        Q1 = cp.Variable((d, d), symmetric=True)
        cons += [Q0 >> 0, Q1 >> 0]
        Gc = Gamma.T @ cj.reshape(n, 1)
        ge = Gc @ e_row
        M = X - ge - ge.T + (float(cj @ cj) - rj ** 2) * np.ones((d + 1, d + 1))
        cons.append(cp.hstack(
            [cp.sum(cp.multiply(seg.Sd[k], M - Q0))
             - cp.sum(cp.multiply(seg.Td[k], Q1))
             for k in range(2 * d + 1)]) == 0)

    # ---- R1: the branching cuts ---------------------------------------
    for c in node.cuts:
        w = bd(c.s_bar, d)
        expr = c.a_vec @ (Gamma @ w)
        cons.append(expr <= c.beta if c.sense < 0 else expr >= c.beta)

    # ---- R4 box, and R2(a) aggregated bound-factor RLT -----------------
    #
    # The box must be imposed on Gamma ITSELF, not only through the RLT
    # inequalities it generates.  The RLT families are *derived* from
    # l <= Gamma <= u but do not imply it, and the proof that a box split
    # produces a tighter child relaxation uses Gamma >= l directly:
    #
    #   child_lower(i,j) - parent_lower(i,j) = delta * (Gamma_{p j} - l_{p j})
    #
    # which is only >= 0 when Gamma is actually confined to the box.  Omitting
    # it let a certified child bound fall 5.4e-4 BELOW its certified parent on
    # a box[0,2] split -- caught by Gate 3a, which is exactly what that gate
    # exists for.
    if node.lo is not None:
        cons += [Gamma >= node.lo, Gamma <= node.hi]
        if use_rlt:
            cons += rlt.cvxpy_constraints(Gamma, X, node.lo, node.hi)

    # ---- R2(b) + R3: one lift block per accumulated direction ---------
    lifts = {}
    if use_lift:
        A_bc, B_bc = seg.A_bc, seg.B_bc          # Gamma A = B
        for key, a in node.directions():
            z = Gamma.T @ a.reshape(n, 1)                     # (d+1, 1)
            Z = cp.Variable((d + 1, d + 1), symmetric=True)
            cons.append(cp.bmat([[np.ones((1, 1)), z.T],
                                 [z, Z]]) >> 0)               # Z >= z z^T
            # X - Z = Gamma^T (I - a a^T) Gamma is valid, but note that
            # rank(I - a a^T) = n - 1, so in the plane (n = 2) it forces a
            # difference of rank <= 1 inside a cone of size d+1.  That kills
            # strict feasibility and is the main source of the
            # 'optimal_inaccurate' node solves measured in Phase 3, so it is
            # switchable and its cost in bound quality is measured.
            if use_XZ:
                cons.append(X >> Z)
            if use_implied:
                aB = (a @ B_bc).reshape(1, -1)                # (1, 2(l+1))
                cons.append(z.T @ A_bc == aB)                 # z^T A = a^T B
                cons.append(Z @ A_bc == z @ aB)               # Z A = z (a^T B)
            lifts[key] = dict(z=z, Z=Z)

        # pairwise products of cuts sharing a direction (this is the part
        # that makes a *second* cut on the same axis do any work at all)
        if use_implied:
            for i1 in range(len(node.cuts)):
                for i2 in range(i1 + 1, len(node.cuts)):
                    c1, c2 = node.cuts[i1], node.cuts[i2]
                    if c1.key() != c2.key():
                        continue
                    L = lifts[c1.key()]
                    ax = np.asarray(c1.key(), float)
                    # re-express each cut against the shared axis `ax`
                    s1 = c1.sense * np.sign(c1.a_vec @ ax)
                    s2 = c2.sense * np.sign(c2.a_vec @ ax)
                    b1 = c1.beta * (c1.a_vec @ ax)
                    b2 = c2.beta * (c2.a_vec @ ax)
                    w1 = bd(c1.s_bar, d).reshape(-1, 1)
                    w2 = bd(c2.s_bar, d).reshape(-1, 1)
                    prod = (w1.T @ L["Z"] @ w2
                            - b2 * (w1.T @ L["z"])
                            - b1 * (w2.T @ L["z"])
                            + b1 * b2)
                    cons.append(s1 * s2 * prod >= 0)

    cost = cp.sum(cp.multiply(seg.Gk, X))
    prob = cp.Problem(cp.Minimize(cost), cons)
    return dict(prob=prob, Gfree=Gfree, Xfree=Xfree, Gamma=Gamma, X=X,
                lifts=lifts, cons=cons, cost=cost)


def solve(seg, node, solver="CLARABEL", ladder=(1e-11, 1e-9, 1e-8), **kw):
    """Solve the node relaxation; returns the same dict shape as Segment.solve."""
    import cvxpy as cp
    import warnings

    h = build(seg, node, **kw)
    last = dict(converged=False, status="not attempted", lb=-np.inf)
    for tol in ladder:
        opts = dict(verbose=False)
        if solver == "CLARABEL":
            opts.update(tol_gap_abs=tol, tol_gap_rel=tol, tol_feas=tol,
                        max_iter=800)
        elif solver == "SCS":
            opts.update(eps=tol, max_iters=200000)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                h["prob"].solve(solver=solver, **opts)
        except cp.error.SolverError as exc:
            last = dict(converged=False, status=f"SolverError: {exc}",
                        lb=-np.inf, infeasible=False)
            continue
        st = h["prob"].status
        if st in ("infeasible", "infeasible_inaccurate"):
            return dict(converged=True, status=st, infeasible=True, lb=np.inf)
        if st not in ("optimal", "optimal_inaccurate") or h["Gfree"].value is None:
            last = dict(converged=False, status=st, lb=-np.inf, infeasible=False)
            continue

        Gfree = np.asarray(h["Gfree"].value, float)
        Xfree = 0.5 * (np.asarray(h["Xfree"].value, float)
                       + np.asarray(h["Xfree"].value, float).T)
        out = seg._package(Gfree, Xfree, float(h["prob"].value))
        out.update(status=st, converged=(st == "optimal"), tol=tol,
                   infeasible=False, lb=float(h["prob"].value))
        if st == "optimal":
            return out
        last = out
    return last


# ----------------------------------------------------------------------
def obbt(seg, node, ub, solver="CLARABEL", tol=1e-9, coords=None, **kw):
    """Optimality-based bound tightening on the node relaxation itself.

    For each free control-point coordinate, minimise and maximise it subject to
    ALL the node's constraints -- obstacle certificates included -- plus the
    incumbent cost bound.  The result is the tightest box implied by everything
    the node knows, not just by the cost:

        lo_new[p,i] = min  Gamma[p,i]   s.t.  node relaxation, cost <= ub
        hi_new[p,i] = max  Gamma[p,i]   s.t.  node relaxation, cost <= ub

    Valid because every point feasible for the node with cost <= ub satisfies
    both.

    The objective is carried by a cp.Parameter so the problem is canonicalised
    ONCE and only re-solved per coordinate.  That matters far more than the
    conic solves do: on an 11-obstacle node in R^3, building a fresh
    cp.Problem per direction costs 75.6 ms each against 3.9 ms for a
    re-solve -- a 19x difference, and canonicalisation was ~95% of the time.

    `coords` optionally restricts the work to a list of (p, i) pairs.

    Returns (lo, hi) or None if the node is infeasible under the cost bound.
    """
    import cvxpy as cp
    import warnings

    if node.lo is None:
        return None
    h = build(seg, node, **kw)
    base = list(h["cons"]) + [h["cost"] <= ub]
    lo = node.lo.copy()
    hi = node.hi.copy()

    if coords is None:
        coords = [(p, i) for p in range(seg.n) for i in seg.Fidx]

    wp = cp.Parameter((seg.n, seg.d + 1), name="obbt_dir")
    prob = cp.Problem(cp.Minimize(cp.sum(cp.multiply(wp, h["Gamma"]))), base)
    opts = dict(verbose=False, tol_gap_abs=tol, tol_gap_rel=tol,
                tol_feas=tol, max_iter=400)

    W = np.zeros((seg.n, seg.d + 1))
    for (p, i) in coords:
        for sense, store in ((1.0, lo), (-1.0, hi)):
            W[:] = 0.0
            W[p, i] = sense
            wp.value = W.copy()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    prob.solve(solver=solver, **opts)
            except Exception:
                continue
            if prob.status in ("infeasible", "infeasible_inaccurate"):
                return None
            if prob.status not in ("optimal", "optimal_inaccurate"):
                continue
            val = sense * float(prob.value)
            if sense > 0:
                store[p, i] = max(store[p, i], val - 1e-9)
            else:
                store[p, i] = min(store[p, i], val + 1e-9)
    if np.any(lo > hi + 1e-12):
        return None
    return lo, hi
