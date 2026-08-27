"""Aggregated (partial-traced) bound-factor RLT -- requirement R2(a).

See docs/rlt_lemma.md for the statement and proofs.  In short: per-entry
McCormick is inexpressible in (Gamma, X) because it needs Gamma_pi Gamma_qj for
p != q, but *sums over the ambient index p* of products of nonnegative bound
factors are still nonnegative and are exactly functions of X.  Four families
result, all linear in (Gamma, X), all adding no PSD block, and there are
O(d^2) of them regardless of n.

Coefficient form used below, for a box l <= Gamma <= u:

  LL   X_ij >= sum_p l_pj Gamma_pi + sum_p l_pi Gamma_pj - sum_p l_pi l_pj
  UU   X_ij >= sum_p u_pj Gamma_pi + sum_p u_pi Gamma_pj - sum_p u_pi u_pj
  LU   X_ij <= sum_p u_pj Gamma_pi + sum_p l_pi Gamma_pj - sum_p l_pi u_pj
  UL   X_ij <= sum_p l_pj Gamma_pi + sum_p u_pi Gamma_pj - sum_p u_pi l_pj
"""
import numpy as np


def residuals(G, X, l, u):
    """Slack of each family at a given (Gamma, X); all must be >= 0 for LL/UU
    and >= 0 for the (upper - X) form of LU/UL.

    Returns a dict of (d+1, d+1) arrays:
      'LL', 'UU'  = X_ij - lower_ij      (>= 0 required)
      'LU', 'UL'  = upper_ij - X_ij      (>= 0 required)
    """
    lo_LL, lo_UU, up_LU, up_UL = bounds(G, l, u)
    return dict(LL=X - lo_LL, UU=X - lo_UU, LU=up_LU - X, UL=up_UL - X)


def bounds(G, l, u):
    """The four affine bounds on X, evaluated at a given Gamma.

    Returns (lower_LL, lower_UU, upper_LU, upper_UL), each (d+1, d+1).
    """
    G = np.asarray(G, float)
    l = np.asarray(l, float)
    u = np.asarray(u, float)
    # sum_p a_pi G_pj  ==  (a^T G)_ij with a, G both (n, d+1)
    lG = l.T @ G                 # [lG]_ij = sum_p l_pi G_pj
    uG = u.T @ G                 # [uG]_ij = sum_p u_pi G_pj
    ll = l.T @ l
    uu = u.T @ u
    lu = l.T @ u                 # [lu]_ij = sum_p l_pi u_pj

    lower_LL = lG.T + lG - ll            # sum_p l_pj G_pi + sum_p l_pi G_pj - ...
    lower_UU = uG.T + uG - uu
    upper_LU = uG.T + lG - lu
    upper_UL = lG.T + uG - lu.T
    return lower_LL, lower_UU, upper_LU, upper_UL


def gap_bound(l, u):
    """Theorem 2: |X_ij - (Gamma^T Gamma)_ij| <= sum_p w_pi w_pj."""
    w = np.asarray(u, float) - np.asarray(l, float)
    return w.T @ w


def cvxpy_constraints(Gamma_expr, X_expr, l, u, families=("LL", "UU", "LU", "UL")):
    """Constraint list for a CVXPY model.

    Gamma_expr is (n, d+1) and X_expr is (d+1, d+1); both may be affine
    expressions rather than raw variables (in the facially reduced model they
    are).  Only the requested families are emitted.
    """
    l = np.asarray(l, float)
    u = np.asarray(u, float)
    lG = l.T @ Gamma_expr
    uG = u.T @ Gamma_expr
    ll, uu, lu = l.T @ l, u.T @ u, l.T @ u

    cons = []
    if "LL" in families:
        cons.append(X_expr >= lG.T + lG - ll)
    if "UU" in families:
        cons.append(X_expr >= uG.T + uG - uu)
    if "LU" in families:
        cons.append(X_expr <= uG.T + lG - lu)
    if "UL" in families:
        cons.append(X_expr <= lG.T + uG - lu.T)
    return cons


# ----------------------------------------------------------------------
# Theorem 3 check: the aggregated bound equals the sum over p of the
# per-entry McCormick bounds on the full Shor lift Y_{pi,pj} = G_pi G_pj.
# ----------------------------------------------------------------------
def mccormick_entrywise(G, l, u):
    """Per-entry McCormick envelope for each product G_pi * G_pj.

    Returns (lo, hi), each (n, d+1, d+1), with
      lo_pij = max(LL, UU) bound on G_pi G_pj,
      hi_pij = min(LU, UL) bound.
    """
    G = np.asarray(G, float)
    l = np.asarray(l, float)
    u = np.asarray(u, float)
    Gi = G[:, :, None]
    Gj = G[:, None, :]
    li, lj = l[:, :, None], l[:, None, :]
    ui, uj = u[:, :, None], u[:, None, :]

    ll = lj * Gi + li * Gj - li * lj
    uu = uj * Gi + ui * Gj - ui * uj
    lu = uj * Gi + li * Gj - li * uj
    ul = lj * Gi + ui * Gj - ui * lj
    return np.maximum(ll, uu), np.minimum(lu, ul)
