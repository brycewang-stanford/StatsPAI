"""M-estimation standard error of entropy balancing, any estimand.

Parameters are ``theta = (m, lambda_a for each reweighted arm, mu1, mu0)``.
With ``R = c(X) - m`` and, for a reweighted arm ``a`` with indicator
``S_a``, ``w_a = q exp(lambda_a' R)`` (``q`` the base weights), the
estimating functions per unit are

* ``T (c - m)``                  -- ``m``: the target moments, with ``T = D``
  (ATT), ``1 - D`` (ATC) or ``1`` (ATE);
* ``S_a w_a R``                  -- the balance conditions of arm ``a``;
* ``S_a w_a (Y - mu_a)``         -- a reweighted arm's outcome mean, or
  ``S_a (Y - mu_a)`` for the arm that is the target (ATT: treated, ATC:
  controls).

The effect is ``mu1 - mu0`` and its variance ``A^{-1} B A^{-T} / n`` with
``A`` the analytic mean Jacobian and ``B`` the mean outer product. The base
weights are fixed constants. A row-wise rescaling of an estimating function
leaves the sandwich unchanged, so the normalising constant of ``w_a`` does
not matter and ``lambda_a`` is recovered from the solver's weights by
regressing ``log(w_a / q)`` on ``[1, R]``.

For the ATT without base weights this is the variance
``WeightIt::lm_weightit(vcov = "asympt")`` reports, as before; the ATE /
ATC and base-weight cases are checked against the same function.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def ebalance_mestimation_se(
    Y: np.ndarray,
    D: np.ndarray,
    C: np.ndarray,
    target: np.ndarray,
    arm_weights: Dict[int, np.ndarray],
    base: Optional[np.ndarray] = None,
    sampling: Optional[np.ndarray] = None,
) -> float:
    """Sandwich SE of ``mu1 - mu0``.

    Parameters
    ----------
    Y, D : (n,) arrays
        Outcome and 0/1 treatment.
    C : (n, k) array
        Constraint functions ``c(X)``.
    target : (n,) bool array
        Rows whose mean of ``c(X)`` is the balancing target.
    arm_weights : {arm: weights}
        For each reweighted arm (1 = treated, 0 = controls) the solver's
        weights over that arm's rows, in row order.
    base : (n,) array, optional
        Base weights ``q`` (default 1).
    sampling : (n,) array, optional
        Sampling weights ``s`` (default 1). Every estimating function is
        multiplied by ``s``; ``arm_weights`` then include ``s``.
    """
    n, k = C.shape
    d = np.asarray(D, dtype=float)
    tgt = np.asarray(target, dtype=float)
    sw = np.ones(n) if sampling is None else np.asarray(sampling, dtype=float)
    q = np.ones(n) if base is None else np.asarray(base, dtype=float)
    q = q * sw  # the reweighted arms' weights are s * q * exp(lambda'R)
    sel = tgt.astype(bool)
    m = (sw[sel, None] * C[sel]).sum(axis=0) / sw[sel].sum()
    tgt = tgt * sw
    R = C - m

    arms = sorted(arm_weights)
    lam: Dict[int, np.ndarray] = {}
    w: Dict[int, np.ndarray] = {}
    for a in arms:
        rows = d == a
        design = np.column_stack([np.ones(int(rows.sum())), R[rows]])
        coef, *_ = np.linalg.lstsq(design, np.log(arm_weights[a] / q[rows]), rcond=None)
        lam[a] = coef[1:]
        wa = np.zeros(n)
        wa[rows] = q[rows] * np.exp(R[rows] @ lam[a])
        w[a] = wa

    # outcome means: reweighted arms use w_a, the target arm is unweighted
    s = {1: d * sw, 0: (1.0 - d) * sw}
    eff = {a: (w[a] if a in w else s[a]) for a in (1, 0)}
    mu = {a: float(np.sum(eff[a] * Y) / np.sum(eff[a])) for a in (1, 0)}

    # parameter layout: m | lambda_a (arms in order) | mu1 | mu0
    off = {a: k + i * k for i, a in enumerate(arms)}
    i_mu = {1: k + len(arms) * k, 0: k + len(arms) * k + 1}
    p = i_mu[0] + 1

    cols = [tgt[:, None] * R]
    for a in arms:
        cols.append(w[a][:, None] * R)
    cols.append((eff[1] * (Y - mu[1]))[:, None])
    cols.append((eff[0] * (Y - mu[0]))[:, None])
    psi = np.column_stack(cols)

    A = np.zeros((p, p))
    A[:k, :k] = -tgt.sum() * np.eye(k)
    for a in arms:
        rows = slice(off[a], off[a] + k)
        wr = (w[a][:, None] * R).sum(axis=0)
        A[rows, :k] = -(w[a].sum() * np.eye(k) + np.outer(wr, lam[a]))
        A[rows, off[a] : off[a] + k] = (w[a][:, None] * R).T @ R
    for a in (1, 0):
        r = i_mu[a]
        if a in w:
            resid = w[a] * (Y - mu[a])
            A[r, :k] = -resid.sum() * lam[a]
            A[r, off[a] : off[a] + k] = resid @ R
        A[r, r] = -eff[a].sum()
    A /= n
    B = psi.T @ psi / n
    A_inv = np.linalg.inv(A)
    V = A_inv @ B @ A_inv.T / n
    g = np.zeros(p)
    g[i_mu[1]], g[i_mu[0]] = 1.0, -1.0
    return float(np.sqrt(g @ V @ g))
