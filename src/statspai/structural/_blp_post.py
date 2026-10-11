"""Post-estimation algebra for ``sp.blp``: diversion ratios and markups.

Both are functions of one market's share Jacobian ``dS[j, k] = ds_j / dp_k``
and nothing else, so they live apart from the estimator.
"""

from __future__ import annotations

import numpy as np


def diversion_matrix(dS: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Diversion ratios from a share Jacobian.

    ``D[j, k] = (ds_k / dp_j) / (-ds_j / dp_j)``: of the buyers product ``j``
    loses when its price rises, the fraction that moves to ``k``. Inside
    shares and the outside share sum to one, so the outside good receives
    ``1 - sum_k D[j, k]``.

    Returns
    -------
    D : np.ndarray
        ``(J, J)`` with a zero diagonal. Rows of products whose own-price
        derivative is not negative are NaN: a diversion ratio is undefined
        when a price rise does not lose buyers.
    outside : np.ndarray
        ``(J,)`` diversion to the outside good.
    undefined : np.ndarray
        Boolean mask of the NaN rows.
    """
    own = -np.diag(dS)
    undefined = ~(own > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        D = dS.T / own[:, np.newaxis]
    np.fill_diagonal(D, 0.0)
    D[undefined] = np.nan
    outside = 1.0 - D.sum(axis=1)
    return D, outside, undefined


def firm_markups(dS: np.ndarray, shares: np.ndarray, firms: np.ndarray) -> np.ndarray:
    """Price minus marginal cost under multi-product Bertrand pricing.

    Firm ``f`` chooses the prices of its products; the first-order condition
    for product ``j`` is ``s_j + sum_{k in f} (p_k - mc_k) ds_k / dp_j = 0``,
    so ``p - mc = -(O * dS')^{-1} s`` with ``O[j, k] = 1`` when ``j`` and
    ``k`` belong to the same firm [@berry1995automobile].
    """
    same_firm = (firms[:, np.newaxis] == firms[np.newaxis, :]).astype(float)
    return np.asarray(-np.linalg.solve(same_firm * dS.T, shares), dtype=float)
