"""Conditional independence tests for categorical variables.

One implementation for the two places that need it: the testable
implications of a declared graph (:meth:`statspai.DAG.test_implications`)
and constraint-based structure learning (:func:`statspai.pc_algorithm`).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = ["discrete_ci_test"]


def discrete_ci_test(
    a: np.ndarray,
    b: np.ndarray,
    strata: Optional[np.ndarray] = None,
    kind: str = "chi-square",
) -> Tuple[float, int, int, int]:
    """Test ``a`` independent of ``b`` within the strata.

    The statistic is computed in each stratum and summed, and so are the
    degrees of freedom, counting in each stratum only the levels that
    occur there (``dagitty::localTests(type = "cis.chisq")``,
    ``bnlearn::ci.test(test = "x2-adf")``; ``"mi-adf"`` for the G test).

    Parameters
    ----------
    a, b : integer arrays
        Category codes, ``0 .. k-1``.
    strata : integer array, optional
        One code per row for the configuration of the conditioning
        variables. ``None`` is the marginal test.
    kind : {'chi-square', 'g-test'}
        Pearson's statistic or the likelihood ratio.

    Returns
    -------
    statistic : float
    df : int
        Zero when no stratum has two levels of both variables: the
        independence cannot be tested on these data.
    n_used : int
        Rows in the strata that contributed.
    width : int
        Largest ``min(rows, columns) - 1`` among those strata, the
        normaliser of Cramer's V.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    ka = int(a.max()) + 1 if a.size else 0
    kb = int(b.max()) + 1 if b.size else 0
    blocks: list[np.ndarray]
    if strata is None:
        blocks = [np.arange(a.size)]
    else:
        order = np.argsort(strata, kind="stable")
        cuts = np.flatnonzero(np.diff(np.asarray(strata)[order])) + 1
        blocks = np.split(order, cuts)

    stat, dof, n_used, width = 0.0, 0, 0, 0
    for rows in blocks:
        obs = np.zeros((ka, kb))
        np.add.at(obs, (a[rows], b[rows]), 1.0)
        obs = obs[obs.sum(axis=1) > 0][:, obs.sum(axis=0) > 0]
        if min(obs.shape) < 2:
            continue
        total = obs.sum()
        expected = np.outer(obs.sum(axis=1), obs.sum(axis=0)) / total
        if kind == "chi-square":
            stat += float(((obs - expected) ** 2 / expected).sum())
        else:
            seen = obs > 0
            stat += float(2 * (obs[seen] * np.log(obs[seen] / expected[seen])).sum())
        dof += (obs.shape[0] - 1) * (obs.shape[1] - 1)
        n_used += int(total)
        width = max(width, min(obs.shape) - 1)
    return stat, dof, n_used, width
