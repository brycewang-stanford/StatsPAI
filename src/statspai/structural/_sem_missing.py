"""Incomplete data for the structural equation engine.

Rows are grouped by the pattern of values they observe; the likelihood of
:mod:`statspai.structural._sem_engine` is then a sum over patterns. The
saturated mean and covariance, against which a model is tested, have no
closed form with missing values and are found by the EM algorithm.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

__all__ = ["_patterns", "_saturated_moments"]


def _patterns(Z: np.ndarray) -> List[Dict[str, Any]]:
    """Rows grouped by which variables they observe, with the group's mean
    and covariance (divisor: the group's size) of the observed ones."""
    seen = ~np.isnan(Z)
    _, first, inverse = np.unique(seen, axis=0, return_index=True, return_inverse=True)
    out = []
    for g, row in enumerate(first):
        idx = np.flatnonzero(seen[row])
        rows = np.flatnonzero(np.ravel(inverse) == g)
        block = Z[np.ix_(rows, idx)]
        mean = block.mean(axis=0)
        dev = block - mean
        out.append({"idx": idx, "rows": rows, "n": len(rows), "mean": mean,
                    "S": dev.T @ dev / len(rows)})  # fmt: skip
    return out


def _saturated_moments(
    Z: np.ndarray, patterns: List[Dict[str, Any]]
) -> Tuple[np.ndarray, np.ndarray]:
    """Maximum-likelihood mean and covariance of incomplete normal data, by
    the EM algorithm."""
    n, p = Z.shape
    mu = np.nanmean(Z, axis=0)
    S = np.diag(np.nanvar(Z, axis=0))
    for _ in range(20000):
        t1 = np.zeros(p)
        t2 = np.zeros((p, p))
        for g in patterns:
            o = g["idx"]
            block = Z[g["rows"]].copy()
            if len(o) < p:
                mis = np.setdiff1d(np.arange(p), o)
                slope = np.linalg.solve(S[np.ix_(o, o)], S[np.ix_(o, mis)])
                block[:, mis] = mu[mis] + (block[:, o] - mu[o]) @ slope
                t2[np.ix_(mis, mis)] += g["n"] * (
                    S[np.ix_(mis, mis)] - S[np.ix_(mis, o)] @ slope
                )
            t1 += block.sum(axis=0)
            t2 += block.T @ block
        mu_new = t1 / n
        S_new = t2 / n - np.outer(mu_new, mu_new)
        S_new = 0.5 * (S_new + S_new.T)
        change = max(np.max(np.abs(mu_new - mu)), np.max(np.abs(S_new - S)))
        mu, S = mu_new, S_new
        if change < 1e-13:
            break
    return mu, S
