"""Exact Borusyak--Jaravel--Spiess variance for the imputation estimator.

Every quantity the imputation estimator reports is linear in the
outcome: ``tau_hat = v' y`` for a weight vector ``v`` that the design
determines.  The variance therefore has a closed form, and the reference
implementations use it --- Stata ``did_imputation`` (Borusyak, SSC) and
R ``didimputation::se_inner`` (Butts & Borusyak) agree with each other to
``3e-9`` on the harness fixture.

StatsPAI previously used an approximation to it, in two places, and both
were wrong in the same direction:

1.  The projection weights on untreated rows were approximated by
    ``n_k(unit) / n_untreated(unit)``-style shares. That is the exact
    least-squares projection only on a balanced panel with no covariates.
    The exact object is ``v* = -Z (Z0'Z0)^-1 Z1' w``.
2.  Treated residuals were centred on the *global* mean effect at that
    horizon. The reference centres each treated cell on the
    ``v^2``-weighted mean within its own (cohort, relative-time) block.

On the module-84 fixture the approximation put the headline standard
error 36 percent below the reference and the horizon standard errors
4.9--13 percent away with non-uniform sign; on ``mpdta`` the headline was
18 percent low. The direction of the headline error is the dangerous one
--- too small --- which is what the estimator's old runtime warning was
describing when it reported roughly 0.87 coverage at a nominal 95
percent level.

References
----------
Borusyak, K., Jaravel, X. and Spiess, J. (2024).  "Revisiting Event-Study
Designs: Robust and Efficient Estimation."  *Review of Economic Studies*,
91(6), 3253-3285.  [@borusyak2024revisiting]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import splu

__all__ = ["bjs_weight_vector", "bjs_exact_se"]


def bjs_weight_vector(
    *,
    design_all: sparse.csr_matrix,
    design_untreated: sparse.csr_matrix,
    treated_mask: np.ndarray,
    target_weights: np.ndarray,
) -> np.ndarray:
    """The exact linear weights ``v`` with ``tau_hat = v' y``.

    ``target_weights`` carries the weights on treated rows only (for a
    horizon, ``1/N_h`` on that horizon's treated cells and zero
    elsewhere).  Untreated rows receive minus the least-squares
    projection of those weights through the Y(0) design, which is what
    makes the estimator an imputation rather than a raw mean.
    """
    treated = np.asarray(treated_mask, dtype=bool)
    w = np.asarray(target_weights, dtype=float)

    gram = (design_untreated.T @ design_untreated).tocsc()
    # Z'w over ALL target rows, not just treated ones. The target of a
    # post-treatment horizon sits on treated cells, so the two agree
    # there; the target of a pre-treatment lead under the in-sample
    # convention sits on UNTREATED cells, and restricting the sum to
    # treated rows would send the whole weight vector to zero and report
    # a standard error of exactly 0.
    rhs = np.asarray(design_all.T @ w).ravel()
    try:
        solved = splu(gram).solve(rhs)
    except (RuntimeError, ValueError):
        # Singular only if the untreated sample fails to identify the
        # Y(0) model, which the caller's coverage checks already reject;
        # fall back to a least-squares solve rather than returning a
        # confident-looking number from a deficient system.
        solved = np.asarray(
            np.linalg.lstsq(gram.toarray(), rhs, rcond=None)[0], dtype=float
        )

    # tau = sum_S w_i (y_i - Z_i a) with a = (Z0'Z0)^-1 Z0'y. The second
    # term loads only on untreated rows, because that is the sample the
    # Y(0) model is fit on; treated rows therefore keep their own weight
    # exactly. This reduces to the reference's expression whenever the
    # target is confined to treated cells.
    projection = np.asarray(design_all @ solved, dtype=float).ravel()
    v = w.copy()
    v[~treated] -= projection[~treated]
    return v


def bjs_exact_se(
    *,
    v: np.ndarray,
    adjusted: np.ndarray,
    treated_mask: np.ndarray,
    cluster: np.ndarray,
    cohort: np.ndarray,
    relative_time: np.ndarray,
) -> float:
    """Cluster-robust standard error of ``v' y`` under the BJS convention.

    Mirrors ``didimputation::se_inner``: treated rows are demeaned within
    their (cohort, relative-time) block using ``v^2`` weights, untreated
    rows keep their Y(0) residual, and the cluster scores are summed.

    No small-sample correction is applied, matching both references.
    """
    treated = np.asarray(treated_mask, dtype=bool)
    v = np.asarray(v, dtype=float)
    adj = np.asarray(adjusted, dtype=float)

    tau = adj.copy()
    if treated.any():
        block = pd.DataFrame(
            {
                "g": pd.Series(np.asarray(cohort)[treated]).astype(str).to_numpy(),
                "e": pd.Series(np.asarray(relative_time)[treated])
                .astype(str)
                .to_numpy(),
                "v2": v[treated] ** 2,
                "adj": adj[treated],
            }
        )
        block["num"] = block["v2"] * block["adj"]
        grouped = block.groupby(["g", "e"], sort=False)[["num", "v2"]].transform("sum")
        denom = grouped["v2"].to_numpy()
        cell_mean = np.divide(
            grouped["num"].to_numpy(),
            denom,
            out=np.zeros_like(denom),
            where=denom > 0,
        )
        tau[treated] = adj[treated] - cell_mean

    scores = pd.Series(v * tau).groupby(pd.Series(np.asarray(cluster))).sum().to_numpy()
    return float(np.sqrt(np.sum(scores**2)))


def bjs_se_for_target(
    *,
    design_all: sparse.csr_matrix,
    design_untreated: sparse.csr_matrix,
    treated_mask: np.ndarray,
    target_weights: np.ndarray,
    adjusted: np.ndarray,
    cluster: np.ndarray,
    cohort: np.ndarray,
    relative_time: np.ndarray,
) -> float:
    """Convenience wrapper: build ``v`` then evaluate the variance."""
    v = bjs_weight_vector(
        design_all=design_all,
        design_untreated=design_untreated,
        treated_mask=treated_mask,
        target_weights=target_weights,
    )
    return bjs_exact_se(
        v=v,
        adjusted=adjusted,
        treated_mask=treated_mask,
        cluster=cluster,
        cohort=cohort,
        relative_time=relative_time,
    )
