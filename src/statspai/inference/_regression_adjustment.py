"""Regression adjustment with an outcome regression per treatment arm.

``teffects ra`` fits a linear outcome regression in each arm and averages the
difference of the two predictions; ``teffects ipwra`` fits the same two
regressions by weighted least squares with inverse-probability weights from a
logit of treatment, which makes the estimator doubly robust. Both are
M-estimators: the regressions, the logit and the two averages are solved
together, and the variance is the sandwich of the stacked moments. Written
out, that sandwich is the variance of the influence function below, which
carries the sampling error of every coefficient into the effect.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["regression_adjustment"]


def _logit(d: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Maximum-likelihood logit coefficients by Newton's method."""
    gamma = np.zeros(Z.shape[1])
    for _ in range(100):
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ gamma, -700, 700)))
        grad = Z.T @ (d - p)
        hess = (Z * (p * (1.0 - p))[:, None]).T @ Z
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError as exc:
            raise MethodIncompatibility(
                "g_computation: the propensity-score logit is not identified "
                "(collinear or perfectly separating covariates).",
                recovery_hint="Check ps_covariates for a variable that "
                "predicts treatment perfectly.",
            ) from exc
        gamma = gamma + step
        if np.max(np.abs(step)) < 1e-12:
            break
    return gamma


def regression_adjustment(
    y: np.ndarray,
    d: np.ndarray,
    X: np.ndarray,
    *,
    estimand: str,
    Z: Optional[np.ndarray] = None,
) -> Tuple[float, float, Dict[str, float]]:
    """Effect, standard error and the untreated potential-outcome mean.

    ``X`` and ``Z`` carry their intercept column. ``Z`` switches on the
    inverse-probability weights: ``1 / p`` and ``1 / (1 - p)`` for the ATE,
    ``1`` and ``p / (1 - p)`` for the ATT.
    """
    n = len(y)
    treated = d == 1
    n1, n0 = int(treated.sum()), int((~treated).sum())
    k = X.shape[1]
    if n1 <= k or n0 <= k:
        raise DataInsufficient(
            "g_computation(by_arm=True): each treatment arm needs more "
            f"observations than outcome-model coefficients ({k}); got "
            f"{n1} treated and {n0} untreated.",
            diagnostics={"n_treated": n1, "n_untreated": n0, "k": k},
        )
    att = estimand == "ATT"

    w1 = np.ones(n)
    w0 = np.ones(n)
    if Z is not None:
        gamma = _logit(d, Z)
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ gamma, -700, 700)))
        odds = p / (1.0 - p)
        if att:
            w0 = odds
            dw1 = np.zeros(n)
        else:
            w1 = 1.0 / p
            w0 = 1.0 / (1.0 - p)
            dw1 = -(1.0 - p) / p  # d w1 / d (z'gamma)
        dw0 = odds  # d w0 / d (z'gamma), for both estimands
        info = (Z * (p * (1.0 - p))[:, None]).T @ Z / n
        if_gamma = (Z * (d - p)[:, None]) @ np.linalg.inv(info)

    a1 = d * w1
    a0 = (1.0 - d) * w0
    Q1 = (X * a1[:, None]).T @ X / n
    Q0 = (X * a0[:, None]).T @ X / n
    beta1 = np.linalg.solve(Q1, (X * a1[:, None]).T @ y / n)
    beta0 = np.linalg.solve(Q0, (X * a0[:, None]).T @ y / n)
    e1 = y - X @ beta1
    e0 = y - X @ beta0

    score1 = X * (a1 * e1)[:, None]
    score0 = X * (a0 * e0)[:, None]
    if Z is not None:
        # the weights are estimated: their sampling error moves the two
        # weighted regressions through d(weight)/d(gamma)
        G1 = (X * (d * dw1 * e1)[:, None]).T @ Z / n
        G0 = (X * ((1.0 - d) * dw0 * e0)[:, None]).T @ Z / n
        score1 = score1 + if_gamma @ G1.T
        score0 = score0 + if_gamma @ G0.T
    if_beta1 = score1 @ np.linalg.inv(Q1)
    if_beta0 = score0 @ np.linalg.inv(Q0)

    m1, m0 = X @ beta1, X @ beta0
    if att:
        share = n1 / n
        tau = float(np.mean(m1[treated] - m0[treated]))
        pom0 = float(np.mean(m0[treated]))
        xbar = X[treated].mean(axis=0)
        psi = d * (m1 - m0 - tau) / share + if_beta1 @ xbar - if_beta0 @ xbar
        psi0 = d * (m0 - pom0) / share + if_beta0 @ xbar
    else:
        tau = float(np.mean(m1 - m0))
        pom0 = float(np.mean(m0))
        xbar = X.mean(axis=0)
        psi = (m1 - m0 - tau) + if_beta1 @ xbar - if_beta0 @ xbar
        psi0 = (m0 - pom0) + if_beta0 @ xbar
    se = float(np.sqrt(np.mean(psi**2) / n))
    return (
        tau,
        se,
        {"pomean0": pom0, "pomean0_se": float(np.sqrt(np.mean(psi0**2) / n))},
    )
