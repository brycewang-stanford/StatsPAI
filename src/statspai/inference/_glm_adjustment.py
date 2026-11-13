"""G-computation with a logistic or Poisson outcome model.

Two layouts, as for the linear model:

* **one model per arm** (``by_arm=True``; Stata ``teffects ra (y x, logit)
  (d)``): the outcome model is fitted in each treatment arm, every unit is
  predicted under both, and the potential-outcome means are the averages.
  The regressions and the averages are one M-estimator, so the variance is
  the sandwich of the stacked moments; written out, it is the variance of
  the influence functions returned here.
* **one model with the treatment as a regressor** (the default layout;
  R ``marginaleffects::avg_comparisons`` of a ``glm``): predictions under
  ``d = 1`` and ``d = 0`` from the single fit.

Both links are canonical, so the score of a coefficient vector is
``X' (y - mu)`` and the information is ``X' diag(mu') X``.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from ..exceptions import ConvergenceFailure, DataInsufficient, MethodIncompatibility

__all__ = [
    "OUTCOME_MODELS",
    "CONTRASTS",
    "check_outcome",
    "fit_glm",
    "glm_mean",
    "glm_by_arm",
    "contrast_of_means",
]

OUTCOME_MODELS: Tuple[str, ...] = ("linear", "logit", "poisson")
CONTRASTS: Tuple[str, ...] = ("difference", "ratio", "odds_ratio")


def check_outcome(y: np.ndarray, model: str) -> None:
    """Refuse an outcome the model is not a model of."""
    if model == "logit":
        if not np.all((y == 0) | (y == 1)):
            raise MethodIncompatibility(
                "g_computation: outcome_model='logit' needs a 0/1 outcome.",
                recovery_hint=(
                    "Recode the outcome to 0/1, or use outcome_model='linear' "
                    "or 'poisson'."
                ),
                diagnostics={"n_not_binary": int(((y != 0) & (y != 1)).sum())},
            )
    elif model == "poisson":
        if np.any(y < 0):
            raise MethodIncompatibility(
                "g_computation: outcome_model='poisson' needs a non-negative "
                "outcome.",
                recovery_hint="Use outcome_model='linear' for this outcome.",
                diagnostics={"n_negative": int((y < 0).sum())},
            )


def glm_mean(eta: np.ndarray, model: str) -> Tuple[np.ndarray, np.ndarray]:
    """Mean and its derivative with respect to the linear index."""
    if model == "linear":
        return eta, np.ones_like(eta)
    if model == "logit":
        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -700, 700)))
        return mu, mu * (1.0 - mu)
    mu = np.exp(np.clip(eta, -700, 700))
    return mu, mu


def fit_glm(y: np.ndarray, X: np.ndarray, model: str, who: str) -> np.ndarray:
    """Maximum-likelihood coefficients by Newton's method (``X`` has the constant)."""
    k = X.shape[1]
    beta = np.zeros(k)
    if model == "poisson":
        beta[0] = np.log(max(float(np.mean(y)), 1e-8))
    for _ in range(200):
        mu, dmu = glm_mean(X @ beta, model)
        grad = X.T @ (y - mu)
        hess = (X * dmu[:, None]).T @ X
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError as exc:
            raise MethodIncompatibility(
                f"{who}: the {model} outcome model is not identified "
                "(collinear covariates, or a covariate that predicts the "
                "outcome perfectly).",
                recovery_hint="Drop the redundant or separating covariate.",
            ) from exc
        if not np.all(np.isfinite(step)):
            break
        beta = beta + step
        if np.max(np.abs(step)) < 1e-12:
            return beta
        if model != "linear" and np.max(np.abs(X @ beta)) > 500:
            break
    raise ConvergenceFailure(
        f"{who}: the {model} outcome model did not converge; a covariate may "
        "separate the outcome in one treatment arm.",
        recovery_hint=(
            "Look for a covariate pattern with only one outcome value in an "
            "arm, drop it, or use outcome_model='linear'."
        ),
    )


def glm_by_arm(
    y: np.ndarray, d: np.ndarray, X: np.ndarray, *, estimand: str, model: str
) -> Dict[str, np.ndarray]:
    """Potential-outcome means and their influence functions, a model per arm.

    Returns ``pom`` (``[E Y(0), E Y(1)]`` over the population of the
    estimand) and ``psi`` (``n x 2``), scaled so that the variance of
    ``pom[a]`` is ``mean(psi[:, a] ** 2) / n``.
    """
    who = "g_computation(by_arm=True)"
    n, k = X.shape
    treated = d == 1
    n1, n0 = int(treated.sum()), int(n - treated.sum())
    if n1 <= k or n0 <= k:
        raise DataInsufficient(
            f"{who}: each treatment arm needs more observations than "
            f"outcome-model coefficients ({k}); got {n1} treated and {n0} "
            "untreated.",
            diagnostics={"n_treated": n1, "n_untreated": n0, "k": k},
        )
    att = estimand == "ATT"
    target = treated if att else np.ones(n, dtype=bool)
    share = target.mean()
    pom = np.zeros(2)
    psi = np.zeros((n, 2))
    for a, arm in ((0, ~treated), (1, treated)):
        beta = fit_glm(y[arm], X[arm], model, who)
        mu, dmu = glm_mean(X @ beta, model)
        ind = arm.astype(float)
        info = (X * (ind * dmu)[:, None]).T @ X / n
        if_beta = (X * (ind * (y - mu))[:, None]) @ np.linalg.inv(info)
        pom[a] = float(mu[target].mean())
        grad = (X[target] * dmu[target, None]).mean(axis=0)
        psi[:, a] = target * (mu - pom[a]) / share + if_beta @ grad
    return {"pom": pom, "psi": psi}


def contrast_of_means(
    pom: np.ndarray, psi: np.ndarray, contrast: str
) -> Tuple[float, float, float, float]:
    """Contrast of ``E Y(1)`` and ``E Y(0)`` with delta-method errors.

    Returns ``(estimate, se, log_estimate, log_se)``. For ``"difference"``
    the last two are ``nan``; for the ratio measures the standard error of
    the estimate is the delta-method one on the natural scale and the log
    pair is what the confidence interval is built from.
    """
    n = len(psi)
    m0, m1 = float(pom[0]), float(pom[1])
    p0, p1 = psi[:, 0], psi[:, 1]
    if contrast == "difference":
        est = m1 - m0
        return est, float(np.sqrt(np.mean((p1 - p0) ** 2) / n)), np.nan, np.nan
    if contrast == "ratio":
        if not (m0 > 0 and m1 > 0):
            raise MethodIncompatibility(
                "g_computation: contrast='ratio' needs positive potential-"
                f"outcome means; got {m1:.6g} and {m0:.6g}.",
                recovery_hint="Use contrast='difference'.",
            )
        log_est = float(np.log(m1 / m0))
        g = p1 / m1 - p0 / m0
    else:
        if not (0 < m0 < 1 and 0 < m1 < 1):
            raise MethodIncompatibility(
                "g_computation: contrast='odds_ratio' needs potential-outcome "
                f"means strictly between 0 and 1; got {m1:.6g} and {m0:.6g}.",
                recovery_hint="Use it with a 0/1 outcome, or contrast='ratio'.",
            )
        log_est = float(np.log(m1 / (1 - m1)) - np.log(m0 / (1 - m0)))
        g = p1 / (m1 * (1 - m1)) - p0 / (m0 * (1 - m0))
    log_se = float(np.sqrt(np.mean(g**2) / n))
    est = float(np.exp(log_est))
    return est, est * log_se, log_est, log_se
