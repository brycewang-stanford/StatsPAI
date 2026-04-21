r"""Model-averaging double/debiased machine learning (Ahrens et al. 2025).

Standard DML picks a single nuisance learner for the outcome regression
:math:`g(X)` and the treatment propensity :math:`m(X)`.  Getting that choice
wrong degrades the :math:`\sqrt n`-rate consistency of the target parameter
:math:`\theta`, so in practice applied researchers run DML under several
candidate learners and hope the estimates agree.

Ahrens, Hansen, Kurz, Schaffer and Wiemann (2025, *JAE*) formalise this as
**model averaging**: fit DML with a *set* of candidate nuisance models, weight
their :math:`\theta`-estimates by a cross-validated measure of nuisance risk,
and report the weighted average with an appropriately adjusted sandwich
variance.  When the true nuisance is close to any single candidate the
average is no worse than the best candidate; when no model dominates the
average outperforms each candidate individually.

This module implements the partially-linear regression (PLR) version:

* Fit DML-PLR under each candidate ``(ml_g_k, ml_m_k)`` pair.
* Form per-candidate :math:`\hat\theta_k` and cross-validated nuisance MSE
  :math:`\text{MSE}_k = \text{MSE}(\hat g_k) + \text{MSE}(\hat m_k)`.
* Compute weights :math:`w_k \propto \text{MSE}_k^{-1}` (or equal weights).
* Output the weighted estimate and a variance that accounts for the
  covariance *between* candidate scores.

References
----------
Ahrens, A., Hansen, C.B., Kurz, M., Schaffer, M.E. and Wiemann, T. (2025).
    "Model averaging for double machine learning."
    *Journal of Applied Econometrics*, 40(3), 381-402.
    DOI 10.1002/jae.3103.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..core.results import CausalResult


__all__ = ["dml_model_averaging", "model_averaging_dml", "DMLAveragingResult"]


def _default_candidates() -> List[Tuple[Any, Any, str]]:
    """Return a reasonable default roster of (g, m, label) triples."""
    from sklearn.linear_model import LassoCV, LogisticRegressionCV, RidgeCV
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

    return [
        (LassoCV(cv=5), LassoCV(cv=5), "lasso"),
        (RidgeCV(), RidgeCV(), "ridge"),
        (RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=1),
         RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=1), "rf"),
        (GradientBoostingRegressor(n_estimators=200, random_state=0),
         GradientBoostingRegressor(n_estimators=200, random_state=0), "gbm"),
    ]


class DMLAveragingResult(CausalResult):
    """CausalResult extended with per-candidate and weight details.

    Attributes stored in ``model_info``:

    * ``candidates``  — list of candidate labels.
    * ``theta_k``     — per-candidate :math:`\\hat\\theta`.
    * ``se_k``        — per-candidate SE.
    * ``mse_k``       — per-candidate nuisance risk (g + m).
    * ``weights``     — averaging weights.
    * ``weight_rule`` — how the weights were computed.
    """


def _fit_candidate_plr(
    Y: np.ndarray,
    D: np.ndarray,
    X: np.ndarray,
    ml_g: Any,
    ml_m: Any,
    n_folds: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Fit one PLR candidate; return (y_resid, d_resid, mse_g, mse_m)."""
    from sklearn.base import clone
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    n = len(Y)
    y_resid = np.zeros(n)
    d_resid = np.zeros(n)

    for tr, te in kf.split(X):
        g = clone(ml_g)
        g.fit(X[tr], Y[tr])
        y_resid[te] = Y[te] - g.predict(X[te])

        m = clone(ml_m)
        m.fit(X[tr], D[tr])
        d_resid[te] = D[te] - m.predict(X[te])

    mse_g = float(np.mean(y_resid ** 2))
    mse_m = float(np.mean(d_resid ** 2))
    return y_resid, d_resid, mse_g, mse_m


def dml_model_averaging(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: Sequence[str],
    candidates: Optional[List[Tuple[Any, Any, str]]] = None,
    n_folds: int = 5,
    seed: int = 0,
    weight_rule: str = "inverse_risk",
    alpha: float = 0.05,
) -> DMLAveragingResult:
    """Model-averaging DML-PLR estimator.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome column.
    treat : str
        Continuous-or-binary treatment column.
    covariates : list of str
        Covariate columns ``X``.
    candidates : list of (ml_g, ml_m, label), optional
        Candidate nuisance learners.  ``ml_g`` regresses ``y`` on ``X``;
        ``ml_m`` regresses ``treat`` on ``X``.  Defaults to a Lasso/Ridge/
        RandomForest/GradientBoosting roster.
    n_folds : int, default 5
        Cross-fitting folds per candidate.
    seed : int, default 0
    weight_rule : {"inverse_risk", "equal", "single_best"}
        How to weight candidate estimators.

        * ``"inverse_risk"`` — :math:`w_k \\propto 1/(\\text{MSE}_g + \\text{MSE}_m)` (default).
        * ``"equal"``         — :math:`w_k = 1/K`.
        * ``"single_best"``   — put all mass on the lowest-risk candidate.
    alpha : float, default 0.05
        Two-sided CI level.

    Returns
    -------
    DMLAveragingResult
        With the weighted :math:`\\hat\\theta`, SE, CI, and per-candidate
        diagnostics under ``result.model_info``.

    Examples
    --------
    >>> import statspai as sp
    >>> r = sp.dml_model_averaging(df, y="y", treat="d",
    ...                             covariates=[f"x{j}" for j in range(10)])
    >>> r.summary()
    >>> r.model_info["weights"]
    {"lasso": 0.31, "ridge": 0.09, "rf": 0.22, "gbm": 0.38}
    """
    from scipy import stats as sp_stats

    for c in [y, treat] + list(covariates):
        if c not in data.columns:
            raise ValueError(f"Column '{c}' not found in data")
    if weight_rule not in {"inverse_risk", "equal", "single_best"}:
        raise ValueError("weight_rule must be 'inverse_risk', 'equal' or 'single_best'")
    if len(covariates) == 0:
        raise ValueError("At least one covariate required")

    cand = list(candidates) if candidates is not None else _default_candidates()
    if len(cand) == 0:
        raise ValueError("No candidate nuisance models supplied")

    Y = data[y].to_numpy(dtype=float)
    D = data[treat].to_numpy(dtype=float)
    X = data[list(covariates)].to_numpy(dtype=float)
    n = len(Y)
    if n != len(D) or n != X.shape[0]:
        raise ValueError("Inconsistent row counts between y, treat, covariates")

    thetas, ses, mses, labels, resids = [], [], [], [], []
    for (ml_g, ml_m, label) in cand:
        y_r, d_r, mse_g, mse_m = _fit_candidate_plr(
            Y, D, X, ml_g, ml_m, n_folds, seed
        )
        denom = float(np.sum(d_r ** 2))
        if denom < 1e-12:
            continue
        theta_k = float(np.sum(d_r * y_r) / denom)
        # Influence function-based SE for candidate k
        psi = (y_r - theta_k * d_r) * d_r
        J = -np.mean(d_r ** 2)
        var_k = float(np.mean(psi ** 2) / (J ** 2) / n)
        ses.append(np.sqrt(max(var_k, 0.0)))
        thetas.append(theta_k)
        mses.append(mse_g + mse_m)
        labels.append(label)
        resids.append((y_r, d_r, theta_k))

    if not thetas:
        raise RuntimeError("No candidate produced a finite estimate")

    thetas = np.array(thetas)
    ses = np.array(ses)
    mses = np.array(mses)

    # --- weights ----------------------------------------------------- #
    if weight_rule == "equal":
        w = np.ones_like(thetas) / len(thetas)
    elif weight_rule == "single_best":
        w = np.zeros_like(thetas)
        w[int(np.argmin(mses))] = 1.0
    else:  # inverse_risk
        inv = 1.0 / np.clip(mses, 1e-12, None)
        w = inv / inv.sum()

    theta_avg = float(np.sum(w * thetas))

    # --- variance: covariance between candidate scores --------------- #
    psi_matrix = np.zeros((n, len(thetas)))
    for k, (y_r, d_r, theta_k) in enumerate(resids):
        J_k = -np.mean(d_r ** 2)
        psi_matrix[:, k] = (y_r - theta_k * d_r) * d_r / (J_k * np.sqrt(n))
    cov = psi_matrix.T @ psi_matrix
    var_avg = float(w @ cov @ w)
    se_avg = float(np.sqrt(max(var_avg, 0.0)))

    z = sp_stats.norm.ppf(1 - alpha / 2)
    ci = (theta_avg - z * se_avg, theta_avg + z * se_avg)
    pvalue = (
        float(2 * (1 - sp_stats.norm.cdf(abs(theta_avg / se_avg))))
        if se_avg > 0 else float("nan")
    )

    model_info: Dict[str, Any] = {
        "method": "Model-averaging DML (PLR)",
        "candidates": labels,
        "theta_k": dict(zip(labels, thetas.tolist())),
        "se_k": dict(zip(labels, ses.tolist())),
        "mse_k": dict(zip(labels, mses.tolist())),
        "weights": dict(zip(labels, w.tolist())),
        "weight_rule": weight_rule,
        "n_folds": n_folds,
        "n_obs": int(n),
        "alpha": alpha,
        "citation": (
            "Ahrens, A., Hansen, C.B., Kurz, M., Schaffer, M.E. and Wiemann, T. (2025). "
            "Model averaging for double machine learning. "
            "Journal of Applied Econometrics 40(3):381-402. DOI 10.1002/jae.3103."
        ),
    }

    return DMLAveragingResult(
        method="DML (PLR) with model averaging",
        estimand="ATE",
        estimate=theta_avg,
        se=se_avg,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=int(n),
        model_info=model_info,
    )


# R-style alias
model_averaging_dml = dml_model_averaging
