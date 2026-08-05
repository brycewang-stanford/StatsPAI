"""Prediction-powered inference (PPI) for AI/ML-labeled data.

PPI (Angelopoulos et al. 2023) combines a small human-labeled sample
with a large sample whose labels come from an arbitrary predictive
model (an LLM annotator, a fitted regressor, …) and delivers valid
confidence intervals for a population quantity — without any
assumption on the prediction model's quality.  The recipe is

.. math::
    \\hat\\theta^{PP} = \\underbrace{\\tilde\\theta_f^{unl}}_{
        \\text{imputed estimate}}
    + \\underbrace{(\\hat\\theta_y^{lab} - \\hat\\theta_f^{lab})}_{
        \\text{rectifier from the labeled sample}},

with the two terms independent, so variances add.  The power-tuned
variant (PPI++, Angelopoulos, Duchi & Zrnic 2023) scales the imputed
term by a data-driven :math:`\\lambda \\in [0, 1]` chosen to minimise
the asymptotic variance — guaranteeing PPI is never worse than
classical labeled-only inference, even when the predictions are junk
(:math:`\\lambda \\to 0` recovers the classical estimator).

This module implements the two most-used targets:

* :func:`ppi_mean` — population mean of an outcome.
* :func:`ppi_ols` — OLS coefficients, via influence-function
  rectification with HC-style sandwich variances.

References
----------
Angelopoulos, A. N., Bates, S., Fannjiang, C., Jordan, M. I., &
Zrnic, T. (2023). "Prediction-powered inference."  *Science*,
382(6671), 669–674. doi:10.1126/science.adi6000.
[@angelopoulos2023prediction]

Angelopoulos, A. N., Duchi, J. C., & Zrnic, T. (2023). "PPI++:
Efficient Prediction-Powered Inference."  arXiv preprint
arXiv:2311.01453. [@angelopoulos2023ppi]
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from ..core.results import CausalResult
from ..exceptions import DataInsufficient, MethodIncompatibility, NumericalInstability

__all__ = ["ppi_mean", "ppi_ols"]


# ----------------------------------------------------------------------
# Shared coercion / validation
# ----------------------------------------------------------------------


def _as_1d(name: str, values: Any) -> np.ndarray:
    """Coerce to a finite 1-D float array; loud failure otherwise."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        raise DataInsufficient(f"{name} is empty.")
    if not np.all(np.isfinite(arr)):
        n_bad = int(np.sum(~np.isfinite(arr)))
        raise MethodIncompatibility(
            f"{name} contains {n_bad} non-finite value(s) (NaN/Inf). "
            "PPI has no missing-data path; drop or impute those rows "
            "explicitly before calling.",
            recovery_hint=f"Filter rows where {name} is NaN before calling.",
        )
    return arr


def _tuned_lambda(cov_yf: float, var_f: float, n: int, N: int) -> float:
    """PPI++ optimal power-tuning weight for a 1-D estimator, clipped
    to [0, 1].  ``lambda = Cov / (Var * (1 + n/N))``; degenerate
    prediction variance yields 0 (predictions carry no information)."""
    if not np.isfinite(var_f) or var_f <= 1e-300:
        return 0.0
    lam = cov_yf / (var_f * (1.0 + n / N))
    return float(np.clip(lam, 0.0, 1.0))


# ----------------------------------------------------------------------
# PPI mean
# ----------------------------------------------------------------------


def ppi_mean(
    *,
    y: Any,
    yhat: Any,
    yhat_unlabeled: Any,
    tune: bool = True,
    alpha: float = 0.05,
) -> CausalResult:
    """Prediction-powered estimate of a population mean.

    Combines ``n`` human-labeled outcomes with ``N`` model-predicted
    outcomes on unlabeled rows.  Valid regardless of prediction
    quality; with ``tune=True`` (PPI++) the interval is asymptotically
    never wider than the classical labeled-only interval.

    Parameters
    ----------
    y : array-like, shape (n,)
        Gold-standard (human) outcomes on the labeled sample.
    yhat : array-like, shape (n,)
        Model predictions on the *same* labeled rows.
    yhat_unlabeled : array-like, shape (N,)
        Model predictions on the unlabeled rows.
    tune : bool, default True
        Use the PPI++ power-tuning weight ``λ ∈ [0, 1]``.  ``False``
        fixes ``λ = 1`` (the original PPI estimator).
    alpha : float, default 0.05
        CI level (1 - alpha confidence).

    Returns
    -------
    CausalResult
        ``estimate`` / ``se`` / ``ci`` / ``pvalue`` for the mean;
        ``model_info`` carries ``lambda``, the classical labeled-only
        estimate and SE, the imputed-only mean, the labeled-sample
        correlation between ``y`` and ``yhat``, and the effective
        variance ratio versus classical inference.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(7)
    >>> theta = 2.0
    >>> n, N = 100, 5000
    >>> y_all = theta + rng.normal(0, 1, n + N)
    >>> f_all = y_all + rng.normal(0, 0.5, n + N)   # decent predictions
    >>> r = sp.ppi_mean(y=y_all[:n], yhat=f_all[:n],
    ...                 yhat_unlabeled=f_all[n:])
    >>> bool(r.ci[0] < theta < r.ci[1])
    True
    >>> bool(r.se < np.std(y_all[:n], ddof=1) / np.sqrt(n))  # beats classical
    True

    References
    ----------
    Angelopoulos et al. (2023), *Science* 382(6671), 669–674 —
    doi:10.1126/science.adi6000.  PPI++: arXiv:2311.01453.
    """
    y_arr = _as_1d("y", y)
    f_arr = _as_1d("yhat", yhat)
    fu_arr = _as_1d("yhat_unlabeled", yhat_unlabeled)
    if len(y_arr) != len(f_arr):
        raise MethodIncompatibility(
            f"y and yhat must be paired rows of the labeled sample; got "
            f"lengths {len(y_arr)} and {len(f_arr)}.",
            recovery_hint="Align y and yhat on the same labeled rows.",
        )
    n, N = len(y_arr), len(fu_arr)
    if n < 4:
        raise DataInsufficient(
            f"PPI needs at least 4 labeled rows for a variance estimate; got {n}."
        )
    if N < 4:
        raise DataInsufficient(
            f"PPI needs at least 4 unlabeled rows; got {N}. With so few "
            "unlabeled rows classical inference on the labeled sample "
            "is the better tool."
        )

    var_f_lab = float(np.var(f_arr, ddof=1))
    cov_yf = float(np.cov(y_arr, f_arr, ddof=1)[0, 1])
    lam = _tuned_lambda(cov_yf, var_f_lab, n, N) if tune else 1.0

    rectified = y_arr - lam * f_arr
    estimate = float(lam * np.mean(fu_arr) + np.mean(rectified))
    var_unl = float(lam**2 * np.var(fu_arr, ddof=1) / N)
    var_lab = float(np.var(rectified, ddof=1) / n)
    se = float(np.sqrt(var_unl + var_lab))
    if not np.isfinite(se) or se <= 0:
        raise NumericalInstability(
            "PPI variance estimate is degenerate (zero or non-finite); "
            "the outcome and predictions are numerically constant."
        )

    z = sp_stats.norm.ppf(1 - alpha / 2)
    ci = (estimate - z * se, estimate + z * se)
    pvalue = float(2 * (1 - sp_stats.norm.cdf(abs(estimate / se))))

    classical_est = float(np.mean(y_arr))
    classical_se = float(np.std(y_arr, ddof=1) / np.sqrt(n))
    sd_y = float(np.std(y_arr, ddof=1))
    sd_f = float(np.std(f_arr, ddof=1))
    corr = cov_yf / (sd_y * sd_f) if sd_y > 0 and sd_f > 0 else float("nan")

    model_info: Dict[str, Any] = {
        "lambda": float(lam),
        "tuned": bool(tune),
        "n_labeled": int(n),
        "n_unlabeled": int(N),
        "classical_estimate": classical_est,
        "classical_se": classical_se,
        "imputed_mean": float(np.mean(fu_arr)),
        "corr_y_yhat": float(corr),
        "var_ratio_vs_classical": (
            float((se / classical_se) ** 2) if classical_se > 0 else float("nan")
        ),
    }

    return CausalResult(
        method="Prediction-Powered Inference (mean)",
        estimand="mean",
        estimate=estimate,
        se=se,
        pvalue=pvalue,
        ci=(float(ci[0]), float(ci[1])),
        alpha=alpha,
        n_obs=int(n + N),
        model_info=model_info,
        _citation_key="ppi",
    )


# ----------------------------------------------------------------------
# PPI OLS
# ----------------------------------------------------------------------


def _design(X: Any, name: str, add_intercept: bool) -> Tuple[np.ndarray, list]:
    """Return (design matrix, term names)."""
    if isinstance(X, pd.DataFrame):
        names = [str(c) for c in X.columns]
        mat = X.to_numpy(dtype=np.float64)
    else:
        mat = np.asarray(X, dtype=np.float64)
        if mat.ndim == 1:
            mat = mat[:, None]
        names = [f"x{j}" for j in range(mat.shape[1])]
    if not np.all(np.isfinite(mat)):
        raise MethodIncompatibility(
            f"{name} contains non-finite values (NaN/Inf). PPI has no "
            "missing-data path; drop or impute those rows explicitly.",
            recovery_hint=f"Filter rows of {name} with NaN before calling.",
        )
    if add_intercept:
        mat = np.hstack([np.ones((mat.shape[0], 1)), mat])
        names = ["const"] + names
    return mat, names


def _ols_influence(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and per-row influence functions.

    ``psi_i = (X'X/n)^{-1} x_i e_i`` so that ``beta_hat ≈ beta +
    mean(psi)`` and ``Var(beta_hat) = Var(psi)/n`` (HC0 sandwich).
    Returns ``(beta, psi)`` with ``psi`` of shape (n, p).
    """
    n = X.shape[0]
    G = X.T @ X / n
    try:
        G_inv = np.linalg.inv(G)
    except np.linalg.LinAlgError as exc:
        raise NumericalInstability(
            f"Design matrix is singular ({exc}); drop collinear columns."
        ) from exc
    beta = G_inv @ (X.T @ y / n)
    resid = y - X @ beta
    psi = (X * resid[:, None]) @ G_inv.T
    return beta, psi


def ppi_ols(
    *,
    X: Any,
    y: Any,
    yhat: Any,
    X_unlabeled: Any,
    yhat_unlabeled: Any,
    target: Optional[Union[str, int]] = None,
    tune: bool = True,
    add_intercept: bool = True,
    alpha: float = 0.05,
) -> CausalResult:
    """Prediction-powered OLS with a labeled audit sample.

    Runs OLS of the model-predicted outcome on the unlabeled
    covariates, then rectifies every coefficient with the labeled
    sample's ``OLS(y) − OLS(yhat)`` contrast.  Standard errors combine
    the two independent sandwich variances; with ``tune=True`` each
    coefficient gets its own PPI++ weight ``λ_j ∈ [0, 1]``.

    Parameters
    ----------
    X : DataFrame or array, shape (n, p)
        Covariates for the labeled rows.
    y : array-like, shape (n,)
        Gold-standard outcomes for the labeled rows.
    yhat : array-like, shape (n,)
        Model-predicted outcomes for the same labeled rows.
    X_unlabeled : DataFrame or array, shape (N, p)
        Covariates for the unlabeled rows (same columns as ``X``).
    yhat_unlabeled : array-like, shape (N,)
        Model-predicted outcomes for the unlabeled rows.
    target : str or int, optional
        Which coefficient is the headline ``estimate``.  Defaults to
        the first non-intercept term.
    tune : bool, default True
        Per-coefficient PPI++ power tuning.  ``False`` fixes ``λ = 1``.
    add_intercept : bool, default True
        Prepend a constant column to both design matrices.
    alpha : float, default 0.05
        CI level (1 - alpha confidence).

    Returns
    -------
    CausalResult
        Headline coefficient in ``estimate`` / ``se`` / ``ci`` /
        ``pvalue``; the full coefficient table (with per-term ``λ`` and
        the classical labeled-only comparison) in ``detail``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(3)
    >>> n, N = 150, 4000
    >>> x_all = rng.normal(size=n + N)
    >>> y_all = 1.0 + 2.0 * x_all + rng.normal(0, 1, n + N)
    >>> f_all = y_all + rng.normal(0, 0.5, n + N)
    >>> r = sp.ppi_ols(
    ...     X=pd.DataFrame({"x": x_all[:n]}), y=y_all[:n], yhat=f_all[:n],
    ...     X_unlabeled=pd.DataFrame({"x": x_all[n:]}),
    ...     yhat_unlabeled=f_all[n:])
    >>> bool(r.ci[0] < 2.0 < r.ci[1])
    True
    >>> list(r.detail["term"])
    ['const', 'x']

    References
    ----------
    Angelopoulos et al. (2023), *Science* 382(6671), 669–674 —
    doi:10.1126/science.adi6000.  PPI++: arXiv:2311.01453.
    """
    y_arr = _as_1d("y", y)
    f_arr = _as_1d("yhat", yhat)
    fu_arr = _as_1d("yhat_unlabeled", yhat_unlabeled)
    X_lab, names = _design(X, "X", add_intercept)
    X_unl, names_unl = _design(X_unlabeled, "X_unlabeled", add_intercept)
    if names != names_unl or X_lab.shape[1] != X_unl.shape[1]:
        raise MethodIncompatibility(
            f"X and X_unlabeled must share the same columns; got "
            f"{names} vs {names_unl}.",
            recovery_hint="Reindex X_unlabeled to X's columns.",
        )
    n, p = X_lab.shape
    N = X_unl.shape[0]
    if len(y_arr) != n or len(f_arr) != n:
        raise MethodIncompatibility(
            f"X, y, yhat must have the same number of labeled rows; got "
            f"{n}, {len(y_arr)}, {len(f_arr)}.",
            recovery_hint="Align labeled rows before calling.",
        )
    if len(fu_arr) != N:
        raise MethodIncompatibility(
            f"X_unlabeled has {N} rows but yhat_unlabeled has " f"{len(fu_arr)}.",
            recovery_hint="Align unlabeled rows before calling.",
        )
    if n < p + 2:
        raise DataInsufficient(
            f"Labeled sample has {n} rows for {p} parameters; need at "
            f"least p + 2 = {p + 2}."
        )
    if N < p + 2:
        raise DataInsufficient(
            f"Unlabeled sample has {N} rows for {p} parameters; need at "
            f"least p + 2 = {p + 2}."
        )

    beta_f_unl, psi_f_unl = _ols_influence(X_unl, fu_arr)
    beta_y_lab, psi_y_lab = _ols_influence(X_lab, y_arr)
    beta_f_lab, psi_f_lab = _ols_influence(X_lab, f_arr)

    var_psi_f_unl = np.var(psi_f_unl, axis=0, ddof=1)
    var_psi_f_lab = np.var(psi_f_lab, axis=0, ddof=1)
    cov_psi = np.array(
        [
            float(np.cov(psi_y_lab[:, j], psi_f_lab[:, j], ddof=1)[0, 1])
            for j in range(p)
        ]
    )
    if tune:
        lam = np.array(
            [_tuned_lambda(cov_psi[j], float(var_psi_f_lab[j]), n, N) for j in range(p)]
        )
    else:
        lam = np.ones(p)

    beta_pp = lam * beta_f_unl + (beta_y_lab - lam * beta_f_lab)
    var_pp = (lam**2) * var_psi_f_unl / N + np.var(
        psi_y_lab - lam[None, :] * psi_f_lab, axis=0, ddof=1
    ) / n
    se_pp = np.sqrt(np.maximum(var_pp, 0.0))
    if not np.all(np.isfinite(se_pp)) or np.any(se_pp <= 0):
        raise NumericalInstability(
            "PPI OLS variance estimates are degenerate (zero or "
            "non-finite); check for constant columns or a constant "
            "outcome."
        )

    se_classical = np.sqrt(np.var(psi_y_lab, axis=0, ddof=1) / n)
    z = sp_stats.norm.ppf(1 - alpha / 2)
    tvals = beta_pp / se_pp
    pvals = 2 * (1 - sp_stats.norm.cdf(np.abs(tvals)))

    detail = pd.DataFrame(
        {
            "term": names,
            "estimate": beta_pp,
            "se": se_pp,
            "t": tvals,
            "pvalue": pvals,
            "ci_low": beta_pp - z * se_pp,
            "ci_high": beta_pp + z * se_pp,
            "lambda": lam,
            "classical_estimate": beta_y_lab,
            "classical_se": se_classical,
        }
    )

    # Headline coefficient.
    if target is None:
        idx = 1 if (add_intercept and p > 1) else 0
    elif isinstance(target, int):
        if not 0 <= target < p:
            raise MethodIncompatibility(
                f"target index {target} out of range for {p} terms.",
                recovery_hint=f"Use an index in [0, {p - 1}] or a term name.",
            )
        idx = target
    else:
        if target not in names:
            raise MethodIncompatibility(
                f"target {target!r} not among terms {names}.",
                recovery_hint="Pass one of the design column names.",
            )
        idx = names.index(str(target))

    model_info: Dict[str, Any] = {
        "target": names[idx],
        "terms": list(names),
        "lambda": [float(v) for v in lam],
        "tuned": bool(tune),
        "n_labeled": int(n),
        "n_unlabeled": int(N),
        "classical_estimate": float(beta_y_lab[idx]),
        "classical_se": float(se_classical[idx]),
        "imputed_estimate": float(beta_f_unl[idx]),
        "var_ratio_vs_classical": (
            float((se_pp[idx] / se_classical[idx]) ** 2)
            if se_classical[idx] > 0
            else float("nan")
        ),
    }

    return CausalResult(
        method="Prediction-Powered Inference (OLS)",
        estimand=f"coef[{names[idx]}]",
        estimate=float(beta_pp[idx]),
        se=float(se_pp[idx]),
        pvalue=float(pvals[idx]),
        ci=(float(beta_pp[idx] - z * se_pp[idx]), float(beta_pp[idx] + z * se_pp[idx])),
        alpha=alpha,
        n_obs=int(n + N),
        detail=detail,
        model_info=model_info,
        _citation_key="ppi",
    )
