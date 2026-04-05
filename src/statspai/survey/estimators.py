"""
Survey estimation routines: means, totals, and GLM with design-corrected
variance estimation (linearisation / Taylor series).

Implements the same variance formulas as R ``survey::svymean``,
``survey::svytotal``, and ``survey::svyglm``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

if TYPE_CHECKING:
    from .design import SurveyDesign


# ====================================================================== #
#  Result container
# ====================================================================== #

@dataclass
class SurveyResult:
    """Container for survey estimation results."""

    estimate: pd.Series
    std_error: pd.Series
    ci_lower: pd.Series
    ci_upper: pd.Series
    deff: pd.Series  # design effect
    dof: float  # degrees of freedom
    alpha: float = 0.05

    @property
    def t_values(self) -> pd.Series:
        return self.estimate / self.std_error

    @property
    def p_values(self) -> pd.Series:
        return 2 * (1 - sp_stats.t.cdf(np.abs(self.t_values), df=self.dof))

    def summary(self) -> pd.DataFrame:
        """Pretty summary table."""
        tbl = pd.DataFrame({
            "Estimate": self.estimate,
            "Std.Err": self.std_error,
            "t": self.t_values,
            "p": self.p_values,
            f"CI({1-self.alpha:.0%}) lo": self.ci_lower,
            f"CI({1-self.alpha:.0%}) hi": self.ci_upper,
            "DEFF": self.deff,
        })
        return tbl

    def __repr__(self) -> str:
        return self.summary().to_string()


# ====================================================================== #
#  Internal helpers
# ====================================================================== #

def _resolve_vars(
    variables: Union[str, List[str]],
    data: pd.DataFrame,
) -> tuple[List[str], np.ndarray]:
    """Return (var_names, values_matrix)."""
    if isinstance(variables, str):
        variables = [variables]
    vals = data[variables].values.astype(np.float64)
    return variables, vals


def _stratified_cluster_var(
    scores: np.ndarray,
    strata: np.ndarray,
    cluster_ids: np.ndarray,
    fpc: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Taylor linearisation variance for clustered, stratified designs.

    Parameters
    ----------
    scores : (n, p)  linearised score contributions per observation
    strata : (n,)
    cluster_ids : (n,)
    fpc : (n,) or None — finite population correction fractions

    Returns
    -------
    var : (p,) estimated variances for each of *p* statistics
    """
    p = scores.shape[1]
    total_var = np.zeros(p)

    unique_strata = np.unique(strata)
    for h in unique_strata:
        mask_h = strata == h
        scores_h = scores[mask_h]
        clusters_h = cluster_ids[mask_h]

        unique_psu = np.unique(clusters_h)
        n_h = len(unique_psu)

        if n_h < 2:
            # Single-PSU stratum — contribute 0 (conservative, matches R survey)
            continue

        # Sum scores within each PSU
        psu_totals = np.zeros((n_h, p))
        for g, psu in enumerate(unique_psu):
            psu_totals[g] = scores_h[clusters_h == psu].sum(axis=0)

        psu_mean = psu_totals.mean(axis=0)
        # Between-PSU variance
        dev = psu_totals - psu_mean[None, :]
        s2 = (dev ** 2).sum(axis=0) / (n_h - 1)

        fpc_factor = 1.0
        if fpc is not None:
            f_h = fpc[mask_h][0]
            fpc_factor = 1 - f_h

        total_var += fpc_factor * n_h * s2

    return total_var


def _design_dof(strata: np.ndarray, cluster_ids: np.ndarray) -> float:
    """Degrees of freedom = (# PSUs) - (# strata)."""
    n_psu = len(np.unique(cluster_ids))
    n_strata = len(np.unique(strata))
    return max(float(n_psu - n_strata), 1.0)


def _srs_var(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Simple random sampling variance for comparison (DEFF denominator)."""
    p = values.shape[1]
    v = np.zeros(p)
    n = len(weights)
    w_sum = weights.sum()
    for j in range(p):
        wm = np.average(values[:, j], weights=weights)
        v[j] = np.sum(weights * (values[:, j] - wm) ** 2) / (w_sum ** 2) * n / (n - 1)
    return v


# ====================================================================== #
#  Public API
# ====================================================================== #

def svymean(
    variables: Union[str, List[str]],
    design: "SurveyDesign",
    alpha: float = 0.05,
) -> SurveyResult:
    """
    Survey-weighted mean with design-corrected standard errors.

    Uses Taylor-series linearisation identical to R ``survey::svymean``.

    Parameters
    ----------
    variables : str or list of str
        Column name(s) in the design's data.
    design : SurveyDesign
    alpha : float

    Returns
    -------
    SurveyResult
    """
    var_names, vals = _resolve_vars(variables, design.data)
    w = design.weights
    w_sum = w.sum()

    # Point estimates
    means = np.array([np.average(vals[:, j], weights=w) for j in range(len(var_names))])

    # Linearised scores for the mean: z_i = w_i * (y_i - mean) / sum(w)
    scores = w[:, None] * (vals - means[None, :]) / w_sum

    # Design variance
    design_var = _stratified_cluster_var(
        scores, design.strata, design.cluster_ids, design.fpc_values,
    )
    se = np.sqrt(design_var)

    # Design effect
    srs_v = _srs_var(vals, w)
    deff = np.where(srs_v > 0, design_var / srs_v, 1.0)

    # Confidence interval
    dof = _design_dof(design.strata, design.cluster_ids)
    t_crit = sp_stats.t.ppf(1 - alpha / 2, df=dof)

    return SurveyResult(
        estimate=pd.Series(means, index=var_names),
        std_error=pd.Series(se, index=var_names),
        ci_lower=pd.Series(means - t_crit * se, index=var_names),
        ci_upper=pd.Series(means + t_crit * se, index=var_names),
        deff=pd.Series(deff, index=var_names),
        dof=dof,
        alpha=alpha,
    )


def svytotal(
    variables: Union[str, List[str]],
    design: "SurveyDesign",
    alpha: float = 0.05,
) -> SurveyResult:
    """
    Survey-weighted total with design-corrected standard errors.

    Parameters
    ----------
    variables : str or list of str
    design : SurveyDesign
    alpha : float

    Returns
    -------
    SurveyResult
    """
    var_names, vals = _resolve_vars(variables, design.data)
    w = design.weights

    totals = np.array([(w * vals[:, j]).sum() for j in range(len(var_names))])

    # Linearised scores for total: z_i = w_i * y_i
    scores = w[:, None] * vals

    design_var = _stratified_cluster_var(
        scores, design.strata, design.cluster_ids, design.fpc_values,
    )
    se = np.sqrt(design_var)

    srs_v = _srs_var(vals, w) * (w.sum()) ** 2
    srs_v = np.where(srs_v > 0, srs_v, 1.0)
    deff = np.where(srs_v > 0, design_var / srs_v, 1.0)

    dof = _design_dof(design.strata, design.cluster_ids)
    t_crit = sp_stats.t.ppf(1 - alpha / 2, df=dof)

    return SurveyResult(
        estimate=pd.Series(totals, index=var_names),
        std_error=pd.Series(se, index=var_names),
        ci_lower=pd.Series(totals - t_crit * se, index=var_names),
        ci_upper=pd.Series(totals + t_crit * se, index=var_names),
        deff=pd.Series(deff, index=var_names),
        dof=dof,
        alpha=alpha,
    )


def svyglm(
    formula: str,
    design: "SurveyDesign",
    family: str = "gaussian",
    alpha: float = 0.05,
):
    """
    Survey-weighted generalised linear model.

    Fits WLS (for gaussian family) or weighted IRLS (for binomial/poisson)
    and computes design-corrected standard errors via the sandwich estimator.

    Parameters
    ----------
    formula : str
        ``"y ~ x1 + x2"`` style formula.
    design : SurveyDesign
    family : str
        ``"gaussian"``, ``"binomial"`` (logistic), or ``"poisson"``.
    alpha : float

    Returns
    -------
    SurveyResult with regression coefficient estimates.
    """
    from patsy import dmatrices

    y_df, X_df = dmatrices(formula, data=design.data, return_type="dataframe")
    y = y_df.values.ravel()
    X = X_df.values
    w = design.weights
    var_names = list(X_df.columns)
    n, k = X.shape

    if family == "gaussian":
        params, working_residuals = _wls_fit(y, X, w)
    elif family == "binomial":
        params, working_residuals = _irls_fit(y, X, w, family="binomial")
    elif family == "poisson":
        params, working_residuals = _irls_fit(y, X, w, family="poisson")
    else:
        raise ValueError(f"Unknown family: {family}. Use 'gaussian', 'binomial', or 'poisson'.")

    # Sandwich variance with survey design correction
    # Score contributions: z_i = w_i * r_i * x_i  (linearised influence)
    scores = w[:, None] * working_residuals[:, None] * X

    design_var = _stratified_cluster_var(
        scores, design.strata, design.cluster_ids, design.fpc_values,
    )

    # Bread: (X'WX)^{-1}
    WX = X * w[:, None]
    bread = np.linalg.inv(WX.T @ X)

    # Full sandwich: bread @ meat @ bread
    # meat is sum of outer products of cluster score totals (already in design_var)
    # But we need the full matrix, not just diagonal — recompute
    p = k
    meat_matrix = np.zeros((p, p))
    unique_strata = np.unique(design.strata)
    for h in unique_strata:
        mask_h = design.strata == h
        scores_h = scores[mask_h]
        clusters_h = design.cluster_ids[mask_h]
        unique_psu = np.unique(clusters_h)
        n_h = len(unique_psu)
        if n_h < 2:
            continue
        psu_totals = np.zeros((n_h, p))
        for g, psu in enumerate(unique_psu):
            psu_totals[g] = scores_h[clusters_h == psu].sum(axis=0)
        psu_mean = psu_totals.mean(axis=0)
        dev = psu_totals - psu_mean[None, :]
        s2 = dev.T @ dev / (n_h - 1)
        fpc_factor = 1.0
        if design.fpc_values is not None:
            f_h = design.fpc_values[mask_h][0]
            fpc_factor = 1 - f_h
        meat_matrix += fpc_factor * n_h * s2

    vcov = bread @ meat_matrix @ bread
    se = np.sqrt(np.diag(vcov))

    dof = _design_dof(design.strata, design.cluster_ids)
    t_crit = sp_stats.t.ppf(1 - alpha / 2, df=dof)
    estimates = pd.Series(params, index=var_names)
    se_s = pd.Series(se, index=var_names)

    return SurveyResult(
        estimate=estimates,
        std_error=se_s,
        ci_lower=estimates - t_crit * se_s,
        ci_upper=estimates + t_crit * se_s,
        deff=pd.Series(np.ones(k), index=var_names),  # DEFF not standard for regression
        dof=dof,
        alpha=alpha,
    )


# ====================================================================== #
#  Internal fitting helpers
# ====================================================================== #

def _wls_fit(
    y: np.ndarray, X: np.ndarray, w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted least squares.  Returns (params, residuals)."""
    W = np.sqrt(w)
    Xw = X * W[:, None]
    yw = y * W
    params = np.linalg.lstsq(Xw, yw, rcond=None)[0]
    residuals = y - X @ params
    return params, residuals


def _irls_fit(
    y: np.ndarray,
    X: np.ndarray,
    w: np.ndarray,
    family: str,
    max_iter: int = 25,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Iteratively Re-weighted Least Squares for GLM families.

    Returns (params, working_residuals) where working_residuals are
    the linearised residuals used for sandwich variance estimation.
    """
    n, k = X.shape
    beta = np.zeros(k)

    for _ in range(max_iter):
        eta = X @ beta

        if family == "binomial":
            mu = 1 / (1 + np.exp(-eta))
            mu = np.clip(mu, 1e-10, 1 - 1e-10)
            var_mu = mu * (1 - mu)
            deriv = var_mu  # d mu / d eta
        elif family == "poisson":
            mu = np.exp(np.clip(eta, -20, 20))
            var_mu = mu
            deriv = mu
        else:
            raise ValueError(f"Unknown family: {family}")

        # Working response and working weights
        z = eta + (y - mu) / deriv
        irls_w = w * deriv ** 2 / var_mu  # = w * var_mu for canonical link

        W = np.sqrt(irls_w)
        Xw = X * W[:, None]
        zw = z * W
        beta_new = np.linalg.lstsq(Xw, zw, rcond=None)[0]

        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    # Working residuals for sandwich
    eta = X @ beta
    if family == "binomial":
        mu = 1 / (1 + np.exp(-eta))
        mu = np.clip(mu, 1e-10, 1 - 1e-10)
    elif family == "poisson":
        mu = np.exp(np.clip(eta, -20, 20))

    working_residuals = (y - mu)

    return beta, working_residuals
