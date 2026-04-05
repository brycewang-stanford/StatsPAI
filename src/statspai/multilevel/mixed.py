"""
Linear mixed effects (hierarchical linear) models.

y_ij = X_ij'beta + Z_ij'u_j + eps_ij
where u_j ~ N(0, G),  eps_ij ~ N(0, sigma^2)

Equivalent to Stata ``mixed`` and R ``lme4::lmer()``.

Estimation via profiled (RE)ML with Newton-type optimisation over
variance components, analytical GLS for fixed effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from ..core.results import EconometricResults


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class MixedResult:
    """Container for mixed model estimation results."""

    fixed_effects: pd.Series
    random_effects: pd.DataFrame  # BLUPs per group
    variance_components: Dict[str, float]
    blups: Dict[str, np.ndarray]
    n_obs: int
    n_groups: int
    icc: float
    log_likelihood: float

    # internal bookkeeping (set after __init__)
    _se_fixed: pd.Series = field(default=None, repr=False)
    _groups: np.ndarray = field(default=None, repr=False)
    _method: str = field(default="reml", repr=False)
    _converged: bool = field(default=True, repr=False)
    _lr_test: Optional[Dict[str, float]] = field(default=None, repr=False)
    _alpha: float = field(default=0.05, repr=False)

    # ------------------------------------------------------------------
    def ranef(self) -> pd.DataFrame:
        """Return random effects (BLUPs) as a DataFrame."""
        return self.random_effects.copy()

    # ------------------------------------------------------------------
    def summary(self) -> str:
        """Formatted summary table (Stata-style)."""
        lines: List[str] = []
        w = 72
        lines.append("=" * w)
        lines.append("Mixed-effects model".center(w))
        lines.append("=" * w)

        lines.append(f"  Method:          {self._method.upper()}")
        lines.append(f"  No. obs:         {self.n_obs}")
        lines.append(f"  No. groups:      {self.n_groups}")
        lines.append(f"  Log-likelihood:  {self.log_likelihood:.4f}")
        lines.append(f"  ICC:             {self.icc:.4f}")
        lines.append(f"  Converged:       {self._converged}")
        lines.append("-" * w)

        # Fixed effects table
        lines.append("Fixed effects:")
        alpha = self._alpha
        t_crit = stats.norm.ppf(1 - alpha / 2)
        hdr = f"{'':>16s} {'Coef':>10s} {'Std.Err':>10s} {'z':>8s} {'P>|z|':>8s}  [{100*(1-alpha):.0f}% CI]"
        lines.append(hdr)
        lines.append("-" * w)
        for var in self.fixed_effects.index:
            b = self.fixed_effects[var]
            se = self._se_fixed[var] if self._se_fixed is not None else np.nan
            z = b / se if se > 0 else np.nan
            p = 2 * (1 - stats.norm.cdf(abs(z))) if not np.isnan(z) else np.nan
            lo, hi = b - t_crit * se, b + t_crit * se
            lines.append(
                f"{var:>16s} {b:10.4f} {se:10.4f} {z:8.3f} {p:8.4f}  [{lo:.4f}, {hi:.4f}]"
            )

        lines.append("-" * w)
        lines.append("Variance components:")
        for name, val in self.variance_components.items():
            lines.append(f"  {name:20s}  {val:.6f}")

        if self._lr_test is not None:
            lines.append("-" * w)
            lr = self._lr_test
            lines.append(
                f"LR test vs. linear model: chi2({lr['df']:.0f}) = {lr['chi2']:.4f}, "
                f"Prob > chi2 = {lr['p']:.4f}"
            )
        lines.append("=" * w)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def to_econometric_results(self) -> EconometricResults:
        """Convert to the StatsPAI unified result class."""
        params = self.fixed_effects
        se = self._se_fixed if self._se_fixed is not None else pd.Series(
            np.nan, index=params.index
        )
        model_info = {
            "model_type": "Mixed-effects (LMM)",
            "method": self._method,
            "converged": self._converged,
        }
        data_info = {
            "n_obs": self.n_obs,
            "n_groups": self.n_groups,
            "df_resid": self.n_obs - len(params),
        }
        diagnostics = {
            "log_likelihood": self.log_likelihood,
            "icc": self.icc,
            "variance_components": self.variance_components,
        }
        if self._lr_test is not None:
            diagnostics["lr_test"] = self._lr_test

        return EconometricResults(
            params=params,
            std_errors=se,
            model_info=model_info,
            data_info=data_info,
            diagnostics=diagnostics,
        )


# ---------------------------------------------------------------------------
# Helper: build X and Z for one group
# ---------------------------------------------------------------------------

def _build_matrices(df_j: pd.DataFrame, x_fixed: List[str],
                    x_random: List[str]):
    """Return (y_j, X_j, Z_j) arrays for a single group."""
    y_j = df_j["__y__"].values.astype(float)
    X_j = df_j[["__intercept__"] + x_fixed].values.astype(float)
    Z_j = df_j[["__intercept__"] + x_random].values.astype(float) if x_random else df_j[["__intercept__"]].values.astype(float)
    return y_j, X_j, Z_j


# ---------------------------------------------------------------------------
# Core estimation
# ---------------------------------------------------------------------------

def _negloglik(theta: np.ndarray, groups_data, p_fixed: int, q_random: int,
               n_total: int, reml: bool):
    """
    Negative (restricted) log-likelihood as a function of variance components.

    theta layout: [log(diag G_1), ..., log(diag G_q), log(sigma2)]
    We optimise on log scale to enforce positivity.
    """
    g_diag = np.exp(theta[:q_random])
    sigma2 = np.exp(theta[q_random])
    G = np.diag(g_diag)

    # Accumulate sufficient statistics for GLS beta
    XtVinvX = np.zeros((p_fixed, p_fixed))
    XtVinvy = np.zeros(p_fixed)
    logdet_sum = 0.0
    quad_sum = 0.0

    for y_j, X_j, Z_j in groups_data:
        n_j = len(y_j)
        V_j = Z_j @ G @ Z_j.T + sigma2 * np.eye(n_j)

        try:
            L_j = np.linalg.cholesky(V_j)
        except np.linalg.LinAlgError:
            return 1e12  # not positive-definite; return large value

        logdet_sum += 2.0 * np.sum(np.log(np.diag(L_j)))

        # V_j^{-1} via Cholesky solve
        Vinv_X = np.linalg.solve(V_j, X_j)
        Vinv_y = np.linalg.solve(V_j, y_j)

        XtVinvX += X_j.T @ Vinv_X
        XtVinvy += X_j.T @ Vinv_y

    # GLS beta
    try:
        beta = np.linalg.solve(XtVinvX, XtVinvy)
    except np.linalg.LinAlgError:
        return 1e12

    # Quadratic form
    for y_j, X_j, Z_j in groups_data:
        r_j = y_j - X_j @ beta
        n_j = len(y_j)
        V_j = Z_j @ G @ Z_j.T + sigma2 * np.eye(n_j)
        quad_sum += r_j @ np.linalg.solve(V_j, r_j)

    nll = 0.5 * (logdet_sum + quad_sum + n_total * np.log(2 * np.pi))

    if reml:
        sign, logdet_xtvinvx = np.linalg.slogdet(XtVinvX)
        if sign <= 0:
            return 1e12
        nll += 0.5 * logdet_xtvinvx
        nll -= 0.5 * p_fixed * np.log(2 * np.pi)

    return nll


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def mixed(
    data: pd.DataFrame,
    y: str,
    x_fixed: list,
    group: str,
    x_random: list | None = None,
    method: str = "reml",
    maxiter: int = 200,
    tol: float = 1e-6,
    alpha: float = 0.05,
) -> MixedResult:
    """
    Fit a linear mixed effects model.

    Parameters
    ----------
    data : DataFrame
        Panel / grouped data.
    y : str
        Dependent variable column.
    x_fixed : list of str
        Fixed-effect regressors (intercept added automatically).
    group : str
        Column identifying level-2 groups.
    x_random : list of str, optional
        Random-slope variables.  ``None`` -> random intercept only.
    method : {'reml', 'ml'}
        Estimation method.
    maxiter : int
        Maximum iterations for the optimiser.
    tol : float
        Convergence tolerance.
    alpha : float
        Significance level for CIs in the summary.

    Returns
    -------
    MixedResult
    """
    if method not in ("reml", "ml"):
        raise ValueError("method must be 'reml' or 'ml'")

    df = data[[y] + x_fixed + [group]].dropna().copy()
    df.rename(columns={y: "__y__"}, inplace=True)
    df["__intercept__"] = 1.0

    if x_random is None:
        x_random_cols: list = []
    else:
        x_random_cols = list(x_random)

    group_labels = df[group].unique()
    n_groups = len(group_labels)
    n_obs = len(df)
    p_fixed = 1 + len(x_fixed)  # intercept + covariates
    q_random = 1 + len(x_random_cols)  # intercept + random slopes

    # Pre-compute per-group matrices
    groups_data = []
    for g in group_labels:
        sub = df[df[group] == g]
        y_j, X_j, Z_j = _build_matrices(sub, x_fixed, x_random_cols)
        groups_data.append((y_j, X_j, Z_j))

    reml = method == "reml"

    # Starting values: OLS residual variance, small random effects
    ols_beta = np.linalg.lstsq(
        np.vstack([X for _, X, _ in groups_data]),
        np.concatenate([yy for yy, _, _ in groups_data]),
        rcond=None,
    )[0]
    resid = np.concatenate([yy - X @ ols_beta for yy, X, _ in groups_data])
    s2_start = max(float(np.var(resid)), 1e-4)

    theta0 = np.zeros(q_random + 1)
    theta0[:q_random] = np.log(s2_start * 0.1)
    theta0[q_random] = np.log(s2_start * 0.9)

    # Optimise
    res = minimize(
        _negloglik,
        theta0,
        args=(groups_data, p_fixed, q_random, n_obs, reml),
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": tol},
    )
    converged = res.success

    # Extract variance components
    g_diag = np.exp(res.x[:q_random])
    sigma2 = np.exp(res.x[q_random])
    G = np.diag(g_diag)

    # Re-compute GLS beta at optimum
    XtVinvX = np.zeros((p_fixed, p_fixed))
    XtVinvy = np.zeros(p_fixed)

    Vinv_list = []
    for y_j, X_j, Z_j in groups_data:
        n_j = len(y_j)
        V_j = Z_j @ G @ Z_j.T + sigma2 * np.eye(n_j)
        Vinv_j = np.linalg.inv(V_j)
        Vinv_list.append(Vinv_j)
        XtVinvX += X_j.T @ Vinv_j @ X_j
        XtVinvy += X_j.T @ Vinv_j @ y_j

    beta_hat = np.linalg.solve(XtVinvX, XtVinvy)

    # Standard errors of fixed effects
    cov_beta = np.linalg.inv(XtVinvX)
    se_beta = np.sqrt(np.diag(cov_beta))

    fixed_names = ["_cons"] + list(x_fixed)
    fixed_effects = pd.Series(beta_hat, index=fixed_names)
    se_fixed = pd.Series(se_beta, index=fixed_names)

    # BLUPs:  u_hat_j = G Z_j' V_j^{-1} (y_j - X_j beta_hat)
    random_names = ["_cons"] + x_random_cols
    blup_dict: Dict[str, np.ndarray] = {}
    blup_rows = []
    for idx, g in enumerate(group_labels):
        y_j, X_j, Z_j = groups_data[idx]
        r_j = y_j - X_j @ beta_hat
        u_hat = G @ Z_j.T @ Vinv_list[idx] @ r_j
        blup_dict[g] = u_hat
        blup_rows.append(dict(zip(random_names, u_hat)))

    random_effects_df = pd.DataFrame(blup_rows, index=group_labels)
    random_effects_df.index.name = group

    # Log-likelihood at optimum
    ll = -res.fun
    if reml:
        # Convert REML to ML log-lik for reporting
        sign, logdet = np.linalg.slogdet(XtVinvX)
        ll_ml = ll - 0.5 * logdet + 0.5 * p_fixed * np.log(2 * np.pi)
    else:
        ll_ml = ll

    # ICC (for random intercept component)
    sigma2_u0 = float(g_diag[0])
    icc = sigma2_u0 / (sigma2_u0 + sigma2)

    # Variance components dict
    vc: Dict[str, float] = {}
    for i, name in enumerate(random_names):
        vc[f"var({name})"] = float(g_diag[i])
    vc["var(Residual)"] = float(sigma2)

    # LR test: random effects vs. pooled OLS (ML based)
    # Fit OLS log-likelihood for comparison
    X_all = np.vstack([X for _, X, _ in groups_data])
    y_all = np.concatenate([yy for yy, _, _ in groups_data])
    resid_ols = y_all - X_all @ ols_beta
    s2_ols = float(np.sum(resid_ols ** 2) / n_obs)
    ll_ols = -0.5 * n_obs * (np.log(2 * np.pi * s2_ols) + 1)
    chi2 = max(2 * (ll_ml - ll_ols), 0.0)
    lr_df = q_random  # number of variance components being tested
    lr_p = 1.0 - stats.chi2.cdf(chi2, lr_df) if chi2 > 0 else 1.0

    lr_test = {"chi2": chi2, "df": lr_df, "p": lr_p}

    result = MixedResult(
        fixed_effects=fixed_effects,
        random_effects=random_effects_df,
        variance_components=vc,
        blups=blup_dict,
        n_obs=n_obs,
        n_groups=n_groups,
        icc=icc,
        log_likelihood=ll_ml,
        _se_fixed=se_fixed,
        _groups=group_labels,
        _method=method,
        _converged=converged,
        _lr_test=lr_test,
        _alpha=alpha,
    )
    return result
