"""
Survey estimation routines: means, totals, and GLM with design-corrected
variance estimation (linearisation / Taylor series).

Implements the same variance formulas as R ``survey::svymean``,
``survey::svytotal``, and ``survey::svyglm``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Union

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .._result_serialize import ResultProtocolMixin

if TYPE_CHECKING:
    from .design import SurveyDesign


# ====================================================================== #
#  Result container
# ====================================================================== #


@dataclass
class SurveyResult(ResultProtocolMixin):
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
        return 2 * sp_stats.t.sf(np.abs(self.t_values), df=self.dof)

    def summary(self) -> pd.DataFrame:
        """Pretty summary table."""
        tbl = pd.DataFrame(
            {
                "Estimate": self.estimate,
                "Std.Err": self.std_error,
                "t": self.t_values,
                "p": self.p_values,
                f"CI({1 - self.alpha:.0%}) lo": self.ci_lower,
                f"CI({1 - self.alpha:.0%}) hi": self.ci_upper,
                "DEFF": self.deff,
            }
        )
        return tbl

    def __repr__(self) -> str:
        return str(self.summary().to_string())


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


def _design_vcov(scores: np.ndarray, design: "SurveyDesign") -> np.ndarray:
    """
    First-stage Taylor-linearisation (ultimate-cluster) covariance.

    Mirrors R ``survey:::onestage`` / ``onestrat`` for a single-stage
    design: within stratum *h* with ``n_h`` sampled PSUs, PSU score totals
    ``t_hj`` are centred at their stratum mean and

        V = sum_h (1 - f_hj) * n_h / (n_h - 1) * sum_j d_hj d_hj'

    with ``f_hj`` the sampling fraction (0 without fpc).  Strata with a
    single PSU follow ``design.lonely_psu`` (see :class:`SurveyDesign`).

    Parameters
    ----------
    scores : (n, p) linearised score contributions per observation
    design : SurveyDesign

    Returns
    -------
    (p, p) covariance matrix
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 1:
        scores = scores[:, None]
    p = scores.shape[1]
    strata = design._strata_codes
    psu = design._psu_codes
    fpc = design.fpc_values
    rule = design.lonely_psu

    n_strata = int(strata.max()) + 1
    # PSU totals, the PSU's stratum and its first-row sampling fraction
    n_psu_total = int(psu.max()) + 1
    t = np.zeros((n_psu_total, p))
    np.add.at(t, psu, scores)
    first = np.unique(psu, return_index=True)[1]
    psu_stratum = strata[first]
    psu_f = np.zeros(n_psu_total) if fpc is None else np.asarray(fpc)[first]

    recentre = scores.sum(axis=0) / n_psu_total if rule == "adjust" else None

    vcov = np.zeros((p, p))
    lonely = []
    n_ok = 0
    for h in range(n_strata):
        idx = np.flatnonzero(psu_stratum == h)
        n_h = idx.size
        one_minus_f = 1.0 - psu_f[idx]
        if n_h > 1:
            n_ok += 1
            if np.all(one_minus_f < 1e-7):
                continue  # certainty stratum
            d = t[idx] - t[idx].mean(axis=0)
            scale = one_minus_f * n_h / (n_h - 1)
            vcov += (d * scale[:, None]).T @ d
            continue
        lonely.append(h)
        if rule == "fail":
            raise ValueError(
                "Stratum with a single sampled PSU (lonely PSU); choose "
                "svydesign(lonely_psu='remove'|'certainty'|'adjust'|'average')"
            )
        if rule == "adjust" and not np.all(one_minus_f < 1e-7):
            d = t[idx] - recentre
            vcov += (d * one_minus_f[:, None]).T @ d
        # "remove" / "certainty": zero; "average": rescaled below
    if lonely and rule == "average":
        if n_ok == 0:
            raise ValueError("lonely_psu='average': every stratum has one PSU")
        vcov *= n_strata / n_ok
    if lonely and not design._lonely_psu_explicit:
        warnings.warn(
            f"{len(lonely)} stratum/strata with a single sampled PSU; they "
            "contribute zero variance (lonely_psu='remove'). R survey fails "
            "and Stata reports a missing SE by default -- set "
            "svydesign(lonely_psu=...) explicitly to choose the rule.",
            UserWarning,
            stacklevel=3,
        )
    return vcov


def _design_dof(design: "SurveyDesign") -> float:
    """Design degrees of freedom = (# PSUs) - (# strata), PSUs within strata.

    Same as R ``survey::degf`` (with ``nest=TRUE``) and Stata ``e(df_r)``.
    """
    n_psu = int(design._psu_codes.max()) + 1
    n_strata = int(design._strata_codes.max()) + 1
    return max(float(n_psu - n_strata), 1.0)


def _deff_denominator(
    values: np.ndarray, weights: np.ndarray, mode: str, total: bool
) -> np.ndarray:
    """SRS variance used as DEFF denominator, as in R ``svymean(deff=)``.

    ``svyvar`` = sum w (y - ybar)^2 / sum w * n / (n - 1).  Without
    replacement (``mode="wor"``, R ``deff=TRUE``; Stata ``estat effects``
    when an fpc is declared): ``svyvar * (N - n) / (N n)`` with N = sum w;
    with replacement (``mode="replace"``, R ``deff="replace"``; Stata
    ``estat effects`` without fpc): ``svyvar / n``.  Totals multiply by N^2.
    """
    n = float(np.sum(weights != 0))
    N = float(weights.sum())
    wbar = values.T @ weights / N
    svyvar = ((values - wbar) ** 2).T @ weights / N * n / (n - 1)
    if mode == "replace":
        v = svyvar / n
    elif mode == "wor":
        if N < n:
            warnings.warn(
                "Sample size greater than population size (sum of weights): "
                "are weights correctly scaled? DEFF set to NaN.",
                UserWarning,
                stacklevel=3,
            )
            return np.full(values.shape[1], np.nan)
        v = svyvar * (N - n) / (N * n)
    else:
        raise ValueError(f"deff must be 'wor' or 'replace'; got {mode!r}")
    return v * N**2 if total else v


# ====================================================================== #
#  Public API
# ====================================================================== #


def svymean(
    variables: Union[str, List[str]],
    design: "SurveyDesign",
    alpha: float = 0.05,
    deff: str = "wor",
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
        CIs use the t distribution with the design df (#PSU - #strata), as
        Stata ``svy:`` and R ``confint(..., df = degf(design))``; R's
        ``confint.svystat`` default is the normal quantile.
    deff : {"wor", "replace"}
        Denominator of the design effect: SRS without replacement with
        N = sum of weights (R ``deff=TRUE``; Stata ``estat effects`` when an
        fpc is declared) or with replacement (R ``deff="replace"``; Stata
        ``estat effects`` without fpc).

    Returns
    -------
    SurveyResult

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> df = pd.DataFrame({
    ...     "income": rng.normal(50, 10, n),
    ...     "region": rng.integers(0, 4, n),   # strata
    ...     "psu_id": rng.integers(0, 20, n),  # clusters
    ...     "pw": rng.uniform(1.0, 3.0, n),    # sampling weights
    ... })
    >>> design = sp.svydesign(data=df, weights="pw", strata="region",
    ...                       cluster="psu_id", nest=True)
    >>> res = sp.svymean("income", design)
    >>> list(res.estimate.index)
    ['income']
    """
    var_names, vals = _resolve_vars(variables, design.data)
    w = design.weights
    w_sum = w.sum()

    # Point estimates
    means = np.array([np.average(vals[:, j], weights=w) for j in range(len(var_names))])

    # Linearised scores for the mean: z_i = w_i * (y_i - mean) / sum(w)
    scores = w[:, None] * (vals - means[None, :]) / w_sum

    design_var = np.diag(_design_vcov(scores, design))
    se = np.sqrt(design_var)

    deff_v = design_var / _deff_denominator(vals, w, deff, total=False)

    dof = _design_dof(design)
    t_crit = sp_stats.t.ppf(1 - alpha / 2, df=dof)

    return SurveyResult(
        estimate=pd.Series(means, index=var_names),
        std_error=pd.Series(se, index=var_names),
        ci_lower=pd.Series(means - t_crit * se, index=var_names),
        ci_upper=pd.Series(means + t_crit * se, index=var_names),
        deff=pd.Series(deff_v, index=var_names),
        dof=dof,
        alpha=alpha,
    )


def svytotal(
    variables: Union[str, List[str]],
    design: "SurveyDesign",
    alpha: float = 0.05,
    deff: str = "wor",
) -> SurveyResult:
    """
    Survey-weighted total with design-corrected standard errors.

    Parameters
    ----------
    variables : str or list of str
    design : SurveyDesign
    alpha : float
        CIs use the t distribution with the design df (#PSU - #strata), as
        Stata ``svy:`` and R ``confint(..., df = degf(design))``; R's
        ``confint.svystat`` default is the normal quantile.
    deff : {"wor", "replace"}
        Denominator of the design effect: SRS without replacement with
        N = sum of weights (R ``deff=TRUE``; Stata ``estat effects`` when an
        fpc is declared) or with replacement (R ``deff="replace"``; Stata
        ``estat effects`` without fpc).

    Returns
    -------
    SurveyResult

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> df = pd.DataFrame({
    ...     "income": rng.normal(50, 10, n),
    ...     "region": rng.integers(0, 4, n),   # strata
    ...     "psu_id": rng.integers(0, 20, n),  # clusters
    ...     "pw": rng.uniform(1.0, 3.0, n),    # sampling weights
    ... })
    >>> design = sp.svydesign(data=df, weights="pw", strata="region",
    ...                       cluster="psu_id", nest=True)
    >>> res = sp.svytotal("income", design)
    >>> list(res.estimate.index)
    ['income']
    """
    var_names, vals = _resolve_vars(variables, design.data)
    w = design.weights

    totals = np.array([(w * vals[:, j]).sum() for j in range(len(var_names))])

    # Linearised scores for total: z_i = w_i * y_i
    scores = w[:, None] * vals

    design_var = np.diag(_design_vcov(scores, design))
    se = np.sqrt(design_var)

    deff_v = design_var / _deff_denominator(vals, w, deff, total=True)

    dof = _design_dof(design)
    t_crit = sp_stats.t.ppf(1 - alpha / 2, df=dof)

    return SurveyResult(
        estimate=pd.Series(totals, index=var_names),
        std_error=pd.Series(se, index=var_names),
        ci_lower=pd.Series(totals - t_crit * se, index=var_names),
        ci_upper=pd.Series(totals + t_crit * se, index=var_names),
        deff=pd.Series(deff_v, index=var_names),
        dof=dof,
        alpha=alpha,
    )


def svyglm(
    formula: str,
    design: "SurveyDesign",
    family: str = "gaussian",
    alpha: float = 0.05,
    dof: str = "design",
) -> SurveyResult:
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
        Canonical links; the sandwich is ``A^{-1} B A^{-1}`` with
        ``A = X' diag(w V(mu)) X`` and ``B`` the design covariance of the
        score totals ``w (y - mu) x`` (R ``survey:::svy.varcoef``).
    alpha : float
    dof : {"design", "residual"}
        Degrees of freedom for t statistics, p-values and CIs.
        ``"design"`` (default): #PSU - #strata, as Stata ``svy:``.
        ``"residual"``: #PSU - #strata + 1 - #coefficients, as R
        ``summary.svyglm`` / ``confint.svyglm``.

    Returns
    -------
    SurveyResult with regression coefficient estimates.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> df = pd.DataFrame({
    ...     "income": rng.normal(50, 10, n),
    ...     "age": rng.integers(20, 65, n),
    ...     "region": rng.integers(0, 4, n),   # strata
    ...     "psu_id": rng.integers(0, 20, n),  # clusters
    ...     "pw": rng.uniform(1.0, 3.0, n),    # sampling weights
    ... })
    >>> design = sp.svydesign(data=df, weights="pw", strata="region",
    ...                       cluster="psu_id", nest=True)
    >>> res = sp.svyglm("income ~ age", design)
    >>> list(res.estimate.index)
    ['Intercept', 'age']
    """
    from patsy import dmatrices

    from ..core.utils import _coerce_string_extension_dtypes

    # pandas >= 3.0 string columns are StringDtype, which patsy cannot sniff.
    _data = _coerce_string_extension_dtypes(design.data)
    y_df, X_df = dmatrices(formula, data=_data, return_type="dataframe")
    y = y_df.values.ravel()
    X = X_df.values
    w = design.weights
    var_names = list(X_df.columns)
    n, k = X.shape

    if family == "gaussian":
        params, working_residuals = _wls_fit(y, X, w)
        var_mu = np.ones(n)
    elif family in ("binomial", "poisson"):
        params, working_residuals, var_mu = _irls_fit(y, X, w, family=family)
    else:
        raise ValueError(
            f"Unknown family: {family}. Use 'gaussian', 'binomial', or 'poisson'."
        )

    # Sandwich (R survey:::svy.varcoef): bread = (X' diag(w V(mu)) X)^{-1},
    # score contributions z_i = w_i (y_i - mu_i) x_i (canonical link).
    scores = w[:, None] * working_residuals[:, None] * X
    bread = np.linalg.inv((X * (w * var_mu)[:, None]).T @ X)
    vcov = bread @ _design_vcov(scores, design) @ bread
    se = np.sqrt(np.diag(vcov))

    dof_v = _design_dof(design)
    if dof == "residual":
        dof_v = dof_v + 1 - k
    elif dof != "design":
        raise ValueError(f"dof must be 'design' or 'residual'; got {dof!r}")
    t_crit = sp_stats.t.ppf(1 - alpha / 2, df=dof_v)
    estimates = pd.Series(params, index=var_names)
    se_s = pd.Series(se, index=var_names)

    return SurveyResult(
        estimate=estimates,
        std_error=se_s,
        ci_lower=estimates - t_crit * se_s,
        ci_upper=estimates + t_crit * se_s,
        deff=pd.Series(np.ones(k), index=var_names),  # DEFF not standard for regression
        dof=dof_v,
        alpha=alpha,
    )


# ====================================================================== #
#  Internal fitting helpers
# ====================================================================== #


def _wls_fit(
    y: np.ndarray,
    X: np.ndarray,
    w: np.ndarray,
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
    max_iter: int = 100,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Iteratively Re-weighted Least Squares for canonical-link GLM families.

    Returns ``(params, y - mu, V(mu))`` at the solution; ``V(mu)`` is the
    variance function, which for a canonical link is also d mu / d eta.
    Converges on the relative change in the weighted deviance (R
    ``glm.control`` criterion) at a tighter default tolerance.
    """
    n, k = X.shape
    beta = np.zeros(k)

    def _mu_var(eta):
        if family == "binomial":
            mu = 1 / (1 + np.exp(-eta))
            mu = np.clip(mu, 1e-10, 1 - 1e-10)
            return mu, mu * (1 - mu)
        if family == "poisson":
            mu = np.exp(np.clip(eta, -20, 20))
            return mu, mu
        raise ValueError(f"Unknown family: {family}")

    def _dev(mu):
        if family == "binomial":
            return -2 * np.sum(w * (y * np.log(mu) + (1 - y) * np.log(1 - mu)))
        with np.errstate(divide="ignore", invalid="ignore"):
            ylog = np.where(y > 0, y * np.log(y / mu), 0.0)
        return 2 * np.sum(w * (ylog - (y - mu)))

    if family == "binomial":
        mu = (w * y + 0.5) / (w + 1)
        eta = np.log(mu / (1 - mu))
    else:
        mu = y + 0.1
        eta = np.log(mu)
    dev_old = np.inf
    for _ in range(max_iter):
        mu, var_mu = _mu_var(eta)
        z = eta + (y - mu) / var_mu
        W = np.sqrt(w * var_mu)
        beta = np.linalg.lstsq(X * W[:, None], z * W, rcond=None)[0]
        eta = X @ beta
        dev = _dev(_mu_var(eta)[0])
        if abs(dev - dev_old) / (abs(dev) + 0.1) < tol:
            break
        dev_old = dev

    mu, var_mu = _mu_var(X @ beta)
    return beta, y - mu, var_mu
