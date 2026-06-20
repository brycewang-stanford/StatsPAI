"""
Propensity score diagnostics for matching and IPW analyses.

Comprehensive toolkit for assessing overlap, balance, and propensity score
quality — the essential diagnostics that existing Python packages handle poorly.

Functions
---------
propensity_score : Estimate propensity scores via logit, probit, or GBM.
overlap_plot     : Mirrored density plot of PS distributions by treatment.
trimming         : Crump et al. (2009) optimal PS trimming.
love_plot        : Dot plot of standardized mean differences (before/after).
ps_balance       : Comprehensive balance table with SMD, KS, variance ratios.

Classes
-------
PSBalanceResult  : Container for balance diagnostics with summary/plot methods.
BalanceDiagnosticsResult : Unified raw/weighted balance diagnostics.

References
----------
Crump, R.K., Hotz, V.J., Imbens, G.W. and Mitnik, O.A. (2009).
    "Dealing with limited overlap in estimation of average treatment effects."
    Biometrika, 96(1), 187-199.
Rosenbaum, P.R. and Rubin, D.B. (1983). Biometrika, 70(1), 41-55.
Austin, P.C. (2011). Multivariate Behavioral Research, 46(3), 399-424. [@crump2009dealing]
"""

from typing import Any, Dict, Optional, List, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats, optimize

# ======================================================================
# Propensity score estimation
# ======================================================================


def propensity_score(
    data: pd.DataFrame,
    treatment: str,
    covariates: List[str],
    method: str = "logit",
    trimming: Optional[str] = None,
) -> pd.Series:
    """Estimate propensity scores P(D=1|X).

    Parameters
    ----------
    data : DataFrame
        Input data.
    treatment : str
        Name of binary treatment column (0/1).
    covariates : list of str
        Covariate column names.
    method : {'logit', 'probit', 'gbm'}
        Estimation method.  ``'logit'`` uses IRLS (no sklearn needed).
        ``'probit'`` uses scipy.optimize.  ``'gbm'`` tries sklearn
        GradientBoostingClassifier, falling back to logit with interactions.
    trimming : {None, 'crump'}
        If ``'crump'``, apply Crump et al. (2009) trimming after estimation.
        Trimmed observations receive ``NaN`` scores.

    Returns
    -------
    pd.Series
        Propensity scores indexed like *data*.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.cps_wage()
    >>> ps = sp.propensity_score(df, treatment='union',
    ...                          covariates=['education', 'experience',
    ...                                      'tenure'])
    >>> round(float(ps.mean()), 3)  # matches the union share of 0.177
    0.177

    GBM scores with Crump trimming — poorly overlapping observations
    receive ``NaN``:

    >>> ps_trim = sp.propensity_score(df, treatment='union',
    ...                               covariates=['education', 'experience',
    ...                                           'tenure'],
    ...                               method='gbm', trimming='crump')

    Typical diagnostics flow afterwards: pass the scores (or derived IPW
    weights) to :func:`sp.overlap_plot`, :func:`sp.ps_balance`, or
    :func:`sp.balance_diagnostics`.
    """
    D = data[treatment].values.astype(float)
    X = data[covariates].values.astype(float)

    if method == "logit":
        ps = _logit_irls(X, D)
    elif method == "probit":
        ps = _probit_mle(X, D)
    elif method == "gbm":
        ps = _gbm_ps(X, D)
    else:
        raise ValueError(f"method must be 'logit', 'probit', or 'gbm', got '{method}'")

    # Clip to avoid exact 0/1
    ps = np.clip(ps, 1e-8, 1 - 1e-8)
    ps_series = pd.Series(ps, index=data.index, name="propensity_score")

    if trimming == "crump":
        alpha = _crump_alpha(ps)
        mask = (ps < alpha) | (ps > 1 - alpha)
        ps_series[mask] = np.nan

    return ps_series


def _logit_irls(
    X: np.ndarray, D: np.ndarray, max_iter: int = 50, tol: float = 1e-8
) -> np.ndarray:
    """Logistic regression via iteratively reweighted least squares."""
    n, k = X.shape
    Xa = np.column_stack([np.ones(n), X])
    beta = np.zeros(Xa.shape[1])

    for _ in range(max_iter):
        eta = Xa @ beta
        eta = np.clip(eta, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        W = mu * (1 - mu)
        W = np.maximum(W, 1e-12)
        z = eta + (D - mu) / W
        # Weighted least squares step
        XtW = Xa.T * W
        try:
            beta_new = np.linalg.solve(XtW @ Xa, XtW @ z)
        except np.linalg.LinAlgError:
            beta_new = np.linalg.lstsq(XtW @ Xa, XtW @ z, rcond=None)[0]
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    eta = Xa @ beta
    eta = np.clip(eta, -30, 30)
    return 1.0 / (1.0 + np.exp(-eta))


def _probit_mle(X: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Probit regression via maximum likelihood (scipy.optimize)."""
    n, k = X.shape
    Xa = np.column_stack([np.ones(n), X])

    def neg_loglik(beta: np.ndarray) -> float:
        eta = Xa @ beta
        eta = np.clip(eta, -30, 30)
        Phi = stats.norm.cdf(eta)
        Phi = np.clip(Phi, 1e-12, 1 - 1e-12)
        return float(-np.sum(D * np.log(Phi) + (1 - D) * np.log(1 - Phi)))

    def grad(beta: np.ndarray) -> np.ndarray:
        eta = Xa @ beta
        eta = np.clip(eta, -30, 30)
        Phi = stats.norm.cdf(eta)
        Phi = np.clip(Phi, 1e-12, 1 - 1e-12)
        phi = stats.norm.pdf(eta)
        lam = D * phi / Phi - (1 - D) * phi / (1 - Phi)
        return np.asarray(-Xa.T @ lam, dtype=float)

    beta0 = np.zeros(Xa.shape[1])
    result = optimize.minimize(
        neg_loglik, beta0, jac=grad, method="BFGS", options={"maxiter": 200}
    )
    eta = Xa @ result.x
    eta = np.clip(eta, -30, 30)
    return np.asarray(stats.norm.cdf(eta), dtype=float)


def _gbm_ps(X: np.ndarray, D: np.ndarray) -> np.ndarray:
    """GBM propensity scores via sklearn, fallback to logit + interactions."""
    try:
        from sklearn.ensemble import GradientBoostingClassifier

        gbm = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )
        gbm.fit(X, D)
        return np.asarray(gbm.predict_proba(X)[:, 1], dtype=float)
    except ImportError:
        import warnings

        warnings.warn(
            "sklearn not available for GBM; falling back to logit with "
            "pairwise interactions.",
            stacklevel=3,
        )
        # Build interaction terms
        from itertools import combinations

        interactions = []
        for i, j in combinations(range(X.shape[1]), 2):
            interactions.append(X[:, i] * X[:, j])
        if interactions:
            X_aug = np.column_stack([X] + interactions)
        else:
            X_aug = X
        return _logit_irls(X_aug, D)


# ======================================================================
# Crump et al. (2009) trimming
# ======================================================================


def _crump_alpha(ps: np.ndarray) -> float:
    """Find optimal Crump trimming threshold alpha.

    Solves: alpha = 1 / (2 * E[1/(e(1-e)) * I(alpha <= e <= 1-alpha)])
    via grid search on [0, 0.5).
    """
    ps_clean = ps[np.isfinite(ps)]
    inv_var = 1.0 / (ps_clean * (1 - ps_clean))

    best_alpha = 0.0
    alphas = np.linspace(0, 0.49, 500)
    for a in alphas:
        mask = (ps_clean >= a) & (ps_clean <= 1 - a)
        if mask.sum() < 2:
            break
        rhs = 1.0 / (2.0 * np.mean(inv_var[mask]))
        if a <= rhs:
            best_alpha = a
        else:
            break
    return best_alpha


def trimming(
    data: pd.DataFrame,
    treatment: str,
    covariates: List[str],
    method: str = "crump",
    ps: Optional[pd.Series] = None,
    ps_method: str = "logit",
) -> pd.DataFrame:
    """Trim sample to optimal overlap region.

    Parameters
    ----------
    data : DataFrame
        Input data.
    treatment : str
        Binary treatment column.
    covariates : list of str
        Covariates for PS estimation (if *ps* not supplied).
    method : {'crump', 'sturmer'}
        ``'crump'`` uses Crump et al. (2009) optimal rule.
        ``'sturmer'`` trims at the fixed [0.1, 0.9] interval.
    ps : Series, optional
        Pre-estimated propensity scores.  If None, estimated via *ps_method*.
    ps_method : str
        Method for PS estimation if *ps* is None.

    Returns
    -------
    DataFrame
        Trimmed data (rows with PS in the overlap region).

    Examples
    --------
    Strong selection on covariates creates limited overlap:

    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 300
    >>> x1 = rng.normal(size=n)
    >>> x2 = rng.normal(size=n)
    >>> p = 1.0 / (1.0 + np.exp(-(2.0 * x1 - 2.0 * x2)))
    >>> d = rng.binomial(1, p)
    >>> df = pd.DataFrame({'d': d, 'x1': x1, 'x2': x2})

    Crump et al. (2009) optimal rule drops poor-overlap rows:

    >>> trimmed = sp.trimming(df, treatment='d',
    ...                       covariates=['x1', 'x2'])
    >>> (len(df), len(trimmed))
    (300, 206)

    Fixed [0.1, 0.9] trimming keeps a narrower sample:

    >>> trimmed_s = sp.trimming(df, treatment='d',
    ...                         covariates=['x1', 'x2'],
    ...                         method='sturmer')
    >>> len(trimmed_s)
    180
    """
    if ps is None:
        ps = propensity_score(data, treatment, covariates, method=ps_method)

    ps_vals = ps.values

    if method == "crump":
        alpha = _crump_alpha(ps_vals[np.isfinite(ps_vals)])
    elif method == "sturmer":
        alpha = 0.1
    else:
        raise ValueError(f"method must be 'crump' or 'sturmer', got '{method}'")

    mask = (ps_vals >= alpha) & (ps_vals <= 1 - alpha) & np.isfinite(ps_vals)
    return data.loc[mask].copy()


# ======================================================================
# Standardized mean difference & balance utilities
# ======================================================================


def _smd(
    x_t: np.ndarray,
    x_c: np.ndarray,
    w_t: Optional[np.ndarray] = None,
    w_c: Optional[np.ndarray] = None,
) -> float:
    """Standardized mean difference (Austin 2011 formula)."""
    if w_t is not None:
        mean_t = np.average(x_t, weights=w_t)
        var_t = np.average((x_t - mean_t) ** 2, weights=w_t)
    else:
        mean_t = np.mean(x_t)
        var_t = np.var(x_t, ddof=1) if len(x_t) > 1 else 0.0

    if w_c is not None:
        mean_c = np.average(x_c, weights=w_c)
        var_c = np.average((x_c - mean_c) ** 2, weights=w_c)
    else:
        mean_c = np.mean(x_c)
        var_c = np.var(x_c, ddof=1) if len(x_c) > 1 else 0.0

    denom = np.sqrt((var_t + var_c) / 2.0)
    if denom < 1e-12:
        return 0.0
    return float((mean_t - mean_c) / denom)


def _variance_ratio(
    x_t: np.ndarray,
    x_c: np.ndarray,
    w_t: Optional[np.ndarray] = None,
    w_c: Optional[np.ndarray] = None,
) -> float:
    """Variance ratio (treated / control)."""
    if w_t is not None:
        mean_t = np.average(x_t, weights=w_t)
        var_t = np.average((x_t - mean_t) ** 2, weights=w_t)
    else:
        var_t = np.var(x_t, ddof=1) if len(x_t) > 1 else 0.0

    if w_c is not None:
        mean_c = np.average(x_c, weights=w_c)
        var_c = np.average((x_c - mean_c) ** 2, weights=w_c)
    else:
        var_c = np.var(x_c, ddof=1) if len(x_c) > 1 else 0.0

    if var_c < 1e-12:
        return np.inf if var_t > 1e-12 else 1.0
    return float(var_t / var_c)


def _ks_stat(
    x_t: np.ndarray,
    x_c: np.ndarray,
    w_t: Optional[np.ndarray] = None,
    w_c: Optional[np.ndarray] = None,
) -> float:
    """Kolmogorov-Smirnov statistic (unweighted or weighted)."""
    if w_t is None and w_c is None:
        return float(stats.ks_2samp(x_t, x_c).statistic)

    # Weighted KS
    all_vals = np.sort(np.concatenate([x_t, x_c]))
    if w_t is None:
        w_t = np.ones(len(x_t))
    if w_c is None:
        w_c = np.ones(len(x_c))

    # Approximate weighted CDF via order-based approach
    idx_t = np.argsort(x_t)
    idx_c = np.argsort(x_c)
    cum_w_t = np.cumsum(w_t[idx_t])
    cum_w_c = np.cumsum(w_c[idx_c])
    ecdf_t = np.interp(all_vals, x_t[idx_t], cum_w_t / cum_w_t[-1], left=0, right=1)
    ecdf_c = np.interp(all_vals, x_c[idx_c], cum_w_c / cum_w_c[-1], left=0, right=1)
    return float(np.max(np.abs(ecdf_t - ecdf_c)))


# ======================================================================
# PSBalanceResult
# ======================================================================


class PSBalanceResult:
    """Container for propensity score balance diagnostics.

    Attributes
    ----------
    table : DataFrame
        Balance statistics per covariate: mean_treat, mean_control,
        smd_raw, smd_weighted, variance_ratio, ks_stat.
    ps : Series
        Estimated propensity scores.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 300
    >>> x1 = rng.normal(size=n)
    >>> x2 = rng.normal(size=n)
    >>> p = 1.0 / (1.0 + np.exp(-(0.5 * x1 - 0.5 * x2 - 0.5)))
    >>> d = rng.binomial(1, p)
    >>> df = pd.DataFrame({'d': d, 'x1': x1, 'x2': x2})
    >>> bal = sp.ps_balance(df, treatment='d', covariates=['x1', 'x2'])
    >>> isinstance(bal, sp.PSBalanceResult)
    True
    >>> bal.table['smd_weighted'].round(2).tolist()
    [0.02, -0.06]
    """

    def __init__(self, table: pd.DataFrame, ps: pd.Series) -> None:
        self.table = table
        self.ps = ps

    def summary(self) -> str:
        """Formatted balance summary table."""
        lines = []
        lines.append("Propensity Score Balance Diagnostics")
        lines.append("=" * 70)
        lines.append("")

        fmt = table_to_string(self.table)
        lines.append(fmt)

        lines.append("")
        n_imbalanced_raw = (self.table["smd_raw"].abs() > 0.1).sum()
        n_imbalanced_wtd = (self.table["smd_weighted"].abs() > 0.1).sum()
        lines.append(
            f"Covariates with |SMD| > 0.1: {n_imbalanced_raw} (raw)"
            f" -> {n_imbalanced_wtd} (weighted)"
        )
        return "\n".join(lines)

    def love_plot(self, threshold: float = 0.1, **kwargs: Any) -> Tuple[Any, Any]:
        """Convenience method: calls ``love_plot()`` from balance data."""
        return _love_plot_from_table(self.table, threshold=threshold, **kwargs)

    def _repr_html_(self) -> str:
        """Rich HTML display for Jupyter notebooks."""
        html = "<h4>Propensity Score Balance Diagnostics</h4>"
        html += str(self.table.to_html(float_format="{:.4f}".format))
        n_raw = (self.table["smd_raw"].abs() > 0.1).sum()
        n_wtd = (self.table["smd_weighted"].abs() > 0.1).sum()
        html += (
            f"<p>Covariates with |SMD| &gt; 0.1: {n_raw} (raw) "
            f"&rarr; {n_wtd} (weighted)</p>"
        )
        return html

    def __repr__(self) -> str:
        return self.summary()


class BalanceDiagnosticsResult:
    """Container for raw/weighted matching balance diagnostics.

    Returned by :func:`sp.balance_diagnostics`. Holds a per-covariate
    ``table`` (raw vs. weighted SMDs, variance ratios, KS stats) and a
    ``summary_stats`` dict (max/mean SMDs, imbalance counts, effective
    sample size, propensity overlap). Call ``.summary()`` for a report.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.cps_wage()
    >>> bal = sp.balance_diagnostics(
    ...     df, treatment='union',
    ...     covariates=['education', 'experience', 'tenure'])
    >>> isinstance(bal, sp.BalanceDiagnosticsResult)
    True
    >>> bal.summary_stats['n_obs']
    3000
    """

    def __init__(
        self,
        table: pd.DataFrame,
        summary: Dict[str, Any],
        ps: Optional[pd.Series] = None,
        weights: Optional[pd.Series] = None,
    ) -> None:
        self.table = table
        self.summary_stats = summary
        self.ps = ps
        self.weights = weights

    def summary(self) -> str:
        lines = [
            "Balance Diagnostics",
            "=" * 70,
            self.table.to_string(float_format=lambda x: f"{x:.4f}"),
            "",
        ]
        s = self.summary_stats
        lines.append(
            "Max |SMD| raw -> weighted: "
            f"{s.get('max_abs_smd_raw', np.nan):.4f} -> "
            f"{s.get('max_abs_smd_weighted', np.nan):.4f}"
        )
        lines.append(
            "Covariates above threshold: "
            f"{s.get('n_imbalanced_raw', 0)} -> "
            f"{s.get('n_imbalanced_weighted', 0)}"
        )
        if "effective_sample_size" in s:
            lines.append(f"Effective sample size: {s['effective_sample_size']:.2f}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table": self.table.to_dict(orient="index"),
            "summary": dict(self.summary_stats),
        }

    def __repr__(self) -> str:
        return self.summary()


def table_to_string(df: pd.DataFrame) -> str:
    """Format a DataFrame as an aligned text table."""
    return str(df.to_string(float_format=lambda x: f"{x:.4f}"))


# ======================================================================
# ps_balance — comprehensive balance table
# ======================================================================


def ps_balance(
    data: pd.DataFrame,
    treatment: str,
    covariates: List[str],
    weights: Optional[Union[np.ndarray, pd.Series]] = None,
    method: str = "logit",
) -> PSBalanceResult:
    """Compute comprehensive propensity score balance table.

    Parameters
    ----------
    data : DataFrame
        Input data.
    treatment : str
        Binary treatment column.
    covariates : list of str
        Covariate columns to assess balance for.
    weights : array-like, optional
        IPW or matching weights.  If None, inverse-PS weights are
        computed automatically from estimated propensity scores.
    method : str
        PS estimation method ('logit', 'probit', 'gbm').

    Returns
    -------
    PSBalanceResult
        Object with ``.table``, ``.ps``, ``.summary()``, ``.love_plot()``.

    Examples
    --------
    Simulated data with confounded treatment assignment:

    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 300
    >>> x1 = rng.normal(size=n)
    >>> x2 = rng.normal(size=n)
    >>> p = 1.0 / (1.0 + np.exp(-(0.5 * x1 - 0.5 * x2 - 0.5)))
    >>> d = rng.binomial(1, p)
    >>> df = pd.DataFrame({'d': d, 'x1': x1, 'x2': x2})

    Without ``weights``, ATE inverse-propensity weights are computed
    from the estimated propensity scores:

    >>> bal = sp.ps_balance(df, treatment='d', covariates=['x1', 'x2'])
    >>> bal.table['smd_raw'].round(2).tolist()
    [0.68, -0.45]
    >>> bal.table['smd_weighted'].round(2).tolist()
    [0.02, -0.06]
    >>> fig, ax = bal.love_plot()  # doctest: +SKIP
    """
    D = data[treatment].values.astype(float)
    ps = propensity_score(data, treatment, covariates, method=method)
    ps_vals = ps.values

    # Compute IPW weights if not supplied
    if weights is not None:
        w = np.asarray(weights, dtype=float)
    else:
        # ATE weights: 1/e for treated, 1/(1-e) for control
        w = np.where(D == 1, 1.0 / ps_vals, 1.0 / (1.0 - ps_vals))

    treat_mask = D == 1
    ctrl_mask = D == 0

    rows = []
    for cov in covariates:
        x = data[cov].values.astype(float)
        x_t = x[treat_mask]
        x_c = x[ctrl_mask]
        w_t = w[treat_mask]
        w_c = w[ctrl_mask]

        mean_t = np.mean(x_t)
        mean_c = np.mean(x_c)
        smd_raw = _smd(x_t, x_c)
        smd_weighted = _smd(x_t, x_c, w_t, w_c)
        vr = _variance_ratio(x_t, x_c, w_t, w_c)
        ks = _ks_stat(x_t, x_c, w_t, w_c)

        rows.append(
            {
                "variable": cov,
                "mean_treat": mean_t,
                "mean_control": mean_c,
                "smd_raw": smd_raw,
                "smd_weighted": smd_weighted,
                "variance_ratio": vr,
                "ks_stat": ks,
            }
        )

    table = pd.DataFrame(rows).set_index("variable")
    return PSBalanceResult(table=table, ps=ps)


def balance_diagnostics(
    data: pd.DataFrame,
    treatment: str,
    covariates: List[str],
    weights: Optional[Union[np.ndarray, pd.Series, str]] = None,
    ps: Optional[Union[np.ndarray, pd.Series, str]] = None,
    method: str = "logit",
    threshold: float = 0.1,
) -> BalanceDiagnosticsResult:
    """Unified balance diagnostics for matching and weighting estimators.

    Parameters
    ----------
    data : DataFrame
        Analysis frame.
    treatment : str
        Binary treatment indicator.
    covariates : list of str
        Covariates to audit.
    weights : array-like or str, optional
        Observation weights after matching/weighting. If omitted, ATE
        inverse-propensity weights are computed from ``ps``.
    ps : array-like or str, optional
        Propensity scores. If omitted, estimated with ``method``.
    method : {'logit', 'probit', 'gbm'}, default 'logit'
        Propensity-score model when ``ps`` is not supplied.
    threshold : float, default 0.1
        Balance threshold for absolute standardized mean differences.

    Returns
    -------
    BalanceDiagnosticsResult
        ``.table`` has one row per covariate; ``.summary_stats`` records
        max/mean SMDs, imbalance counts, effective sample size, and
        propensity-score overlap.

    Examples
    --------
    With no ``weights``, ATE inverse-propensity weights are computed from
    the estimated propensity scores:

    >>> import statspai as sp
    >>> df = sp.cps_wage()
    >>> bal = sp.balance_diagnostics(
    ...     df, treatment='union',
    ...     covariates=['education', 'experience', 'tenure'])
    >>> bal.summary_stats['n_obs']
    3000
    >>> bool(bal.summary_stats['n_imbalanced_weighted']
    ...      <= bal.summary_stats['n_imbalanced_raw'])
    True

    Typical post-estimation flow — audit your own weights and scores:

    >>> import numpy as np
    >>> ps = sp.propensity_score(
    ...     df, 'union', ['education', 'experience', 'tenure'])
    >>> w = np.where(df['union'] == 1, 1 / ps, 1 / (1 - ps))
    >>> bal = sp.balance_diagnostics(
    ...     df, treatment='union',
    ...     covariates=['education', 'experience', 'tenure'],
    ...     weights=w, ps=ps)
    >>> bool(bal.summary_stats['effective_sample_size'] > 0)
    True
    """
    cols = [treatment] + list(covariates)
    df = data[cols].dropna().copy()
    if df.empty:
        raise ValueError("No complete observations for balance diagnostics.")

    D = df[treatment].to_numpy(dtype=float)
    if not set(np.unique(D)).issubset({0, 1}):
        raise ValueError(f"Treatment '{treatment}' must be binary.")

    ps_series = _coerce_vector(ps, data, df.index, "propensity_score")
    if ps_series is None:
        ps_series = propensity_score(df, treatment, covariates, method=method)
    ps_vals = np.clip(ps_series.to_numpy(dtype=float), 1e-8, 1 - 1e-8)

    w_series = _coerce_vector(weights, data, df.index, "weights")
    if w_series is None:
        w_vals = np.where(D == 1, 1.0 / ps_vals, 1.0 / (1.0 - ps_vals))
        w_series = pd.Series(w_vals, index=df.index, name="weights")
    else:
        w_vals = w_series.to_numpy(dtype=float)

    treat_mask = D == 1
    ctrl_mask = D == 0
    rows = []
    for cov in covariates:
        x = df[cov].to_numpy(dtype=float)
        x_t, x_c = x[treat_mask], x[ctrl_mask]
        w_t, w_c = w_vals[treat_mask], w_vals[ctrl_mask]
        smd_raw = _smd(x_t, x_c)
        smd_weighted = _smd(x_t, x_c, w_t, w_c)
        rows.append(
            {
                "variable": cov,
                "mean_treat": float(np.mean(x_t)),
                "mean_control": float(np.mean(x_c)),
                "weighted_mean_treat": float(np.average(x_t, weights=w_t)),
                "weighted_mean_control": float(np.average(x_c, weights=w_c)),
                "smd_raw": float(smd_raw),
                "smd_weighted": float(smd_weighted),
                "variance_ratio_weighted": float(_variance_ratio(x_t, x_c, w_t, w_c)),
                "ks_stat_weighted": float(_ks_stat(x_t, x_c, w_t, w_c)),
                "balanced": bool(abs(smd_weighted) <= threshold),
            }
        )

    table = pd.DataFrame(rows).set_index("variable")
    abs_raw = table["smd_raw"].abs()
    abs_wtd = table["smd_weighted"].abs()
    ess = _effective_sample_size(w_vals)
    ps_t, ps_c = ps_vals[treat_mask], ps_vals[ctrl_mask]
    common_low = float(max(ps_t.min(), ps_c.min()))
    common_high = float(min(ps_t.max(), ps_c.max()))
    summary = {
        "threshold": float(threshold),
        "n_obs": int(len(df)),
        "n_treated": int(treat_mask.sum()),
        "n_control": int(ctrl_mask.sum()),
        "max_abs_smd_raw": float(abs_raw.max()) if len(abs_raw) else np.nan,
        "max_abs_smd_weighted": float(abs_wtd.max()) if len(abs_wtd) else np.nan,
        "mean_abs_smd_raw": float(abs_raw.mean()) if len(abs_raw) else np.nan,
        "mean_abs_smd_weighted": float(abs_wtd.mean()) if len(abs_wtd) else np.nan,
        "n_imbalanced_raw": int((abs_raw > threshold).sum()),
        "n_imbalanced_weighted": int((abs_wtd > threshold).sum()),
        "effective_sample_size": float(ess),
        "pscore_min": float(ps_vals.min()),
        "pscore_max": float(ps_vals.max()),
        "common_support_low": common_low,
        "common_support_high": common_high,
        "common_support_width": max(0.0, common_high - common_low),
    }
    return BalanceDiagnosticsResult(
        table=table,
        summary=summary,
        ps=pd.Series(ps_vals, index=df.index, name="propensity_score"),
        weights=w_series,
    )


def _coerce_vector(
    value: Optional[Union[np.ndarray, pd.Series, str]],
    data: pd.DataFrame,
    index: pd.Index,
    name: str,
) -> Optional[pd.Series]:
    if value is None:
        return None
    if isinstance(value, str):
        if value not in data.columns:
            raise ValueError(f"Column {value!r} not found for {name}.")
        return data.loc[index, value].astype(float).rename(name)
    if isinstance(value, pd.Series):
        return value.reindex(index).astype(float).rename(name)
    arr = np.asarray(value, dtype=float).ravel()
    if len(arr) == len(data):
        return pd.Series(arr, index=data.index, name=name).loc[index]
    if len(arr) == len(index):
        return pd.Series(arr, index=index, name=name)
    raise ValueError(
        f"{name} length must match data ({len(data)}) or complete-case "
        f"rows ({len(index)}); got {len(arr)}."
    )


def _effective_sample_size(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    denom = float(np.sum(w**2))
    if denom <= 0:
        return np.nan
    return float((np.sum(w) ** 2) / denom)


# ======================================================================
# Plotting helpers
# ======================================================================


def _require_matplotlib() -> Any:
    """Import matplotlib or raise a helpful error."""
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install matplotlib"
        )


# ======================================================================
# overlap_plot
# ======================================================================


def overlap_plot(
    data: pd.DataFrame,
    treatment: str,
    covariates: List[str],
    ps: Optional[pd.Series] = None,
    method: str = "logit",
    ax: Any = None,
    figsize: Tuple[float, float] = (8, 4),
    title: str = "Propensity Score Overlap",
) -> Tuple[Any, Any]:
    """Mirrored density plot of propensity scores by treatment group.

    Parameters
    ----------
    data : DataFrame
        Input data.
    treatment : str
        Binary treatment column.
    covariates : list of str
        Covariates for PS estimation (ignored if *ps* supplied).
    ps : Series, optional
        Pre-estimated propensity scores.
    method : str
        PS estimation method if *ps* is None.
    ax : matplotlib Axes, optional
        Axes to plot on.  If None, a new figure is created.
    figsize : tuple
        Figure size (width, height).
    title : str
        Plot title.

    Returns
    -------
    (fig, ax) : tuple
        Matplotlib figure and axes.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 300
    >>> x1 = rng.normal(size=n)
    >>> x2 = rng.normal(size=n)
    >>> p = 1.0 / (1.0 + np.exp(-(0.5 * x1 - 0.5 * x2 - 0.5)))
    >>> d = rng.binomial(1, p)
    >>> df = pd.DataFrame({'d': d, 'x1': x1, 'x2': x2})
    >>> fig, ax = sp.overlap_plot(df, treatment='d',
    ...                           covariates=['x1', 'x2'])
    >>> fig.savefig('overlap.png')  # doctest: +SKIP

    Reuse pre-estimated propensity scores and set a custom title:

    >>> ps = sp.propensity_score(df, treatment='d',
    ...                          covariates=['x1', 'x2'])
    >>> fig, ax = sp.overlap_plot(df, treatment='d',
    ...                           covariates=['x1', 'x2'], ps=ps,
    ...                           title='PS overlap')
    """
    plt = _require_matplotlib()

    if ps is None:
        ps = propensity_score(data, treatment, covariates, method=method)

    D = data[treatment].values.astype(float)
    ps_vals = ps.values

    ps_treat = ps_vals[D == 1]
    ps_ctrl = ps_vals[D == 0]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Kernel density estimation
    grid = np.linspace(0, 1, 300)

    kde_t = stats.gaussian_kde(ps_treat, bw_method="scott")
    kde_c = stats.gaussian_kde(ps_ctrl, bw_method="scott")

    density_t = kde_t(grid)
    density_c = kde_c(grid)

    # Mirrored: treated above, control below
    ax.fill_between(
        grid,
        density_t,
        alpha=0.35,
        color="#2171b5",
        label=f"Treated (n={len(ps_treat)})",
    )
    ax.fill_between(
        grid,
        -density_c,
        alpha=0.35,
        color="#cb181d",
        label=f"Control (n={len(ps_ctrl)})",
    )
    ax.plot(grid, density_t, color="#2171b5", linewidth=1.2)
    ax.plot(grid, -density_c, color="#cb181d", linewidth=1.2)

    # Common support region
    cs_low = max(ps_treat.min(), ps_ctrl.min())
    cs_high = min(ps_treat.max(), ps_ctrl.max())
    ax.axvspan(
        cs_low,
        cs_high,
        alpha=0.08,
        color="green",
        label=f"Common support [{cs_low:.2f}, {cs_high:.2f}]",
    )

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Propensity Score")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    ax.set_xlim(0, 1)

    fig.tight_layout()
    return fig, ax


# ======================================================================
# love_plot
# ======================================================================


def love_plot(
    data: pd.DataFrame,
    treatment: str,
    covariates: List[str],
    weights: Optional[Union[np.ndarray, pd.Series]] = None,
    threshold: float = 0.1,
    ps_method: str = "logit",
    ax: Any = None,
    figsize: Tuple[float, Optional[float]] = (7, None),
    title: str = "Covariate Balance (Love Plot)",
) -> Tuple[Any, Any]:
    """Love plot: dot plot of standardized mean differences before/after.

    Parameters
    ----------
    data : DataFrame
        Input data.
    treatment : str
        Binary treatment column.
    covariates : list of str
        Covariate columns.
    weights : array-like, optional
        IPW or matching weights.  If None, inverse-PS weights are computed.
    threshold : float
        SMD threshold for the vertical dashed line (default 0.1).
    ps_method : str
        PS estimation method for balance computation.
    ax : matplotlib Axes, optional
    figsize : tuple
        (width, height).  Height defaults to 0.4 * n_covariates + 1.
    title : str
        Plot title.

    Returns
    -------
    (fig, ax) : tuple

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.cps_wage()
    >>> fig, ax = sp.love_plot(df, treatment='union',
    ...                        covariates=['education', 'experience',
    ...                                    'tenure'])
    >>> fig.savefig('love_plot.png')  # doctest: +SKIP

    >>> # Tighter balance threshold and custom title
    >>> fig, ax = sp.love_plot(df, treatment='union',
    ...                        covariates=['education', 'experience',
    ...                                    'tenure'],
    ...                        threshold=0.05,
    ...                        title='Balance: union vs non-union')
    """
    result = ps_balance(data, treatment, covariates, weights=weights, method=ps_method)
    return _love_plot_from_table(
        result.table, threshold=threshold, ax=ax, figsize=figsize, title=title
    )


def _love_plot_from_table(
    table: pd.DataFrame,
    threshold: float = 0.1,
    ax: Any = None,
    figsize: Tuple[float, Optional[float]] = (7, None),
    title: str = "Covariate Balance (Love Plot)",
) -> Tuple[Any, Any]:
    """Internal love plot renderer from a balance table."""
    plt = _require_matplotlib()

    n_vars = len(table)
    if figsize[1] is None:
        figsize = (figsize[0], max(3, 0.4 * n_vars + 1))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    variables = table.index.tolist()
    smd_raw = table["smd_raw"].abs().values
    smd_wtd = table["smd_weighted"].abs().values
    y_pos = np.arange(n_vars)

    # Horizontal connecting lines
    for i in range(n_vars):
        ax.plot(
            [smd_raw[i], smd_wtd[i]],
            [y_pos[i], y_pos[i]],
            color="gray",
            linewidth=0.8,
            zorder=1,
        )

    # Dots
    ax.scatter(
        smd_raw,
        y_pos,
        color="#cb181d",
        s=50,
        zorder=2,
        label="Raw",
        marker="o",
        edgecolors="white",
        linewidth=0.5,
    )
    ax.scatter(
        smd_wtd,
        y_pos,
        color="#2171b5",
        s=50,
        zorder=2,
        label="Weighted",
        marker="s",
        edgecolors="white",
        linewidth=0.5,
    )

    # Threshold line
    ax.axvline(
        threshold,
        color="black",
        linestyle="--",
        linewidth=0.8,
        label=f"Threshold ({threshold})",
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(variables)
    ax.set_xlabel("|Standardized Mean Difference|")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.invert_yaxis()
    ax.set_xlim(left=0)

    fig.tight_layout()
    return fig, ax
