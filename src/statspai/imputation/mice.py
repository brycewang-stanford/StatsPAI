"""
Multiple Imputation by Chained Equations (MICE).

Implements the MICE algorithm (van Buuren & Groothuis-Oudshoorn 2011)
for handling missing data, plus Rubin's rules for combining estimates.

Equivalent to Stata's ``mi impute chained`` and R's ``mice::mice()``.

References
----------
van Buuren, S. & Groothuis-Oudshoorn, K. (2011).
"mice: Multivariate Imputation by Chained Equations in R."
*Journal of Statistical Software*, 45(3), 1-67. [@buuren2011mice]

Rubin, D.B. (1987).
"Multiple Imputation for Nonresponse in Surveys."
*Wiley*. [@rubin1987multiple]
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, MethodIncompatibility


class MICEResult(ResultProtocolMixin):
    """Results from MICE imputation.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(7)
    >>> x1 = rng.normal(size=120)
    >>> x2 = 0.5 * x1 + rng.normal(size=120)
    >>> y = 1.0 + x1 + x2 + rng.normal(size=120)
    >>> df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    >>> df.loc[rng.choice(120, size=15, replace=False), "x2"] = np.nan
    >>> res = sp.mice(df, m=3, method="pmm", seed=0)
    >>> type(res).__name__
    'MICEResult'
    >>> res.n_imputations
    3
    >>> bool(res.complete(0)["x2"].notna().all())
    True
    """

    def __init__(
        self,
        imputed_datasets: Any,
        n_imputations: Any,
        n_obs: Any,
        n_missing: Any,
        variables_imputed: Any,
        methods: Any,
        convergence: Any,
    ) -> None:
        self.imputed_datasets = imputed_datasets
        self.n_imputations = n_imputations
        self.n_obs = n_obs
        self.n_missing = n_missing  # dict: var -> count
        self.variables_imputed = variables_imputed
        self.methods = methods  # dict: var -> method used
        self.convergence = convergence
        self.fit_failures: List[Dict[str, Any]] = []

    def summary(self) -> str:
        lines = [
            "Multiple Imputation by Chained Equations (MICE)",
            "=" * 55,
            f"Imputations: {self.n_imputations}",
            f"Observations: {self.n_obs}",
            "",
            f"{'Variable':<20s} {'Missing':>8s} {'%':>6s} {'Method':<15s}",
            "-" * 55,
        ]
        for var in self.variables_imputed:
            n_miss = self.n_missing[var]
            pct = n_miss / self.n_obs * 100
            method = self.methods.get(var, "pmm")
            lines.append(f"{var:<20s} {n_miss:>8d} {pct:>5.1f}% {method:<15s}")
        lines.append("=" * 55)
        return "\n".join(lines)

    def complete(self, m: int = 0) -> pd.DataFrame:
        """Return the m-th completed dataset (0-indexed)."""
        if m >= self.n_imputations:
            raise ValueError(f"Only {self.n_imputations} imputations available")
        return self.imputed_datasets[m].copy()

    def combine(self, estimates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply Rubin's rules to combine estimates across imputations."""
        return _rubins_rules(estimates)


def _barnard_rubin_df(m: int, b: np.ndarray, t: np.ndarray, dfcom: float) -> np.ndarray:
    """Barnard & Rubin (1999) small-sample degrees of freedom.

    ``lambda = (1 + 1/m) b / t``; ``nu_old = (m - 1) / lambda^2`` (Rubin
    1987) and, for finite ``dfcom``,
    ``nu = (m - 1) tmp / ((dfcom + 3)(m - 1) + lambda^2 tmp)`` with
    ``tmp = (1 - lambda)(1 + dfcom) dfcom`` -- algebraically
    ``1 / (1/nu_old + 1/nu_obs)``, written as R ``mice:::barnard.rubin``
    writes it so that ``b = 0`` (no between-imputation variance) gives
    ``nu_obs`` instead of ``inf / inf``.
    """
    lam = (1.0 + 1.0 / m) * b / t
    with np.errstate(divide="ignore", invalid="ignore"):
        if not np.isfinite(dfcom):
            return (m - 1) / lam**2
        tmp = (1.0 - lam) * (1.0 + dfcom) * dfcom
        return (m - 1) * tmp / ((dfcom + 3.0) * (m - 1) + lam**2 * tmp)


def _rubins_rules(
    estimates: List[Dict[str, Any]],
    dfcom: Optional[float] = None,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Apply Rubin's (1987) combination rules.

    Parameters
    ----------
    estimates : list of dict
        Each dict has 'params' (array) and 'var_cov' (matrix), and
        optionally 'df_resid' (complete-data residual degrees of freedom).
    dfcom : float, optional
        Complete-data degrees of freedom. Defaults to the ``df_resid`` the
        estimates carry (all must agree), else infinity (large sample).
    alpha : float, default 0.05
        Level of the returned confidence intervals.

    Returns
    -------
    dict
        ``params`` (Q-bar), ``se``, ``tvalues``, ``pvalues``, ``ci_lower``,
        ``ci_upper``, ``df``, ``dfcom``, ``ubar``, ``b``, ``t``, ``riv``,
        ``lambda``, ``fmi``, ``fmi_barnard_rubin`` (all per coefficient),
        ``var_cov`` (the total
        covariance ``U-bar + (1 + 1/m) B``), ``ubar_matrix`` / ``b_matrix``
        (the full within / between covariances, for :func:`mi_test`) and
        ``n_imputations``.

    Notes
    -----
    These are the quantities R ``mice::pool`` reports. The degrees of
    freedom are Barnard & Rubin's (1999) small-sample df when ``dfcom`` is
    finite and Rubin's (1987) ``(m - 1) / lambda^2`` otherwise; the
    fraction of missing information is ``(riv + 2/(df + 3)) / (riv + 1)``
    with that same df (R ``mice``); ``fmi_barnard_rubin`` is Barnard &
    Rubin's (1999) small-sample version ``1 - [l(df)/l(dfcom)] U/T`` with
    ``l(u) = (u + 1)/(u + 3)``, which is what Stata ``mi estimate`` reports
    (it equals ``fmi`` when ``dfcom`` is infinite). Before 1.30 StatsPAI
    always used Rubin's large-sample df (while labelling it Barnard-Rubin),
    which overstates the df -- and understates p-values -- when the
    complete-data df is small.
    """
    m = len(estimates)
    if m < 2:
        raise DataInsufficient(
            "Rubin's rules need at least two imputations.",
            recovery_hint="Pool the estimates from m >= 2 imputed datasets.",
        )
    params_list = np.asarray([np.asarray(e["params"], dtype=float) for e in estimates])
    vcov_list = np.asarray([np.asarray(e["var_cov"], dtype=float) for e in estimates])

    if dfcom is None:
        dfs = [e.get("df_resid") for e in estimates]
        if all(d is not None for d in dfs):
            if len(set(float(d) for d in dfs)) != 1:
                raise MethodIncompatibility(
                    f"complete-data df differ across imputations ({sorted(set(dfs))}); "
                    "pass dfcom= explicitly",
                    recovery_hint="Pass dfcom= explicitly.",
                )
            dfcom = float(dfs[0])
        else:
            dfcom = float("inf")
    dfcom = float(dfcom)

    # Combined point estimate: mean across imputations
    Q_bar = params_list.mean(axis=0)
    # Within-imputation variance
    U_bar = vcov_list.mean(axis=0)
    # Between-imputation variance (m - 1 divisor)
    dev = params_list - Q_bar
    B = dev.T @ dev / (m - 1)
    # Total variance
    T = U_bar + (1 + 1 / m) * B

    ubar = np.diag(U_bar)
    b = np.diag(B)
    t = np.diag(T)
    se = np.sqrt(t)
    tvalues = Q_bar / se
    riv = (1 + 1 / m) * b / ubar
    lam = (1 + 1 / m) * b / t
    df = _barnard_rubin_df(m, b, t, dfcom)
    pvalues = 2 * stats.t.sf(np.abs(tvalues), df)
    tcrit = stats.t.ppf(1 - alpha / 2, df)
    fmi = (riv + 2 / (df + 3)) / (riv + 1)
    # Barnard & Rubin (1999, p. 953) small-sample FMI, as Stata's
    # `mi estimate` reports it: 1 - [l(df) / l(dfcom)] U/T, l(u) = (u+1)/(u+3).
    if np.isfinite(dfcom):
        fmi_br = 1 - ((df + 1) / (df + 3)) / ((dfcom + 1) / (dfcom + 3)) * ubar / t
    else:
        fmi_br = fmi

    return {
        "params": Q_bar,
        "se": se,
        "tvalues": tvalues,
        "pvalues": pvalues,
        "ci_lower": Q_bar - tcrit * se,
        "ci_upper": Q_bar + tcrit * se,
        "df": df,
        "dfcom": dfcom,
        "ubar": ubar,
        "b": b,
        "t": t,
        "riv": riv,
        "lambda": lam,
        "fmi": fmi,
        "fmi_barnard_rubin": fmi_br,
        "var_cov": T,
        "ubar_matrix": U_bar,
        "b_matrix": B,
        "n_imputations": m,
    }


def _impute_pmm(y_obs: Any, x_obs: Any, x_miss: Any, rng: Any, k: int = 5) -> Any:
    """Predictive mean matching imputation."""
    n_obs = len(y_obs)
    if n_obs < 2:
        return rng.choice(y_obs, size=len(x_miss))

    # Fit OLS on observed
    X_obs = np.column_stack([np.ones(n_obs), x_obs])
    try:
        beta = np.linalg.lstsq(X_obs, y_obs, rcond=None)[0]
    except np.linalg.LinAlgError:
        return rng.choice(y_obs, size=len(x_miss))

    # Draw beta from posterior (Bayesian bootstrap)
    resid = y_obs - X_obs @ beta
    sigma2 = np.sum(resid**2) / max(n_obs - X_obs.shape[1], 1)
    try:
        XtX_inv = np.linalg.inv(X_obs.T @ X_obs)
    except np.linalg.LinAlgError:
        return rng.choice(y_obs, size=len(x_miss))

    beta_star = rng.multivariate_normal(beta, sigma2 * XtX_inv)

    # Predicted values
    y_hat_obs = X_obs @ beta_star
    X_miss = np.column_stack([np.ones(len(x_miss)), x_miss])
    y_hat_miss = X_miss @ beta_star

    # Match: for each missing, find k nearest observed predictions
    imputed = np.empty(len(x_miss))
    for i, yh in enumerate(y_hat_miss):
        distances = np.abs(y_hat_obs - yh)
        nearest_idx = np.argsort(distances)[:k]
        imputed[i] = rng.choice(y_obs[nearest_idx])

    return imputed


def _impute_norm(y_obs: Any, x_obs: Any, x_miss: Any, rng: Any) -> Any:
    """Normal (Bayesian linear regression) imputation."""
    n_obs = len(y_obs)
    X_obs = np.column_stack([np.ones(n_obs), x_obs])
    try:
        beta = np.linalg.lstsq(X_obs, y_obs, rcond=None)[0]
        resid = y_obs - X_obs @ beta
        sigma2 = np.sum(resid**2) / max(n_obs - X_obs.shape[1], 1)
        XtX_inv = np.linalg.inv(X_obs.T @ X_obs)
    except np.linalg.LinAlgError:
        return rng.normal(y_obs.mean(), y_obs.std(), size=len(x_miss))

    # Draw from posterior
    sigma2_star = (
        sigma2 * (n_obs - X_obs.shape[1]) / rng.chisquare(n_obs - X_obs.shape[1])
    )
    beta_star = rng.multivariate_normal(beta, sigma2_star * XtX_inv)

    X_miss = np.column_stack([np.ones(len(x_miss)), x_miss])
    y_hat = X_miss @ beta_star + rng.normal(0, np.sqrt(sigma2_star), size=len(x_miss))
    return y_hat


def _logit_newton(
    y: np.ndarray, X: np.ndarray, ridge: float = 1e-6, max_iter: int = 50
) -> Any:
    """Logistic MLE by Newton-Raphson; returns ``(beta, cov)`` or None.

    A tiny ridge keeps the Hessian invertible under (quasi-)separation, the
    role R ``mice`` gives to its data augmentation.
    """
    k = X.shape[1]
    beta = np.zeros(k)
    pen = ridge * np.eye(k)
    pen[0, 0] = 0.0
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        W = p * (1.0 - p)
        H = (X * W[:, None]).T @ X + pen
        g = X.T @ (y - p) - pen @ beta
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return None
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    eta = np.clip(X @ beta, -30, 30)
    p = 1.0 / (1.0 + np.exp(-eta))
    H = (X * (p * (1.0 - p))[:, None]).T @ X + pen
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(beta)) or not np.all(np.isfinite(cov)):
        return None
    return beta, (cov + cov.T) / 2.0


def _impute_logreg(y_obs: Any, x_obs: Any, x_miss: Any, rng: Any) -> Any:
    """Proper logistic imputation for a binary variable (R ``logreg``).

    Draws ``beta* ~ N(beta_hat, (X'WX)^-1)`` before predicting, so the
    between-imputation variance carries the parameter uncertainty Rubin's
    rules need. (Before 1.32 the MLE itself was used -- an improper
    imputation that understates the pooled variance.) Returns None when the
    model cannot be fitted; the caller records and reports that.
    """
    y_obs = np.asarray(y_obs, dtype=float)
    X_obs = np.column_stack([np.ones(len(y_obs)), x_obs])
    fit = _logit_newton(y_obs, X_obs)
    if fit is None:
        return None
    beta, cov = fit
    beta_star = rng.multivariate_normal(beta, cov)
    X_miss = np.column_stack([np.ones(len(x_miss)), x_miss])
    probs = 1.0 / (1.0 + np.exp(-np.clip(X_miss @ beta_star, -30, 30)))
    return (rng.random(len(x_miss)) < probs).astype(float)


def _mlogit_fit(codes: np.ndarray, X: np.ndarray, K: int, ridge: float = 1e-6) -> Any:
    """Multinomial logit (baseline category 0) by Newton; ``(theta, cov)``."""
    n, k = X.shape
    Y = np.zeros((n, K))
    Y[np.arange(n), codes] = 1.0
    theta = np.zeros((K - 1) * k)
    pen = ridge * np.eye(len(theta))
    for j in range(K - 1):
        pen[j * k, j * k] = 0.0

    def probs(th: np.ndarray) -> np.ndarray:
        E = np.column_stack([np.zeros(n), X @ th.reshape(K - 1, k).T])
        E = E - E.max(axis=1, keepdims=True)
        P = np.exp(E)
        out: np.ndarray = P / P.sum(axis=1, keepdims=True)
        return out

    def hessian(P: np.ndarray) -> np.ndarray:
        H = np.zeros((len(theta), len(theta)))
        for a in range(1, K):
            for b in range(1, K):
                w = P[:, a] * ((a == b) - P[:, b])
                H[(a - 1) * k : a * k, (b - 1) * k : b * k] = (X * w[:, None]).T @ X
        return H + pen

    for _ in range(50):
        P = probs(theta)
        g = (X.T @ (Y[:, 1:] - P[:, 1:])).T.ravel() - pen @ theta
        try:
            step = np.linalg.solve(hessian(P), g)
        except np.linalg.LinAlgError:
            return None
        theta = theta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    try:
        cov = np.linalg.inv(hessian(probs(theta)))
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(theta)) or not np.all(np.isfinite(cov)):
        return None
    return theta, (cov + cov.T) / 2.0


def _impute_polyreg(
    codes_obs: np.ndarray, x_obs: Any, x_miss: Any, K: int, rng: Any
) -> Any:
    """Proper multinomial-logit imputation for an unordered categorical.

    Parameters are drawn from their asymptotic posterior before the
    categories are sampled from the predicted probabilities. Returns integer
    category codes, or None when the model cannot be fitted.
    """
    X_obs = np.column_stack([np.ones(len(codes_obs)), x_obs])
    fit = _mlogit_fit(np.asarray(codes_obs, dtype=int), X_obs, K)
    if fit is None:
        return None
    theta, cov = fit
    th = rng.multivariate_normal(theta, cov).reshape(K - 1, X_obs.shape[1])
    X_miss = np.column_stack([np.ones(len(x_miss)), x_miss])
    E = np.column_stack([np.zeros(len(X_miss)), X_miss @ th.T])
    E = E - E.max(axis=1, keepdims=True)
    P = np.exp(E)
    P = P / P.sum(axis=1, keepdims=True)
    u = rng.random(len(X_miss))[:, None]
    return (u > np.cumsum(P, axis=1)).sum(axis=1).clip(0, K - 1)


def _is_categorical(s: pd.Series) -> bool:
    return not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)


def _predictor_matrix(
    df_imp: pd.DataFrame, pred_vars: List[str], categorical: set
) -> np.ndarray:
    """Numeric predictors as-is; categorical ones as treatment dummies."""
    blocks = []
    for v in pred_vars:
        if v in categorical:
            d = pd.get_dummies(df_imp[v].astype("object"), drop_first=True, dtype=float)
            if d.shape[1]:
                blocks.append(d.to_numpy(dtype=float))
        else:
            blocks.append(df_imp[v].to_numpy(dtype=float)[:, None])
    if not blocks:
        return np.empty((len(df_imp), 0))
    return np.column_stack(blocks)


def mice(
    data: pd.DataFrame,
    m: int = 5,
    max_iter: int = 10,
    method: Union[str, Dict[str, str]] = "pmm",
    predictors: Optional[Dict[str, List[str]]] = None,
    seed: Optional[int] = None,
    print_progress: bool = False,
) -> MICEResult:
    """
    Multiple Imputation by Chained Equations (MICE).

    Equivalent to Stata's ``mi impute chained`` and R's ``mice::mice()``.

    Parameters
    ----------
    data : pd.DataFrame
        Data with missing values (NaN).
    m : int, default 5
        Number of imputations.
    max_iter : int, default 10
        Number of MICE iterations per imputation.
    method : str or dict, default 'pmm'
        Imputation method per variable:
        'pmm' (predictive mean matching), 'norm' (Bayesian linear),
        'logreg' (proper logistic, for binary variables), 'polyreg'
        (proper multinomial logit, for unordered categoricals), 'sample'
        (random draw from the observed values -- ignores every other
        variable). Dict maps variable names to methods. With a string, the
        string applies to numeric variables with more than two values;
        two-valued variables get 'logreg' and non-numeric variables with
        more than two levels get 'polyreg', as R ``mice``'s defaults.
        (Before 1.32 non-numeric variables got 'sample', which attenuates
        every association with them.)
    predictors : dict, optional
        Dict mapping variable -> list of predictor variables.
        If None, uses all other variables; categorical ones enter as
        treatment dummies.
    seed : int, optional
        Random seed.
    print_progress : bool, default False

    Returns
    -------
    MICEResult
        Result with .complete(m), .summary(), and .combine() methods.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(7)
    >>> x1 = rng.normal(size=120)
    >>> x2 = 0.5 * x1 + rng.normal(size=120)
    >>> y = 1.0 + x1 + x2 + rng.normal(size=120)
    >>> df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    >>> df.loc[rng.choice(120, size=15, replace=False), "x2"] = np.nan
    >>> result = sp.mice(df, m=3, method='pmm', seed=0)
    >>> df_complete = result.complete(0)  # first imputed dataset
    >>> type(df_complete).__name__
    'DataFrame'
    >>> result.n_imputations
    3
    >>> # Combine estimates across imputations with Rubin's rules
    >>> estimates = []
    >>> for i in range(3):
    ...     df_i = result.complete(i)
    ...     r = sp.regress("y ~ x1 + x2", data=df_i)
    ...     estimates.append({'params': r.params.values,
    ...                       'var_cov': np.diag(r.std_errors.values ** 2)})
    >>> combined = result.combine(estimates)
    >>> {'fmi', 'df', 'n_imputations'} <= set(combined)
    True
    """
    rng = np.random.default_rng(seed)
    df = data.copy()

    # Identify variables with missing data
    missing_vars = [col for col in df.columns if df[col].isna().any()]
    n_missing = {col: df[col].isna().sum() for col in missing_vars}
    categorical = {c for c in df.columns if _is_categorical(df[c])}
    # A string column with very many levels (an id, a name) would enter every
    # equation as hundreds of dummies; it is left out as a predictor.
    max_levels = max(10, min(50, len(df) // 10))
    wide = sorted(c for c in categorical if df[c].nunique() > max_levels)
    usable_cols = [c for c in df.columns if c not in wide]
    if wide and predictors is None:
        import warnings as _warnings

        _warnings.warn(
            f"mice: categorical column(s) {wide} have more than {max_levels} "
            "levels and are not used as predictors (pass predictors= to "
            "override).",
            UserWarning,
            stacklevel=2,
        )
    valid_methods = ("pmm", "norm", "logreg", "polyreg", "sample")

    # Determine methods
    if isinstance(method, str):
        methods = {}
        for var in missing_vars:
            n_lev = df[var].dropna().nunique()
            if n_lev == 2:
                methods[var] = "logreg"
            elif var in categorical:
                methods[var] = "polyreg"
            else:
                methods[var] = method
    else:
        methods = dict(method)
        for var in missing_vars:
            methods.setdefault(
                var,
                (
                    "logreg"
                    if df[var].dropna().nunique() == 2
                    else ("polyreg" if var in categorical else "pmm")
                ),
            )
    bad = {v: mth for v, mth in methods.items() if mth not in valid_methods}
    if bad:
        raise MethodIncompatibility(
            f"mice: unknown imputation method(s) {bad}.",
            recovery_hint=f"Use one of {valid_methods}.",
        )
    for var in missing_vars:
        if var in categorical and methods[var] in ("pmm", "norm"):
            raise MethodIncompatibility(
                f"mice: {var!r} is categorical; method {methods[var]!r} needs a "
                "numeric variable.",
                recovery_hint="Use 'logreg' (2 levels), 'polyreg' or 'sample'.",
            )
    fit_failures: List[Dict[str, Any]] = []

    # Sort by amount of missing (least to most)
    missing_vars_sorted = sorted(missing_vars, key=lambda v: n_missing[v])

    imputed_datasets = []

    for imp in range(m):
        if print_progress:
            print(f"Imputation {imp + 1}/{m}")

        df_imp = df.copy()

        # Initial imputation: random sample from observed
        for var in missing_vars_sorted:
            mask = df_imp[var].isna()
            observed = df_imp.loc[~mask, var].values
            if len(observed) > 0:
                df_imp.loc[mask, var] = rng.choice(observed, size=mask.sum())

        # Chained equations iterations
        for iteration in range(max_iter):
            for var in missing_vars_sorted:
                mask = df[var].isna()  # Original missing pattern
                if mask.sum() == 0:
                    continue

                # Predictor variables
                if predictors is not None and var in predictors:
                    pred_vars = predictors[var]
                else:
                    pred_vars = [c for c in usable_cols if c != var]

                m_method = methods.get(var, "pmm")
                observed = df.loc[~mask, var].values
                if len(pred_vars) == 0 or m_method == "sample":
                    df_imp.loc[mask, var] = rng.choice(observed, size=mask.sum())
                    continue

                # Predictor matrix at the current imputed values; categorical
                # predictors enter as treatment dummies.
                X_all = _predictor_matrix(df_imp, pred_vars, categorical)
                for j in range(X_all.shape[1]):
                    col_nan = np.isnan(X_all[:, j])
                    if col_nan.any():
                        X_all[col_nan, j] = np.nanmean(X_all[:, j])
                x_obs = X_all[~mask.values]
                x_miss = X_all[mask.values]

                imputed: Any
                if m_method in ("logreg", "polyreg"):
                    levels = pd.unique(df.loc[~mask, var])
                    levels = sorted(levels, key=lambda v: (str(type(v)), v))
                    lut = {lv: i for i, lv in enumerate(levels)}
                    codes = df_imp.loc[~mask, var].map(lut).to_numpy()
                    if m_method == "logreg" and len(levels) == 2:
                        drawn = _impute_logreg(codes, x_obs, x_miss, rng)
                    else:
                        drawn = _impute_polyreg(codes, x_obs, x_miss, len(levels), rng)
                    if drawn is None:
                        fit_failures.append(
                            {
                                "imputation": imp,
                                "iteration": iteration,
                                "variable": var,
                                "method": m_method,
                            }
                        )
                        imputed = rng.choice(observed, size=mask.sum())
                    else:
                        imputed = np.asarray(levels, dtype=object)[
                            np.asarray(drawn, dtype=int)
                        ]
                        if not _is_categorical(df[var]):
                            imputed = imputed.astype(float)
                else:
                    y_obs = df_imp.loc[~mask, var].values.astype(float)
                    if m_method == "pmm":
                        imputed = _impute_pmm(y_obs, x_obs, x_miss, rng)
                    else:
                        imputed = _impute_norm(y_obs, x_obs, x_miss, rng)

                df_imp.loc[mask, var] = imputed

        imputed_datasets.append(df_imp)

    if fit_failures:
        import warnings as _warnings

        from ..exceptions import ConvergenceWarning

        by_var: Dict[str, int] = {}
        for f in fit_failures:
            by_var[f["variable"]] = by_var.get(f["variable"], 0) + 1
        _warnings.warn(
            ConvergenceWarning(
                "mice: the imputation model could not be fitted in "
                f"{len(fit_failures)} step(s) ({by_var}); those steps drew "
                "from the observed values instead, which ignores the other "
                "variables.",
                recovery_hint="Reduce predictors (predictors=), merge sparse "
                "categories, or check for perfect prediction.",
                diagnostics={"by_variable": by_var},
            ),
            stacklevel=2,
        )

    _result = MICEResult(
        imputed_datasets=imputed_datasets,
        n_imputations=m,
        n_obs=len(df),
        n_missing=n_missing,
        variables_imputed=missing_vars_sorted,
        methods=methods,
        convergence=not fit_failures,
    )
    _result.fit_failures = fit_failures
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.imputation.mice",
            params={
                "m": m,
                "max_iter": max_iter,
                "method": method if isinstance(method, str) else dict(method),
                "predictors": (
                    {k: list(v) for k, v in predictors.items()} if predictors else None
                ),
                "seed": seed,
                "print_progress": print_progress,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


def mi_estimate(
    mice_result: MICEResult,
    estimator: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Run an estimator on each imputed dataset and combine using Rubin's rules.

    Parameters
    ----------
    mice_result : MICEResult
        Result from mice().
    estimator : callable
        Estimation function that returns an EconometricResults object.
    **kwargs
        Arguments passed to the estimator.

    Returns
    -------
    dict
        Combined estimates (Rubin's rules), keyed as in
        :meth:`MICEResult.combine`, plus ``var_names``. The within-imputation
        covariance is the estimator's full ``var_cov`` when it exposes one
        (``data_info['var_cov']``), else the diagonal of squared standard
        errors; the complete-data df for Barnard-Rubin is its ``df_resid``
        when exposed, else infinity. This reproduces R ``mice::pool`` and
        Stata ``mi estimate`` (small-sample df) for ``sp.regress``.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(7)
    >>> x1 = rng.normal(size=120)
    >>> x2 = 0.5 * x1 + rng.normal(size=120)
    >>> y = 1.0 + x1 + x2 + rng.normal(size=120)
    >>> df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    >>> df.loc[rng.choice(120, size=15, replace=False), "x2"] = np.nan
    >>> mice_res = sp.mice(df, m=3, seed=0)
    >>> combined = sp.mi_estimate(mice_res, sp.regress, formula="y ~ x1 + x2")
    >>> combined['n_imputations']
    3
    >>> combined['var_names']
    ['Intercept', 'x1', 'x2']
    """
    estimates = []
    var_names: Optional[List[str]] = None
    for i in range(mice_result.n_imputations):
        df_i = mice_result.complete(i)
        result = estimator(data=df_i, **kwargs)
        params = result.params
        info = getattr(result, "data_info", None) or {}
        V = info.get("var_cov")
        k = len(params)
        if V is None or np.shape(V) != (k, k):
            # No full covariance exposed: pool the variances only.
            V = np.diag(np.asarray(result.std_errors, dtype=float) ** 2)
        est = {"params": np.asarray(params, dtype=float), "var_cov": np.asarray(V)}
        if info.get("df_resid") is not None:
            est["df_resid"] = float(info["df_resid"])
        estimates.append(est)
        if var_names is None:
            var_names = list(getattr(params, "index", range(k)))

    combined = _rubins_rules(estimates)
    combined["var_names"] = var_names
    return combined
