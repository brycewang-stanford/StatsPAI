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

from typing import Optional, List, Dict, Any, Union
import numpy as np
import pandas as pd
from scipy import stats
import warnings


class MICEResult:
    """Results from MICE imputation."""

    def __init__(self, imputed_datasets, n_imputations, n_obs,
                 n_missing, variables_imputed, methods, convergence):
        self.imputed_datasets = imputed_datasets
        self.n_imputations = n_imputations
        self.n_obs = n_obs
        self.n_missing = n_missing  # dict: var -> count
        self.variables_imputed = variables_imputed
        self.methods = methods  # dict: var -> method used
        self.convergence = convergence

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
            method = self.methods.get(var, 'pmm')
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


def _rubins_rules(estimates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Apply Rubin's (1987) combination rules.

    Parameters
    ----------
    estimates : list of dict
        Each dict has 'params' (array) and 'var_cov' (matrix).

    Returns
    -------
    dict
        Combined estimates with keys: 'params', 'se', 'tvalues',
        'pvalues', 'fmi' (fraction of missing information).
    """
    m = len(estimates)
    params_list = [e['params'] for e in estimates]
    vcov_list = [e['var_cov'] for e in estimates]

    # Combined point estimate: mean across imputations
    Q_bar = np.mean(params_list, axis=0)

    # Within-imputation variance
    U_bar = np.mean(vcov_list, axis=0)

    # Between-imputation variance
    B = np.zeros_like(U_bar)
    for q in params_list:
        diff = q - Q_bar
        B += np.outer(diff, diff)
    B /= (m - 1)

    # Total variance
    T = U_bar + (1 + 1/m) * B

    se = np.sqrt(np.diag(T))
    tvalues = Q_bar / se

    # Degrees of freedom (Barnard-Rubin 1999)
    r = (1 + 1/m) * np.diag(B) / np.diag(U_bar)
    nu_old = (m - 1) * (1 + 1/r)**2
    # Large sample df
    pvalues = 2 * (1 - stats.t.cdf(np.abs(tvalues), np.maximum(nu_old, 1)))

    # Fraction of missing information
    fmi = (r + 2 / (nu_old + 3)) / (r + 1)

    return {
        'params': Q_bar,
        'se': se,
        'tvalues': tvalues,
        'pvalues': pvalues,
        'fmi': fmi,
        'var_cov': T,
        'n_imputations': m,
    }


def _impute_pmm(y_obs, x_obs, x_miss, rng, k=5):
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


def _impute_norm(y_obs, x_obs, x_miss, rng):
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
    sigma2_star = sigma2 * (n_obs - X_obs.shape[1]) / rng.chisquare(n_obs - X_obs.shape[1])
    beta_star = rng.multivariate_normal(beta, sigma2_star * XtX_inv)

    X_miss = np.column_stack([np.ones(len(x_miss)), x_miss])
    y_hat = X_miss @ beta_star + rng.normal(0, np.sqrt(sigma2_star), size=len(x_miss))
    return y_hat


def _impute_logreg(y_obs, x_obs, x_miss, rng):
    """Logistic regression imputation for binary variables."""
    from scipy.optimize import minimize

    n_obs = len(y_obs)
    X_obs = np.column_stack([np.ones(n_obs), x_obs])

    def neg_ll(beta):
        xb = X_obs @ beta
        xb = np.clip(xb, -500, 500)
        p = 1 / (1 + np.exp(-xb))
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return -np.sum(y_obs * np.log(p) + (1 - y_obs) * np.log(1 - p))

    beta0 = np.zeros(X_obs.shape[1])
    try:
        result = minimize(neg_ll, beta0, method='BFGS')
        beta = result.x
    except Exception:
        p_obs = y_obs.mean()
        return (rng.random(len(x_miss)) < p_obs).astype(float)

    X_miss = np.column_stack([np.ones(len(x_miss)), x_miss])
    xb = X_miss @ beta
    xb = np.clip(xb, -500, 500)
    probs = 1 / (1 + np.exp(-xb))
    return (rng.random(len(x_miss)) < probs).astype(float)


def mice(
    data: pd.DataFrame,
    m: int = 5,
    max_iter: int = 10,
    method: Union[str, Dict[str, str]] = "pmm",
    predictors: Dict[str, List[str]] = None,
    seed: int = None,
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
        'logreg' (logistic for binary), 'sample' (random sample).
        Dict maps variable names to methods.
    predictors : dict, optional
        Dict mapping variable -> list of predictor variables.
        If None, uses all other variables.
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
    >>> result = sp.mice(df, m=5, method='pmm')
    >>> df_complete = result.complete(0)  # first imputed dataset
    >>> print(result.summary())
    >>>
    >>> # Combine estimates across imputations
    >>> estimates = []
    >>> for i in range(5):
    ...     df_i = result.complete(i)
    ...     r = sp.regress("y ~ x1 + x2", data=df_i)
    ...     estimates.append({'params': r.params.values, 'var_cov': np.diag(r.std_errors.values**2)})
    >>> combined = result.combine(estimates)
    """
    rng = np.random.default_rng(seed)
    df = data.copy()

    # Identify variables with missing data
    missing_vars = [col for col in df.columns if df[col].isna().any()]
    n_missing = {col: df[col].isna().sum() for col in missing_vars}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Determine methods
    if isinstance(method, str):
        methods = {}
        for var in missing_vars:
            if var not in numeric_cols:
                methods[var] = 'sample'
            elif df[var].dropna().nunique() == 2:
                methods[var] = 'logreg'
            else:
                methods[var] = method
    else:
        methods = method

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
                    pred_vars = [c for c in numeric_cols if c != var and c in df_imp.columns]

                if len(pred_vars) == 0:
                    # No predictors: sample from observed
                    observed = df.loc[~mask, var].values
                    df_imp.loc[mask, var] = rng.choice(observed, size=mask.sum())
                    continue

                # Get predictor matrix (using current imputed values)
                X_all = df_imp[pred_vars].values.astype(float)

                # Handle any remaining NaN in predictors
                for j in range(X_all.shape[1]):
                    col_nan = np.isnan(X_all[:, j])
                    if col_nan.any():
                        X_all[col_nan, j] = np.nanmean(X_all[:, j])

                y_obs = df_imp.loc[~mask, var].values.astype(float)
                x_obs = X_all[~mask.values]
                x_miss = X_all[mask.values]

                m_method = methods.get(var, 'pmm')

                if m_method == 'pmm':
                    imputed = _impute_pmm(y_obs, x_obs, x_miss, rng)
                elif m_method == 'norm':
                    imputed = _impute_norm(y_obs, x_obs, x_miss, rng)
                elif m_method == 'logreg':
                    imputed = _impute_logreg(y_obs, x_obs, x_miss, rng)
                elif m_method == 'sample':
                    observed = df.loc[~mask, var].values
                    imputed = rng.choice(observed, size=mask.sum())
                else:
                    observed = df.loc[~mask, var].values
                    imputed = rng.choice(observed, size=mask.sum())

                df_imp.loc[mask, var] = imputed

        imputed_datasets.append(df_imp)

    _result = MICEResult(
        imputed_datasets=imputed_datasets,
        n_imputations=m,
        n_obs=len(df),
        n_missing=n_missing,
        variables_imputed=missing_vars_sorted,
        methods=methods,
        convergence=True,
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov
        _attach_prov(
            _result,
            function="sp.imputation.mice",
            params={
                "m": m, "max_iter": max_iter,
                "method": method if isinstance(method, str)
                          else dict(method),
                "predictors": (
                    {k: list(v) for k, v in predictors.items()}
                    if predictors else None
                ),
                "seed": seed, "print_progress": print_progress,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


def mi_estimate(
    mice_result: MICEResult,
    estimator,
    **kwargs,
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
        Combined estimates (Rubin's rules).

    Examples
    --------
    >>> import statspai as sp
    >>> mice_res = sp.mice(df, m=5)
    >>> combined = sp.mi_estimate(mice_res, sp.regress, formula="y ~ x1 + x2")
    """
    estimates = []

    for i in range(mice_result.n_imputations):
        df_i = mice_result.complete(i)
        result = estimator(data=df_i, **kwargs)
        estimates.append({
            'params': result.params.values,
            'var_cov': np.diag(result.std_errors.values**2),
        })

    combined = _rubins_rules(estimates)
    # Add variable names from the first result
    df_0 = mice_result.complete(0)
    first_result = estimator(data=df_0, **kwargs)
    combined['var_names'] = list(first_result.params.index)

    return combined
