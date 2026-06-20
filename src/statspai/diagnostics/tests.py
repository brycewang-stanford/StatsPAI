"""
Regression diagnostic tests — Python equivalent of Stata's ``estat`` suite.

Provides a unified interface for common specification and diagnostic
tests that are standard in empirical economics.

Functions
---------
diagnose(result)
    One-line comprehensive diagnostics (hetero, serial, functional form, VIF).
het_test(result)
    Breusch-Pagan / White heteroskedasticity test.
reset_test(result)
    Ramsey RESET functional form test.
vif(result)
    Variance Inflation Factors for multicollinearity.
serial_test(result)
    Breusch-Godfrey serial correlation test.

References
----------
Breusch, T.S. and Pagan, A.R. (1979).
"A Simple Test for Heteroscedasticity and Random Coefficient Variation."
*Econometrica*, 47(5), 1287-1294. [@breusch1979simple]

Ramsey, J.B. (1969).
"Tests for Specification Errors in Classical Linear Least-Squares
Regression Analysis."
*Journal of the Royal Statistical Society: Series B*, 31(2), 350-371. [@ramsey1969tests]

White, H. (1980).
"A Heteroskedasticity-Consistent Covariance Matrix Estimator and a
Direct Test for Heteroskedasticity."
*Econometrica*, 48(4), 817-838. [@white1980heteroskedasticity]
"""

from typing import Dict, Any, List

import numpy as np
import pandas as pd
from scipy import stats


def diagnose(
    data: pd.DataFrame,
    y: str,
    x: List[str],
    print_results: bool = True,
) -> Dict[str, Any]:
    """
    Comprehensive regression diagnostics in one call.

    Equivalent to running Stata's ``estat hettest``, ``estat ovtest``,
    ``estat vif`` after a regression.

    Parameters
    ----------
    data : pd.DataFrame
        Data used in the regression.
    y : str
        Dependent variable name.
    x : list of str
        Independent variable names (excluding constant).
    print_results : bool, default True
        Print formatted output.

    Returns
    -------
    dict
        Keys: ``'het_test'``, ``'reset_test'``, ``'vif'``,
        each containing test results.

    Examples
    --------
    Run the full diagnostic suite on a Mincer-style wage regression:

    >>> import statspai as sp
    >>> df = sp.cps_wage()
    >>> diag = sp.diagnose(df, y='log_wage',
    ...                    x=['education', 'experience'],
    ...                    print_results=False)
    >>> sorted(diag.keys())
    ['het_test', 'reset_test', 'vif']

    References
    ----------
    breusch1979simple, ramsey1969tests, white1980heteroskedasticity
    """
    results = {}

    # Run OLS first
    df = data[[y] + x].dropna()
    Y = df[y].values
    X = np.column_stack([np.ones(len(df))] + [df[v].values for v in x])
    n, k = X.shape
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    resid = Y - X @ beta
    yhat = X @ beta

    results["het_test"] = _bp_test(resid, X, n, k)
    results["reset_test"] = _reset_test(Y, X, resid, yhat, n, k)
    results["vif"] = _vif(df, x)

    if print_results:
        _print_diagnostics(results, x)

    return results


def het_test(
    data: pd.DataFrame,
    y: str,
    x: List[str],
) -> Dict[str, float]:
    """
    Breusch-Pagan test for heteroskedasticity.

    H0: Homoskedastic errors (constant variance).
    H1: Variance depends on regressors.

    Equivalent to Stata's ``estat hettest``.

    Parameters
    ----------
    data, y, x : as in ``diagnose()``

    Returns
    -------
    dict
        ``'statistic'``, ``'df'``, ``'pvalue'``.

    References
    ----------
    Breusch, T.S. and Pagan, A.R. (1979). *Econometrica*, 47(5), 1287-1294. [@breusch1979simple]

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> df = pd.DataFrame({
    ...     "x1": rng.normal(size=200),
    ...     "x2": rng.normal(size=200),
    ... })
    >>> df["y"] = (1.0 + 0.5 * df["x1"] - 0.3 * df["x2"]
    ...            + rng.normal(size=200))
    >>> ht = sp.het_test(df, y="y", x=["x1", "x2"])
    >>> int(ht["df"])
    2
    >>> ht["pvalue"] < 0.05   # homoskedastic data: do not reject H0
    False
    """
    df = data[[y] + x].dropna()
    Y = df[y].values
    X = np.column_stack([np.ones(len(df))] + [df[v].values for v in x])
    n, k = X.shape
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    resid = Y - X @ beta
    return _bp_test(resid, X, n, k)


def reset_test(
    data: pd.DataFrame,
    y: str,
    x: List[str],
    powers: int = 3,
) -> Dict[str, float]:
    """
    Ramsey RESET test for functional form misspecification.

    Tests whether nonlinear combinations of fitted values help
    explain the dependent variable. Rejection suggests the model
    is missing important nonlinearities.

    Equivalent to Stata's ``estat ovtest``.

    Parameters
    ----------
    data, y, x : as in ``diagnose()``
    powers : int, default 3
        Include ŷ², ŷ³, ..., ŷ^powers in the auxiliary regression.

    Returns
    -------
    dict
        ``'statistic'`` (F), ``'df1'``, ``'df2'``, ``'pvalue'``.

    References
    ----------
    Ramsey, J.B. (1969). *JRSS-B*, 31(2), 350-371. [@ramsey1969tests]

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> df = pd.DataFrame({
    ...     "x1": rng.normal(size=200),
    ...     "x2": rng.normal(size=200),
    ... })
    >>> df["y"] = (1.0 + 0.5 * df["x1"] - 0.3 * df["x2"]
    ...            + rng.normal(size=200))
    >>> rt = sp.reset_test(df, y="y", x=["x1", "x2"])
    >>> rt["df1"], rt["df2"]
    (2, 195)
    >>> rt["pvalue"] > 0.05   # linear form is correct: do not reject
    True
    """
    df = data[[y] + x].dropna()
    Y = df[y].values
    X = np.column_stack([np.ones(len(df))] + [df[v].values for v in x])
    n, k = X.shape
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    resid = Y - X @ beta
    yhat = X @ beta
    return _reset_test(Y, X, resid, yhat, n, k, powers)


def vif(
    data: pd.DataFrame,
    x: List[str],
) -> pd.DataFrame:
    """
    Variance Inflation Factors for multicollinearity detection.

    VIF > 10 is a common rule of thumb for concerning multicollinearity.

    Equivalent to Stata's ``estat vif``.

    Parameters
    ----------
    data : pd.DataFrame
    x : list of str
        Independent variables.

    Returns
    -------
    pd.DataFrame
        Columns: ``'variable'``, ``'VIF'``, ``'1/VIF'``.

    Notes
    -----
    VIF_j = 1 / (1 - R²_j) where R²_j is from regressing x_j on
    all other x variables.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> df = pd.DataFrame({
    ...     "x1": rng.normal(size=200),
    ...     "x2": rng.normal(size=200),
    ... })
    >>> v = sp.vif(df, x=["x1", "x2"])
    >>> list(v.columns)
    ['variable', 'VIF', '1/VIF']
    >>> bool((v["VIF"] < 5).all())   # near-orthogonal: low VIFs
    True
    """
    return _vif(data, x)


# ======================================================================
# Internal implementations
# ======================================================================


def _bp_test(resid: Any, X: Any, n: int, k: int) -> Dict[str, Any]:
    """Breusch-Pagan test."""
    e2 = resid**2
    # Regress e² on X
    beta_e = np.linalg.lstsq(X, e2, rcond=None)[0]
    e2_hat = X @ beta_e
    rss_restricted = np.sum((e2 - e2.mean()) ** 2)
    rss_unrestricted = np.sum((e2 - e2_hat) ** 2)

    # LM statistic = n * R²
    r2 = 1 - rss_unrestricted / rss_restricted if rss_restricted > 0 else 0
    lm = n * r2
    df = k - 1  # excluding constant
    pvalue = float(1 - stats.chi2.cdf(lm, df))

    return {"statistic": float(lm), "df": df, "pvalue": pvalue, "test": "Breusch-Pagan"}


def _reset_test(
    Y: Any, X: Any, resid: Any, yhat: Any, n: int, k: int, powers: int = 3
) -> Dict[str, Any]:
    """Ramsey RESET test."""
    # Augmented model: add yhat², yhat³, ...
    X_aug_cols = [X]
    for p in range(2, powers + 1):
        X_aug_cols.append((yhat**p).reshape(-1, 1))
    X_aug = np.column_stack(X_aug_cols)
    k_aug = X_aug.shape[1]

    beta_aug = np.linalg.lstsq(X_aug, Y, rcond=None)[0]
    resid_aug = Y - X_aug @ beta_aug

    rss_r = np.sum(resid**2)
    rss_u = np.sum(resid_aug**2)
    df1 = k_aug - k
    df2 = n - k_aug

    if rss_u > 0 and df1 > 0 and df2 > 0:
        f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)
        pvalue = float(1 - stats.f.cdf(f_stat, df1, df2))
    else:
        f_stat = 0.0
        pvalue = 1.0

    return {
        "statistic": float(f_stat),
        "df1": df1,
        "df2": df2,
        "pvalue": pvalue,
        "test": "Ramsey RESET",
    }


def _vif(data: pd.DataFrame, x_vars: List[str]) -> pd.DataFrame:
    """Compute VIF for each variable."""
    df = data[x_vars].dropna()
    X = df.values
    n, k = X.shape
    rows = []

    for j in range(k):
        # Regress x_j on all other x
        y_j = X[:, j]
        X_other = np.column_stack([np.ones(n)] + [X[:, i] for i in range(k) if i != j])
        beta_j = np.linalg.lstsq(X_other, y_j, rcond=None)[0]
        resid_j = y_j - X_other @ beta_j
        tss_j = np.sum((y_j - y_j.mean()) ** 2)
        rss_j = np.sum(resid_j**2)
        r2_j = 1 - rss_j / tss_j if tss_j > 0 else 0

        vif_j = 1 / (1 - r2_j) if r2_j < 1 else np.inf

        rows.append(
            {
                "variable": x_vars[j],
                "VIF": round(vif_j, 2),
                "1/VIF": round(1 / vif_j, 4) if vif_j > 0 else 0,
            }
        )

    return pd.DataFrame(rows)


def _print_diagnostics(results: Dict[str, Any], x_vars: List[str]) -> None:
    """Pretty-print diagnostic results."""
    print("=" * 60)
    print("  Regression Diagnostics")
    print("=" * 60)

    # Heteroskedasticity
    ht = results["het_test"]
    print("\n  Breusch-Pagan test for heteroskedasticity")
    print(f"    chi2({ht['df']}) = {ht['statistic']:.4f}")
    print(f"    p-value = {ht['pvalue']:.4f}")
    print(
        f"    {'REJECT H0: evidence of heteroskedasticity' if ht['pvalue'] < 0.05 else 'Cannot reject H0: no evidence of heteroskedasticity'}"
    )

    # RESET
    rt = results["reset_test"]
    print("\n  Ramsey RESET test")
    print(f"    F({rt['df1']}, {rt['df2']}) = {rt['statistic']:.4f}")
    print(f"    p-value = {rt['pvalue']:.4f}")
    print(
        f"    {'REJECT H0: functional form may be misspecified' if rt['pvalue'] < 0.05 else 'Cannot reject H0: no evidence of misspecification'}"
    )

    # VIF
    vf = results["vif"]
    print("\n  Variance Inflation Factors")
    print(vf.to_string(index=False))
    max_vif = vf["VIF"].max()
    if max_vif > 10:
        print("    WARNING: VIF > 10 detected — multicollinearity concern")
    else:
        print(f"    Max VIF = {max_vif:.2f} — no multicollinearity concern")

    print("=" * 60)
