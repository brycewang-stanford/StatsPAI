"""What is read off a VAR besides its coefficients.

Lag-order selection before the fit (:func:`varsoc`), and after it the Wald
test that a lag can be dropped, the LM test for residual autocorrelation,
the stability condition, Granger causality and forecasts. The post-fit
pieces are methods of :class:`~statspai.timeseries.var.VARResult` and are
reachable through :func:`statspai.estat`.

Every statistic uses the maximum-likelihood residual covariance (divisor
``T``), which is what Stata's ``var`` suite reports.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = [
    "varsoc",
    "lag_exclusion",
    "lm_autocorrelation",
    "stability",
    "granger_table",
    "forecast",
    "equation_table",
]


# ------------------------------------------------------------------ helpers
def _design(values: np.ndarray, lags: int, trend: str, skip: int) -> tuple:
    """``(Y, X)`` of a VAR(``lags``) whose first usable row is ``skip``."""
    n = values.shape[0]
    Y = values[skip:]
    parts = [values[skip - lag : n - lag] for lag in range(1, lags + 1)]
    T = Y.shape[0]
    if trend in ("c", "ct"):
        parts.append(np.ones((T, 1)))
    if trend == "ct":
        parts.append(np.arange(1, T + 1, dtype=float).reshape(-1, 1))
    X = np.hstack(parts) if parts else np.zeros((T, 0))
    return Y, X


def _ml_sigma(Y: np.ndarray, X: np.ndarray) -> np.ndarray:
    if X.shape[1]:
        resid = Y - X @ np.linalg.lstsq(X, Y, rcond=None)[0]
    else:
        resid = Y
    return resid.T @ resid / Y.shape[0]


def _require_fit(result: Any) -> None:
    if getattr(result, "_B", None) is None or getattr(result, "_X", None) is None:
        raise MethodIncompatibility(
            "This diagnostic needs a VAR fitted by sp.var.",
            recovery_hint="Fit the model with sp.var(...) and pass the result.",
        )


# ---------------------------------------------------------- lag selection
def varsoc(
    data: pd.DataFrame,
    variables: Optional[Sequence[str]] = None,
    *,
    maxlag: int = 4,
    trend: str = "c",
    alpha: float = 0.05,
    exog: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Lag-order selection statistics for a VAR.

    Fits VAR(0) .. VAR(``maxlag``) on the same observations (the first
    ``maxlag`` are set aside for every model, so the likelihoods are
    comparable) and reports, for each order, the log likelihood, the
    likelihood-ratio test against the order below, the final prediction
    error and the Akaike, Hannan-Quinn and Schwarz criteria.

    Parameters
    ----------
    data : pandas.DataFrame
        The series, in time order.
    variables : sequence of str, optional
        Endogenous variables. Default: every numeric column.
    maxlag : int, default 4
        Highest order considered.
    trend : {'c', 'ct', 'n'}, default 'c'
        Deterministic terms: constant, constant and trend, or none.
    alpha : float, default 0.05
        Level of the sequential likelihood-ratio test.

    Returns
    -------
    pandas.DataFrame
        Indexed by lag, with columns ``LL``, ``LR``, ``df``, ``p``, ``FPE``,
        ``AIC``, ``HQIC``, ``SBIC``. ``attrs['selected']`` maps each
        criterion to the order it picks: the minimum of FPE / AIC / HQIC /
        SBIC, and for ``LR`` the highest order whose test against the order
        below rejects. ``attrs['n']`` is the common number of observations.

    Notes
    -----
    The criteria are scaled by the number of observations, ``AIC = (-2 LL +
    2 m) / T`` with ``m`` the total number of coefficients, as in Stata's
    ``varsoc`` and Lütkepohl's text; R's ``vars::VARselect`` reports
    ``log|Sigma| + 2 m / T``, which ranks the orders the same way.

    SBIC and HQIC are consistent for the true order; AIC and FPE
    overestimate it with positive probability in large samples but can
    forecast better in small ones. When they disagree, check the residuals
    of the smaller model (``sp.estat(fit, 'varlmar')``) before settling.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(1)
    >>> T = 300
    >>> y = np.zeros((T, 2))
    >>> for t in range(1, T):
    ...     y[t] = 0.5 * y[t - 1] + rng.normal(size=2)
    >>> table = sp.varsoc(pd.DataFrame(y, columns=["a", "b"]), maxlag=3)
    >>> int(table.attrs["selected"]["SBIC"])
    1

    References
    ----------
    lutkepohl2005new
    """
    if variables is None:
        variables = data.select_dtypes(include=[np.number]).columns.tolist()
    variables = list(variables)
    unknown = [v for v in variables if v not in data.columns]
    if unknown or not variables:
        raise MethodIncompatibility(
            (
                f"sp.varsoc: variable(s) {unknown} are not in the data."
                if unknown
                else "sp.varsoc: no variables given."
            ),
            recovery_hint="Name the endogenous series.",
        )
    if trend not in ("c", "ct", "n"):
        raise MethodIncompatibility(
            f"sp.varsoc: trend={trend!r} is not 'c', 'ct' or 'n'.",
            recovery_hint="Use trend='c'.",
        )
    exog_names = [str(v) for v in (exog or [])]
    if [v for v in exog_names if v not in data.columns or v in variables]:
        raise MethodIncompatibility(
            f"sp.varsoc: exog={exog_names} must name columns that are not "
            "among the endogenous variables.",
            recovery_hint="Check the names passed to exog=.",
        )
    complete = data[variables + exog_names].dropna()
    values = complete[variables].to_numpy(dtype=float)
    extra = complete[exog_names].to_numpy(dtype=float)[maxlag:]
    n, k = values.shape
    T = n - maxlag
    deterministic = {"c": 1, "ct": 2, "n": 0}[trend] + len(exog_names)
    if maxlag < 1 or T <= k * maxlag + deterministic:
        raise DataInsufficient(
            f"sp.varsoc: {n} observations cannot support maxlag={maxlag} "
            f"with {k} variable(s).",
            recovery_hint="Lower maxlag.",
        )
    rows: List[Dict[str, float]] = []
    previous = np.nan
    for p in range(maxlag + 1):
        Y, X = _design(values, p, trend, maxlag)
        X = np.hstack([X, extra])
        sigma = _ml_sigma(Y, X)
        det = float(np.linalg.det(sigma))
        ll = -0.5 * T * k * (1.0 + np.log(2.0 * np.pi)) - 0.5 * T * np.log(det)
        per_eq = k * p + deterministic
        total = k * per_eq
        lr = 2.0 * (ll - previous) if p else np.nan
        rows.append(
            {
                "LL": ll,
                "LR": lr,
                "df": float(k * k) if p else np.nan,
                "p": float(stats.chi2.sf(lr, k * k)) if p else np.nan,
                "FPE": det * ((T + per_eq) / (T - per_eq)) ** k,
                "AIC": (-2.0 * ll + 2.0 * total) / T,
                "HQIC": (-2.0 * ll + 2.0 * np.log(np.log(T)) * total) / T,
                "SBIC": (-2.0 * ll + np.log(T) * total) / T,
            }
        )
        previous = ll
    table = pd.DataFrame(rows, index=pd.RangeIndex(maxlag + 1, name="lag"))
    selected = {c: int(table[c].idxmin()) for c in ("FPE", "AIC", "HQIC", "SBIC")}
    rejecting = [p for p in range(maxlag, 0, -1) if table.loc[p, "p"] < alpha]
    selected["LR"] = rejecting[0] if rejecting else 0
    table.attrs.update(selected=selected, n=int(T), variables=variables)
    return table


# ------------------------------------------------------------- after a fit
def _lag_columns(result: Any, lag: int) -> List[int]:
    k = len(result.var_names)
    return list(range((lag - 1) * k, lag * k))


def lag_exclusion(result: Any) -> pd.DataFrame:
    """Wald tests that all endogenous variables at a lag can be dropped,
    equation by equation and in all equations at once (Stata ``varwle``)."""
    _require_fit(result)
    B, bread = result._B, result._XtX_inv
    sigma = np.asarray(result._coef_sigma_u, dtype=float)
    names = list(result.var_names)
    k = len(names)
    rows = []
    for lag in range(1, result.lags + 1):
        cols = _lag_columns(result, lag)
        inv_block = np.linalg.inv(bread[np.ix_(cols, cols)])
        for i, name in enumerate(names):
            b = B[cols, i]
            chi2 = float(b @ inv_block @ b) / sigma[i, i]
            rows.append((name, lag, chi2, k))
        # all equations: Var(vec B_lag) = Sigma (x) bread_block
        stacked = B[cols, :]
        chi2 = float(np.trace(np.linalg.inv(sigma) @ stacked.T @ inv_block @ stacked))
        rows.append(("All", lag, chi2, k * k))
    table = pd.DataFrame(rows, columns=["equation", "lag", "chi2", "df"])
    table["p"] = stats.chi2.sf(table["chi2"], table["df"])
    order = {name: i for i, name in enumerate(names + ["All"])}
    table = table.sort_values(
        ["equation", "lag"], key=lambda s: s.map(order) if s.name == "equation" else s
    )
    return table.reset_index(drop=True)


def lm_autocorrelation(result: Any, lags: int = 2) -> pd.DataFrame:
    """LM test for residual autocorrelation at each lag order (Stata
    ``varlmar``).

    For lag ``s`` the VAR is re-estimated with the residuals lagged ``s``
    periods added (zeros where the lag does not exist). The statistic is
    ``(T - d - 0.5) * log(|Sigma| / |Sigma_s|)`` with ``d`` the number of
    coefficients in one equation of the augmented model, chi-squared with
    ``K^2`` degrees of freedom under no autocorrelation at that order.
    """
    _require_fit(result)
    X, Y = result._X, result._Y
    T, k = Y.shape
    resid = np.asarray(result.residuals, dtype=float)
    det0 = float(np.linalg.det(resid.T @ resid / T))
    rows = []
    for s in range(1, lags + 1):
        if s >= T:
            raise DataInsufficient(
                f"lag {s} is not available with {T} observations.",
                recovery_hint="Lower the number of lags.",
            )
        lagged = np.zeros_like(resid)
        lagged[s:] = resid[:-s]
        aug = np.column_stack([X, lagged])
        det_s = float(np.linalg.det(_ml_sigma(Y, aug)))
        d = aug.shape[1]
        chi2 = (T - d - 0.5) * np.log(det0 / det_s)
        rows.append((s, chi2, k * k, float(stats.chi2.sf(chi2, k * k))))
    return pd.DataFrame(rows, columns=["lag", "chi2", "df", "p"])


def stability(result: Any) -> pd.DataFrame:
    """Eigenvalues of the companion matrix and their moduli (Stata
    ``varstable``). The VAR is stable when every modulus is below one;
    ``attrs['stable']`` says whether it is."""
    _require_fit(result)
    k, p = len(result.var_names), result.lags
    A = result._B[: k * p, :].T  # K x Kp
    companion = np.zeros((k * p, k * p))
    companion[:k, :] = A
    if p > 1:
        companion[k:, :-k] = np.eye(k * (p - 1))
    eig = np.linalg.eigvals(companion)
    order = np.argsort(-np.abs(eig), kind="stable")
    eig = eig[order]
    table = pd.DataFrame(
        {"real": eig.real, "imaginary": eig.imag, "modulus": np.abs(eig)}
    )
    table.attrs["stable"] = bool(np.all(np.abs(eig) < 1.0))
    return table


def granger_table(result: Any) -> pd.DataFrame:
    """Granger-causality Wald tests for every equation (Stata
    ``vargranger``): each other variable's lags, and all of them together."""
    _require_fit(result)
    B, bread = result._B, result._XtX_inv
    sigma = np.asarray(result._coef_sigma_u, dtype=float)
    names = list(result.var_names)
    k, p = len(names), result.lags
    rows = []
    for i, eq in enumerate(names):
        others = [j for j in range(k) if j != i]
        groups = [(names[j], [lag * k + j for lag in range(p)]) for j in others]
        groups.append(("ALL", [lag * k + j for lag in range(p) for j in others]))
        for label, cols in groups:
            b = B[cols, i]
            chi2 = float(b @ np.linalg.inv(bread[np.ix_(cols, cols)]) @ b) / sigma[i, i]
            rows.append(
                (eq, label, chi2, len(cols), float(stats.chi2.sf(chi2, len(cols))))
            )
    return pd.DataFrame(rows, columns=["equation", "excluded", "chi2", "df", "p"])


def equation_table(result: Any) -> pd.DataFrame:
    """Per-equation fit: number of coefficients, root MSE (divisor ``T -
    coefficients``), R-squared and the Wald chi-squared that every
    coefficient except the constant is zero."""
    _require_fit(result)
    X, Y, B = result._X, result._Y, result._B
    T, m = X.shape
    resid = np.asarray(result.residuals, dtype=float)
    sigma = np.asarray(result._coef_sigma_u, dtype=float)
    trend = result._trend or "c"
    constant = result._k * result.lags if trend in ("c", "ct") else -1
    slopes = [j for j in range(m) if j != constant]
    inv_block = np.linalg.inv(result._XtX_inv[np.ix_(slopes, slopes)])
    rows = []
    for i, name in enumerate(result.var_names):
        rss = float(resid[:, i] @ resid[:, i])
        y = Y[:, i]
        tss = float(((y - y.mean()) ** 2).sum()) if trend != "n" else float(y @ y)
        b = B[slopes, i]
        chi2 = float(b @ inv_block @ b) / sigma[i, i]
        rows.append(
            {
                "equation": name,
                "parms": m,
                "rmse": float(np.sqrt(rss / (T - m))),
                "r2": 1.0 - rss / tss,
                "chi2": chi2,
                "p": float(stats.chi2.sf(chi2, len(slopes))),
            }
        )
    return pd.DataFrame(rows).set_index("equation")


def forecast(result: Any, steps: int = 1, alpha: float = 0.05) -> pd.DataFrame:
    """Dynamic forecasts ``steps`` periods past the estimation sample.

    Each forecast feeds the next. The standard error is the root of the
    forecast-error variance ``sum_i Phi_i Sigma Phi_i'`` built from the
    moving-average coefficients; it treats the VAR coefficients as known, so
    it understates the uncertainty in short samples.

    Returns a DataFrame indexed by horizon ``1 .. steps`` with, for each
    variable, the columns ``<name>``, ``<name>_se``, ``<name>_lower`` and
    ``<name>_upper``.
    """
    _require_fit(result)
    if steps < 1:
        raise MethodIncompatibility(
            "forecast: steps must be at least 1.", recovery_hint="Use steps=1."
        )
    if getattr(result, "_exog", None):
        raise MethodIncompatibility(
            "forecast: the VAR has exogenous variables "
            f"({', '.join(result._exog)}); their future values are not known.",
            recovery_hint="Refit without exog=, or forecast from the "
            "coefficients with your own path for the exogenous variables.",
        )
    names = list(result.var_names)
    k, p = len(names), result.lags
    B = result._B
    trend = result._trend or "c"
    T = result._Y.shape[0]
    history = [result._levels[-lag] for lag in range(1, p + 1)]  # most recent first
    sigma = np.asarray(result.sigma_u, dtype=float)
    A = [B[lag * k : (lag + 1) * k, :].T for lag in range(p)]
    phi = [np.eye(k)]
    mse = np.zeros((k, k))
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    rows = []
    for h in range(1, steps + 1):
        x = np.concatenate(history[:p]) if p else np.zeros(0)
        if trend in ("c", "ct"):
            x = np.append(x, 1.0)
        if trend == "ct":
            x = np.append(x, float(T + h))
        point = x @ B
        mse = mse + phi[h - 1] @ sigma @ phi[h - 1].T
        nxt = np.zeros((k, k))
        for j in range(1, min(h, p) + 1):
            nxt += phi[h - j] @ A[j - 1]
        phi.append(nxt)
        se = np.sqrt(np.diag(mse))
        row: Dict[str, float] = {}
        for i, name in enumerate(names):
            row[name] = float(point[i])
            row[f"{name}_se"] = float(se[i])
            row[f"{name}_lower"] = float(point[i] - z * se[i])
            row[f"{name}_upper"] = float(point[i] + z * se[i])
        rows.append(row)
        history.insert(0, point)
    return pd.DataFrame(rows, index=pd.RangeIndex(1, steps + 1, name="horizon"))
