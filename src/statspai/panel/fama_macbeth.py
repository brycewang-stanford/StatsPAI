"""
Fama-MacBeth two-step regression.

``sp.fama_macbeth`` runs the same cross-sectional regression in every
period and reports the time-series mean of the coefficients, with the
standard error taken from how much the per-period estimates vary
[@fama1973risk]. The standard error is robust to any correlation of the
errors *within* a period (the whole cross-section moves together) and, in
its plain form, assumes the per-period estimates are uncorrelated *over
time*. A persistent firm effect breaks that assumption and the plain
standard error is then too small [@petersen2009estimating;
@gow2010correcting]; ``lags=`` applies the Newey-West correction to the
coefficient series, which handles serial correlation that dies out but not
a permanent firm effect. Cluster by firm, or by firm and period, for that.

Reference implementations: R ``plm::pmg`` (mean-groups over the time
index), Stata ``xtfmb`` and ``asreg, fmb``.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..core.results import EconometricResults
from ..core.utils import create_design_matrices
from ..exceptions import DataInsufficient, MethodIncompatibility


def fama_macbeth(
    formula: str,
    data: pd.DataFrame,
    time: str,
    *,
    lags: int = 0,
    alpha: float = 0.05,
) -> EconometricResults:
    """Fama-MacBeth regression: cross-sections by period, then averages.

    Step one fits ``formula`` by OLS separately in each period. Step two
    averages the ``T`` coefficient vectors; their sample covariance divided
    by ``T`` is the covariance of the average.

    Parameters
    ----------
    formula : str
        Regression formula, e.g. ``"ret ~ beta + size"``.
    data : pd.DataFrame
        Long-format panel, one row per unit and period.
    time : str
        The period column. One cross-sectional regression is run for each
        of its values.
    lags : int, default 0
        Newey-West lags for the coefficient series. With ``lags=0`` the
        covariance is the textbook ``S / T``, ``S`` the sample covariance
        of the per-period estimates. With ``lags=L`` the Bartlett-weighted
        autocovariances up to lag ``L`` are added and the result carries
        the ``T / (T - 1)`` factor, so that ``lags=0`` is the same number
        either way: Stata ``xtfmb, lag(L)`` and
        ``sandwich::NeweyWest(lm(b ~ 1), lag = L, prewhite = FALSE,
        adjust = TRUE)`` on each coefficient series.
    alpha : float, default 0.05
        Significance level of the confidence intervals.

    Returns
    -------
    EconometricResults
        ``params`` are the time-series means and inference uses a t
        distribution with ``T - 1`` degrees of freedom (as ``xtfmb``;
        ``plm::pmg`` prints normal p-values for the same standard errors).
        ``model_info['period_coefs']`` holds the per-period estimates
        (indexed by period, with ``nobs`` and ``r2``),
        ``model_info['avg_r2']`` the average cross-sectional R-squared.

    Notes
    -----
    The periods need not hold the same units. A period with no more
    observations than coefficients, or whose regressors are collinear,
    cannot be estimated; it is left out with a warning and listed in
    ``model_info['skipped_periods']``.

    With ``lags > 0`` the full Newey-West covariance matrix is reported.
    ``xtfmb`` sets its off-diagonal elements to zero; the standard errors
    agree, a joint test does not.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"firm": np.repeat(np.arange(200), 10),
    ...                    "year": np.tile(np.arange(10), 200)})
    >>> df["x"] = rng.normal(size=len(df))
    >>> df["y"] = df["x"] + rng.normal(size=10)[df["year"]] + rng.normal(size=len(df))
    >>> res = sp.fama_macbeth("y ~ x", df, time="year")
    >>> int(res.model_info["n_periods"])
    10
    >>> bool(abs(res.params["x"] - 1) < 0.1)
    True

    References
    ----------
    fama1973risk, petersen2009estimating, gow2010correcting,
    newey1987simple
    """
    if not isinstance(data, pd.DataFrame):
        raise MethodIncompatibility("fama_macbeth: data must be a pandas DataFrame.")
    if time not in data.columns:
        raise MethodIncompatibility(
            f"fama_macbeth: time={time!r} is not a column in the data."
        )
    if isinstance(lags, bool) or int(lags) != lags or lags < 0:
        raise MethodIncompatibility(
            f"fama_macbeth: lags must be a non-negative integer, got {lags!r}."
        )
    lags = int(lags)

    # one design for the whole panel, so every period has the same columns
    # (a factor level absent from one period is a collinear column there)
    y_df, X_df = create_design_matrices(formula, data)
    names = [str(c) for c in X_df.columns]
    period = data[time].reindex(y_df.index)
    if period.isna().any():
        keep = period.notna().to_numpy()
        y_df, X_df, period = y_df[keep], X_df[keep], period[keep]
    y = np.asarray(y_df, dtype=float).ravel()
    X = np.asarray(X_df, dtype=float)
    k = X.shape[1]

    rows: List[Dict[str, Any]] = []
    skipped: List[Any] = []
    codes, uniques = pd.factorize(period, sort=True)
    for code, label in enumerate(uniques):
        mask = codes == code
        Xt, yt = X[mask], y[mask]
        n_t = int(mask.sum())
        if n_t <= k or np.linalg.matrix_rank(Xt) < k:
            skipped.append(label)
            continue
        b, *_ = np.linalg.lstsq(Xt, yt, rcond=None)
        resid = yt - Xt @ b
        tss = float(np.sum((yt - yt.mean()) ** 2))
        r2 = 1.0 - float(resid @ resid) / tss if tss > 0 else np.nan
        rows.append({"period": label, "b": b, "nobs": n_t, "r2": r2})

    T = len(rows)
    if T < 2:
        raise DataInsufficient(
            f"fama_macbeth: {T} period(s) could be estimated; the standard "
            "error is the variation of the estimates across periods and "
            "needs at least two.",
            recovery_hint="Check time= and that each period has more "
            "observations than regressors.",
            diagnostics={"n_periods": T, "skipped_periods": list(skipped)},
        )
    if skipped:
        warnings.warn(
            f"fama_macbeth: {len(skipped)} period(s) left out because the "
            f"cross-section could not identify the {k} coefficients "
            f"(first: {skipped[:5]}).",
            UserWarning,
            stacklevel=2,
        )
    if lags >= T:
        raise DataInsufficient(
            f"fama_macbeth: lags={lags} needs more than {T} periods.",
            recovery_hint="Lower lags.",
        )

    B = np.vstack([r["b"] for r in rows])
    beta = B.mean(axis=0)
    dev = B - beta
    S = dev.T @ dev
    for j in range(1, lags + 1):
        gamma = dev[j:].T @ dev[:-j]
        S = S + (1.0 - j / (lags + 1.0)) * (gamma + gamma.T)
    cov = S / (T * (T - 1.0))
    se = np.sqrt(np.diag(cov))

    period_coefs = pd.DataFrame(
        B, index=pd.Index([r["period"] for r in rows], name=time), columns=names
    )
    period_coefs["nobs"] = [r["nobs"] for r in rows]
    period_coefs["r2"] = [r["r2"] for r in rows]
    nobs = int(period_coefs["nobs"].sum())
    avg_r2 = float(np.nanmean(period_coefs["r2"]))
    fitted = X @ beta

    model_info: Dict[str, Any] = {
        "model_type": "Fama-MacBeth",
        "method": "Fama-MacBeth two-step"
        + (f" (Newey-West, {lags} lag{'s' if lags != 1 else ''})" if lags else ""),
        "robust": "newey-west" if lags else "fama-macbeth",
        "vcov_type": "Fama-MacBeth" + (f" Newey-West({lags})" if lags else ""),
        "time": time,
        "lags": lags,
        "n_periods": T,
        "avg_r2": avg_r2,
        "period_coefs": period_coefs,
        "skipped_periods": list(skipped),
        "vcov": pd.DataFrame(cov, index=names, columns=names),
        "formula": formula,
        "alpha": alpha,
    }
    data_info: Dict[str, Any] = {
        "nobs": nobs,
        "dependent_var": str(y_df.columns[-1]) if hasattr(y_df, "columns") else "y",
        "df_resid": T - 1,
        "df_model": k - 1 if "Intercept" in names else k,
        "var_cov": cov,
        "fitted_values": fitted,
        "residuals": y - fitted,
    }
    diagnostics: Dict[str, Any] = {
        "Periods": T,
        "Avg. R-squared": avg_r2,
        "Obs. per period (min)": int(period_coefs["nobs"].min()),
        "Obs. per period (max)": int(period_coefs["nobs"].max()),
    }
    if lags:
        diagnostics["Newey-West lags"] = lags
    return EconometricResults(
        params=pd.Series(beta, index=names),
        std_errors=pd.Series(se, index=names),
        model_info=model_info,
        data_info=data_info,
        diagnostics=diagnostics,
    )
