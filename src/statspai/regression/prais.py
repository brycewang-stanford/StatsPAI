"""Linear regression with AR(1) errors: Prais-Winsten and Cochrane-Orcutt.

The model is ``y_t = x_t'b + u_t`` with ``u_t = rho * u_{t-1} + e_t``. Both
estimators quasi-difference the data with an estimate of ``rho`` and run OLS
on the result; they differ in the first observation. Cochrane-Orcutt drops
it. Prais-Winsten keeps it, scaled by ``sqrt(1 - rho^2)``, which is the
feasible GLS estimator and matters in short series.

``rho`` is re-estimated from the residuals of the last fit until it stops
changing (or once, with ``twostep=True``).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import EconometricResults
from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["prais"]

_RHOTYPES = ("regress", "freg", "tscorr", "dw", "theil", "nagar")


def _rho(e: np.ndarray, kind: str, k: int) -> float:
    """The estimate of rho from OLS-scale residuals ``e``."""
    n = e.size
    lag, cur = e[:-1], e[1:]
    if kind == "regress":  # e_t on e_{t-1}
        return float(cur @ lag / (lag @ lag))
    if kind == "freg":  # e_t on e_{t+1}
        return float(lag @ cur / (cur @ cur))
    tscorr = float(cur @ lag / (e @ e))
    if kind == "tscorr":
        return tscorr
    if kind == "theil":
        # degrees-of-freedom adjustment over the n - 1 residual pairs
        return tscorr * (n - 1 - k) / (n - 1)
    dw = float(np.sum(np.diff(e) ** 2) / (e @ e))
    rho_dw = 1.0 - dw / 2.0
    if kind == "dw":
        return rho_dw
    # nagar: Stata counts rho among the parameters (k + 1)
    return (rho_dw * n**2 + (k + 1) ** 2) / (n**2 - (k + 1) ** 2)


def _transform(a: np.ndarray, rho: float, keep_first: bool) -> np.ndarray:
    out = a[1:] - rho * a[:-1]
    if not keep_first:
        return out
    first = np.sqrt(1.0 - rho**2) * a[:1]
    return np.concatenate([first, out], axis=0)


def prais(
    formula: str,
    data: pd.DataFrame,
    *,
    method: str = "prais",
    rhotype: str = "regress",
    twostep: bool = False,
    time: Optional[str] = None,
    vce: str = "ols",
    tol: float = 1e-6,
    maxiter: int = 100,
    alpha: float = 0.05,
) -> EconometricResults:
    """Regression with first-order autoregressive errors.

    Parameters
    ----------
    formula : str
        ``"y ~ x1 + x2"``.
    data : pandas.DataFrame
        One row per period, with no missing value in the model variables and
        no gap in time: the transformation uses the row above as the previous
        period.
    method : {'prais', 'corc'}, default 'prais'
        ``'prais'`` is Prais-Winsten, which keeps the first observation;
        ``'corc'`` is Cochrane-Orcutt, which drops it.
    rhotype : {'regress', 'freg', 'tscorr', 'dw', 'theil', 'nagar'}
        How ``rho`` is computed from the residuals: the slope of ``e_t`` on
        ``e_{t-1}`` (default), the slope of ``e_t`` on ``e_{t+1}``, the
        first autocorrelation, ``1 - d/2`` from the Durbin-Watson statistic,
        or Theil's and Nagar's small-sample adjustments of the last two.
    twostep : bool, default False
        Stop after the first estimate of ``rho`` instead of iterating to
        convergence.
    time : str, optional
        Column to sort by first. Without it the rows are taken as ordered.
    vce : {'ols', 'robust', 'hc2', 'hc3'}, default 'ols'
        Covariance of the transformed regression. ``'robust'`` is HC1.
    tol : float, default 1e-6
        Iteration stops when ``rho`` changes by less than this.
    maxiter : int, default 100
        Iteration limit.
    alpha : float, default 0.05
        One minus the confidence level.

    Returns
    -------
    EconometricResults
        Coefficients and standard errors of the transformed regression.
        ``model_info`` holds ``rho``, ``dw_original``, ``dw_transformed``,
        ``iterations``, ``rho_path`` and ``converged``; ``diagnostics``
        holds the transformed regression's R-squared, F and root MSE.

    Notes
    -----
    The constant is the coefficient on the transformed column of ones, so it
    is on the scale of the original model. R-squared and F describe the
    transformed regression and are not comparable with those of OLS on the
    levels. This follows Stata's ``prais``; R's ``prais::prais_winsten``
    gives the same coefficients.

    If ``rho`` is close to one the data are near a unit root and the quasi-
    difference removes almost all the level information; test for a unit
    root (:func:`statspai.unitroot`) before reading the estimates.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> x = rng.normal(size=n)
    >>> u = np.zeros(n)
    >>> for t in range(1, n):
    ...     u[t] = 0.6 * u[t - 1] + rng.normal()
    >>> df = pd.DataFrame({"x": x, "y": 1 + 2 * x + u})
    >>> fit = sp.prais("y ~ x", data=df)
    >>> bool(abs(fit.params["x"] - 2) < 0.2)
    True
    >>> bool(0.4 < fit.model_info["rho"] < 0.8)
    True

    References
    ----------
    cochrane1949application
    """
    from .ols import regress

    method = method.lower()
    if method not in ("prais", "corc"):
        raise MethodIncompatibility(
            f"sp.prais: method={method!r} is not 'prais' or 'corc'.",
            recovery_hint="Use method='prais' (Prais-Winsten) or 'corc'.",
        )
    if rhotype not in _RHOTYPES:
        raise MethodIncompatibility(
            f"sp.prais: rhotype={rhotype!r} is not one of {', '.join(_RHOTYPES)}.",
            recovery_hint="Use rhotype='regress' (the default).",
        )
    vce = {"robust": "hc1"}.get(vce.lower(), vce.lower())
    if vce not in ("ols", "hc1", "hc2", "hc3"):
        raise MethodIncompatibility(
            f"sp.prais: vce={vce!r} is not 'ols', 'robust', 'hc2' or 'hc3'.",
            recovery_hint="Use vce='ols' or vce='robust'.",
        )
    if time is not None:
        if time not in data.columns:
            raise MethodIncompatibility(
                f"sp.prais: time={time!r} is not a column.",
                recovery_hint="Pass the name of the time variable.",
            )
        data = data.sort_values(time, kind="stable")

    ols = regress(formula, data=data)
    info = ols.data_info
    X = np.asarray(info["X"], dtype=float)
    y = np.asarray(info["y"], dtype=float)
    names = [str(v) for v in info["var_names"]]
    n, k = X.shape
    if n != len(data):
        raise DataInsufficient(
            f"sp.prais: {len(data) - n} row(s) have a missing value in the "
            "model variables, which leaves a gap in the series.",
            recovery_hint="Drop or fill the incomplete periods so that the "
            "rows are consecutive.",
        )
    if n - (method == "corc") <= k:
        raise DataInsufficient(
            "sp.prais: not enough observations for the transformed regression.",
            recovery_hint="Use fewer regressors or a longer series.",
        )

    keep_first = method == "prais"
    beta = np.asarray(ols.params, dtype=float)
    e_ols = y - X @ beta
    dw_original = float(np.sum(np.diff(e_ols) ** 2) / (e_ols @ e_ols))
    rho, path, converged = 0.0, [0.0], False
    for _ in range(1 if twostep else maxiter):
        new = _rho(y - X @ beta, rhotype, k)
        if not np.isfinite(new) or abs(new) >= 1:
            raise MethodIncompatibility(
                f"sp.prais: the estimate of rho is {new:.4f}; the "
                "transformation needs |rho| < 1.",
                recovery_hint="The errors look non-stationary: difference the "
                "data, or test for a unit root with sp.unitroot.",
            )
        Xs, ys = _transform(X, new, keep_first), _transform(y, new, keep_first)
        beta = np.linalg.lstsq(Xs, ys, rcond=None)[0]
        path.append(float(new))
        done = abs(new - rho) < tol
        rho = float(new)
        if done:
            converged = True
            break
    converged = converged or twostep

    Xs, ys = _transform(X, rho, keep_first), _transform(y, rho, keep_first)
    m = Xs.shape[0]
    resid = ys - Xs @ beta
    bread = np.linalg.inv(Xs.T @ Xs)
    rss = float(resid @ resid)
    df_resid = m - k
    if vce == "ols":
        cov = bread * rss / df_resid
    else:
        h = np.einsum("ij,jk,ik->i", Xs, bread, Xs)
        scale = {"hc1": np.full(m, m / df_resid), "hc2": 1 / (1 - h)}.get(
            vce, 1 / (1 - h) ** 2
        )
        meat = (Xs * (resid**2 * scale)[:, None]).T @ Xs
        cov = bread @ meat @ bread
    se = np.sqrt(np.diag(cov))

    # Fit statistics of the transformed regression: the total sum of squares
    # is taken about the mean of the transformed outcome, as Stata does.
    const = next((j for j in range(k) if np.ptp(X[:, j]) == 0 and X[0, j] != 0), None)
    if const is not None:
        tss = float(((ys - ys.mean()) ** 2).sum())
        df_model = k - 1
    else:
        tss, df_model = float(ys @ ys), k
    r2 = 1.0 - rss / tss if tss > 0 else np.nan
    r2_adj = 1.0 - (rss / df_resid) / (tss / (m - (const is not None)))
    if vce == "ols":
        f_stat = ((tss - rss) / df_model) / (rss / df_resid) if df_model else np.nan
    else:
        tested = [j for j in range(k) if j != const]
        b_t = beta[tested]
        f_stat = float(b_t @ np.linalg.solve(cov[np.ix_(tested, tested)], b_t)) / len(
            tested
        )
    f_p = float(stats.f.sf(f_stat, df_model, df_resid)) if df_model else np.nan
    e_final = y - X @ beta
    u = e_final[1:] - rho * e_final[:-1]
    if keep_first:
        u = np.concatenate([[np.sqrt(1.0 - rho**2) * e_final[0]], u])
    dw_transformed = float(np.sum(np.diff(u) ** 2) / (u @ u))

    index = pd.Index(names)
    model_info: Dict[str, Any] = {
        "model_type": "Prais-Winsten" if keep_first else "Cochrane-Orcutt",
        "method": "AR(1) feasible GLS, " + ("two-step" if twostep else "iterated"),
        "rho": rho,
        "rhotype": rhotype,
        "rho_path": path,
        "iterations": len(path) - 1,
        "converged": converged,
        "dw_original": dw_original,
        "dw_transformed": dw_transformed,
        "robust": "nonrobust" if vce == "ols" else vce,
        "alpha": alpha,
    }
    data_info: Dict[str, Any] = {
        "nobs": m,
        "df_model": df_model,
        "df_resid": df_resid,
        "dependent_var": info.get("dependent_var"),
        "var_names": names,
        "var_cov": cov,
        "X": Xs,
        "y": ys,
        "residuals": resid,
        "fitted_values": Xs @ beta,
        "rss": rss,
        "tss": tss,
    }
    diagnostics = {
        "R-squared": r2,
        "Adj. R-squared": r2_adj,
        "F-statistic": f_stat,
        "Prob (F-statistic)": f_p,
        "Root MSE": float(np.sqrt(rss / df_resid)),
        "rho": rho,
        "Durbin-Watson (original)": dw_original,
        "Durbin-Watson (transformed)": dw_transformed,
    }
    if not converged:
        import warnings

        from ..exceptions import ConvergenceWarning

        warnings.warn(
            f"sp.prais: rho did not converge in {maxiter} iterations "
            f"(last change {abs(path[-1] - path[-2]):.2e}).",
            ConvergenceWarning,
            stacklevel=2,
        )
    return EconometricResults(
        params=pd.Series(beta, index=index),
        std_errors=pd.Series(se, index=index),
        model_info=model_info,
        data_info=data_info,
        diagnostics=diagnostics,
    )
