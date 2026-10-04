"""Small panel-data tools that sit around the estimators.

- :func:`xtsum`: overall, between and within variation of each variable;
- :func:`xtserial`: Wooldridge's test for first-order serial correlation;
- :func:`xtoverid`: the cluster-robust test of random against fixed
  effects (a Hausman test that stays valid under heteroskedasticity and
  within-panel correlation);
- :func:`xt_statistics`: the variance components and the three R-squared
  that Stata's ``xtreg`` reports next to a fixed- or random-effects fit.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["xtsum", "xtserial", "xtoverid", "xt_statistics"]


def _columns(data: pd.DataFrame, names: Sequence[str], what: str) -> List[str]:
    names = [names] if isinstance(names, str) else list(names)
    unknown = [v for v in names if v not in data.columns]
    if unknown or not names:
        raise MethodIncompatibility(
            (
                f"{what}: column(s) {unknown} are not in the data."
                if unknown
                else f"{what}: no variables given."
            ),
            recovery_hint="Check the variable names.",
        )
    return names


def _group_mean(values: np.ndarray, codes: np.ndarray, n: int) -> np.ndarray:
    total = np.zeros((n,) + values.shape[1:])
    np.add.at(total, codes, values)
    counts = np.bincount(codes, minlength=n).astype(float)
    return total / counts.reshape((-1,) + (1,) * (values.ndim - 1))


# ---------------------------------------------------------------- xtsum
def xtsum(
    data: pd.DataFrame,
    variables: Optional[Sequence[str]] = None,
    *,
    id: str,
) -> pd.DataFrame:
    """Overall, between and within summary statistics of panel variables.

    Parameters
    ----------
    data : pandas.DataFrame
        Panel data in long format.
    variables : sequence of str, optional
        Variables to describe. Default: every numeric column except ``id``.
    id : str
        Panel identifier.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``(variable, component)`` with ``component`` in
        ``overall`` / ``between`` / ``within`` and columns ``mean``, ``sd``,
        ``min``, ``max``, ``obs``. ``obs`` is the number of observations
        (overall), of panels (between) and the average number of periods
        per panel (within).

    Notes
    -----
    The between statistics describe the panel means ``xbar_i``. The within
    statistics describe ``x_it - xbar_i + xbar``, the deviation from the
    panel mean with the grand mean added back so that it is on the scale of
    ``x``; its standard deviation uses the divisor ``N - 1``. This is Stata's
    ``xtsum``.

    A variable with almost no within variation (a small within ``sd``) is
    barely identified in a fixed-effects regression, whatever its overall
    variation.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.dgp_panel(n_units=30, n_periods=6, seed=0)
    >>> table = sp.xtsum(df, ["y", "x"], id="unit")
    >>> list(table.columns)
    ['mean', 'sd', 'min', 'max', 'obs']
    >>> float(table.loc[("y", "between"), "obs"])
    30.0
    """
    if id not in data.columns:
        raise MethodIncompatibility(
            f"sp.xtsum: id={id!r} is not a column.",
            recovery_hint="Pass the panel identifier.",
        )
    if variables is None:
        variables = [
            c for c in data.select_dtypes(include=[np.number]).columns if c != id
        ]
    names = _columns(data, variables, "sp.xtsum")
    rows: Dict[Any, Dict[str, float]] = {}
    for name in names:
        frame = data[[id, name]].dropna()
        x = frame[name].to_numpy(dtype=float)
        if x.size == 0:
            raise DataInsufficient(
                f"sp.xtsum: {name!r} has no observed value.",
                recovery_hint="Drop the variable.",
            )
        codes, _ = pd.factorize(frame[id], sort=True)
        n = int(codes.max()) + 1
        means = _group_mean(x, codes, n)
        within = x - means[codes] + x.mean()
        rows[(name, "overall")] = {
            "mean": float(x.mean()),
            "sd": float(x.std(ddof=1)) if x.size > 1 else np.nan,
            "min": float(x.min()),
            "max": float(x.max()),
            "obs": float(x.size),
        }
        rows[(name, "between")] = {
            "mean": np.nan,
            "sd": float(means.std(ddof=1)) if n > 1 else np.nan,
            "min": float(means.min()),
            "max": float(means.max()),
            "obs": float(n),
        }
        rows[(name, "within")] = {
            "mean": np.nan,
            "sd": (
                float(np.sqrt(((within - x.mean()) ** 2).sum() / (x.size - 1)))
                if x.size > 1
                else np.nan
            ),
            "min": float(within.min()),
            "max": float(within.max()),
            "obs": float(x.size / n),
        }
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index = pd.MultiIndex.from_tuples(
        table.index, names=["variable", "component"]
    )
    return table[["mean", "sd", "min", "max", "obs"]]


# ------------------------------------------------------------ xtserial
def xtserial(
    data: pd.DataFrame,
    y: str,
    x: Sequence[str],
    *,
    id: str,
    time: str,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Wooldridge test for first-order serial correlation in panel errors.

    The model is estimated in first differences, which removes the unit
    effect. If the level errors are serially uncorrelated, the differenced
    errors have first-order autocorrelation ``-0.5``; the test regresses the
    differenced residuals on their lag and tests that coefficient against
    ``-0.5`` with a panel-clustered variance.

    Parameters
    ----------
    data : pandas.DataFrame
        Panel data in long format.
    y : str
        Outcome.
    x : sequence of str
        Regressors (time-varying; a time-invariant one differences to zero).
    id, time : str
        Panel and time identifiers. ``time`` must count periods (integers):
        a difference is taken only between consecutive periods.
    alpha : float, default 0.05
        Level used in the interpretation.

    Returns
    -------
    dict
        ``statistic`` (F), ``df`` ``(1, G - 1)``, ``pvalue``, ``rho`` (the
        coefficient of the residual on its lag), ``n_obs`` and ``params`` /
        ``std_errors`` of the first-difference regression.

    Notes
    -----
    Rejection means the usual fixed- or random-effects standard errors are
    wrong (clustering on the panel repairs them) and that a dynamic model
    may be called for. The test needs at least three consecutive periods
    per panel and works with unbalanced panels and gaps.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for firm in range(80):
    ...     e = 0.0
    ...     for year in range(8):
    ...         e = 0.7 * e + rng.normal()  # AR(1) errors
    ...         x = rng.normal()
    ...         rows.append((firm, year, 1 + x + e, x))
    >>> df = pd.DataFrame(rows, columns=["firm", "year", "y", "x"])
    >>> out = sp.xtserial(df, "y", ["x"], id="firm", time="year")
    >>> out["df"], bool(out["pvalue"] < 0.01)
    ((1, 79), True)

    References
    ----------
    wooldridge2010econometric; drukker2003testing
    """
    xs = _columns(data, x, "sp.xtserial")
    _columns(data, [y, id, time], "sp.xtserial")
    frame = data[[id, time, y] + xs].dropna().sort_values([id, time], kind="stable")
    t = frame[time].to_numpy(dtype=float)
    if not np.all(t == np.round(t)):
        raise MethodIncompatibility(
            f"sp.xtserial: time={time!r} must count periods (integers).",
            recovery_hint="Convert dates to a period index first.",
        )
    codes, _ = pd.factorize(frame[id], sort=True)
    values = frame[[y] + xs].to_numpy(dtype=float)

    def difference(a: np.ndarray, codes_: np.ndarray, t_: np.ndarray) -> tuple:
        ok = (codes_[1:] == codes_[:-1]) & (t_[1:] - t_[:-1] == 1)
        return a[1:][ok] - a[:-1][ok], codes_[1:][ok], t_[1:][ok]

    d, d_codes, d_time = difference(values, codes, t)
    if d.shape[0] <= len(xs):
        raise DataInsufficient(
            "sp.xtserial: too few consecutive observations to difference.",
            recovery_hint="The test needs at least three consecutive periods "
            "per panel.",
        )
    dy, dX = d[:, 0], d[:, 1:]
    beta = np.linalg.lstsq(dX, dy, rcond=None)[0]
    resid = dy - dX @ beta
    # lag of the differenced residual, within panel and at consecutive periods
    ok = (d_codes[1:] == d_codes[:-1]) & (d_time[1:] - d_time[:-1] == 1)
    e, e_lag, g = resid[1:][ok], resid[:-1][ok], d_codes[1:][ok]
    groups = np.unique(g)
    if e.size < 2 or groups.size < 2:
        raise DataInsufficient(
            "sp.xtserial: no panel has three consecutive periods.",
            recovery_hint="The test cannot be computed on this panel.",
        )
    sxx = float(e_lag @ e_lag)
    rho = float(e_lag @ e) / sxx
    u = e - rho * e_lag
    scores = np.zeros(int(g.max()) + 1)
    np.add.at(scores, g, e_lag * u)
    G = groups.size
    # one regressor and no constant: Stata's (N-1)/(N-K) factor is one
    var = (scores @ scores) / sxx**2 * (G / (G - 1.0))
    stat = (rho + 0.5) ** 2 / var
    pvalue = float(stats.f.sf(stat, 1, G - 1))

    # first-difference regression with panel-clustered standard errors
    k = dX.shape[1]
    bread = np.linalg.inv(dX.T @ dX)
    panel_scores = np.zeros((int(d_codes.max()) + 1, k))
    np.add.at(panel_scores, d_codes, dX * resid[:, None])
    Gd = np.unique(d_codes).size
    n_d = dX.shape[0]
    cov = (
        bread
        @ (panel_scores.T @ panel_scores)
        @ bread
        * (Gd / (Gd - 1.0))
        * ((n_d - 1.0) / (n_d - k))
    )
    reject = pvalue < alpha
    return {
        "test": "Wooldridge test for autocorrelation in panel data",
        "H0": "no first-order autocorrelation",
        "statistic": float(stat),
        "df": (1, int(G - 1)),
        "pvalue": pvalue,
        "rho": rho,
        "n_obs": int(n_d),
        "params": pd.Series(beta, index=[f"D.{v}" for v in xs]),
        "std_errors": pd.Series(np.sqrt(np.diag(cov)), index=[f"D.{v}" for v in xs]),
        "interpretation": (
            f"F(1, {G - 1}) = {stat:.3f}, p = {pvalue:.4f}. "
            + (
                "Reject H0: the errors are serially correlated; cluster the "
                "standard errors on the panel."
                if reject
                else "Cannot reject H0: no evidence of first-order serial "
                "correlation."
            )
        ),
    }


# ----------------------------------------------------- variance components
def _complete(data: pd.DataFrame, y: str, xs: List[str], id: str) -> tuple:
    frame = data[[id, y] + xs].dropna()
    codes, _ = pd.factorize(frame[id], sort=True)
    return (
        frame[y].to_numpy(dtype=float),
        frame[xs].to_numpy(dtype=float),
        codes,
        int(codes.max()) + 1,
    )


def _swamy_arora(
    y: np.ndarray, X: np.ndarray, codes: np.ndarray, n: int, components: bool = True
) -> dict:
    N, K = X.shape
    Ti = np.bincount(codes, minlength=n).astype(float)
    yb, Xb = _group_mean(y, codes, n), _group_mean(X, codes, n)
    yw, Xw = y - yb[codes], X - Xb[codes]
    bw = np.linalg.lstsq(Xw, yw, rcond=None)[0]
    ew = yw - Xw @ bw
    if not components:
        # the fixed-effects statistics need the two transforms only; the
        # between regression behind sigma_u of the random-effects model
        # does not exist with more regressors than panels (year dummies
        # on a short list of firms)
        return {"Ti": Ti, "yb": yb, "Xb": Xb, "yw": yw, "Xw": Xw, "bw": bw, "ew": ew}
    if N - n - K <= 0 or n - K - 1 <= 0:
        raise DataInsufficient(
            "Too few panels or periods for the variance components.",
            recovery_hint="Use fewer regressors.",
        )
    s2e = float(ew @ ew) / (N - n - K)
    Zb = np.column_stack([np.ones(n), Xb])
    eb = yb - Zb @ np.linalg.lstsq(Zb, yb, rcond=None)[0]
    t_harmonic = n / float(np.sum(1.0 / Ti))
    s2u = max(0.0, float(eb @ eb) / (n - K - 1) - s2e / t_harmonic)
    theta = 1.0 - np.sqrt(s2e / (Ti * s2u + s2e))
    return {
        "s2e": s2e, "s2u": s2u, "theta": theta, "Ti": Ti,
        "yb": yb, "Xb": Xb, "yw": yw, "Xw": Xw, "bw": bw, "ew": ew,
    }  # fmt: skip


def _corr2(a: np.ndarray, b: np.ndarray) -> float:
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1] ** 2)


def xt_statistics(
    data: pd.DataFrame,
    y: str,
    x: Sequence[str],
    *,
    id: str,
    params: Any,
    method: str,
    cov: Optional[Any] = None,
) -> Dict[str, Any]:
    """Variance components and R-squared of a static panel fit, as Stata's
    ``xtreg`` reports them.

    ``params`` are the slopes of the fit (a mapping from regressor name to
    coefficient; a constant is ignored), ``method`` is ``'fe'``, ``'re'`` or
    ``'be'`` and ``cov`` the covariance of the slopes (used for the standard
    error of the fixed-effects constant).

    Returns ``r2_within`` / ``r2_between`` / ``r2_overall`` (squared
    correlations of the within-deviated, panel-mean and raw outcome with the
    corresponding linear prediction), and

    - ``'fe'``: ``sigma_e`` (within residuals, ``N - n - K`` degrees of
      freedom), ``sigma_u`` (standard deviation of the estimated unit
      effects), ``rho``, ``corr_u_xb``, and the constant ``cons`` -- the
      average unit effect, ``ybar - xbar'b`` -- with ``cons_se`` and its
      covariance with the slopes ``cons_cov``;
    - ``'re'``: the Swamy-Arora ``sigma_e``, ``sigma_u``, ``rho`` and
      ``theta`` (the share of the panel mean removed by the GLS transform;
      ``theta_min`` / ``theta_max`` differ in an unbalanced panel).
    """
    xs = _columns(data, x, "xt_statistics")
    yv, X, codes, n = _complete(data, y, xs, id)
    b = np.array([float(dict(params)[name]) for name in xs])
    parts = _swamy_arora(yv, X, codes, n, components=method != "fe")
    out: Dict[str, Any] = {
        "r2_within": _corr2(parts["Xw"] @ b, parts["yw"]),
        "r2_between": _corr2(parts["Xb"] @ b, parts["yb"]),
        "r2_overall": _corr2(X @ b, yv),
        "n_obs": int(yv.size),
        "n_groups": int(n),
    }
    if method == "fe":
        N, K = X.shape
        xbar = X.mean(axis=0)
        cons = float(yv.mean() - xbar @ b)
        resid = parts["yw"] - parts["Xw"] @ b
        s2e = float(resid @ resid) / (N - n - K)
        u = parts["yb"] - parts["Xb"] @ b - cons
        s2u = float(u.var(ddof=1)) if n > 1 else float("nan")
        out.update(
            sigma_e=float(np.sqrt(s2e)),
            sigma_u=float(np.sqrt(s2u)),
            rho=s2u / (s2u + s2e),
            corr_u_xb=float(np.corrcoef(u[codes], X @ b)[0, 1]),
            cons=cons,
        )
        if cov is not None:
            V = np.asarray(cov, dtype=float)
            if V.shape == (K, K):
                # classical: the mean residual adds sigma_e^2 / N; with a
                # panel-clustered variance the within residuals sum to zero
                # in every cluster and that term vanishes
                classical = np.allclose(
                    V, s2e * np.linalg.inv(parts["Xw"].T @ parts["Xw"]), rtol=1e-6
                )
                extra = s2e / N if classical else 0.0
                out["cons_se"] = float(np.sqrt(xbar @ V @ xbar + extra))
                out["cons_cov"] = pd.Series(-(V @ xbar), index=xs)
    elif method == "re":
        s2e, s2u = parts["s2e"], parts["s2u"]
        theta = parts["theta"]
        out.update(
            sigma_e=float(np.sqrt(s2e)),
            sigma_u=float(np.sqrt(s2u)),
            rho=s2u / (s2u + s2e) if s2u + s2e > 0 else float("nan"),
            theta=float(np.median(theta)),
            theta_min=float(theta.min()),
            theta_max=float(theta.max()),
        )
    return out


# ------------------------------------------------------------- xtoverid
def xtoverid(
    data: pd.DataFrame,
    y: str,
    x: Sequence[str],
    *,
    id: str,
    cluster: Optional[str] = None,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Robust test of random effects against fixed effects.

    Random effects assumes the regressors are uncorrelated with the unit
    effect; those are over-identifying restrictions that fixed effects does
    not impose. The random-effects equation is re-estimated with the
    regressors in deviation-from-panel-mean form added, and the test is the
    Wald statistic that the added terms are zero, with a cluster-robust
    variance. Unlike the classical Hausman test it remains valid under
    heteroskedasticity and serial correlation, and it is never negative.

    Parameters
    ----------
    data : pandas.DataFrame
        Panel data in long format.
    y : str
        Outcome.
    x : sequence of str
        Regressors. Time-invariant ones are kept in the model but cannot be
        tested (their deviation is zero) and do not count as restrictions.
    id : str
        Panel identifier.
    cluster : str, optional
        Cluster variable. Default: the panel identifier.
    alpha : float, default 0.05
        Level used in the interpretation.

    Returns
    -------
    dict
        ``statistic`` (chi-squared), ``df``, ``pvalue``, ``tested`` (the
        regressors that vary within panels) and ``interpretation``.

    Notes
    -----
    The cluster-robust variance carries the factor ``G/(G-1) *
    (N-1)/(N-K)`` and the statistic is referred to the chi-squared
    distribution, which reproduces the Stata command ``xtoverid`` after
    ``xtreg, re vce(cluster)``. The same hypothesis is tested by the
    Mundlak regression, ``sp.panel(method='mundlak')``.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.dgp_panel(n_units=80, n_periods=6, seed=1)
    >>> out = sp.xtoverid(df, "y", ["x"], id="unit")
    >>> int(out["df"])
    1

    References
    ----------
    arellano1993testing; wooldridge2010econometric
    """
    xs = _columns(data, x, "sp.xtoverid")
    cols = [id, y] + xs + ([cluster] if cluster and cluster != id else [])
    _columns(data, cols, "sp.xtoverid")
    frame = data[cols].dropna()
    yv = frame[y].to_numpy(dtype=float)
    X = frame[xs].to_numpy(dtype=float)
    codes, _ = pd.factorize(frame[id], sort=True)
    n = int(codes.max()) + 1
    parts = _swamy_arora(yv, X, codes, n)
    theta = parts["theta"][codes]
    ys = yv - theta * parts["yb"][codes]
    Xs = np.column_stack([1.0 - theta, X - theta[:, None] * parts["Xb"][codes]])
    Xw = parts["Xw"]
    varying = [j for j in range(X.shape[1]) if np.abs(Xw[:, j]).max() > 1e-10]
    if not varying:
        raise MethodIncompatibility(
            "sp.xtoverid: no regressor varies within panels.",
            recovery_hint="The test compares within and between variation.",
        )
    Z = np.column_stack([Xs, Xw[:, varying]])
    coef = np.linalg.lstsq(Z, ys, rcond=None)[0]
    resid = ys - Z @ coef
    bread = np.linalg.inv(Z.T @ Z)
    g_codes = codes
    if cluster and cluster != id:
        g_codes, _ = pd.factorize(frame[cluster], sort=True)
    G = int(g_codes.max()) + 1
    scores = np.zeros((G, Z.shape[1]))
    np.add.at(scores, g_codes, Z * resid[:, None])
    n_obs, n_coef = Z.shape
    cov = bread @ (scores.T @ scores) @ bread
    cov *= (G / (G - 1.0)) * ((n_obs - 1.0) / (n_obs - n_coef))
    idx = list(range(Xs.shape[1], Z.shape[1]))
    tested = coef[idx]
    stat = float(tested @ np.linalg.solve(cov[np.ix_(idx, idx)], tested))
    df = len(idx)
    pvalue = float(stats.chi2.sf(stat, df))
    return {
        "test": "Test of overidentifying restrictions: fixed vs random effects",
        "H0": "the regressors are uncorrelated with the unit effect (RE is "
        "consistent)",
        "statistic": stat,
        "df": df,
        "pvalue": pvalue,
        "tested": [xs[j] for j in varying],
        "n_clusters": G,
        "interpretation": (
            f"chi2({df}) = {stat:.3f}, p = {pvalue:.4f}. "
            + (
                "Reject H0: use fixed effects."
                if pvalue < alpha
                else "Cannot reject H0: random effects is admissible."
            )
        ),
    }
