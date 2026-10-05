"""Chow test for a structural break at a known date.

The date is chosen before looking at the data, so the statistic has an
ordinary F reference. When the date is searched for, the reference is the
sup-F law instead: :func:`statspai.structural_break` with
``method='sup-f'``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["chow_test"]


def _first_regime_sizes(
    data: pd.DataFrame,
    break_point: Any,
    time: Optional[str],
    n: int,
) -> List[int]:
    """Translate ``break_point`` into sizes of the regimes that end there."""
    points = (
        list(break_point)
        if isinstance(break_point, (list, tuple, np.ndarray, pd.Index))
        else [break_point]
    )
    if not points:
        raise MethodIncompatibility("chow_test: break_point is empty.")
    sizes: List[int] = []
    for bp in points:
        if time is not None:
            if time not in data.columns:
                raise MethodIncompatibility(
                    f"chow_test: time column {time!r} not found.",
                    recovery_hint="Check the column name.",
                )
            col = data[time]
            value = pd.Timestamp(bp) if np.issubdtype(col.dtype, np.datetime64) else bp
            size = int((col < value).sum())
        elif isinstance(bp, (int, np.integer)) and not isinstance(bp, bool):
            size = int(bp)
        else:
            idx = data.index
            value = pd.Timestamp(bp) if isinstance(idx, pd.DatetimeIndex) else bp
            try:
                size = int((idx < value).sum())
            except TypeError as exc:
                raise MethodIncompatibility(
                    f"chow_test: break_point {bp!r} cannot be compared with "
                    "the index of data.",
                    recovery_hint=(
                        "Pass the number of observations in the first regime, "
                        "or name the time column with time=."
                    ),
                ) from exc
        sizes.append(size)
    if sorted(set(sizes)) != sizes:
        raise MethodIncompatibility(
            "chow_test: break points must be distinct and in increasing order.",
            diagnostics={"first_regime_sizes": sizes},
        )
    if sizes[0] <= 0 or sizes[-1] >= n:
        raise MethodIncompatibility(
            "chow_test: every break point must leave observations on both "
            "sides of it.",
            diagnostics={"first_regime_sizes": sizes, "n_obs": n},
        )
    return sizes


def chow_test(
    data: pd.DataFrame,
    y: str,
    x: Optional[Sequence[str]] = None,
    *,
    break_point: Any,
    time: Optional[str] = None,
    break_vars: Optional[Sequence[str]] = None,
    vce: str = "nonrobust",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Chow test for a structural break at a known date.

    Tests that the regression coefficients are the same before and after
    ``break_point``. The model is fitted once with every coefficient in
    ``break_vars`` allowed to differ between the regimes, and the
    restrictions that the differences are zero are tested. With every
    coefficient free and classical errors this is Chow's statistic
    ``[(RSS - RSS_1 - RSS_2) / k] / [(RSS_1 + RSS_2) / (n - 2k)]``.

    Parameters
    ----------
    data : pandas.DataFrame
        Observations in time order.
    y : str
        Dependent variable.
    x : sequence of str, optional
        Regressors. A constant is always included; with ``x=None`` the test
        is for a shift in the mean.
    break_point : int, label or list of these
        Where the second regime starts. An ``int`` is the number of
        observations in the first regime, so ``break_point=50`` puts rows
        ``0..49`` before the break. With ``time=`` it is a value of that
        column, and anything else is looked up in the index of ``data``; in
        both cases the row carrying the value is the first of the new
        regime (Stata ``estat sbknown, break()``). A list tests several
        breaks jointly.
    time : str, optional
        Column that dates the observations.
    break_vars : sequence of str, optional
        Coefficients allowed to change, as names from ``x`` plus
        ``"const"``. Default: all of them.
    vce : {'nonrobust', 'hc0', 'hc1'}, default 'nonrobust'
        Covariance estimator of the Wald statistic. With ``'nonrobust'``
        the F statistic is exact under normal errors with one variance in
        every regime; ``'hc1'`` allows the variance to differ between
        regimes and observations.
    alpha : float, default 0.05
        Level used for ``reject``.

    Returns
    -------
    dict
        ``statistic`` (F), ``df1``, ``df2`` and ``pvalue``; ``chi2`` (the
        Wald statistic ``df1 * F``) and ``chi2_pvalue`` (Stata reports
        this pair); ``reject``; ``break_point`` (sizes of the regimes that
        end at each break); ``n_obs``; ``rss_restricted`` and
        ``rss_unrestricted``; ``regimes`` (a DataFrame of the coefficients
        in each regime); ``vce`` and ``test``.

    Notes
    -----
    A regime needs at least as many observations as it has free
    coefficients. Chow's second test, for a regime too short to fit, is not
    computed.

    The F reference holds for a date fixed in advance. Picking the date
    with the largest statistic and then reading its p-value here rejects a
    stable relation far more often than ``alpha``.

    References
    ----------
    chow1960tests

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 120
    >>> x = rng.normal(size=n)
    >>> shift = np.where(np.arange(n) < 60, 0.0, 1.5)
    >>> df = pd.DataFrame({"y": 1 + 0.5 * x + shift + rng.normal(size=n), "x": x})
    >>> res = sp.chow_test(df, "y", ["x"], break_point=60)
    >>> res["df1"], res["df2"]
    (2, 116)
    >>> bool(res["reject"])
    True
    """
    if vce not in ("nonrobust", "hc0", "hc1"):
        raise MethodIncompatibility(
            f"chow_test: vce must be 'nonrobust', 'hc0' or 'hc1', got {vce!r}."
        )
    x_names = [x] if isinstance(x, str) else list(x or [])
    missing = [c for c in [y] + x_names if c not in data.columns]
    if missing:
        raise MethodIncompatibility(
            f"chow_test: column(s) {missing} not found in data.",
            recovery_hint="Check the column names.",
        )
    frame = data[[y] + x_names]
    if frame.isna().to_numpy().any():
        raise MethodIncompatibility(
            "chow_test: missing values in y or x.",
            recovery_hint=(
                "Drop or fill them first; dropping rows silently would move "
                "the break date."
            ),
        )
    yv = frame[y].to_numpy(dtype=float)
    n = len(yv)
    X = np.column_stack(
        [np.ones(n)] + [frame[c].to_numpy(dtype=float) for c in x_names]
    )
    names = ["const"] + x_names
    k = X.shape[1]

    chosen = names if break_vars is None else list(dict.fromkeys(break_vars))
    unknown = [v for v in chosen if v not in names]
    if unknown or not chosen:
        raise MethodIncompatibility(
            f"chow_test: break_vars {unknown or chosen} must be drawn from {names}."
        )
    cols = [names.index(v) for v in chosen]
    q = len(cols)

    sizes = _first_regime_sizes(data, break_point, time, n)
    edges = [0] + sizes + [n]
    m = len(sizes)
    kk = k + m * q
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a < q:
            raise DataInsufficient(
                f"chow_test: a regime of {b - a} observations cannot identify "
                f"{q} coefficients.",
                recovery_hint="Move the break, or let fewer coefficients change.",
                diagnostics={"regime_sizes": list(np.diff(edges))},
            )
    if n - kk <= 0:
        raise DataInsufficient(
            "chow_test: no residual degrees of freedom.",
            diagnostics={"n_obs": n, "n_coefficients": kk},
        )

    # Regime j >= 1 adds its own shift of the breaking coefficients, so the
    # last m*q coefficients are differences from the first regime.
    blocks = [X]
    for j in range(1, m + 1):
        d = np.zeros(n)
        d[edges[j] : edges[j + 1]] = 1.0
        blocks.append(X[:, cols] * d[:, None])
    W = np.column_stack(blocks)
    if np.linalg.matrix_rank(W) < kk:
        raise MethodIncompatibility(
            "chow_test: the regressors are collinear within a regime.",
            recovery_hint=(
                "A regressor that is constant inside a regime cannot have its "
                "own coefficient there; leave it out of break_vars."
            ),
        )
    WtW_inv = np.linalg.inv(W.T @ W)
    beta = WtW_inv @ (W.T @ yv)
    e = yv - W @ beta
    rss_u = float(e @ e)
    b0 = np.linalg.lstsq(X, yv, rcond=None)[0]
    rss_r = float(np.sum((yv - X @ b0) ** 2))

    df1, df2 = m * q, n - kk
    if vce == "nonrobust":
        V = WtW_inv * rss_u / df2
    else:
        meat = (W * (e**2)[:, None]).T @ W
        V = WtW_inv @ meat @ WtW_inv
        if vce == "hc1":
            V = V * (n / df2)
    delta = beta[k:]
    wald = float(delta @ np.linalg.solve(V[k:, k:], delta))
    f_stat = wald / df1
    pvalue = float(stats.f.sf(f_stat, df1, df2))

    # Coefficients regime by regime: the base plus that regime's shift.
    regimes = {}
    for j in range(m + 1):
        coef = beta[:k].copy()
        if j >= 1:
            coef[cols] += beta[k + (j - 1) * q : k + j * q]
        regimes[f"regime_{j + 1}"] = coef
    regime_table = pd.DataFrame(regimes, index=names)
    regime_table.loc["n_obs"] = np.diff(edges)

    return {
        "test": "Chow test for a break at a known date",
        "statistic": f_stat,
        "df1": df1,
        "df2": df2,
        "pvalue": pvalue,
        "chi2": wald,
        "chi2_pvalue": float(stats.chi2.sf(wald, df1)),
        "reject": bool(pvalue < alpha),
        "break_point": sizes if m > 1 else sizes[0],
        "break_vars": chosen,
        "n_obs": n,
        "rss_restricted": rss_r,
        "rss_unrestricted": rss_u,
        "regimes": regime_table,
        "vce": vce,
    }
