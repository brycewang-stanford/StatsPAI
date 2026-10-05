"""Rolling and recursive least squares.

The same regression is fitted on a window that moves through the sample,
or on a sample that grows from a fixed start. The path of the
coefficients shows whether a relation is stable: a rolling market beta, or
the coefficients of a forecasting equation over time.
"""

from __future__ import annotations

import warnings
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["rolling"]

_VCE = ("nonrobust", "hc0", "hc1", "hc2", "hc3")


def _window_fit(X: np.ndarray, y: np.ndarray, vce: str) -> Optional[dict]:
    """OLS on one window; None when the design is rank deficient there."""
    n, k = X.shape
    XtX = X.T @ X
    if np.linalg.matrix_rank(XtX) < k:
        return None
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ (X.T @ y)
    e = y - X @ beta
    rss = float(e @ e)
    df = n - k
    if vce == "nonrobust":
        V = XtX_inv * (rss / df) if df > 0 else np.full((k, k), np.nan)
    else:
        w = e**2
        if vce in ("hc2", "hc3"):
            h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
            w = w / (1.0 - h) ** (1 if vce == "hc2" else 2)
        V = XtX_inv @ ((X * w[:, None]).T @ X) @ XtX_inv
        if vce == "hc1":
            V = V * (n / df) if df > 0 else np.full((k, k), np.nan)
    centred = y - y.mean()
    tss = float(centred @ centred)
    return {
        "beta": beta,
        "se": np.sqrt(np.diag(V)),
        "r2": 1.0 - rss / tss if tss > 0 else np.nan,
        "rmse": np.sqrt(rss / df) if df > 0 else np.nan,
    }


def rolling(
    formula: str,
    data: pd.DataFrame,
    window: int,
    *,
    step: int = 1,
    recursive: bool = False,
    vce: str = "nonrobust",
) -> pd.DataFrame:
    """Rolling or recursive least squares.

    Fits ``formula`` by OLS on each window of ``window`` consecutive
    observations, moving ``step`` observations at a time. With
    ``recursive=True`` the start stays at the first observation and the
    sample grows instead (recursive least squares, Stata ``rolling,
    recursive``).

    Parameters
    ----------
    formula : str
        Regression formula, as for :func:`statspai.regress`.
    data : pandas.DataFrame
        Observations in time order.
    window : int
        Observations in each window; with ``recursive=True``, in the first.
    step : int, default 1
        Observations between the ends of consecutive windows.
    recursive : bool, default False
        Keep the start fixed and let the sample grow.
    vce : {'nonrobust', 'hc0', 'hc1', 'hc2', 'hc3'}, default 'nonrobust'
        Covariance estimator of the standard errors in each window.

    Returns
    -------
    pandas.DataFrame
        One row per window, indexed by the label of its last observation.
        Columns ``start`` and ``end`` (index labels of the first and last
        observation), ``nobs``, one column per coefficient under its name,
        one per standard error under ``se_<name>``, then ``r2`` and
        ``rmse``. ``attrs`` records ``window``, ``step``, ``recursive``,
        ``vce``, ``formula`` and ``params`` (the coefficient names).

    Notes
    -----
    Windows are counted over the rows the regression can use. A row with a
    missing value in any variable is dropped first, so with gaps inside
    the sample a window spans more calendar time than ``window`` rows; the
    ``start`` and ``end`` columns show what each window covers.

    A window in which the regressors are collinear -- a dummy that is
    constant there, say -- has no unique fit. Its coefficients are returned
    as missing and a warning counts such windows.

    Each window is a separate regression on few observations, so the
    paths are noisy, and consecutive windows share all but ``step``
    observations, so their estimates move together. A drift in the path is
    not by itself evidence of instability; :func:`statspai.cusum_test` and
    :func:`statspai.structural_break` test it.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 120
    >>> market = rng.normal(size=n)
    >>> beta = np.where(np.arange(n) < 60, 0.8, 1.4)
    >>> df = pd.DataFrame({"market": market})
    >>> df["stock"] = beta * market + rng.normal(scale=0.3, size=n)
    >>> path = sp.rolling("stock ~ market", df, window=36)
    >>> path.shape[0]  # 120 - 36 + 1 windows
    85
    >>> bool(path["market"].iloc[0] < path["market"].iloc[-1])
    True
    """
    from .ols import regress

    if vce not in _VCE:
        raise MethodIncompatibility(
            f"rolling: vce must be one of {_VCE}, got {vce!r}.",
            recovery_hint="Use 'nonrobust' or 'hc1'.",
        )
    window, step = int(window), int(step)
    if step < 1:
        raise MethodIncompatibility("rolling: step must be at least 1.")

    # One full-sample fit builds the design: the formula's transformations
    # (lags, powers, categorical terms) are applied to the whole series, as
    # they must be, and then sliced.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = regress(formula, data)
    info = full.data_info
    X = np.asarray(info["X"], dtype=float)
    y = np.asarray(info["y"], dtype=float).ravel()
    names: List[str] = [str(c) for c in full.params.index]
    if X.shape[1] != len(names):
        raise MethodIncompatibility(
            "rolling: the regressors are collinear in the full sample.",
            recovery_hint="Drop the redundant term from the formula.",
            diagnostics={"omitted": full.model_info.get("omitted")},
        )
    n, k = X.shape
    if window <= k:
        raise DataInsufficient(
            f"rolling: a window of {window} observations cannot fit "
            f"{k} coefficients.",
            recovery_hint=f"Use window > {k}.",
        )
    if window > n:
        raise DataInsufficient(
            f"rolling: window={window} exceeds the {n} usable observations.",
            recovery_hint="Use a shorter window.",
        )
    labels: Any = info.get("sample_index")
    if labels is None or len(labels) != n:
        labels = data.index if len(data.index) == n else pd.RangeIndex(n)
    labels = pd.Index(labels)

    rows = []
    n_singular = 0
    for end in range(window, n + 1, step):
        start = 0 if recursive else end - window
        fit = _window_fit(X[start:end], y[start:end], vce)
        row = {"start": labels[start], "end": labels[end - 1], "nobs": end - start}
        if fit is None:
            n_singular += 1
            beta = se = np.full(k, np.nan)
            r2 = rmse = np.nan
        else:
            beta, se, r2, rmse = fit["beta"], fit["se"], fit["r2"], fit["rmse"]
        row.update(dict(zip(names, beta)))
        row.update({f"se_{nm}": s for nm, s in zip(names, se)})
        row["r2"], row["rmse"] = r2, rmse
        rows.append(row)
    if n_singular:
        warnings.warn(
            f"rolling: the regressors are collinear in {n_singular} of "
            f"{len(rows)} windows; their coefficients are missing.",
            UserWarning,
            stacklevel=2,
        )
    out = pd.DataFrame(rows, index=pd.Index([r["end"] for r in rows], name=labels.name))
    out.attrs.update(
        {
            "window": window,
            "step": step,
            "recursive": bool(recursive),
            "vce": vce,
            "formula": formula,
            "params": names,
        }
    )
    return out
